"""Per-row orchestrator for the support-triage pipeline.

Drives one ticket through the deterministic state machine::

    triage -> hard-rule gate -> retrieve (with Visa fallback) ->
    weak-grounding gate -> specialist -> verifier -> [optional revise] ->
    assemble row | escalate

Every step is timed (``time.monotonic``) and recorded in an
:class:`OrchestratorTrace` that the caller can persist to disk for
post-hoc auditing.  The orchestrator never raises out of
:meth:`Orchestrator.process_row`; any unexpected exception is converted
to a ``validation_failure`` escalation so the CSV stays well-formed.

A ``stub_mode`` switch short-circuits the entire pipeline to a canned
"replied" row.  This is what ``main.py --dry-run`` uses to prove the
wiring without spending API tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make ``code/`` importable when this file is run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import safety  # noqa: E402
from llm import LLMError  # noqa: E402
from schema import normalize_company  # noqa: E402

from agents.escalation import EscalationResult, escalate as _escalate_writer  # noqa: E402
from agents.specialist import (  # noqa: E402
    SpecialistResult,
    _fallback_product_area,
    specialist as _run_specialist,
)
from agents.triage import TriageResult, triage as _run_triage  # noqa: E402
from agents.verifier import VerifierResult, verify as _run_verify  # noqa: E402

# ---------------------------------------------------------------------------
# Trace dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorTrace:
    """Machine-readable record of one row's full pipeline pass."""

    row_id: str
    input: dict
    triage: dict | None
    retrieval: list[dict]
    specialist: dict | None
    specialist_revised: dict | None
    verifier: dict | None
    escalation: dict | None
    decision: str
    escalation_trigger: str | None
    elapsed_ms: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_row_id(issue: str, subject: str, company: str) -> str:
    """Stable 12-char id derived from the three input columns.

    SHA-256 over ``issue \x1f subject \x1f company`` (utf-8); we keep the
    first 12 hex chars for log readability.  Collisions are vanishingly
    unlikely at the 56-row scale of this challenge.
    """
    parts = [issue or "", subject or "", company or ""]
    raw = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _synthetic_triage(company: str) -> TriageResult:
    """Build a minimal :class:`TriageResult` when real triage failed.

    Mirrors :func:`agents.escalation._synthetic_triage` but lives here
    so the orchestrator does not depend on a private helper.
    """
    try:
        canon = normalize_company(company)
    except ValueError:
        canon = "none"
    return TriageResult(
        request_type="invalid",
        inferred_company=canon,
        scope="partial",
        ambiguity="med",
        risk_flags=[],
        sub_requests=[],
        rationale="synthetic triage (orchestrator fallback)",
        deterministic_flags=[],
    )


def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _elapsed(start_ms: float) -> int:
    return int(_now_ms() - start_ms)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Drive one row at a time through the support-triage pipeline."""

    def __init__(
        self,
        retriever: Any,
        *,
        weak_grounding_threshold: float = safety.WEAK_GROUNDING_THRESHOLD,
        enable_revise: bool = True,
        trace_dir: str | Path | None = None,
        stub_mode: bool = False,
    ) -> None:
        self.retriever = retriever
        self.weak_grounding_threshold = float(weak_grounding_threshold)
        self.enable_revise = bool(enable_revise)
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.stub_mode = bool(stub_mode)

    # ---- public ----

    def process_row(
        self, issue: str, subject: str, company: str
    ) -> tuple[dict, OrchestratorTrace]:
        """Run one ticket through the pipeline.

        Returns the final CSV-shaped row dict and the :class:`OrchestratorTrace`.
        Never raises: unexpected exceptions become ``validation_failure``
        escalations.
        """
        row_id = compute_row_id(issue, subject, company)
        input_payload = {
            "issue": issue or "",
            "subject": subject or "",
            "company": company or "",
        }
        elapsed_ms: dict[str, int] = {}
        triage_obj: TriageResult | None = None
        retrieval_chunks: list[Any] = []
        specialist_draft: SpecialistResult | None = None
        specialist_revised: SpecialistResult | None = None
        verifier_result: VerifierResult | None = None

        try:
            # 1. Stub mode short-circuit.
            if self.stub_mode:
                row, trace = self._stub_row(
                    row_id=row_id, input_payload=input_payload, company=company
                )
                self._write_trace_safe(trace)
                return row, trace

            # 2. Triage.
            t0 = _now_ms()
            try:
                triage_obj = _run_triage(issue=issue, subject=subject, company=company)
            except Exception as exc:  # noqa: BLE001 - we wrap any failure
                elapsed_ms["triage"] = _elapsed(t0)
                triage_obj = _synthetic_triage(company)
                just = (
                    f"validation_failure -- triage error: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="validation_failure",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                    free_form_reason=just,
                )
            elapsed_ms["triage"] = _elapsed(t0)

            # 3. Hard-rule gate.
            t0 = _now_ms()
            hard_trigger = safety.hard_escalate_trigger(triage_obj)
            elapsed_ms["safety_gate"] = _elapsed(t0)
            if hard_trigger is not None:
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger=hard_trigger,
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                )

            # 4. Multi-request unresolved gate.
            if safety.is_multi_request_unresolved(triage_obj):
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="multi_request_unresolved",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                )

            # 5. Retrieve.
            t0 = _now_ms()
            query = (f"{subject or ''}\n{issue or ''}".strip()) or (issue or "")
            company_filter = (
                triage_obj.inferred_company
                if triage_obj.inferred_company != "none"
                else None
            )
            try:
                retrieval_chunks = list(
                    self.retriever.retrieve(query, company=company_filter, k=5)
                )
            except Exception as exc:  # noqa: BLE001
                elapsed_ms["retrieval"] = _elapsed(t0)
                just = (
                    f"validation_failure -- retrieval error: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="validation_failure",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=[],
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                    free_form_reason=just,
                )

            # Visa fallback: when the company filter starved retrieval, retry
            # cross-company.  The visa corpus is thin and frequently gives
            # zero hits for the user's literal phrasing.
            if not retrieval_chunks and company_filter == "visa":
                try:
                    retrieval_chunks = list(
                        self.retriever.retrieve(query, company=None, k=5)
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms["retrieval"] = _elapsed(t0)
                    just = (
                        f"validation_failure -- retrieval (visa fallback) error: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                    return self._escalate(
                        issue=issue,
                        subject=subject,
                        company=company,
                        triage_obj=triage_obj,
                        trigger="validation_failure",
                        row_id=row_id,
                        input_payload=input_payload,
                        retrieval_chunks=[],
                        specialist_draft=None,
                        specialist_revised=None,
                        verifier_result=None,
                        elapsed_ms=elapsed_ms,
                        free_form_reason=just,
                    )
            elapsed_ms["retrieval"] = _elapsed(t0)

            if not retrieval_chunks:
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="out_of_scope",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=[],
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                )

            # 6. Weak grounding gate.
            if safety.is_weak_grounding(retrieval_chunks, self.weak_grounding_threshold):
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="weak_grounding",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                )

            # 7. Specialist (first pass).
            t0 = _now_ms()
            try:
                specialist_draft = _run_specialist(
                    issue=issue,
                    subject=subject,
                    triage_obj=triage_obj,
                    chunks=retrieval_chunks,
                )
            except LLMError as exc:
                elapsed_ms["specialist"] = _elapsed(t0)
                just = (
                    f"validation_failure -- specialist LLMError: "
                    f"{str(exc)[:200]}"
                )
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="validation_failure",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=None,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                    free_form_reason=just,
                )
            elapsed_ms["specialist"] = _elapsed(t0)

            if specialist_draft.insufficient_evidence:
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="insufficient_evidence",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=specialist_draft,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                )

            # 8. Verifier.
            t0 = _now_ms()
            try:
                verifier_result = _run_verify(
                    issue=issue,
                    subject=subject,
                    chunks=retrieval_chunks,
                    draft=specialist_draft,
                )
            except LLMError as exc:
                elapsed_ms["verifier"] = _elapsed(t0)
                just = (
                    f"validation_failure -- verifier LLMError: "
                    f"{str(exc)[:200]}"
                )
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="validation_failure",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=specialist_draft,
                    specialist_revised=None,
                    verifier_result=None,
                    elapsed_ms=elapsed_ms,
                    free_form_reason=just,
                )
            elapsed_ms["verifier"] = _elapsed(t0)

            # 9. Action gate.
            action = verifier_result.suggested_action
            final_draft: SpecialistResult = specialist_draft
            if action == "revise" and self.enable_revise:
                t0 = _now_ms()
                revised_issue = (
                    f"{issue}\n\n[REVISION NOTES from verifier]\n"
                    f"{verifier_result.verifier_notes}\n"
                    f"Unsupported claims to avoid: "
                    f"{verifier_result.unsupported_claims}\n"
                )
                try:
                    specialist_revised = _run_specialist(
                        issue=revised_issue,
                        subject=subject,
                        triage_obj=triage_obj,
                        chunks=retrieval_chunks,
                    )
                except LLMError as exc:
                    elapsed_ms["specialist_revised"] = _elapsed(t0)
                    just = (
                        f"validation_failure -- specialist (revise) LLMError: "
                        f"{str(exc)[:200]}"
                    )
                    return self._escalate(
                        issue=issue,
                        subject=subject,
                        company=company,
                        triage_obj=triage_obj,
                        trigger="validation_failure",
                        row_id=row_id,
                        input_payload=input_payload,
                        retrieval_chunks=retrieval_chunks,
                        specialist_draft=specialist_draft,
                        specialist_revised=None,
                        verifier_result=verifier_result,
                        elapsed_ms=elapsed_ms,
                        free_form_reason=just,
                    )
                elapsed_ms["specialist_revised"] = _elapsed(t0)

                t0 = _now_ms()
                try:
                    verifier_result = _run_verify(
                        issue=issue,
                        subject=subject,
                        chunks=retrieval_chunks,
                        draft=specialist_revised,
                    )
                except LLMError as exc:
                    elapsed_ms["verifier_revised"] = _elapsed(t0)
                    just = (
                        f"validation_failure -- verifier (revised) LLMError: "
                        f"{str(exc)[:200]}"
                    )
                    return self._escalate(
                        issue=issue,
                        subject=subject,
                        company=company,
                        triage_obj=triage_obj,
                        trigger="validation_failure",
                        row_id=row_id,
                        input_payload=input_payload,
                        retrieval_chunks=retrieval_chunks,
                        specialist_draft=specialist_draft,
                        specialist_revised=specialist_revised,
                        verifier_result=None,
                        elapsed_ms=elapsed_ms,
                        free_form_reason=just,
                    )
                elapsed_ms["verifier_revised"] = _elapsed(t0)

                if verifier_result.suggested_action != "accept":
                    return self._escalate(
                        issue=issue,
                        subject=subject,
                        company=company,
                        triage_obj=triage_obj,
                        trigger="verifier_rejected",
                        row_id=row_id,
                        input_payload=input_payload,
                        retrieval_chunks=retrieval_chunks,
                        specialist_draft=specialist_draft,
                        specialist_revised=specialist_revised,
                        verifier_result=verifier_result,
                        elapsed_ms=elapsed_ms,
                    )
                final_draft = specialist_revised
            elif action == "accept":
                final_draft = specialist_draft
            else:
                # "escalate", or "revise" with retries disabled.
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_obj,
                    trigger="verifier_rejected",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=specialist_draft,
                    specialist_revised=specialist_revised,
                    verifier_result=verifier_result,
                    elapsed_ms=elapsed_ms,
                )

            # 10. Build the replied row.
            request_type = (
                final_draft.request_type_override or triage_obj.request_type
            )
            product_area = final_draft.product_area or _fallback_product_area(
                triage_obj.inferred_company
            )
            row = safety.assemble_output_row(
                issue=issue,
                subject=subject,
                company=company,
                decision="replied",
                response_text=final_draft.response,
                justification=final_draft.justification,
                product_area=product_area,
                request_type=request_type,
            )

            # If validation forced a coercion, the Status will now read
            # "Escalated"; capture that so the trace tells the truth.
            decision_str = "replied"
            escalation_trigger: str | None = None
            if row.get("Status") == "Escalated":
                decision_str = "escalated"
                escalation_trigger = "validation_failure"

            trace = OrchestratorTrace(
                row_id=row_id,
                input=input_payload,
                triage=triage_obj.to_dict(),
                retrieval=[c.to_dict() for c in retrieval_chunks],
                specialist=specialist_draft.to_dict(),
                specialist_revised=(
                    specialist_revised.to_dict() if specialist_revised else None
                ),
                verifier=verifier_result.to_dict(),
                escalation=None,
                decision=decision_str,
                escalation_trigger=escalation_trigger,
                elapsed_ms=elapsed_ms,
                error=None,
            )
            self._write_trace_safe(trace)
            return row, trace
        except Exception as exc:  # noqa: BLE001 - last-line safety net
            # Anything we missed: convert to a validation_failure escalation
            # so the CSV stays well-formed.
            err_text = (
                f"{type(exc).__name__}: {str(exc)[:200]}\n"
                f"{traceback.format_exc()[:1000]}"
            )
            triage_for_esc = triage_obj or _synthetic_triage(company)
            try:
                return self._escalate(
                    issue=issue,
                    subject=subject,
                    company=company,
                    triage_obj=triage_for_esc,
                    trigger="validation_failure",
                    row_id=row_id,
                    input_payload=input_payload,
                    retrieval_chunks=retrieval_chunks,
                    specialist_draft=specialist_draft,
                    specialist_revised=specialist_revised,
                    verifier_result=verifier_result,
                    elapsed_ms=elapsed_ms,
                    free_form_reason=f"unhandled: {err_text[:200]}",
                    error=err_text,
                )
            except Exception:  # noqa: BLE001 - escalation itself failed
                # Build a minimal stub so we still return a row.
                row = safety.assemble_output_row(
                    issue=issue,
                    subject=subject,
                    company=company,
                    decision="escalated",
                    response_text=(
                        "Thank you for reaching out. We're routing this to a human "
                        "agent who will follow up with you shortly."
                    ),
                    justification=f"validation_failure -- {err_text[:200]}",
                    product_area="",
                    request_type="invalid",
                )
                trace = OrchestratorTrace(
                    row_id=row_id,
                    input=input_payload,
                    triage=triage_obj.to_dict() if triage_obj else None,
                    retrieval=[],
                    specialist=None,
                    specialist_revised=None,
                    verifier=None,
                    escalation=None,
                    decision="escalated",
                    escalation_trigger="validation_failure",
                    elapsed_ms=elapsed_ms,
                    error=err_text,
                )
                self._write_trace_safe(trace)
                return row, trace

    # ---- internals ----

    def _stub_row(
        self,
        *,
        row_id: str,
        input_payload: dict,
        company: str,
    ) -> tuple[dict, OrchestratorTrace]:
        """Build a canned ``Replied`` row for ``--dry-run``."""
        try:
            company_canon = normalize_company(company)
        except ValueError:
            company_canon = "none"
        product_area = _fallback_product_area(company_canon)
        row = safety.assemble_output_row(
            issue=input_payload["issue"],
            subject=input_payload["subject"],
            company=input_payload["company"],
            decision="replied",
            response_text=(
                "Thank you for reaching out. Stub mode is active."
            ),
            justification="stub_mode=True; --dry-run path",
            product_area=product_area,
            request_type="product_issue",
        )
        trace = OrchestratorTrace(
            row_id=row_id,
            input=input_payload,
            triage=None,
            retrieval=[],
            specialist=None,
            specialist_revised=None,
            verifier=None,
            escalation=None,
            decision="replied",
            escalation_trigger=None,
            elapsed_ms={"stub": 0},
            error=None,
        )
        return row, trace

    def _escalate(
        self,
        *,
        issue: str,
        subject: str,
        company: str,
        triage_obj: TriageResult,
        trigger: str,
        row_id: str,
        input_payload: dict,
        retrieval_chunks: list[Any],
        specialist_draft: SpecialistResult | None,
        specialist_revised: SpecialistResult | None,
        verifier_result: VerifierResult | None,
        elapsed_ms: dict,
        free_form_reason: str | None = None,
        error: str | None = None,
    ) -> tuple[dict, OrchestratorTrace]:
        """Build an escalated row and matching trace, write it, and return."""
        try:
            esc: EscalationResult = _escalate_writer(
                issue=issue,
                subject=subject,
                triage_obj=triage_obj,
                reason=trigger,
                free_form_reason=free_form_reason,
            )
        except ValueError:
            # Unknown trigger: route to "unknown" template, preserve original
            # trigger string in the audit trail.
            esc = _escalate_writer(
                issue=issue,
                subject=subject,
                triage_obj=triage_obj,
                reason="unknown",
                free_form_reason=(
                    free_form_reason or f"orchestrator trigger={trigger}"
                ),
            )
            trigger = "unknown"

        request_type = (
            triage_obj.request_type if triage_obj is not None else "invalid"
        )

        row = safety.assemble_output_row(
            issue=issue,
            subject=subject,
            company=company,
            decision="escalated",
            response_text=esc.response,
            justification=esc.justification,
            product_area=esc.product_area,
            request_type=request_type,
        )

        trace = OrchestratorTrace(
            row_id=row_id,
            input=input_payload,
            triage=triage_obj.to_dict() if triage_obj else None,
            retrieval=[c.to_dict() for c in retrieval_chunks],
            specialist=specialist_draft.to_dict() if specialist_draft else None,
            specialist_revised=(
                specialist_revised.to_dict() if specialist_revised else None
            ),
            verifier=verifier_result.to_dict() if verifier_result else None,
            escalation=esc.to_dict(),
            decision="escalated",
            escalation_trigger=trigger,
            elapsed_ms=elapsed_ms,
            error=error,
        )
        self._write_trace_safe(trace)
        return row, trace

    def _write_trace_safe(self, trace: OrchestratorTrace) -> None:
        """Best-effort trace write; never raises."""
        if self.trace_dir is None:
            return
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"{trace.row_id}.json"
            tmp = path.with_suffix(path.suffix + ".tmp")
            text = json.dumps(
                trace.to_dict(), sort_keys=True, indent=2, ensure_ascii=False
            )
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.write("\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - trace I/O is non-fatal
            pass
