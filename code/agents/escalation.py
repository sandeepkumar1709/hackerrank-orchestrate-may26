"""Stage-5 escalation writer for the support-triage pipeline.

The escalation agent is intentionally **deterministic** -- no LLM call. It
takes a triage result and a categorical ``reason`` from a closed vocabulary
and emits a templated, neutral acknowledgement plus a one-line justification
that names the trigger.

We keep this dumb on purpose. Escalation messages must:

1. Never assert a policy or speculate.
2. Cite the trigger so downstream agents can audit.
3. Be safe to write even when every other component failed (LLMError,
   retrieval empty, validation errors, etc).

CLI
---
``python code/agents/escalation.py --reason payments_fraud --issue "..."``
prints a JSON ``EscalationResult``. Returns exit 3 if ``--reason`` is not in
``ESCALATION_TRIGGERS``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Make ``code/`` importable when this file is run as a script.
_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from agents.triage import TriageResult, triage  # noqa: E402

# ---------------------------------------------------------------------------
# Public vocab
# ---------------------------------------------------------------------------

ESCALATION_TRIGGERS: tuple[str, ...] = (
    # Triage-level risk flags
    "account_access",
    "payments_fraud",
    "security_incident",
    "identity_verification",
    "legal",
    "pii_in_request",
    "prompt_injection",
    "out_of_scope",
    # Pipeline-level outcomes
    "weak_grounding",
    "verifier_rejected",
    "insufficient_evidence",
    "multi_request_unresolved",
    "validation_failure",
    "unknown",
)


@dataclass(frozen=True)
class EscalationResult:
    response: str
    justification: str
    product_area: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Each value is a ``(response_template, justification_template)`` pair.
# Both are formatted with ``.format()`` over a small, fixed keyword set:
# ``{free_form_reason}`` is always defined (may be empty). No other variables
# appear in any template -- so every key is truly safe even if a future caller
# forgets to pass anything else.

_TEMPLATES: dict[str, tuple[str, str]] = {
    "account_access": (
        "Thank you for reaching out. Because this involves account access, "
        "we're routing your request to a human agent who can verify your "
        "identity and help you regain access. They will follow up with you "
        "shortly.",
        "Escalated: account_access risk flag -- identity verification required "
        "before any account change.",
    ),
    "payments_fraud": (
        "Thank you for reaching out. Because this involves a possible "
        "unauthorized charge, we're routing your request to a human agent. "
        "If your card may have been compromised, please also contact your "
        "issuing bank directly using the number on the back of your card.",
        "Escalated: payments_fraud risk flag -- financial dispute requires "
        "human handling and bank coordination.",
    ),
    "security_incident": (
        "Thank you for reaching out. Because this involves a possible "
        "security incident, we're routing your request to our security team. "
        "They will follow up with you shortly. In the meantime, please change "
        "your password and review recent account activity if you can.",
        "Escalated: security_incident risk flag -- incident response handled "
        "by security team.",
    ),
    "identity_verification": (
        "Thank you for reaching out. Identity verification needs to be "
        "handled by a human agent for your safety. We're routing your "
        "request now and they will follow up with you shortly.",
        "Escalated: identity_verification risk flag -- KYC/ID flow requires "
        "human review.",
    ),
    "legal": (
        "Thank you for reaching out. Because your request involves a legal "
        "matter, we're routing it to the team that handles those inquiries. "
        "They will follow up with you shortly.",
        "Escalated: legal risk flag -- routed to legal-handling team.",
    ),
    "pii_in_request": (
        "Thank you for reaching out. We noticed your message contains "
        "sensitive personal information. For your safety we're routing this "
        "to a human agent who can handle it through a secure channel. Please "
        "do not share additional sensitive details in chat.",
        "Escalated: pii_in_request risk flag -- sensitive data must be "
        "handled out-of-band.",
    ),
    "prompt_injection": (
        "Thank you for reaching out. Your request will be reviewed by a "
        "human agent who will follow up with you shortly.",
        "Escalated: prompt_injection risk flag -- refused automated handling, "
        "neutral acknowledgement only.",
    ),
    "out_of_scope": (
        "Thank you for reaching out. This question is outside the scope of "
        "the support topics we cover here, so we're routing it to a human "
        "agent who can help direct you to the right place.",
        "Escalated: out_of_scope -- ticket falls outside the supported "
        "HackerRank/Claude/Visa corpora.",
    ),
    "weak_grounding": (
        "Thank you for reaching out. We don't have enough information in our "
        "knowledge base to give you a confident answer here, so we're "
        "routing this to a human agent who will follow up with you shortly.",
        "Escalated: weak_grounding -- top retrieved chunks scored below the "
        "confidence threshold.",
    ),
    "verifier_rejected": (
        "Thank you for reaching out. We want to make sure we give you an "
        "accurate answer, so a human agent is going to take this one and "
        "follow up with you shortly.",
        "Escalated: verifier_rejected -- independent faithfulness check did "
        "not approve the draft response.",
    ),
    "insufficient_evidence": (
        "Thank you for reaching out. We don't have a definitive answer for "
        "this in our knowledge base, so we're routing your request to a "
        "human agent who will follow up shortly.",
        "Escalated: insufficient_evidence -- no retrieved chunk supported "
        "the answer.",
    ),
    "multi_request_unresolved": (
        "Thank you for reaching out. Your message contains multiple "
        "requests, and we want to handle each one properly, so we're "
        "routing this to a human agent who will follow up with you shortly.",
        "Escalated: multi_request_unresolved -- chunks covered only part of "
        "the bundled requests.",
    ),
    "validation_failure": (
        "Thank you for reaching out. We hit an issue preparing your reply, "
        "so a human agent will pick this up and follow up with you shortly.",
        "Escalated: validation_failure -- output failed schema/enum checks "
        "before write.",
    ),
    "unknown": (
        "Thank you for reaching out. We're routing your request to a human "
        "agent who will follow up with you shortly.",
        # Note: when 'unknown' is used we append free_form_reason verbatim
        # below so the audit trail still says *something* useful.
        "Escalated: unknown trigger.{free_form_reason}",
    ),
}


# ---------------------------------------------------------------------------
# Product area mapping (intentionally coarse -- escalations don't have to be
# specific; the human picking up the ticket can re-categorize. We use the
# triage company as the source.)
# ---------------------------------------------------------------------------

_ESCALATION_PRODUCT_AREA: dict[str, str] = {
    "hackerrank": "screen",
    "claude": "conversation_management",
    "visa": "general_support",
    "none": "",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def escalate(
    issue: str,
    subject: str | None,
    triage_obj: TriageResult,
    reason: str,
    *,
    free_form_reason: str | None = None,
) -> EscalationResult:
    """Render a templated escalation row.

    Parameters
    ----------
    issue, subject:
        The original ticket. Currently unused by the templates (they're
        deliberately generic), but accepted so callers can swap in custom
        templates later without changing the call site.
    triage_obj:
        Source of truth for ``inferred_company`` (drives ``product_area``).
    reason:
        Must be in :data:`ESCALATION_TRIGGERS` -- raises ``ValueError``
        otherwise so callers can't silently drop an escalation off the
        whitelist.
    free_form_reason:
        Free-text used only when ``reason == "unknown"`` so the audit trail
        captures *something* about the trigger. Empty/None for the templated
        cases.
    """
    # Suppress unused-arg lints; we accept these for API stability.
    _ = (issue, subject)

    if reason not in ESCALATION_TRIGGERS:
        raise ValueError(
            f"unknown escalation reason {reason!r}; "
            f"must be one of {ESCALATION_TRIGGERS}"
        )

    response_tmpl, justification_tmpl = _TEMPLATES[reason]

    # Only the 'unknown' template references {free_form_reason}; build the
    # appended fragment safely for all other templates (it'll just be unused).
    free = (free_form_reason or "").strip()
    free_fragment = f" Detail: {free}" if free else ""

    response = response_tmpl  # all non-unknown templates have no placeholders
    justification = justification_tmpl.format(free_form_reason=free_fragment)

    # justification on max 280 chars, single line.
    if len(justification) > 280:
        justification = justification[:280].rstrip()
    if "\n" in justification:
        justification = justification.replace("\n", " ").strip()

    product_area = _ESCALATION_PRODUCT_AREA.get(triage_obj.inferred_company, "")

    return EscalationResult(
        response=response,
        justification=justification,
        product_area=product_area,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="escalation",
        description=(
            "Render a templated escalation row from a categorical reason. "
            "No LLM call."
        ),
    )
    p.add_argument(
        "--reason",
        required=True,
        help=f"Escalation trigger; one of {list(ESCALATION_TRIGGERS)}.",
    )
    p.add_argument("--issue", default="", help="The user's issue text (optional).")
    p.add_argument(
        "--subject",
        default="",
        help="Optional subject line (currently unused by templates).",
    )
    p.add_argument(
        "--company",
        default="none",
        help="Company hint used to derive product_area (hackerrank|claude|visa|none).",
    )
    p.add_argument(
        "--free-form-reason",
        default="",
        help=(
            "Free-text reason text appended to the audit trail when --reason "
            "is 'unknown'. Ignored for templated reasons."
        ),
    )
    p.add_argument(
        "--no-triage-llm",
        action="store_true",
        default=True,
        help=(
            "Skip the LLM-backed triage call; use a synthetic TriageResult "
            "(default: enabled, since the escalation writer only needs the "
            "company hint)."
        ),
    )
    return p


def _synthetic_triage(issue: str, subject: str, company: str) -> TriageResult:
    """Build a minimal TriageResult from inputs without calling the LLM.

    Escalation only needs ``inferred_company`` to pick a product_area, so we
    skip the LLM-backed pipeline by default.
    """
    from schema import normalize_company  # local import to avoid cycle at module import

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
        rationale="synthetic triage (escalation CLI)",
        deterministic_flags=[],
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.reason not in ESCALATION_TRIGGERS:
        # Match argparse-style error text but exit 3 (validation, per plan).
        print(
            f"error: --reason {args.reason!r} is not a valid escalation trigger. "
            f"Allowed: {list(ESCALATION_TRIGGERS)}",
            file=sys.stderr,
        )
        return 3

    if args.no_triage_llm:
        triage_obj = _synthetic_triage(args.issue, args.subject, args.company)
    else:  # pragma: no cover - LLM path, not exercised in CI
        from llm import LLMError  # noqa: E402

        try:
            triage_obj = triage(
                issue=args.issue,
                subject=args.subject,
                company=args.company,
            )
        except LLMError as exc:
            print(f"LLMError (triage): {exc}", file=sys.stderr)
            return 2

    try:
        result = escalate(
            issue=args.issue,
            subject=args.subject,
            triage_obj=triage_obj,
            reason=args.reason,
            free_form_reason=args.free_form_reason or None,
        )
    except ValueError as exc:
        print(f"ValidationError: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
