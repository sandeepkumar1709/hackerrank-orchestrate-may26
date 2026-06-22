"""Safety / gating helpers shared by the orchestrator.

Three concerns live here:

1. Mapping triage ``risk_flags`` (an open-ended set) to the closed
   :data:`escalation.ESCALATION_TRIGGERS` vocabulary used by the
   escalation writer (:func:`hard_escalate_trigger`).
2. Detecting weak retrieval grounding (:func:`is_weak_grounding`) and
   unresolved multi-request tickets (:func:`is_multi_request_unresolved`).
3. Assembling the final CSV row dict from per-stage outputs and
   coercing it to an escalated stub when it fails schema validation
   (:func:`assemble_output_row`).

This module performs no LLM calls and has no I/O. It is safe to import
from anywhere in the pipeline and is exercised end-to-end by the
orchestrator's dry-run path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

# Make ``code/`` importable when this file is run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from schema import (  # noqa: E402
    coerce_to_escalated,
    format_request_type,
    format_status,
    normalize_request_type,
    validate_output_row,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEAK_GROUNDING_THRESHOLD = 0.3
RESPONSE_MAX_CHARS = 4000

# Maps the open ``triage.risk_flags`` set to the closed
# ``escalation.ESCALATION_TRIGGERS`` vocabulary.  Only the flags that should
# unconditionally divert to escalation appear here — flags like ``urgency``
# and ``multi_request`` are intentionally absent because they are handled
# elsewhere (multi-request via :func:`is_multi_request_unresolved`, urgency
# is informational only).
RISK_FLAG_TO_TRIGGER: dict[str, str] = {
    "account_access": "account_access",
    "payments_fraud": "payments_fraud",
    "security_incident": "security_incident",
    "identity_verification": "identity_verification",
    "legal": "legal",
    "pii_in_request": "pii_in_request",
    "prompt_injection": "prompt_injection",
    "out_of_scope": "out_of_scope",
}

# Priority order for hard-escalate evaluation.  When multiple flags are set
# we report the highest-priority one so the audit trail names the most
# severe cause.  ``prompt_injection`` and ``pii_in_request`` come first
# because they need to be handled before anything else.
_HARD_ESCALATE_PRIORITY: tuple[str, ...] = (
    "prompt_injection",
    "pii_in_request",
    "security_incident",
    "payments_fraud",
    "identity_verification",
    "legal",
    "account_access",
    "out_of_scope",
)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def hard_escalate_trigger(triage: Any) -> str | None:
    """Return the trigger name to escalate on, or ``None`` if no hard rule fires.

    ``triage`` is a :class:`agents.triage.TriageResult`-shaped object; we
    only read ``risk_flags`` so a synthetic/duck-typed stand-in works
    too.
    """
    flags = set(getattr(triage, "risk_flags", None) or [])
    for flag in _HARD_ESCALATE_PRIORITY:
        if flag in flags and flag in RISK_FLAG_TO_TRIGGER:
            return RISK_FLAG_TO_TRIGGER[flag]
    return None


def is_weak_grounding(
    chunks: Iterable[Any] | None,
    threshold: float = WEAK_GROUNDING_THRESHOLD,
) -> bool:
    """True when retrieval is empty or the top chunk falls below ``threshold``.

    ``chunks`` is treated as an ordered list-like (the retriever sorts by
    score desc).  Each chunk must expose a ``score`` attribute.
    """
    if not chunks:
        return True
    chunk_list = list(chunks)
    if not chunk_list:
        return True
    top = chunk_list[0]
    score = getattr(top, "score", None)
    if score is None:
        return True
    return float(score) < float(threshold)


def is_multi_request_unresolved(triage: Any) -> bool:
    """True when triage flagged a multi-part ticket we cannot fully serve.

    Two clauses must both hold:
    * ``multi_request`` was flagged by triage.
    * Two or more sub-requests were extracted AND triage did not put the
      ticket fully ``in`` scope.

    A single-sub-request ``multi_request`` flag (rare but possible from
    the LLM) doesn't trigger this path; the regular pipeline can handle
    it.
    """
    flags = set(getattr(triage, "risk_flags", None) or [])
    if "multi_request" not in flags:
        return False
    sub_requests = list(getattr(triage, "sub_requests", None) or [])
    scope = getattr(triage, "scope", None)
    return len(sub_requests) >= 2 and scope != "in"


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def assemble_output_row(
    issue: str,
    subject: str,
    company: str,
    *,
    decision: str,
    response_text: str,
    justification: str,
    product_area: str,
    request_type: str,
) -> dict:
    """Build the final CSV row dict from per-stage outputs.

    Steps (in order):

    1. Format ``Status`` via :func:`format_status`; on failure fall back
       to ``"Escalated"``.
    2. Normalize and format ``Request Type``; on failure use
       ``"invalid"``.
    3. Build the CSV-shaped dict with the seven output columns.
    4. Validate with :func:`validate_output_row`. If the row is invalid,
       coerce it via :func:`coerce_to_escalated` and drop the internal
       ``_coerced_reason`` marker.
    5. Truncate ``Response`` to :data:`RESPONSE_MAX_CHARS`.

    The ``justification`` field is intentionally NOT written to the
    output CSV — the contract is the seven columns of
    :data:`schema.OUTPUT_COLUMNS`.  The orchestrator preserves
    justification in its trace JSON instead.
    """
    # 1. Status.
    try:
        status_str = format_status(decision)
    except (ValueError, TypeError):
        status_str = format_status("escalated")

    # 2. Request type.
    try:
        request_type_str = format_request_type(normalize_request_type(request_type))
    except (ValueError, TypeError):
        request_type_str = format_request_type("invalid")

    # 3. Build the row.
    row: dict[str, Any] = {
        "Issue": issue if issue is not None else "",
        "Subject": subject if subject is not None else "",
        "Company": company if company is not None else "",
        "Response": response_text if response_text is not None else "",
        "Product Area": product_area if product_area is not None else "",
        "Status": status_str,
        "Request Type": request_type_str,
    }

    # 4. Validate; coerce on failure.
    problems = validate_output_row(row)
    if problems:
        reason = ";".join(problems)
        row = coerce_to_escalated(row, reason)
        # The _coerced_reason marker is internal; never write it to CSV.
        row.pop("_coerced_reason", None)

    # 5. Truncate Response.
    resp = row.get("Response") or ""
    if len(resp) > RESPONSE_MAX_CHARS:
        row["Response"] = resp[:RESPONSE_MAX_CHARS].rstrip()

    return row


# ---------------------------------------------------------------------------
# Self-test (run as `python code/safety.py`)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    from dataclasses import dataclass

    @dataclass
    class _T:
        risk_flags: list
        sub_requests: list
        scope: str

    @dataclass
    class _C:
        score: float

    # hard_escalate_trigger: priority order.
    t = _T(risk_flags=["account_access", "prompt_injection"], sub_requests=[], scope="in")
    assert hard_escalate_trigger(t) == "prompt_injection", hard_escalate_trigger(t)
    t2 = _T(risk_flags=["urgency"], sub_requests=[], scope="in")
    assert hard_escalate_trigger(t2) is None
    t3 = _T(risk_flags=[], sub_requests=[], scope="in")
    assert hard_escalate_trigger(t3) is None

    # is_weak_grounding.
    assert is_weak_grounding([]) is True
    assert is_weak_grounding(None) is True
    assert is_weak_grounding([_C(score=0.1)]) is True
    assert is_weak_grounding([_C(score=0.5)]) is False
    assert is_weak_grounding([_C(score=0.5)], threshold=0.6) is True

    # is_multi_request_unresolved.
    t_mr = _T(
        risk_flags=["multi_request"],
        sub_requests=["a", "b"],
        scope="partial",
    )
    assert is_multi_request_unresolved(t_mr) is True
    t_mr_in = _T(
        risk_flags=["multi_request"],
        sub_requests=["a", "b"],
        scope="in",
    )
    assert is_multi_request_unresolved(t_mr_in) is False
    t_mr_one = _T(risk_flags=["multi_request"], sub_requests=["a"], scope="partial")
    assert is_multi_request_unresolved(t_mr_one) is False

    # assemble_output_row: replied path.
    row = assemble_output_row(
        issue="hi",
        subject="subj",
        company="HackerRank",
        decision="replied",
        response_text="ok",
        justification="why",
        product_area="screen",
        request_type="product_issue",
    )
    assert row["Status"] == "Replied", row
    assert row["Request Type"] == "product_issue", row
    assert row["Response"] == "ok"
    assert row["Issue"] == "hi"
    assert "_coerced_reason" not in row

    # assemble_output_row: bad request_type forces invalid.
    row2 = assemble_output_row(
        issue="hi",
        subject="",
        company="None",
        decision="replied",
        response_text="ok",
        justification="",
        product_area="",
        request_type="???",
    )
    assert row2["Request Type"] == "invalid"

    # assemble_output_row: response truncation.
    big = "x" * (RESPONSE_MAX_CHARS + 100)
    row3 = assemble_output_row(
        issue="hi",
        subject="",
        company="HackerRank",
        decision="replied",
        response_text=big,
        justification="",
        product_area="screen",
        request_type="product_issue",
    )
    assert len(row3["Response"]) == RESPONSE_MAX_CHARS

    print("OK: safety self-test passed")


if __name__ == "__main__":
    _self_test()
