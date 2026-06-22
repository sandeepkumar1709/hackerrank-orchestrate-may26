"""Stage-4 faithfulness verifier for the support-triage pipeline.

The verifier is a *separate* judge that did NOT write the specialist's draft.
It receives the original ticket, the same numbered chunk list the specialist
saw, and the specialist's draft response, and produces a rubric-anchored
faithfulness verdict::

    faithfulness        : "ok" | "partial" | "fail"
    suggested_action    : "accept" | "revise" | "escalate"
    unsupported_claims  : list[str]
    missing_citations   : list[int]
    verifier_notes      : str (<=280 chars)

We use a different model family (default: ``Qwen/Qwen3-30B-A3B-Instruct-2507``)
than the specialist (``meta-llama/Llama-3.3-70B-Instruct``) so the judge
is independent.

Hard rule: if the specialist's draft already raised ``insufficient_evidence``,
the verifier's response is forced to ``("fail", "escalate")`` regardless of
what the LLM returns. The orchestrator should not be able to re-accept a
self-flagged refusal.

CLI
---
``python code/agents/verifier.py --issue "..." --company hackerrank`` runs the
full triage -> retrieve -> specialist -> verifier chain. ``--no-llm`` shortcuts
to a stub (offline-friendly).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make ``code/`` importable when this file is run as a script.
_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from llm import LLMError, call_json  # noqa: E402
from retriever import RetrievalResult  # noqa: E402

from agents.specialist import (  # noqa: E402
    SpecialistResult,
    _format_chunk,
    specialist,
    _stub_result as _specialist_stub,
)
from agents.triage import triage  # noqa: E402

# ---------------------------------------------------------------------------
# Public schema
# ---------------------------------------------------------------------------

VERIFIER_SCHEMA_KEYS: list[str] = [
    "faithfulness",
    "unsupported_claims",
    "missing_citations",
    "suggested_action",
    "verifier_notes",
]

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

_FAITHFULNESS_VOCAB: tuple[str, ...] = ("ok", "partial", "fail")
_ACTION_VOCAB: tuple[str, ...] = ("accept", "revise", "escalate")

UNSUPPORTED_CLAIM_MAX_CHARS = 200
NOTES_MAX_CHARS = 280


@dataclass(frozen=True)
class VerifierResult:
    faithfulness: str  # "ok" | "partial" | "fail"
    unsupported_claims: list[str]
    missing_citations: list[int]
    suggested_action: str  # "accept" | "revise" | "escalate"
    verifier_notes: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an independent faithfulness judge. You did NOT write the draft below -- your job is to decide whether each factual claim in it is anchored in the supplied chunks. You cannot use outside knowledge.

## INPUTS
You will be given:
- ISSUE: the user's original support ticket
- CHUNKS: a numbered corpus list (`[#N]`) the specialist used as evidence
- DRAFT_RESPONSE: the specialist's draft reply (may include `[#N]` anchors)
- DRAFT_CITATIONS: the list of chunk numbers the specialist declared
- DRAFT_INSUFFICIENT_EVIDENCE: whether the specialist refused for lack of evidence

## SCORING RUBRIC (faithfulness)
- "ok" -- every factual claim in DRAFT_RESPONSE has at least one supporting chunk AND every cited chunk supports the claim it is attached to. The answer is on-topic and complete.
- "partial" -- at most one minor unsupported phrase OR at most one weak citation, but the core answer holds. Use this when the gist is right but a sentence drifts beyond the chunks.
- "fail" -- any major claim is unsupported, contradicted by a chunk, fabricated, or the answer is off-topic / asks the user to do something not in the chunks.

## ACTION MAPPING
- "ok" -> "accept"
- "partial" -> "revise"
- "fail" -> "escalate"

HARD RULE: if DRAFT_INSUFFICIENT_EVIDENCE is true, return faithfulness="fail" and suggested_action="escalate". The specialist already flagged itself.

## ANTI-RUBBER-STAMP
Do NOT default to "ok". When in doubt, prefer "partial". A rubber-stamp judge is worse than no judge -- your value is in catching unsupported claims, so be skeptical.

## OUTPUT SCHEMA
Return ONLY a JSON object. No prose, no markdown fences. Exact shape:

{
  "faithfulness": "ok" | "partial" | "fail",
  "unsupported_claims": [<string>, ...],
  "missing_citations": [<int>, ...],
  "suggested_action": "accept" | "revise" | "escalate",
  "verifier_notes": "<string <=280 chars, single line>"
}

Field semantics:
- `unsupported_claims`: short paraphrases (<=200 chars each) of any sentences in DRAFT_RESPONSE that are not anchored in the chunks. Empty list when faithfulness is "ok".
- `missing_citations`: chunk numbers (1..len(CHUNKS)) that the specialist should have cited but did not. Empty list when none.
- `verifier_notes`: one or two sentences explaining the verdict. No newlines.

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# User-message construction
# ---------------------------------------------------------------------------


def _build_user_message(
    issue: str | None,
    subject: str | None,
    chunks: list[RetrievalResult],
    draft: SpecialistResult,
) -> str:
    """Assemble the per-call user message."""
    subj_line = subject.strip() if (subject and subject.strip()) else "(empty)"
    issue_text = (issue or "").strip()

    chunk_blocks = [_format_chunk(i, ch) for i, ch in enumerate(chunks, start=1)]
    chunks_section = "\n\n".join(chunk_blocks) if chunk_blocks else "(no chunks)"

    draft_citations = json.dumps(list(draft.citations), ensure_ascii=False)

    return (
        f"SUBJECT: {subj_line}\n\n"
        f"ISSUE:\n{issue_text}\n\n"
        f"CHUNKS:\n{chunks_section}\n\n"
        f"DRAFT_RESPONSE:\n{draft.response}\n\n"
        f"DRAFT_CITATIONS: {draft_citations}\n\n"
        f"DRAFT_INSUFFICIENT_EVIDENCE: "
        f"{'true' if draft.insufficient_evidence else 'false'}"
    )


# ---------------------------------------------------------------------------
# Validation / merge
# ---------------------------------------------------------------------------


class _ValidationError(Exception):
    """Raised on enum/shape failures from the LLM."""


def _validate_enum(value: Any, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in allowed else None


def _coerce_int_list(raw: Any, max_n: int) -> list[int]:
    """Same idea as specialist._coerce_int_list -- keep ints in 1..max_n."""
    if not isinstance(raw, list):
        return []
    seen: set[int] = set()
    for v in raw:
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            if 1 <= v <= max_n:
                seen.add(v)
            continue
        if isinstance(v, str):
            try:
                vi = int(v.strip())
            except ValueError:
                continue
            if 1 <= vi <= max_n:
                seen.add(vi)
        elif isinstance(v, float):
            vi = int(v)
            if vi == v and 1 <= vi <= max_n:
                seen.add(vi)
    return sorted(seen)


def _coerce_string_list(
    raw: Any, *, max_chars: int, cap: int = 20
) -> list[str]:
    """Trim, dedupe (preserving order), truncate, cap length."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        s = re.sub(r"\s+", " ", item).strip()
        if not s:
            continue
        if len(s) > max_chars:
            s = s[:max_chars].rstrip()
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _truncate_short(text: str, max_chars: int) -> str:
    one_line = re.sub(r"\s*\n\s*", " ", text or "").strip()
    if len(one_line) > max_chars:
        one_line = one_line[:max_chars].rstrip()
    return one_line


def _merge_results(
    llm_obj: dict,
    chunks: list[RetrievalResult],
    draft: SpecialistResult,
) -> VerifierResult:
    """Validate the LLM output and apply consistency overrides."""
    n_chunks = len(chunks)

    # 1. Enums (with safe coercion).
    faithfulness = _validate_enum(llm_obj.get("faithfulness"), _FAITHFULNESS_VOCAB)
    suggested_action = _validate_enum(llm_obj.get("suggested_action"), _ACTION_VOCAB)
    if faithfulness is None or suggested_action is None:
        # Coerce to the safest verdict.
        faithfulness = "fail"
        suggested_action = "escalate"

    # 2. Lists.
    unsupported_claims = _coerce_string_list(
        llm_obj.get("unsupported_claims"),
        max_chars=UNSUPPORTED_CLAIM_MAX_CHARS,
    )
    missing_citations = _coerce_int_list(llm_obj.get("missing_citations"), n_chunks)

    # 3. Notes.
    raw_notes = llm_obj.get("verifier_notes") or ""
    if not isinstance(raw_notes, str):
        raw_notes = ""
    verifier_notes = _truncate_short(raw_notes, NOTES_MAX_CHARS)

    # 4. Consistency rules.
    # If faithfulness is "fail" but action is "accept", upgrade to escalate.
    if faithfulness == "fail" and suggested_action == "accept":
        suggested_action = "escalate"
    # If faithfulness is "ok" but action is "escalate", trust the action
    # (judge sees something the rubric missed).
    # No coercion in that direction.

    # 5. HARD OVERRIDE: specialist self-flagged refusal -> always escalate.
    if draft.insufficient_evidence:
        faithfulness = "fail"
        suggested_action = "escalate"

    return VerifierResult(
        faithfulness=faithfulness,
        unsupported_claims=unsupported_claims,
        missing_citations=missing_citations,
        suggested_action=suggested_action,
        verifier_notes=verifier_notes,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def verify(
    issue: str,
    subject: str | None,
    chunks: list[RetrievalResult],
    draft: SpecialistResult,
    *,
    llm_client: Any = None,
    model: str = DEFAULT_MODEL,
) -> VerifierResult:
    """Independently judge the specialist's draft response.

    Parameters
    ----------
    issue, subject:
        The original ticket.
    chunks:
        The same chunk list the specialist saw (positional anchoring is
        preserved).
    draft:
        The :class:`SpecialistResult` to evaluate.
    llm_client, model:
        Pass-through to :func:`llm.call_json`.
    """
    # Short-circuit: if the specialist already refused, don't pay for an LLM call.
    if draft.insufficient_evidence:
        return VerifierResult(
            faithfulness="fail",
            unsupported_claims=[],
            missing_citations=[],
            suggested_action="escalate",
            verifier_notes=(
                "Specialist self-flagged insufficient_evidence; escalate by rule."
            ),
        )

    user_message = _build_user_message(issue, subject, chunks, draft)

    llm_obj = call_json(
        _SYSTEM_PROMPT,
        user_message,
        VERIFIER_SCHEMA_KEYS,
        model=model,
        max_tokens=1500,
        cache_system=True,
        client=llm_client,
    )

    return _merge_results(llm_obj, chunks, draft)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _stub_result() -> VerifierResult:
    """Offline stub -- paired with the specialist stub it always escalates."""
    return VerifierResult(
        faithfulness="fail",
        unsupported_claims=[],
        missing_citations=[],
        suggested_action="escalate",
        verifier_notes="Stub: --no-llm path; specialist had insufficient evidence.",
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verifier",
        description=(
            "Run the stage-4 faithfulness verifier. By default this also runs "
            "the upstream triage, retrieval, and specialist stages."
        ),
    )
    p.add_argument("--issue", required=True, help="The user's issue text.")
    p.add_argument("--subject", default="", help="Optional subject line.")
    p.add_argument(
        "--company",
        default="none",
        help="Optional company hint (hackerrank|claude|visa|none).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved chunks to use (default: 5).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Nebius model id for the verifier call.",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking in retrieval.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip every LLM call; return a stub VerifierResult (offline).",
    )
    p.add_argument(
        "--index-dir",
        default="data/index",
        help="Path to the retriever index dir (default: data/index).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.no_llm:
        # Pair the stub specialist with the stub verifier so the CLI exercises
        # the same shape an orchestrator would see end-to-end.
        spec_stub = _specialist_stub()
        ver_stub = _stub_result()
        out = {
            "specialist": spec_stub.to_dict(),
            "verifier": ver_stub.to_dict(),
        }
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    # Full chain.
    try:
        triage_obj = triage(
            issue=args.issue,
            subject=args.subject,
            company=args.company,
        )
    except LLMError as exc:
        print(f"LLMError (triage): {exc}", file=sys.stderr)
        return 2

    from retriever import Retriever  # noqa: E402

    try:
        retriever = Retriever(
            index_dir=args.index_dir,
            enable_rerank=not args.no_rerank,
        )
    except FileNotFoundError as exc:
        print(f"RetrievalError: {exc}", file=sys.stderr)
        return 4

    company_for_filter = (
        triage_obj.inferred_company
        if triage_obj.inferred_company != "none"
        else None
    )
    chunks = retriever.retrieve(args.issue, company=company_for_filter, k=args.k)

    try:
        spec_result = specialist(
            issue=args.issue,
            subject=args.subject,
            triage_obj=triage_obj,
            chunks=chunks,
        )
        verifier_result = verify(
            issue=args.issue,
            subject=args.subject,
            chunks=chunks,
            draft=spec_result,
            model=args.model,
        )
    except LLMError as exc:
        print(f"LLMError: {exc}", file=sys.stderr)
        if exc.raw:
            preview = exc.raw[:300].replace("\n", " ")
            print(f"  raw preview: {preview!r}", file=sys.stderr)
        return 2
    except _ValidationError as exc:
        print(f"ValidationError: {exc}", file=sys.stderr)
        return 3

    out = {
        "triage": triage_obj.to_dict(),
        "n_chunks": len(chunks),
        "specialist": spec_result.to_dict(),
        "verifier": verifier_result.to_dict(),
    }
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
