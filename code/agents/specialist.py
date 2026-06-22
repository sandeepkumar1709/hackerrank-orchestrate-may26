"""Stage-3 domain-specialist agent for the support-triage pipeline.

Given a triaged ticket plus a list of retrieved corpus chunks, the specialist
drafts a citation-anchored response. It MUST cite chunk anchors (`[#N]`) for
every factual sentence and refuses (sets ``insufficient_evidence=True``) when
no chunk supports the answer. Domain guidance is injected per
``triage.inferred_company`` so each company's preferred ``product_area``
vocabulary and tone is encouraged without being hard-coded into the LLM.

Pipeline
--------
1. Build a single prompt-cached system prompt (role + citation contract +
   schema + refusal rule + multi-request handling + override rule + 3
   few-shot anchors).
2. Build a per-call user message: SUBJECT, ISSUE, TRIAGE_SUMMARY,
   DOMAIN_GUIDANCE, CHUNKS (numbered ``[#1]..[#N]`` with ``(<company> |
   <title> | <heading_path>)`` headers).
3. Call Claude Sonnet 4.6 via ``llm.call_json``.
4. Post-merge: validate enums, validate citations against the supplied
   chunk count, cross-check inline anchors, force ``insufficient_evidence``
   when citations are empty, normalize ``product_area``, truncate fields.

CLI
---
``python code/agents/specialist.py --issue "..." [--subject ""] --company hackerrank``
runs the full triage -> retrieve -> specialist chain. ``--no-llm`` returns a
stub ``SpecialistResult`` (offline-friendly).
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
from schema import (  # noqa: E402
    PRODUCT_AREA_OBSERVED,
    normalize_product_area,
    normalize_request_type,
)

from agents.triage import TriageResult, triage  # noqa: E402

# ---------------------------------------------------------------------------
# Public schema
# ---------------------------------------------------------------------------

SPECIALIST_SCHEMA_KEYS: list[str] = [
    "response",
    "justification",
    "product_area",
    "citations",
    "confidence",
    "insufficient_evidence",
    "request_type_override",
]

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

RESPONSE_MAX_CHARS = 2500
JUSTIFICATION_MAX_CHARS = 280


@dataclass(frozen=True)
class SpecialistResult:
    response: str
    justification: str
    product_area: str
    citations: list[int]
    confidence: float
    insufficient_evidence: bool
    request_type_override: str | None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class _ValidationError(Exception):
    """Raised when the LLM response fails enum or shape checks."""


# ---------------------------------------------------------------------------
# Domain guidance table (one paragraph per company)
# ---------------------------------------------------------------------------

_DOMAIN_GUIDANCE: dict[str, str] = {
    "hackerrank": (
        "HackerRank questions are typically procedural; step-list answers are "
        "preferred. Preferred product_area values: screen, community."
    ),
    "claude": (
        "Claude (Anthropic) questions are typically short Q&A. Preferred "
        "product_area values: privacy, conversation_management."
    ),
    "visa": (
        "Visa questions usually involve phone numbers and an escalate-to-issuer "
        "pattern (the cardholder must contact the issuing bank). Preferred "
        "product_area values: travel_support, general_support."
    ),
    "none": (
        "No specific company was identified. Be concise; if the chunks do not "
        "clearly cover the question, prefer insufficient_evidence over guessing."
    ),
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You answer one support ticket using ONLY the numbered chunks supplied; you have no other knowledge.

## ROLE
You are a careful customer-support writer. The user will give you a ticket plus a numbered list of corpus chunks (each prefixed `[#N]`). Your job is to draft a faithful, citation-anchored response. You have no other knowledge -- if the chunks do not contain an answer, you must say so via the schema rather than guess.

## CITATION CONTRACT
- Every factual sentence in `response` MUST end with one or more `[#N]` anchors that match the supplied chunks.
- Greetings ("Hi,") and closers ("Hope this helps.") are exempt and need no anchors.
- Do not invent anchor numbers. Every `N` you cite must appear in the chunk list provided.
- Do not paste long verbatim strings from a chunk; paraphrase tightly while preserving facts.
- If multiple chunks support a sentence, list them: `[#1][#3]`.

## OUTPUT SCHEMA
Return ONLY a JSON object. No prose, no markdown fences. Exact shape:

{
  "response": "<string; the user-facing reply with [#N] anchors on factual sentences>",
  "justification": "<string <=280 chars, single line, explains which anchors carry which claims>",
  "product_area": "<snake_case_lower string; pick from the company's preferred list when possible>",
  "citations": [<int>, ...],
  "confidence": <float 0.0..1.0>,
  "insufficient_evidence": <bool>,
  "request_type_override": null | "product_issue" | "feature_request" | "bug" | "invalid"
}

`citations` is the deduplicated, sorted list of anchor numbers actually used in `response`.

## REFUSAL RULE
If NO chunk supports the answer:
- Set `insufficient_evidence` to true.
- Put a neutral one-or-two-sentence acknowledgement in `response` (no anchors, no fabricated facts).
- Set `citations` to [].
- Set `confidence` to 0.3 or lower.
- `product_area` may be empty `""` if you cannot infer it.

## MULTI-REQUEST HANDLING
If the triage summary lists more than one `sub_request`, address each in order. If the chunks only cover one of the sub-requests, set `insufficient_evidence` to true (you cannot half-answer).

## REQUEST_TYPE OVERRIDE
Set `request_type_override` to a non-null value ONLY when the corpus clearly contradicts triage's choice (rare). Otherwise leave it null.

## ANTI-HALLUCINATION NOTES
- Never reference a chunk you were not given.
- Never present uncited facts as established. If you cannot anchor a claim, drop it or refuse.
- Confidence must reflect actual chunk-coverage, not stylistic certainty.

## EXAMPLES

EXAMPLE A (HackerRank -- test active):
INPUT:
SUBJECT: (empty)
ISSUE: I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.
TRIAGE_SUMMARY: {"request_type": "product_issue", "inferred_company": "hackerrank", "scope": "in", "ambiguity": "low", "risk_flags": [], "sub_requests": []}
DOMAIN_GUIDANCE: HackerRank questions are typically procedural; step-list answers are preferred. Preferred product_area values: screen, community.
CHUNKS:
[#1] (hackerrank | Invite Candidates to a Test | )
Tests stay active indefinitely unless a start and end time are set. To configure expiration: 1) Open Active Tests, 2) Set start/end times for the test, 3) Save changes.

OUTPUT:
{"response": "Hi,\\n\\nTests stay active indefinitely unless a start and end time are set [#1]. To set expiration, open Active Tests and set start/end times [#1].", "justification": "Cited [#1] for indefinite-active rule and expiration steps.", "product_area": "screen", "citations": [1], "confidence": 0.9, "insufficient_evidence": false, "request_type_override": null}

EXAMPLE B (Claude -- delete conversation with private info):
INPUT:
SUBJECT: (empty)
ISSUE: One of my Claude conversations has private info, I forgot to make a temporary chat. Can it be deleted?
TRIAGE_SUMMARY: {"request_type": "product_issue", "inferred_company": "claude", "scope": "in", "ambiguity": "low", "risk_flags": ["pii_in_request"], "sub_requests": []}
DOMAIN_GUIDANCE: Claude (Anthropic) questions are typically short Q&A. Preferred product_area values: privacy, conversation_management.
CHUNKS:
[#1] (claude | Delete a conversation | )
To delete an individual conversation: navigate to the conversation, click the conversation name at top, select Delete.

OUTPUT:
{"response": "To delete an individual conversation: 1) Navigate to the conversation. 2) Click the conversation name at top. 3) Select Delete [#1].", "justification": "Cited [#1] which provides the delete-conversation steps.", "product_area": "privacy", "citations": [1], "confidence": 0.85, "insufficient_evidence": false, "request_type_override": null}

EXAMPLE C (Visa -- stolen traveller's cheques):
INPUT:
SUBJECT: (empty)
ISSUE: I bought Visa Traveller's Cheques from Citicorp and they were stolen in Lisbon last night. What do I do?
TRIAGE_SUMMARY: {"request_type": "product_issue", "inferred_company": "visa", "scope": "in", "ambiguity": "low", "risk_flags": ["urgency"], "sub_requests": []}
DOMAIN_GUIDANCE: Visa questions usually involve phone numbers and an escalate-to-issuer pattern (the cardholder must contact the issuing bank). Preferred product_area values: travel_support, general_support.
CHUNKS:
[#1] (visa | Visa Traveller's Cheques | Report a lost cheque)
Call Citicorp immediately at 1-800-645-6556 or +1-813-623-1709 collect Mon-Fri 6:30am-2:30pm EST. Have cheque serials and purchase details ready.
[#2] (visa | Visa Traveller's Cheques | )
If your cheques are lost or stolen, notify local police and contact the issuing bank.

OUTPUT:
{"response": "Call Citicorp immediately at 1-800-645-6556 (US toll-free) or +1-813-623-1709 (collect), Mon-Fri 6:30am-2:30pm EST [#1]. Have cheque serial numbers and purchase details ready [#1]. Notify local police in Lisbon [#2].", "justification": "[#1] gives Citicorp contact and required info; [#2] for police-notification guidance.", "product_area": "travel_support", "citations": [1, 2], "confidence": 0.8, "insufficient_evidence": false, "request_type_override": null}

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# User-message construction
# ---------------------------------------------------------------------------


def _format_triage_summary(triage_obj: TriageResult) -> str:
    """Produce the short JSON summary the prompt expects."""
    summary = {
        "request_type": triage_obj.request_type,
        "inferred_company": triage_obj.inferred_company,
        "scope": triage_obj.scope,
        "ambiguity": triage_obj.ambiguity,
        "risk_flags": list(triage_obj.risk_flags),
        "sub_requests": list(triage_obj.sub_requests),
    }
    return json.dumps(summary, sort_keys=True, ensure_ascii=False)


def _format_chunk(idx: int, chunk: RetrievalResult) -> str:
    """Render one chunk in the form the prompt expects.

    Layout::

        [#N] (<company> | <title> | <heading_path>)
        <full text>
    """
    heading = " > ".join(list(chunk.heading_path or [])) if chunk.heading_path else ""
    header = f"[#{idx}] ({chunk.company} | {chunk.title} | {heading})"
    return f"{header}\n{chunk.text}".rstrip()


def _build_user_message(
    issue: str | None,
    subject: str | None,
    triage_obj: TriageResult,
    chunks: list[RetrievalResult],
) -> str:
    """Assemble the per-call user message."""
    subj_line = subject.strip() if (subject and subject.strip()) else "(empty)"
    issue_text = (issue or "").strip()
    triage_summary = _format_triage_summary(triage_obj)
    domain_guidance = _DOMAIN_GUIDANCE.get(
        triage_obj.inferred_company, _DOMAIN_GUIDANCE["none"]
    )

    chunk_blocks: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        chunk_blocks.append(_format_chunk(i, ch))
    chunks_section = "\n\n".join(chunk_blocks) if chunk_blocks else "(no chunks)"

    return (
        f"SUBJECT: {subj_line}\n\n"
        f"ISSUE:\n{issue_text}\n\n"
        f"TRIAGE_SUMMARY: {triage_summary}\n\n"
        f"DOMAIN_GUIDANCE: {domain_guidance}\n\n"
        f"CHUNKS:\n{chunks_section}"
    )


# ---------------------------------------------------------------------------
# Post-merge helpers
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(r"\[#(\d+)\]")


def _coerce_int_list(raw: Any, max_n: int) -> list[int]:
    """Return only the ints in ``raw`` that fall in 1..max_n, deduped + sorted."""
    if not isinstance(raw, list):
        return []
    seen: set[int] = set()
    for v in raw:
        if isinstance(v, bool):
            continue  # bool is an int subclass; reject explicitly
        if isinstance(v, int):
            if 1 <= v <= max_n:
                seen.add(v)
            continue
        # Tolerate "1" / "1.0" inputs from sloppy LLMs.
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


def _strip_out_of_range_anchors(text: str, max_n: int) -> str:
    """Remove ``[#N]`` tokens whose N is not in 1..max_n."""

    def _rep(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except ValueError:
            return ""
        if 1 <= n <= max_n:
            return m.group(0)
        return ""

    return _ANCHOR_RE.sub(_rep, text)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` at the last sentence boundary.

    A trailing ``[#N]`` token (or several) on the final sentence is preserved
    so we don't strand citations.
    """
    if len(text) <= max_chars:
        return text
    # Cut window
    window = text[:max_chars]
    # Find last sentence-ending punctuation in the window.
    last = -1
    for ch in (".", "!", "?", "\n"):
        idx = window.rfind(ch)
        if idx > last:
            last = idx
    if last < 0:
        # No sentence boundary; hard cut.
        return window.rstrip()
    cut = last + 1
    # Preserve any trailing [#N] (including consecutive ones) that come right
    # after the punctuation in the original text.
    rest = text[cut:]
    m = re.match(r"\s*((?:\[#\d+\])+)", rest)
    if m:
        cut += len(m.group(0))
    return text[:cut].rstrip()


def _truncate_short(text: str, max_chars: int) -> str:
    """Collapse newlines and truncate to ``max_chars`` (used for justification)."""
    one_line = re.sub(r"\s*\n\s*", " ", text).strip()
    if len(one_line) > max_chars:
        one_line = one_line[:max_chars].rstrip()
    return one_line


def _validate_request_type_override(raw: Any) -> str | None:
    """Validate ``request_type_override``. Null/empty -> None; else normalize or raise."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip() == "":
        return None
    if not isinstance(raw, str):
        raise _ValidationError(
            f"request_type_override: expected string or null, got {type(raw).__name__}"
        )
    try:
        return normalize_request_type(raw)
    except ValueError as exc:
        raise _ValidationError(f"request_type_override: {exc}") from exc


def _fallback_product_area(company: str) -> str:
    """First (sorted) preferred product_area for ``company``, or ''."""
    observed = PRODUCT_AREA_OBSERVED.get(company) or set()
    if not observed:
        return ""
    return sorted(observed)[0]


# ---------------------------------------------------------------------------
# Post-merge
# ---------------------------------------------------------------------------


def _merge_results(
    llm_obj: dict,
    chunks: list[RetrievalResult],
    triage_obj: TriageResult,
) -> SpecialistResult:
    """Validate the LLM output, enforce invariants, and return a SpecialistResult."""
    n_chunks = len(chunks)

    # --- response ---
    raw_response = llm_obj.get("response")
    if not isinstance(raw_response, str):
        raise _ValidationError(
            f"response: expected string, got {type(raw_response).__name__}"
        )
    response = raw_response

    # --- justification ---
    raw_justification = llm_obj.get("justification") or ""
    if not isinstance(raw_justification, str):
        raise _ValidationError("justification: expected string")
    justification = _truncate_short(raw_justification, JUSTIFICATION_MAX_CHARS)

    # --- product_area ---
    raw_pa = llm_obj.get("product_area") or ""
    if not isinstance(raw_pa, str):
        raise _ValidationError("product_area: expected string")
    product_area = normalize_product_area(raw_pa)

    # --- citations ---
    raw_citations = llm_obj.get("citations")
    citations = _coerce_int_list(raw_citations, n_chunks)

    # --- confidence ---
    raw_conf = llm_obj.get("confidence", 0.0)
    if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
        raise _ValidationError(
            f"confidence: expected number, got {type(raw_conf).__name__}"
        )
    confidence = float(raw_conf)
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    # --- insufficient_evidence ---
    raw_insuf = llm_obj.get("insufficient_evidence", False)
    if not isinstance(raw_insuf, bool):
        raise _ValidationError(
            f"insufficient_evidence: expected bool, got {type(raw_insuf).__name__}"
        )
    insufficient_evidence = bool(raw_insuf)

    # --- request_type_override ---
    request_type_override = _validate_request_type_override(
        llm_obj.get("request_type_override")
    )

    # --- response anchor cross-check ---
    # Strip any out-of-range anchors from the text first.
    if n_chunks > 0:
        response = _strip_out_of_range_anchors(response, n_chunks)
    else:
        # No chunks supplied; remove every anchor token.
        response = _ANCHOR_RE.sub("", response)

    # Collect inline anchors and merge with declared citations.
    inline_anchors: set[int] = set()
    for m in _ANCHOR_RE.finditer(response):
        try:
            v = int(m.group(1))
        except ValueError:
            continue
        if 1 <= v <= n_chunks:
            inline_anchors.add(v)

    if n_chunks > 0:
        all_anchors = sorted(set(citations) | inline_anchors)
        citations = [a for a in all_anchors if 1 <= a <= n_chunks]
    else:
        citations = []

    # --- forced refusal when nothing is cited ---
    if not citations and not insufficient_evidence:
        insufficient_evidence = True
        confidence = min(confidence, 0.3)

    # --- product_area fallback ---
    if not product_area and not insufficient_evidence:
        product_area = _fallback_product_area(triage_obj.inferred_company)

    # --- response truncation (keeps trailing anchor tokens) ---
    response = _truncate_at_sentence(response, RESPONSE_MAX_CHARS)

    return SpecialistResult(
        response=response,
        justification=justification,
        product_area=product_area,
        citations=citations,
        confidence=confidence,
        insufficient_evidence=insufficient_evidence,
        request_type_override=request_type_override,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def specialist(
    issue: str,
    subject: str | None,
    triage_obj: TriageResult,
    chunks: list[RetrievalResult],
    *,
    llm_client: Any = None,
    model: str = DEFAULT_MODEL,
) -> SpecialistResult:
    """Draft a citation-anchored response for one ticket.

    Parameters
    ----------
    issue:
        The user's free-text issue.
    subject:
        Optional subject line (may be ``None`` or empty).
    triage_obj:
        The :class:`TriageResult` produced by stage-1 triage.
    chunks:
        Retrieved chunks (already top-k filtered) in the order to display.
        Numbering is positional: ``chunks[0]`` corresponds to ``[#1]``.
    llm_client, model:
        Pass-through to :func:`llm.call_json`.
    """
    # If no chunks at all, short-circuit to a refusal stub.
    if not chunks:
        return SpecialistResult(
            response=(
                "Thank you for reaching out. We don't have enough information "
                "in our knowledge base to give you a confident answer here, so "
                "we're routing this to a human agent."
            ),
            justification="No retrieved chunks; refusing to answer.",
            product_area="",
            citations=[],
            confidence=0.0,
            insufficient_evidence=True,
            request_type_override=None,
        )

    user_message = _build_user_message(issue, subject, triage_obj, chunks)

    llm_obj = call_json(
        _SYSTEM_PROMPT,
        user_message,
        SPECIALIST_SCHEMA_KEYS,
        model=model,
        max_tokens=2000,
        cache_system=True,
        client=llm_client,
    )

    return _merge_results(llm_obj, chunks, triage_obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _stub_result() -> SpecialistResult:
    """Offline-friendly stub used by ``--no-llm`` and by callers without an API key."""
    return SpecialistResult(
        response=(
            "Thank you for reaching out. We're routing this to a human agent "
            "for follow-up."
        ),
        justification="Stub: --no-llm path; insufficient evidence by construction.",
        product_area="",
        citations=[],
        confidence=0.0,
        insufficient_evidence=True,
        request_type_override=None,
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="specialist",
        description=(
            "Run the stage-3 domain-specialist agent. By default this also "
            "runs the upstream triage and retrieval stages."
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
        help="Nebius model id for the specialist call.",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking in retrieval.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip every LLM call; return a stub SpecialistResult (offline).",
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
        # Pure offline stub: don't even build the retriever (which loads a
        # heavy embedding model). The whole point of --no-llm is no IO.
        result = _stub_result()
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
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

    # Retrieval (lazy import to avoid heavy deps in the --no-llm path).
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
        result = specialist(
            issue=args.issue,
            subject=args.subject,
            triage_obj=triage_obj,
            chunks=chunks,
            model=args.model,
        )
    except LLMError as exc:
        print(f"LLMError (specialist): {exc}", file=sys.stderr)
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
        "specialist": result.to_dict(),
    }
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
