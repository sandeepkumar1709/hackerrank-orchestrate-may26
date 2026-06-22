# Build Plan — Grounded Multi-Domain Support Triage Agent

Companion to `ARCHITECTURE.md`. This plan sequences the work, names the evals that gate each step, and makes our anti-hallucination contract concrete. We update both files as decisions firm up.

---

## Build status (live)

| Phase | Status | Notes |
| --- | --- | --- |
| 0 — Domain knowledge | ✅ done | Corpus surveyed; key findings folded into corpus.py decisions (see §1 below). |
| Corpus build (`code/corpus.py`) | ✅ done | 4,987 chunks (HR 3,189 / Claude 1,705 / Visa 93). p50=231 tok, p95=480, max=480. Byte-deterministic. 4 `index.md` files skipped. Smoke test passes. |
| Schema (`code/schema.py`) | ✅ done | Closed enums (status, request_type, company); product_area observed-vocab from 10 labeled rows; normalize/format/coerce helpers; self-test passes. |
| Retrieval (`code/retriever.py`) | ✅ done (rerank pending API key) | Hybrid BM25 + BGE-small embeddings + RRF (k=60) + Sonnet 4.6 rerank. 705 LOC. Embeddings cached (`data/index/embeddings.npy`, 7.3 MB, 4987×384 float32). 3 sample queries verified BM25+dense+RRF path on Visa/Claude/HR tickets. Rerank wired (prompt-cached system prompt, JSON validation + 1 retry + RRF fallback) but `NEBIUS_API_KEY` is not set so rerank unverified end-to-end. |
| LLM wrapper (`code/llm.py`) | ✅ done (Nebius/OpenAI-compatible) | ~290 LOC. `LLMError`, lazy `get_client()` singleton (uses `openai` SDK pointed at `NEBIUS_BASE_URL`), `call_json(system, user, schema_keys)` with lenient JSON parser, one bounded retry on parse/missing-key failure, optional `LOG_LLM_USAGE` token-usage stderr line. Defaults: triage/specialist/rerank → `meta-llama/Llama-3.3-70B-Instruct`; verifier → `Qwen/Qwen2.5-72B-Instruct` (different family preserves "independent judge" property). Anthropic-specific `cache_control` removed; Nebius auto-caches prefixes on supported models. Self-check confirms missing-key error path. |
| Triage agent (`code/agents/triage.py`) | ✅ done (LLM path pending API key) | 653 LOC. Deterministic-first pipeline: 7 pre-LLM rules (empty/thanks/feature-request shortcuts, 8 risk-flag regexes for account_access/payments_fraud/security/identity/legal/PII/prompt_injection/urgency, multi-request count, out-of-scope token gate). Single LLM call with cached system prompt + 4 few-shot anchors from sample CSV. Post-merge enforces det-flag floor + request_type lock + company-input-wins. Rules-only path verified on 3 representative tickets. |
| Specialist agent (`code/agents/specialist.py`) | ✅ done (LLM path pending API key) | 692 LOC. Single shared system prompt + per-call DOMAIN_GUIDANCE block (HR/Claude/Visa/Generic). Strict citation contract (every factual sentence must end `[#N]`; uncited factual sentences forbidden). 3 few-shot anchors inline in user message. Post-merge: validates citation ints in range, cross-checks inline `[#N]` against citations array, forces insufficient_evidence when citations empty, falls back to first observed product_area for company. `--no-llm` stub path verified. |
| Verifier agent (`code/agents/verifier.py`) | ✅ done (Opus path pending API key) | 488 LOC. Independent judge using **Opus 4.7** (different from Sonnet specialist to avoid self-grading). Zero-shot scoring rubric (ok/partial/fail → accept/revise/escalate). Hard override: `draft.insufficient_evidence=True` short-circuits to `(fail, escalate)` without LLM call. Anti-rubber-stamp clause. `--no-llm` stub path verified. |
| Escalation writer (`code/agents/escalation.py`) | ✅ done (deterministic, no LLM) | 384 LOC. 14-trigger templated dispatch (account_access, payments_fraud, security_incident, identity_verification, legal, pii_in_request, prompt_injection, out_of_scope, weak_grounding, verifier_rejected, insufficient_evidence, multi_request_unresolved, validation_failure, unknown). Each trigger has neutral ≤400-char response template + ≤280-char justification template. NO LLM call by default — deterministic, byte-stable. Coarse company→product_area mapping. CLI verified for 3 reasons + invalid-trigger error path. |
| Safety rules (`code/safety.py`) | ✅ done | 309 LOC. Pure deterministic rules. `hard_escalate_trigger` (priority-ordered risk-flag→trigger map), `is_weak_grounding` (top-1 score < 0.3), `is_multi_request_unresolved`, `assemble_output_row` (validates + coerces to escalated on failure, truncates Response at 4000 chars). 8-assertion self-test passes. |
| Orchestrator (`code/orchestrator.py`) | ✅ done | 769 LOC. Pure-Python state machine: triage → hard rule → multi-request gate → retrieve → weak-grounding gate → specialist → verifier → action gate (accept / revise-once / escalate) → assemble row. Stub mode for `--dry-run`. Visa-empty-chunks fallback retries with `company=None`. Two-layer try/except: per-step exception → `validation_failure` escalation; outer wrapper guarantees process_row never raises. Per-row trace JSON written to `data/index/traces/<row_id>.json` (best-effort, swallows errors). |
| CLI (`code/main.py`) | ✅ done | 363 LOC. Reads `support_tickets/support_tickets.csv`, drives orchestrator, writes `support_tickets/output.csv`. Resume-safe (row_id = sha256(issue+subject+company)[:12]). Append-with-flush+fsync after every row. CLI: --input, --output, --limit, --start, --resume, --no-rerank, --no-revise, --dry-run, --quiet. Pre-flight checks: NEBIUS_API_KEY (exit 4), corpus index (exit 5). Dry-run produces byte-identical CSV across runs (verified by diff). |
| Real run | ✅ done | 29 rows in 5:07 (10.6 s/row avg). **Replied: 15, Escalated: 14.** Top triggers: insufficient_evidence (3), out_of_scope (3), payments_fraud (2), verifier_rejected (2), security_incident (2), account_access (1), prompt_injection (1). Spot-checked 6 rows: replied responses correctly cite [#N] anchors; escalations match templated triggers. Output written to `support_tickets/output.csv`; per-row traces under `data/index/traces/`. |
| README + requirements | ✅ done | `code/README.md` (install/configure/run/troubleshooting) + `code/requirements.txt` (pinned deps). |
| Submission package | ⏳ pending | Zip `code/` for upload. |
| 4 — Guardrails | woven through | Boundary checks at every stage. |
| 5 — Iterate | ⏳ pending | Eval-driven fixes. |
| 6 — Final + package | ⏳ pending | Run on `support_tickets.csv`, write README. |

---

## 0. Non-negotiables (these drive every later choice)

1. **No claim without a citation.** Every sentence in `response` must trace to one or more retrieved chunks. The responder is forbidden from drawing on parametric knowledge.
2. **Eval before tune.** We do not improve a component without a measurement that says it got better. The eval harness is built **second**, before the agent itself.
3. **Escalate when grounding is weak.** Low retrieval score, contradictory chunks, or out-of-corpus topics → `escalated`, never a guess.
4. **Determinism.** `temperature=0`, fixed seeds, sorted I/O. Re-running on the same CSV produces the same `output.csv`.
5. **Evidence in the transcript.** Every architectural change goes through a logged turn that explains *why*, so the AI Judge interview and AI Fluency score have a paper trail.

---

## 1. Phase 0 — Domain knowledge ✅ done

We can't classify what we don't understand. We surveyed the territory before any code; key findings drove the corpus.py decisions.

### 1.1 Corpus shape — findings

| Company | Articles | ~Tokens | Avg/article | Notes |
| --- | --- | --- | --- | --- |
| HackerRank | 434 (4 index files skipped) | ~982k | ~2.2k | YAML frontmatter; **uses repeated H1s as section breaks** (frontmatter `title` is the join of those H1s). Rich help center. |
| Claude | 317 (1 index skipped) | ~449k | ~1.4k | Lighter Q&A style with frontmatter; many articles fit whole under cap. |
| Visa | 14 (1 index skipped) | ~13k | ~950 | Hierarchical tree; `support.md` is content-rich and large; coverage thin → escalation thresholds will need to lean per-company. |

**Information unit:** heading section for HackerRank (forced by H1-section quirk above), whole article when it fits the token cap for Claude/Visa. This drove the "whole-article-when-it-fits, split-on-`#{1,3}` otherwise" rule in `corpus.py`.

### 1.2 Label vocabulary — pending extraction (Phase 1 task)

To be derived from `support_tickets/sample_support_tickets.csv` (108 labeled rows). Will be loaded by `schema.py` at module import. Closed enums: `status`, `request_type`. Closed-per-company vocab: `product_area`.

### 1.3 Failure-mode catalog — pending (Phase 1 task)

Will be tagged on a manual read of the dev split before the eval harness fires.

### Phase-0 artifacts

- `data/index/chunks.jsonl` — 4,987 chunks, schema `{id, company, path, title, breadcrumb_path, source_url, heading_path, text, n_tokens}`.
- `data/index/manifest.json` — pinned: `tokenizer_name=BAAI/bge-small-en-v1.5`, `chunker_version=1.0.0`, `corpus_checksum`, per-company counts, skipped files.
- `code/corpus.py` (788 LOC) + `code/corpus_test.py` (167 LOC).

---

## 2. Phase 1 — Eval harness first (~45 min)

If we build the agent before the eval, we tune by vibes. Build the meter first.

### 2.1 Splits

`sample_support_tickets.csv` → **dev** (80%) and **holdout** (20%, untouched until the very end). We tune on dev. We report dev score in commit messages, holdout only once at the end.

### 2.2 Metrics

| Metric | What it measures | Why |
| --- | --- | --- |
| `status_acc` | exact match `replied/escalated` | the highest-leverage axis — escalation is asymmetric in cost |
| `escalate_precision` / `escalate_recall` | among escalations | flags both "trigger-happy" and "reckless" agents |
| `request_type_acc` | exact match | enum classification |
| `product_area_acc` | exact match against vocabulary | strict, since vocabulary is closed |
| `response_faithfulness` | LLM-as-judge: are all factual claims supported by cited chunks? | the anti-hallucination probe |
| `response_helpfulness` | LLM-as-judge vs. expected response | quality, not just safety |
| `justification_quality` | LLM-as-judge: does justification name the trigger or the citations? | rubric explicitly checks this |

### 2.3 Implementation

`code/eval.py` (CLI: `python -m code.eval --pred output.csv --gold sample_support_tickets.csv`):

- Deterministic exact-match scoring for the four enum/vocab columns.
- LLM-judge calls for `response` and `justification`, run with a *separate* judge prompt and a *different* model invocation (Opus 4.7 as judge over Sonnet 4.6 outputs) to avoid self-grading bias.
- Confusion matrices printed per company.
- Per-row diff dump so we can read the worst 20 cases by hand.

**Gate to Phase 2:** `eval.py` runs end-to-end on a stub agent that always escalates, producing baseline numbers.

---

## 3. Phase 2 — Retrieval that doesn't suck (~1.5 h)

Retrieval is the single biggest lever on faithfulness. Bad retrieval → either hallucination or over-escalation.

### 3.1 Chunking ✅ implemented in `code/corpus.py`

- Walk the three trees once; sorted POSIX traversal for determinism.
- Skip `index.md` files at any level; skip files whose body tokenizes to < 20 tokens.
- Whole-article-when-it-fits: if `len(prefix + body)` ≤ HARD_CAP, emit one chunk with empty `heading_path`. Otherwise split on `^#{1,3}` (H1/H2/H3); the first H1 matching the article title is consumed as the title, not a section break.
- Hard cap **480 tokens** (BGE-small's 512 ceiling minus headroom for the prefix); soft target 300–400; floor 80 (merge sibling-into-next when below floor).
- Prefix every chunk text with `<title> > <heading_path>\n\n` for self-containment.
- Code fences and pipe tables are atomic — never split mid-block.
- Tokenizer: **`BAAI/bge-small-en-v1.5`** (HuggingFace `transformers`). Pinned in `manifest.json`. Used both for chunk sizing and as the embedder downstream — same tokenizer end-to-end avoids silent truncation.
- Persist `data/index/chunks.jsonl`: `{id, company, path, title, breadcrumb_path, source_url, heading_path, text, n_tokens}` (9 keys, all required).

### 3.2 Hybrid retrieval — pending

- **BM25** (sparse, keyword) — `rank_bm25` over tokenized chunks.
- **Embeddings** (dense, semantic) — local `BAAI/bge-small-en-v1.5` (384-dim; same tokenizer as chunking, so chunk lengths are guaranteed under the embedder's context window). Cached to `data/index/embeddings.npy`.
- **Fusion** — Reciprocal Rank Fusion (RRF, k=60) over both rankings. Top 20 fused candidates.
- **Rerank** — single Sonnet 4.6 call scoring each candidate's relevance as JSON. Keep top 5; drop anything below 0.3 → escalate path.
- Per-company filter when `Company` is known; otherwise search global with company as a feature for the reranker.

### 3.3 Query rewriting — deferred to v2

Original plan called for 1–2 paraphrases via Sonnet. Deferred: with 4,987 chunks and the corpus being topical/keyword-rich, the marginal recall gain may not justify the extra LLM call per row. Eval first; add if recall@5 is below the gate.

### 3.4 Retrieval evals (run BEFORE agent code)

`code/eval_retrieval.py` measures **recall@k** against a hand-labeled set of 20 sample tickets where we manually flag the right article(s). This is small but invaluable — it tells us whether retrieval is the bottleneck before we blame the LLM.

**Gate to Phase 3:** recall@5 ≥ 0.85 on the 20-row probe. If not, fix retrieval before moving on.

---

## 4. Phase 3 — Subagent topology (~2 h)

We use focused, single-purpose Claude API calls — each with its own system prompt, prompt-cached, and a narrow output schema. This keeps each prompt short, reviewable, and individually evaluable.

```
                   ┌──────────────────┐
                   │  Orchestrator    │  (deterministic Python, not an LLM)
                   │  (main.py)       │
                   └─────────┬────────┘
                             │
   ┌────────┬────────────────┼─────────────────┬──────────┐
   ▼        ▼                ▼                 ▼          ▼
┌────────┐┌──────────┐┌─────────────────┐┌──────────┐┌─────────┐
│Triage  ││Retrieval ││Domain Specialist││Verifier  ││Escalation│
│Agent   ││Agent     ││  (one of three) ││ (Judge)  ││ Writer  │
│        ││(rerank + ││  HR | Claude |  ││          ││         │
│        ││expand)   ││  Visa | Generic ││          ││         │
└────────┘└──────────┘└─────────────────┘└──────────┘└─────────┘
```

### 4.1 Triage Agent

- Inputs: raw `Issue, Subject, Company`.
- Output (JSON): `{request_type, inferred_company, sub_requests[], risk_flags[], scope: in|partial|out, ambiguity: low|med|high}`.
- Few-shot anchors from Phase 0.
- This is deterministic-first: regex/keyword rules pre-fill obvious risk flags before the LLM call, and the LLM can only escalate confidence, not override hard flags.

### 4.2 Retrieval Agent (a thin wrapper)

- Pure Python pipeline from §3, but exposed as a single callable.
- Returns top-k chunks with provenance and per-chunk relevance scores.
- Records retrieval metadata into the per-row trace (used by `eval.py`).

### 4.3 Domain Specialist (3 + 1)

One per company plus a `Generic` fallback. Each has a system prompt that:

- Names its domain ("You answer **only** HackerRank platform questions…").
- Lists the `product_area` vocabulary for that domain (closed set).
- Pastes the retrieved chunks verbatim with `[#1]`-style anchors.
- Demands the response cite anchors inline (e.g. `…tests stay active indefinitely [#2]…`).
- Refuses if no anchor supports the claim — must instead emit an `insufficient_evidence` flag for the orchestrator.

Output schema: `{response, justification, citations: [chunk_ids], product_area, confidence: 0..1, insufficient_evidence: bool}`.

### 4.4 Verifier (Judge) Agent

- Inputs: ticket, retrieved chunks, specialist's draft response.
- Output: `{faithfulness: ok|partial|fail, unsupported_claims: [...], suggested_action: accept|revise|escalate}`.
- Different model from the specialist (Opus 4.7 over Sonnet 4.6) — independent reviewer, not self-grader.
- If `fail` → orchestrator forces `escalated`. If `revise` → one retry with the verifier's notes appended.

### 4.5 Escalation Writer

- Triggered when triage hard-flags, retrieval is too weak, or the verifier rejects.
- Output: short, neutral acknowledgement that does **not** assert any policy. Cites the triggering reason in `justification`.

### 4.6 Orchestrator (the only "agent" without an LLM)

- Pure Python. Sequences the calls. Owns the state machine. Owns determinism.
- One trace JSON per row written to `data/index/traces/<row_id>.json` (gitignored) for debugging and for the AI Judge interview.

**Gate to Phase 4:** end-to-end run on 10 dev rows, manual inspection of traces, no obvious hallucinations.

---

## 5. Phase 4 — Guardrails (~1 h, woven through)

Guardrails are not a single component; they're checks at every boundary.

| Boundary | Guardrail |
| --- | --- |
| **Input** | Strip control chars; flag prompt-injection patterns ("ignore previous instructions"); cap input length; redact obvious PII before sending to LLM (logged copy as `[REDACTED]`). |
| **Triage output** | Validate enum values; if invalid, retry once, then escalate. |
| **Retrieval** | Top-1 score below threshold → mark `weak_grounding`. |
| **Specialist output** | JSON schema validation; check that every cited anchor exists in the supplied chunks; check that response references at least one anchor unless escalating. |
| **Verifier** | Independent faithfulness check (above). |
| **Final write** | Re-validate row against `schema.py` enums and `product_area` vocabulary; on any failure → coerce to `escalated` with `justification = "validation failure: <reason>"`. |
| **Privacy** | Never log raw API keys or full PII; `log.txt` redaction is enforced (already in AGENTS.md §5.4). |

---

## 6. Phase 5 — Iterate against eval (~2 h)

This is where most of the *quality* comes from. Loop:

1. Run agent on dev split.
2. Run `eval.py`.
3. Read the worst 20 rows by hand.
4. Identify the single biggest failure mode (retrieval, classification, escalation policy, response style).
5. Fix only that. Re-run.
6. Stop when marginal gains drop or time budget says stop.

Track each iteration in `code/notes/eval_runs.md` with date, scores, change made, scores after.

---

## 7. Phase 6 — Final run + packaging (~30 min)

- Re-run on `support_tickets.csv` end-to-end.
- Sanity-check `output.csv` schema with `eval.py --schema-only`.
- Run on the holdout split once; record the number; do not tune.
- `code/README.md`: install (`pip install -r requirements.txt`), env vars (`NEBIUS_API_KEY`), run command, design summary, eval reproduction steps.
- Final `log.txt` review for completeness.

---

## 8. Time budget (against ~10h remaining)

| Phase | Budget | Cumulative |
| --- | --- | --- |
| 0 — Domain knowledge | 0:45 | 0:45 |
| 1 — Eval harness | 0:45 | 1:30 |
| 2 — Retrieval | 1:30 | 3:00 |
| 3 — Subagents | 2:00 | 5:00 |
| 4 — Guardrails | woven in | 5:00 |
| 5 — Iterate | 2:00 | 7:00 |
| 6 — Final + package | 0:30 | 7:30 |
| Buffer | 2:30 | 10:00 |

Buffer is real, not optimistic. Hackathons miss when buffers are zero.

---

## 9. What we will NOT do

- **No frameworks** (LangChain/LlamaIndex). Hidden control flow makes the AI Judge interview hard. Plain Python + `openai` SDK pointed at Nebius.
- **No fine-tuning.** Out of scope for 10 hours and the spec.
- **No new training data.** Corpus-only is a hard rule.
- **No multi-turn agents looping until they "feel done".** Bounded steps with explicit gates. Loops are bug factories.
- **No clever auto-retry on bad LLM JSON beyond one structured retry.** Second failure → escalate. Resilience over cleverness.

---

## 10. Architecture deltas (sync to ARCHITECTURE.md)

The following decisions in this plan **supersede or extend** ARCHITECTURE.md:

- Retrieval is **hybrid (BM25 + embeddings + RRF + LLM rerank)**, not BM25-only. (Was deferred; promoted to default because faithfulness depends on it.)
- A **Verifier (Judge) Agent** is added between the specialist and the writer. (New.)
- **Per-company Domain Specialist agents** replace the single responder. (Refinement of `responder.py`.)
- **Eval harness comes before the agent code**, not after. (Order change.)
- **Determinism** explicitly includes "pin the embedding model + checksum the index" beyond just `temperature=0`.

`ARCHITECTURE.md` will be updated to point to these sections so the two files don't drift.

---

## 11. Open questions to confirm before we start

1. Embedding choice: hosted (network call, costs) **or** local `sentence-transformers` (free, slower first run)? Decision: **local `BAAI/bge-small-en-v1.5`** for determinism + zero ongoing cost.
2. Concurrency: sequential is simplest. Parallelize per-row only if eval shows wall-clock is the constraint. Confirm OK to start sequential.
3. Should `code/notes/` be committed (audit trail for AI Judge) or gitignored (cleaner submission)? Default: **commit `domain.md` and `eval_runs.md`**, gitignore raw traces. Confirm.

Once you answer those three (or say "your call"), I start with Phase 0.
