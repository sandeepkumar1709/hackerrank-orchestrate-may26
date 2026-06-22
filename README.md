# Support Triage Agent

A terminal-based AI agent that triages real support tickets across three product ecosystems — **HackerRank**, **Claude**, and **Visa** — grounded entirely in a local markdown help-center corpus. Built solo in 24 hours for the **HackerRank Orchestrate** hackathon (May 1–2, 2026).

For each ticket the agent decides whether to answer or escalate, classifies the request, and produces a cited, policy-safe response — without ever calling out to the live web for ground truth.

```
Issue, Subject, Company  →  agent  →  status, product_area, response, justification, request_type
```

---

## What it does

Given a raw ticket (`support_tickets/support_tickets.csv`), the agent:

1. **Classifies** the request (`product_issue` | `feature_request` | `bug` | `invalid`) and infers the company when it's missing.
2. **Retrieves** evidence from the local corpus (`data/{hackerrank,claude,visa}/`, ~765 help-center articles) via hybrid BM25 + dense embedding search with reciprocal rank fusion and an LLM rerank pass.
3. **Runs a safety gate** — 14 deterministic triggers (account lockout, fraud/chargeback, security incident, legal threat, weak retrieval coverage, etc.) decide whether the ticket is even eligible for a generated reply.
4. **Replies or escalates**:
   - *Reply* — a specialist agent drafts a response, every factual sentence carrying an inline `[#N]` citation back to a retrieved chunk.
   - *Escalate* — a deterministic template, no LLM call, no invented policy.
5. **Verifies** every reply with an independent judge model (different model family from the specialist, to avoid self-grading bias) before it's allowed to ship.
6. **Writes** a structured, schema-validated row to `support_tickets/output.csv`.

## Why it's built this way

- **No hallucinated policy.** The agent can only say what's in the corpus. If retrieval is weak or the topic is uncovered, it escalates instead of guessing — verified by a faithfulness judge that double-checks every citation.
- **Deterministic where it matters.** `temperature=0`, pinned embedder + corpus checksum, fixed-stopword BM25 tokenizer, stable tie-breaking sort order. Two `--dry-run` passes produce byte-identical output.
- **Resume-safe.** Output is append-with-flush, keyed by row hash — kill the process mid-run and `--resume` picks up where it left off.
- **Auditable.** Every row's full decision trail (triage → retrieval → specialist → verifier → escalation) is written to a per-row trace JSON for the judge interview.

## Architecture

```
CSV row → Preprocess → Classify → Retrieve → Safety gate → (Reply | Escalate) → Structured row → CSV
```

See [`code/ARCHITECTURE.md`](./code/ARCHITECTURE.md) for the full pipeline diagram and module-by-module design rationale, and [`code/PLAN.md`](./code/PLAN.md) for the build decision log.

| Module | Role |
| --- | --- |
| `code/corpus.py` | Walks the corpus, chunks markdown by heading, builds the searchable index. |
| `code/retriever.py` | Hybrid BM25 + BGE-small embeddings + RRF + LLM rerank. |
| `code/agents/triage.py` | Rule-first classification + risk-flag sweep + LLM merge. |
| `code/agents/specialist.py` | Grounded, citation-enforced responder. |
| `code/agents/verifier.py` | Independent faithfulness judge. |
| `code/agents/escalation.py` | Deterministic templated escalation writer. |
| `code/safety.py` | Hard-rule triggers and output coercion. |
| `code/orchestrator.py` | State machine sequencing the agents per row. |
| `code/main.py` | CLI entry point — CSV in, CSV out, resume-safe. |

## Tech stack

Python · [Nebius AI Studio](https://studio.nebius.com/) (Llama 3.3 70B for triage/specialist/rerank, Qwen3 30B as the independent verifier) · `sentence-transformers` (BGE-small local embeddings) · `rank-bm25` · `tiktoken`

## Running it

```bash
python -m pip install -r code/requirements.txt
cp .env.example .env   # set NEBIUS_API_KEY
python code/corpus.py  # one-time: build the searchable index
python code/main.py    # read support_tickets/support_tickets.csv → write output.csv
```

Full install/configure/run instructions, CLI flags, and troubleshooting are in [`code/README.md`](./code/README.md).

## Repository layout

```
.
├── code/                  # the agent (see code/README.md)
├── data/                  # local-only support corpus: hackerrank/, claude/, visa/
├── support_tickets/       # input CSVs + the agent's output.csv
├── problem_statement.md   # original hackathon task spec
└── evalutation_criteria.md# original hackathon scoring rubric
```

## Context

Built for the HackerRank Orchestrate hackathon: a 24-hour solo challenge to design, build, and ship a support-ticket triage agent, then defend the design in a live AI judge interview. `problem_statement.md` and `evalutation_criteria.md` are the original task brief and rubric provided by the organizers.
