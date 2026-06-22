# Support Triage Agent — `code/`

Terminal-based agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** corpora. Reads `support_tickets/support_tickets.csv`, writes `support_tickets/output.csv` with five answer columns (`Status`, `Product Area`, `Response`, plus the original `Issue/Subject/Company` and `Request Type`).

The agent is grounded **only** in the local markdown corpus under `data/`. No web access at runtime.

---

## Install

```bash
python -m pip install -r code/requirements.txt
```

Tested on Python 3.11+, Windows 11 and Linux. First run downloads the `BAAI/bge-small-en-v1.5` embedding model (~130 MB) once; subsequent runs reuse the cache.

## Configure

1. Copy the env template and fill in your Nebius API key:
   ```bash
   cp .env.example .env
   # edit .env, set NEBIUS_API_KEY=...
   ```
2. `code/llm.py` auto-loads `.env` via `python-dotenv` — no shell sourcing needed.

Required env: `NEBIUS_API_KEY`. Optional: `NEBIUS_BASE_URL` (default `https://api.studio.nebius.ai/v1/`), `LOG_LLM_USAGE=1` for per-call token logging.

## One-time corpus build

```bash
python code/corpus.py --print-stats
```

Walks `data/{hackerrank,claude,visa}/`, chunks markdown by heading with whole-article-when-it-fits, writes `data/index/chunks.jsonl` (~5k chunks) + `data/index/manifest.json`. Idempotent: re-runs without `--force` short-circuit on corpus-checksum match.

## Run

```bash
python code/main.py
```

Reads `support_tickets/support_tickets.csv` → drives the per-row pipeline → writes `support_tickets/output.csv` (append-with-flush, resume-safe).

Useful flags:
- `--limit N` / `--start N` — process a slice
- `--resume` — skip rows whose `row_id` is already in the output
- `--no-rerank` — disable the LLM rerank step (BM25 + dense + RRF only)
- `--no-revise` — disable the verifier-driven specialist retry
- `--dry-run` — stub mode; no API calls (proves wiring)
- `--quiet` — suppress per-row progress lines

Smoke (3 rows, no API spend):
```bash
python code/main.py --dry-run --limit 3 --output support_tickets/output.dryrun.csv
```

## Pipeline (per row)

```
ticket → triage → safety gate → retrieve (BM25 + BGE-small + RRF + LLM rerank)
       → specialist (cited [#N] response) → verifier (independent judge)
       → reply  OR  templated escalation (deterministic)
       → row written to output.csv  +  trace JSON to data/index/traces/
```

Modules (`code/`):

| File | Purpose |
| --- | --- |
| `corpus.py` | Walk + chunk + manifest. |
| `schema.py` | Closed enums + `product_area` vocab + validation. |
| `retriever.py` | Hybrid retrieval (BM25 + dense + RRF + LLM rerank). |
| `llm.py` | Tiny Nebius (OpenAI-compatible) JSON wrapper with bounded retry. |
| `agents/triage.py` | Deterministic-first triage: 7 pre-LLM rules + risk-flag regex sweep + LLM merge. |
| `agents/specialist.py` | Per-company grounded responder; strict `[#N]` citation contract. |
| `agents/verifier.py` | Independent faithfulness judge (different model family from specialist). |
| `agents/escalation.py` | 14-trigger templated escalation writer (no LLM call). |
| `safety.py` | Hard-rule triggers, weak-grounding gate, output-row coercion. |
| `orchestrator.py` | Pure-Python state machine sequencing the agents. |
| `main.py` | CSV in/out CLI; resume-safe; summary stats. |

See `code/ARCHITECTURE.md` and `code/PLAN.md` for the design rationale and decision log.

## Models (Nebius)

| Stage | Default model |
| --- | --- |
| Triage / Specialist / Rerank | `meta-llama/Llama-3.3-70B-Instruct` |
| Verifier (independent judge) | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Embedder (corpus + queries) | `BAAI/bge-small-en-v1.5` (local) |

Override per call via the `--model` / `--rerank-model` flags or the `model=` kwarg on each agent function. Find the full Nebius catalog at <https://studio.nebius.com/>.

## Outputs

- `support_tickets/output.csv` — the deliverable. 7 columns: `Issue, Subject, Company, Response, Product Area, Status, Request Type`. Status is `Replied` or `Escalated`.
- `data/index/traces/<row_id>.json` — per-row audit trail (triage / retrieval / specialist / verifier / escalation / timing). Useful for the AI Judge interview.
- `data/index/chunks.jsonl` + `manifest.json` — the searchable corpus index.
- `data/index/embeddings.npy` + `embedding_manifest.json` — cached dense vectors.

## Determinism

- `temperature=0` on every LLM call.
- BGE embedder pinned in `manifest.json` with corpus checksum.
- BM25 tokenizer is pure regex + a fixed stopword list.
- All sorts use `(score desc, chunk_id asc)`.
- Output CSV writes are append-with-flush + fsync, sorted by input row order.
- Two consecutive `--dry-run` runs produce byte-identical CSVs (verified).

## Reproducing the run

```bash
python code/corpus.py                 # build chunks + manifest (cached after first run)
python code/main.py                   # full 29-row run (~13 min wall-clock)
```

Resume-safe: kill mid-run, re-run with `--resume`.

## Troubleshooting

- `LLMError: NEBIUS_API_KEY is not set` → fill `.env`.
- `FileNotFoundError: missing manifest` → run `python code/corpus.py` first.
- `404: model does not exist` → that Nebius model id was wrong. List the catalog with the snippet in `code/llm.py` docstring or via `https://studio.nebius.com/models`.
