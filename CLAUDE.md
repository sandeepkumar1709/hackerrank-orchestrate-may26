# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

`AGENTS.md` is the authoritative source for session rules: onboarding gate, the mandatory append-only log at `%USERPROFILE%\hackerrank_orchestrate\log.txt` (Windows) / `$HOME/hackerrank_orchestrate/log.txt` (Unix), per-turn log format, and the evaluable-submission contract. Read it in full before acting. The notes below add repo-specific context that AGENTS.md does not cover.

## What this repo is for

Starter for the **HackerRank Orchestrate** 24-hour hackathon (May 1–2, 2026). The deliverable is a terminal-based agent that, for each row in `support_tickets/support_tickets.csv`, writes five columns to `support_tickets/output.csv`: `status` (`replied`|`escalated`), `product_area`, `response`, `justification`, `request_type` (`product_issue`|`feature_request`|`bug`|`invalid`).

See `problem_statement.md` for the full I/O schema and `evalutation_criteria.md` for the scoring rubric (Agent Design, AI Judge Interview, Output CSV accuracy, AI Fluency from `log.txt`).

## Big-picture architecture

The repo ships only a skeleton — there is no agent code yet. `code/main.py` is empty; you build the agent inside `code/`. The architecture the evaluator expects (per `evalutation_criteria.md` §1) is a clear separation of:

- **Retrieval** over the local corpus in `data/` — three siblings: `data/hackerrank/`, `data/claude/`, `data/visa/`. Each is a tree of markdown FAQ articles exported from the respective help center; filenames are `<id>-<slug>.md`. The corpus is local-only and must be the sole grounding source — no live web fetches for ground-truth answers.
- **Routing/classification** — pick `product_area` and `request_type`; route by the `Company` column (`HackerRank`|`Claude`|`Visa`|`None`), inferring from content when `None`.
- **Reasoning + escalation policy** — high-risk, sensitive, or out-of-scope tickets must `escalate` rather than guess. Hallucinated policies are penalized.
- **Structured output** — write rows to `support_tickets/output.csv` with exactly the five required columns and allowed values.

Inputs you'll work against:
- `support_tickets/sample_support_tickets.csv` — has expected outputs; use for development and self-eval. Columns: `Issue, Subject, Company, Response, Product Area, Status, Request Type`.
- `support_tickets/support_tickets.csv` — inputs only (`Issue, Subject, Company`); this is the file your final agent must score.

A row may contain multiple requests, irrelevant/misleading text, or a blank/`None` company. Plan for noisy inputs.

## Conventions and constraints

- All your code lives in `code/`. Add modules as needed (`agent.py`, `retriever.py`, etc.) and keep a `code/README.md` documenting install + run commands — the evaluator reads it.
- Read secrets from env vars only (`NEBIUS_API_KEY`, …). Copy `.env.example` → `.env` (gitignored). `code/llm.py` auto-loads `.env` via python-dotenv.
- Be deterministic where possible — seed sampling; pin dependencies.
- The submission zip should include `code/` only — exclude virtualenvs, `node_modules`, build artifacts, and the `data/` + `support_tickets/` CSVs (those are provided by the evaluator).
- `.gitignore` already excludes `data/index/` and `data/embeddings/` — put any built indices/vectors there so they don't get committed.

## Commands

There is no build/lint/test scaffolding yet — language and tooling are the participant's choice (Python, JS, or TS recommended). Once you create `code/`, add the actual run command to `code/README.md`. Conventional invocation will be something like:

```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

Adjust the entry point as you build it out.
