"""Main CLI entry point for the support-triage agent.

Reads the input ticket CSV, runs each row through the
:class:`Orchestrator`, and writes a 7-column output CSV that conforms
to :data:`schema.OUTPUT_COLUMNS`.

Examples
--------
Real run (requires ``NEBIUS_API_KEY``)::

    python code/main.py

Smoke test without spending tokens::

    python code/main.py --dry-run --limit 3 \\
        --output support_tickets/output.dryrun.csv

Resume an interrupted run (skips rows whose row_id is already in the
output file)::

    python code/main.py --resume

Exit codes
----------
* 0 — success
* 1 — catastrophic crash
* 4 — missing ``NEBIUS_API_KEY`` in real (non-dry-run) mode
* 5 — missing index dir; user must run ``python code/corpus.py`` first
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Make ``code/`` importable when this file is run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import safety  # noqa: E402
from orchestrator import Orchestrator, compute_row_id  # noqa: E402
from schema import OUTPUT_COLUMNS  # noqa: E402

REPO_ROOT = _HERE.parent

DEFAULT_INPUT = "support_tickets/support_tickets.csv"
DEFAULT_OUTPUT = "support_tickets/output.csv"
DEFAULT_INDEX_DIR = "data/index"
DEFAULT_TRACE_DIR = "data/index/traces"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="support-triage",
        description="Run the support-triage agent on a CSV of tickets.",
    )
    p.add_argument(
        "--input", default=DEFAULT_INPUT, help=f"Input CSV (default: {DEFAULT_INPUT})"
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    p.add_argument(
        "--start", type=int, default=0, help="Skip the first N input rows (default: 0)."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose row_id is already in the output CSV.",
    )
    p.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Path to the retriever index dir (default: {DEFAULT_INDEX_DIR}).",
    )
    p.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=f"Path to the per-row trace directory (default: {DEFAULT_TRACE_DIR}).",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking inside the retriever.",
    )
    p.add_argument(
        "--no-revise",
        action="store_true",
        help="Disable the verifier-driven specialist revise loop.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the API key check and the retriever; emit canned 'Replied' "
            "rows. Useful for smoke testing the wiring."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress per-row progress output."
    )
    return p


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_path(raw: str) -> Path:
    """Resolve ``raw`` to an absolute path, anchored at the repo root.

    Repo root is the parent of the ``code/`` directory.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_resume_set(output_path: Path) -> set[str]:
    """Read existing output CSV and return the set of completed row_ids.

    Each output row's ``Issue/Subject/Company`` triple is hashed back to
    its row_id via :func:`compute_row_id`; that's the same hash the
    orchestrator computes per input row, so the two sets line up.
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    try:
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = compute_row_id(
                    row.get("Issue") or "",
                    row.get("Subject") or "",
                    row.get("Company") or "",
                )
                done.add(rid)
    except Exception as exc:  # noqa: BLE001
        print(
            f"warning: failed to read resume set from {output_path}: {exc}",
            file=sys.stderr,
        )
    return done


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    index_dir = _resolve_path(args.index_dir)
    trace_dir = _resolve_path(args.trace_dir)

    # Pre-flight: API key check (real mode only).
    if not args.dry_run:
        api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
        if not api_key:
            print(
                "error: NEBIUS_API_KEY is not set. Export it before running, "
                "or pass --dry-run for a wiring smoke test.",
                file=sys.stderr,
            )
            return 4

    # Build the retriever (skip in dry-run).
    retriever = None
    if not args.dry_run:
        try:
            from retriever import Retriever  # noqa: E402

            retriever = Retriever(
                index_dir=index_dir,
                enable_rerank=not args.no_rerank,
            )
        except FileNotFoundError as exc:
            print(
                f"error: retriever index not found ({exc}). "
                f"Run `python code/corpus.py` to build it.",
                file=sys.stderr,
            )
            return 5
        except Exception as exc:  # noqa: BLE001
            print(f"error: failed to construct Retriever: {exc}", file=sys.stderr)
            return 1

    # Build the orchestrator.
    orch = Orchestrator(
        retriever,
        enable_revise=not args.no_revise,
        trace_dir=trace_dir,
        stub_mode=args.dry_run,
    )

    # Read the input CSV.
    if not input_path.exists():
        print(f"error: input CSV not found: {input_path}", file=sys.stderr)
        return 1
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Resume map.
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        done_ids = _load_resume_set(output_path)

    # Open output file (append on resume, write otherwise).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    write_header = not (args.resume and file_exists and done_ids)
    mode = "a" if (args.resume and file_exists and not write_header) else "w"
    out_f = open(output_path, mode, encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(
            out_f,
            fieldnames=list(OUTPUT_COLUMNS),
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        if write_header:
            writer.writeheader()
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except OSError:
                pass

        # Per-row loop.
        total = len(all_rows)
        processed = 0
        replied = 0
        escalated = 0
        trigger_counter: Counter[str] = Counter()
        t_start = time.monotonic()

        for i, raw in enumerate(all_rows):
            if i < args.start:
                continue
            issue = (raw.get("Issue") or "").strip()
            subject = (raw.get("Subject") or "").strip()
            company = (raw.get("Company") or "none").strip() or "none"

            row_id = compute_row_id(issue, subject, company)
            if row_id in done_ids:
                if not args.quiet:
                    print(f"[{i + 1}/{total}] {row_id} skipped (resume)")
                continue
            if args.limit is not None and processed >= args.limit:
                break

            t_row = time.monotonic()
            try:
                row, trace = orch.process_row(issue, subject, company)
            except Exception as exc:  # noqa: BLE001 - belt-and-suspenders
                row = safety.assemble_output_row(
                    issue=issue,
                    subject=subject,
                    company=company,
                    decision="escalated",
                    response_text=(
                        "Thank you for reaching out. We're routing this to a human "
                        "agent who will follow up with you shortly."
                    ),
                    justification=(
                        f"validation_failure -- {type(exc).__name__}: "
                        f"{str(exc)[:200]}"
                    ),
                    product_area="",
                    request_type="invalid",
                )
                trace = None
            row_elapsed_ms = int((time.monotonic() - t_row) * 1000)

            writer.writerow(row)
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except OSError:
                pass

            processed += 1
            status = row.get("Status", "")
            if status == "Replied":
                replied += 1
            else:
                escalated += 1
                if trace is not None and trace.escalation_trigger:
                    trigger_counter[trace.escalation_trigger] += 1
                else:
                    trigger_counter["unknown"] += 1

            if not args.quiet:
                print(
                    f"[{i + 1}/{total}] {row_id} {status} ({row_elapsed_ms}ms)"
                )
    finally:
        out_f.close()

    total_elapsed = time.monotonic() - t_start
    avg = total_elapsed / processed if processed else 0.0
    print()
    print(
        f"Processed: {processed} rows in {total_elapsed:.1f}s "
        f"({avg:.1f}s/row avg)"
    )
    print(f"Replied: {replied}")
    print(f"Escalated: {escalated}")
    if trigger_counter:
        print("Top triggers:")
        print(f"  {'trigger':<28} count")
        for trig, n in trigger_counter.most_common():
            print(f"  {trig:<28} {n}")
    try:
        rel = output_path.relative_to(REPO_ROOT)
        print(f"Output: {rel.as_posix()}")
    except ValueError:
        print(f"Output: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback as _tb

        _tb.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
