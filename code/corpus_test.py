"""Smoke test for ``code/corpus.py``.

Run with: ``python code/corpus_test.py`` (from the repo root).

Asserts the manifest + chunks.jsonl shape, byte-determinism across rebuilds,
and that the skip-rebuild path is honoured. Prints ``OK: ...`` on success.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

# Make ``code/`` importable regardless of where Python is invoked from.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import corpus  # noqa: E402


REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "data" / "index"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
MANIFEST_PATH = OUT_DIR / "manifest.json"

REQUIRED_KEYS = {
    "id",
    "company",
    "path",
    "title",
    "breadcrumb_path",
    "source_url",
    "heading_path",
    "text",
    "n_tokens",
}


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print(">> first build (force=True)", flush=True)
    manifest = corpus.build_corpus(DATA_DIR, OUT_DIR, force=True)

    # 2. manifest sanity.
    assert MANIFEST_PATH.exists(), "manifest.json was not written"
    on_disk = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert on_disk["tokenizer_name"] == "BAAI/bge-small-en-v1.5", on_disk[
        "tokenizer_name"
    ]
    assert on_disk["chunker_version"] == "1.0.0", on_disk["chunker_version"]
    assert re.fullmatch(r"[0-9a-f]{64}", on_disk["corpus_checksum"]), on_disk[
        "corpus_checksum"
    ]

    chunk_count = on_disk["chunk_count"]
    counts = on_disk["per_company_counts"]
    print(
        f">> chunk_count={chunk_count} per_company={counts}",
        flush=True,
    )

    # 3. count ranges. Calibrated to actual data with H1-as-section-break and
    # whole-article-when-it-fits chunking. HR articles use repeated H1s for
    # section breaks (frontmatter title is the join of those H1s), so the
    # corpus produces more chunks than a naive ceil(tokens/cap) would suggest.
    assert 4000 <= chunk_count <= 6000, f"chunk_count out of range: {chunk_count}"
    assert 2500 <= counts["hackerrank"] <= 4000, counts
    assert 1300 <= counts["claude"] <= 2200, counts
    assert 70 <= counts["visa"] <= 120, counts

    # 4. validate chunks.jsonl line-by-line.
    rows: list[dict] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_KEYS - row.keys()
            assert not missing, f"line {lineno} missing keys: {missing}"
            assert 0 < row["n_tokens"] <= corpus.HARD_CAP, (
                f"line {lineno} bad n_tokens={row['n_tokens']}"
            )
            assert "\\" not in row["path"], (
                f"line {lineno} backslash in path: {row['path']!r}"
            )
            assert row["path"].startswith("data/"), (
                f"line {lineno} path doesn't start with data/: {row['path']!r}"
            )
            rows.append(row)

    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids found"

    # 5. byte determinism across rebuilds.
    print(">> second build (force=True)", flush=True)
    sha_before = _file_sha(CHUNKS_PATH)
    corpus.build_corpus(DATA_DIR, OUT_DIR, force=True)
    sha_after = _file_sha(CHUNKS_PATH)
    assert sha_before == sha_after, (
        f"chunks.jsonl not byte-identical across rebuilds: {sha_before} != {sha_after}"
    )

    # 6. skip-rebuild leaves the file untouched.
    print(">> third build (force=False, expect skip)", flush=True)
    mtime_before = CHUNKS_PATH.stat().st_mtime_ns
    # Sleep a tick so that any accidental rewrite produces a different mtime
    # on filesystems with coarse timestamp resolution.
    time.sleep(0.01)
    corpus.build_corpus(DATA_DIR, OUT_DIR, force=False)
    mtime_after = CHUNKS_PATH.stat().st_mtime_ns
    assert mtime_before == mtime_after, (
        f"force=False triggered a rebuild: mtime changed "
        f"{mtime_before} -> {mtime_after}"
    )

    # 7. at least one Visa file skipped as index.
    skipped_indexes = [
        s for s in on_disk["skipped_files"] if s["reason"] == "index"
    ]
    assert any(
        s["path"].startswith("data/visa/") for s in skipped_indexes
    ), f"no Visa file was skipped as index: {skipped_indexes}"

    # 8. at least one HackerRank chunk has a non-empty heading_path.
    assert any(
        r["company"] == "hackerrank" and r["heading_path"] for r in rows
    ), "no HackerRank chunk has a non-empty heading_path"

    # 9. at least one Claude article produced >= 2 chunks.
    by_path: dict[str, int] = {}
    for r in rows:
        if r["company"] == "claude":
            by_path[r["path"]] = by_path.get(r["path"], 0) + 1
    assert any(v >= 2 for v in by_path.values()), (
        "no Claude article produced >= 2 chunks"
    )

    # 10. median tokens within sane window.
    median_tokens = statistics.median(r["n_tokens"] for r in rows)
    assert 150 <= median_tokens <= 400, f"median tokens out of window: {median_tokens}"

    print(
        f"OK: {len(rows)} chunks across "
        f"{{hackerrank={counts['hackerrank']}, "
        f"claude={counts['claude']}, "
        f"visa={counts['visa']}}}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
