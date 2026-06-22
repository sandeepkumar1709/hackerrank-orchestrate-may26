"""Corpus chunker for the support triage agent.

Walks data/{hackerrank,claude,visa}, splits markdown by heading with
section-and-paragraph packing, prepends a ``title > heading_path`` context
line to every chunk, and persists ``data/index/chunks.jsonl`` plus a manifest.

Designed to be deterministic: same inputs -> byte-identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARD_CAP = 480                 # tokens; leaves headroom under BGE-small's 512 ceiling
SOFT_TARGET_MIN = 300
SOFT_TARGET_MAX = 400
FLOOR = 80                     # merge sibling chunks below this
EMPTY_BODY_TOKEN_THRESHOLD = 20  # skip files whose body has fewer tokens than this
TITLE_PREFIX_TOKEN_CAP = 64    # truncate prefix if it alone would dominate
CHUNKER_VERSION = "1.0.0"
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
COMPANIES = ("hackerrank", "claude", "visa")

SECTION_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
LAST_UPDATED_RE = re.compile(r"^_Last updated:.*?_\s*$", re.MULTILINE)
IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
TABLE_BLOCK_RE = re.compile(r"(?:^\|.*\|\s*$\n?)+", re.MULTILINE)


# ---------------------------------------------------------------------------
# Tokenizer wrapper (cached for the duration of a build)
# ---------------------------------------------------------------------------


def _load_tokenizer():
    """Load the BGE-small tokenizer. Fail loudly on any error; no fallback."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for corpus.py. "
            "Install with `pip install transformers tokenizers`."
        ) from exc
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # network failure, missing files, etc.
        raise RuntimeError(
            f"Failed to load tokenizer {TOKENIZER_NAME!r}: {exc}. "
            "This is required; we do not fall back to a different tokenizer."
        ) from exc
    return tok


def _count_tokens(s: str, tokenizer) -> int:
    """Count tokens excluding any special tokens the model would otherwise add."""
    if not s:
        return 0
    return len(tokenizer.encode(s, add_special_tokens=False))


# ---------------------------------------------------------------------------
# File discovery + I/O
# ---------------------------------------------------------------------------


def _iter_markdown_files(data_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted ``(company, abs_path)`` pairs for every .md file.

    Sort key is the POSIX relative path inside the company root so the order
    is deterministic across platforms.
    """
    out: list[tuple[str, Path]] = []
    for company in COMPANIES:
        root = data_dir / company
        if not root.exists():
            continue
        files = list(root.rglob("*.md"))
        # Sort by POSIX relative path inside the company root.
        files.sort(key=lambda p: p.relative_to(root).as_posix())
        for f in files:
            out.append((company, f))
    return out


def _corpus_checksum(files: list[tuple[str, Path]]) -> str:
    """Deterministic sha256 over (rel_posix_path, sha256_of_bytes) pairs."""
    h = hashlib.sha256()
    pairs: list[tuple[str, str]] = []
    for _company, path in files:
        # Use POSIX relative-to-cwd path so checksums are stable.
        rel = path.as_posix()
        # We want the path slug to be stable across OSes, so keep the
        # relative form from the data dir using the company anchor.
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        pairs.append((rel, digest))
    pairs.sort()
    for rel, digest in pairs:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Frontmatter + body cleanup
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Empty dict if no frontmatter."""
    if not text.startswith("---"):
        return {}, text
    # Find the closing fence on its own line (allow optional trailing whitespace).
    m = re.search(r"^---\s*$", text[3:], flags=re.MULTILINE)
    if not m:
        return {}, text
    yaml_block = text[3 : 3 + m.start()]
    body_start = 3 + m.end()
    # Skip the newline after the closing fence, if any.
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(yaml_block) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return data, text[body_start:]


def _strip_last_updated(body: str) -> str:
    return LAST_UPDATED_RE.sub("", body)


def _strip_images(body: str) -> str:
    return IMAGE_MD_RE.sub("", body)


# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------


def _should_skip_index(file_path: Path) -> bool:
    return file_path.name == "index.md"


def _should_skip_empty(body: str, tokenizer) -> bool:
    return _count_tokens(body.strip(), tokenizer) < EMPTY_BODY_TOKEN_THRESHOLD


# ---------------------------------------------------------------------------
# Title / breadcrumbs
# ---------------------------------------------------------------------------


def _extract_title(frontmatter: dict, body: str, file_path: Path) -> str:
    t = frontmatter.get("title")
    if isinstance(t, str) and t.strip():
        return t.strip()
    m = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return file_path.stem


def _extract_breadcrumbs(
    frontmatter: dict,
    file_path: Path,
    data_dir: Path,
    company: str,
) -> list[str]:
    bc = frontmatter.get("breadcrumbs")
    if isinstance(bc, list) and all(isinstance(x, str) for x in bc):
        return [x.strip() for x in bc if x.strip()]
    rel = file_path.relative_to(data_dir / company)
    parts = list(rel.parts[:-1])  # exclude the file name
    return parts


# ---------------------------------------------------------------------------
# Section + paragraph splitting
# ---------------------------------------------------------------------------


def _split_sections(body: str, title: str) -> list[tuple[list[str], str]]:
    """Split ``body`` on H1/H2/H3 headings.

    The first H1 whose text equals ``title`` (case-insensitive, trimmed) is
    consumed and not treated as a section break; all subsequent headings (any
    of #/##/###) are section breaks. Returns a list of
    ``(heading_path, section_text)`` tuples. ``heading_path`` is empty for
    the pre-heading prelude. If no headings remain, returns
    ``[([], body_stripped)]``.
    """
    matches = list(SECTION_HEADING_RE.finditer(body))
    title_norm = title.strip().lower()

    # Find and strip the leading title H1 if present.
    consume_idx: int | None = None
    if matches and len(matches[0].group(1)) == 1:  # first heading is H1
        heading_text = matches[0].group(2).strip().lower()
        if (
            heading_text == title_norm
            or heading_text.rstrip(".:!?") == title_norm
        ):
            consume_idx = 0

    breaks = matches[consume_idx + 1 :] if consume_idx is not None else matches

    # The "prelude" runs from the end of the consumed title heading (or 0)
    # to the first remaining break (or end of body).
    prelude_start = matches[consume_idx].end() if consume_idx is not None else 0

    sections: list[tuple[list[str], str]] = []

    if not breaks:
        text = body[prelude_start:].strip()
        if text:
            sections.append(([], text))
        return sections or [([], body.strip())]

    # Prelude (text before the first break) -> heading_path = []
    prelude = body[prelude_start : breaks[0].start()].strip()
    if prelude:
        sections.append(([], prelude))

    # Walk the breaks. Track a depth-indexed heading path so subheadings nest.
    path_stack: list[tuple[int, str]] = []  # (level, heading text)
    for i, m in enumerate(breaks):
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        # Pop any stack entries at the same or deeper level.
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, heading_text))
        heading_path = [h for _lv, h in path_stack]

        text_start = m.end()
        text_end = breaks[i + 1].start() if i + 1 < len(breaks) else len(body)
        section_text = body[text_start:text_end].strip()
        if section_text:
            sections.append((heading_path, section_text))

    if not sections:
        # Fallback: nothing extracted (e.g. pure heading-only file).
        return [([], body.strip())]
    return sections


def _replace_atomic_blocks(text: str) -> tuple[str, list[str]]:
    """Replace code fences and tables with placeholders; return (text, blocks)."""
    blocks: list[str] = []

    def _stash(match: re.Match) -> str:
        idx = len(blocks)
        blocks.append(match.group(0))
        return f"__ATOMIC_BLOCK_{idx}__"

    # Stash code fences first so a fenced table isn't double-stashed.
    text = CODE_FENCE_RE.sub(_stash, text)
    text = TABLE_BLOCK_RE.sub(_stash, text)
    return text, blocks


_ATOMIC_PLACEHOLDER_RE = re.compile(r"__ATOMIC_BLOCK_(\d+)__")


def _restore_atomic_blocks(text: str, blocks: list[str]) -> str:
    def _unstash(match: re.Match) -> str:
        idx = int(match.group(1))
        return blocks[idx] if 0 <= idx < len(blocks) else match.group(0)

    return _ATOMIC_PLACEHOLDER_RE.sub(_unstash, text)


def _hard_cut_tokens(text: str, max_tokens: int, tokenizer) -> list[str]:
    """Split a long blob by encoded token windows. Last resort."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces: list[str] = []
    for i in range(0, len(ids), max_tokens):
        window = ids[i : i + max_tokens]
        decoded = tokenizer.decode(window, skip_special_tokens=True).strip()
        if decoded:
            pieces.append(decoded)
    return pieces


def _split_paragraph_into_sentences(
    paragraph: str, max_tokens: int, tokenizer
) -> list[str]:
    """Split an over-cap paragraph by sentence; hard-cut as last resort."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out: list[str] = []
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = sent if not buf else f"{buf} {sent}"
        if _count_tokens(candidate, tokenizer) <= max_tokens:
            buf = candidate
            continue
        # candidate too big -- flush buf, then handle sent on its own.
        if buf:
            out.append(buf)
            buf = ""
        if _count_tokens(sent, tokenizer) <= max_tokens:
            buf = sent
        else:
            # Single sentence still too big: token-window cut.
            out.extend(_hard_cut_tokens(sent, max_tokens, tokenizer))
    if buf:
        out.append(buf)
    return out


def _split_paragraphs(text: str, max_tokens: int, tokenizer) -> list[str]:
    """Greedy paragraph packing under ``max_tokens``.

    Code fences and tables are atomic. Atomic blocks larger than ``max_tokens``
    are emitted as their own chunk (we don't shred them).
    """
    if max_tokens <= 0:
        # Defensive: prefix already exceeds the cap. Use a small floor to keep
        # progress; later merge will rebalance.
        max_tokens = 16

    stashed_text, blocks = _replace_atomic_blocks(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", stashed_text) if p.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def _flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append(_restore_atomic_blocks("\n\n".join(buf), blocks))
            buf = []
            buf_tokens = 0

    for para in paragraphs:
        # If this paragraph is a single atomic placeholder, treat the
        # underlying block atomically (it's never split).
        is_atomic_only = bool(_ATOMIC_PLACEHOLDER_RE.fullmatch(para))
        restored = _restore_atomic_blocks(para, blocks)
        para_tokens = _count_tokens(restored, tokenizer)

        if para_tokens <= max_tokens:
            if buf_tokens + para_tokens <= max_tokens:
                buf.append(para)
                buf_tokens += para_tokens
            else:
                _flush()
                buf.append(para)
                buf_tokens = para_tokens
            continue

        # Paragraph alone exceeds max_tokens.
        _flush()
        if is_atomic_only:
            # Prefer keeping atomic blocks intact; fall back to a token-window
            # hard-cut only when the block is too big to honor the cap.
            # This keeps the schema invariant (n_tokens <= max_tokens).
            if para_tokens <= max_tokens:
                chunks.append(restored)
            else:
                chunks.extend(_hard_cut_tokens(restored, max_tokens, tokenizer))
            continue
        # Sentence-split (the paragraph may still contain placeholders for
        # in-line code/tables; restore before sentence-splitting).
        sub_chunks = _split_paragraph_into_sentences(restored, max_tokens, tokenizer)
        chunks.extend(sub_chunks)

    _flush()
    return chunks


# ---------------------------------------------------------------------------
# Small-chunk merging
# ---------------------------------------------------------------------------


def _merge_small(
    chunks: list[str], floor: int, cap: int, tokenizer
) -> list[str]:
    """Merge chunks below ``floor`` tokens into siblings without exceeding ``cap``."""
    if len(chunks) <= 1:
        return list(chunks)

    out: list[str] = list(chunks)

    # Forward pass: merge a small chunk into the next sibling if it fits.
    i = 0
    while i < len(out) - 1:
        cur_t = _count_tokens(out[i], tokenizer)
        if cur_t < floor:
            nxt_t = _count_tokens(out[i + 1], tokenizer)
            if cur_t + nxt_t + 2 <= cap:  # +2 for the joining "\n\n"
                out[i + 1] = out[i] + "\n\n" + out[i + 1]
                del out[i]
                continue
        i += 1

    # Backward pass: merge a trailing small chunk into the previous one.
    if len(out) >= 2:
        last_t = _count_tokens(out[-1], tokenizer)
        if last_t < floor:
            prev_t = _count_tokens(out[-2], tokenizer)
            if prev_t + last_t + 2 <= cap:
                out[-2] = out[-2] + "\n\n" + out[-1]
                out.pop()

    return out


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "section"


def _truncate_prefix_to_cap(prefix: str, cap: int, tokenizer) -> str:
    """Hard-cut a too-long prefix at ``cap`` tokens, keeping a trailing newlines."""
    if _count_tokens(prefix, tokenizer) <= cap:
        return prefix
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    truncated = tokenizer.decode(ids[:cap], skip_special_tokens=True).strip()
    return truncated + "\n\n"


def _make_chunk(
    company: str,
    file_rel_path: str,
    title: str,
    breadcrumbs: list[str],
    source_url: str | None,
    heading_path: list[str],
    text: str,
    idx_within_file: int,
    tokenizer,
) -> dict:
    file_stem = Path(file_rel_path).stem
    if heading_path:
        slug = _slugify(" ".join(heading_path))
        chunk_id = f"{company}:{file_stem}:{slug}"
    else:
        chunk_id = f"{company}:{file_stem}:{idx_within_file}"
    return {
        "id": chunk_id,
        "company": company,
        "path": file_rel_path,
        "title": title,
        "breadcrumb_path": list(breadcrumbs),
        "source_url": source_url,
        "heading_path": list(heading_path),
        "text": text,
        "n_tokens": _count_tokens(text, tokenizer),
    }


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def _per_company_counts(rows: Iterable[dict]) -> dict[str, int]:
    counts = {c: 0 for c in COMPANIES}
    for r in rows:
        c = r["company"]
        counts[c] = counts.get(c, 0) + 1
    return counts


def _format_stats(rows: list[dict], skipped_files: list[dict]) -> str:
    counts = _per_company_counts(rows)
    tokens = sorted(r["n_tokens"] for r in rows) if rows else []
    p50 = statistics.median(tokens) if tokens else 0
    p95 = tokens[int(0.95 * (len(tokens) - 1))] if tokens else 0
    mx = max(tokens) if tokens else 0
    skipped_index = sum(1 for s in skipped_files if s["reason"] == "index")
    skipped_empty = sum(1 for s in skipped_files if s["reason"] == "empty")
    lines = [
        f"corpus: {len(rows)} chunks",
        f"  per company: hackerrank={counts['hackerrank']} claude={counts['claude']} visa={counts['visa']}",
        f"  tokens p50={p50} p95={p95} max={mx}",
        f"  skipped: index={skipped_index} empty={skipped_empty}",
    ]
    return "\n".join(lines)


def build_corpus(
    data_dir: Path,
    out_dir: Path,
    force: bool = False,
    print_stats: bool = False,
) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer()
    files = _iter_markdown_files(data_dir)
    checksum = _corpus_checksum(files)

    manifest_path = out_dir / "manifest.json"
    chunks_path = out_dir / "chunks.jsonl"

    if (
        not force
        and manifest_path.exists()
        and chunks_path.exists()
    ):
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("corpus_checksum") == checksum:
                print("skip rebuild (corpus unchanged)", file=sys.stderr)
                if print_stats:
                    # Best-effort stats from the existing file.
                    rows = [
                        json.loads(line)
                        for line in chunks_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    print(
                        _format_stats(rows, existing.get("skipped_files", [])),
                        flush=True,
                    )
                return existing
        except Exception:
            pass  # corrupt manifest -> rebuild.

    chunks: list[dict] = []
    skipped_files: list[dict] = []

    for company, abs_path in files:
        # We want a POSIX path beginning with "data/...". data_dir is e.g.
        # Path("data") (relative) or an absolute Path. In either case,
        # join the data_dir's leaf name with the part inside data_dir.
        rel_inside_data = abs_path.relative_to(data_dir).as_posix()
        data_leaf = data_dir.name or "data"
        file_rel_path = f"{data_leaf}/{rel_inside_data}"

        if _should_skip_index(abs_path):
            skipped_files.append({"path": file_rel_path, "reason": "index"})
            continue

        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"Failed to read {abs_path}: {exc}") from exc

        frontmatter, body = _parse_frontmatter(text)
        body = _strip_last_updated(_strip_images(body))

        if _should_skip_empty(body, tokenizer):
            skipped_files.append({"path": file_rel_path, "reason": "empty"})
            continue

        title = _extract_title(frontmatter, body, abs_path)
        breadcrumbs = _extract_breadcrumbs(frontmatter, abs_path, data_dir, company)
        source_url_raw = frontmatter.get("source_url")
        source_url = source_url_raw if isinstance(source_url_raw, str) else None

        # Whole-article-when-it-fits: if the entire body plus the title prefix
        # fits under HARD_CAP, emit a single chunk with empty heading_path.
        # Otherwise split on H1/H2/H3 (consuming the first matching title H1).
        whole_prefix = _truncate_prefix_to_cap(f"{title}\n\n", TITLE_PREFIX_TOKEN_CAP, tokenizer)
        if _count_tokens(whole_prefix + body, tokenizer) <= HARD_CAP:
            sections = [([], body)]
        else:
            sections = _split_sections(body, title)

        # Build a flat list of (heading_path, text) chunks for this file.
        file_texts: list[tuple[list[str], str]] = []
        for heading_path, section_text in sections:
            if heading_path:
                prefix = f"{title} > {' > '.join(heading_path)}\n\n"
            else:
                prefix = f"{title}\n\n"
            prefix = _truncate_prefix_to_cap(prefix, TITLE_PREFIX_TOKEN_CAP, tokenizer)

            prefix_tokens = _count_tokens(prefix, tokenizer)
            full_text = prefix + section_text
            if _count_tokens(full_text, tokenizer) <= HARD_CAP:
                file_texts.append((heading_path, full_text))
                continue

            # Paragraph-split with prefix budget.
            budget = HARD_CAP - prefix_tokens
            pieces = _split_paragraphs(section_text, budget, tokenizer)
            for piece in pieces:
                file_texts.append((heading_path, prefix + piece))

        # Apply _merge_small across the whole file (treats them as siblings).
        # The merged-into chunk keeps the heading_path of its anchor (the
        # chunk that absorbed the smaller neighbor). We track heading_path
        # in lockstep with the texts so chunk metadata stays correct.
        merged: list[tuple[list[str], str]] = []
        if not file_texts:
            pass
        else:
            heading_paths = [hp for hp, _t in file_texts]
            texts = [t for _hp, t in file_texts]

            # Forward pass: merge small chunk into next sibling.
            i = 0
            while i < len(texts) - 1:
                cur_t = _count_tokens(texts[i], tokenizer)
                if cur_t < FLOOR:
                    nxt_t = _count_tokens(texts[i + 1], tokenizer)
                    if cur_t + nxt_t + 2 <= HARD_CAP:
                        # Prefer the more specific (deeper) heading path.
                        if len(heading_paths[i + 1]) >= len(heading_paths[i]):
                            keep_hp = heading_paths[i + 1]
                        else:
                            keep_hp = heading_paths[i]
                        texts[i + 1] = texts[i] + "\n\n" + texts[i + 1]
                        heading_paths[i + 1] = keep_hp
                        del texts[i]
                        del heading_paths[i]
                        continue
                i += 1
            # Backward merge for trailing small chunk.
            if len(texts) >= 2:
                last_t = _count_tokens(texts[-1], tokenizer)
                if last_t < FLOOR:
                    prev_t = _count_tokens(texts[-2], tokenizer)
                    if prev_t + last_t + 2 <= HARD_CAP:
                        keep_hp = (
                            heading_paths[-2]
                            if len(heading_paths[-2]) >= len(heading_paths[-1])
                            else heading_paths[-1]
                        )
                        texts[-2] = texts[-2] + "\n\n" + texts[-1]
                        heading_paths[-2] = keep_hp
                        texts.pop()
                        heading_paths.pop()

            for hp, t in zip(heading_paths, texts):
                merged.append((hp, t))

        for idx, (heading_path, txt) in enumerate(merged):
            chunks.append(
                _make_chunk(
                    company=company,
                    file_rel_path=file_rel_path,
                    title=title,
                    breadcrumbs=breadcrumbs,
                    source_url=source_url,
                    heading_path=heading_path,
                    text=txt,
                    idx_within_file=idx,
                    tokenizer=tokenizer,
                )
            )

    # Safety net: any chunk still over cap (shouldn't happen, but be defensive)
    # gets hard-cut into per-chunk-cap pieces. Rare in practice; keeps the
    # downstream schema invariant strict.
    safe_chunks: list[dict] = []
    for c in chunks:
        if c["n_tokens"] <= HARD_CAP:
            safe_chunks.append(c)
            continue
        pieces = _hard_cut_tokens(c["text"], HARD_CAP, tokenizer)
        for j, p in enumerate(pieces):
            new = dict(c)
            new["text"] = p
            new["n_tokens"] = _count_tokens(p, tokenizer)
            new["id"] = f"{c['id']}-cut{j}"
            safe_chunks.append(new)
    chunks = safe_chunks

    # Stable sort for byte-identical output.
    company_order = {c: i for i, c in enumerate(COMPANIES)}
    chunks.sort(
        key=lambda r: (company_order.get(r["company"], 99), r["path"], r["id"])
    )

    # Disambiguate any rare id collisions deterministically.
    seen: dict[str, int] = {}
    for c in chunks:
        if c["id"] in seen:
            seen[c["id"]] += 1
            c["id"] = f"{c['id']}-{seen[c['id']]}"
        else:
            seen[c["id"]] = 0
    # Re-sort after id rewriting (shouldn't change order, but be safe).
    chunks.sort(
        key=lambda r: (company_order.get(r["company"], 99), r["path"], r["id"])
    )

    _write_jsonl_atomic(chunks_path, chunks)

    counts = _per_company_counts(chunks)
    skipped_index = sum(1 for s in skipped_files if s["reason"] == "index")
    skipped_empty = sum(1 for s in skipped_files if s["reason"] == "empty")

    manifest = {
        "corpus_checksum": checksum,
        "chunk_count": len(chunks),
        "build_timestamp_iso": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "chunker_version": CHUNKER_VERSION,
        "tokenizer_name": TOKENIZER_NAME,
        "per_company_counts": {c: counts.get(c, 0) for c in COMPANIES},
        "skipped_index_count": skipped_index,
        "skipped_empty_count": skipped_empty,
        "skipped_files": sorted(
            skipped_files, key=lambda s: (s["reason"], s["path"])
        ),
    }
    _write_json_atomic(manifest_path, manifest)

    if print_stats:
        print(_format_stats(chunks, skipped_files), flush=True)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data/index")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-stats", action="store_true")
    args = parser.parse_args()
    build_corpus(
        Path(args.data_dir),
        Path(args.out_dir),
        force=args.force,
        print_stats=args.print_stats,
    )
