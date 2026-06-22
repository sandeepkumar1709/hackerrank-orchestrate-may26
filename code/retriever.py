"""Hybrid retriever for the support-triage agent.

Pipeline: BM25 (lexical) + dense embeddings (BAAI/bge-small-en-v1.5),
combined via Reciprocal Rank Fusion, optionally re-ranked with Claude
Sonnet (latest). Determinism: stable sorts on (score desc, chunk_id asc),
temperature=0 on the rerank, fixed stopwords + regex tokenizer for BM25.

Public API
----------
::

    from retriever import Retriever
    r = Retriever()  # loads data/index/*; lazily builds embeddings.npy
    hits = r.retrieve("how long do tests stay active?", company="hackerrank", k=5)

CLI
---
``python code/retriever.py --query "..." [--company X] [--k 5] [--no-rerank] [--json]``

Dependencies (install if missing): rank_bm25, sentence-transformers (or
transformers+torch as a manual fallback), numpy, openai, python-dotenv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# rank_bm25 is light; required.
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "rank_bm25 is required. Install with `pip install rank-bm25`."
    ) from exc

# Make ``code/`` importable so we can use schema.normalize_company whether
# launched as a script or as a module.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from schema import normalize_company  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
EMBED_BATCH = 64
RRF_K = 60
BM25_TOPN = 30
DENSE_TOPN = 30
RRF_TOPN = 20
DEFAULT_RERANK_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # Nebius default
RERANK_MAX_TOKENS = 1500
USER_QUERY_CAP_CHARS = 500
USER_CHUNK_CAP_CHARS = 600
# 2 / (RRF_K + 1) is the maximum possible RRF score for two rankings (rank 1 in both)
MAX_RRF_SCORE = 2.0 / (RRF_K + 1)  # ~= 0.0328

# Fixed English stopword list (40 words) for BM25 tokenization.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "do", "does", "for", "from", "had", "has", "have", "i",
        "if", "in", "is", "it", "its", "me", "my", "no", "not", "of",
        "on", "or", "so", "that", "the", "to", "was", "were", "what",
        "will", "with", "you",
    }
)
# Note: 41 entries above; trimmed in the regex stripping pass to keep tests stable.

_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9_\-]*\b")


SYSTEM_PROMPT_RERANK = (
    "You are a relevance scorer for a customer-support retrieval pipeline.\n"
    "You will be given a user's support question and a numbered list of corpus "
    "chunks. For each chunk, output a relevance score from 0.0 to 1.0:\n"
    "- 1.0 = directly answers the question\n"
    "- 0.5 = related but partial / tangential\n"
    "- 0.0 = unrelated\n"
    "Respond with ONLY a JSON object: "
    "{\"scores\": [{\"id\": <int>, \"score\": <float>}, ...]}\n"
    "Include every input id exactly once. No prose. No markdown fences."
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    company: str
    path: str
    title: str
    heading_path: list[str]
    source_url: str | None
    text: str
    n_tokens: int
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float
    rerank_score: float | None
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_index_dir(raw: str | Path) -> Path:
    """Resolve ``index_dir``. If relative, anchor it at the repo root.

    The repo root is the parent of ``code/`` (where this file lives).
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    # Try as-is (cwd-relative) first; fall back to repo-rooted.
    if p.exists():
        return p.resolve()
    repo_root = _HERE.parent
    return (repo_root / p).resolve()


def _tokenize(text: str) -> list[str]:
    """Lowercase, regex-tokenize, drop stopwords and tokens of length <2."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


class _EmbedderST:
    """sentence-transformers backend (preferred path)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = EMBED_BATCH) -> np.ndarray:
        vec = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32, copy=False)


class _EmbedderTransformers:
    """Manual transformers + torch fallback (mean-pool + L2-normalize)."""

    def __init__(self, model_name: str):
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name

    def _mean_pool(self, last_hidden, attention_mask):
        torch = self.torch
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int = EMBED_BATCH) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self.model(**enc)
                pooled = self._mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                # L2 normalize.
                norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
                pooled = pooled / norms
                out[i : i + len(batch)] = pooled.cpu().numpy().astype(np.float32)
        return out


def _make_embedder(model_name: str):
    """Try sentence-transformers first; fall back to transformers+torch."""
    try:
        return _EmbedderST(model_name)
    except Exception as exc_st:
        try:
            return _EmbedderTransformers(model_name)
        except Exception as exc_tr:
            raise RuntimeError(
                f"Could not load any embedding backend for {model_name!r}. "
                f"sentence-transformers error: {exc_st}. "
                f"transformers fallback error: {exc_tr}."
            ) from exc_tr


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """Hybrid BM25 + dense retriever with optional LLM rerank (Nebius)."""

    def __init__(
        self,
        index_dir: str | Path = "data/index",
        llm_client: Any = None,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        enable_rerank: bool = True,
    ):
        self.index_dir = _resolve_index_dir(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"index dir not found: {self.index_dir}")

        manifest_path = self.index_dir / "manifest.json"
        chunks_path = self.index_dir / "chunks.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"missing chunks: {chunks_path}")

        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._corpus_checksum: str = self._manifest["corpus_checksum"]

        # Load chunks. JSONL is ~5k rows; loading once is fine.
        self._chunks: list[dict] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._chunks.append(json.loads(line))
        if not self._chunks:
            raise RuntimeError(f"no chunks loaded from {chunks_path}")

        # ID lookup + per-company index pools.
        self._id_to_idx: dict[str, int] = {c["id"]: i for i, c in enumerate(self._chunks)}
        self._company_idxs: dict[str, list[int]] = {}
        for i, c in enumerate(self._chunks):
            self._company_idxs.setdefault(c["company"], []).append(i)

        # BM25 (in-memory, rebuilt each session).
        self._bm25_tokens: list[list[str]] = [
            _tokenize(c["text"]) for c in self._chunks
        ]
        self._bm25 = BM25Okapi(self._bm25_tokens)

        # Embeddings: load-or-build.
        self._embed_path = self.index_dir / "embeddings.npy"
        self._embed_manifest_path = self.index_dir / "embedding_manifest.json"
        self._embeddings: np.ndarray | None = None
        self._embedder: Any = None  # lazy
        self._ensure_embeddings()

        # Rerank wiring.
        self._rerank_model = rerank_model
        self._enable_rerank = enable_rerank
        self._llm_client = llm_client  # may be None; lazy below.
        self._llm_attempted = False

    # ---------- embeddings ----------

    def _embed_cache_valid(self) -> bool:
        if not (self._embed_path.exists() and self._embed_manifest_path.exists()):
            return False
        try:
            em = json.loads(self._embed_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            em.get("model_name") == EMBED_MODEL_NAME
            and em.get("dim") == EMBED_DIM
            and em.get("count") == len(self._chunks)
            and em.get("corpus_checksum") == self._corpus_checksum
            and em.get("normalize") is True
        )

    def _ensure_embeddings(self) -> None:
        if self._embed_cache_valid():
            arr = np.load(self._embed_path)
            if arr.shape == (len(self._chunks), EMBED_DIM) and arr.dtype == np.float32:
                self._embeddings = arr
                return
            # Shape mismatch -> rebuild.

        self._build_embeddings()

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = _make_embedder(EMBED_MODEL_NAME)
        return self._embedder

    def _build_embeddings(self) -> None:
        embedder = self._get_embedder()
        texts = [c["text"] for c in self._chunks]
        t0 = time.time()
        vec = embedder.encode(texts, batch_size=EMBED_BATCH)
        if vec.shape != (len(self._chunks), EMBED_DIM):
            raise RuntimeError(
                f"unexpected embedding shape {vec.shape}; "
                f"expected ({len(self._chunks)}, {EMBED_DIM})"
            )
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        # Defensive renormalization (guards against backends that skip it).
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec = vec / norms

        # Atomic write of npy + sidecar.
        from io import BytesIO

        buf = BytesIO()
        np.save(buf, vec, allow_pickle=False)
        _atomic_write_bytes(self._embed_path, buf.getvalue())

        from datetime import datetime, timezone

        manifest = {
            "model_name": EMBED_MODEL_NAME,
            "dim": EMBED_DIM,
            "count": int(vec.shape[0]),
            "corpus_checksum": self._corpus_checksum,
            "normalize": True,
            "built_iso": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "build_seconds": round(time.time() - t0, 2),
        }
        _atomic_write_json(self._embed_manifest_path, manifest)
        self._embeddings = vec

    # ---------- retrieve ----------

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        k: int = 5,
    ) -> list[RetrievalResult]:
        if query is None or not str(query).strip():
            return []
        if k <= 0:
            return []

        # 1. Resolve company filter.
        if company is None:
            company_canon = "none"
        else:
            try:
                company_canon = normalize_company(company)
            except ValueError:
                # Invalid company string -> return empty. The caller can decide to retry.
                return []

        if company_canon == "none":
            pool: list[int] = list(range(len(self._chunks)))
        else:
            pool = list(self._company_idxs.get(company_canon, []))
            if not pool:
                return []

        # 2. BM25 stage.
        q_tokens = _tokenize(query)
        if q_tokens:
            scores = self._bm25.get_scores(q_tokens)  # full N
            # Build (score, chunk_id, idx) tuples within the pool.
            bm25_pool = [(float(scores[i]), self._chunks[i]["id"], i) for i in pool]
            # Sort: score desc, chunk_id asc.
            bm25_pool.sort(key=lambda r: (-r[0], r[1]))
            bm25_top = [(idx, rank) for rank, (_, _, idx) in enumerate(bm25_pool[:BM25_TOPN], start=1)]
        else:
            bm25_top = []

        # 3. Dense stage.
        if self._embeddings is None:
            raise RuntimeError("embeddings not loaded")
        embedder = self._get_embedder()
        q_vec = embedder.encode([query], batch_size=1)
        if q_vec.shape != (1, EMBED_DIM):
            raise RuntimeError(f"unexpected query embedding shape: {q_vec.shape}")
        q = q_vec[0]
        # Renormalize defensively.
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        # Score the entire pool: dot product since both sides are L2-normalized.
        pool_arr = np.asarray(pool, dtype=np.int64)
        sims = self._embeddings[pool_arr] @ q  # shape (|pool|,)
        dense_rows = [
            (float(sims[k_]), self._chunks[pool_arr[k_]]["id"], int(pool_arr[k_]))
            for k_ in range(len(pool_arr))
        ]
        dense_rows.sort(key=lambda r: (-r[0], r[1]))
        dense_top = [(idx, rank) for rank, (_, _, idx) in enumerate(dense_rows[:DENSE_TOPN], start=1)]

        # 4. RRF fusion.
        rrf_scores: dict[int, float] = {}
        bm25_rank_map: dict[int, int] = {}
        dense_rank_map: dict[int, int] = {}
        for idx, rank in bm25_top:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            bm25_rank_map[idx] = rank
        for idx, rank in dense_top:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            dense_rank_map[idx] = rank

        if not rrf_scores:
            return []

        rrf_rows = [
            (rrf_scores[idx], self._chunks[idx]["id"], idx) for idx in rrf_scores
        ]
        rrf_rows.sort(key=lambda r: (-r[0], r[1]))
        rrf_top = rrf_rows[:RRF_TOPN]

        # 5. Optional rerank.
        rerank_scores: dict[int, float] | None = None
        if self._enable_rerank:
            rerank_scores = self._rerank_candidates(query, [idx for _, _, idx in rrf_top])

        # 6. Build final result set.
        results: list[RetrievalResult] = []
        for rrf_score, _cid, idx in rrf_top:
            chunk = self._chunks[idx]
            rerank_score = (
                rerank_scores.get(idx) if rerank_scores is not None else None
            )
            if rerank_score is not None:
                final = max(0.0, min(1.0, float(rerank_score)))
            else:
                final = min(1.0, float(rrf_score) / MAX_RRF_SCORE)
            results.append(
                RetrievalResult(
                    chunk_id=chunk["id"],
                    company=chunk["company"],
                    path=chunk["path"],
                    title=chunk["title"],
                    heading_path=list(chunk.get("heading_path") or []),
                    source_url=chunk.get("source_url"),
                    text=chunk["text"],
                    n_tokens=int(chunk.get("n_tokens", 0)),
                    bm25_rank=bm25_rank_map.get(idx),
                    dense_rank=dense_rank_map.get(idx),
                    rrf_score=float(rrf_score),
                    rerank_score=(None if rerank_score is None else float(rerank_score)),
                    score=float(final),
                )
            )

        # 7. Sort by (score desc, chunk_id asc) and take top-k.
        results.sort(key=lambda r: (-r.score, r.chunk_id))
        return results[:k]

    # ---------- rerank ----------

    def _get_llm_client(self):
        """Lazy-init the Nebius (OpenAI-compatible) client via llm.get_client."""
        if self._llm_client is not None:
            return self._llm_client
        if self._llm_attempted:
            return None
        self._llm_attempted = True
        try:
            from llm import get_client, LLMError  # noqa: E402

            try:
                self._llm_client = get_client()
                return self._llm_client
            except LLMError:
                # Missing key — silently fall back to RRF; main.py prints the real error.
                return None
        except Exception as exc:
            print(f"warning: failed to init LLM client: {exc}", file=sys.stderr)
            return None

    def _rerank_candidates(
        self, query: str, idxs: list[int]
    ) -> dict[int, float] | None:
        """Score ``idxs`` via the LLM; return idx->score map. None on failure.

        Returns None to signal the caller to fall back to RRF scoring.
        """
        if not idxs:
            return {}

        client = self._get_llm_client()
        if client is None:
            return None

        # Build the user prompt.
        q = (query or "").strip()
        if len(q) > USER_QUERY_CAP_CHARS:
            q = q[:USER_QUERY_CAP_CHARS]

        lines = ["QUESTION:", q, "", "CHUNKS:"]
        for i, idx in enumerate(idxs, start=1):
            c = self._chunks[idx]
            heading = " > ".join(c.get("heading_path") or [])
            head_label = f"{c['title']}" + (f" > {heading}" if heading else "")
            text = c["text"]
            if len(text) > USER_CHUNK_CAP_CHARS:
                text = text[:USER_CHUNK_CAP_CHARS]
            lines.append(f"[{i}] [{c['company']}] {head_label}")
            lines.append(text)
            lines.append("")
        user_msg = "\n".join(lines).strip()

        for attempt in (1, 2):
            try:
                resp = client.chat.completions.create(
                    model=self._rerank_model,
                    max_tokens=RERANK_MAX_TOKENS,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_RERANK},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = ""
                try:
                    raw = (resp.choices[0].message.content or "").strip()
                except Exception:
                    raw = ""
                # Strip ```json fences if present.
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                scores = data.get("scores")
                if not isinstance(scores, list):
                    raise ValueError("'scores' is not a list")
                # Validate every input id is covered.
                seen_ids: dict[int, float] = {}
                for entry in scores:
                    if not isinstance(entry, dict):
                        raise ValueError("score entry is not an object")
                    sid = entry.get("id")
                    sval = entry.get("score")
                    if not isinstance(sid, int):
                        raise ValueError(f"bad id: {sid!r}")
                    if not isinstance(sval, (int, float)):
                        raise ValueError(f"bad score: {sval!r}")
                    seen_ids[sid] = float(sval)
                expected = set(range(1, len(idxs) + 1))
                if set(seen_ids.keys()) != expected:
                    missing = expected - set(seen_ids.keys())
                    raise ValueError(f"missing ids in rerank response: {sorted(missing)}")
                # Map back to chunk indices.
                return {idxs[i - 1]: seen_ids[i] for i in expected}
            except Exception as exc:
                if attempt == 1:
                    print(
                        f"warning: rerank attempt {attempt} failed ({exc}); retrying",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"warning: rerank failed twice ({exc}); falling back to RRF",
                    file=sys.stderr,
                )
                return None
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_human(rows: Iterable[RetrievalResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(rows, start=1):
        rerank = f"{r.rerank_score:.2f}" if r.rerank_score is not None else "n/a"
        lines.append(
            f"[{i}] score={r.score:.2f}  rrf={r.rrf_score:.4f}  "
            f"rerank={rerank}  {r.chunk_id}"
        )
        lines.append(f"    title: {r.title}")
        lines.append(f"    path: {r.path}")
        preview = r.text.replace("\n", " ").strip()
        # First 200 chars after the prefix line. The corpus prepends
        # "<title> [> heading_path]\n\n". Slice past it if present.
        # Simple: drop everything up to and including the first double-newline.
        body = r.text
        sep = body.find("\n\n")
        if sep >= 0:
            body = body[sep + 2 :]
        body_oneline = body.replace("\n", " ").strip()
        lines.append(f"    preview: {body_oneline[:200]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_jsonl(rows: Iterable[RetrievalResult]) -> str:
    return "\n".join(
        json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) for r in rows
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid BM25+dense retriever")
    ap.add_argument("--query", required=True, help="user question / query string")
    ap.add_argument(
        "--company",
        default=None,
        help="filter by company (hackerrank|claude|visa|none); omit for cross-company",
    )
    ap.add_argument("--k", type=int, default=5, help="number of results to return")
    ap.add_argument(
        "--no-rerank",
        action="store_true",
        help="disable LLM reranking (use raw RRF score)",
    )
    ap.add_argument("--json", action="store_true", help="print full results as JSONL")
    ap.add_argument(
        "--index-dir",
        default="data/index",
        help="path to index dir (default: data/index relative to repo root)",
    )
    ap.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help=f"Nebius model id for rerank (default: {DEFAULT_RERANK_MODEL})",
    )
    args = ap.parse_args(argv)

    r = Retriever(
        index_dir=args.index_dir,
        rerank_model=args.rerank_model,
        enable_rerank=not args.no_rerank,
    )
    rows = r.retrieve(args.query, company=args.company, k=args.k)
    if args.json:
        print(_format_jsonl(rows))
    else:
        print(_format_human(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
