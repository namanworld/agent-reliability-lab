"""The code index: chunks + keyword search + semantic search, combined.

How a query is answered (mode="hybrid"):

    query ──► BM25 ranking      (exact words: good for names like `is_blocked`)
          └─► semantic ranking  (meaning: good for "where are secrets blocked?")
                     │
          Reciprocal Rank Fusion: score = sum over lists of 1 / (60 + rank)
                     │
          × path weight (source code first, notes and tests later)
                     │
                 top results

Why RRF: the two searches score on different scales (BM25 ~0-20, cosine 0-1), so
their scores can't be added. RRF only uses each result's *rank*, so a chunk that
either search puts near the top ends up near the top. 60 is the usual constant;
it stops the first few ranks from dominating.

The index is cached in .pulse_index/. When files change, only those files are
re-chunked and re-embedded; everything else is reused.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pulse.retrieval.bm25 import BM25
from pulse.retrieval.chunks import Chunk, chunk_file
from pulse.retrieval.embeddings import Embedder
from pulse.tools.paths import BLOCKED_DIRS, SEARCH_SKIPPED_DIRS, is_blocked, path_weight

INDEXED_EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".toml", ".txt"}
MAX_FILE_BYTES = 300_000
RRF_K = 60
CACHE_DIR = ".pulse_index"


def list_files(root: Path) -> list[str]:
    """Every indexable file, using the same rules as the tools (no secrets, no symlinks)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in BLOCKED_DIRS and not d.startswith(".")
            and os.path.normpath(os.path.join(rel_dir, d)) not in SEARCH_SKIPPED_DIRS
        )
        for name in sorted(filenames):
            rel = os.path.normpath(os.path.join(rel_dir, name))
            full = Path(dirpath) / name
            if (Path(name).suffix in INDEXED_EXTENSIONS and not name.startswith(".")
                    and not is_blocked(rel) and not full.is_symlink()
                    and full.stat().st_size <= MAX_FILE_BYTES):
                found.append(rel)
    return found


@dataclass
class Hit:
    chunk: Chunk
    score: float


class CodeIndex:
    def __init__(self, root: Path, chunks: list[Chunk], vectors: np.ndarray | None):
        self.root = root
        self.chunks = chunks
        self.vectors = vectors  # one row per chunk, or None when semantic search is off
        self.bm25 = BM25([embedding_text(c) for c in chunks])
        self.reembedded = 0  # chunks embedded in the last build (0 = everything reused)

    # ---------- building and caching ----------

    @classmethod
    def build(cls, root: Path, embedder: Embedder | None, use_cache: bool = True) -> CodeIndex:
        """Load the cached index, re-chunking and re-embedding only files that changed.

        Embedding is the slow part (M1.5: a full rebuild took ~40 s inside a user's
        request). The cache remembers each file's modification time, chunks and
        vectors, so an unchanged file is reused and only changed files are embedded.
        """
        root = root.resolve()
        files = list_files(root)
        fingerprint = {f: (root / f).stat().st_mtime_ns for f in files}
        cache = root / CACHE_DIR / (embedder.name if embedder else "bm25")

        # 1. What the cache knows: file -> list of (chunk, vector) from the last build.
        cached: dict[str, list[tuple[Chunk, np.ndarray | None]]] = {}
        old_fingerprint: dict[str, int] = {}
        if use_cache and (cache / "manifest.json").exists():
            old_fingerprint = json.loads((cache / "manifest.json").read_text())
            old_chunks = [Chunk(**c) for c in json.loads((cache / "chunks.json").read_text())]
            old_vectors = np.load(cache / "vectors.npy") if embedder else [None] * len(old_chunks)
            for chunk, vector in zip(old_chunks, old_vectors, strict=True):
                cached.setdefault(chunk.path, []).append((chunk, vector))

        # 2. Reuse unchanged files; chunk changed or new files (their vectors are None for now).
        chunks: list[Chunk] = []
        vectors: list[np.ndarray | None] = []
        for f in files:
            if f in cached and old_fingerprint.get(f) == fingerprint[f]:
                for chunk, vector in cached[f]:
                    chunks.append(chunk)
                    vectors.append(vector)
            else:
                for chunk in chunk_file(f, (root / f).read_text(errors="replace")):
                    chunks.append(chunk)
                    vectors.append(None)

        # 3. Embed only the chunks that have no vector yet.
        missing = [i for i, v in enumerate(vectors) if v is None]
        matrix = None
        if embedder is not None:
            new = embedder.embed_documents([embedding_text(chunks[i]) for i in missing])
            for i, vector in zip(missing, new, strict=True):
                vectors[i] = vector
            matrix = np.array(vectors, dtype=np.float32) if chunks else np.zeros((0, 1), np.float32)

        # 4. Save the cache if anything changed.
        if use_cache and (missing or fingerprint != old_fingerprint):
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks]))
            (cache / "manifest.json").write_text(json.dumps(fingerprint))
            if matrix is not None:
                np.save(cache / "vectors.npy", matrix)

        index = cls(root, chunks, matrix)
        index.reembedded = len(missing) if embedder is not None else 0
        return index

    # ---------- searching ----------

    def rank_bm25(self, query: str) -> list[int]:
        scores = self.bm25.scores(query)
        return [i for i in np.argsort(scores)[::-1] if scores[i] > 0]

    def rank_semantic(self, query_vector: np.ndarray) -> list[int]:
        if self.vectors is None or len(self.chunks) == 0:
            return []
        similarity = self.vectors @ query_vector  # cosine, because all vectors have length 1
        return list(np.argsort(similarity)[::-1])

    def search(
        self,
        query: str,
        embedder: Embedder | None,
        mode: str = "hybrid",
        k: int = 5,
        use_path_weight: bool = True,
        candidates: int = 50,
    ) -> list[Hit]:
        lists: list[list[int]] = []
        if mode in ("bm25", "hybrid"):
            lists.append(self.rank_bm25(query)[:candidates])
        if mode in ("semantic", "hybrid") and embedder is not None:
            lists.append(self.rank_semantic(embedder.embed_query(query))[:candidates])

        fused: dict[int, float] = {}
        for ranking in lists:
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        if use_path_weight:
            fused = {i: s * path_weight(self.chunks[i].path) for i, s in fused.items()}

        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(self.chunks[i], score) for i, score in best]


def embedding_text(chunk: Chunk) -> str:
    """What gets embedded/indexed: the path and name help, the code is truncated."""
    return f"{chunk.path} {chunk.name}\n{chunk.text[:2000]}"
