"""Semantic search: turn text into vectors so that similar meaning = nearby vectors.

An embedding model maps a piece of text to a list of numbers (here 384 of them).
Texts about the same thing end up pointing in similar directions, even when they
share no words: "which files are off-limits" can land near `def is_blocked(...)`.
Similarity is the cosine of the angle between two vectors (1.0 = same direction).

Two embedders:

- FastEmbedder: a real model, BAAI/bge-small-en-v1.5 (~67 MB, downloaded once on
  first use), run locally through ONNX: no PyTorch, no API key, no per-call cost.
- HashingEmbedder: NOT semantic. It hashes words into a vector, so it only knows
  about shared words. It exists so tests run offline and instantly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from pulse.retrieval.bm25 import tokenize

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every vector to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        from fastembed import (
            TextEmbedding,  # imported here so tests don't need the model
        )

        self.name = model.replace("/", "_")
        # Keep the downloaded model in ~/.cache so it survives reboots (the default is /tmp).
        self.model = TextEmbedding(model, cache_dir=str(Path.home() / ".cache" / "fastembed"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # passage_embed / query_embed add the prefixes this model was trained with.
        return normalize(np.array(list(self.model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self.model.query_embed([text])))
        return normalize(np.array(vector, dtype=np.float32))


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.name = f"hashing{dim}"
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in tokenize(text):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(self._embed(text))
