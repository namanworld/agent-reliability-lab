"""find_code: find code by describing what it does (hybrid keyword + semantic search).

search_code is for exact text ("def percentile", an error message). find_code is
for questions in words ("how is the index cache rebuilt?"), where the exact
identifier isn't known yet. It returns whole functions/classes with line ranges,
so the model can go straight to read_file.

Try it directly (the first run downloads the ~67 MB embedding model):
    uv run python -m pulse.tools.find_code "how is the index cache rebuilt"
    uv run python -m pulse.tools.find_code "how is the index cache rebuilt" --mode bm25
    uv run python -m pulse.tools.find_code "turn tool errors into messages" --mode semantic --no-path-weight

Keep example queries here different from evals/suites/retrieval.yaml: this file is
indexed, so a copied query would "find itself" and inflate the scores.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pulse.retrieval.embeddings import Embedder, FastEmbedder, HashingEmbedder
from pulse.retrieval.index import CodeIndex
from pulse.tools.registry import Tool

SNIPPET_LINES = 6

_embedders: dict[str, Embedder | None] = {}
_indexes: dict[tuple[str, str], CodeIndex] = {}
_notes: dict[str, str] = {}


def get_embedder(kind: str | None = None) -> Embedder | None:
    """fastembed by default; PULSE_EMBEDDER=hashing for offline tests, =none for BM25 only."""
    kind = kind or os.environ.get("PULSE_EMBEDDER", "fastembed")
    if kind not in _embedders:
        if kind == "none":
            _embedders[kind] = None
        elif kind == "hashing":
            _embedders[kind] = HashingEmbedder()
        else:
            try:
                _embedders[kind] = FastEmbedder()
            except Exception as e:  # noqa: BLE001 - no model available: keyword search still works
                _embedders[kind] = None
                _notes[kind] = f"semantic search unavailable ({type(e).__name__}); keyword results only"
    return _embedders[kind]


def get_index(root: Path, kind: str | None = None) -> tuple[CodeIndex, Embedder | None]:
    kind = kind or os.environ.get("PULSE_EMBEDDER", "fastembed")
    embedder = get_embedder(kind)
    key = (str(root.resolve()), kind)
    # Rebuild checks are cheap (file modification times), so do them on every call.
    _indexes[key] = CodeIndex.build(root, embedder)
    return _indexes[key], embedder


class FindCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3, max_length=300,
        description="What you're looking for, in words, e.g. 'where is the search index cache rebuilt'.",
    )
    max_results: int = Field(default=5, ge=1, le=10, description="How many code chunks to return.")


def run_find_code(args: FindCodeArgs, repo_root: Path, mode: str = "hybrid",
                  use_path_weight: bool = True, embedder_kind: str | None = None) -> str:
    index, embedder = get_index(repo_root, embedder_kind)
    hits = index.search(args.query, embedder, mode=mode, k=args.max_results,
                        use_path_weight=use_path_weight)
    if not hits:
        return f"No code found for {args.query!r}. Try different words, or search_code for exact text."

    lines = []
    note = _notes.get(embedder_kind or os.environ.get("PULSE_EMBEDDER", "fastembed"))
    if note:
        lines.append(f"[{note}]")
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        label = f"{c.kind} {c.name}".strip()
        lines.append(f"{n}. {c.location}  ({label})")
        for line in c.text.splitlines()[:SNIPPET_LINES]:
            lines.append(f"     {line[:160]}")
    return "\n".join(lines)


FIND_CODE = Tool(
    name="find_code",
    description=(
        "Find code by describing what it does, when you don't know the exact name. Returns the "
        "best-matching functions/classes with file paths and line ranges. Use search_code instead "
        "for exact text such as a function name or an error message."
    ),
    args_model=FindCodeArgs,
    run=run_find_code,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run find_code directly, without a model.")
    p.add_argument("query")
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--mode", choices=["hybrid", "bm25", "semantic"], default="hybrid")
    p.add_argument("--no-path-weight", action="store_true")
    p.add_argument("--embedder", choices=["fastembed", "hashing", "none"], default=None)
    p.add_argument("--repo", type=Path, default=Path("."))
    a = p.parse_args()
    args = FindCodeArgs(query=a.query, max_results=a.max_results)
    print(run_find_code(args, a.repo, mode=a.mode, use_path_weight=not a.no_path_weight,
                        embedder_kind=a.embedder))


if __name__ == "__main__":
    main()
