"""Score search quality on its own: does the expected file come back near the top?

No model calls, so it costs nothing and runs in seconds. Compare modes side by side:

    uv run python scripts/eval_retrieval.py                               # bm25, semantic, hybrid
    uv run python scripts/eval_retrieval.py --modes hybrid -v             # show each query's top 5
    uv run python scripts/eval_retrieval.py --no-path-weight              # without the src-first boost
    uv run python scripts/eval_retrieval.py --embedder hashing            # offline, not semantic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pulse.tools.find_code import get_index


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate code retrieval.")
    p.add_argument("--suite", type=Path, default=Path("evals/suites/retrieval.yaml"))
    p.add_argument("--modes", default="bm25,semantic,hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-path-weight", action="store_true")
    p.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    cases = yaml.safe_load(a.suite.read_text())["cases"]
    index, embedder = get_index(a.repo, a.embedder)
    if embedder is None:
        print("embedding model unavailable: semantic and hybrid fall back to keywords only\n")
    print(f"{len(index.chunks)} chunks indexed, {len(cases)} queries, "
          f"path weight {'off' if a.no_path_weight else 'on'}\n")

    rows = []
    for mode in a.modes.split(","):
        hits_at_1 = hits_at_k = 0
        reciprocal_ranks = []
        misses = []
        for case in cases:
            results = index.search(case["query"], embedder, mode=mode, k=a.k,
                                   use_path_weight=not a.no_path_weight)
            paths = [h.chunk.path for h in results]
            rank = paths.index(case["expect"]) + 1 if case["expect"] in paths else None
            hits_at_1 += rank == 1
            hits_at_k += rank is not None
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            if rank != 1:
                misses.append((case, rank, results))
        n = len(cases)
        rows.append((mode, hits_at_1 / n, hits_at_k / n, sum(reciprocal_ranks) / n))

        if a.verbose:
            print(f"== {mode}: queries where the expected file wasn't first")
            for case, rank, results in misses:
                where = f"rank {rank}" if rank else f"not in top {a.k}"
                print(f"  [{where}] {case['query']}  (expected {case['expect']})")
                for r in results[:3]:
                    print(f"        {r.chunk.location}  {r.chunk.name}")
            print()

    print(f"{'mode':<10}{'recall@1':>10}{'recall@' + str(a.k):>10}{'MRR':>8}")
    for mode, r1, rk, mrr in rows:
        print(f"{mode:<10}{r1:>10.0%}{rk:>10.0%}{mrr:>8.2f}")


if __name__ == "__main__":
    main()
