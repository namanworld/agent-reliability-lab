"""Re-score saved eval results with the current suite and checks, without calling a model.

Use it after fixing a check: old traces are scored again, and every case whose
verdict changed is listed.

    uv run python scripts/rescore.py evals/results/pulse_seed_20261126-005802.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pulse.evals import check, load_suite, result_from_trace


def main() -> None:
    p = argparse.ArgumentParser(description="Re-score saved eval results.")
    p.add_argument("results", type=Path)
    p.add_argument("--suite", type=Path, default=Path("evals/suites/pulse_seed.yaml"))
    a = p.parse_args()

    cases = {c.id: c for c in load_suite(a.suite).cases}
    before, after = defaultdict(list), defaultdict(list)
    for line in a.results.read_text().splitlines():
        row = json.loads(line)
        case = cases.get(row["case"])
        if case is None or row["trace"] is None:
            continue
        failures = check(case, result_from_trace(row["trace"]))
        now = not failures
        before[row["model"]].append(row["passed"])
        after[row["model"]].append(now)
        if now != row["passed"]:
            verdict = "FAIL -> PASS" if now else "PASS -> FAIL"
            print(f"{verdict}  {row['model']:<20} {row['case']}")
            for f in failures:
                print(f"      - {f}")

    print()
    for model in before:
        b, n = sum(before[model]), len(before[model])
        print(f"{model:<20} before {b}/{n}   after {sum(after[model])}/{n}")


if __name__ == "__main__":
    main()
