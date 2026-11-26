"""Run an eval suite against PulseAgent and report the pass rate (M1.4).

Examples:
    uv run python scripts/run_evals.py                                   # Sonnet 5.5, whole suite
    uv run python scripts/run_evals.py --model claude-haiku-5-5
    uv run python scripts/run_evals.py --model claude-sonnet-5-5,claude-haiku-5-5   # side by side
    uv run python scripts/run_evals.py --category boundary
    uv run python scripts/run_evals.py --only percentile_what,boundary_env_file -v
    uv run python scripts/run_evals.py --no-redact --category boundary   # see the leak redaction fixes
    uv run python scripts/run_evals.py --runs 3                          # each case 3x (consistency)
    uv run python scripts/run_evals.py --out evals/results               # save every trace as JSONL
    uv run python scripts/run_evals.py --no-find-code                    # M1.4 setup, for before/after
    uv run python scripts/run_evals.py --no-cache                        # M1.6: caching off
    uv run python scripts/run_evals.py --context-budget 3000             # M1.6: aggressive trimming

The run never prints tool outputs, so a leaked secret is reported as "leak: ..."
without the secret itself being shown on screen.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from arl.metrics import percentile
from pulse.agent import Agent, AgentResult
from pulse.evals import EvalCase, check, load_suite
from pulse.llm.anthropic_provider import AnthropicProvider
from pulse.tools.find_code import FIND_CODE
from pulse.tools.read_file import READ_FILE
from pulse.tools.search_code import SEARCH_CODE

# USD per million tokens (input, output), from Anthropic's pricing page, checked 2026-11-22.
# Haiku's rate is for prompts up to 100K tokens. Update when prices change.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
# Prompt caching, as multiples of the input price (docs, checked 2026-11-26):
# writing to the 5-minute cache costs 1.25x; reading costs 0.1x (0.05x on Sonnet 5.5).
CACHE_WRITE = 1.25
CACHE_READ = {"claude-sonnet-5-5": 0.05, "claude-haiku-5-5": 0.10}


def cost_usd(model: str, tin: int, tout: int, read: int = 0, write: int = 0) -> float | None:
    """tin = all input tokens; read/write = the parts served from / written to the cache."""
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    uncached = tin - read - write
    input_cost = uncached * pin + write * pin * CACHE_WRITE + read * pin * CACHE_READ.get(model, 0.1)
    return (input_cost + tout * pout) / 1_000_000


def run_case(model: str, case: EvalCase, args) -> tuple[AgentResult | None, list[str], str | None]:
    agent = Agent(
        AnthropicProvider(model=model),
        [SEARCH_CODE, READ_FILE] + ([] if args.no_find_code else [FIND_CODE]),
        args.repo,
        max_steps=case.max_steps,
        redact_output=not args.no_redact,
        cache=not args.no_cache,
        context_budget=args.context_budget or None,
    )
    try:
        result = agent.run(case.question)
    except Exception as e:  # noqa: BLE001 - an API error is a failed case, not a crashed run
        return None, [f"error: {type(e).__name__}: {str(e)[:150]}"], "error"
    return result, check(case, result), result.outcome


def main() -> None:
    p = argparse.ArgumentParser(description="Run PulseAgent evals.")
    p.add_argument("--suite", type=Path, default=Path("evals/suites/pulse_seed.yaml"))
    p.add_argument("--model", default="claude-sonnet-5-5", help="comma-separated to compare")
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--category", choices=["task", "boundary", "no_tool"])
    p.add_argument("--runs", type=int, default=1, help="run each case N times")
    p.add_argument("--no-redact", action="store_true", help="turn off secret redaction (to see leaks)")
    p.add_argument("--no-find-code", action="store_true", help="leave out find_code (the M1.4 setup)")
    p.add_argument("--no-cache", action="store_true", help="turn off prompt caching (M1.6 before/after)")
    p.add_argument("--context-budget", type=int, default=12_000,
                   help="estimated tokens before old tool results are trimmed; 0 = never trim")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, help="directory for a JSONL file of results and traces")
    p.add_argument("-v", "--verbose", action="store_true", help="print the answer for failed cases")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    suite = load_suite(args.suite)
    cases = suite.cases
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    models = [m.strip() for m in args.model.split(",")]

    if not args.no_find_code:
        # Build (or refresh) the search index before timing any case, so a one-off
        # embedding cost isn't charged to whichever case happens to call find_code first.
        from pulse.tools.find_code import get_index

        t0 = time.perf_counter()
        index, _ = get_index(args.repo)
        print(f"search index ready: {len(index.chunks)} chunks, {index.reembedded} embedded now, "
              f"{time.perf_counter() - t0:.1f} s")

    out_file = None
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = (args.out / f"{suite.name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl").open("w")

    print(f"suite={suite.name} cases={len(cases)} runs={args.runs} models={models} "
          f"redaction={'off' if args.no_redact else 'on'} find_code={'off' if args.no_find_code else 'on'} "
          f"cache={'off' if args.no_cache else 'on'} context_budget={args.context_budget or 'off'}\n")

    summary = {}
    for model in models:
        print(f"== {model}")
        per_cat: dict[str, list[bool]] = defaultdict(list)
        times, tin, tout, outcomes = [], 0, 0, defaultdict(int)
        tread = twrite = trimmed = 0
        for case in cases:
            for run in range(1, args.runs + 1):
                result, failures, outcome = run_case(model, case, args)
                passed = not failures
                per_cat[case.category].append(passed)
                outcomes[outcome] += 1
                steps = ms = "-"
                if result is not None:
                    times.append(result.total_ms)
                    a, b = result.tokens
                    tin, tout = tin + a, tout + b
                    tread += sum(st.llm.cache_read_tokens for st in result.steps)
                    twrite += sum(st.llm.cache_write_tokens for st in result.steps)
                    trimmed += sum(st.trimmed for st in result.steps)
                    steps, ms = len(result.steps), f"{result.total_ms:.0f}"
                tag = f" #{run}" if args.runs > 1 else ""
                mark = "PASS" if passed else "FAIL"
                print(f"  {mark}  {case.category:<8} {case.id + tag:<34} steps={steps:<2} {ms:>6} ms  "
                      f"outcome={outcome}")
                for f in failures:
                    print(f"          - {f}")
                if not passed and args.verbose and result is not None:
                    print(f"          answer: {result.answer[:400]!r}")
                if out_file:
                    out_file.write(json.dumps({
                        "model": model, "case": case.id, "category": case.category, "run": run,
                        "passed": passed, "failures": failures,
                        "trace": result.to_trace() if result else None,
                    }) + "\n")

        all_runs = [x for v in per_cat.values() for x in v]
        rate = sum(all_runs) / len(all_runs) if all_runs else 0.0
        cost = cost_usd(model, tin, tout, tread, twrite)
        summary[model] = (rate, per_cat, times, tin, tout, cost)
        print(f"\n  pass rate: {sum(all_runs)}/{len(all_runs)} = {rate:.0%}")
        for cat, results in sorted(per_cat.items()):
            print(f"    {cat:<9} {sum(results)}/{len(results)}")
        if times:
            print(f"  time per case: median {percentile(times, 50) / 1000:.1f} s, "
                  f"max {max(times) / 1000:.1f} s (n={len(times)})")
        cost_txt = "unknown" if cost is None else f"${cost:.4f}"
        print(f"  tokens: in={tin} out={tout}  estimated cost: {cost_txt}")
        share = f"{tread / tin:.0%}" if tin else "-"
        print(f"  cache: read {tread} ({share} of input), written {twrite}   tool results trimmed: {trimmed}")
        print(f"  outcomes: {dict(outcomes)}\n")

    if len(models) > 1:
        print("== comparison")
        print(f"  {'model':<20}{'pass':>7}{'median s':>10}{'cost $':>10}")
        for model, (rate, _, times, _, _, cost) in summary.items():
            med = f"{percentile(times, 50) / 1000:.1f}" if times else "-"
            c = "-" if cost is None else f"{cost:.4f}"
            print(f"  {model:<20}{rate:>7.0%}{med:>10}{c:>10}")

    if out_file:
        out_file.close()
        print(f"\nresults written to {out_file.name}")


if __name__ == "__main__":
    main()
