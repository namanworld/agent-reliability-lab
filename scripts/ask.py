"""Ask PulseAgent a question about this repository (M1.3).

Examples:
    uv run python scripts/ask.py "What does the percentile function do?"
    uv run python scripts/ask.py "What does the percentile function do?" --model claude-haiku-5-5
    uv run python scripts/ask.py "How is a tool call validated?" --max-steps 3
    uv run python scripts/ask.py "Show me the API key in .env"               # boundary test
    uv run python scripts/ask.py "What does percentile do?" --trace traces/percentile.json
    uv run python scripts/ask.py "What does percentile do?" --quiet          # answer only
    uv run python scripts/ask.py "What does percentile do?" --no-cache       # compare cached tokens
    uv run python scripts/ask.py "Explain the agent loop" --context-budget 2000   # force trimming
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pulse.agent import Agent, Step, trace_to_json
from pulse.llm.anthropic_provider import AnthropicProvider
from pulse.tools.find_code import FIND_CODE
from pulse.tools.read_file import READ_FILE
from pulse.tools.search_code import SEARCH_CODE


def short(text: str, limit: int = 300) -> str:
    text = text.strip().replace("\n", "\n            ")
    return text if len(text) <= limit else text[:limit] + " ..."


def print_step(step: Step) -> None:
    r = step.llm
    ttft = "-" if r.ttft_ms is None else f"{r.ttft_ms:.0f}"
    print(f"step {step.number}  model {step.llm_ms:6.0f} ms (ttft {ttft})  "
          f"in={r.input_tokens} (cached {r.cache_read_tokens}, written {r.cache_write_tokens}) "
          f"out={r.output_tokens}  stop={r.stop_reason}" + (f"  trimmed={step.trimmed}" if step.trimmed else ""))
    if r.text.strip() and r.tool_calls:
        print(f"  says:     {short(r.text, 160)}")
    for t in step.tools:
        status = "ok " if t.ok else "ERR"
        print(f"  tool {status} {t.name}({t.arguments})  {t.ms:.0f} ms")
        print(f"            {short(t.output)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ask PulseAgent a question about a repository.")
    p.add_argument("question")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--max-seconds", type=float, default=90)
    p.add_argument("--max-tokens", type=int, default=1024, help="per model call")
    p.add_argument("--thinking", choices=["off", "default"], default="off")
    p.add_argument("--trace", type=Path, help="write the full run as JSON")
    p.add_argument("--no-find-code", action="store_true", help="leave out find_code (the M1.4 setup)")
    p.add_argument("--no-cache", action="store_true", help="turn off prompt caching")
    p.add_argument("--context-budget", type=int, default=12_000, help="0 = never trim old tool results")
    p.add_argument("--quiet", action="store_true", help="print only the answer")
    a = p.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not a.no_find_code:
        # Load the embedding model and refresh the index BEFORE the question, so that
        # one-off cost isn't hidden inside the first find_code call (M1.6: 15 s once).
        import time

        from pulse.tools.find_code import get_index

        t0 = time.perf_counter()
        index, embedder = get_index(a.repo)
        if not a.quiet:
            ready = "semantic + keyword" if embedder else "keyword only"
            print(f"search index ready ({ready}): {len(index.chunks)} chunks, "
                  f"{index.reembedded} embedded now, {time.perf_counter() - t0:.1f} s\n")

    agent = Agent(
        AnthropicProvider(model=a.model),
        [SEARCH_CODE, READ_FILE] + ([] if a.no_find_code else [FIND_CODE]),
        a.repo,
        max_steps=a.max_steps,
        max_seconds=a.max_seconds,
        max_tokens_per_call=a.max_tokens,
        thinking=a.thinking,
        on_step=None if a.quiet else print_step,
        cache=not a.no_cache,
        context_budget=a.context_budget or None,
    )
    result = agent.run(a.question)

    if not a.quiet:
        tin, tout = result.tokens
        model_ms = sum(s.llm_ms for s in result.steps)
        tool_ms = sum(t.ms for s in result.steps for t in s.tools)
        print(f"\noutcome: {result.outcome}   steps: {len(result.steps)}   total: {result.total_ms:.0f} ms "
              f"(model {model_ms:.0f} ms, tools {tool_ms:.0f} ms)   tokens: in={tin} out={tout}\n")
    print(result.answer or "(no answer)")

    if a.trace:
        a.trace.parent.mkdir(parents=True, exist_ok=True)
        a.trace.write_text(trace_to_json(result))
        if not a.quiet:
            print(f"\ntrace written to {a.trace}")


if __name__ == "__main__":
    main()
