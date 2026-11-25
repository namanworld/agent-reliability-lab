"""M1.1 demo: ask Claude a question with one tool available, then validate its tool call.

Examples:
    uv run python scripts/tool_call_demo.py
    uv run python scripts/tool_call_demo.py --prompt "Hi, how are you?"     # no tool needed
    uv run python scripts/tool_call_demo.py --model claude-haiku-5-5
    uv run python scripts/tool_call_demo.py --thinking default
    uv run python scripts/tool_call_demo.py --record tests/fixtures/search_call.jsonl
    uv run python scripts/tool_call_demo.py --replay tests/fixtures/search_call.jsonl
    uv run python scripts/tool_call_demo.py --break-args               # simulate a bad call
    uv run python scripts/tool_call_demo.py --show-schema
    uv run python scripts/tool_call_demo.py --runs 6                 # 1 warm-up + 5 measured
    uv run python scripts/tool_call_demo.py --runs 6 --no-tools      # same prompt, no tools sent
    uv run python scripts/tool_call_demo.py --runs 6 --terse         # ask for no preamble text
    uv run python scripts/tool_call_demo.py --execute                # also run the tool (M1.2)
    uv run python scripts/tool_call_demo.py --execute --prompt "Show me what's in ../../.ssh"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arl.metrics import percentile
from pulse.llm.anthropic_provider import AnthropicProvider
from pulse.llm.base import ToolCall
from pulse.llm.replay import ReplayProvider
from pulse.tools.registry import ToolCallError, ToolExecutionError, validate_call
from pulse.tools.search_code import SEARCH_CODE

SYSTEM = (
    "You are a coding assistant working in a Python repository. "
    "Use the search_code tool to find code before answering questions about it."
)
TERSE = (
    " Call tools directly, without announcing them first. Keep answers short."
)
TOOLS = {SEARCH_CODE.name: SEARCH_CODE}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Get one tool call from Claude and validate it.")
    p.add_argument("--prompt", default="Where is the percentile function defined in this repo?")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--thinking", choices=["off", "default"], default="off")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-system", action="store_true", help="send no system prompt")
    p.add_argument("--record", type=Path, help="save the raw stream to this .jsonl file")
    p.add_argument("--replay", type=Path, help="replay a recorded .jsonl instead of calling Claude")
    p.add_argument("--break-args", action="store_true",
                   help="corrupt the tool call's arguments to see validation fail")
    p.add_argument("--show-schema", action="store_true", help="print the tool schema sent to Claude")
    p.add_argument("--runs", type=int, default=1,
                   help="repeat the call on one connection; the first counts as warm-up")
    p.add_argument("--no-tools", action="store_true", help="send the same prompt without tools")
    p.add_argument("--execute", action="store_true", help="run the validated tool call")
    p.add_argument("--repo", type=Path, default=Path("."), help="repository the tool may search")
    p.add_argument("--terse", action="store_true",
                   help="system prompt asks for tool calls without preamble text")
    return p.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = parse_args()
    if args.show_schema:
        print(json.dumps(SEARCH_CODE.spec(), indent=2), "\n")

    if args.replay:
        provider = ReplayProvider(args.replay)
        source = f"replay of {args.replay}"
    else:
        provider = AnthropicProvider(model=args.model, record_to=args.record)
        source = f"live {args.model}, thinking={args.thinking}"

    system = None if args.no_system else SYSTEM + (TERSE if args.terse else "")
    tools = None if args.no_tools else [t.spec() for t in TOOLS.values()]

    def call_once():
        return provider.complete(
            [{"role": "user", "content": args.prompt}],
            tools=tools,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
            system=system,
        )

    if args.runs > 1:
        print(f"{args.runs} calls on one connection (tools={'off' if args.no_tools else 'on'}, "
              f"terse={args.terse}); the first is warm-up\n")
        ttfts, totals = [], []
        for i in range(args.runs):
            resp = call_once()
            label = "warm-up" if i == 0 else f"run {i}"
            print(f"{label:<8} ttft={resp.ttft_ms or 0:6.0f} ms  total={resp.total_ms or 0:6.0f} ms  "
                  f"out={resp.output_tokens}  stop={resp.stop_reason}  "
                  f"tool_calls={len(resp.tool_calls)}  text_chars={len(resp.text.strip())}")
            if i > 0:
                if resp.ttft_ms is not None:
                    ttfts.append(resp.ttft_ms)
                if resp.total_ms is not None:
                    totals.append(resp.total_ms)
        if ttfts:
            print(f"\nmedian over n={len(ttfts)}: ttft={percentile(ttfts, 50):.0f} ms  "
                  f"total={percentile(totals, 50):.0f} ms\n")
        print("details of the last run:")
    else:
        resp = call_once()

    print(f"source:      {source}")
    print(f"stop_reason: {resp.stop_reason}")
    ttft = "-" if resp.ttft_ms is None else f"{resp.ttft_ms:.0f} ms"
    total = "-" if resp.total_ms is None else f"{resp.total_ms:.0f} ms"
    print(f"ttft:        {ttft}   total: {total}")
    print(f"tokens:      in={resp.input_tokens} out={resp.output_tokens} "
          f"thinking={resp.thinking_tokens}")
    if resp.text.strip():
        print(f"text:        {resp.text.strip()!r}")
    if resp.refused:
        print("RESULT:      the model refused (stop_reason=refusal); text may be cut off")
    if resp.empty:
        print("RESULT:      empty response (no text, no tool call) -> ARL check #1 fails")
    if args.record and not args.replay:
        print(f"recorded:    {args.record}")

    for call in resp.tool_calls:
        if args.break_args:
            call = ToolCall(id=call.id, name=call.name,
                            arguments_json='{"query": "", "max_results": 500, "color": "red"}')
        print(f"\ntool call:   {call.name}  id={call.id}")
        print(f"raw args:    {call.arguments_json}")
        try:
            validated = validate_call(call, TOOLS)
            print(f"VALID:       {validated.model_dump()}")
        except ToolCallError as e:
            print(f"INVALID:     {e}")
            continue
        tool = TOOLS[call.name]
        if args.execute and tool.run is not None:
            try:
                result = tool.run(validated, args.repo)
                print(f"RESULT:\n{result}")
            except ToolExecutionError as e:
                print(f"TOOL ERROR (would be sent back to the model): {e}")

    if not resp.tool_calls:
        print("\nno tool call (the model answered directly)")


if __name__ == "__main__":
    main()
