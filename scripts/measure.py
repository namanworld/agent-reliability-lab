"""Measure time-to-first-token (TTFT) and time-per-output-token (TPOT).

Every setting is a command-line flag, so you can change one thing at a time
and compare the results.

Examples:
    uv run python scripts/measure.py                        # Claude Sonnet 5.5, 1 run
    uv run python scripts/measure.py --runs 10              # medians over 10 runs
    uv run python scripts/measure.py --pad-words 750        # a ~1,000-token prompt
    uv run python scripts/measure.py --model claude-haiku-5-5
    uv run python scripts/measure.py --provider local       # mlx_lm.server on :8080
    uv run python scripts/measure.py --provider local --think
    uv run python scripts/measure.py --bust-cache           # defeat prefix caching
    uv run python scripts/measure.py --events               # print every stream event
    uv run python scripts/measure.py --runs 10 --pad-words 0,750,2250   # interleaved comparison
    uv run python scripts/measure.py --runs 10 --model claude-sonnet-5-5,claude-haiku-5-5
    uv run python scripts/measure.py --long                 # long answer, so TPOT is measurable
    uv run python scripts/measure.py --long --thinking off  # Claude: no up-front thinking
    uv run python scripts/measure.py --thinking summarized --events   # see the thinking stream

Definitions (see NOTES.md, M0.1 Step 2):
    ttft_visible  request sent -> first non-whitespace answer text (headline number)
    ttft_model    request sent -> start of generation (first text, or the start of a
                  thinking block when the thinking text is hidden)
    tpot          (last answer text time - ttft_visible) / (answer tokens - 1), where
                  answer tokens = output tokens - thinking tokens. Shown only when the
                  answer arrived in at least MIN_DELTAS pieces.
                  Claude sends several tokens per event, so a short answer can arrive
                  in 1-2 events and give a meaningless TPOT.

Claude 5.x models think by default (adaptive thinking), and on Sonnet 5.5 the
thinking text is hidden by default: the stream carries empty thinking deltas while
the tokens are still generated and billed. --thinking off sends
the right "off" value for each model: {"type": "between_tools"} on Sonnet 5.5,
{"type": "disabled"} on Haiku 5.5 (see pulse.llm.anthropic_provider.THINKING_OFF).

Comma-separated --model / --pad-words values are run interleaved: every round runs
each setting once in a shuffled order, so slow minutes on the network hit all
settings equally instead of whichever ran last.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import uuid
from dataclasses import dataclass
from functools import cache
from itertools import product

from arl.metrics import percentile
from pulse.llm.anthropic_provider import thinking_off_param

FILLER = "The quick brown fox jumps over the lazy dog. "  # 9 words
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "local": "default_model"}
MIN_DELTAS = 5
LONG_PROMPT = "Write a 300-word story about a lighthouse keeper."


@dataclass
class RunResult:
    ttft_model_ms: float | None = None
    ttft_visible_ms: float | None = None
    last_text_ms: float | None = None
    total_ms: float = 0.0
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_exact: bool = True
    thinking_tokens: int | None = None
    thinking_start_ms: float | None = None
    stop_reason: str | None = None
    text_deltas: int = 0
    text: str = ""

    @property
    def tpot_ms(self) -> float | None:
        if (
            self.ttft_visible_ms is None
            or self.last_text_ms is None
            or not self.output_tokens
            or self.output_tokens < 2
            or self.text_deltas < MIN_DELTAS
        ):
            return None
        # Hidden thinking tokens are generated before any visible text, so they
        # must not be spread over the visible-text window.
        tokens = self.output_tokens - (self.thinking_tokens or 0)
        if tokens < 2:
            return None
        return (self.last_text_ms - self.ttft_visible_ms) / (tokens - 1)

    @property
    def cut_off(self) -> bool:
        return self.stop_reason in ("max_tokens", "length")


def record_text(result: RunResult, elapsed_ms: float, kind: str, text: str) -> None:
    """Update the timing marks when generated text arrives.

    kind is "answer" (visible text), "tool" (tool-call arguments) or "thinking".
    """
    if not text:
        return
    if kind in ("answer", "tool"):
        result.text_deltas += 1
    if result.ttft_model_ms is None:
        result.ttft_model_ms = elapsed_ms
    if kind in ("answer", "tool") and text.strip() and result.ttft_visible_ms is None:
        result.ttft_visible_ms = elapsed_ms
    result.last_text_ms = elapsed_ms
    if kind == "answer":
        result.text += text


# ---------- Anthropic (Claude) ----------


@cache
def anthropic_client():
    from anthropic import Anthropic

    return Anthropic()  # reads ANTHROPIC_API_KEY; one client = one reused connection


def run_anthropic(args: argparse.Namespace, model: str, prompt: str) -> RunResult:
    client = anthropic_client()
    r = RunResult()
    t0 = time.perf_counter()
    extra = {}
    if args.thinking == "off":
        extra["thinking"] = thinking_off_param(model)  # differs per model
    elif args.thinking == "summarized":
        extra["thinking"] = {"type": "adaptive", "display": "summarized"}
    stream = client.messages.create(
        model=model,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        **extra,
    )
    for event in stream:
        ms = (time.perf_counter() - t0) * 1000
        if args.events:
            print(f"  {ms:8.1f} ms  {event.type}", end="")

        if event.type == "content_block_start":
            block_type = event.content_block.type
            if block_type == "thinking" and r.thinking_start_ms is None:
                r.thinking_start_ms = ms
                if r.ttft_model_ms is None:  # hidden thinking: no text, but generation began
                    r.ttft_model_ms = ms
            if args.events:
                print(f"  [{block_type}]", end="")

        elif event.type == "message_start":
            u = event.message.usage
            cached = u.cache_read_input_tokens or 0
            r.input_tokens = u.input_tokens + cached + (u.cache_creation_input_tokens or 0)
            r.cached_tokens = cached
        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "text_delta":
                record_text(r, ms, "answer", d.text)
                piece = d.text
            elif d.type == "thinking_delta":
                record_text(r, ms, "thinking", d.thinking)
                piece = d.thinking
            elif d.type == "signature_delta":
                piece = "<signature>"  # end of a (possibly hidden) thinking block
            elif d.type == "input_json_delta":
                record_text(r, ms, "tool", d.partial_json)
                piece = d.partial_json
            else:
                piece = ""
            if args.events:
                print(f"  {d.type} {piece!r}", end="")
        elif event.type == "message_delta":
            r.stop_reason = event.delta.stop_reason
            r.output_tokens = event.usage.output_tokens
            details = getattr(event.usage, "output_tokens_details", None)
            if details is not None and details.thinking_tokens is not None:
                r.thinking_tokens = details.thinking_tokens

        if args.events:
            print()
    r.total_ms = (time.perf_counter() - t0) * 1000
    return r


# ---------- Local model (mlx_lm.server, OpenAI-compatible) ----------


@cache
def local_client(base_url: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key="not-needed")


def run_local(args: argparse.Namespace, model: str, prompt: str) -> RunResult:
    client = local_client(args.base_url)
    content = prompt if args.think else f"{prompt} /no_think"
    r = RunResult(output_tokens_exact=False)
    chunks_with_text = 0
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": content}],
        stream=True,
    )
    for chunk in stream:
        ms = (time.perf_counter() - t0) * 1000
        if chunk.usage is not None:  # only if the server sends usage
            r.output_tokens = chunk.usage.completion_tokens
            r.input_tokens = chunk.usage.prompt_tokens
            r.output_tokens_exact = True
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        answer = delta.content or ""
        thinking = getattr(delta, "reasoning", None) or ""  # non-standard field
        for kind, text in (("thinking", thinking), ("answer", answer)):
            if text:
                chunks_with_text += 1
                record_text(r, ms, kind, text)
        if args.events:
            print(f"  {ms:8.1f} ms  thinking={thinking!r} answer={answer!r} "
                  f"finish={choice.finish_reason}")
        if choice.finish_reason:
            r.stop_reason = choice.finish_reason
    r.total_ms = (time.perf_counter() - t0) * 1000
    if not r.output_tokens_exact:
        r.output_tokens = chunks_with_text  # ~1 token per chunk (seen in Step 2)
    return r


# ---------- Reporting ----------


def fmt(ms: float | None) -> str:
    return "     -" if ms is None else f"{ms:6.0f}"


def print_run(label: str, r: RunResult) -> None:
    tpot = "    -" if r.tpot_ms is None else f"{r.tpot_ms:5.1f}"
    approx = "" if r.output_tokens_exact else "~"
    print(
        f"{label:<10} ttft_visible={fmt(r.ttft_visible_ms)} ms  "
        f"ttft_model={fmt(r.ttft_model_ms)} ms  tpot={tpot} ms  "
        f"total={fmt(r.total_ms)} ms  in={r.input_tokens}  cached={r.cached_tokens}  "
        f"out={approx}{r.output_tokens}  thinking={r.thinking_tokens}  "
        f"deltas={r.text_deltas}  stop={r.stop_reason}"
    )
    if r.cut_off and r.ttft_visible_ms is None:
        print("           WARNING: cut off before any answer text "
              "(tokens went to thinking?); try --thinking off or more --max-tokens")
    elif r.cut_off:
        print("           WARNING: cut off by max_tokens; the answer is incomplete")


def median_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return percentile(present, 50) if present else None


def print_summary(title: str, results: list[RunResult]) -> None:
    n = len(results)
    print(f"\n{title}: n={n} measured runs (warm-up excluded)")
    rows = {
        "ttft_visible": [r.ttft_visible_ms for r in results],
        "ttft_model": [r.ttft_model_ms for r in results],
        "tpot": [r.tpot_ms for r in results],
        "total": [r.total_ms for r in results],
    }
    for name, raw in rows.items():
        values = [v for v in raw if v is not None]
        if not values:
            print(f"  {name:<13} no data")
            continue
        line = f"  {name:<13} median={percentile(values, 50):7.1f} ms"
        line += f"  min={min(values):7.1f}  max={max(values):7.1f}"
        if len(values) >= 100:
            line += f"  p95={percentile(values, 95):7.1f}"
        print(line)
    if n < 100:
        print("  (p95 is shown only when n >= 100)")
    if all(r.tpot_ms is None for r in results):
        print(f"  tpot needs >= {MIN_DELTAS} text deltas per run; try --long")
    cut = sum(r.cut_off for r in results)
    if cut:
        print(f"  WARNING: {cut}/{n} runs were cut off by max_tokens")


def build_prompt(question: str, pad_words: int, bust: bool) -> str:
    parts = []
    if bust:
        # A unique first line changes the prefix, so no cache can reuse it.
        parts.append(f"Request id: {uuid.uuid4().hex}")
    if pad_words:
        repeats = max(1, pad_words // 9)
        parts.append("Background text (ignore it):\n" + FILLER * repeats)
    parts.append(question)
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure TTFT and TPOT.")
    p.add_argument("--provider", choices=["anthropic", "local"], default="anthropic")
    p.add_argument("--model",
                   help="comma-separated to compare (default: claude-sonnet-5-5 / default_model)")
    p.add_argument("--prompt", default="Count from 1 to 5.")
    p.add_argument("--pad-words", default="0",
                   help="~N words of filler before the question (~1.3 tokens/word); "
                        "comma-separated to compare, e.g. 0,750,2250")
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument("--runs", type=int, default=1, help="measured runs")
    p.add_argument("--warmup", type=int, default=1, help="runs discarded first")
    p.add_argument("--bust-cache", action="store_true",
                   help="unique prefix per run so prefix caching can't help")
    p.add_argument("--thinking", choices=["default", "off", "summarized"], default="default",
                   help="Claude only: default=model decides (hidden on 5.5 models), "
                        "off=no up-front thinking, summarized=show thinking summaries")
    p.add_argument("--think", action="store_true",
                   help="local only: leave Qwen3 thinking on (default adds /no_think)")
    p.add_argument("--base-url",
                   default=os.environ.get("LOCAL_MODEL_URL", "http://localhost:8080/v1"))
    p.add_argument("--events", action="store_true", help="print every stream event")
    p.add_argument("--show-text", action="store_true", help="print the last answer")
    p.add_argument("--long", action="store_true",
                   help="ask for a ~300-word answer (max_tokens 600) so TPOT is measurable")
    p.add_argument("--seed", type=int, default=None, help="fix the shuffled run order")
    args = p.parse_args()
    args.models = [m.strip() for m in (args.model or DEFAULT_MODELS[args.provider]).split(",")]
    args.pads = [int(x) for x in args.pad_words.split(",")]
    if args.long:
        args.prompt = LONG_PROMPT
        args.max_tokens = max(args.max_tokens, 600)
    return args


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = parse_args()
    runner = run_anthropic if args.provider == "anthropic" else run_local
    configs = list(product(args.models, args.pads))
    print(
        f"provider={args.provider} models={args.models} pad_words={args.pads} "
        f"max_tokens={args.max_tokens} bust_cache={args.bust_cache} thinking={args.thinking} "
        f"warmup={args.warmup} runs={args.runs} per setting\n"
    )

    def one(label: str, model: str, pad: int) -> RunResult:
        prompt = build_prompt(args.prompt, pad, args.bust_cache)
        if args.events:
            print(f"--- {label} events ---")
        result = runner(args, model, prompt)
        print_run(label, result)
        return result

    last: RunResult | None = None
    for i in range(args.warmup):  # opens the connection; not measured
        last = one(f"warm-up {i + 1}", *configs[0])

    rng = random.Random(args.seed)
    measured: dict[tuple[str, int], list[RunResult]] = {c: [] for c in configs}
    for round_no in range(1, args.runs + 1):
        order = configs[:]
        rng.shuffle(order)  # interleave so drift over time hits every setting equally
        for model, pad in order:
            tag = f" {model} pad={pad}" if len(configs) > 1 else ""
            last = one(f"run {round_no}{tag}", model, pad)
            measured[(model, pad)].append(last)

    for (model, pad), results in measured.items():
        if results:
            print_summary(f"{model} pad={pad}", results)

    if len(configs) > 1 and args.runs:
        print("\nComparison (medians, ms)")
        print(f"  {'model':<22}{'pad':>6}{'in_tok':>8}{'ttft_visible':>14}{'tpot':>8}{'total':>9}")
        for (model, pad), results in measured.items():
            def cell(v: float | None, w: int) -> str:
                return f"{'-':>{w}}" if v is None else f"{v:>{w}.1f}"
            in_tok = results[-1].input_tokens
            print(f"  {model:<22}{pad:>6}{in_tok!s:>8}"
                  f"{cell(median_or_none([r.ttft_visible_ms for r in results]), 14)}"
                  f"{cell(median_or_none([r.tpot_ms for r in results]), 8)}"
                  f"{cell(median_or_none([r.total_ms for r in results]), 9)}")

    if args.show_text and last is not None:
        print(f"\nLast answer:\n{last.text}")

if __name__ == "__main__":
    main()
