"""Anthropic (Claude) provider: streams a Messages API call into an LLMResponse.

Parsing is a separate pure function (`parse_events`) that works on plain dicts,
so the same code handles a live stream and a recorded one (see replay.py).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pulse.llm.base import LLMResponse, ThinkingMode, ToolCall

TimedEvent = tuple[float | None, dict[str, Any]]  # (ms since request, raw event)

# Each model turns thinking off with a different value, and the wrong one is a 400 error
# (docs configuration table, checked 2026-11-25). Provider quirks like this are exactly
# what an adapter layer exists to hide from the agent.
THINKING_OFF: dict[str, dict[str, str]] = {
    "claude-sonnet-5-5": {"type": "between_tools"},  # rejects "disabled"
    "claude-haiku-5-5": {"type": "disabled"},  # rejects "between_tools"
}


def thinking_off_param(model: str) -> dict[str, str]:
    for prefix, param in THINKING_OFF.items():
        if model.startswith(prefix):
            return param
    known = ", ".join(THINKING_OFF)
    raise ValueError(
        f"Don't know how to turn thinking off for {model!r}. Add it to THINKING_OFF (known: {known})."
    )


def parse_events(timed_events: Iterable[TimedEvent]) -> LLMResponse:
    """Turn raw Anthropic stream events into an LLMResponse.

    Tool calls arrive in pieces: `content_block_start` gives the tool's id and
    name, then `input_json_delta` events carry fragments of the arguments JSON,
    which only form valid JSON once the block is finished.
    """
    resp = LLMResponse()
    open_tools: dict[int, dict[str, Any]] = {}  # block index -> partial tool call
    order: list[int] = []
    blocks: dict[int, dict[str, Any]] = {}  # block index -> block being rebuilt

    def mark_visible(ms: float | None) -> None:
        if resp.ttft_ms is None and ms is not None:
            resp.ttft_ms = ms

    for ms, ev in timed_events:
        etype = ev.get("type")
        if etype == "message_start":
            usage = ev["message"]["usage"]
            # input_tokens from the API counts only the uncached part; total = all three.
            resp.cache_read_tokens = usage.get("cache_read_input_tokens") or 0
            resp.cache_write_tokens = usage.get("cache_creation_input_tokens") or 0
            resp.input_tokens = usage.get("input_tokens", 0) + resp.cache_read_tokens + resp.cache_write_tokens
        elif etype == "content_block_start":
            block = ev["content_block"]
            btype = block.get("type")
            if btype == "text":
                blocks[ev["index"]] = {"type": "text", "text": ""}
            elif btype == "thinking":
                blocks[ev["index"]] = {"type": "thinking", "thinking": "", "signature": ""}
            elif btype == "redacted_thinking":
                blocks[ev["index"]] = {"type": "redacted_thinking", "data": block.get("data", "")}
            elif btype == "tool_use":
                blocks[ev["index"]] = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            if btype == "tool_use":
                open_tools[ev["index"]] = {"id": block["id"], "name": block["name"], "json": ""}
                order.append(ev["index"])
                mark_visible(ms)
        elif etype == "content_block_delta":
            delta = ev["delta"]
            block = blocks.get(ev.get("index", -1), {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                resp.text += text
                if block.get("type") == "text":
                    block["text"] += text
                if text.strip():
                    mark_visible(ms)
            elif delta.get("type") == "thinking_delta" and block.get("type") == "thinking":
                block["thinking"] += delta.get("thinking", "")
            elif delta.get("type") == "signature_delta" and block.get("type") == "thinking":
                block["signature"] += delta.get("signature", "")
            elif delta.get("type") == "input_json_delta":
                open_tools[ev["index"]]["json"] += delta.get("partial_json", "")
        elif etype == "message_delta":
            resp.stop_reason = ev["delta"].get("stop_reason")
            usage = ev.get("usage") or {}
            resp.output_tokens = usage.get("output_tokens")
            details = usage.get("output_tokens_details") or {}
            resp.thinking_tokens = details.get("thinking_tokens")
        if ms is not None:
            resp.total_ms = ms

    for index in order:
        t = open_tools[index]
        # A tool with no arguments streams no JSON at all; that means "{}".
        args_json = t["json"] or "{}"
        resp.tool_calls.append(ToolCall(id=t["id"], name=t["name"], arguments_json=args_json))
        try:
            parsed = json.loads(args_json)
        except json.JSONDecodeError:
            parsed = {}  # the history needs an object; the error goes back in the tool_result
        blocks[index]["input"] = parsed if isinstance(parsed, dict) else {}
    resp.content_blocks = [
        b for _, b in sorted(blocks.items())
        if not (b["type"] == "text" and not b["text"])  # the API rejects empty text blocks
    ]
    return resp


class AnthropicProvider:
    def __init__(self, model: str = "claude-sonnet-5-5", record_to: Path | None = None):
        from anthropic import Anthropic

        self.client = Anthropic()  # one client = one reused HTTPS connection
        self.model = model
        self.record_to = record_to

    def _timed_stream(self, **request: Any) -> Iterator[TimedEvent]:
        t0 = time.perf_counter()
        recorded: list[dict[str, Any]] = []
        for event in self.client.messages.create(stream=True, **request):
            ms = (time.perf_counter() - t0) * 1000
            raw = event.model_dump(mode="json", exclude_none=True)
            recorded.append({"ms": round(ms, 1), "event": raw})
            yield ms, raw
        if self.record_to:
            self.record_to.parent.mkdir(parents=True, exist_ok=True)
            with self.record_to.open("w") as f:
                for line in recorded:
                    f.write(json.dumps(line) + "\n")

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        thinking: ThinkingMode = "off",
        system: str | None = None,
        tool_choice: dict[str, Any] | None = None,
        cache: bool = False,
    ) -> LLMResponse:
        request = build_request(self.model, messages, tools=tools, max_tokens=max_tokens,
                                thinking=thinking, system=system, tool_choice=tool_choice, cache=cache)
        return parse_events(self._timed_stream(**request))


CACHE = {"type": "ephemeral"}  # 5-minute cache, refreshed each time it's used


def build_request(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 1024,
    thinking: ThinkingMode = "off",
    system: str | None = None,
    tool_choice: dict[str, Any] | None = None,
    cache: bool = False,
) -> dict[str, Any]:
    """Build the Messages API request. A pure function, so tests can inspect it.

    With cache=True, up to 3 cache breakpoints are set. The API caches everything
    *up to and including* a marked block, in the order tools -> system -> messages:

      1. the last tool definition  -> caches all tool definitions
      2. the system prompt         -> caches tools + system
      3. the last message block    -> caches the whole conversation so far, so the
                                      next step only pays full price for new turns

    Cached reads cost 0.1x the input price (0.05x on Sonnet 5.5); writing to the
    cache costs 1.25x. Prompts under 512 tokens are never cached (no error).
    Inputs are copied, never modified: the caller's message history stays clean.
    """
    request: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if tools:
        request["tools"] = [dict(t) for t in tools]
    if system:
        request["system"] = system
    if tool_choice:
        request["tool_choice"] = tool_choice
    if thinking == "off":
        request["thinking"] = thinking_off_param(model)

    if cache:
        if tools:
            request["tools"][-1]["cache_control"] = CACHE
        if system:
            request["system"] = [{"type": "text", "text": system, "cache_control": CACHE}]
        if messages:
            msgs = [dict(m) for m in messages]
            last = msgs[-1]
            content = last["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) for b in content]
            blocks[-1]["cache_control"] = CACHE
            last["content"] = blocks
            request["messages"] = msgs
    return request
