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

    def mark_visible(ms: float | None) -> None:
        if resp.ttft_ms is None and ms is not None:
            resp.ttft_ms = ms

    for ms, ev in timed_events:
        etype = ev.get("type")
        if etype == "message_start":
            usage = ev["message"]["usage"]
            resp.input_tokens = (
                usage.get("input_tokens", 0)
                + (usage.get("cache_read_input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
            )
        elif etype == "content_block_start":
            block = ev["content_block"]
            if block.get("type") == "tool_use":
                open_tools[ev["index"]] = {"id": block["id"], "name": block["name"], "json": ""}
                order.append(ev["index"])
                mark_visible(ms)
        elif etype == "content_block_delta":
            delta = ev["delta"]
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                resp.text += text
                if text.strip():
                    mark_visible(ms)
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
        resp.tool_calls.append(ToolCall(id=t["id"], name=t["name"], arguments_json=t["json"] or "{}"))
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
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools
        if system:
            request["system"] = system
        if thinking == "off":
            request["thinking"] = thinking_off_param(self.model)
        return parse_events(self._timed_stream(**request))
