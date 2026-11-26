"""Replay a recorded stream instead of calling a model.

Record once with a real call (`--record`), then tests replay the file: same
parsing code, no API key, no network, same result every time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pulse.llm.anthropic_provider import TimedEvent, parse_events
from pulse.llm.base import LLMResponse, ThinkingMode


def load_recording(path: Path) -> list[TimedEvent]:
    events: list[TimedEvent] = []
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            events.append((item.get("ms"), item["event"]))
    return events


class ReplayProvider:
    def __init__(self, path: Path):
        self.path = path

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
        return parse_events(load_recording(self.path))
