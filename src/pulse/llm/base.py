"""Provider-neutral types for talking to any LLM.

The agent only sees these types. Each provider (Anthropic, local mlx-lm, ...)
translates its own request and stream format into them, so the agent code never
changes when the model does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ThinkingMode = Literal["off", "default"]


@dataclass
class ToolCall:
    """One tool call exactly as the model produced it (arguments not yet validated)."""

    id: str
    name: str
    arguments_json: str  # raw text, because the model can produce invalid JSON


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    ttft_ms: float | None = None  # first visible output: answer text or tool call
    total_ms: float | None = None

    @property
    def cut_off(self) -> bool:
        return self.stop_reason in ("max_tokens", "length")

    @property
    def empty(self) -> bool:
        """No answer text and no tool call: ARL check #1 from M0.1."""
        return not self.text.strip() and not self.tool_calls


class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        thinking: ThinkingMode = "off",
        system: str | None = None,
    ) -> LLMResponse: ...
