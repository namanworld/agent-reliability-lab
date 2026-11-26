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
    cache_read_tokens: int = 0  # part of input_tokens served from the prompt cache
    cache_write_tokens: int = 0  # part of input_tokens written to the cache this call
    ttft_ms: float | None = None  # first visible output: answer text or tool call
    total_ms: float | None = None
    # The assistant turn exactly as it must be sent back in the next request
    # (text, tool_use and any thinking blocks with their signatures).
    content_blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cut_off(self) -> bool:
        return self.stop_reason in ("max_tokens", "length")

    @property
    def refused(self) -> bool:
        """The model declined for safety reasons. Generation stops early, so any text
        is often cut mid-sentence. Not an answer, and not something to retry blindly."""
        return self.stop_reason == "refusal"

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
        tool_choice: dict[str, Any] | None = None,
        cache: bool = False,
    ) -> LLMResponse: ...
