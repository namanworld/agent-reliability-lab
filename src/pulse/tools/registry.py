"""Tool definitions and validation of the model's tool calls.

Each tool's arguments are a Pydantic model. The same model produces the JSON
Schema sent to the LLM and validates what comes back, so the description the
model sees and the check the code runs can never drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from pulse.llm.base import ToolCall


class ToolCallError(Exception):
    """A tool call the agent must not run. The message is written for the model,
    so the agent loop (M1.3) can send it back and let the model correct itself."""


class ToolExecutionError(Exception):
    """A valid call that failed while running (path outside the repo, timeout, ...).
    Also written for the model, which can retry with different arguments."""


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[..., str] | None = None  # (validated_args, repo_root) -> text for the model

    def spec(self) -> dict[str, Any]:
        """The tool definition in the Anthropic `tools` format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


def validate_call(call: ToolCall, tools: dict[str, Tool]) -> BaseModel:
    """Return validated arguments, or raise ToolCallError explaining what's wrong."""
    tool = tools.get(call.name)
    if tool is None:
        known = ", ".join(sorted(tools)) or "none"
        raise ToolCallError(f"Unknown tool '{call.name}'. Available tools: {known}.")

    try:
        raw = json.loads(call.arguments_json)
    except json.JSONDecodeError as e:
        raise ToolCallError(
            f"Arguments for '{call.name}' are not valid JSON ({e.msg} at position {e.pos})."
        ) from e
    if not isinstance(raw, dict):
        raise ToolCallError(f"Arguments for '{call.name}' must be a JSON object.")

    try:
        return tool.args_model.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
            for err in e.errors()
        )
        raise ToolCallError(f"Invalid arguments for '{call.name}': {problems}.") from e
