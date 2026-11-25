"""search_code: arguments only for now. The implementation (ripgrep with a path
allowlist) comes in M1.2; M1.1 is about getting a valid call from the model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pulse.tools.registry import Tool


class SearchCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject arguments the tool doesn't have

    query: str = Field(
        min_length=1,
        max_length=200,
        description="Text or regular expression to search for, e.g. 'def percentile'.",
    )
    path: str = Field(
        default=".",
        description="Directory to search, relative to the repository root.",
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of matching lines to return."
    )


SEARCH_CODE = Tool(
    name="search_code",
    description=(
        "Search the repository's source files for a text pattern and return matching "
        "lines with file paths and line numbers. Use it to find where something is "
        "defined or used before answering questions about the code."
    ),
    args_model=SearchCodeArgs,
)
