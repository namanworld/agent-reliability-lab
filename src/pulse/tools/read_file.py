"""read_file: read a slice of one file, with line numbers, inside the repository.

Same boundary as search_code (pulse.tools.paths). Output is numbered so the
model can cite exact lines, and paged so a big file can't flood the context.

Try it directly:
    uv run python -m pulse.tools.read_file src/arl/metrics.py
    uv run python -m pulse.tools.read_file src/arl/metrics.py --start-line 5 --max-lines 3
    uv run python -m pulse.tools.read_file .env                       # blocked
    uv run python -m pulse.tools.read_file ../../etc/hosts            # outside the repo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pulse.tools.paths import resolve_inside
from pulse.tools.registry import Tool, ToolExecutionError

MAX_FILE_BYTES = 1_000_000
MAX_OUTPUT_CHARS = 8000
MAX_LINE_CHARS = 300


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="File path relative to the repository root.")
    start_line: int = Field(default=1, ge=1, description="First line to return (1-based).")
    max_lines: int = Field(default=120, ge=1, le=400, description="How many lines to return.")


def run_read_file(args: ReadFileArgs, repo_root: Path) -> str:
    target = resolve_inside(repo_root, args.path, must_be="file")
    if target.stat().st_size > MAX_FILE_BYTES:
        raise ToolExecutionError(f"'{args.path}' is larger than 1 MB; use search_code to find the lines you need.")
    data = target.read_bytes()
    if b"\0" in data[:1024]:
        raise ToolExecutionError(f"'{args.path}' is a binary file.")

    lines = data.decode("utf-8", errors="replace").splitlines()
    total = len(lines)
    if total == 0:
        return f"'{args.path}' is empty."
    if args.start_line > total:
        raise ToolExecutionError(f"'{args.path}' has only {total} lines; start_line {args.start_line} is past the end.")

    first = args.start_line
    last = min(total, first + args.max_lines - 1)
    width = len(str(last))
    out_lines: list[str] = []
    size = 0
    for no in range(first, last + 1):
        text = lines[no - 1]
        if len(text) > MAX_LINE_CHARS:
            text = text[:MAX_LINE_CHARS] + " [line truncated]"
        line = f"{no:>{width}}| {text}"
        if size + len(line) > MAX_OUTPUT_CHARS:
            last = no - 1
            break
        out_lines.append(line)
        size += len(line) + 1

    header = f"{args.path} (lines {first}-{last} of {total})"
    footer = ""
    if last < total:
        footer = f"\n[more below; call read_file with start_line={last + 1} to continue]"
    return header + "\n" + "\n".join(out_lines) + footer


READ_FILE = Tool(
    name="read_file",
    description=(
        "Read lines from one file in the repository, with line numbers. Use it after "
        "search_code has found the file, and read only the lines you need."
    ),
    args_model=ReadFileArgs,
    run=run_read_file,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run read_file directly, without a model.")
    p.add_argument("path")
    p.add_argument("--start-line", type=int, default=1)
    p.add_argument("--max-lines", type=int, default=120)
    p.add_argument("--repo", type=Path, default=Path("."))
    a = p.parse_args()
    try:
        args = ReadFileArgs(path=a.path, start_line=a.start_line, max_lines=a.max_lines)
        print(run_read_file(args, a.repo))
    except ToolExecutionError as e:
        print(f"ERROR (sent back to the model): {e}")


if __name__ == "__main__":
    main()
