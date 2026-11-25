"""search_code: find text in the repository, without ever leaving it.

The model chooses the arguments, so every argument is treated as hostile input:

- `path` must resolve (symlinks included) to a directory inside the repo root.
- `query` is passed as a separate argv item after `--`, never through a shell,
  so a query like "--version" or "; rm -rf ~" is just text to search for.
- Secrets and VCS internals (.env*, *.pem, *.key, .git/) are never searched.
- Output is capped (lines, columns, characters) and the search has a timeout.

Two backends with the same guarantees: ripgrep when installed (fast), and a
pure-Python fallback so the tool still works (and is testable) without it.

Try it directly:
    uv run python -m pulse.tools.search_code "def percentile"
    uv run python -m pulse.tools.search_code "percentile" --path src --max-results 3
    uv run python -m pulse.tools.search_code "x" --path ../..          # rejected
    uv run python -m pulse.tools.search_code "def " --backend python
    uv run python -m pulse.tools.search_code -- --version          # "--" so the CLI treats it as text
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pulse.tools.registry import Tool, ToolExecutionError

BLOCKED_GLOBS = [".env*", "*.pem", "*.key", "id_rsa*", "*.p12"]
BLOCKED_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
MAX_LINE_CHARS = 200
MAX_OUTPUT_CHARS = 4000
MAX_FILE_BYTES = 1_000_000
TIMEOUT_S = 5.0


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


# ---------- the security boundary ----------


def resolve_inside(repo_root: Path, user_path: str) -> Path:
    """Resolve user_path against repo_root and refuse anything that lands outside it.

    resolve() follows symlinks and collapses '..', so the check happens on the real
    location, not on how the path is spelled.
    """
    root = repo_root.resolve()
    target = (root / user_path).resolve()
    if target != root and root not in target.parents:
        raise ToolExecutionError(
            f"Path '{user_path}' is outside the repository. Use a path relative to the repository root."
        )
    if not target.exists():
        raise ToolExecutionError(f"Path '{user_path}' does not exist in the repository.")
    if not target.is_dir():
        raise ToolExecutionError(f"Path '{user_path}' is a file; search_code takes a directory.")
    return target


def is_blocked(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if any(p in BLOCKED_DIRS for p in parts):
        return True
    return any(fnmatch.fnmatch(parts[-1], g) for g in BLOCKED_GLOBS)


# ---------- backends: each returns (matches, error); a match is (rel_path, line_no, text) ----------

Match = tuple[str, int, str]


def search_with_ripgrep(rg: str, root: Path, target: Path, query: str) -> list[Match]:
    cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--with-filename",
           "--max-columns", str(MAX_LINE_CHARS), "--max-filesize", "1M"]
    for g in BLOCKED_GLOBS:
        cmd += ["--glob", f"!{g}"]
    for d in BLOCKED_DIRS:
        cmd += ["--glob", f"!{d}/"]
    # No --follow: symlinks are not traversed. "--" ends the options, so a query
    # starting with "-" can't be read as a flag (argument injection).
    cmd += ["--", query, str(target)]
    try:
        proc = subprocess.run(  # exit codes are handled below, so no check=True
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S, shell=False, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise ToolExecutionError(f"Search timed out after {TIMEOUT_S:.0f}s. Narrow the path or query.") from e
    if proc.returncode == 1:  # ripgrep: 1 = no matches
        return []
    if proc.returncode != 0:
        first = (proc.stderr.strip().splitlines() or ["unknown error"])[0]
        raise ToolExecutionError(f"Search failed: {first[:200]}")
    matches: list[Match] = []
    for line in proc.stdout.splitlines():
        file_part, _, rest = line.partition(":")
        line_no, _, text = rest.partition(":")
        if not line_no.isdigit():
            continue
        rel = os.path.relpath(file_part, root)
        if not is_blocked(rel):
            matches.append((rel, int(line_no), text))
    return matches


def search_with_python(root: Path, target: Path, query: str) -> list[Match]:
    try:
        pattern = re.compile(query)
    except re.error as e:
        raise ToolExecutionError(f"Invalid regular expression: {e}") from e
    matches: list[Match] = []
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in BLOCKED_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = os.path.relpath(full, root)
            if name.startswith(".") or is_blocked(rel) or full.is_symlink():
                continue
            try:
                if full.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = full.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:1024]:  # binary file
                continue
            for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    matches.append((rel, i, line[:MAX_LINE_CHARS]))
    return matches


# ---------- the tool ----------


def run_search_code(args: SearchCodeArgs, repo_root: Path, backend: str = "auto") -> str:
    root = repo_root.resolve()
    target = resolve_inside(root, args.path)
    rg = shutil.which("rg")
    if backend == "rg" or (backend == "auto" and rg):
        if not rg:
            raise ToolExecutionError("ripgrep is not installed.")
        matches = search_with_ripgrep(rg, root, target, args.query)
    else:
        matches = search_with_python(root, target, args.query)

    if not matches:
        return f"No matches for {args.query!r} in '{args.path}'."
    matches.sort()
    shown = matches[: args.max_results]
    lines = [f"{rel}:{no}: {text.strip()}" for rel, no, text in shown]
    out = "\n".join(lines)
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
    if len(matches) > len(shown):
        out += f"\n[showing {len(shown)} of {len(matches)} matches; raise max_results or narrow the query]"
    return out


SEARCH_CODE = Tool(
    name="search_code",
    description=(
        "Search the repository's source files for a text pattern and return matching "
        "lines with file paths and line numbers. Use it to find where something is "
        "defined or used before answering questions about the code."
    ),
    args_model=SearchCodeArgs,
    run=run_search_code,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Run search_code directly, without a model.")
    p.add_argument("query")
    p.add_argument("--path", default=".")
    p.add_argument("--max-results", type=int, default=10)
    p.add_argument("--repo", type=Path, default=Path("."), help="repository root")
    p.add_argument("--backend", choices=["auto", "rg", "python"], default="auto")
    a = p.parse_args()
    try:
        args = SearchCodeArgs(query=a.query, path=a.path, max_results=a.max_results)
        print(run_search_code(args, a.repo, backend=a.backend))
    except ToolExecutionError as e:
        print(f"ERROR (sent back to the model): {e}")


if __name__ == "__main__":
    main()
