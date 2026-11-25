"""The repository boundary shared by every tool that touches files.

One implementation, used by search_code and read_file, so a fix to the boundary
fixes every tool at once.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Literal

from pulse.tools.registry import ToolExecutionError

BLOCKED_GLOBS = [".env*", "*.pem", "*.key", "id_rsa*", "*.p12"]
BLOCKED_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
# Readable with read_file, but left out of searches and the index:
# - evals/: eval questions and attack fixtures. Searchable eval data leaks the test
#   into the answer (a retrieval query would find itself in retrieval.yaml), and the
#   injection fixture showed up in unrelated searches in M1.4.
# - .pulse_index/: the search index's own cache.
SEARCH_SKIPPED_DIRS = {"evals", ".pulse_index"}


def path_weight(rel_path: str) -> float:
    """How likely a file is to be the answer: code > scripts > tests > notes and evals."""
    if rel_path.startswith("src/"):
        return 1.0
    if rel_path.startswith("scripts/"):
        return 0.8
    if rel_path.startswith("tests/"):
        return 0.6
    if rel_path.endswith(".md") or rel_path.startswith("evals/"):
        return 0.4
    return 0.7


def is_blocked(rel_path: str) -> bool:
    """Secrets and tool/VCS internals that no tool may read or search."""
    parts = Path(rel_path).parts
    if any(p in BLOCKED_DIRS for p in parts):
        return True
    return bool(parts) and any(fnmatch.fnmatch(parts[-1], g) for g in BLOCKED_GLOBS)


def resolve_inside(
    repo_root: Path, user_path: str, must_be: Literal["dir", "file"] = "dir"
) -> Path:
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
    if target != root and is_blocked(str(target.relative_to(root))):
        raise ToolExecutionError(f"Path '{user_path}' is not accessible (secrets and VCS files are blocked).")
    if not target.exists():
        raise ToolExecutionError(f"Path '{user_path}' does not exist in the repository.")
    if must_be == "dir" and not target.is_dir():
        raise ToolExecutionError(f"Path '{user_path}' is a file; this tool takes a directory.")
    if must_be == "file" and not target.is_file():
        raise ToolExecutionError(f"Path '{user_path}' is a directory; this tool takes a file.")
    return target
