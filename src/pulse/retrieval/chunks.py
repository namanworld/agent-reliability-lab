"""Step 1 of retrieval: cut the repository into chunks.

A chunk is the unit we search over and return. Good chunks are small enough to be
specific and large enough to make sense on their own:

- Python files: one chunk per top-level function or class (found with `ast`),
  plus one chunk for the module header (imports, constants) if it has content.
- Other text files (Markdown, YAML, TOML): windows of 40 lines.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

WINDOW_LINES = 40


@dataclass
class Chunk:
    path: str  # relative to the repo root
    start_line: int  # 1-based, inclusive
    end_line: int  # inclusive
    kind: str  # "function", "class", "module", or "text"
    name: str  # function/class name, or "" for module/text chunks
    text: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def chunk_file(path: str, source: str) -> list[Chunk]:
    if path.endswith(".py"):
        try:
            return chunk_python(path, source)
        except SyntaxError:
            pass  # not valid Python: fall back to line windows
    return chunk_text(path, source)


def chunk_python(path: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    tree = ast.parse(source)
    chunks: list[Chunk] = []

    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    for node in definitions:
        # Include decorators, which sit above the `def` line.
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = node.end_lineno or node.lineno
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.append(Chunk(path, start, end, kind, node.name, "\n".join(lines[start - 1:end])))

    # Everything before the first definition: docstring, imports, constants.
    header_end = (min(c.start_line for c in chunks) - 1) if chunks else len(lines)
    header = "\n".join(lines[:header_end]).strip()
    if header:
        chunks.insert(0, Chunk(path, 1, max(header_end, 1), "module", "", header))
    return chunks


def chunk_text(path: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks = []
    for start in range(0, len(lines), WINDOW_LINES):
        window = lines[start:start + WINDOW_LINES]
        text = "\n".join(window).strip()
        if text:
            chunks.append(Chunk(path, start + 1, start + len(window), "text", "", text))
    return chunks
