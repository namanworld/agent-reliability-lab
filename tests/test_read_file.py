from pathlib import Path

import pytest

from pulse.tools.read_file import ReadFileArgs, run_read_file
from pulse.tools.registry import ToolExecutionError


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text("".join(f"line {i}\n" for i in range(1, 501)))
    (root / "empty.py").write_text("")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02" * 10)
    (root / ".env").write_text("KEY=sk-secret\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("sk-secret-outside\n")
    (root / "link.txt").symlink_to(outside)
    return root


def read(repo: Path, **kw) -> str:
    return run_read_file(ReadFileArgs(**kw), repo)


def test_reads_numbered_slice_and_says_where_to_continue(repo):
    out = read(repo, path="src/m.py", start_line=10, max_lines=3)
    assert out.splitlines()[0] == "src/m.py (lines 10-12 of 500)"
    assert "10| line 10" in out and "12| line 12" in out and "13| line 13" not in out
    assert "start_line=13" in out


def test_whole_small_file_has_no_continue_hint(repo):
    (repo / "small.py").write_text("a\nb\n")
    out = read(repo, path="small.py")
    assert "lines 1-2 of 2" in out and "more below" not in out


@pytest.mark.parametrize("bad", ["../outside.txt", "/etc/hosts", "src/../../outside.txt", "link.txt"])
def test_outside_paths_and_symlinks_out_are_rejected(repo, bad):
    with pytest.raises(ToolExecutionError, match="outside the repository"):
        read(repo, path=bad)


def test_secret_file_is_blocked(repo):
    with pytest.raises(ToolExecutionError, match="not accessible"):
        read(repo, path=".env")


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"path": "nope.py"}, "does not exist"),
        ({"path": "src"}, "is a directory"),
        ({"path": "blob.bin"}, "binary"),
        ({"path": "src/m.py", "start_line": 999}, "only 500 lines"),
    ],
)
def test_errors_are_explained(repo, kw, message):
    with pytest.raises(ToolExecutionError, match=message):
        read(repo, **kw)


def test_empty_file(repo):
    assert "is empty" in read(repo, path="empty.py")
