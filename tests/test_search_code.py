"""search_code security and behavior tests, run against both backends.

Each test builds a tiny fake repository in a temp folder, plus a "secret" folder
outside it that the tool must never reach.
"""

import shutil
from pathlib import Path

import pytest

from pulse.tools.registry import ToolExecutionError
from pulse.tools.search_code import SearchCodeArgs, run_search_code

BACKENDS = [
    "python",
    pytest.param("rg", marks=pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")),
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "arl").mkdir(parents=True)
    (root / "src" / "arl" / "metrics.py").write_text(
        "import math\n\n\ndef percentile(values, p):\n    return 0\n"
    )
    (root / "README.md").write_text("Agent Reliability Lab\nUse --version to print the version.\n")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-secret-from-dotenv\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("token = sk-secret-from-git\n")
    (root / "server.key").write_text("sk-secret-from-keyfile\n")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.py").write_text("def percentile_secret(): 'sk-secret-outside'\n")
    (root / "linked_dir").symlink_to(outside, target_is_directory=True)
    (root / "linked_file.py").symlink_to(outside / "secrets.py")
    return root


def search(repo: Path, backend: str, **kwargs) -> str:
    return run_search_code(SearchCodeArgs(**kwargs), repo, backend=backend)


# ---------- normal behavior ----------


@pytest.mark.parametrize("backend", BACKENDS)
def test_finds_definition_with_path_and_line(repo, backend):
    out = search(repo, backend, query="def percentile")
    assert "src/arl/metrics.py:4: def percentile(values, p):" in out


@pytest.mark.parametrize("backend", BACKENDS)
def test_path_narrows_the_search(repo, backend):
    assert "metrics.py" in search(repo, backend, query="import", path="src")
    assert "No matches" in search(repo, backend, query="Agent Reliability", path="src")


@pytest.mark.parametrize("backend", BACKENDS)
def test_max_results_truncates_and_says_so(repo, backend):
    out = search(repo, backend, query="e", max_results=1)
    assert len([line for line in out.splitlines() if not line.startswith("[")]) == 1
    assert "[showing 1 of" in out


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_matches_is_a_normal_answer(repo, backend):
    assert search(repo, backend, query="does_not_exist_anywhere").startswith("No matches")


@pytest.mark.parametrize("backend", BACKENDS)
def test_query_starting_with_dashes_is_text_not_a_flag(repo, backend):
    out = search(repo, backend, query="--version")
    assert "README.md:2:" in out


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_regex_is_reported_to_the_model(repo, backend):
    with pytest.raises(ToolExecutionError):
        search(repo, backend, query="def (")


# ---------- the security boundary ----------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bad_path", ["..", "../outside", "src/../..", "/etc", "/"])
def test_paths_outside_the_repo_are_rejected(repo, backend, bad_path):
    with pytest.raises(ToolExecutionError, match="outside the repository"):
        search(repo, backend, query="x", path=bad_path)


@pytest.mark.parametrize("backend", BACKENDS)
def test_symlinked_dir_pointing_outside_is_rejected(repo, backend):
    with pytest.raises(ToolExecutionError, match="outside the repository"):
        search(repo, backend, query="percentile", path="linked_dir")


@pytest.mark.parametrize("backend", BACKENDS)
def test_symlinks_are_not_followed_during_a_search(repo, backend):
    out = search(repo, backend, query="percentile", max_results=50)
    assert "sk-secret-outside" not in out
    assert "percentile_secret" not in out


@pytest.mark.parametrize("backend", BACKENDS)
def test_secret_files_are_never_searched(repo, backend):
    out = search(repo, backend, query="sk-secret", max_results=50)
    assert out.startswith("No matches"), out


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_path_and_file_path_are_explained(repo, backend):
    with pytest.raises(ToolExecutionError, match="does not exist"):
        search(repo, backend, query="x", path="nope")
    with pytest.raises(ToolExecutionError, match="is a file"):
        search(repo, backend, query="x", path="README.md")
