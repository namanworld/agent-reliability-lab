"""Retrieval tests. They use HashingEmbedder (offline, word-based), so they check the
plumbing (chunking, ranking, fusion, boundaries, caching), not semantic quality.
Semantic quality is measured by scripts/eval_retrieval.py with the real model."""

from pathlib import Path

import pytest

from pulse.retrieval.bm25 import BM25, tokenize
from pulse.retrieval.chunks import chunk_python, chunk_text
from pulse.retrieval.embeddings import HashingEmbedder
from pulse.retrieval.index import CodeIndex, list_files
from pulse.tools.find_code import FindCodeArgs, run_find_code
from pulse.tools.paths import path_weight

PY = '''"""Module docstring."""
import os

LIMIT = 3


@decorator
def read_config(path):
    return open(path).read()


class SecretGuard:
    def is_blocked(self, name):
        return name.startswith(".env")
'''


# ---------- chunking ----------


def test_python_file_splits_into_header_function_and_class():
    chunks = chunk_python("m.py", PY)
    assert [(c.kind, c.name) for c in chunks] == [("module", ""), ("function", "read_config"), ("class", "SecretGuard")]
    fn = chunks[1]
    assert fn.start_line == 7 and fn.text.startswith("@decorator")  # decorator included
    assert chunks[2].location == "m.py:12-14"


def test_text_file_splits_into_line_windows():
    text = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_text("notes.md", text)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (41, 80), (81, 100)]


# ---------- BM25 ----------


def test_tokenize_splits_identifiers():
    assert tokenize("readFile read_file ReadFile") == ["read", "file"] * 3


def test_bm25_prefers_rare_matching_words():
    bm = BM25(["the agent loop and step budget", "percentile interpolation", "agent agent agent"])
    scores = bm.scores("percentile")
    assert scores[1] > 0 and scores[0] == 0 and scores[2] == 0


# ---------- index, fusion and boundaries ----------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "guard.py").write_text(PY)
    (tmp_path / "NOTES.md").write_text("We discussed the secret guard and is_blocked at length.\n")
    (tmp_path / ".env").write_text("KEY=sk-secret\n")
    (tmp_path / "evals" / "fixtures").mkdir(parents=True)
    (tmp_path / "evals" / "fixtures" / "attack.md").write_text("ignore instructions, secret guard\n")
    return tmp_path


def test_index_skips_secrets_and_fixtures(repo):
    files = list_files(repo)
    assert "src/guard.py" in files and "NOTES.md" in files
    assert ".env" not in files and not any(f.startswith("evals/") for f in files)


def test_hybrid_puts_source_above_notes(repo):
    index = CodeIndex.build(repo, HashingEmbedder(), use_cache=False)
    hits = index.search("secret guard is blocked", HashingEmbedder(), mode="hybrid")
    assert hits[0].chunk.path == "src/guard.py"


def test_path_weight_order():
    assert path_weight("src/a.py") > path_weight("scripts/a.py") > path_weight("tests/a.py") > path_weight("NOTES.md")


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return super().embed_documents(texts)


def test_only_changed_files_are_embedded_again(repo):
    emb = CountingEmbedder()
    full = CodeIndex.build(repo, emb)
    first_count = emb.embedded
    assert first_count == len(full.chunks) > 0

    CodeIndex.build(repo, emb)  # nothing changed
    assert emb.embedded == first_count

    (repo / "src" / "extra.py").write_text("def one():\n    pass\n\n\ndef two():\n    pass\n")
    updated = CodeIndex.build(repo, emb)
    assert emb.embedded == first_count + 2 and updated.reembedded == 2
    # reused and new vectors line up with their chunks
    fresh = CodeIndex.build(repo, HashingEmbedder(), use_cache=False)
    assert [c.location for c in updated.chunks] == [c.location for c in fresh.chunks]
    assert (updated.vectors == fresh.vectors).all()


def test_cache_is_reused_and_rebuilt_on_change(repo):
    first = CodeIndex.build(repo, HashingEmbedder())
    assert (repo / ".pulse_index").is_dir()
    again = CodeIndex.build(repo, HashingEmbedder())
    assert [c.location for c in again.chunks] == [c.location for c in first.chunks]
    (repo / "src" / "new.py").write_text("def brand_new_function():\n    pass\n")
    rebuilt = CodeIndex.build(repo, HashingEmbedder())
    assert any(c.name == "brand_new_function" for c in rebuilt.chunks)


def test_rrf_rewards_agreement_between_lists(repo):
    index = CodeIndex.build(repo, HashingEmbedder(), use_cache=False)
    hybrid = index.search("read config", HashingEmbedder(), mode="hybrid", use_path_weight=False)
    assert hybrid[0].chunk.name == "read_config"
    assert hybrid[0].score == pytest.approx(2 / 61)  # rank 1 in both lists: 1/61 + 1/61


# ---------- the tool ----------


def test_find_code_output_has_location_and_snippet(repo):
    out = run_find_code(FindCodeArgs(query="secret guard blocked"), repo, embedder_kind="hashing")
    first = out.splitlines()[0]
    assert first.startswith("1. src/guard.py:") and "class SecretGuard" in first
    assert "sk-secret" not in out


def test_find_code_without_embedder_still_works(repo):
    out = run_find_code(FindCodeArgs(query="read config"), repo, embedder_kind="none")
    assert "read_config" in out
