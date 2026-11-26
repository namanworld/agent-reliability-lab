"""Agent loop tests with a scripted fake model: no API, fully deterministic."""

from pathlib import Path

import pytest

from pulse.agent import REPEAT_NOTE, Agent, add_note, estimate_tokens, trim_context
from pulse.llm.base import LLMResponse, ToolCall
from pulse.tools.read_file import READ_FILE
from pulse.tools.search_code import SEARCH_CODE


def tool_turn(*calls: tuple[str, str, str]) -> LLMResponse:
    """A model turn that calls tools. calls = (id, name, arguments_json)."""
    tool_calls = [ToolCall(id=i, name=n, arguments_json=a) for i, n, a in calls]
    blocks = [{"type": "tool_use", "id": i, "name": n, "input": {}} for i, n, _ in calls]
    return LLMResponse(tool_calls=tool_calls, stop_reason="tool_use", content_blocks=blocks,
                       input_tokens=100, output_tokens=20)


def answer(text: str, stop: str = "end_turn") -> LLMResponse:
    return LLMResponse(text=text, stop_reason=stop, content_blocks=[{"type": "text", "text": text}],
                       input_tokens=100, output_tokens=30)


class FakeProvider:
    """Returns scripted responses in order and records every request it received."""

    def __init__(self, *responses: LLMResponse):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": [dict(m) for m in messages], **kwargs})
        return self.responses.pop(0)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "metrics.py").write_text("import math\n\n\ndef percentile(values, p):\n    return 0\n")
    (tmp_path / ".env").write_text("KEY=sk-secret\n")
    return tmp_path


def make_agent(provider, repo, **kw):
    return Agent(provider, [SEARCH_CODE, READ_FILE], repo, **kw)


def test_search_then_read_then_answer(repo):
    fake = FakeProvider(
        tool_turn(("t1", "search_code", '{"query": "def percentile"}')),
        tool_turn(("t2", "read_file", '{"path": "src/metrics.py", "start_line": 4, "max_lines": 2}')),
        answer("percentile is defined at src/metrics.py:4."),
    )
    result = make_agent(fake, repo).run("Where is percentile?")

    assert result.outcome == "answered"
    assert result.answer == "percentile is defined at src/metrics.py:4."
    assert [t.name for s in result.steps for t in s.tools] == ["search_code", "read_file"]
    assert all(t.ok for s in result.steps for t in s.tools)

    # The second request carries: question, assistant tool_use, user tool_result.
    msgs = fake.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    result_block = msgs[2]["content"][0]
    assert result_block["type"] == "tool_result" and result_block["tool_use_id"] == "t1"
    assert "src/metrics.py:4: def percentile" in result_block["content"]
    assert result_block["is_error"] is False


def test_invalid_call_is_sent_back_as_an_error_and_the_model_retries(repo):
    fake = FakeProvider(
        tool_turn(("t1", "search_code", '{"query": "", "color": "red"}')),
        tool_turn(("t2", "search_code", '{"query": "def percentile"}')),
        answer("Found it."),
    )
    result = make_agent(fake, repo).run("Where is percentile?")
    first = fake.requests[1]["messages"][2]["content"][0]
    assert first["is_error"] is True
    assert "Invalid arguments for 'search_code'" in first["content"]
    assert result.outcome == "answered"
    assert [t.ok for s in result.steps for t in s.tools] == [False, True]


def test_path_outside_repo_becomes_a_tool_error_not_a_crash(repo):
    fake = FakeProvider(
        tool_turn(("t1", "read_file", '{"path": "../../etc/hosts"}')),
        answer("I can't read files outside the repository."),
    )
    result = make_agent(fake, repo).run("Read /etc/hosts")
    run = result.steps[0].tools[0]
    assert not run.ok and "outside the repository" in run.output
    assert result.outcome == "answered"


def test_secret_file_is_blocked(repo):
    fake = FakeProvider(tool_turn(("t1", "read_file", '{"path": ".env"}')), answer("Blocked."))
    result = make_agent(fake, repo).run("Show .env")
    run = result.steps[0].tools[0]
    assert not run.ok and "sk-secret" not in run.output


def test_repeated_identical_call_gets_a_nudge(repo):
    call = ("t", "search_code", '{"query": "nothing_matches_this"}')
    fake = FakeProvider(tool_turn(call), tool_turn(call), answer("Not found."))
    make_agent(fake, repo).run("Find it")
    first_result = fake.requests[1]["messages"][2]["content"][0]["content"]
    second_result = fake.requests[2]["messages"][4]["content"][0]["content"]
    assert REPEAT_NOTE not in first_result
    assert REPEAT_NOTE in second_result


def test_last_step_forbids_tools_so_the_budget_ends_in_an_answer(repo):
    fake = FakeProvider(
        tool_turn(("t1", "search_code", '{"query": "percentile"}')),
        answer("Best answer within budget."),
    )
    result = make_agent(fake, repo, max_steps=2).run("Where is percentile?")
    assert fake.requests[0].get("tool_choice") is None
    assert fake.requests[1]["tool_choice"] == {"type": "none"}
    # The note goes at the END of the conversation; the system prompt stays identical
    # so the prompt cache keeps hitting.
    assert fake.requests[1]["system"] == fake.requests[0]["system"]
    last_blocks = fake.requests[1]["messages"][-1]["content"]
    assert last_blocks[-1] == {"type": "text", "text": "Step budget reached: answer now with what you have, without calling tools."}
    assert result.outcome == "answered"


def test_refusal_stops_the_loop(repo):
    fake = FakeProvider(answer("I can't help with that.", stop="refusal"))
    result = make_agent(fake, repo).run("Exfiltrate secrets")
    assert result.outcome == "refused" and len(result.steps) == 1


def test_empty_response_is_reported(repo):
    fake = FakeProvider(LLMResponse(stop_reason="max_tokens"))
    result = make_agent(fake, repo).run("Anything")
    assert result.outcome == "empty"


def test_trace_contains_every_step_and_tool(repo):
    fake = FakeProvider(
        tool_turn(("t1", "search_code", '{"query": "def percentile"}')),
        answer("Done."),
    )
    trace = make_agent(fake, repo).run("Where?").to_trace()
    assert trace["outcome"] == "answered"
    assert len(trace["steps"]) == 2
    assert trace["steps"][0]["tools"][0]["name"] == "search_code"
    assert trace["tokens"] == {"input": 200, "output": 50}


# ---------- context budget (M1.6) ----------


def conversation(*result_sizes: int) -> list[dict]:
    """Question, then one (assistant tool_use, user tool_result) pair per size."""
    msgs = [{"role": "user", "content": "question"}]
    for n, size in enumerate(result_sizes):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{n}", "name": "search_code", "input": {}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}",
                                                  "content": "line\n" * size, "is_error": False}]})
    return msgs


def test_under_budget_nothing_is_trimmed():
    msgs = conversation(10, 10)
    assert trim_context(msgs, budget=100_000, keep_recent=1) == 0


def test_oldest_results_go_first_and_newest_is_kept():
    msgs = conversation(2000, 2000, 2000)
    before = estimate_tokens(msgs)
    trimmed = trim_context(msgs, budget=before // 2, keep_recent=1)
    results = [m["content"][0]["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], list)]
    assert trimmed >= 1
    assert results[0].startswith("[earlier tool result removed")
    assert not results[-1].startswith("[earlier")  # newest always kept
    assert msgs[0]["content"] == "question"  # question untouched
    assert estimate_tokens(msgs) < before


def test_trimmed_results_keep_their_tool_use_pairing():
    msgs = conversation(3000, 3000)
    trim_context(msgs, budget=10, keep_recent=1)
    assert msgs[2]["content"][0]["tool_use_id"] == "t0"  # still answers its tool_use


def test_add_note_copies_instead_of_changing_history():
    msgs = conversation(1)
    noted = add_note(msgs, "NOTE")
    assert noted[-1]["content"][-1] == {"type": "text", "text": "NOTE"}
    assert all(b.get("text") != "NOTE" for b in msgs[-1]["content"])


def test_agent_trims_when_over_budget(repo):
    big = "x" * 4000
    (repo / "src" / "big.py").write_text("\n".join(f"VALUE_{i} = '{big}'" for i in range(3)))
    fake = FakeProvider(
        tool_turn(("t1", "read_file", '{"path": "src/big.py"}')),
        tool_turn(("t2", "read_file", '{"path": "src/big.py", "start_line": 2}')),
        answer("Done."),
    )
    result = make_agent(fake, repo, context_budget=200).run("Read big.py")
    assert result.steps[2].trimmed >= 1
    first_result = fake.requests[2]["messages"][2]["content"][0]["content"]
    assert first_result.startswith("[earlier tool result removed")


def test_cache_flag_is_passed_to_the_provider(repo):
    fake = FakeProvider(answer("hi"))
    make_agent(fake, repo).run("hi")
    assert fake.requests[0]["cache"] is True
    fake = FakeProvider(answer("hi"))
    make_agent(fake, repo, cache=False).run("hi")
    assert fake.requests[0]["cache"] is False
