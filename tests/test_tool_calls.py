from pathlib import Path

import pytest

from pulse.llm.anthropic_provider import parse_events, thinking_off_param
from pulse.llm.base import ToolCall
from pulse.llm.replay import ReplayProvider, load_recording
from pulse.tools.registry import ToolCallError, validate_call
from pulse.tools.search_code import SEARCH_CODE, SearchCodeArgs

FIXTURE = Path(__file__).parent / "fixtures" / "handwritten_search_call.jsonl"
TOOLS = {SEARCH_CODE.name: SEARCH_CODE}


def call(args_json: str, name: str = "search_code") -> ToolCall:
    return ToolCall(id="toolu_test", name=name, arguments_json=args_json)


# ---------- parsing a recorded stream ----------


def test_replay_reassembles_tool_call_from_fragments():
    resp = ReplayProvider(FIXTURE).complete([{"role": "user", "content": "x"}])
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search_code"
    assert resp.tool_calls[0].arguments_json == '{"query": "def percentile", "path": "src"}'


def test_replay_reads_text_usage_and_timing():
    resp = parse_events(load_recording(FIXTURE))
    assert resp.text == "I'll search for it."
    assert resp.input_tokens == 612
    assert resp.output_tokens == 58
    assert resp.ttft_ms == pytest.approx(702.4)  # first visible text, not message_start
    assert not resp.empty


def test_tool_with_no_arguments_becomes_empty_object():
    events = [
        (None, {"type": "content_block_start", "index": 0,
                "content_block": {"type": "tool_use", "id": "t1", "name": "list", "input": {}}}),
        (None, {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
    ]
    assert parse_events(events).tool_calls[0].arguments_json == "{}"


def test_response_with_nothing_is_empty():
    events = [(None, {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}})]
    resp = parse_events(events)
    assert resp.empty and resp.cut_off


# ---------- validating tool calls ----------


def test_valid_call_applies_defaults():
    args = validate_call(call('{"query": "def percentile"}'), TOOLS)
    assert isinstance(args, SearchCodeArgs)
    assert args.path == "." and args.max_results == 10


def test_unknown_tool_lists_available_tools():
    with pytest.raises(ToolCallError, match="Unknown tool 'delete_repo'.*search_code"):
        validate_call(call("{}", name="delete_repo"), TOOLS)


def test_invalid_json_is_rejected():
    with pytest.raises(ToolCallError, match="not valid JSON"):
        validate_call(call('{"query": "def perc'), TOOLS)  # stream cut mid-way


def test_non_object_json_is_rejected():
    with pytest.raises(ToolCallError, match="must be a JSON object"):
        validate_call(call('["def percentile"]'), TOOLS)


@pytest.mark.parametrize(
    ("args_json", "problem"),
    [
        ("{}", "query: Field required"),
        ('{"query": ""}', "query: String should have at least 1 character"),
        ('{"query": "x", "max_results": 500}', "max_results: Input should be less than or equal to 50"),
        ('{"query": "x", "color": "red"}', "color: Extra inputs are not permitted"),
    ],
)
def test_schema_violations_explain_the_problem(args_json, problem):
    with pytest.raises(ToolCallError) as exc:
        validate_call(call(args_json), TOOLS)
    assert problem in str(exc.value)


def test_spec_matches_validation_model():
    schema = SEARCH_CODE.spec()["input_schema"]
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False


# ---------- provider quirks ----------


def test_thinking_off_differs_per_model():
    assert thinking_off_param("claude-sonnet-5-5") == {"type": "between_tools"}
    assert thinking_off_param("claude-haiku-5-5") == {"type": "disabled"}


def test_unknown_model_fails_loudly_instead_of_guessing():
    with pytest.raises(ValueError, match="THINKING_OFF"):
        thinking_off_param("claude-some-new-model")


# ---------- a real recorded response (Haiku 5.5, 2026-11-25) ----------

REAL = Path(__file__).parent / "fixtures" / "haiku_search_call.jsonl"


def test_real_recording_replays_to_a_valid_call():
    resp = ReplayProvider(REAL).complete([{"role": "user", "content": "x"}])
    assert resp.stop_reason == "tool_use"
    assert [c.name for c in resp.tool_calls] == ["search_code"]
    args = validate_call(resp.tool_calls[0], TOOLS)
    assert "percentile" in args.query
    assert resp.ttft_ms is not None and resp.total_ms is not None
    assert resp.ttft_ms < resp.total_ms
