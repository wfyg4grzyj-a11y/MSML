from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from alpha_lab.llm_runtime import LLMRunError, LLMRuntime, LLMSettings
from alpha_lab.provider_openai import OpenAIProvider
from alpha_lab.agent import AgentLoop
from alpha_lab.context import ContextManager
from alpha_lab.provider import ToolCall


def settings(**kwargs):
    return LLMSettings("gpt-5.6-sol", "medium", **kwargs)


def test_budget_reserves_failed_requests_and_redacts_error(tmp_path):
    runtime = LLMRuntime(settings(max_calls=1), tmp_path / "usage.json")
    with pytest.raises(LLMRunError, match="HTTP 429") as error:
        with runtime.request("system", [], []):
            bad = RuntimeError("secret-token-in-server-body")
            bad.status_code = 429
            raise bad
    assert "secret-token" not in str(error.value)
    assert runtime.calls == 1 and runtime.reserved_tokens > 0
    with pytest.raises(LLMRunError, match="budget"):
        with runtime.request("system", [], []):
            pytest.fail("Must not send another request")


@pytest.mark.parametrize("kind", ["response.incomplete", "response.failed", "error", "empty"])
def test_terminal_failure_never_becomes_completed_response(kind):
    client = MagicMock()
    events = [] if kind == "empty" else [NS(type=kind, response=NS(usage=None))]
    stream = MagicMock()
    stream.__iter__.return_value = iter(events)
    client.responses.create.return_value = stream
    provider = OpenAIProvider(client)
    with pytest.raises(LLMRunError):
        list(provider.stream_response(model="model", system="s", history=[], tools=[], reasoning_effort="medium"))
    stream.close.assert_called_once()


def test_bounded_summary_uses_selected_responses_settings():
    client = MagicMock()
    message = NS(type="message", content=[NS(type="output_text", text="Summary")])
    message.model_dump = lambda: {"type":"message", "role":"assistant", "content":[]}
    response = NS(id="r", output=[message], usage=NS(input_tokens=20, output_tokens=8))
    stream = MagicMock()
    stream.__iter__.return_value = iter([NS(type="response.completed", response=response)])
    client.responses.create.return_value = stream
    runtime = LLMRuntime(settings())
    provider = OpenAIProvider(client, runtime=runtime)
    assert provider.complete(model="gpt-5.6-sol",system="summarize",messages=[],max_tokens=2000)=="Summary"
    request = client.responses.create.call_args.kwargs
    assert request["reasoning"] == {"effort":"medium"}
    assert request["max_output_tokens"] == 2000 and request["store"] is False
    assert request["truncation"] == "disabled" and request["parallel_tool_calls"] is False
    assert runtime.calls == 1 and runtime.output_tokens == 8
    client.chat.completions.create.assert_not_called()


def test_custom_tool_allowlist_blocks_unadvertised_subagent(tmp_path):
    provider = OpenAIProvider(MagicMock())
    executor = MagicMock(return_value={"output":"ok"})
    agent = AgentLoop(provider, "gpt-5.6-sol", ContextManager(provider,"gpt-5.6-sol",str(tmp_path)),
                      lambda event:None, tools=[{"type":"function","name":"evaluate"}],
                      tool_executor=executor)
    result = agent._handle_tool_calls([ToolCall("c","spawn_sub_agent",'{"task":"do work"}')])
    assert "not enabled" in result[0]["output"]
    executor.assert_not_called()


def test_terminal_error_not_retried_by_agent(tmp_path):
    provider = MagicMock()
    provider.stream_response.side_effect = LLMRunError("budget reached")
    agent = AgentLoop(provider,"gpt-5.6-sol",ContextManager(provider,"gpt-5.6-sol",str(tmp_path)),
                      lambda event:None,prompt_builder=lambda *a:"s")
    assert agent._call_api([]) is None
    assert provider.stream_response.call_count == 1


def test_failed_summary_preserves_provider_history_and_prior_summary(tmp_path):
    provider = MagicMock()
    provider.runtime = None
    provider.complete.side_effect = LLMRunError("Summary budget reached")
    context = ContextManager(provider,"gpt-5.6-sol",str(tmp_path))
    context.summary = "Earlier findings"
    context.should_summarize = lambda: True
    context.add_entry("user","Earlier user message")
    agent = AgentLoop(provider,"gpt-5.6-sol",context,lambda event:None)
    agent._input_history = [{"role":"user","content":"Preserve this history"}]
    agent._run_loop = MagicMock()
    agent.send_user_message("Continue")
    assert agent._input_history[0]["content"] == "Preserve this history"
    assert context.summary == "Earlier findings"
    assert context.last_summarization_succeeded is False


def test_empty_summary_does_not_destroy_context():
    provider = MagicMock()
    provider.complete.return_value = ""
    context = ContextManager(provider,"gpt-5.6-sol")
    context.summary = "Prior summary"
    context.add_entry("user","Old message")
    assert context.summarize_and_fork() == "Prior summary"
    assert len(context.entries)==1 and not context.last_summarization_succeeded


def test_bounded_config_requires_intentional_model_and_effort(tmp_path):
    import json
    from alpha_lab.config import load_config
    path=tmp_path/"config.json"
    raw={"data_path":"data", "description":"test", "llm_runtime":{"max_calls":4}}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="explicit"):
        load_config(path)
    raw.update(model="gpt-6-sol",reasoning_effort="high")
    path.write_text(json.dumps(raw))
    config=load_config(path)
    assert config.llm_runtime.max_calls==4 and config.llm_runtime.reasoning_effort=="high"


def test_unexpected_gateway_usage_blocks_further_calls():
    runtime=LLMRuntime(settings(max_reserved_tokens=50_000))
    with runtime.request("s",[],[]) as record:
        runtime.record_usage(record,NS(usage=NS(input_tokens=60_000,output_tokens=10)))
    with pytest.raises(LLMRunError,match="budget"):
        with runtime.request("s",[],[]):
            pytest.fail("Gateway-added usage must count against the next-call budget")
