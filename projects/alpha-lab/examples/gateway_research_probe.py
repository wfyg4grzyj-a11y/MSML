"""Bounded medium/high reasoning, tools and summary compatibility probe."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from alpha_lab.llm_runtime import LLMSettings, bounded_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = LLMSettings(args.model, args.reasoning_effort, max_calls=4,
                           max_reserved_tokens=100_000, max_run_seconds=900)
    (args.output / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    load_dotenv(".env")
    provider = bounded_provider(settings, args.output / "usage.json")
    result = {"model": args.model, "reasoning_effort": args.reasoning_effort}
    tool = {"type": "function", "name": "add", "description": "Add two integers",
            "strict": True, "parameters": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"], "additionalProperties": False}}
    history = provider.build_user_items("Call add with 2 and 3, then report its returned answer.")
    try:
        responses = []
        for event in provider.stream_response(model=args.model, system="Use the add tool exactly once.",
                                              history=history, tools=[tool],
                                              reasoning_effort=args.reasoning_effort):
            if event.type == "done":
                responses.append(event.response)
        response = responses[-1]
        if len(response.tool_calls) != 1 or response.tool_calls[0].name != "add":
            raise RuntimeError("Expected add tool call")
        call = response.tool_calls[0]
        values = json.loads(call.arguments)
        assert values == {"a": 2, "b": 3}
        provider.append_response_to_history(history, response)
        history += provider.build_tool_result_items([{"call_id": call.call_id, "output": "5"}])
        response = None
        for event in provider.stream_response(model=args.model, system="Report the returned sum in one sentence.",
                                              history=history, tools=[tool],
                                              reasoning_effort=args.reasoning_effort):
            if event.type == "done":
                response = event.response
        result["tool_continuation"] = bool(response and "5" in response.text and not response.tool_calls)
        summary = provider.complete(model=args.model, system="Summarize in one sentence.",
                                    messages=[{"role": "user", "content": "The test added 2 and 3 and returned 5."}],
                                    max_tokens=2048)
        result["summary"] = bool(summary.strip())
        result["success"] = result["tool_continuation"] and result["summary"]
    except Exception as error:
        result.update(success=False, error_type=type(error).__name__)
        if type(error).__name__ == "LLMRunError":
            result["safe_error"] = str(error)
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
