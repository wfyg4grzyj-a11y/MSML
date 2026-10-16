"""OpenAI Responses API provider for alpha-lab.

Implements the Provider protocol using the OpenAI SDK's Responses API
with ZDR-compatible settings (store=False, local history tracking).
"""

from __future__ import annotations

from collections.abc import Iterator
import threading
import time
from typing import Any

import httpx
from openai import OpenAI

from alpha_lab.provider import Provider, Response, StreamEvent, ToolCall
from alpha_lab.llm_runtime import LLMRunError


class OpenAIProvider:
    """Provider backed by the OpenAI Responses API."""

    def __init__(self, client: OpenAI, *, runtime=None) -> None:
        self._client = client
        self.runtime = runtime

    def close(self) -> None:
        """Release the SDK client, including when legacy proxy access is disabled."""
        self._client.close()

    @property
    def openai_client(self) -> OpenAI | None:
        """Expose the underlying OpenAI client (used for web search proxy)."""
        if self.runtime is not None:
            # Legacy proxy tools must not bypass shared budgets/model selection.
            return None
        return self._client

    # ------------------------------------------------------------------
    # stream_response
    # ------------------------------------------------------------------

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str,
    ) -> Iterator[StreamEvent]:
        if self.runtime is None:
            yield from self._stream_response(model=model, system=system, history=history,
                                            tools=tools, reasoning_effort=reasoning_effort)
            return
        settings = self.runtime.settings
        if model != settings.model:
            raise LLMRunError("Request model differs from the intentional run selection")
        with self.runtime.request(system, history, tools) as record:
            yield from self._stream_response(
                model=model, system=system, history=history, tools=tools,
                reasoning_effort=settings.reasoning_effort, record=record,
                options={"max_output_tokens": settings.max_output_tokens,
                         "parallel_tool_calls": False, "truncation": "disabled",
                         "text": {"verbosity": settings.verbosity}})

    def _stream_response(self, *, model, system, history, tools, reasoning_effort,
                         record=None, options=None):
        """Stream a response via the OpenAI Responses API."""
        started=time.monotonic()
        request = dict(
            model=model,
            instructions=system,
            input=history,
            tools=tools,
            store=False,  # Does not establish the gateway's retention policy.
            truncation="auto",
            stream=True,
            reasoning={"effort": reasoning_effort},
            include=["reasoning.encrypted_content"],
        )
        request.update(options or {})
        request_budget = None
        if self.runtime:
            request_budget = self.runtime.request_timeout_seconds()
            if request_budget<=0:
                raise LLMRunError("LLM run deadline reached before request")
            # Per-phase SDK timeouts are also shortened near the absolute cutoff.
            request['timeout'] = httpx.Timeout(request_budget,
                read=min(self.runtime.settings.read_timeout_seconds,request_budget),
                connect=min(20.,request_budget))
        stream = self._client.responses.create(**request)
        timer = None
        expired = threading.Event()
        if self.runtime:
            def close_at_deadline():
                expired.set()
                try:
                    stream.close()
                except (OSError, RuntimeError):
                    pass
            remaining=min(request_budget-(time.monotonic()-started),self.runtime.remaining_seconds())
            if remaining<=0:
                stream.close()
                raise LLMRunError("LLM request deadline reached before streaming")
            timer = threading.Timer(remaining, close_at_deadline)
            timer.daemon = True
            timer.start()

        full_text = ""
        raw_response = None

        try:
            for event in stream:
                if event.type == "response.output_text.delta":
                    full_text += event.delta
                    yield StreamEvent(type="text_delta", delta=event.delta)
                elif event.type == "response.completed":
                    raw_response = event.response
                elif event.type in {"response.failed", "response.incomplete"}:
                    if self.runtime and record is not None:
                        self.runtime.record_usage(record, event.response)
                    raise LLMRunError(f"LLM stream terminated with {event.type}")
                elif event.type == "error":
                    raise LLMRunError("LLM stream reported an error event")
                if expired.is_set() or (self.runtime and self.runtime.remaining_seconds()<=0):
                    if self.runtime and record is not None and raw_response is not None:
                        self.runtime.record_usage(record,raw_response)
                    raise LLMRunError("LLM request deadline reached")
        finally:
            if timer:
                timer.cancel()
            try:
                stream.close()
            except (OSError, RuntimeError):
                pass  # Stream already closed or connection lost

        if raw_response is None:
            raise LLMRunError("LLM stream ended without a completed response")
        if self.runtime and record is not None:
            self.runtime.record_usage(record, raw_response)
        if getattr(raw_response,"status",None) not in {None,"completed"}:
            raise LLMRunError("LLM response did not have completed status")

        # Parse the completed response into normalized form
        text_output = ""
        tool_calls: list[ToolCall] = []
        has_web_search = False
        raw_output_items: list[dict[str, Any]] = []

        for item in raw_response.output:
            # Serialize to dict for raw_output_items
            try:
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "to_dict"):
                    item_dict = item.to_dict()
                else:
                    item_dict = dict(item)
                raw_output_items.append(item_dict)
            except (TypeError, ValueError, AttributeError):
                pass  # Non-serializable item, skip

            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text_output += content.text
            elif item.type == "function_call":
                tool_calls.append(ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                ))
            elif item.type == "web_search_call":
                has_web_search = True

        input_tokens = 0
        output_tokens = 0
        if hasattr(raw_response, "usage") and raw_response.usage:
            input_tokens = raw_response.usage.input_tokens
            output_tokens = raw_response.usage.output_tokens

        response = Response(
            id=raw_response.id,
            text=text_output,
            tool_calls=tool_calls,
            has_web_search=has_web_search,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_output_items=raw_output_items,
        )

        yield StreamEvent(type="done", response=response)

    # ------------------------------------------------------------------
    # complete (non-streaming, for summarization)
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> str:
        if self.runtime is not None:
            settings = self.runtime.settings
            if model != settings.model:
                raise LLMRunError("Summary model differs from the intentional run selection")
            with self.runtime.request(system, messages, [], max_tokens) as record:
                completed = None
                for event in self._stream_response(
                    model=model, system=system, history=messages, tools=[],
                    reasoning_effort=settings.reasoning_effort, record=record,
                    options={"max_output_tokens": min(max_tokens, settings.max_output_tokens),
                             "parallel_tool_calls": False, "truncation": "disabled",
                             "text": {"verbosity": settings.verbosity}}):
                    if event.type == "done":
                        completed = event.response
                if completed is None or not completed.text.strip():
                    raise LLMRunError("Summary produced no completed text")
                return completed.text
        """Simple non-streaming completion via Chat Completions API."""
        chat_messages = [{"role": "system", "content": system}] + messages
        # Newer models (gpt-5-mini, gpt-5.4+) require max_completion_tokens
        # instead of max_tokens
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=chat_messages,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            if "max_completion_tokens" in str(e):
                response = self._client.chat.completions.create(
                    model=model,
                    messages=chat_messages,
                    max_tokens=max_tokens,
                )
            else:
                raise
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        """Build OpenAI Responses API user input items."""
        return [{"role": "user", "content": message}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build function_call_output items + optional image injection."""
        items: list[dict[str, Any]] = []

        for r in results:
            items.append({
                "type": "function_call_output",
                "call_id": r["call_id"],
                "output": r["output"],
            })

        if images:
            image_content: list[dict[str, Any]] = []
            for b64_data, media_type in images:
                image_content.append({
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{b64_data}",
                })
            image_content.append({
                "type": "input_text",
                "text": "Here is the image you requested. Analyze it carefully.",
            })
            items.append({
                "role": "user",
                "content": image_content,
            })

        return items

    def append_response_to_history(
        self,
        history: list[dict[str, Any]],
        response: Response,
    ) -> None:
        """Append cleaned output items to the conversation history."""
        for item_dict in response.raw_output_items:
            cleaned = self._clean_output_item(item_dict)
            if cleaned:
                history.append(cleaned)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_output_item(item: dict[str, Any]) -> dict[str, Any] | None:
        """Clean an output item for use as input in next turn.

        Removes response-specific fields that aren't valid for input.
        """
        item_type = item.get("type", "")
        exclude_fields = {"status", "id"}

        if item_type == "message":
            return {
                "type": item_type,
                "role": item.get("role", "assistant"),
                "content": item.get("content", ""),
            }

        if item_type == "reasoning":
            return {k: v for k, v in item.items() if k not in exclude_fields}

        if item_type == "function_call":
            return {
                "type": item_type,
                "call_id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            }

        cleaned = {k: v for k, v in item.items() if k not in exclude_fields}
        return cleaned if cleaned else None
