"""Explicit, shared LLM controls for bounded unattended research runs."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


class LLMRunError(RuntimeError):
    """A safe terminal error; agent-level retries must not repeat this call."""


@dataclass(frozen=True)
class LLMSettings:
    model: str
    reasoning_effort: str
    max_output_tokens: int = 4096
    max_calls: int = 16
    max_reserved_tokens: int = 200_000
    max_run_seconds: int = 3600
    max_request_seconds: int = 240
    read_timeout_seconds: int = 120
    max_context_bytes: int = 64_000
    verbosity: str = "low"

    def __post_init__(self):
        if self.model not in {"gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol"}:
            raise ValueError("Select an approved gateway model explicitly")
        if self.reasoning_effort not in {"medium", "high"}:
            raise ValueError("Select medium or high reasoning explicitly")
        for name, value in asdict(self).items():
            if name not in {"model", "reasoning_effort", "verbosity"}:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        if self.verbosity not in {"low", "medium", "high"}:
            raise ValueError("Invalid text verbosity")


class LLMRuntime:
    """Serializes calls and reserves conservative tokens before sending requests.

    Failed calls keep their reservation because billed usage may be unavailable.
    Input UTF-8 bytes plus an 8192-token gateway allowance are reserved; upstream
    gateway-injected prompts cannot be bounded exactly by this client.
    """

    def __init__(self, settings: LLMSettings, ledger: Path | None = None):
        self.settings = settings
        self.ledger = ledger
        self.started = time.monotonic()
        self.calls = 0
        self.reserved_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.records = []
        self._lock = threading.Lock()

    def save(self):
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            payload = {"settings": asdict(self.settings), "calls": self.calls,
                       "reserved_tokens": self.reserved_tokens,
                       "reported_input_tokens": self.input_tokens,
                       "reported_output_tokens": self.output_tokens,
                       "records": self.records}
            temporary = self.ledger.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self.ledger)

    @contextmanager
    def request(self, system, history, tools, output_cap=None):
        with self._lock:
            s = self.settings
            size = len(json.dumps([system, history, tools], ensure_ascii=False).encode("utf-8"))
            cap = min(output_cap or s.max_output_tokens, s.max_output_tokens)
            reserve = size + 8192 + cap
            if size > s.max_context_bytes:
                raise LLMRunError("Explicit context byte limit reached; no automatic truncation")
            if time.monotonic() - self.started >= s.max_run_seconds:
                raise LLMRunError("LLM run deadline reached")
            used=max(self.reserved_tokens,self.input_tokens+self.output_tokens)
            if self.calls >= s.max_calls or used + reserve > s.max_reserved_tokens:
                raise LLMRunError("LLM call or reserved-token budget reached")
            self.calls += 1
            self.reserved_tokens += reserve
            record = {"call": self.calls, "reserved_tokens": reserve, "status": "started"}
            self.records.append(record)
            self.save()
            request_start = time.monotonic()
            try:
                yield record
            except LLMRunError:
                record["status"] = "failed"
                raise
            except Exception as error:
                record["status"] = "failed"
                status = getattr(error, "status_code", None)
                # Never serialize arbitrary gateway error strings or request bodies.
                raise LLMRunError(f"LLM request failed: {type(error).__name__}, HTTP {status}") from None
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - request_start, 3)
                self.save()

    def record_usage(self, record, response):
        usage = getattr(response, "usage", None)
        if usage:
            record["input_tokens"] = int(usage.input_tokens)
            record["output_tokens"] = int(usage.output_tokens)
            self.input_tokens += record["input_tokens"]
            self.output_tokens += record["output_tokens"]
            details=getattr(usage,"output_tokens_details",None)
            reasoning=getattr(details,"reasoning_tokens",None)
            if reasoning is not None:
                record["reasoning_tokens"]=int(reasoning)
        record["status"] = "completed"


def bounded_provider(settings: LLMSettings, ledger: Path):
    """No SDK retries; one runtime controls research and summary requests."""
    from alpha_lab.client import get_provider
    return get_provider("openai", runtime_settings=settings, ledger=ledger)
