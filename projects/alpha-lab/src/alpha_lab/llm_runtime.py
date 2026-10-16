"""Explicit, shared LLM controls for bounded unattended research runs."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class LLMRunError(RuntimeError):
    """A safe terminal error; agent-level retries must not repeat this call."""


def utc_now():
    return datetime.now(timezone.utc)


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
    deadline_utc: str | None = None

    def __post_init__(self):
        if self.model not in {"gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol"}:
            raise ValueError("Select an approved gateway model explicitly")
        if self.reasoning_effort not in {"medium", "high"}:
            raise ValueError("Select medium or high reasoning explicitly")
        for name, value in asdict(self).items():
            if name not in {"model", "reasoning_effort", "verbosity", "deadline_utc"}:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        if self.verbosity not in {"low", "medium", "high"}:
            raise ValueError("Invalid text verbosity")
        if self.deadline_utc is not None:
            try:
                deadline = datetime.fromisoformat(self.deadline_utc)
                if deadline.tzinfo is None or deadline.utcoffset() is None:
                    raise ValueError
            except (ValueError,TypeError):
                raise ValueError("deadline_utc requires an ISO timestamp with an explicit timezone") from None
            object.__setattr__(self,'deadline_utc',deadline.astimezone(timezone.utc).isoformat())


class LLMRuntime:
    """Serializes calls and reserves conservative tokens before sending requests.

    Failed calls keep their reservation because billed usage may be unavailable.
    Input UTF-8 bytes plus an 8192-token gateway allowance are reserved; upstream
    gateway-injected prompts cannot be bounded exactly by this client.
    """

    def __init__(self, settings: LLMSettings, ledger: Path | None = None):
        self.settings = settings
        self.ledger = Path(ledger).resolve() if ledger is not None else None
        self.intent = (self.ledger.with_suffix(self.ledger.suffix + ".intent.json")
                       if self.ledger is not None else None)
        self.run_id = uuid.uuid4().hex
        self.created_utc = utc_now().isoformat()
        self.started = time.monotonic()
        self.deadline = datetime.fromisoformat(settings.deadline_utc) if settings.deadline_utc else None
        self.calls = 0
        self.reserved_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.records = []
        self._lock = threading.Lock()
        # Request serialization spans network I/O; lifecycle inspection must not.
        self._state_lock = threading.Lock()
        self._ledger_lock = threading.Lock()
        self._request_active = False
        self._terminal_reason = None
        if self.ledger is not None:
            self._initialize_ledger()

    def _initialize_ledger(self) -> None:
        """Claim a fresh path once; reused evidence must never reset a budget."""
        try:
            if self.ledger.exists() or self.intent.exists():
                raise FileExistsError
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation handles competing initializers on one filesystem.
            # Keep this intent even after setup failure or usage-file loss.
            with self.intent.open("x", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "run_id": self.run_id,
                           "created_utc": self.created_utc,
                           "ledger": str(self.ledger), "settings": asdict(self.settings),
                           "restart_policy": "fresh_run_required; no automatic resume"},
                          handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.ledger.exists():
                raise FileExistsError
            self.save()
        except FileExistsError:
            raise LLMRunError("Existing LLM ledger or intent found; use a new run directory with reviewed controls") from None
        except OSError:
            raise LLMRunError("LLM ledger initialization failed; preserve its intent and use a new run directory") from None

    def remaining_seconds(self):
        remaining = self.settings.max_run_seconds - (time.monotonic()-self.started)
        if self.deadline is not None:
            remaining = min(remaining,(self.deadline-utc_now()).total_seconds())
        return remaining

    def request_timeout_seconds(self):
        return max(0.,min(self.settings.max_request_seconds,self.remaining_seconds()))

    @property
    def request_active(self) -> bool:
        with self._state_lock:
            return self._request_active

    def stop_reason(self) -> str | None:
        """Inspect global limits without waiting for a running SDK request."""
        if self.remaining_seconds() <= 0:
            return "deadline"
        with self._state_lock:
            if self._terminal_reason is not None:
                return self._terminal_reason
            if self.calls >= self.settings.max_calls:
                return "call_budget"
            used = max(self.reserved_tokens, self.input_tokens + self.output_tokens)
            # Even a one-token request needs this fixed reservation allowance.
            minimum = 8192 + 1 + len(json.dumps(["", [], []]).encode("utf-8"))
            if used + minimum > self.settings.max_reserved_tokens:
                return "token_budget"
        return None

    def check_deadline(self) -> None:
        """Already admitted tool results may finish only before the time cutoff."""
        if self.remaining_seconds() <= 0:
            raise LLMRunError("LLM run deadline reached")

    def check_active(self) -> None:
        reason = self.stop_reason()
        if reason == "deadline":
            raise LLMRunError("LLM run deadline reached")
        if reason is not None:
            raise LLMRunError("LLM call or reserved-token budget reached")

    def save(self):
        if self.ledger:
            with self._ledger_lock:
                with self._state_lock:
                    payload = {"schema_version": 2, "run_id": self.run_id,
                               "created_utc": self.created_utc,
                               "settings": asdict(self.settings), "calls": self.calls,
                               "reserved_tokens": self.reserved_tokens,
                               "reported_input_tokens": self.input_tokens,
                               "reported_output_tokens": self.output_tokens,
                               "request_active": self._request_active,
                               "terminal_reason": self._terminal_reason,
                               "records": self.records}
                    serialized = json.dumps(payload, indent=2)
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.ledger.with_suffix(".tmp")
                temporary.write_text(serialized, encoding="utf-8")
                temporary.replace(self.ledger)

    @contextmanager
    def request(self, system, history, tools, output_cap=None):
        with self._lock:
            s = self.settings
            if output_cap is not None and (isinstance(output_cap, bool)
                                          or not isinstance(output_cap, int) or output_cap <= 0):
                raise LLMRunError("Output token cap must be a positive integer")
            size = len(json.dumps([system, history, tools], ensure_ascii=False).encode("utf-8"))
            cap = min(output_cap or s.max_output_tokens, s.max_output_tokens)
            reserve = size + 8192 + cap
            if size > s.max_context_bytes:
                raise LLMRunError("Explicit context byte limit reached; no automatic truncation")
            self.check_deadline()
            with self._state_lock:
                used = max(self.reserved_tokens, self.input_tokens + self.output_tokens)
                if self._terminal_reason is None:
                    if self.calls >= s.max_calls:
                        self._terminal_reason = "call_budget"
                    elif used + reserve > s.max_reserved_tokens:
                        self._terminal_reason = "token_budget"
                reason = self._terminal_reason
                if reason is None:
                    self.calls += 1
                    self.reserved_tokens += reserve
                    record = {"call": self.calls, "reserved_tokens": reserve, "status": "started",
                              "remaining_run_seconds":round(self.remaining_seconds(),3)}
                    self.records.append(record)
                    self._request_active = True
            if reason is not None:
                self.save()
                raise LLMRunError("LLM call or reserved-token budget reached")
            request_start = time.monotonic()
            try:
                self.save()
                yield record
            except LLMRunError:
                with self._state_lock:
                    record["status"] = "failed"
                raise
            except Exception as error:
                with self._state_lock:
                    record["status"] = "failed"
                status = getattr(error, "status_code", None)
                # Never serialize arbitrary gateway error strings or request bodies.
                raise LLMRunError(f"LLM request failed: {type(error).__name__}, HTTP {status}") from None
            finally:
                with self._state_lock:
                    record["elapsed_seconds"] = round(time.monotonic() - request_start, 3)
                    self._request_active = False
                self.save()

    def record_usage(self, record, response):
        with self._state_lock:
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
