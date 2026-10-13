"""Persistent intent rejects budget reset without ever creating SDK requests."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.client import get_provider
from alpha_lab.llm_runtime import LLMRunError, LLMRuntime, LLMSettings


def settings(**kwargs):
    return LLMSettings('gpt-6-sol', 'high', **kwargs)


def test_fresh_intent_and_zero_usage_precede_first_request(tmp_path):
    control = LLMRuntime(settings(), tmp_path / 'usage.json')
    intent = json.loads(control.intent.read_text())
    usage = json.loads(control.ledger.read_text())
    assert intent['settings'] == usage['settings'] == asdict(control.settings)
    assert intent['run_id'] == usage['run_id'] == control.run_id
    assert intent['ledger'] == str(control.ledger.resolve())
    assert usage['calls'] == 0 and usage['records'] == []
    assert usage['schema_version'] == 2 and 'no automatic resume' in intent['restart_policy']


@pytest.mark.parametrize('state', ['zero_calls', 'completed', 'failed'])
def test_existing_ledger_cannot_reset_and_preserves_both_files(tmp_path, state):
    control = LLMRuntime(settings(max_calls=1), tmp_path / 'usage.json')
    if state != 'zero_calls':
        if state == 'completed':
            with control.request('synthetic', [], []) as record:
                control.record_usage(record, NS(usage=NS(input_tokens=2, output_tokens=3)))
        else:
            with pytest.raises(LLMRunError):
                with control.request('synthetic', [], []):
                    raise LLMRunError('synthetic failure')
    before = (control.ledger.read_bytes(), control.intent.read_bytes())
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
        LLMRuntime(control.settings, control.ledger)
    assert (control.ledger.read_bytes(), control.intent.read_bytes()) == before


@pytest.mark.parametrize('contents', ['{}', '{truncated', 'legacy-sensitive-fixture'])
def test_legacy_or_corrupt_evidence_is_rejected_without_reading_or_overwrite(tmp_path, contents):
    ledger = tmp_path / 'usage.json'
    ledger.write_text(contents)
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent') as error:
        LLMRuntime(settings(), ledger)
    assert ledger.read_text() == contents and contents not in str(error.value)
    assert not ledger.with_suffix('.json.intent.json').exists()


def test_usage_file_loss_does_not_release_immutable_intent(tmp_path):
    control = LLMRuntime(settings(), tmp_path / 'usage.json')
    original = control.intent.read_bytes()
    control.ledger.unlink()
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
        LLMRuntime(control.settings, control.ledger)
    assert control.intent.read_bytes() == original and not control.ledger.exists()


def test_relative_alias_cannot_claim_same_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    control = LLMRuntime(settings(), Path('usage.json'))
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
        LLMRuntime(settings(), tmp_path / '.' / 'usage.json')
    assert control.ledger == (tmp_path / 'usage.json').resolve()


def test_concurrent_thread_initializers_have_one_winner(tmp_path):
    ledger = tmp_path / 'usage.json'
    gate = threading.Barrier(5)
    def initialize():
        gate.wait(timeout=3)
        try:
            return LLMRuntime(settings(), ledger)
        except LLMRunError:
            return None
    with ThreadPoolExecutor(max_workers=5) as pool:
        controls = list(pool.map(lambda _: initialize(), range(5)))
    winners = [control for control in controls if control is not None]
    assert len(winners) == 1
    payload = json.loads(ledger.read_text())
    assert payload['calls'] == 0 and payload['run_id'] == winners[0].run_id


def test_independent_process_initializers_have_one_winner(tmp_path):
    ledger, gate = tmp_path / 'usage.json', tmp_path / 'ready'
    code = '''
import sys,time
from pathlib import Path
from alpha_lab.llm_runtime import LLMRuntime,LLMSettings,LLMRunError
for _ in range(100):
    if Path(sys.argv[2]).exists(): break
    time.sleep(.05)
else: raise RuntimeError('fixture gate timeout')
try:
    LLMRuntime(LLMSettings('gpt-6-sol','high'),Path(sys.argv[1]))
except LLMRunError:
    sys.exit(3)
'''
    children = [subprocess.Popen([sys.executable, '-c', code, str(ledger), str(gate)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        gate.write_text('ready')
        for child in children:
            child.communicate(timeout=10)
        assert sorted(child.returncode for child in children) == [0, 3]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=3)
    assert json.loads(ledger.read_text())['calls'] == 0
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
        LLMRuntime(settings(), ledger)


def test_crash_after_durable_intent_cannot_reset_budget(tmp_path):
    ledger = tmp_path / 'usage.json'
    code = '''
import os,sys
from pathlib import Path
from alpha_lab.llm_runtime import LLMRuntime,LLMSettings
LLMRuntime.save=lambda self: os._exit(31)
LLMRuntime(LLMSettings('gpt-6-sol','high'),Path(sys.argv[1]))
'''
    result = subprocess.run([sys.executable, '-c', code, str(ledger)], capture_output=True, timeout=10)
    assert result.returncode == 31 and not ledger.exists()
    intent = ledger.with_suffix('.json.intent.json')
    preserved = intent.read_bytes()
    assert json.loads(preserved)['run_id']
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
        LLMRuntime(settings(), ledger)
    assert intent.read_bytes() == preserved


def test_initial_usage_write_failure_preserves_intent(tmp_path):
    ledger = tmp_path / 'usage.json'
    with patch.object(LLMRuntime, 'save', side_effect=OSError('synthetic persistence failure')):
        with pytest.raises(LLMRunError, match='initialization failed'):
            LLMRuntime(settings(), ledger)
    assert ledger.with_suffix('.json.intent.json').exists()
    with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
        LLMRuntime(settings(), ledger)


def test_factory_rejects_old_ledger_before_sdk_setup(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    control = LLMRuntime(settings(), tmp_path / 'usage.json')
    with patch('alpha_lab.client.OpenAI') as sdk:
        with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
            get_provider(runtime_settings=settings(), ledger=control.ledger)
        sdk.assert_not_called()


def test_factory_checks_missing_key_before_claiming_intent(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    ledger = tmp_path / 'usage.json'
    with patch('alpha_lab.client.OpenAI') as sdk:
        with pytest.raises(ValueError, match='OPENAI_API_KEY'):
            get_provider(runtime_settings=settings(), ledger=ledger)
        sdk.assert_not_called()
    assert not ledger.exists() and not ledger.with_suffix('.json.intent.json').exists()


def test_new_bounded_provider_requires_durable_ledger(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    with patch('alpha_lab.client.OpenAI') as sdk:
        with pytest.raises(ValueError, match='fresh ledger'):
            get_provider(runtime_settings=settings())
        sdk.assert_not_called()


def test_sdk_setup_failure_retains_zero_usage_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    ledger = tmp_path / 'usage.json'
    with patch('alpha_lab.client.OpenAI', side_effect=RuntimeError('synthetic SDK setup failure')) as sdk:
        with pytest.raises(RuntimeError, match='SDK setup failure'):
            get_provider(runtime_settings=settings(), ledger=ledger)
        assert json.loads(ledger.read_text())['calls'] == 0
        with pytest.raises(LLMRunError, match='Existing LLM ledger or intent'):
            get_provider(runtime_settings=settings(), ledger=ledger)
        assert sdk.call_count == 1


def test_shared_live_runtime_preserves_intent_and_counters(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    control = LLMRuntime(settings(), tmp_path / 'usage.json')
    with control.request('synthetic', [], []) as record:
        control.record_usage(record, NS(usage=None))
    original = control.intent.read_bytes()
    started = control.started
    with patch('alpha_lab.client.OpenAI', return_value=MagicMock()) as sdk:
        provider = get_provider(runtime=control)
        assert provider.runtime is control and control.calls == 1 and control.started == started
        sdk.assert_called_once()
        provider.close()
    assert control.intent.read_bytes() == original


@pytest.mark.parametrize('limit', ['deadline', 'call_budget'])
def test_factory_rejects_exhausted_shared_runtime_before_sdk(limit, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    control = LLMRuntime(settings(max_calls=1))
    if limit == 'deadline':
        control.started -= control.settings.max_run_seconds + 1
    else:
        with control.request('synthetic', [], []) as record:
            control.record_usage(record, NS(usage=None))
    with patch('alpha_lab.client.OpenAI') as sdk:
        with pytest.raises(LLMRunError):
            get_provider(runtime=control)
        sdk.assert_not_called()
