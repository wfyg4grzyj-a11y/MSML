"""Shared limits constrain orchestration and tools, not just SDK requests."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from alpha_lab.agent import AgentLoop
from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.events import PhaseEvent, ToolCallEvent, ToolResultEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.llm_runtime import LLMRunError, LLMRuntime, LLMSettings
from alpha_lab.provider import ToolCall
from alpha_lab.provider_openai import OpenAIProvider
from alpha_lab.workspace_lock import WorkspaceLock


def runtime(**kwargs):
    return LLMRuntime(LLMSettings('gpt-6.1-sol', 'high', **kwargs))


def consume_one(control):
    with control.request('public synthetic definition', [], []) as record:
        control.record_usage(record, NS(usage=None))


def test_lifecycle_checks_do_not_wait_for_network_request(monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr('alpha_lab.llm_runtime.utc_now', lambda: now)
    control = runtime(max_calls=1, deadline_utc=(now + timedelta(seconds=10)).isoformat())
    started, release, inspected = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def request():
        try:
            with control.request('synthetic', [], []):
                started.set()
                assert release.wait(3)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=request)
    worker.start()
    assert started.wait(1)
    now += timedelta(seconds=10)
    observer = None
    try:
        # Deadline takes precedence over a full call budget.
        assert control.stop_reason() == 'deadline'
        now -= timedelta(seconds=1)
        assert control.stop_reason() == 'call_budget'
        now += timedelta(seconds=1)
        # The observer must finish while the request still owns serialization.
        def final_inspect():
            try:
                assert control.request_active and control.stop_reason() == 'deadline'
                with pytest.raises(LLMRunError, match='deadline'):
                    control.check_deadline()
                inspected.set()
            except BaseException as error:
                errors.append(error)
        observer = threading.Thread(target=final_inspect)
        observer.start()
        assert inspected.wait(1)
        assert worker.is_alive()
    finally:
        release.set()
        worker.join(2)
        if observer is not None and observer.ident is not None:
            observer.join(2)
    assert not errors and not control.request_active


@pytest.mark.parametrize('kind', ['elapsed', 'calls', 'tokens', 'reported_usage'])
def test_global_limit_reasons_are_side_effect_free(kind):
    control = runtime(max_calls=1)
    if kind == 'elapsed':
        control.started -= control.settings.max_run_seconds + 1
        expected = 'deadline'
    elif kind == 'calls':
        consume_one(control)
        expected = 'call_budget'
    elif kind == 'tokens':
        control = runtime(max_reserved_tokens=1)
        expected = 'token_budget'
    else:
        control.input_tokens = control.settings.max_reserved_tokens + 1
        expected = 'token_budget'
    before = (control.calls, control.reserved_tokens, len(control.records))
    assert control.stop_reason() == expected
    with pytest.raises(LLMRunError):
        control.check_active()
    assert (control.calls, control.reserved_tokens, len(control.records)) == before


def test_rejected_token_reservation_stops_later_smaller_requests(tmp_path):
    control = LLMRuntime(LLMSettings('gpt-6.1-sol', 'high', max_reserved_tokens=10_000), tmp_path / 'usage.json')
    with pytest.raises(LLMRunError, match='budget'):
        with control.request('synthetic', [], []):
            pytest.fail('Over-budget request must not be admitted')
    assert control.calls == 0 and control.stop_reason() == 'token_budget'
    with pytest.raises(LLMRunError, match='budget'):
        with control.request('', [], [], output_cap=1):
            pytest.fail('A rejected run must not automatically reset or shrink its budget')
    assert json.loads(control.ledger.read_text())['terminal_reason'] == 'token_budget'


@pytest.mark.parametrize('cap', [0, -1, True, 1.5, '20'])
def test_invalid_request_output_caps_do_not_reserve(cap):
    control = runtime()
    with pytest.raises(LLMRunError, match='positive integer'):
        with control.request('', [], [], output_cap=cap):
            pytest.fail('Invalid cap must not be admitted')
    assert control.calls == 0 and not control.request_active


def test_failed_initial_ledger_save_releases_request_state():
    control = runtime()
    control.save = MagicMock(side_effect=[OSError('synthetic failure'), None])
    with pytest.raises(LLMRunError, match='OSError'):
        with control.request('synthetic', [], []):
            pytest.fail('Failed reservation persistence must not admit network I/O')
    assert not control.request_active and control.records[0]['status'] == 'failed'
    assert control.calls == 1


def test_concurrent_ledger_saves_and_usage_leave_complete_snapshot(tmp_path):
    control = LLMRuntime(LLMSettings('gpt-6.1-sol', 'high'), tmp_path / 'usage.json')
    errors = []
    def task():
        try:
            with control.request('synthetic', [], []) as record:
                control.record_usage(record, NS(usage=NS(input_tokens=2, output_tokens=3)))
            control.save()
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=task) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert not errors and not any(thread.is_alive() for thread in threads)
    payload = json.loads(control.ledger.read_text())
    assert payload['calls'] == 5 and payload['reported_input_tokens'] == 10
    assert payload['reported_output_tokens'] == 15 and payload['request_active'] is False
    assert len(payload['records']) == 5 and all(row['status'] == 'completed' for row in payload['records'])


def make_dispatcher(tmp_path, control):
    events = []
    config = TaskConfig(data_path='synthetic', description='Offline', model='gpt-6.1-sol', reasoning_effort='high',
                        pipeline=PipelineConfig(phases=['phase3'], phase3=Phase3Config(gpu_ids=[], max_experiments=1)))
    executor = MagicMock()
    executor.can_submit.return_value = True
    executor.submit_experiment.return_value = 'cpu-synthetic'
    db = ExperimentDB(str(tmp_path / 'experiments.db'))
    dispatcher = Dispatcher(OpenAIProvider(MagicMock(), runtime=control), config, str(tmp_path), db,
                            None, events.append, worker_count=0, cpu_executor=executor)
    worker = MagicMock()
    worker.worker_id, worker.busy = 'synthetic-worker', False
    worker._thread = None
    worker.last_event_at = 0
    worker.current_experiment = None
    dispatcher.workers = [worker]
    dispatcher.recover = MagicMock()
    dispatcher._run_strategist = MagicMock()
    return dispatcher, db, worker, executor, events


@pytest.mark.parametrize('kind', ['deadline', 'call_budget', 'token_budget'])
def test_dispatcher_limit_stops_new_work_and_preserves_checked_state(tmp_path, kind):
    control = runtime()
    if kind == 'deadline':
        control.started -= control.settings.max_run_seconds + 1
    elif kind == 'call_budget':
        control = runtime(max_calls=1)
        consume_one(control)
    else:
        control = runtime(max_reserved_tokens=1)
    dispatcher, db, worker, executor, events = make_dispatcher(tmp_path, control)
    identifier = db.create('checked_fixture', 'D', 'H', '{}')
    db.update_status(identifier, 'checked')
    with pytest.raises(LLMRunError):
        dispatcher.run()
    assert events[-1].status == 'failed' and db.get(identifier).status == 'checked'
    dispatcher._run_strategist.assert_not_called()
    worker.implement.assert_not_called()
    worker.generate_report.assert_not_called()
    executor.submit_experiment.assert_not_called()
    executor.cleanup_all.assert_called_once()
    with WorkspaceLock(tmp_path):
        pass


def test_last_allowed_response_can_finish_board_normally(tmp_path):
    control = runtime(max_calls=1)
    consume_one(control)
    dispatcher, db, worker, executor, events = make_dispatcher(tmp_path, control)
    identifier = db.create('complete_fixture', 'D', 'H', '{}')
    db.update_status(identifier, 'done')
    assert dispatcher.run() is True
    assert events[-1].status == 'completed'
    dispatcher._run_strategist.assert_not_called()
    worker.implement.assert_not_called()
    executor.submit_experiment.assert_not_called()


def test_full_budget_drains_current_worker_without_new_starts(tmp_path):
    control = runtime(max_calls=1)
    consume_one(control)
    dispatcher, db, worker, executor, events = make_dispatcher(tmp_path, control)
    identifier = db.create('admitted_response', 'D', 'H', '{}')
    worker.busy = True
    dispatcher._poll_slurm = MagicMock()
    def finish():
        db.update_status(identifier, 'done')
        worker.busy = False
    dispatcher._wait_for_next_poll = MagicMock(side_effect=finish)
    assert dispatcher.run() is True
    dispatcher._poll_slurm.assert_called_once()
    dispatcher._run_strategist.assert_not_called()
    worker.implement.assert_not_called()
    worker.generate_report.assert_not_called()
    executor.submit_experiment.assert_not_called()
    assert events[-1].status == 'completed'


def test_deadline_interrupts_budget_drain(tmp_path, monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr('alpha_lab.llm_runtime.utc_now', lambda: now)
    control = runtime(max_calls=1, deadline_utc=(now + timedelta(seconds=1)).isoformat())
    consume_one(control)
    dispatcher, _, worker, executor, events = make_dispatcher(tmp_path, control)
    worker.busy = True
    dispatcher._poll_slurm = MagicMock()
    def elapse(seconds):
        nonlocal now
        now += timedelta(seconds=seconds)
    monkeypatch.setattr('alpha_lab.dispatcher.time.sleep', elapse)
    with pytest.raises(LLMRunError, match='deadline'):
        dispatcher.run()
    dispatcher._poll_slurm.assert_called_once()
    worker.stop.assert_called_once()
    executor.cleanup_all.assert_called_once()
    assert events[-1].status == 'failed'


def test_budget_reached_mid_iteration_admits_no_more_jobs(tmp_path):
    control = runtime(max_calls=1)
    dispatcher, db, worker, executor, _ = make_dispatcher(tmp_path, control)
    identifier = db.create('checked_fixture', 'D', 'H', '{}')
    db.update_status(identifier, 'checked')
    def last_request():
        consume_one(control)
        worker.busy = True
    dispatcher._run_strategist.side_effect = last_request
    dispatcher._wait_for_next_poll = lambda: setattr(worker, 'busy', False)
    with pytest.raises(LLMRunError, match='budget'):
        dispatcher.run()
    assert db.get(identifier).status == 'checked'
    executor.submit_experiment.assert_not_called()
    worker.implement.assert_not_called()


def test_submission_budget_race_retains_unsubmitted_reservation(tmp_path):
    control = runtime()
    dispatcher, db, _, executor, _ = make_dispatcher(tmp_path, control)
    identifier = db.create('checked_fixture', 'D', 'H', '{}')
    db.update_status(identifier, 'checked')
    control.stop_reason = MagicMock(side_effect=[None, None, 'call_budget'])
    dispatcher._submit_checked()
    row = db.get(identifier)
    assert row.status == 'queued' and row.slurm_job_id is None
    assert 'before submission' in row.error
    executor.submit_experiment.assert_not_called()


def make_agent(control, callback=None):
    provider = OpenAIProvider(MagicMock(), runtime=control)
    execute = MagicMock(return_value={'output': 'synthetic result', 'done': True})
    agent = AgentLoop(provider, 'gpt-6.1-sol', ContextManager(provider, 'gpt-6.1-sol'), callback or (lambda e: None),
                      tools=[{'type': 'function', 'name': 'evaluate'}], tool_executor=execute, min_report_attempts=1)
    return agent, execute


def test_expired_response_tools_do_not_execute_or_complete():
    control = runtime(deadline_utc='2000-01-01T00:00:00Z')
    agent, execute = make_agent(control)
    with pytest.raises(LLMRunError, match='deadline'):
        agent._handle_tool_calls([ToolCall('synthetic', 'evaluate', '{}')])
    execute.assert_not_called()
    assert not agent._done


def test_last_allowed_response_tools_finish_before_deadline():
    control = runtime(max_calls=1)
    consume_one(control)
    agent, execute = make_agent(control)
    agent._handle_tool_calls([ToolCall('synthetic', 'evaluate', '{}')])
    execute.assert_called_once()
    assert agent._done and control.calls == 1


@pytest.mark.parametrize('crossing', ['before_execute', 'after_execute'])
def test_tool_deadline_crossing_never_promotes_success(crossing, monkeypatch):
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr('alpha_lab.llm_runtime.utc_now', lambda: now)
    control = runtime(deadline_utc=(now + timedelta(seconds=1)).isoformat())
    events = []
    def callback(event):
        nonlocal now
        events.append(event)
        if crossing == 'before_execute' and isinstance(event, ToolCallEvent):
            now += timedelta(seconds=1)
    agent, execute = make_agent(control, callback)
    if crossing == 'after_execute':
        def late_tool(*args):
            nonlocal now
            now += timedelta(seconds=1)
            return {'output': 'synthetic result', 'done': True}
        execute.side_effect = late_tool
    with pytest.raises(LLMRunError, match='deadline'):
        agent._handle_tool_calls([ToolCall('synthetic', 'evaluate', '{}')])
    assert execute.call_count == (crossing == 'after_execute')
    assert not agent._done
    assert any(isinstance(event, ToolResultEvent) for event in events) is (crossing == 'after_execute')
