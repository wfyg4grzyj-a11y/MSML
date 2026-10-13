"""Failed or interrupted orchestration must not advance or report success."""
from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.config import Phase3Config, PipelineConfig, TaskConfig
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.events import PhaseEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.pipeline import Pipeline, StepResult, detect_phase1_complete
from alpha_lab.workspace_lock import WorkspaceLock


def prepare_dispatcher(tmp_path):
    events = []
    executor = MagicMock()
    db = ExperimentDB(str(tmp_path / 'experiments.db'))
    dispatcher = Dispatcher(MagicMock(), TaskConfig(data_path='synthetic', description='Offline'),
                            str(tmp_path), db, executor, events.append, worker_count=0)
    for name in ['recover', '_submit_checked', '_maybe_generate_report', '_assign_workers',
                 '_track_analyzed', '_check_stale', '_check_stuck_workers',
                 '_maybe_supervisor_check', '_emit_board_summary', '_poll_slurm']:
        setattr(dispatcher, name, MagicMock())
    dispatcher._should_run_strategist = MagicMock(return_value=False)
    dispatcher._should_terminate = MagicMock(return_value=True)
    return dispatcher, events, executor, db


@pytest.mark.parametrize('stage', ['recover', '_poll_slurm', '_submit_checked', '_assign_workers'])
def test_dispatcher_failure_propagates_and_releases_resources(tmp_path, stage):
    dispatcher, events, executor, db = prepare_dispatcher(tmp_path)
    identifier = db.create('pending', 'D', 'H', '{}')
    getattr(dispatcher, stage).side_effect = RuntimeError('synthetic failure')
    with pytest.raises(RuntimeError, match='synthetic failure'):
        dispatcher.run()
    assert [e.status for e in events if isinstance(e, PhaseEvent)] == ['starting', 'failed']
    assert db.get(identifier).status == 'to_implement'
    if stage == 'recover':
        dispatcher._should_run_strategist.assert_not_called()
        dispatcher._poll_slurm.assert_not_called()
        dispatcher._submit_checked.assert_not_called()
    executor.cleanup_all.assert_called_once()
    assert dispatcher._log_file is None
    with WorkspaceLock(tmp_path):
        pass


@pytest.mark.parametrize('mode', ['completed', 'requested_stop', 'keyboard_interrupt'])
def test_dispatcher_terminal_status_distinguishes_stop(tmp_path, mode):
    dispatcher, events, executor, _ = prepare_dispatcher(tmp_path)
    if mode == 'requested_stop':
        dispatcher._poll_slurm.side_effect = lambda: setattr(dispatcher, '_stop_requested', True)
    elif mode == 'keyboard_interrupt':
        dispatcher._poll_slurm.side_effect = KeyboardInterrupt()
    if mode == 'keyboard_interrupt':
        with pytest.raises(KeyboardInterrupt):
            dispatcher.run()
    else:
        assert dispatcher.run() is (mode == 'completed')
    assert events[-1].status == ('completed' if mode == 'completed' else 'stopped')
    executor.cleanup_all.assert_called_once()
    with WorkspaceLock(tmp_path):
        pass


def prepare_pipeline(tmp_path):
    events = []
    pipeline = Pipeline(MagicMock(), TaskConfig(data_path='synthetic', description='Offline'),
                        str(tmp_path), events.append)
    pipeline._load_phase1_context = MagicMock(return_value='')
    pipeline._read_review = MagicMock(return_value='Verdict: PASS')
    pipeline._run_tests = MagicMock(return_value='[exit code: 0]')
    pipeline._run_step = MagicMock(side_effect=lambda step, **kwargs: StepResult(step.name, True))
    return pipeline, events


@pytest.mark.parametrize('stage', ['builder', 'critic', 'tester'])
def test_failed_phase2_step_rejects_stale_pass_evidence(tmp_path, stage):
    pipeline, events = prepare_pipeline(tmp_path)
    pipeline._run_step.side_effect = lambda step, **kwargs: StepResult(step.name, step.name != stage)
    with patch('alpha_lab.pipeline.detect_phase2_progress', return_value=stage):
        assert pipeline.run_phase2() is False
    assert events[-1].step == 'complete' and events[-1].status == 'failed'
    assert pipeline._run_step.call_count == 1
    pipeline._read_review.assert_not_called()
    pipeline._run_tests.assert_not_called()


@pytest.mark.parametrize('start', ['builder', 'critic', 'tester', 'done'])
def test_phase2_success_returns_explicit_true(tmp_path, start):
    pipeline, events = prepare_pipeline(tmp_path)
    with patch('alpha_lab.pipeline.detect_phase2_progress', return_value=start):
        assert pipeline.run_phase2() is True
    assert events[-1].step == 'complete' and events[-1].status == 'completed'
    if start == 'done':
        pipeline._run_step.assert_not_called()


@pytest.mark.parametrize('when', ['before', 'during'])
def test_stopped_phase2_never_returns_success(tmp_path, when):
    pipeline, events = prepare_pipeline(tmp_path)
    if when == 'before':
        pipeline.stop()
    else:
        def stop_step(step, **kwargs):
            pipeline.stop()
            return StepResult(step.name, True)
        pipeline._run_step.side_effect = stop_step
    with patch('alpha_lab.pipeline.detect_phase2_progress', return_value='builder'):
        assert pipeline.run_phase2() is False
    assert not any(e.step == 'complete' and e.status == 'completed' for e in events)
    pipeline._run_tests.assert_not_called()


def test_failed_framework_tests_exhaust_limit_without_success(tmp_path):
    pipeline, events = prepare_pipeline(tmp_path)
    pipeline._run_tests.return_value = '[exit code: 1]'
    pipeline.config.pipeline.max_fix_iterations = 1
    with patch('alpha_lab.pipeline.detect_phase2_progress', return_value='tester'):
        assert pipeline.run_phase2() is False
    assert events[-1].status == 'failed'


@pytest.mark.parametrize('report', ['', ' \n\t', '# Schema\nObserved fields'])
def test_empty_phase1_report_is_not_resume_evidence(tmp_path, report):
    (tmp_path / 'learnings.md').write_text('Synthetic learnings')
    directory = tmp_path / 'data_report'
    directory.mkdir()
    (directory / 'schema.md').write_text(report)
    assert detect_phase1_complete(str(tmp_path)) is bool(report.strip())


@pytest.fixture
def headless(tmp_path, monkeypatch):
    from alpha_lab import run
    workspace = tmp_path / 'workspace'
    monkeypatch.setattr(sys, 'argv', ['alpha-lab-run', '--config', 'synthetic.yaml', '--workspace', str(workspace)])
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    config = TaskConfig(data_path='synthetic', description='Offline', pipeline=PipelineConfig(
        phases=['phase1', 'phase2', 'phase3'], phase3=Phase3Config(gpu_ids=[])))
    provider = MagicMock(spec=['close'])
    agent = MagicMock()
    agent._done, agent._stop_requested = True, False
    pipeline = MagicMock()
    pipeline.run_phase2.return_value = True
    pipeline._stop_requested = False
    dispatcher = MagicMock()
    dispatcher.run.return_value = True
    with ExitStack() as stack:
        stack.enter_context(patch('alpha_lab.run.load_config', return_value=config))
        stack.enter_context(patch('alpha_lab.client.get_provider', return_value=provider))
        stack.enter_context(patch('alpha_lab.phase0.run_phase0', return_value=MagicMock()))
        supervisor = stack.enter_context(patch('alpha_lab.supervisor.Supervisor')).return_value
        stack.enter_context(patch('alpha_lab.run.ContextManager'))
        stack.enter_context(patch('alpha_lab.run.AgentLoop', return_value=agent))
        pipeline_factory = stack.enter_context(patch('alpha_lab.run.Pipeline', return_value=pipeline))
        dispatcher_factory = stack.enter_context(patch('alpha_lab.dispatcher.Dispatcher', return_value=dispatcher))
        cpu_factory = stack.enter_context(patch('alpha_lab.local_cpu.LocalCPUManager'))
        detection = stack.enter_context(patch('alpha_lab.run.detect_phase1_complete', side_effect=[False, True, True]))
        yield dict(run=run, workspace=workspace, config=config, provider=provider, agent=agent,
                   pipeline=pipeline, pipeline_factory=pipeline_factory, dispatcher=dispatcher,
                   dispatcher_factory=dispatcher_factory, cpu_factory=cpu_factory,
                   detection=detection, supervisor=supervisor)


def assert_headless_closed(headless, expected):
    run = headless['run']
    headless['provider'].close.assert_called_once()
    assert run._event_log_file is None
    rows = [json.loads(line) for line in (headless['workspace'].parent / 'events.jsonl').read_text().splitlines()]
    terminal = [row for row in rows if row.get('phase') == 'run' and row.get('step') == 'complete']
    assert len(terminal) == 1 and terminal[0]['status'] == expected
    with WorkspaceLock(headless['workspace']):
        pass


@pytest.mark.parametrize('mode', ['agent_failure', 'missing_output', 'stop'])
def test_headless_phase1_failure_prevents_later_work(headless, mode):
    if mode == 'agent_failure':
        headless['agent']._done = False
    elif mode == 'missing_output':
        headless['detection'].side_effect = [False, False]
    else:
        headless['agent']._stop_requested = True
    with pytest.raises(RuntimeError, match='Phase 1'):
        headless['run'].run_main()
    headless['supervisor'].review_phase1.assert_not_called()
    headless['pipeline_factory'].assert_not_called()
    headless['dispatcher_factory'].assert_not_called()
    assert_headless_closed(headless, 'stopped' if mode == 'stop' else 'failed')


@pytest.mark.parametrize('mode', ['failure', 'stop'])
def test_headless_phase2_failure_prevents_experiments(headless, mode):
    headless['pipeline'].run_phase2.return_value = False
    headless['pipeline']._stop_requested = mode == 'stop'
    with pytest.raises(RuntimeError, match='Phase 2'):
        headless['run'].run_main()
    headless['supervisor'].review_phase2.assert_not_called()
    headless['cpu_factory'].assert_not_called()
    headless['dispatcher_factory'].assert_not_called()
    assert_headless_closed(headless, 'stopped' if mode == 'stop' else 'failed')


@pytest.mark.parametrize('mode', ['failure', 'stop', 'interrupt'])
def test_headless_phase3_outcome_propagates(headless, mode):
    if mode == 'stop':
        headless['dispatcher'].run.return_value = False
        expected_exception = RuntimeError
    elif mode == 'interrupt':
        headless['dispatcher'].run.side_effect = KeyboardInterrupt()
        expected_exception = KeyboardInterrupt
    else:
        headless['dispatcher'].run.side_effect = RuntimeError('synthetic dispatcher failure')
        expected_exception = RuntimeError
    with pytest.raises(expected_exception):
        headless['run'].run_main()
    headless['cpu_factory'].return_value.cleanup_all.assert_called_once()
    if mode == 'interrupt':
        headless['agent'].stop.assert_called_once()
        headless['pipeline'].stop.assert_called_once()
        headless['dispatcher'].stop.assert_called_once()
    assert_headless_closed(headless, 'failed' if mode == 'failure' else 'stopped')


def test_headless_success_and_cleanup_failure_are_distinct(headless):
    headless['provider'].close.side_effect = RuntimeError('synthetic cleanup failure')
    with pytest.raises(RuntimeError, match='cleanup failure'):
        headless['run'].run_main()
    assert_headless_closed(headless, 'failed')


def test_headless_success_records_terminal_event_before_log_close(headless):
    headless['run'].run_main()
    headless['dispatcher'].run.assert_called_once()
    assert_headless_closed(headless, 'completed')
