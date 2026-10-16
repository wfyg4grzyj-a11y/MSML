"""Deterministic races with real SQLite and fake workers/executors, no APIs."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.config import TaskConfig
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.tools import execute_tool
from alpha_lab.worker import Worker


def make_dispatcher(db, tmp_workspace, executor=None, cpu=None, workers=0):
    executor = executor if executor is not None else MagicMock()
    executor.can_submit.return_value = True
    executor.submit_experiment.return_value = "fake-job"
    executor.poll_jobs.return_value = {}
    return Dispatcher(
        MagicMock(), TaskConfig(data_path="synthetic", description="Offline claims"),
        tmp_workspace, db, executor, lambda event: None,
        worker_count=workers, cpu_executor=cpu,
    )


def make_worker(db, tmp_workspace, name="owner"):
    return Worker(name, MagicMock(), TaskConfig(data_path="synthetic", description="Offline"),
                  tmp_workspace, db, lambda event: None)


def create(db, name="synthetic", status="to_implement"):
    identifier = db.create(name, "Synthetic", "Offline", "{}")
    db.update_status(identifier, status)
    return db.get(identifier)


@pytest.mark.parametrize("operation", ["claim", "submit"])
def test_concurrent_database_handles_have_one_winner(db, operation):
    snapshot = create(db, status="checked" if operation == "submit" else "to_implement")
    other = ExperimentDB(db.db_path)
    barrier = threading.Barrier(2)

    def contend(handle, name):
        barrier.wait(5)
        if operation == "submit":
            return handle.reserve_submission(snapshot)
        return handle.claim_worker(snapshot, name, "implement")

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(contend, handle, name) for handle, name in [(db, "a"), (other, "b")]]
        outcomes = [future.result(5) for future in futures]
    assert sum(outcome is not None for outcome in outcomes) == 1


@pytest.mark.parametrize("status", ["cancelled", "queued", "running", "analyzed", "done", "checked"])
def test_assignment_does_not_reopen_ineligible_work(db, status):
    snapshot = create(db)
    db.update_status(snapshot.id, status)
    assert db.claim_worker(snapshot, "owner", "implement") is None
    assert db.assign_worker(snapshot.id, "owner") is False
    assert db.get(snapshot.id).status == status


def test_claim_cannot_replace_an_owner_or_use_an_updated_snapshot(db):
    snapshot = create(db)
    assert db.assign_worker(snapshot.id, "first")
    assert not db.assign_worker(snapshot.id, "second")
    assert db.claim_worker(snapshot, "second", "implement") is None
    assert not db.release_worker(snapshot.id, worker_id="second")
    assert db.release_worker(snapshot.id, worker_id="first")
    db.set_error(snapshot.id, "New information after the snapshot")
    assert db.claim_worker(snapshot, "owner", "implement") is None


def test_stale_cleanup_cannot_release_a_refreshed_claim(db):
    snapshot = create(db)
    assert db.assign_worker(snapshot.id, "same-owner")
    old = db.get(snapshot.id)
    assert db.release_worker(snapshot.id, worker_id="same-owner")
    assert db.assign_worker(snapshot.id, "same-owner")
    assert not db.release_worker(snapshot.id, worker_id="same-owner", updated_at=old.updated_at)
    assert db.get(snapshot.id).worker_id == "same-owner"


def test_fixer_budget_counts_successful_claims_not_losing_snapshots(db):
    snapshot = create(db, status="finished")
    db.set_error(snapshot.id, "Synthetic failed execution")
    snapshot = db.get(snapshot.id)
    for attempt in (1, 2):
        claimed = db.claim_worker(snapshot, "fixer", "fix")
        assert claimed.fix_attempts == attempt
        assert db.claim_worker(snapshot, "loser", "fix") is None
        assert db.get(snapshot.id).fix_attempts == attempt
        db.release_worker(snapshot.id, worker_id="fixer")
        snapshot = db.get(snapshot.id)
    assert db.claim_worker(snapshot, "fixer", "fix") is None
    assert db.get(snapshot.id).fix_attempts == 2


def test_fixer_rejects_successful_execution_without_spending_attempt(db):
    snapshot = create(db, status="finished")
    db.set_error(snapshot.id, "Historical diagnostic")
    db.set_results(snapshot.id, '{"synthetic":true}')
    assert db.claim_worker(db.get(snapshot.id), "fixer", "fix") is None
    assert db.get(snapshot.id).fix_attempts == 0


@pytest.mark.parametrize("task,status", [("implement", "to_implement"), ("analyze", "finished"), ("fix", "finished")])
def test_cancelled_snapshot_never_starts_a_worker(db, tmp_workspace, task, status):
    snapshot = create(db, status=status)
    if task == "fix":
        db.set_error(snapshot.id, "Synthetic error")
        snapshot = db.get(snapshot.id)
    assert db.cancel_if_idle(snapshot.id)
    worker = make_worker(db, tmp_workspace)
    with patch("alpha_lab.worker.threading.Thread") as thread:
        assert getattr(worker, task)(snapshot) is False
        thread.assert_not_called()
    assert worker.current_experiment is None
    worker.provider.assert_not_called()


def test_two_workers_with_one_snapshot_start_only_one_thread(db, tmp_workspace):
    snapshot = create(db)
    first, second = (make_worker(db, tmp_workspace, name) for name in ["first", "second"])
    release = threading.Event()
    first._run_implement = lambda experiment: release.wait(5)
    second._run_implement = MagicMock()
    try:
        assert first.implement(snapshot) is True
        assert second.implement(snapshot) is False
        assert db.get(snapshot.id).worker_id == "first"
        second._run_implement.assert_not_called()
        second.provider.assert_not_called()
    finally:
        release.set()
        first._thread.join(5)


def test_thread_start_failure_releases_only_its_own_claim(db, tmp_workspace):
    snapshot = create(db)
    worker = make_worker(db, tmp_workspace)
    with patch("alpha_lab.worker.threading.Thread.start", side_effect=RuntimeError("synthetic start failure")):
        with pytest.raises(RuntimeError, match="synthetic start failure"):
            worker.implement(snapshot)
    assert not worker.busy and worker.current_experiment is None
    assert db.get(snapshot.id).worker_id is None


def test_stop_prevents_new_tasks_and_agent_activation(db, tmp_workspace):
    snapshot = create(db)
    worker = make_worker(db, tmp_workspace)
    worker.stop()
    with patch("alpha_lab.worker.threading.Thread") as thread:
        assert not worker.implement(snapshot)
        assert not worker.analyze(snapshot)
        assert not worker.fix(snapshot)
        assert not worker.generate_report(1, 10)
        thread.assert_not_called()
    agent = MagicMock()
    assert not worker._activate_agent(agent)
    agent.stop.assert_called_once()
    agent.run.assert_not_called()
    assert db.get(snapshot.id).worker_id is None


def test_prune_after_capacity_check_never_submits(db, tmp_workspace):
    snapshot = create(db, status="checked")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher.executor.can_submit.side_effect = lambda: db.cancel_if_idle(snapshot.id)
    dispatcher._submit_checked()
    dispatcher.executor.submit_experiment.assert_not_called()
    assert db.get(snapshot.id).status == "cancelled"


def test_assigned_checked_row_is_not_submitted(db, tmp_workspace):
    snapshot = create(db)
    db.assign_worker(snapshot.id, "implementation-still-running")
    db.update_status(snapshot.id, "checked")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher._submit_checked()
    dispatcher.executor.submit_experiment.assert_not_called()


def test_submission_is_reserved_before_executor_and_duplicate_dispatch(db, tmp_workspace):
    snapshot = create(db, status="checked")
    first = make_dispatcher(db, tmp_workspace)
    second = make_dispatcher(ExperimentDB(db.db_path), tmp_workspace)

    def submit(experiment, workspace):
        assert db.get(snapshot.id).status == "queued"
        assert not db.cancel_if_idle(snapshot.id)
        second._submit_checked()
        return "fake-job"

    first.executor.submit_experiment.side_effect = submit
    first._submit_checked()
    second.executor.submit_experiment.assert_not_called()
    current = db.get(snapshot.id)
    assert current.slurm_job_id == "fake-job" and current.error is None


def test_submission_error_is_quarantined_across_recovery_without_retry(db, tmp_workspace):
    snapshot = create(db, status="checked")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher.executor.submit_experiment.side_effect = RuntimeError("synthetic ambiguity")
    dispatcher._submit_checked()
    dispatcher.recover()
    dispatcher._submit_checked()
    dispatcher._assign_workers()
    assert dispatcher.executor.submit_experiment.call_count == 1
    current = db.get(snapshot.id)
    assert current.status == "queued" and current.worker_id is None
    assert "reconcile before retry" in current.error


def test_changed_reservation_cancels_returned_job_without_reopening_row(db, tmp_workspace):
    snapshot = create(db, status="checked")
    dispatcher = make_dispatcher(db, tmp_workspace)

    def submit(experiment, workspace):
        # Simulate an administrative state change outside the supported prune API.
        db.update_status(experiment.id, "cancelled")
        return "orphan-returned-job"

    dispatcher.executor.submit_experiment.side_effect = submit
    dispatcher._submit_checked()
    dispatcher.executor.cancel.assert_called_once_with("orphan-returned-job")
    assert db.get(snapshot.id).status == "cancelled"
    assert db.get(snapshot.id).slurm_job_id is None


def test_observer_error_does_not_cancel_an_attached_job(db, tmp_workspace):
    snapshot = create(db, status="checked")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher.event_callback = MagicMock(side_effect=RuntimeError("synthetic observer failure"))
    with pytest.raises(RuntimeError, match="observer failure"):
        dispatcher._submit_checked()
    dispatcher.executor.cancel.assert_not_called()
    current = db.get(snapshot.id)
    assert current.status == "queued" and current.slurm_job_id == "fake-job" and current.error is None


def test_reserved_job_without_identity_is_retained_after_restart(db, tmp_workspace):
    snapshot = create(db, status="checked")
    db.reserve_submission(snapshot)
    restarted = make_dispatcher(ExperimentDB(db.db_path), tmp_workspace, workers=1)
    summary = restarted.recover()
    restarted._submit_checked()
    restarted._assign_workers()
    assert summary["unresolved_submissions"] == 1
    restarted.executor.submit_experiment.assert_not_called()
    assert db.get(snapshot.id).status == "queued"


@pytest.mark.parametrize("terminal", ["RUNNING", "COMPLETED", "UNKNOWN"])
def test_cpu_recovery_uses_persisted_native_job_prefix(db, tmp_workspace, terminal):
    snapshot = create(db, status="queued")
    db.set_slurm_job(snapshot.id, "cpu-persisted-id")
    cpu = MagicMock()
    cpu.poll_jobs.return_value = {"cpu-persisted-id": terminal}
    restarted = make_dispatcher(db, tmp_workspace, cpu=cpu)
    restarted.recover()
    cpu.poll_jobs.assert_called_once_with(["cpu-persisted-id"])
    restarted.executor.poll_jobs.assert_not_called()
    expected = {"RUNNING": "running", "COMPLETED": "finished", "UNKNOWN": "queued"}[terminal]
    assert db.get(snapshot.id).status == expected
    if terminal == "UNKNOWN":
        assert "reconcile" in db.get(snapshot.id).error
        restarted._assign_workers()
        assert db.get(snapshot.id).worker_id is None


def test_shutdown_timeout_does_not_release_a_live_workers_claim(db, tmp_workspace):
    snapshot = create(db)
    dispatcher = make_dispatcher(db, tmp_workspace, workers=1)
    worker = dispatcher.workers[0]
    release = threading.Event()
    worker._run_implement = lambda experiment: release.wait(5)
    try:
        worker.implement(snapshot)
        dispatcher.stop(join_timeout=0)
        assert worker.busy and db.get(snapshot.id).worker_id == worker.worker_id
    finally:
        release.set()
        worker._thread.join(5)


def test_startup_recovery_is_rejected_while_own_worker_is_running(db, tmp_workspace):
    snapshot = create(db)
    dispatcher = make_dispatcher(db, tmp_workspace, workers=1)
    worker = dispatcher.workers[0]
    release = threading.Event()
    worker._run_implement = lambda experiment: release.wait(5)
    try:
        worker.implement(snapshot)
        with pytest.raises(RuntimeError, match="before workers start"):
            dispatcher.recover()
        assert db.get(snapshot.id).worker_id == worker.worker_id
    finally:
        release.set()
        worker._thread.join(5)


def test_strategist_cannot_prune_assigned_work_or_a_stale_snapshot(db, tmp_workspace, monkeypatch):
    snapshot = create(db)
    original_get = db.get

    def stale_get(identifier):
        old = original_get(identifier)
        db.assign_worker(identifier, "concurrent-worker")
        return old

    monkeypatch.setattr(db, "get", stale_get)
    result = execute_tool("cancel_experiments", {"experiment_ids": [snapshot.id], "reason": "Synthetic"},
                          tmp_workspace, db=db)
    assert "Skipped 1" in result["output"]
    assert original_get(snapshot.id).status == "to_implement"
    assert original_get(snapshot.id).worker_id == "concurrent-worker"


def test_stale_fixer_candidate_does_not_consume_an_attempt_or_an_idle_slot(db, tmp_workspace, monkeypatch):
    stale = create(db, name="stale-fixer", status="finished")
    db.set_error(stale.id, "Synthetic failed execution")
    usable = create(db, name="usable-implementation")
    dispatcher = make_dispatcher(db, tmp_workspace, workers=1)
    worker = dispatcher.workers[0]
    release = threading.Event()
    worker._run_implement = lambda experiment: release.wait(5)
    original_list = db.list_by_status

    def stale_list(*statuses):
        snapshots = original_list(*statuses)
        if statuses == ("finished",):
            db.cancel_if_idle(stale.id)
        return snapshots

    monkeypatch.setattr(db, "list_by_status", stale_list)
    try:
        dispatcher._assign_workers()
        assert db.get(stale.id).fix_attempts == 0
        assert db.get(usable.id).worker_id == worker.worker_id
    finally:
        release.set()
        if worker._thread: worker._thread.join(5)


@pytest.mark.parametrize("known_state", ["PENDING", "RUNNING", "COMPLETED"])
def test_unknown_job_diagnostic_is_cleared_when_state_becomes_known(db, tmp_workspace, known_state):
    snapshot = create(db, status="queued")
    db.set_slurm_job(snapshot.id, "persisted-job")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher.executor.poll_jobs.return_value = {"persisted-job": "UNKNOWN"}
    dispatcher._poll_slurm()
    assert "UNKNOWN" in db.get(snapshot.id).error
    dispatcher.executor.poll_jobs.return_value = {"persisted-job": known_state}
    dispatcher._poll_slurm()
    assert db.get(snapshot.id).error is None


def test_known_job_state_does_not_erase_a_different_failure(db, tmp_workspace):
    snapshot = create(db, status="queued")
    db.set_slurm_job(snapshot.id, "persisted-job")
    db.set_error(snapshot.id, "Separate synthetic diagnostic")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher.executor.poll_jobs.return_value = {"persisted-job": "COMPLETED"}
    dispatcher._poll_slurm()
    assert db.get(snapshot.id).error == "Separate synthetic diagnostic"


def test_missing_executor_or_omitted_job_status_stays_unresolved(db, tmp_workspace):
    snapshot = create(db, status="queued")
    db.set_slurm_job(snapshot.id, "cpu-persisted")
    dispatcher = make_dispatcher(db, tmp_workspace)
    dispatcher._poll_slurm()
    assert db.get(snapshot.id).status == "queued"
    assert "UNKNOWN" in db.get(snapshot.id).error


@pytest.mark.parametrize("task,status", [("implement", "to_implement"), ("analyze", "finished"), ("fix", "finished")])
def test_stop_during_agent_construction_prevents_run_and_automatic_promotion(db, tmp_workspace, task, status):
    snapshot = create(db, status=status)
    worker = make_worker(db, tmp_workspace)
    agent = MagicMock()

    def prepare_agent(**kwargs):
        worker.stop()
        return agent

    with patch("alpha_lab.worker.AgentLoop", side_effect=prepare_agent):
        getattr(worker, f"_run_{task}")(snapshot)
    agent.run.assert_not_called()
    agent.stop.assert_called_once()
    assert db.get(snapshot.id).status == status


def test_reporter_preserves_configured_reasoning_effort(db, tmp_workspace):
    worker = make_worker(db, tmp_workspace)
    worker.config.reasoning_effort = "high"
    with patch("alpha_lab.worker.AgentLoop") as agent_factory:
        worker._run_report(1, 10)
    assert agent_factory.call_args.kwargs["reasoning_effort"] == "high"
    agent_factory.return_value.run.assert_called_once()
    worker.provider.assert_not_called()


def test_stop_during_capacity_check_prevents_reservation_and_launch(db, tmp_workspace):
    snapshot = create(db, status="checked")
    dispatcher = make_dispatcher(db, tmp_workspace)

    def request_stop():
        dispatcher._stop_requested = True
        return True

    dispatcher.executor.can_submit.side_effect = request_stop
    dispatcher._submit_checked()
    dispatcher.executor.submit_experiment.assert_not_called()
    assert db.get(snapshot.id).status == "checked"


def test_old_database_migrates_fixer_counter_before_atomic_claim(tmp_path):
    import sqlite3
    from alpha_lab.experiment_db import _CREATE_TABLE
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(_CREATE_TABLE.replace(",\n    fix_attempts INTEGER NOT NULL DEFAULT 0", ""))
    connection.close()
    db = ExperimentDB(str(path))
    snapshot = create(db, status="finished")
    db.set_error(snapshot.id, "Synthetic failure")
    assert db.claim_worker(db.get(snapshot.id), "fixer", "fix").fix_attempts == 1
