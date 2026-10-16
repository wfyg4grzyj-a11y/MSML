"""Dispatcher — the main orchestration loop for Phase 3.

Pure Python (no LLM). Manages strategist turns, worker assignments,
job submission/polling, and kanban state transitions.

Supports executors:
- SlurmManager: submits jobs via sbatch
- LocalGPUManager: spawns subprocesses directly on local GPUs
- LocalCPUManager: runs tree-based models on CPU in parallel with GPU jobs
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from alpha_lab.config import Phase3Config, TaskConfig
from alpha_lab.events import (
    AgentEvent,
    BoardSummaryEvent,
    ExperimentEvent,
    PhaseEvent,
)
from alpha_lab.experiment_db import Experiment, ExperimentDB
from alpha_lab.llm_runtime import LLMRunError, LLMRuntime
from alpha_lab.provider import Provider
from alpha_lab.strategist import Strategist
from alpha_lab.worker import Worker
from alpha_lab.workspace_lock import WorkspaceLock


class JobExecutor(Protocol):
    """Protocol for job executors (SLURM or local)."""

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Submit experiment, return job ID."""
        ...

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Poll job statuses. Returns {job_id: status}."""
        ...

    def cancel(self, job_id: str) -> None:
        """Cancel a job."""
        ...

    def can_submit(self) -> bool:
        """Check if capacity available."""
        ...

    def running_gpu_count(self) -> int:
        """Count running jobs."""
        ...


class CPUExecutor(Protocol):
    """Protocol for CPU executor (optional)."""

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        ...

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        ...

    def cancel(self, job_id: str) -> None:
        ...

    def can_submit(self) -> bool:
        ...

    def running_count(self) -> int:
        ...

logger = logging.getLogger("alpha_lab.dispatcher")

POLL_INTERVAL = 10  # seconds
_UNKNOWN_JOB_ERROR = "Job state UNKNOWN; reconcile its process/job identity before retry."


class Dispatcher:
    """Main orchestration loop for Phase 3 experiment system."""

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        db: ExperimentDB,
        executor: JobExecutor,
        event_callback: Callable[[AgentEvent], None],
        worker_count: int = 4,
        metrics: object | None = None,
        cpu_executor: CPUExecutor | None = None,
        adapter: object | None = None,
        supervisor: object | None = None,
        workspace_lock: WorkspaceLock | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.db = db
        self.executor = executor  # GPU executor
        self.cpu_executor = cpu_executor  # Optional CPU executor
        self.event_callback = event_callback
        self.metrics = metrics
        self.adapter = adapter
        self.supervisor = supervisor
        self._stop_requested = False
        self._workspace_lock = workspace_lock
        self._run_lock = threading.Lock()
        # Track which jobs are CPU vs GPU for polling
        self._cpu_job_ids: set[str] = set()

        p3 = config.pipeline.phase3

        # Create workers
        self.workers = [
            Worker(
                worker_id=f"worker_{i}",
                provider=provider,
                config=config,
                workspace=workspace,
                db=db,
                event_callback=event_callback,
                metrics=metrics,
                adapter=adapter,
            )
            for i in range(worker_count)
        ]

        # Create strategist
        self.strategist = Strategist(
            provider=provider,
            config=config,
            workspace=workspace,
            db=db,
            event_callback=event_callback,
            adapter=adapter,
        )

        # Strategist scheduling (protected by _state_lock)
        self._state_lock = threading.Lock()
        self._strategist_interval = p3.strategist_interval
        self._last_strategist_time = 0.0
        self._analyzed_since_strategist = 0
        self._last_analyzed_count = 0
        self._strategist_running = False
        self._strategist_thread: threading.Thread | None = None

        # Report scheduling
        self._report_interval = p3.report_interval
        self._last_report_at_done_count = 0
        self._report_number = 0
        self._current_report_number = 0
        self._report_in_progress = False
        self._report_worker: Worker | None = None

        # Limits
        self._max_experiments = p3.max_experiments

        # Convergence tracking
        self._convergence_threshold = p3.convergence_threshold
        # Resolve metric: config override > adapter primary > fallback "sharpe"
        if p3.convergence_metric:
            self._convergence_metric = p3.convergence_metric
        elif adapter is not None:
            self._convergence_metric = adapter.metric.primary_metric
        else:
            self._convergence_metric = "sharpe"
        # Determine metric direction for convergence comparison
        self._metric_direction = "maximize"
        if adapter is not None:
            self._metric_direction = adapter.metric.direction
        if self._metric_direction == "minimize":
            self._best_metric_value: float = float("inf")
        else:
            self._best_metric_value: float = float("-inf")
        self._experiments_since_improvement: int = 0

        # Dispatcher JSONL log
        self._log_file = None

    def _init_log(self) -> None:
        """Open the dispatcher JSONL log."""
        log_dir = Path(self.workspace) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_dir / "dispatcher.jsonl", "a")

    def _log(self, action: str, **details: object) -> None:
        """Write a structured log entry for dispatcher decisions."""
        if self._log_file is not None:
            entry = {"t": time.time(), "action": action, **details}
            try:
                self._log_file.write(json.dumps(entry, default=str) + "\n")
                self._log_file.flush()
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to write dispatcher log entry: %s", e)

    def _bounded_runtime(self) -> LLMRuntime | None:
        runtime = getattr(self.provider, "runtime", None)
        return runtime if isinstance(runtime, LLMRuntime) else None

    def _can_start_runtime_work(self) -> bool:
        runtime = self._bounded_runtime()
        if runtime is not None:
            runtime.check_deadline()
            return runtime.stop_reason() is None
        return True

    def _runtime_work_active(self, runtime: LLMRuntime) -> bool:
        with self._state_lock:
            strategist_running = self._strategist_running
        return strategist_running or any(w.busy for w in self.workers) or runtime.request_active

    def _wait_for_next_poll(self) -> None:
        for _ in range(POLL_INTERVAL):
            if self._stop_requested:
                break
            runtime = self._bounded_runtime()
            if runtime is not None:
                runtime.check_deadline()
            time.sleep(1)

    def emit(self, event: AgentEvent) -> None:
        self.event_callback(event)
        # Also log to dispatcher JSONL
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps(event.to_dict(), default=str) + "\n")
                self._log_file.flush()
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to write event to dispatcher log: %s", e)

    def stop(self, join_timeout: float = 30) -> None:
        """Stop the dispatcher and all workers, waiting for threads to finish."""
        self._stop_requested = True
        self.strategist.stop()
        for w in self.workers:
            w.stop()

        # Join worker threads
        for w in self.workers:
            if w._thread is not None and w._thread.is_alive():
                w._thread.join(timeout=join_timeout)

        # Join strategist thread
        if self._strategist_thread is not None and self._strategist_thread.is_alive():
            self._strategist_thread.join(timeout=join_timeout)

        # Kill all running executor jobs to prevent orphans
        if self.executor is not None:
            try:
                self.executor.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup GPU executor: %s", e)
        if self.cpu_executor is not None:
            try:
                self.cpu_executor.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup CPU executor: %s", e)

        self._cleanup()
        if self._workspace_lock is not None:
            threads = [w._thread for w in self.workers if w._thread is not None]
            if self._strategist_thread is not None:
                threads.append(self._strategist_thread)
            self._workspace_lock.retain_until(threads)

    def _cleanup(self) -> None:
        """Release this dispatcher's idle claims, retaining claims of live threads."""
        idle_ids = {w.worker_id for w in self.workers if not w.busy}
        try:
            for exp in self.db.list_all():
                if exp.worker_id in idle_ids:
                    self.db.release_worker(exp.id, worker_id=exp.worker_id, updated_at=exp.updated_at)
        except Exception as e:
            logger.warning("Failed to release idle worker claims during cleanup: %s", e)

        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError as e:
                logger.warning("Failed to close dispatcher log file: %s", e)
            self._log_file = None

    def recover(self) -> dict:
        """Exclude other workspace owners before releasing any orphaned claim."""
        if self._workspace_lock is not None:
            if not self._workspace_lock.held or self._workspace_lock.workspace != Path(self.workspace).resolve():
                raise RuntimeError("Dispatcher requires its workspace's acquired ownership lock")
            return self._recover_owned()
        with WorkspaceLock(self.workspace):
            return self._recover_owned()

    def _recover_owned(self) -> dict:
        """Recover from a crash: release orphaned workers and reconcile SLURM jobs.

        Returns a summary dict of recovery actions taken.
        """
        if any(worker.busy for worker in self.workers):
            raise RuntimeError("Recovery is only valid before workers start")
        summary: dict = {"released_workers": 0, "slurm_reconciled": 0, "unresolved_submissions": 0}

        # 1. Release orphaned worker assignments (no workers exist yet at startup)
        for exp in self.db.list_all():
            if exp.worker_id is not None:
                if self.db.release_worker(exp.id, worker_id=exp.worker_id, updated_at=exp.updated_at):
                    summary["released_workers"] += 1

        # 2. Reconcile SLURM jobs
        slurm_exps = self.db.list_by_status("queued", "running")
        for exp in slurm_exps:
            if not exp.slurm_job_id:
                summary["unresolved_submissions"] += 1
                if exp.status == "queued" and not exp.error:
                    self.db.flag_submission(exp, "Submission identity unavailable; reconcile before retry.")
                self._log("unresolved_submission", experiment_id=exp.id)
            elif exp.slurm_job_id.startswith("cpu-"):
                self._cpu_job_ids.add(exp.slurm_job_id)
        # Native GPU executors can reattach only identity-verified local processes.
        recover_jobs = getattr(self.executor, "recover_running_jobs", None)
        if callable(recover_jobs):
            recover_jobs(self.workspace, {
                exp.slurm_job_id: exp.name for exp in slurm_exps
                if exp.slurm_job_id and exp.slurm_job_id not in self._cpu_job_ids
            })
        recover_cpu = getattr(self.cpu_executor, "recover_running_jobs", None)
        if callable(recover_cpu):
            recover_cpu(self.workspace, {
                exp.slurm_job_id: exp.name for exp in slurm_exps
                if exp.slurm_job_id and exp.slurm_job_id in self._cpu_job_ids
            })
        self._poll_slurm()
        summary["slurm_reconciled"] = sum(
            self.db.get(exp.id).status != exp.status for exp in slurm_exps
        )

        self._log("recovery", **summary)
        logger.info(f"Recovery complete: {summary}")
        return summary

    def run(self) -> bool:
        """Hold workspace ownership through recovery, execution and shutdown."""
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Dispatcher is already running")
        owned = self._workspace_lock is None
        try:
            if owned:
                self._workspace_lock = WorkspaceLock(self.workspace).acquire()
            if not self._workspace_lock.held or self._workspace_lock.workspace != Path(self.workspace).resolve():
                raise RuntimeError("Dispatcher workspace ownership is unavailable")
            try:
                return self._run_owned()
            finally:
                self.stop()
        finally:
            if owned and self._workspace_lock is not None:
                self._workspace_lock.release()
            self._run_lock.release()

    def _run_owned(self) -> bool:
        """Main loop — blocks until stopped or max_experiments reached."""
        logger.info("Dispatcher starting")
        self._init_log()
        self._log("dispatcher_start", max_experiments=self._max_experiments,
                  worker_count=len(self.workers), report_interval=self._report_interval)
        self.emit(PhaseEvent(
            phase="phase3",
            step="dispatcher",
            status="starting",
            detail="Phase 3 experiment loop starting",
        ))

        # Ensure experiments directory exists
        Path(self.workspace, "experiments").mkdir(parents=True, exist_ok=True)

        terminal_status = "failed"
        terminal_detail = "Phase 3 experiment loop failed"
        try:
            # Recovery must succeed before any new worker/API/submission work.
            try:
                self.recover()
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                self._log("recovery_error", error=str(e))
                raise

            while not self._stop_requested:
                runtime = self._bounded_runtime()
                reason = runtime.stop_reason() if runtime is not None else None
                if reason is not None:
                    if reason != "deadline" and self._runtime_work_active(runtime):
                        # Let the admitted response finish its local tools, but
                        # admit no further agents, reports or experiment jobs.
                        self._poll_slurm()
                        self._emit_board_summary()
                        self._wait_for_next_poll()
                        continue
                    if reason != "deadline" and self._should_terminate():
                        break  # Last allowed response already completed the board.
                    runtime.check_active()

                # 1. Strategist turn (if due)
                if self._should_run_strategist():
                    self._run_strategist()

                # 2. Poll SLURM jobs
                self._poll_slurm()

                # 3. Submit checked experiments to SLURM
                self._submit_checked()

                # 4. Milestone report (if due) — checked BEFORE assigning
                #    workers so a report can claim an idle slot.
                self._maybe_generate_report()

                # 5. Assign idle workers
                self._assign_workers()

                # 6. Track newly analyzed experiments (for strategist trigger)
                self._track_analyzed()

                # 7. Detect stale workers
                self._check_stale()

                # 7b. Detect stuck workers (observability)
                self._check_stuck_workers()

                # 7c. Supervisor health check (if error rate high)
                self._maybe_supervisor_check()

                # 8. Emit board summary
                self._emit_board_summary()

                # 9. Check termination
                if self._should_terminate():
                    logger.info("Max experiments reached, stopping")
                    break

                # Sleep (interruptible)
                self._wait_for_next_poll()

            terminal_status = "stopped" if self._stop_requested else "completed"
            terminal_detail = ("Phase 3 experiment loop stopped" if self._stop_requested
                               else "Phase 3 experiment loop finished")
        except KeyboardInterrupt:
            terminal_status = "stopped"
            terminal_detail = "Phase 3 experiment loop interrupted"
            raise
        except Exception as e:
            logger.error(f"Dispatcher error: {e}")
            self._log("dispatcher_error", error=str(e), traceback=traceback.format_exc())
            raise
        finally:
            self._log("dispatcher_stop", status=terminal_status)
            try:
                self.emit(PhaseEvent(
                    phase="phase3",
                    step="dispatcher",
                    status=terminal_status,
                    detail=terminal_detail,
                ))
            finally:
                if self._log_file is not None:
                    try:
                        self._log_file.close()
                    except OSError as e:
                        logger.warning("Failed to close dispatcher log on shutdown: %s", e)
            logger.info("Dispatcher stopped")
        return terminal_status == "completed"

    def _should_run_strategist(self) -> bool:
        """Determine if it's time for a strategist turn."""
        if self._stop_requested:
            return False
        with self._state_lock:
            if self._strategist_running:
                return False

            now = time.time()
            elapsed = now - self._last_strategist_time

            # First turn: immediately
            if self._last_strategist_time == 0:
                return True

            # After N experiments analyzed
            if self._analyzed_since_strategist >= 3:
                return True

        # DB/worker checks don't need the lock
        pending = self.db.list_by_status("to_implement")
        idle_workers = [w for w in self.workers if not w.busy]

        with self._state_lock:
            elapsed = time.time() - self._last_strategist_time
            if not pending and idle_workers and elapsed > 60:
                return True
            if elapsed >= self._strategist_interval:
                return True

        return False

    def _run_strategist(self) -> None:
        """Run a strategist turn in a background thread (non-blocking).

        In no_strategist ablation mode, proposes experiments via a simple
        one-shot LLM call with no feedback about what's working — testing
        the value of strategic planning vs naive proposals.
        """
        if not self._can_start_runtime_work():
            return
        with self._state_lock:
            self._strategist_running = True

        if self.config.pipeline.phase3.no_strategist:
            self._log("random_proposer_start")
            self.emit(ExperimentEvent(
                name="random_proposer",
                status="running",
                detail="Random proposer generating experiments (no strategist)",
            ))

            def _random_proposer_thread() -> None:
                try:
                    self._propose_random_experiments()
                    with self._state_lock:
                        self._last_strategist_time = time.time()
                        self._analyzed_since_strategist = 0
                    self._log("random_proposer_done")
                except Exception as e:
                    logger.error(f"Random proposer error: {e}")
                    self._log("random_proposer_error", error=str(e),
                              traceback=traceback.format_exc())
                finally:
                    with self._state_lock:
                        self._strategist_running = False

            t = threading.Thread(target=_random_proposer_thread, daemon=True)
            self._strategist_thread = t
            t.start()
            return

        self._log("strategist_start")
        self.emit(ExperimentEvent(
            name="strategist",
            status="running",
            detail="Strategist proposing experiments",
        ))

        def _strategist_thread() -> None:
            try:
                self.strategist.run_turn()
                with self._state_lock:
                    self._last_strategist_time = time.time()
                    self._analyzed_since_strategist = 0
                self._log("strategist_done")
            except Exception as e:
                logger.error(f"Strategist error: {e}")
                self._log("strategist_error", error=str(e), traceback=traceback.format_exc())
            finally:
                with self._state_lock:
                    self._strategist_running = False

        t = threading.Thread(target=_strategist_thread, daemon=True)
        self._strategist_thread = t
        t.start()

    def _propose_random_experiments(self) -> None:
        """Propose experiments without strategic context (ablation mode).

        Uses a one-shot LLM call with only framework description and learnings
        from Phase 1 — no board state, no leaderboard, no playbook. This tests
        the value of the strategist's iterative learning and planning.
        """
        # Check budget — cancelled experiments do not consume budget slots
        summary = self.db.board_summary()
        total_proposed = sum(v for k, v in summary.items() if k != "cancelled")
        remaining = self._max_experiments - total_proposed
        if remaining <= 0:
            logger.info("Random proposer: budget exhausted")
            return

        # Propose in batches of 5 (or remaining, whichever is less)
        batch_size = min(5, remaining)

        # Read Phase 1 learnings for minimal domain context
        learnings = ""
        learnings_path = Path(self.workspace) / "learnings.md"
        if learnings_path.exists():
            learnings = learnings_path.read_text()[:3000]

        # Read framework code for context on what configs are valid
        framework_code = ""
        adapter = self.adapter
        if adapter is not None:
            fw_name = adapter.experiment.framework_dir or "backtest"
            fw_dir = Path(self.workspace) / fw_name
            if fw_dir.is_dir():
                snippets = []
                for f in sorted(fw_dir.rglob("*.py"))[:10]:
                    try:
                        snippets.append(f"### {f.name}\n```python\n{f.read_text()[:2000]}\n```")
                    except OSError:
                        pass
                framework_code = "\n\n".join(snippets)

        metric_name = "metric"
        direction = "maximize"
        domain = "unknown"
        if adapter is not None:
            metric_name = adapter.metric.primary_metric
            direction = adapter.metric.direction
            domain = adapter.domain_name

        prompt = f"""You are proposing experiment configs for domain '{domain}'.
Goal: {direction} '{metric_name}'.

## Learnings
{learnings if learnings else "No prior exploration."}

## Framework Code (for valid config fields)
{framework_code[:8000] if framework_code else "No framework code available."}

## Instructions
Propose {batch_size} DIVERSE experiment configs. Vary architectures, hyperparameters,
and strategies broadly. Each experiment should be meaningfully different.

Return a JSON array of objects, each with:
- "name": short unique name (alphanumeric + underscores)
- "description": one-line description
- "hypothesis": what you expect
- "config": the experiment config JSON object

Return ONLY the JSON array. No explanation."""

        import re as _re
        response = self.provider.complete(
            model=self.config.model,
            system="Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
        )
        runtime = self._bounded_runtime()
        if runtime is not None:
            runtime.check_deadline()

        # Parse response
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            proposals = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Random proposer: failed to parse LLM response")
            return

        if not isinstance(proposals, list):
            proposals = [proposals]

        for p in proposals[:batch_size]:
            if runtime is not None:
                runtime.check_deadline()
            name = _re.sub(r"[^a-zA-Z0-9_\-]", "_", str(p.get("name", "random")))[:80]
            desc = str(p.get("description", ""))
            hyp = str(p.get("hypothesis", ""))
            config = p.get("config", {})
            config_str = json.dumps(config) if isinstance(config, dict) else str(config)
            try:
                exp_id = self.db.create(name, desc, hyp, config_str)
                logger.info(f"Random proposer: created experiment #{exp_id} '{name}'")
            except Exception as e:
                logger.warning(f"Random proposer: failed to create experiment: {e}")

    def _poll_slurm(self) -> None:
        """Poll GPU and CPU executors for job status updates."""
        # Get all queued/running experiments with job IDs
        active = self.db.list_by_status("queued", "running")
        job_ids = [exp.slurm_job_id for exp in active if exp.slurm_job_id]
        if not job_ids:
            return

        # Split into CPU and GPU jobs
        cpu_job_ids = [jid for jid in job_ids if jid in self._cpu_job_ids or jid.startswith("cpu-")]
        gpu_job_ids = [jid for jid in job_ids if jid not in cpu_job_ids]

        # Poll both executors
        statuses: dict[str, str] = {}
        if gpu_job_ids and self.executor is not None:
            statuses.update(self.executor.poll_jobs(gpu_job_ids))
        if cpu_job_ids and self.cpu_executor:
            statuses.update(self.cpu_executor.poll_jobs(cpu_job_ids))
        for job_id in job_ids:
            statuses.setdefault(job_id, "UNKNOWN")

        for exp in active:
            if not exp.slurm_job_id or exp.slurm_job_id not in statuses:
                continue

            slurm_status = statuses[exp.slurm_job_id]
            if slurm_status in ("PENDING", "RUNNING", "COMPLETED") and exp.error == _UNKNOWN_JOB_ERROR:
                self.db.clear_matching_error(exp.id, _UNKNOWN_JOB_ERROR)

            if exp.status == "queued" and slurm_status == "RUNNING":
                self.db.update_status(
                    exp.id, "running",
                    started_at=time.time(),
                )
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="running",
                    prev_status="queued",
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"SLURM job {exp.slurm_job_id} running",
                ))
                logger.info(f"Experiment #{exp.id} {exp.name}: queued -> running")

            elif exp.status in ("queued", "running") and slurm_status == "COMPLETED":
                self.db.update_status(
                    exp.id, "finished",
                    finished_at=time.time(),
                )
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="finished",
                    prev_status=exp.status,
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"SLURM job {exp.slurm_job_id} completed",
                ))
                logger.info(f"Experiment #{exp.id} {exp.name}: {exp.status} -> finished")

            elif exp.status in ("queued", "running") and slurm_status in (
                "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"
            ):
                self.db.set_error_and_finish(exp.id, f"SLURM {slurm_status}")
                self.emit(ExperimentEvent(
                    experiment_id=exp.id,
                    name=exp.name,
                    status="finished",
                    prev_status=exp.status,
                    slurm_job_id=exp.slurm_job_id,
                    detail=f"SLURM job {exp.slurm_job_id} {slurm_status}",
                ))
                logger.warning(
                    f"Experiment #{exp.id} {exp.name}: SLURM {slurm_status}"
                )
            elif slurm_status == "UNKNOWN":
                message = _UNKNOWN_JOB_ERROR
                if exp.error != message:
                    self.db.set_error(exp.id, message)
                    self._log("unresolved_job", experiment_id=exp.id, job_id=exp.slurm_job_id)

    def _is_cpu_experiment(self, exp: Experiment) -> bool:
        """Check if an experiment should run on CPU based on its config."""
        try:
            config = json.loads(exp.config_json or "{}")
            model_type = config.get("model_type", "").lower()
            resource = config.get("resource", "").lower()

            # Explicit resource tag takes precedence
            if resource == "cpu":
                return True
            if resource == "gpu":
                return False

            # Infer from model type
            cpu_keywords = [
                "xgboost", "lightgbm", "gbdt", "gradient_boost",
                "random_forest", "decision_tree", "tree",
                "linear", "lasso", "ridge", "elastic",
                "catboost", "sklearn",
            ]
            return any(kw in model_type for kw in cpu_keywords)
        except (json.JSONDecodeError, TypeError):
            return False

    def _submit_checked(self) -> None:
        """Submit checked experiments to GPU or CPU executor."""
        checked = self.db.list_by_status("checked")
        for exp in checked:
            if self._stop_requested:
                break
            if not self._can_start_runtime_work():
                break

            # Determine which executor to use
            # CPU-only mode: route everything to CPU when no GPU executor
            use_cpu = (
                self.executor is None
                or (self.cpu_executor is not None and self._is_cpu_experiment(exp))
            )

            if use_cpu and self.cpu_executor is not None:
                if not self.cpu_executor.can_submit():
                    logger.debug(f"CPU slots full, skipping {exp.name}")
                    continue
                executor = self.cpu_executor
                executor_name = "CPU"
            elif self.executor is not None:
                if not self.executor.can_submit():
                    logger.debug("GPU budget exhausted, waiting")
                    continue
                executor = self.executor
                executor_name = "GPU"
            else:
                logger.warning(f"No executor available for {exp.name}, skipping")
                continue

            if self._stop_requested:
                break
            if not self._can_start_runtime_work():
                break
            reservation = self.db.reserve_submission(exp)
            if reservation is None:
                continue  # Snapshot was pruned, changed, or claimed after listing.
            job_id = None
            try:
                if not self._can_start_runtime_work():
                    self.db.flag_submission(reservation, "Run limit reached before submission; reconcile before retry.")
                    break
                job_id = executor.submit_experiment(reservation, self.workspace)
                self._log("submission_returned", experiment_id=exp.id, job_id=job_id)
                if not self.db.attach_submission(reservation, job_id):
                    raise RuntimeError("Submission reservation changed before job attachment")
                if use_cpu:
                    self._cpu_job_ids.add(job_id)
            except LLMRunError:
                self.db.flag_submission(reservation, "Run limit interrupted submission; reconcile before retry.")
                raise
            except Exception as e:
                logger.error("Submit failed for experiment #%s (%s)", exp.id, type(e).__name__)
                if job_id:
                    try:
                        executor.cancel(job_id)
                        self._log("submission_cancel_requested", experiment_id=exp.id, job_id=job_id)
                    except Exception as cancel_error:
                        self._log("submission_cancel_failed", experiment_id=exp.id, job_id=job_id,
                                  error_type=type(cancel_error).__name__)
                self.db.flag_submission(
                    reservation, f"Submit failed ({type(e).__name__}); reconcile before retry."
                )
                continue
            # Observer failures must not turn an already attached job into a failed submission.
            self.emit(ExperimentEvent(
                experiment_id=exp.id, name=exp.name, status="queued", prev_status="checked",
                slurm_job_id=job_id, detail=f"Submitted {executor_name} job {job_id}",
            ))
            logger.info(f"Experiment #{exp.id} {exp.name}: submitted as {executor_name} job {job_id}")

    def _assign_workers(self) -> None:
        """Assign idle workers to pending tasks. Priority: fix > analyze > implement."""
        # First: release experiments assigned to workers that are no longer busy
        self._release_dead_assignments()

        if self._stop_requested:
            return
        if not self._can_start_runtime_work():
            return

        idle = [w for w in self.workers if not w.busy]
        if not idle:
            return

        # Priority 1: fix failed experiments (finished with error, no results, < max attempts)
        MAX_FIX_ATTEMPTS = 2
        finished = self.db.list_by_status("finished")
        fixable = [
            exp for exp in finished
            if exp.worker_id is None
            and exp.error
            and not exp.results_json
            and exp.fix_attempts < MAX_FIX_ATTEMPTS
        ]

        for exp in fixable:
            if not idle or self._stop_requested:
                break
            if not self._can_start_runtime_work():
                break
            worker = idle[0]
            if worker.fix(exp) is False:
                continue
            idle.pop(0)
            attempts = self.db.get(exp.id).fix_attempts
            logger.info(
                f"Assigning {worker.worker_id} to fix #{exp.id} {exp.name} "
                f"(attempt {attempts}/{MAX_FIX_ATTEMPTS})"
            )
            self._log("assign_worker", worker=worker.worker_id, task="fix",
                       experiment_id=exp.id, experiment_name=exp.name,
                       fix_attempt=attempts)

        # Priority 2: analyze finished experiments (with results or max fix attempts reached)
        unassigned_finished = [
            exp for exp in finished
            if exp.worker_id is None and exp not in fixable
        ]

        for exp in unassigned_finished:
            if not idle or self._stop_requested:
                break
            if not self._can_start_runtime_work():
                break
            worker = idle[0]
            if worker.analyze(exp) is False:
                continue
            idle.pop(0)
            logger.info(
                f"Assigning {worker.worker_id} to analyze #{exp.id} {exp.name}"
            )
            self._log("assign_worker", worker=worker.worker_id, task="analyze",
                       experiment_id=exp.id, experiment_name=exp.name)

        # Priority 3: implement new experiments (include "implemented" stuck experiments)
        to_implement = self.db.list_by_status("to_implement", "implemented")
        unassigned_impl = [
            exp for exp in to_implement if exp.worker_id is None
        ]

        for exp in unassigned_impl:
            if not idle or self._stop_requested:
                break
            if not self._can_start_runtime_work():
                break
            worker = idle[0]
            if worker.implement(exp) is False:
                continue
            idle.pop(0)
            logger.info(
                f"Assigning {worker.worker_id} to implement #{exp.id} {exp.name}"
            )
            self._log("assign_worker", worker=worker.worker_id, task="implement",
                       experiment_id=exp.id, experiment_name=exp.name)

    def _release_dead_assignments(self) -> None:
        """Release experiments assigned to workers that are no longer busy.

        This catches the case where a worker thread died or finished but
        the experiment still has worker_id set (e.g. because the LLM agent
        didn't call update_experiment).

        To avoid racing with thread startup, only release if the assignment
        is at least 10 seconds old.
        """
        # Build set of currently-busy worker IDs
        busy_worker_ids = {w.worker_id for w in self.workers if w.busy}
        now = time.time()

        # Check experiments in states where workers should be active
        for status in ("to_implement", "implemented", "finished"):
            assigned = self.db.list_by_status(status)
            for exp in assigned:
                if exp.worker_id and exp.worker_id not in busy_worker_ids:
                    # Grace period: don't release if assignment is very recent
                    # (thread may still be starting up)
                    if now - exp.updated_at < 10:
                        continue
                    logger.warning(
                        f"Experiment #{exp.id} {exp.name} assigned to "
                        f"{exp.worker_id} but worker is idle — releasing"
                    )
                    self._log("release_dead_assignment",
                              experiment_id=exp.id, worker=exp.worker_id,
                              status=exp.status)
                    self.db.release_worker(exp.id, worker_id=exp.worker_id, updated_at=exp.updated_at)

    def _check_stale(self) -> None:
        """Detect stale workers (>5min) and release their assignments.

        Only releases if the assigned worker thread is no longer alive,
        to avoid prematurely releasing experiments from workers that are
        just running long shell commands.
        """
        busy_worker_ids = {w.worker_id for w in self.workers if w.busy}
        stale = self.db.stale_workers(timeout_s=300)
        for exp in stale:
            if exp.worker_id in busy_worker_ids:
                continue  # Worker thread still alive, don't release
            logger.warning(
                f"Stale worker on experiment #{exp.id} {exp.name} — releasing"
            )
            self.db.release_worker(exp.id, worker_id=exp.worker_id, updated_at=exp.updated_at)

    def _check_stuck_workers(self) -> None:
        """Detect workers with no events for >10 minutes (observability only)."""
        now = time.time()
        for w in self.workers:
            if w.busy and w.last_event_at > 0 and now - w.last_event_at > 600:
                logger.warning(
                    f"Worker {w.worker_id} stuck: no events for "
                    f"{int(now - w.last_event_at)}s"
                )
                self._log(
                    "stuck_worker",
                    worker=w.worker_id,
                    seconds_since_event=int(now - w.last_event_at),
                )

    def _maybe_supervisor_check(self) -> None:
        """Run supervisor health check if error rate exceeds 40%.

        Checked every 10 analyzed experiments. If the supervisor patches
        the adapter, reload it.
        """
        if self.supervisor is None:
            return

        analyzed = self.db.list_by_status("analyzed", "done")
        if len(analyzed) < 10 or len(analyzed) % 10 != 0:
            return

        # Calculate error rate
        all_finished = self.db.list_by_status("analyzed", "done")
        errors = [e for e in all_finished if e.error]
        error_rate = len(errors) / max(len(all_finished), 1)

        if error_rate <= 0.4:
            return
        if not self._can_start_runtime_work():
            return

        logger.warning(
            f"Error rate {error_rate:.0%} exceeds 40% — running supervisor health check"
        )
        self._log("supervisor_health_check", error_rate=error_rate)

        try:
            self.supervisor.phase3_health_check()
            # Reload adapter in case supervisor patched it
            from alpha_lab.adapter_loader import resolve_adapter
            new_adapter = resolve_adapter(self.workspace)
            self.adapter = new_adapter
            # Update workers and strategist
            for w in self.workers:
                w.adapter = new_adapter
            self.strategist.adapter = new_adapter
        except Exception as e:
            logger.error(f"Supervisor health check failed: {e}")
            self._log("supervisor_error", error=str(e))

    def _emit_board_summary(self) -> None:
        """Emit a board summary event."""
        summary = self.db.board_summary()
        recent = self.db.list_all()[-10:]
        _metric = self._convergence_metric
        leaders = self.db.leaderboard(_metric, 5)

        experiments = []
        for exp in recent:
            experiments.append({
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "worker_id": exp.worker_id,
                "slurm_job_id": exp.slurm_job_id,
            })

        leaderboard = []
        for exp in leaders:
            metrics = {}
            if exp.results_json:
                try:
                    metrics = json.loads(exp.results_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            leaderboard.append({
                "id": exp.id,
                "name": exp.name,
                "metrics": metrics,
            })

        self.emit(BoardSummaryEvent(
            counts=summary,
            experiments=experiments,
            leaderboard=leaderboard,
        ))

    def _maybe_generate_report(self) -> None:
        """Trigger a milestone report if enough experiments are done."""
        if self._stop_requested:
            return
        if self._report_in_progress:
            if self._report_worker is not None:
                if self._report_worker.busy:
                    return  # still running
                # Reporter thread finished (completed or died) — either way, reset state
            else:
                # _report_worker is None but flag is stuck — reset
                logger.warning("Report flag stuck with no worker — resetting")
                self._report_in_progress = False
                return
            # Reporter just finished — copy to output/
            try:
                from alpha_lab.output_generator import OutputGenerator
                gen = OutputGenerator(self.workspace, adapter=self.adapter)
                gen.copy_milestone_report(self._current_report_number)
                gen.generate_index()
            except Exception as e:
                logger.error(f"Output copy failed: {e}")
            self._report_in_progress = False
            self._report_worker = None
            return

        done_count = len(self.db.list_by_status("done", "analyzed"))
        experiments_since_report = done_count - self._last_report_at_done_count

        if experiments_since_report >= self._report_interval and done_count > 0:
            # Find an idle worker for reporting
            idle = [w for w in self.workers if not w.busy]
            if not idle:
                return  # all busy, try next cycle
            if not self._can_start_runtime_work():
                return

            self._report_number += 1
            worker = idle[0]
            logger.info(
                f"Triggering milestone report #{self._report_number} "
                f"({done_count} experiments done)"
            )
            self.emit(ExperimentEvent(
                name="reporter",
                status="running",
                detail=f"Generating milestone report #{self._report_number} ({done_count} done)",
            ))

            # Ensure reports directory exists
            Path(self.workspace, "reports").mkdir(parents=True, exist_ok=True)

            if not self._can_start_runtime_work():
                self._report_number -= 1
                return
            if worker.generate_report(self._report_number, done_count) is False:
                self._report_number -= 1
                return
            self._last_report_at_done_count = done_count
            self._report_in_progress = True
            self._report_worker = worker
            self._current_report_number = self._report_number

    def _track_analyzed(self) -> None:
        """Track how many experiments have been analyzed since last strategist turn.

        Also tracks convergence: if no improvement in top metric for N experiments,
        sets a flag for early stopping.
        """
        current_analyzed = len(self.db.list_by_status("analyzed", "done"))
        new_analyzed = current_analyzed - self._last_analyzed_count
        if new_analyzed > 0:
            with self._state_lock:
                self._analyzed_since_strategist += new_analyzed
            self._last_analyzed_count = current_analyzed

            # Check for improvement in best metric
            leaders = self.db.leaderboard(self._convergence_metric, 1)
            if leaders:
                try:
                    metrics = json.loads(leaders[0].results_json or "{}")
                    default_val = float("inf") if self._metric_direction == "minimize" else float("-inf")
                    current_best = float(metrics.get(self._convergence_metric, default_val))
                    # For maximize: improvement = current > best
                    # For minimize: improvement = current < best
                    if self._metric_direction == "minimize":
                        improved = current_best < self._best_metric_value
                    else:
                        improved = current_best > self._best_metric_value
                    if improved:
                        improvement = abs(current_best - self._best_metric_value)
                        self._best_metric_value = current_best
                        self._experiments_since_improvement = 0
                        logger.info(
                            f"New best {self._convergence_metric}: {current_best:.4f} "
                            f"(improvement: {improvement:.4f})"
                        )
                        self._log("new_best", metric=self._convergence_metric,
                                  value=current_best, improvement=improvement)
                    else:
                        self._experiments_since_improvement += new_analyzed
                except (json.JSONDecodeError, ValueError, TypeError):
                    self._experiments_since_improvement += new_analyzed

    def _check_convergence(self) -> bool:
        """Check if we've converged (no improvement for N experiments)."""
        if self._convergence_threshold <= 0:
            return False
        if self._experiments_since_improvement >= self._convergence_threshold:
            logger.info(
                f"Convergence detected: no improvement in {self._convergence_metric} "
                f"for {self._experiments_since_improvement} experiments"
            )
            self._log("convergence",
                      experiments_since_improvement=self._experiments_since_improvement,
                      best_value=self._best_metric_value)
            return True
        return False

    def _should_terminate(self) -> bool:
        """Check if we should stop: max_experiments fully completed (analyzed+done)."""
        done = self.db.list_by_status("analyzed", "done")

        # Convergence is logged for observability but does not terminate the run.
        # We always run until max_experiments are fully completed so that cross-run
        # comparisons use the same number of real experiments.
        self._check_convergence()  # logs convergence state; result intentionally ignored

        # Stop only when max_experiments have been fully completed
        if len(done) < self._max_experiments:
            return False
        # Don't terminate while experiments are still in progress
        in_flight = self.db.list_by_status(
            "to_implement", "implemented", "checked", "queued", "running", "finished"
        )
        if in_flight:
            return False
        return True
