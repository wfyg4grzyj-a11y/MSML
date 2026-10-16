"""Local CPU job management for Phase 3 experiment dispatch.

Runs tree-based models and data preprocessing jobs on CPU while GPU
experiments run on LocalGPUManager. Same interface as LocalGPUManager.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from alpha_lab.experiment_db import Experiment
from alpha_lab.local_process import ManagedProcess, RecoveredProcess, start_process
from alpha_lab.local_job_state import (
    job_directory, load_record, inspect_process, receipt_path, metadata_path,
    record_terminal, terminal_status, write_atomic, effective_time_limit,
)
import psutil

logger = logging.getLogger("alpha_lab.local_cpu")


@dataclass
class LocalCPUJob:
    """Tracks a running local CPU subprocess."""
    proc: ManagedProcess | RecoveredProcess
    exp_name: str
    output_file: TextIO
    workspace: str
    start_time: float = 0.0
    terminal_status: str | None = None
    time_limit: float = 0


class LocalCPUManager:
    """Manages local CPU job spawning for tree-based models.

    Same interface as LocalGPUManager:
    - submit_experiment(exp, workspace) -> job_id
    - poll_jobs(job_ids) -> {job_id: status}
    - cancel(job_id)
    - can_submit() -> bool
    - running_count() -> int
    """

    def __init__(
        self,
        max_parallel: int = 4,
        time_limit_seconds: int = 3600,
        python_executable: str = "",
    ) -> None:
        """
        Parameters
        ----------
        max_parallel : int
            Max concurrent CPU experiments.
        time_limit_seconds : int
            Subprocess timeout.
        python_executable : str
            Path to the Python interpreter for experiment subprocesses.
            Empty string (default) uses sys.executable.
        """
        self.max_parallel = max_parallel
        self.time_limit = time_limit_seconds
        self.python_executable = python_executable or sys.executable
        self._jobs: dict[str, LocalCPUJob] = {}
        self._terminal_jobs: dict[str, str] = {}

    def recover_running_jobs(self, workspace: str, job_id_map: dict[str, str]) -> int:
        """Adopt verified live CPU identities or replay matching terminal evidence."""
        workspace = str(Path(workspace).resolve())
        recovered = 0
        runner = str(Path(__file__).with_name("local_runner.py"))
        for job_id, name in job_id_map.items():
            if job_id in self._jobs or job_id in self._terminal_jobs:
                continue
            try:
                directory = job_directory(workspace, name)
                record = load_record(directory, workspace, name, "cpu", job_id)
                if record is None:
                    continue
                process, live = inspect_process(record, directory, runner)
                if process is None:
                    status = terminal_status(record, directory) if not live else None
                    if status is not None:
                        self._terminal_jobs[job_id] = status
                        recovered += 1
                    continue
                if os.name == "nt" and not record.get("job_name"):
                    continue
                proc = RecoveredProcess(record["pid"], created=record["created"],
                                        job_name=record.get("job_name"),
                                        exit_file=receipt_path(directory, job_id),
                                        exit_identity=(job_id, "cpu"))
                self._jobs[job_id] = LocalCPUJob(
                    proc=proc, exp_name=name, workspace=workspace, start_time=record["start_time"],
                    time_limit=record["time_limit_seconds"],
                    output_file=open(directory / "cpu_job.out", "a", encoding="utf-8"),
                )
                recovered += 1
            except (OSError, ValueError, KeyError, psutil.Error):
                continue
        return recovered

    def _record_terminal(self, job_id: str, status: str, code: int | None = None) -> None:
        job = self._jobs[job_id]
        try:
            record_terminal(job.workspace, job.exp_name, "cpu", job_id, status, code)
        except OSError:
            logger.warning("Could not persist terminal CPU identity for %s", job_id)

    def running_count(self) -> int:
        """Count number of jobs currently running."""
        count = 0
        for job in self._jobs.values():
            if job.proc.poll() is None:
                count += 1
        return count

    def can_submit(self) -> bool:
        """Check if we have capacity to submit another job."""
        return self.running_count() < self.max_parallel

    def submit_experiment(self, exp: Experiment, workspace: str) -> str:
        """Spawn experiment as subprocess. Returns job ID (UUID)."""
        if not self.can_submit():
            raise RuntimeError(f"No CPU slot available (all {self.max_parallel} in use)")

        workspace = str(Path(workspace).resolve())
        exp_dir = job_directory(workspace, exp.name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        job_id = f"cpu-{uuid.uuid4().hex}"
        (exp_dir / "local_exit_code.txt").unlink(missing_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = workspace + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONIOENCODING"] = "utf-8"
        env["ALPHALAB_JOB_ID"] = job_id
        runner = str(Path(__file__).with_name("local_runner.py"))

        # Open output file and spawn subprocess (try/finally to avoid fd leak)
        output_path = exp_dir / "cpu_job.out"
        output_file = open(output_path, "w", encoding="utf-8")
        try:
            proc = start_process(
                [self.python_executable, "-u", runner, "cpu"],
                stdout=output_file,
                stderr=subprocess.STDOUT,
                cwd=str(exp_dir),
                env=env,
                python_executable=self.python_executable,
            )
        except BaseException:
            output_file.close()
            raise

        self._jobs[job_id] = LocalCPUJob(
            proc=proc,
            exp_name=exp.name,
            output_file=output_file,
            workspace=workspace,
            start_time=time.time(),
            time_limit=max(0, self.time_limit),
        )
        try:
            write_atomic(metadata_path(exp_dir, "cpu"), {
                "schema_version": 1, "job_id": job_id, "resource": "cpu", "workspace": workspace,
                "experiment_name": exp.name, "pid": proc.pid, "created": proc.created,
                "start_time": self._jobs[job_id].start_time,
                "time_limit_seconds": self._jobs[job_id].time_limit,
                "job_name": proc.job.name if proc.job else None,
            })
        except BaseException:
            self.cancel(job_id)
            raise

        logger.info(f"Submitted CPU job {job_id} for {exp.name} (PID {proc.pid})")
        return job_id

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Poll for job statuses."""
        result: dict[str, str] = {}
        now = time.time()

        for job_id in job_ids:
            if job_id in self._terminal_jobs:
                result[job_id] = self._terminal_jobs[job_id]
                continue
            if job_id not in self._jobs:
                result[job_id] = "UNKNOWN"
                continue

            job = self._jobs[job_id]
            if job.terminal_status:
                result[job_id] = job.terminal_status
                continue
            retcode = job.proc.poll()

            if retcode is None:
                elapsed = now - job.start_time
                limit = effective_time_limit(job.time_limit, self.time_limit)
                if limit > 0 and elapsed > limit:
                    logger.warning(
                        f"CPU job {job_id} ({job.exp_name}) exceeded time limit "
                        f"({elapsed:.0f}s > {limit}s), killing"
                    )
                    self._kill_job(job_id)
                    job.terminal_status = "TIMEOUT"
                    self._record_terminal(job_id, "TIMEOUT")
                    result[job_id] = "TIMEOUT"
                else:
                    result[job_id] = "RUNNING"
            elif getattr(job.proc, "unverified_exit", False):
                job.terminal_status = "UNKNOWN"
                result[job_id] = "UNKNOWN"
                self._cleanup_job(job_id)
            elif retcode == 0:
                job.terminal_status = "COMPLETED"
                self._record_terminal(job_id, "COMPLETED", retcode)
                result[job_id] = "COMPLETED"
                self._cleanup_job(job_id)
            else:
                job.terminal_status = "FAILED"
                self._record_terminal(job_id, "FAILED", retcode)
                result[job_id] = "FAILED"
                logger.warning(f"CPU job {job_id} ({job.exp_name}) failed with exit code {retcode}")
                self._cleanup_job(job_id)

        return result

    def _kill_job(self, job_id: str) -> None:
        """Kill a job's process."""
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.proc.terminate_tree()
        self._cleanup_job(job_id)

    def cancel(self, job_id: str) -> None:
        """Kill a running job."""
        if job_id not in self._jobs:
            logger.warning(f"Cannot cancel unknown CPU job {job_id}")
            return
        job = self._jobs[job_id]
        if job.terminal_status:
            return
        logger.info(f"Cancelling CPU job {job_id} ({job.exp_name})")
        self._kill_job(job_id)
        job.terminal_status = "CANCELLED"
        self._record_terminal(job_id, "CANCELLED")

    def _cleanup_job(self, job_id: str) -> None:
        """Close output file."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            job.proc.close()
            try:
                job.output_file.close()
            except OSError as e:
                logger.warning("Failed to close output file for CPU job %s: %s", job_id, e)

    def cleanup_all(self) -> None:
        """Kill all running jobs. Call on shutdown."""
        for job_id in list(self._jobs.keys()):
            job = self._jobs[job_id]
            if job.proc.poll() is None:
                self.cancel(job_id)
            else:
                self._cleanup_job(job_id)
        self._jobs.clear()


def is_cpu_experiment(exp: Experiment) -> bool:
    """Check if an experiment should run on CPU based on its config.

    Experiments with model_type containing these keywords run on CPU:
    - xgboost, lightgbm, gbdt, gradient_boost
    - random_forest, decision_tree
    - linear, lasso, ridge, elastic
    - catboost
    """
    import json
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
