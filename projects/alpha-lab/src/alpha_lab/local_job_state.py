"""Invocation-scoped native job identity and terminal evidence."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import psutil


def _native_file(path: Path) -> Path:
    absolute = str(path.absolute())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = ("\\\\?\\UNC\\"+absolute[2:]) if absolute.startswith("\\\\") else "\\\\?\\"+absolute
    return Path(absolute)


def job_directory(workspace: str, name: str) -> Path:
    base = (Path(workspace).resolve() / "experiments").resolve()
    directory = (base / name).resolve()
    directory.relative_to(base)
    if directory == base:
        raise ValueError("Experiment requires its own directory")
    return directory


def metadata_path(directory: Path, resource: str) -> Path:
    return _native_file(directory / ("cpu_job.json" if resource == "cpu" else "local_job.json"))


def receipt_path(directory: Path, job_id: str) -> Path:
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_id):
        raise ValueError("Invalid native job identity")
    return _native_file(directory / f"local_job_result-{job_id}.json")


def write_atomic(path: Path, record: dict) -> None:
    path = _native_file(path)
    descriptor, temporary = tempfile.mkstemp(prefix="job.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: Path) -> dict | None:
    try:
        with _native_file(path).open("r", encoding="utf-8") as stream:
            text = stream.read(65537)
        if len(text.encode("utf-8")) > 65536:
            return None
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _positive_number(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def load_record(directory: Path, workspace: str, name: str, resource: str, job_id: str) -> dict | None:
    record = read_json(metadata_path(directory, resource))
    if (not record or type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("job_id") != job_id
            or record.get("resource") != resource or record.get("experiment_name") != name
            or record.get("workspace") != str(Path(workspace).resolve())
            or type(record.get("pid")) is not int or record["pid"] <= 0
            or not _positive_number(record.get("created"))
            or not _positive_number(record.get("start_time"))
            or type(record.get("time_limit_seconds")) not in (int, float)
            or not math.isfinite(record["time_limit_seconds"]) or record["time_limit_seconds"] < 0):
        return None
    job_name = record.get("job_name")
    if os.name == "nt":
        if not isinstance(job_name, str) or not job_name.startswith("Local\\AlphaLab-") or len(job_name) > 128:
            return None
    elif job_name is not None:
        return None
    try:
        receipt_path(directory, job_id)
    except ValueError:
        return None
    return record


def effective_time_limit(original: float, current: float) -> float:
    """A recovered invocation cannot extend its original positive timeout."""
    positive = [value for value in (original, current) if value > 0]
    return min(positive) if positive else 0


def inspect_process(record: dict, directory: Path, runner: str) -> tuple[psutil.Process | None, bool]:
    """Return verified process, and whether this PID vintage is still live."""
    try:
        process = psutil.Process(record["pid"])
        if process.create_time() != record["created"]:
            return None, False  # PID reused: never adopt or signal the replacement.
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None, False
        if Path(process.cwd()).resolve() == directory and runner in process.cmdline():
            return process, True
        return None, True  # Same live vintage but wrong command/directory: quarantine.
    except psutil.NoSuchProcess:
        return None, False
    except psutil.Error:
        return None, True  # Cannot attest that a process is gone.


def read_exit_code(path: Path, job_id: str, resource: str) -> int | None:
    receipt = read_json(path)
    if (not receipt or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or receipt.get("job_id") != job_id
            or receipt.get("resource") != resource or type(receipt.get("exit_code")) is not int
            or not _positive_number(receipt.get("finished_at"))):
        return None
    return receipt["exit_code"]


def terminal_status(record: dict, directory: Path) -> str | None:
    status = record.get("terminal_status")
    code = record.get("exit_code")
    if _positive_number(record.get("finished_at")):
        if status in ("CANCELLED", "TIMEOUT"):
            return status
        if type(code) is int and status == ("COMPLETED" if code == 0 else "FAILED"):
            return status
    code = read_exit_code(receipt_path(directory, record["job_id"]), record["job_id"], record["resource"])
    return None if code is None else ("COMPLETED" if code == 0 else "FAILED")


def record_terminal(workspace: str, name: str, resource: str, job_id: str,
                    status: str, exit_code: int | None = None) -> None:
    directory = job_directory(workspace, name)
    record = load_record(directory, workspace, name, resource, job_id)
    if record is not None:
        record.update(terminal_status=status, exit_code=exit_code, finished_at=time.time())
        write_atomic(metadata_path(directory, resource), record)
