"""Local process containment: Windows Job Objects or POSIX process groups."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import uuid
from ctypes import wintypes
from typing import Any

import psutil

IS_WINDOWS = os.name == "nt"


class WindowsJob:
    """A named, non-inheritable job that owns every descendant process."""

    def __init__(self, name: str | None = None, *, existing: bool = False):
        self.name = name or f"Local\\AlphaLab-{uuid.uuid4()}"
        self.handle = None
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel = kernel
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.OpenJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        if existing:
            self.handle = kernel.OpenJobObjectW(0x0001 | 0x0008, False, self.name)
        else:
            self.handle = kernel.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if existing:
            return

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t), ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits), ("io", ctypes.c_ulonglong * 6),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t),
            ]

        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, proc: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self) -> None:
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __del__(self):
        # Also release handles if setup/recovery fails before a manager owns us.
        try:
            self.close()
        except (AttributeError, OSError):
            pass


class ManagedProcess:
    def __init__(self, proc: subprocess.Popen, job: WindowsJob | None = None):
        self.proc = proc
        self.job = job
        self.pid = proc.pid
        try:
            self.created = psutil.Process(proc.pid).create_time()
        except psutil.NoSuchProcess:
            self.created = None

    def poll(self):
        return self.proc.poll()

    def wait(self, timeout=None):
        return self.proc.wait(timeout=timeout)

    def communicate(self, timeout=None):
        return self.proc.communicate(timeout=timeout)

    def terminate_tree(self) -> None:
        if self.job:
            self.job.terminate()
            self.wait(timeout=5)
        else:
            # The saved session leader PID remains the process group ID even
            # if the leader exits before its descendants.
            try:
                os.killpg(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.wait(timeout=5)

    def close(self) -> None:
        if self.job:
            self.job.close()


def start_process(command: list[str], *, cwd: str, env: dict[str, str] | None = None,
                  python_executable: str = sys.executable, **kwargs: Any) -> ManagedProcess:
    """Contain before executing user code, eliminating the child-spawn race."""
    if not IS_WINDOWS:
        return ManagedProcess(subprocess.Popen(command, cwd=cwd, env=env,
                                              start_new_session=True, **kwargs))
    job = WindowsJob()
    # This small bootstrap waits until assigned to the job. Every subsequent
    # process (including the actual interpreter) inherits job membership.
    bootstrap = (
        "import subprocess,sys; sys.stdin.buffer.read(1); "
        "sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL))"
    )
    proc = None
    try:
        proc = subprocess.Popen([python_executable, "-u", "-c", bootstrap, *command],
                                cwd=cwd, env=env, stdin=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW, **kwargs)
        job.assign(proc)
        managed = ManagedProcess(proc, job)
        proc.stdin.write("\n" if proc.text_mode else b"\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.stdin = None
        return managed
    except BaseException:
        if proc is not None:
            proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    stream.close()
        job.close()
        raise


class RecoveredProcess(ManagedProcess):
    """Identity-checked process reference; never call os.kill(pid, 0) on Windows."""

    def __init__(self, pid: int, *, created: float | None = None,
                 job_name: str | None = None, exit_file=None):
        self.process = psutil.Process(pid)
        if created is not None and self.process.create_time() != created:
            raise ValueError("PID belongs to a different process")
        self.pid = pid
        self.job = WindowsJob(job_name, existing=True) if job_name else None
        self.exit_file = exit_file
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            try:
                if self.process.is_running() and self.process.status() != psutil.STATUS_ZOMBIE:
                    return None
            except psutil.NoSuchProcess:
                pass
            try:
                self.returncode = int(self.exit_file.read_text())
            except (AttributeError, OSError, ValueError):
                self.returncode = 1  # Never report unverifiable work as successful.
        return self.returncode

    def wait(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except psutil.TimeoutExpired as error:
            raise subprocess.TimeoutExpired(str(self.pid), timeout) from error
        return self.poll()
