"""Process-owned workspace exclusion for local AlphaLab research runners."""
from __future__ import annotations

import errno
import os
import threading
from pathlib import Path


class WorkspaceOwnedError(RuntimeError):
    """Another local runner owns the workspace."""


class WorkspaceLock:
    """Keep a persistent lock file; never unlink it while another handle may exist."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / ".alpha_lab_runner.lock"
        self._file = None
        self._pid = os.getpid()
        self._mutex = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._reaper = None

    @property
    def held(self) -> bool:
        return self._file is not None and self._pid == os.getpid()

    def acquire(self) -> WorkspaceLock:
        with self._mutex:
            if self._file is not None:
                raise RuntimeError("Workspace lock is already acquired by this object")
            self.workspace.mkdir(parents=True, exist_ok=True)
            handle = open(self.path, "a+b")
            os.set_inheritable(handle.fileno(), False)
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    # Windows permits locking a byte beyond EOF. No pre-lock write.
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                handle.close()
                if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise WorkspaceOwnedError(f"Workspace already owned by a runner: {self.workspace}") from None
                raise
            self._file = handle
            self._pid = os.getpid()
            return self

    def retain_until(self, threads: list[threading.Thread]) -> None:
        """A timed-out shutdown keeps exclusion until its existing threads end."""
        with self._mutex:
            self._threads.update(thread for thread in threads if thread.is_alive())

    def release(self) -> bool:
        with self._mutex:
            if not self.held:
                return True
            alive = [thread for thread in self._threads if thread.is_alive()]
            if alive:
                if self._reaper is None:
                    self._reaper = threading.Thread(target=self._release_after_threads, daemon=True)
                    self._reaper.start()
                return False
            handle, self._file = self._file, None
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
            return True

    def _release_after_threads(self) -> None:
        while True:
            with self._mutex:
                alive = [thread for thread in self._threads if thread.is_alive()]
            if not alive:
                self.release()
                return
            for thread in alive:
                thread.join()

    def __enter__(self) -> WorkspaceLock:
        return self.acquire()

    def __exit__(self, *_args) -> None:
        self.release()
