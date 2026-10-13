"""Containment regressions and the POSIX dispatch contract."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest

from alpha_lab import local_process
from alpha_lab.tools import execute_shell
from tests.test_tools import python_command


def test_posix_launch_and_group_termination(monkeypatch, tmp_path):
    monkeypatch.setattr(local_process, 'IS_WINDOWS', False)
    process = MagicMock(pid=99999999)
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(local_process.subprocess, 'Popen', popen)
    kill_group = MagicMock()
    monkeypatch.setattr(local_process.os, 'killpg', kill_group, raising=False)
    monkeypatch.setattr(local_process, 'signal', SimpleNamespace(SIGTERM=15, SIGKILL=9))
    managed = local_process.start_process(['python', 'run_experiment.py'], cwd=str(tmp_path), env={'TEST': 'yes'})
    popen.assert_called_once_with(['python', 'run_experiment.py'], cwd=str(tmp_path), env={'TEST': 'yes'}, start_new_session=True)
    managed.terminate_tree()
    assert [call.args for call in kill_group.call_args_list] == [(process.pid, 15), (process.pid, 9)]


def test_shell_timeout_terminates_child(tmp_path):
    script = tmp_path / 'shell child.py'
    script.write_text("import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); Path('child.pid').write_text(str(p.pid)); time.sleep(120)")
    command = python_command(f"import runpy; runpy.run_path({str(script)!r},run_name='__main__')")
    result = execute_shell(command, str(tmp_path), timeout=5)
    assert 'timed out' in result
    pid = int((tmp_path / 'child.pid').read_text())
    assert not psutil.pid_exists(pid)


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object lifetime')
def test_job_close_kills_children_after_parent_exits(tmp_path):
    command = [sys.executable, '-c', "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); Path('child.pid').write_text(str(p.pid))"]
    managed = local_process.start_process(command, cwd=str(tmp_path))
    try:
        managed.wait(timeout=10)
        child = psutil.Process(int((tmp_path / 'child.pid').read_text()))
        managed.close()
        child.wait(timeout=5)
        assert not child.is_running()
    finally:
        managed.close()
