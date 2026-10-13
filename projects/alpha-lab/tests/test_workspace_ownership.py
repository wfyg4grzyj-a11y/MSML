"""Native workspace exclusion, crash release and runner resource boundaries."""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.config import TaskConfig
from alpha_lab.dispatcher import Dispatcher
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.workspace_lock import WorkspaceLock, WorkspaceOwnedError


def test_same_workspace_alias_is_excluded_and_lock_file_is_retained(tmp_path):
    workspace = tmp_path/'workspace with spaces 測試'
    first = WorkspaceLock(workspace).acquire()
    try:
        with pytest.raises(WorkspaceOwnedError):
            WorkspaceLock(workspace/'..'/workspace.name).acquire()
        assert not os.get_inheritable(first._file.fileno())
    finally:
        first.release()
    assert first.path.exists()
    with WorkspaceLock(workspace) as replacement:
        assert replacement.held


def test_second_process_cannot_enter_owned_workspace(tmp_path):
    code = '''import sys
from alpha_lab.workspace_lock import WorkspaceLock,WorkspaceOwnedError
try:
    with WorkspaceLock(sys.argv[1]): print('entered')
except WorkspaceOwnedError: print('blocked')
'''
    with WorkspaceLock(tmp_path):
        result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0 and result.stdout.strip() == 'blocked'


def test_crashed_process_releases_os_lock_without_deleting_file(tmp_path):
    code = '''import os,sys
from alpha_lab.workspace_lock import WorkspaceLock
owner=WorkspaceLock(sys.argv[1]).acquire()
print('ready',flush=True)
sys.stdin.readline()
os._exit(17)
'''
    child = subprocess.Popen([sys.executable, '-u', '-c', code, str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(WorkspaceOwnedError):
            WorkspaceLock(tmp_path).acquire()
        child.communicate('exit\n', timeout=10)
        assert child.returncode == 17
        assert (tmp_path/'.alpha_lab_runner.lock').exists()
        with WorkspaceLock(tmp_path):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def test_timed_out_worker_keeps_lock_until_its_thread_finishes(tmp_path):
    release = threading.Event()
    worker = threading.Thread(target=lambda: release.wait(5))
    worker.start()
    owner = WorkspaceLock(tmp_path).acquire()
    try:
        owner.retain_until([worker])
        assert owner.release() is False
        with pytest.raises(WorkspaceOwnedError):
            WorkspaceLock(tmp_path).acquire()
    finally:
        release.set()
        worker.join(5)
    deadline = time.monotonic()+5
    while owner.held and time.monotonic()<deadline:
        time.sleep(.01)
    assert not owner.held
    with WorkspaceLock(tmp_path):
        pass


def make_dispatcher(workspace, db):
    return Dispatcher(MagicMock(), TaskConfig(data_path='synthetic', description='Offline ownership'),
                      str(workspace), db, MagicMock(), lambda event: None, worker_count=0)


@pytest.mark.parametrize('entrypoint', ['run', 'recover'])
def test_second_dispatcher_cannot_release_existing_claim(tmp_path, entrypoint):
    db = ExperimentDB(str(tmp_path/'experiments.db'))
    identifier = db.create('synthetic', 'Offline', 'No API', '{}')
    db.assign_worker(identifier, 'first-runner')
    second = make_dispatcher(tmp_path, db)
    with WorkspaceLock(tmp_path):
        with pytest.raises(WorkspaceOwnedError):
            getattr(second, entrypoint)()
    assert db.get(identifier).worker_id == 'first-runner'
    second.executor.cleanup_all.assert_not_called()


def test_standalone_dispatcher_holds_ownership_and_cleans_up_on_error(tmp_path):
    db = ExperimentDB(str(tmp_path/'experiments.db'))
    dispatcher = make_dispatcher(tmp_path, db)

    def body():
        with pytest.raises(WorkspaceOwnedError):
            WorkspaceLock(tmp_path).acquire()
        raise RuntimeError('synthetic body error')

    dispatcher._run_owned = body
    with pytest.raises(RuntimeError, match='body error'):
        dispatcher.run()
    dispatcher.executor.cleanup_all.assert_called_once()
    with WorkspaceLock(tmp_path):
        pass


def test_runner_refuses_before_provider_or_event_log(tmp_path, monkeypatch):
    from alpha_lab.run import run_main
    workspace = tmp_path/'workspace'
    monkeypatch.setattr(sys, 'argv', ['alpha-lab-run', '--config', 'synthetic.yaml', '--workspace', str(workspace)])
    with patch('alpha_lab.run.load_config', return_value=TaskConfig(data_path='synthetic', description='Offline')):
        with patch('alpha_lab.client.get_provider') as factory, WorkspaceLock(workspace):
            with pytest.raises(WorkspaceOwnedError):
                run_main()
            factory.assert_not_called()
    assert not (tmp_path/'events.jsonl').exists()


def test_phase0_failure_closes_provider_log_and_releases_workspace(tmp_path, monkeypatch):
    from alpha_lab import run
    workspace = tmp_path/'workspace'
    monkeypatch.setattr(sys, 'argv', ['alpha-lab-run', '--config', 'synthetic.yaml', '--workspace', str(workspace)])
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    provider = MagicMock()
    with patch('alpha_lab.run.load_config', return_value=TaskConfig(data_path='synthetic', description='Offline')):
        with patch('alpha_lab.client.get_provider', return_value=provider):
            with patch('alpha_lab.phase0.run_phase0', side_effect=RuntimeError('synthetic phase0 failure')):
                with pytest.raises(RuntimeError, match='phase0 failure'):
                    run.run_main()
    provider.close.assert_called_once()
    assert run._event_log_file is None
    with WorkspaceLock(workspace):
        pass


def test_interactive_cli_requires_explicit_workspace_before_provider(monkeypatch):
    from alpha_lab.cli import main
    monkeypatch.setattr(sys, 'argv', ['alpha-lab'])
    with patch('alpha_lab.client.get_provider') as factory:
        with pytest.raises(SystemExit) as exited:
            main()
        assert exited.value.code == 2
        factory.assert_not_called()


def test_interactive_cli_is_excluded_before_provider(tmp_path, monkeypatch):
    from alpha_lab.cli import main
    monkeypatch.setattr(sys, 'argv', ['alpha-lab', '--workspace', str(tmp_path)])
    with patch('alpha_lab.client.get_provider') as factory, WorkspaceLock(tmp_path):
        with pytest.raises(WorkspaceOwnedError):
            main()
        factory.assert_not_called()


def test_interactive_setup_failure_closes_provider_and_releases_workspace(tmp_path, monkeypatch):
    from alpha_lab.cli import main
    monkeypatch.setattr(sys, 'argv', ['alpha-lab', '--workspace', str(tmp_path)])
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-not-a-real-key')
    provider = MagicMock()
    with patch('alpha_lab.client.get_provider', return_value=provider):
        with patch('alpha_lab.cli.PromptSession', side_effect=RuntimeError('synthetic prompt failure')):
            with pytest.raises(RuntimeError, match='prompt failure'):
                main()
    provider.close.assert_called_once()
    with WorkspaceLock(tmp_path):
        pass
