"""Real subprocess tests; no providers or paid API calls are used."""

import json
import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager


@pytest.fixture(params=['cpu', 'gpu'])
def manager(request):
    if request.param == 'gpu':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA hardware required')
        value = LocalGPUManager(gpu_ids=[0], time_limit_seconds=60)
    else:
        value = LocalCPUManager(max_parallel=1, time_limit_seconds=60)
    yield value
    value.cleanup_all()


def prepare(tmp_path, script, name='experiment with spaces 測試'):
    workspace = tmp_path / 'workspace with spaces 測試'
    directory = workspace / 'experiments' / name
    directory.mkdir(parents=True)
    (directory / 'run_experiment.py').write_text(script, encoding='utf-8')
    (directory / 'local_helper.py').write_text('VALUE = 11', encoding='utf-8')
    (workspace / 'workspace_helper.py').write_text('VALUE = 42', encoding='utf-8')
    db = ExperimentDB(str(workspace / 'experiments.db'))
    experiment_id = db.create(name, 'Local verification', 'No LLM API', '{}')
    experiment = db.get(experiment_id)
    return workspace, directory, experiment


def until(check, seconds=40):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail('Local experiment did not reach the expected state')


def finished(manager, job_id):
    state = manager.poll_jobs([job_id])[job_id]
    return state if state != 'RUNNING' else None


def test_cpu_and_gpu_calculation(manager, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # An existing PYTHONPATH entry must survive, and a relative workspace must
    # resolve before the experiment changes its working directory.
    helper = tmp_path / 'existing import path'
    helper.mkdir()
    (helper / 'existing_helper.py').write_text('VALUE = 17')
    monkeypatch.setenv('PYTHONPATH', str(helper))
    gpu = isinstance(manager, LocalGPUManager)
    calculation = '''
import torch
assert torch.cuda.is_available()
torch.use_deterministic_algorithms(True)
torch.use_deterministic_algorithms(False)
a = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)
torch.testing.assert_close((a @ a).cpu(), a.cpu() @ a.cpu())
result = {'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
''' if gpu else '''
import numpy as np
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(np.arange(10).reshape(-1, 1), np.arange(10) * 3 + 2)
assert abs(model.predict([[11]])[0] - 35) < 1e-6
result = {'device': 'cpu', 'prediction': float(model.predict([[11]])[0])}
'''
    script = '''import json, os
from pathlib import Path
import workspace_helper, existing_helper, local_helper
assert workspace_helper.VALUE == 42 and existing_helper.VALUE == 17
assert local_helper.VALUE == 11
assert os.environ['CUDA_VISIBLE_DEVICES'] == %r
print('output 測試', flush=True)
''' % ('0' if gpu else '') + calculation + "\nPath('result.json').write_text(json.dumps(result))\n"
    workspace, directory, experiment = prepare(tmp_path, script)
    job = manager.submit_experiment(experiment, os.path.relpath(workspace, tmp_path))
    assert until(lambda: finished(manager, job)) == 'COMPLETED'
    assert manager.poll_jobs([job])[job] == 'COMPLETED'
    result = json.loads((directory / 'result.json').read_text())
    if gpu:
        assert result['device'] != 'cpu'
    else:
        assert result['device'] == 'cpu'
    output = 'local_job.out' if gpu else 'cpu_job.out'
    assert 'output 測試' in (directory / output).read_text(encoding='utf-8')
    assert manager.can_submit()


def test_failure_retains_traceback(manager, tmp_path):
    workspace, directory, experiment = prepare(tmp_path, "raise RuntimeError('intentional local failure')")
    job = manager.submit_experiment(experiment, str(workspace))
    assert until(lambda: finished(manager, job)) == 'FAILED'
    output = 'local_job.out' if isinstance(manager, LocalGPUManager) else 'cpu_job.out'
    assert 'intentional local failure' in (directory / output).read_text(encoding='utf-8')


@pytest.mark.parametrize('action', ['cancel', 'timeout'])
def test_cancel_and_timeout_kill_descendants(manager, tmp_path, action):
    script = '''import subprocess, sys, time, json, os
from pathlib import Path
grandchild_code = "import time; time.sleep(120)"
child_code = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c',%r]); Path('grandchild.pid').write_text(str(p.pid)); time.sleep(120)" % grandchild_code
child = subprocess.Popen([sys.executable, '-c', child_code])
Path('child.pid').write_text(str(child.pid))
while not Path('grandchild.pid').exists(): time.sleep(0.01)
Path('ready').write_text('ready')
time.sleep(120)
'''
    workspace, directory, experiment = prepare(tmp_path, script)
    job = manager.submit_experiment(experiment, str(workspace))
    until(lambda: (directory / 'ready').exists())
    processes = [psutil.Process(int((directory / file).read_text()))
                 for file in ['child.pid', 'grandchild.pid']]
    assert not manager.can_submit()
    if action == 'cancel':
        manager.cancel(job)
        expected = 'CANCELLED'
    else:
        manager.time_limit = 1
        manager._jobs[job].start_time = time.time() - 2
        expected = 'TIMEOUT'
    assert manager.poll_jobs([job])[job] == expected
    assert manager.poll_jobs([job])[job] == expected
    _, alive = psutil.wait_procs(processes, timeout=5)
    try:
        assert not alive, 'Child or grandchild survived termination'
    finally:
        for process in alive:
            process.kill()
    assert manager.can_submit()


def test_gpu_recovery_checks_identity(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware required')
    manager = LocalGPUManager(gpu_ids=[0], time_limit_seconds=60)
    recovered = LocalGPUManager(gpu_ids=[0], time_limit_seconds=60)
    workspace, directory, experiment = prepare(tmp_path, "import time; from pathlib import Path; Path('ready').write_text('ready'); time.sleep(120)")
    try:
        job = manager.submit_experiment(experiment, str(workspace))
        until(lambda: (directory / 'ready').exists())
        metadata = directory / 'local_job.json'
        record = json.loads(metadata.read_text())
        altered = dict(record, created=record['created'] - 1)
        metadata.write_text(json.dumps(altered))
        assert recovered.recover_running_jobs(str(workspace), {job: experiment.name}) == 0
        metadata.write_text(json.dumps(record))
        assert recovered.recover_running_jobs(str(workspace), {job: experiment.name}) == 1
        assert recovered.poll_jobs([job])[job] == 'RUNNING'
        recovered.cancel(job)
        assert recovered.poll_jobs([job])[job] == 'CANCELLED'
    finally:
        recovered.cleanup_all()
        manager.cleanup_all()
