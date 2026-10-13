"""Verify local CPU/CUDA execution without constructing an LLM provider."""

import argparse
import json
import time
from pathlib import Path

from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager


CPU_SCRIPT = '''
import json, os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
model = LinearRegression().fit(np.arange(10).reshape(-1, 1), np.arange(10) * 3 + 2)
prediction = float(model.predict([[11]])[0])
assert abs(prediction - 35) < 1e-6
Path('result.json').write_text(json.dumps({'device': 'cpu', 'prediction': prediction}))
'''

GPU_SCRIPT = '''
import json
from pathlib import Path
import torch
assert torch.cuda.is_available()
torch.manual_seed(7)
x = torch.randn(128, 8, device='cuda')
target = x.sum(dim=1, keepdim=True)
model = torch.nn.Linear(8, 1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
losses = []
for _ in range(25):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
assert losses[-1] < losses[0] * 0.1
Path('result.json').write_text(json.dumps({'device': torch.cuda.get_device_name(0),
    'cuda': torch.version.cuda, 'initial_loss': losses[0], 'final_loss': losses[-1]}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--gpu', type=int, help='Also run CUDA training on this GPU index')
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    db = ExperimentDB(str(workspace / 'experiments.db'))
    checks = [('cpu', LocalCPUManager(time_limit_seconds=60), CPU_SCRIPT)]
    if args.gpu is not None:
        checks.append(('gpu', LocalGPUManager(gpu_ids=[args.gpu], time_limit_seconds=60), GPU_SCRIPT))
    report = {}
    for name, manager, script in checks:
        experiment_name = f'local_{name}_{time.time_ns()}'
        directory = workspace / 'experiments' / experiment_name
        directory.mkdir(parents=True)
        (directory / 'run_experiment.py').write_text(script, encoding='utf-8')
        (directory / 'config.yaml').write_text('model_type: linear\nbatch_size: 128\ninput_size: 8\n')
        experiment = db.get(db.create(experiment_name, 'Offline local check', 'Local execution works', '{}'))
        try:
            job = manager.submit_experiment(experiment, str(workspace))
            while True:
                status = manager.poll_jobs([job])[job]
                if status != 'RUNNING':
                    break
                time.sleep(0.1)
            if status != 'COMPLETED':
                raise RuntimeError(f'{name} experiment {status}; inspect job output in {directory}')
            report[name] = json.loads((directory / 'result.json').read_text())
            print(f'{name.upper()} passed: {report[name]}', flush=True)
        finally:
            manager.cleanup_all()
    (workspace / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
