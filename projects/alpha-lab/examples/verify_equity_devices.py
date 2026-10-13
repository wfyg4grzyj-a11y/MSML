"""Verify actual dated factor projections through native CPU/GPU job managers."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager
from alpha_lab.market_neutral import neutral_weights


SCRIPT='''
import json, os, time
from pathlib import Path
import numpy as np
import torch
config=json.loads(Path('device-check.json').read_text())
device=config['device']
if device=='cpu':
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
else:
    assert torch.cuda.is_available()
raw=np.load(config['inputs'],allow_pickle=False)
errors=[]; nets=[]; betas=[]; caps=[]
started=time.perf_counter()
for i in range(int(raw['batch_count'])):
    score=torch.as_tensor(raw[f'score_{i}'],dtype=torch.float64,device=device)
    beta=torch.as_tensor(raw[f'beta_{i}'],dtype=torch.float64,device=device)
    design=torch.stack((torch.ones_like(beta),beta),dim=1)
    residual=score-design@(torch.linalg.pinv(design)@score)
    weight=residual/torch.abs(residual).sum()
    weight=weight*torch.minimum(torch.as_tensor(1.,device=device),.02/torch.abs(weight).max())
    value=weight.cpu().numpy()
    errors.append(float(np.max(np.abs(value-raw[f'reference_{i}']))))
    nets.append(abs(float(value.sum())))
    betas.append(abs(float(value@raw[f'beta_{i}'])))
    caps.append(float(np.max(np.abs(value))))
if device=='cuda':torch.cuda.synchronize()
result={'device':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        'torch':torch.__version__,'cuda':torch.version.cuda,'batches':int(raw['batch_count']),
        'max_numpy_difference':max(errors),'max_net_exposure':max(nets),
        'max_estimated_beta_exposure':max(betas),'max_name_weight':max(caps),
        'calculation_seconds':time.perf_counter()-started}
assert result['max_numpy_difference']<1e-8
assert result['max_net_exposure']<1e-10
assert result['max_estimated_beta_exposure']<1e-10
assert result['max_name_weight']<=.0200000001
Path('result.json').write_text(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    if workspace.exists():raise ValueError('Use a new device verification directory')
    workspace.mkdir(parents=True)
    spec=json.loads(args.selection.read_text())['selected']['result']['spec']
    features=duckdb.read_parquet(str(args.panel/'features.parquet')).df()
    batches={}
    count=0
    for date,cross in features.groupby('date',sort=True):
        cross=cross.dropna(subset=list(spec['factors'])+['beta','filing_date'])
        if len(cross)<30:continue
        score=sum((cross[f].rank(pct=True).to_numpy()-.5)*c for f,c in spec['factors'].items())
        beta=cross.beta.to_numpy()
        batches[f'score_{count}']=score; batches[f'beta_{count}']=beta
        batches[f'reference_{count}']=neutral_weights(score,beta)
        count+=1
    batches['batch_count']=count
    inputs=workspace/'projection-inputs.npz'
    np.savez(inputs,**batches)
    db=ExperimentDB(str(workspace/'experiments.db'))
    report={}
    for label,device,manager in [('cpu','cpu',LocalCPUManager(max_parallel=1,time_limit_seconds=120)),
                                 ('gpu','cuda',LocalGPUManager(gpu_ids=[0],time_limit_seconds=120))]:
        name=f'equity_projection_{label}'
        directory=workspace/'experiments'/name
        directory.mkdir(parents=True)
        (directory/'run_experiment.py').write_text(SCRIPT,encoding='utf-8')
        (directory/'device-check.json').write_text(json.dumps({'inputs':str(inputs),'device':device}))
        (directory/'config.yaml').write_text('model_type: factor_projection\n')
        experiment=db.get(db.create(name,'Dated equity factor projection','CPU/GPU parity','{}'))
        try:
            job=manager.submit_experiment(experiment,str(workspace))
            while True:
                status=manager.poll_jobs([job])[job]
                if status!='RUNNING':break
                time.sleep(.1)
            if status!='COMPLETED':raise RuntimeError(f'{label} job {status}; inspect local job output')
            report[label]=json.loads((directory/'result.json').read_text())
            print(json.dumps(report[label]),flush=True)
        finally:
            manager.cleanup_all()
    (workspace/'verification.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
