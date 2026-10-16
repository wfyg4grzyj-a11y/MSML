"""Verify actual dated factor projections through native CPU/GPU job managers."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.local_cpu import LocalCPUManager
from alpha_lab.local_gpu import LocalGPUManager
from alpha_lab.market_neutral import MarketNeutralBacktest, neutral_weights


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
def project_values(score_values,beta_values):
    score=torch.as_tensor(score_values,dtype=torch.float64,device=device)
    beta=torch.as_tensor(beta_values,dtype=torch.float64,device=device)
    design=torch.stack((torch.ones_like(beta),beta),dim=1)
    residual=score-design@(torch.linalg.pinv(design)@score)
    gross=torch.abs(residual).sum()
    if float(gross.item())<1e-10:
        return np.zeros(len(score_values))
    weight=residual/gross
    weight=weight*torch.minimum(torch.as_tensor(1.,device=device),.02/torch.abs(weight).max())
    return weight.cpu().numpy()
def project(i):
    return project_values(raw[f'score_{i}'],raw[f'beta_{i}'])
started=time.perf_counter()
project(0)  # Initialize allocation, BLAS and CUDA/SVD libraries before timing.
sanity_beta=np.linspace(.2,2.,100)
assert np.max(np.abs(project_values(3*sanity_beta+5,sanity_beta)))==0
assert np.max(np.abs(project_values(np.zeros(100),sanity_beta)))==0
if device=='cuda':torch.cuda.synchronize()
initialization=time.perf_counter()-started
timings=[]
for trial in range(3):
    if device=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    for i in range(int(raw['batch_count'])):
        value=project(i)
        errors.append(float(np.max(np.abs(value-raw[f'reference_{i}']))))
        nets.append(abs(float(value.sum())))
        betas.append(abs(float(value@raw[f'beta_{i}'])))
        caps.append(float(np.max(np.abs(value))))
    if device=='cuda':torch.cuda.synchronize()
    timings.append(time.perf_counter()-started)
result={'device':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        'torch':torch.__version__,'cuda':torch.version.cuda,'batches':int(raw['batch_count']),
        'max_numpy_difference':max(errors),'max_net_exposure':max(nets),
        'max_estimated_beta_exposure':max(betas),'max_name_weight':max(caps),
        'calculation_seconds':float(np.median(timings)),
        'initialization_seconds':initialization,'warm_trial_seconds':timings,
        'degenerate_projection_checks':2,
        'timing_scope':'All supplied monthly projections, host/device transfers and parity checks'}
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
    features=MarketNeutralBacktest(args.panel).features
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
