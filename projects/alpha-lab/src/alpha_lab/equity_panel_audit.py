"""Compare a panel correction using the frozen spec and pre-holdout periods only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import FEATURES, MarketNeutralBacktest


def compare_panels(original:Path, corrected:Path, selection:Path, output:Path):
    if output.exists():
        raise ValueError("Choose a new comparison directory; original studies are immutable")
    freeze_hash=file_hash(selection)
    frozen=json.loads(selection.read_text())
    spec=frozen['selected']['result']['spec']
    if frozen['panel_manifest_sha256']!=file_hash(original/'panel.json'):
        raise ValueError("Original panel does not match the frozen study")
    engines=[MarketNeutralBacktest(panel) for panel in (original,corrected)]
    if not all(engine.panel_integrity_verified for engine in engines):
        raise ValueError("Panel comparison requires complete verified panel hashes")
    if json.loads((original/'sources.json').read_text())!=json.loads((corrected/'sources.json').read_text()):
        raise ValueError("A methodology comparison must use identical raw-source snapshots")
    output.mkdir(parents=True)
    commitment={'created_utc':datetime.now(timezone.utc).isoformat(),'spec':spec,
                'purpose':'Methodology correction diagnostic; no candidate selection or holdout reuse',
                'evaluation_periods':['train 2014-2019','validation 2020-2022'],
                'holdout_evaluated':False,'new_strategy_trials':0,
                'selection_sha256':freeze_hash,'original_panel_sha256':file_hash(original/'panel.json'),
                'corrected_panel_sha256':file_hash(corrected/'panel.json')}
    atomic_json(output/'comparison-frozen.json',commitment)
    old,new=(engine.features for engine in engines)
    keys=['date','ticker']
    identity=old[keys].merge(new[keys],on=keys,how='outer',indicator=True)
    changed=identity.loc[identity['_merge']!='both'].copy()
    changed.to_csv(output/'universe-changes.csv',index=False)
    shared=old.merge(new,on=keys,suffixes=('_old','_new'))
    differences={}
    for name in FEATURES:
        left,right=shared[name+'_old'].to_numpy(float),shared[name+'_new'].to_numpy(float)
        finite=np.isfinite(left)&np.isfinite(right)
        count=int(np.count_nonzero((np.isfinite(left)!=np.isfinite(right))
                                  | (finite & ~np.isclose(left,right,rtol=1e-10,atol=1e-12))))
        differences[name]={'shared_rows_changed':count,
                           'maximum_absolute_change':float(np.max(np.abs(left[finite]-right[finite]))) if finite.any() else None}
    diagnostic=[]
    for label,engine in zip(('original','corrected'),engines):
        result,daily=engine.evaluate(spec,include_holdout=False)
        if 'holdout' in result or (daily.date>pd.Timestamp('2022-12-31')).any():
            raise ValueError("Panel comparison accidentally accessed a protected evaluation period")
        atomic_json(output/(label+'-preholdout.json'),result)
        daily.to_csv(output/(label+'-preholdout-daily.csv'),index=False)
        diagnostic.append({'panel':label,'train':result['train'],'validation':result['validation'],
                           'missing_price_observations':result['missing_price_observations']})
    if file_hash(selection)!=freeze_hash:
        raise ValueError("Frozen selection changed during the comparison")
    summary={**commitment,'original_selection_preserved':True,
             'original_rows':len(old),'corrected_rows':len(new),'shared_rows':len(shared),
             'removed_monthly_security_rows':int((identity['_merge']=='left_only').sum()),
             'added_monthly_security_rows':int((identity['_merge']=='right_only').sum()),
             'changed_features':differences,'preholdout_diagnostics':diagnostic,
             'investment_ready':False,'licensed_data_and_results':'local_only'}
    atomic_json(output/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-panel',type=Path,required=True)
    parser.add_argument('--corrected-panel',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compare_panels(args.original_panel,args.corrected_panel,args.selection,args.output)


if __name__=='__main__':
    main()
