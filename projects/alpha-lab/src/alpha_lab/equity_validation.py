"""Local-only selection freeze, one holdout evaluation and robustness artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import MarketNeutralBacktest


def block_bootstrap_sharpe(returns, *, seed=20261013, trials=2000, block=20):
    rng=np.random.default_rng(seed)
    x=np.asarray(returns,float)
    values=[]
    for _ in range(trials):
        starts=rng.integers(0,len(x),size=(len(x)+block-1)//block)
        indices=((starts[:,None]+np.arange(block))%len(x)).ravel()[:len(x)]
        sample=x[indices]
        values.append(float(sample.mean()/sample.std(ddof=1)*np.sqrt(252)))
    return {"method":"Circular moving-block bootstrap, 20-session blocks",
            "seed":seed,"trials":trials,"sharpe_95_percentile_interval":np.quantile(values,[.025,.975]).tolist()}


def validate_research(panel:Path, runs:list[Path], output:Path):
    if output.exists():
        raise ValueError("Existing validation directory; never overwrite a frozen holdout study")
    output.mkdir(parents=True)
    engine=MarketNeutralBacktest(panel)
    candidates=[]
    for run in runs:
        for spec in json.loads((run/"candidates.json").read_text()):
            result,_=engine.evaluate(spec)
            eligible=(result["max_abs_net_exposure"]<1e-10
                      and abs(result["train"]["realized_spy_beta"])<=.1
                      and abs(result["validation"]["realized_spy_beta"])<=.1)
            score=min(result["train"]["sharpe_zero_rf"],result["validation"]["sharpe_zero_rf"])
            candidates.append({"run":str(run),"score":score,"eligible":eligible,"result":result})
    eligible=[item for item in candidates if item["eligible"]]
    if not eligible:
        raise ValueError("No candidates satisfy pre-holdout neutrality checks")
    selected=max(eligible,key=lambda item:item["score"])
    import alpha_lab.market_neutral as implementation
    freeze={"frozen_utc":datetime.now(timezone.utc).isoformat(),
            "rule":"Maximize minimum(train, validation) net Sharpe; abs realized SPY beta <=0.1 in both; net exposure <1e-10",
            "candidate_count":len(candidates),"selected":selected,
            "panel_manifest_sha256":file_hash(panel/"panel.json"),
            "backtester_sha256":file_hash(Path(implementation.__file__)),
            "design_sha256":{str(run):file_hash(run/"candidates.json") for run in runs},
            "holdout":"2023-2025; no selection or tuning uses these observations"}
    atomic_json(output/"preholdout-candidates.json",candidates)
    # Persist this commitment BEFORE the first holdout computation.
    atomic_json(output/"selection-frozen.json",freeze)
    shutil.copyfile(implementation.__file__,output/"backtester-frozen.py")
    spec=selected["result"]["spec"]
    base,daily=engine.evaluate(spec,include_holdout=True)
    scenarios=[]
    for name,cost,borrow,terminal in (("base",10,300,0),("frictionless",0,0,0),
                                      ("cost25_borrow1000",25,1000,0),
                                      ("terminal_adverse30",10,300,.3)):
        metrics,_=engine.evaluate(spec,include_holdout=True,cost_bps=cost,
                                  borrow_bps=borrow,terminal_stress=terminal)
        scenarios.append({"scenario":name,"metrics":metrics})
    atomic_json(output/"holdout.json",base)
    atomic_json(output/"robustness.json",scenarios)
    daily.to_csv(output/"selected-daily.csv",index=False)
    holdout=daily.loc[daily.date.between("2023-01-01","2025-12-31"),"net"].to_numpy()
    uncertainty=block_bootstrap_sharpe(holdout)
    atomic_json(output/"holdout-uncertainty.json",uncertainty)
    actions=duckdb.read_parquet(str(panel/"actions.parquet")).df()
    records=[]
    for event in base["missing_event_details"]:
        event_date=np.datetime64(event["date"])
        matches=actions[(actions.ticker==event["ticker"])
                        & (actions.date>=event_date-np.timedelta64(14,"D"))
                        & (actions.date<=event_date+np.timedelta64(3,"D"))]
        records.append({**event,"nearby_actions":matches.action.unique().tolist()})
    atomic_json(output/"missing-price-audit.json",records)
    summary={"selected":spec,"candidate_count":len(candidates),
             "train":base["train"],"validation":base["validation"],"holdout":base["holdout"],
             "uncertainty":uncertainty,"days_with_missing_exposure":base["missing_exposure_events"],
             "unresolved_price_observations":len(base["missing_event_details"]),
             "events_with_nearby_actions":sum(bool(record["nearby_actions"]) for record in records),
             "investment_ready":False,"holdout_tuning_performed":False,
             "limitations":json.loads((panel/"panel.json").read_text())["limitations"]}
    atomic_json(output/"summary.json",summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--run",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    validate_research(args.panel,args.run,args.output)


if __name__=="__main__":
    main()
