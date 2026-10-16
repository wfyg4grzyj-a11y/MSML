"""Export fixed strategy specifications as normal AlphaLab CPU experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import MarketNeutralBacktest, validate_spec


def export_experiments(panel:Path, runs:list[Path], workspace:Path):
    if workspace.exists():
        raise ValueError("Use a new experiment workspace")
    workspace.mkdir(parents=True)
    db=ExperimentDB(str(workspace/"experiments.db"))
    exported=[]
    for run in runs:
        for spec in json.loads((run/"candidates.json").read_text()):
            validate_spec(spec)
            fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
            name=f"candidate_{fingerprint}"
            directory=workspace/"experiments"/name
            directory.mkdir(parents=True)
            config={"model_type":"linear","panel":str(panel.resolve()),
                    "panel_manifest_sha256":file_hash(panel/"panel.json"),
                    "cost_bps":10,"borrow_bps":300,"terminal_stress":0,
                    "holdout_access":False}
            atomic_json(directory/"experiment.json",config)
            # JSON is also valid YAML; use exact quoting for Windows paths.
            atomic_json(directory/"config.yaml",config)
            atomic_json(directory/"strategy.json",spec)
            (directory/"strategy.py").write_text(
                "import json\nfrom pathlib import Path\n"
                "STRATEGY_SPEC=json.loads(Path(__file__).with_name('strategy.json').read_text())\n",
                encoding="utf-8")
            (directory/"run_experiment.py").write_text(
                "from alpha_lab.equity_experiment import run_experiment\n"
                "if __name__=='__main__':\n    run_experiment()\n",encoding="utf-8")
            identifier=db.create(name,spec["name"],"LLM-designed fixed rank-factor hypothesis",json.dumps(config))
            db.update_status(identifier,"implemented")
            db.update_status(identifier,"checked")
            exported.append({"id":identifier,"name":name,"spec":spec,"directory":str(directory)})
    atomic_json(workspace/"exported.json",exported)
    return exported


def run_experiment(config_path=Path("experiment.json")):
    config=json.loads(config_path.read_text())
    if config.get("holdout_access") is not False:
        raise ValueError("Normal experiment workers have no holdout access")
    panel=Path(config["panel"])
    if file_hash(panel/"panel.json")!=config["panel_manifest_sha256"]:
        raise ValueError("Pinned panel manifest changed")
    spec=validate_spec(json.loads(config_path.with_name("strategy.json").read_text()))
    engine=MarketNeutralBacktest(panel)
    result,daily=engine.evaluate(spec,cost_bps=config["cost_bps"],borrow_bps=config["borrow_bps"],
                                 terminal_stress=config["terminal_stress"])
    output=config_path.parent/"results"
    output.mkdir(exist_ok=True)
    import alpha_lab.market_neutral as implementation
    metrics={"sharpe":result["validation"]["sharpe_zero_rf"],
             "max_drawdown":result["validation"]["max_drawdown"],
             "primary_metric_period":"validation 2020-2022; no holdout access",
             "backtester_sha256":file_hash(Path(implementation.__file__)),
             "panel_manifest_sha256":config["panel_manifest_sha256"],
             "research_only":True,"details":result}
    atomic_json(output/"metrics.json",metrics)
    daily.to_csv(output/"daily.csv",index=False)
    print(json.dumps({"strategy":spec["name"],"sharpe":metrics["sharpe"],"research_only":True}),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--run",type=Path,action="append",required=True)
    parser.add_argument("--workspace",type=Path,required=True)
    args=parser.parse_args()
    exported=export_experiments(args.panel,args.run,args.workspace)
    print(json.dumps({"exported":len(exported),"workspace":str(args.workspace)}),flush=True)


if __name__=="__main__":main()
