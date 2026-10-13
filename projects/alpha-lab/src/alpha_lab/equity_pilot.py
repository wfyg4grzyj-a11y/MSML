"""Run AlphaLab's AgentLoop against explicit, offline equity research tools."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from alpha_lab.llm_runtime import LLMSettings, bounded_provider
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import FEATURES, MarketNeutralBacktest, prepare_panel, validate_spec


def function(name, description, properties, required):
    return {"type":"function", "name":name, "description":description, "strict":True,
            "parameters":{"type":"object", "properties":properties, "required":required,
                          "additionalProperties":False}}


TOOLS = [
    function("research_rules", "Read public factor definitions and predetermined research rules", {}, []),
    function("register_strategy", "Save a strategy design for later local testing; returns no market data or results",
             {"name":{"type":"string"}, "factors":{"type":"array", "items":{
                 "type":"object", "properties":{"feature":{"type":"string","enum":list(FEATURES)},
                                                    "coefficient":{"type":"number"}},
                 "required":["feature","coefficient"], "additionalProperties":False}}},
             ["name","factors"]),
    function("report_to_user", "Finish with economic hypotheses and testing recommendations; performance is unknown",
             {"summary":{"type":"string"}}, ["summary"]),
]


class PublicStrategyDesigner:
    """LLM boundary: no data-store or backtester reference, only public definitions.

    Candidate backtests run AFTER this agent finishes. Their results never enter
    this tool's responses or the model conversation.
    """
    def __init__(self, workspace, max_experiments=8):
        self.workspace=workspace
        self.max_experiments=max_experiments
        self.candidates=[]
        self.report=None

    def __call__(self,name,args):
        if name=="research_rules":
            output={"features":list(FEATURES),
                    "definitions":{"book_yield":"book equity / market capitalization",
                                   "earnings_yield":"as-reported trailing net income / market capitalization",
                                   "fcf_yield":"as-reported trailing free cash flow / market capitalization",
                                   "roa":"reported return on assets",
                                   "gross_profitability":"trailing gross profit / assets",
                                   "leverage":"debt / assets; lower can be preferable",
                                   "momentum_12_1":"12-month return excluding the most recent month",
                                   "momentum_3_1":"3-month return excluding the most recent month",
                                   "reversal_1m":"negative most recent 1-month return",
                                   "low_vol":"negative trailing 63-session return volatility"},
                    "construction":"Percentile ranks, project off intercept and trailing market beta, gross <=1, name cap 2%",
                    "predetermined_costs":"10 bps one-way traded notional; 3% annual short borrow; no cash interest",
                    "timing_rule":"Monthly close signals; trade following session close; earn subsequent returns",
                    "design_limit":self.max_experiments,
                    "privacy":"Strategy design only. No licensed data, dataset statistics or backtest results are available."}
        elif name=="register_strategy":
            factors=args["factors"]
            if len({f["feature"] for f in factors})!=len(factors):
                raise ValueError("Duplicate factors")
            spec=validate_spec({"name":args["name"],"factors":{f["feature"]:f["coefficient"] for f in factors}})
            if len(self.candidates)>=self.max_experiments:
                raise ValueError("Design budget exhausted")
            if any(c["name"]==spec["name"] for c in self.candidates):
                raise ValueError("Use distinct strategy names")
            self.candidates.append(spec)
            atomic_json(self.workspace/"candidates.json",self.candidates)
            output={"registered":spec["name"],"testing":"Deferred to local backtester after strategy design finishes"}
        elif name=="report_to_user":
            if len(self.candidates)<3:
                raise ValueError("Register at least three candidates before reporting")
            self.report=args["summary"]
            (self.workspace/"agent-report.txt").write_text(self.report,encoding="utf-8")
            return {"output":"Strategy design report saved locally", "done":True}
        else:
            raise ValueError("Unknown public design tool")
        return {"output":json.dumps(output)}


class EquityResearchTools:
    def __init__(self, panel:Path, workspace:Path, max_experiments=8):
        self.engine = MarketNeutralBacktest(panel)
        self.manifest = json.loads((panel/"panel.json").read_text())
        self.workspace = workspace
        self.max_experiments = max_experiments
        self.experiments = []
        self.report = None

    def evaluate(self, spec):
        validate_spec(spec)
        fingerprint = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
        if len(self.experiments) >= self.max_experiments:
            raise ValueError("Experiment budget exhausted; report existing results")
        result, daily = self.engine.evaluate(spec)
        result["id"] = fingerprint
        atomic_json(self.workspace/f"experiment-{fingerprint}.json",result)
        daily.to_csv(self.workspace/f"experiment-{fingerprint}-daily.csv",index=False)
        self.experiments.append(result)
        atomic_json(self.workspace/"experiments.json",self.experiments)
        return result

def run_pilot(panel,workspace,model,reasoning_effort):
    from alpha_lab.agent import AgentLoop
    from alpha_lab.context import ContextManager
    if workspace.exists():
        raise ValueError("Choose a new run directory to avoid duplicate campaigns")
    workspace.mkdir(parents=True)
    settings=LLMSettings(model,reasoning_effort,max_calls=16,max_reserved_tokens=500_000,
                         max_output_tokens=4096,max_run_seconds=5400,max_context_bytes=64_000)
    manifest={"started_utc":datetime.now(timezone.utc).isoformat(),"panel":str(panel.resolve()),
              "llm":asdict(settings),"max_experiments":8,"seed":20261013,
              "parameters":{"sdk_retries":0,"agent_retries":0,"concurrency":1,
                            "temperature":"omitted: reasoning model default",
                            "top_p":"omitted: reasoning model default",
                            "store":False,"retention":"Gateway policy unknown; store=false is not a retention guarantee",
                            "tools":[t["name"] for t in TOOLS],"parallel_tool_calls":False,
                            "structured_outputs":"Strict function schemas; text report",
                            "context":"Explicit byte cap; no automatic truncation; small bounded campaign",
                            "summary":"Responses, same selected model/reasoning; shared budget",
                            "failed_stream":"Terminal failure; keep conservative token reservation",
                            "cost_budget":"No dollar estimate: gateway pricing unavailable; call/token caps enforced"},
              "data_flow":"LLM sees only public definitions and generated candidate specs; all market data and results remain local",
              "holdout":"Frozen 2023-2025; no agent access; selection on validation only",
              "authorization":"User authorized unattended LLM research for this 10-hour period; simulations only"}
    atomic_json(workspace/"run.json",manifest)
    load_dotenv(".env")
    provider=bounded_provider(settings,workspace/"llm-usage.json")
    designer=PublicStrategyDesigner(workspace)
    events=[]
    def on_event(event):
        if event.type in {"status","error"}:
            safe=event.to_dict()
            print(json.dumps(safe),flush=True)
            events.append(safe)
    prompt=("You are AlphaLab researching a US equity market-neutral strategy from fundamentals and market data. "
            "Use only the provided strategy-design tools. Inspect research rules, then register 3 to 8 economically "
            "motivated linear combinations. Include value, quality and momentum baselines plus combinations. "
            "Coefficient sign matters: leverage and volatility prefer negative exposure (low_vol is already negative volatility). "
            "Design for stability, realistic trading costs and low market beta. You have no actual data or performance access. "
            "Do not claim audited point-in-time accuracy, borrow feasibility or investment readiness. "
            "Each additional candidate is another trial; acknowledge selection bias and unresolved terminal returns. "
            "Local testing happens after you finish. Call report_to_user with strategy names, exact factors, "
            "economic hypotheses, proposed tests and limitations. Never invent metrics. Do not request user input.")
    agent=AgentLoop(provider,model,ContextManager(provider,model,str(workspace)),on_event,
                    reasoning_effort=reasoning_effort,tools=TOOLS,tool_executor=designer,
                    prompt_builder=lambda *args:prompt,min_report_attempts=1)
    agent.run("Design bounded fundamental/market-data US equity market-neutral candidate strategies now.")
    # No LLM calls below this boundary: private data is evaluated exclusively here.
    local=EquityResearchTools(panel,workspace)
    for spec in designer.candidates:
        local.evaluate(spec)
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                    agent_finished=bool(agent._done and designer.report),
                    experiment_count=len(local.experiments),events=events)
    atomic_json(workspace/"run.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prepare=commands.add_parser("prepare")
    prepare.add_argument("--config",type=Path,default=Path("data/market/sharadar.json"))
    prepare.add_argument("--output",type=Path,required=True)
    pilot=commands.add_parser("run")
    pilot.add_argument("--panel",type=Path,required=True)
    pilot.add_argument("--workspace",type=Path,required=True)
    pilot.add_argument("--model",required=True)
    pilot.add_argument("--reasoning-effort",required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        prepare_panel(args.config,args.output)
    else:
        result=run_pilot(args.panel,args.workspace,args.model,args.reasoning_effort)
        print(json.dumps({"agent_finished":result["agent_finished"],"experiments":result["experiment_count"]}),flush=True)
        if not result["agent_finished"]:
            raise SystemExit(1)


if __name__=="__main__":
    main()
