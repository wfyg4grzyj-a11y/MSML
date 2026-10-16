"""Registered owned CPU workflow for the large-cap factor platform."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Annotated,Literal
import argparse,json,subprocess,sys,time
import psutil
from pydantic import Field,model_validator
from alpha_lab.local_process import start_process
from alpha_lab.market_data.sharadar_store import atomic_json
from .contracts import StrictModel,ResearchError,digest,parse,read_json,file_hash,Hash,Identifier
from .bundles import artifact,verify_files
from .factor_inputs import load_inputs,prepare_inputs
from .registry import Registry,stamp
from .runtime import implementation_identity,environment_identity,child_environment


def numerical_environment():
    return environment_identity()|{'scipy':version('scipy'),'clarabel':version('clarabel')}


class PlatformSpec(StrictModel):
    schema_version:Literal[5]
    family_id:Identifier
    hypothesis_id:Identifier
    title:str
    bundle_id:Hash
    fundamental_sha256:Hash
    catalog_sha256:Hash
    policy:dict
    implementation:dict[str,Hash]
    environment:dict[str,str]
    seed:Annotated[int,Field(ge=0,lt=2**32)]
    threads:Annotated[int,Field(ge=1,le=8)]
    timeout_seconds:Annotated[int,Field(ge=1,le=7200)]
    memory_limit_mb:Annotated[int,Field(ge=128,le=16000)]
    max_output_bytes:Annotated[int,Field(ge=1024,le=3000000000)]

    @model_validator(mode='after')
    def supported(self):
        p=self.policy
        if not '1999-01-01'<=p['requested_window']['start']<p['requested_window']['end']<='2022-12-31':
            raise ValueError('Only explicit pre-2023 extended research dates')
        fixed={'gross':1.,'name':.05,'turnover':2.,'abs_net':.01,'abs_beta':.01,'tolerance':1e-6,
               'industry_abs_net':.005,'industry_abs_beta':.005,'annual_vol':.1}
        if p['hard_limits']!=fixed:raise ValueError('This fixed study does not silently change hard accounting/control bounds')
        for key,val in {'starting_nav_usd':1e6,'one_way_commission_bps':1.,'one_way_slippage_bps':5.,
                        'annual_hypothetical_borrow':.03,'day_count':'ACT365F','cash_credit':0.,'cash_debit':.05,'participation':.01}.items():
            if p['execution'].get(key)!=val:raise ValueError('Unsupported change to the fixed diagnostic accounting convention')
        if p['risk']['lookback_sessions']!=252 or p['risk']['minimum_observations']!=126 or p['risk']['annualization']!=252:
            raise ValueError('Unsupported risk window/unit convention')
        if p['risk']['diagonal_shrinkage']!=.3 or p['risk']['variance_floor']!=1e-8:
            raise ValueError('Unsupported risk shrinkage/variance floor')
        if not 0<p['expected_returns']['risk_aversion_gamma']<=100 or not 1<=p['expected_returns']['horizon_sessions']<=63:
            raise ValueError('Expected return scale/horizon must be intentional')
        opt=p['optimization']
        if not 0<opt['inner_gross']<=1 or not 0<opt['inner_name']<=.05 or not 0<opt['inner_turnover']<=2 or not 0<opt['annual_vol_cap']<=.1:
            raise ValueError('Construction cannot loosen hard bounds')
        return self

    def computation(self):
        return self.model_dump(exclude={'family_id','hypothesis_id','title','timeout_seconds','memory_limit_mb','max_output_bytes'})

    def semantic_id(self):return digest(self.computation())
    def comparison_id(self):return digest(self.computation())


def resolve_platform(inputs,fundamentals,policy,request):
    bundle=load_inputs(inputs)
    if (bundle['start'],bundle['end'])!=(policy['requested_window']['start'],policy['requested_window']['end']):
        raise ResearchError('Study window differs from immutable input scope')
    if 'universe_policy' in policy and bundle.get('universe_policy')!=policy['universe_policy']:
        raise ResearchError('Run policy differs from the pinned shared universe')
    return parse(PlatformSpec,request|{'schema_version':5,'bundle_id':bundle['bundle_id'],
        'fundamental_sha256':file_hash(fundamentals),'catalog_sha256':file_hash(Path(__file__).with_name('jkp_catalog.json')),
        'policy':policy,'implementation':implementation_identity(),'environment':numerical_environment()})


def validate_spec(spec,inputs,fundamentals):
    if (spec.implementation!=implementation_identity() or spec.environment!=numerical_environment()
            or spec.catalog_sha256!=file_hash(Path(__file__).with_name('jkp_catalog.json'))
            or spec.fundamental_sha256!=file_hash(fundamentals) or spec.bundle_id!=load_inputs(inputs)['bundle_id']):
        raise ResearchError('Resolved platform source/input/runtime changed')


def platform_report(outcome):
    return ('US large-cap JKP factor platform: provisional long-history diagnostic\n'
      f"Window: {outcome['start']}..{outcome['end']}; {outcome['sessions']} sessions / {outcome['months']} formations\n"
      f"Catalog153: {outcome['implemented_provider_reconstructions']} implemented provider reconstructions; "
      f"{outcome['blocked_factors']} blocked. Nonzero at least once: {outcome['factors_nonzero_at_least_once']}\n"
      'No exact JKP/WRDS replication or raw-share/broker execution claim. Full factor status: coverage.json\n'
      f"Account final observed NAV: {outcome['final_nav_observed_usd']}; provisional estimate: {outcome['final_nav_estimate_usd']}\n"
      f"Complete-path return: {outcome['complete_path_return']}; unresolved sessions: {outcome['unresolved_sessions']}\n"
      f"Actual simulated nonzero fills: {outcome['nonzero_fills']}; daily interventions: {outcome['daily_controls']}\n"
      f"Trading costs: {outcome['trade_cost_usd']}; hypothetical borrow: {outcome['borrow_usd']}\n"
      'Same-calendar S&P500 comparison uses explicitly identified adjusted SPY ETF total-return proxy.\n'
      'Industry rank/standardization and joint industry dollar/beta neutrality; dated causal risk; '
      'ideal10% volatility factor/theme/model layers; equal factors/themes. Ideal holdings are not executable.\n'
      'Gamma5 unconstrained reverse-optimization prior,21-session horizon. Sparse conic implementation optimizer '
      'net of costs/borrow, with unchanged gross/name/net/beta/turnover hard caps, '
      'additional per-industry0.5% caps and10% forecast risk. No implicit relaxation.\n'
      'Next-close fixed total-return units,1% dollar-volume participation,6bp one-way trading cost, '
      'hypothetical3% borrow, ACT365F, zero credit/5% debit. Close-observed reductions act next session.\n'
      'Missing held marks retain positions and censor observed NAV/new sizing. '
      'No stale executable exits, missing-return zero fill or fabricated corporate-action proceeds.\n'
      'Look through month-*.npz for raw signals/factor/model/forecast/implementation weights. '
      'Forward ideal-model/factor intervals are separate diagnostics, with incomplete outcomes retained.\n'
      'Limitations:\n'+'\n'.join('- '+x for x in outcome['known_limitations'])+'\n'
      'No LLM/data-provider calls,2023-2025 holdout evaluation, actual orders or sibling-project edits.\n')


def inspect_platform(directory):
    directory=Path(directory).resolve();record=read_json(directory/'run.json')
    spec=parse(PlatformSpec,read_json(directory/'spec.json'))
    base={'directory':str(directory),'trial_id':spec.semantic_id(),'attempt_id':record['attempt_id'],
          'hypothesis_id':spec.hypothesis_id,'start':spec.policy['requested_window']['start'],'end':spec.policy['requested_window']['end'],
          'effective_status':record['status'],'artifacts_verified':False,'result_hash':None,
          'evidence_kind':'development_signal_diagnostics_published','investment_ready':False}
    if not (directory/'COMPLETE.json').exists():return base
    marker=read_json(directory/'COMPLETE.json')
    if marker.get('kind')!='large_cap_factor_study_v1' or marker.get('trial_id')!=spec.semantic_id() or record['status']!='completed':
        raise ResearchError('Invalid factor study completion')
    verify_files(directory,marker['files']);verify_files(directory,record['artifacts'])
    outcome=read_json(directory/'outcome.json')
    coverage=read_json(directory/'coverage.json')
    if len(coverage)!=153 or len({x['jkp_id'] for x in coverage})!=153:raise ResearchError('Factor coverage inventory is incomplete')
    if outcome['end']>='2023-01-01' or outcome['investment_ready'] or outcome['holdout_evaluations']:
        raise ResearchError('Unsupported historical/acceptance boundary')
    return base|{'artifacts_verified':True,'result_hash':file_hash(directory/'COMPLETE.json'),
                 'report':str(directory/'report.txt'),'summary':outcome}


def run_platform(inputs,fundamentals,spec_path,destination,registry_path,*,retry_of=None):
    spec=parse(PlatformSpec,read_json(spec_path));inputs,fundamentals=Path(inputs).resolve(),Path(fundamentals).resolve()
    validate_spec(spec,inputs,fundamentals)
    destination=Path(destination).resolve()
    if destination==inputs or destination.is_relative_to(inputs) or inputs.is_relative_to(destination):
        raise ResearchError('Outputs must be separate from immutable inputs')
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/'spec.json',spec.model_dump());atomic_json(destination/'bundle.json',load_inputs(inputs))
    atomic_json(destination/'fundamentals-binding.json',{'path':str(fundamentals),'sha256':file_hash(fundamentals)})
    with Registry(registry_path) as registry:
        attempt=registry.begin(spec,destination,retry_of,access_bounds=(spec.policy['requested_window']['start'],spec.policy['requested_window']['end']))
        record={'schema_version':5,'kind':'large_cap_factor_study','attempt_id':attempt,'trial_id':spec.semantic_id(),
            'hypothesis_id':spec.hypothesis_id,'status':'running','stage':'evaluating','started_at':stamp(),'updated_at':stamp(),
            'finished_at':None,'failure':None,'worker_pid':None,'peak_rss_mb':0.,'artifacts':{},'llm_calls':0,
            'scientific_disposition':'provisional_total_return_unit_factor_research_not_verified_execution'}
        atomic_json(destination/'run.json',record);work=destination/'worker-output';work.mkdir()
        proc=None;failure='execution_failure';published=False;started=time.monotonic()
        try:
            with (destination/'worker.out').open('w',encoding='utf-8') as log:
                proc=start_process([sys.executable,'-s','-P','-m','alpha_lab.research.factor_workflow','worker',
                    '--inputs',str(inputs),'--fundamentals',str(fundamentals),'--spec',str(destination/'spec.json'),
                    '--output',str(work)],cwd=str(destination),env=child_environment(spec),stdout=log,stderr=subprocess.STDOUT,text=True)
                record['worker_pid']=proc.pid
                while proc.poll() is None:
                    try:
                        root=psutil.Process(proc.pid)
                        rss=sum(p.memory_info().rss for p in [root]+root.children(recursive=True) if p.is_running())/2**20
                    except psutil.Error:rss=record['peak_rss_mb']
                    record['peak_rss_mb']=max(record['peak_rss_mb'],rss);record['updated_at']=stamp()
                    atomic_json(destination/'run.json',record)
                    size=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
                    failure=('timeout' if time.monotonic()-started>spec.timeout_seconds else
                             'memory_limit' if rss>spec.memory_limit_mb else 'output_limit' if size>spec.max_output_bytes else 'execution_failure')
                    if failure!='execution_failure':raise ResearchError('Factor resource guard reached')
                    time.sleep(1.)
                if proc.wait()!=0:raise ResearchError('Factor worker failed; preserved worker.out contains diagnostics')
            failure='publication_failure'
            validate_spec(spec,inputs,fundamentals)
            outcome=read_json(work/'outcome.json')
            for p in work.iterdir():p.replace(destination/p.name)
            work.rmdir()
            (destination/'report.txt').write_text(platform_report(outcome),encoding='utf-8')
            files={p.name:artifact(p) for p in destination.iterdir() if p.is_file() and p.name!='run.json'}
            record.update(status='completed',stage='finished',updated_at=stamp(),finished_at=stamp(),artifacts=files)
            atomic_json(destination/'run.json',record)
            atomic_json(destination/'COMPLETE.json',{'kind':'large_cap_factor_study_v1','trial_id':spec.semantic_id(),
                        'files':files|{'run.json':artifact(destination/'run.json')}})
            evidence=inspect_platform(destination);published=True;registry.finish(attempt,complete=evidence)
            return evidence
        except BaseException as error:
            if proc is not None and proc.poll() is None:proc.terminate_tree()
            if published:raise ResearchError('Published factor study requires registry reconciliation') from None
            record.update(status='failed',stage='finished',updated_at=stamp(),finished_at=stamp(),failure=failure)
            (destination/'COMPLETE.json').unlink(missing_ok=True)
            atomic_json(destination/'run.json',record);registry.finish(attempt,failure=failure)
            raise ResearchError('Factor study failed: '+failure) from error
        finally:
            if proc is not None:proc.close()


def reconcile_platform(registry_path,directory):
    evidence=inspect_platform(directory)
    with Registry(registry_path) as registry:
        registry.verify_attempt(evidence)
        if evidence['artifacts_verified']:registry.finish(evidence['attempt_id'],complete=evidence)
        elif evidence['effective_status']=='failed':
            registry.finish(evidence['attempt_id'],failure=read_json(Path(directory)/'run.json')['failure'])
        else:raise ResearchError('Unfinished factor worker requires attention, not a duplicate run')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--sources',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    for command in ('worker','run'):
        p=commands.add_parser(command)
        for flag in ('inputs','fundamentals','spec','output'):p.add_argument('--'+flag,type=Path,required=True)
        if command=='run':p.add_argument('--registry',type=Path,required=True)
    check=commands.add_parser('inspect');check.add_argument('directory',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare_inputs(args.sources,args.output)
    elif args.command=='inspect':result=inspect_platform(args.directory)
    elif args.command=='run':result=run_platform(args.inputs,args.fundamentals,args.spec,args.output,args.registry)
    else:
        from .factor_study import evaluate_platform
        spec=parse(PlatformSpec,read_json(args.spec));validate_spec(spec,args.inputs,args.fundamentals)
        result=evaluate_platform(args.inputs,args.fundamentals,args.output,args.spec)
        import os
        atomic_json(args.output/'runtime.json',{'implementation':spec.implementation,'environment':spec.environment,
            'seed':spec.seed,'threads':spec.threads,'provider_keys_present':any(k.endswith('_API_KEY') for k in os.environ),
            'proxy_present':any(k.lower().endswith('_proxy') for k in os.environ),
            'isolation':'owned_credential_filtered_process_not_filesystem_or_network_sandbox'})
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
