"""Versioned raw-accounting study execution, publication, inspection and recovery."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import time

import psutil

from alpha_lab.local_process import start_process
from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact, safe_file, verify_files
from .contracts import ResearchError, digest, parse, read_json
from .registry import Registry, stamp
from .runtime import child_environment, environment_identity, implementation_identity
from .study_contracts import StudyBundle, StudyPacket, StudyRun, StudySpec
from .study_inputs import load_study

RESULTS={'ledger.json','portfolio.json','runtime.json'}
FILES=RESULTS | {'spec.json','bundle.json','packet.json','report.txt'}


def study_environment():
    return environment_identity() | {'scipy':version('scipy')}


def validate_study_spec(spec,bundle):
    if spec.bundle_id!=bundle.bundle_id:raise ResearchError('Study specification identifies a different bundle')
    if spec.implementation!=implementation_identity():raise ResearchError('Study implementation changed; resolve a new explicit specification')
    if spec.environment!=study_environment():raise ResearchError('Study numerical environment changed; resolve a new specification')


def resolve_study(bundle_path,request):
    bundle,_=load_study(bundle_path)
    if set(request)&{'schema_version','bundle_id','methodology','implementation','environment'}:
        raise ResearchError('Study request contains reserved resolved fields')
    return parse(StudySpec,request | {'schema_version':3,'bundle_id':bundle.bundle_id,'methodology':'raw_share_cash_action_ledger_v1',
                                     'implementation':implementation_identity(),'environment':study_environment()})


def validate_results(directory,spec,bundle,packet):
    ledger=read_json(safe_file(directory,'ledger.json'))
    portfolio=read_json(safe_file(directory,'portfolio.json'))
    runtime=read_json(safe_file(directory,'runtime.json'))
    if (ledger.get('methodology')!=spec.methodology or ledger.get('investment_ready') is not False
            or ledger.get('point_in_time_attested') is not packet.scenario.point_in_time_attested
            or ledger.get('terminal_audit_complete') is not packet.scenario.terminal_audit_complete
            or ledger.get('history_mode')!=packet.scenario.history_mode or ledger.get('llm_calls')!=0
            or [d.get('date') for d in ledger.get('daily',[])]!=[s.date for s in packet.scenario.sessions]
            or portfolio.get('investment_ready') is not False or portfolio.get('selection')!='none'
            or portfolio.get('risk_mode')!=packet.risk_mode):
        raise ResearchError('Study result identity, readiness or complete date coverage failed')
    required={'date','cash_usd','receivables_net_usd','nav_estimate_usd','known_components_nav_usd','nav_observed_usd',
              'return_observed','valuation_resolved','cash_interest_usd','funding_usd','borrow_usd','action_pnl_usd',
              'raw_mark_pnl_usd','trade_cost_usd','reconciliation_residual_usd','missing_marks','unresolved_terminal',
              'unpriced_cost_count','constraint_violations'}
    previous=None
    for day in ledger['daily']:
        if (set(day)!=required or abs(day['reconciliation_residual_usd'])>packet.scenario.policy.tolerance_usd
                or (day['nav_observed_usd'] is not None)!=day['valuation_resolved']
                or (day['return_observed'] is not None and (previous is None or not previous['valuation_resolved']))):
            raise ResearchError('Ledger valuation/reconciliation evidence is inconsistent')
        if previous:
            change=day['known_components_nav_usd']-previous['known_components_nav_usd']
            explained=day['cash_interest_usd']-day['funding_usd']-day['borrow_usd']+day['action_pnl_usd']+day['raw_mark_pnl_usd']-day['trade_cost_usd']
            if abs(change-explained)>packet.scenario.policy.tolerance_usd:raise ResearchError('Published ledger NAV does not reconcile')
        previous=day
    if runtime!={'environment':spec.environment,'implementation':spec.implementation,'threads':spec.threads,'seed':spec.seed,
                  'provider_keys_present':False,'proxy_present':False,
                  'isolation':'credential_filtered_process_not_filesystem_or_network_sandbox'}:
        raise ResearchError('Study worker runtime does not match its requested boundary')
    if len(portfolio.get('plans',[]))!=len(packet.contexts):raise ResearchError('Portfolio decisions are missing')
    return ledger,portfolio


def study_report(spec,bundle,packet,ledger,portfolio):
    totals={key:sum(day[key] for day in ledger['daily']) for key in
            ('cash_interest_usd','funding_usd','borrow_usd','action_pnl_usd','raw_mark_pnl_usd','trade_cost_usd')}
    from .contracts import canonical
    target_detail=''
    if packet.portfolio_policy and packet.portfolio_policy.target_limits is not None:
        target_detail=('Construction-only target limits: '+canonical(packet.portfolio_policy.target_limits.model_dump())+'\n'
                       'Fill admission and retained-position monitoring use unchanged hard limits; no buffered-to-hard fallback.\n')
    return ('US equity integrated research report — raw-share accounting v1\n'
            f'Hypothesis: {packet.hypothesis}\nFamily/hypothesis: {spec.family_id}/{spec.hypothesis_id}\n'
            f'Trial: {spec.semantic_id()}\nBundle: {bundle.bundle_id}\nDevelopment sessions: {bundle.start}..{bundle.end}\n'
            'Scientific disposition: PROVISIONAL RESEARCH; engineering completion is not strategy acceptance.\n'
            'Raw unadjusted USD prices and explicit actions; adjusted-return labels are not credited to the ledger.\n'
            'Decision close -> fixed shares -> later-session close fill; whole batches reject infeasible partial legs.\n'
            'Day orders expire; later decisions cancel/replace pending same-security targets.\n'
            'Prior-close shares determine ex-session entitlement; payment/delivery changes signed claims into cash/shares.\n'
            'T0 research settlement; short proceeds segregated without interest/rebate. No broker settlement is modeled.\n'
            f'Accounting policies: {canonical(packet.scenario.policy.model_dump())}\n'
            f'{target_detail}'
            f'Initial NAV: {ledger["initial_nav_usd"]}; observed final NAV: {ledger["final_nav_observed_usd"]}; estimated final NAV: {ledger["final_nav_estimate_usd"]}\n'
            f'Components USD: {canonical(totals)}\n'
            f'Unresolved valuation sessions: {sum(not d["valuation_resolved"] for d in ledger["daily"])}\n'
            f'Unpriced costs/rights: {len(ledger["unpriced_costs"])}; outstanding signed claims: {len(ledger["claims_outstanding"])}\n'
            f'Orders/fills: {len(ledger["orders"])}/{sum(bool(o["filled_delta"]) for o in ledger["orders"])}\n'
            f'Risk: {packet.risk_mode}; unitless scores are not expected returns. Covariance stays additive excess-return variance.\n'
            f'Portfolio plans: {len(portfolio["plans"])}; infeasible/missing: {sum(p["status"] not in ("feasible","zero_target") for p in portfolio["plans"])}\n'
            f'Actual risk missing/violated sessions: {sum(p["status"]!="covered" for p in portfolio["actual_position_risk"])}\n'
            f'Source history: {packet.scenario.history_mode}; PIT attestation: {packet.scenario.point_in_time_attested}; terminal audit: {packet.scenario.terminal_audit_complete}\n'
            'Terminal audit completion does not establish every economic outcome. Stale marks are provisional, never executable exits.\n'
            'Missing marks/costs suppress observed returns. No implicit liquidation, zero-cost truth or optimized relaxation.\n'
            'Read ledger.json for every balance/action/order; portfolio.json for targets and actual retained-position checks.\n'
            'Risk baseline/borrow assumptions and external evidence remain separate from investment readiness.\n'
            'No original holdout evaluation, LLM/provider calls, automatic selection or real trades.\n'
            'Owned process/filtered credentials; wall/output and sampled RSS guards; no filesystem/network sandbox.\n')


def run_study(bundle_path,spec_path,destination,registry_path,*,retry_of=None):
    spec=parse(StudySpec,read_json(spec_path))
    bundle_path,destination=Path(bundle_path).resolve(),Path(destination).resolve()
    if destination==bundle_path or destination.is_relative_to(bundle_path) or bundle_path.is_relative_to(destination):
        raise ResearchError('Keep study outputs outside immutable inputs')
    bundle,packet=load_study(bundle_path);validate_study_spec(spec,bundle)
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/'spec.json',spec.model_dump());atomic_json(destination/'bundle.json',bundle.model_dump())
    atomic_json(destination/'packet.json',packet.model_dump())
    (destination/'INCOMPLETE').write_text('Require validated COMPLETE.json and reconciled registry; no automatic retries\n')
    with Registry(registry_path) as registry:
        attempt=registry.begin(spec,destination,retry_of,access_bounds=(bundle.start,bundle.end))
        record=StudyRun(schema_version=3,attempt_id=attempt,trial_id=spec.semantic_id(),spec_sha256=digest(spec.model_dump()),
                        bundle_id=bundle.bundle_id,hypothesis_id=spec.hypothesis_id,status='running',stage='validating',
                        started_at=stamp(),updated_at=stamp(),finished_at=None,failure=None,peak_rss_mb=0.,worker_pid=None,
                        retry_of=retry_of,artifacts={},llm_calls=0,scientific_disposition='provisional_raw_share_accounting_research_not_accepted')
        proc,failure,published=None,'input_validation',False
        try:
            atomic_json(destination/'run.json',record.model_dump())
            work=destination/'worker-output';work.mkdir()
            failure='execution_failure';start=time.monotonic()
            with (destination/'worker.out').open('w',encoding='utf-8') as log:
                proc=start_process([sys.executable,'-s','-P','-m','alpha_lab.research.study_worker','--bundle',str(bundle_path),
                                    '--spec',str(destination/'spec.json'),'--output',str(work)],cwd=str(destination),
                                   env=child_environment(spec),stdout=log,stderr=subprocess.STDOUT,text=True)
                record=record.model_copy(update={'stage':'evaluating','worker_pid':proc.pid})
                while proc.poll() is None:
                    try:
                        root=psutil.Process(proc.pid)
                        rss=sum(p.memory_info().rss for p in [root]+root.children(recursive=True) if p.is_running())/1024**2
                    except psutil.Error:rss=record.peak_rss_mb
                    record=record.model_copy(update={'peak_rss_mb':max(float(rss),record.peak_rss_mb),'updated_at':stamp()})
                    atomic_json(destination/'run.json',record.model_dump())
                    size=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
                    failure_next=('timeout' if time.monotonic()-start>spec.timeout_seconds else
                                  'output_limit' if size>spec.max_output_bytes else 'memory_limit' if rss>spec.memory_limit_mb else None)
                    if failure_next:failure=failure_next;raise ResearchError('Bounded study guard reached')
                    time.sleep(.25)
                if proc.wait()!=0:raise ResearchError('Study worker failed')
                if time.monotonic()-start>spec.timeout_seconds:failure='timeout';raise ResearchError('Study exceeded deadline')
            failure='publication_failure'
            if {p.name for p in work.iterdir()}!=RESULTS:raise ResearchError('Study worker omitted required outputs')
            ledger,portfolio=validate_results(work,spec,bundle,packet)
            validate_study_spec(spec,load_study(bundle_path)[0])
            record=record.model_copy(update={'stage':'publishing','updated_at':stamp()})
            atomic_json(destination/'run.json',record.model_dump())
            for name in RESULTS:(work/name).replace(destination/name)
            work.rmdir()
            (destination/'report.txt').write_text(study_report(spec,bundle,packet,ledger,portfolio),encoding='utf-8')
            record=parse(StudyRun,record.model_dump() | {'status':'completed','stage':'finished','updated_at':stamp(),'finished_at':stamp(),
                                                       'artifacts':{n:artifact(destination/n) for n in sorted(FILES)}})
            atomic_json(destination/'run.json',record.model_dump())
            atomic_json(destination/'COMPLETE.json',{'kind':'raw_share_study_v3','trial_id':spec.semantic_id(),
                        'files':{n:artifact(destination/n) for n in sorted(FILES | {'run.json'})}})
            if sum(p.stat().st_size for p in destination.iterdir() if p.is_file())>spec.max_output_bytes:
                failure='output_limit';raise ResearchError('Final study output exceeds cap')
            evidence=inspect_study(destination);published=True
            (destination/'INCOMPLETE').unlink()
            registry.finish(attempt,complete=evidence)
            return evidence
        except BaseException as error:
            if proc is not None and proc.poll() is None:proc.terminate_tree()
            if published:raise ResearchError('Study files completed; explicit registry reconciliation required') from None
            failure='interrupted' if isinstance(error,KeyboardInterrupt) else failure
            record=parse(StudyRun,record.model_dump() | {'status':'failed','stage':'finished','updated_at':stamp(),'finished_at':stamp(),'failure':failure})
            atomic_json(destination/'run.json',record.model_dump());(destination/'COMPLETE.json').unlink(missing_ok=True)
            registry.finish(attempt,failure=failure)
            raise ResearchError('Study attempt failed: '+failure) from None
        finally:
            if proc is not None:proc.close()


def inspect_study(directory):
    directory=Path(directory).resolve()
    record=parse(StudyRun,read_json(safe_file(directory,'run.json')))
    spec=parse(StudySpec,read_json(safe_file(directory,'spec.json')))
    if (record.spec_sha256!=digest(spec.model_dump()) or record.trial_id!=spec.semantic_id() or record.bundle_id!=spec.bundle_id
            or record.hypothesis_id!=spec.hypothesis_id):raise ResearchError('Saved study identity mismatch')
    packet=parse(StudyPacket,read_json(safe_file(directory,'packet.json')))
    base={'attempt_id':record.attempt_id,'trial_id':record.trial_id,'hypothesis_id':spec.hypothesis_id,
          'directory':str(directory),'start':packet.scenario.sessions[0].date,'end':packet.scenario.sessions[-1].date,
          'result_hash':None,'failure':record.failure,'evidence_kind':'development_raw_share_ledger_published'}
    if not (directory/'COMPLETE.json').exists():
        return base | {'effective_status':'failed' if record.status=='failed' else 'unfinished_requires_reconciliation','artifacts_verified':False}
    marker=read_json(safe_file(directory,'COMPLETE.json'))
    if (set(marker)!= {'kind','trial_id','files'} or marker['kind']!='raw_share_study_v3' or marker['trial_id']!=record.trial_id
            or set(marker['files'])!=FILES | {'run.json'} or record.status!='completed' or set(record.artifacts)!=FILES):
        raise ResearchError('Invalid study publication marker')
    verify_files(directory,marker['files']);verify_files(directory,record.artifacts)
    bundle=parse(StudyBundle,read_json(directory/'bundle.json'))
    if bundle.identity()!=bundle.bundle_id or bundle.bundle_id!=spec.bundle_id or bundle.packet_sha256!=artifact(directory/'packet.json')['sha256']:
        raise ResearchError('Study bundle/packet identity mismatch')
    ledger,portfolio=validate_results(directory,spec,bundle,packet)
    return base | {'effective_status':'completed','artifacts_verified':True,'result_hash':artifact(directory/'COMPLETE.json')['sha256'],
                   'report':str(directory/'report.txt'),'methodology':spec.methodology,'investment_ready':False,
                   'summary':{'initial_nav_usd':ledger['initial_nav_usd'],'final_nav_observed_usd':ledger['final_nav_observed_usd'],
                              'unresolved_sessions':sum(not d['valuation_resolved'] for d in ledger['daily']),
                              'portfolio_plans':len(portfolio['plans']),'unpriced_costs':len(ledger['unpriced_costs'])}}


def reconcile_study(registry_path,directory):
    evidence=inspect_study(directory)
    with Registry(registry_path) as registry:
        registry.verify_attempt(evidence)
        if evidence['artifacts_verified']:registry.finish(evidence['attempt_id'],complete=evidence)
        elif evidence['effective_status']=='failed':registry.finish(evidence['attempt_id'],failure=evidence['failure'])
        else:raise ResearchError('Study outcome remains unresolved; do not reset or submit a duplicate')
    return evidence
