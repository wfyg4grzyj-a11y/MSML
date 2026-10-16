"""Explicit P2-score -> raw-ledger packet bridge, without performance selection."""
from pathlib import Path

from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import load_bundle
from .contracts import ResearchError, file_hash, parse, read_json
from .features import read_parquet
from .signal_contracts import SignalSpec
from .signal_workflow import inspect_signal
from .study_contracts import StudyPacket


def packet_from_signal(signal_run,development_bundle,template_path,output):
    signal_run,development_bundle=Path(signal_run),Path(development_bundle)
    evidence=inspect_signal(signal_run)
    if not evidence['artifacts_verified']:raise ResearchError('Require a completed verified signal study')
    original=parse(SignalSpec,read_json(signal_run/'spec.json'))
    bundle=load_bundle(development_bundle)
    if original.bundle_id!=bundle.bundle_id:raise ResearchError('Signal source and development bundle differ')
    template=read_json(template_path)
    if not isinstance(template,dict) or template.get('contexts')!=[] or template.get('risk_mode')!='diagonal_reference':
        raise ResearchError('Bridge requires an empty-context template with an explicitly declared diagonal reference baseline')
    scenario=template.get('scenario',{})
    if scenario.get('history_mode')=='synthetic_reference' or scenario.get('point_in_time_attested') is not False:
        raise ResearchError('Legacy P2 scores require explicit provisional reconstructed history, not synthetic/PIT promotion')
    if template.get('identity_namespace')!='sharadar_permaticker_share_class':
        raise ResearchError('P2 bridge requires the pinned share-class identity namespace')
    if template.get('portfolio_policy',{}).get('max_issuer_gross') is not None or template.get('portfolio_policy',{}).get('max_sector_net') is not None:
        raise ResearchError('P2 does not supply confirmed issuer/sector links; provide evidenced contexts for those constraints')
    selected=list(scenario.get('securities',{}))
    if not 2<=len(selected)<=512:raise ResearchError('Explicitly choose a bounded cohort before bridging scores')
    identities={str(s['security_id']):s['ticker'] for s in read_json(development_bundle/'identities.json')['securities']}
    if not set(selected)<=identities.keys():raise ResearchError('Raw accounting cohort includes unmatched share-class identities')
    prices=read_parquet(development_bundle/'features.parquet')[['date','ticker','beta']]
    scores=read_parquet(signal_run/'predictions.parquet')
    scores=scores.loc[scores.role.eq('validation'),['date','security_id','score','source_member']].copy()
    if scores.duplicated(['date','security_id']).any():raise ResearchError('Signal validation decisions overlap')
    sessions={s['date']:s for s in scenario.get('sessions',[])}
    dates=sorted(set(scores.date.dt.strftime('%Y-%m-%d')) & sessions.keys())
    contexts=[]
    for date in dates:
        current=scores.loc[scores.date.eq(date)].set_index('security_id')
        betas=prices.loc[prices.date.eq(date)].set_index('ticker').beta
        points=[]
        for sid in selected:
            key=int(sid);row=current.loc[key] if key in current.index else None
            score=float(row.score) if row is not None and row.score==row.score else None
            beta=betas.get(identities[sid])
            beta=float(beta) if beta is not None and beta==beta else None
            points.append({'security_id':sid,'score':score,'eligible':bool(row.source_member) if row is not None else False,
                           'issuer_id':None,'sector_id':None,'beta':beta})
        contexts.append({'context_id':'signal-'+date,'date':date,'information_time':sessions[date]['close_utc'],
                         'signal_evidence_sha256':evidence['result_hash'],'source_policy_id':'legacy-P2-provisional-share-class',
                         'risk_binding_id':None,'fill_lag_sessions':original.label.execution_lag_sessions,'points':points})
    if not contexts:raise ResearchError('Explicit raw-session window has no validation signal decisions')
    packet=parse(StudyPacket,template | {'contexts':contexts})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as target:
        from .contracts import canonical
        target.write(canonical(packet.model_dump())+'\n')
    return {'packet':str(output),'source_signal_manifest_sha256':file_hash(signal_run/'COMPLETE.json'),
            'contexts':len(contexts),'cohort_predeclared_in_template':True,'future_outcome_filter':False,
            'point_in_time_attested':False,'investment_ready':False}
