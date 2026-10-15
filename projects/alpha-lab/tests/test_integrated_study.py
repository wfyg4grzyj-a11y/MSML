"""Integrated portfolio/accounting, bounded Windows worker and recovery gates."""
from pathlib import Path

import pytest
np=pytest.importorskip('numpy')
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.contracts import ResearchError, parse, read_json
from alpha_lab.research.registry import Registry, initialize_registry
from alpha_lab.research.signal_contracts import Family, Hypothesis
from alpha_lab.research.study_contracts import StudyPacket
from alpha_lab.research.study_engine import evaluate_study
from alpha_lab.research.study_inputs import load_study, pin_study
from alpha_lab.research import study_workflow as flow
from tests.test_accounting_ledger import scenario
from tests.test_risk_portfolio import policy


def packet(integrated=True):
    data=scenario()
    for session in data['sessions']:
        session['quotes']={sid:dict(session['quotes']['A']) for sid in ('A','B','C','D')}
    data['securities']={sid:sid for sid in ('A','B','C','D')}
    context={'context_id':'first-context','date':data['sessions'][0]['date'],
             'information_time':data['sessions'][0]['close_utc'],'signal_evidence_sha256':'c'*64,
             'source_policy_id':'reference','risk_binding_id':None,'fill_lag_sessions':1,
             'points':[{'security_id':sid,'score':score,'eligible':True,'issuer_id':sid,'sector_id':'same','beta':beta}
                       for sid,score,beta in zip(['A','B','C','D'],[1.,1.,-1.,-1.],[.7,1.3,.7,1.3])]}
    return {'schema_version':3,'study_id':'integration','hypothesis':'A fixed synthetic neutral score portfolio',
            'scenario':data,'portfolio_policy':policy().model_dump() if integrated else None,
            'risk_mode':'diagonal_reference' if integrated else 'none','variance_per_session':.0001 if integrated else None,
            'risk_contract_sha256':None,'risk_source_policy_id':None,'benchmark_id':'SPY','cash_id':'zero',
            'calendar_id':'XNYS_modeled_sessions','identity_namespace':'synthetic_share_class',
            'max_risk_age_calendar_days':3,'contexts':[context] if integrated else []}


def prepared(tmp_path, value=None):
    source=tmp_path/'packet.json';atomic_json(source,value or packet())
    bundle=tmp_path/'bundle';pin_study(source,bundle)
    spec=flow.resolve_study(bundle,{'family_id':'integration','hypothesis_id':'reference','title':'Reference integration',
                                  'seed':42,'threads':1,'timeout_seconds':45,'max_output_bytes':4000000,'memory_limit_mb':1024})
    specpath=tmp_path/'spec.json';atomic_json(specpath,spec.model_dump())
    ledger=tmp_path/'registry.sqlite';initialize_registry(ledger)
    with Registry(ledger) as db:
        db.register(Family(family_id='integration',premise='Synthetic integration of fixed scores and raw accounting',max_unique_trials=1,max_attempts_per_trial=2),
                    Hypothesis(hypothesis_id='reference',family_id='integration',premise='No-cost neutral portfolio has constant NAV',tests=['Independent cash/share reconciliation','Coverage and failure preservation']),spec)
    return bundle,specpath,ledger


def test_planner_feeds_fixed_share_orders_and_checks_actual_retained_positions(tmp_path):
    value=parse(StudyPacket,packet())
    ledger,port=evaluate_study(value,tmp_path)
    assert ledger['final_nav_observed_usd']==pytest.approx(1000.)
    assert len(ledger['orders'])==4 and all(o['date']=='2020-01-03' for o in ledger['orders'])
    assert [o['filled_delta'] for o in ledger['orders']]==pytest.approx([2.5,2.5,-2.5,-2.5])
    assert port['plans'][0]['status']=='feasible'
    assert all(p['status']=='covered' for p in port['actual_position_risk'])
    assert all(abs(p['exposures']['net'])<1e-6 for p in port['actual_position_risk'])


def test_partial_one_leg_rejects_whole_non_neutral_batch(tmp_path):
    value=packet();value['scenario']['sessions'][1]['quotes']['D']['volume_shares']=1.
    ledger,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    assert all(o['filled_delta']==0 for o in ledger['orders'])
    assert all(o['status']=='rejected_batch_constraints' for o in ledger['orders'])
    assert ledger['final_nav_observed_usd']==1000


def test_future_price_change_preserves_earlier_plan_and_orders(tmp_path):
    first=packet();changed=packet();changed['scenario']['sessions'][2]['quotes']['A']['raw_close_usd']=9999.
    a,pa=evaluate_study(parse(StudyPacket,first),tmp_path)
    b,pb=evaluate_study(parse(StudyPacket,changed),tmp_path)
    assert pa['plans']==pb['plans'] and a['orders']==b['orders']
    assert a['daily'][:2]==b['daily'][:2]
    assert pb['actual_position_risk'][-1]['status']=='constraints_violated'


def test_held_position_never_vanishes_when_later_context_omits_it(tmp_path):
    value=packet();context=dict(value['contexts'][0]);context.update(context_id='second',date='2020-01-06',
                         information_time=value['scenario']['sessions'][2]['close_utc'],points=value['contexts'][0]['points'][:3])
    value['contexts'].append(context)
    ledger,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    assert port['plans'][1]['status']=='insufficient_evidence'
    assert any(h['security_id']=='D' and h['shares']<0 for h in ledger['holdings'] if h['date']=='2020-01-06')
    assert port['actual_position_risk'][-1]['status']=='insufficient_evidence'


def test_pinned_inputs_reject_tampering_and_holdout_dates(tmp_path):
    source=tmp_path/'source.json';atomic_json(source,packet())
    pin_study(source,tmp_path/'bundle')
    (tmp_path/'bundle/packet.json').write_text('{}')
    with pytest.raises(ResearchError):load_study(tmp_path/'bundle')
    value=packet();value['scenario']['sessions'][-1]['date']='2023-01-03'
    with pytest.raises(ResearchError):parse(StudyPacket,value)


def test_actual_windows_worker_and_saved_inspection_are_credential_free(tmp_path,monkeypatch):
    bundle,spec,registry=prepared(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-private');monkeypatch.setenv('HTTPS_PROXY','http://unreachable.invalid')
    result=flow.run_study(bundle,spec,tmp_path/'run',registry)
    assert result['artifacts_verified'] and result['summary']['final_nav_observed_usd']==pytest.approx(1000.)
    assert read_json(tmp_path/'run/runtime.json')['provider_keys_present'] is False
    before={p.name:p.stat().st_mtime_ns for p in (tmp_path/'run').iterdir()}
    assert flow.inspect_study(tmp_path/'run')['result_hash']==result['result_hash']
    assert before=={p.name:p.stat().st_mtime_ns for p in (tmp_path/'run').iterdir()}
    with Registry(registry,readonly=True) as db:
        assert db.snapshot()['exposures'][0]['kind']=='development_raw_share_ledger_published'
    with pytest.raises(ResearchError,match='completed'):flow.run_study(bundle,spec,tmp_path/'duplicate',registry)


def test_zero_exit_without_outputs_and_failed_recovery_do_not_complete(tmp_path,monkeypatch):
    class Empty:
        pid=123
        def poll(self):return 0
        def wait(self):return 0
        def close(self):pass
    bundle,spec,registry=prepared(tmp_path)
    monkeypatch.setattr(flow,'start_process',lambda *a,**k:Empty())
    with pytest.raises(ResearchError,match='publication_failure'):flow.run_study(bundle,spec,tmp_path/'run',registry)
    assert flow.inspect_study(tmp_path/'run')['effective_status']=='failed'
    assert flow.reconcile_study(registry,tmp_path/'run')['effective_status']=='failed'
    with Registry(registry) as db:
        assert len(db.snapshot()['attempts'])==1


def test_complete_file_database_gap_is_reconciled_without_new_attempt(tmp_path,monkeypatch):
    bundle,spec,registry=prepared(tmp_path)
    original=Registry.finish
    def interrupted(self,attempt,**kw):
        if kw.get('complete'):raise OSError('synthetic publication interruption')
        return original(self,attempt,**kw)
    monkeypatch.setattr(Registry,'finish',interrupted)
    with pytest.raises(ResearchError,match='reconciliation'):flow.run_study(bundle,spec,tmp_path/'run',registry)
    assert flow.inspect_study(tmp_path/'run')['artifacts_verified']
    monkeypatch.setattr(Registry,'finish',original)
    assert flow.reconcile_study(registry,tmp_path/'run')['artifacts_verified']
    with Registry(registry,readonly=True) as db:assert len(db.snapshot()['attempts'])==1


@pytest.mark.parametrize('guard',['timeout','memory_limit','output_limit'])
def test_integrated_study_resource_limits_preserve_failure(tmp_path,monkeypatch,guard):
    import os
    from alpha_lab.research.study_contracts import StudySpec
    bundle,spec_path,registry=prepared(tmp_path)
    spec=parse(StudySpec,read_json(spec_path) | {'timeout_seconds':1 if guard=='timeout' else 45,
              'memory_limit_mb':128 if guard=='memory_limit' else 1024,'max_output_bytes':1024 if guard=='output_limit' else 4000000})
    atomic_json(spec_path,spec.model_dump())
    class Waiting:
        pid=os.getpid();stopped=False
        def poll(self):return 1 if self.stopped else None
        def terminate_tree(self):self.stopped=True
        def close(self):pass
    proc=Waiting();monkeypatch.setattr(flow,'start_process',lambda *a,**k:proc)
    if guard=='timeout':
        moments=iter([0.,2.]);monkeypatch.setattr(flow.time,'monotonic',lambda:next(moments))
    if guard=='memory_limit':
        class Root:
            def children(self,recursive):return []
            def is_running(self):return True
            def memory_info(self):return type('Memory',(),{'rss':2*1024**3})()
        monkeypatch.setattr(flow.psutil,'Process',lambda pid:Root())
    with pytest.raises(ResearchError,match=guard):flow.run_study(bundle,spec_path,tmp_path/'run',registry)
    assert proc.stopped and flow.inspect_study(tmp_path/'run')['failure']==guard


def test_cached_risk_still_rejects_stale_use_and_retains_one_loaded_snapshot(tmp_path,monkeypatch):
    from tests.test_risk_portfolio import binding_request
    from alpha_lab.research.risk_adapter import pin_forecast
    from alpha_lab.research import study_engine
    source=tmp_path/'source';source.mkdir()
    atomic_json(source/'forecast.json',{'model':'reference','as_of':'2020-01-02','horizons':[1],
                 'units':'squared_decimal_return','target':'additive_excess_returns','diagnostics':{}})
    np.savez_compressed(source/'forecast.npz',security_ids=np.array(['A','B','C','D']),covariance_1=np.eye(4)*.0001)
    pin_forecast(source,tmp_path/'forecast',binding_request(calendar_id='XNYS_modeled_sessions',cash_id='zero'))
    value=packet();value.update(risk_mode='equity_risk_forecast',variance_per_session=None,
                                risk_contract_sha256='a'*64,risk_source_policy_id='reference')
    value['contexts'][0]['risk_binding_id']='forecast'
    parsed=parse(StudyPacket,value);controller=study_engine.PortfolioController(parsed,tmp_path)
    original=study_engine.load_risk;calls=[]
    monkeypatch.setattr(study_engine,'load_risk',lambda *a,**k:(calls.append(1) or original(*a,**k)))
    controller.risk(parsed.contexts[0],parsed.scenario.sessions[0])
    controller.risk(parsed.contexts[0],parsed.scenario.sessions[1])
    assert len(calls)==1
    with pytest.raises(ResearchError,match='stale'):controller.risk(parsed.contexts[0],parsed.scenario.sessions[2])
