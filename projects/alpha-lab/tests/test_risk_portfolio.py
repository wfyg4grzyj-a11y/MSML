"""Risk format compatibility and independent constrained-portfolio references."""
from pathlib import Path

import pytest
np=pytest.importorskip('numpy')
pytest.importorskip('scipy')
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.contracts import ResearchError, parse
from alpha_lab.research.portfolio import PositionInput, PortfolioPolicy, exposures, project_portfolio
from alpha_lab.research.risk_adapter import RiskUse, diagonal_baseline, load_risk, pin_forecast


def binding_request(**changes):
    return {'contract_sha256':'a'*64,'source_policy_id':'reference','available_at':'2020-01-02T21:00:00+00:00',
            'training_cutoff':'2019-12-31','availability_kind':'synthetic_reference','availability_evidence_sha256':'b'*64,
            'benchmark_id':'SPY','cash_id':'ACT365-zero','calendar_id':'XNYS','identity_namespace':'synthetic_share_class',
            'history_mode':'synthetic_reference','point_in_time_attested':False,'terminal_audit_complete':True,**changes}


def risk_use(**changes):
    return parse(RiskUse,{'decision_time':'2020-01-03T21:00:00+00:00','horizon_sessions':1,
                         'max_age_calendar_days':3,'contract_sha256':'a'*64,'source_policy_id':'reference',
                         'benchmark_id':'SPY','cash_id':'ACT365-zero','calendar_id':'XNYS',
                         'identity_namespace':'synthetic_share_class','expected_security_order':['A','B','C','D'],
                         'use':'synthetic_validation',**changes})


@pytest.fixture
def risk_source(tmp_path):
    source=tmp_path/'source';source.mkdir()
    atomic_json(source/'forecast.json',{'model':'reference','as_of':'2020-01-02','horizons':[1],
                 'units':'squared_decimal_return','target':'additive_excess_returns','diagnostics':{}})
    np.savez_compressed(source/'forecast.npz',security_ids=np.array(['A','B','C','D']),covariance_1=np.eye(4)*.0001)
    return source


def test_exact_official_forecast_format_pins_without_source_mutation(risk_source,tmp_path):
    before={p.name:p.read_bytes() for p in risk_source.iterdir()}
    binding=pin_forecast(risk_source,tmp_path/'pinned',binding_request())
    got,cov=load_risk(tmp_path/'pinned',risk_use())
    assert got==binding and not cov.flags.writeable
    np.testing.assert_array_equal(cov,np.eye(4)*.0001)
    assert before=={p.name:p.read_bytes() for p in risk_source.iterdir()}


@pytest.mark.parametrize('change',[{'expected_security_order':['B','A','C','D']},
    {'expected_security_order':['A','B','C','D','HELD']},{'horizon_sessions':2},
    {'decision_time':'2020-01-01T21:00:00+00:00'},{'decision_time':'2020-02-01T21:00:00+00:00'},
    {'contract_sha256':'0'*64},{'source_policy_id':'different'},{'cash_id':'different'},
    {'benchmark_id':'different'},{'identity_namespace':'sharadar_permaticker_share_class'},{'use':'attested_research'}])
def test_semantic_mismatches_fail_without_scaling_or_reordering(risk_source,tmp_path,change):
    pin_forecast(risk_source,tmp_path/'pinned',binding_request())
    with pytest.raises(ResearchError):load_risk(tmp_path/'pinned',risk_use(**change))


def test_future_training_intraday_availability_and_unattested_real_data_rejected(risk_source,tmp_path):
    for i,change in enumerate([{'training_cutoff':'2020-01-03'},{'available_at':'2020-01-02T00:00:00+00:00'},
                               {'history_mode':'public_information_reconstruction'}]):
        with pytest.raises(ResearchError):pin_forecast(risk_source,tmp_path/str(i),binding_request(**change))


@pytest.mark.parametrize('invalid',['units','target','asymmetric','indefinite','pickle','nonfinite'])
def test_corrupt_or_wrong_risk_inputs_fail(risk_source,tmp_path,invalid):
    from alpha_lab.research.contracts import read_json
    if invalid in ('units','target'):
        meta=read_json(risk_source/'forecast.json');meta[invalid]='wrong';atomic_json(risk_source/'forecast.json',meta)
    else:
        ids=np.array(['A','B','C','D'],dtype=object if invalid=='pickle' else str)
        cov=np.eye(4)*.0001
        if invalid=='asymmetric':cov[0,1]=1.
        if invalid=='indefinite':cov[0,0]=-1.
        if invalid=='nonfinite':cov[0,0]=np.nan
        np.savez_compressed(risk_source/'forecast.npz',security_ids=ids,covariance_1=cov)
    with pytest.raises((ResearchError,ValueError)):pin_forecast(risk_source,tmp_path/'pinned',binding_request())


def policy(**changes):
    return parse(PortfolioPolicy,{'methodology':'score_projection_covariance_penalty_v1',
             'score_units':'unitless_rank_not_expected_return','borrow_recall':'block_increases_report_retained_short',
             'risk_target':'additive_excess_returns','risk_units':'squared_decimal_return','risk_horizon_sessions':1,
             'max_gross':1.,'max_name':.3,'max_issuer_gross':.6,'max_sector_net':.01,'max_turnover':2.,
             'net_tolerance':0.,'beta_tolerance':0.,'covariance_penalty':0.,'numerical_tolerance':.000001,
             'max_iterations':200,'max_securities':128,**changes})


def positions():
    return [PositionInput(security_id=sid,score=score,raw_price_usd=100.,held_shares=0.,eligible=True,tradable=True,
                          borrow_available_shares=1000.,issuer_id=sid,sector_id='same-sector',beta=beta)
            for sid,score,beta in zip(['A','B','C','D'],[1.,1.,-1.,-1.],[.7,1.3,.7,1.3])]


def solve(items=None, **changes):
    items=items or positions()
    return project_portfolio(items,[p.security_id for p in items],np.eye(len(items))*.0001,1000.,policy(**changes))


def test_independent_equal_weight_neutral_projection_reference():
    result=solve()
    assert result['status']=='feasible'
    np.testing.assert_allclose(list(result['target_weights'].values()),[.25,.25,-.25,-.25],atol=1e-6)
    assert abs(result['exposures']['net'])<1e-6 and abs(result['exposures']['beta'])<1e-6
    assert result['exposures']['forecast_variance']==pytest.approx(.000025,abs=1e-9)


def test_issuer_and_sector_caps_zero_or_reduce_targets_without_hidden_relaxation():
    items=positions()
    items=[p.model_copy(update={'issuer_id':'shared' if i in (0,1) else p.issuer_id}) for i,p in enumerate(items)]
    result=solve(items,max_issuer_gross=.2)
    assert result['status']=='feasible' and result['exposures']['issuer_gross']['shared']<=.200001
    assert result['fallback']=='none'


def test_missing_issuer_beta_held_risk_and_valuation_evidence_are_not_zero_filled():
    for field in ('issuer_id','sector_id','beta','raw_price_usd'):
        items=positions();items[0]=items[0].model_copy(update={field:None,'held_shares':1.})
        result=solve(items);assert result['status']=='insufficient_evidence' and result['target_weights'] is None
    items=positions()
    result=project_portfolio(items,['A','B','C','MISSING'],np.eye(4)*.0001,1000.,policy())
    assert result['status']=='insufficient_evidence'


def test_frozen_ineligible_position_is_retained_and_infeasibility_explicit():
    items=positions();items[0]=items[0].model_copy(update={'held_shares':4.,'eligible':False,'tradable':False})
    result=solve(items)
    assert result['status']=='infeasible' and result['target_weights'] is None and result['fallback']=='none'
    items[0]=items[0].model_copy(update={'held_shares':2.})
    result=solve(items)
    assert result['status']=='feasible' and result['target_weights']['A']==pytest.approx(.2)


def test_unknown_short_availability_blocks_increases_but_exposes_retained_short():
    items=[p.model_copy(update={'borrow_available_shares':None}) for p in positions()]
    result=solve(items)
    assert all(w>=-1e-6 for w in result['target_weights'].values())
    items[2]=items[2].model_copy(update={'held_shares':-1.})
    values,violations=exposures(items,np.array([.1,0.,-.1,0.]),np.eye(4)*.0001,1000.,policy())
    assert values['retained_short_without_current_locate']==['C']


def test_zero_turnover_and_reference_size_limits_do_not_trigger_fallback():
    result=solve(max_turnover=0.)
    assert result['status']=='zero_target'
    assert solve(max_securities=2)['reason']=='reference_solver_size_cap'
    with pytest.raises(ResearchError):diagonal_baseline(['A','B'],.0001,21)


def test_missing_scores_are_reduce_only_without_erasing_held_risk():
    items=positions();items[0]=items[0].model_copy(update={'score':None})
    result=solve(items)
    assert result['status'] in ('feasible','zero_target') and abs(result['target_weights']['A'])<1e-6
    items[0]=items[0].model_copy(update={'held_shares':1.})
    result=solve(items)
    assert result['target_weights']['A']<=.1+1e-6


def test_broader_forecast_union_keeps_inactive_zero_positions_and_all_held_names():
    items=positions()+[PositionInput(security_id=sid,score=None,raw_price_usd=None,held_shares=0.,
                   eligible=False,tradable=False,borrow_available_shares=None,issuer_id=None,sector_id=None,beta=None)
                   for sid in ('INACTIVE1','INACTIVE2')]
    ids=[p.security_id for p in items]
    result=project_portfolio(items,ids,np.eye(6)*.0001,1000.,policy(max_securities=4))
    assert result['status']=='feasible' and result['target_weights']['INACTIVE1']==0.
    assert result['inactive_zero_weight_ids']==['INACTIVE1','INACTIVE2']
    items[-1]=items[-1].model_copy(update={'raw_price_usd':100.,'held_shares':1.,'beta':1.,'issuer_id':'extra','sector_id':'same-sector'})
    result=project_portfolio(items,ids,np.eye(6)*.0001,1000.,policy(max_securities=4))
    assert result['reason']=='reference_solver_size_cap' and result['target_weights'] is None


def test_producer_attested_real_format_accepts_audit_completion_with_unresolved_outcomes(risk_source,tmp_path):
    from alpha_lab.research.contracts import read_json
    meta=read_json(risk_source/'forecast.json')
    meta['diagnostics']={'data_provenance':{'point_in_time':True,'terminal_actions_reconciled':True,
             'cap_basis':'total_issuer','industry_scheme':'SIC','return_convention':'simple_total',
             'benchmark_id':'SPY','cash_id':'ACT365-zero','calendar_id':'XNYS','unresolved_economic_outcomes':1},
             'prior_calibration':{'calibration_end':'2019-12-31'},'parameter_selection_end':None}
    atomic_json(risk_source/'forecast.json',meta)
    request=binding_request(history_mode='public_information_reconstruction',point_in_time_attested=True,
                            availability_kind='archived_observation')
    pin_forecast(risk_source,tmp_path/'pinned',request)
    binding,covariance=load_risk(tmp_path/'pinned',risk_use(use='attested_research'))
    assert binding.terminal_audit_complete and covariance.shape==(4,4)
    assert not hasattr(binding,'economic_outcomes_resolved')
