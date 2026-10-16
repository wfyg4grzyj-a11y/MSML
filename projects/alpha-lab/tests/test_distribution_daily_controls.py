"""Signed spin-off arithmetic and causal, reduction-only risk intervention."""
from copy import deepcopy

import numpy as np
import pytest

from alpha_lab.research.contracts import ResearchError, parse
from alpha_lab.research.ledger_contracts import LedgerScenario
from alpha_lab.research.study_contracts import DailyRiskControls, StudyPacket
from alpha_lab.research.daily_controls import reduction_targets
from tests.test_risk_portfolio import policy, positions
from alpha_lab.research.study_engine import evaluate_study
from tests.test_accounting_ledger import action, decision, run, scenario
from tests.test_integrated_study import packet

CONVENTION='signed_fractional_prior_close_same_session_delivery_research'


def controls(value):
    value['portfolio_policy'].update(max_name=.3,net_tolerance=.01,beta_tolerance=.01,
                                    max_issuer_gross=None,max_sector_net=None)
    value['daily_risk_controls']={'methodology':'close_breach_uniform_reduction_v1','fill_lag_sessions':1,
        'max_gross':.8,'max_name':.2,'net_tolerance':.005,'beta_tolerance':.005,'exit_ineligible_holdings':True}
    return value


@pytest.mark.parametrize('quantity,cash',[(100.,0.),(-100.,20000.)])
def test_signed_fractional_distribution_retains_parent_and_ex_close_trade_cannot_erase_entitlement(quantity,cash):
    value=scenario((100.,99.,99.),initial_cash=cash,initial_shares={'A':quantity})
    for s in value['sessions']:s['quotes']['B']={'raw_close_usd':200.,'volume_shares':10000.,'tradable':True}
    value['actions']=[action(value,'stock_distribution',ratio=.005,recipient_id='B',
                            payment_date='2020-01-03',distribution_convention=CONVENTION)]
    out=run(value)
    # $1 parent mark loss exactly offset by .005 * $200 of recipient stock.
    assert out['final_nav_observed_usd']==pytest.approx(10000.)
    holdings={h['security_id']:h['shares'] for h in out['holdings'] if h['date']=='2020-01-03'}
    assert holdings=={'A':quantity,'B':quantity*.005}
    assert not out['unpriced_costs'] and not out['claims_outstanding']
    value['decisions']=[decision(value,{'A':0.})]
    out=run(value)
    delivery=next(j for j in out['journal'] if j['kind']=='delivery')
    assert delivery['amount']==quantity*.005
    assert any(h['security_id']=='B' and h['shares']==quantity*.005 for h in out['holdings'])


def test_ex_session_buyer_has_no_prior_close_distribution_and_raw_marks_not_tax_value():
    value=scenario((100.,99.,99.))
    value['actions']=[action(value,'stock_distribution',ratio=.005,recipient_id='B',
                            payment_date='2020-01-03',distribution_convention=CONVENTION)]
    value['decisions']=[decision(value,{'A':1.})]
    assert not any(j['kind']=='delivery' for j in run(value)['journal'])


def test_distributed_short_stock_incurred_borrow_is_not_omitted():
    value=scenario((100.,99.,99.),initial_cash=20000.,initial_shares={'A':-100.})
    value['policy']['hypothetical_borrow_rate']=.03
    for s in value['sessions']:s['quotes']['B']={'raw_close_usd':200.,'volume_shares':10000.,'tradable':True}
    value['actions']=[action(value,'stock_distribution',ratio=.005,recipient_id='B',
                            payment_date='2020-01-03',distribution_convention=CONVENTION)]
    out=run(value)
    recipient=[j for j in out['journal'] if j['kind']=='borrow' and j['security_id']=='B']
    assert len(recipient)==1 and recipient[0]['calendar_days']==3
    assert recipient[0]['amount_usd']==pytest.approx(.5*200*.03*3/365)
    assert out['final_nav_observed_usd']==pytest.approx(10000-10000*.03*4/365)


@pytest.mark.parametrize('change',[{'distribution_convention':None},{'payment_date':'2020-01-06'},
                                  {'evidence_kind':'verified_terms'},{'recipient_id':'A'}])
def test_real_distribution_terms_never_silently_attest_research_settlement(change):
    value=scenario();value['actions']=[action(value,'stock_distribution',ratio=.005,recipient_id='B',
        payment_date='2020-01-03',distribution_convention=CONVENTION,**{}) | change]
    with pytest.raises(ResearchError):parse(LedgerScenario,value)


def test_assumed_distribution_forbids_pit_attestation_and_old_shape_is_stable():
    original=packet()
    parsed=parse(StudyPacket,original).model_dump()
    assert 'daily_risk_controls' not in parsed and 'intent' not in parsed['contexts'][0]
    assert parse(StudyPacket,parsed).model_dump()==parsed
    value=scenario();value.update(history_mode='archived_vendor_feed',point_in_time_attested=True)
    value['actions']=[action(value,'stock_distribution',ratio=.005,recipient_id='B',payment_date='2020-01-03',
        distribution_convention=CONVENTION,evidence_kind='declared_distribution_assumption')]
    with pytest.raises(ResearchError):parse(LedgerScenario,value)


def test_missing_recipient_quote_censors_nav_instead_of_zero_valuing_obligation():
    value=scenario(initial_cash=2000.,initial_shares={'A':-10.})
    value['actions']=[action(value,'stock_distribution',ratio=.005,recipient_id='B',payment_date='2020-01-03',
                            distribution_convention=CONVENTION)]
    out=run(value)
    assert out['final_nav_observed_usd'] is None and out['daily'][1]['missing_marks']==['B']


@pytest.mark.parametrize('field',['max_gross','max_name','net_tolerance','beta_tolerance'])
def test_daily_policy_cannot_relax_hard_limits(field):
    value=controls(packet());value['daily_risk_controls'][field]=value['portfolio_policy'][field]+.01
    with pytest.raises(ResearchError):parse(StudyPacket,value)


def drift_packet():
    value=controls(packet())
    # On Jan 3 the original neutral entry is accepted; on Jan 6 an already-held
    # long appreciates. The Jan 6 control can only fill Jan 7.
    value['scenario']['sessions'][2]['quotes']['A']['raw_close_usd']=120.
    last=deepcopy(value['scenario']['sessions'][2]);last.update(date='2020-01-07',close_utc='2020-01-07T21:00:00+00:00')
    value['scenario']['sessions'].append(last)
    return value


def test_daily_breach_fixed_next_close_reductions_keep_hard_policy_and_no_new_alpha(tmp_path):
    value=drift_packet();hard=deepcopy(value['portfolio_policy'])
    out,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    control=port['risk_controls'][2]
    assert control['status']=='reduction_planned'
    assert set(control['violations'])>={'net','beta'}
    # Before reduction: long $300+$250, short $250+$250, NAV $1050,
    # net 50/1050; to .005 net requires scale .105.
    assert control['uniform_scale']==pytest.approx(.105)
    orders=[o for o in out['orders'] if o['decision_id'].startswith('daily-control-')]
    assert len(orders)==4 and {o['date'] for o in orders}=={'2020-01-07'}
    assert sorted(o['target_shares'] for o in orders)==pytest.approx([-.2625,-.2625,.2625,.2625])
    assert port['actual_position_risk'][2]['status']=='constraints_violated'
    assert port['actual_position_risk'][3]['status']=='covered'
    assert value['portfolio_policy']==hard


def test_future_gap_does_not_resize_control_and_partial_fill_retains_violation(tmp_path):
    first=drift_packet();changed=deepcopy(first)
    changed['scenario']['sessions'][3]['quotes']['A']['raw_close_usd']=500.
    changed['scenario']['sessions'][3]['quotes']['A']['volume_shares']=.001
    a,pa=evaluate_study(parse(StudyPacket,first),tmp_path)
    b,pb=evaluate_study(parse(StudyPacket,changed),tmp_path)
    assert pa['risk_controls'][:3]==pb['risk_controls'][:3]
    assert a['daily'][:3]==b['daily'][:3]
    assert pb['actual_position_risk'][-1]['status']=='constraints_violated'
    assert pb['risk_controls'][-1]['status']=='right_censored_no_next_session'
    assert any(o['status']=='filled_derisk_preexisting_violation' for o in b['orders'])


def test_monthly_alpha_is_suppressed_when_daily_reductions_are_needed(tmp_path):
    value=drift_packet();context=deepcopy(value['contexts'][0])
    context.update(context_id='rebalance',date='2020-01-06',information_time='2020-01-06T21:00:00+00:00')
    value['contexts'].append(context)
    out,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    assert port['plans'][-1]['status']=='suppressed_daily_risk_control'
    assert not any(o['decision_id']=='rebalance' for o in out['orders'])


def test_scheduled_zero_close_has_priority_and_no_repeat_alpha_search(tmp_path):
    value=drift_packet();context=deepcopy(value['contexts'][0])
    context.update(context_id='closing',date='2020-01-06',information_time='2020-01-06T21:00:00+00:00')
    context['intent']='close_all'
    for point in context['points']:point.update(score=None,eligible=False)
    value['contexts'].append(context)
    out,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    assert port['risk_controls'][2]['status']=='scheduled_close_priority'
    assert all(h['date']!='2020-01-07' for h in out['holdings'])


def test_unresolved_nav_blocks_control_sizing_without_silently_replacing_marks(tmp_path):
    value=drift_packet();value['scenario']['sessions'][2]['quotes'].pop('A')
    out,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    assert port['risk_controls'][2]['status']=='insufficient_evidence'
    assert not any(o['decision_id']=='daily-control-2020-01-06' for o in out['orders'])


def test_received_non_alpha_asset_exits_after_receipt_even_without_hard_breach(tmp_path):
    value=controls(packet())
    value['scenario'].update(initial_cash_usd=1100.,initial_shares={'A':-1.})
    for s in value['scenario']['sessions']:
        s['quotes']['E']={'raw_close_usd':20.,'volume_shares':10000.,'tradable':True}
    value['scenario']['securities']['E']='RECEIVED'
    value['contexts'][0]['points'].append({'security_id':'E','score':None,'eligible':False,
                                          'issuer_id':None,'sector_id':None,'beta':1.})
    value['scenario']['actions']=[action(value['scenario'],'stock_distribution',ratio=.005,recipient_id='E',
        payment_date='2020-01-03',distribution_convention=CONVENTION)]
    out,port=evaluate_study(parse(StudyPacket,value),tmp_path)
    assert port['risk_controls'][1]['exit_security_ids']==['E']
    recipient=[o for o in out['orders'] if o['security_id']=='E']
    assert len(recipient)==1 and recipient[0]['date']=='2020-01-06'
    assert recipient[0]['filled_delta']==pytest.approx(.005)
    assert recipient[0]['decision_id']=='daily-control-2020-01-03'


def test_daily_turnover_cap_and_zero_tolerance_can_require_staged_reductions():
    hard=policy(max_turnover=.1,net_tolerance=0.,beta_tolerance=0.)
    inputs=positions()
    inputs=[p.model_copy(update={'held_shares':2.}) for p in inputs]
    control=DailyRiskControls(methodology='close_breach_uniform_reduction_v1',fill_lag_sessions=1,
              max_gross=.8,max_name=.2,net_tolerance=0.,beta_tolerance=0.,exit_ineligible_holdings=True)
    result=reduction_targets(inputs,[p.security_id for p in inputs],np.eye(4)*.0001,1000.,hard,control)
    assert result['turnover_fraction']<1.
    assert result['planned_exposures_at_decision_close']['turnover']==pytest.approx(.09)
    assert 'net' in result['planned_residual_violations']
    assert all(0 <= q <= 2. for q in result['target_shares'].values())
