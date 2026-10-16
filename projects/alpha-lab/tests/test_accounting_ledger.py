"""Hand-derived economic references, not parity with the legacy backtester."""
import copy

import pytest
pytest.importorskip('numpy')
from alpha_lab.equity_calendar import session_schedule
from alpha_lab.research.contracts import ResearchError, parse
from alpha_lab.research.ledger import Ledger
from alpha_lab.research.ledger_contracts import LedgerScenario


def scenario(prices=(100.,100.,100.), *, initial_cash=1000., initial_shares=None, start='2020-01-02', end='2020-01-06'):
    calendar,_=session_schedule(start,end)
    sessions=[]
    for row,price in zip(calendar.itertuples(),prices):
        quotes={'A':{'raw_close_usd':price,'volume_shares':100000.,'tradable':True}} if price is not None else {}
        sessions.append({'date':str(row.date.date()),'close_utc':str(row.session_close_utc),'quotes':quotes})
    return {'schema_version':3,'scenario_id':'reference','source_id':'a'*64,'history_mode':'synthetic_reference',
            'point_in_time_attested':False,'terminal_audit_complete':True,'economic_outcomes_resolved':True,
            'securities':{'A':'ALPHA','B':'BETA'},'initial_cash_usd':initial_cash,'initial_shares':initial_shares or {},
            'policy':{'methodology':'raw_share_cash_action_ledger_v1','price_basis':'unadjusted_usd_per_share_explicit_actions',
                      'trade_settlement':'T0_research_cash_not_broker_settlement','short_proceeds':'segregated_no_cash_interest_no_rebate',
                      'dividend_entitlement':'prior_close_position_before_ex_session_orders',
                      'missing_marks':'last_raw_mark_provisional_no_executable_exit','day_count':'ACT365F',
                      'annual_cash_credit':0.,'annual_cash_debit':0.,'commission_bps':0.,'slippage_bps':0.,
                      'participation':1.,'allow_cash_debit':True,'borrow_mode':'explicit_hypothetical',
                      'hypothetical_borrow_rate':0.,'max_gross':4.,'max_name':4.,'tolerance_usd':.000001},
            'sessions':sessions,'actions':[],'lending':[],'decisions':[]}


def decision(value, targets, index=0, name='order'):
    session=value['sessions'][index]
    return {'decision_id':name,'date':session['date'],'information_time':session['close_utc'],
            'target_shares':targets,'fill_lag_sessions':1}


def action(value, kind, *, security='A', index=1, name='action', **terms):
    return {'action_id':name,'security_id':security,'date':value['sessions'][index]['date'],
            'known_at':value['sessions'][0]['close_utc'],'kind':kind,'ratio':None,'cash_usd_per_share':None,
            'recipient_id':None,'payment_date':None,'new_ticker':None,'evidence':'b'*64,
            'evidence_kind':'unresolved' if kind.startswith('unresolved_') else 'synthetic_reference',**terms}


def run(value):return Ledger(parse(LedgerScenario,value)).run()


def test_cash_only_and_long_short_proceeds_are_not_profit():
    assert run(scenario())['final_nav_observed_usd']==1000
    for shares, cash in ((10.,0.),(-10.,2000.)):
        value=scenario();value['decisions']=[decision(value,{'A':shares})]
        out=run(value)
        assert out['daily'][1]['cash_usd']==cash
        assert out['final_nav_observed_usd']==1000
        assert out['orders'][0]['filled_delta']==shares


def test_next_close_causality_future_fill_price_does_not_redesign_quantity():
    value=scenario((100.,200.,220.));value['decisions']=[decision(value,{'A':5.})]
    out=run(value)
    assert out['daily'][0]['nav_observed_usd']==1000
    assert out['daily'][1]['nav_observed_usd']==1000
    assert out['daily'][2]['nav_observed_usd']==1100
    revised=copy.deepcopy(value);revised['sessions'][2]['quotes']['A']['raw_close_usd']=9999.
    assert run(revised)['orders']==out['orders']
    assert run(revised)['daily'][:2]==out['daily'][:2]


def test_signed_costs_flat_price_round_trip_and_slippage():
    value=scenario();value['policy']['commission_bps']=10.
    value['decisions']=[decision(value,{'A':10.}),decision(value,{'A':0.},1,'close')]
    out=run(value)
    assert out['final_nav_observed_usd']==998
    assert sum(d['trade_cost_usd'] for d in out['daily'])==2
    value['policy']['commission_bps']=0.;value['policy']['slippage_bps']=10.
    assert run(value)['final_nav_observed_usd']==pytest.approx(998.)


def test_split_adjusts_holdings_marks_and_prior_decision_share_basis():
    value=scenario((100.,50.,50.),initial_cash=0.,initial_shares={'A':10.})
    value['actions']=[action(value,'split',ratio=2.)]
    value['decisions']=[decision(value,{'A':10.})]
    out=run(value)
    assert out['final_nav_observed_usd']==1000
    assert out['holdings'][-1]['shares']==20
    assert out['orders'][0]['requested_delta']==0


@pytest.mark.parametrize('shares,cash',[(10.,0.),(-10.,2000.)])
def test_dividend_ex_entitlement_receivable_payment_and_post_ex_trades(shares,cash):
    value=scenario((100.,99.,99.),initial_cash=cash,initial_shares={'A':shares})
    value['actions']=[action(value,'cash_dividend',cash_usd_per_share=1.,payment_date='2020-01-06')]
    value['decisions']=[decision(value,{'A':0.})]
    out=run(value)
    assert out['daily'][1]['receivables_net_usd']==shares
    assert out['daily'][1]['nav_observed_usd']==1000
    assert out['final_nav_observed_usd']==1000
    value=scenario((100.,99.,99.));value['actions']=[action(value,'cash_dividend',cash_usd_per_share=1.,payment_date='2020-01-06')]
    value['decisions']=[decision(value,{'A':10.})]
    assert not any(j['kind']=='delivery' for j in run(value)['journal'])


def test_weekend_act365_cash_funding_and_short_borrow_are_calendar_accruals():
    value=scenario((100.,100.),initial_cash=2000.,initial_shares={'A':-10.},start='2020-01-03',end='2020-01-06')
    value['policy'].update(annual_cash_credit=.1,hypothetical_borrow_rate=.03)
    out=run(value)
    assert out['daily'][1]['borrow_usd']==pytest.approx(1000*.03*3/365)
    assert out['daily'][1]['cash_interest_usd']==pytest.approx(1000*.1*3/365)
    value['initial_cash_usd']=0.;value['initial_shares']={'A':10.}
    value['decisions']=[decision(value,{'A':20.})]
    # Cash debits occur at the close, never accrued over an earlier interval.
    assert run(value)['daily'][1]['funding_usd']==0


def test_partial_unfilled_and_missing_quote_never_liquidate_at_stale_mark():
    value=scenario();value['sessions'][1]['quotes']['A']['volume_shares']=3.
    value['decisions']=[decision(value,{'A':10.})]
    out=run(value)
    assert out['orders'][0]['filled_delta']==3 and out['orders'][0]['unfilled_delta']==7
    assert out['holdings'][-1]['shares']==3
    missing=scenario((100.,None,105.),initial_cash=0.,initial_shares={'A':10.})
    missing['decisions']=[decision(missing,{'A':0.})]
    result=run(missing)
    assert result['orders'][0]['filled_delta']==0
    assert result['daily'][1]['nav_observed_usd'] is None
    assert result['daily'][2]['nav_observed_usd']==1050
    assert result['daily'][2]['return_observed'] is None
    assert result['endpoint_return']==pytest.approx(.05)


def test_missing_borrow_and_unavailable_locate_are_visible_not_zero_cost_truth():
    value=scenario();value['policy'].update(borrow_mode='evidenced_only',hypothetical_borrow_rate=None)
    value['decisions']=[decision(value,{'A':-10.})]
    assert run(value)['orders'][0]['status']=='rejected_short_locate'
    value['initial_cash_usd']=2000.;value['initial_shares']={'A':-10.};value['decisions']=[]
    result=run(value)
    assert result['final_nav_observed_usd'] is None and len(result['unpriced_costs'])==2


@pytest.mark.parametrize('kind',['cash_merger','stock_merger','mixed_merger'])
def test_cash_stock_and_mixed_merger_delivery(kind):
    value=scenario((100.,None,None),initial_cash=0.,initial_shares={'A':10.})
    for session in value['sessions']:
        session['quotes']['B']={'raw_close_usd':50.,'volume_shares':10000.,'tradable':True}
    terms={'payment_date':'2020-01-06'}
    if kind!='stock_merger':terms['cash_usd_per_share']=100. if kind=='cash_merger' else 50.
    if kind!='cash_merger':terms.update(ratio=2. if kind=='stock_merger' else 1.,recipient_id='B')
    value['actions']=[action(value,kind,**terms)]
    out=run(value)
    assert out['final_nav_observed_usd']==1000 and not out['claims_outstanding']
    assert all(h['security_id']=='B' for h in out['holdings'] if h['date']=='2020-01-06')


def test_rename_preserves_identity_and_unresolved_terminal_has_no_guessed_proceeds():
    value=scenario(initial_cash=0.,initial_shares={'A':10.})
    value['actions']=[action(value,'rename',new_ticker='RENAMED')]
    out=run(value);assert out['holdings'][-1]['security_id']=='A' and out['holdings'][-1]['ticker']=='RENAMED'
    value['actions']=[action(value,'unresolved_terminal')]
    value['economic_outcomes_resolved']=False
    out=run(value);assert out['final_nav_observed_usd'] is None
    assert out['daily'][-1]['cash_usd']==0 and out['holdings'][-1]['shares']==10


def test_calendar_early_close_and_strict_basis_future_information_are_enforced():
    value=scenario((100.,100.),start='2020-11-25',end='2020-11-27')
    assert value['sessions'][1]['close_utc'].startswith('2020-11-27 18:00:00')
    assert run(value)['final_nav_observed_usd']==1000
    value['sessions'][1]['close_utc']='2020-11-27T21:00:00+00:00'
    with pytest.raises(ResearchError,match='official'):run(value)
    value=scenario();value['policy']['price_basis']='adjusted_total_return'
    with pytest.raises(ResearchError):run(value)
    value=scenario();value['decisions']=[decision(value,{'A':1.})]
    value['decisions'][0]['information_time']=value['sessions'][1]['close_utc']
    with pytest.raises(ResearchError):run(value)


def test_cash_debit_accrues_only_after_fill_and_spans_weekend():
    value=scenario();value['policy']['annual_cash_debit']=.1
    value['decisions']=[decision(value,{'A':20.})]
    result=run(value)
    assert result['daily'][1]['funding_usd']==0
    assert result['daily'][2]['funding_usd']==pytest.approx(1000*.1*3/365)
    assert result['final_nav_observed_usd']==pytest.approx(1000-1000*.1*3/365)


def test_newer_expired_stock_loan_does_not_fall_back_to_old_observation():
    from alpha_lab.research.ledger import lending_at
    value=scenario();value['policy'].update(borrow_mode='evidenced_only',hypothetical_borrow_rate=None)
    old={'security_id':'A','known_at':'2020-01-02T20:00:00+00:00','valid_through':'2020-01-10T21:00:00+00:00',
         'available_shares':100.,'annual_fee':.03,'evidence':'d'*64,'evidence_kind':'synthetic_reference'}
    newer=old | {'known_at':'2020-01-02T20:30:00+00:00','valid_through':'2020-01-02T21:00:00+00:00'}
    value['lending']=[old,newer];value['decisions']=[decision(value,{'A':-10.})]
    parsed=parse(LedgerScenario,value)
    assert lending_at(parsed,'A',parsed.sessions[0].close_utc,parsed.sessions[1].close_utc) is None
    assert run(value)['orders'][0]['filled_delta']==0


def test_later_recall_rejects_earlier_short_decision_without_changing_order_size():
    value=scenario();value['policy'].update(borrow_mode='evidenced_only',hypothetical_borrow_rate=None)
    old={'security_id':'A','known_at':'2020-01-02T20:00:00+00:00','valid_through':'2020-01-10T21:00:00+00:00',
         'available_shares':100.,'annual_fee':.03,'evidence':'d'*64,'evidence_kind':'synthetic_reference'}
    value['lending']=[old,old | {'known_at':'2020-01-03T20:00:00+00:00','available_shares':0.}]
    value['decisions']=[decision(value,{'A':-10.})]
    result=run(value);assert result['orders'][0]['target_shares']==-10. and result['orders'][0]['filled_delta']==0


def test_undelivered_short_stock_financing_and_never_marked_long_claim_are_unresolved():
    value=scenario((100.,None,None),initial_cash=2000.,initial_shares={'A':-10.})
    for session in value['sessions']:session['quotes']['B']={'raw_close_usd':50.,'volume_shares':1000.,'tradable':True}
    value['actions']=[action(value,'stock_merger',ratio=2.,recipient_id='B',payment_date='2020-01-06')]
    out=run(value)
    assert out['final_nav_observed_usd'] is None
    assert out['unpriced_costs'][0]['reason']=='short_stock_claim_financing_before_delivery_unknown'
    value=scenario((100.,None,None),initial_cash=0.,initial_shares={'A':10.})
    value['actions']=[action(value,'stock_merger',ratio=2.,recipient_id='B',payment_date='2020-01-06')]
    out=run(value)
    assert out['final_nav_estimate_usd'] is None and out['final_nav_observed_usd'] is None
    assert out['holdings'][-1]['shares']==20.


def test_cancel_replace_censored_decision_removes_old_pending_order():
    value=scenario()
    first=decision(value,{'A':10.});first['fill_lag_sessions']=2
    replacement=decision(value,{'A':0.},1,'replacement');replacement['fill_lag_sessions']=2
    value['decisions']=[first,replacement]
    out=run(value)
    assert not out['orders'] and any(j['kind']=='cancel_replaced_pending' for j in out['journal'])


def test_unresolved_distribution_retains_position_without_declaring_a_delisting():
    value=scenario(initial_cash=0.,initial_shares={'A':10.})
    value['actions']=[action(value,'unresolved_distribution')];value['economic_outcomes_resolved']=False
    result=run(value)
    assert result['final_nav_observed_usd'] is None
    assert result['unpriced_costs'][0]['reason']=='unresolved_distribution_entitlement'
    assert result['holdings'][-1]['shares']==10. and not result['holdings'][-1]['economic_terminal_unknown']
    assert not result['daily'][-1]['unresolved_terminal']
