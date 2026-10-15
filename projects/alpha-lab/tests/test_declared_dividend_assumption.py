"""Research cash assumptions remain distinct from attested corporate evidence."""
import pytest

from alpha_lab.research.contracts import ResearchError, parse
from alpha_lab.research.ledger_contracts import LedgerScenario
from tests.test_accounting_ledger import action, run, scenario


def dividend_scenario():
    value=scenario(prices=(100.,99.,99.),initial_shares={'A':10.})
    value.update(history_mode='public_information_reconstruction',terminal_audit_complete=False,
                 economic_outcomes_resolved=False)
    value['actions']=[action(value,'cash_dividend',cash_usd_per_share=1.,
                            payment_date=value['sessions'][1]['date'],
                            evidence_kind='declared_dividend_assumption')]
    return value


def test_explicit_research_dividend_settlement_reconciles_and_retains_label():
    result=run(dividend_scenario())
    assert result['final_nav_observed_usd']==2000.
    assert result['investment_ready'] is False and result['point_in_time_attested'] is False
    event=next(row for row in result['journal'] if row['kind']=='action')
    assert event['evidence_kind']=='declared_dividend_assumption'
    assert result['economic_outcomes_resolved'] is False


def test_declared_dividend_assumption_cannot_become_point_in_time_attested():
    value=dividend_scenario();value['point_in_time_attested']=True
    with pytest.raises(ResearchError):parse(LedgerScenario,value)


def test_declared_dividend_assumption_cannot_supply_merger_terms():
    value=dividend_scenario();value['actions'][0]['kind']='cash_merger'
    with pytest.raises(ResearchError):parse(LedgerScenario,value)
