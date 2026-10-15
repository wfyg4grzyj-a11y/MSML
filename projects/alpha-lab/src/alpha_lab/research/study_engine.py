"""Trusted integration: current holdings -> risk/constraints -> next-close share orders."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .contracts import ResearchError
from .ledger import Ledger, lending_at
from .ledger_contracts import Decision
from .portfolio import PositionInput, coverage, exposures, project_portfolio
from .risk_adapter import RiskUse, diagonal_baseline, load_risk, validate_risk_use


class PortfolioController:
    def __init__(self, packet, directory):
        self.packet,self.directory=packet,directory
        self.plans,self.observed=[] ,[]
        self.risk_cache={}

    def context(self, session):
        candidates=[c for c in self.packet.contexts if c.date<=session.date
                    and datetime.fromisoformat(c.information_time)<=datetime.fromisoformat(session.close_utc)]
        return candidates[-1] if candidates else None

    def risk(self, context, session):
        ids=[p.security_id for p in context.points]
        if self.packet.risk_mode=='diagonal_reference':
            key=('diagonal',tuple(ids))
            if key not in self.risk_cache:
                if len(self.risk_cache)>=2:self.risk_cache.clear()
                matrix=diagonal_baseline(ids,self.packet.variance_per_session,self.packet.portfolio_policy.risk_horizon_sessions)
                matrix.setflags(write=False);self.risk_cache[key]=matrix
            return ids,self.risk_cache[key],{
                'kind':'diagonal_reference_not_fitted_equity_risk','variance_per_session':self.packet.variance_per_session}
        use=RiskUse(decision_time=session.close_utc,horizon_sessions=self.packet.portfolio_policy.risk_horizon_sessions,
                    max_age_calendar_days=self.packet.max_risk_age_calendar_days,
                    contract_sha256=self.packet.risk_contract_sha256,source_policy_id=self.packet.risk_source_policy_id,
                    benchmark_id=self.packet.benchmark_id,cash_id=self.packet.cash_id,calendar_id=self.packet.calendar_id,
                    identity_namespace=self.packet.identity_namespace,expected_security_order=ids,
                    use='synthetic_validation' if self.packet.scenario.history_mode=='synthetic_reference' else 'attested_research')
        key=(context.risk_binding_id,use.horizon_sessions,tuple(ids))
        if key not in self.risk_cache:
            if len(self.risk_cache)>=2:self.risk_cache.clear()
            self.risk_cache[key]=load_risk(self.directory/context.risk_binding_id,use)
        binding,covariance=self.risk_cache[key]
        validate_risk_use(binding,use)
        return ids,covariance,{'kind':'equity_risk_forecast','forecast_id':binding.forecast_id,
                              'as_of':binding.as_of,'available_at':binding.available_at,'training_cutoff':binding.training_cutoff}

    def inputs(self, ledger, session, context):
        ids={p.security_id for p in context.points}
        if any(q and sid not in ids for sid,q in ledger.shares.items()):
            raise ResearchError('Held-position risk coverage disappeared')
        if any(c['kind']=='stock' and c['amount'] for c in ledger.claims):
            raise ResearchError('Pending stock consideration needs separate risk coverage before new targets')
        result=[]
        for point in context.points:
            sid=point.security_id;quote=session.quotes.get(sid)
            loan=lending_at(ledger.spec,sid,session.close_utc,session.close_utc)
            available=loan.available_shares if loan else (1e30 if ledger.policy.borrow_mode=='explicit_hypothetical' else None)
            result.append(PositionInput(security_id=sid,score=point.score,raw_price_usd=quote.raw_close_usd if quote else None,
                         held_shares=ledger.shares.get(sid,0.),eligible=point.eligible and sid not in ledger.terminal,
                         tradable=quote.tradable and sid not in ledger.terminal if quote else False,
                         borrow_available_shares=available,issuer_id=point.issuer_id,sector_id=point.sector_id,beta=point.beta))
        return result

    def decisions(self, ledger, session):
        context=next((c for c in self.packet.contexts if c.date==session.date),None)
        if context is None:return []
        evidence={'date':session.date,'context_id':context.context_id,'signal_evidence_sha256':context.signal_evidence_sha256,
                  'source_policy_id':context.source_policy_id,'decision_nav_usd':ledger.daily[-1]['nav_observed_usd']}
        if not ledger.daily[-1]['valuation_resolved']:
            self.plans.append(evidence | {'status':'insufficient_evidence','reason':'unresolved_accounting_nav','target_weights':None})
            return []
        try:
            ids,covariance,risk=self.risk(context,session)
            inputs=self.inputs(ledger,session,context)
            result=project_portfolio(inputs,ids,covariance,ledger.nav(),self.packet.portfolio_policy)
        except ResearchError as error:
            self.plans.append(evidence | {'status':'insufficient_evidence','reason':str(error),'target_weights':None})
            return []
        self.plans.append(evidence | result | {'risk':risk})
        if result['target_weights'] is None:return []
        return [Decision(decision_id=context.context_id,date=session.date,information_time=context.information_time,
                         target_shares=result['target_shares'],fill_lag_sessions=context.fill_lag_sessions)]

    def candidate(self, ledger, session, shares, cash):
        context=self.context(session)
        if context is None:return ['risk_context_unavailable']
        try:
            ids,covariance,_=self.risk(context,session)
            inputs=self.inputs(ledger,session,context)
            nav=ledger.nav()+cash-ledger.cash+sum((shares.get(sid,0.)-ledger.shares.get(sid,0.))*ledger.marks.get(sid,0.) for sid in shares)
            reason=coverage(inputs,ids,covariance,self.packet.portfolio_policy,nav,covariance_already_validated=True)
            if reason:return [reason]
            weights=np.array([shares.get(p.security_id,0.)*(p.raw_price_usd or 0.)/nav for p in inputs])
            _,violations=exposures(inputs,weights,covariance,nav,self.packet.portfolio_policy)
            return violations
        except ResearchError as error:
            return [str(error)]

    def positions(self, ledger, session):
        context=self.context(session)
        if context is None:
            self.observed.append({'date':session.date,'status':'no_risk_context_yet'})
            return
        try:
            ids,covariance,risk=self.risk(context,session)
            inputs=self.inputs(ledger,session,context)
            if not ledger.daily[-1]['valuation_resolved']:raise ResearchError('Accounting valuation is unresolved')
            reason=coverage(inputs,ids,covariance,self.packet.portfolio_policy,ledger.nav(),covariance_already_validated=True)
            if reason:raise ResearchError(reason)
            weights=np.array([p.held_shares*(p.raw_price_usd or 0.)/ledger.nav() for p in inputs])
            values,violations=exposures(inputs,weights,covariance,ledger.nav(),self.packet.portfolio_policy)
            self.observed.append({'date':session.date,'status':'constraints_violated' if violations else 'covered',
                                  'exposures':values,'violations':violations,'risk':risk})
        except ResearchError as error:
            self.observed.append({'date':session.date,'status':'insufficient_evidence','reason':str(error)})


def evaluate_study(packet, directory):
    ledger=Ledger(packet.scenario)
    if packet.portfolio_policy is None:
        result=ledger.run();plans,positions=[],[]
    else:
        controller=PortfolioController(packet,directory)
        result=ledger.run(decision_provider=controller.decisions,candidate_check=controller.candidate,position_reporter=controller.positions)
        plans,positions=controller.plans,controller.observed
    return result,{'plans':plans,'actual_position_risk':positions,'selection':'none','fallback':'none',
                   'objective':'unitless_score_projection_not_expected_return','investment_ready':False,
                   'risk_mode':packet.risk_mode,'portfolio_policy':packet.portfolio_policy.model_dump() if packet.portfolio_policy else None}
