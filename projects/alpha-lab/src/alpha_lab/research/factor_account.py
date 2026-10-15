"""Provisional total-return-unit account, distinct from the raw-share action ledger."""
from __future__ import annotations

import numpy as np
from .contracts import ResearchError
from .factor_construction import annual_volatility


def position_exposures(weights,beta,industry):
    groups={str(g):{'net':float(weights[industry==g].sum()),
                     'beta':float((weights[industry==g]*beta[industry==g]).sum())} for g in set(industry)}
    return {'gross':float(abs(weights).sum()),'name':float(abs(weights).max(initial=0.)),
            'net':float(weights.sum()),'beta':float(weights@beta),'industries':groups}


def breaches(values, *, industry_limit=.005):
    limits={'gross':1.,'name':.05,'net':.01,'beta':.01}
    result=[key for key,limit in limits.items() if abs(values[key])>limit+1e-6]
    if any(abs(v['net'])>industry_limit+1e-6 for v in values['industries'].values()):result.append('industry_net')
    if any(abs(v['beta'])>industry_limit+1e-6 for v in values['industries'].values()):result.append('industry_beta')
    if values.get('forecast_vol') is not None and values['forecast_vol']>.1+1e-6:result.append('forecast_risk')
    return result


def risk_at(weights,risk):
    if not np.any(abs(weights)>1e-12):return 0.
    if risk is None:return None
    idx,b,d,supported=risk
    outside=np.ones(len(weights),dtype=bool);outside[idx]=False
    if np.any(abs(weights[outside])>1e-12) or np.any((abs(weights[idx])>1e-12)&~supported):return None
    return annual_volatility(b,d,weights[idx])


class DiagnosticAccount:
    def __init__(self,n):
        self.units=np.zeros(n);self.cash=1e6;self.marks=np.full(n,np.nan)
        self.pending=None;self.daily=[];self.orders=[];self.controls=[]
        self.last_day=None;self.last_resolved=True

    def value(self,units=None,cash=None):
        units=self.units if units is None else units;cash=self.cash if cash is None else cash
        return float(cash+np.where(units!=0,units*np.nan_to_num(self.marks),0.).sum())

    def schedule(self,index,targets,kind):
        self.pending={'fill_index':index+1,'targets':np.array(targets,copy=True),'kind':kind,'decision_index':index}

    def step(self,index,day,price,dollar_volume,beta,industry,risk=None):
        from datetime import date
        before=self.value();borrow=funding=0.
        if self.last_day:
            days=(date.fromisoformat(day)-date.fromisoformat(self.last_day)).days
            borrow=float((-np.minimum(self.units,0)*np.nan_to_num(self.marks)).sum()*.03*days/365)
            funding=max(-self.cash,0.)*.05*days/365
            self.cash-=borrow+funding
        observed=np.isfinite(price)&(price>0);self.marks[observed]=price[observed]
        costs=turnover=0.;order_status='none'
        missing=(self.units!=0)&~observed
        if self.pending is not None and self.pending['fill_index']==index:
            old=self.units.copy();target=self.pending['targets'];requested=target-old
            # A total-return unit's notional maps to raw dollar volume; no fake
            # executable stale quote, raw shares or corporate-action payout.
            available=observed&np.isfinite(dollar_volume)&(dollar_volume>=0)
            cap=np.divide(dollar_volume*.01,price,out=np.zeros(len(price)),where=available)
            delta=np.where(available,np.sign(requested)*np.minimum(abs(requested),cap),0.)
            proposed=old+delta
            reducing=(abs(proposed)<=abs(old)+1e-10)&(proposed*old>=-1e-12)
            if missing.any():delta=np.where(reducing,delta,0.)
            proposed=old+delta
            execution=np.where(available,price*(1+np.sign(delta)*.0005),0.)
            commission=abs(delta)*execution*.0001
            slip=abs(delta)*np.where(available,price,0.)*.0005
            trial_cash=self.cash-float((delta*execution+commission).sum())
            nav=self.value(proposed,trial_cash)
            weights=proposed*np.nan_to_num(self.marks)/nav if nav>0 else np.zeros(len(price))
            exposure=position_exposures(weights,beta,industry);exposure['forecast_vol']=risk_at(weights,risk)
            violations=breaches(exposure) if nav>0 else ['nonpositive_nav']
            if risk is not None and exposure['forecast_vol'] is None:violations.append('missing_held_risk')
            changed=abs(delta)>1e-12
            turnover=float((abs(delta)*np.nan_to_num(price)).sum()/nav) if nav>0 else float('inf')
            if turnover>2.+1e-6:violations.append('turnover')
            only_reduce=bool(np.all(abs(proposed)<=abs(old)+1e-10)&np.all(proposed*old>=-1e-12))
            if violations and not only_reduce:
                order_status='rejected_batch_constraints';delta[:]=0.;turnover=0.
            else:
                self.units=proposed;self.cash=trial_cash;costs=float((commission+slip).sum())
                order_status='filled_derisk_residual_violation' if violations else 'filled'
            self.orders.append({'date':day,'index':index,'decision_index':self.pending['decision_index'],
                 'kind':self.pending['kind'],'status':order_status,'nonzero_fills':int(np.count_nonzero(abs(delta)>1e-12)),
                 'trade_cost_usd':costs,'turnover':turnover,'violations':violations,
                 'unexecutable_requested_names':int(np.count_nonzero((abs(requested)>1e-12)&~available)),
                 'filled_delta':delta.copy(),'requested_delta':requested.copy()})
            self.pending=None
        nav=self.value();missing=(abs(self.units)>1e-12)&~observed
        resolved=not missing.any()
        weights=self.units*np.nan_to_num(self.marks)/nav if nav>0 else np.zeros(len(price))
        values=position_exposures(weights,beta,industry);values['forecast_vol']=risk_at(weights,risk)
        violations=breaches(values)
        actual_return=nav/before-1 if self.last_day and self.last_resolved and resolved and before>0 else None
        self.daily.append({'date':day,'nav_observed_usd':nav if resolved else None,'nav_estimate_usd':nav,
          'valuation_resolved':resolved,'missing_held_marks':int(missing.sum()),'borrow_usd':borrow,
          'funding_usd':funding,'trade_cost_usd':costs,'return_observed':actual_return,'cash_usd':self.cash,
          'gross':values['gross'],'net':values['net'],'beta':values['beta'],'name_max':values['name'],
          'annual_ex_ante_vol':values['forecast_vol'],
          'hard_violations':','.join(violations),'held_names':int(np.count_nonzero(abs(self.units)>1e-12))})
        self.last_day=day;self.last_resolved=resolved
        return values,violations,resolved

    def daily_control(self,index,day,values,violations,resolved,final_index):
        if not violations:return False
        if not resolved or self.value()<=0:
            self.controls.append({'date':day,'status':'blocked_unresolved_nav','violations':violations});return False
        scale=1.
        for size,limit in [(values['gross'],.8),(values['name'],.04),(abs(values['net']),.005),(abs(values['beta']),.005)]:
            if size>limit:scale=min(scale,limit/size)
        for group in values['industries'].values():
            for key in ('net','beta'):
                if abs(group[key])>.0025:scale=min(scale,.0025/abs(group[key]))
        if values.get('forecast_vol') is not None and values['forecast_vol']>.09:
            scale=min(scale,.09/values['forecast_vol'])
        if index==final_index:
            self.controls.append({'date':day,'status':'right_censored','violations':violations});return False
        self.schedule(index,self.units*scale,'daily_reduction')
        self.controls.append({'date':day,'status':'scheduled_next_close','uniform_scale':scale,'violations':violations})
        return True
