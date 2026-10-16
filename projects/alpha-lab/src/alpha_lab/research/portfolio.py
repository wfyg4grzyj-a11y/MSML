"""Bounded reference portfolio projection with explicit feasibility and coverage."""
from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_serializer, model_validator

from .contracts import Identifier, ResearchError, StrictModel


class PositionInput(StrictModel):
    security_id: Identifier
    score: float | None
    raw_price_usd: Annotated[float, Field(gt=0)] | None
    held_shares: float
    eligible: bool
    tradable: bool
    borrow_available_shares: Annotated[float, Field(ge=0)] | None
    issuer_id: Identifier | None
    sector_id: Identifier | None
    beta: float | None


class TargetLimits(StrictModel):
    """Optional stricter construction limits; never used for fills or monitoring."""
    max_gross: Annotated[float, Field(gt=0, le=4)]
    max_name: Annotated[float, Field(gt=0, le=1)]
    max_turnover: Annotated[float, Field(ge=0, le=8)]
    net_tolerance: Annotated[float, Field(ge=0, le=.1)]
    beta_tolerance: Annotated[float, Field(ge=0, le=.1)]


class PortfolioPolicy(StrictModel):
    methodology: Literal['score_projection_covariance_penalty_v1']
    score_units: Literal['unitless_rank_not_expected_return']
    borrow_recall: Literal['block_increases_report_retained_short']
    risk_target: Literal['additive_excess_returns']
    risk_units: Literal['squared_decimal_return']
    risk_horizon_sessions: Annotated[int, Field(ge=1, le=63)]
    max_gross: Annotated[float, Field(gt=0, le=4)]
    max_name: Annotated[float, Field(gt=0, le=1)]
    max_issuer_gross: Annotated[float, Field(gt=0, le=1)] | None
    max_sector_net: Annotated[float, Field(ge=0, le=1)] | None
    max_turnover: Annotated[float, Field(ge=0, le=8)]
    net_tolerance: Annotated[float, Field(ge=0, le=.1)]
    beta_tolerance: Annotated[float, Field(ge=0, le=.1)]
    covariance_penalty: Annotated[float, Field(ge=0, le=100)]
    numerical_tolerance: Annotated[float, Field(gt=0, le=.00001)]
    max_iterations: Annotated[int, Field(ge=1, le=1000)]
    max_securities: Annotated[int, Field(ge=2, le=512)]
    target_limits: TargetLimits | None = None

    @model_validator(mode='after')
    def conservative_targets(self):
        if self.target_limits is not None and any(
                value > getattr(self, name) for name, value in self.target_limits.model_dump().items()):
            raise ValueError('Construction target limits cannot exceed hard portfolio limits')
        return self

    @model_serializer(mode='wrap')
    def archive_compatible(self, handler):
        data=handler(self)
        if self.target_limits is None:
            data.pop('target_limits',None)
        return data


def coverage(inputs, ids, covariance, policy, nav, *, covariance_already_validated=False):
    if not np.isfinite(nav) or nav <= 0:
        return 'nonpositive_or_unobserved_nav'
    if not ids or [p.security_id for p in inputs] != list(ids) or len(set(ids)) != len(ids):
        return 'risk_identity_order_or_held_coverage_mismatch'
    active=[p for p in inputs if p.held_shares or (p.eligible and p.tradable and p.score is not None)]
    if len(active) > policy.max_securities:
        return 'reference_solver_size_cap'
    matrix = np.asarray(covariance)
    if matrix.shape != (len(ids), len(ids)) or (not covariance_already_validated and
            (not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T,rtol=0,atol=1e-12)
             or np.linalg.eigvalsh(matrix).min() < -1e-12)):
        return 'invalid_covariance'
    for p in inputs:
        active = bool(p.held_shares) or (p.eligible and p.tradable and p.score is not None)
        if active and p.raw_price_usd is None:
            return 'missing_position_valuation'
        if active and p.beta is None:
            return 'missing_beta_coverage'
        if active and policy.max_issuer_gross is not None and p.issuer_id is None:
            return 'missing_issuer_evidence'
        if active and policy.max_sector_net is not None and p.sector_id is None:
            return 'missing_sector_evidence'
    return None


def exposures(inputs, weights, covariance, nav, policy):
    current = np.array([p.held_shares*(p.raw_price_usd or 0.)/nav for p in inputs])
    beta = np.array([p.beta if p.beta is not None else 0. for p in inputs])
    issuer = {key:float(sum(abs(w) for p,w in zip(inputs,weights) if p.issuer_id==key))
              for key in sorted({p.issuer_id for p in inputs if p.issuer_id is not None})}
    sector = {key:float(sum(w for p,w in zip(inputs,weights) if p.sector_id==key))
              for key in sorted({p.sector_id for p in inputs if p.sector_id is not None})}
    values = {'gross':float(np.abs(weights).sum()), 'net':float(weights.sum()), 'beta':float(weights@beta),
              'turnover':float(np.abs(weights-current).sum()), 'name_max':float(np.abs(weights).max(initial=0.)),
              'issuer_gross':issuer, 'sector_net':sector,
              'forecast_variance':float(weights@covariance@weights),
              'retained_short_without_current_locate':[p.security_id for p,w in zip(inputs,weights)
                   if w<0 and (p.borrow_available_shares is None or -w*nav/(p.raw_price_usd or 1.)>p.borrow_available_shares+1e-10)],
              'risk_target':policy.risk_target,'risk_units':policy.risk_units,'horizon_sessions':policy.risk_horizon_sessions}
    tol=policy.numerical_tolerance
    errors=[]
    for name,actual,limit in [('gross',values['gross'],policy.max_gross),('net',abs(values['net']),policy.net_tolerance),
                               ('beta',abs(values['beta']),policy.beta_tolerance),('turnover',values['turnover'],policy.max_turnover),
                               ('name',values['name_max'],policy.max_name)]:
        if actual>limit+tol:errors.append(name)
    if policy.max_issuer_gross is not None and any(v>policy.max_issuer_gross+tol for v in issuer.values()):errors.append('issuer')
    if policy.max_sector_net is not None and any(abs(v)>policy.max_sector_net+tol for v in sector.values()):errors.append('sector')
    for p,w,old in zip(inputs,weights,current):
        if not p.tradable and abs(w-old)>tol:errors.append('untradable_position_changed')
        if (not p.eligible or p.score is None) and (abs(w)>abs(old)+tol or w*old < -tol):errors.append('ineligible_or_unscored_position_increased')
        available=p.borrow_available_shares or 0.
        short_limit=max(available*(p.raw_price_usd or 0.)/nav,-min(old,0.))
        if w < -short_limit-tol:errors.append('short_availability')
    return values,sorted(set(errors))


def project_portfolio(inputs, ids, covariance, nav, policy):
    """Construct buffered targets, keeping the caller's hard policy unchanged.

    The fill/retained-position paths call exposures with the original policy.
    A failed buffered optimization has no unbuffered fallback.
    """
    if policy.target_limits is None:
        return _project_portfolio(inputs,ids,covariance,nav,policy)
    limits=policy.target_limits.model_dump()
    construction=policy.model_copy(update=limits | {'target_limits':None})
    result=_project_portfolio(inputs,ids,covariance,nav,construction)
    evidence={'construction_limits':limits,'hard_limits':{name:getattr(policy,name) for name in limits}}
    if result['target_weights'] is None:
        return result | evidence
    weights=np.array([result['target_weights'][sid] for sid in ids])
    values,violations=exposures(inputs,weights,covariance,nav,policy)
    if violations:
        return {'status':'infeasible','reason':'constructed_target_breaches_hard_limits',
                'violations':violations,'target_weights':None,'fallback':'none'} | evidence
    return result | evidence | {'exposures':values,'hard_violations':[]}


def _project_portfolio(inputs, ids, covariance, nav, policy):
    """Fixed scores shape a target; this is not an expected-return forecast."""
    from scipy.optimize import Bounds, LinearConstraint, minimize
    failure=coverage(inputs,ids,covariance,policy,nav)
    if failure:return {'status':'insufficient_evidence','reason':failure,'target_weights':None,'fallback':'none'}
    active=[i for i,p in enumerate(inputs) if p.held_shares or (p.eligible and p.tradable and p.score is not None)]
    if len(active)!=len(inputs):
        if active:
            sub=_project_portfolio([inputs[i] for i in active],[ids[i] for i in active],covariance[np.ix_(active,active)],nav,policy)
            if sub['target_weights'] is None:return sub
            weights=np.array([sub['target_weights'].get(sid,0.) for sid in ids])
        else:
            sub={'status':'zero_target','methodology':policy.methodology,'iterations':0,'fallback':'none','score_interpretation':policy.score_units}
            weights=np.zeros(len(ids))
        values,violations=exposures(inputs,weights,covariance,nav,policy)
        if violations:return {'status':'infeasible','reason':'expanded_coverage_constraints','violations':violations,'target_weights':None,'fallback':'none'}
        return sub | {'target_weights':dict(zip(ids,map(float,weights))),
                      'target_shares':{p.security_id:float(w*nav/p.raw_price_usd) if p.raw_price_usd else 0. for p,w in zip(inputs,weights)},
                      'exposures':values,'active_security_ids':[ids[i] for i in active],
                      'inactive_zero_weight_ids':[sid for i,sid in enumerate(ids) if i not in active]}
    n=len(inputs)
    current=np.array([p.held_shares*(p.raw_price_usd or 0.)/nav for p in inputs])
    beta=np.array([p.beta if p.beta is not None else 0. for p in inputs])
    scores=np.array([p.score if p.score is not None and p.eligible and p.tradable else 0. for p in inputs])
    mask=np.array([p.score is not None and p.eligible and p.tradable for p in inputs])
    target=np.zeros(n)
    if mask.any():
        design=np.column_stack([np.ones(mask.sum()),beta[mask]])
        target[mask]=scores[mask]-design@np.linalg.lstsq(design,scores[mask],rcond=None)[0]
        norm=np.abs(target).sum()
        if norm>1e-12:target*=policy.max_gross/norm
    low,high=np.full(n,-policy.max_name),np.full(n,policy.max_name)
    for i,p in enumerate(inputs):
        short_limit=max((p.borrow_available_shares or 0.)*(p.raw_price_usd or 0.)/nav,-min(current[i],0.))
        low[i]=max(low[i],-short_limit)
        if not p.eligible or p.score is None:
            low[i]=max(low[i],min(current[i],0.));high[i]=min(high[i],max(current[i],0.))
        if not p.tradable:
            low[i]=high[i]=current[i]
    if np.any(low>high) or np.any(np.abs(current[np.array([not p.tradable for p in inputs])])>policy.max_name+policy.numerical_tolerance):
        return {'status':'infeasible','reason':'frozen_positions_breach_bounds','target_weights':None,'fallback':'none'}
    # x=(weights, absolute weights, absolute changes). Bounds and linear
    # epigraphs impose gross/name/issuer/turnover without sign approximations.
    rows,lower,upper=[],[],[]
    def constraint(w=None,a=None,t=None,lo=-np.inf,hi=np.inf):
        rows.append(np.concatenate([np.zeros(n) if w is None else w,np.zeros(n) if a is None else a,np.zeros(n) if t is None else t]))
        lower.append(lo);upper.append(hi)
    eye=np.eye(n)
    for i in range(n):
        constraint(w=eye[i],a=-eye[i],hi=0);constraint(w=-eye[i],a=-eye[i],hi=0)
        constraint(w=eye[i],t=-eye[i],hi=current[i]);constraint(w=-eye[i],t=-eye[i],hi=-current[i])
    constraint(a=np.ones(n),hi=policy.max_gross)
    constraint(t=np.ones(n),hi=policy.max_turnover)
    constraint(w=np.ones(n),lo=-policy.net_tolerance,hi=policy.net_tolerance)
    if not (policy.beta_tolerance==policy.net_tolerance==0 and np.allclose(beta,beta[0],rtol=0,atol=1e-12)):
        constraint(w=beta,lo=-policy.beta_tolerance,hi=policy.beta_tolerance)
    if policy.max_issuer_gross is not None:
        for key in sorted({p.issuer_id for p in inputs if p.issuer_id is not None}):
            constraint(a=np.array([float(p.issuer_id==key) for p in inputs]),hi=policy.max_issuer_gross)
    if policy.max_sector_net is not None:
        for key in sorted({p.sector_id for p in inputs if p.sector_id is not None}):
            constraint(w=np.array([float(p.sector_id==key) for p in inputs]),lo=-policy.max_sector_net,hi=policy.max_sector_net)
    scale=max(float(np.diag(covariance).mean()),1e-12)
    def objective(x):
        w=x[:n]
        return float(np.square(w-target).sum()+policy.covariance_penalty*(w@covariance@w)/scale)
    def gradient(x):
        return np.r_[2*(x[:n]-target)+2*policy.covariance_penalty*(covariance@x[:n])/scale,np.zeros(2*n)]
    matrix=np.asarray(rows);lo=np.asarray(lower);hi=np.asarray(upper);equal=lo==hi
    constraints=[LinearConstraint(matrix[mask],lo[mask],hi[mask]) for mask in (equal,~equal) if mask.any()]
    result=minimize(objective,np.r_[np.clip(current,low,high),np.abs(current),np.zeros(n)],jac=gradient,
                    method='SLSQP',bounds=Bounds(np.r_[low,np.zeros(2*n)],np.r_[high,np.full(n,policy.max_name),np.full(n,8.)]),
                    constraints=constraints,
                    options={'maxiter':policy.max_iterations,'ftol':policy.numerical_tolerance**2,'disp':False})
    weights=result.x[:n]
    values,violations=exposures(inputs,weights,covariance,nav,policy)
    if not result.success or not np.isfinite(weights).all() or violations:
        return {'status':'infeasible' if result.status in (4,8) else 'solver_failed','reason':'constrained_projection_failed',
                'solver_status':int(result.status),'violations':violations,'target_weights':None,'fallback':'none'}
    return {'status':'zero_target' if values['gross']<=policy.numerical_tolerance else 'feasible',
            'methodology':policy.methodology,'target_weights':dict(zip(ids,map(float,weights))),
            'target_shares':{p.security_id:float(w*nav/p.raw_price_usd) if p.raw_price_usd else 0. for p,w in zip(inputs,weights)},
            'exposures':values,'iterations':int(result.nit),'fallback':'none','score_interpretation':policy.score_units}
