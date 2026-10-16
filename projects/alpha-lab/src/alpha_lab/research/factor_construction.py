"""Industry-neutral factors, causal PSD risk and explicit reverse optimization."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from .contracts import ResearchError


def estimate_risk(returns, market, *, minimum=126, shrinkage=.3):
    """PSD market + shrunken residual covariance; no forward observations.

    Missing residuals are zero only inside this covariance estimator, with each
    series normalized by its OWN observed count. They are never P&L zero returns.
    Missing-overlap correlations shrink toward zero. Unsupported IDs stay flagged.
    """
    returns=np.asarray(returns,dtype=float);market=np.asarray(market,dtype=float)
    if returns.ndim!=2 or market.shape!=(len(returns),) or not 0 < shrinkage <= 1:
        raise ResearchError('Invalid historical risk window')
    good=np.isfinite(returns) & np.isfinite(market[:,None])
    count=good.sum(axis=0);supported=count>=minimum
    stock_mean=np.divide(np.where(good,returns,0.).sum(axis=0),count,out=np.zeros(returns.shape[1]),where=count>0)
    market_mean=np.divide(np.where(good,market[:,None],0.).sum(axis=0),count,out=np.zeros(returns.shape[1]),where=count>0)
    x=np.where(good,market[:,None]-market_mean,0.);y=np.where(good,returns-stock_mean,0.)
    denom=np.square(x).sum(axis=0)
    beta=np.divide((x*y).sum(axis=0),denom,out=np.zeros(returns.shape[1]),where=denom>1e-16)
    residual=np.where(good,y-x*beta,0.)
    variance=np.divide(np.square(residual).sum(axis=0),np.maximum(count-2,1))
    variance=np.maximum(variance,1e-8)
    # B'B + diag(d) is PSD even with disjoint missingness patterns.
    b=np.sqrt(1-shrinkage)*residual/np.sqrt(np.maximum(count-2,1))
    mvar=np.var(market[np.isfinite(market)],ddof=1)
    if not np.isfinite(mvar) or mvar<=0:raise ResearchError('Market risk history is missing/constant')
    b=np.vstack([b,np.sqrt(mvar)*beta])
    d=shrinkage*variance
    beta=np.where(supported & (denom>1e-16),beta,np.nan)
    return b,d,beta,supported & np.isfinite(beta)


def covariance_product(b,d,weights):
    return b.T@(b@weights)+d*weights


def annual_volatility(b,d,weights):
    return float(np.sqrt(252*(np.square(b@weights).sum()+(d*np.square(weights)).sum())))


def industry_constraints(industry,beta):
    industry=np.asarray(industry);beta=np.asarray(beta,dtype=float)
    rows=[]
    for group in sorted(set(industry)):
        mask=industry==group
        rows.append(mask.astype(float))
        values=np.where(mask,np.nan_to_num(beta),0.)
        # Demeaning improves conditioning; a constant industry beta is already
        # neutralized by its dollar constraint and must not create a rank defect.
        values[mask]-=values[mask].mean()
        norm=np.linalg.norm(values)
        if norm>1e-8:rows.append(values/norm)
    return np.array(rows)


def construct_factor(raw,sign,industry,beta,b,d, *, minimum_group=20,target_vol=.1):
    raw=np.asarray(raw,dtype=float);industry=np.asarray(industry)
    if sign not in (-1,1) or raw.shape!=beta.shape or len(industry)!=len(raw):
        raise ResearchError('Factor identity/sign/dimension mismatch')
    weights=np.zeros(len(raw));usable=np.isfinite(raw)&np.isfinite(beta)&(industry!='UNKNOWN')
    accepted=0
    for group in sorted(set(industry[usable])):
        idx=np.where(usable&(industry==group))[0]
        if len(idx)<minimum_group:continue
        ranks=rankdata(sign*raw[idx],method='average')
        sd=ranks.std(ddof=0)
        if sd<=1e-12:continue
        z=(ranks-ranks.mean())/sd
        design=np.column_stack([np.ones(len(idx)),beta[idx]-beta[idx].mean()])
        residual=z-design@np.linalg.lstsq(design,z,rcond=1e-10)[0]
        weights[idx]=residual;accepted+=1
    volatility=annual_volatility(b,d,weights)
    if volatility>1e-12:weights*=target_vol/volatility
    else:weights[:]=0.
    constraints=industry_constraints(industry,np.nan_to_num(beta))
    residual=float(np.abs(constraints@weights).max(initial=0.))
    if residual>1e-8:raise ResearchError('Constructed factor lost industry dollar/beta neutrality')
    return weights,{'usable_names':int(usable.sum()),'active_names':int(np.count_nonzero(weights)),
                    'accepted_industries':accepted,'annual_ex_ante_vol':annual_volatility(b,d,weights),
                    'industry_constraint_residual':residual,'ideal_unconstrained_factor_not_executable':True}


def combine_tree(factors,themes,b,d,target_vol=.1):
    """Equal factor weights within themes; equal theme weights across the model."""
    theme_weights={}
    for theme in sorted(set(themes.values())):
        leaves=[w for name,w in factors.items() if themes[name]==theme and np.any(w)]
        if not leaves:continue
        w=np.mean(leaves,axis=0);vol=annual_volatility(b,d,w)
        if vol>1e-12:theme_weights[theme]=w*target_vol/vol
    if not theme_weights:return np.zeros(len(d)),{},[]
    model=np.mean(list(theme_weights.values()),axis=0)
    vol=annual_volatility(b,d,model)
    if vol>1e-12:model*=target_vol/vol
    return model,theme_weights,sorted(set(themes.values())-theme_weights.keys())


def implied_returns(b,d,model, *, risk_aversion,horizon):
    if risk_aversion<=0 or horizon<1:raise ResearchError('Require intentional expected-return scale and horizon')
    return risk_aversion*horizon*covariance_product(b,d,model)
