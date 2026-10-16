"""Sparse conic implementation portfolio; independent post-solve hard checks."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from .contracts import ResearchError
from .factor_construction import annual_volatility,industry_constraints


def optimize(mu,current,industry,beta,b,d,eligible, *, max_gross=.8,max_name=.04,max_turnover=1.8,
             target_vol=.1,trading_cost=.0006,annual_borrow=.03,horizon_calendar_days=30,time_limit=30.):
    import clarabel
    mu=np.asarray(mu,dtype=float);current=np.asarray(current,dtype=float);eligible=np.asarray(eligible,dtype=bool)
    n=len(mu)
    if (n<2 or n>2500 or current.shape!=(n,) or eligible.shape!=(n,) or beta.shape!=(n,)
            or not all(np.isfinite(v).all() for v in (mu,current,b,d)) or np.any(d<=0)):
        raise ResearchError('Optimizer inputs have invalid shape, coverage or values')
    if np.any(~np.isfinite(beta)):raise ResearchError('Optimizer beta coverage is incomplete')
    # Variables: w, |w|, |w-current|. Borrow cost = rate*(|w|-w)/2.
    borrow=annual_borrow*horizon_calendar_days/365/2
    q=np.r_[-mu-borrow,np.full(n,borrow),np.full(n,trading_cost)]
    eye=sparse.eye(n,format='csc');zero=sparse.csc_matrix((n,n))
    eq=industry_constraints(industry,beta)
    aeq=sparse.hstack([sparse.csc_matrix(eq),sparse.csc_matrix((len(eq),2*n))],format='csc')
    low=np.full(n,-max_name);high=np.full(n,max_name)
    low[~eligible]=np.minimum(current[~eligible],0.);high[~eligible]=np.maximum(current[~eligible],0.)
    # Unknown/ineligible alpha may only reduce an existing position.
    aineq=sparse.vstack([
        sparse.hstack([eye,-eye,zero]),sparse.hstack([-eye,-eye,zero]),
        sparse.hstack([eye,zero,-eye]),sparse.hstack([-eye,zero,-eye]),
        sparse.hstack([eye,zero,zero]),sparse.hstack([-eye,zero,zero]),
        sparse.csc_matrix(np.r_[np.zeros(n),np.ones(n),np.zeros(n)][None,:]),
        sparse.csc_matrix(np.r_[np.zeros(2*n),np.ones(n)][None,:])],format='csc')
    bineq=np.r_[np.zeros(2*n),current,-current,high,-low,max_gross,max_turnover]
    risk=sparse.vstack([sparse.csc_matrix(b),sparse.diags(np.sqrt(d))],format='csc')*np.sqrt(252)
    asoc=sparse.vstack([sparse.csc_matrix((1,3*n)),
        sparse.hstack([-risk,sparse.csc_matrix((risk.shape[0],2*n))])],format='csc')
    bsoc=np.r_[target_vol,np.zeros(risk.shape[0])]
    a=sparse.vstack([aeq,aineq,asoc],format='csc');rhs=np.r_[np.zeros(len(eq)),bineq,bsoc]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=120;settings.time_limit=time_limit
    settings.max_threads=1;settings.tol_gap_abs=1e-8;settings.tol_feas=1e-9
    solver=clarabel.DefaultSolver(sparse.csc_matrix((3*n,3*n)),q,a,rhs,
        [clarabel.ZeroConeT(len(eq)),clarabel.NonnegativeConeT(len(bineq)),clarabel.SecondOrderConeT(len(bsoc))],settings)
    result=solver.solve()
    status=str(result.status)
    if status!='Solved':return {'status':'solver_failed','solver_status':status,'weights':None,'fallback':'none'}
    w=np.array(result.x[:n]);w[abs(w)<1e-10]=0.
    gross=float(np.abs(w).sum());turnover=float(np.abs(w-current).sum())
    exposure=float(np.abs(eq@w).max(initial=0.));vol=annual_volatility(b,d,w)
    if (gross>max_gross+1e-6 or abs(w).max(initial=0.)>max_name+1e-6 or turnover>max_turnover+1e-6
            or exposure>1e-6 or vol>target_vol+1e-6 or not np.isfinite(w).all()
            or np.any(w<low-1e-6) or np.any(w>high+1e-6)):
        raise ResearchError('Conic solution failed independent hard constraint validation')
    return {'status':'feasible','solver_status':status,'weights':w,'gross':gross,'turnover':turnover,
            'name_max':float(np.abs(w).max(initial=0.)),'industry_constraint_residual':exposure,
            'annual_ex_ante_vol':vol,'expected_return_horizon':float(mu@w),
            'trading_cost_estimate':trading_cost*turnover,'solve_seconds':result.solve_time,'fallback':'none'}
