"""Explicit provider reconstructions of JKP signals; not exact CRSP/Compustat replication.

Definitions/signs reference bkelly-lab/jkp-data commit
666e8960ed81a664f1e9f189f92925ab2eeffcd5 (MIT source). Provider field/basis,
arrival, currency and security-master differences remain explicit approximations.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
from scipy.stats import rankdata, skew
from .contracts import read_json

FUND_FIELDS=('assets','assetsc','assetsnc','cashneq','debt','debtc','debtnc','equity','liabilities',
 'liabilitiesc','liabilitiesnc','inventory','investmentsc','investmentsnc','receivables','revenue',
 'netinccmn','netinc','ncfo','ebit','ebitda','intexp','rnd','sgna','capex','ncfinv','ncfcommon','intangibles',
 'ncfdiv','ncfdebt','depamor','gp','opex','fcf','taxexp','ebt','ppnenet','retearn','sharesbas','fxusd')

BLOCKED={
 'emp_gr1':'Historical employee counts are absent from pinned Sharadar fundamentals',
 'sale_emp_gr1':'Historical employee counts are absent',
 'iskew_ff3_21d':'Dated internally matched FF3 return factors are not available',
 'iskew_hxz4_21d':'Dated internally matched HXZ4 return factors are not available',
 'ivol_ff3_21d':'Dated internally matched FF3 return factors are not available',
 'ivol_hxz4_21d':'Dated internally matched HXZ4 return factors are not available',
 'resff3_12_1':'Matched FF3 history and verified residual-momentum construction are not available',
 'resff3_6_1':'Matched FF3 history and verified residual-momentum construction are not available',
 'o_score':'Ohlson real-asset deflator/FFO and exact component mapping are not verified',
 'kz_index':'Exact capital/dividend/Q component mapping is not verified',
 'ival_me':'Required forecast/valuation inputs and exact implementation are not available',
 'eq_dur':'Required long-horizon earnings/valuation forecasts are not available',
 'mispricing_mgmt':'All canonical composite leaves and exact composite procedure are not available',
 'mispricing_perf':'All canonical composite leaves and exact composite procedure are not available',
 'qmj':'Full canonical quality composite and its leaves are not verified',
 'qmj_growth':'Full canonical quality growth composite is not verified',
 'qmj_prof':'Full canonical profitability composite is not verified',
 'qmj_safety':'Full canonical safety composite is not verified',
 'tangibility':'Gross PPE required by the published formula is absent; net PPE is not substituted',
 'ppeinv_gr1a':'Gross PPE required by the published formula is absent',
 'lnoa_gr1a':'Gross/other asset and liability component mapping is not verified',
 'fnl_gr1a':'Dated preferred-stock financing balance is absent',
 'nfna_gr1a':'Dated preferred-stock financing balance is absent',
 'taccruals_at':'Total-accrual financial asset/liability components are incomplete',
 'taccruals_ni':'Total-accrual financial asset/liability components are incomplete',
 'eqpo_me':'Gross equity repurchases/issuance cannot be recovered from net common-stock cash flow',
 'earnings_variability':'Canonical multi-year earnings/cash-earnings volatility construction not verified',
 'ni_ar1':'Canonical multi-year earnings persistence construction not verified',
 'ni_ivol':'Canonical multi-year earnings innovation volatility construction not verified',
 'ni_inc8q':'Eight consecutive year-on-year increases require a verified quarterly sequence',
 'niq_su':'Seasonal earnings surprise history and scale construction not verified',
 'saleq_su':'Seasonal sales surprise history and scale construction not verified',
 'ocfq_saleq_std':'Canonical historical quarterly cash-flow ratio variance construction not verified',
 'rd5_at':'Five-year R&D capital vintages not implemented',
 'eqnpo_12m':'Market-based payout reconstruction not verified; net financing cash-flow yield is separately identified',
}


def ratio(a,b,positive=True):
    a,b=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float))
    valid=np.isfinite(a)&np.isfinite(b)&((b>0) if positive else (b!=0))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=valid)


def fundamental_values(current,lags,quarter,quarter_year,market_cap, *, market_cap_year=None,quarter_previous=None,quarter_year_previous=None):
    """Inputs are already USD-converted, stale/future values masked independently."""
    missing=np.full(len(market_cap),np.nan)
    def f(name,lag=0):return (current if lag==0 else lags.get(lag,{})).get(name,missing)
    def derived(lag=0):
        at=f('assets',lag);cash=f('cashneq',lag);debt=f('debt',lag);eq=f('equity',lag)
        accrual=f('netinccmn',lag)-f('ncfo',lag)
        # Missing R&D stays missing. No assumed zero merely to expand coverage.
        operating=f('ebitda',lag)+f('rnd',lag)
        coa=f('assetsc',lag)-cash;col=f('liabilitiesc',lag)-f('debtc',lag)
        ncoa=f('assetsnc',lag)-f('investmentsnc',lag)
        ncol=f('liabilitiesnc',lag)-f('debtnc',lag)
        return {'at':at,'be':eq,'sale':f('revenue',lag),'inv':f('inventory',lag),'capx':-f('capex',lag),
                'debt':debt,'coa':coa,'col':col,'cowc':coa-col,'ncoa':ncoa,'ncol':ncol,'nncoa':ncoa-ncol,
                'noa':coa+ncoa-col-ncol,'lti':f('investmentsnc',lag),'sti':f('investmentsc',lag),
                'gp':f('gp',lag),'ocf':f('ncfo',lag),'tax':f('taxexp',lag),'cash':cash,
                'op':operating,'cop':operating-accrual,'ope':f('ebitda',lag)-f('intexp',lag),'accrual':accrual,
                'bev':eq+debt-cash}
    d=derived();d1=derived(1);d2=derived(2);d3=derived(3)
    me=market_cap;ev=me+f('debt')-f('cashneq')
    out={'at_me':ratio(f('assets'),me),'be_me':ratio(f('equity'),me),
         'debt_me':ratio(f('debt'),me),'netdebt_me':ratio(f('debt')-f('cashneq'),me),
         'ni_me':ratio(f('netinccmn'),me),'fcf_me':ratio(f('fcf'),me),'ocf_me':ratio(f('ncfo'),me),
         'sale_me':ratio(f('revenue'),me),'rd_me':ratio(f('rnd'),me),
         'at_be':ratio(f('assets'),f('equity')),'cash_at':ratio(f('cashneq'),f('assets')),
         'rd_sale':ratio(f('rnd'),f('revenue')),'gp_at':ratio(f('gp'),f('assets')),
         'gp_atl1':ratio(f('gp'),f('assets',1)),'op_at':ratio(d['op'],f('assets')),
         'op_atl1':ratio(d['op'],f('assets',1)),'cop_at':ratio(d['cop'],f('assets')),
         'cop_atl1':ratio(d['cop'],f('assets',1)),'ope_be':ratio(d['ope'],f('equity')),
         'ope_bel1':ratio(d['ope'],f('equity',1)),'ocf_at':ratio(f('ncfo'),f('assets')),
         'ni_be':ratio(f('netinccmn'),f('equity')),'ebit_sale':ratio(f('ebit'),f('revenue')),
         'ebitda_mev':ratio(f('ebitda'),ev),'bev_mev':ratio(d['bev'],ev),
         'ebit_bev':ratio(f('ebit'),d['bev']),'sale_bev':ratio(f('revenue'),d['bev']),
         'at_turnover':ratio(f('revenue'),(f('assets')+f('assets',1))/2),'opex_at':ratio(f('opex'),f('assets')),
         'pi_nix':np.where(f('ebt')>0,ratio(f('ebt'),f('netinc')),np.nan),
         'oaccruals_at':ratio(d['accrual'],f('assets')),
         'oaccruals_ni':ratio(d['accrual'],np.abs(f('netinccmn'))),
         'noa_at':ratio(d['noa'],f('assets')),
         'ocf_at_chg1':ratio(d['ocf'],d['at'])-ratio(d1['ocf'],d1['at']),
         'dbnetis_at':ratio(f('ncfdebt'),f('assets')),
         'eqnetis_at':ratio(f('ncfcommon'),f('assets')),
         'netis_at':ratio(f('ncfcommon')+f('ncfdebt'),f('assets')),
         'eqnpo_me':ratio(-f('ncfdiv')-f('ncfcommon'),me),
         'div12m_me':ratio(-f('ncfdiv'),me),
         'aliq_at':ratio(d['cash']+.75*d['coa']+.5*(f('assetsnc')-f('intangibles')),d['at'])}
    # No positive financing sign, gross-vs-net, or preferred/minority omission is
    # labeled as an exact JKP match; catalog exposes all provider approximations.
    old_me=missing if market_cap_year is None else market_cap_year
    out['aliq_mat']=ratio(d['cash']+.75*d['coa']+.5*(f('assetsnc')-f('intangibles')),f('assets',1)+old_me-f('equity',1))
    for name,var,lag in [('at_gr1','at',1),('sale_gr1','sale',1),('sale_gr3','sale',3),
                         ('inv_gr1','inv',1),('capx_gr1','capx',1),('capx_gr2','capx',2),('capx_gr3','capx',3),('debt_gr3','debt',3)]:
        out[name]=ratio(d[var],{1:d1,2:d2,3:d3}[lag][var])-1
    for var in ('be','coa','col','cowc','ncoa','ncol','nncoa','noa','inv','lti','sti','tax'):
        # JKP source uses current assets for these generic _gr1a signals, except
        # NOA which explicitly uses lagged assets.
        out[var+'_gr1a']=ratio(d[var]-d1[var],d1['at'] if var=='noa' else d['at'])
    for name,a,bvar in [('dgp_dsale','gp','sale'),('dsale_dinv','sale','inv'),
                         ('dsale_drec','sale','receivables'),('dsale_dsga','sale','sgna')]:
        def value(var,lag=0):return derived(lag)[var] if var in d else f(var,lag)
        out[name]=ratio(value(a),(value(a,1)+value(a,2))/2)-ratio(value(bvar),(value(bvar,1)+value(bvar,2))/2)
    out['capex_abn']=ratio(ratio(d['capx'],d['sale']),
        (ratio(d1['capx'],d1['sale'])+ratio(d2['capx'],d2['sale'])+ratio(d3['capx'],d3['sale']))/3)-1
    q=lambda field:quarter.get(field,missing)
    q1=lambda field:quarter_year.get(field,missing)
    qp=lambda field:(quarter_previous or {}).get(field,missing)
    q1p=lambda field:(quarter_year_previous or {}).get(field,missing)
    out.update(niq_at=ratio(q('netinccmn'),qp('assets')),niq_be=ratio(q('netinccmn'),qp('equity')),
               saleq_gr1=ratio(q('revenue'),q1('revenue'))-1)
    out['niq_at_chg1']=out['niq_at']-ratio(q1('netinccmn'),q1p('assets'))
    out['niq_be_chg1']=out['niq_be']-ratio(q1('netinccmn'),q1p('equity'))
    out['z_score']=(1.2*ratio(f('assetsc')-f('liabilitiesc'),f('assets'))+1.4*ratio(f('retearn'),f('assets'))
        +3.3*ratio(f('ebit'),f('assets'))+.6*ratio(me,f('liabilities'))+ratio(f('revenue'),f('assets')))
    # Piotroski requires all nine components; missing information remains missing.
    roa=ratio(f('netinccmn'),f('assets'));old_roa=ratio(f('netinccmn',1),f('assets',1))
    nine=[(roa,0.,'>'),(f('ncfo'),0.,'>'),(roa,old_roa,'>'),(f('ncfo'),f('netinccmn'),'>'),
          (ratio(f('debtnc'),f('assets')),ratio(f('debtnc',1),f('assets',1)),'<'),
          (ratio(f('assetsc'),f('liabilitiesc')),ratio(f('assetsc',1),f('liabilitiesc',1)),'>'),
          (f('ncfcommon'),0.,'<='),(ratio(f('gp'),f('revenue')),ratio(f('gp',1),f('revenue',1)),'>'),
          (ratio(f('revenue'),f('assets')),ratio(f('revenue',1),f('assets',1)),'>')]
    score=np.zeros(len(me));known=np.ones(len(me),dtype=bool)
    for a,bv,op in nine:
        known &= np.isfinite(a)&np.isfinite(bv)
        score+=np.asarray(a<bv if op=='<' else a<=bv if op=='<=' else a>bv,dtype=float)
    out['f_score']=np.where(known,score,np.nan)
    return {name:np.where(np.isfinite(value),value,np.nan) for name,value in out.items()}


MARKET_NAMES={'age','market_equity','prc','ret_1_0','ret_3_1','ret_6_1','ret_9_1','ret_12_1','ret_12_7','ret_60_12',
 'prc_highprc_252d','rvol_21d','rmax1_21d','rmax5_21d','rmax5_rvol_21d','rskew_21d','dolvol_126d',
 'dolvol_var_126d','ami_126d','turnover_126d','turnover_var_126d','beta_60m','beta_dimson_21d',
 'betadown_252d','betabab_1260d','corr_1260d','ivol_capm_21d','ivol_capm_252d','iskew_capm_21d',
 'bidaskhl_21d','zero_trades_21d','zero_trades_126d','zero_trades_252d','chcsho_12m','eqnpo_12m'}


def market_values(index,adjusted,split_close,dollar_volume,high,low,cap,market_returns,monthly_indices,first_index):
    n=adjusted.shape[1];missing=np.full(n,np.nan);out={}
    ret=np.full_like(adjusted[max(0,index-1260):index+1],np.nan)
    start=max(0,index-1260);p=adjusted[start:index+1]
    if len(p)>1:ret[1:]=ratio(p[1:],p[:-1])-1
    mr=market_returns[start:index+1]
    months=[i for i in monthly_indices if i<=index]
    month_prices=adjusted[months]
    def mom(old,recent):
        if len(months)<=old:return missing.copy()
        values=month_prices[-old-1:len(month_prices)-recent]
        result=ratio(month_prices[-recent-1],month_prices[-old-1])-1
        return np.where(np.isfinite(values).all(axis=0),result,np.nan)
    for old,recent in ((1,0),(3,1),(6,1),(9,1),(12,1),(12,7),(60,12)):
        out[f'ret_{old}_{recent}']=mom(old,recent)
    out['age']=(index-first_index)/21
    out['market_equity']=cap[index]
    # prc is filled from raw prices by the adapter, not the split-adjusted series.
    for days in (21,126,252):
        sample=ret[-days:];valid=np.isfinite(sample);count=valid.sum(axis=0)
        dv=dollar_volume[max(0,index-days+1):index+1]
        minobs={21:15,126:84,252:126}[days]
        if days==21:
            with np.errstate(invalid='ignore',divide='ignore'):
                sd=np.nanstd(sample,axis=0,ddof=1)
                maximum=np.nanmax(sample,axis=0)
                top=np.sort(np.where(valid,sample,-np.inf),axis=0)[-5:]
                topmean=np.where(np.isfinite(top).all(axis=0),top.mean(axis=0),np.nan)
                sk=skew(sample,axis=0,bias=False,nan_policy='omit')
            for name,value in [('rvol_21d',sd),('rmax1_21d',maximum),('rmax5_21d',topmean),
                               ('rmax5_rvol_21d',ratio(topmean,sd)),('rskew_21d',sk)]:
                out[name]=np.where(count>=minobs,value,np.nan)
        if days==126:
            with np.errstate(invalid='ignore',divide='ignore'):
                avg=np.nanmean(dv,axis=0)
                out['dolvol_126d']=np.where(np.isfinite(dv).sum(axis=0)>=84,avg,np.nan)
                out['dolvol_var_126d']=np.where(np.isfinite(dv).sum(axis=0)>=84,ratio(np.nanstd(dv,axis=0,ddof=1),avg),np.nan)
                out['ami_126d']=np.where(count>=84,np.nanmean(ratio(np.abs(sample),dv),axis=0),np.nan)
            turnover=ratio(dv,cap[max(0,index-days+1):index+1])
            out['turnover_126d']=np.where(np.isfinite(turnover).sum(axis=0)>=84,np.nanmean(turnover,axis=0),np.nan)
            out['turnover_var_126d']=np.where(np.isfinite(turnover).sum(axis=0)>=84,ratio(np.nanstd(turnover,axis=0,ddof=1),out['turnover_126d']),np.nan)
        # Liu turnover tiebreaker, with explicit daily-window normalization.
        turn=ratio(dv,cap[max(0,index-days+1):index+1])
        zero=ratio((dv==0).sum(axis=0),np.isfinite(dv).sum(axis=0))*21
        average=np.nanmean(turn,axis=0);good=(count>=minobs)&np.isfinite(average)
        tie=np.full(n,np.nan)
        if good.any():tie[good]=rankdata(-average[good],method='average')/good.sum()/100
        out[f'zero_trades_{days}d']=np.where(good,zero+tie,np.nan)
    def regress(window):
        y=ret[-window:];x=mr[-window:,None]
        good=np.isfinite(y)&np.isfinite(x);counts=good.sum(axis=0)
        ym=ratio(np.where(good,y,0.).sum(axis=0),counts)
        xm=ratio(np.where(good,x,0.).sum(axis=0),counts)
        xx=np.where(good,x-xm,0.);yy=np.where(good,y-ym,0.)
        beta=ratio((xx*yy).sum(axis=0),np.square(xx).sum(axis=0))
        residual=np.where(good,yy-xx*beta,np.nan)
        return beta,residual,counts
    for window in (21,252):
        beta,residual,counts=regress(window)
        out[f'ivol_capm_{window}d']=np.where(counts>={21:15,252:126}[window],np.nanstd(residual,axis=0,ddof=1),np.nan)
        if window==21:out['iskew_capm_21d']=np.where(counts>=15,skew(residual,axis=0,bias=False,nan_policy='omit'),np.nan)
    for window,name in ((252,'betadown_252d'),):
        y=ret[-window:];x=mr[-window:,None]
        good=np.isfinite(y)&np.isfinite(x)&(x<0);cnt=good.sum(axis=0)
        ym=ratio(np.where(good,y,0.).sum(axis=0),cnt);xm=ratio(np.where(good,x,0.).sum(axis=0),cnt)
        xx=np.where(good,x-xm,0.);yy=np.where(good,y-ym,0.)
        out[name]=np.where(cnt>=60,ratio((xx*yy).sum(axis=0),np.square(xx).sum(axis=0)),np.nan)
    # Dimson lead/lag observations are all inside the already-observed window.
    out['beta_dimson_21d']=missing.copy()
    if len(ret)>=24:
        x=np.column_stack([np.ones(21),mr[-23:-2],mr[-22:-1],mr[-21:]])
        y=ret[-22:-1]
        for j in range(n):
            good=np.isfinite(x).all(axis=1)&np.isfinite(y[:,j])
            if good.sum()>=15 and np.linalg.matrix_rank(x[good])==4:
                out['beta_dimson_21d'][j]=np.linalg.lstsq(x[good],y[good,j],rcond=None)[0][1:].sum()
    # These are documented provider/SPY variants, not full CRSP market replications.
    out['corr_1260d']=missing.copy();out['betabab_1260d']=missing.copy()
    if len(ret)>=1260:
        y=ret[-1260:];x=mr[-1260:]
        y=y[2:]+y[1:-1]+y[:-2];x=(x[2:]+x[1:-1]+x[:-2])[:,None]
        good=np.isfinite(y)&np.isfinite(x);count=good.sum(axis=0)
        ym=ratio(np.where(good,y,0.).sum(axis=0),count);xm=ratio(np.where(good,x,0.).sum(axis=0),count)
        xx=np.where(good,x-xm,0.);yy=np.where(good,y-ym,0.)
        corr=ratio((xx*yy).sum(axis=0),np.sqrt(np.square(xx).sum(axis=0)*np.square(yy).sum(axis=0)))
        out['corr_1260d']=np.where(count>=750,corr,np.nan)
        out['betabab_1260d']=np.where(count>=750,corr*ratio(np.nanstd(ret[-252:],axis=0,ddof=1),np.nanstd(mr[-252:],ddof=1)),np.nan)
    out['beta_60m']=missing.copy()
    if len(months)>=61:
        mret=ratio(month_prices[-60:],month_prices[-61:-1])-1
        mkt=np.array([np.prod(1+market_returns[a+1:z+1])-1 for a,z in zip(months[-61:-1],months[-60:])])[:,None]
        good=np.isfinite(mret)&np.isfinite(mkt);cnt=good.sum(axis=0)
        ym=ratio(np.where(good,mret,0.).sum(axis=0),cnt);xm=ratio(np.where(good,mkt,0.).sum(axis=0),cnt)
        x=np.where(good,mkt-xm,0.);y=np.where(good,mret-ym,0.)
        out['beta_60m']=np.where(cnt>=36,ratio((x*y).sum(axis=0),np.square(x).sum(axis=0)),np.nan)
    # Source split-adjusted OHLC are on one common share basis within a series.
    h=split_close[max(0,index-251):index+1];out['prc_highprc_252d']=ratio(split_close[index],np.nanmax(h,axis=0))
    out['prc_highprc_252d']=np.where(np.isfinite(h).sum(axis=0)>=126,out['prc_highprc_252d'],np.nan)
    h=high[max(0,index-21):index+1];l=low[max(0,index-21):index+1]
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        log_range=np.log(ratio(h,l))
        bs=np.square(log_range[1:])+np.square(log_range[:-1])
        gs=np.square(np.log(ratio(np.maximum(h[1:],h[:-1]),np.minimum(l[1:],l[:-1]))))
        alpha=(np.sqrt(2*bs)-np.sqrt(bs))/(3-2*np.sqrt(2))-np.sqrt(gs/(3-2*np.sqrt(2)))
        alpha=np.clip(alpha,0,20);spread=2*np.expm1(alpha)/(1+np.exp(alpha))
        out['bidaskhl_21d']=np.where(np.isfinite(spread).sum(axis=0)>=15,np.nanmean(spread,axis=0),np.nan)
    for lo,hi in ((1,1),(2,5),(6,10),(11,15),(16,20)):
        for suffix in ('an','na'):
            name=f'seas_{lo}_{hi}{suffix}'
            if len(months)<hi*12:out[name]=missing.copy();continue
            lags=list(range(lo*12-1,hi*12,12)) if suffix=='an' else [i for i in range((lo-1)*12,hi*12) if i%12!=11]
            observations=[]
            for lag in lags:
                j=len(months)-1-lag
                if j<1:observations.append(missing)
                else:observations.append(ratio(month_prices[j],month_prices[j-1])-1)
            sample=np.array(observations)
            out[name]=np.where(np.isfinite(sample).all(axis=0),np.mean(sample,axis=0),np.nan)
    if len(months)>12:
        shares=ratio(cap[index],split_close[index]);old_idx=months[-13]
        old_shares=ratio(cap[old_idx],split_close[old_idx])
        out['chcsho_12m']=ratio(shares,old_shares)-1
    return out


def catalog():
    return read_json(Path(__file__).with_name('jkp_catalog.json'))
