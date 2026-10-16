"""Independent arithmetic, causality and constraint references for the platform."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from alpha_lab.research.contracts import ResearchError
from alpha_lab.research.universes import UniversePolicy,select_members
from alpha_lab.research.factor_construction import (estimate_risk,annual_volatility,construct_factor,
    combine_tree,implied_returns,covariance_product)
from alpha_lab.research.large_portfolio import optimize
from alpha_lab.research.jkp_signals import FUND_FIELDS,ratio,fundamental_values,market_values
from alpha_lab.research.factor_inputs import asof_fundamentals
from alpha_lab.research.factor_account import DiagnosticAccount
from alpha_lab.research.factor_study import forward_interval


def candidates(n=12):
    return pd.DataFrame({'security_id':[str(i).zfill(3) for i in range(n)],'cap_median':np.arange(n,0,-1)*1e9,
        'dollar_volume_median':np.full(n,2e7),'cap_count':63,'quote_count':63,'raw_price':10.,'listing_age':252})


def test_universe_hysteresis_is_not_reset_to_top_n_and_has_no_signal_inputs():
    policy=UniversePolicy(target=10,lower=9,upper=12,entry_rank=8)
    frame=candidates(15);first,summary=select_members(frame,set(),policy)
    assert first.loc[first.member,'security_id'].tolist()==[str(i).zfill(3) for i in range(10)]
    frame.loc[0,'cap_median']=1.5e9
    second,summary=select_members(frame,set(first.loc[first.member,'security_id']),policy)
    assert '000' not in second.loc[second.member,'security_id'].tolist()
    frame=candidates(15);frame.loc[0,'cap_median']=4.5e9 # rank12, still in retention buffer
    second,summary=select_members(frame,set(first.loc[first.member,'security_id']),policy)
    assert '000' in set(second.loc[second.member,'security_id'])
    assert 9<=summary['names']<=12


def test_short_universe_and_missing_metric_do_not_invent_membership_or_future_filters():
    frame=candidates(5);frame.loc[0,'cap_count']=0
    selected,summary=select_members(frame,set(),UniversePolicy(target=10,lower=9,upper=12,entry_rank=8))
    assert summary['shortfall'] and summary['names']==4 and not selected.iloc[0].member
    with pytest.raises(ResearchError):select_members(pd.concat([frame,frame.iloc[:1]]),set(),UniversePolicy())


def test_asof_fundamentals_preserve_vintages_lagged_periods_and_staleness():
    grid=pd.DataFrame({'ticker':['A','A'],'date':pd.to_datetime(['2020-01-31','2020-03-31'])})
    source=pd.DataFrame({'ticker':['A','A','A'],'available_at':pd.to_datetime(['2019-02-01','2020-02-01','2020-03-01']),
       'calendardate':pd.to_datetime(['2018-12-31','2019-12-31','2018-12-31']),
       'assets':[100.,150.,999.],'fxusd':[1.,1.,1.]})
    cur=asof_fundamentals(grid,source)
    assert cur.assets.tolist()==[100.,150.] # an old-period amendment cannot displace the current period
    # Fixed-quarter request can never see its March revision in January.
    fixed=pd.DataFrame({'calendardate':pd.to_datetime(['2019-12-31','2019-12-31'])})
    lag=asof_fundamentals(grid,source,lag_years=1,current=fixed)
    assert lag.assets.tolist()==[100.,999.]
    far=pd.DataFrame({'ticker':['A'],'date':pd.to_datetime(['2023-01-01'])})
    assert np.isnan(asof_fundamentals(far,source).assets.iloc[0])
    fixed=pd.DataFrame({'calendardate':pd.to_datetime(['2020-12-31','2020-12-31'])})
    assert asof_fundamentals(grid,source,lag_years=2,current=fixed).assets.tolist()==[100.,999.]


def test_psd_risk_with_missing_overlap_and_exact_diagonal_shrinkage():
    rng=np.random.default_rng(42);market=rng.normal(0,.01,252)
    returns=market[:,None]+rng.normal(0,.02,(252,8));returns[:80,0]=np.nan;returns[90:180,1]=np.nan
    b,d,beta,usable=estimate_risk(returns,market)
    covariance=b.T@b+np.diag(d)
    assert np.linalg.eigvalsh(covariance).min()>0 and usable.all()
    w=np.arange(8)/20
    assert covariance_product(b,d,w)==pytest.approx(covariance@w)
    returns[:200,2]=np.nan
    assert not estimate_risk(returns,market)[3][2]


def test_industry_factor_equations_and_ten_percent_risk_are_independently_satisfied():
    n=60;industry=np.array(['A']*30+['B']*30);beta=np.tile(np.linspace(.6,1.5,30),2)
    raw=np.sin(np.arange(n)*.7);b=np.array([beta*.01]);d=np.full(n,.0004)
    w,detail=construct_factor(raw,1,industry,beta,b,d)
    for group in ('A','B'):
        assert abs(w[industry==group].sum())<1e-10
        assert abs((w[industry==group]*beta[industry==group]).sum())<1e-10
    assert np.sqrt(252*(w@(b.T@b+np.diag(d))@w))==pytest.approx(.1)
    negative,_=construct_factor(raw,-1,industry,beta,b,d)
    assert negative==pytest.approx(-w)
    zero,_=construct_factor(np.ones(n),1,industry,beta,b,d)
    assert not zero.any()


def test_small_industries_missing_values_and_future_raw_inputs_do_not_rename_universe():
    beta=np.ones(24);industry=np.array(['A']*21+['SMALL']*3);raw=np.arange(24,dtype=float)
    raw[0]=np.nan;raw[1]=np.nan
    w,detail=construct_factor(raw,1,industry,beta,np.zeros((1,24)),np.ones(24)*.0001)
    assert len(w)==24 and not w.any() # both groups below required count


def test_theme_tree_equal_themes_and_inverse_return_units():
    n=60;beta=np.ones(n);industry=np.array(['A']*n);b=np.zeros((1,n));d=np.full(n,.0004)
    a,_=construct_factor(np.sin(np.arange(n)),1,industry,beta,b,d)
    c,_=construct_factor(np.cos(np.arange(n)),1,industry,beta,b,d)
    model,themes,missing=combine_tree({'a':a,'b':a,'c':c},{'a':'ONE','b':'ONE','c':'TWO'},b,d)
    assert len(themes)==2 and not missing
    expected=(a+c)/2;expected*=.1/annual_volatility(b,d,expected)
    assert model==pytest.approx(expected)
    assert implied_returns(b,d,model,risk_aversion=5.,horizon=21)==pytest.approx(5*21*d*model)


def test_conic_optimizer_hand_reference_and_ineligible_reduction_only():
    pytest.importorskip('clarabel')
    n=4;industry=np.array(['A']*n);beta=np.ones(n);b=np.zeros((1,n));d=np.full(n,.0001)
    answer=optimize(np.array([.02,.01,-.01,-.02]),np.zeros(n),industry,beta,b,d,np.ones(n,dtype=bool),
                    max_gross=.8,max_name=.2,target_vol=.1,trading_cost=0.,annual_borrow=0.)
    assert answer['weights']==pytest.approx([.2,.2,-.2,-.2],abs=1e-6)
    eligible=np.array([False,True,True,True])
    answer=optimize(np.array([.02,.01,-.01,-.02]),np.zeros(n),industry,beta,b,d,eligible,
                    max_gross=.8,max_name=.2,target_vol=.1,trading_cost=0.,annual_borrow=0.)
    assert abs(answer['weights'][0])<1e-7


def test_fundamental_sign_units_and_missing_is_not_filled_with_zero():
    fields={name:np.array([10.,np.nan]) for name in FUND_FIELDS}
    fields.update(assets=np.array([100.,100.]),gp=np.array([40.,40.]),netinccmn=np.array([8.,8.]),
                  ncfo=np.array([9.,9.]),capex=np.array([-10.,-10.]),fxusd=np.ones(2))
    old={k:v/2 for k,v in fields.items()}
    result=fundamental_values(fields,{1:old,2:old,3:old},fields,old,np.array([200.,200.]))
    assert result['gp_at']==pytest.approx([.4,.4])
    assert result['at_gr1']==pytest.approx([1.,1.])
    assert result['oaccruals_at']==pytest.approx([-.01,-.01])
    assert np.isnan(result['op_at'][1])
    assert np.isnan(ratio([1.],[0.]))[0]


def test_quarter_profitability_uses_prior_quarter_denominators():
    current={n:np.array([10.]) for n in FUND_FIELDS}
    q=current|{'netinccmn':np.array([5.]),'assets':np.array([900.])}
    previous=current|{'assets':np.array([100.]),'equity':np.array([50.])}
    result=fundamental_values(current,{1:current,2:current,3:current},q,q,np.array([200.]),
        quarter_previous=previous,quarter_year_previous=previous,market_cap_year=np.array([100.]))
    assert result['niq_at'][0]==pytest.approx(.05)
    assert result['niq_be'][0]==pytest.approx(.1)


def test_catalog_has_every_published_factor_and_explicit_blocked_reasons():
    from alpha_lab.research.jkp_signals import catalog
    entries=catalog()['items']
    assert len(entries)==153 and len({e['jkp_id'] for e in entries})==153
    assert len({e['theme'] for e in entries})==13
    assert all(e['sign'] in (-1,1) and e['exact_JKP_replication'] is False for e in entries)
    assert all(e['blocked_reason'] for e in entries if e['implementation_status']=='blocked')


def test_market_window_is_future_invariant_and_seasonality_uses_calendar_lags():
    import warnings
    n=3;length=1600;rng=np.random.default_rng(9)
    adjusted=100*np.exp(np.cumsum(rng.normal(0,.01,(length,n)),axis=0))
    close=adjusted.copy();dv=np.full((length,n),1e7);high=close*1.01;low=close*.99;cap=np.full((length,n),1e9)
    market=rng.normal(0,.01,length);months=list(range(20,length,21))
    i=months[30]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        first=market_values(i,adjusted,close,dv,high,low,cap,market,months,np.zeros(n))
        changed=adjusted.copy();changed[i+1:]*=9
        second=market_values(i,changed,close,dv,high,low,cap,market,months,np.zeros(n))
    for key in first:np.testing.assert_allclose(first[key],second[key],equal_nan=True)
    lag=months[30-11];expected=adjusted[lag]/adjusted[months[30-12]]-1
    assert first['seas_1_1an']==pytest.approx(expected)


def test_account_fixed_next_close_units_and_round_trip_cost_and_borrow_math():
    a=DiagnosticAccount(4);beta=np.ones(4);industry=np.array(['A']*4)
    price=np.full(4,100.);volume=np.full(4,1e8)
    a.step(0,'2020-01-02',price,volume,beta,industry)
    a.schedule(0,np.array([100.,100.,-100.,-100.]),'entry')
    a.step(1,'2020-01-03',price,volume,beta,industry)
    assert not a.orders[0]['violations'] and a.orders[0]['nonzero_fills']==4
    # $40k turnover, commission at slipped execution price; signed legs cancel.
    assert a.daily[1]['trade_cost_usd']==pytest.approx(24.)
    a.schedule(1,np.zeros(4),'close')
    a.step(2,'2020-01-06',price,volume,beta,industry)
    assert not a.units.any()
    assert a.value()==pytest.approx(1e6-48-20000*.03*3/365)


def test_missing_held_price_censors_and_never_executes_stale_exit():
    a=DiagnosticAccount(4);beta=np.ones(4);industry=np.array(['A']*4)
    p=np.full(4,100.);v=np.full(4,1e8)
    a.step(0,'2020-01-02',p,v,beta,industry);a.schedule(0,np.array([100.,100.,-100.,-100.]),'entry')
    a.step(1,'2020-01-03',p,v,beta,industry);a.schedule(1,np.zeros(4),'close')
    p[0]=np.nan;a.step(2,'2020-01-06',p,v,beta,industry)
    assert a.units[0]==100. and a.daily[-1]['nav_observed_usd'] is None
    assert a.orders[-1]['unexecutable_requested_names']==1


def test_forward_diagnostic_missingness_is_not_zero_return_or_position_deletion():
    prices=np.array([[100.,100.],[100.,100.],[110.,np.nan],[120.,120.]])
    result=forward_interval(np.array([.5,-.5]),prices,0,1,3)
    assert result['return_complete'] is None and result['missing_names']==1
    assert result['known_component_pnl']==pytest.approx(.1)


def synthetic_platform_inputs(tmp_path):
    from alpha_lab.equity_calendar import session_schedule
    from alpha_lab.research.factor_inputs import save_json
    from alpha_lab.research.contracts import file_hash,digest
    import duckdb
    root=tmp_path/'inputs';root.mkdir();n=80
    schedule,_=session_schedule('2018-01-02','2020-04-30')
    calendar=pd.DataFrame({'date':pd.to_datetime(schedule.date),'ordinal':np.arange(len(schedule)),
                           'close_utc':schedule.session_close_utc.astype(str)})
    calendar['month_end']=calendar.date.eq(calendar.groupby(calendar.date.dt.to_period('M')).date.transform('max'))
    calendar.to_csv(root/'calendar.csv',index=False)
    rng=np.random.default_rng(17);market=rng.normal(.0001,.009,len(calendar))
    returns=market[:,None]+rng.normal(0,.015,(len(calendar),n))
    p=100*np.cumprod(1+returns,axis=0)
    for name,value in {'adjusted':p,'split_close':p,'raw':p,'high':p*1.01,'low':p*.99,
                        'dollar_volume':np.ones_like(p)*1e8,'cap':np.ones_like(p)*1e10}.items():
        np.save(root/(name+'.npy'),value,allow_pickle=False)
    np.save(root/'benchmark.npy',100*np.cumprod(1+market),allow_pickle=False)
    ids=[{'security_id':str(j),'ticker':str(j),'industry':'A' if j<40 else 'B','first_index':0} for j in range(n)]
    save_json(root/'identities.json',ids)
    rows=[];fund=[]
    for day in calendar.loc[calendar.month_end&calendar.date.ge('2020-01-31'),'date']:
        for j in range(n):
            rows.append({'date':day,'ticker':str(j),'security_id':str(j)})
            record={'date':day,'ticker':str(j),'security_id':str(j),'information_date':day-pd.Timedelta(days=20),'reportperiod':day-pd.Timedelta(days=40)}
            for prefix in ('','l1_','l2_','l3_','q_','q1_','qp_','q1p_'):
                for k,name in enumerate(FUND_FIELDS):
                    record[prefix+name]=1. if name=='fxusd' else 100.+k*3+np.sin(j*.7+k)*10+j
            record['capex']=-100.;fund.append(record)
    db=duckdb.connect();db.from_df(pd.DataFrame(rows)).write_parquet(str(root/'universe.parquet'))
    features=tmp_path/'features.parquet';db.from_df(pd.DataFrame(fund)).write_parquet(str(features));db.close()
    body={'start':'2020-01-31','end':'2020-04-30','files':{p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in root.iterdir()}}
    body['bundle_id']=digest(body);save_json(root/'manifest.json',body)
    save_json(root/'COMPLETE.json',{'bundle_id':body['bundle_id'],'manifest_sha256':file_hash(root/'manifest.json')})
    return root,features


def reference_platform_policy():
    return {'requested_window':{'start':'2020-01-31','end':'2020-04-30'},
      'hard_limits':{'gross':1.,'name':.05,'turnover':2.,'abs_net':.01,'abs_beta':.01,'tolerance':1e-6,
                     'industry_abs_net':.005,'industry_abs_beta':.005,'annual_vol':.1},
      'execution':{'starting_nav_usd':1e6,'one_way_commission_bps':1.,'one_way_slippage_bps':5.,'annual_hypothetical_borrow':.03,
                    'day_count':'ACT365F','cash_credit':0.,'cash_debit':.05,'participation':.01},
      'risk':{'lookback_sessions':252,'minimum_observations':126,'annualization':252,'diagonal_shrinkage':.3,'variance_floor':1e-8},
      'construction':{'minimum_industry_names':20,'ideal_factor_annual_volatility':.1,'ideal_theme_annual_volatility':.1,'ideal_model_annual_volatility':.1},
      'expected_returns':{'risk_aversion_gamma':5.,'horizon_sessions':21},
      'optimization':{'inner_gross':.8,'inner_name':.04,'inner_turnover':1.8,'annual_vol_cap':.1,
                        'horizon_borrow_days':30,'solver_time_limit_seconds':10.}}


def test_native_registered_factor_worker_publication_inspection_and_duplicate_guard(tmp_path,monkeypatch):
    pytest.importorskip('clarabel')
    from alpha_lab.research.factor_inputs import save_json
    from alpha_lab.research.factor_workflow import resolve_platform,run_platform,inspect_platform
    from alpha_lab.research.registry import Registry,initialize_registry
    from alpha_lab.research.signal_contracts import Family,Hypothesis
    from alpha_lab.research.contracts import read_json
    inputs,features=synthetic_platform_inputs(tmp_path)
    spec=resolve_platform(inputs,features,reference_platform_policy(),{'family_id':'native-factor-test','hypothesis_id':'reference',
         'title':'Synthetic fixed factor platform integration','seed':42,'threads':1,'timeout_seconds':60,
         'memory_limit_mb':2048,'max_output_bytes':10000000})
    save_json(tmp_path/'spec.json',spec.model_dump());initialize_registry(tmp_path/'registry.sqlite')
    with Registry(tmp_path/'registry.sqlite') as db:
        db.register(Family(family_id='native-factor-test',premise='Independent synthetic platform integration',max_unique_trials=1,max_attempts_per_trial=2),
                    Hypothesis(family_id='native-factor-test',hypothesis_id='reference',premise='A causal fixed pipeline publishes all coverage and outcome evidence',
                               tests=['Native worker complete publication and hash validation','No credentials or duplicate completed trial']),spec)
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-private')
    result=run_platform(inputs,features,tmp_path/'spec.json',tmp_path/'run',tmp_path/'registry.sqlite')
    assert result['artifacts_verified'] and result['summary']['months']==4
    runtime=read_json(tmp_path/'run/runtime.json')
    assert not runtime['provider_keys_present'] and not runtime['proxy_present']
    assert inspect_platform(tmp_path/'run')['result_hash']==result['result_hash']
    with pytest.raises(ResearchError,match='active|completed'):run_platform(inputs,features,tmp_path/'spec.json',tmp_path/'duplicate',tmp_path/'registry.sqlite')
