"""Fixed long-history factor/theme/model study and provisional implementation account."""
from __future__ import annotations

from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import duckdb
from .contracts import ResearchError,read_json
from .factor_inputs import FUND_FIELDS,load_inputs,save_json
from .jkp_signals import catalog,fundamental_values,market_values,ratio
from .factor_construction import estimate_risk,construct_factor,combine_tree,implied_returns,annual_volatility
from .large_portfolio import optimize
from .factor_account import DiagnosticAccount,position_exposures,breaches,risk_at


def forward_interval(weights,prices,decision,entry,exit):
    active=abs(weights)>1e-10
    if not active.any():return {'return_complete':0.,'missing_names':0,'known_component_pnl':0.}
    window=prices[entry:exit+1,active]
    full=np.isfinite(window).all(axis=0)&np.isfinite(prices[decision,active])&(prices[decision,active]>0)
    units=weights[active]/prices[decision,active]
    pnl=np.where(full,units*(prices[exit,active]-prices[entry,active]),0.)
    return {'return_complete':float(pnl.sum()) if full.all() else None,'missing_names':int((~full).sum()),
            'known_component_pnl':float(pnl.sum())}


def evaluate_platform(inputs,fundamental_path,output,settings):
    inputs,output=Path(inputs),Path(output)
    manifest=load_inputs(inputs)
    spec=read_json(settings);np.random.seed(spec['seed'])
    config=spec['policy']
    identities=read_json(inputs/'identities.json');names=[str(x['security_id']) for x in identities]
    lookup={sid:i for i,sid in enumerate(names)};n=len(names)
    industries=np.array([x['industry'] for x in identities]);first=np.array([x['first_index'] for x in identities])
    calendar=pd.read_csv(inputs/'calendar.csv');calendar['date']=pd.to_datetime(calendar.date)
    dates=calendar.date.dt.strftime('%Y-%m-%d').tolist();month_idx=np.where(calendar.month_end)[0].tolist()
    maps={name:np.load(inputs/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in
          ('adjusted','split_close','raw','dollar_volume','high','low','cap')}
    spy=np.load(inputs/'benchmark.npy',allow_pickle=False)
    market=np.r_[np.nan,spy[1:]/spy[:-1]-1]
    db=duckdb.connect(config={'threads':'1','memory_limit':'2GB'})
    universe=db.execute('SELECT * FROM read_parquet(?) ORDER BY date,security_id',[str(inputs/'universe.parquet')]).df()
    fundamental=db.execute('SELECT * FROM read_parquet(?) ORDER BY date,security_id',[str(fundamental_path)]).df()
    if fundamental.duplicated(['date','security_id']).any():raise ResearchError('Repeated feature keys')
    grouped={str(day.date()):g.set_index(g.security_id.astype(str)) for day,g in fundamental.groupby('date',sort=False)}
    members={str(day.date()):set(g.security_id.astype(str)) for day,g in universe.groupby('date',sort=False)}
    entries=catalog()['items'];supported=[x for x in entries if x['implementation_status']=='provider_reconstruction']
    factor_names=[x['jkp_id'] for x in supported];themes={x['jkp_id']:x['theme'] for x in supported}
    coverage={x['jkp_id']:{'computed_months':0,'nonzero_months':0,'finite_stock_months':0,'first_nonzero':None,
                         'max_raw_names':0} for x in entries}
    account=DiagnosticAccount(n);beta_global=np.zeros(n);months=[];factor_diagnostics=[];forward=[]
    risk_supported_global=np.zeros(n,dtype=bool)
    risk_context=None
    first_day=next(i for i,d in enumerate(dates) if d>=manifest['start'])
    final_idx=len(dates)-1;target_dates=sorted(members)
    boundary={day:(target_dates[j+1] if j+1<len(target_dates) else None) for j,day in enumerate(target_dates)}
    index_by_day={d:i for i,d in enumerate(dates)}
    for i in range(first_day,final_idx+1):
        day=dates[i]
        price=np.asarray(maps['adjusted'][i]);dv=np.asarray(maps['dollar_volume'][i])
        values,violations,resolved=account.step(i,day,price,dv,beta_global,industries,risk_context)
        if i==final_idx-1:
            account.schedule(i,np.zeros(n),'predeclared_final_close')
            account.controls.append({'date':day,'status':'final_close_priority'})
            continue
        if day in members:
            selected=members[day];held={names[j] for j in np.where(abs(account.units)>1e-12)[0]}
            ids=sorted(selected|held);idx=np.array([lookup[sid] for sid in ids]);industry=industries[idx]
            p=np.asarray(maps['adjusted'][max(0,i-252):i+1,idx])
            history=ratio(p[1:],p[:-1])-1
            b,d,beta,supported_risk=estimate_risk(history,market[max(0,i-251):i+1],
                minimum=config['risk']['minimum_observations'],shrinkage=config['risk']['diagonal_shrinkage'])
            beta_safe=np.nan_to_num(beta)
            beta_global[idx[supported_risk]]=beta[supported_risk]
            risk_context=(idx,b,d,supported_risk)
            risk_supported_global[:]=False;risk_supported_global[idx[supported_risk]]=True
            marked_weights=account.units*np.nan_to_num(account.marks)/account.value() if account.value()>0 else np.zeros(n)
            values=position_exposures(marked_weights,beta_global,industries)
            values['forecast_vol']=risk_at(marked_weights,risk_context);violations=breaches(values)
            account.daily[-1].update(beta=values['beta'],annual_ex_ante_vol=values['forecast_vol'],hard_violations=','.join(violations))
            group=grouped[day].reindex(ids)
            def fields(prefix):return {name:group[prefix+name].to_numpy(dtype=float) for name in FUND_FIELDS}
            old_index=month_idx[month_idx.index(i)-12] if month_idx.index(i)>=12 else None
            old_cap=np.asarray(maps['cap'][old_index,idx]) if old_index is not None else np.full(len(idx),np.nan)
            with warnings.catch_warnings():
                for message in ('Mean of empty slice','All-NaN slice encountered','Degrees of freedom <= 0 for slice',
                                'After omitting NaNs, one or more axis-slices of one or more sample arguments is too small',
                                'Precision loss occurred in moment calculation'):
                    warnings.filterwarnings('ignore',message=message)
                raw=fundamental_values(fields(''),{j:fields(f'l{j}_') for j in (1,2,3)},fields('q_'),fields('q1_'),
                    np.asarray(maps['cap'][i,idx]),market_cap_year=old_cap,
                    quarter_previous=fields('qp_'),quarter_year_previous=fields('q1p_'))
                market_raw=market_values(i,*(np.asarray(maps[name][:,idx]) for name in
                    ('adjusted','split_close','dollar_volume','high','low','cap')),market,month_idx,first[idx])
            market_raw['prc']=np.asarray(maps['raw'][i,idx]);raw.update(market_raw)
            factors={};raw_matrix=[];weight_matrix=[]
            for entry in supported:
                name=entry['jkp_id']
                if name not in raw:raise ResearchError('Catalog claims an unimplemented raw signal: '+name)
                value=np.array(raw[name],copy=True)
                value[[sid not in selected for sid in ids]]=np.nan
                # Coverage masks do not redefine membership or consult outcomes.
                weights,detail=construct_factor(value,entry['sign'],industry,beta,b,d,
                    minimum_group=config['construction']['minimum_industry_names'],
                    target_vol=config['construction']['ideal_factor_annual_volatility'])
                factors[name]=weights;raw_matrix.append(value);weight_matrix.append(weights)
                c=coverage[name];c['computed_months']+=1;c['finite_stock_months']+=int(np.isfinite(value).sum())
                c['max_raw_names']=max(c['max_raw_names'],int(np.isfinite(value).sum()))
                if np.any(weights):
                    c['nonzero_months']+=1
                    if c['first_nonzero'] is None:c['first_nonzero']=day
                factor_diagnostics.append({'date':day,'factor':name,**detail})
            model,theme_weights,missing_themes=combine_tree(factors,themes,b,d,
                target_vol=config['construction']['ideal_theme_annual_volatility'])
            model_vol=annual_volatility(b,d,model)
            if model_vol>1e-12:model*=config['construction']['ideal_model_annual_volatility']/model_vol
            mu=implied_returns(b,d,model,risk_aversion=config['expected_returns']['risk_aversion_gamma'],
                               horizon=config['expected_returns']['horizon_sessions'])
            nav=account.daily[-1]['nav_observed_usd']
            eligible=np.array([sid in selected for sid in ids])&supported_risk&(industry!='UNKNOWN')
            current=np.zeros(len(idx)) if nav is None or nav<=0 else account.units[idx]*np.nan_to_num(price[idx])/nav
            if day==target_dates[-1]:
                result={'status':'right_censored','weights':None}
            elif nav is None or not np.all(supported_risk[abs(account.units[idx])>1e-10]):
                result={'status':'blocked_account_valuation_or_held_risk','weights':None}
            else:
                limits=config['optimization']
                result=optimize(mu,current,industry,beta_safe,b,d,eligible,max_gross=limits['inner_gross'],
                    max_name=limits['inner_name'],max_turnover=limits['inner_turnover'],target_vol=limits['annual_vol_cap'],
                    trading_cost=.0006,annual_borrow=.03,horizon_calendar_days=limits['horizon_borrow_days'],
                    time_limit=limits['solver_time_limit_seconds'])
            actual_weights=result.pop('weights',None)
            if actual_weights is not None:
                actual_weights[~eligible&(abs(current)<1e-12)]=0.
                target=np.zeros(n)
                target[idx]=np.divide(actual_weights*nav,price[idx],out=np.zeros(len(idx)),where=np.isfinite(price[idx])&(price[idx]>0))
                account.schedule(i,target,'monthly_alpha')
            record={'date':day,'universe_names':len(selected),'risk_only_held_names':len(held-selected),
                    'raw_factors_implemented':len(supported),'nonzero_factors':sum(np.any(v) for v in factors.values()),
                    'active_themes':len(theme_weights),'missing_themes':','.join(missing_themes),
                    'risk_supported_names':int(supported_risk.sum()),'model_annual_vol':annual_volatility(b,d,model),
                    'model_ideal_gross':float(abs(model).sum()),'account_nav_observed':nav,'optimizer_status':result['status'],
                    'optimizer_gross':result.get('gross'),'optimizer_turnover':result.get('turnover'),
                    'optimizer_annual_vol':result.get('annual_ex_ante_vol'),'solver_seconds':result.get('solve_seconds')}
            months.append(record)
            # Saved raw/weights are formed before the separate outcome function.
            np.savez_compressed(output/f'month-{day[:7]}.npz',security_ids=np.array(ids),raw=np.array(raw_matrix),
                  factor_weights=np.array(weight_matrix),model_weights=model,expected_returns=mu,
                  optimized_weights=np.full(len(idx),np.nan) if actual_weights is None else actual_weights,
                  beta=beta,risk_diagonal=d,risk_loadings=b)
            next_day=boundary[day]
            if next_day is not None and i+1<=final_idx:
                end_index=min(index_by_day[next_day]+1,final_idx)
                for name,w in list(factors.items())+[('MODEL',model)]:
                    full=np.zeros(n);full[idx]=w
                    observation=forward_interval(full,maps['adjusted'],i,i+1,end_index)
                    forward.append({'date':day,'factor':name,'entry_date':dates[i+1],'exit_date':dates[end_index],**observation})
            print(day,'universe',len(selected),'factors',record['nonzero_factors'],'themes',len(theme_weights),
                  'portfolio',result['status'],flush=True)
        # Reductions take priority over a pending monthly alpha batch.
        if i not in (final_idx-1,final_idx):
            account.daily_control(i,day,values,violations,resolved,final_idx)
    # Store dense trade vectors separately from the compact human audit table.
    filled=[];requested=[];orders=[]
    for order in account.orders:
        filled.append(order.pop('filled_delta'));requested.append(order.pop('requested_delta'));orders.append(order)
    np.savez_compressed(output/'trade-vectors.npz',filled=np.array(filled),requested=np.array(requested),
                        final_units=account.units,security_ids=np.array(names))
    pd.DataFrame(account.daily).to_csv(output/'daily.csv',index=False)
    pd.DataFrame(months).to_csv(output/'months.csv',index=False)
    db.from_df(pd.DataFrame(factor_diagnostics)).write_parquet(str(output/'factor-diagnostics.parquet'))
    db.from_df(pd.DataFrame(forward)).write_parquet(str(output/'factor-forward.parquet'))
    db.close()
    save_json(output/'orders.json',orders);save_json(output/'controls.json',account.controls)
    save_json(output/'coverage.json',[entry|coverage[entry['jkp_id']] for entry in entries])
    complete=[row for row in account.daily if row['valuation_resolved']]
    unresolved=[row for row in account.daily if not row['valuation_resolved']]
    final=account.daily[-1]
    outcome={'methodology':'vendor_adjusted_total_return_unit_diagnostic_v1','start':account.daily[0]['date'],
       'end':final['date'],'sessions':len(account.daily),'months':len(months),'catalog_count':len(entries),
       'implemented_provider_reconstructions':len(supported),'blocked_factors':len(entries)-len(supported),
       'factors_nonzero_at_least_once':sum(v['nonzero_months']>0 for v in coverage.values()),
       'first_execution_date':next((o['date'] for o in orders if o['nonzero_fills']),None),
       'nonzero_fills':sum(o['nonzero_fills'] for o in orders),'orders_status':pd.Series([o['status'] for o in orders]).value_counts().to_dict(),
       'daily_controls':len(account.controls),'unresolved_sessions':len(unresolved),
       'first_unresolved_date':unresolved[0]['date'] if unresolved else None,
       'final_nav_observed_usd':final['nav_observed_usd'],'final_nav_estimate_usd':final['nav_estimate_usd'],
       'complete_path_return':final['nav_observed_usd']/1e6-1 if not unresolved else None,
       'last_fully_observed_nav':complete[-1]['nav_observed_usd'] if complete else None,
       'borrow_usd':sum(x['borrow_usd'] for x in account.daily),'trade_cost_usd':sum(x['trade_cost_usd'] for x in account.daily),
       'final_held_names':final['held_names'],'benchmark':'SPY adjusted total-return proxy for S&P500',
       'benchmark_period_return':float(spy[-1]/spy[first_day]-1),
       'scientific_disposition':'Provisional diagnostics; no exact JKP replication, raw-share or broker execution claim',
       'point_in_time_attested':False,'investment_ready':False,'LLM_calls':0,'provider_calls':0,'holdout_evaluations':0,
       'known_limitations':['Current domestic/primary and SIC2 metadata are backcast, not dated eligibility or industry proof',
          'Provider approximations and missing characteristics are itemized; no WRDS inputs',
          'Ideal factor/model portfolios are uncapped analytical directions; executable targets are separately constrained',
          'A gamma5 inverse-optimization prior is not empirical evidence of expected alpha',
          'Analytical total-return units assume vendor-adjusted reinvestment and do not reproduce stock distributions/settlement',
          'Missing held marks retain positions, censor observed NAV and block new sizing; no fictitious exits',
          'Hypothetical unlimited3% borrow is not actual historical availability/fee evidence',
          'Any valid-interval-only performance is conditional coverage, not a complete long-history return']}
    outcome['known_limitations'].append('The JKP catalog was published/updated after much of this history; this is an academic backcast, not a claim the complete library was historically available or an unexamined holdout')
    save_json(output/'outcome.json',outcome)
    return outcome
