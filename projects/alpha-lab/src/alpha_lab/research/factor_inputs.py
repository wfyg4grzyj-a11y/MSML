"""Immutable long-history inputs; membership never consults signal/outcome masks."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
from alpha_lab.equity_calendar import session_schedule
from .contracts import ResearchError, digest, file_hash, read_json
from .jkp_signals import FUND_FIELDS
from .universes import UniversePolicy,select_members


def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def asof_fundamentals(grid,source, *, lag_years=0,lag_months=0,current=None):
    """Latest available vintage for the requested exact standardized fiscal quarter."""
    left=grid[['ticker','date']].copy();left['row']=np.arange(len(left))
    by=['ticker']
    lagged=bool(lag_years or lag_months)
    if lagged:
        left['calendardate']=current['calendardate']-pd.DateOffset(years=lag_years,months=lag_months)
        by+=['calendardate']
        left=left.dropna(subset=['calendardate'])
    right=source.copy()
    if not lagged:
        right=right.sort_values(['ticker','available_at','calendardate'])
        latest_period=right.groupby('ticker').calendardate.cummax()
        right=right.loc[right.calendardate.eq(latest_period)].copy()
    if right.duplicated(by+['available_at']).any():
        # More recent fiscal period is the current report. For a fixed period,
        # competing rows at one information time must agree, not be arbitrarily selected.
        if lagged:raise ResearchError('Ambiguous contemporaneous fundamental vintage')
        right=right.sort_values('calendardate').drop_duplicates(by+['available_at'],keep='last')
    merged=pd.merge_asof(left.sort_values(['date','ticker']),right.sort_values(['available_at','ticker']),
                         left_on='date',right_on='available_at',by=by,direction='backward',allow_exact_matches=True)
    merged=merged.set_index('row').reindex(np.arange(len(grid)))
    future=merged.available_at.gt(pd.Series(grid.date.to_numpy(),index=merged.index))
    if future.any():raise ResearchError('Fundamental as-of join used future information')
    stale=((pd.Series(grid.date.to_numpy(),index=merged.index)-merged.available_at).dt.days.gt(550)
           if not lagged else pd.Series(False,index=merged.index))
    for name in FUND_FIELDS:
        if name not in merged:merged[name]=np.nan
        if name not in ('fxusd','sharesbas'):
            merged[name]=pd.to_numeric(merged[name],errors='coerce')*merged.fxusd
        merged.loc[stale,name]=np.nan
    return merged


def prepare_fundamental_overlay(inputs,sources_path,output):
    """New immutable corrected feature artifact; never rewrite a completed bundle."""
    inputs,output=Path(inputs),Path(output)
    manifest=load_inputs(inputs);sources=read_json(sources_path)
    path=Path('data/market/sharadar/tables/fundamentals')/sources['fundamentals']['version']/'data.parquet'
    if file_hash(path)!=sources['fundamentals']['sha256']:raise ResearchError('Fundamental source changed')
    db=duckdb.connect(config={'threads':'1','memory_limit':'2GB'})
    db.execute('SET enable_progress_bar=false')
    grid=db.execute('SELECT ticker,date,security_id FROM read_parquet(?)',[str(inputs/'universe.parquet')]).df()
    db.register('selected',pd.DataFrame({'ticker':grid.ticker.unique()}))
    fields=','.join(FUND_FIELDS)
    query='SELECT ticker,calendardate,reportperiod,date+INTERVAL 2 DAY AS available_at,'+fields+" FROM read_parquet(?) JOIN selected USING(ticker) WHERE dimension=? AND date<=CAST(? AS DATE)"
    annual=db.execute(query,[str(path),'ART',manifest['end']]).df()
    quarterly=db.execute(query,[str(path),'ARQ',manifest['end']]).df()
    cur=asof_fundamentals(grid,annual);quarter=asof_fundamentals(grid,quarterly)
    blocks=[grid,cur[list(FUND_FIELDS)].reset_index(drop=True)]
    blocks.append(cur[['available_at','reportperiod']].rename(columns={'available_at':'information_date'}).reset_index(drop=True))
    for lag in (1,2,3):
        got=asof_fundamentals(grid,annual,lag_years=lag,current=cur)
        blocks.append(got[list(FUND_FIELDS)].add_prefix(f'l{lag}_').reset_index(drop=True))
    for prefix,got in [('q_',quarter),('q1_',asof_fundamentals(grid,quarterly,lag_years=1,current=quarter)),
                       ('qp_',asof_fundamentals(grid,quarterly,lag_months=3,current=quarter)),
                       ('q1p_',asof_fundamentals(grid,quarterly,lag_years=1,lag_months=3,current=quarter))]:
        blocks.append(got[list(FUND_FIELDS)].add_prefix(prefix).reset_index(drop=True))
    wide=pd.concat(blocks,axis=1)
    if output.exists():raise ResearchError('Refuse to replace an existing feature artifact')
    db.from_df(wide).write_parquet(str(output));db.close()
    record={'parent_bundle_id':manifest['bundle_id'],'fundamental_source_sha256':file_hash(path),
            'feature_sha256':file_hash(output),'rows':len(wide),'implementation_sha256':file_hash(__file__),
            'current_staleness_days':550,'lagged_vintages':'Exact fiscal quarter, latest revision known by decision; not invalidated solely for being intentionally historical',
            'quarter_profitability_denominator':'Prior fiscal quarter assets/equity, including matching year-prior comparison',
            'point_in_time_attested':False,'performance_computed':False}
    save_json(output.with_suffix('.json'),record)
    return record


def prepare_inputs(sources_path,output, *, start='1999-03-31',end='2022-12-30',policy=None):
    policy=policy or UniversePolicy()
    if not '1999-01-01'<=start<end<='2022-12-31':raise ResearchError('Extended research remains entirely before 2023')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=read_json(sources_path)
    required=('stocks','funds','daily','fundamentals','tickers')
    paths={name:Path('data/market/sharadar/tables')/name/sources[name]['version']/'data.parquet' for name in required}
    if not all(file_hash(path)==sources[name]['sha256'] for name,path in paths.items()):raise ResearchError('Raw source snapshot hash mismatch')
    schedule,metadata=session_schedule('1997-12-31',end)
    cal=pd.DataFrame({'date':pd.to_datetime(schedule.date),'ordinal':np.arange(len(schedule)),
                      'close_utc':schedule.session_close_utc.astype(str)})
    cal['month_end']=cal.date.eq(cal.groupby(cal.date.dt.to_period('M')).date.transform('max'))
    cal.to_csv(output/'calendar.csv',index=False)
    db=duckdb.connect(config={'threads':'2','memory_limit':'4GB','temp_directory':str(output/'duckdb-temp')})
    db.execute('SET enable_progress_bar=false')
    db.register('calendar',cal)
    for name,path in paths.items():db.read_parquet(str(path)).create_view(name)
    identity=db.execute('''SELECT ticker,min(permaticker) AS security_id,min(siccode) AS siccode,
       string_agg(DISTINCT exchange,',') AS exchanges FROM tickers
       WHERE category IN ('Domestic Common Stock','Domestic Common Stock Primary Class')
       GROUP BY ticker HAVING count(DISTINCT permaticker)=1 AND count(DISTINCT siccode)<=1''').df()
    if identity.security_id.duplicated().any():
        # Historical rename tickers are not safely merged from a current mapping.
        # Keep every ambiguous mapping in the exclusion record, not as two copies.
        dup=identity.loc[identity.security_id.duplicated(False)].copy()
        db.from_df(dup).write_parquet(str(output/'ambiguous-identities.parquet'))
        identity=identity.loc[~identity.security_id.duplicated(False)]
    identity['security_id']=identity.security_id.astype(str)
    identity['industry']=identity.siccode.map(lambda x:'SIC2-'+str(int(x)//100).zfill(2) if pd.notna(x) and 100<=int(x)<=9999 else 'UNKNOWN')
    identity=identity.loc[identity.exchanges.str.contains('NASDAQ|NYSE|AMEX|BATS',case=False,na=False)].copy()
    db.register('identities',identity)
    print('Preparing independent cap and liquidity windows',flush=True)
    db.execute('''CREATE TEMP TABLE quote_windows AS WITH rows AS (
      SELECT q.ticker,q.date,c.ordinal,q.closeunadj,q.close*q.volume AS dv,
       min(c.ordinal) OVER(PARTITION BY q.ticker) AS first_index
      FROM stocks q JOIN calendar c USING(date) JOIN identities i USING(ticker)
      WHERE isfinite(q.closeunadj) AND q.closeunadj>0 AND isfinite(q.close) AND q.close>0
       AND isfinite(q.volume) AND q.volume>=0), windows AS (
      SELECT ticker,date,closeunadj,ordinal-first_index AS listing_age,first_index,
       median(dv) OVER w AS dollar_volume_median,count(dv) OVER w AS quote_count
      FROM rows WINDOW w AS (PARTITION BY ticker ORDER BY ordinal RANGE BETWEEN 62 PRECEDING AND CURRENT ROW))
      SELECT w.* FROM windows w JOIN calendar c USING(date) WHERE c.month_end AND w.date>=CAST(? AS DATE)''',[start])
    db.execute('''CREATE TEMP TABLE cap_windows AS WITH rows AS (
       SELECT d.ticker,d.date,c.ordinal,d.marketcap*1000000 AS cap FROM daily d
       JOIN calendar c USING(date) JOIN identities i USING(ticker)
       WHERE isfinite(d.marketcap) AND d.marketcap>0), windows AS (
       SELECT ticker,date,median(cap) OVER w AS cap_median,count(cap) OVER w AS cap_count
       FROM rows WINDOW w AS (PARTITION BY ticker ORDER BY ordinal RANGE BETWEEN 62 PRECEDING AND CURRENT ROW))
       SELECT w.* FROM windows w JOIN calendar c USING(date) WHERE c.month_end AND w.date>=CAST(? AS DATE)''',[start])
    candidates=db.execute('''SELECT i.security_id,i.ticker,i.industry,q.date,q.closeunadj AS raw_price,
       q.dollar_volume_median,q.quote_count,q.listing_age,q.first_index,c.cap_median,c.cap_count
       FROM quote_windows q JOIN identities i USING(ticker) LEFT JOIN cap_windows c USING(ticker,date)
       ORDER BY date,security_id''').df()
    members=[];stats=[];audits=[];previous=set();previous_snapshot=None
    for day,group in candidates.groupby('date',sort=True):
        audit,stat=select_members(group,previous,policy)
        audit['date']=day;audits.append(audit)
        selected=audit.loc[audit.member].copy();members.append(selected)
        today=cal.loc[cal.date.eq(day)].iloc[0]
        effective=int(today.ordinal)+1
        snapshot={'date':str(day.date()),'decision_time':today.close_utc,
           'effective_date':str(cal.iloc[effective].date.date()) if effective<len(cal) else None,
           'previous_snapshot':previous_snapshot,'policy_id':stat['policy_id'],
           'members':sorted(selected.security_id),'source_versions':sources}
        previous_snapshot=digest(snapshot)
        stats.append({'date':str(day.date()),**stat,'snapshot_id':previous_snapshot,
                      'decision_time':today.close_utc,'effective_date':snapshot['effective_date'],
                      'previous_snapshot':snapshot['previous_snapshot']})
        previous=set(selected.security_id)
    members=pd.concat(members,ignore_index=True)
    db.from_df(pd.concat(audits,ignore_index=True)).write_parquet(str(output/'universe-audit.parquet'))
    db.from_df(members).write_parquet(str(output/'universe.parquet'))
    save_json(output/'membership-stats.json',stats)
    used=identity.loc[identity.security_id.isin(members.security_id)].sort_values('security_id').reset_index(drop=True)
    used['first_index']=used.ticker.map(candidates.groupby('ticker').first_index.min()).astype(int)
    used.to_json(output/'identities.json',orient='records',indent=2)
    db.register('used',used)
    print(f'Loading extended daily history for {len(used)} permanent identities',flush=True)
    n=len(used);shape=(len(cal),n);indices=dict(zip(used.ticker,np.arange(n)))
    arrays={name:np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype='float64',shape=shape)
            for name in ('adjusted','split_close','raw','dollar_volume','high','low','cap')}
    for value in arrays.values():value[:]=np.nan
    for year in range(1997,int(end[:4])+1):
        year_start=f'{year}-01-01';year_end=min(f'{year}-12-31',end)
        q=db.execute('''SELECT q.ticker,c.ordinal,q.closeadj,q.close,q.closeunadj,q.volume,q.high,q.low
             FROM stocks q JOIN calendar c USING(date) JOIN used u USING(ticker)
             WHERE q.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)''',[year_start,year_end]).df()
        if q.duplicated(['ticker','ordinal']).any():raise ResearchError('Ambiguous daily raw quotes')
        row=q.ordinal.to_numpy();col=q.ticker.map(indices).to_numpy()
        for name,values in [('adjusted',q.closeadj),('split_close',q.close),('raw',q.closeunadj),
                            ('dollar_volume',q.close*q.volume),('high',q.high),('low',q.low)]:
            x=values.to_numpy(dtype=float);valid=np.isfinite(x)&(x>=0 if name=='dollar_volume' else x>0)
            arrays[name][row[valid],col[valid]]=x[valid]
        c=db.execute('''SELECT d.ticker,c.ordinal,d.marketcap*1000000 AS cap FROM daily d
          JOIN calendar c USING(date) JOIN used u USING(ticker)
          WHERE d.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)''',[year_start,year_end]).df()
        if c.duplicated(['ticker','ordinal']).any():raise ResearchError('Ambiguous daily capitalization')
        value=c.cap.to_numpy(dtype=float);good=np.isfinite(value)&(value>0)
        arrays['cap'][c.ordinal.to_numpy()[good],c.ticker.map(indices).to_numpy()[good]]=value[good]
        if year%5==0:print('Daily inputs through',year,flush=True)
    for value in arrays.values():value.flush()
    spy=db.execute('''SELECT c.date,c.ordinal,f.closeadj FROM calendar c LEFT JOIN
         (SELECT date,closeadj FROM funds WHERE ticker='SPY') f USING(date) ORDER BY c.ordinal''').df()
    if len(spy)!=len(cal) or not np.isfinite(spy.closeadj).all() or spy.closeadj.le(0).any():raise ResearchError('S&P500 SPY proxy needs complete official-session coverage')
    np.save(output/'benchmark.npy',spy.closeadj.to_numpy(dtype=float),allow_pickle=False)
    print('Preparing dated fundamental vintages and fiscal-period lags',flush=True)
    grid=members[['ticker','date','security_id']].copy().reset_index(drop=True)
    fields=','.join(FUND_FIELDS)
    annual=db.execute('SELECT ticker,calendardate,reportperiod,date+INTERVAL 2 DAY AS available_at,'+fields+
         " FROM fundamentals JOIN used USING(ticker) WHERE dimension='ART' AND date<=CAST(? AS DATE)",[end]).df()
    quarterly=db.execute('SELECT ticker,calendardate,reportperiod,date+INTERVAL 2 DAY AS available_at,'+fields+
         " FROM fundamentals JOIN used USING(ticker) WHERE dimension='ARQ' AND date<=CAST(? AS DATE)",[end]).df()
    cur=asof_fundamentals(grid,annual)
    blocks=[grid,cur[list(FUND_FIELDS)].reset_index(drop=True),
            cur[['available_at','reportperiod']].rename(columns={'available_at':'information_date'}).reset_index(drop=True)]
    for lag in (1,2,3):
        got=asof_fundamentals(grid,annual,lag_years=lag,current=cur)
        blocks.append(got[list(FUND_FIELDS)].add_prefix(f'l{lag}_').reset_index(drop=True))
    quarter=asof_fundamentals(grid,quarterly)
    quarter_year=asof_fundamentals(grid,quarterly,lag_years=1,current=quarter)
    for prefix,frame in [('q_',quarter),('q1_',quarter_year),
                        ('qp_',asof_fundamentals(grid,quarterly,lag_months=3,current=quarter)),
                        ('q1p_',asof_fundamentals(grid,quarterly,lag_years=1,lag_months=3,current=quarter))]:
        blocks.append(frame[list(FUND_FIELDS)].add_prefix(prefix).reset_index(drop=True))
    wide=pd.concat(blocks,axis=1)
    db.from_df(wide).write_parquet(str(output/'fundamentals.parquet'))
    db.close()
    paths_list=[p for p in output.iterdir() if p.is_file()]
    body={'methodology':'extended_factor_inputs_v1','start':start,'end':end,'source_versions':sources,
          'source_files':{k:str(v.resolve()) for k,v in paths.items()},'universe_policy':policy.model_dump(),
          'names':n,'calendar_sessions':len(cal),'months':len(stats),'source_implementation_sha256':file_hash(__file__),
          'industry':'current_snapshot_SIC2_backcast_not_PIT','issuer':'current_primary_share_class_metadata_not_dated_issuer_proof',
          'availability':'prices/caps assumed usable at close; ART/ARQ filing+2 calendar days, stale550days; current snapshots not replayed arrivals',
          'benchmark':'Sharadar adjusted SPY total-return ETF proxy for S&P500, not exact official index',
          'files':{p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in paths_list},
          'point_in_time_attested':False,'performance_computed':False}
    body['bundle_id']=digest(body);save_json(output/'manifest.json',body)
    save_json(output/'COMPLETE.json',{'bundle_id':body['bundle_id'],'manifest_sha256':file_hash(output/'manifest.json')})
    return body


def load_inputs(path):
    path=Path(path).resolve();manifest=read_json(path/'manifest.json');marker=read_json(path/'COMPLETE.json')
    if (marker!={'bundle_id':manifest['bundle_id'],'manifest_sha256':file_hash(path/'manifest.json')}
            or digest({k:v for k,v in manifest.items() if k!='bundle_id'})!=manifest['bundle_id']):
        raise ResearchError('Unverified factor input completion')
    for name,item in manifest['files'].items():
        p=path/name
        if p.is_symlink() or p.parent!=path or not p.is_file() or p.stat().st_size!=item['bytes'] or file_hash(p)!=item['sha256']:
            raise ResearchError('Factor input inventory changed')
    return manifest
