import json
import pytest
np=pytest.importorskip('numpy')
pd=pytest.importorskip('pandas')
pytest.importorskip('exchange_calendars')
import duckdb
from alpha_lab.equity_quality import audit_panel
from alpha_lab.market_neutral import MarketNeutralBacktest
from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.equity_calendar import session_schedule,research_availability


def quality_panel(tmp_path,*,schema=3,omit_date=None,bad_returns=False):
    panel=tmp_path/'panel';panel.mkdir()
    schedule,meta=session_schedule('2012-01-01','2014-01-31')
    benchmark=schedule[['date']].assign(ticker='SPY',closeadj=100.+np.arange(len(schedule)))
    sources={}
    for table in ('stocks','funds','daily','fundamentals','tickers','actions'):
        path=tmp_path/(table+'.parquet')
        if table=='funds':
            duckdb.from_df(benchmark).write_parquet(str(path))
        else:
            path.write_bytes(b'pinned fixture')
        sources[table]={'path':str(path),'parquet_sha256':file_hash(path)}
    (panel/'sources.json').write_text(json.dumps(sources))
    feature_date = '2014-01-30' if omit_date=='2014-01-31' else '2014-01-31'
    duckdb.from_df(pd.DataFrame({'ticker':['S'],'date':pd.to_datetime([feature_date]),
                                'beta':[1.],'filing_date':pd.to_datetime(['2014-01-20']),
                                'available_date':pd.to_datetime(['2014-01-22'])})).write_parquet(str(panel/'features.parquet'))
    (panel/'actions.parquet').write_bytes(b'empty action fixture')
    dates=pd.DatetimeIndex(schedule.date)
    spy=benchmark.closeadj.pct_change(fill_method=None).to_numpy(dtype=np.float32)
    if omit_date is not None:
        mask=dates!=pd.Timestamp(omit_date);dates=dates[mask];spy=spy[mask]
    if bad_returns:
        spy[10]=.1
    np.savez(panel/'returns.npz',dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
             tickers=np.array(['S']),returns=np.zeros((len(dates),1)),spy=spy)
    names=['sources.json','features.parquet','actions.parquet','returns.npz']
    if schema>=4:
        schedule.to_csv(panel/'calendar.csv',index=False);names.append('calendar.csv')
    manifest={'schema_version':schema,'start':'2014-01-01','end':'2014-01-31',
              'calendar':meta,'availability_assumptions':research_availability(),
              'files':{name:file_hash(panel/name) for name in names}}
    (panel/'panel.json').write_text(json.dumps(manifest))
    return panel


def test_quality_audit_checks_sources_calendar_and_accounting_without_backtesting(tmp_path,monkeypatch):
    panel=quality_panel(tmp_path)
    before=file_hash(panel/'panel.json')
    monkeypatch.setattr(MarketNeutralBacktest,'evaluate',lambda *a,**k:pytest.fail('No performance or holdout evaluation'))
    summary=audit_panel(panel,tmp_path/'audit')
    assert summary['calendar']['panel_sessions_complete'] is True
    assert summary['calendar']['benchmark_returns_match'] is True
    assert summary['raw_sources_verified']==6 and summary['accounting_quality']['minimum_filing_age_days']==11
    assert summary['point_in_time_attested'] is False and summary['investment_ready'] is False
    assert file_hash(panel/'panel.json')==before
    assert summary['open_issues']==5
    with pytest.raises(ValueError,match='immutable'):
        audit_panel(panel,tmp_path/'audit')


@pytest.mark.parametrize('missing',['2012-01-03','2012-06-04','2014-01-31'])
def test_legacy_missing_calendar_edges_and_interior_become_blocking_queue_items(tmp_path,missing):
    panel=quality_panel(tmp_path,omit_date=missing)
    summary=audit_panel(panel,tmp_path/'audit')
    assert summary['calendar']['missing_panel_sessions']==1
    queue=json.loads((tmp_path/'audit'/'quality-queue.json').read_text())
    assert any(item['id']=='panel_session_mismatch' and item['blocks_investment_ready'] for item in queue)


def test_pinned_raw_benchmark_returns_are_checked_independently(tmp_path):
    summary=audit_panel(quality_panel(tmp_path,bad_returns=True),tmp_path/'audit')
    assert summary['calendar']['benchmark_returns_match'] is False


def test_modified_raw_source_cannot_pass_quality_verification(tmp_path):
    panel=quality_panel(tmp_path)
    (tmp_path/'funds.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw-source hash'):
        audit_panel(panel,tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_new_panels_require_complete_calendar_evidence(tmp_path):
    panel=quality_panel(tmp_path,schema=4)
    engine=MarketNeutralBacktest(panel)
    assert engine.calendar_verified is True
    manifest=json.loads((panel/'panel.json').read_text())
    del manifest['files']['calendar.csv']
    (panel/'panel.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='calendar evidence'):
        MarketNeutralBacktest(panel)


def test_calendar_dates_and_early_close_evidence_are_validated_even_with_updated_file_hash(tmp_path):
    panel=quality_panel(tmp_path,schema=4)
    evidence=pd.read_csv(panel/'calendar.csv');evidence.loc[0,'session_close_utc']='2012-01-03 00:00:00+00:00'
    evidence.to_csv(panel/'calendar.csv',index=False)
    manifest=json.loads((panel/'panel.json').read_text());manifest['files']['calendar.csv']=file_hash(panel/'calendar.csv')
    (panel/'panel.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='exchange schedule'):
        MarketNeutralBacktest(panel)


def test_new_calendar_cannot_hide_a_missing_boundary_by_shortening_the_arrays(tmp_path):
    panel=quality_panel(tmp_path,schema=4)
    evidence=pd.read_csv(panel/'calendar.csv').iloc[1:].copy()
    evidence['session']=np.arange(1,len(evidence)+1)
    evidence.to_csv(panel/'calendar.csv',index=False)
    with np.load(panel/'returns.npz') as raw:
        arrays={name:raw[name] for name in raw.files}
    for name in ('dates','returns','spy'):
        arrays[name]=arrays[name][1:]
    np.savez(panel/'returns.npz',**arrays)
    manifest=json.loads((panel/'panel.json').read_text())
    for name in ('calendar.csv','returns.npz'):
        manifest['files'][name]=file_hash(panel/name)
    (panel/'panel.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='exchange schedule'):
        MarketNeutralBacktest(panel)


def test_separate_terminal_evidence_retains_its_original_panel_scope(tmp_path):
    panel=quality_panel(tmp_path)
    terminal=tmp_path/'terminal';terminal.mkdir()
    (terminal/'sources.json').write_text('{}')
    (terminal/'events.json').write_text(json.dumps([{'ticker':'OLD','date':'2014-01-03','unresolved_reasons':['unverified_settlement']}]))
    (terminal/'summary.json').write_text(json.dumps({'event_observations':1,'original_panel_sha256':'original-panel-hash'}))
    summary=audit_panel(panel,tmp_path/'audit',terminal_audit=terminal)
    assert summary['terminal_evidence']['original_panel_sha256']=='original-panel-hash'
    assert summary['open_issues']==6 and summary['terminal_actions_reconciled'] is False
    queue=json.loads((tmp_path/'audit'/'quality-queue.json').read_text())
    assert queue[-1]['scope']=='terminal_event' and queue[-1]['id']=='terminal:OLD:2014-01-03'
