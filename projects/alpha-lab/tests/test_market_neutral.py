import pytest
np=pytest.importorskip("numpy")
pd=pytest.importorskip("pandas")
import duckdb

from alpha_lab.market_neutral import (MarketNeutralBacktest, neutral_weights, validate_spec,
                                      create_session_prices, create_research_calendar)


def test_neutral_projection_and_position_cap():
    rng=np.random.default_rng(13)
    scores=rng.normal(size=100)
    beta=rng.uniform(.3,2.,size=100)
    weights=neutral_weights(scores,beta,max_weight=.01)
    assert abs(weights.sum())<1e-12
    assert abs(weights@beta)<1e-12
    assert np.abs(weights).sum()<=1+1e-12
    assert np.abs(weights).max()<=.01+1e-12


@pytest.mark.parametrize("factors", [{"future_return":1},{"roa":float('nan')},{"roa":True},{"roa":0}])
def test_reject_invalid_or_future_factors(factors):
    with pytest.raises(ValueError):
        validate_spec({"name":"bad","factors":factors})


def synthetic_panel(tmp_path):
    dates=pd.bdate_range('2014-01-29','2014-02-14')
    tickers=np.asarray([f'S{i}' for i in range(40)],dtype='U32')
    features=pd.DataFrame({"ticker":tickers,"date":pd.Timestamp('2014-01-31'),
                           "filing_date":pd.Timestamp('2014-01-20'),
                           "beta":np.linspace(.5,1.5,40),"book_yield":np.sin(np.arange(40))})
    duckdb.from_df(features).write_parquet(str(tmp_path/'features.parquet'))
    returns=np.zeros((len(dates),40))
    # Massive return during the execution session must not be earned by signal.
    returns[dates.get_loc('2014-02-03')]=np.sin(np.arange(40))*.1
    returns[dates.get_loc('2014-02-04')]=np.sin(np.arange(40))*.01
    np.savez(tmp_path/'returns.npz',dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
             tickers=tickers,returns=returns,spy=np.linspace(-.01,.01,len(dates)))
    return tmp_path


def test_signal_executes_next_close_without_same_day_profit(tmp_path):
    engine=MarketNeutralBacktest(synthetic_panel(tmp_path))
    result,daily=engine.evaluate({"name":"value","factors":{"book_yield":1}},cost_bps=0,borrow_bps=0)
    assert daily.loc[daily.date==pd.Timestamp('2014-02-03'),'net'].item()==0
    assert daily.loc[daily.date==pd.Timestamp('2014-02-04'),'net'].item()>0
    assert result['max_abs_net_exposure']<1e-12


def test_costs_and_borrow_reduce_returns_and_holdout_not_exposed(tmp_path):
    engine=MarketNeutralBacktest(synthetic_panel(tmp_path))
    spec={"name":"value","factors":{"book_yield":1}}
    clean,_=engine.evaluate(spec,cost_bps=0,borrow_bps=0)
    costly,_=engine.evaluate(spec,cost_bps=100,borrow_bps=1000)
    assert costly['train']['total_return']<clean['train']['total_return']
    assert 'holdout' not in costly


def test_missing_return_is_recorded_and_adverse_stress_applies(tmp_path):
    engine=MarketNeutralBacktest(synthetic_panel(tmp_path))
    engine.returns[engine.dates.get_loc('2014-02-05'),0]=np.nan
    result,daily=engine.evaluate({"name":"value","factors":{"book_yield":1}},terminal_stress=.3)
    assert result['missing_exposure_events']==1 and result['missing_exposure_sum']>0
    assert daily.stress.sum()<daily.net.sum()
    assert np.max(np.abs(daily.net_exposure))<1e-12


@pytest.mark.parametrize("parameters",[{"cost_bps":-1},{"borrow_bps":float('nan')},{"terminal_stress":-1}])
def test_reject_impossible_cost_assumptions(tmp_path,parameters):
    engine=MarketNeutralBacktest(synthetic_panel(tmp_path))
    with pytest.raises(ValueError,match="nonnegative"):
        engine.evaluate({"name":"value","factors":{"book_yield":1}},**parameters)


def test_session_windows_do_not_stretch_over_missing_quotes():
    dates=pd.bdate_range('2013-01-01',periods=330)
    calendar=pd.DataFrame({"date":dates,"session":np.arange(330)})
    full=pd.DataFrame({"ticker":"FULL","date":dates,"close":200.+np.arange(330),
                       "closeadj":200.+np.arange(330),"closeunadj":200.+np.arange(330),"volume":1.})
    gap=full.copy();gap['ticker']='GAP'
    gap=gap.drop(index=[279,295])
    gap.loc[280,'volume']=1000.  # exactly outside session 300's twenty-session window
    with duckdb.connect() as db:
        db.register('calendar',calendar)
        db.register('stocks',pd.concat([full,gap]))
        db.register('securities',pd.DataFrame({"ticker":['FULL','GAP']}))
        create_session_prices(db)
        rows=db.execute('SELECT * FROM prices WHERE session=300 ORDER BY ticker').df().set_index('ticker')
        assert rows.loc['FULL','p21']==479 and rows.loc['FULL','p63']==437
        assert rows.loc['FULL','p252']==248 and rows.loc['FULL','observations']==252
        assert rows.loc['FULL','liquidity_observations']==20
        assert np.isnan(rows.loc['GAP','p21'])  # no substitution with an earlier observed row
        assert rows.loc['GAP','p63']==437 and rows.loc['GAP','p252']==248
        assert rows.loc['GAP','observations']==250 and rows.loc['GAP','liquidity_observations']==19
        assert rows.loc['GAP','adv20']==pytest.approx(np.mean([200+i for i in range(281,301) if i!=295]))
        assert db.execute("SELECT ret FROM prices WHERE ticker='GAP' AND session=280").fetchone()[0] is None
        assert db.execute("SELECT ret FROM prices WHERE ticker='GAP' AND session=281").fetchone()[0]==pytest.approx(481/480-1)
        # A future price change must not alter any feature available at session 300.
        before=db.execute('SELECT * FROM prices WHERE session<=300 ORDER BY ticker,session').df()
        db.execute('DROP TABLE prices')
        full.loc[full.index>300,['close','closeadj','closeunadj']]=1e9
        db.register('stocks',pd.concat([full,gap]))
        create_session_prices(db)
        after=db.execute('SELECT * FROM prices WHERE session<=300 ORDER BY ticker,session').df()
        pd.testing.assert_frame_equal(before,after)


def test_research_calendar_uses_sorted_unique_benchmark_sessions():
    funds=pd.DataFrame({"ticker":['SPY','OTHER','SPY','SPY'],
                        "date":pd.to_datetime(['2014-01-06','2014-01-07','2014-01-03','2014-01-02'])})
    with duckdb.connect() as db:
        db.register('funds',funds)
        create_research_calendar(db,'2014-01-03','2014-01-06')
        assert db.execute('SELECT "session" FROM calendar ORDER BY date').fetchall()==[(1,),(2,)]
        db.execute('DROP TABLE calendar')
        db.register('funds',pd.concat([funds,funds.iloc[:1]]))
        with pytest.raises(ValueError,match='duplicated'):
            create_research_calendar(db,'2014-01-03','2014-01-06')


def rewrite_arrays(panel,change):
    with np.load(panel/'returns.npz',allow_pickle=False) as raw:
        arrays={key:raw[key] for key in raw.files}
    change(arrays)
    np.savez(panel/'returns.npz',**arrays)


@pytest.mark.parametrize('change',[
    lambda x:x.update(returns=x['returns'][:,:-1]),
    lambda x:x['dates'].__setitem__(1,x['dates'][0]),
    lambda x:x['tickers'].__setitem__(1,x['tickers'][0]),
    lambda x:x['spy'].__setitem__(3,np.nan),
    lambda x:x['spy'].__setitem__(0,np.inf),
    lambda x:x['returns'].__setitem__((3,0),np.inf),
    lambda x:x['returns'].__setitem__((3,0),-1.1),
    lambda x:x.update(price_observed=np.ones_like(x['returns'],dtype=float)),
    lambda x:x.update(price_observed=np.arange(x['returns'].size).reshape(x['returns'].shape)!=120),
])
def test_corrupt_arrays_are_rejected_before_backtesting(tmp_path,change):
    panel=synthetic_panel(tmp_path)
    rewrite_arrays(panel,change)
    with pytest.raises(ValueError,match='Research panel'):
        MarketNeutralBacktest(panel)


@pytest.mark.parametrize('change',[
    lambda frame:pd.concat([frame,frame.iloc[:1]]),
    lambda frame:frame.assign(filing_date=pd.Timestamp('2014-02-01')),
    lambda frame:frame.assign(available_date=pd.Timestamp('2014-02-01')),
    lambda frame:frame.assign(ticker='UNKNOWN'),
])
def test_feature_identity_and_availability_failures_are_rejected(tmp_path,change):
    panel=synthetic_panel(tmp_path)
    features=duckdb.read_parquet(str(panel/'features.parquet')).df()
    duckdb.from_df(change(features)).write_parquet(str(panel/'features.parquet'))
    with pytest.raises(ValueError,match='Research panel'):
        MarketNeutralBacktest(panel)


def test_panel_hash_is_checked_before_reading_modified_data(tmp_path):
    import json
    from alpha_lab.market_data.sharadar import file_hash
    panel=synthetic_panel(tmp_path)
    expected=file_hash(panel/'features.parquet')
    (panel/'panel.json').write_text(json.dumps({"files":{"features.parquet":expected}}))
    (panel/'features.parquet').write_bytes(b'corrupt data')
    with pytest.raises(ValueError,match='file hash mismatch'):
        MarketNeutralBacktest(panel)


def test_unverified_legacy_panel_and_terminal_policy_are_explicit(tmp_path):
    engine=MarketNeutralBacktest(synthetic_panel(tmp_path))
    result,_=engine.evaluate({"name":"value","factors":{"book_yield":1}})
    assert result['panel_integrity_verified'] is False
    assert result['terminal_actions_reconciled'] is False and result['investment_ready'] is False
    assert 'economic proceeds unresolved' in result['missing_price_policy']


def test_panel_preparation_rejects_modified_raw_source_before_querying(tmp_path,monkeypatch):
    from alpha_lab.market_neutral import prepare_panel
    from alpha_lab.market_data.sharadar import file_hash
    source=tmp_path/'source.parquet';source.write_bytes(b'original')
    original_hash=file_hash(source)
    source.write_bytes(b'modified')
    class Store:
        def latest(self,table):
            return {'path':str(source),'parquet_sha256':original_hash}
        def _connect(self):
            pytest.fail('Modified raw snapshot must fail before any database access')
    monkeypatch.setattr('alpha_lab.market_neutral.load_config',lambda config:(Store(),[],0))
    with pytest.raises(ValueError,match='raw-source hash mismatch'):
        prepare_panel(tmp_path/'config.json',tmp_path/'panel')
