import pytest
np=pytest.importorskip("numpy")
pd=pytest.importorskip("pandas")
import duckdb

from alpha_lab.market_neutral import MarketNeutralBacktest, neutral_weights, validate_spec


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
