import pytest
pd = pytest.importorskip('pandas')
np = pytest.importorskip('numpy')
pytest.importorskip('exchange_calendars')
from alpha_lab.equity_calendar import session_schedule,validate_benchmark
from alpha_lab.market_neutral import create_research_calendar
import duckdb


def test_exceptional_closures_and_early_close_times():
    schedule,_ = session_schedule('2012-10-26','2012-11-02')
    assert schedule.date.dt.strftime('%Y-%m-%d').tolist()==['2012-10-26','2012-10-31','2012-11-01','2012-11-02']
    schedule,_ = session_schedule('2018-12-03','2018-12-07')
    assert pd.Timestamp('2018-12-05') not in pd.DatetimeIndex(schedule.date)
    schedule,meta = session_schedule('2024-11-27','2024-11-29')
    assert len(schedule)==2 and meta['early_close_sessions']==1
    assert schedule.iloc[-1].session_close_utc=='2024-11-29 18:00:00+00:00'


@pytest.mark.parametrize('change',[
    lambda f:f.drop(index=0),lambda f:f.drop(index=1),lambda f:f.drop(index=len(f)-1),
    lambda f:pd.concat([f,f.iloc[:1].assign(date=pd.Timestamp('2014-01-04'))]),
    lambda f:pd.concat([f,f.iloc[:1]]),
    lambda f:f.assign(closeadj=[np.nan,100.,100.]),
    lambda f:f.assign(closeadj=[np.inf,100.,100.]),
    lambda f:f.assign(closeadj=[0.,100.,100.]),
    lambda f:f.assign(closeadj=[-1.,100.,100.]),
])
def test_bad_benchmark_fails_before_a_calendar_is_created(change):
    schedule,_=session_schedule('2014-01-02','2014-01-06')
    benchmark=schedule[['date']].assign(ticker='SPY',closeadj=100.)
    with duckdb.connect() as db:
        db.register('funds',change(benchmark))
        with pytest.raises(ValueError,match='Benchmark'):
            create_research_calendar(db,'2014-01-02','2014-01-06')
        assert db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='calendar'").fetchone()[0]==0


@pytest.mark.parametrize('start,end',[
    ('2014-01-06','2014-01-02'),('2014-01-02T12:00','2014-01-06'),
    ('2014-01-02T00:00Z','2014-01-06'),('NaT','2014-01-06'),
])
def test_invalid_date_bounds_do_not_silently_normalize(start,end):
    with pytest.raises(ValueError,match='timezone-free'):
        session_schedule(start,end)
