"""Offline, dated US equity research panel and constrained long/short backtester.

No brokerage connectivity. Corrected vendor history and terminal prices still
require an independent audit before these results support investment decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

from alpha_lab.market_data.sharadar_cli import load_config
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.equity_calendar import session_schedule, validate_benchmark, research_availability

FEATURES = ("book_yield", "earnings_yield", "fcf_yield", "roa", "gross_profitability",
            "leverage", "momentum_12_1", "momentum_3_1", "reversal_1m", "low_vol")


def create_session_prices(db):
    """Use exact benchmark sessions; missing quotes never extend a lookback.

    Inputs are stocks, securities and calendar(date, session) views/tables.
    RANGE frames avoid materializing a dense security/session cross product.
    """
    db.execute("""CREATE TABLE prices AS
        SELECT p.ticker,p.date,c.session,p.closeunadj,p.closeadj,
          CASE WHEN c.session-lag(c.session) OVER w=1
                    AND p.closeadj>0 AND isfinite(p.closeadj)
                    AND lag(p.closeadj) OVER w>0 AND isfinite(lag(p.closeadj) OVER w)
               THEN p.closeadj/lag(p.closeadj) OVER w-1 END ret,
          avg(CASE WHEN p.close>0 AND isfinite(p.close)
                        AND p.volume>=0 AND isfinite(p.volume)
                   THEN p.close*p.volume END) OVER recent adv20,
          count(CASE WHEN p.close>0 AND isfinite(p.close)
                          AND p.volume>=0 AND isfinite(p.volume) THEN 1 END)
              OVER recent liquidity_observations,
          count(CASE WHEN p.closeadj>0 AND isfinite(p.closeadj) THEN 1 END)
              OVER (PARTITION BY p.ticker ORDER BY c.session
                    RANGE BETWEEN 251 PRECEDING AND CURRENT ROW) observations,
          max(CASE WHEN p.closeadj>0 AND isfinite(p.closeadj) THEN p.closeadj END)
              OVER (PARTITION BY p.ticker ORDER BY c.session
                    RANGE BETWEEN 21 PRECEDING AND 21 PRECEDING) p21,
          max(CASE WHEN p.closeadj>0 AND isfinite(p.closeadj) THEN p.closeadj END)
              OVER (PARTITION BY p.ticker ORDER BY c.session
                    RANGE BETWEEN 63 PRECEDING AND 63 PRECEDING) p63,
          max(CASE WHEN p.closeadj>0 AND isfinite(p.closeadj) THEN p.closeadj END)
              OVER (PARTITION BY p.ticker ORDER BY c.session
                    RANGE BETWEEN 252 PRECEDING AND 252 PRECEDING) p252
        FROM stocks p JOIN securities USING(ticker) JOIN calendar c USING(date)
        WINDOW w AS (PARTITION BY p.ticker ORDER BY c.session),
               recent AS (PARTITION BY p.ticker ORDER BY c.session
                          RANGE BETWEEN 19 PRECEDING AND CURRENT ROW)""")


def create_research_calendar(db, start, end):
    schedule, metadata = session_schedule(start, end)
    benchmark = db.execute('''SELECT date,closeadj FROM funds WHERE ticker='SPY'
                            AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                            ORDER BY date''', [start, end]).df()
    validate_benchmark(benchmark, schedule)
    db.register('research_schedule', schedule)
    db.execute('CREATE TABLE calendar AS SELECT * FROM research_schedule ORDER BY date')
    db.unregister('research_schedule')
    return metadata


def prepare_panel(config: Path, output: Path, *, start="2014-01-01", end="2025-12-31",
                  universe_size=500):
    """Pin snapshots; monthly universes use dated price, liquidity and market cap.

    The static domestic primary-common-stock metadata restriction is disclosed:
    it is not a dated security master. Never restrict to currently active names.
    """
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(start_date) or pd.isna(end_date) or start_date > end_date or isinstance(universe_size, bool)
            or not isinstance(universe_size,int) or universe_size < 30):
        raise ValueError("Invalid date range or universe size; at least 30 securities required")
    warmup = (start_date - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new panel directory; existing panels are immutable")
    output.mkdir(parents=True, exist_ok=True)
    store, _, _ = load_config(config)
    sources = {table: store.latest(table) for table in
               ("stocks", "funds", "daily", "fundamentals", "tickers", "actions")}
    for source in sources.values():
        if file_hash(Path(source['path']))!=source['parquet_sha256']:
            raise ValueError("Pinned raw-source hash mismatch")
    preparation_hash = file_hash(Path(__file__))
    atomic_json(output / "sources.json", sources)
    with store._connect() as db:
        for table, source in sources.items():
            db.read_parquet(source["path"]).create_view(table)
        calendar_metadata = create_research_calendar(db,warmup,end)
        db.execute('SELECT * FROM calendar ORDER BY date').df().to_csv(output/'calendar.csv',index=False)
        db.execute("""CREATE TABLE months AS SELECT MAX(date) date FROM calendar
                      WHERE date >= CAST(? AS DATE) GROUP BY date_trunc('month',date)""", [start])
        db.execute("""CREATE TABLE securities AS SELECT DISTINCT ticker FROM tickers
                      WHERE "table"='SF1' AND currency='USD'
                      AND category IN ('Domestic Common Stock','Domestic Common Stock Primary Class')""")
        print("Preparing dated prices and rolling liquidity", flush=True)
        create_session_prices(db)
        db.execute("""CREATE TABLE snapshots AS SELECT p.ticker,p.date,p.closeunadj,p.adv20,
                      d.marketcap*1000000 marketcap, p.p21/p.p252-1 momentum_12_1,
                      p.p21/p.p63-1 momentum_3_1, -(p.closeadj/p.p21-1) reversal_1m
                      FROM prices p JOIN months m USING(date) JOIN daily d USING(ticker,date)
                      WHERE p.observations=252 AND p.liquidity_observations=20
                      AND p.p21>0 AND p.p63>0 AND p.p252>0
                      AND p.closeunadj>=5 AND p.adv20>=5000000
                      AND d.marketcap>=500
                      QUALIFY row_number() OVER (PARTITION BY p.date ORDER BY d.marketcap DESC,p.ticker)<=?""", [universe_size])
        snapshots = db.execute("SELECT * FROM snapshots ORDER BY ticker,date").df()
        tickers = sorted(snapshots.ticker.unique())
        db.register("selected", pd.DataFrame({"ticker": tickers}))
        print(f"Loading {len(tickers)} historical securities", flush=True)
        prices = db.execute("""SELECT ticker,date,closeadj,ret FROM prices JOIN selected USING(ticker)
                               ORDER BY date,ticker""").df()
        filings = db.execute("""SELECT f.ticker,f.date filing_date,f.reportperiod,f.fxusd,
                              f.equityusd,f.netinccmnusd,f.fcf,f.roa,f.gp,f.assets,f.debtusd
                              FROM fundamentals f JOIN selected USING(ticker)
                              WHERE dimension='ART' AND date <= CAST(? AS DATE)
                              QUALIFY row_number() OVER(PARTITION BY ticker,date
                              ORDER BY reportperiod DESC,lastupdated DESC)=1
                              ORDER BY filing_date,ticker""", [end]).df()
        benchmark = db.execute("SELECT date,closeadj FROM funds WHERE ticker='SPY' ORDER BY date").df()
        actions = db.execute("SELECT ticker,date,action,value,contraticker FROM actions JOIN selected USING(ticker)").df()
    snapshots["date"] = pd.to_datetime(snapshots.date)
    filings["filing_date"] = pd.to_datetime(filings.filing_date)
    filings["available_date"] = filings.filing_date + pd.Timedelta(days=2)
    features = pd.merge_asof(snapshots.sort_values("date"), filings.sort_values("available_date"),
                             left_on="date", right_on="available_date", by="ticker",
                             direction="backward", tolerance=pd.Timedelta(days=550))
    features["book_yield"] = features.equityusd / features.marketcap
    features["earnings_yield"] = features.netinccmnusd / features.marketcap
    features["fcf_yield"] = features.fcf * features.fxusd / features.marketcap
    features["gross_profitability"] = features.gp / features.assets.where(features.assets > 0)
    features["leverage"] = features.debtusd / (features.assets * features.fxusd).where(features.assets > 0)
    features["filing_age_days"] = (features.date - features.filing_date).dt.days
    prices["date"] = pd.to_datetime(prices.date)
    benchmark["date"] = pd.to_datetime(benchmark.date)
    calendar = pd.DatetimeIndex(benchmark.loc[benchmark.date.between(warmup, end), "date"])
    closes = prices.pivot(index="date", columns="ticker", values="closeadj").reindex(index=calendar, columns=tickers)
    closes = closes.where((closes>0) & np.isfinite(closes))
    # No forward-filled returns. Gaps are observed by the backtester explicitly.
    returns = closes.pct_change(fill_method=None)
    spy = benchmark.set_index("date").closeadj.reindex(calendar).pct_change(fill_method=None)
    variance = spy.rolling(252, min_periods=200).var()
    beta = returns.rolling(252, min_periods=200).cov(spy).div(variance, axis=0)
    vol = returns.rolling(63, min_periods=50).std()
    for column, frame in (("beta", beta), ("low_vol", -vol)):
        lookup = frame.stack().rename(column).reset_index()
        features = features.merge(lookup, on=["date", "ticker"], how="left")
    features = features.replace([np.inf, -np.inf], np.nan)
    duckdb.from_df(features).write_parquet(str(output / "features.parquet"))
    duckdb.from_df(actions).write_parquet(str(output / "actions.parquet"))
    np.savez_compressed(output / "returns.npz", dates=calendar.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                        tickers=np.asarray(tickers, dtype="U32"), returns=returns.to_numpy(dtype=np.float32),
                        price_observed=np.isfinite(closes.to_numpy()),
                        spy=spy.to_numpy(dtype=np.float32))
    manifest = {"schema_version": 4, "preparation_sha256": preparation_hash,
                "calendar_preparation_sha256": file_hash(Path(__file__).with_name('equity_calendar.py')),
                "raw_source_hashes_verified": True,
                "calendar": {**calendar_metadata, "benchmark_complete": True},
                "availability_assumptions": research_availability(),
                "start": start, "end": end, "universe_size": universe_size,
                "historical_tickers": len(tickers), "monthly_rows": len(features),
                "months": features.date.nunique(), "features": list(FEATURES),
                "filing_lag_calendar_days": 2, "max_available_age_days": 550, "max_filing_age_days": 552,
                "minimum_raw_price": 5, "minimum_adv20_usd": 5_000_000,
                "minimum_marketcap_usd": 500_000_000,
                "price_window_policy": {
                    "calendar": "XNYS modeled sessions; SPY completeness validated; no quote forward fill",
                    "momentum_offsets_sessions": [21,63,252],
                    "liquidity_sessions": 20, "minimum_liquidity_observations": 20,
                    "history_sessions": 252, "minimum_history_observations": 252,
                    "gap_spanning_daily_returns": "missing, never assigned to one session"},
                "point_in_time_attested": False, "terminal_actions_reconciled": False,
                "limitations": ["Corrected vendor snapshot, not historical feed replay",
                                "Static domestic primary common-stock metadata filter",
                                "Ticker continuity and issuer class mapping are not fully audited",
                                "Missing and terminal returns require stress analysis",
                                "No historical borrow availability or locate fees",
                                "Monthly estimated beta neutrality does not ensure realized beta neutrality",
                                "Historical source delivery and market-cap share vintages are unverified",
                                "Exchange-rule calendar is modeled, not an independent historical-feed attestation"]}
    manifest["files"] = {name: file_hash(output/name) for name in
                         ("sources.json","features.parquet","actions.parquet","returns.npz","calendar.csv")}
    atomic_json(output / "panel.json", manifest)
    print(json.dumps(manifest), flush=True)
    return manifest


def neutral_weights(scores, beta, *, max_weight=.02):
    """Continuous scores projected off intercept/beta; cap by reducing gross."""
    scores, beta = np.asarray(scores, float), np.asarray(beta, float)
    if len(scores) < 30 or not np.isfinite(scores).all() or not np.isfinite(beta).all():
        return np.zeros_like(scores)
    design = np.column_stack([np.ones(len(scores)), beta])
    residual = scores - design @ np.linalg.lstsq(design, scores, rcond=None)[0]
    gross = np.abs(residual).sum()
    if gross < 1e-10:
        return np.zeros_like(scores)
    weights = residual / gross
    if np.abs(weights).max() > max_weight:
        weights *= max_weight / np.abs(weights).max()
    return weights


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"name", "factors"}:
        raise ValueError("Strategy requires exactly name and factors")
    if not isinstance(spec["name"], str) or not 1 <= len(spec["name"]) <= 80:
        raise ValueError("Invalid strategy name")
    factors = spec["factors"]
    if not isinstance(factors, dict) or not factors or set(factors) - set(FEATURES):
        raise ValueError("Unknown or empty factors")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or abs(v)>5
           for v in factors.values()) or sum(abs(v) for v in factors.values()) == 0:
        raise ValueError("Invalid factor coefficients")
    return spec


def performance(returns, spy):
    returns, spy = np.asarray(returns, float), np.asarray(spy, float)
    if len(returns)<2 or not np.isfinite(returns).all() or np.any(returns<=-1):
        raise ValueError("Insufficient finite returns or portfolio insolvency")
    wealth = np.cumprod(1 + returns)
    stdev = returns.std(ddof=1)
    var = spy.var(ddof=1)
    return {"sessions": len(returns), "annualized_return": float(wealth[-1] ** (252/len(returns))-1),
            "annualized_volatility": float(stdev*np.sqrt(252)),
            "sharpe_zero_rf": float(returns.mean()/stdev*np.sqrt(252)) if stdev else 0.,
            "max_drawdown": float(np.min(wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1)),
            "realized_spy_beta": float(np.cov(returns,spy,ddof=1)[0,1]/var) if var else 0.,
            "total_return": float(wealth[-1]-1)}


class MarketNeutralBacktest:
    def __init__(self, panel: Path):
        self.panel = panel = Path(panel)
        self.panel_integrity_verified = False
        self.calendar_verified = False
        self.availability_assumptions = {"status": "Legacy panel: availability assumptions not fully recorded",
                                        "point_in_time_attested": False}
        manifest = {}
        if (panel/"panel.json").exists():
            manifest=json.loads((panel/"panel.json").read_text())
            files=manifest.get("files",{})
            if not isinstance(files,dict):
                raise ValueError("Research panel file hashes are invalid")
            for name, expected in files.items():
                if name not in {"sources.json","features.parquet","actions.parquet","returns.npz","calendar.csv"}:
                    raise ValueError("Unknown research panel file")
                if file_hash(panel/name)!=expected:
                    raise ValueError("Research panel file hash mismatch")
            required_files = {"sources.json","features.parquet","actions.parquet","returns.npz"}
            if manifest.get('schema_version',0) >= 4:
                required_files.add('calendar.csv')
                if set(files) != required_files:
                    raise ValueError("Research panel calendar evidence or file hashes are incomplete")
            self.panel_integrity_verified = set(files)==required_files
            self.availability_assumptions = manifest.get('availability_assumptions',self.availability_assumptions)
        self.features = duckdb.read_parquet(str(panel / "features.parquet")).df()
        with np.load(panel / "returns.npz", allow_pickle=False) as raw:
            self.dates = pd.DatetimeIndex(raw["dates"])
            self.tickers = raw["tickers"].tolist()
            self.returns = raw["returns"].astype(float)
            self.spy = raw["spy"].astype(float)
            self.price_observed = raw["price_observed"] if "price_observed" in raw else None
        if (self.dates.hasnans or not self.dates.is_monotonic_increasing or not self.dates.is_unique
                or len(self.dates)<2 or len(set(self.tickers))!=len(self.tickers)
                or not self.tickers or any(not isinstance(t,str) or not t for t in self.tickers)):
            raise ValueError("Research panel dates or security identities are invalid")
        if manifest.get('schema_version',0) >= 4:
            recorded = pd.read_csv(panel/'calendar.csv',parse_dates=['date'])
            if not {'start','end'} <= set(manifest):
                raise ValueError("Research panel calendar date bounds are incomplete")
            warmup = (pd.Timestamp(manifest['start'])-pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            expected, _ = session_schedule(warmup,manifest['end'])
            if not pd.DatetimeIndex(recorded.date).equals(self.dates):
                raise ValueError("Research panel calendar does not match return dates")
            try:
                pd.testing.assert_frame_equal(recorded,expected,check_dtype=False)
            except AssertionError:
                raise ValueError("Research panel calendar evidence differs from the exchange schedule") from None
            self.calendar_verified = True
        if self.returns.shape!=(len(self.dates),len(self.tickers)) or self.spy.shape!=(len(self.dates),):
            raise ValueError("Research panel array dimensions are inconsistent")
        if (np.isinf(self.returns).any() or np.any(self.returns < -1) or np.isinf(self.spy).any()
                or not np.isfinite(self.spy[1:]).all() or np.any(self.spy < -1)):
            raise ValueError("Research panel contains impossible returns or missing benchmark sessions")
        if self.price_observed is not None and (
                self.price_observed.shape!=self.returns.shape or self.price_observed.dtype!=np.dtype(bool)):
            raise ValueError("Research panel price-observation mask is invalid")
        if self.price_observed is not None and np.any(np.isfinite(self.returns[1:]) & ~(
                self.price_observed[1:] & self.price_observed[:-1])):
            raise ValueError("Research panel assigns finite price returns to missing quote pairs")
        required={"ticker","date","beta","filing_date"}
        if not required<=set(self.features):
            raise ValueError("Research panel feature columns are incomplete")
        if (self.features.duplicated(["ticker","date"]).any()
                or not self.features.ticker.isin(self.tickers).all()
                or not self.features.date.isin(self.dates).all()):
            raise ValueError("Research panel features have duplicate or unmatched identities/dates")
        if (self.features.filing_date>self.features.date).any() or (
                "available_date" in self.features
                and (self.features.available_date>self.features.date).any()):
            raise ValueError("Research panel contains future accounting information")
        self.positions = {ticker:i for i,ticker in enumerate(self.tickers)}

    def evaluate(self, spec, *, include_holdout=False, cost_bps=10., borrow_bps=300.,
                 terminal_stress=.0):
        validate_spec(spec)
        if any(isinstance(value,bool) or not np.isfinite(value) or value<0
               for value in (cost_bps,borrow_bps,terminal_stress)):
            raise ValueError("Costs, borrow and terminal stress must be finite and nonnegative")
        factors = list(spec["factors"])
        targets = {}
        for date, cross in self.features.groupby("date", sort=True):
            cross = cross.dropna(subset=factors+["beta", "filing_date"])
            if len(cross) < 30:
                continue
            score = np.zeros(len(cross))
            for factor, coefficient in spec["factors"].items():
                score += (cross[factor].rank(pct=True).to_numpy()-.5)*coefficient
            weights = neutral_weights(score, cross.beta.to_numpy())
            target = np.zeros(len(self.tickers))
            indices = [self.positions[t] for t in cross.ticker]
            target[indices] = weights
            target_beta = np.full(len(self.tickers), np.nan)
            target_beta[indices] = cross.beta.to_numpy()
            # Signal month-end -> next session close execution -> subsequent return.
            activation = self.dates.searchsorted(pd.Timestamp(date)) + 2
            targets[activation] = (target, target_beta)
        end = pd.Timestamp("2025-12-31" if include_holdout else "2022-12-31")
        active = np.zeros(len(self.tickers))
        previous = active.copy()
        active_beta = np.full(len(self.tickers), np.nan)
        net, gross, stress, turnover, exposures, missing = [], [], [], [], [], []
        used_dates = []
        missing_details = []
        for i, date in enumerate(self.dates):
            if date < pd.Timestamp("2014-01-01"):
                continue
            if date > end:
                break
            if i in targets:
                active, active_beta = (value.copy() for value in targets[i])
                if self.price_observed is not None:
                    known_available = self.price_observed[i-1] & np.isfinite(active_beta)
                    active[~known_available]=0.
                    remaining = known_available & (np.abs(active)>1e-12)
                    repaired = np.zeros_like(active)
                    repaired[remaining] = neutral_weights(active[remaining],active_beta[remaining])
                    active = repaired
            bad = (~np.isfinite(self.returns[i])) & (np.abs(active)>1e-12)
            # Mark stale at last available price, liquidate now, record unresolved
            # exposure. Adverse stress charges BOTH sides instead of rewarding shorts.
            missing_exposure = float(np.abs(active[bad]).sum())
            for j in np.flatnonzero(bad):
                missing_details.append({"date":str(date.date()),"ticker":self.tickers[j],
                                        "weight":float(active[j])})
            r = np.nan_to_num(self.returns[i], nan=0., posinf=0., neginf=0.)
            # Today's missing observation must not change today's other holdings.
            # Repair the portfolio only for the following return interval.
            desired = active.copy()
            traded = float(np.abs(desired-previous).sum())
            pnl = float(desired @ r)
            fee = traded * cost_bps/10_000
            borrow = float(-desired[desired<0].sum())*borrow_bps/10_000/252
            daily_net = pnl-fee-borrow
            daily_stress = daily_net-missing_exposure*terminal_stress
            gross.append(pnl); net.append(daily_net); stress.append(daily_stress)
            turnover.append(traded); missing.append(missing_exposure)
            exposures.append([float(desired.sum()), float(np.abs(desired).sum())])
            used_dates.append(date)
            active = desired.copy()
            if bad.any():
                active[bad] = 0.
                remaining = (np.abs(active)>1e-12) & np.isfinite(active_beta)
                repaired = np.zeros_like(active)
                repaired[remaining] = neutral_weights(active[remaining], active_beta[remaining])
                active = repaired
            # End-of-session marked weights; cash/fees handled at portfolio level.
            previous = desired*(1+r)/(1+daily_net)
        dates = pd.DatetimeIndex(used_dates)
        net = np.asarray(net); stress=np.asarray(stress)
        result = {"spec": spec, "cost_bps": cost_bps, "borrow_bps_annual": borrow_bps,
                  "terminal_stress_fraction": terminal_stress,
                  "execution": "Monthly signals executed following close; targets maintained daily",
                  "cash": "No cash interest or financing spread; Sharpe uses zero risk-free rate",
                  "total_notional_turnover": float(np.sum(turnover)),
                  "missing_exposure_events": int(np.count_nonzero(missing)),
                  "missing_price_observations": len(missing_details),
                  "missing_exposure_sum": float(np.sum(missing)),
                  "missing_event_details": missing_details,
                  "max_abs_net_exposure": float(np.max(np.abs(np.asarray(exposures)[:,0]))),
                  "mean_gross_exposure": float(np.mean(np.asarray(exposures)[:,1])),
                  "panel_integrity_verified": self.panel_integrity_verified,
                  "calendar_verified": self.calendar_verified,
                  "availability_assumptions": self.availability_assumptions,
                  "missing_price_policy": "Zero marked return, assumed stale-price exit and subsequent repair; economic proceeds unresolved",
                  "terminal_actions_reconciled": False,
                  "investment_ready": False,
                  "research_only": True}
        for name, begin, finish in (("train","2014-01-01","2019-12-31"),
                                     ("validation","2020-01-01","2022-12-31"),
                                     ("holdout","2023-01-01","2025-12-31")):
            mask = (dates>=begin)&(dates<=finish)
            if mask.any():
                result[name] = performance(net[mask], self.spy[self.dates.get_indexer(dates[mask])])
                result[name]["terminal_stress_total_return"] = float(np.prod(1+stress[mask])-1)
        return result, pd.DataFrame({"date":dates,"gross":gross,"net":net,"stress":stress,
                                     "turnover":turnover,"missing_exposure":missing,
                                     "net_exposure":np.asarray(exposures)[:,0],
                                     "gross_exposure":np.asarray(exposures)[:,1]})
