"""Sharadar first, Tiingo next; Arrays only for uncovered US-equity datasets."""

from dataclasses import asdict, dataclass
from datetime import date

from .sharadar import TABLES, SharadarError, table_name
from .sharadar import file_hash
from pathlib import Path
from .tiingo import ticker_symbol

# These feeds require new adapters and the relevant account entitlements.
TIINGO_EXTRA_DATASETS = ("intraday_prices", "news", "fund_fees", "dividend_details", "overnight_prices")
ARRAYS_EXTRA_DATASETS = ("consensus_estimates", "company_guidance", "earnings_transcripts",
                         "finra_short_interest", "dark_pool_activity", "equity_ratings", "etf_holdings")
# Reviewed US-equity mandates; do not infer equity exposure from a US fund listing.
US_EQUITY_ETFS = {"SPY", "QQQ", "IWM"}
US_EXCHANGES = {"NYSE", "NASDAQ", "NYSEARCA", "NYSEMKT", "NYSEAMERICAN", "BATS", "OTC"}


def us_equity_scope(store, ticker, *, etf=False):
    symbol = ticker_symbol(ticker).replace('-', '.')
    if etf and symbol not in US_EQUITY_ETFS:
        raise SharadarError("Arrays ETF coverage is limited to reviewed US-equity funds SPY, QQQ and IWM")
    snapshot = store.latest('tickers')
    if snapshot is None or file_hash(Path(snapshot['path'])) != snapshot['parquet_sha256']:
        raise SharadarError("Arrays scope requires a verified local Sharadar tickers snapshot")
    with store._connect() as db:
        rows = db.execute('SELECT permaticker,exchange,currency,category FROM read_parquet(?) WHERE ticker=? AND "table"=?',
                          [snapshot['path'], symbol, 'SFP' if etf else 'SEP']).fetchall()
    if len(rows) != 1:
        raise SharadarError("US-equity scope is absent or ambiguous; Arrays request refused")
    permanent, exchange, currency, category = rows[0]
    category = str(category or '')
    if currency != 'USD' or str(exchange).upper() not in US_EXCHANGES or not permanent:
        raise SharadarError("Arrays accepts verified US-listed USD equities only")
    if (etf and category != 'ETF') or (not etf and 'Common Stock' not in category and 'ADR' not in category):
        raise SharadarError("Security type is outside Arrays US-equity scope")
    return {'symbol': symbol, 'asset_scope': 'us_equity_etf' if etf else 'us_listed_equity',
            'security_id': int(permanent), 'reference_version': snapshot['version'],
            'reference_sha256': snapshot['parquet_sha256'], 'historical_eligibility_attested': False}


@dataclass(frozen=True)
class Route:
    dataset: str
    provider: str
    table: str
    reason: str
    ticker: str | None = None
    sharadar_start: str | None = None
    sharadar_end: str | None = None
    scope: dict | None = None

    def manifest(self):
        return asdict(self) | {"policy": "sharadar_first", "series_policy": "one_provider_per_request",
                              "provider_priority": ["sharadar", "tiingo", "arrays"],
                              "arrays_policy": "uncovered_us_equity_only; never_on_provider_errors"}


def catalog():
    return ([{"dataset": t, "provider": "sharadar", "adapter": "ready"} for t in TABLES]
            + [{"dataset": "daily_prices", "provider": "sharadar", "fallback": "tiingo",
                "adapter": "ready", "fallback_conditions": ["symbol_absent", "earlier_history"]}]
            + [{"dataset": t, "provider": "tiingo", "adapter": "not_implemented"} for t in TIINGO_EXTRA_DATASETS]
            + [{"dataset": t, "provider": "arrays", "adapter": "ready", "scope": "us_equity_only",
                "reason": "not_covered_by_prior_providers"} for t in ARRAYS_EXTRA_DATASETS])


def select_route(store, dataset, *, ticker=None, start=None, end=None, asset_table=None):
    """Select using cached full Sharadar snapshots, never by catching provider failures.

    An absent local table is not evidence that the provider lacks a dataset.
    Earlier-history fallback selects the entire requested series from Tiingo,
    preserving each provider's adjustment basis. It does not promise Tiingo coverage.
    """
    if start:
        date.fromisoformat(str(start))
    if end:
        date.fromisoformat(str(end))
    if start and end and str(start) > str(end):
        raise SharadarError("Start date must precede end date")
    if dataset in TIINGO_EXTRA_DATASETS:
        raise SharadarError(f"{dataset} routes to Tiingo, but its adapter is not implemented; check entitlement before adding it")
    if dataset in ARRAYS_EXTRA_DATASETS:
        if not ticker:
            raise SharadarError("Arrays datasets require one explicit US-equity ticker")
        if asset_table is not None and asset_table != ('funds' if dataset == 'etf_holdings' else 'stocks'):
            raise SharadarError('Asset table conflicts with the requested Arrays dataset')
        scope = us_equity_scope(store, ticker, etf=dataset == 'etf_holdings')
        return Route(dataset, 'arrays', dataset, 'not_covered_by_prior_providers', scope['symbol'], scope=scope)
    if dataset != "daily_prices":
        table = table_name(dataset)
        return Route(table, "sharadar", table, "dataset_available", ticker_symbol(ticker) if ticker else None)
    if not ticker:
        raise SharadarError("daily_prices requires one ticker per request")
    ticker = ticker_symbol(ticker)
    if asset_table is not None and asset_table not in {"stocks", "funds"}:
        raise SharadarError("asset_table must be stocks or funds")
    tables = [asset_table] if asset_table else ["stocks", "funds"]
    snapshots = [(t, store.latest(t)) for t in tables]
    if any(snapshot is None for _, snapshot in snapshots):
        raise SharadarError("Sync Sharadar stocks and funds before checking daily-price availability")
    matches = []
    with store._connect() as db:
        for table, snapshot in snapshots:
            first, last = db.execute(
                'SELECT min(date), max(date) FROM read_parquet(?) WHERE ticker = ?',
                [snapshot["path"], ticker],
            ).fetchone()
            if first is not None:
                matches.append((table, str(first), str(last)))
    if len(matches) > 1:
        raise SharadarError("Ticker exists in stocks and funds; specify asset_table to avoid combining different assets")
    if not matches:
        return Route(dataset, "tiingo", "daily", "symbol_absent", ticker)
    table, first, last = matches[0]
    if (start and str(start) < first) or (end and str(end) < first):
        return Route(dataset, "tiingo", "daily", "earlier_history", ticker, first, last)
    return Route(dataset, "sharadar", table, "dataset_available", ticker, first, last)
