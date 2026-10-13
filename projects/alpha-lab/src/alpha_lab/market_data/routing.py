"""Sharadar-first selection, with explicit availability-based Tiingo fallback."""

from dataclasses import asdict, dataclass
from datetime import date

from .sharadar import TABLES, SharadarError, table_name
from .tiingo import ticker_symbol

# These feeds require new adapters and the relevant account entitlements.
TIINGO_EXTRA_DATASETS = ("intraday_prices", "news", "fund_fees", "dividend_details", "overnight_prices")


@dataclass(frozen=True)
class Route:
    dataset: str
    provider: str
    table: str
    reason: str
    ticker: str | None = None
    sharadar_start: str | None = None
    sharadar_end: str | None = None

    def manifest(self):
        return asdict(self) | {"policy": "sharadar_first", "series_policy": "one_provider_per_request"}


def catalog():
    return ([{"dataset": t, "provider": "sharadar", "adapter": "ready"} for t in TABLES]
            + [{"dataset": "daily_prices", "provider": "sharadar", "fallback": "tiingo",
                "adapter": "ready", "fallback_conditions": ["symbol_absent", "earlier_history"]}]
            + [{"dataset": t, "provider": "tiingo", "adapter": "not_implemented"} for t in TIINGO_EXTRA_DATASETS])


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
