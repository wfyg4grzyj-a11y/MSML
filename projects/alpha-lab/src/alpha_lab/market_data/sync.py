"""Idempotent per-ticker updates, refreshing history when adjustments change."""

from datetime import date, timedelta

from .store import FIELDS, MarketStore, normalize_bars, utc_now
from .tiingo import TiingoClient, TiingoError, ticker_symbol


def sync_ticker(client: TiingoClient, store: MarketStore, ticker: str,
                start: date, end: date, *, overlap_days: int = 7,
                full_refresh: bool = False) -> dict:
    ticker = ticker_symbol(ticker)
    if start > end or overlap_days < 1:
        raise ValueError("Invalid date range or overlap_days")
    started = utc_now()
    first, last = store.bounds(ticker)
    coverage_start = store.coverage_start(ticker) or first
    # Never combine earlier requested history with a newer adjustment vintage.
    history_start = min(start, coverage_start) if coverage_start else start
    history_end = max(end, last) if last else end
    full = full_refresh or first is None or start < coverage_start
    fetch_start = history_start if full else max(history_start, last - timedelta(days=overlap_days))
    try:
        metadata = client.metadata(ticker)
        if ticker_symbol(metadata["ticker"]) != ticker:
            raise ValueError("Metadata ticker does not match request")
        responses = [("metadata", None, None, metadata)]

        def fetch(begin):
            payload = client.prices(ticker, begin, history_end)
            bars = normalize_bars(ticker, payload, begin, history_end)
            responses.append(("prices", begin.isoformat(), history_end.isoformat(), payload))
            return bars

        bars = fetch(fetch_start)
        if not bars:
            raise ValueError("Tiingo returned no bars; existing history was preserved")
        if not full:
            old = {row["date"]: row for row in store.prices([ticker], start=fetch_start.isoformat())}
            adjustment_changed = any(
                (bar["date"] not in old and (bar["split_factor"] != 1 or bar["div_cash"] != 0))
                or (bar["date"] in old and any(bar[key] != old[bar["date"]][key] for key in FIELDS))
                for bar in bars
            )
            if adjustment_changed:
                bars = fetch(history_start)
                full = True
                if not bars:
                    raise ValueError("History refresh returned no bars; existing history was preserved")
        # A truncated provider response must not silently remove stored dates.
        if full:
            existing = {row["date"] for row in store.prices([ticker], start=history_start.isoformat(), end=history_end.isoformat())}
            if not existing.issubset({bar["date"] for bar in bars}):
                raise ValueError("History refresh omitted cached dates; existing history was preserved")
        sync_id = store.commit_sync(ticker=ticker, metadata=metadata, bars=bars,
                                    responses=responses, started_at=started,
                                    start=history_start if full else fetch_start,
                                    end=history_end, full_refresh=full)
        return {"ticker": ticker, "sync_id": sync_id, "rows": len(bars), "full_refresh": full,
                "start_date": bars[0]["date"], "end_date": bars[-1]["date"]}
    except (TiingoError, ValueError):
        # Known errors are safe, but do not persist arbitrary upstream response text.
        store.record_failure(ticker, started, history_start, history_end, "Tiingo request or data validation failed")
        raise
