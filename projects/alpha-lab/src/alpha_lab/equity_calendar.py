"""Offline US equity sessions and benchmark completeness checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def session_schedule(start, end):
    """Return XNYS modeled sessions, including exceptional closures/early closes.

    This community-maintained calendar is a pinned model of exchange rules,
    not an independently attested historical exchange feed.
    """
    try:
        import exchange_calendars as calendars
    except ImportError:
        raise ValueError("Install AlphaLab's equity-research extra for session validation") from None
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(begin) or pd.isna(finish) or begin.tzinfo is not None or finish.tzinfo is not None
            or begin != begin.normalize() or finish != finish.normalize() or begin > finish):
        raise ValueError("Research calendar requires increasing, timezone-free dates")
    calendar = calendars.get_calendar("XNYS", start=begin, end=finish)
    dates = calendar.sessions
    if len(dates) < 2:
        raise ValueError("Research calendar requires at least two exchange sessions")
    schedule = pd.DataFrame({
        "date": dates, "session": np.arange(1, len(dates) + 1),
        "session_close_utc": calendar.schedule.loc[dates, "close"].astype(str).to_numpy(),
        "early_close": dates.isin(calendar.early_closes),
    })
    metadata = {
        "calendar_id": "XNYS", "library": "exchange_calendars", "version": calendars.__version__,
        "reference": "https://github.com/gerrymanoim/exchange_calendars",
        "method": "Modeled regular US equity sessions including exceptional closures and early closes",
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "sessions": len(dates), "early_close_sessions": int(schedule.early_close.sum()),
        "independently_attested": False,
    }
    return schedule, metadata


def validate_benchmark(benchmark, schedule):
    """Reject missing/extra/duplicate sessions and unusable adjusted SPY quotes."""
    dates = pd.DatetimeIndex(benchmark.date)
    expected = pd.DatetimeIndex(schedule.date)
    if dates.hasnans or not dates.is_unique:
        raise ValueError("Benchmark session calendar is empty or duplicated")
    missing, extra = expected.difference(dates), dates.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"Benchmark/exchange session mismatch: {len(missing)} missing, {len(extra)} extra")
    values = benchmark.closeadj.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Benchmark adjusted prices must be positive and finite on every session")


def research_availability():
    """Recorded assumptions, not a point-in-time data attestation."""
    return {
        "source_vintage": "Current corrected vendor snapshots; historical arrivals and revisions not replayed",
        "market_data": "Signal-date EOD prices/volume assumed usable after that close; no same-close execution",
        "marketcap": "Same-date Sharadar DAILY marketcap in USD millions; historical delivery/share vintages unverified",
        "marketcap_limitation": "No independently dated share-count reconstruction or issuer/class aggregation",
        "fundamentals": "ART filing DateKey plus two calendar days; backward as-of join; at most 550 days after assumed availability (552 after filing)",
        "vendor_schedule": "Published daily delivery/lag is context, not proof of each historical record arrival",
        "security_metadata": "Static domestic USD financial-reporting primary-common-stock filter, includes delisted names",
        "price_adjustments": "Current adjusted historical prices; adjustment revisions are not independently replayed",
        "execution": "Monthly signal after close; trade following session close; first earned return one session later",
        "terminal_returns": "Assumed stale-price exits; economic proceeds remain unresolved",
        "point_in_time_attested": False,
    }
