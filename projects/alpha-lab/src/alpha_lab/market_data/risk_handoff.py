"""Pinned read-only risk-model sources and shared benchmark/cash/calendar inputs.

This is a source handoff, not an attested RiskPanel. Historical identity,
eligibility, accounting and terminal-action preparation remain separate audits.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

import duckdb
import httpx

from .sharadar import SharadarError, file_hash
from .sharadar_cli import load_config
from .sharadar_store import utc_now
from .tiingo import ticker_symbol

REQUIRED = {
    "stocks": "ticker date close closeunadj closeadj volume lastupdated".split(),
    "funds": "ticker date close closeunadj closeadj volume lastupdated".split(),
    "daily": "ticker date marketcap lastupdated".split(),
    "fundamentals": ("ticker dimension date reportperiod calendardate fiscalperiod lastupdated assets debt liabilities "
                     "netinc netinccmnusd ncfo fxusd equity equityusd prefdivis sharesbas sharefactor").split(),
    "tickers": ("table permaticker ticker category exchange isdelisted siccode currency location relatedtickers "
                "firstpricedate lastpricedate firstquarter lastquarter secfilings lastupdated name").split(),
    "actions": "date action ticker name value contraticker contraname".split(),
    "descriptions": "table indicator title description unittype".split(),
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_PAGE = "https://fred.stlouisfed.org/series/DGS3MO"


def pin_sources(store):
    """Hash pinned versions; never follow latest again during this preparation."""
    sources = {}
    with duckdb.connect(":memory:", config={"memory_limit": "256MB", "threads": "2"}) as db:
        for table, required in REQUIRED.items():
            snapshot = store.latest(table)
            if snapshot is None:
                raise SharadarError("Download all risk-model source tables before creating a handoff")
            path = Path(snapshot["path"]).resolve(strict=True)
            if not path.is_relative_to(store.root) or path.parent.name != snapshot["version"]:
                raise SharadarError("Source snapshot escapes its versioned storage root")
            columns = db.from_parquet(str(path)).columns
            if set(required) - set(columns):
                raise SharadarError(f"Required equity-risk fields are absent in {table}")
            if file_hash(path) != snapshot["parquet_sha256"]:
                raise SharadarError("Pinned source hash mismatch; stop preparation")
            sources[table] = snapshot | {"path": str(path), "hash_verified": True,
                "manifest_path": str(path.parent / "manifest.json"),
                "manifest_sha256": file_hash(path.parent / "manifest.json"),
                "required_fields_present": True}
    return sources


def parse_cash_csv(payload):
    """FRED daily observations; missing holidays remain absent, not zero yields."""
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if rows.fieldnames not in (["observation_date", "DGS3MO"], ["DATE", "DGS3MO"]):
        raise SharadarError("Expected a FRED DGS3MO daily CSV")
    dates, yields = [], []
    previous = None
    for row in rows:
        day = date.fromisoformat(row[rows.fieldnames[0]])
        if previous is not None and day <= previous:
            raise SharadarError("Cash observation dates must be unique and increasing")
        previous = day
        if row["DGS3MO"] in {"", "."}:
            continue
        value = float(row["DGS3MO"])
        if not math.isfinite(value) or not -100 < value < 100:
            raise SharadarError("Invalid annual cash-proxy yield")
        dates.append(day)
        yields.append(value)
    if not dates:
        raise SharadarError("No valid cash yield observations")
    return dates, yields


def cash_accrual(sessions, previous_sessions, observation_dates, yields, *, max_age_days=7):
    """ACT/365F simple accrual at the strictly lagged rate known at prior close.

    A date-only observation first becomes admissible on the next US session.
    For the interval (previous close, current close], freeze the latest
    observation dated strictly before the previous session, including weekends.
    """
    if len(observation_dates) != len(yields) or len(sessions) != len(previous_sessions):
        raise SharadarError("Cash input lengths differ")
    returns, chosen_dates, chosen_yields = [], [], []
    for current, previous in zip(sessions, previous_sessions):
        current, previous = date.fromisoformat(str(current)), date.fromisoformat(str(previous))
        index = bisect_left(observation_dates, previous) - 1
        if index < 0 or not 0 < (previous - observation_dates[index]).days <= max_age_days:
            raise SharadarError("Cash proxy is missing or exceeds the permitted carry-forward age")
        days = (current - previous).days
        if days <= 0:
            raise SharadarError("Cash intervals must span increasing session closes")
        returns.append(yields[index] / 100 * days / 365)
        chosen_dates.append(str(observation_dates[index]))
        chosen_yields.append(yields[index])
    return returns, chosen_dates, chosen_yields


def fetch_cash(start, end):
    """Public series only; no API key, subscription or LLM client."""
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.get(FRED_URL, params={"id": "DGS3MO", "cosd": start, "coed": end})
            if response.status_code != 200 or len(response.content) > 5_000_000:
                raise SharadarError("Cannot download the public FRED cash series; supply --cash-csv")
            payload = response.content
            parse_cash_csv(payload)
            return payload
    except httpx.HTTPError:
        raise SharadarError("FRED request failed; retry or supply --cash-csv") from None


def build_series(sources, cash_payload, *, start=None, end=None):
    try:
        import exchange_calendars as calendars
        import numpy as np
    except ImportError:
        raise SharadarError("Install AlphaLab's risk-data extra for benchmark/calendar preparation") from None
    with duckdb.connect(":memory:", config={"memory_limit": "256MB", "threads": "2"}) as db:
        rows = db.execute('SELECT date, closeadj FROM read_parquet(?) WHERE ticker = ? ORDER BY date',
                          [sources["funds"]["path"], "SPY"]).fetchall()
    prices = {str(day): value for day, value in rows}
    if len(prices) != len(rows) or not rows:
        raise SharadarError("Benchmark has duplicate dates or no SPY observations")
    stock_start, stock_end = sources["stocks"]["start_date"], sources["stocks"]["end_date"]
    anchor = max(stock_start, str(rows[0][0]))
    finish = end or min(stock_end, str(rows[-1][0]))
    if start and start <= anchor:
        raise SharadarError("Benchmark returns need a preceding Sharadar close; choose a start after its first price date")
    if end and end > min(stock_end, str(rows[-1][0])):
        raise SharadarError("Requested end exceeds complete stock/benchmark coverage")
    if finish <= anchor:
        raise SharadarError("No common stock/benchmark session range")
    calendar = calendars.get_calendar("XNYS", start=anchor, end=finish)
    all_sessions = [str(day.date()) for day in calendar.sessions]
    if len(all_sessions) < 2 or all_sessions[0] != anchor:
        raise SharadarError("Price-history anchor is not an XNYS session")
    sessions, previous = [], []
    for before, current in zip(all_sessions, all_sessions[1:]):
        if start is None or current >= start:
            sessions.append(current)
            previous.append(before)
    if not sessions:
        raise SharadarError("Requested range has no complete sessions")
    missing = sorted(set(sessions + previous) - set(prices))
    if missing:
        raise SharadarError("Benchmark prices do not cover every supplied XNYS session")
    values = [prices[t] / prices[p] - 1 if prices[p] is not None and prices[t] is not None
              and math.isfinite(prices[p]) and math.isfinite(prices[t]) and min(prices[p], prices[t]) > 0
              else float("nan") for t, p in zip(sessions, previous)]
    if not all(math.isfinite(v) and v >= -1 for v in values):
        raise SharadarError("Benchmark returns must be complete and valid")
    observation_dates, yields = parse_cash_csv(cash_payload)
    cash, chosen_dates, chosen_yields = cash_accrual(sessions, previous, observation_dates, yields)
    schedule = calendar.schedule.loc[sessions]
    early = {str(day.date()) for day in calendar.early_closes}
    arrays = {"dates": np.array(sessions, dtype="datetime64[D]"),
              "previous_dates": np.array(previous, dtype="datetime64[D]"),
              "market_returns": np.array(values, dtype=float), "cash_returns": np.array(cash, dtype=float),
              "cash_yield_observation_dates": np.array(chosen_dates, dtype="datetime64[D]"),
              "cash_yield_percent": np.array(chosen_yields, dtype=float),
              "session_close_utc": np.array([str(v) for v in schedule["close"]]),
              "early_close": np.array([day in early for day in sessions], dtype=bool)}
    metadata = {
        "start_date": sessions[0], "end_date": sessions[-1], "sessions": len(sessions),
        "benchmark_id": "SP500_SPY_ETF_SHARADAR_TOTAL_RETURN_PROXY",
        "benchmark_method": "SPY closeadj[t]/closeadj[previous XNYS session]-1; includes ETF expenses/tracking differences; not official SPTR",
        "benchmark_reference": "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy",
        "cash_id": "FRED_DGS3MO_LAGGED_ACT365F_SIMPLE_PROXY",
        "cash_method": "yield_percent/100 * calendar_days_between_adjacent_closes/365; latest observation strictly before the previous close date; maximum carry-forward age 7 calendar days",
        "cash_note": "Treasury-yield cash-accrual proxy, not a traded bill total return or actual overnight deposit rate",
        "cash_reference": FRED_PAGE,
        "calendar_id": f"XNYS_exchange_calendars_{calendars.__version__}",
        "calendar_method": "XNYS modeled exchange rules including ad-hoc holidays and early closes; benchmark completeness validated for every session",
        "calendar_reference": "https://github.com/gerrymanoim/exchange_calendars",
        "early_close_sessions": int(arrays["early_close"].sum()),
        "source_vintage_mode": "current_vendor_snapshot",
        "availability_limitation": "Conservative observation-date lag; historical source arrival times and revision vintages are not independently verified",
    }
    return arrays, metadata


def load_handoff(directory, *, verify_sources=False):
    """Check completion and file hashes before letting a consumer use the bundle."""
    directory = Path(directory).resolve(strict=True)
    if (directory / "INCOMPLETE").exists():
        raise SharadarError("Handoff was interrupted and is incomplete")
    complete = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    names = {"sources.json", "shared_series.npz", "cash-DGS3MO.csv", "DATA_INTERFACE.md"}
    if set(complete["sha256"]) != names:
        raise SharadarError("Invalid handoff completion record")
    if any(file_hash(directory / name) != complete["sha256"][name] for name in names):
        raise SharadarError("Handoff file hash mismatch")
    bundle = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    if bundle.get("kind") != "equity_risk_source_handoff" or bundle.get("schema_version") != 1:
        raise SharadarError("Unsupported risk source handoff")
    root = Path(bundle["sharadar_root"]).resolve(strict=True)
    if set(bundle["sources"]) != set(REQUIRED):
        raise SharadarError("Risk source bundle is incomplete")
    for table, source in bundle["sources"].items():
        path = Path(source["path"]).resolve(strict=True)
        expected = root / "tables" / table / source["version"] / "data.parquet"
        if path != expected or not path.is_relative_to(root):
            raise SharadarError("Pinned source path does not match the recorded version")
        manifest_path = path.parent / "manifest.json"
        if file_hash(manifest_path) != source["manifest_sha256"]:
            raise SharadarError("Pinned source manifest hash mismatch")
        if verify_sources and file_hash(path) != source["parquet_sha256"]:
            raise SharadarError("Pinned source data hash mismatch")
    return bundle


def iter_source_rows(bundle, table, *, columns=None, tickers=None, start=None, end=None, dimensions=None):
    """Filtered, streamed dictionary rows from pinned paths; never repin latest.

    Consumers should project only needed columns and filter dates/tickers.
    Quoting every field also supports the provider's reserved `table` column.
    """
    if table not in REQUIRED:
        raise SharadarError("Table is not part of this handoff")
    source = bundle["sources"][table]
    names = [c["name"] for c in source["schema"]["columns"]]
    columns = names if columns is None else columns
    if not columns or len(set(columns)) != len(columns) or not set(columns).issubset(names):
        raise SharadarError("Unknown or duplicate source projection columns")
    from .sharadar_store import identifier
    conditions, parameters = [], [source["path"]]
    if tickers is not None:
        if not tickers or "ticker" not in names:
            raise SharadarError("Supply nonempty tickers for a table with a ticker column")
        conditions.append('"ticker" IN (' + ",".join("?" for _ in tickers) + ")")
        parameters.extend(ticker_symbol(t) for t in tickers)
    for value, operator in ((start, ">="), (end, "<=")):
        if value:
            if "date" not in names:
                raise SharadarError("This source table has no date column")
            date.fromisoformat(str(value))
            conditions.append(f'CAST("date" AS DATE) {operator} CAST(? AS DATE)')
            parameters.append(str(value))
    if start and end and str(start) > str(end):
        raise SharadarError("Start date must precede end date")
    if dimensions is not None:
        if not dimensions or "dimension" not in names or set(dimensions) - {"ARQ", "ART", "ARY"}:
            raise SharadarError("Only explicit as-reported accounting dimensions are permitted")
        conditions.append('"dimension" IN (' + ",".join("?" for _ in dimensions) + ")")
        parameters.extend(dimensions)
    query = "SELECT " + ", ".join(identifier(c) for c in columns) + " FROM read_parquet(?)"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    with duckdb.connect(":memory:", config={"memory_limit": "256MB", "threads": "2"}) as db:
        cursor = db.execute(query, parameters)
        while batch := cursor.fetchmany(4096):
            yield from (dict(zip(columns, row)) for row in batch)


def create_handoff(store, destination, *, cash_payload, interface, start=None, end=None):
    """Create a unique handoff directory after all sources and series validate."""
    import numpy as np
    destination = Path(destination).resolve()
    if destination.exists():
        raise SharadarError("Handoff already exists; choose a new output directory")
    if destination.is_relative_to(store.root):
        raise SharadarError("Keep consumer handoffs outside the Sharadar source store")
    interface = Path(interface).resolve(strict=True)
    interface_bytes = interface.read_bytes()
    interface_hash = hashlib.sha256(interface_bytes).hexdigest()
    sources = pin_sources(store)
    arrays, shared = build_series(sources, cash_payload, start=start, end=end)
    if any(a.dtype.hasobject for a in arrays.values()):
        raise SharadarError("Object arrays are forbidden in consumer artifacts")
    readiness = {
        "raw_snapshot_access_ready": True, "required_source_fields_present": True,
        "shared_series_ready": True, "risk_panel_ready": False,
        "point_in_time_attested": False, "terminal_actions_reconciled": False,
        "pending_preparation": [
            "Effective-dated security/issuer/class mappings and audited total issuer cap",
            "Dated domicile/exchange/common-stock/equity-REIT eligibility",
            "SIC dictionary validation and dated anchors or documented complete change chains",
            "ARQ/ART accounting selection with next-session lag, fiscal-period matching and currency checks",
            "Verified common book equity or explicit missing-book fallback",
            "Terminal-action reconciliation, with unresolved economic outcomes marked missing",
            "Listing-age evidence or separately labeled price-coverage lower bound",
            "Historical availability/revision assessment before public_information_reconstruction attestation",
        ],
        "field_conventions": {
            "market_cap": "daily.marketcap USD millions * 1e6, strictly lagged and repriced; issuer/class audit required",
            "dollar_volume": "stocks.close * stocks.volume on the same split basis",
            "raw_close": "stocks.closeunadj",
            "total_returns": "stocks.closeadj ratios on adjacent official sessions; action audit required",
            "accounting": "ARQ/ART (ARY only explicit fallback), date strictly before forecast session; never MR dimensions",
            "identity": "tickers.permaticker is a security identifier, not a dated issuer relationship",
            "metadata_tables": "tickers.table/descriptions.table use legacy uppercase SEP/SFP/SF1/DAILY identifiers",
            "missingness": "NaN for unavailable numeric facts; empty Unicode for unresolved issuer/SIC; no zero-fill",
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    # INCOMPLETE remains on write failure; consumers must require COMPLETE.json.
    (destination / "INCOMPLETE").write_text("Consumer must require COMPLETE.json\n", encoding="utf-8")
    (destination / "cash-DGS3MO.csv").write_bytes(cash_payload)
    (destination / "DATA_INTERFACE.md").write_bytes(interface_bytes)
    np.savez_compressed(destination / "shared_series.npz", **arrays)
    bundle = {"kind": "equity_risk_source_handoff", "schema_version": 1, "created_at": utc_now(),
              "sharadar_root": str(store.root), "interface_path": str(interface), "interface_sha256": interface_hash,
              "sources": sources, "shared_series": shared, "readiness": readiness,
              "usage": "Read pinned sources[table].path directly with filtered Parquet queries; do not follow latest.json again. shared_series.npz loads with allow_pickle=False. This directory is not a RiskPanel."}
    (destination / "sources.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    complete = {name: file_hash(destination / name) for name in ("sources.json", "shared_series.npz", "cash-DGS3MO.csv", "DATA_INTERFACE.md")}
    (destination / "COMPLETE.json").write_text(json.dumps({"sha256": complete}, indent=2) + "\n", encoding="utf-8")
    (destination / "INCOMPLETE").unlink()
    return {"directory": str(destination), "sources": len(sources), "shared_series": shared, "readiness": readiness}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only equity-risk source/benchmark/cash handoff; no LLM calls")
    parser.add_argument("--config", type=Path, default=Path("data/market/sharadar.json"))
    parser.add_argument("--interface", type=Path, default=Path("../equity-risk/docs/DATA_INTERFACE.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    cash = parser.add_mutually_exclusive_group(required=True)
    cash.add_argument("--cash-csv", type=Path)
    cash.add_argument("--download-cash", action="store_true")
    args = parser.parse_args(argv)
    try:
        store, _, _ = load_config(args.config)
        if args.output.exists():
            raise SharadarError("Handoff already exists; choose a new output directory")
        if args.start_date and args.end_date and args.start_date > args.end_date:
            raise SharadarError("Start date must precede end date")
        snapshot = store.latest("stocks")
        if snapshot is None:
            raise SharadarError("Stock snapshot is not cached")
        begin = (args.start_date or date.fromisoformat(snapshot["start_date"])) - timedelta(days=21)
        end = args.end_date or date.fromisoformat(snapshot["end_date"])
        payload = args.cash_csv.read_bytes() if args.cash_csv else fetch_cash(str(begin), str(end))
        result = create_handoff(store, args.output, cash_payload=payload, interface=args.interface,
                                start=str(args.start_date) if args.start_date else None,
                                end=str(args.end_date) if args.end_date else None)
        print(json.dumps(result, indent=2))
        return 0
    except SharadarError as exc:
        print(str(exc), file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, ImportError, duckdb.Error):
        print("Cannot prepare handoff; check cached sources, cash data, calendar and interface paths", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
