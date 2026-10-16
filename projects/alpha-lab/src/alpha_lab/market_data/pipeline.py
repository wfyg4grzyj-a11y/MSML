"""Default market-data pipeline. Sharadar first; Tiingo only for availability gaps."""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
from dotenv import load_dotenv

from .cli import export_snapshot, load_config as load_tiingo_config
from .routing import catalog, select_route
from .routing import ARRAYS_EXTRA_DATASETS
from .sharadar import SharadarClient, SharadarError
from .sharadar_cli import load_config as load_sharadar_config
from .store import MarketStore
from .sync import sync_ticker
from .tiingo import TiingoClient, TiingoError, ticker_symbol
from .arrays import ArraysError, arrays_request


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) - {"sharadar_config", "tiingo_config", "arrays_config", "tickers"}:
        raise SharadarError("Invalid pipeline configuration; API keys belong in .env")
    tickers = raw.get("tickers", [])
    if not isinstance(tickers, list) or any(not isinstance(t, str) for t in tickers):
        raise SharadarError("tickers must be a list of symbols")
    def relative(name):
        value = Path(raw[name])
        return value if value.is_absolute() else path.parent / value
    store, tables, rate = load_sharadar_config(relative("sharadar_config"))
    tiingo = load_tiingo_config(relative("tiingo_config"))
    return store, tables, rate, tiingo, list(dict.fromkeys(ticker_symbol(t) for t in tickers))


def emit_event(value):
    print(value, flush=True)


def sync_pipeline(store, tables, rate, tiingo, tickers, *, dataset=None, start=None, end=None,
                  asset_table=None, force=False, emit=emit_event, arrays_pipeline_config=None,
                  metrics='EPS', period_type='quarterly', fiscal_year=None, fiscal_quarter=None):
    """Update primary tables before inspecting availability; fail closed on errors."""
    if dataset in ARRAYS_EXTRA_DATASETS:
        if len(tickers) != 1 or arrays_pipeline_config is None:
            raise ArraysError('Arrays requires one explicit ticker and pipeline config')
        route = select_route(store,dataset,ticker=tickers[0],start=start,end=end,asset_table=asset_table)
        result = arrays_request(store,arrays_pipeline_config,route,'sync',start=start,end=end,
                                metrics=metrics,period_type=period_type,fiscal_year=fiscal_year,fiscal_quarter=fiscal_quarter)
        emit(json.dumps(result))
        return [route]
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    if start and start > end:
        raise SharadarError("Start date must precede end date")
    if dataset and dataset != "daily_prices":
        chosen = [select_route(store, dataset).table]
    elif dataset == "daily_prices":
        chosen = [asset_table] if asset_table else ["stocks", "funds"]
    else:
        chosen = tables
        if tickers:
            chosen = list(dict.fromkeys(chosen + ["stocks", "funds"]))
    with SharadarClient(os.environ.get("SHARADAR_API_KEY", ""), requests_per_minute=rate) as client:
        for table in chosen:
            emit(json.dumps({"provider": "sharadar", "table": table, "status": "checking_source"}))
            result = store.sync(client, table, force=force,
                event=lambda table, status: emit(json.dumps({"provider": "sharadar", "table": table, "status": status})),
                progress=lambda table, done, total: emit(json.dumps({"table": table, "downloaded_bytes": done, "total_bytes": total})))
            emit(json.dumps(result))
    if dataset not in {None, "daily_prices"}:
        return []
    routes = [select_route(store, "daily_prices", ticker=t, start=start, end=end, asset_table=asset_table) for t in tickers]
    for route in routes:
        emit(json.dumps(route.manifest()))
    fallback = [r for r in routes if r.provider == "tiingo"]
    if fallback:
        with TiingoClient(os.environ.get("TIINGO_API_KEY", ""), requests_per_minute=tiingo["rate"]) as client, MarketStore(tiingo["database"]) as cache:
            for route in fallback:
                metadata = client.metadata(route.ticker)
                first = date.fromisoformat(metadata["startDate"][:10])
                if route.sharadar_start and first >= date.fromisoformat(route.sharadar_start):
                    raise SharadarError("Tiingo has no earlier history for this symbol; requested gap cannot be filled")
                requested_start = date.fromisoformat(start) if start else first
                result = sync_ticker(client, cache, route.ticker, max(first, requested_start), date.fromisoformat(end),
                                     overlap_days=tiingo["overlap"], full_refresh=force)
                emit(json.dumps(result | {"provider": "tiingo", "routing": route.manifest()}))
    return routes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sharadar-first market data; no LLM calls")
    parser.add_argument("--config", type=Path, default=Path("data/market/pipeline.json") if Path("data/market/pipeline.json").exists() else Path("data/market.example.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="List routing and adapter availability")
    commands.add_parser("status", help="Inspect cached primary tables offline")
    for command in ("route", "sync", "export"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", default="daily_prices" if command in {"route", "export"} else None)
        child.add_argument("--ticker", action="append")
        child.add_argument("--start-date", type=date.fromisoformat)
        child.add_argument("--end-date", type=date.fromisoformat)
        child.add_argument("--asset-table", choices=["stocks", "funds"])
        child.add_argument("--metrics", default="EPS", help="Arrays forecast metrics, comma-separated")
        child.add_argument("--period-type", choices=["annual","quarterly","semi-annual"], default="quarterly")
        child.add_argument("--fiscal-year", type=int)
        child.add_argument("--fiscal-quarter", choices=["Q1","Q2","Q3","Q4"])
        if command == "sync":
            child.add_argument("--env-file", type=Path, default=Path(".env"))
            child.add_argument("--full-refresh", action="store_true")
        if command == "export":
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--dimension")
            child.add_argument("--columns")
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            print(json.dumps(catalog(), indent=2))
            return 0
        store, tables, rate, tiingo, tickers = load_config(args.config)
        if args.command == "status":
            print(json.dumps({"policy": "sharadar_first", "primary": store.summary(tables),
                              "tiingo_fallback_database": str(tiingo["database"]), "tickers": tickers}, indent=2))
            return 0
        tickers = ([ticker_symbol(t) for t in args.ticker] if args.ticker else
                   tickers if args.dataset in {None, "daily_prices"} else [])
        start = args.start_date.isoformat() if args.start_date else None
        end = args.end_date.isoformat() if args.end_date else None
        if args.dataset in ARRAYS_EXTRA_DATASETS:
            if len(tickers) != 1:
                raise ArraysError("Arrays requires exactly one explicit US-equity ticker")
            if args.command == 'export' and (args.dimension or args.columns):
                raise ArraysError("Arrays exports its native schema; dimension/columns are unsupported")
            route = select_route(store,args.dataset,ticker=tickers[0],start=start,end=end,asset_table=args.asset_table)
            if args.command == 'route':
                print(json.dumps([route.manifest()],indent=2))
                return 0
            if args.command == 'sync':
                load_dotenv(args.env_file,override=False)
            result = arrays_request(store,args.config,route,args.command,
                                    output=args.output if args.command == 'export' else None,
                                    start=start,end=end,metrics=args.metrics,period_type=args.period_type,
                                    fiscal_year=args.fiscal_year,fiscal_quarter=args.fiscal_quarter)
            print(json.dumps(result,indent=2))
            return 0
        if args.dataset in {None, "daily_prices"} and not tickers:
            raise SharadarError("Configure tickers or supply --ticker")
        if args.command == "sync":
            load_dotenv(args.env_file, override=False)
            sync_pipeline(store, tables, rate, tiingo, tickers, dataset=args.dataset, start=start, end=end,
                          asset_table=args.asset_table, force=args.full_refresh)
            return 0
        if args.dataset == "daily_prices" and args.command == "export" and len(tickers) != 1:
            raise SharadarError("Export daily_prices for one ticker at a time to preserve provider-specific columns")
        routes = [select_route(store, args.dataset, ticker=t, start=start, end=end, asset_table=args.asset_table)
                  for t in tickers] if args.dataset == "daily_prices" else [select_route(store, args.dataset, start=start, end=end)]
        if args.command == "route":
            print(json.dumps([r.manifest() for r in routes], indent=2))
            return 0
        route = routes[0]
        if route.provider == "sharadar":
            result = store.export(route.table, args.output, tickers=tickers or None, start=start, end=end,
                                  dimension=args.dimension, columns=args.columns.split(",") if args.columns else None,
                                  routing=route.manifest())
        else:
            if args.dimension or args.columns:
                raise SharadarError("Tiingo fallback exports its complete native schema; dimension/columns are unsupported")
            if not tiingo["database"].is_file():
                raise SharadarError("Tiingo fallback is not cached; run sync with the same ticker/date request")
            with MarketStore(tiingo["database"]) as cache:
                if route.sharadar_start:
                    first, _ = cache.bounds(route.ticker)
                    if first is None or first >= date.fromisoformat(route.sharadar_start):
                        raise SharadarError("Earlier Tiingo history is not cached; run sync with the same ticker/date request")
                result = export_snapshot(cache, args.output, [route.ticker], start, end, routing=route.manifest())
        print(json.dumps(result, indent=2))
        return 0
    except (SharadarError, TiingoError, ArraysError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error):
        print("Cannot read pipeline configuration or local data; check settings and paths", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
