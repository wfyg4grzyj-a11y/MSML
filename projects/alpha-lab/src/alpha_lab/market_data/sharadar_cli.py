"""Direct Sharadar bulk sync and offline queries; no LLM calls."""

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import duckdb
from dotenv import load_dotenv

from .sharadar import SharadarClient, SharadarError, table_name
from .sharadar_store import SharadarStore


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) - {"root", "tables", "requests_per_minute", "memory_limit", "threads"}:
        raise SharadarError("Invalid Sharadar configuration; API keys belong in .env")
    tables = raw.get("tables")
    if not isinstance(tables, list) or not tables or any(not isinstance(t, str) for t in tables):
        raise SharadarError("Configure a nonempty tables list")
    rate = raw.get("requests_per_minute", 30)
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 0 < rate <= 600:
        raise SharadarError("Invalid requests_per_minute")
    threads = raw.get("threads", 4)
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise SharadarError("Invalid threads setting")
    root = Path(raw.get("root", "market/sharadar"))
    root = root if root.is_absolute() else path.parent / root
    store = SharadarStore(root, memory_limit=raw.get("memory_limit", "4GB"), threads=threads)
    return store, list(dict.fromkeys(table_name(t) for t in tables)), rate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Direct sharadar.com full-history data")
    parser.add_argument("--config", type=Path, default=Path("data/sharadar.example.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("sync", "probe"):
        child = commands.add_parser(command)
        child.add_argument("--table", action="append")
        child.add_argument("--env-file", type=Path, default=Path(".env"))
        if command == "sync":
            child.add_argument("--force", action="store_true")
    commands.add_parser("status", help="Inspect local snapshots offline")
    export = commands.add_parser("export", help="Write a filtered offline CSV")
    export.add_argument("--table", required=True)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--ticker", action="append")
    export.add_argument("--start-date", type=date.fromisoformat)
    export.add_argument("--end-date", type=date.fromisoformat)
    export.add_argument("--dimension")
    export.add_argument("--observed-before", help="Use only a snapshot captured by this ISO timestamp with timezone")
    export.add_argument("--columns", help="Comma-separated original field names")
    args = parser.parse_args(argv)
    try:
        store, tables, rate = load_config(args.config)
        if args.command == "status":
            print(json.dumps(store.summary(tables), indent=2))
        elif args.command == "export":
            result = store.export(args.table, args.output, tickers=args.ticker,
                                  start=str(args.start_date) if args.start_date else None,
                                  end=str(args.end_date) if args.end_date else None,
                                  dimension=args.dimension, columns=args.columns.split(",") if args.columns else None,
                                  observed_before=args.observed_before)
            print(json.dumps(result, indent=2))
        else:
            load_dotenv(args.env_file, override=False)
            tables = [table_name(t) for t in args.table] if args.table else tables
            failures = 0
            def progress(table, received, total):
                print(json.dumps({"table": table, "downloaded_bytes": received, "total_bytes": total}), flush=True)
            def event(table, status):
                print(json.dumps({"table": table, "status": status}), flush=True)
            with SharadarClient(os.environ.get("SHARADAR_API_KEY", ""), requests_per_minute=rate) as client:
                for table in tables:
                    try:
                        event(table, "checking_source")
                        result = client.status(table) if args.command == "probe" else store.sync(client, table, force=args.force, progress=progress, event=event)
                        print(json.dumps(result), flush=True)
                    except SharadarError as exc:
                        failures += 1
                        print(json.dumps({"table": table, "status": "failed", "error": str(exc)}), file=sys.stderr, flush=True)
                    except OSError:
                        failures += 1
                        print(json.dumps({"table": table, "status": "failed", "error": "Download or validation failed; retry sync to resume. Previous snapshot retained."}), file=sys.stderr, flush=True)
            return int(bool(failures))
        return 0
    except SharadarError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error):
        # Raw database/HTTP diagnostics may contain input values or authenticated URLs.
        print("Cannot read configuration or local data; check settings and paths", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
