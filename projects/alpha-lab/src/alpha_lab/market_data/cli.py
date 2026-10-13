"""Independent market data CLI; no LLM providers are loaded."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from .store import COLUMNS, MarketStore, utc_now
from .sync import sync_ticker
from .tiingo import TiingoClient, TiingoError, ticker_symbol


def load_config(path: Path) -> dict:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = {"tickers", "start_date", "end_date", "database", "requests_per_minute", "overlap_days"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Config has unsupported fields; keep API keys in .env")
    tickers = raw.get("tickers")
    if not isinstance(tickers, list) or not tickers or any(not isinstance(t, str) for t in tickers):
        raise ValueError("Config requires a nonempty tickers list")
    symbols = list(dict.fromkeys(ticker_symbol(t) for t in tickers))
    start = date.fromisoformat(raw["start_date"])
    end = date.fromisoformat(raw["end_date"]) if raw.get("end_date") else datetime.now(timezone.utc).date() - timedelta(days=1)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    rate, overlap = raw.get("requests_per_minute", 30), raw.get("overlap_days", 7)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 600:
        raise ValueError("requests_per_minute must be between 0 and 600")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 1 <= overlap <= 365:
        raise ValueError("overlap_days must be between 1 and 365")
    database = Path(raw.get("database", "market/tiingo.sqlite3"))
    if not database.is_absolute():
        database = path.parent / database
    return {"tickers": symbols, "start": start, "end": end, "database": database.resolve(),
            "rate": rate, "overlap": overlap}


def export_snapshot(store: MarketStore, destination: Path, tickers: list[str],
                    start: str | None = None, end: str | None = None, *, routing: dict | None = None) -> dict:
    """Write a unique CSV snapshot with a content hash and provenance manifest."""
    destination = destination.resolve()
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ValueError("Snapshot already exists; choose a new filename")
    rows = store.prices(tickers, start=start, end=end)
    if not rows:
        raise ValueError("No stored bars match the export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 1, "provider": "tiingo", "frequency": "daily",
            "created_at": utc_now(), "rows": len(rows),
            "filters": {"tickers": tickers, "start": start, "end": end},
            "tickers": sorted({row["ticker"] for row in rows}),
            "start_date": min(row["date"] for row in rows), "end_date": max(row["date"] for row in rows),
            "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(), "columns": COLUMNS,
            "adjustments": "adj_* are provider split/dividend-adjusted; raw OHLCV retained",
            "point_in_time": False,
            "note": "Current back-adjusted history; fetched_at is retrieval time, not historical publication time.",
        }
        # Exclusive creation protects snapshots from accidental overwrites.
        if routing is not None:
            manifest["routing"] = routing
        with destination.open("xb") as output, temporary.open("rb") as source:
            shutil.copyfileobj(source, output)
        try:
            with manifest_path.open("x", encoding="utf-8") as output:
                json.dump(manifest, output, indent=2)
                output.write("\n")
        except Exception:
            destination.unlink()
            raise
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tiingo daily market data (no LLM calls)")
    parser.add_argument("--config", type=Path, default=Path("data/tiingo.example.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync", help="Backfill/update configured tickers")
    sync.add_argument("--full-refresh", action="store_true")
    sync.add_argument("--end-date", type=date.fromisoformat)
    sync.add_argument("--env-file", type=Path, default=Path(".env"))
    commands.add_parser("status", help="Inspect cached data without network access")
    export = commands.add_parser("export", help="Export stored bars without network access")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--start-date", type=date.fromisoformat)
    export.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args(argv)
    try:
        try:
            config = load_config(args.config)
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError("Invalid market data configuration; check fields, dates and file path") from None
        if args.command == "sync":
            load_dotenv(args.env_file, override=False)
            key = os.environ.get("TIINGO_API_KEY", "")
            if not key.strip():
                raise ValueError("Set TIINGO_API_KEY in your environment or local .env")
            end = args.end_date or config["end"]
            if config["start"] > end:
                raise ValueError("start_date must be on or before end_date")
            failures = 0
            with TiingoClient(key, requests_per_minute=config["rate"]) as client, MarketStore(config["database"]) as store:
                for ticker in config["tickers"]:
                    try:
                        result = sync_ticker(client, store, ticker, config["start"], end,
                                             overlap_days=config["overlap"], full_refresh=args.full_refresh)
                        print(json.dumps(result))
                    except (TiingoError, ValueError) as exc:
                        failures += 1
                        print(f"{ticker}: {exc}", file=sys.stderr)
            return 1 if failures else 0
        with MarketStore(config["database"]) as store:
            if args.command == "status":
                print(json.dumps({"database": str(store.path), "assets": store.summary(config["tickers"])}, indent=2))
            else:
                start = args.start_date.isoformat() if args.start_date else None
                end = args.end_date.isoformat() if args.end_date else None
                if start and end and start > end:
                    raise ValueError("start_date must be on or before end_date")
                print(json.dumps(export_snapshot(store, args.output, config["tickers"], start, end), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, TiingoError) as exc:
        # Never dump a malformed config or arbitrary provider response body.
        message = str(exc) if isinstance(exc, TiingoError) or type(exc) is ValueError else "Cannot read configuration or local data file"
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
