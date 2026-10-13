"""SQLite canonical bars plus append-only provider response and sync audits."""

import hashlib
import json
import math
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from .tiingo import ticker_symbol

FIELDS = {
    "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
    "adj_open": "adjOpen", "adj_high": "adjHigh", "adj_low": "adjLow",
    "adj_close": "adjClose", "adj_volume": "adjVolume",
    "div_cash": "divCash", "split_factor": "splitFactor",
}
COLUMNS = ["ticker", "date", *FIELDS, "fetched_at"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_bars(ticker: str, payload: list[dict], start: date, end: date) -> list[dict]:
    ticker = ticker_symbol(ticker)
    bars, seen = [], set()
    for row in payload:
        try:
            stamp = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
            day = stamp.date()
            if stamp.tzinfo is not None and stamp.utcoffset().total_seconds() != 0:
                raise ValueError("Expected a UTC daily timestamp")
            if stamp.time().replace(tzinfo=None).isoformat() != "00:00:00":
                raise ValueError("Expected midnight daily timestamp")
            if not start <= day <= end or day in seen:
                raise ValueError("Duplicate or out-of-range date")
            bar = {"ticker": ticker, "date": day.isoformat()}
            for column, field in FIELDS.items():
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {field}")
                bar[column] = value
            if bar["volume"] < 0 or int(bar["volume"]) != bar["volume"] or bar["volume"] > 2**63 - 1:
                raise ValueError("Invalid volume")
            bar["volume"] = int(bar["volume"])
            if bar["adj_volume"] < 0 or bar["div_cash"] < 0 or bar["split_factor"] <= 0:
                raise ValueError("Invalid corporate action or adjusted volume")
            for prefix in ("", "adj_"):
                low, high = bar[prefix + "low"], bar[prefix + "high"]
                tolerance = max(abs(high), 1) * 1e-8
                if low <= 0 or high + tolerance < low:
                    raise ValueError("Invalid price range")
                for field in ("open", "close"):
                    if not low - tolerance <= bar[prefix + field] <= high + tolerance:
                        raise ValueError("OHLC prices outside low/high")
            bars.append(bar)
            seen.add(day)
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            # Do not include arbitrary upstream values in logs or exceptions.
            raise ValueError(f"Invalid daily bar for {ticker}; missing fields, dates or OHLCV values") from None
    return sorted(bars, key=lambda bar: bar["date"])


class MarketStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
            self.db.close()
            raise ValueError("Unsupported market database version")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        numeric = ", ".join(f"{name} {'INTEGER' if name == 'volume' else 'REAL'} NOT NULL" for name in FIELDS)
        self.db.executescript(f"""
            CREATE TABLE IF NOT EXISTS assets (
                ticker TEXT PRIMARY KEY, metadata_json TEXT NOT NULL, fetched_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bars (
                ticker TEXT NOT NULL REFERENCES assets(ticker), date TEXT NOT NULL,
                {numeric}, fetched_at TEXT NOT NULL, PRIMARY KEY(ticker, date)
            );
            CREATE TABLE IF NOT EXISTS syncs (
                id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
                status TEXT NOT NULL, row_count INTEGER NOT NULL, full_refresh INTEGER NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY, sync_id INTEGER NOT NULL REFERENCES syncs(id),
                kind TEXT NOT NULL, start_date TEXT, end_date TEXT,
                sha256 TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            PRAGMA user_version=1;
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def bounds(self, ticker: str) -> tuple[date | None, date | None]:
        row = self.db.execute("SELECT MIN(date), MAX(date) FROM bars WHERE ticker=?", (ticker_symbol(ticker),)).fetchone()
        return tuple(date.fromisoformat(value) if value else None for value in row)

    def coverage_start(self, ticker: str) -> date | None:
        """Track requested coverage even when the first day is a holiday or pre-IPO."""
        value = self.db.execute("SELECT MIN(start_date) FROM syncs WHERE ticker=? AND status='ok' AND full_refresh=1",
                                (ticker_symbol(ticker),)).fetchone()[0]
        return date.fromisoformat(value) if value else None

    def prices(self, tickers: list[str] | None = None, *, start: str | None = None,
               end: str | None = None) -> list[dict]:
        clauses, args = [], []
        if tickers is not None:
            if not tickers:
                return []
            clauses.append("ticker IN (" + ",".join("?" for _ in tickers) + ")")
            args.extend(ticker_symbol(t) for t in tickers)
        for field, value, operator in (("date", start, ">="), ("date", end, "<=")):
            if value is not None:
                clauses.append(f"{field} {operator} ?")
                args.append(date.fromisoformat(value).isoformat())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return [dict(row) for row in self.db.execute("SELECT * FROM bars" + where + " ORDER BY ticker, date", args)]

    def summary(self, tickers: list[str] | None = None) -> list[dict]:
        if tickers == []:
            return []
        where = " WHERE ticker IN (" + ",".join("?" for _ in tickers) + ")" if tickers else ""
        args = [ticker_symbol(t) for t in tickers] if tickers else []
        return [dict(row) for row in self.db.execute("""
            SELECT ticker, COUNT(*) AS rows, MIN(date) AS start_date, MAX(date) AS end_date,
                   MAX(fetched_at) AS fetched_at FROM bars
        """ + where + " GROUP BY ticker ORDER BY ticker", args)]

    def commit_sync(self, *, ticker, metadata, bars, responses, started_at, start, end, full_refresh):
        fetched = utc_now()
        with self.db:
            self.db.execute("INSERT INTO assets VALUES (?, ?, ?) ON CONFLICT(ticker) DO UPDATE SET metadata_json=excluded.metadata_json, fetched_at=excluded.fetched_at",
                            (ticker, json.dumps(metadata, allow_nan=False), fetched))
            if full_refresh:
                self.db.execute("DELETE FROM bars WHERE ticker=? AND date BETWEEN ? AND ?",
                                (ticker, start.isoformat(), end.isoformat()))
            columns = ",".join(COLUMNS)
            self.db.executemany(f"INSERT OR REPLACE INTO bars ({columns}) VALUES ({','.join('?' for _ in COLUMNS)})",
                                [tuple(bar.get(key) if key != "fetched_at" else fetched for key in COLUMNS) for bar in bars])
            cursor = self.db.execute("""INSERT INTO syncs
                (ticker,started_at,finished_at,start_date,end_date,status,row_count,full_refresh)
                VALUES (?,?,?,?,?,'ok',?,?)""",
                (ticker, started_at, fetched, start.isoformat(), end.isoformat(), len(bars), int(full_refresh)))
            for kind, response_start, response_end, payload in responses:
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
                self.db.execute("INSERT INTO responses (sync_id,kind,start_date,end_date,sha256,payload_json) VALUES (?,?,?,?,?,?)",
                                (cursor.lastrowid, kind, response_start, response_end,
                                 hashlib.sha256(encoded.encode()).hexdigest(), encoded))
        return cursor.lastrowid

    def record_failure(self, ticker: str, started_at: str, start: date, end: date, error: str):
        with self.db:
            self.db.execute("""INSERT INTO syncs
                (ticker,started_at,finished_at,start_date,end_date,status,row_count,full_refresh,error)
                VALUES (?,?,?,?,?,'failed',0,0,?)""",
                (ticker, started_at, utc_now(), start.isoformat(), end.isoformat(), error))
