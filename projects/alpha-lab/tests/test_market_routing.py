"""Availability-based provider selection and fail-closed default pipeline."""

import hashlib
import json
import zipfile
from contextlib import nullcontext

import pytest

from alpha_lab.market_data import pipeline
from alpha_lab.market_data.routing import catalog, select_route
from alpha_lab.market_data.sharadar import SharadarError, file_hash, parse_schema
from alpha_lab.market_data.sharadar_store import SharadarStore


@pytest.fixture
def lake(tmp_path):
    store = SharadarStore(tmp_path / "lake", memory_limit="256MB", threads=1)
    for table, symbol in (("stocks", "AAPL"), ("funds", "SPY")):
        ddl = f'''CREATE TABLE IF NOT EXISTS {table} (
  ticker text NOT NULL,
  date date NOT NULL,
  close double precision,
  volume double precision,
  PRIMARY KEY (ticker, date)
);
'''
        archive = tmp_path / f"{table}.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(table + ".csv", f"ticker,date,close,volume\n{symbol},1997-12-31,10,100\n{symbol},2026-10-12,20,200\n")
        store.publish(table, archive, parse_schema(ddl, table),
                      {"table": table, "modified": "now", "size": archive.stat().st_size, "history": "full"},
                      {"sha256": file_hash(archive), "bytes": archive.stat().st_size})
    return store


def test_default_fund_and_stock_routes(lake):
    assert select_route(lake, "daily_prices", ticker="spy").table == "funds"
    route = select_route(lake, "daily_prices", ticker="AAPL", start="2000-01-01")
    assert route.provider == "sharadar"
    assert route.sharadar_start == "1997-12-31"


def test_earlier_history_selects_one_tiingo_series(lake):
    route = select_route(lake, "daily_prices", ticker="SPY", start="1993-01-29")
    assert route.provider == "tiingo"
    assert route.reason == "earlier_history"
    assert route.manifest()["series_policy"] == "one_provider_per_request"


def test_end_before_sharadar_history(lake):
    assert select_route(lake, "daily_prices", ticker="SPY", end="1995-01-01").provider == "tiingo"


def test_future_end_does_not_treat_staleness_as_missing_dataset(lake):
    assert select_route(lake, "daily_prices", ticker="SPY", end="2027-01-01").provider == "sharadar"


def test_absent_symbol_falls_back(lake):
    assert select_route(lake, "daily_prices", ticker="QQQ").reason == "symbol_absent"


def test_missing_cache_is_not_provider_unavailability(tmp_path):
    with pytest.raises(SharadarError, match="Sync Sharadar"):
        select_route(SharadarStore(tmp_path), "daily_prices", ticker="SPY")


def test_corrupt_primary_never_falls_back(lake):
    (lake.root / "tables/stocks/latest.json").write_text("invalid")
    with pytest.raises(SharadarError, match="Invalid local"):
        select_route(lake, "daily_prices", ticker="SPY")


def test_ambiguous_symbol_requires_asset_table(lake):
    # The fixture's two assets now share a symbol, simulating namespace overlap.
    with lake._connect() as db:
        funds = lake.latest("funds")["path"]
        altered = db.sql("SELECT 'AAPL' AS ticker, date, close, volume FROM read_parquet(?)", params=[funds])
        replacement = lake.root / "replacement.parquet"
        altered.write_parquet(str(replacement))
    original_latest = lake.latest
    lake.latest = lambda table: (original_latest(table) | {"path": str(replacement)}) if table == "funds" else original_latest(table)
    with pytest.raises(SharadarError, match="specify asset_table"):
        select_route(lake, "daily_prices", ticker="AAPL")
    assert select_route(lake, "daily_prices", ticker="AAPL", asset_table="stocks").table == "stocks"


@pytest.mark.parametrize("dataset", ["fundamentals", "daily", "sp500", "SF1", "SEP", "holdings"])
def test_sharadar_datasets_never_use_tiingo(lake, dataset):
    assert select_route(lake, dataset, start="1900-01-01").provider == "sharadar"


@pytest.mark.parametrize("dataset", ["news", "intraday_prices", "fund_fees", "dividend_details", "overnight_prices"])
def test_catalog_does_not_claim_unimplemented_adapters(lake, dataset):
    assert next(x for x in catalog() if x["dataset"] == dataset)["adapter"] == "not_implemented"
    with pytest.raises(SharadarError, match="adapter is not implemented"):
        select_route(lake, dataset)


def fake_primary(monkeypatch, lake):
    monkeypatch.setattr(pipeline, "SharadarClient", lambda *a, **k: nullcontext(object()))
    monkeypatch.setattr(lake, "sync", lambda client, table, **k: {"table": table, "status": "unchanged"})


def test_sync_default_never_constructs_tiingo_when_primary_covers(lake, monkeypatch):
    fake_primary(monkeypatch, lake)
    def forbidden(*a, **k):
        pytest.fail("Tiingo should not be called for covered requests")
    monkeypatch.setattr(pipeline, "TiingoClient", forbidden)
    routes = pipeline.sync_pipeline(lake, ["stocks", "funds"], 30, {}, ["SPY"], emit=lambda _: None)
    assert routes[0].provider == "sharadar"


def test_primary_failure_does_not_trigger_fallback(lake, monkeypatch):
    fake_primary(monkeypatch, lake)
    def failure(*a, **k):
        raise SharadarError("HTTP 401")
    monkeypatch.setattr(lake, "sync", failure)
    monkeypatch.setattr(pipeline, "TiingoClient", lambda *a, **k: pytest.fail("No fallback on errors"))
    with pytest.raises(SharadarError, match="HTTP 401"):
        pipeline.sync_pipeline(lake, ["stocks"], 30, {}, ["SPY"], start="1993-01-29", emit=lambda _: None)


def bar(day):
    return {"date": day + "T00:00:00.000Z", "open": 10, "high": 11, "low": 9, "close": 10,
            "volume": 100, "adjOpen": 5, "adjHigh": 5.5, "adjLow": 4.5, "adjClose": 5,
            "adjVolume": 200, "divCash": 0, "splitFactor": 1}


def test_fallback_sync_uses_requested_range_and_native_export(lake, monkeypatch, tmp_path):
    fake_primary(monkeypatch, lake)
    requests = []
    class Tiingo:
        def metadata(self, ticker):
            return {"ticker": ticker, "startDate": "1993-01-29"}
        def prices(self, ticker, start, end):
            requests.append((ticker, str(start), str(end)))
            return [bar("1993-01-29"), bar("2026-10-12")]
    monkeypatch.setattr(pipeline, "TiingoClient", lambda *a, **k: nullcontext(Tiingo()))
    config = {"rate": 30, "database": tmp_path / "tiingo.sqlite3", "overlap": 7}
    routes = pipeline.sync_pipeline(lake, ["stocks", "funds"], 30, config, ["SPY"],
                                    start="1993-01-29", end="2026-10-12", emit=lambda _: None)
    assert requests == [("SPY", "1993-01-29", "2026-10-12")]
    monkeypatch.setattr(pipeline, "load_config", lambda _: (lake, [], 30, config, ["SPY"]))
    out = tmp_path / "prices.csv"
    assert pipeline.main(["export", "--start-date", "1993-01-29", "--output", str(out)]) == 0
    manifest = json.loads(out.with_suffix(".csv.manifest.json").read_text())
    assert manifest["provider"] == "tiingo"
    assert manifest["routing"] == routes[0].manifest()
    assert "adj_close" in manifest["columns"]
    assert manifest["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_tiingo_with_no_earlier_coverage_fails(lake, monkeypatch, tmp_path):
    fake_primary(monkeypatch, lake)
    class Tiingo:
        def metadata(self, ticker):
            return {"ticker": ticker, "startDate": "1997-12-31"}
        def prices(self, *a):
            pytest.fail("No price download when there is no earlier coverage")
    monkeypatch.setattr(pipeline, "TiingoClient", lambda *a, **k: nullcontext(Tiingo()))
    with pytest.raises(SharadarError, match="no earlier history"):
        pipeline.sync_pipeline(lake, [], 30, {"database": tmp_path / "tiingo.db", "rate": 30}, ["SPY"],
                               start="1993-01-29", emit=lambda _: None)


def test_primary_export_provenance_and_immutable_output(lake, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "load_config", lambda _: (lake, [], 30, {}, ["SPY"]))
    out = tmp_path / "prices.csv"
    assert pipeline.main(["export", "--output", str(out)]) == 0
    manifest = json.loads(out.with_suffix(".csv.manifest.json").read_text())
    assert manifest["provider"] == "sharadar.com"
    assert manifest["routing"]["reason"] == "dataset_available"
    assert "split-adjusted" in manifest["adjustments"]
    before = out.read_bytes()
    assert pipeline.main(["export", "--output", str(out)]) == 1
    assert out.read_bytes() == before


def test_reversed_dates_fail_before_network(lake, monkeypatch):
    monkeypatch.setattr(pipeline, "SharadarClient", lambda *a, **k: pytest.fail("No network"))
    with pytest.raises(SharadarError, match="Start date"):
        pipeline.sync_pipeline(lake, [], 30, {}, ["SPY"], start="2026-10-12", end="2026-10-01")


def test_multi_ticker_export_rejected(lake, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "load_config", lambda _: (lake, [], 30, {}, ["SPY", "AAPL"]))
    assert pipeline.main(["export", "--output", str(tmp_path / "out.csv")]) == 1
    assert not (tmp_path / "out.csv").exists()


def test_keys_cannot_be_embedded_in_pipeline_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"SHARADAR_API_KEY":"secret"}')
    with pytest.raises(SharadarError, match="API keys belong"):
        pipeline.load_config(config)
