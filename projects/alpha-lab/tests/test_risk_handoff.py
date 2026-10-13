"""Source pinning, nonfabricated attestations and dated shared input preparation."""

import json
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pytest

from alpha_lab.market_data.risk_handoff import (
    REQUIRED, build_series, cash_accrual, create_handoff, parse_cash_csv, pin_sources, load_handoff, iter_source_rows,
)
from alpha_lab.market_data.sharadar import SharadarError, file_hash
from alpha_lab.market_data.sharadar_store import SharadarStore

CASH = b"observation_date,DGS3MO\n2024-11-26,5\n2024-11-27,6\n2024-11-28,.\n2024-11-29,7\n2024-12-02,8\n"


@pytest.fixture
def lake(tmp_path):
    store = SharadarStore(tmp_path / "lake")
    with duckdb.connect(":memory:") as db:
        for table, columns in REQUIRED.items():
            base = store.root / "tables" / table
            folder = base / ("a" * 32)
            folder.mkdir(parents=True)
            parquet = folder / "data.parquet"
            if table in {"stocks", "funds"}:
                query = "SELECT 'SPY' AS ticker, CAST(day AS DATE) AS date, price AS closeadj, price AS closeunadj, price AS close, 100 AS volume, CAST(day AS DATE) AS lastupdated FROM (VALUES ('2024-11-27',10.0),('2024-11-29',11.0),('2024-12-02',12.0),('2024-12-03',13.0)) AS p(day,price)"
            else:
                query = "SELECT " + ", ".join(f'NULL::VARCHAR AS "{c}"' for c in columns)
            db.sql(query).write_parquet(str(parquet))
            manifest = {"table": table, "parquet_sha256": file_hash(parquet), "rows": 4 if table in {"stocks", "funds"} else 1,
                        "start_date": "2024-11-27", "end_date": "2024-12-03", "source": {"history": "full"},
                        "schema": {"columns": [{"name": c} for c in columns]}}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            (base / "latest.json").write_text(json.dumps({"version": "a" * 32}))
    return store


def test_missing_yields_are_not_zero_filled():
    days, yields = parse_cash_csv(CASH)
    assert date(2024, 11, 28) not in days
    assert yields == [5, 6, 7, 8]


@pytest.mark.parametrize("payload", [b"date,yield\n2024-01-01,1\n",
    b"DATE,DGS3MO\n2024-01-01,1\n2024-01-01,2\n", b"DATE,DGS3MO\n2024-01-01,nan\n",
    b"DATE,DGS3MO\n2024-01-01,inf\n", b"DATE,DGS3MO\n2024-01-01,.\n"])
def test_reject_invalid_cash_csv(payload):
    with pytest.raises(SharadarError):
        parse_cash_csv(payload)


def test_cash_strict_lag_and_weekend_accrual():
    days, yields = parse_cash_csv(CASH)
    returns, chosen, values = cash_accrual(["2024-12-02"], ["2024-11-29"], days, yields)
    assert chosen == ["2024-11-27"]  # Same-day Friday yield cannot fund Friday's close.
    assert values == [6]
    assert returns[0] == pytest.approx(.06 * 3 / 365)


@pytest.mark.parametrize("observations", [[date(2024, 11, 29)], [date(2024, 11, 1)]])
def test_cash_missing_or_stale_is_an_error(observations):
    with pytest.raises(SharadarError, match="missing or exceeds"):
        cash_accrual(["2024-12-02"], ["2024-11-29"], observations, [5])


def test_pinning_does_not_modify_sources(lake):
    before = {str(p): p.read_bytes() for p in lake.root.rglob("*") if p.is_file()}
    sources = pin_sources(lake)
    assert len(sources) == 7
    assert all(s["hash_verified"] for s in sources.values())
    assert before == {str(p): p.read_bytes() for p in lake.root.rglob("*") if p.is_file()}


def test_tampered_parquet_rejected(lake):
    path = Path(lake.latest("funds")["path"])
    with duckdb.connect(":memory:") as db:
        db.sql("SELECT 'SPY' AS ticker, DATE '2024-11-27' AS date, 1 AS close, 1 AS closeadj, 1 AS closeunadj, 1 AS volume, DATE '2024-11-27' AS lastupdated").write_parquet(str(path))
    with pytest.raises(SharadarError, match="hash mismatch"):
        pin_sources(lake)


def test_missing_source_table_rejected(lake):
    (lake.root / "tables/actions/latest.json").unlink()
    with pytest.raises(SharadarError, match="Download all"):
        pin_sources(lake)


def test_missing_physical_column_rejected(lake):
    path = Path(lake.latest("daily")["path"])
    with duckdb.connect(":memory:") as db:
        db.sql("SELECT 1 AS marketcap").write_parquet(str(path))
    with pytest.raises(SharadarError, match="fields are absent"):
        pin_sources(lake)


def test_shared_series_complete_sessions_and_early_close(lake):
    arrays, metadata = build_series(pin_sources(lake), CASH)
    assert arrays["dates"].astype(str).tolist() == ["2024-11-29", "2024-12-02", "2024-12-03"]
    np.testing.assert_allclose(arrays["market_returns"], [.1, 12/11-1, 13/12-1])
    assert arrays["early_close"].tolist() == [True, False, False]
    assert metadata["early_close_sessions"] == 1
    assert "not official SPTR" in metadata["benchmark_method"]
    assert metadata["source_vintage_mode"] == "current_vendor_snapshot"
    assert all(a.dtype.kind != "O" for a in arrays.values())


def test_missing_benchmark_session_rejected(lake):
    sources = pin_sources(lake)
    path = Path(sources["funds"]["path"])
    with duckdb.connect(":memory:") as db:
        filtered = db.sql("SELECT * FROM read_parquet(?) WHERE date != DATE '2024-12-02'", params=[str(path)])
        replacement = path.with_name("incomplete.parquet")
        filtered.write_parquet(str(replacement))
    sources["funds"]["path"] = str(replacement)
    with pytest.raises(SharadarError, match="every supplied"):
        build_series(sources, CASH)


def test_unsupported_range_not_silently_truncated(lake):
    sources = pin_sources(lake)
    with pytest.raises(SharadarError, match="preceding Sharadar close"):
        build_series(sources, CASH, start="2024-11-26")
    with pytest.raises(SharadarError, match="end exceeds"):
        build_series(sources, CASH, end="2024-12-04")


def test_handoff_is_immutable_and_not_an_attested_model_panel(lake, tmp_path):
    interface = tmp_path / "DATA_INTERFACE.md"
    interface.write_text("RiskPanel contract")
    output = tmp_path / "handoff"
    result = create_handoff(lake, output, cash_payload=CASH, interface=interface)
    assert not result["readiness"]["risk_panel_ready"]
    assert not result["readiness"]["point_in_time_attested"]
    assert not result["readiness"]["terminal_actions_reconciled"]
    assert not (output / "panel.npz").exists()
    assert not (output / "INCOMPLETE").exists()
    with np.load(output / "shared_series.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["market_returns"]).all()
        assert np.isfinite(arrays["cash_returns"]).all()
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert all(file_hash(output / name) == value for name, value in complete["sha256"].items())
    before = {str(p): p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SharadarError, match="already exists"):
        create_handoff(lake, output, cash_payload=CASH, interface=interface)
    assert before == {str(p): p.read_bytes() for p in output.iterdir()}


def test_handoff_cannot_write_inside_source_store(lake, tmp_path):
    with pytest.raises(SharadarError, match="outside"):
        create_handoff(lake, lake.root / "handoff", cash_payload=CASH, interface=tmp_path / "missing")


def test_reader_uses_pinned_versions_and_handles_reserved_columns(lake, tmp_path):
    interface = tmp_path / "interface.md"
    interface.write_text("contract")
    output = tmp_path / "bundle"
    create_handoff(lake, output, cash_payload=CASH, interface=interface)
    # Later ingestion changes latest, while the old immutable version is retained.
    (lake.root / "tables/stocks/latest.json").write_text('{"version":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}')
    bundle = load_handoff(output, verify_sources=True)
    rows = list(iter_source_rows(bundle, "stocks", columns=["ticker", "date"], tickers=["SPY"], start="2024-12-02"))
    assert len(rows) == 2
    assert all(row["ticker"] == "SPY" for row in rows)
    assert list(iter_source_rows(bundle, "descriptions", columns=["table", "indicator"])) == [{"table": None, "indicator": None}]
    with pytest.raises(SharadarError, match="as-reported"):
        list(iter_source_rows(bundle, "fundamentals", dimensions=["MRQ"]))


def test_incomplete_or_tampered_handoff_refuses_read(lake, tmp_path):
    interface = tmp_path / "interface.md"
    interface.write_text("contract")
    output = tmp_path / "bundle"
    create_handoff(lake, output, cash_payload=CASH, interface=interface)
    marker = output / "INCOMPLETE"
    marker.write_text("interrupted")
    with pytest.raises(SharadarError, match="incomplete"):
        load_handoff(output)
    marker.unlink()
    (output / "cash-DGS3MO.csv").write_bytes(b"tampered")
    with pytest.raises(SharadarError, match="file hash mismatch"):
        load_handoff(output)
