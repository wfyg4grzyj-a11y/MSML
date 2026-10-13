"""Deterministic infrastructure tests; no live API credentials or LLM calls."""

import hashlib
import json
from datetime import date

import httpx
import pytest

from alpha_lab.market_data import MarketStore, TiingoClient, TiingoError, sync_ticker
from alpha_lab.market_data.cli import export_snapshot, load_config, main
from alpha_lab.market_data.store import normalize_bars
from alpha_lab.market_data.tiingo import ticker_symbol

D1, D2, D3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)


def bar(day=D1, *, adj=1, dividend=0, split=1):
    return {"date": day.isoformat() + "T00:00:00.000Z", "open": 9, "high": 11,
            "low": 8, "close": 10, "volume": 1000,
            "adjOpen": 9 * adj, "adjHigh": 11 * adj, "adjLow": 8 * adj,
            "adjClose": 10 * adj, "adjVolume": 1000 / adj,
            "divCash": dividend, "splitFactor": split}


class FakeClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def metadata(self, ticker):
        return {"ticker": ticker, "name": "Test asset", "startDate": "2024-01-02"}

    def prices(self, ticker, start, end):
        self.requests.append((ticker, start, end))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def store(tmp_path):
    with MarketStore(tmp_path / "market data" / "prices.sqlite3") as result:
        yield result


def test_client_headers_parameters_and_pacing():
    requests, sleeps = [], []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"ticker": "SPY"} if not request.url.path.endswith("prices") else [bar()])
    with TiingoClient("private-test-token", requests_per_minute=30,
                      transport=httpx.MockTransport(respond), sleep=sleeps.append, clock=lambda: 0) as client:
        client.metadata("spy")
        assert client.prices("SPY", D1, D2) == [bar()]
    assert sleeps == [2]
    assert requests[0].headers["Authorization"] == "Token private-test-token"
    assert "token" not in requests[1].url.params
    assert requests[1].url.params["startDate"] == "2024-01-02"
    assert requests[1].url.params["resampleFreq"] == "daily"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retry_and_retry_after(status):
    responses = iter([httpx.Response(status, headers={"Retry-After": "2"}), httpx.Response(200, json=[bar()])])
    sleeps = []
    with TiingoClient("key", transport=httpx.MockTransport(lambda _: next(responses)),
                      sleep=sleeps.append, clock=lambda: 0) as client:
        assert client.prices("SPY", D1, D2) == [bar()]
    assert sleeps == [2, 2]


@pytest.mark.parametrize("status", [401, 403, 404, 429, 302])
def test_errors_do_not_leak_secrets(status):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(status, text="private-test-token", headers={"Location": "https://example.com"})
    with TiingoClient("private-test-token", retries=0, transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(TiingoError) as error:
            client.prices("SPY", D1, D2)
    assert "private-test-token" not in str(error.value)
    assert len(requests) == 1


def test_network_retries_are_bounded_and_safe():
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ConnectError("private-test-token", request=request)
    with TiingoClient("private-test-token", retries=2, transport=httpx.MockTransport(respond), sleep=lambda _: None) as client:
        with pytest.raises(TiingoError, match="connection failed") as error:
            client.metadata("SPY")
    assert len(calls) == 3
    assert "private-test-token" not in str(error.value)


def test_backfill_incremental_resume_and_audit(store):
    sync_ticker(FakeClient([bar(), bar(D2)]), store, "spy", D1, D2)
    client = FakeClient([bar(), bar(D2), bar(D3)])
    result = sync_ticker(client, store, "SPY", D1, D3, overlap_days=1)
    assert client.requests == [("SPY", D1, D3)]
    assert not result["full_refresh"]
    assert len(store.prices(["spy"])) == 3
    sync_ticker(FakeClient([bar(), bar(D2), bar(D3)]), store, "SPY", D1, D3)
    assert len(store.prices()) == 3
    assert store.db.execute("SELECT COUNT(*) FROM syncs WHERE status='ok'").fetchone()[0] == 3
    assert store.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 6
    assert store.summary()[0]["end_date"] == D3.isoformat()


@pytest.mark.parametrize("action", [{"dividend": 1}, {"split": 2}])
def test_new_action_refreshes_earlier_adjusted_history(store, action):
    sync_ticker(FakeClient([bar(), bar(D2)]), store, "SPY", D1, D2)
    client = FakeClient([bar(D2), bar(D3, **action)], [bar(adj=.5), bar(D2, adj=.5), bar(D3, **action)])
    result = sync_ticker(client, store, "SPY", D1, D3, overlap_days=1)
    assert result["full_refresh"]
    assert len(client.requests) == 2
    assert store.prices()[0]["adj_close"] == 5
    assert store.prices()[0]["close"] == 10


def test_existing_dividend_does_not_refresh_unchanged_history(store):
    sync_ticker(FakeClient([bar(), bar(D2, dividend=1)]), store, "SPY", D1, D2)
    client = FakeClient([bar(), bar(D2, dividend=1), bar(D3)])
    assert not sync_ticker(client, store, "SPY", D1, D3)["full_refresh"]
    assert len(client.requests) == 1


def test_holiday_or_pre_ipo_start_does_not_force_repeated_backfill(store):
    start = date(2000, 1, 1)
    sync_ticker(FakeClient([bar(), bar(D2)]), store, "SPY", start, D2)
    client = FakeClient([bar(), bar(D2), bar(D3)])
    assert not sync_ticker(client, store, "SPY", start, D3)["full_refresh"]
    assert len(client.requests) == 1
    assert client.requests[0][1] > start


def test_bad_config_values_are_not_echoed(tmp_path, capsys):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"tickers": ["SPY"], "start_date": "secret-should-not-appear"}))
    assert main(["--config", str(config), "status"]) == 1
    assert "secret-should-not-appear" not in capsys.readouterr().err


def test_corrected_overlap_refreshes_history(store):
    sync_ticker(FakeClient([bar(), bar(D2)]), store, "SPY", D1, D2)
    client = FakeClient([bar(D1, adj=.8), bar(D2)], [bar(D1, adj=.8), bar(D2), bar(D3)])
    assert sync_ticker(client, store, "SPY", D1, D3)["full_refresh"]
    assert store.prices()[0]["adj_close"] == 8


@pytest.mark.parametrize("response", [[], [bar(D2)], [bar(D1), bar(D1)], TiingoError("Tiingo HTTP 503")])
def test_failed_refresh_preserves_history_and_records_failure(store, response):
    sync_ticker(FakeClient([bar(), bar(D2)]), store, "SPY", D1, D2)
    original = store.prices()
    with pytest.raises((ValueError, TiingoError)):
        sync_ticker(FakeClient(response), store, "SPY", D1, D3, full_refresh=True)
    assert store.prices() == original
    assert store.db.execute("SELECT status FROM syncs ORDER BY id DESC LIMIT 1").fetchone()[0] == "failed"


def test_backfill_can_extend_earlier_without_losing_newer_history(store):
    sync_ticker(FakeClient([bar(D2), bar(D3)]), store, "SPY", D2, D3)
    client = FakeClient([bar(), bar(D2), bar(D3)])
    sync_ticker(client, store, "SPY", D1, D2)
    assert client.requests[0] == ("SPY", D1, D3)
    assert len(store.prices()) == 3


@pytest.mark.parametrize("field,value", [("close", 99), ("volume", -1), ("volume", 1.2),
    ("adjLow", 0), ("divCash", -1), ("splitFactor", 0), ("adjClose", float("nan")), ("open", True)])
def test_reject_invalid_bars(field, value):
    row = bar()
    row[field] = value
    with pytest.raises(ValueError, match="Invalid daily bar"):
        normalize_bars("SPY", [row], D1, D2)


def test_export_offline_snapshot_and_hash(store, tmp_path):
    sync_ticker(FakeClient([bar(), bar(D2)]), store, "SPY", D1, D2)
    path = tmp_path / "research snapshot.csv"
    manifest = export_snapshot(store, path, ["SPY"])
    assert manifest["rows"] == 2
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not manifest["point_in_time"]
    assert "adj_close" in path.read_text()
    assert json.loads(path.with_suffix(".csv.manifest.json").read_text())["provider"] == "tiingo"
    with pytest.raises(ValueError, match="already exists"):
        export_snapshot(store, path, ["SPY"])


def test_database_transaction_rolls_back_on_invalid_metadata(store):
    sync_ticker(FakeClient([bar()]), store, "SPY", D1, D1)
    original = store.prices()
    client = FakeClient([bar(D1, adj=.5), bar(D2)])
    client.metadata = lambda _: {"ticker": "SPY", "bad": float("nan")}
    with pytest.raises(ValueError):
        sync_ticker(client, store, "SPY", D1, D2, full_refresh=True)
    assert store.prices() == original


def test_config_relative_paths_and_offline_cli(tmp_path, capsys, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tickers": ["spy", "SPY"], "start_date": D1.isoformat(), "database": "cache/prices.db"}))
    config = load_config(path)
    assert config["tickers"] == ["SPY"]
    assert config["database"] == tmp_path / "cache" / "prices.db"
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert main(["--config", str(path), "status"]) == 0
    assert '"assets": []' in capsys.readouterr().out
    assert main(["--config", str(path), "sync", "--env-file", str(tmp_path / "no.env")]) == 1
    assert "TIINGO_API_KEY" in capsys.readouterr().err


def test_ticker_path_injection_rejected():
    with pytest.raises(ValueError):
        ticker_symbol("../SPY?token=secret")
