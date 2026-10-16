"""Knowledge-time archive selection never substitutes today's corrected file."""
import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip('duckdb')
pd = pytest.importorskip('pandas')

from alpha_lab.market_data.sharadar import SharadarError, file_hash
from alpha_lab.market_data.sharadar_store import SharadarStore


def snapshot(store, version, observed, close):
    folder = store.root / 'tables/stocks' / version
    folder.mkdir(parents=True)
    duckdb.from_df(pd.DataFrame({'ticker': ['A'], 'date': [pd.Timestamp('2014-01-02')], 'close': [close]})).write_parquet(str(folder / 'data.parquet'))
    manifest = {'table': 'stocks', 'retrieved_at': observed, 'parquet_sha256': file_hash(folder / 'data.parquet'),
                'source': {'modified': observed}, 'schema': {'primary_key': ['ticker', 'date'],
                'columns': [{'name': 'ticker'}, {'name': 'date'}, {'name': 'close'}]}}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (folder.parent / 'latest.json').write_text(json.dumps({'version': version}))
    return folder


def test_cutoff_selects_original_even_when_latest_has_revised_history(tmp_path):
    store = SharadarStore(tmp_path)
    snapshot(store, 'a' * 32, '2026-10-13T10:00:00Z', 10.)
    snapshot(store, 'b' * 32, '2026-10-14T10:00:00Z', 11.)
    assert store.latest('stocks')['version'] == 'b' * 32
    assert store.asof('stocks', '2026-10-13T18:00:00+08:00')['version'] == 'a' * 32
    assert store.asof('stocks', '2026-10-12T23:59:59Z') is None
    assert store.asof('stocks', '2014-01-02T23:59:59Z') is None
    result = store.export('stocks', tmp_path / 'asobserved.csv', observed_before='2026-10-13T10:00:00Z')
    assert result['source_version'] == 'a' * 32 and result['point_in_time'] is False
    assert result['source_retrieved_at'] == '2026-10-13T10:00:00Z'
    assert pd.read_csv(tmp_path / 'asobserved.csv').close.tolist() == [10.]


def test_missing_observed_history_never_creates_an_export(tmp_path):
    store = SharadarStore(tmp_path)
    snapshot(store, 'a' * 32, '2026-10-13T10:00:00Z', 10.)
    with pytest.raises(SharadarError, match='No archived snapshot'):
        store.export('stocks', tmp_path / 'old.csv', observed_before='2014-01-02T23:59:59Z')
    assert not (tmp_path / 'old.csv').exists()


@pytest.mark.parametrize('invalid', ['2026-10-13', '2026-10-13T10:00:00', 'invalid', 1])
def test_naive_and_invalid_cutoffs_fail_closed(tmp_path, invalid):
    with pytest.raises(SharadarError, match='timezone'):
        SharadarStore(tmp_path).asof('stocks', invalid)


def test_same_time_multiple_snapshots_require_reconciliation(tmp_path):
    store = SharadarStore(tmp_path)
    for char in 'ab':
        snapshot(store, char * 32, '2026-10-13T10:00:00Z', 10.)
    with pytest.raises(SharadarError, match='Ambiguous'):
        store.asof('stocks', '2026-10-13T10:00:00Z')


def test_modified_archived_data_rejected(tmp_path):
    store = SharadarStore(tmp_path)
    folder = snapshot(store, 'a' * 32, '2026-10-13T10:00:00Z', 10.)
    (folder / 'data.parquet').write_bytes(b'modified')
    with pytest.raises(SharadarError, match='hash mismatch'):
        store.asof('stocks', '2026-10-13T10:00:00Z')


@pytest.mark.parametrize('field,value', [('retrieved_at', '2014-01-01'), ('table', 'daily'), ('parquet_sha256', None)])
def test_bad_archive_evidence_rejected_without_latest_fallback(tmp_path, field, value):
    store = SharadarStore(tmp_path)
    folder = snapshot(store, 'a' * 32, '2026-10-13T10:00:00Z', 10.)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest[field] = value
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SharadarError, match='Invalid archived'):
        store.asof('stocks', '2026-10-13T10:00:00Z')
