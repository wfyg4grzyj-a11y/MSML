import csv
import io
import json
import zipfile
from pathlib import Path

import duckdb
import httpx
import pytest

from alpha_lab.market_data.sharadar import (
    SharadarClient, SharadarError, file_hash, parse_schema, table_name,
)
from alpha_lab.market_data.sharadar_cli import load_config, main
from alpha_lab.market_data.sharadar_store import SharadarStore


DDL = '''CREATE TABLE IF NOT EXISTS stocks (
  ticker text NOT NULL,
  date date NOT NULL,
  close double precision,
  volume double precision,
  PRIMARY KEY (ticker, date)
);
'''
CSV = 'ticker,date,close,volume\nSPY,2026-10-09,100.5,123\nSPY,2026-10-12,101,456\nAAPL,2026-10-12,200,789\n'


def zipped(text=CSV, member='stocks.csv'):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return output.getvalue()


def publish(tmp_path, text=CSV):
    store = SharadarStore(tmp_path / 'lake', memory_limit='256MB', threads=1)
    archive = tmp_path / 'stocks.zip'
    archive.write_bytes(zipped(text))
    source = {'table': 'stocks', 'modified': '2026-10-13', 'size': archive.stat().st_size, 'history': 'full'}
    store.publish('stocks', archive, parse_schema(DDL, 'stocks'), source,
                  {'sha256': file_hash(archive), 'bytes': archive.stat().st_size})
    return store, archive, source


def client(handler, **kwargs):
    return SharadarClient('SECRET-DO-NOT-LOG', transport=httpx.MockTransport(handler),
                         sleep=lambda _: None, requests_per_minute=600, **kwargs)


def test_alias_and_reject_unknown():
    assert table_name('SEP') == 'stocks'
    assert table_name('SF1') == 'fundamentals'
    with pytest.raises(SharadarError):
        table_name('../../stocks')


def test_schema_quoted_names_and_unique_index():
    schema = parse_schema('''CREATE TABLE IF NOT EXISTS holdings_investor (
  date text,
  investorid text,
  totalvalue double precision
);
CREATE UNIQUE INDEX IF NOT EXISTS hi_pk ON public.holdings_investor USING btree (date, investorid);
''', 'holdings_investor')
    assert schema['primary_key'] == ['date', 'investorid']
    quoted = parse_schema('''CREATE TABLE IF NOT EXISTS descriptions (
  "table" text NOT NULL,
  indicator text NOT NULL,
  PRIMARY KEY ("table", indicator)
);
''', 'descriptions')
    assert quoted['columns'][0]['name'] == 'table'


@pytest.mark.parametrize('ddl', [DDL.replace('double precision', 'text; DROP TABLE x;'),
                               DDL.replace('PRIMARY KEY (ticker, date)', 'PRIMARY KEY (missing)'),
                               DDL.replace('ticker text', 'ticker$ text')])
def test_schema_rejects_unrecognized_ddl(ddl):
    with pytest.raises(SharadarError):
        parse_schema(ddl, 'stocks')


def test_status_chooses_full_history_and_does_not_persist_remote_urls():
    def handler(request):
        assert request.url.params['api_key'] == 'SECRET-DO-NOT-LOG'
        return httpx.Response(200, json={'files': [
            {'history': '5y', 'size': 12, 'modified': 'old'},
            {'history': 'full', 'size': 123, 'modified': 'new', 'url': 'secret-signed-url'}]})
    with client(handler) as c:
        assert c.status('stocks') == {'table': 'stocks', 'size': 123, 'modified': 'new', 'history': 'full'}


def test_http_error_never_exposes_key_or_response_body():
    with client(lambda _: httpx.Response(401, text='SECRET-DO-NOT-LOG')) as c:
        with pytest.raises(SharadarError) as exc:
            c.status('stocks')
    assert 'SECRET' not in str(exc.value)
    assert '401' in str(exc.value)


def test_connection_errors_are_sanitized():
    def handler(request):
        raise httpx.ConnectError(str(request.url))
    with client(handler, retries=0) as c:
        with pytest.raises(SharadarError) as exc:
            c.status('stocks')
    assert 'SECRET' not in str(exc.value)


def test_rate_limit_retries():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(429, headers={'Retry-After': '0'}) if len(requests) == 1 else httpx.Response(200, json={'size': 123, 'modified': 'new'})
    with client(handler) as c:
        assert c.status('stocks')['size'] == 123
    assert len(requests) == 2


def test_schema_requests_are_public():
    def handler(request):
        assert 'api_key' not in request.url.params
        return httpx.Response(200, text=DDL)
    with client(handler) as c:
        assert c.schema('stocks')['primary_key'] == ['ticker', 'date']


def test_bulk_auth_not_forwarded_to_storage(tmp_path):
    blob = zipped()
    def handler(request):
        if request.url.host == 'api.sharadar.com':
            assert request.url.params['api_key'] == 'SECRET-DO-NOT-LOG'
            return httpx.Response(302, headers={'Location': 'https://storage.example/stocks.zip?signature=temporary'})
        assert 'api_key' not in request.url.params
        assert 'authorization' not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(blob), headers={'ETag': 'abc'})
    path = tmp_path / 'stocks.zip'
    with client(handler) as c:
        result = c.download('stocks', {'size': len(blob), 'modified': 'new'}, path)
    assert path.read_bytes() == blob
    assert result['sha256'] == file_hash(path)
    assert not list(tmp_path.glob('*.part*'))


def test_resume_partial_download(tmp_path):
    blob = zipped()
    path = tmp_path / 'stocks.zip'
    partial = path.with_suffix('.zip.part')
    partial.write_bytes(blob[:30])
    partial.with_suffix('.part.json').write_text(json.dumps({'source': {'table': 'stocks', 'size': len(blob), 'modified': 'new'}, 'etag': 'same'}))
    def handler(request):
        if request.url.host == 'api.sharadar.com':
            return httpx.Response(302, headers={'Location': 'https://storage.example/object'})
        assert request.headers['Range'] == 'bytes=30-'
        assert request.headers['If-Range'] == 'same'
        return httpx.Response(206, stream=httpx.ByteStream(blob[30:]), headers={'ETag': 'same', 'Content-Range': f'bytes 30-{len(blob)-1}/{len(blob)}'})
    with client(handler) as c:
        c.download('stocks', {'size': len(blob), 'modified': 'new'}, path)
    assert path.read_bytes() == blob


def test_changed_source_discards_partial(tmp_path):
    blob = zipped()
    path = tmp_path / 'stocks.zip'
    partial = path.with_suffix('.zip.part')
    partial.write_bytes(b'old')
    partial.with_suffix('.part.json').write_text(json.dumps({'source': {'table': 'stocks', 'size': len(blob), 'modified': 'old'}}))
    def handler(request):
        if request.url.host == 'api.sharadar.com':
            return httpx.Response(302, headers={'Location': 'https://storage.example/object'})
        assert 'range' not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(blob))
    with client(handler) as c:
        c.download('stocks', {'size': len(blob), 'modified': 'new'}, path)
    assert path.read_bytes() == blob


def test_interrupted_stream_resumes_with_a_fresh_link(tmp_path):
    blob = b'a' * (1024 * 1024) + b'end'
    tickets, storage = [], []
    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield blob[:-3]
            raise httpx.ReadError('private-signed-url')
    def handler(request):
        if request.url.host == 'api.sharadar.com':
            tickets.append(request)
            return httpx.Response(302, headers={'Location': f'https://storage.example/object?signature={len(tickets)}'})
        storage.append(request)
        if len(storage) == 1:
            return httpx.Response(200, stream=Interrupted(), headers={'ETag': 'same'})
        assert request.headers['Range'] == 'bytes=1048576-'
        return httpx.Response(206, stream=httpx.ByteStream(blob[-3:]),
                              headers={'ETag': 'same', 'Content-Range': f'bytes 1048576-{len(blob)-1}/{len(blob)}'})
    path = tmp_path / 'stocks.zip'
    with client(handler) as c:
        c.download('stocks', {'size': len(blob), 'modified': 'new'}, path)
    assert path.read_bytes() == blob
    assert len(tickets) == 2
    assert storage[0].url != storage[1].url


def test_invalid_range_is_not_published(tmp_path):
    def handler(request):
        if request.url.host == 'api.sharadar.com':
            return httpx.Response(302, headers={'Location': 'https://storage.example/object'})
        return httpx.Response(206, stream=httpx.ByteStream(b'bad'), headers={'Content-Range': 'bytes 9-11/123'})
    path = tmp_path / 'stocks.zip'
    with client(handler, retries=0) as c:
        with pytest.raises(SharadarError) as error:
            c.download('stocks', {'size': 123, 'modified': 'new'}, path)
    assert not path.exists()
    assert 'signature' not in str(error.value)


@pytest.mark.parametrize('location', ['', 'http://storage.example/object', 'https://user:password@storage.example/object'])
def test_invalid_download_redirect(location, tmp_path):
    with client(lambda _: httpx.Response(302, headers={'Location': location}), retries=0) as c:
        with pytest.raises(SharadarError):
            c.download('stocks', {'size': 123, 'modified': 'new'}, tmp_path / 's.zip')


def test_typed_snapshot_and_offline_filtered_export(tmp_path):
    store, _, _ = publish(tmp_path)
    snapshot = store.latest('stocks')
    assert snapshot['rows'] == 3
    assert snapshot['start_date'] == '2026-10-09'
    assert snapshot['end_date'] == '2026-10-12'
    with duckdb.connect() as db:
        types = dict((r[0], r[1]) for r in db.execute('DESCRIBE SELECT * FROM read_parquet(?)', [snapshot['path']]).fetchall())
        assert types['date'] == 'DATE'
        assert types['close'] == 'DOUBLE'
    output = tmp_path / 'spy.csv'
    manifest = store.export('stocks', output, tickers=['SPY'], start='2026-10-12', columns=['ticker', 'date', 'close'])
    with output.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['close'] == '101.0'
    assert manifest['source_version'] == snapshot['version']
    assert manifest['sha256'] == file_hash(output)
    with pytest.raises(SharadarError):
        store.export('stocks', output)


@pytest.mark.parametrize('invalid', [CSV + 'SPY,2026-10-12,102,123\n',
                                    CSV.replace('100.5', 'not-a-number'),
                                    CSV.replace('100.5', 'NaN'),
                                    CSV.replace('SPY,2026-10-09', ',2026-10-09'),
                                    CSV.replace('2026-10-09', 'not-a-date'),
                                    CSV.replace('ticker,date', 'symbol,date')])
def test_bad_snapshot_retains_previous_version(tmp_path, invalid):
    store, archive, source = publish(tmp_path)
    version = store.latest('stocks')['version']
    archive.write_bytes(zipped(invalid))
    with pytest.raises(SharadarError):
        store.publish('stocks', archive, parse_schema(DDL, 'stocks'), source,
                      {'sha256': file_hash(archive), 'bytes': archive.stat().st_size})
    assert store.latest('stocks')['version'] == version
    assert not list((store.root / 'staging').iterdir())


def test_zip_member_paths_never_extracted(tmp_path):
    store, archive, source = publish(tmp_path)
    archive.write_bytes(zipped(member='../../escape.csv'))
    store.publish('stocks', archive, parse_schema(DDL, 'stocks'), source,
                  {'sha256': file_hash(archive), 'bytes': archive.stat().st_size})
    assert not (tmp_path / 'escape.csv').exists()


def test_truncated_zip_does_not_publish(tmp_path):
    store, archive, source = publish(tmp_path)
    version = store.latest('stocks')['version']
    archive.write_bytes(b'broken zip')
    with pytest.raises(SharadarError):
        store.publish('stocks', archive, parse_schema(DDL, 'stocks'), source, {'sha256': 'bad', 'bytes': 10})
    assert store.latest('stocks')['version'] == version


def test_unchanged_source_skips_schema_and_download(tmp_path):
    store, _, source = publish(tmp_path)
    class Unchanged:
        def status(self, table): return source
        def schema(self, table): pytest.fail('Unchanged source must not refetch')
    assert store.sync(Unchanged(), 'stocks')['status'] == 'unchanged'


def test_concurrent_writer_rejected_and_lock_released(tmp_path):
    store, _, source = publish(tmp_path)
    class Unchanged:
        def status(self, table): return source
    with store._writer_lock():
        with pytest.raises(SharadarError, match='Another sync'):
            SharadarStore(store.root).sync(Unchanged(), 'stocks')
    assert store.sync(Unchanged(), 'stocks')['status'] == 'unchanged'


def test_export_filters_cannot_inject_sql(tmp_path):
    store, _, _ = publish(tmp_path)
    output = tmp_path / 'empty.csv'
    store.export('stocks', output, tickers=["SPY') OR 1=1 --"])
    assert len(output.read_text().splitlines()) == 1
    with pytest.raises(SharadarError):
        store.export('stocks', tmp_path / 'other.csv', columns=['close; DROP TABLE x'])


def test_fundamental_dimensions_remain_distinct(tmp_path):
    ddl = DDL.replace('stocks', 'fundamentals').replace('  close double precision,', '  dimension text NOT NULL,\n  close double precision,').replace('PRIMARY KEY (ticker, date)', 'PRIMARY KEY (ticker, date, dimension)')
    blob = zipped('ticker,date,dimension,close,volume\nAAPL,2026-10-12,ARQ,1,1\nAAPL,2026-10-12,MRQ,2,2\n')
    archive = tmp_path / 'f.zip'; archive.write_bytes(blob)
    store = SharadarStore(tmp_path / 'lake')
    store.publish('fundamentals', archive, parse_schema(ddl, 'fundamentals'), {'modified': 'now'}, {'sha256': file_hash(archive), 'bytes': len(blob)})
    output = tmp_path / 'arq.csv'
    store.export('fundamentals', output, dimension='ARQ')
    assert len(output.read_text().splitlines()) == 2
    assert 'MRQ' not in output.read_text()


def test_config_keeps_keys_out_and_resolves_local_root(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'tables': ['SEP'], 'root': 'lake'}))
    store, tables, _ = load_config(path)
    assert store.root == tmp_path / 'lake'
    assert tables == ['stocks']
    path.write_text(json.dumps({'tables': ['stocks'], 'api_key': 'SECRET'}))
    with pytest.raises(SharadarError):
        load_config(path)


def test_status_is_offline(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'tables': ['stocks'], 'root': 'lake'}))
    monkeypatch.setattr(SharadarClient, '__init__', lambda *a, **kw: pytest.fail('No network for status'))
    assert main(['--config', str(path), 'status']) == 0
    assert json.loads(capsys.readouterr().out)[0]['snapshot'] is None
