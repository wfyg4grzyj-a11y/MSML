"""Bounded Arrays ingestion for explicitly uncovered US-equity datasets."""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx

from .routing import ARRAYS_EXTRA_DATASETS, select_route, US_EXCHANGES
from .sharadar import file_hash
from .sharadar_store import atomic_json, SharadarStore

BASE_URL = 'https://data-tools.prd.arrays.org/api'
PATHS = {'consensus_estimates': '/v1/stocks/estimates-guidance',
         'company_guidance': '/v1/stocks/estimates-guidance',
         'earnings_transcripts': '/v1/stocks/earnings-transcript',
         'finra_short_interest': '/v1/stocks/short-interest',
         'dark_pool_activity': '/v1/stocks/darkpool',
         'equity_ratings': '/v1/stocks/ratings', 'etf_holdings': '/v1/etf/holdings'}


class ArraysError(RuntimeError):
    """Safe message; upstream bodies/URLs/headers never enter errors or logs."""


def query_parameters(dataset, ticker, *, start=None, end=None, metrics='EPS', period_type='quarterly',
                     fiscal_year=None, fiscal_quarter=None):
    if dataset not in PATHS:
        raise ArraysError('Dataset is covered by earlier providers or outside the Arrays allowlist')
    if period_type not in {'annual', 'quarterly', 'semi-annual'}:
        raise ArraysError('Invalid forecast period type')
    if fiscal_year is not None and (isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int) or not 2005 <= fiscal_year <= 2100):
        raise ArraysError('Invalid fiscal year')
    if fiscal_quarter is not None and fiscal_quarter not in {'Q1', 'Q2', 'Q3', 'Q4'}:
        raise ArraysError('Invalid fiscal quarter')
    params = {'symbol': ticker}
    if dataset == 'etf_holdings':
        if start or end or fiscal_year or fiscal_quarter:
            raise ArraysError('ETF holdings is a current snapshot; historical dates are unsupported')
        return params
    if dataset == 'earnings_transcripts':
        if start or end or fiscal_year is None or period_type == 'semi-annual' or period_type == 'quarterly' and fiscal_quarter is None:
            raise ArraysError('Transcripts require issuer fiscal year/period and quarterly quarter; date ranges are unsupported')
        return params | {'fiscal_year': fiscal_year, 'period_type': period_type} | ({'fiscal_quarter': fiscal_quarter} if fiscal_quarter else {})
    last = date.fromisoformat(str(end)) if end else datetime.now(timezone.utc).date() - timedelta(days=1)
    first = date.fromisoformat(str(start)) if start else last - timedelta(days=29)
    if first > last:
        raise ArraysError('Start date must precede end date')
    if dataset == 'dark_pool_activity' and (last - first).days > 31:
        raise ArraysError('Dark-pool requests are bounded to 32 days; use separate immutable windows')
    params.update(start_time=int(datetime.combine(first, datetime.min.time(), timezone.utc).timestamp()),
                  end_time=int(datetime.combine(last + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp()) - 1)
    if dataset in {'consensus_estimates', 'company_guidance'}:
        selected = metrics.split(',')
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= {'EPS','SALES','DPS','CFPS','EBITDA','EBIT','BPS','ASSETS'}:
            raise ArraysError('Unsupported forecast metrics')
        params.update(metrics=metrics, type='estimate' if dataset == 'consensus_estimates' else 'guidance', period_type=period_type)
        if fiscal_year is not None:
            params['fiscal_year'] = fiscal_year
        if fiscal_quarter is not None:
            params['fiscal_quarter'] = fiscal_quarter
    elif fiscal_year is not None or fiscal_quarter is not None:
        raise ArraysError('Fiscal filters only apply to estimates/guidance/transcripts')
    if dataset == 'finra_short_interest':
        params['time_type'] = 'UPDATED_AT'
    return params


class ArraysClient:
    def __init__(self, key, *, page_size=100, max_pages=10, max_rows=5000, max_bytes=8_000_000,
                 timeout_seconds=15, run_seconds=90, requests_per_minute=30, transport=None):
        if not isinstance(key, str) or not key.strip():
            raise ArraysError('ARRAYS_API_KEY is not configured')
        limits = [page_size,max_pages,max_rows,max_bytes,timeout_seconds,run_seconds,requests_per_minute]
        if any(isinstance(v,bool) or not isinstance(v,int) or v <= 0 for v in limits) or page_size > 1000 or requests_per_minute > 600:
            raise ArraysError('Invalid Arrays request limits')
        self.page_size,self.max_pages,self.max_rows,self.max_bytes = page_size,max_pages,max_rows,max_bytes
        self.timeout,self.run_seconds,self.rate = timeout_seconds,run_seconds,requests_per_minute
        self._key,self._last = key,0.
        self._client = httpx.Client(headers={'X-API-Key': key}, follow_redirects=False, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def fetch(self, route, params, *, equity_symbols=None):
        if route.provider != 'arrays' or route.dataset not in ARRAYS_EXTRA_DATASETS or not route.scope or params.get('symbol') != route.ticker:
            raise ArraysError('A verified uncovered US-equity route is required')
        if route.scope.get('symbol') != route.ticker or route.scope.get('asset_scope') not in {'us_listed_equity','us_equity_etf'}:
            raise ArraysError('Invalid Arrays asset scope')
        if route.dataset == 'etf_holdings' and equity_symbols is None:
            raise ArraysError('US equity constituent scope is required for ETF holdings')
        started = time.monotonic()
        records, pages, dropped, offset, total_bytes = [],0,0,0,0
        seen_pages = set()
        paged = route.dataset in {'consensus_estimates','company_guidance'}
        while True:
            remaining = self.run_seconds - (time.monotonic() - started)
            wait = max(0.,60 / self.rate - (time.monotonic() - self._last))
            if remaining <= wait or pages >= self.max_pages:
                raise ArraysError('Arrays run/page cap reached; prior snapshot retained')
            if wait:
                time.sleep(wait)
            request = params | ({'limit':self.page_size,'offset':offset} if paged else
                                {'limit':min(self.max_rows,1000)} if route.dataset == 'equity_ratings' else {})
            try:
                self._last = time.monotonic()
                timeout = min(self.timeout,self.run_seconds - (self._last - started))
                with self._client.stream('GET',BASE_URL+PATHS[route.dataset],params=request,
                                         timeout=httpx.Timeout(timeout,connect=min(5,timeout))) as response:
                    if response.status_code != 200:
                        raise ArraysError(f'Arrays HTTP {response.status_code}; no retries, payments or provider fallback')
                    chunks,size = [],0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if total_bytes + size > self.max_bytes or time.monotonic()-started >= self.run_seconds:
                            raise ArraysError('Arrays response/time cap reached')
                        chunks.append(chunk)
                body = b''.join(chunks)
                if self._key.encode() in body:
                    raise ArraysError('Credential echo rejected')
                payload = json.loads(body,parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (httpx.HTTPError, ValueError, UnicodeError):
                raise ArraysError('Arrays request or JSON validation failed') from None
            data = payload.get('data') if isinstance(payload,dict) else None
            if not isinstance(payload,dict) or payload.get('success') is not True or not isinstance(data,list) or any(not isinstance(r,dict) for r in data):
                raise ArraysError('Invalid Arrays success/data envelope')
            pages += 1
            total_bytes += size
            fingerprint = hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
            if paged and data and fingerprint in seen_pages:
                raise ArraysError('Arrays pagination did not advance; prior snapshot retained')
            seen_pages.add(fingerprint)
            for row in data:
                symbol = row.get('symbol')
                normalized = symbol.upper().replace('-','.') if isinstance(symbol,str) else None
                if route.dataset == 'etf_holdings':
                    if normalized != route.ticker:
                        raise ArraysError('Holdings response names an unexpected parent fund')
                    asset = row.get('asset')
                    normalized_asset = asset.upper().replace('-','.') if isinstance(asset,str) else None
                    if normalized_asset not in equity_symbols:
                        dropped += 1
                        continue
                elif normalized != route.ticker:
                    raise ArraysError('Arrays response contains an unexpected security; snapshot rejected')
                records.append(row)
            if len(records) > self.max_rows:
                raise ArraysError('Arrays row cap reached; prior snapshot retained')
            if time.monotonic() - started >= self.run_seconds:
                raise ArraysError('Arrays total time cap reached')
            pagination = payload.get('pagination') or (payload.get('metadata',{}).get('pagination') if isinstance(payload.get('metadata'),dict) else None)
            has_more = pagination.get('has_more') if isinstance(pagination,dict) else None
            if has_more is not None and type(has_more) is not bool:
                raise ArraysError('Invalid pagination completeness flag')
            if not paged:
                if has_more is True:
                    raise ArraysError('Endpoint returned a truncated page; narrow the window')
                if route.dataset == 'equity_ratings' and len(data) >= min(self.max_rows,1000):
                    raise ArraysError('Ratings limit may truncate data; narrow the date window')
                break
            if has_more is False or has_more is None and len(data) < self.page_size:
                break
            if not data:
                raise ArraysError('Empty advancing page is inconsistent')
            offset += len(data)
        if route.dataset == 'etf_holdings' and dropped and not records:
            raise ArraysError('No verified US equity constituents remain; snapshot rejected')
        return {'data':records,'pages':pages,'excluded_out_of_scope_rows':dropped,
                'request_params':params,'complete_requested_window':True,
                'scope_filtered_portfolio':route.dataset == 'etf_holdings', 'point_in_time_attested':False}


def equity_constituent_symbols(store):
    snapshot = store.latest('tickers')
    if snapshot is None or file_hash(Path(snapshot['path'])) != snapshot['parquet_sha256']:
        raise ArraysError('Verified equity reference required')
    with store._connect() as db:
        rows = db.execute('SELECT ticker,exchange,category FROM read_parquet(?) WHERE "table"=\'SEP\' AND currency=\'USD\'', [snapshot['path']]).fetchall()
    return {ticker for ticker,exchange,category in rows if str(exchange).upper() in US_EXCHANGES
            and ('Common Stock' in str(category) or 'ADR' in str(category))}


class ArraysStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def latest(self, dataset, ticker):
        if dataset not in PATHS:
            raise ArraysError('Unsupported Arrays dataset')
        # Tickers come from verified routes, but validate independent offline reads.
        from .tiingo import ticker_symbol
        ticker = ticker_symbol(ticker).replace('-','.')
        base = self.root / dataset / ticker
        if not (base/'latest.json').exists():
            return None
        try:
            version = json.loads((base/'latest.json').read_text())['version']
            if len(version) != 32 or any(c not in '0123456789abcdef' for c in version):
                raise ValueError
            folder = base/version
            manifest = json.loads((folder/'manifest.json').read_text())
            data = folder/'data.json'
            if folder.resolve().parent != base.resolve() or manifest['dataset'] != dataset or manifest['ticker'] != ticker or file_hash(data) != manifest['sha256']:
                raise ValueError
            return manifest | {'path':str(data),'version':version}
        except (OSError,ValueError,KeyError,TypeError):
            raise ArraysError('Invalid cached Arrays snapshot') from None

    def sync(self, client, route, params, *, equity_symbols=None):
        # Same local OS writer lock discipline as the existing data lake.
        with SharadarStore(self.root)._writer_lock():
            result = client.fetch(route,params,equity_symbols=equity_symbols)
            version = uuid.uuid4().hex
            folder = self.root / route.dataset / route.ticker / version
            folder.mkdir(parents=True)
            data = folder/'data.json'
            data.write_text(json.dumps(result['data'],ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
            manifest = {'provider':'arrays.org','dataset':route.dataset,'ticker':route.ticker,
                        'retrieved_at':datetime.now(timezone.utc).isoformat(),'rows':len(result['data']),
                        'sha256':file_hash(data),'routing':route.manifest(),
                        **{k:v for k,v in result.items() if k != 'data'},
                        'availability':'Vendor observation/publication fields retained separately from local capture; not independently attested',
                        'limitations':['No historical borrow/locate evidence inferred from short interest',
                                       'ETF holdings are a current US-equity subset; no historical date or weight renormalization',
                                       'Dark-pool prices never replace the existing price series; ratings are vendor scores, not broker recommendations']}
            atomic_json(folder/'manifest.json',manifest)
            atomic_json(folder.parent/'latest.json',{'version':version})
            return manifest

    def export(self, route, output):
        snapshot = self.latest(route.dataset,route.ticker)
        if snapshot is None:
            raise ArraysError('Arrays data is not cached; sync the explicit dataset/ticker first')
        destination = Path(output).resolve()
        sidecar = destination.with_suffix(destination.suffix+'.manifest.json')
        if destination.exists() or sidecar.exists():
            raise ArraysError('Export exists; choose a new immutable filename')
        records = json.loads(Path(snapshot['path']).read_text(encoding='utf-8'))
        destination.parent.mkdir(parents=True,exist_ok=True)
        created = sidecar_created = False
        try:
            with destination.open('x',encoding='utf-8',newline='') as target:
                created = True
                if destination.suffix.lower() == '.csv':
                    columns = sorted({key for row in records for key in row})
                    writer = csv.DictWriter(target,fieldnames=columns)
                    writer.writeheader()
                    writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in records)
                else:
                    json.dump(records,target,ensure_ascii=False,allow_nan=False)
            manifest = snapshot | {'source_snapshot_sha256':snapshot['sha256'],'sha256':file_hash(destination),
                                   'export_created_at':datetime.now(timezone.utc).isoformat()}
            with sidecar.open('x',encoding='utf-8') as target:
                sidecar_created = True
                json.dump(manifest,target,indent=2)
            return manifest
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            if sidecar_created:
                sidecar.unlink(missing_ok=True)
            raise


def load_arrays_config(pipeline_path):
    path = Path(pipeline_path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    configured = raw.get('arrays_config')
    if not isinstance(configured,str) or not configured:
        raise ArraysError('Configure arrays_config for uncovered Arrays datasets')
    config_path = Path(configured)
    if not config_path.is_absolute():
        config_path = path.parent/config_path
    values = json.loads(config_path.read_text(encoding='utf-8-sig'))
    allowed = {'root','page_size','max_pages','max_rows','max_bytes','timeout_seconds','run_seconds','requests_per_minute'}
    if not isinstance(values,dict) or set(values)-allowed or not isinstance(values.get('root'),str) or not values['root'].strip():
        raise ArraysError('Invalid Arrays config; credentials belong only in .env')
    root = Path(values.pop('root'))
    return ArraysStore(root if root.is_absolute() else config_path.parent/root),values


def arrays_request(store, config_path, route, command, *, output=None, start=None, end=None,
                   metrics='EPS',period_type='quarterly',fiscal_year=None,fiscal_quarter=None):
    import os
    cache,settings = load_arrays_config(config_path)
    if command == 'export':
        if start or end or fiscal_year or fiscal_quarter:
            raise ArraysError('Arrays offline export returns the pinned request; date/fiscal filtering requires a new sync')
        return cache.export(route,output)
    params = query_parameters(route.dataset,route.ticker,start=start,end=end,metrics=metrics,
                              period_type=period_type,fiscal_year=fiscal_year,fiscal_quarter=fiscal_quarter)
    eligible = equity_constituent_symbols(store) if route.dataset == 'etf_holdings' else None
    with ArraysClient(os.environ.get('ARRAYS_API_KEY',''),**settings) as client:
        return cache.sync(client,route,params,equity_symbols=eligible)
