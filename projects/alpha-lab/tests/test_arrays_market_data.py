"""Arrays is an explicit US-equity coverage extension, never an outage fallback."""
import json
from pathlib import Path

import httpx
import pytest

pd = pytest.importorskip('pandas')
duckdb = pytest.importorskip('duckdb')

from alpha_lab.market_data.arrays import ArraysClient, ArraysError, ArraysStore, query_parameters, load_arrays_config
from alpha_lab.market_data import pipeline
from alpha_lab.market_data.routing import ARRAYS_EXTRA_DATASETS, Route, select_route
from alpha_lab.market_data.sharadar import SharadarError, file_hash
from alpha_lab.market_data.sharadar_store import SharadarStore


@pytest.fixture
def reference(tmp_path):
    path = tmp_path/'metadata.parquet'
    frame = pd.DataFrame({'table':['SEP','SEP','SEP','SFP','SFP'],
                          'ticker':['AAPL','BRK.B','VOD.L','SPY','GLD'],
                          'permaticker':[1,2,3,4,5], 'exchange':['NASDAQ','NYSE','LSE','NYSEARCA','NYSEARCA'],
                          'currency':['USD']*5,'category':['Domestic Common Stock']*3+['ETF']*2})
    duckdb.from_df(frame).write_parquet(str(path))
    class Store:
        def latest(self,table):
            if table == 'tickers':return {'path':str(path),'version':'a'*32,'parquet_sha256':file_hash(path)}
            return None
        def _connect(self):return duckdb.connect(':memory:')
    return Store()


def route(dataset='finra_short_interest', symbol='AAPL'):
    return Route(dataset,'arrays',dataset,'not_covered_by_prior_providers',symbol,
                 scope={'symbol':symbol,'asset_scope':'us_equity_etf' if dataset=='etf_holdings' else 'us_listed_equity','security_id':1})


def response(data, **extra):
    return httpx.Response(200,json={'success':True,'data':data,**extra})


@pytest.mark.parametrize('dataset',ARRAYS_EXTRA_DATASETS)
def test_only_distinct_equity_datasets_route_to_arrays(reference,dataset):
    result=select_route(reference,dataset,ticker='SPY' if dataset=='etf_holdings' else 'AAPL')
    assert result.provider=='arrays' and result.scope['historical_eligibility_attested'] is False


@pytest.mark.parametrize('dataset',['fundamentals','SF1','stocks','SEP','actions','insiders','holdings','sp500','metrics','daily'])
def test_preexisting_coverage_never_routes_to_arrays(reference,dataset):
    assert select_route(reference,dataset).provider=='sharadar'


@pytest.mark.parametrize('dataset',['news','intraday_prices','fund_fees','dividend_details','overnight_prices'])
def test_unimplemented_prior_adapter_is_not_a_coverage_gap(reference,dataset):
    with pytest.raises(SharadarError,match='Tiingo'):
        select_route(reference,dataset,ticker='AAPL')


@pytest.mark.parametrize('dataset',['crypto','forex','macro','options','borrow_availability','borrow_cost','prices'])
def test_other_asset_classes_and_unknown_datasets_are_blocked(reference,dataset):
    with pytest.raises(SharadarError):select_route(reference,dataset,ticker='AAPL')


def test_foreign_listing_non_equity_fund_and_corrupt_reference_fail_closed(reference,tmp_path):
    with pytest.raises(SharadarError,match='US-listed'):
        select_route(reference,'consensus_estimates',ticker='VOD.L')
    with pytest.raises(SharadarError,match='reviewed'):
        select_route(reference,'etf_holdings',ticker='GLD')
    assert select_route(reference,'consensus_estimates',ticker='BRK-B').ticker=='BRK.B'
    with pytest.raises(SharadarError,match='Asset table'):
        select_route(reference,'consensus_estimates',ticker='AAPL',asset_table='funds')
    bad=reference.latest('tickers')|{'parquet_sha256':'0'*64}
    reference.latest=lambda _:bad
    with pytest.raises(SharadarError,match='verified'):
        select_route(reference,'finra_short_interest',ticker='AAPL')


def test_auth_header_no_query_secret_and_no_redirects():
    calls=[]
    def handler(request):
        calls.append(request)
        assert request.headers['X-API-Key']=='synthetic-secret'
        assert 'synthetic-secret' not in str(request.url)
        return httpx.Response(302,headers={'Location':'https://elsewhere.example/steal'})
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArraysError,match='HTTP 302'):client.fetch(route(),{'symbol':'AAPL'})
    assert len(calls)==1


@pytest.mark.parametrize('code',[401,402,403,429,500])
def test_provider_failures_do_not_retry_pay_or_fallback(code):
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(code,json={'error':'synthetic-secret private request payload'})
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArraysError) as error:client.fetch(route(),{'symbol':'AAPL'})
    assert len(calls)==1 and 'synthetic-secret' not in str(error.value) and 'private' not in str(error.value)


def test_offset_pagination_preserves_all_observation_period_rows():
    offsets=[]
    def handler(request):
        offset=int(request.url.params['offset']);offsets.append(offset)
        return response([{'symbol':'AAPL','observed_at':'2026-10-01','fiscal_period':'2027Q1'},
                         {'symbol':'AAPL','observed_at':'2026-10-01','fiscal_period':'2027Q2'}] if offset==0 else
                        [{'symbol':'AAPL','observed_at':'2026-10-02','fiscal_period':'2027Q1'}])
    with ArraysClient('synthetic-secret',page_size=2,requests_per_minute=600,transport=httpx.MockTransport(handler)) as client:
        result=client.fetch(route('consensus_estimates'),{'symbol':'AAPL','metrics':'EPS','type':'estimate'})
    assert offsets==[0,2] and len(result['data'])==3 and result['complete_requested_window'] is True


def test_caps_and_repeated_pages_preserve_previous_snapshot(tmp_path):
    good=httpx.MockTransport(lambda r:response([{'symbol':'AAPL','short_interest':10}]))
    cache=ArraysStore(tmp_path/'cache')
    with ArraysClient('synthetic-secret',transport=good) as client:cache.sync(client,route(),{'symbol':'AAPL'})
    before=cache.latest('finra_short_interest','AAPL')['version']
    bad=httpx.MockTransport(lambda r:response([{'symbol':'AAPL','short_interest':10},{'symbol':'AAPL','short_interest':20}]))
    with ArraysClient('synthetic-secret',max_rows=1,transport=bad) as client:
        with pytest.raises(ArraysError,match='row cap'):cache.sync(client,route(),{'symbol':'AAPL'})
    assert cache.latest('finra_short_interest','AAPL')['version']==before
    with ArraysClient('synthetic-secret',page_size=2,requests_per_minute=600,transport=bad) as client:
        with pytest.raises(ArraysError,match='did not advance'):client.fetch(route('consensus_estimates'),{'symbol':'AAPL'})


@pytest.mark.parametrize('data',[{},[{'symbol':'VOD.L'}],[{'symbol':'AAPL','echo':'synthetic-secret'}]])
def test_bad_or_mixed_scope_payload_is_not_persisted(data):
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(lambda r:response(data))) as client:
        with pytest.raises(ArraysError):client.fetch(route(),{'symbol':'AAPL'})


def test_etf_portfolio_is_filtered_to_verified_us_equities_without_renormalization():
    data=[{'symbol':'SPY','asset':'AAPL','weight_percentage':5},{'symbol':'SPY','asset':'VOD.L','weight_percentage':5},
          {'symbol':'SPY','asset':'USD','weight_percentage':2},{'symbol':'SPY','asset':'BTC','weight_percentage':1}]
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(lambda r:response(data))) as client:
        result=client.fetch(route('etf_holdings','SPY'),{'symbol':'SPY'},equity_symbols={'AAPL'})
    assert result['data']==[data[0]] and result['excluded_out_of_scope_rows']==3
    assert result['scope_filtered_portfolio'] is True


@pytest.mark.parametrize('data',[[{'symbol':'GLD','asset':'AAPL'}],[{'symbol':'SPY','asset':'BTC'}]])
def test_holdings_parent_and_all_dropped_results_are_rejected(data):
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(lambda r:response(data))) as client:
        with pytest.raises(ArraysError):client.fetch(route('etf_holdings','SPY'),{'symbol':'SPY'},equity_symbols={'AAPL'})


def test_nested_exports_are_immutable_and_hash_checked(tmp_path):
    cache=ArraysStore(tmp_path/'cache')
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(lambda r:response([{'symbol':'AAPL','transcript':[{'content':'test'}]}]))) as client:
        result=cache.sync(client,route('earnings_transcripts'),{'symbol':'AAPL'})
    output=tmp_path/'transcripts.csv';manifest=cache.export(route('earnings_transcripts'),output)
    assert manifest['source_snapshot_sha256']==result['sha256'] and manifest['sha256']==file_hash(output)
    with pytest.raises(ArraysError,match='exists'):cache.export(route('earnings_transcripts'),output)
    Path(cache.latest('earnings_transcripts','AAPL')['path']).write_text('modified')
    with pytest.raises(ArraysError,match='Invalid cached'):cache.export(route('earnings_transcripts'),tmp_path/'other.json')


def test_request_time_basis_and_current_holdings_limits():
    p=query_parameters('finra_short_interest','AAPL',start='2026-10-01',end='2026-10-13')
    assert p['time_type']=='UPDATED_AT' and p['start_time']<p['end_time']
    with pytest.raises(ArraysError,match='current snapshot'):query_parameters('etf_holdings','SPY',start='2020-01-01')
    with pytest.raises(ArraysError,match='fiscal year'):query_parameters('earnings_transcripts','AAPL')
    with pytest.raises(ArraysError,match='32 days'):query_parameters('dark_pool_activity','AAPL',start='2020-01-01',end='2022-01-01')


def test_programmatic_arrays_sync_never_constructs_previous_provider_clients(reference,monkeypatch):
    monkeypatch.setattr(pipeline,'SharadarClient',lambda *a,**k:pytest.fail('No covered provider request'))
    monkeypatch.setattr(pipeline,'TiingoClient',lambda *a,**k:pytest.fail('No covered provider request'))
    monkeypatch.setattr(pipeline,'arrays_request',lambda *a,**k:{'rows':1})
    routes=pipeline.sync_pipeline(reference,[],30,{},['AAPL'],dataset='finra_short_interest',arrays_pipeline_config='config.json',emit=lambda _:None)
    assert routes[0].provider=='arrays'


@pytest.mark.parametrize('extra',[{'api_key':'private'}, {'base_url':'https://elsewhere.example'}, {'root':''}])
def test_config_rejects_embedded_credentials_custom_hosts_and_empty_roots(tmp_path,extra):
    config=tmp_path/'pipeline.json';config.write_text(json.dumps({'arrays_config':'arrays.json'}))
    (tmp_path/'arrays.json').write_text(json.dumps({'root':'cache',**extra}))
    with pytest.raises(ArraysError,match='Invalid Arrays config'):load_arrays_config(config)


@pytest.mark.parametrize('dataset,data,extra',[('equity_ratings',[{'symbol':'AAPL'}],{}),
                                              ('finra_short_interest',[{'symbol':'AAPL'}],{'pagination':{'has_more':True}}),
                                              ('consensus_estimates',[{'symbol':'AAPL'}],{'metadata':{'pagination':{'has_more':'true'}}})])
def test_truncated_or_ambiguous_completeness_is_rejected(dataset,data,extra):
    with ArraysClient('synthetic-secret',max_rows=1,transport=httpx.MockTransport(lambda r:response(data,**extra))) as client:
        with pytest.raises(ArraysError):client.fetch(route(dataset),{'symbol':'AAPL'})


def test_cli_requires_explicit_scope_and_offline_export_preserves_native_request(reference,monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,'load_config',lambda _: (reference,[],30,{},['SPY']))
    calls=[]
    def request(*args,**kwargs):
        calls.append((args,kwargs));return {'rows':1}
    monkeypatch.setattr(pipeline,'arrays_request',request)
    assert pipeline.main(['sync','--dataset','etf_holdings'])==1
    assert pipeline.main(['sync','--dataset','finra_short_interest','--ticker','AAPL','--ticker','BRK.B'])==1
    assert pipeline.main(['route','--dataset','finra_short_interest','--ticker','AAPL'])==0
    assert not calls
    assert pipeline.main(['export','--dataset','finra_short_interest','--ticker','AAPL','--output',str(tmp_path/'out.json')])==0
    assert calls[0][0][3]=='export' and calls[0][0][2].provider=='arrays'


def test_failed_export_sidecar_write_cleans_only_own_partial_files(tmp_path,monkeypatch):
    cache=ArraysStore(tmp_path/'cache')
    with ArraysClient('synthetic-secret',transport=httpx.MockTransport(lambda r:response([{'symbol':'AAPL'}]))) as client:
        cache.sync(client,route(),{'symbol':'AAPL'})
    original=json.dump
    def dump(value,target,**kwargs):
        if str(target.name).endswith('.manifest.json'):
            target.write('{');raise OSError('simulated disk failure')
        return original(value,target,**kwargs)
    monkeypatch.setattr(json,'dump',dump)
    output=tmp_path/'export.json'
    with pytest.raises(OSError):cache.export(route(),output)
    assert not output.exists() and not output.with_suffix('.json.manifest.json').exists()
    assert cache.latest('finra_short_interest','AAPL')['rows']==1
