import pytest
pd=pytest.importorskip('pandas')
from alpha_lab.equity_actions import diagnose_terminal_event


def diagnose(legs, *, counterparty_date='2020-01-06', counterparty_currency='USD'):
    actions=pd.DataFrame([{'ticker':'TARGET','date':pd.Timestamp('2020-01-03'),
                           'action':action,'value':value,'contraticker':contra}
                          for action,value,contra in [('delisted',1_000_000,'N/A'),*legs]])
    prices=pd.DataFrame({'ticker':['TARGET','BUYER'],
                         'date':pd.to_datetime(['2020-01-03',counterparty_date]),
                         'closeunadj':[40.,80.],'closeadj':[35.,75.]})
    return diagnose_terminal_event({'date':'2020-01-06','ticker':'TARGET','weight':.01},
                                   actions,prices,{'TARGET':{'USD'},'BUYER':{counterparty_currency}})


def test_standard_cash_and_stock_are_additive_on_raw_share_basis():
    result=diagnose([('acquisitioncash',10.,'BUYER'),('acquisitionstock',.5,'BUYER')])
    assert result['classification']=='cash_and_stock'
    assert result['diagnostic_consideration_usd_per_share']==50.
    assert result['diagnostic_difference_from_last_raw_close']==pytest.approx(.25)
    assert result['received_stock_marks'][0]['raw_close']==80.  # adjusted quote would use the wrong share basis
    assert result['economic_return_ready'] is False and result['terminal_actions_reconciled'] is False
    assert 'settlement_and_information_timing_not_verified' in result['unresolved_reasons']


def test_cash_only_uses_per_share_cash_not_delisting_market_cap():
    result=diagnose([('acquisitioncash',50.,'N/A'),('acquisitionby',1_000_000,'N/A')])
    assert result['classification']=='cash' and result['diagnostic_consideration_usd_per_share']==50.


def test_bankruptcy_market_cap_is_never_interpreted_as_a_payout_or_zero():
    result=diagnose([('bankruptcyliquidation',1_000_000,'N/A')])
    assert result['classification']=='insolvency'
    assert result['diagnostic_consideration_usd_per_share'] is None
    assert 'terminal_proceeds_unknown' in result['unresolved_reasons']


@pytest.mark.parametrize('complex_action',['acquisitionelectcash','acquisitioncasheur','acquisitionspecial'])
def test_elections_foreign_cash_and_special_securities_are_not_added_to_default(complex_action):
    result=diagnose([('acquisitioncash',50.,'BUYER'),(complex_action,25.,'BUYER')])
    assert result['classification']=='election_foreign_cash_or_special_security'
    assert result['diagnostic_consideration_usd_per_share'] is None


@pytest.mark.parametrize('parameters',[{'counterparty_date':'2020-01-03'},
                                     {'counterparty_date':'2020-01-07'}, {'counterparty_currency':'CAD'}])
def test_received_stock_requires_exact_date_quote_and_verified_usd_currency(parameters):
    result=diagnose([('acquisitionstock',.5,'BUYER')],**parameters)
    assert result['diagnostic_consideration_usd_per_share'] is None
    assert len(result['unresolved_reasons'])>=2


def test_duplicate_cash_legs_are_ambiguous_instead_of_double_counted():
    result=diagnose([('acquisitioncash',50.,'BUYER'),('acquisitioncash',50.,'N/A')])
    assert result['diagnostic_consideration_usd_per_share'] is None
    assert 'ambiguous_duplicate_consideration_legs' in result['unresolved_reasons']


@pytest.mark.parametrize('has_event',[False,True])
def test_audit_handles_empty_studies_and_uses_price_not_reporting_currency(tmp_path,monkeypatch,has_event):
    import json
    import duckdb
    from alpha_lab.equity_actions import audit_terminal_events
    from alpha_lab.market_data.sharadar import file_hash
    sources={}
    date=pd.Timestamp('2014-01-02')
    frames={
        'stocks':pd.DataFrame({'ticker':['A'],'date':[date],'closeunadj':[10.],'closeadj':[10.]}),
        'actions':pd.DataFrame({'ticker':['A','A'],'date':[date,date],
                               'action':['delisted','acquisitioncash'],'value':[100.,12.],'contraticker':['N/A','N/A']}),
        'tickers':pd.DataFrame({'table':['SEP','SF1'],'ticker':['A','A'],'currency':['USD','CAD']}),
        'descriptions':pd.DataFrame({'table':['ACTIONTYPES','SEP'],'indicator':['acquisitioncash','closeunadj'],
                                     'description':['USD per share','raw close'],'unittype':['numeric','USD/share']})}
    for name,frame in frames.items():
        path=tmp_path/(name+'.parquet')
        duckdb.from_df(frame).write_parquet(str(path))
        sources[name]={'path':str(path),'parquet_sha256':file_hash(path)}
    panel=tmp_path/'panel';panel.mkdir()
    (panel/'sources.json').write_text(json.dumps(sources))
    (panel/'actions.parquet').write_bytes((tmp_path/'actions.parquet').read_bytes())
    (panel/'panel.json').write_text(json.dumps({'start':'2014-01-01','end':'2014-12-31',
        'files':{name:file_hash(panel/name) for name in ('sources.json','actions.parquet')}}))
    validation=tmp_path/'validation';validation.mkdir()
    events=[{'date':'2014-01-03','ticker':'A','weight':.01}] if has_event else []
    (validation/'holdout.json').write_text(json.dumps({'missing_event_details':events}))
    expected=file_hash(validation/'holdout.json')
    class Store:
        def latest(self,name):return sources[name]
        def _connect(self):return duckdb.connect()
    monkeypatch.setattr('alpha_lab.equity_actions.load_config',lambda config:(Store(),[],0))
    result=audit_terminal_events(panel,validation,tmp_path/'audit',tmp_path/'config.json')
    assert result['event_observations']==int(has_event)
    assert result['diagnostic_standard_consideration_marks']==int(has_event)
    if has_event:
        records=json.loads((tmp_path/'audit'/'events.json').read_text())
        assert records[0]['diagnostic_consideration_usd_per_share']==12.
    assert result['pnl_modified'] is False and file_hash(validation/'holdout.json')==expected
