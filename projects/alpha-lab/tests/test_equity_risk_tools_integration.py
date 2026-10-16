"""Consumer integration over independent synthetic saved covariances, not fits."""
import json
from pathlib import Path

import pytest
import numpy as np

pytest.importorskip('equity_risk_tools')
from equity_risk.data import Forecast

from alpha_lab.research.contracts import ResearchError, digest, file_hash, read_json
from alpha_lab.research.risk_tools import (MODELS, PortfolioRiskService, inspect_risk_operation,
                                         pin_library, save_risk_operation, verify_library)
from alpha_lab.research.service import ResearchService
from tests.test_risk_portfolio import policy

DAY = '2020-01-02'
IDS = ['101', '102', '103', '104']
BASE = np.diag([.0001, .0002, .0003, .0004])+np.ones((4,4))*.0001


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'export';root.mkdir()
    for index, name in enumerate(MODELS):
        diagnostics={'data_provenance':{'point_in_time':False,'benchmark_id':'SPY','cash_id':'cash','calendar_id':'XNYS'},
                     'parameter_selection_end':'2019-12-31','correlation_fits':[{'converged':True}]}
        covariance={h:BASE*h*(index+1) for h in (1,5,21)}
        components={h:{} for h in covariance}
        if name=='market_residual':diagnostics['effective_beta_1']=[1.,.8,1.2,1.]
        if name=='fundamental_latent':
            diagnostics.update(local_fit={'converged':True,'gradient_max':0.},latent_rank=0,
                named_columns=['market','size','beta'],exposures=[[1.,float(i),1.] for i in range(4)])
            components={h:{'named':c.copy()} for h,c in covariance.items()}
        Forecast(name,DAY,tuple(IDS),covariance,diagnostics,components).save(root/name)
    write(root/'manifest.json',{'models':list(MODELS),'as_of':DAY,'research_only':True,'point_in_time_attested':False})
    (root/'security-order.csv').write_text('position,security_id,ticker,sic\n0,101,AAA,3571\n1,102,BBB,3674\n2,103,CCC,6020\n3,104,DDD,6020\n')
    config=tmp_path/'registry.json'
    write(config,{'schema_version':1,'exports':['export'],'research_runs':[],'default_model':'market','cache_size':3})
    return config


@pytest.fixture
def pinned(source,tmp_path):
    root=tmp_path/'library'
    pin_library(source,root)
    return root


def args(**changes):
    return {'as_of':DAY,'positions':[{'security':'101','weight':.6},{'security':'102','weight':-.4}],
            'model':'market','horizon':1,'identifier':'security_id',**changes}


def projection(**changes):
    positions=[{'security_id':sid,'score':score,'raw_price_usd':100.,'held_shares':0.,'eligible':True,
                'tradable':True,'borrow_available_shares':10000.,'issuer_id':sid,'sector_id':'sector','beta':1.}
               for sid,score in zip(IDS,[1.,-1.,.5,-.5])]
    return {'as_of':DAY,'model':'market','decision_time':'2020-01-03T21:00:00+00:00',
            'available_at':'2020-01-02T21:05:00+00:00','availability_kind':'declared_assumption',
            'availability_evidence_sha256':'a'*64,'beta_evidence_sha256':'b'*64,'max_age_calendar_days':3,
            'benchmark_id':'SPY','cash_id':'cash','calendar_id':'XNYS','nav_usd':1_000_000.,
            'positions':positions,'policy':policy(covariance_penalty=.5).model_dump(),**changes}


def test_pin_copies_only_documented_inputs_and_preserves_producer(source,tmp_path):
    before={str(p):file_hash(p) for p in (source.parent/'export').rglob('*') if p.is_file()}
    write(source.parent/'export'/'unrelated.json',{'unrelated':True})
    result=pin_library(source,tmp_path/'library')
    assert result['snapshots']==3
    assert not list((tmp_path/'library').rglob('unrelated.json'))
    assert before=={p:file_hash(Path(p)) for p in before}
    assert verify_library(tmp_path/'library')['library_id']==result['library_id']
    with pytest.raises(FileExistsError):pin_library(source,tmp_path/'library')


def test_dated_catalog_and_weighted_reference(pinned):
    risk=PortfolioRiskService(pinned)
    assert len(risk.catalog()['snapshots'])==3
    result=risk.invoke('risk_portfolio',args(benchmark=None))['result']
    w=np.array([.6,-.4,0.,0.])
    assert result['variance']==pytest.approx(w@BASE@w)
    assert result['gross_exposure']==1.
    assert result['net_exposure']==pytest.approx(.2)
    assert result['metadata']['research_only'] is True
    assert result['metadata']['point_in_time_attested'] is False
    assert len(risk.schemas)==12
    assert json.loads(risk('risk_catalog',{})['output'])['ok'] is True


@pytest.mark.parametrize('arguments,code',[
    (args(as_of='2020-01-03',benchmark=None),'snapshot_unavailable'),
    (args(horizon=2,benchmark=None),'unsupported_horizon'),
    (args(positions=[{'security':'UNKNOWN','weight':0.}],benchmark=None),'unknown_security'),
    (args(positions=[{'security':'101','weight':.1},{'security':'101','weight':.2}],benchmark=None),'duplicate_position')])
def test_unavailable_date_horizon_or_identity_is_explicit(pinned,arguments,code):
    result=PortfolioRiskService(pinned).invoke('risk_portfolio',arguments)
    assert result['ok'] is False
    assert result['error']['code']==code


def test_covariance_aligned_to_permanent_ids_and_readonly(pinned):
    matrix,meta=PortfolioRiskService(pinned).covariance(as_of=DAY,model='direct',horizon=5,security_ids=['103','101'])
    np.testing.assert_allclose(matrix,BASE[np.ix_([2,0],[2,0])]*15)
    assert not matrix.flags.writeable
    assert meta['horizon_sessions']==5


@pytest.mark.parametrize('name',['exports/0/market_residual/forecast.npz','exports/0/manifest.json','exports/0/security-order.csv','risk-tools.json'])
def test_cached_service_rechecks_all_artifact_bytes(pinned,name):
    risk=PortfolioRiskService(pinned)
    risk.invoke('risk_portfolio',args(benchmark=None))
    target=pinned/name
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ResearchError,match='bytes changed'):risk.catalog()


def test_incomplete_library_and_package_change_are_rejected(pinned,monkeypatch):
    from alpha_lab.research import risk_tools
    monkeypatch.setattr(risk_tools,'_package_identity',lambda:{'different':True})
    with pytest.raises(ResearchError,match='runtime differs'):PortfolioRiskService(pinned)
    (pinned/'COMPLETE.json').unlink()
    with pytest.raises(ResearchError):verify_library(pinned)


def test_failed_numerical_fit_cannot_be_pinned(source,tmp_path):
    path=source.parent/'export/fundamental_latent/forecast.json'
    info=read_json(path);info['diagnostics']['local_fit']['converged']=False;write(path,info)
    with pytest.raises(ValueError):pin_library(source,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_saved_rejection_and_offline_dashboard_inspection(pinned,tmp_path):
    output=tmp_path/'reports/rejected'
    saved=save_risk_operation(pinned,{'tool':'risk_portfolio','arguments':args(as_of='2021-01-01',benchmark=None)},output)
    assert saved['status']=='rejected' and saved['artifacts_verified']
    # The archived report stays inspectable without a source library or runtime.
    (pinned/'COMPLETE.json').unlink()
    service=ResearchService(output.parent)
    assert service.inspect('rejected')==saved
    assert service.list_runs()['runs'][0]['stage']=='local_risk_query'
    assert 'snapshot_unavailable' in service.report('rejected')
    (output/'response.json').write_text('{}')
    with pytest.raises(ResearchError):inspect_risk_operation(output)


@pytest.mark.parametrize('changes',[
    {'decision_time':'2020-01-01T21:00:00+00:00'},
    {'decision_time':'2020-02-01T21:00:00+00:00'},
    {'available_at':'2020-01-02T00:00:00+00:00'},
    {'available_at':'2020-01-04T21:00:00+00:00'},
    {'benchmark_id':'OTHER'}, {'cash_id':'OTHER'}, {'calendar_id':'OTHER'},
    {'decision_time':'2020-01-03T21:00:00'}])
def test_projection_rejects_future_stale_availability_and_bindings(pinned,changes):
    with pytest.raises(ResearchError):PortfolioRiskService(pinned).project(projection(**changes))


def test_projection_uses_saved_covariance_without_losing_constraints(pinned,tmp_path):
    request=projection()
    result=PortfolioRiskService(pinned).project(request)
    assert result['portfolio']['status']=='feasible'
    weights=np.array(list(result['portfolio']['target_weights'].values()))
    assert abs(weights.sum())<1e-6
    assert max(abs(weights))<=.300001
    assert result['portfolio']['exposures']['forecast_variance']==pytest.approx(weights@BASE@weights)
    assert result['historical_study_attestation'] is False
    assert result['investment_ready'] is False and result['orders_submitted']==0
    saved=save_risk_operation(pinned,request,tmp_path/'projection',projection=True)
    assert saved['status']=='completed'
    request['positions'][0]['beta']=None
    missing=PortfolioRiskService(pinned).project(request)['portfolio']
    assert missing['status']=='insufficient_evidence'
    request=projection()
    for position in request['positions']:position['borrow_available_shares']=None
    no_locate=PortfolioRiskService(pinned).project(request)['portfolio']
    assert no_locate['status']=='zero_target'


def test_saved_risk_tools_do_not_enter_public_llm_toolset(pinned):
    from alpha_lab.research.public_design import DESIGN_TOOLS
    assert not any(schema['name'].startswith('risk_') for schema in DESIGN_TOOLS)
