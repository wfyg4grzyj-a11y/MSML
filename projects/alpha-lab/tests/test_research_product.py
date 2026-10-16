"""Shared API/viewer, public-only tool authority and explicit LLM admission."""
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import json

import pytest
pytest.importorskip('numpy');pytest.importorskip('scipy')
from fastapi import FastAPI
from fastapi.testclient import TestClient
from alpha_lab.llm_runtime import LLMSettings
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research import cli
from alpha_lab.research.contracts import ResearchError, read_json
from alpha_lab.research.public_design import PublicSignalTools, run_public_design
from alpha_lab.research.service import ResearchService
from alpha_lab.research.web import research_router
from tests.test_integrated_study import prepared


@pytest.fixture
def saved(tmp_path):
    bundle,spec,registry=prepared(tmp_path)
    service=ResearchService(tmp_path)
    service.run(bundle,spec,tmp_path/'run',registry)
    return service


def test_notebook_cli_and_dashboard_inspect_identical_saved_evidence(saved,capsys,monkeypatch):
    import alpha_lab.research.study_workflow as flow
    monkeypatch.setattr(flow,'start_process',lambda *a,**k:pytest.fail('Viewing started an experiment'))
    expected=saved.inspect('run')
    assert cli.main(['research-inspect','--root',str(saved.root),'--run','run'])==0
    assert json.loads(capsys.readouterr().out)==expected
    app=FastAPI();app.include_router(research_router(lambda:saved.root))
    with TestClient(app) as client:
        assert client.get('/api/research/inspect',params={'run':'run'}).json()==expected
        response=client.get('/research');assert response.status_code==200
        assert 'Opening this viewer starts no experiments' in response.text
        assert 'frame-ancestors' in response.headers['content-security-policy']
        assert client.get('/api/research/report',params={'run':'run'}).text==saved.report('run')
        listing=client.get('/api/research/runs').json()
        assert any(r['id']=='run' and r['status']=='completed' for r in listing['runs'])
        assert client.post('/api/research/run').status_code in (404,405)


@pytest.mark.parametrize('name',['../secret','C:/secret','run/../../secret','.env','run\\..\\secret','/absolute'])
def test_research_viewer_paths_do_not_expose_other_workspace_files(tmp_path,name):
    service=ResearchService(tmp_path)
    with pytest.raises(ResearchError):service.path(name)


def test_tampered_and_partial_artifacts_have_progress_but_no_verified_report(saved):
    (saved.root/'run/ledger.json').write_text('{}')
    assert saved.progress('run')['status']=='completed'
    with pytest.raises(ResearchError):saved.report('run')
    app=FastAPI();app.include_router(research_router(lambda:saved.root))
    with TestClient(app) as client:
        response=client.get('/api/research/report',params={'run':'run'})
        assert response.status_code==400 and 'ledger.json' not in response.text


def test_public_toolset_has_no_evaluation_filesystem_budget_or_attestation_authority(tmp_path):
    tools=PublicSignalTools(tmp_path,1)
    for name in ('evaluate','write_file','read_file','shell','change_budget','reveal_holdout','attest_data'):
        with pytest.raises(ResearchError):tools(name,{})
    proposal={'name':'public-value','factors':[{'feature':'book_yield','coefficient':1.}],
              'premise':'A public valuation hypothesis for explicit later local testing'}
    output=json.loads(tools('draft_recipe',proposal)['output'])
    assert output['performance']=='unknown'
    duplicate=proposal | {'name':'renamed','factors':[{'feature':'book_yield','coefficient':7.}]}
    assert json.loads(tools('draft_recipe',duplicate)['output'])['draft_id']==output['draft_id']
    with pytest.raises(ResearchError,match='budget'):
        tools('draft_recipe',proposal | {'factors':[{'feature':'roa','coefficient':1.}]})
    with pytest.raises(ResearchError):tools('draft_recipe',proposal | {'output_path':'../core.py'})
    assert tools('validate_draft',{'draft_id':output['draft_id']})
    assert not (tmp_path/'registry.sqlite').exists()
    assert tools('report_to_user',{'summary':'Economic premise and proposed tests; performance is unknown.'})['done']


def design_config(tmp_path):
    settings=LLMSettings(model='gpt-6.1-sol',reasoning_effort='high',max_calls=4,max_reserved_tokens=40000,
                         max_run_seconds=60,max_request_seconds=30,read_timeout_seconds=20,max_context_bytes=16000,
                         max_output_tokens=1000,verbosity='low',deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat())
    config={'data_scope':'public_only','base_url':'https://test-gateway.invalid/v1','llm':asdict(settings),'max_drafts':1}
    path=tmp_path/'config.json';atomic_json(path,config)
    return path,config


def test_public_design_requires_explicit_complete_controls_before_provider_setup(tmp_path,monkeypatch):
    import alpha_lab.llm_runtime as runtime
    monkeypatch.setattr(runtime,'bounded_provider',lambda *a,**k:pytest.fail('Invalid config reached provider'))
    path,config=design_config(tmp_path)
    config['llm'].pop('verbosity');atomic_json(path,config)
    with pytest.raises(ResearchError):run_public_design(path,tmp_path/'run')
    path,config=design_config(tmp_path);config['llm']['deadline_utc']='2020-01-01T00:00:00+00:00';atomic_json(path,config)
    with pytest.raises(ResearchError):run_public_design(path,tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_explicit_public_design_uses_only_domain_tools_and_never_evaluates(tmp_path,monkeypatch):
    import alpha_lab.llm_runtime as runtime
    import alpha_lab.agent as agent
    import alpha_lab.context as context
    calls=[]
    class Provider:
        def close(self):calls.append('close')
    monkeypatch.setattr(runtime,'bounded_provider',lambda *a,**k:(calls.append('provider') or Provider()))
    monkeypatch.setattr(context,'ContextManager',lambda *a,**k:None)
    class Agent:
        _done=True
        def __init__(self,*args,**kwargs):self.tools=kwargs['tool_executor'];self.schemas=kwargs['tools']
        def run(self,prompt):
            assert {t['name'] for t in self.schemas}=={'research_catalog','draft_recipe','validate_draft','report_to_user'}
            self.tools('research_catalog',{})
            self.tools('draft_recipe',{'name':'public','factors':[{'feature':'roa','coefficient':1.}],
                                     'premise':'A public profitability hypothesis without private observations'})
            self.tools('report_to_user',{'summary':'Proposed local tests, no performance claimed.'})
    monkeypatch.setattr(agent,'AgentLoop',Agent)
    monkeypatch.setenv('OPENAI_BASE_URL','https://test-gateway.invalid/v1');monkeypatch.setenv('OPENAI_API_KEY','synthetic-private')
    path,_=design_config(tmp_path)
    state=run_public_design(path,tmp_path/'run')
    assert state['status']=='completed' and state['local_evaluation_started'] is False
    assert calls==['provider','close'] and not (tmp_path/'run/ledger.json').exists()
    assert 'synthetic-private' not in (tmp_path/'run/design-run.json').read_text()
    with pytest.raises(FileExistsError):run_public_design(path,tmp_path/'run')


@pytest.mark.parametrize('failure',['agent','close'])
def test_public_design_failure_or_cleanup_error_retains_failed_intent(tmp_path,monkeypatch,failure):
    import alpha_lab.llm_runtime as runtime
    import alpha_lab.agent as agent
    import alpha_lab.context as context
    class Provider:
        def close(self):
            if failure=='close':raise OSError('synthetic cleanup failure')
    monkeypatch.setattr(runtime,'bounded_provider',lambda *a,**k:Provider())
    monkeypatch.setattr(context,'ContextManager',lambda *a,**k:None)
    class Agent:
        _done=failure!='agent'
        def __init__(self,*a,**k):self.tools=k['tool_executor']
        def run(self,prompt):
            self.tools('draft_recipe',{'name':'public','factors':[{'feature':'roa','coefficient':1.}],
                                      'premise':'Synthetic public proposal without market observations'})
            self.tools('report_to_user',{'summary':'No performance claim.'})
    monkeypatch.setattr(agent,'AgentLoop',Agent)
    monkeypatch.setenv('OPENAI_BASE_URL','https://test-gateway.invalid/v1');monkeypatch.setenv('OPENAI_API_KEY','synthetic-private')
    path,_=design_config(tmp_path)
    with pytest.raises(ResearchError):run_public_design(path,tmp_path/'failed')
    assert read_json(tmp_path/'failed/design-run.json')['status']=='failed'
    with pytest.raises(FileExistsError):run_public_design(path,tmp_path/'failed')
