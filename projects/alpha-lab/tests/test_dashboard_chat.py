"""Public chat admission and shared controls with a fully mocked SDK."""
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi.testclient import TestClient
import pytest

import alpha_lab.server as srv
from alpha_lab.dashboard_chat import PublicChatSession,ChatError
from alpha_lab.llm_runtime import LLMRuntime,LLMSettings


@pytest.fixture(autouse=True)
def chat_workspace(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace';(workspace/'logs').mkdir(parents=True)
    (workspace/'logs'/'dispatcher.jsonl').write_text(json.dumps({'type':'status','detail':'PRIVATE_WORKSPACE_SENTINEL'})+'\n')
    monkeypatch.setattr(srv,'_workspace',str(workspace))
    monkeypatch.setattr(srv,'_chat_config_path',None)
    monkeypatch.setattr(srv,'manager',srv.DashboardState())
    monkeypatch.setenv('OPENAI_BASE_URL','https://mock.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY','mock-api-key-never-send-as-question')
    yield workspace
    assert srv._chat_session is None and not srv._chat_request_tasks


def config_file(tmp_path,**overrides):
    settings=LLMSettings('gpt-6.1-sol','high',max_output_tokens=512,max_calls=2,
                         max_reserved_tokens=200000,max_run_seconds=600,max_request_seconds=30,
                         read_timeout_seconds=10,max_context_bytes=16000,verbosity='low',
                         deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    config={'data_scope':'public_only','base_url':'https://mock.invalid/v1',
            'run_directory':str(tmp_path/'chat-run'),'max_question_bytes':2000,'llm':asdict(settings)}
    config.update(overrides)
    path=tmp_path/'chat.json';path.write_text(json.dumps(config),encoding='utf-8')
    return path,config


def stream_answer(text='Mock public answer'):
    message=NS(type='message',content=[NS(type='output_text',text=text)])
    message.model_dump=lambda:{'type':'message','role':'assistant','content':[]}
    response=NS(id='mock-r',output=[message],status='completed',usage=NS(input_tokens=20,output_tokens=8))
    stream=MagicMock();stream.__iter__.return_value=iter([NS(type='response.completed',response=response)])
    return stream


@pytest.fixture
def mocked_sdk(monkeypatch):
    client=MagicMock()
    client.responses.create.side_effect=lambda **kwargs:stream_answer()
    factory=MagicMock(return_value=client)
    monkeypatch.setattr('alpha_lab.client.OpenAI',factory)
    return client,factory


def test_default_chat_is_disabled_without_constructing_a_provider(mocked_sdk):
    client,factory=mocked_sdk
    with TestClient(srv.app) as api:
        assert api.get('/api/chat-config').json()['enabled'] is False
        assert api.post('/api/chat',json={'question':'Explain momentum','consent_to_send':True}).status_code==503
    factory.assert_not_called();client.responses.create.assert_not_called()


@pytest.mark.parametrize('model',['gpt-5.6-sol','gpt-6-sol','gpt-6.1-sol'])
@pytest.mark.parametrize('effort',['medium','high'])
def test_public_payload_uses_intentional_model_effort_and_shared_bounded_responses(tmp_path,monkeypatch,mocked_sdk,model,effort):
    path,config=config_file(tmp_path);config['llm'].update(model=model,reasoning_effort=effort)
    path.write_text(json.dumps(config));monkeypatch.setattr(srv,'_chat_config_path',path)
    client,factory=mocked_sdk
    with TestClient(srv.app) as api:
        session=srv._chat_session
        assert api.get('/api/chat-config').json()['workspace_context'] is False
        factory.assert_not_called()
        answer=api.post('/api/chat',json={'question':'Explain public factor concepts','consent_to_send':True})
        assert answer.status_code==200 and answer.json()['answer']=='Mock public answer'
        request=client.responses.create.call_args.kwargs
        assert request['input']==[{'role':'user','content':'Explain public factor concepts'}]
        assert 'PRIVATE_WORKSPACE_SENTINEL' not in json.dumps(request,default=str)
        assert request['model']==model and request['reasoning']=={'effort':effort}
        assert request['max_output_tokens']==512 and request['store'] is False
        assert request['tools']==[] and request['parallel_tool_calls'] is False and request['truncation']=='disabled'
        assert session.runtime.calls==1 and session.runtime.output_tokens==8
        assert factory.call_args.kwargs['max_retries']==0
        client.chat.completions.create.assert_not_called()
    client.close.assert_called_once()


def test_questions_share_one_call_budget_and_are_not_written_to_artifacts(tmp_path,monkeypatch,mocked_sdk):
    path,config=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    client,factory=mocked_sdk
    with TestClient(srv.app) as api:
        for question in ('Question one not saved','Question two not saved'):
            assert api.post('/api/chat',json={'question':question,'consent_to_send':True}).status_code==200
        assert api.post('/api/chat',json={'question':'Over budget','consent_to_send':True}).status_code==410
        assert api.get('/api/chat-config').json()['enabled'] is False
    assert factory.call_count==1 and client.responses.create.call_count==2
    for artifact in Path(config['run_directory']).iterdir():
        text=artifact.read_text()
        assert 'Question one not saved' not in text and 'Question two not saved' not in text
        assert 'Mock public answer' not in text and 'mock-api-key-never-send-as-question' not in text
    assert json.loads((Path(config['run_directory'])/'usage.json').read_text())['calls']==2
    assert json.loads((Path(config['run_directory'])/'intent.json').read_text())['status']=='configured'


@pytest.mark.parametrize('body',[
    {'question':'Q','consent_to_send':False},{'question':'Q','consent_to_send':1},
    {'question':'Q'},{'question':'Q','consent_to_send':True,'include_workspace':True},
    {'question':'Q','consent_to_send':True,'model':'gpt-4.1-mini'},
    ['request-body-sentinel'],None,
])
def test_invalid_requests_and_context_or_model_overrides_never_construct_provider(tmp_path,monkeypatch,mocked_sdk,body):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    client,factory=mocked_sdk
    with TestClient(srv.app) as api:
        response=api.post('/api/chat',content=json.dumps(body),headers={'Content-Type':'application/json'})
        assert response.status_code==400 and 'request-body-sentinel' not in response.text
    factory.assert_not_called();client.responses.create.assert_not_called()


@pytest.mark.parametrize('question,code',[('',400),(123,400),('🙂'*501,413),('mock-api-key-never-send-as-question',400)])
def test_bad_questions_fail_before_client_creation(tmp_path,monkeypatch,mocked_sdk,question,code):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    client,factory=mocked_sdk
    with TestClient(srv.app) as api:
        response=api.post('/api/chat',json={'question':question,'consent_to_send':True})
        assert response.status_code==code and 'mock-api-key-never-send-as-question' not in response.text
    factory.assert_not_called();client.responses.create.assert_not_called()


@pytest.mark.parametrize('change',[
    lambda c:c['llm'].pop('max_calls'),lambda c:c['llm'].update(model=None),
    lambda c:c['llm'].update(reasoning_effort='low'),lambda c:c['llm'].update(deadline_utc=None),
    lambda c:c.update(data_scope='workspace'),lambda c:c.update(api_key='config-secret-sentinel'),
    lambda c:c.update(base_url='https://user:config-secret-sentinel@mock.invalid/v1'),
])
def test_incomplete_or_private_config_disables_chat_but_keeps_the_viewer(tmp_path,monkeypatch,mocked_sdk,change):
    path,config=config_file(tmp_path);change(config);path.write_text(json.dumps(config))
    monkeypatch.setattr(srv,'_chat_config_path',path)
    client,factory=mocked_sdk
    with TestClient(srv.app) as api:
        result=api.get('/api/chat-config')
        assert result.json()['enabled'] is False and 'config-secret-sentinel' not in result.text
        assert api.get('/api/health').status_code==200
    factory.assert_not_called();client.responses.create.assert_not_called()


def test_existing_chat_run_cannot_reset_the_budget_on_viewer_restart(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    with TestClient(srv.app) as api:assert api.get('/api/chat-config').json()['enabled'] is True
    with TestClient(srv.app) as api:assert api.get('/api/chat-config').json()['enabled'] is False
    mocked_sdk[1].assert_not_called()


def test_elapsed_time_and_endpoint_changes_block_requests_before_provider_creation(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    with TestClient(srv.app) as api:
        monkeypatch.setenv('OPENAI_BASE_URL','https://different.invalid/v1')
        assert api.post('/api/chat',json={'question':'Q','consent_to_send':True}).status_code==503
        srv._chat_session.runtime.started-=601
        assert api.post('/api/chat',json={'question':'Q','consent_to_send':True}).status_code==410
    mocked_sdk[1].assert_not_called()


def test_gateway_error_is_redacted_and_not_retried(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    client,_=mocked_sdk;client.responses.create.side_effect=RuntimeError('SDK_BODY_SECRET_SENTINEL')
    with TestClient(srv.app) as api:
        response=api.post('/api/chat',json={'question':'Public question not saved','consent_to_send':True})
        assert response.status_code==502 and 'SDK_BODY_SECRET_SENTINEL' not in response.text
    assert client.responses.create.call_count==1
    for artifact in (tmp_path/'chat-run').iterdir():assert 'SDK_BODY_SECRET_SENTINEL' not in artifact.read_text()


def test_slow_question_does_not_block_health_or_queue_another_request(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    started=threading.Event();release=threading.Event();client,_=mocked_sdk
    def slow_response(**kwargs):
        started.set();assert release.wait(timeout=5);return stream_answer()
    client.responses.create.side_effect=slow_response
    with TestClient(srv.app) as api:
        with ThreadPoolExecutor() as pool:
            first=pool.submit(api.post,'/api/chat',json={'question':'First','consent_to_send':True})
            try:
                assert started.wait(timeout=2)
                assert api.get('/api/health').status_code==200
                assert api.post('/api/chat',json={'question':'Second','consent_to_send':True}).status_code==429
            finally:release.set()
            assert first.result(timeout=5).status_code==200
    assert client.responses.create.call_count==1


def test_existing_runtime_factory_does_not_reset_elapsed_time_or_counters(tmp_path,monkeypatch,mocked_sdk):
    from alpha_lab.client import get_provider
    path,_=config_file(tmp_path);session=PublicChatSession(path)
    runtime=session.runtime;runtime.calls=1;runtime.started-=10
    started=runtime.started
    provider=get_provider('openai',runtime=runtime)
    assert provider.runtime is runtime and runtime.calls==1 and runtime.started==started
    with pytest.raises(ValueError):get_provider('openai',runtime=runtime,runtime_settings=runtime.settings)
    provider.close();session.close()


def test_credential_echo_in_a_model_answer_is_not_exposed(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    client,_=mocked_sdk
    client.responses.create.side_effect=lambda **kwargs:stream_answer('mock-api-key-never-send-as-question')
    with TestClient(srv.app) as api:
        response=api.post('/api/chat',json={'question':'Public question','consent_to_send':True})
        assert response.status_code==502 and 'mock-api-key-never-send-as-question' not in response.text


def test_reported_usage_blocks_the_next_question_before_the_sdk(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    with TestClient(srv.app) as api:
        srv._chat_session.runtime.input_tokens=200000
        assert api.get('/api/chat-config').json()['enabled'] is False
        assert api.post('/api/chat',json={'question':'Q','consent_to_send':True}).status_code==410
    mocked_sdk[1].assert_not_called()


def test_missing_key_disables_chat_before_sdk_construction(tmp_path,monkeypatch,mocked_sdk):
    path,_=config_file(tmp_path);monkeypatch.setattr(srv,'_chat_config_path',path)
    with TestClient(srv.app) as api:
        monkeypatch.setenv('OPENAI_API_KEY','')
        assert api.get('/api/chat-config').json()['enabled'] is False
        assert api.post('/api/chat',json={'question':'Q','consent_to_send':True}).status_code==503
    mocked_sdk[1].assert_not_called()


def test_impossible_token_budget_is_rejected_at_configuration(tmp_path,mocked_sdk):
    path,config=config_file(tmp_path);config['llm']['max_reserved_tokens']=100
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='fund one question'):PublicChatSession(path)
    mocked_sdk[1].assert_not_called()
