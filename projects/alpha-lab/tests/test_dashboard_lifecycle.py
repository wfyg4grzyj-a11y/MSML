"""Native viewer lifecycle and append-only log integrity; no LLM calls."""
import asyncio
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

import alpha_lab.server as srv


@pytest.fixture(autouse=True)
def dashboard_workspace(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace';(workspace/'logs').mkdir(parents=True)
    monkeypatch.setattr(srv,'_workspace',str(workspace))
    monkeypatch.setattr(srv,'manager',srv.DashboardState())
    srv.event_history.clear();srv.connected_websockets.clear();srv._adapter_cache.clear()
    yield workspace
    assert not srv._runtime_active


class ImmediateLoop:
    def is_closed(self):
        return False

    def call_soon_threadsafe(self,callback,event):
        callback(event)


def reader(workspace):
    queue=asyncio.Queue()
    return srv.LogTailer(str(workspace),ImmediateLoop(),queue),queue


def test_partial_multibyte_utf8_record_is_not_consumed_until_newline(dashboard_workspace):
    tailer,queue=reader(dashboard_workspace)
    path=dashboard_workspace/'logs'/'run.jsonl'
    payload=(json.dumps({'type':'status','detail':'完整🙂'},ensure_ascii=False)+'\n').encode('utf-8')
    cut=payload.index('🙂'.encode('utf-8'))+2
    path.write_bytes(payload[:cut]);tailer._read_new_lines(str(path))
    assert queue.empty() and tailer._offsets[str(path)]==0
    with path.open('ab') as stream:stream.write(payload[cut:-1])
    tailer._read_new_lines(str(path));assert queue.empty()
    with path.open('ab') as stream:stream.write(b'\n')
    tailer._read_new_lines(str(path))
    assert queue.get_nowait().to_dict()['detail']=='完整🙂'
    tailer._read_new_lines(str(path));assert queue.empty()


@pytest.mark.parametrize('mode',['truncate','replace','overwrite_larger'])
def test_rotation_and_truncation_restart_the_cursor(dashboard_workspace,mode):
    tailer,queue=reader(dashboard_workspace)
    path=dashboard_workspace/'logs'/'run.jsonl'
    path.write_text(json.dumps({'type':'status','detail':'old'*30})+'\n',encoding='utf-8')
    tailer._read_new_lines(str(path));queue.get_nowait()
    replacement={'type':'status','detail':'new'*(60 if mode=='overwrite_larger' else 1)}
    if mode=='replace':
        renamed=path.with_suffix('.old');path.replace(renamed)
    path.write_text(json.dumps(replacement)+'\n',encoding='utf-8')
    tailer._read_new_lines(str(path))
    assert queue.get_nowait().to_dict()==replacement
    assert queue.empty()


def test_malformed_nonobject_and_invalid_utf8_lines_do_not_break_the_reader(dashboard_workspace):
    tailer,queue=reader(dashboard_workspace)
    path=dashboard_workspace/'logs'/'run.jsonl'
    path.write_bytes(b'[]\nnull\n42\n"text"\n{bad}\n\xff\n\n{"type":"status"}\n')
    tailer._read_new_lines(str(path))
    assert queue.qsize()==1 and queue.get_nowait().to_dict()=={'type':'status'}


def test_history_is_bounded_across_files_and_preserves_cursor_for_new_events(dashboard_workspace,monkeypatch):
    tailer,queue=reader(dashboard_workspace)
    monkeypatch.setattr(srv,'MAX_EVENT_HISTORY',3)
    for name,timestamps in [('a',[1,5,3]),('b',[2,6,4])]:
        path=dashboard_workspace/'logs'/(name+'.jsonl')
        path.write_text(''.join(json.dumps({'type':'status','timestamp':t})+'\n' for t in timestamps),encoding='utf-8')
    srv._load_jsonl_history(tailer)
    assert [event['timestamp'] for event in srv.event_history]==[4,5,6]
    path=dashboard_workspace/'logs'/'a.jsonl'
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'type':'status','timestamp':7})+'\n')
    tailer._read_new_lines(str(path))
    assert queue.qsize()==1 and queue.get_nowait().to_dict()['timestamp']==7


def test_history_accepts_only_dicts_and_sorts_invalid_timestamps_safely(dashboard_workspace):
    tailer,_=reader(dashboard_workspace)
    path=dashboard_workspace/'logs'/'run.jsonl'
    path.write_text('[]\n'+json.dumps({'type':'status','timestamp':'bad'})+'\n'+
                    json.dumps({'type':'status','timestamp':float('inf')})+'\n'+
                    json.dumps({'type':'status','timestamp':2})+'\n',encoding='utf-8')
    srv._load_jsonl_history(tailer)
    assert len(srv.event_history)==3 and srv.event_history[-1]['timestamp']==2


def test_viewer_restart_replays_once_and_stops_owned_threads_and_tasks(dashboard_workspace):
    path=dashboard_workspace/'logs'/'run.jsonl'
    path.write_text(json.dumps({'type':'status','timestamp':1,'status':'saved'})+'\n',encoding='utf-8')
    for _ in range(2):
        with TestClient(srv.app) as client:
            owners=[srv.log_tailer,srv.file_watcher]
            tasks=[srv._broadcast_task,srv._status_report_task]
            assert client.get('/api/health').status_code==200
            with client.websocket_connect('/ws') as websocket:
                assert websocket.receive_json()['status']=='saved'
            assert sum(event.get('status')=='saved' for event in srv.event_history)==1
        assert all(not owner._thread.is_alive() for owner in owners)
        assert all(task.done() for task in tasks)
        assert srv.file_watcher is None and srv.log_tailer is None
        assert srv.manager._queue is None and srv.manager._loop is None
        assert not srv.event_history and not srv.connected_websockets


def test_startup_failure_cleans_already_started_resources(dashboard_workspace,monkeypatch):
    owned=[];tasks=[]
    def fail_start(self):
        owned.append(srv.log_tailer)
        tasks.extend([srv._broadcast_task,srv._status_report_task])
        raise RuntimeError('Synthetic watcher startup failure')
    monkeypatch.setattr(srv.FileWatcher,'start',fail_start)
    with pytest.raises(RuntimeError,match='Synthetic watcher'):
        with TestClient(srv.app):pass
    assert owned and not owned[0]._thread.is_alive()
    assert all(task.done() for task in tasks)
    assert srv._broadcast_task is None and srv._status_report_task is None
    assert srv.manager._queue is None


def test_overlapping_lifespan_is_rejected_without_stopping_the_active_viewer(dashboard_workspace):
    with TestClient(srv.app) as first:
        owners=[srv.log_tailer,srv.file_watcher]
        with pytest.raises(RuntimeError,match='active lifespan'):
            with TestClient(srv.app):pass
        assert first.get('/api/health').status_code==200
        assert all(owner._thread.is_alive() for owner in owners)


@pytest.mark.parametrize('kind',['LogTailer','FileWatcher'])
def test_start_and_stop_are_idempotent(dashboard_workspace,kind):
    owner=getattr(srv,kind)(str(dashboard_workspace),ImmediateLoop(),asyncio.Queue())
    try:
        owner.start();first=owner._thread
        owner.start();assert owner._thread is first
        owner.stop();owner.stop()
        assert not first.is_alive()
    finally:
        owner.stop()


def test_pending_manual_status_task_is_cancelled_on_shutdown(dashboard_workspace,monkeypatch):
    started=threading.Event();cancelled=threading.Event()
    async def slow_report():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(srv,'_generate_and_broadcast_status',slow_report)
    with TestClient(srv.app) as client:
        with client.websocket_connect('/ws') as websocket:
            websocket.send_json({'type':'generate_status'})
            assert started.wait(timeout=2)
    assert cancelled.is_set() and not srv._manual_status_tasks


def test_manually_generated_status_reports_use_the_same_history_cap(dashboard_workspace,monkeypatch):
    from unittest.mock import patch
    monkeypatch.setattr(srv,'MAX_EVENT_HISTORY',3)
    async def generate():
        with patch('alpha_lab.output_generator.OutputGenerator') as generator:
            generator.return_value.generate_status_report.return_value={'check':'synthetic'}
            for _ in range(10):await srv._generate_and_broadcast_status()
    asyncio.run(generate())
    assert len(srv.event_history)<=3


@pytest.mark.parametrize('status,worker',[('queued',None),('running',None),('to_implement','worker-1'),('finished','reviewer-1')])
def test_viewer_cannot_mark_active_work_cancelled_without_stopping_its_owner(dashboard_workspace,status,worker):
    from alpha_lab.experiment_db import ExperimentDB
    board=ExperimentDB(str(dashboard_workspace/'experiments.db'))
    identifier=board.create('active_fixture','Synthetic','No process started','{}')
    board.update_status(identifier,status)
    if worker is not None:board.assign_worker(identifier,worker)
    with TestClient(srv.app) as client:
        response=client.post(f'/api/experiments/{identifier}/cancel')
        assert response.status_code==409
    assert board.get(identifier).status==status


def test_viewer_can_prune_unassigned_work_without_changing_results(dashboard_workspace):
    from alpha_lab.experiment_db import ExperimentDB
    board=ExperimentDB(str(dashboard_workspace/'experiments.db'))
    identifier=board.create('idle_fixture','Synthetic','No process started','{}')
    board.set_results(identifier,'{"synthetic_evidence":true}')
    with TestClient(srv.app) as client:
        assert client.post(f'/api/experiments/{identifier}/cancel').status_code==200
    experiment=board.get(identifier)
    assert experiment.status=='cancelled' and experiment.results_json=='{"synthetic_evidence":true}'


@pytest.mark.parametrize('change',['queued','worker'])
def test_claim_between_lookup_and_cancellation_is_not_overwritten(dashboard_workspace,monkeypatch,change):
    from alpha_lab.experiment_db import ExperimentDB
    board=ExperimentDB(str(dashboard_workspace/'experiments.db'))
    identifier=board.create('race_fixture','Synthetic','No process started','{}')
    with TestClient(srv.app) as client:
        original_get=srv.manager.db.get
        def stale_lookup(exp_id):
            old=original_get(exp_id)
            if change=='queued':board.update_status(exp_id,'queued')
            else:board.assign_worker(exp_id,'concurrent-worker')
            return old
        monkeypatch.setattr(srv.manager.db,'get',stale_lookup)
        assert client.post(f'/api/experiments/{identifier}/cancel').status_code==409
    current=board.get(identifier)
    assert current.status==('queued' if change=='queued' else 'to_implement')
    if change=='worker':assert current.worker_id=='concurrent-worker'


def test_atomic_pruning_rejects_missing_and_already_cancelled_rows(dashboard_workspace):
    from alpha_lab.experiment_db import ExperimentDB
    board=ExperimentDB(str(dashboard_workspace/'experiments.db'))
    identifier=board.create('idle_atomic_fixture','Synthetic','No process started','{}')
    assert board.cancel_if_idle(identifier) is True
    assert board.cancel_if_idle(identifier) is False
    assert board.cancel_if_idle(identifier+1) is False
