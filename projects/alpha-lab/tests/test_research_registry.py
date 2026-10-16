"""Independent-process admission, immutable lineage and crash/publication gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
pytest.importorskip('numpy')
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.contracts import ResearchError, parse, read_json
from alpha_lab.research.registry import Registry, initialize_registry
from alpha_lab.research.runtime import child_environment
from alpha_lab.research.signal_contracts import Family, Hypothesis, SignalSpec, baseline_recipes
from alpha_lab.research import signal_workflow as flow
from tests.test_signal_lab import signal_bundle, request, registration


def setup_ledger(tmp_path, spec):
    path = tmp_path/'ledger.sqlite'; initialize_registry(path)
    with Registry(path) as db: registration(db, spec)
    source = tmp_path/'spec.json'; atomic_json(source, spec.model_dump())
    return path, source


def test_immutable_budgets_parent_lineage_and_cross_family_dedup(signal_bundle, tmp_path):
    spec = flow.resolve_signal(signal_bundle, request())
    path = tmp_path/'ledger.sqlite'; initialize_registry(path)
    with Registry(path) as db:
        trial = registration(db,spec,max_trials=1)
        with pytest.raises(ResearchError,match='immutable'):registration(db,spec,max_trials=2)
        changed = parse(SignalSpec,spec.model_dump() | {'hypothesis_id':'quality', 'recipe':baseline_recipes()[1]})
        with pytest.raises(ResearchError,match='unique-trial'):registration(db,changed,max_trials=1,parent=trial)
        new_family = parse(SignalSpec,spec.model_dump() | {'family_id':'other-family','hypothesis_id':'other-name'})
        assert registration(db,new_family,max_trials=1) == trial
        first = db.begin(spec,tmp_path/'first')
        with pytest.raises(ResearchError,match='active'):db.begin(new_family,tmp_path/'new-family')
        db.finish(first,failure='execution_failure')
        second = db.begin(new_family,tmp_path/'retry',retry_of=first)
        db.finish(second,failure='execution_failure')
        with pytest.raises(ResearchError,match='budget'):db.begin(spec,tmp_path/'third',retry_of=second)
        history=db.snapshot()
        assert len(history['trials']) == 1 and len(history['registrations']) == 2
        assert history['attempts'][1]['retry_of'] == first
        assert len(history['exposures']) == 2


def test_parent_registration_is_immutable_and_must_exist(signal_bundle,tmp_path):
    spec = flow.resolve_signal(signal_bundle,request())
    path=tmp_path/'ledger.sqlite'; initialize_registry(path)
    with Registry(path) as db:
        first=registration(db,spec)
        changed=parse(SignalSpec,spec.model_dump() | {'hypothesis_id':'quality','recipe':baseline_recipes()[1]})
        with pytest.raises(ResearchError,match='Parent'):registration(db,changed,parent='0'*64)
        registration(db,changed,parent=first)
        with pytest.raises(ResearchError,match='parent'):registration(db,changed)


def test_claim_cannot_mislabel_registered_family(signal_bundle,tmp_path):
    spec=flow.resolve_signal(signal_bundle,request())
    ledger,_=setup_ledger(tmp_path,spec)
    mislabeled=parse(SignalSpec,spec.model_dump() | {'family_id':'unregistered-family'})
    with Registry(ledger) as db:
        with pytest.raises(ResearchError,match='Register'):db.begin(mislabeled,tmp_path/'false-family')
        assert not db.snapshot()['attempts']


def test_two_processes_cannot_claim_one_trial(signal_bundle,tmp_path):
    spec=flow.resolve_signal(signal_bundle,request())
    ledger,source=setup_ledger(tmp_path,spec)
    code = '''
import sys
from alpha_lab.research.contracts import parse,read_json,ResearchError
from alpha_lab.research.signal_contracts import SignalSpec
from alpha_lab.research.registry import Registry
try:
    with Registry(sys.argv[1]) as db: db.begin(parse(SignalSpec,read_json(sys.argv[2])),sys.argv[3])
    print('claimed')
except ResearchError: print('blocked')
'''
    def invoke(n):
        return subprocess.run([sys.executable,'-c',code,str(ledger),str(source),str(tmp_path/f'run{n}')],
                              env=child_environment(spec),capture_output=True,text=True,timeout=20,check=True).stdout.strip()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(invoke,range(2)))
    assert sorted(results)==['blocked','claimed']
    with Registry(ledger,readonly=True) as db:
        assert len(db.snapshot()['attempts'])==1
        assert db.snapshot()['exposures'][0]['kind'].endswith('unresolved')


def test_concurrent_registration_cannot_exceed_family_budget(signal_bundle,tmp_path):
    spec=flow.resolve_signal(signal_bundle,request())
    ledger=tmp_path/'ledger.sqlite';initialize_registry(ledger)
    first=tmp_path/'one.json';atomic_json(first,spec.model_dump())
    other=parse(SignalSpec,spec.model_dump() | {'hypothesis_id':'quality','recipe':baseline_recipes()[1]})
    second=tmp_path/'two.json';atomic_json(second,other.model_dump())
    code='''
import sys
from alpha_lab.research.contracts import parse,read_json,ResearchError
from alpha_lab.research.signal_contracts import SignalSpec,Family,Hypothesis
from alpha_lab.research.registry import Registry
s=parse(SignalSpec,read_json(sys.argv[2]))
f=Family(family_id=s.family_id,premise='A fixed shared synthetic family',max_unique_trials=1,max_attempts_per_trial=2)
h=Hypothesis(hypothesis_id=s.hypothesis_id,family_id=s.family_id,premise='A fixed synthetic hypothesis',tests=['coverage'])
try:
    with Registry(sys.argv[1]) as db: db.register(f,h,s)
    print('registered')
except ResearchError: print('blocked')
'''
    def invoke(source):
        return subprocess.run([sys.executable,'-c',code,str(ledger),str(source)],env=child_environment(spec),
                              capture_output=True,text=True,timeout=20,check=True).stdout.strip()
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(invoke,[first,second]))
    assert sorted(results)==['blocked','registered']
    with Registry(ledger,readonly=True) as db:assert len(db.snapshot()['trials'])==1


def test_completed_files_survive_database_publication_gap_and_reconcile(signal_bundle,tmp_path,monkeypatch):
    spec=flow.resolve_signal(signal_bundle,request())
    ledger,source=setup_ledger(tmp_path,spec)
    original=Registry.finish
    def fail_once(self,attempt,**values):
        if values.get('complete') is not None:raise OSError('synthetic DB interruption')
        return original(self,attempt,**values)
    monkeypatch.setattr(Registry,'finish',fail_once)
    output=tmp_path/'run'
    with pytest.raises(ResearchError,match='reconciliation'):flow.run_signal(signal_bundle,source,output,tmp_path/'cache',ledger)
    evidence=flow.inspect_signal(output)
    assert evidence['artifacts_verified']
    with Registry(ledger) as db:
        assert db.snapshot()['attempts'][0]['status']=='running'
        with pytest.raises(ResearchError,match='active'):db.begin(spec,tmp_path/'duplicate')
    monkeypatch.setattr(Registry,'finish',original)
    assert flow.reconcile_signal(ledger,output)['result_hash']==evidence['result_hash']
    assert flow.reconcile_signal(ledger,output)['result_hash']==evidence['result_hash']
    with Registry(ledger,readonly=True) as db:assert db.snapshot()['attempts'][0]['status']=='completed'


def test_marker_disk_failure_is_not_completion(signal_bundle,tmp_path,monkeypatch):
    spec=flow.resolve_signal(signal_bundle,request())
    ledger,source=setup_ledger(tmp_path,spec)
    original=flow.atomic_json
    def interrupted(path,value):
        if Path(path).name=='COMPLETE.json':raise OSError('synthetic disk failure')
        return original(path,value)
    monkeypatch.setattr(flow,'atomic_json',interrupted)
    with pytest.raises(ResearchError,match='publication_failure'):flow.run_signal(signal_bundle,source,tmp_path/'run',tmp_path/'cache',ledger)
    assert not (tmp_path/'run/COMPLETE.json').exists()
    assert flow.inspect_signal(tmp_path/'run')['effective_status']=='failed'


@pytest.mark.parametrize('guard',['timeout','memory_limit'])
def test_signal_resource_guards_terminate_owned_process(signal_bundle,tmp_path,monkeypatch,guard):
    spec=flow.resolve_signal(signal_bundle,request(timeout_seconds=1 if guard=='timeout' else 45,
                                                  memory_limit_mb=128 if guard=='memory_limit' else 1024))
    ledger,source=setup_ledger(tmp_path,spec)
    class Waiting:
        pid=os.getpid()
        stopped=False
        def poll(self):return 1 if self.stopped else None
        def terminate_tree(self):self.stopped=True
        def close(self):pass
    proc=Waiting()
    monkeypatch.setattr(flow,'start_process',lambda *a,**k:proc)
    if guard=='timeout':
        times=iter([0.,2.]);monkeypatch.setattr(flow.time,'monotonic',lambda:next(times))
    else:
        class Memory:
            rss=2*1024**3
        class Root:
            def memory_info(self):return Memory()
            def children(self,recursive):return []
            def is_running(self):return True
        monkeypatch.setattr(flow.psutil,'Process',lambda pid:Root())
    with pytest.raises(ResearchError,match=guard):flow.run_signal(signal_bundle,source,tmp_path/'run',tmp_path/'cache',ledger)
    assert proc.stopped
    assert read_json(tmp_path/'run/run.json')['failure']==guard


def test_output_cap_never_publishes_completion(signal_bundle,tmp_path):
    spec=flow.resolve_signal(signal_bundle,request(max_output_bytes=1024))
    ledger,source=setup_ledger(tmp_path,spec)
    with pytest.raises(ResearchError,match='output_limit'):flow.run_signal(signal_bundle,source,tmp_path/'run',tmp_path/'cache',ledger)
    assert not (tmp_path/'run/COMPLETE.json').exists()


def test_comparison_compatible_studies_keeps_coverage_visible(signal_bundle,tmp_path):
    base=flow.resolve_signal(signal_bundle,request())
    other=flow.resolve_signal(signal_bundle,request(hypothesis_id='quality',recipe=baseline_recipes()[1]))
    ledger=tmp_path/'ledger.sqlite';initialize_registry(ledger)
    paths=[]
    for spec in (base,other):
        source=tmp_path/(spec.hypothesis_id+'.json');atomic_json(source,spec.model_dump())
        with Registry(ledger) as db:registration(db,spec)
        paths.append(tmp_path/spec.hypothesis_id)
        flow.run_signal(signal_bundle,source,paths[-1],tmp_path/'cache',ledger)
    result=flow.compare_signals(paths)
    assert result['selection']=='none' and result['outcome_filter_for_score_overlap'] is False
    assert all(row['scored_a']==40 and row['scored_b']==35 for row in result['score_comparisons'][0]['sessions'])


def test_unknown_registry_is_not_overwritten(tmp_path):
    path=tmp_path/'unknown.sqlite';path.write_bytes(b'unchanged')
    with pytest.raises(FileExistsError):initialize_registry(path)
    assert path.read_bytes()==b'unchanged'


def test_prior_knowledge_is_hash_verified_and_cannot_be_reset(signal_bundle,tmp_path):
    from alpha_lab.research.contracts import file_hash
    from alpha_lab.research.signal_contracts import PriorExposure
    spec=flow.resolve_signal(signal_bundle,request())
    ledger,source=setup_ledger(tmp_path,spec)
    exposure=PriorExposure(exposure_id='already-observed-holdout',start='2023-01-01',end='2025-12-31',
                           evidence_sha256=file_hash(source),description='Synthetic prior exposure evidence only',
                           kind='previously_observed_legacy_evaluation')
    with Registry(ledger) as db:
        db.note_prior_exposure(exposure,source)
        db.note_prior_exposure(exposure,source)
        assert len(db.snapshot()['prior_exposures'])==1
        altered=parse(PriorExposure,exposure.model_dump() | {'end':'2024-12-31'})
        with pytest.raises(ResearchError,match='immutable'):db.note_prior_exposure(altered,source)
        source.write_text('changed')
        with pytest.raises(ResearchError,match='hash'):db.note_prior_exposure(exposure,source)


def test_feature_cache_is_exclusive_across_processes(signal_bundle,tmp_path):
    from alpha_lab.workspace_lock import WorkspaceLock
    from alpha_lab.research.features import feature_identity
    from alpha_lab.research.bundles import load_bundle
    spec=flow.resolve_signal(signal_bundle,request())
    cache=tmp_path/'cache'
    key=feature_identity(load_bundle(signal_bundle),'momentum_12_1')[0]
    code='''
import sys
from alpha_lab.research.features import cached_feature
from alpha_lab.research.bundles import load_bundle
from alpha_lab.research.contracts import ResearchError
try:
    cached_feature(sys.argv[1],load_bundle(sys.argv[1]),'momentum_12_1',sys.argv[2],2000000)
    print('unexpected')
except ResearchError: print('blocked')
'''
    with WorkspaceLock(cache/'locks'/key):
        result=subprocess.run([sys.executable,'-c',code,str(signal_bundle),str(cache)],env=child_environment(spec),
                              capture_output=True,text=True,timeout=20,check=True)
    assert result.stdout.strip()=='blocked' and not (cache/key).exists()


def test_cli_check_and_inspect_do_not_submit_a_worker(signal_bundle,tmp_path,monkeypatch,capsys):
    from alpha_lab.research import cli
    spec=flow.resolve_signal(signal_bundle,request())
    path=tmp_path/'spec.json';atomic_json(path,spec.model_dump())
    monkeypatch.setattr(flow,'start_process',lambda *a,**k:pytest.fail('Read-only operation submitted a worker'))
    assert cli.main(['signal-check','--bundle',str(signal_bundle),'--spec',str(path)])==0
    assert json.loads(capsys.readouterr().out)['llm_calls']==0
