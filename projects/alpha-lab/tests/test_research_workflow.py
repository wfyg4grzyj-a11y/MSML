"""P1 identity, causal materialization, actual child execution and failure gates."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')
duckdb = pytest.importorskip('duckdb')

from alpha_lab.equity_calendar import session_schedule
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import MarketNeutralBacktest
from alpha_lab.research import cli, workflow
from alpha_lab.research.bundles import artifact, load_bundle, pin_panel
from alpha_lab.research.contracts import (DEVELOPMENT_END, ResearchError, ResearchSpec, digest,
                                         file_hash, parse, read_json)
from alpha_lab.research.references import accounting_reference
from alpha_lab.research.runtime import child_environment


@pytest.fixture
def source_panel(tmp_path):
    root = tmp_path/'source';root.mkdir()
    calendar, meta = session_schedule('2012-01-01', '2023-01-05')
    dates = pd.DatetimeIndex(calendar.date)
    tickers = [f'S{i:02d}' for i in range(40)] + ['FUTURE']
    features = pd.DataFrame({'ticker': tickers[:40], 'date': pd.Timestamp('2014-01-31'),
                             'filing_date': pd.Timestamp('2014-01-20'),
                             'beta': np.linspace(.5, 1.5, 40), 'book_yield': np.sin(np.arange(40))})
    future = features.iloc[:1].assign(ticker='FUTURE', date=pd.Timestamp('2023-01-03'), book_yield=999999.)
    features = pd.concat([features, future], ignore_index=True)
    actions = pd.DataFrame({'ticker': ['S00', 'FUTURE'], 'date': pd.to_datetime(['2014-02-05', '2023-01-03']),
                            'action': ['split', 'delisted'], 'value': [2., 0.], 'contraticker': ['', '']})
    metadata = pd.DataFrame({'ticker': tickers, 'permaticker': list(range(1, 42)), 'table': ['SEP']*41})
    with duckdb.connect(':memory:') as db:
        db.from_df(features).write_parquet(str(root/'features.parquet'))
        db.from_df(actions).write_parquet(str(root/'actions.parquet'))
        db.from_df(metadata).write_parquet(str(root/'metadata.parquet'))
    returns = np.zeros((len(dates), len(tickers)), dtype=np.float32)
    returns[dates.get_loc('2014-02-03'), :40] = np.sin(np.arange(40))*.1
    returns[dates.get_loc('2014-02-04'), :40] = np.sin(np.arange(40))*.01
    returns[dates >= '2023-01-01'] = 12345.
    np.savez_compressed(root/'returns.npz', dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                        tickers=np.asarray(tickers, dtype='U32'), returns=returns,
                        price_observed=np.ones_like(returns, dtype=bool), spy=np.linspace(-.01, .01, len(dates)))
    calendar.to_csv(root/'calendar.csv', index=False)
    sources = {name: {'version': 'a'*32, 'parquet_sha256': file_hash(root/'metadata.parquet'),
                      'path': str(root/'metadata.parquet')}
               for name in ('stocks', 'funds', 'daily', 'fundamentals', 'tickers', 'actions')}
    atomic_json(root/'sources.json', sources)
    manifest = {'schema_version': 4, 'start': '2014-01-01', 'end': '2023-01-05', 'calendar': meta,
                'availability_assumptions': {'point_in_time_attested': False},
                'files': {name: file_hash(root/name) for name in
                          ('sources.json', 'features.parquet', 'actions.parquet', 'returns.npz', 'calendar.csv')}}
    atomic_json(root/'panel.json', manifest)
    return root


@pytest.fixture
def bundle(source_panel, tmp_path):
    path = tmp_path/'bundle'
    pin_panel(source_panel, path, end='2014-02-14')
    return path


def spec_for(bundle, tmp_path, **changes):
    values = dict(hypothesis_id='legacy-baseline', family_id='compatibility', title='Synthetic value baseline',
                  strategy={'name': 'value', 'factors': {'book_yield': 1.}}, cost_bps=10.,
                  borrow_bps_annual=300., terminal_stress=0., seed=42, threads=1,
                  timeout_seconds=45, max_output_bytes=2_000_000, memory_limit_mb=1024)
    values.update(changes)
    spec = workflow.make_spec(bundle, **values)
    path = tmp_path/'spec.json'
    atomic_json(path, spec.model_dump())
    return path, spec


def test_materialized_files_exclude_holdout_and_future_only_names(source_panel, bundle):
    contract = load_bundle(bundle)
    assert contract.end == '2014-02-14' and contract.securities == 40
    assert str(source_panel) not in (bundle/'sources.json').read_text()
    assert all(set(v) == {'version', 'sha256'} for v in read_json(bundle/'sources.json').values())
    with np.load(bundle/'returns.npz', allow_pickle=False) as data:
        assert max(data['dates']) <= contract.end and 'FUTURE' not in data['tickers']
        assert np.max(data['returns']) < 1
    for name in ('features.parquet', 'actions.parquet'):
        frame = duckdb.read_parquet(str(bundle/name)).df()
        assert frame.date.max() <= pd.Timestamp(contract.end) and 'FUTURE' not in frame.ticker.values
    with pytest.raises((ResearchError, FileExistsError)):
        pin_panel(source_panel, bundle, end='2014-02-14')


@pytest.mark.parametrize('end', ['2023-01-01', '2025-12-31', '2013-12-31', '2022-12-31T00:00Z'])
def test_protected_or_invalid_pin_bounds_are_rejected(source_panel, tmp_path, end):
    with pytest.raises(ResearchError):pin_panel(source_panel, tmp_path/'forbidden', end=end)
    assert not (tmp_path/'forbidden').exists()


def test_corrupt_input_and_incomplete_bundle_are_rejected(bundle):
    (bundle/'features.parquet').write_bytes(b'corrupt')
    with pytest.raises(ResearchError, match='hash/size'):load_bundle(bundle)
    (bundle/'COMPLETE.json').unlink()
    with pytest.raises(ResearchError):load_bundle(bundle)


def test_future_revision_changes_identity_but_not_prior_materialized_values(source_panel, bundle, tmp_path):
    with np.load(source_panel/'returns.npz', allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    raw['returns'][pd.DatetimeIndex(raw['dates']) > DEVELOPMENT_END] = 987654.
    np.savez_compressed(source_panel/'returns.npz', **raw)
    manifest = read_json(source_panel/'panel.json')
    manifest['files']['returns.npz'] = file_hash(source_panel/'returns.npz')
    atomic_json(source_panel/'panel.json', manifest)
    revised = tmp_path/'revised'
    pin_panel(source_panel, revised, end='2014-02-14')
    assert load_bundle(revised).bundle_id != load_bundle(bundle).bundle_id
    with np.load(bundle/'returns.npz') as a, np.load(revised/'returns.npz') as b:
        for name in a.files:np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize('change', [{'API_KEY': 'never-print-this-private-value'}, {'seed': True},
                                   {'cost_bps': float('nan')}, {'threads': 0}, {'timeout_seconds': '45'},
                                   {'methodology': 'new_hidden_method'}, {'cash_policy': 'new_cash'}])
def test_strict_specs_refuse_coercion_and_silent_policies(bundle, tmp_path, change):
    _, spec = spec_for(bundle, tmp_path)
    with pytest.raises(ResearchError) as error:parse(ResearchSpec, spec.model_dump() | change)
    assert 'never-print-this-private-value' not in str(error.value)


def test_identity_excludes_display_but_includes_cost_and_source(bundle, tmp_path):
    _, spec = spec_for(bundle, tmp_path)
    renamed = parse(ResearchSpec, spec.model_dump() | {'title': 'Another label',
                                                    'strategy': {'name': 'Other name', 'factors': {'book_yield': 1.}}})
    assert spec.semantic_id() == renamed.semantic_id()
    relabeled = parse(ResearchSpec, spec.model_dump() | {'hypothesis_id': 'renamed', 'family_id': 'renamed'})
    assert relabeled.semantic_id() == spec.semantic_id()
    changed = parse(ResearchSpec, spec.model_dump() | {'cost_bps': 25.})
    assert changed.semantic_id() != spec.semantic_id()
    with pytest.raises(ResearchError, match='identify'):
        workflow.compatible(parse(ResearchSpec, spec.model_dump() | {'bundle_id': '0'*64}), load_bundle(bundle))


def test_filtered_worker_environment_never_inherits_keys_or_proxy(bundle, tmp_path, monkeypatch):
    _, spec = spec_for(bundle, tmp_path)
    for name in ('OPENAI_API_KEY', 'ARRAYS_API_KEY', 'SHARADAR_API_KEY', 'TIINGO_API_KEY', 'AWS_SECRET_ACCESS_KEY',
                 'HTTPS_PROXY', 'PRIVATE_TOKEN', 'PYTHONSTARTUP', 'PYTHONPATH'):
        monkeypatch.setenv(name, 'synthetic-private')
    env = child_environment(spec)
    assert 'synthetic-private' not in env.values() and env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['PYTHONHASHSEED'] == '42' and env['OPENBLAS_NUM_THREADS'] == '1'


def test_actual_child_matches_unchanged_evaluator_and_inspection_is_read_only(source_panel, bundle, tmp_path, monkeypatch):
    spec_path, spec = spec_for(bundle, tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-private')
    monkeypatch.setenv('HTTPS_PROXY', 'http://unreachable.invalid')
    run = tmp_path/'run'
    record = workflow.run_research(bundle, spec_path, run)
    assert record.status == 'completed' and record.llm_calls == 0
    old, old_daily = MarketNeutralBacktest(source_panel).evaluate(spec.strategy.model_dump(), cost_bps=10., borrow_bps=300.)
    expected = old_daily.loc[old_daily.date <= '2014-02-14'].reset_index(drop=True)
    actual = pd.read_csv(run/'daily.csv', parse_dates=['date'])
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-12, rtol=1e-12)
    assert actual.loc[actual.date == '2014-02-03', 'gross'].item() == 0
    assert actual.loc[actual.date == '2014-02-04', 'gross'].item() > 0
    assert read_json(run/'metrics.json')['legacy_result'].get('holdout') is None
    before = {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in run.iterdir() if p.is_file()}
    info = workflow.inspect_run(run)
    assert info['effective_status'] == 'completed' and info['artifacts_verified'] is True
    assert before == {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):workflow.run_research(bundle, spec_path, run)
    (run/'daily.csv').write_text('date,net\n2025-01-01,99\n')
    with pytest.raises(ResearchError):workflow.inspect_run(run)


class EmptyProcess:
    pid = 123
    def poll(self):return 0
    def wait(self):return 0
    def close(self):pass


def test_zero_exit_without_artifacts_is_failed_and_not_retried(bundle, tmp_path, monkeypatch):
    spec_path, _ = spec_for(bundle, tmp_path)
    calls=[]
    monkeypatch.setattr(workflow, 'start_process', lambda *a, **k: (calls.append(1) or EmptyProcess()))
    with pytest.raises(ResearchError, match='publication_failure'):
        workflow.run_research(bundle, spec_path, tmp_path/'run')
    assert len(calls) == 1 and not (tmp_path/'run'/'COMPLETE.json').exists()
    assert workflow.inspect_run(tmp_path/'run')['record']['failure_code'] == 'publication_failure'


def test_changed_code_fails_before_worker_admission(bundle, tmp_path, monkeypatch):
    spec_path, _ = spec_for(bundle, tmp_path)
    monkeypatch.setattr(workflow, 'implementation_identity', lambda: {})
    monkeypatch.setattr(workflow, 'start_process', lambda *a, **k: pytest.fail('No job for changed code'))
    with pytest.raises(ResearchError, match='input_validation'):
        workflow.run_research(bundle, spec_path, tmp_path/'run')
    assert workflow.inspect_run(tmp_path/'run')['effective_status'] == 'failed'


def test_interrupted_completion_publication_never_reports_success(bundle, tmp_path, monkeypatch):
    spec_path, _ = spec_for(bundle, tmp_path)
    original = workflow.atomic_json
    def fail_marker(path, value):
        if Path(path).name == 'COMPLETE.json':raise OSError('simulated disk failure')
        return original(path, value)
    monkeypatch.setattr(workflow, 'atomic_json', fail_marker)
    with pytest.raises(ResearchError, match='publication_failure'):
        workflow.run_research(bundle, spec_path, tmp_path/'run')
    info = workflow.inspect_run(tmp_path/'run')
    assert info['effective_status'] == 'failed' and info['artifacts_verified'] is False


def test_cap_refuses_even_successful_worker_output(bundle, tmp_path):
    spec_path, _ = spec_for(bundle, tmp_path, max_output_bytes=1024)
    with pytest.raises(ResearchError, match='output_limit'):
        workflow.run_research(bundle, spec_path, tmp_path/'run')
    assert workflow.inspect_run(tmp_path/'run')['record']['failure_code'] == 'output_limit'


def test_cli_check_and_inspect_never_execute(bundle, tmp_path, monkeypatch, capsys):
    spec_path, _ = spec_for(bundle, tmp_path)
    monkeypatch.setattr(workflow, 'start_process', lambda *a, **k: pytest.fail('Read-only command started work'))
    before = {p.name: p.stat().st_mtime_ns for p in bundle.iterdir()}
    assert cli.main(['check', '--bundle', str(bundle), '--spec', str(spec_path)]) == 0
    assert json.loads(capsys.readouterr().out)['llm_calls'] == 0
    assert before == {p.name: p.stat().st_mtime_ns for p in bundle.iterdir()}


def test_check_refuses_absent_feature_dependencies(bundle, tmp_path, capsys):
    spec_path, _ = spec_for(bundle, tmp_path, strategy={'name': 'missing', 'factors': {'fcf_yield': 1.}})
    assert cli.main(['check', '--bundle', str(bundle), '--spec', str(spec_path)]) == 1
    assert 'absent or nonnumeric' in capsys.readouterr().err


@pytest.mark.parametrize('guard', ['timeout', 'memory_limit'])
def test_resource_guards_stop_owned_process_and_preserve_failed_attempt(bundle, tmp_path, monkeypatch, guard):
    import os
    class WaitingProcess:
        pid = os.getpid()
        stopped = False
        def poll(self):return 1 if self.stopped else None
        def terminate_tree(self):self.stopped = True
        def close(self):pass
    proc = WaitingProcess()
    monkeypatch.setattr(workflow, 'start_process', lambda *a, **k: proc)
    if guard == 'timeout':
        clock = iter([0, 2])
        monkeypatch.setattr(workflow, 'time', SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None))
        spec_path, _ = spec_for(bundle, tmp_path, timeout_seconds=1)
    else:
        process = SimpleNamespace(children=lambda **k: [], is_running=lambda: True,
                                  memory_info=lambda: SimpleNamespace(rss=256*1024**2))
        monkeypatch.setattr(workflow.psutil, 'Process', lambda _: process)
        spec_path, _ = spec_for(bundle, tmp_path, memory_limit_mb=128)
    with pytest.raises(ResearchError, match=guard):workflow.run_research(bundle, spec_path, tmp_path/'run')
    assert proc.stopped and not (tmp_path/'run'/'COMPLETE.json').exists()
    assert workflow.inspect_run(tmp_path/'run')['record']['failure_code'] == guard


def test_resealed_hashes_cannot_hide_future_dates(bundle):
    from alpha_lab.research.contracts import BUNDLE_FILES, PANEL_FILES
    with np.load(bundle/'returns.npz', allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    raw['dates'][-1] = '2023-01-03'
    np.savez_compressed(bundle/'returns.npz', **raw)
    panel = read_json(bundle/'panel.json')
    panel['files'] = {n: file_hash(bundle/n) for n in sorted(PANEL_FILES)}
    atomic_json(bundle/'panel.json', panel)
    body = read_json(bundle/'bundle.json');body.pop('bundle_id')
    body['files'] = {n: artifact(bundle/n) for n in sorted(BUNDLE_FILES)}
    body['bundle_id'] = digest(body)
    atomic_json(bundle/'bundle.json', body)
    atomic_json(bundle/'COMPLETE.json', {'kind': 'development_bundle', 'bundle_id': body['bundle_id'],
                                       'files': {n: artifact(bundle/n) for n in BUNDLE_FILES | {'bundle.json'}}})
    with pytest.raises(ResearchError, match='date/security'):load_bundle(bundle)


def test_accounting_expectations_are_independent_and_not_a_ledger_attestation():
    reference = accounting_reference()
    cases = {c['case']: c for c in reference['cases']}
    short = cases['short_sale']
    assert short['expected_cash_usd'] + short['shares'] * short['price_usd'] == short['expected_nav_usd']
    div = cases['long_cash_dividend']
    assert div['shares']*div['ex_price_usd'] + div['expected_receivable_usd'] == 1000
    assert cases['unresolved_terminal']['expected_economic_proceeds'] is None
    assert reference['legacy_accounting_verified'] is False


def test_gpu_wrapper_honors_experiment_determinism(monkeypatch, tmp_path):
    from alpha_lab import local_runner
    calls=[]
    request = lambda value: calls.append(value)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(use_deterministic_algorithms=request))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_runner.runpy, 'run_path', lambda *a, **k: sys.modules['torch'].use_deterministic_algorithms(True))
    local_runner.run(True)
    assert calls == [True] and sys.modules['torch'].use_deterministic_algorithms is request
