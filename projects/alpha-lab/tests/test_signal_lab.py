"""Independent timing, missingness, fit, cache and native-worker P2 acceptance cases."""
import json
from pathlib import Path

import pytest
np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')
duckdb = pytest.importorskip('duckdb')

from alpha_lab.equity_calendar import session_schedule
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.bundles import load_bundle, pin_panel
from alpha_lab.research.contracts import ResearchError, digest, file_hash, parse, read_json
from alpha_lab.research.features import (KEYS, cached_feature, feature_identity, prepare_feature,
                                          read_parquet, write_parquet)
from alpha_lab.research.registry import Registry, initialize_registry
from alpha_lab.research.signal_contracts import Family, Fold, Hypothesis, Label, Recipe, SignalSpec, baseline_recipes
from alpha_lab.research.signals import diagnose, make_labels, score_fold, training_mask, transform_feature
from alpha_lab.research import signal_workflow as flow


@pytest.fixture
def signal_bundle(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    calendar, meta = session_schedule('2014-01-01', '2014-05-30')
    dates = pd.DatetimeIndex(calendar.date)
    tickers = [f'S{i:02d}' for i in range(40)]
    rows = []
    for day in ('2014-01-31', '2014-02-28', '2014-03-31', '2014-04-30', '2014-05-30'):
        for i, ticker in enumerate(tickers):
            row = {'date': pd.Timestamp(day), 'ticker': ticker, 'beta': 1.,
                   'filing_date': pd.Timestamp(day)-pd.Timedelta(days=10),
                   'available_date': pd.Timestamp(day)-pd.Timedelta(days=8)}
            for j, name in enumerate(('book_yield', 'earnings_yield', 'fcf_yield', 'roa', 'gross_profitability',
                                       'leverage', 'momentum_12_1', 'momentum_3_1', 'reversal_1m', 'low_vol')):
                row[name] = float(np.sin(i+j))
            if i < 5:
                row['filing_date'] = row['available_date'] = pd.NaT
            rows.append(row)
    features = pd.DataFrame(rows)
    metadata = pd.DataFrame({'ticker': tickers, 'permaticker': list(range(1, 41)), 'table': ['SEP']*40})
    actions = pd.DataFrame({'ticker': ['S00'], 'date': [pd.Timestamp('2014-02-03')], 'action': ['split']})
    for name, table in [('features', features), ('actions', actions), ('metadata', metadata)]:
        write_parquet(table, source/(name+'.parquet'))
    returns = np.tile(np.linspace(-.003, .003, 40), (len(dates), 1))
    returns[dates.get_loc('2014-04-02'), 0] = np.nan
    observed = np.ones_like(returns, dtype=bool)
    observed[dates.get_loc('2014-04-02'), 0] = False
    returns[dates.get_loc('2014-04-03'), 0] = np.nan
    np.savez_compressed(source/'returns.npz', dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                        tickers=np.asarray(tickers, dtype='U32'), returns=returns,
                        price_observed=observed, spy=np.full(len(dates), .001))
    calendar.to_csv(source/'calendar.csv', index=False)
    sources = {name: {'version': 'a'*32, 'parquet_sha256': file_hash(source/'metadata.parquet'),
                      'path': str(source/'metadata.parquet')}
               for name in ('stocks', 'funds', 'daily', 'fundamentals', 'tickers', 'actions')}
    atomic_json(source/'sources.json', sources)
    atomic_json(source/'panel.json', {'schema_version': 4, 'start': '2014-01-01', 'end': '2014-05-30',
                'calendar': meta, 'files': {n: file_hash(source/n) for n in
                ('features.parquet', 'actions.parquet', 'sources.json', 'returns.npz', 'calendar.csv')}})
    destination = tmp_path/'bundle'
    pin_panel(source, destination, end='2014-05-30')
    return destination


def request(**changes):
    value = {'family_id': 'test-family', 'hypothesis_id': 'momentum', 'recipe': baseline_recipes()[2],
             'label': {'target': 'simple_excess_spy', 'horizon_sessions': 2, 'execution_lag_sessions': 1,
                       'availability_lag_sessions': 1, 'basis': 'vendor_adjusted_close_to_close_no_gap_fill'},
             'folds': [{'name': 'first', 'train_start': '2014-01-01', 'train_end': '2014-02-28',
                        'validation_start': '2014-03-01', 'validation_end': '2014-05-30', 'embargo_sessions': 0}],
             'quantiles': 5, 'min_cross_section': 10, 'seed': 42, 'threads': 1, 'timeout_seconds': 45,
             'max_output_bytes': 4_000_000, 'memory_limit_mb': 1024}
    return value | changes


def registration(db, spec, *, max_trials=4, max_attempts=2, parent=None):
    family = Family(family_id=spec.family_id, premise='Synthetic signal comparisons only',
                    max_unique_trials=max_trials, max_attempts_per_trial=max_attempts)
    hypothesis = Hypothesis(hypothesis_id=spec.hypothesis_id, family_id=spec.family_id,
                            premise='Prespecified synthetic momentum diagnostic', tests=['coverage', 'causal timing'])
    return db.register(family, hypothesis, spec, parent)


def test_price_only_does_not_require_filing_and_fundamental_gates_are_independent():
    source = pd.DataFrame({'ticker': ['A','B','C','D','E'], 'date': pd.Timestamp('2018-05-01'),
                            'momentum_12_1': [1.,2.,3.,4.,5.], 'roa': [1.,2.,3.,4.,5.],
                            'filing_date': pd.to_datetime([None, '2018-04-01', '2018-05-01', '2016-01-01', '2018-04-01']),
                            'available_date': pd.to_datetime([None, '2018-04-03', '2018-05-03', '2016-01-03', '2018-04-02'])})
    ids = dict(zip(source.ticker, range(1,6)))
    momentum = prepare_feature(source, 'momentum_12_1', ids)
    assert momentum.feature_available.all() and momentum.source_member.all()
    fundamental = prepare_feature(source, 'roa', ids)
    assert fundamental.feature_available.tolist() == [False, True, False, False, False]
    assert fundamental.availability_reason.tolist() == ['undated_information','available','future_information',
                                                        'stale_information','inconsistent_availability']


def test_label_math_entry_close_not_decision_close_and_simple_excess():
    dates = pd.bdate_range('2018-01-01', periods=7)
    member = pd.DataFrame({'date': [dates[0]], 'ticker': ['A'], 'security_id': [1]})
    arrays = {'dates': dates, 'tickers': np.array(['A']), 'returns': np.array([[0.],[.9],[.1],[-.2],[.8],[0.],[0.]]),
              'price_observed': np.ones((7,1), bool), 'spy': np.array([0.,.9,.01,.02,0.,0.,0.])}
    policy = parse(Label, request()['label'])
    actual = make_labels(member, arrays, policy).iloc[0]
    assert actual.label == pytest.approx((1.1*.8-1)-(1.01*1.02-1))
    assert actual.label_entry == dates[1] and actual.label_end == dates[3] and actual.label_available == dates[4]
    arrays['returns'][5:] = 9999.
    assert make_labels(member, arrays, policy).label.item() == actual.label
    arrays['returns'][2,0] = np.nan
    missing = make_labels(member, arrays, policy)
    assert len(missing) == 1 and not missing.label_observed.item() and missing.label_status.item() == 'missing_asset_path'


def test_label_right_and_availability_censoring_retain_rows():
    dates = pd.bdate_range('2018-01-01', periods=7)
    members = pd.DataFrame({'date': dates[-4:], 'ticker': ['A']*4, 'security_id': [1]*4})
    arrays = {'dates': dates, 'tickers': np.array(['A']), 'returns': np.zeros((7,1)),
              'price_observed': np.ones((7,1), bool), 'spy': np.zeros(7)}
    labels = make_labels(members, arrays, parse(Label, request()['label']))
    assert labels.label_status.tolist() == ['availability_censored','right_censored','right_censored','right_censored']
    assert labels.label.isna().all()


def test_overlap_and_availability_purge_on_actual_session_cutoff():
    sessions = pd.bdate_range('2014-01-01', '2014-04-30')
    fold = parse(Fold, request()['folds'][0])
    frame = pd.DataFrame({'date': pd.to_datetime(['2014-01-31']*4), 'label_observed': [True,True,True,False],
                          'label_end': pd.to_datetime(['2014-02-28','2014-03-03','2014-02-28',None]),
                          'label_available': pd.to_datetime(['2014-02-28','2014-03-04','2014-03-03',None])})
    fit, evidence = training_mask(frame, fold, sessions)
    assert fit.tolist() == [True,False,False,False]
    assert evidence['cutoff'].startswith('2014-03-03') and evidence['purged_overlap_or_availability'] == 2
    embargo = parse(Fold, fold.model_dump() | {'embargo_sessions': 1})
    assert not training_mask(frame, embargo, sessions)[0].any()


def test_zscore_fit_uses_only_admitted_training_and_constant_or_missing_is_explicit():
    frame = pd.DataFrame({'date': pd.to_datetime(['2014-01-31']*4), 'x': [1.,3.,1000.,np.nan]})
    fit = pd.Series([True,True,False,False])
    values, evidence = transform_feature(frame, 'x', fit, 'train_zscore')
    np.testing.assert_allclose(values[:3], [-1.,1.,998.])
    assert evidence['mean'] == 2 and evidence['std_population'] == 1
    frame.loc[2, 'x'] = -99999.
    assert transform_feature(frame, 'x', fit, 'train_zscore')[1] == evidence
    assert transform_feature(frame.assign(x=1.), 'x', fit, 'train_zscore')[0].eq(0).all()
    assert transform_feature(frame, 'x', fit & False, 'train_zscore')[0].isna().all()


def test_future_missing_labels_do_not_change_feature_scores_or_bucket_membership(signal_bundle):
    spec = flow.resolve_signal(signal_bundle, request())
    raw = pd.DataFrame({'date': pd.Timestamp('2014-03-31'), 'ticker': [f'S{i}' for i in range(20)],
                        'security_id': range(1,21), 'source_member': True, 'raw_momentum_12_1': np.arange(20.),
                        'label': np.arange(20.)/100, 'label_observed': True, 'label_status': 'observed',
                        'label_end': pd.Timestamp('2014-04-04'), 'label_available': pd.Timestamp('2014-04-07')})
    sessions = pd.bdate_range('2014-01-01','2014-05-30')
    original, _ = score_fold(raw, spec, spec.folds[0], sessions)
    changed = raw.copy(); changed.loc[15:, ['label','label_observed','label_status']] = [np.nan,False,'missing_asset_path']
    revised, _ = score_fold(changed, spec, spec.folds[0], sessions)
    pd.testing.assert_series_equal(original.score, revised.score)
    pd.testing.assert_series_equal(original['quantile'], revised['quantile'])
    daily, buckets, _ = diagnose(revised, spec)
    assert daily.members.item() == 20 and daily.paired.item() == 15
    assert buckets.loc[buckets['quantile'].eq(5),'scored'].item() == 4
    assert buckets.loc[buckets['quantile'].eq(5),'observed'].item() == 0


def test_computation_identity_ignores_renames_positive_rescales_and_operational_budgets(signal_bundle):
    spec = flow.resolve_signal(signal_bundle, request(recipe=baseline_recipes()[0]))
    renamed = parse(SignalSpec, spec.model_dump() | {'family_id': 'other', 'hypothesis_id': 'other', 'timeout_seconds': 90,
                          'recipe': spec.recipe.model_dump() | {'name': 'renamed', 'factors': {'book_yield': 7., 'earnings_yield': 7., 'roa': 0.}}})
    assert renamed.semantic_id() == spec.semantic_id()
    reverse = parse(SignalSpec, spec.model_dump() | {'recipe': spec.recipe.model_dump() | {'factors': {'book_yield': -1., 'earnings_yield': -1.}}})
    assert reverse.semantic_id() != spec.semantic_id()


@pytest.mark.parametrize('change', [{'seed': True}, {'threads': 0}, {'API_KEY': 'synthetic-private'},
                                   {'timeout_seconds': '45'}, {'quantiles': 1}])
def test_strict_signal_settings_do_not_echo_inputs(signal_bundle, change):
    with pytest.raises(ResearchError) as error:
        flow.resolve_signal(signal_bundle, request(**change))
    assert 'synthetic-private' not in str(error.value)


def test_fold_overlap_and_future_dates_are_denied(signal_bundle):
    base = request()
    for folds in ([base['folds'][0], base['folds'][0] | {'name': 'overlapping'}],
                  [base['folds'][0] | {'validation_end': '2023-01-01'}],
                  [base['folds'][0] | {'validation_end': '2014-06-30'}]):
        with pytest.raises(ResearchError): flow.resolve_signal(signal_bundle, base | {'folds': folds})


def test_cache_reuses_raw_computation_and_rejects_corruption(signal_bundle, tmp_path):
    bundle = load_bundle(signal_bundle)
    first, key = cached_feature(signal_bundle, bundle, 'momentum_12_1', tmp_path/'cache', 2_000_000)
    path = tmp_path/'cache'/key/'values.parquet'; before = path.stat().st_mtime_ns
    second, other = cached_feature(signal_bundle, bundle, 'momentum_12_1', tmp_path/'cache', 2_000_000)
    assert key == other and before == path.stat().st_mtime_ns
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    assert first.feature_available.all()
    fundamental, _ = cached_feature(signal_bundle, bundle, 'roa', tmp_path/'cache', 2_000_000)
    assert fundamental.feature_available.sum() == 175 and len(fundamental) == 200
    changed = bundle.model_copy(update={'bundle_id': '0'*64})
    assert feature_identity(changed, 'roa')[0] != feature_identity(bundle, 'roa')[0]
    path.write_bytes(b'bad')
    with pytest.raises(ResearchError, match='hash/size'):cached_feature(signal_bundle, bundle, 'momentum_12_1', tmp_path/'cache', 2_000_000)


def test_native_worker_publication_and_read_only_inspection(signal_bundle, tmp_path, monkeypatch):
    spec = flow.resolve_signal(signal_bundle, request())
    spec_path = tmp_path/'spec.json'; atomic_json(spec_path, spec.model_dump())
    ledger = tmp_path/'registry.sqlite'; initialize_registry(ledger)
    with Registry(ledger) as db: registration(db, spec)
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-private')
    monkeypatch.setenv('HTTPS_PROXY','http://unreachable.invalid')
    output = tmp_path/'run'
    result = flow.run_signal(signal_bundle, spec_path, output, tmp_path/'cache', ledger)
    assert result['artifacts_verified'] and result['effective_status'] == 'completed'
    predictions = read_parquet(output/'predictions.parquet')
    assert len(predictions) == 200 and predictions.score_available.all()
    assert predictions.loc[predictions.date.eq('2014-05-30'), 'label_status'].eq('right_censored').all()
    assert read_json(output/'runtime.json')['provider_keys_present'] is False
    before = {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in output.iterdir() if p.is_file()}
    assert flow.inspect_signal(output)['result_hash'] == result['result_hash']
    assert before == {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in output.iterdir() if p.is_file()}
    with Registry(ledger, readonly=True) as db:
        data = db.snapshot()
        assert data['attempts'][0]['status'] == 'completed'
        assert data['exposures'][0]['result_hash'] == result['result_hash']
    with pytest.raises(ResearchError, match='completed'):
        flow.run_signal(signal_bundle, spec_path, tmp_path/'duplicate', tmp_path/'cache', ledger)
    (output/'predictions.parquet').write_bytes(b'corrupt')
    with pytest.raises(ResearchError):flow.inspect_signal(output)


def test_zero_exit_without_results_is_failed_and_retry_is_explicit(signal_bundle, tmp_path, monkeypatch):
    class Empty:
        pid = 123
        def poll(self): return 0
        def wait(self): return 0
        def close(self): pass
    spec = flow.resolve_signal(signal_bundle, request())
    path = tmp_path/'spec.json'; atomic_json(path,spec.model_dump())
    ledger = tmp_path/'registry.sqlite'; initialize_registry(ledger)
    with Registry(ledger) as db:registration(db,spec)
    monkeypatch.setattr(flow, 'start_process', lambda *a, **k: Empty())
    with pytest.raises(ResearchError, match='publication_failure'):
        flow.run_signal(signal_bundle,path,tmp_path/'failed',tmp_path/'cache',ledger)
    evidence = flow.inspect_signal(tmp_path/'failed')
    assert evidence['effective_status'] == 'failed'
    assert flow.reconcile_signal(ledger,tmp_path/'failed')['effective_status'] == 'failed'
    with Registry(ledger, readonly=True) as db:
        history = db.snapshot()
        assert history['exposures'][0]['kind'].startswith('development_failed')
    with pytest.raises(ResearchError, match='Retry'):
        flow.run_signal(signal_bundle,path,tmp_path/'unlinked',tmp_path/'cache',ledger)
    with pytest.raises(ResearchError, match='publication_failure'):
        flow.run_signal(signal_bundle,path,tmp_path/'retry',tmp_path/'cache',ledger,retry_of=evidence['attempt_id'])
    with Registry(ledger) as db:
        assert len(db.snapshot()['attempts']) == 2
        with pytest.raises(ResearchError, match='budget'):db.begin(spec,tmp_path/'third',evidence['attempt_id'])


def test_comparison_rejects_incompatible_versions_before_reading_predictions(monkeypatch):
    monkeypatch.setattr(flow,'inspect_signal',lambda p: {'artifacts_verified':True,'comparison_id':p})
    with pytest.raises(ResearchError,match='incompatible'):flow.compare_signals(['one','two'])


def test_raw_feature_coverage_survives_an_unfittable_transform(signal_bundle):
    spec=flow.resolve_signal(signal_bundle,request(recipe=baseline_recipes()[2] | {'transform':'train_zscore'}))
    frame=pd.DataFrame({'date':pd.Timestamp('2014-03-31'),'security_id':[1,2], 'ticker':['A','B'],
                        'raw_momentum_12_1':[1.,2.], 'label_observed':False, 'label':np.nan,
                        'label_end':pd.NaT,'label_available':pd.NaT})
    actual,evidence=score_fold(frame,spec,spec.folds[0],pd.bdate_range('2014-01-01','2014-05-30'))
    assert actual.feature_available.all() and not actual.transform_available.any() and not actual.score_available.any()
    assert evidence['features']['momentum_12_1']['status']=='no_training_values'


@pytest.mark.parametrize('field',['bundle_id','label','folds','implementation','environment','quantiles'])
def test_comparison_identity_includes_semantic_compatibility(signal_bundle,field):
    spec=flow.resolve_signal(signal_bundle,request())
    changes={'bundle_id':'0'*64, 'label':spec.label.model_dump() | {'horizon_sessions':3},
             'folds':[spec.folds[0].model_dump() | {'embargo_sessions':1}],
             'implementation':{},'environment':{},'quantiles':4}
    changed=parse(SignalSpec,spec.model_dump() | {field:changes[field]})
    assert changed.comparison_id()!=spec.comparison_id()


def test_equal_score_ties_stay_together_and_constant_ic_is_explicit(signal_bundle):
    spec=flow.resolve_signal(signal_bundle,request(min_cross_section=2))
    frame=pd.DataFrame({'date':pd.Timestamp('2014-03-31'),'security_id':[1,2], 'ticker':['A','B'],
                        'raw_momentum_12_1':[1.,1.], 'label_observed':True, 'label':[.1,.2],
                        'label_end':pd.Timestamp('2014-04-04'),'label_available':pd.Timestamp('2014-04-07')})
    actual,_=score_fold(frame,spec,spec.folds[0],pd.bdate_range('2014-01-01','2014-05-30'))
    assert actual['quantile'].nunique()==1
    daily,buckets,_=diagnose(actual,spec)
    assert daily.ic_status.item()=='constant_values' and daily.rank_ic.isna().all()
    assert (buckets.scored==0).sum()==4


def test_dependency_check_projects_only_requested_features(signal_bundle,tmp_path):
    spec=flow.resolve_signal(signal_bundle,request())
    contract=load_bundle(signal_bundle)
    projected=tmp_path/'projected';projected.mkdir()
    write_parquet(pd.DataFrame({'momentum_12_1':[1.]}),projected/'features.parquet')
    flow.validate_signal(spec,contract,projected)
    quality=parse(SignalSpec,spec.model_dump() | {'recipe':baseline_recipes()[1]})
    with pytest.raises(ResearchError,match='dependency'):flow.validate_signal(quality,contract,projected)
    write_parquet(pd.DataFrame({'roa':[1.],'gross_profitability':[1.],'leverage':[1.]}),projected/'features.parquet')
    with pytest.raises(ResearchError,match='availability'):flow.validate_signal(quality,contract,projected)
