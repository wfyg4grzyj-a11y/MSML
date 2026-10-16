"""Registered, bounded P2 execution and read-only evidence inspection/comparison."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from alpha_lab.local_process import start_process
from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact, load_bundle, safe_file, verify_files
from .contracts import DataBundle, ResearchError, digest, file_hash, parse, read_json
from .features import KEYS, feature_identity, read_parquet
from .registry import Registry, stamp
from .runtime import child_environment
from .signal_contracts import SignalRun, SignalSpec
from .signals import rank_correlation
from .workflow import compatible

SIGNAL_RESULTS = {'predictions.parquet', 'daily.parquet', 'quantiles.parquet', 'fits.json', 'metrics.json', 'runtime.json'}
SIGNAL_FILES = SIGNAL_RESULTS | {'spec.json', 'bundle.json', 'report.txt'}


def resolve_signal(bundle_path, request):
    if set(request) & {'schema_version', 'bundle_id', 'implementation', 'environment',
                       'feature_policy', 'coverage_policy', 'numerical_policy'}:
        raise ResearchError('Unresolved study contains reserved resolved fields')
    from .runtime import environment_identity, implementation_identity
    bundle = load_bundle(bundle_path)
    spec = parse(SignalSpec, request | {
        'schema_version': 2, 'bundle_id': bundle.bundle_id,
        'implementation': implementation_identity(), 'environment': environment_identity(),
        'feature_policy': 'independent_dependencies_filing_plus2_max_age550_v1',
        'coverage_policy': 'legacy_source_membership_no_future_outcome_filter_v1',
        'numerical_policy': 'cpu_float64_fixed_threads'})
    validate_signal(spec, bundle, bundle_path)
    return spec


def validate_signal(spec, bundle, bundle_path=None):
    compatible(spec, bundle)
    if any(f.train_start < bundle.start or f.validation_end > bundle.end for f in spec.folds):
        raise ResearchError('Folds exceed this pinned development bundle')
    if bundle_path is not None:
        import duckdb
        from .signal_contracts import FEATURES
        with duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB'}) as db:
            kinds = {row[0]: row[1] for row in db.execute('DESCRIBE SELECT * FROM read_parquet(?)',
                     [str(Path(bundle_path)/'features.parquet')]).fetchall()}
        for name in spec.recipe.computation()['factors']:
            if kinds.get(name) not in {'FLOAT', 'DOUBLE'}:
                raise ResearchError('Signal requires an absent or nonnumeric feature dependency')
            if FEATURES[name][0] == 'fundamental' and not {'filing_date', 'available_date'} <= kinds.keys():
                raise ResearchError('Fundamental signal requires explicit filing and availability fields')


def validate_signal_results(directory, spec, bundle):
    directory = Path(directory)
    metrics = read_json(safe_file(directory, 'metrics.json'))
    if (set(metrics) != {'kind', 'trial_id', 'comparison_id', 'bundle_id', 'feature_ids', 'folds',
                         'investment_ready', 'point_in_time_attested', 'rows', 'llm_calls'}
            or metrics['kind'] != 'development_signal_diagnostics_v1'
            or metrics['trial_id'] != spec.semantic_id() or metrics['comparison_id'] != spec.comparison_id()
            or metrics['bundle_id'] != bundle.bundle_id or metrics['investment_ready'] is not False
            or metrics['point_in_time_attested'] is not False or metrics['llm_calls'] != 0
            or set(metrics['feature_ids']) != set(spec.recipe.computation()['factors'])):
        raise ResearchError('Signal result identity/readiness mismatch')
    predictions = read_parquet(safe_file(directory, 'predictions.parquet'))
    expected = set(KEYS + ['source_member', 'label_entry', 'label_end', 'label_available', 'label',
                          'label_observed', 'label_status', 'fold', 'role', 'fit_admitted',
                          'feature_available', 'transform_available', 'score', 'score_available', 'quantile'])
    for feature in spec.recipe.computation()['factors']:
        expected.update(prefix+feature for prefix in ('raw_', 'available_', 'reason_', 'information_', 'transformed_'))
    if (set(predictions) != expected or len(predictions) != metrics['rows'] or predictions.empty
            or predictions.duplicated(['fold']+KEYS).any()
            or not predictions.date.between(bundle.start, bundle.end).all()
            or not predictions.source_member.eq(True).all()
            or set(predictions.fold) != {f.name for f in spec.folds}
            or not set(predictions.role) <= {'train', 'validation'}
            or not predictions.label_observed.equals(predictions.label_status.eq('observed'))
            or not predictions.label_observed.equals(pd.Series(np.isfinite(predictions.label), name='label_observed'))
            or not predictions.score_available.equals(pd.Series(np.isfinite(predictions.score), name='score_available'))
            or not predictions['quantile'].notna().equals(predictions.score_available)
            or not predictions.loc[predictions.score_available, 'quantile'].between(1, spec.quantiles).all()):
        raise ResearchError('Signal prediction coverage/schema is invalid')
    for fold in spec.folds:
        subset = predictions.loc[predictions.fold.eq(fold.name)]
        if not subset.loc[subset.role.eq('train'), 'date'].between(fold.train_start, fold.train_end).all() or not subset.loc[
                subset.role.eq('validation'), 'date'].between(fold.validation_start, fold.validation_end).all():
            raise ResearchError('Prediction rows violate fold bounds')
    daily = read_parquet(safe_file(directory, 'daily.parquet'))
    quantiles = read_parquet(safe_file(directory, 'quantiles.parquet'))
    if (daily.empty or daily.duplicated(['fold', 'date']).any()
            or set(daily) != {'fold', 'date', 'members', 'feature_available', 'scored', 'labels_observed', 'paired',
                              'rank_ic', 'ic_status', 'top_minus_bottom_label', 'score_rank_autocorrelation',
                              'score_overlap', 'top_membership_churn', 'top_entrants', 'top_exits'}
            or set(quantiles) != {'fold', 'date', 'quantile', 'scored', 'observed', 'mean_label'}
            or quantiles.duplicated(['fold', 'date', 'quantile']).any()
            or len(quantiles) != len(daily) * spec.quantiles):
        raise ResearchError('Signal diagnostics are incomplete')
    for frame in (predictions, daily, quantiles):
        if np.isinf(frame.select_dtypes(include='number').to_numpy(dtype=float, na_value=np.nan)).any():
            raise ResearchError('Infinite diagnostic values')
    # Verify coverage counts independently of the worker's summary JSON.
    val = predictions.loc[predictions.role.eq('validation')]
    counts = val.groupby(['fold', 'date']).agg(members=('security_id', 'size'), scored=('score_available', 'sum'),
                                              labels_observed=('label_observed', 'sum')).reset_index()
    if not daily[['fold', 'date', 'members', 'scored', 'labels_observed']].equals(counts):
        raise ResearchError('Diagnostic coverage differs from retained predictions')
    runtime = read_json(safe_file(directory, 'runtime.json'))
    if runtime != {'environment': spec.environment, 'implementation': spec.implementation,
                   'threads': spec.threads, 'seed': spec.seed, 'numerical_policy': spec.numerical_policy,
                   'provider_keys_present': False, 'proxy_present': False,
                   'isolation': 'credential_filtered_process_not_filesystem_or_network_sandbox'}:
        raise ResearchError('Worker runtime differs from requested isolation/numerical policy')
    fits = read_json(safe_file(directory, 'fits.json'))
    if set(fits) != {f.name for f in spec.folds} or len(metrics['folds']) != len(spec.folds):
        raise ResearchError('Fold fit evidence is incomplete')
    return metrics, predictions


def report_text(spec, bundle, record, metrics):
    from .contracts import canonical
    return (f'US equity signal laboratory — P2\nAttempt: {record.attempt_id}\nTrial: {spec.semantic_id()}\n'
            f'Hypothesis/family: {spec.hypothesis_id} / {spec.family_id}\nBundle: {bundle.bundle_id}\n'
            f'Recipe: {spec.recipe.name}; computation: {canonical(spec.recipe.computation())}\n'
            f'Label: {canonical(spec.label.model_dump())}\n'
            'Decision: session close; entry: following declared session close; outcomes begin AFTER entry.\n'
            'Excess label: compounded asset total return minus compounded SPY proxy total return.\n'
            'Training: outcome end and assumed availability strictly precede validation cutoff minus embargo.\n'
            'Rank transforms use feature-available members on that date; z-scores fit admitted training rows only.\n'
            'Quantiles use ALL scored members before label missingness; ties stay together.\n'
            'IC pairs require observed labels; report all source members, score coverage and censored outcomes.\n'
            'Top churn = 1 - shared top names / previous top count; entrants/exits and overlap are explicit.\n'
            'Monthly row population, static domestic eligibility and cap/liquidity/price screens are inherited.\n'
            'No historical eligibility/issuer, vendor-arrival or terminal-action settlement attestation.\n'
            'Fundamentals: current ART reconstruction, filing+2 calendar days, age<=550 days; revisions possible.\n'
            'Features are legacy prepared formulas, not a new raw-data recomputation or corrected PIT history.\n'
            'No transaction costs, borrow, financing, capacity, ledger, or risk neutrality demonstrated here.\n'
            'Quantile spreads are descriptive forward labels, NOT executable portfolio P&L or expected returns.\n'
            'Operational completion is NOT strategy acceptance. No holdout evaluated; no LLM/provider calls.\n'
            'Original 2023–2025 holdout was previously observed and remains protected.\n'
            'Worker: filtered credentials, CPU float64/fixed threads, owned process tree. Same-user files/network\n'
            'remain accessible; this is not an OS security sandbox. Time/output guards and sampled RSS cap.\n'
            'Raw cache: at most one artifact per requested feature; each capped at max_output_bytes, outside run cap.\n'
            'Registry publication follows COMPLETE.json; reconcile explicitly after an interrupted DB update.\n\n'
            'Per-fold descriptive evidence (no automatic selection):\n'+canonical(metrics['folds'])+'\n')


def run_signal(bundle_path, spec_path, destination, cache, registry_path, *, retry_of=None):
    spec = parse(SignalSpec, read_json(spec_path))
    bundle_path, destination, cache = [Path(p).resolve() for p in (bundle_path, destination, cache)]
    for a, b in ((destination, bundle_path), (cache, bundle_path), (cache, destination)):
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ResearchError('Keep input bundle, run directory and cache in separate directory trees')
    bundle = load_bundle(bundle_path)
    validate_signal(spec, bundle, bundle_path)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination/'spec.json', spec.model_dump())
    atomic_json(destination/'bundle.json', bundle.model_dump())
    (destination/'INCOMPLETE').write_text('Require COMPLETE.json and registry reconciliation; no automatic retries\n')
    with Registry(registry_path) as registry:
        attempt = registry.begin(spec, destination, retry_of, access_bounds=(bundle.start, bundle.end))
        record = SignalRun(schema_version=2, attempt_id=attempt, trial_id=spec.semantic_id(),
                           spec_sha256=digest(spec.model_dump()), bundle_id=bundle.bundle_id,
                           hypothesis_id=spec.hypothesis_id, status='running', stage='validating',
                           started_at=stamp(), updated_at=stamp(), finished_at=None, failure=None,
                           peak_rss_mb=0., worker_pid=None, retry_of=retry_of, artifacts={}, llm_calls=0,
                           scientific_disposition='provisional_development_signal_diagnostics_not_portfolio_pnl')
        proc, failure, published = None, 'input_validation', False
        try:
            atomic_json(destination/'run.json', record.model_dump())
            work = destination/'worker-output'
            work.mkdir()
            failure = 'execution_failure'
            record = record.model_copy(update={'stage': 'evaluating', 'updated_at': stamp()})
            start = time.monotonic()
            with (destination/'worker.out').open('w', encoding='utf-8') as log:
                proc = start_process([sys.executable, '-s', '-P', '-m', 'alpha_lab.research.signal_worker',
                                      '--bundle', str(bundle_path), '--spec', str(destination/'spec.json'),
                                      '--output', str(work), '--cache', str(cache)], cwd=str(destination),
                                     env=child_environment(spec), stdout=log, stderr=subprocess.STDOUT, text=True)
                record = record.model_copy(update={'worker_pid': proc.pid})
                while proc.poll() is None:
                    try:
                        root = psutil.Process(proc.pid)
                        rss = sum(p.memory_info().rss for p in [root]+root.children(recursive=True) if p.is_running()) / 1024**2
                    except psutil.Error:
                        rss = record.peak_rss_mb
                    record = record.model_copy(update={'peak_rss_mb': max(float(rss), record.peak_rss_mb), 'updated_at': stamp()})
                    atomic_json(destination/'run.json', record.model_dump())
                    size = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
                    if time.monotonic()-start > spec.timeout_seconds:
                        failure = 'timeout'
                        raise ResearchError('Signal worker timed out')
                    if size > spec.max_output_bytes:
                        failure = 'output_limit'
                        raise ResearchError('Signal output cap exceeded')
                    if rss > spec.memory_limit_mb:
                        failure = 'memory_limit'
                        raise ResearchError('Signal sampled RSS cap exceeded')
                    time.sleep(.25)
                if proc.wait() != 0:
                    raise ResearchError('Signal worker failed')
                if time.monotonic()-start > spec.timeout_seconds:
                    failure = 'timeout'
                    raise ResearchError('Signal worker deadline exceeded')
            failure = 'publication_failure'
            if {p.name for p in work.iterdir()} != SIGNAL_RESULTS:
                raise ResearchError('Worker omitted required signal artifacts')
            metrics, predictions = validate_signal_results(work, spec, bundle)
            for name, key in metrics['feature_ids'].items():
                if feature_identity(bundle, name)[0] != key:
                    raise ResearchError('Worker used an incompatible feature computation')
            # Cover every source row in each declared fold, including missing outcomes.
            members = read_parquet(bundle_path/'features.parquet')[['date', 'ticker']]
            for fold in spec.folds:
                expected = members.loc[members.date.between(fold.train_start, fold.train_end) |
                                       members.date.between(fold.validation_start, fold.validation_end)]
                actual = predictions.loc[predictions.fold.eq(fold.name), ['date', 'ticker']]
                if not actual.sort_values(['date', 'ticker']).reset_index(drop=True).equals(
                        expected.sort_values(['date', 'ticker']).reset_index(drop=True)):
                    raise ResearchError('Worker lost source population rows')
            validate_signal(spec, load_bundle(bundle_path), bundle_path)
            record = record.model_copy(update={'stage': 'publishing', 'updated_at': stamp()})
            atomic_json(destination/'run.json', record.model_dump())
            for name in sorted(SIGNAL_RESULTS):
                os.replace(work/name, destination/name)
            work.rmdir()
            (destination/'report.txt').write_text(report_text(spec, bundle, record, metrics), encoding='utf-8')
            if sum(p.stat().st_size for p in destination.iterdir() if p.is_file()) > spec.max_output_bytes:
                failure = 'output_limit'
                raise ResearchError('Final artifact set exceeds output cap')
            record = parse(SignalRun, record.model_dump() | {'status': 'completed', 'stage': 'finished',
                           'updated_at': stamp(), 'finished_at': stamp(),
                           'artifacts': {n: artifact(destination/n) for n in sorted(SIGNAL_FILES)}})
            atomic_json(destination/'run.json', record.model_dump())
            atomic_json(destination/'COMPLETE.json', {'kind': 'signal_run_v2', 'trial_id': spec.semantic_id(),
                        'files': {n: artifact(destination/n) for n in sorted(SIGNAL_FILES | {'run.json'})}})
            if sum(p.stat().st_size for p in destination.iterdir() if p.is_file()) > spec.max_output_bytes:
                failure = 'output_limit'
                raise ResearchError('Completion inventory exceeds output cap')
            evidence = inspect_signal(destination)
            published = True
            (destination/'INCOMPLETE').unlink()
            registry.finish(attempt, complete=evidence)
            return evidence
        except BaseException as error:
            if proc is not None and proc.poll() is None:
                proc.terminate_tree()
            if published:
                # Files are already valid and must not be rewritten as a failure.
                raise ResearchError('Artifacts completed; explicit registry reconciliation required') from None
            failure = 'interrupted' if isinstance(error, KeyboardInterrupt) else failure
            record = parse(SignalRun, record.model_dump() | {'status': 'failed', 'stage': 'finished',
                           'updated_at': stamp(), 'finished_at': stamp(), 'failure': failure})
            atomic_json(destination/'run.json', record.model_dump())
            (destination/'COMPLETE.json').unlink(missing_ok=True)
            registry.finish(attempt, failure=failure)
            raise ResearchError('Signal attempt failed: '+failure) from None
        finally:
            if proc is not None:
                proc.close()


def inspect_signal(directory):
    directory = Path(directory).resolve()
    record = parse(SignalRun, read_json(safe_file(directory, 'run.json')))
    spec = parse(SignalSpec, read_json(safe_file(directory, 'spec.json')))
    if (record.trial_id != spec.semantic_id() or record.spec_sha256 != digest(spec.model_dump())
            or record.bundle_id != spec.bundle_id or record.hypothesis_id != spec.hypothesis_id):
        raise ResearchError('Attempt/resolved specification identity mismatch')
    base = {'attempt_id': record.attempt_id, 'trial_id': record.trial_id,
            'hypothesis_id': spec.hypothesis_id, 'directory': str(directory),
            'start': min(f.train_start for f in spec.folds), 'end': max(f.validation_end for f in spec.folds),
            'result_hash': None, 'failure': record.failure}
    if not (directory/'COMPLETE.json').is_file():
        return base | {'effective_status': 'failed' if record.status == 'failed' else 'unfinished_requires_reconciliation',
                       'artifacts_verified': False}
    marker = read_json(safe_file(directory, 'COMPLETE.json'))
    if (set(marker) != {'kind', 'trial_id', 'files'} or marker['kind'] != 'signal_run_v2'
            or marker['trial_id'] != spec.semantic_id() or record.status != 'completed'
            or set(marker['files']) != SIGNAL_FILES | {'run.json'} or set(record.artifacts) != SIGNAL_FILES):
        raise ResearchError('Invalid signal completion marker')
    verify_files(directory, marker['files'])
    verify_files(directory, record.artifacts)
    bundle = parse(DataBundle, read_json(safe_file(directory, 'bundle.json')))
    if bundle.bundle_id != spec.bundle_id:
        raise ResearchError('Archived bundle/spec mismatch')
    metrics, _ = validate_signal_results(directory, spec, bundle)
    return base | {'effective_status': 'completed', 'artifacts_verified': True,
                   'result_hash': file_hash(directory/'COMPLETE.json'), 'comparison_id': spec.comparison_id(),
                   'metrics': metrics, 'report': str(directory/'report.txt')}


def reconcile_signal(registry_path, directory):
    evidence = inspect_signal(directory)
    with Registry(registry_path) as registry:
        registry.verify_attempt(evidence)
        if evidence['effective_status'] == 'completed':
            registry.finish(evidence['attempt_id'], complete=evidence)
        elif evidence['effective_status'] == 'failed':
            registry.finish(evidence['attempt_id'], failure=evidence['failure'])
        else:
            raise ResearchError('Attempt outcome is unresolved; do not submit a duplicate or erase it')
    return evidence


def compare_signals(directories):
    if len(directories) < 2:
        raise ResearchError('Comparison requires at least two completed studies')
    evidence = [inspect_signal(p) for p in directories]
    if any(not e['artifacts_verified'] for e in evidence) or len({e['comparison_id'] for e in evidence}) != 1:
        raise ResearchError('Reject incompatible bundle, label, fold, numerical or implementation versions')
    tables = [read_parquet(Path(p)/'predictions.parquet') for p in directories]
    minimum = parse(SignalSpec, read_json(Path(directories[0])/'spec.json')).min_cross_section
    comparisons = []
    for a in range(len(tables)):
        for b in range(a+1, len(tables)):
            left = tables[a].loc[tables[a].role.eq('validation'), ['fold']+KEYS+['score']]
            right = tables[b].loc[tables[b].role.eq('validation'), ['fold']+KEYS+['score']]
            joined = left.merge(right, on=['fold']+KEYS, suffixes=('_a', '_b'), validate='one_to_one', how='outer')
            rows = []
            for (fold, date), group in joined.groupby(['fold', 'date']):
                correlation, reason = rank_correlation(group.score_a, group.score_b, minimum)
                rows.append({'fold': fold, 'date': str(date.date()), 'score_overlap': int(group[['score_a', 'score_b']].notna().all(axis=1).sum()),
                             'scored_a': int(group.score_a.notna().sum()), 'scored_b': int(group.score_b.notna().sum()),
                             'rank_correlation': correlation, 'status': reason})
            comparisons.append({'a': evidence[a]['trial_id'], 'b': evidence[b]['trial_id'], 'sessions': rows})
    return {'kind': 'descriptive_signal_comparison_v1', 'comparison_id': evidence[0]['comparison_id'],
            'selection': 'none', 'outcome_filter_for_score_overlap': False,
            'studies': [{'trial_id': e['trial_id'], 'folds': e['metrics']['folds']} for e in evidence],
            'score_comparisons': comparisons}
