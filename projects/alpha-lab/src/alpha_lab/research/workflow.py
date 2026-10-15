"""One local development-only execution, with fail-closed artifact publication."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import psutil

from alpha_lab.local_process import start_process
from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact, load_bundle, safe_file, verify_files
from .contracts import (DAILY_COLUMNS, DataBundle, ResearchError, ResearchSpec, RunRecord,
                        canonical, file_hash, parse, read_json)
from .references import accounting_reference
from .runtime import child_environment, environment_identity, implementation_identity

RESULT_FILES = {'metrics.json', 'daily.csv', 'runtime.json'}
RUN_FILES = RESULT_FILES | {'spec.json', 'bundle.json', 'reference-cases.json', 'report.txt'}


def now():
    return datetime.now(timezone.utc).isoformat()


def compatible(spec, bundle):
    if spec.bundle_id != bundle.bundle_id:
        raise ResearchError('Spec does not identify this pinned bundle')
    if spec.implementation != implementation_identity():
        raise ResearchError('Implementation changed; resolve a new explicit spec')
    if spec.environment != environment_identity():
        raise ResearchError('Environment changed; resolve a new explicit spec')


def validate_dependencies(bundle_path, spec):
    import duckdb
    with duckdb.connect(':memory:') as db:
        columns = db.execute('DESCRIBE SELECT * FROM read_parquet(?)',
                             [str(Path(bundle_path)/'features.parquet')]).fetchall()
    kinds = dict((row[0], row[1]) for row in columns)
    for name in list(spec.strategy.factors) + ['beta']:
        if kinds.get(name) not in {'FLOAT', 'DOUBLE'}:
            raise ResearchError('Strategy requires absent or nonnumeric panel features')


def make_spec(bundle, *, hypothesis_id, family_id, title, strategy, cost_bps, borrow_bps_annual,
              terminal_stress, seed, threads, timeout_seconds, max_output_bytes, memory_limit_mb):
    bundle = load_bundle(bundle)
    return parse(ResearchSpec, {
        'schema_version': 1, 'hypothesis_id': hypothesis_id, 'family_id': family_id, 'title': title,
        'bundle_id': bundle.bundle_id, 'strategy': strategy,
        'methodology': 'legacy_market_neutral_v1', 'universe_policy': bundle.universe_policy,
        'feature_policy': 'legacy_percentile_ranks_requires_filings',
        'fold_policy': 'train_2014_2019_validation_2020_2022',
        'timing_policy': 'month_end_signal_next_close_daily_target_maintenance',
        'risk_policy': 'intercept_spy_beta_projection_gross1_namecap2pct',
        'cash_policy': 'no_cash_interest_or_financing_zero_rf_sharpe',
        'cost_bps': cost_bps, 'borrow_bps_annual': borrow_bps_annual,
        'borrow_day_count': '252_sessions_legacy_assumption', 'terminal_stress': terminal_stress,
        'numerical_policy': 'cpu_float64_fixed_threads', 'seed': seed, 'threads': threads,
        'timeout_seconds': timeout_seconds, 'max_output_bytes': max_output_bytes,
        'memory_limit_mb': memory_limit_mb,
        'implementation': implementation_identity(), 'environment': environment_identity(),
    })


def validate_results(directory, bundle, spec):
    directory = Path(directory)
    envelope = read_json(safe_file(directory, 'metrics.json'))
    if (set(envelope) != {'kind', 'spec_id', 'bundle_id', 'legacy_result'}
            or envelope['kind'] != 'provisional_legacy_development_result'
            or envelope['spec_id'] != spec.semantic_id() or envelope['bundle_id'] != bundle.bundle_id):
        raise ResearchError('Result identity/schema mismatch')
    result = envelope['legacy_result']
    if (not isinstance(result, dict) or 'holdout' in result
            or result.get('spec') != spec.strategy.model_dump()
            or result.get('cost_bps') != spec.cost_bps
            or result.get('borrow_bps_annual') != spec.borrow_bps_annual
            or result.get('terminal_stress_fraction') != spec.terminal_stress
            or result.get('investment_ready') is not False or result.get('research_only') is not True
            or result.get('terminal_actions_reconciled') is not False):
        raise ResearchError('Result policies/readiness differ from the contract')
    canonical(result)  # Reject any nonfinite JSON result, including nested metrics.
    daily = pd.read_csv(safe_file(directory, 'daily.csv'))
    if daily.columns.tolist() != DAILY_COLUMNS or daily.empty:
        raise ResearchError('Daily result schema is incomplete')
    dates = pd.to_datetime(daily.date, errors='coerce')
    if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
            or not dates.between(bundle.start, bundle.end).all()
            or not np.isfinite(daily[DAILY_COLUMNS[1:]].to_numpy(dtype=float)).all()
            or (daily[['turnover', 'missing_exposure', 'gross_exposure']] < -1e-12).any().any()
            or (daily.gross_exposure > 1 + 1e-10).any()):
        raise ResearchError('Daily result values/dates are invalid')
    runtime = read_json(safe_file(directory, 'runtime.json'))
    expected_keys = {'numerical_policy', 'threads', 'seed', 'provider_keys_present', 'proxy_present',
                     'environment', 'implementation', 'isolation', 'resource_guards'}
    if (set(runtime) != expected_keys or runtime['threads'] != spec.threads or runtime['seed'] != spec.seed
            or runtime['numerical_policy'] != spec.numerical_policy
            or runtime['provider_keys_present'] is not False or runtime['proxy_present'] is not False
            or runtime['environment'] != spec.environment or runtime['implementation'] != spec.implementation
            or runtime['isolation'] != 'credential_filtered_process_not_filesystem_or_network_sandbox'):
        raise ResearchError('Worker runtime evidence differs from the requested policy')
    return result, daily


def _report(record, spec, bundle, result, rows):
    reference = accounting_reference()
    return ('US equity research — P1 legacy compatibility workflow\n\n'
            f'Run: {record.run_id}\nSpec: {record.spec_id}\nBundle: {bundle.bundle_id}\n'
            f'Hypothesis/family: {spec.hypothesis_id} / {spec.family_id}\n'
            f'Methodology: {spec.methodology}\nDevelopment bounds: {bundle.start}..{bundle.end}\n'
            f'Daily rows: {rows}\nCPU threads: {spec.threads}; seed: {spec.seed}\n'
            'Operational result: validated completed artifact set, subject to COMPLETE.json\n'
            'Scientific disposition: PROVISIONAL LEGACY RESEARCH; not strategy acceptance\n'
            'LLM/provider calls: 0; original holdout not evaluated\n'
            'Isolation: filtered environment and process lifetime containment; no filesystem/network sandbox\n'
            'Resource guards: wall time/output caps and sampled process-tree RSS, not hard OS RAM isolation\n\n'
            f'Timing: {spec.timing_policy}\nRisk: {spec.risk_policy}\nCash: {spec.cash_policy}\n'
            f'One-way cost: {spec.cost_bps} bps; borrow: {spec.borrow_bps_annual} bps/{spec.borrow_day_count}\n'
            f'Terminal stress fraction: {spec.terminal_stress}\n'
            f'Input history: {bundle.history_mode}; availability: {bundle.availability_policy}\n'
            f'Universe: {bundle.universe_policy}\n'
            'No historical eligibility, arrival, terminal settlement or borrow attestation is supplied.\n\n'
            'Legacy assumptions and limitations:\n' + ''.join('- '+s+'\n' for s in reference['legacy_limitations']) +
            '\nIndependent accounting expectations are in reference-cases.json. They are specifications;\n'
            'the new share/cash/action ledger is deferred to P3 and has NOT passed those cases.\n\n'
            'Development metrics (legacy definitions):\n' +
            json.dumps({k: result[k] for k in ('train', 'validation') if k in result}, indent=2) + '\n')


def run_research(bundle_path, spec_path, destination):
    spec = parse(ResearchSpec, read_json(spec_path))
    destination = Path(destination).resolve()
    bundle_path = Path(bundle_path).resolve()
    if destination == bundle_path or destination.is_relative_to(bundle_path):
        raise ResearchError('Keep runs outside immutable input bundles')
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination/'spec.json', spec.model_dump())
    record = RunRecord(schema_version=1, run_id=uuid.uuid4().hex, spec_id=spec.semantic_id(),
                       bundle_id=spec.bundle_id, status='running', stage='validating',
                       scientific_disposition='provisional_legacy_research_not_accepted',
                       started_at=now(), updated_at=now(), finished_at=None, failure_code=None,
                       parent_attempt=None, llm_calls=0, artifacts={}, input_validation='pending',
                       worker_pid=None, peak_rss_mb=0.,
                       isolation='credential_filtered_process_not_filesystem_or_network_sandbox')
    atomic_json(destination/'run.json', record.model_dump())
    (destination/'INCOMPLETE').write_text('Require validated COMPLETE.json; do not retry in this directory\n')
    proc = None
    failure = 'input_validation'
    try:
        bundle = load_bundle(bundle_path)
        compatible(spec, bundle)
        validate_dependencies(bundle_path, spec)
        atomic_json(destination/'bundle.json', bundle.model_dump())
        work = destination/'worker-output'
        work.mkdir()
        record = record.model_copy(update={'stage': 'evaluating', 'input_validation': 'passed', 'updated_at': now()})
        atomic_json(destination/'run.json', record.model_dump())
        failure = 'execution_failure'
        started = time.monotonic()
        with (destination/'worker.out').open('w', encoding='utf-8') as log:
            proc = start_process([sys.executable, '-s', '-P', '-m', 'alpha_lab.research.worker',
                                  '--bundle', str(bundle_path), '--spec', str(destination/'spec.json'),
                                  '--output', str(work)], cwd=str(destination), env=child_environment(spec),
                                 stdout=log, stderr=subprocess.STDOUT, text=True)
            record = record.model_copy(update={'worker_pid': proc.pid})
            while proc.poll() is None:
                try:
                    root = psutil.Process(proc.pid)
                    processes = [root] + root.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in processes if p.is_running()) / 1024**2
                except psutil.Error:
                    rss = record.peak_rss_mb
                record = record.model_copy(update={'peak_rss_mb': max(float(rss), record.peak_rss_mb), 'updated_at': now()})
                atomic_json(destination/'run.json', record.model_dump())
                size = sum(p.stat().st_size for p in work.iterdir() if p.is_file()) + (destination/'worker.out').stat().st_size
                if time.monotonic() - started > spec.timeout_seconds:
                    failure = 'timeout'
                    raise ResearchError('Research worker timed out')
                if size > spec.max_output_bytes:
                    failure = 'output_limit'
                    raise ResearchError('Research worker output cap exceeded')
                if rss > spec.memory_limit_mb:
                    failure = 'memory_limit'
                    raise ResearchError('Research worker sampled RSS cap exceeded')
                time.sleep(.25)
            if proc.wait() != 0:
                raise ResearchError('Research worker failed; inspect its local run evidence')
            if time.monotonic() - started > spec.timeout_seconds:
                failure = 'timeout'
                raise ResearchError('Research worker exceeded the request deadline')
        failure = 'publication_failure'
        if {p.name for p in work.iterdir()} != RESULT_FILES:
            raise ResearchError('Worker did not produce the exact required artifact set')
        result, daily = validate_results(work, bundle, spec)
        expected_dates = pd.read_csv(bundle_path/'calendar.csv', parse_dates=['date']).date
        expected_dates = expected_dates.loc[expected_dates >= pd.Timestamp(bundle.start)].reset_index(drop=True)
        if not pd.to_datetime(daily.date).equals(expected_dates):
            raise ResearchError('Daily output does not cover every requested session')
        # Detect changed inputs or evaluator code while the worker was running.
        compatible(spec, load_bundle(bundle_path))
        record = record.model_copy(update={'stage': 'publishing', 'updated_at': now()})
        atomic_json(destination/'run.json', record.model_dump())
        for name in sorted(RESULT_FILES):
            os.replace(work/name, destination/name)
        work.rmdir()
        atomic_json(destination/'reference-cases.json', accounting_reference())
        (destination/'report.txt').write_text(_report(record, spec, bundle, result, len(daily)), encoding='utf-8')
        if sum((destination/n).stat().st_size for n in RUN_FILES | {'worker.out'}) > spec.max_output_bytes:
            failure = 'output_limit'
            raise ResearchError('Published artifact set exceeds output cap')
        artifacts = {name: artifact(destination/name) for name in sorted(RUN_FILES)}
        record = parse(RunRecord, record.model_dump() | {'status': 'completed', 'stage': 'finished',
                            'updated_at': now(), 'finished_at': now(), 'artifacts': artifacts})
        atomic_json(destination/'run.json', record.model_dump())
        (destination/'INCOMPLETE').unlink()
        atomic_json(destination/'COMPLETE.json', {'kind': 'research_run', 'spec_id': spec.semantic_id(),
                                                'files': {name: artifact(destination/name) for name in sorted(RUN_FILES | {'run.json'})}})
        inspect_run(destination)  # Final read-only verification, not just exit-code success.
        return record
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            proc.terminate_tree()
        stopped = isinstance(error, KeyboardInterrupt)
        record = parse(RunRecord, record.model_dump() | {'status': 'stopped' if stopped else 'failed',
                      'stage': 'finished', 'updated_at': now(), 'finished_at': now(),
                      'failure_code': 'interrupted' if stopped else failure})
        atomic_json(destination/'run.json', record.model_dump())
        (destination/'COMPLETE.json').unlink(missing_ok=True)
        raise ResearchError('Research attempt stopped' if stopped else 'Research attempt failed: '+failure) from None
    finally:
        if proc is not None:
            proc.close()


def inspect_run(directory):
    """Read/verify only; does not launch, repair, retry, load dotenv or write."""
    directory = Path(directory).resolve()
    record = parse(RunRecord, read_json(safe_file(directory, 'run.json')))
    if not (directory/'COMPLETE.json').is_file():
        state = {'completed': 'incomplete', 'running': 'unfinished_requires_reconciliation'}.get(record.status, record.status)
        return {'effective_status': state,
                'artifacts_verified': False, 'record': record.model_dump()}
    complete = read_json(safe_file(directory, 'COMPLETE.json'))
    if (not isinstance(complete, dict) or set(complete) != {'kind', 'spec_id', 'files'} or complete['kind'] != 'research_run'
            or not isinstance(complete['files'], dict)
            or set(complete['files']) != RUN_FILES | {'run.json'} or record.status != 'completed'):
        raise ResearchError('Invalid run completion marker')
    verify_files(directory, complete['files'])
    verify_files(directory, record.artifacts)
    if set(record.artifacts) != RUN_FILES:
        raise ResearchError('Incomplete run artifact inventory')
    spec = parse(ResearchSpec, read_json(safe_file(directory, 'spec.json')))
    bundle = parse(DataBundle, read_json(safe_file(directory, 'bundle.json')))
    if (spec.semantic_id() != record.spec_id or spec.bundle_id != bundle.bundle_id
            or record.bundle_id != bundle.bundle_id or complete['spec_id'] != record.spec_id):
        raise ResearchError('Run specification/bundle identity mismatch')
    validate_results(directory, bundle, spec)
    return {'effective_status': 'completed', 'artifacts_verified': True, 'record': record.model_dump(),
            'policies': {key: value for key, value in spec.model_dump().items() if key.endswith('policy')},
            'bounds': [bundle.start, bundle.end], 'report': str(directory/'report.txt')}
