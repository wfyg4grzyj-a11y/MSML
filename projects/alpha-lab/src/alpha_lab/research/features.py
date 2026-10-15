"""Independently gated, raw feature artifacts; no fitted transform is cached here."""
from __future__ import annotations

from pathlib import Path
from importlib.metadata import version
import os
import uuid

import duckdb
import numpy as np
import pandas as pd

from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.workspace_lock import WorkspaceLock, WorkspaceOwnedError
from .bundles import artifact, safe_file, verify_files
from .contracts import ResearchError, digest, file_hash, read_json
from .signal_contracts import FEATURES

KEYS = ['date', 'ticker', 'security_id']
FEATURE_COLUMNS = KEYS + ['value', 'source_member', 'feature_available', 'availability_reason', 'information_date']


def read_parquet(path):
    with duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB'}) as db:
        return db.execute('SELECT * FROM read_parquet(?)', [str(path)]).df()


def write_parquet(frame, path):
    with duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB'}) as db:
        db.from_df(frame).write_parquet(str(path))


def feature_definition(name):
    if name not in FEATURES:
        raise ResearchError('Unsupported feature')
    kind, units, dependencies = FEATURES[name]
    return {'name': name, 'version': 1, 'source_column': name, 'kind': kind, 'units': units,
            'dependencies': dependencies, 'source': 'pinned_legacy_prepared_panel',
            'availability': ('decision_session_close' if kind == 'price' else
                             'filing_plus_2_calendar_days_age_at_most_550_days'),
            'historical_arrival_attested': False, 'missing_policy': 'retain_member_missing_value'}


def feature_identity(bundle, name):
    definition = feature_definition(name)
    body = {'bundle_id': bundle.bundle_id, 'definition': definition,
            'implementation': {p.name: file_hash(p) for p in
                               [Path(__file__), Path(__file__).with_name('signal_contracts.py')]},
            'environment': {k: version(k) for k in ('numpy', 'pandas', 'duckdb')}}
    return digest(body), body


def prepare_feature(source, name, identities):
    """Feature gates never consult future outcomes or another feature's mask."""
    definition = feature_definition(name)
    if name not in source:
        raise ResearchError('Pinned panel lacks a requested feature dependency')
    out = source[['date', 'ticker']].copy()
    out['date'] = pd.to_datetime(out.date)
    out['security_id'] = out.ticker.map(identities)
    if out.security_id.isna().any() or out.duplicated(KEYS).any():
        raise ResearchError('Feature has ambiguous security/date identity')
    out['security_id'] = out.security_id.astype('int64')
    value = pd.to_numeric(source[name], errors='coerce').astype('float64')
    reason = pd.Series('available', index=source.index)
    reason.loc[~np.isfinite(value)] = 'missing_or_nonfinite_source'
    information = out.date.copy()
    if definition['kind'] == 'fundamental':
        # An absent timestamp cannot be inferred from a nonmissing ratio.
        filing = pd.to_datetime(source.get('filing_date', pd.Series(pd.NaT, index=source.index)))
        information = pd.to_datetime(source.get('available_date', pd.Series(pd.NaT, index=source.index)))
        reason.loc[(out.date - information).dt.days > 550] = 'stale_information'
        reason.loc[information > out.date] = 'future_information'
        reason.loc[information.ne(filing + pd.Timedelta(days=2))] = 'inconsistent_availability'
        reason.loc[filing.isna() | information.isna()] = 'undated_information'
    available = reason.eq('available')
    out['value'] = value.where(available)
    out['source_member'] = True
    out['feature_available'] = available
    out['availability_reason'] = reason
    out['information_date'] = information
    return out[FEATURE_COLUMNS].sort_values(KEYS).reset_index(drop=True)


def inspect_feature(directory, expected_id, expected_body):
    marker = read_json(safe_file(directory, 'COMPLETE.json'))
    if set(marker) != {'kind', 'feature_id', 'files'} or marker['kind'] != 'raw_feature_v1' or marker['feature_id'] != expected_id:
        raise ResearchError('Invalid feature completion identity')
    if set(marker['files']) != {'manifest.json', 'values.parquet'}:
        raise ResearchError('Incomplete feature artifact inventory')
    verify_files(directory, marker['files'])
    manifest = read_json(safe_file(directory, 'manifest.json'))
    if manifest.get('identity') != expected_body or digest(manifest['identity']) != expected_id:
        raise ResearchError('Feature computation identity mismatch')
    result = read_parquet(safe_file(directory, 'values.parquet'))
    if (result.columns.tolist() != FEATURE_COLUMNS or len(result) != manifest.get('rows')
            or result.duplicated(KEYS).any() or not result.source_member.all()
            or not result.feature_available.equals(pd.Series(np.isfinite(result.value), name='feature_available'))
            or not result.feature_available.equals(result.availability_reason.eq('available'))):
        raise ResearchError('Invalid raw feature coverage/value schema')
    return result


def cached_feature(bundle_path, bundle, name, cache_root, max_bytes):
    """One artifact per computation; concurrent builds fail visibly, never overwrite."""
    cache_root = Path(cache_root).resolve()
    bundle_path = Path(bundle_path).resolve()
    if cache_root == bundle_path or cache_root.is_relative_to(bundle_path):
        raise ResearchError('Feature cache must be outside the immutable bundle')
    key, body = feature_identity(bundle, name)
    target = cache_root/key
    try:
        with WorkspaceLock(cache_root/'locks'/key):
            if target.exists():
                frame = inspect_feature(target, key, body)
                if sum(p.stat().st_size for p in target.iterdir()) > max_bytes:
                    raise ResearchError('Existing feature artifact exceeds the requested per-artifact cap')
                return frame, key
            # Project only the requested inputs; do not read labels or the raw vendor store.
            columns = ['date', 'ticker', name]
            if FEATURES[name][0] == 'fundamental':
                columns += ['filing_date', 'available_date']
            with duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB'}) as db:
                # Identifiers come exclusively from the fixed catalog.
                source = db.execute('SELECT '+','.join(columns)+' FROM read_parquet(?) ORDER BY date,ticker',
                                    [str(bundle_path/'features.parquet')]).df()
            identities = {s['ticker']: s['security_id'] for s in read_json(bundle_path/'identities.json')['securities']}
            frame = prepare_feature(source, name, identities)
            if len(frame) != bundle.rows or not frame.date.between(bundle.start, bundle.end).all():
                raise ResearchError('Feature rows differ from the pinned development population')
            # Keep staging paths shorter than the final hash path on Windows.
            stage = cache_root/('incomplete-'+uuid.uuid4().hex)
            stage.mkdir(parents=True, exist_ok=False)
            write_parquet(frame, stage/'values.parquet')
            atomic_json(stage/'manifest.json', {'identity': body, 'rows': len(frame)})
            atomic_json(stage/'COMPLETE.json', {'kind': 'raw_feature_v1', 'feature_id': key,
                        'files': {n: artifact(stage/n) for n in ('manifest.json', 'values.parquet')}})
            if sum(p.stat().st_size for p in stage.iterdir()) > max_bytes:
                raise ResearchError('Feature artifact exceeds its per-artifact output cap')
            inspect_feature(stage, key, body)
            os.rename(stage, target)
            return frame, key
    except WorkspaceOwnedError:
        raise ResearchError('Feature computation is already owned; inspect it before retrying') from None
