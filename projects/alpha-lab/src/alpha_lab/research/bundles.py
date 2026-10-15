"""Coordinator materialization of bounded panels. Jobs never follow raw paths."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_lab.market_data.sharadar_store import atomic_json
from .contracts import (Artifact, BUNDLE_FILES, DataBundle, DEVELOPMENT_END, IdentityMap,
                        PANEL_FILES, ResearchError, digest, file_hash, iso_day, parse, read_json)


def artifact(path):
    return Artifact(sha256=file_hash(path), bytes=Path(path).stat().st_size).model_dump()


def safe_file(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).name != name or path.is_symlink() or path.resolve().parent != root or not path.is_file():
        raise ResearchError('Artifact is absent or outside its declared directory')
    return path


def verify_files(root, files):
    for name, expected in files.items():
        expected = parse(Artifact, expected) if isinstance(expected, dict) else expected
        path = safe_file(root, name)
        if path.stat().st_size != expected.bytes or file_hash(path) != expected.sha256:
            raise ResearchError('Artifact hash/size mismatch')


def _check_arrays(raw, calendar, identities, end):
    expected = {'dates', 'tickers', 'returns', 'price_observed', 'spy'}
    if set(raw) != expected or any(value.dtype.hasobject for value in raw.values()):
        raise ResearchError('Unsupported return arrays')
    dates = pd.DatetimeIndex(raw['dates'])
    tickers = raw['tickers'].tolist()
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing or len(dates) < 2
            or dates[-1] > pd.Timestamp(end) or not dates.equals(pd.DatetimeIndex(calendar.date))
            or tickers != [s.ticker for s in identities.securities]):
        raise ResearchError('Bundle date/security coverage is invalid')
    if (raw['returns'].shape != (len(dates), len(tickers)) or raw['spy'].shape != (len(dates),)
            or raw['price_observed'].shape != raw['returns'].shape or raw['price_observed'].dtype != bool):
        raise ResearchError('Bundle array dimensions are invalid')
    returns, spy, observed = raw['returns'], raw['spy'], raw['price_observed']
    if (returns.dtype.kind != 'f' or spy.dtype.kind != 'f' or np.isinf(returns).any()
            or np.any(returns < -1) or not np.isfinite(spy[1:]).all() or np.isinf(spy).any()
            or np.any(spy < -1) or np.any(np.isfinite(returns[1:]) & ~(observed[1:] & observed[:-1]))):
        raise ResearchError('Bundle returns or quote-observation mask are invalid')
    return dates, tickers


def _inspect_panel(root, bundle):
    """Read only bounded materialized files; validates actual contents, not labels."""
    panel = read_json(safe_file(root, 'panel.json'))
    if panel.get('schema_version') != 4 or panel.get('start') != bundle.start or panel.get('end') != bundle.end:
        raise ResearchError('Bundle panel bounds/schema differ from its contract')
    if panel.get('files') != {n: bundle.files[n].sha256 for n in PANEL_FILES}:
        raise ResearchError('Panel source hashes differ from bundle hashes')
    identities = parse(IdentityMap, read_json(safe_file(root, 'identities.json')))
    if identities.reference_sha256 != bundle.sources['tickers'].sha256:
        raise ResearchError('Identity reference differs from pinned source')
    calendar = pd.read_csv(safe_file(root, 'calendar.csv'), parse_dates=['date'])
    with np.load(safe_file(root, 'returns.npz'), allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    dates, tickers = _check_arrays(raw, calendar, identities, bundle.end)
    with duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB'}) as db:
        features = db.execute('SELECT * FROM read_parquet(?)', [str(safe_file(root, 'features.parquet'))]).df()
        actions = db.execute('SELECT * FROM read_parquet(?)', [str(safe_file(root, 'actions.parquet'))]).df()
    if (len(features) != bundle.rows or len(dates) != bundle.sessions or len(tickers) != bundle.securities
            or features.empty or not pd.to_datetime(features.date).between(bundle.start, bundle.end).all()
            or set(features.ticker) != set(tickers)
            or features.duplicated(['ticker', 'date']).any()
            or not actions.ticker.isin(tickers).all()
            or not pd.to_datetime(actions.date).le(pd.Timestamp(bundle.end)).all()):
        raise ResearchError('Bundle row/date/security counts are inconsistent')
    if not {'ticker', 'date', 'beta', 'filing_date'} <= set(features):
        raise ResearchError('Bundle feature dependencies are incomplete')
    if (pd.to_datetime(features.filing_date) > pd.to_datetime(features.date)).any() or (
            'available_date' in features and
            (pd.to_datetime(features.available_date) > pd.to_datetime(features.date)).any()):
        raise ResearchError('Bundle accounting availability is in the future')
    # sources.json has only identifiers and hashes; no full-history path access.
    refs = read_json(safe_file(root, 'sources.json'))
    if refs != {name: source.model_dump() for name, source in bundle.sources.items()}:
        raise ResearchError('Unexpected source provenance fields')


def load_bundle(root):
    root = Path(root).resolve()
    complete = read_json(safe_file(root, 'COMPLETE.json'))
    if not isinstance(complete, dict) or set(complete) != {'kind', 'bundle_id', 'files'} or complete['kind'] != 'development_bundle':
        raise ResearchError('Invalid bundle completion marker')
    if not isinstance(complete['files'], dict) or set(complete['files']) != BUNDLE_FILES | {'bundle.json'}:
        raise ResearchError('Incomplete bundle artifact set')
    verify_files(root, complete['files'])
    bundle = parse(DataBundle, read_json(safe_file(root, 'bundle.json')))
    if complete['bundle_id'] != bundle.bundle_id:
        raise ResearchError('Bundle completion identity mismatch')
    verify_files(root, bundle.files)
    _inspect_panel(root, bundle)
    return bundle


def pin_panel(panel, destination, *, end):
    """Pin only legacy development observations; no backtest or provider calls."""
    try:
        iso_day(end)
    except (ValueError, TypeError):
        raise ResearchError('Supply an ISO development end date') from None
    if not '2014-01-01' <= end <= DEVELOPMENT_END:
        raise ResearchError('Development end must be within 2014-2022; holdout denied')
    panel, destination = Path(panel).resolve(), Path(destination).resolve()
    if destination == panel or destination.is_relative_to(panel):
        raise ResearchError('Materialize the bundle outside the source panel')
    source_manifest = read_json(safe_file(panel, 'panel.json'))
    if (source_manifest.get('schema_version') != 4 or source_manifest.get('start') != '2014-01-01'
            or source_manifest.get('end', '') < end or set(source_manifest.get('files', {})) != PANEL_FILES):
        raise ResearchError('P1 requires a hashed schema-4 legacy panel covering the requested dates')
    source_manifest_hash = file_hash(panel/'panel.json')
    source_files = {name: {'sha256': sha, 'bytes': safe_file(panel, name).stat().st_size}
                    for name, sha in source_manifest['files'].items()}
    verify_files(panel, source_files)
    provenance = read_json(panel/'sources.json')
    if set(provenance) != {'stocks', 'funds', 'daily', 'fundamentals', 'tickers', 'actions'}:
        raise ResearchError('Incomplete upstream source identity')
    refs = {name: {'version': p['version'], 'sha256': p['parquet_sha256']} for name, p in provenance.items()}
    # Verify the small identity reference separately, without rehashing the huge
    # raw price bundle or treating its current capture as historical knowledge.
    metadata = Path(provenance['tickers']['path'])
    if file_hash(metadata) != refs['tickers']['sha256']:
        raise ResearchError('Pinned security reference hash mismatch')
    with duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '256MB'}) as db:
        features = db.execute('SELECT * FROM read_parquet(?) WHERE date <= CAST(? AS DATE) ORDER BY date,ticker',
                              [str(panel/'features.parquet'), end]).df()
        tickers = sorted(features.ticker.unique())
        db.register('selected', pd.DataFrame({'ticker': tickers}))
        identity_rows = db.execute('SELECT ticker,permaticker FROM read_parquet(?) JOIN selected USING(ticker) '
                                  'WHERE "table"=\'SEP\' ORDER BY ticker', [str(metadata)]).fetchall()
        actions = db.execute('SELECT * FROM read_parquet(?) WHERE date <= CAST(? AS DATE) '
                             'AND ticker IN (SELECT ticker FROM selected)', [str(panel/'actions.parquet'), end]).df()
    identity = parse(IdentityMap, {'namespace': 'sharadar_permaticker_share_class',
                                  'reference_sha256': refs['tickers']['sha256'],
                                  'historical_eligibility_attested': False,
                                  'securities': [{'ticker': ticker, 'security_id': value} for ticker, value in identity_rows]})
    if [s.ticker for s in identity.securities] != tickers or len(tickers) < 30:
        raise ResearchError('A unique stable security ID is required for every retained ticker')
    with np.load(panel/'returns.npz', allow_pickle=False) as original:
        if set(original.files) != {'dates', 'tickers', 'returns', 'price_observed', 'spy'}:
            raise ResearchError('Unsupported source return arrays')
        dates = pd.DatetimeIndex(original['dates'])
        keep = dates <= pd.Timestamp(end)
        original_tickers = original['tickers'].tolist()
        columns = [original_tickers.index(t) for t in tickers]
        raw = {'dates': original['dates'][keep], 'tickers': np.asarray(tickers, dtype='U32'),
               'returns': original['returns'][keep][:, columns],
               'price_observed': original['price_observed'][keep][:, columns], 'spy': original['spy'][keep]}
    calendar = pd.read_csv(panel/'calendar.csv', parse_dates=['date'])
    calendar = calendar.loc[calendar.date <= pd.Timestamp(end)]
    _check_arrays(raw, calendar, identity, end)
    # Recheck source versions after reading; failure never publishes a complete bundle.
    verify_files(panel, source_files)
    if file_hash(panel/'panel.json') != source_manifest_hash:
        raise ResearchError('Source manifest changed during materialization')
    if file_hash(metadata) != refs['tickers']['sha256']:
        raise ResearchError('Security reference changed during materialization')
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'INCOMPLETE').write_text('Require COMPLETE.json; no resume or overwrite\n')
    with duckdb.connect(':memory:') as db:
        db.from_df(features).write_parquet(str(destination/'features.parquet'))
        db.from_df(actions).write_parquet(str(destination/'actions.parquet'))
    calendar.to_csv(destination/'calendar.csv', index=False)
    np.savez_compressed(destination/'returns.npz', **raw)
    atomic_json(destination/'identities.json', identity.model_dump())
    atomic_json(destination/'sources.json', refs)
    manifest = source_manifest | {'end': end, 'historical_tickers': len(tickers), 'monthly_rows': len(features),
                                  'months': int(features.date.nunique()),
                                  'calendar': source_manifest['calendar'] | {'end': str(calendar.date.iloc[-1].date()),
                                                                           'sessions': len(calendar),
                                                                           'early_close_sessions': int(calendar.early_close.sum())}}
    manifest['files'] = {name: file_hash(destination/name) for name in sorted(PANEL_FILES)}
    atomic_json(destination/'panel.json', manifest)
    body = {'schema_version': 1, 'start': manifest['start'], 'end': end,
            'source_panel_sha256': source_manifest_hash, 'sources': refs,
            'files': {name: artifact(destination/name) for name in sorted(BUNDLE_FILES)},
            'history_mode': 'current_vendor_snapshot_reconstruction',
            'availability_policy': 'legacy_art_filing_plus_2_calendar_days_unattested',
            'universe_policy': 'legacy_panel_membership_static_metadata_unattested',
            'return_basis': 'vendor_adjusted_simple_total_returns_unattested',
            'identity_namespace': identity.namespace, 'calendar_id': 'XNYS_modeled_sessions',
            'calendar_sha256': file_hash(destination/'calendar.csv'),
            'rows': len(features), 'sessions': len(calendar), 'securities': len(tickers),
            'point_in_time_attested': False, 'terminal_actions_reconciled': False}
    bundle = parse(DataBundle, body | {'bundle_id': digest(body)})
    atomic_json(destination/'bundle.json', bundle.model_dump())
    _inspect_panel(destination, bundle)
    (destination/'INCOMPLETE').unlink()
    atomic_json(destination/'COMPLETE.json', {'kind': 'development_bundle', 'bundle_id': bundle.bundle_id,
                                            'files': {name: artifact(destination/name) for name in sorted(BUNDLE_FILES | {'bundle.json'})}})
    return load_bundle(destination)
