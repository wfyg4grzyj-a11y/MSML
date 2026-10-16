"""Immutable terminal-action inputs and signed total-return-unit conversions.

The bundle is an ex-post accounting aid.  It does not attest when action terms
were known, and it never exposes action terms to signal or universe formation.
Only standard Sharadar USD cash and common-stock exchange legs are normalized.
All other terminal packages remain explicit unresolved events.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pandas as pd

from .contracts import ResearchError, digest, file_hash, read_json


PACKAGE_KEYS = {
    'package_id', 'parent_security_id', 'parent_ticker', 'action_date', 'classification',
    'cash_usd_per_raw_share', 'stock_legs', 'source_action_types', 'source_rows_sha256',
    'unresolved_reasons',
}
LEG_KEYS = {'recipient_id', 'recipient_ticker', 'ratio'}
MANIFEST_KEYS = {
    'schema_version', 'kind', 'methodology', 'bundle_id', 'parent_bundle_id', 'start', 'end',
    'point_in_time_attested', 'settlement_timing_attested', 'input_bindings', 'source_pins',
    'counts', 'files', 'implementation_sha256',
}
QUOTE_COLUMNS = [
    'security_id', 'ticker', 'date', 'raw_close_usd', 'adjusted_close_usd',
    'dollar_volume_usd',
]
STANDARD_ACTIONS = {'acquisitioncash', 'acquisitionstock', 'acquisitionby', 'acquisitionof'}
TERMINAL_ACTIONS = {'delisted', 'bankruptcyliquidation', 'voluntarydelisting'}


def _finite(value):
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _artifact(path):
    path = Path(path)
    return {'sha256': file_hash(path), 'bytes': path.stat().st_size}


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _source_path(inputs, manifest, name):
    files = manifest.get('source_files', {})
    if name in files:
        path = Path(files[name])
        return path if path.is_absolute() else (Path(inputs) / path).resolve()
    if name != 'actions' or 'stocks' not in files:
        raise ResearchError('Input manifest lacks a pinned source path')
    stocks = Path(files['stocks'])
    stocks = stocks if stocks.is_absolute() else (Path(inputs) / stocks).resolve()
    try:
        tables = stocks.parents[2]
    except IndexError:
        raise ResearchError('Cannot resolve the pinned actions snapshot') from None
    return tables / 'actions' / manifest['source_versions']['actions']['version'] / 'data.parquet'


def _input_bindings(inputs):
    inputs = Path(inputs).resolve()
    manifest_path, identities_path = inputs / 'manifest.json', inputs / 'identities.json'
    manifest = read_json(manifest_path)
    required = {'stocks', 'actions', 'tickers'}
    versions = manifest.get('source_versions')
    if not isinstance(versions, dict) or not required <= versions.keys():
        raise ResearchError('Input manifest lacks required action source pins')
    for name in required:
        ref = versions[name]
        if (not isinstance(ref, dict) or not isinstance(ref.get('version'), str)
                or not ref['version'] or not isinstance(ref.get('sha256'), str)
                or len(ref['sha256']) != 64):
            raise ResearchError('Input manifest has an invalid action source pin')
    if not isinstance(manifest.get('bundle_id'), str) or len(manifest['bundle_id']) != 64:
        raise ResearchError('Input manifest has no valid bundle identity')
    if not isinstance(manifest.get('start'), str) or not isinstance(manifest.get('end'), str):
        raise ResearchError('Input manifest lacks an explicit date scope')
    if not manifest['start'] < manifest['end'] <= '2022-12-31':
        raise ResearchError('Action bundle is restricted to the frozen pre-2023 scope')
    declared = manifest.get('files', {}).get('identities.json')
    if (not isinstance(declared, dict) or declared.get('sha256') != file_hash(identities_path)
            or declared.get('bytes') != identities_path.stat().st_size):
        raise ResearchError('Identity input differs from its parent manifest')
    paths = {name: _source_path(inputs, manifest, name).resolve() for name in required}
    pins = {}
    for name, path in paths.items():
        ref = versions[name]
        if not path.is_file() or path.is_symlink() or ref['sha256'] != file_hash(path):
            raise ResearchError('Pinned raw action source changed')
        pins[name] = {'path': str(path), 'version': str(ref['version']), 'sha256': ref['sha256']}
    identities = read_json(identities_path)
    if (not isinstance(identities, list) or not identities
            or any(not isinstance(row, dict) or 'ticker' not in row or 'security_id' not in row for row in identities)):
        raise ResearchError('Invalid initial identity inventory')
    ticker_ids = [(str(row['ticker']), str(row['security_id'])) for row in identities]
    if len(set(ticker_ids)) != len(ticker_ids) or len({x[0] for x in ticker_ids}) != len(ticker_ids):
        raise ResearchError('Initial identities are ambiguous')
    return manifest, identities, pins, {
        'manifest_sha256': file_hash(manifest_path), 'identities_sha256': file_hash(identities_path),
    }


def _verify_dictionary(path):
    frame = duckdb.connect().execute('SELECT * FROM read_parquet(?)', [str(path)]).df()
    if not {'table', 'indicator', 'unittype'} <= set(frame.columns):
        raise ResearchError('Description snapshot lacks unit metadata')
    required = {
        ('ACTIONTYPES', 'acquisitioncash'): 'USD/share',
        ('ACTIONTYPES', 'acquisitionstock'): 'ratio',
        ('ACTIONTYPES', 'acquisitionby'): 'USD millions',
        ('ACTIONTYPES', 'delisted'): 'USD millions',
        ('ACTIONTYPES', 'bankruptcyliquidation'): 'USD millions',
        ('SEP', 'closeunadj'): 'USD/share',
        ('SEP', 'closeadj'): 'USD/share',
    }
    found = {}
    for row in frame.itertuples(index=False):
        key = (str(getattr(row, 'table')), str(row.indicator))
        if key in required:
            found.setdefault(key, set()).add(str(row.unittype))
    if any(found.get(key) != {unit} for key, unit in required.items()):
        raise ResearchError('Pinned dictionary does not verify action and quote units')


def _ticker_map(tickers, initial):
    if not {'ticker', 'permaticker'} <= set(tickers.columns):
        raise ResearchError('Ticker source lacks stable share-class identities')
    mapping = {}
    for ticker, group in tickers.dropna(subset=['ticker', 'permaticker']).groupby('ticker', sort=False):
        values = {str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
                  for value in group.permaticker}
        if len(values) == 1:
            mapping[str(ticker)] = next(iter(values))
    for row in initial:
        ticker, security = str(row['ticker']), str(row['security_id'])
        if mapping.get(ticker) != security:
            raise ResearchError('Initial identity differs from the pinned ticker source')
    return mapping


def _row_records(frame):
    records = []
    for row in frame.sort_values(['action', 'contraticker', 'value'], na_position='last').itertuples(index=False):
        records.append({
            'ticker': str(row.ticker), 'date': pd.Timestamp(row.date).date().isoformat(),
            'action': str(row.action),
            'value': float(row.value) if _finite(row.value) else None,
            'contraticker': str(row.contraticker) if pd.notna(row.contraticker) else None,
        })
    return records


def _normalize_package(parent, security, day, frame, mapping):
    records = _row_records(frame)
    types = sorted({row['action'] for row in records})
    cash = [row for row in records if row['action'] == 'acquisitioncash']
    stock = [row for row in records if row['action'] == 'acquisitionstock']
    unsupported = sorted(name for name in types if name.startswith('acquisition') and name not in STANDARD_ACTIONS)
    reasons = []
    if unsupported:
        reasons.append('unsupported_acquisition_types:' + ','.join(unsupported))
    if len(cash) > 1:
        reasons.append('duplicate_cash_legs')
    cash_value = None
    if len(cash) == 1:
        if cash[0]['value'] is None or cash[0]['value'] <= 0:
            reasons.append('invalid_cash_usd_per_share')
        else:
            cash_value = cash[0]['value']
    legs, seen_tickers, seen_ids = [], set(), set()
    for row in stock:
        ticker = row['contraticker']
        if ticker in seen_tickers:
            reasons.append('duplicate_stock_recipient:' + str(ticker))
            continue
        seen_tickers.add(ticker)
        if not ticker or ticker == 'N/A' or ticker not in mapping:
            reasons.append('unmapped_stock_recipient:' + str(ticker))
            continue
        recipient = mapping[ticker]
        if recipient in seen_ids:
            reasons.append('duplicate_stock_recipient_id:' + recipient)
            continue
        seen_ids.add(recipient)
        if recipient == security:
            reasons.append('self_stock_recipient:' + ticker)
            continue
        if row['value'] is None or row['value'] <= 0:
            reasons.append('invalid_stock_ratio:' + ticker)
            continue
        legs.append({'recipient_id': recipient, 'recipient_ticker': ticker, 'ratio': row['value']})
    if not cash and not stock:
        if 'bankruptcyliquidation' in types:
            reasons.append('insolvency_terms_unknown')
        else:
            reasons.append('no_standard_consideration_terms')
    source_hash = digest(records)
    if reasons:
        classification, cash_value, legs = 'unresolved', None, []
    elif cash_value is not None and legs:
        classification = 'mixed'
    elif cash_value is not None:
        classification = 'cash'
    elif legs:
        classification = 'stock'
    else:
        classification = 'unresolved'
        reasons = ['no_standard_consideration_terms']
    identity = {'parent_security_id': security, 'parent_ticker': parent,
                'action_date': day, 'source_rows_sha256': source_hash}
    return {
        'package_id': digest(identity), **identity, 'classification': classification,
        'cash_usd_per_raw_share': cash_value, 'stock_legs': sorted(legs, key=lambda x: (x['recipient_id'], x['recipient_ticker'])),
        'source_action_types': types, 'unresolved_reasons': sorted(set(reasons)),
    }


def _validate_package(package):
    if not isinstance(package, dict) or set(package) != PACKAGE_KEYS:
        raise ResearchError('Invalid normalized action package fields')
    text_fields = ('package_id', 'parent_security_id', 'parent_ticker', 'action_date', 'source_rows_sha256')
    if any(not isinstance(package[key], str) or not package[key] for key in text_fields):
        raise ResearchError('Invalid normalized action package identity fields')
    if len(package['package_id']) != 64 or len(package['source_rows_sha256']) != 64:
        raise ResearchError('Invalid normalized action package hashes')
    if package['classification'] not in {'cash', 'stock', 'mixed', 'unresolved'}:
        raise ResearchError('Invalid normalized action classification')
    identity = {key: package[key] for key in ('parent_security_id', 'parent_ticker', 'action_date', 'source_rows_sha256')}
    if package['package_id'] != digest(identity):
        raise ResearchError('Normalized action package identity mismatch')
    try:
        if pd.Timestamp(package['action_date']).date().isoformat() != package['action_date']:
            raise ValueError
    except (TypeError, ValueError):
        raise ResearchError('Invalid normalized action date') from None
    if (not isinstance(package['source_action_types'], list)
            or any(not isinstance(value, str) or not value for value in package['source_action_types'])
            or package['source_action_types'] != sorted(set(package['source_action_types']))):
        raise ResearchError('Invalid normalized source action inventory')
    if (not isinstance(package['unresolved_reasons'], list)
            or any(not isinstance(value, str) or not value for value in package['unresolved_reasons'])
            or package['unresolved_reasons'] != sorted(set(package['unresolved_reasons']))):
        raise ResearchError('Invalid unresolved action reasons')
    legs = package['stock_legs']
    if not isinstance(legs, list) or any(not isinstance(leg, dict) or set(leg) != LEG_KEYS for leg in legs):
        raise ResearchError('Invalid normalized stock legs')
    recipients = []
    for leg in legs:
        if (not isinstance(leg['recipient_id'], str) or not leg['recipient_id']
                or not isinstance(leg['recipient_ticker'], str) or not leg['recipient_ticker']):
            raise ResearchError('Invalid normalized stock recipient')
        if not _finite(leg['ratio']) or float(leg['ratio']) <= 0:
            raise ResearchError('Invalid normalized stock ratio')
        recipients.append(str(leg['recipient_id']))
    if len(recipients) != len(set(recipients)):
        raise ResearchError('Repeated normalized stock recipient')
    if legs != sorted(legs, key=lambda x: (x['recipient_id'], x['recipient_ticker'])):
        raise ResearchError('Normalized stock legs are unordered')
    cash = package['cash_usd_per_raw_share']
    valid_cash = _finite(cash) and float(cash) > 0
    shape = package['classification']
    if ((shape == 'cash' and not (valid_cash and not legs and not package['unresolved_reasons']))
            or (shape == 'stock' and not (cash is None and legs and not package['unresolved_reasons']))
            or (shape == 'mixed' and not (valid_cash and legs and not package['unresolved_reasons']))
            or (shape == 'unresolved' and not (cash is None and not legs and package['unresolved_reasons']))):
        raise ResearchError('Normalized action terms differ from classification')
    if ((shape in {'cash', 'mixed'} and 'acquisitioncash' not in package['source_action_types'])
            or (shape in {'stock', 'mixed'} and 'acquisitionstock' not in package['source_action_types'])):
        raise ResearchError('Normalized action terms lack their source type')
    return package


def build_action_bundle(inputs: Path, descriptions: Path, output: Path):
    """Build a new immutable sidecar from the parent's exact local source pins."""
    inputs, descriptions, output = Path(inputs).resolve(), Path(descriptions).resolve(), Path(output).resolve()
    if output.exists():
        raise ResearchError('Choose a new action-bundle directory')
    manifest, identities, pins, bindings = _input_bindings(inputs)
    if not descriptions.is_file() or descriptions.is_symlink():
        raise ResearchError('Pinned descriptions snapshot is unavailable')
    _verify_dictionary(descriptions)
    pins['descriptions'] = {'path': str(descriptions), 'version': descriptions.parent.name,
                            'sha256': file_hash(descriptions)}
    db = duckdb.connect(config={'threads': '1', 'memory_limit': '2GB'})
    try:
        tickers = db.execute('SELECT ticker,permaticker FROM read_parquet(?)', [pins['tickers']['path']]).df()
        mapping = _ticker_map(tickers, identities)
        actions = db.execute('''SELECT ticker,date,action,value,contraticker FROM read_parquet(?)
                                WHERE date<=CAST(? AS DATE) ORDER BY ticker,date,action,contraticker''',
                             [pins['actions']['path'], manifest['end']]).df()
        actions['ticker'] = actions.ticker.astype(str)
        initial_tickers = {str(row['ticker']) for row in identities}
        closure = set(initial_tickers)
        while True:
            recipients = {str(x) for x in actions.loc[
                actions.ticker.isin(closure) & actions.action.eq('acquisitionstock'), 'contraticker'].dropna()
                          if str(x) not in {'', 'N/A'}}
            expanded = closure | recipients
            if expanded == closure:
                break
            closure = expanded
        packages = []
        selected = actions.loc[actions.ticker.isin(closure)].copy()
        selected['date_text'] = pd.to_datetime(selected.date).dt.date.astype(str)
        for (ticker, day), group in selected.groupby(['ticker', 'date_text'], sort=True):
            types = set(group.action.astype(str))
            # Only a target's terminal event may alter its position.  In
            # particular, acquisitionof is acquirer metadata and must never
            # turn a continuing acquirer position into an unresolved terminal.
            if not TERMINAL_ACTIONS & types:
                continue
            security = mapping.get(str(ticker))
            if security is None:
                continue
            packages.append(_normalize_package(str(ticker), security, str(day), group.drop(columns='date_text'), mapping))
        packages.sort(key=lambda row: (row['action_date'], row['parent_security_id'], row['package_id']))
        for package in packages:
            _validate_package(package)
        target_days = [(row['parent_security_id'], row['action_date']) for row in packages]
        if len(target_days) != len(set(target_days)):
            raise ResearchError('Terminal target/date is identity-ambiguous')
        # Quote every uniquely mapped transitive recipient, including recipients
        # named in packages that were conservatively left unresolved.  That
        # keeps the immutable evidence complete without exposing partial terms.
        supplemental = sorted(ticker for ticker in closure
                              if ticker in mapping and ticker not in initial_tickers)
        if supplemental:
            selected_quotes = pd.DataFrame({'ticker': supplemental})
            db.register('supplemental', selected_quotes)
            candidates = db.execute('''SELECT CAST(t.permaticker AS VARCHAR) AS security_id,s.ticker,s.date,
                                      s.closeunadj AS raw_close_usd,s.closeadj AS adjusted_close_usd,
                                      s.close*s.volume AS dollar_volume_usd
                                   FROM read_parquet(?) s JOIN supplemental x USING(ticker)
                                   JOIN (SELECT ticker,min(permaticker) AS permaticker FROM read_parquet(?)
                                         GROUP BY ticker HAVING count(DISTINCT permaticker)=1) t USING(ticker)
                                   WHERE s.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                                     AND isfinite(s.closeunadj) AND s.closeunadj>0
                                     AND isfinite(s.closeadj) AND s.closeadj>0
                                     AND isfinite(s.close*s.volume) AND s.close*s.volume>=0
                                   ORDER BY security_id,date,ticker''',
                                [pins['stocks']['path'], pins['tickers']['path'], manifest['start'], manifest['end']]).df()
            # Ticker aliases can map to one stable share-class identity.  When
            # two aliases quote that identity on the same date, neither record
            # is a defensible automatic choice, so omit that key and leave any
            # resulting claim pending until an unambiguous session.
            ambiguous = candidates.duplicated(['security_id', 'date'], keep=False)
            ambiguous_quote_keys = int(candidates.loc[ambiguous, ['security_id', 'date']]
                                       .drop_duplicates().shape[0])
            quotes = candidates.loc[~ambiguous, QUOTE_COLUMNS].reset_index(drop=True)
        else:
            ambiguous_quote_keys = 0
            quotes = pd.DataFrame({
                'security_id': pd.Series(dtype='str'), 'ticker': pd.Series(dtype='str'),
                'date': pd.Series(dtype='datetime64[ns]'), 'raw_close_usd': pd.Series(dtype='float64'),
                'adjusted_close_usd': pd.Series(dtype='float64'), 'dollar_volume_usd': pd.Series(dtype='float64'),
            })
        if quotes.duplicated(['security_id', 'date']).any():
            raise ResearchError('Supplemental recipient quotes are identity-ambiguous')
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / 'packages.json', packages)
        db.from_df(quotes[QUOTE_COLUMNS]).write_parquet(str(output / 'supplemental-quotes.parquet'))
    finally:
        db.close()
    files = {name: _artifact(output / name) for name in ('packages.json', 'supplemental-quotes.parquet')}
    counts = {
        'initial_identities': len(identities), 'closure_tickers': len(closure), 'packages': len(packages),
        'resolved_packages': sum(row['classification'] != 'unresolved' for row in packages),
        'unresolved_packages': sum(row['classification'] == 'unresolved' for row in packages),
        'supplemental_quote_rows': len(quotes),
        'supplemental_securities': int(quotes.security_id.nunique()) if len(quotes) else 0,
        'supplemental_ambiguous_quote_keys': ambiguous_quote_keys,
    }
    body = {
        'schema_version': 1, 'kind': 'factor_action_bundle',
        'methodology': 'sharadar_terminal_action_normalization_v1',
        'parent_bundle_id': manifest['bundle_id'], 'start': manifest['start'], 'end': manifest['end'],
        'point_in_time_attested': False, 'settlement_timing_attested': False,
        'input_bindings': bindings, 'source_pins': pins, 'counts': counts, 'files': files,
        'implementation_sha256': file_hash(__file__),
    }
    result = {'bundle_id': digest(body), **body}
    _write_json(output / 'manifest.json', result)
    return result


def load_action_bundle(path: Path, *, verify_sources=True):
    """Verify and load a complete action bundle; partial/tampered inputs fail closed."""
    path = Path(path).resolve()
    if not path.is_dir() or path.is_symlink():
        raise ResearchError('Action bundle is not a regular directory')
    manifest = read_json(path / 'manifest.json')
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ResearchError('Invalid action bundle manifest fields')
    if (manifest['schema_version'] != 1 or manifest['kind'] != 'factor_action_bundle'
            or manifest['methodology'] != 'sharadar_terminal_action_normalization_v1'
            or manifest['point_in_time_attested'] is not False
            or manifest['settlement_timing_attested'] is not False):
        raise ResearchError('Unsupported action bundle methodology')
    body = {key: value for key, value in manifest.items() if key != 'bundle_id'}
    if manifest['bundle_id'] != digest(body):
        raise ResearchError('Action bundle semantic identity mismatch')
    bindings = manifest.get('input_bindings')
    if (not isinstance(bindings, dict)
            or set(bindings) != {'manifest_sha256', 'identities_sha256'}
            or any(not isinstance(value, str) or len(value) != 64 for value in bindings.values())):
        raise ResearchError('Invalid parent input bindings')
    pins = manifest.get('source_pins')
    if not isinstance(pins, dict) or set(pins) != {'stocks', 'actions', 'tickers', 'descriptions'}:
        raise ResearchError('Action source pin inventory is incomplete')
    for pin in pins.values():
        if (not isinstance(pin, dict) or set(pin) != {'path', 'version', 'sha256'}
                or not isinstance(pin['path'], str) or not pin['path']
                or not isinstance(pin['version'], str) or not pin['version']
                or not isinstance(pin['sha256'], str) or len(pin['sha256']) != 64):
            raise ResearchError('Invalid action source pin')
    if set(manifest.get('files', {})) != {'packages.json', 'supplemental-quotes.parquet'}:
        raise ResearchError('Action bundle file inventory is incomplete')
    for name, expected in manifest['files'].items():
        file = path / name
        if (file.parent != path or file.is_symlink() or not file.is_file()
                or not isinstance(expected, dict) or set(expected) != {'sha256', 'bytes'}
                or file_hash(file) != expected['sha256'] or file.stat().st_size != expected['bytes']):
            raise ResearchError('Action bundle artifact changed')
    packages = read_json(path / 'packages.json')
    if not isinstance(packages, list):
        raise ResearchError('Action packages must be a list')
    for package in packages:
        _validate_package(package)
    order = [(row['action_date'], row['parent_security_id'], row['package_id']) for row in packages]
    target_days = [(row['parent_security_id'], row['action_date']) for row in packages]
    if (order != sorted(order) or len({row['package_id'] for row in packages}) != len(packages)
            or len(target_days) != len(set(target_days))):
        raise ResearchError('Action packages are unordered or repeated')
    db = duckdb.connect()
    try:
        quotes = db.execute('SELECT * FROM read_parquet(?) ORDER BY security_id,date,ticker',
                            [str(path / 'supplemental-quotes.parquet')]).df()
    finally:
        db.close()
    if list(quotes.columns) != QUOTE_COLUMNS or quotes.duplicated(['security_id', 'date']).any():
        raise ResearchError('Invalid supplemental quote keys or columns')
    for name in ('raw_close_usd', 'adjusted_close_usd'):
        if len(quotes) and (not quotes[name].map(_finite).all() or (quotes[name] <= 0).any()):
            raise ResearchError('Invalid supplemental recipient quote')
    if len(quotes) and (not quotes.dollar_volume_usd.map(_finite).all() or (quotes.dollar_volume_usd < 0).any()):
        raise ResearchError('Invalid supplemental recipient liquidity')
    counts = manifest['counts']
    if not isinstance(counts, dict) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                                           for value in counts.values()):
        raise ResearchError('Invalid action bundle counts')
    actual = {
        'packages': len(packages),
        'resolved_packages': sum(row['classification'] != 'unresolved' for row in packages),
        'unresolved_packages': sum(row['classification'] == 'unresolved' for row in packages),
        'supplemental_quote_rows': len(quotes),
        'supplemental_securities': int(quotes.security_id.nunique()) if len(quotes) else 0,
    }
    if any(counts.get(key) != value for key, value in actual.items()):
        raise ResearchError('Action bundle counts do not match its artifacts')
    if verify_sources:
        for pin in pins.values():
            source = Path(pin['path'])
            if not source.is_file() or source.is_symlink() or file_hash(source) != pin['sha256']:
                raise ResearchError('Pinned action source changed')
    return {'manifest': manifest, 'packages': packages, 'quotes': quotes}


def analytical_to_raw(units, adjusted_price, raw_price):
    """Translate signed analytical units to signed raw shares at one quote."""
    if not all(_finite(value) for value in (units, adjusted_price, raw_price)):
        raise ResearchError('Cannot bridge analytical units without finite prices')
    if float(adjusted_price) <= 0 or float(raw_price) <= 0:
        raise ResearchError('Cannot bridge analytical units with nonpositive prices')
    return float(units) * float(adjusted_price) / float(raw_price)


def claim_to_analytical(raw_shares, recipient_adjusted, recipient_raw):
    """Translate a signed raw recipient claim when a valid quote becomes available.

    ``None`` is returned for an absent/nonfinite quote so callers retain the raw
    claim.  Observed nonpositive prices are rejected as invalid evidence.
    """
    if not _finite(raw_shares):
        raise ResearchError('Invalid signed raw recipient claim')
    if recipient_adjusted is None or recipient_raw is None:
        return None
    if not _finite(recipient_adjusted) or not _finite(recipient_raw):
        return None
    if float(recipient_adjusted) <= 0 or float(recipient_raw) <= 0:
        raise ResearchError('Invalid recipient quote for analytical conversion')
    return float(raw_shares) * float(recipient_raw) / float(recipient_adjusted)


def resolve_package(package, parent_units, parent_adjusted, parent_raw):
    """Resolve one package into signed cash and raw-stock claims.

    Unresolved packages retain the parent position and never expose partial
    standard terms.  Resolved packages remove the parent exactly once.
    """
    package = _validate_package(package)
    if not _finite(parent_units):
        raise ResearchError('Invalid analytical parent position')
    base = {'package_id': package['package_id'], 'action_date': package['action_date']}
    if package['classification'] == 'unresolved':
        return {**base, 'resolved': False, 'parent_retained': True, 'raw_parent_shares': None,
                'cash_usd': 0., 'stock_claims': [],
                'unresolved_reasons': list(package['unresolved_reasons'])}
    raw = analytical_to_raw(parent_units, parent_adjusted, parent_raw)
    cash = raw * float(package['cash_usd_per_raw_share'] or 0.)
    claims = [{'recipient_id': leg['recipient_id'], 'recipient_ticker': leg['recipient_ticker'],
               'raw_shares': raw * float(leg['ratio'])} for leg in package['stock_legs']]
    return {**base, 'resolved': True, 'parent_retained': False, 'raw_parent_shares': raw,
            'cash_usd': cash, 'stock_claims': claims, 'unresolved_reasons': []}
