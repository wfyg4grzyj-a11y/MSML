"""Action-bundle normalization and signed unit-bridge references."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_lab.research.contracts import ResearchError, file_hash
from alpha_lab.research.factor_actions import (analytical_to_raw, build_action_bundle,
    claim_to_analytical, load_action_bundle, resolve_package)


def package(kind, *, cash=None, legs=(), reasons=()):
    from alpha_lab.research.contracts import digest
    identity = {'parent_security_id': '1', 'parent_ticker': 'A', 'action_date': '2020-01-02',
                'source_rows_sha256': 'a' * 64}
    types = sorted((['acquisitioncash'] if cash is not None else [])
                   + (['acquisitionstock'] if legs else []))
    return {'package_id': digest(identity), **identity, 'classification': kind,
            'cash_usd_per_raw_share': cash,
            'stock_legs': [{'recipient_id': str(i + 2), 'recipient_ticker': ticker, 'ratio': ratio}
                           for i, (ticker, ratio) in enumerate(legs)],
            'source_action_types': types or ['delisted'],
            'unresolved_reasons': list(reasons)}


@pytest.mark.parametrize('units', [10., -10.])
def test_signed_cash_stock_mixed_and_multiple_stock_legs(units):
    # adjusted80/raw100 means ten analytical units are eight signed raw shares.
    raw = analytical_to_raw(units, 80., 100.)
    assert raw == pytest.approx(units * .8)
    cash = resolve_package(package('cash', cash=100.), units, 80., 100.)
    assert cash['cash_usd'] == pytest.approx(units * 80.) and not cash['stock_claims']
    stock = resolve_package(package('stock', legs=[('B', 2.)]), units, 80., 100.)
    assert stock['stock_claims'][0]['raw_shares'] == pytest.approx(units * 1.6)
    assert claim_to_analytical(stock['stock_claims'][0]['raw_shares'], 40., 50.) == pytest.approx(units * 2.)
    mixed = resolve_package(package('mixed', cash=50., legs=[('B', .6), ('C', .4)]), units, 80., 100.)
    assert mixed['cash_usd'] == pytest.approx(units * 40.)
    assert sum(x['raw_shares'] for x in mixed['stock_claims']) == pytest.approx(units * .8)


def test_unresolved_package_retains_parent_and_missing_quote_retains_signed_claim():
    unresolved = package('unresolved', reasons=['insolvency_terms_unknown'])
    result = resolve_package(unresolved, -10., 80., 100.)
    assert not result['resolved'] and result['parent_retained'] and result['raw_parent_shares'] is None
    claim = resolve_package(package('stock', legs=[('B', 2.)]), -10., 80., 100.)['stock_claims'][0]
    assert claim['raw_shares'] == -16.
    assert claim_to_analytical(claim['raw_shares'], None, None) is None
    assert claim_to_analytical(claim['raw_shares'], float('nan'), 50.) is None


def write_parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.from_df(frame).write_parquet(str(path))


def fixture(tmp_path, actions):
    tables = tmp_path / 'tables'
    versions = {name: 'v-' + name for name in ('stocks', 'actions', 'tickers')}
    paths = {name: tables / name / versions[name] / 'data.parquet' for name in versions}
    tickers = pd.DataFrame({
        'table': ['SEP'] * 4, 'ticker': ['A', 'B', 'C', 'D'], 'permaticker': [1, 2, 3, 4],
        'name': ['A', 'B', 'C', 'D'], 'exchange': ['NYSE'] * 4,
        'category': ['Domestic Common Stock'] * 4, 'currency': ['USD'] * 4,
    })
    dates = pd.to_datetime(['2020-01-02', '2020-01-03'] * 4)
    stocks = pd.DataFrame({
        'ticker': sum(([ticker] * 2 for ticker in ['A', 'B', 'C', 'D']), []), 'date': dates,
        'closeunadj': [100., 101., 50., 51., 25., 26., 10., 11.],
        'closeadj': [80., 81., 40., 41., 20., 21., 8., 9.],
        'close': [100., 101., 50., 51., 25., 26., 10., 11.], 'volume': [1000.] * 8,
    })
    write_parquet(paths['stocks'], stocks)
    write_parquet(paths['actions'], pd.DataFrame(actions))
    write_parquet(paths['tickers'], tickers)
    descriptions = tables / 'descriptions' / 'v-descriptions' / 'data.parquet'
    units = [
        ('ACTIONTYPES', 'acquisitioncash', 'USD/share'),
        ('ACTIONTYPES', 'acquisitionstock', 'ratio'),
        ('ACTIONTYPES', 'acquisitionby', 'USD millions'),
        ('ACTIONTYPES', 'delisted', 'USD millions'),
        ('ACTIONTYPES', 'bankruptcyliquidation', 'USD millions'),
        ('SEP', 'closeunadj', 'USD/share'), ('SEP', 'closeadj', 'USD/share'),
    ]
    write_parquet(descriptions, pd.DataFrame(units, columns=['table', 'indicator', 'unittype']))
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    identities = [{'ticker': 'A', 'security_id': '1', 'industry': 'SIC2-00', 'first_index': 0}]
    (inputs / 'identities.json').write_text(json.dumps(identities))
    manifest = {
        'bundle_id': 'b' * 64, 'start': '2020-01-02', 'end': '2020-01-03',
        'source_versions': {name: {'version': versions[name], 'sha256': file_hash(path)}
                            for name, path in paths.items()},
        'source_files': {'stocks': str(paths['stocks']), 'tickers': str(paths['tickers'])},
        'files': {'identities.json': {'sha256': file_hash(inputs / 'identities.json'),
                                      'bytes': (inputs / 'identities.json').stat().st_size}},
    }
    (inputs / 'manifest.json').write_text(json.dumps(manifest))
    return inputs, descriptions, paths


def row(ticker, action, value=None, contra='N/A', day='2020-01-02'):
    return {'ticker': ticker, 'date': pd.Timestamp(day), 'action': action, 'value': value,
            'contraticker': contra}


def test_bundle_normalizes_additive_multi_leg_and_transitive_recipients(tmp_path):
    actions = [row('A', 'acquisitionby', 900.), row('A', 'delisted', 900.),
               row('A', 'acquisitioncash', 50.), row('A', 'acquisitionstock', .6, 'B'),
               row('A', 'acquisitionstock', .4, 'C'),
               row('B', 'delisted', 500., day='2020-01-03'),
               row('B', 'acquisitionstock', 2., 'D', day='2020-01-03')]
    inputs, descriptions, _ = fixture(tmp_path, actions)
    output = tmp_path / 'bundle'
    manifest = build_action_bundle(inputs, descriptions, output)
    loaded = load_action_bundle(output)
    assert manifest['counts']['packages'] == 2 and manifest['counts']['resolved_packages'] == 2
    first = loaded['packages'][0]
    assert first['classification'] == 'mixed' and first['cash_usd_per_raw_share'] == 50.
    assert [(x['recipient_ticker'], x['ratio']) for x in first['stock_legs']] == [('B', .6), ('C', .4)]
    assert set(loaded['quotes'].security_id) == {'2', '3', '4'}
    assert (loaded['quotes'].dollar_volume_usd >= 0).all()


def test_acquirer_metadata_does_not_schedule_a_terminal_package(tmp_path):
    actions = [row('A', 'acquisitionof', 900.)]
    inputs, descriptions, _ = fixture(tmp_path, actions)
    output = tmp_path / 'bundle'; manifest = build_action_bundle(inputs, descriptions, output)
    assert manifest['counts']['packages'] == 0
    assert load_action_bundle(output)['packages'] == []


@pytest.mark.parametrize('bad_rows,reason', [
    ([row('A', 'acquisitioncash', 50.), row('A', 'acquisitioncash', 50.), row('A', 'delisted', 1e6)],
     'duplicate_cash_legs'),
    ([row('A', 'acquisitionstock', 1., 'B'), row('A', 'acquisitionstock', 2., 'B'), row('A', 'delisted', 1e6)],
     'duplicate_stock_recipient:B'),
    ([row('A', 'acquisitionelectcash', 50., 'B'), row('A', 'delisted', 1e6)],
     'unsupported_acquisition_types:acquisitionelectcash'),
    ([row('A', 'acquisitioncasheur', 50., 'B'), row('A', 'delisted', 1e6)],
     'unsupported_acquisition_types:acquisitioncasheur'),
    ([row('A', 'acquisitionspecial', 1., 'B'), row('A', 'delisted', 1e6)],
     'unsupported_acquisition_types:acquisitionspecial'),
    ([row('A', 'acquisitioncash', -1.), row('A', 'delisted', 1e6)],
     'invalid_cash_usd_per_share'),
    ([row('A', 'acquisitionstock', 0., 'B'), row('A', 'delisted', 1e6)],
     'invalid_stock_ratio:B'),
    ([row('A', 'acquisitionstock', 1., 'MISSING'), row('A', 'delisted', 1e6)],
     'unmapped_stock_recipient:MISSING'),
    ([row('A', 'acquisitionby', 1e6), row('A', 'delisted', 1e6)],
     'no_standard_consideration_terms'),
])
def test_duplicates_market_cap_elections_foreign_and_unmapped_stay_unresolved(tmp_path, bad_rows, reason):
    inputs, descriptions, _ = fixture(tmp_path, bad_rows)
    output = tmp_path / 'bundle'; build_action_bundle(inputs, descriptions, output)
    package = load_action_bundle(output)['packages'][0]
    assert package['classification'] == 'unresolved'
    assert package['cash_usd_per_raw_share'] is None and package['stock_legs'] == []
    assert reason in package['unresolved_reasons']


def test_two_ticker_aliases_for_one_recipient_are_a_duplicate_leg(tmp_path):
    actions = [row('A', 'acquisitionstock', .5, 'B'),
               row('A', 'acquisitionstock', .5, 'D'), row('A', 'delisted', 1e6)]
    inputs, descriptions, paths = fixture(tmp_path, actions)
    db = duckdb.connect()
    tickers = db.execute('SELECT * FROM read_parquet(?)', [str(paths['tickers'])]).df()
    db.close()
    tickers.loc[tickers.ticker.eq('D'), 'permaticker'] = 2
    paths['tickers'].unlink(); write_parquet(paths['tickers'], tickers)
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_versions']['tickers']['sha256'] = file_hash(paths['tickers'])
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / 'bundle'; build_action_bundle(inputs, descriptions, output)
    action = load_action_bundle(output)['packages'][0]
    assert action['classification'] == 'unresolved'
    assert 'duplicate_stock_recipient_id:2' in action['unresolved_reasons']
    assert load_action_bundle(output)['manifest']['counts']['supplemental_ambiguous_quote_keys'] == 2


def test_bundle_and_source_hash_tamper_fail_closed(tmp_path):
    actions = [row('A', 'acquisitioncash', 100.), row('A', 'delisted', 1e6)]
    inputs, descriptions, paths = fixture(tmp_path, actions)
    output = tmp_path / 'bundle'; build_action_bundle(inputs, descriptions, output)
    original = (output / 'packages.json').read_bytes()
    (output / 'packages.json').write_bytes(original + b' ')
    with pytest.raises(ResearchError, match='artifact changed'):
        load_action_bundle(output)
    (output / 'packages.json').write_bytes(original)
    assert load_action_bundle(output)['packages'][0]['classification'] == 'cash'
    with paths['actions'].open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ResearchError, match='source changed'):
        load_action_bundle(output)


def test_dictionary_unit_tamper_is_rejected_before_output(tmp_path):
    actions = [row('A', 'acquisitioncash', 100.), row('A', 'delisted', 1e6)]
    inputs, descriptions, _ = fixture(tmp_path, actions)
    frame = duckdb.connect().execute('SELECT * FROM read_parquet(?)', [str(descriptions)]).df()
    frame.loc[frame.indicator.eq('acquisitioncash'), 'unittype'] = 'EUR/share'
    descriptions.unlink(); write_parquet(descriptions, frame)
    with pytest.raises(ResearchError, match='does not verify'):
        build_action_bundle(inputs, descriptions, tmp_path / 'bundle')
