import pytest

pd = pytest.importorskip('pandas')
pytest.importorskip('duckdb')

from alpha_lab.equity_reference import paired_transitions, share_basis_diagnostics, share_flag_scope, stable_identities, terminal_work_items


def metadata():
    return pd.DataFrame({'table': ['SEP', 'SEP', 'SF1'], 'ticker': ['A', 'B', 'A'],
                         'permaticker': [1, 2, 1], 'exchange': ['NYSE', 'NYSE', 'NYSE'],
                         'siccode': [2000, 2000, 2000]})


def transitions(legs):
    return pd.DataFrame([{'ticker': 'A', 'date': pd.Timestamp(date), 'action': action,
                          'contraticker': contra, 'contraname': name, 'value': value}
                         for date, action, contra, name, value in legs])


def test_permanent_ids_are_security_classes_and_never_issuer_aliases():
    result = stable_identities(metadata(), ['A', 'B', 'MISSING'])
    assert [row['security_id'] for row in result] == [1, 2, None]
    assert all(row['issuer_id'] is None and row['historical_class_eligibility_attested'] is False for row in result)
    duplicate = pd.concat([metadata(), metadata().iloc[[0]]])
    assert stable_identities(duplicate, ['A'])[0]['status'] == 'unresolved'


def test_dated_pairs_retain_identity_without_equating_acquisition_counterparties():
    actions = transitions([('2014-01-02', 'tickerchangefrom', 'OLD', 'N/A', None),
                           ('2014-01-02', 'tickerchangeto', 'A', 'N/A', None),
                           ('2015-01-02', 'acquisitionby', 'B', 'N/A', 100.)])
    pairs, problems = paired_transitions(metadata(), actions)
    assert len(pairs) == 1 and not problems and pairs[0]['from'] == 'OLD'
    assert pairs[0]['to'] == 'A' and pairs[0]['historical_arrival_attested'] is False


@pytest.mark.parametrize('legs,reason', [
    ([('2014-01-02', 'exchangefrom', None, 'NASDAQ', None)], 'unique_from_to_pair_required'),
    ([('2014-01-02', 'exchangefrom', None, 'NASDAQ', None), ('2014-01-02', 'exchangeto', None, 'N/A', None)], 'transition_value_missing_or_invalid'),
    ([('2014-01-02', 'exchangefrom', None, 'NASDAQ', None), ('2014-01-02', 'exchangeto', None, 'OTC', None)], 'current_snapshot_anchor_conflict'),
])
def test_incomplete_or_conflicting_pair_is_not_silently_accepted(legs, reason):
    _, problems = paired_transitions(metadata(), transitions(legs))
    assert any(problem['reason'] == reason for problem in problems)


def test_transition_chain_conflict_is_visible():
    actions = transitions([('2014-01-02', 'exchangefrom', None, 'OTC', None),
                           ('2014-01-02', 'exchangeto', None, 'NASDAQ', None),
                           ('2015-01-02', 'exchangefrom', None, 'OTHER', None),
                           ('2015-01-02', 'exchangeto', None, 'NYSE', None)])
    _, problems = paired_transitions(metadata(), actions)
    assert [row['reason'] for row in problems] == ['transition_chain_conflict']


@pytest.mark.parametrize('unit,cap', [('USD', 100e6), ('USD millions', 100.)])
def test_cap_formula_uses_pinned_units_share_factor_and_split_price_not_weighted_average(unit, cap):
    rows = pd.DataFrame({'marketcap': [cap] * 4, 'sharesbas': [1e6, 1e6, 1e6, 0.],
                         'price': [50., 50., 50., 50.], 'sharefactor': [2., 1., None, 2.]})
    result, flagged = share_basis_diagnostics(rows, marketcap_unit=unit)
    assert result['formula_difference_rows'] == 1 and result['missing_or_nonpositive_basis_rows'] == 2
    assert len(flagged) == 3 and flagged.iloc[0].relative_difference == pytest.approx(.5)


def test_unknown_cap_unit_is_not_guessed():
    with pytest.raises(ValueError, match='units'):
        share_basis_diagnostics(pd.DataFrame(), marketcap_unit='unknown')


def test_quote_scope_distinguishes_expected_missing_prices_and_used_records():
    flags = pd.DataFrame({'ticker': ['A', 'A', 'A', 'MISSING'],
                          'date': pd.to_datetime(['2013-01-01', '2021-01-01', '2015-01-01', '2015-01-01']),
                          'reportperiod': pd.to_datetime(['2012-12-31'] * 4)})
    bounds = pd.DataFrame({'ticker': ['A'], 'first_quote': [pd.Timestamp('2014-01-01')], 'last_quote': [pd.Timestamp('2020-01-01')]})
    features = pd.DataFrame({'ticker': ['A', 'A'], 'date': pd.to_datetime(['2016-01-01', '2023-01-01']),
                             'filing_date': pd.to_datetime(['2013-01-01', '2021-01-01']),
                             'reportperiod': pd.to_datetime(['2012-12-31'] * 2)})
    result, records = share_flag_scope(flags, features, bounds)
    assert result['explained_outside_quote_history'] == 1 and result['records_requiring_review'] == 3
    assert result['used_train_validation_rows'] == 1
    assert records.review_status.tolist() == ['requires_review', 'explained_outside_quote_history', 'requires_review', 'requires_review']
    assert records.quote_history_scope.tolist() == ['before_first_quote', 'after_last_quote', 'within_quote_history', 'unknown']


def test_quote_scope_rejects_duplicate_bounds():
    bounds = pd.DataFrame({'ticker': ['A', 'A']})
    with pytest.raises(ValueError, match='Unique'):
        share_flag_scope(pd.DataFrame(), pd.DataFrame(), bounds)


def test_terminal_tasks_never_promote_diagnostic_proceeds_or_infer_bankruptcy_zero():
    rows = terminal_work_items([{'ticker': 'A', 'date': '2020-01-01', 'classification': 'insolvency'},
                               {'ticker': 'B', 'date': '2020-01-02', 'classification': 'stock', 'diagnostic_consideration_usd_per_share': 50.}])
    assert all(row['status'] == 'open' and row['economic_return_ready'] is False for row in rows)
    assert 'do not infer zero' in ' '.join(rows[0]['required_checks'])
    assert 'successor share-class' in ' '.join(rows[1]['required_checks'])


@pytest.fixture
def reference_inputs(tmp_path):
    import json
    import duckdb
    from alpha_lab.market_data.sharadar import file_hash
    panel, quality, terminal = [tmp_path / tag for tag in ('panel', 'quality', 'terminal')]
    for directory in (panel, quality, terminal):
        directory.mkdir()
    frames = {
        'tickers': metadata(),
        'actions': transitions([('2014-01-02', 'tickerchangefrom', 'OLD', 'N/A', None),
                                ('2014-01-02', 'tickerchangeto', 'A', 'N/A', None)]),
        'fundamentals': pd.DataFrame({'ticker': ['A'], 'dimension': ['ART'], 'date': [pd.Timestamp('2014-01-02')],
                                     'reportperiod': [pd.Timestamp('2013-12-31')], 'lastupdated': [pd.Timestamp('2026-10-13')],
                                     'marketcap': [100e6], 'sharesbas': [1e6], 'price': [50.], 'sharefactor': [2.]}),
        'descriptions': pd.DataFrame({'table': ['SF1'] * 4 + ['DAILY'],
                                     'indicator': ['marketcap', 'price', 'sharesbas', 'sharefactor', 'marketcap'],
                                     'unittype': ['USD', 'USD/share', 'units', 'ratio', 'USD millions']}),
        'stocks': pd.DataFrame({'ticker': ['A'], 'date': [pd.Timestamp('2014-01-02')], 'closeunadj': [50.]})}
    pins = {}
    for tag, frame in frames.items():
        path = tmp_path / (tag + '.parquet')
        duckdb.from_df(frame).write_parquet(str(path))
        pins[tag] = {'path': str(path), 'parquet_sha256': file_hash(path), 'retrieved_at': '2026-10-13T10:00:00Z'}
    duckdb.from_df(pd.DataFrame({'ticker': ['A'], 'date': [pd.Timestamp('2014-01-31')],
                                 'filing_date': [pd.Timestamp('2014-01-02')],
                                 'reportperiod': [pd.Timestamp('2013-12-31')]})).write_parquet(str(panel / 'features.parquet'))
    (panel / 'sources.json').write_text(json.dumps({tag: pin for tag, pin in pins.items() if tag != 'descriptions'}))
    (panel / 'panel.json').write_text(json.dumps({'files': {name: file_hash(panel / name) for name in ('features.parquet', 'sources.json')}}))
    (terminal / 'sources.json').write_text(json.dumps(pins))
    (terminal / 'summary.json').write_text('{}')
    (terminal / 'events.json').write_text(json.dumps([{'ticker': 'A', 'date': '2014-02-03', 'classification': 'cash',
                                                     'diagnostic_consideration_usd_per_share': 55.}]))
    (quality / 'quality-queue.json').write_text(json.dumps([{'id': 'terminal:A:2014-02-03', 'status': 'open'}]))
    (quality / 'summary.json').write_text(json.dumps({
        'files': {'quality-queue.json': file_hash(quality / 'quality-queue.json')},
        'input_hashes': {name: file_hash(panel / name) for name in ('panel.json', 'sources.json')},
        'terminal_evidence': {'hashes': {name: file_hash(terminal / name) for name in ('summary.json', 'events.json', 'sources.json')}}}))
    return panel, quality, terminal


def test_reference_audit_preserves_inputs_and_keeps_cash_terms_open(reference_inputs, tmp_path):
    import json
    from alpha_lab.equity_reference import audit_reference
    from alpha_lab.market_data.sharadar import file_hash
    panel, quality, terminal = reference_inputs
    original = file_hash(terminal / 'events.json')
    output = tmp_path / 'audit'
    result = audit_reference(panel, quality, terminal, output)
    assert result['share_basis']['formula_difference_rows'] == 0 and result['stable_security_ids'] == 1
    assert result['point_in_time_attested'] is result['pnl_modified'] is False
    assert result['open_original_issues'] == 1
    assert file_hash(terminal / 'events.json') == original
    assert json.loads((output / 'quality-queue.json').read_text())[0]['status'] == 'open'
    for name, digest in result['files'].items():
        assert file_hash(output / name) == digest
    with pytest.raises(ValueError, match='immutable'):
        audit_reference(panel, quality, terminal, output)


@pytest.mark.parametrize('target', ['panel', 'quality', 'terminal'])
def test_reference_audit_rejects_tampered_evidence_before_output(reference_inputs, tmp_path, target):
    from alpha_lab.equity_reference import audit_reference
    panel, quality, terminal = reference_inputs
    path = {'panel': panel / 'features.parquet', 'quality': quality / 'quality-queue.json', 'terminal': terminal / 'events.json'}[target]
    path.write_bytes(b'changed')
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='hash mismatch'):
        audit_reference(panel, quality, terminal, output)
    assert not output.exists()
