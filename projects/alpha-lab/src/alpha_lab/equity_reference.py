"""Offline security identity, share-basis and terminal-evidence remediation.

Business dates and local observation times are separate. This audit does not
change a panel, compute performance, assign terminal payouts or attest a feed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_data.sharadar_store import atomic_json, observation_time


TRANSITIONS = {
    'ticker': ('tickerchangefrom', 'tickerchangeto', 'contraticker'),
    'exchange': ('exchangefrom', 'exchangeto', 'contraname'),
    'sic': ('sicchangefrom', 'sicchangeto', 'value'),
}


def stable_identities(metadata, tickers):
    """A vendor permaticker identifies a security/share class, not an issuer."""
    records = []
    for ticker in sorted(set(tickers)):
        rows = metadata[(metadata['table'] == 'SEP') & (metadata.ticker == ticker)]
        ids = rows.permaticker.dropna().unique()
        valid = len(rows) == 1 and len(ids) == 1 and float(ids[0]).is_integer() and ids[0] > 0
        records.append({'ticker': ticker, 'security_id': int(ids[0]) if valid else None,
                        'namespace': 'sharadar_permaticker_share_class',
                        'issuer_id': None, 'status': 'mapped' if valid else 'unresolved',
                        'category_basis': 'current_snapshot_only',
                        'historical_class_eligibility_attested': False})
    return records


def paired_transitions(metadata, actions):
    """Retain verified pair structure and report anchor/chain conflicts.

    Related tickers and acquisition counterparties are not identity equivalence.
    A valid pair is retrospective evidence, not proof of historical arrival or
    of an otherwise complete attribute history.
    """
    valid, issues = [], []
    def value(row, field):
        item = row[field]
        if pd.isna(item) or str(item).strip() in {'', 'N/A', 'NA'}:
            return None
        if field == 'value':
            try:
                number = float(item)
                return int(number) if np.isfinite(number) and number.is_integer() and 0 < number <= 9999 else None
            except (ValueError, TypeError):
                return None
        return str(item).strip()
    for attribute, (before, after, field) in TRANSITIONS.items():
        selected = actions[actions.action.isin([before, after])]
        for (ticker, date), group in selected.groupby(['ticker', 'date'], dropna=False):
            key = {'ticker': ticker, 'date': str(pd.Timestamp(date).date()), 'attribute': attribute}
            old, new = group[group.action == before], group[group.action == after]
            if pd.isna(date) or len(old) != 1 or len(new) != 1:
                issues.append({**key, 'reason': 'unique_from_to_pair_required'})
                continue
            prior, subsequent = value(old.iloc[0], field), value(new.iloc[0], field)
            if prior is None or subsequent is None:
                issues.append({**key, 'reason': 'transition_value_missing_or_invalid'})
                continue
            valid.append({**key, 'from': prior, 'to': subsequent,
                          'basis': 'retrospective_vendor_transition',
                          'historical_arrival_attested': False})
    for (ticker, attribute), group in pd.DataFrame(valid).groupby(['ticker', 'attribute']) if valid else []:
        chain = sorted(group.to_dict('records'), key=lambda row: row['date'])
        for left, right in zip(chain, chain[1:]):
            if left['to'] != right['from']:
                issues.append({'ticker': ticker, 'date': right['date'], 'attribute': attribute,
                               'reason': 'transition_chain_conflict'})
        current = metadata[(metadata['table'] == 'SEP') & (metadata.ticker == ticker)]
        field = {'sic': 'siccode'}.get(attribute, attribute)
        if len(current) == 1:
            anchor = current.iloc[0][field]
            anchor = int(anchor) if attribute == 'sic' and pd.notna(anchor) else anchor
            if pd.isna(anchor) or chain[-1]['to'] != anchor:
                issues.append({'ticker': ticker, 'date': chain[-1]['date'], 'attribute': attribute,
                               'reason': 'current_snapshot_anchor_conflict'})
        else:
            issues.append({'ticker': ticker, 'date': chain[-1]['date'], 'attribute': attribute,
                           'reason': 'unique_current_metadata_anchor_required'})
    return valid, issues


def share_basis_diagnostics(rows, tolerance=0.01, *, marketcap_unit='USD'):
    """SF1's reported cap identity is distinct from a dated DAILY share vintage."""
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('Choose an explicit positive relative tolerance below one')
    if marketcap_unit not in {'USD', 'USD millions'}:
        raise ValueError('Verify market-cap units from the pinned table dictionary')
    divisor = 1e6 if marketcap_unit == 'USD millions' else 1.
    values = rows[['marketcap', 'sharesbas', 'price', 'sharefactor']].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    difference = np.full(len(rows), np.nan)
    expected = np.full(len(rows), np.nan)
    expected[valid] = values[valid, 1] * values[valid, 2] * values[valid, 3] / divisor
    difference[valid] = np.abs(expected[valid] - values[valid, 0]) / values[valid, 0]
    bad = ~valid | (difference > tolerance)
    flagged = rows.loc[bad].copy()
    flagged['reconstructed_marketcap'] = expected[bad]
    flagged['relative_difference'] = difference[bad]
    flagged['reason'] = np.where(valid[bad], 'cap_formula_difference', 'missing_or_nonpositive_basis')
    return {'rows': len(rows), 'valid_basis_rows': int(valid.sum()),
            'missing_or_nonpositive_basis_rows': int((~valid).sum()),
            'formula_difference_rows': int((valid & (difference > tolerance)).sum()),
            'relative_tolerance': tolerance,
            'marketcap_unit': marketcap_unit,
            'formula': 'sharesbas * split-adjusted price * sharefactor' + (' / 1e6' if divisor == 1e6 else ''),
            'scope': 'ART filings through 2022 for current panel securities; not DAILY share-vintage or issuer aggregation attestation'}, flagged


def terminal_work_items(events):
    """Separate conditions required to move diagnostic terms into economic data."""
    records = []
    for event in events:
        kind = event['classification']
        tasks = ['Verify effective time and standard-holder entitlement from primary documents',
                 'Reconcile raw share units, splits and intervening dividends/distributions',
                 'Separate cash receivable, actual payment and stock-delivery dates',
                 'Record when each source was first observed; do not backdate knowledge']
        if kind in {'stock', 'cash_and_stock'}:
            tasks.append('Verify exchange ratio, successor share-class identity and fractional-share treatment')
        if kind == 'election_foreign_cash_or_special_security':
            tasks.append('Resolve election/default package, foreign currency or special-security rights separately')
        if kind in {'no_standard_terms', 'insolvency'}:
            tasks.append('Obtain cancellation/recovery terms; do not infer zero proceeds from bankruptcy or delisting')
        records.append({'id': f"terminal:{event['ticker']}:{event['date']}",
                        'ticker': event['ticker'], 'date': event['date'], 'classification': kind,
                        'status': 'open', 'vendor_diagnostic_terms_present': event.get('diagnostic_consideration_usd_per_share') is not None,
                        'source_records_are_ex_post': True, 'required_checks': tasks,
                        'economic_return_ready': False})
    return records


def audit_reference(panel, quality, terminal, output):
    panel, quality, terminal, output = map(Path, (panel, quality, terminal, output))
    if output.exists():
        raise ValueError('Choose a new immutable reference audit directory')
    manifest = json.loads((panel / 'panel.json').read_text())
    pins = json.loads((panel / 'sources.json').read_text())
    quality_summary = json.loads((quality / 'summary.json').read_text())
    for name, expected in manifest['files'].items():
        if file_hash(panel / name) != expected:
            raise ValueError('Panel input hash mismatch')
    for name, expected in quality_summary['files'].items():
        if file_hash(quality / name) != expected:
            raise ValueError('Prior quality audit hash mismatch')
    for name, expected in quality_summary['input_hashes'].items():
        if file_hash(panel / name) != expected:
            raise ValueError('Quality audit belongs to a different panel')
    attribution = quality_summary['terminal_evidence']
    for name, expected in attribution['hashes'].items():
        if file_hash(terminal / name) != expected:
            raise ValueError('Terminal evidence hash mismatch')
    terminal_pins = json.loads((terminal / 'sources.json').read_text())
    pins['descriptions'] = terminal_pins['descriptions']
    for name in ('tickers', 'actions', 'fundamentals', 'descriptions'):
        if file_hash(Path(pins[name]['path'])) != pins[name]['parquet_sha256']:
            raise ValueError('Raw reference-source hash mismatch')
    observed = max(observation_time(pins[name]['retrieved_at']) for name in ('tickers', 'actions', 'fundamentals'))
    with duckdb.connect(':memory:', config={'memory_limit': '1GB', 'threads': '2'}) as db:
        for name in ('tickers', 'actions', 'fundamentals', 'descriptions'):
            db.read_parquet(pins[name]['path']).create_view(name)
        db.read_parquet(str(panel / 'features.parquet')).create_view('features')
        selected = db.execute('SELECT DISTINCT ticker FROM features ORDER BY ticker').df()
        db.register('selected', selected)
        metadata = db.execute('SELECT t.* FROM tickers t JOIN selected USING(ticker)').df()
        actions = db.execute('SELECT a.* FROM actions a JOIN selected USING(ticker)').df()
        facts = db.execute("""SELECT f.ticker,f.date,f.reportperiod,f.lastupdated,f.marketcap,f.sharesbas,f.price,f.sharefactor
                              FROM fundamentals f JOIN selected USING(ticker)
                              WHERE dimension='ART' AND date BETWEEN DATE '2012-01-01' AND DATE '2022-12-31'
                              ORDER BY ticker,date""").df()
        units = db.execute("""SELECT \"table\",indicator,unittype FROM descriptions
                              WHERE (\"table\"='SF1' AND indicator IN ('marketcap','price','sharesbas','sharefactor'))
                                 OR (\"table\"='DAILY' AND indicator='marketcap') ORDER BY \"table\",indicator""").df()
    cap_units = units[(units['table'] == 'SF1') & (units.indicator == 'marketcap')]
    if len(cap_units) != 1:
        raise ValueError('Unique pinned SF1 market-cap unit required')
    for indicator, expected in {'price': 'USD/share', 'sharesbas': 'units', 'sharefactor': 'ratio'}.items():
        selected_units = units[(units['table'] == 'SF1') & (units.indicator == indicator)]
        if len(selected_units) != 1 or selected_units.iloc[0].unittype != expected:
            raise ValueError('Pinned price/share-factor units do not support cap reconstruction')
    identities = stable_identities(metadata, selected.ticker)
    transitions, conflicts = paired_transitions(metadata, actions)
    for row in identities + transitions:
        row['observed_at'] = observed.isoformat()
    shares, flagged = share_basis_diagnostics(facts, marketcap_unit=cap_units.iloc[0].unittype)
    events = json.loads((terminal / 'events.json').read_text())
    work = terminal_work_items(events)
    queue = json.loads((quality / 'quality-queue.json').read_text())
    reasons = {item['id']: item for item in work}
    for item in queue:
        item['remediation_status'] = 'external_evidence_required'
        if item['id'] == 'historical_delivery_and_revision_vintages':
            item['remediation_status'] = 'forward_archive_selection_implemented; historical_gap_open'
        elif item['id'] == 'dated_security_and_issuer_mapping':
            item['remediation_status'] = 'stable_share_class_ids_and_paired_transitions_audited; issuer_and_eligibility_gap_open'
        elif item['id'] == 'marketcap_share_vintages':
            item['remediation_status'] = 'ART_formula_diagnostics_complete; DAILY_delivery_and_share_vintage_gap_open'
        elif item['id'] in reasons:
            item['required_checks'] = reasons[item['id']]['required_checks']
            item['vendor_diagnostic_terms_present'] = reasons[item['id']]['vendor_diagnostic_terms_present']
    output.mkdir(parents=True)
    atomic_json(output / 'source-pins.json', {name: pins[name] for name in ('tickers', 'actions', 'fundamentals', 'descriptions')})
    atomic_json(output / 'unit-evidence.json', units.to_dict('records'))
    atomic_json(output / 'security-identities.json', identities)
    atomic_json(output / 'attribute-transitions.json', transitions)
    atomic_json(output / 'transition-conflicts.json', conflicts)
    atomic_json(output / 'terminal-work-items.json', work)
    atomic_json(output / 'quality-queue.json', queue)
    flagged.to_csv(output / 'share-basis-flags.csv', index=False)
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'observed_at': observed.isoformat(),
               'panel_sha256': file_hash(panel / 'panel.json'),
               'stable_security_ids': len([row for row in identities if row['status'] == 'mapped']),
               'security_id_unresolved': len([row for row in identities if row['status'] != 'mapped']),
               'paired_attribute_transitions': len(transitions), 'transition_conflicts': len(conflicts),
               'share_basis': shares, 'terminal_observations': len(work),
               'terminal_classifications': dict(Counter(row['classification'] for row in work)),
               'open_original_issues': len([row for row in queue if row['status'] == 'open']),
               'point_in_time_attested': False, 'terminal_actions_reconciled': False, 'investment_ready': False,
               'llm_calls': 0, 'new_strategy_trials': 0, 'holdout_performance_evaluated': False, 'pnl_modified': False,
               'borrow_dataset_check': {'sharadar': 'No documented lending availability/fee dataset or matching local indicators',
                                        'tiingo': 'No documented lending availability/fee endpoint in published catalog',
                                        'status': 'external_source_required; no shortability inferred'},
               'references': ['https://sharadar.com/docs/tickers', 'https://sharadar.com/docs/faqs',
                              'https://sharadar.com/docs/fundamentals', 'https://www.tiingo.com/documentation/'],
               'limits': ['Vendor permanent ID is per share class, not an issuer ID',
                          'Paired dated changes are retrospective; history completeness and arrivals unverified',
                          'Current category/country/primary status is not dated eligibility',
                          'SF1 formula agreement cannot prove DAILY share vintages or tradability',
                          'Terminal work items require primary evidence; no payouts inserted'],
               'files': {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}}
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--quality-audit', type=Path, required=True)
    parser.add_argument('--terminal-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_reference(args.panel, args.quality_audit, args.terminal_audit, args.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
