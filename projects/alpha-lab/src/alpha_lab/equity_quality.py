"""Local research data-quality evidence and unresolved audit queue; no backtests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_lab.equity_calendar import session_schedule, validate_benchmark, research_availability
from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import MarketNeutralBacktest


def audit_panel(panel: Path, output: Path, *, terminal_audit: Path | None = None):
    """Inspect immutable inputs; leave failed quality checks and missing evidence open.

    Calendar integrity covers the full requested date range, but feature-quality
    summaries stop in 2022. No strategy, P&L evaluation or candidate selection.
    A supplied terminal audit is separately attributed evidence, never a claim
    that the current panel has verified economic proceeds.
    """
    panel, output = Path(panel), Path(output)
    if output.exists():
        raise ValueError("Choose a new quality directory; prior evidence is immutable")
    engine = MarketNeutralBacktest(panel)
    if not engine.panel_integrity_verified:
        raise ValueError("Quality audit requires complete verified panel file hashes")
    manifest = json.loads((panel/'panel.json').read_text())
    sources = json.loads((panel/'sources.json').read_text())
    required = {'stocks','funds','daily','fundamentals','tickers','actions'}
    if not required <= set(sources):
        raise ValueError("Quality audit requires all six pinned raw-source snapshots")
    for source in sources.values():
        if file_hash(Path(source['path'])) != source['parquet_sha256']:
            raise ValueError("Quality audit raw-source hash mismatch")
    input_hashes = {'panel.json': file_hash(panel/'panel.json'), 'sources.json': file_hash(panel/'sources.json')}
    output.mkdir(parents=True)
    findings = []

    def issue(code, evidence, resolution, *, scope='panel'):
        findings.append({'id':code,'scope':scope,'status':'open','evidence':evidence,
                         'required_evidence':resolution,'blocks_investment_ready':True})

    warmup = (pd.Timestamp(manifest['start'])-pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    schedule, calendar_metadata = session_schedule(warmup, manifest['end'])
    expected = pd.DatetimeIndex(schedule.date)
    missing, extra = expected.difference(engine.dates), engine.dates.difference(expected)
    calendar_complete = not len(missing) and not len(extra)
    if not calendar_complete:
        issue('panel_session_mismatch', {'missing':missing.strftime('%Y-%m-%d').tolist(),
                                        'extra':extra.strftime('%Y-%m-%d').tolist()},
              'Rebuild a new panel from every modeled exchange session; never compress missing dates')
    with duckdb.connect(':memory:',config={'memory_limit':'256MB','threads':'2'}) as db:
        benchmark = db.execute("SELECT date,closeadj FROM read_parquet(?) WHERE ticker='SPY' AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) ORDER BY date",
                               [sources['funds']['path'],warmup,manifest['end']]).df()
    benchmark_complete = True
    try:
        validate_benchmark(benchmark,schedule)
    except ValueError as error:
        benchmark_complete = False
        issue('benchmark_session_or_price_failure',str(error),'Repair source coverage; adjusted SPY closes must be positive and finite on each session')
    benchmark_returns_match = False
    if calendar_complete and benchmark_complete:
        prices = benchmark.set_index('date').closeadj.reindex(expected)
        raw_returns = prices.pct_change(fill_method=None).to_numpy(dtype=np.float32).astype(float)
        benchmark_returns_match = np.array_equal(raw_returns,engine.spy,equal_nan=True)
        if not benchmark_returns_match:
            issue('benchmark_returns_mismatch','Panel benchmark returns differ from pinned raw SPY adjacent-session returns',
                  'Rebuild an immutable panel and investigate construction before further experiments')
    schedule.to_csv(output/'exchange-sessions.csv',index=False)

    features = engine.features.loc[engine.features.date<=pd.Timestamp('2022-12-31')].copy()
    filing = pd.to_datetime(features.filing_date)
    available = (pd.to_datetime(features.available_date) if 'available_date' in features
                 else filing + pd.Timedelta(days=2))
    age = (features.date-filing).dt.days
    available_age = (features.date-available).dt.days
    timing_bad = filing.notna() & ((available!=filing+pd.Timedelta(days=2)) | (age<2) | (available_age>550))
    if timing_bad.any():
        issue('accounting_availability_violation',int(timing_bad.sum()),'Reconcile filing-date lag and availability-age limits')
    if 'reportperiod' in features:
        future_period = pd.to_datetime(features.reportperiod)>features.date
        if future_period.any():
            issue('future_reporting_period',int(future_period.sum()),'Investigate dated accounting joins and source records')
    facts = {
        'period':'Train/validation feature quality through 2022 only; no performance calculations',
        'monthly_security_rows':len(features),'missing_accounting_rows':int(filing.isna().sum()),
        'minimum_filing_age_days':int(age.min()) if age.notna().any() else None,
        'maximum_filing_age_days':int(age.max()) if age.notna().any() else None,
        'maximum_available_age_days':int(available_age.max()) if available_age.notna().any() else None,
        'availability_rule_violations':int(timing_bad.sum()),
        'assumed_filing_lag_calendar_days':2,'maximum_permitted_available_age_days':550,
        'maximum_permitted_filing_age_days':552,
    }
    issue('historical_delivery_and_revision_vintages',
          'Current corrected snapshots; date-only source records and no historical arrival log',
          'Historical arrival/revision evidence for prices, accounting, share counts and market cap')
    issue('dated_security_and_issuer_mapping',
          'Static SF1 domestic primary-common-stock metadata; dataset tickers are not a dated issuer/class master',
          'Dated stable security/issuer identifiers, share classes and eligibility evidence')
    issue('marketcap_share_vintages',
          'Same-date DAILY marketcap; no reconstructed historically available issuer share count',
          'Dated share-count and corporate-action evidence with explicit repricing/issuer aggregation policy')
    issue('terminal_proceeds',
          'Missing returns use assumed stale-price exits; economic proceeds unverified',
          'Reconcile share units, adjustments, distributions, consideration, settlement and information timing')
    issue('borrow_and_execution',
          'Fixed 3% annual borrow and 10bp costs; no historical locate availability or actual financing',
          'Dated locate/borrow, capacity, transaction costs, financing and execution evidence')

    terminal_evidence = None
    if terminal_audit is not None:
        terminal_audit = Path(terminal_audit)
        audit_hashes = {name:file_hash(terminal_audit/name) for name in ('summary.json','events.json','sources.json')}
        terminal_summary = json.loads((terminal_audit/'summary.json').read_text())
        events = json.loads((terminal_audit/'events.json').read_text())
        if not isinstance(events,list) or len(events)!=terminal_summary['event_observations']:
            raise ValueError("Terminal audit event count is inconsistent")
        terminal_evidence = {'path':str(terminal_audit.resolve()),'hashes':audit_hashes,
                             'original_panel_sha256':terminal_summary['original_panel_sha256'],
                             'scope':'Separately frozen ex-post event evidence; no P&L or current-panel proceeds attestation',
                             'event_observations':len(events)}
        for event in events:
            issue(f"terminal:{event['ticker']}:{event['date']}",event['unresolved_reasons'],
                  'Obtain event-specific evidence before assigning an economic terminal return',scope='terminal_event')
    summary = {
        'created_utc':datetime.now(timezone.utc).isoformat(),'panel':str(panel.resolve()),
        'input_hashes':input_hashes,'raw_sources_verified':len(sources),
        'calendar':{**calendar_metadata,'panel_sessions_complete':calendar_complete,
                    'benchmark_complete':benchmark_complete,'benchmark_returns_match':benchmark_returns_match,
                    'missing_panel_sessions':len(missing),'extra_panel_sessions':len(extra)},
        'accounting_quality':facts,'availability_assumptions':research_availability(),
        'availability_note':'Recorded audit interpretation of current builder; legacy manifests may not record all assumptions',
        'terminal_evidence':terminal_evidence,'open_issues':len(findings),
        'holdout_performance_evaluated':False,'new_strategy_trials':0,'pnl_modified':False,
        'point_in_time_attested':False,'terminal_actions_reconciled':False,'investment_ready':False,
        'licensed_data_and_results':'local_only',
    }
    if any(file_hash(panel/name)!=expected_hash for name,expected_hash in input_hashes.items()):
        raise ValueError("Quality audit inputs changed during inspection")
    atomic_json(output/'quality-queue.json',findings)
    summary['files'] = {name:file_hash(output/name) for name in ('exchange-sessions.csv','quality-queue.json')}
    atomic_json(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--terminal-audit',type=Path)
    args = parser.parse_args()
    summary = audit_panel(args.panel,args.output,terminal_audit=args.terminal_audit)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
