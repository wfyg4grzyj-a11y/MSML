"""Local terminal-event evidence audit; never inject inferred proceeds into P&L.

Standard USD cash and common-stock exchange terms support a diagnostic mark.
Payment dates, holder elections and historical availability require further
evidence. This audit cannot attest terminal returns or a point-in-time panel.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_data.sharadar_cli import load_config
from alpha_lab.market_data.sharadar_store import atomic_json

STANDARD_ACTIONS = {"acquisitioncash", "acquisitionstock", "acquisitionby", "acquisitionof"}


def diagnose_terminal_event(event, actions, prices, quote_currencies):
    """Ex-post evidence only: prices at/after the gap may identify resumed trading.

    Never combine default consideration with an elective or special-security
    package. No terminal market-cap value is treated as a per-share entitlement.
    """
    date, ticker = pd.Timestamp(event["date"]), event["ticker"]
    target_prices = prices[prices.ticker==ticker].sort_values("date")
    before = target_prices[target_prices.date<date]
    after = target_prices[target_prices.date>=date]
    nearby = actions[(actions.ticker==ticker)
                     & actions.date.between(date-pd.Timedelta(days=14),date)]
    record = {**event, "audit_mode":"ex_post_diagnostic_only",
              "terminal_actions_reconciled":False,
              "economic_return_ready":False,
              "nearby_action_types":sorted(nearby.action.unique().tolist()),
              "resumed_quote_date":str(after.date.min().date()) if len(after) else None,
              "unresolved_reasons":["settlement_and_information_timing_not_verified"],
              "diagnostic_consideration_usd_per_share":None,
              "diagnostic_valuation_date":str(date.date()),
              "diagnostic_difference_from_last_raw_close":None}
    if len(before):
        last = before.iloc[-1]
        record.update(last_quote_date=str(last.date.date()),last_raw_close=float(last.closeunadj),
                      last_adjusted_close=float(last.closeadj))
    else:
        record["unresolved_reasons"].append("no_prior_quote")
    delists = nearby.loc[nearby.action=="delisted","date"].unique()
    if len(delists)!=1:
        record["classification"]="ambiguous_or_missing_delisting"
        record["unresolved_reasons"].append("unique_delisting_date_required")
        return record
    terminal_date = pd.Timestamp(delists[0])
    package = nearby[nearby.date==terminal_date]
    types = set(package.action)
    record["terminal_date"] = str(terminal_date.date())
    record["package"] = [{"action":row.action, "value":float(row.value) if pd.notna(row.value) else None,
                          "contraticker":str(row.contraticker) if pd.notna(row.contraticker) else None}
                         for row in package.itertuples()]
    if len(after):
        record["unresolved_reasons"].append("quotes_resume_after_reported_delisting")
    if not len(before) or before.iloc[-1].date!=terminal_date:
        record["unresolved_reasons"].append("last_quote_and_action_date_differ")
    complex_types = {name for name in types if name.startswith("acquisition") and name not in STANDARD_ACTIONS}
    if complex_types:
        record["classification"]="election_foreign_cash_or_special_security"
        record["unresolved_reasons"].append("nonstandard_or_mutually_exclusive_consideration")
        return record
    cash = package[package.action=="acquisitioncash"]
    stock = package[package.action=="acquisitionstock"]
    if not len(cash) and not len(stock):
        record["classification"]="insolvency" if "bankruptcyliquidation" in types else "no_standard_terms"
        record["unresolved_reasons"].append("terminal_proceeds_unknown")
        return record
    record["classification"]="cash_and_stock" if len(cash) and len(stock) else "cash" if len(cash) else "stock"
    problems=[]
    if len(cash)>1 or stock.contraticker.duplicated().any():
        problems.append("ambiguous_duplicate_consideration_legs")
    values=pd.concat([cash.value,stock.value])
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (values<0).any():
        problems.append("invalid_consideration_units")
    consideration=float(cash.value.sum())
    stock_marks=[]
    for row in stock.itertuples():
        contra = row.contraticker
        # Price the received shares at the event/session being audited. A newly
        # created merger successor may not have existed on the last target day.
        # No later quote is used when this session has no receiving-security quote.
        quote = prices[(prices.ticker==contra) & (prices.date==date)]
        if not isinstance(contra,str) or contra in {"", "N/A"} or len(quote)!=1:
            problems.append("received_security_quote_unavailable_on_audit_session")
            continue
        raw = float(quote.iloc[0].closeunadj)
        if not np.isfinite(raw) or raw<=0 or quote_currencies.get(contra)!={"USD"}:
            problems.append("received_security_price_or_currency_unverified")
            continue
        stock_marks.append({"ticker":contra,"units":float(row.value),"raw_close":raw,
                            "mark_date":str(date.date())})
        consideration += float(row.value)*raw
    record["received_stock_marks"] = stock_marks
    if quote_currencies.get(ticker)!={"USD"}:
        problems.append("target_listing_currency_unverified")
    record["unresolved_reasons"].extend(problems)
    if not problems and len(before):
        raw=float(before.iloc[-1].closeunadj)
        if np.isfinite(raw) and raw>0:
            record["diagnostic_consideration_usd_per_share"]=consideration
            record["diagnostic_difference_from_last_raw_close"]=consideration/raw-1
            record["unresolved_reasons"].append("adjusted_price_and_distribution_basis_not_reconciled")
    return record


def audit_terminal_events(panel:Path, validation:Path, output:Path, config:Path):
    if output.exists():
        raise ValueError("Choose a new audit directory; existing evidence is immutable")
    manifest=json.loads((panel/"panel.json").read_text())
    for name in ("sources.json","actions.parquet"):
        if file_hash(panel/name)!=manifest["files"][name]:
            raise ValueError("Panel audit source hash mismatch")
    sourcepins=json.loads((panel/"sources.json").read_text())
    held=json.loads((validation/"holdout.json").read_text())
    events=held["missing_event_details"]
    store,_,_=load_config(config)
    sourcepins["descriptions"]=store.latest("descriptions")
    tables=("stocks","actions","tickers","descriptions")
    for table in tables:
        if file_hash(Path(sourcepins[table]["path"]))!=sourcepins[table]["parquet_sha256"]:
            raise ValueError("Pinned raw-source hash mismatch")
    output.mkdir(parents=True)
    atomic_json(output/"sources.json",{table:sourcepins[table] for table in tables})
    with store._connect() as db:
        for table in tables:
            db.read_parquet(sourcepins[table]["path"]).create_view(table)
        targets=pd.DataFrame({"ticker":pd.Series(sorted({event["ticker"] for event in events}),dtype='str')})
        db.register("targets",targets)
        actions=db.execute("SELECT a.ticker,a.date,a.action,a.value,a.contraticker FROM actions a JOIN targets USING(ticker)").df()
        contra=actions.loc[actions.action=='acquisitionstock','contraticker'].dropna().unique().tolist()
        selected=pd.DataFrame({"ticker":pd.Series(sorted(set(targets.ticker)|set(contra)),dtype='str')})
        db.register("selected",selected)
        first=min((event["date"] for event in events),default=manifest['start'])
        last=max((event["date"] for event in events),default=manifest['end'])
        # Raw and adjusted bases are retained separately; no quote forward fill.
        prices=db.execute("""SELECT ticker,date,closeunadj,closeadj FROM stocks JOIN selected USING(ticker)
                             WHERE date BETWEEN CAST(? AS DATE)-INTERVAL 30 DAY
                                            AND CAST(? AS DATE)+INTERVAL 30 DAY""",[first,last]).df()
        # SF1 currency is functional reporting currency, not the SEP quote unit.
        metadata=db.execute('SELECT ticker,currency FROM tickers JOIN selected USING(ticker) WHERE "table"=\'SEP\'').df()
        price_units=db.execute('SELECT unittype FROM descriptions WHERE "table"=\'SEP\' AND indicator=\'closeunadj\'').fetchall()
        if price_units!=[('USD/share',)]:
            raise ValueError("Pinned dictionary does not verify the stock quote unit")
        dictionary=db.execute('SELECT indicator,description FROM descriptions WHERE "table"=\'ACTIONTYPES\'').df()
    quote_currencies={ticker:set(cross.currency.dropna()) for ticker,cross in metadata.groupby('ticker')}
    records=[diagnose_terminal_event(event,actions,prices,quote_currencies) for event in events]
    atomic_json(output/"events.json",records)
    dictionary.to_csv(output/"action-dictionary.csv",index=False)
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),
             "event_observations":len(records),"event_days":len({event['date'] for event in records}),
             "classifications":dict(Counter(record['classification'] for record in records)),
             "diagnostic_standard_consideration_marks":sum(record['diagnostic_consideration_usd_per_share'] is not None for record in records),
             "unresolved_reasons":dict(Counter(reason for record in records for reason in set(record['unresolved_reasons']))),
             "original_panel_sha256":file_hash(panel/"panel.json"),
             "original_results_sha256":file_hash(validation/"holdout.json"),
             "terminal_actions_reconciled":False,"investment_ready":False,
             "pnl_modified":False,"licensed_data_and_results":"local_only",
             "limitations":["Ex-post evidence is not historically admissible trading information",
                            "Diagnostic consideration uses raw receiving-security closes on the audited missing-price session",
                            "Settlement date, holder elections, distributions and share-unit basis need separate evidence",
                            "Delisting, merger and bankruptcy market-cap values are not per-share payouts"]}
    atomic_json(output/"summary.json",summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('data/market/sharadar.json'))
    args=parser.parse_args()
    audit_terminal_events(args.panel,args.validation,args.output,args.config)


if __name__=='__main__':
    main()
