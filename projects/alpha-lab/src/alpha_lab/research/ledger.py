"""One event ledger: raw shares, USD cash and signed action entitlements.

Missing valuations and unpriced costs are explicit; stale marks never execute.
"""
from __future__ import annotations

from datetime import datetime
import math

from .contracts import ResearchError


def validate_calendar(scenario):
    from alpha_lab.equity_calendar import session_schedule
    expected, _ = session_schedule(scenario.sessions[0].date, scenario.sessions[-1].date)
    actual = [(s.date, datetime.fromisoformat(s.close_utc)) for s in scenario.sessions]
    wanted = [(str(row.date.date()), datetime.fromisoformat(str(row.session_close_utc))) for row in expected.itertuples()]
    if actual != wanted:
        raise ResearchError('Ledger needs every official XNYS session and its exact UTC close')


def lending_at(scenario, security, known, through):
    known, through = datetime.fromisoformat(known), datetime.fromisoformat(through)
    candidates = [l for l in scenario.lending if l.security_id == security
                  and datetime.fromisoformat(l.known_at) <= known]
    if not candidates:
        return None
    latest = max(datetime.fromisoformat(l.known_at) for l in candidates)
    selected = [l for l in candidates if datetime.fromisoformat(l.known_at) == latest]
    if len(selected) != 1:
        raise ResearchError('Ambiguous contemporaneous stock-loan evidence')
    return selected[0] if datetime.fromisoformat(selected[0].valid_through)>=through else None


class Ledger:
    def __init__(self, scenario):
        self.spec, self.policy = scenario, scenario.policy
        self.cash = scenario.initial_cash_usd
        self.shares = dict(scenario.initial_shares)
        self.marks = {sid: q.raw_close_usd for sid, q in scenario.sessions[0].quotes.items()}
        if any(q and sid not in self.marks for sid, q in self.shares.items()):
            raise ResearchError('Opening positions need observed raw marks')
        self.claims, self.pending, self.journal, self.orders, self.holdings = [], [], [], [], []
        self.daily, self.terminal, self.unpriced_costs = [], set(), []
        self.tickers = dict(scenario.securities)

    def nav(self):
        position = sum(q * self.marks.get(sid, 0.) for sid, q in self.shares.items() if q)
        claims = sum(c['amount'] if c['kind'] == 'cash' else c['amount'] * self.marks.get(c['security_id'], 0.) for c in self.claims)
        return self.cash + position + claims

    def note(self, day, kind, **values):
        self.journal.append({'date': day, 'kind': kind, **values})

    def actions(self, session):
        priority = {'split': 0, 'cash_dividend': 1, 'rename': 2, 'unresolved_terminal': 3, 'unresolved_distribution': 3,
                    'cash_merger': 4, 'stock_merger': 4, 'mixed_merger': 4}
        actions = sorted((a for a in self.spec.actions if a.date == session.date), key=lambda a: (priority[a.kind], a.action_id))
        merger_ids = [a.security_id for a in actions if 'merger' in a.kind]
        if len(set(merger_ids)) != len(merger_ids):
            raise ResearchError('Represent mixed consideration in one evidenced merger package')
        for a in actions:
            sid, quantity = a.security_id, self.shares.get(a.security_id, 0.)
            if a.kind == 'split':
                self.shares[sid] = quantity * a.ratio
                if sid in self.marks:
                    self.marks[sid] /= a.ratio
                for order in self.pending:
                    if order['security_id'] == sid:
                        order['target_shares'] *= a.ratio
                for claim in self.claims:
                    if claim['kind'] == 'stock' and claim['security_id'] == sid:
                        claim['amount'] *= a.ratio
            elif a.kind == 'rename':
                self.tickers[sid] = a.new_ticker
            elif a.kind == 'unresolved_terminal':
                if quantity:
                    self.terminal.add(sid)
            elif a.kind == 'unresolved_distribution':
                if quantity:
                    self.unpriced_costs.append({'date':session.date,'security_id':sid,
                                                'reason':'unresolved_distribution_entitlement','entitled_shares':quantity})
            else:
                if a.kind != 'cash_dividend':
                    self.shares[sid] = 0.
                    self.terminal.add(sid)  # Former security cannot trade after conversion.
                if a.cash_usd_per_share is not None and quantity:
                    self.claims.append({'action_id': a.action_id, 'kind': 'cash', 'security_id': sid,
                                        'amount': quantity*a.cash_usd_per_share, 'due': a.payment_date})
                if a.recipient_id and quantity:
                    recipient = a.recipient_id
                    if recipient not in self.marks and recipient in session.quotes:
                        self.marks[recipient] = session.quotes[recipient].raw_close_usd
                    self.claims.append({'action_id': a.action_id, 'kind': 'stock', 'security_id': recipient,
                                        'amount': quantity*a.ratio, 'due': a.payment_date})
                # Rights of unsettled stock consideration need separate evidence.
                if a.kind == 'cash_dividend' and any(c['kind'] == 'stock' and c['security_id'] == sid for c in self.claims):
                    self.unpriced_costs.append({'date': session.date, 'security_id': sid,
                                                'reason': 'unsettled_stock_dividend_entitlement_unknown'})
            self.note(session.date, 'action', action_id=a.action_id, security_id=sid, action_kind=a.kind,
                      entitled_prior_shares=quantity, evidence=a.evidence, evidence_kind=a.evidence_kind)
        remaining = []
        for claim in self.claims:
            if claim['due'] <= session.date:
                if claim['kind'] == 'cash':
                    self.cash += claim['amount']
                else:
                    sid = claim['security_id']
                    self.shares[sid] = self.shares.get(sid, 0.) + claim['amount']
                self.note(session.date, 'delivery', claim_kind=claim['kind'], **{k:v for k,v in claim.items() if k != 'kind'})
            else:
                remaining.append(claim)
        self.claims = remaining

    def accrue(self, previous, session):
        days = (datetime.fromisoformat(session.date) - datetime.fromisoformat(previous.date)).days
        year = 365 if self.policy.day_count == 'ACT365F' else 360
        short_collateral = sum(-q*self.marks.get(sid, 0.) for sid, q in self.shares.items() if q < 0)
        short_collateral += sum(-c['amount']*(1. if c['kind']=='cash' else self.marks.get(c['security_id'],0.))
                                for c in self.claims if c['amount']<0)
        credit = max(self.cash-short_collateral, 0.)*self.policy.annual_cash_credit*days/year
        debit = max(-self.cash, 0.)*self.policy.annual_cash_debit*days/year
        borrow = 0.
        for claim in self.claims:
            if claim['kind']=='stock' and claim['amount']<0:
                self.unpriced_costs.append({'date':session.date,'security_id':claim['security_id'],
                                            'reason':'short_stock_claim_financing_before_delivery_unknown'})
        for sid, q in self.shares.items():
            if q >= 0:
                continue
            evidence = lending_at(self.spec, sid, previous.close_utc, session.close_utc)
            rate = evidence.annual_fee if evidence else self.policy.hypothetical_borrow_rate
            if sid not in self.marks:
                rate = None
            if rate is None:
                self.unpriced_costs.append({'date': session.date, 'security_id': sid,
                                            'reason': 'missing_period_stock_loan_rate_or_mark', 'notional_usd': -q*self.marks[sid] if sid in self.marks else None})
            else:
                fee = -q*self.marks[sid]*rate*days/year
                borrow += fee
                self.note(session.date, 'borrow', security_id=sid, calendar_days=days, amount_usd=fee,
                          assumption=evidence is None, evidence=evidence.evidence if evidence else None)
        self.cash += credit-debit-borrow
        return credit, debit, borrow

    def constraint_violations(self, shares=None, cash=None):
        shares, cash = (self.shares if shares is None else shares), (self.cash if cash is None else cash)
        if any(q and sid not in self.marks for sid,q in shares.items()):
            return ['unmarked_position']
        equity = {sid: q*self.marks[sid] for sid, q in shares.items() if q}
        nav = self.nav() + cash-self.cash + sum(equity.values()) - sum(q*self.marks.get(sid,0.) for sid, q in self.shares.items() if q)
        if nav <= 0:
            return ['nonpositive_nav']
        errors = []
        if sum(abs(v) for v in equity.values())/nav > self.policy.max_gross+1e-10:
            errors.append('max_gross')
        if any(abs(v)/nav > self.policy.max_name+1e-10 for v in equity.values()):
            errors.append('max_name')
        return errors

    def fills(self, index, session, candidate_check=None):
        current = [o for o in self.pending if o['fill_index'] == index]
        self.pending = [o for o in self.pending if o['fill_index'] != index]
        proposed, cash, records = dict(self.shares), self.cash, []
        for order in sorted(current, key=lambda o: o['security_id']):
            sid = order['security_id']; held = self.shares.get(sid, 0.)
            requested = order['target_shares']-held; quote = session.quotes.get(sid)
            status, filled, fee, slip = 'filled', 0., 0., 0.
            if sid in self.terminal or quote is None or not quote.tradable:
                status = 'unfilled_no_executable_quote_or_ceased_security'
            elif abs(requested) > 1e-12:
                filled = math.copysign(min(abs(requested), quote.volume_shares*self.policy.participation), requested)
                candidate = held+filled
                reducing = abs(candidate) <= abs(held) and candidate*held >= 0
                unpriced = (self.unpriced_costs or any(q and key not in session.quotes for key,q in self.shares.items())
                            or any(c['kind']=='stock' and c['security_id'] not in session.quotes for c in self.claims))
                if unpriced and not reducing:
                    status, filled = 'rejected_unresolved_nav', 0.
                if filled and candidate < min(held, 0.):
                    evidence = lending_at(self.spec, sid, order['decision_time'], session.close_utc)
                    limit = evidence.available_shares if evidence else (math.inf if self.policy.borrow_mode == 'explicit_hypothetical' else 0.)
                    now = lending_at(self.spec, sid, session.close_utc, session.close_utc)
                    if now is not None: limit = min(limit, now.available_shares)
                    elif self.policy.borrow_mode=='evidenced_only':limit=0.
                    if -candidate > limit+1e-12: status, filled = 'rejected_short_locate', 0.
                if filled:
                    price = quote.raw_close_usd*(1+math.copysign(self.policy.slippage_bps/10000, filled))
                    fee = abs(filled)*price*self.policy.commission_bps/10000
                    slip = filled*(price-quote.raw_close_usd)
                    proposed[sid] = held+filled; cash -= filled*price+fee
                    if abs(filled) < abs(requested)-1e-12: status = 'partial_remainder_expired'
            records.append({**order, 'date': session.date, 'requested_delta': requested, 'filled_delta': filled,
                            'unfilled_delta': requested-filled, 'status': status, 'commission_usd': fee,
                            'slippage_usd': slip, 'raw_mark_usd': quote.raw_close_usd if quote else None})
        violations = self.constraint_violations(proposed,cash)
        if not self.policy.allow_cash_debit and cash < -self.policy.tolerance_usd: violations.append('cash_debit')
        if candidate_check is not None: violations += candidate_check(self,session,proposed,cash)
        reductions_only = all(abs(proposed.get(sid,0.)) <= abs(self.shares.get(sid,0.))+1e-12
                              and proposed.get(sid,0.)*self.shares.get(sid,0.) >= 0 for sid in proposed)
        reject = bool(violations) and (not reductions_only or 'cash_debit' in violations)
        if reject:
            for record in records:
                if record['filled_delta']:
                    record.update(status='rejected_batch_constraints',filled_delta=0.,unfilled_delta=record['requested_delta'],
                                  commission_usd=0.,slippage_usd=0.)
                record['batch_violations'] = sorted(set(violations))
        else:
            self.shares,self.cash = proposed,cash
            if violations:
                for record in records:
                    record['batch_violations'] = sorted(set(violations))
                    if record['filled_delta']:record['status'] = 'filled_derisk_preexisting_violation'
        self.orders.extend(records)

    def run(self, *, decision_provider=None, candidate_check=None, position_reporter=None):
        validate_calendar(self.spec)
        opening = self.nav()
        previous, previous_resolved = None, True
        for index, session in enumerate(self.spec.sessions):
            before = self.nav()
            credit, debit, borrow = self.accrue(previous, session) if previous else (0., 0., 0.)
            accrued = self.nav()
            self.actions(session)
            action_nav = self.nav()
            for sid, quote in session.quotes.items():
                self.marks[sid] = quote.raw_close_usd
            marked = self.nav()
            start_orders = len(self.orders)
            self.fills(index, session, candidate_check)
            end = self.nav()
            newly = self.orders[start_orders:]
            costs = sum(o['commission_usd']+o['slippage_usd'] for o in newly)
            residual = end-before-(credit-debit-borrow+(action_nav-accrued)+(marked-action_nav)-costs)
            if abs(residual) > self.policy.tolerance_usd or not math.isfinite(end):
                raise ResearchError('Ledger NAV does not reconcile to cash, actions, marks and signed trade costs')
            held = {sid for sid, q in self.shares.items() if q}
            stock_claims = {c['security_id'] for c in self.claims if c['kind'] == 'stock' and c['amount']}
            missing = sorted((held | stock_claims)-session.quotes.keys())
            terminal = sorted(held & self.terminal)
            resolved = not missing and not terminal and not self.unpriced_costs
            actual_return = end/before-1 if previous and previous_resolved and resolved and before > 0 else None
            never_marked = any(sid not in self.marks for sid in held | stock_claims)
            self.daily.append({'date': session.date, 'cash_usd': self.cash, 'receivables_net_usd':
                               sum(c['amount'] if c['kind'] == 'cash' else c['amount']*self.marks.get(c['security_id'],0.) for c in self.claims),
                               'nav_estimate_usd': None if never_marked else end,
                               'known_components_nav_usd': end, 'nav_observed_usd': end if resolved else None,
                               'return_observed': actual_return, 'valuation_resolved': resolved,
                               'cash_interest_usd': credit, 'funding_usd': debit, 'borrow_usd': borrow,
                               'action_pnl_usd': action_nav-accrued, 'raw_mark_pnl_usd': marked-action_nav,
                               'trade_cost_usd': costs, 'reconciliation_residual_usd': residual,
                               'missing_marks': missing, 'unresolved_terminal': terminal,
                               'unpriced_cost_count': len(self.unpriced_costs), 'constraint_violations': self.constraint_violations()})
            for sid in sorted(held | stock_claims):
                self.holdings.append({'date': session.date, 'security_id': sid, 'ticker': self.tickers[sid],
                                      'shares': self.shares.get(sid,0.), 'raw_mark_usd': self.marks.get(sid),
                                      'quote_observed': sid in session.quotes, 'economic_terminal_unknown': sid in terminal})
            if position_reporter is not None:
                position_reporter(self,session)
            decisions = [d for d in self.spec.decisions if d.date == session.date]
            if decision_provider is not None:
                decisions.extend(decision_provider(self,session))
            for decision in decisions:
                for sid, target in decision.target_shares.items():
                    cancelled = [o for o in self.pending if o['security_id']==sid]
                    self.pending = [o for o in self.pending if o['security_id']!=sid]
                    for old in cancelled:self.note(session.date,'cancel_replaced_pending',decision_id=old['decision_id'],security_id=sid)
                    if index+decision.fill_lag_sessions >= len(self.spec.sessions):
                        self.note(session.date,'decision_right_censored',decision_id=decision.decision_id,security_id=sid)
                        continue
                    self.pending.append({'decision_id': decision.decision_id, 'security_id': sid, 'target_shares': target,
                                         'decision_time': session.close_utc, 'fill_index': index+decision.fill_lag_sessions})
            previous, previous_resolved = session, resolved
        final = self.daily[-1]
        return {'methodology': self.policy.methodology, 'initial_nav_usd': opening,
                'final_nav_observed_usd': final['nav_observed_usd'], 'final_nav_estimate_usd': final['nav_estimate_usd'],
                'endpoint_return': final['nav_observed_usd']/opening-1 if opening>0 and final['valuation_resolved'] else None,
                'daily': self.daily, 'holdings': self.holdings, 'orders': self.orders, 'journal': self.journal,
                'claims_outstanding': self.claims, 'unpriced_costs': self.unpriced_costs,
                'investment_ready': False, 'history_mode': self.spec.history_mode,
                'point_in_time_attested': self.spec.point_in_time_attested,
                'terminal_audit_complete': self.spec.terminal_audit_complete,
                'source_economic_outcomes_resolved': self.spec.economic_outcomes_resolved,
                'economic_outcomes_resolved': self.spec.economic_outcomes_resolved and not self.unpriced_costs
                    and not any(d['unresolved_terminal'] for d in self.daily),
                'borrow_evidence': self.policy.borrow_mode, 'llm_calls': 0}
