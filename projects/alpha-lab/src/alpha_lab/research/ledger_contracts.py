"""Raw-share USD accounting contracts, distinct from adjusted-return diagnostics."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from .contracts import DEVELOPMENT_END, Hash, Identifier, StrictModel, iso_day

Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]


def utc_stamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('Require timezone-aware information time')
    return stamp.astimezone(timezone.utc).isoformat()


class Quote(StrictModel):
    raw_close_usd: Positive
    volume_shares: Nonnegative
    tradable: bool


class Session(StrictModel):
    date: str
    close_utc: str
    quotes: dict[Identifier, Quote]

    _date = field_validator('date')(iso_day)
    _time = field_validator('close_utc')(utc_stamp)


class Lending(StrictModel):
    security_id: Identifier
    known_at: str
    valid_through: str
    available_shares: Nonnegative
    annual_fee: Annotated[float, Field(ge=0, le=10)]
    evidence: Hash
    evidence_kind: Literal['broker_observation', 'synthetic_reference']

    _times = field_validator('known_at', 'valid_through')(utc_stamp)

    @model_validator(mode='after')
    def ordered(self):
        if datetime.fromisoformat(self.known_at) > datetime.fromisoformat(self.valid_through):
            raise ValueError('Lending observation expires before it is known')
        return self


class Action(StrictModel):
    action_id: Identifier
    security_id: Identifier
    date: str
    known_at: str
    kind: Literal['split', 'cash_dividend', 'stock_distribution', 'cash_merger', 'stock_merger', 'mixed_merger', 'rename', 'unresolved_terminal', 'unresolved_distribution']
    ratio: Positive | None
    cash_usd_per_share: Nonnegative | None
    recipient_id: Identifier | None
    payment_date: str | None
    new_ticker: str | None
    evidence: Hash
    evidence_kind: Literal['verified_terms', 'synthetic_reference', 'unresolved', 'declared_dividend_assumption', 'declared_distribution_assumption']
    distribution_convention: Literal['signed_fractional_prior_close_same_session_delivery_research'] | None = None

    @model_serializer(mode='wrap')
    def archived_shape(self, handler):
        value = handler(self)
        if self.distribution_convention is None:
            value.pop('distribution_convention', None)
        return value

    _date = field_validator('date')(iso_day)
    _time = field_validator('known_at')(utc_stamp)

    @model_validator(mode='after')
    def terms(self):
        if self.payment_date is not None:
            iso_day(self.payment_date)
            if self.payment_date < self.date:
                raise ValueError('Payment precedes entitlement')
        expected = {'split': {'ratio'}, 'cash_dividend': {'cash_usd_per_share', 'payment_date'},
                    'cash_merger': {'cash_usd_per_share', 'payment_date'},
                    'stock_merger': {'ratio', 'recipient_id', 'payment_date'}, 'rename': {'new_ticker'},
                    'stock_distribution': {'ratio', 'recipient_id', 'payment_date'},
                    'mixed_merger': {'ratio', 'recipient_id', 'cash_usd_per_share', 'payment_date'},
                    'unresolved_terminal': set(), 'unresolved_distribution': set()}[self.kind]
        terms = {key for key in ('ratio', 'cash_usd_per_share', 'recipient_id', 'payment_date', 'new_ticker')
                 if getattr(self, key) is not None}
        if terms != expected or (self.evidence_kind == 'unresolved') != self.kind.startswith('unresolved_'):
            raise ValueError('Require exact evidenced terms; unknown terms remain unresolved')
        if self.evidence_kind == 'declared_dividend_assumption' and self.kind != 'cash_dividend':
            raise ValueError('Declared dividend assumptions cannot supply merger or other action terms')
        if self.kind == 'stock_distribution':
            if (self.distribution_convention is None or self.payment_date != self.date
                    or self.recipient_id == self.security_id
                    or self.evidence_kind not in ('synthetic_reference', 'declared_distribution_assumption')):
                raise ValueError('Fractional stock distributions require explicit same-session research delivery conventions')
        elif self.distribution_convention is not None or self.evidence_kind == 'declared_distribution_assumption':
            raise ValueError('Distribution conventions apply only to stock distributions')
        return self


class Decision(StrictModel):
    decision_id: Identifier
    date: str
    information_time: str
    target_shares: dict[Identifier, float]
    fill_lag_sessions: Annotated[int, Field(ge=1, le=20)]

    _date = field_validator('date')(iso_day)
    _time = field_validator('information_time')(utc_stamp)


class AccountingPolicy(StrictModel):
    methodology: Literal['raw_share_cash_action_ledger_v1']
    price_basis: Literal['unadjusted_usd_per_share_explicit_actions']
    trade_settlement: Literal['T0_research_cash_not_broker_settlement']
    short_proceeds: Literal['segregated_no_cash_interest_no_rebate']
    dividend_entitlement: Literal['prior_close_position_before_ex_session_orders']
    missing_marks: Literal['last_raw_mark_provisional_no_executable_exit']
    day_count: Literal['ACT365F', 'ACT360']
    annual_cash_credit: Annotated[float, Field(ge=-1, le=10)]
    annual_cash_debit: Annotated[float, Field(ge=0, le=10)]
    commission_bps: Annotated[float, Field(ge=0, le=1000)]
    slippage_bps: Annotated[float, Field(ge=0, le=1000)]
    participation: Annotated[float, Field(gt=0, le=1)]
    allow_cash_debit: bool
    borrow_mode: Literal['evidenced_only', 'explicit_hypothetical']
    hypothetical_borrow_rate: Annotated[float, Field(ge=0, le=10)] | None
    max_gross: Annotated[float, Field(gt=0, le=10)]
    max_name: Annotated[float, Field(gt=0, le=10)]
    tolerance_usd: Annotated[float, Field(gt=0, le=.01)]

    @model_validator(mode='after')
    def borrow(self):
        if (self.borrow_mode == 'explicit_hypothetical') != (self.hypothetical_borrow_rate is not None):
            raise ValueError('A hypothetical borrow scenario requires its explicit rate')
        return self


class LedgerScenario(StrictModel):
    schema_version: Literal[3]
    scenario_id: Identifier
    source_id: Hash
    history_mode: Literal['synthetic_reference', 'public_information_reconstruction', 'archived_vendor_feed']
    point_in_time_attested: bool
    terminal_audit_complete: bool
    economic_outcomes_resolved: bool
    securities: dict[Identifier, str]
    initial_cash_usd: Nonnegative
    initial_shares: dict[Identifier, float]
    policy: AccountingPolicy
    sessions: Annotated[list[Session], Field(min_length=2, max_length=3000)]
    actions: list[Action]
    lending: list[Lending]
    decisions: list[Decision]

    @model_validator(mode='after')
    def consistent(self):
        if self.history_mode=='synthetic_reference' and self.point_in_time_attested:
            raise ValueError('Synthetic fixtures cannot attest historical point-in-time data')
        if self.history_mode!='synthetic_reference' and (
                any(a.evidence_kind=='synthetic_reference' for a in self.actions)
                or any(l.evidence_kind=='synthetic_reference' for l in self.lending)):
            raise ValueError('Real-source scenarios cannot promote synthetic action/loan evidence')
        if self.point_in_time_attested and any(a.evidence_kind in ('declared_dividend_assumption', 'declared_distribution_assumption') for a in self.actions):
            raise ValueError('Assumed dividend terms/settlement cannot attest historical source availability')
        if any(a.kind.startswith('unresolved_') for a in self.actions) and self.economic_outcomes_resolved:
            raise ValueError('Unknown terminal terms cannot be declared economically resolved')
        dates = [s.date for s in self.sessions]
        if dates != sorted(set(dates)) or dates[-1] > DEVELOPMENT_END or dates[0] < '2014-01-01':
            raise ValueError('Only ordered development sessions 2014-2022 are admitted')
        if len({a.action_id for a in self.actions}) != len(self.actions) or len({d.decision_id for d in self.decisions}) != len(self.decisions):
            raise ValueError('Repeated action or decision identity')
        ids = set(self.securities)
        referenced = set(self.initial_shares)
        for session in self.sessions:
            referenced.update(session.quotes)
        for decision in self.decisions:
            referenced.update(decision.target_shares)
            if decision.date not in dates or datetime.fromisoformat(decision.information_time) > datetime.fromisoformat(self.sessions[dates.index(decision.date)].close_utc):
                raise ValueError('Decision uses future information or an absent session')
        for action in self.actions:
            referenced.add(action.security_id)
            if action.recipient_id:
                referenced.add(action.recipient_id)
            if action.date not in dates:
                raise ValueError('Action is outside supplied sessions')
            if action.date == dates[0]:
                raise ValueError('Opening raw marks must precede economic action accounting')
            if action.payment_date is not None and dates[0] <= action.payment_date <= dates[-1] and action.payment_date not in dates:
                raise ValueError('Payment must be a supplied session, not an inferred weekend cash flow')
            if datetime.fromisoformat(action.known_at) > datetime.fromisoformat(self.sessions[dates.index(action.date)].close_utc):
                raise ValueError('Action terms became known after the accounting session')
        referenced.update(l.security_id for l in self.lending)
        if not referenced <= ids:
            raise ValueError('Unknown share-class identity')
        if len({(d.date, sid) for d in self.decisions for sid in d.target_shares}) != sum(len(d.target_shares) for d in self.decisions):
            raise ValueError('Ambiguous same-session targets')
        return self
