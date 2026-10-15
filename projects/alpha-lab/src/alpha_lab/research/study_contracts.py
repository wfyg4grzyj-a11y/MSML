"""An integrated development study ties signals, constraints and raw accounting."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from .contracts import Artifact, Hash, Identifier, StrictModel, digest, iso_day
from .ledger_contracts import LedgerScenario, utc_stamp
from .portfolio import PortfolioPolicy
from .signal_contracts import SignalRun


class SignalPoint(StrictModel):
    security_id: Identifier
    score: float | None
    eligible: bool
    issuer_id: Identifier | None
    sector_id: Identifier | None
    beta: float | None


class PortfolioContext(StrictModel):
    context_id: Identifier
    date: str
    information_time: str
    signal_evidence_sha256: Hash
    source_policy_id: Identifier
    risk_binding_id: Identifier | None
    fill_lag_sessions: Annotated[int, Field(ge=1, le=20)]
    points: Annotated[list[SignalPoint], Field(min_length=2, max_length=5000)]
    intent: Literal['close_all'] | None = None

    @model_serializer(mode='wrap')
    def archived_shape(self, handler):
        value=handler(self)
        if self.intent is None:
            value.pop('intent',None)
        return value

    @model_validator(mode='after')
    def close_intent(self):
        if self.intent == 'close_all' and any(p.eligible or p.score is not None for p in self.points):
            raise ValueError('An explicit close has no eligible alpha signals')
        return self

    _date = field_validator('date')(iso_day)
    _time = field_validator('information_time')(utc_stamp)


class DailyRiskControls(StrictModel):
    methodology: Literal['close_breach_uniform_reduction_v1']
    fill_lag_sessions: Literal[1]
    max_gross: Annotated[float, Field(gt=0, le=10)]
    max_name: Annotated[float, Field(gt=0, le=1)]
    net_tolerance: Annotated[float, Field(ge=0, le=10)]
    beta_tolerance: Annotated[float, Field(ge=0, le=10)]
    exit_ineligible_holdings: bool


class StudyPacket(StrictModel):
    schema_version: Literal[3]
    study_id: Identifier
    hypothesis: Annotated[str, Field(min_length=10, max_length=2000)]
    scenario: LedgerScenario
    portfolio_policy: PortfolioPolicy | None
    risk_mode: Literal['none', 'diagonal_reference', 'equity_risk_forecast']
    variance_per_session: Annotated[float, Field(gt=0)] | None
    risk_contract_sha256: Hash | None
    risk_source_policy_id: Identifier | None
    benchmark_id: Identifier
    cash_id: Identifier
    calendar_id: Literal['XNYS_modeled_sessions']
    identity_namespace: Literal['sharadar_permaticker_share_class', 'synthetic_share_class']
    max_risk_age_calendar_days: Annotated[int, Field(ge=0, le=30)]
    contexts: list[PortfolioContext]
    daily_risk_controls: DailyRiskControls | None = None

    @model_serializer(mode='wrap')
    def archived_shape(self, handler):
        value = handler(self)
        if self.daily_risk_controls is None:
            value.pop('daily_risk_controls', None)
        return value

    @model_validator(mode='after')
    def consistent(self):
        if self.daily_risk_controls is not None:
            if self.portfolio_policy is None or self.risk_mode == 'none':
                raise ValueError('Daily controls require an explicit portfolio risk policy')
            for field in ('max_gross','max_name','net_tolerance','beta_tolerance'):
                if getattr(self.daily_risk_controls, field) > getattr(self.portfolio_policy, field):
                    raise ValueError('Daily controls cannot weaken hard portfolio limits')
        if self.risk_mode == 'none':
            if self.portfolio_policy is not None or self.contexts or self.variance_per_session is not None:
                raise ValueError('Ledger-only studies have no hidden portfolio policy')
        elif self.portfolio_policy is None or not self.contexts or self.scenario.decisions:
            raise ValueError('Integrated studies need contexts and one authoritative portfolio decision path')
        if self.risk_mode == 'diagonal_reference' and (self.variance_per_session is None or self.portfolio_policy.risk_horizon_sessions != 1):
            raise ValueError('The explicit diagonal baseline supports one session only')
        if self.risk_mode == 'equity_risk_forecast' and (self.risk_contract_sha256 is None or self.risk_source_policy_id is None or self.variance_per_session is not None):
            raise ValueError('Real/synthetic forecast use requires explicit binding conventions')
        dates=[c.date for c in self.contexts]
        if dates != sorted(set(dates)):
            raise ValueError('One ordered portfolio context per decision session')
        sessions={s.date:s for s in self.scenario.sessions}
        for context in self.contexts:
            ids=[p.security_id for p in context.points]
            if (len(set(ids)) != len(ids) or not set(ids)<=self.scenario.securities.keys() or context.date not in sessions
                    or datetime.fromisoformat(context.information_time)>datetime.fromisoformat(sessions[context.date].close_utc)):
                raise ValueError('Portfolio context identity/date/information time mismatch')
            if (self.risk_mode=='equity_risk_forecast') != (context.risk_binding_id is not None):
                raise ValueError('Context risk mode does not match its binding reference')
        return self


class StudyBundle(StrictModel):
    schema_version: Literal[3]
    bundle_id: Hash
    start: str
    end: str
    packet_sha256: Hash
    risk_ids: list[Identifier]
    files: dict[str, Artifact]

    def identity(self):
        return digest(self.model_dump(exclude={'bundle_id'}))


class StudySpec(StrictModel):
    schema_version: Literal[3]
    family_id: Identifier
    hypothesis_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=120)]
    bundle_id: Hash
    methodology: Literal['raw_share_cash_action_ledger_v1']
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    threads: Annotated[int, Field(ge=1, le=8)]
    timeout_seconds: Annotated[int, Field(ge=1, le=600)]
    max_output_bytes: Annotated[int, Field(ge=1024, le=100_000_000)]
    memory_limit_mb: Annotated[int, Field(ge=128, le=8192)]
    implementation: dict[str, Hash]
    environment: dict[str, str]

    def computation(self):
        return self.model_dump(exclude={'family_id','hypothesis_id','title','timeout_seconds','max_output_bytes','memory_limit_mb'})

    def semantic_id(self):
        return digest(self.computation())

    def comparison_id(self):
        return digest(self.computation())  # No pooling across economic bundles.


class StudyRun(SignalRun):
    schema_version: Literal[3]
    scientific_disposition: Literal['provisional_raw_share_accounting_research_not_accepted']
