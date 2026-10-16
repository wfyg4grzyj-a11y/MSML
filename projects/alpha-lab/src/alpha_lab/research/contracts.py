"""Strict P1 contracts. These describe legacy research, not verified economics."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEVELOPMENT_END = '2022-12-31'
Hash = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')]
PANEL_FILES = {'sources.json', 'features.parquet', 'actions.parquet', 'returns.npz', 'calendar.csv'}
BUNDLE_FILES = PANEL_FILES | {'panel.json', 'identities.json'}
DAILY_COLUMNS = ['date', 'gross', 'net', 'stress', 'turnover', 'missing_exposure',
                 'net_exposure', 'gross_exposure']


class ResearchError(RuntimeError):
    """Sanitized local workflow failure; no provider payloads or credentials."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    except (OSError, ValueError, UnicodeError):
        raise ResearchError('Cannot read valid finite JSON') from None


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True, allow_inf_nan=False)


def parse(model, value):
    try:
        return model.model_validate(value)
    except ValidationError as error:
        fields = sorted({'.'.join(map(str, e['loc'])) or 'root' for e in error.errors(include_input=False)})
        raise ResearchError(f'Invalid {model.__name__} fields: ' + ', '.join(fields)) from None


def iso_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Expected an ISO date')
    return value


class Artifact(StrictModel):
    sha256: Hash
    bytes: Annotated[int, Field(ge=0)]


class SourceRef(StrictModel):
    version: Identifier
    sha256: Hash


class Identity(StrictModel):
    ticker: Annotated[str, Field(pattern=r'^[A-Z0-9][A-Z0-9.\^-]{0,31}$')]
    security_id: Annotated[int, Field(gt=0)]


class IdentityMap(StrictModel):
    namespace: Literal['sharadar_permaticker_share_class']
    reference_sha256: Hash
    historical_eligibility_attested: Literal[False]
    securities: list[Identity]

    @model_validator(mode='after')
    def unique(self):
        if not self.securities or any(len({getattr(s, key) for s in self.securities}) != len(self.securities)
                                      for key in ('ticker', 'security_id')):
            raise ValueError('Absent or ambiguous security identities')
        return self


class DataBundle(StrictModel):
    schema_version: Literal[1]
    bundle_id: Hash
    start: str
    end: str
    source_panel_sha256: Hash
    sources: dict[str, SourceRef]
    files: dict[str, Artifact]
    history_mode: Literal['current_vendor_snapshot_reconstruction']
    availability_policy: Literal['legacy_art_filing_plus_2_calendar_days_unattested']
    universe_policy: Literal['legacy_panel_membership_static_metadata_unattested']
    return_basis: Literal['vendor_adjusted_simple_total_returns_unattested']
    identity_namespace: Literal['sharadar_permaticker_share_class']
    calendar_id: Literal['XNYS_modeled_sessions']
    calendar_sha256: Hash
    rows: Annotated[int, Field(gt=0)]
    sessions: Annotated[int, Field(gt=1)]
    securities: Annotated[int, Field(ge=30)]
    point_in_time_attested: Literal[False]
    terminal_actions_reconciled: Literal[False]

    _dates = field_validator('start', 'end')(iso_day)

    @model_validator(mode='after')
    def bounded(self):
        if self.start != '2014-01-01' or not self.start <= self.end <= DEVELOPMENT_END:
            raise ValueError('P1 is restricted to legacy development dates through 2022')
        if set(self.files) != BUNDLE_FILES or set(self.sources) != {'stocks', 'funds', 'daily', 'fundamentals', 'tickers', 'actions'}:
            raise ValueError('Incomplete bundle files/sources')
        if self.calendar_sha256 != self.files['calendar.csv'].sha256:
            raise ValueError('Calendar identity mismatch')
        body = self.model_dump(exclude={'bundle_id'})
        if self.bundle_id != digest(body):
            raise ValueError('Bundle semantic identity mismatch')
        return self


class Strategy(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=80)]
    factors: dict[str, float]

    @model_validator(mode='after')
    def supported(self):
        from alpha_lab.market_neutral import validate_spec
        validate_spec(self.model_dump())
        return self


class ResearchSpec(StrictModel):
    schema_version: Literal[1]
    hypothesis_id: Identifier
    family_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=120)]
    bundle_id: Hash
    strategy: Strategy
    methodology: Literal['legacy_market_neutral_v1']
    universe_policy: Literal['legacy_panel_membership_static_metadata_unattested']
    feature_policy: Literal['legacy_percentile_ranks_requires_filings']
    fold_policy: Literal['train_2014_2019_validation_2020_2022']
    timing_policy: Literal['month_end_signal_next_close_daily_target_maintenance']
    risk_policy: Literal['intercept_spy_beta_projection_gross1_namecap2pct']
    cash_policy: Literal['no_cash_interest_or_financing_zero_rf_sharpe']
    cost_bps: Annotated[float, Field(ge=0, le=10000)]
    borrow_bps_annual: Annotated[float, Field(ge=0, le=100000)]
    borrow_day_count: Literal['252_sessions_legacy_assumption']
    terminal_stress: Annotated[float, Field(ge=0, le=1)]
    numerical_policy: Literal['cpu_float64_fixed_threads']
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    threads: Annotated[int, Field(ge=1, le=8)]
    timeout_seconds: Annotated[int, Field(ge=1, le=600)]
    max_output_bytes: Annotated[int, Field(ge=1024, le=100_000_000)]
    memory_limit_mb: Annotated[int, Field(ge=128, le=8192)]
    implementation: dict[str, Hash]
    environment: dict[str, str]

    def semantic_id(self):
        body = self.model_dump(exclude={'title', 'hypothesis_id', 'family_id'})
        body['strategy'].pop('name')
        return digest(body)


class RunRecord(StrictModel):
    schema_version: Literal[1]
    run_id: Identifier
    spec_id: Hash
    bundle_id: Hash
    status: Literal['running', 'failed', 'stopped', 'completed']
    stage: Literal['validating', 'evaluating', 'publishing', 'finished']
    scientific_disposition: Literal['provisional_legacy_research_not_accepted']
    started_at: str
    updated_at: str
    finished_at: str | None
    failure_code: Literal['input_validation', 'execution_failure', 'timeout', 'interrupted', 'publication_failure', 'output_limit', 'memory_limit'] | None
    parent_attempt: Literal[None]
    llm_calls: Literal[0]
    artifacts: dict[str, Artifact]
    input_validation: Literal['pending', 'passed']
    worker_pid: int | None
    peak_rss_mb: Annotated[float, Field(ge=0)]
    isolation: Literal['credential_filtered_process_not_filesystem_or_network_sandbox']

    @model_validator(mode='after')
    def terminal(self):
        try:
            stamps = [datetime.fromisoformat(s) for s in (self.started_at, self.updated_at, self.finished_at) if s is not None]
            if any(s.tzinfo is None or s.utcoffset() is None for s in stamps) or stamps != sorted(stamps):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Invalid run timestamps') from None
        if (self.status == 'running') != (self.finished_at is None):
            raise ValueError('Inconsistent terminal timestamp')
        if self.status in {'failed', 'stopped'} and self.failure_code is None:
            raise ValueError('Missing failure reason')
        if self.status == 'completed' and (self.failure_code is not None or self.input_validation != 'passed' or not self.artifacts):
            raise ValueError('Incomplete completion evidence')
        if self.status != 'running' and self.stage != 'finished':
            raise ValueError('Terminal run must finish its stage')
        return self
