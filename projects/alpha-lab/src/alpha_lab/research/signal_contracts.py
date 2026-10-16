"""P2 signal studies: diagnostic scores and outcomes, never portfolio P&L."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .contracts import Artifact, DEVELOPMENT_END, Hash, Identifier, StrictModel, digest, iso_day

FEATURES = {
    'momentum_12_1': ('price', 'simple_return', 'prior_12_to_1_month_adjusted_prices'),
    'momentum_3_1': ('price', 'simple_return', 'prior_3_to_1_month_adjusted_prices'),
    'reversal_1m': ('price', 'negative_simple_return', 'prior_month_adjusted_prices'),
    'low_vol': ('price', 'negative_daily_return_std', '63_sessions_min50_adjusted_returns'),
    'book_yield': ('fundamental', 'ratio', 'book_equity_and_decision_market_cap'),
    'earnings_yield': ('fundamental', 'ratio', 'net_income_and_decision_market_cap'),
    'fcf_yield': ('fundamental', 'ratio', 'free_cash_flow_and_decision_market_cap'),
    'roa': ('fundamental', 'ratio', 'net_income_and_assets'),
    'gross_profitability': ('fundamental', 'ratio', 'gross_profit_and_assets'),
    'leverage': ('fundamental', 'ratio', 'debt_and_assets'),
}


class Recipe(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=80)]
    operator: Literal['linear', 'quality_scaled_momentum']
    factors: dict[str, float]
    transform: Literal['cross_sectional_rank', 'train_zscore']

    @model_validator(mode='after')
    def supported(self):
        if not self.factors or set(self.factors) - FEATURES.keys() or not any(self.factors.values()):
            raise ValueError('Require supported nonzero feature coefficients')
        if self.operator == 'quality_scaled_momentum' and (
                self.transform != 'cross_sectional_rank' or self.factors != {
                    'momentum_12_1': 1., 'roa': 1., 'gross_profitability': 1., 'leverage': -1.}):
            raise ValueError('The quality multiplier has fixed inputs and centered ranks')
        return self

    def computation(self):
        # Decimal normalization makes renamed and positively rescaled linear
        # recipes identical. The rounded coefficients are ALSO used to compute.
        factors = {k: Decimal(str(v)) for k, v in self.factors.items() if v}
        if self.operator == 'linear':
            scale = sum(abs(v) for v in factors.values())
            factors = {k: (v / scale).quantize(Decimal('0.000000000001'), rounding=ROUND_HALF_EVEN)
                       for k, v in factors.items()}
        return {'operator': self.operator, 'transform': self.transform,
                'factors': {k: float(v) for k, v in sorted(factors.items()) if v}}


class Label(StrictModel):
    target: Literal['total_return', 'simple_excess_spy']
    horizon_sessions: Annotated[int, Field(ge=1, le=63)]
    execution_lag_sessions: Annotated[int, Field(ge=1, le=20)]
    availability_lag_sessions: Annotated[int, Field(ge=0, le=20)]
    basis: Literal['vendor_adjusted_close_to_close_no_gap_fill']


class Fold(StrictModel):
    name: Identifier
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    embargo_sessions: Annotated[int, Field(ge=0, le=63)]

    _dates = field_validator('train_start', 'train_end', 'validation_start', 'validation_end')(iso_day)

    @model_validator(mode='after')
    def order(self):
        if not '2014-01-01' <= self.train_start <= self.train_end < self.validation_start <= self.validation_end <= DEVELOPMENT_END:
            raise ValueError('Folds must be ordered development dates through 2022')
        return self


class SignalSpec(StrictModel):
    schema_version: Literal[2]
    family_id: Identifier
    hypothesis_id: Identifier
    bundle_id: Hash
    recipe: Recipe
    label: Label
    folds: Annotated[list[Fold], Field(min_length=1, max_length=12)]
    quantiles: Annotated[int, Field(ge=2, le=10)]
    min_cross_section: Annotated[int, Field(ge=2, le=10000)]
    feature_policy: Literal['independent_dependencies_filing_plus2_max_age550_v1']
    coverage_policy: Literal['legacy_source_membership_no_future_outcome_filter_v1']
    numerical_policy: Literal['cpu_float64_fixed_threads']
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    threads: Annotated[int, Field(ge=1, le=8)]
    timeout_seconds: Annotated[int, Field(ge=1, le=600)]
    max_output_bytes: Annotated[int, Field(ge=1024, le=100_000_000)]
    memory_limit_mb: Annotated[int, Field(ge=128, le=8192)]
    implementation: dict[str, Hash]
    environment: dict[str, str]

    @model_validator(mode='after')
    def separated(self):
        ordered = sorted(self.folds, key=lambda f: f.validation_start)
        if len({f.name for f in ordered}) != len(ordered) or any(
                a.validation_end >= b.validation_start for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Validation folds may not overlap or repeat names')
        return self

    def comparison(self):
        return self.model_dump(exclude={'family_id', 'hypothesis_id', 'recipe', 'timeout_seconds',
                                        'max_output_bytes', 'memory_limit_mb'})

    def comparison_id(self):
        return digest(self.comparison())

    def computation(self):
        return self.comparison() | {'recipe': self.recipe.computation()}

    def semantic_id(self):
        return digest(self.computation())


class Family(StrictModel):
    family_id: Identifier
    premise: Annotated[str, Field(min_length=10, max_length=2000)]
    max_unique_trials: Annotated[int, Field(ge=1, le=100)]
    max_attempts_per_trial: Annotated[int, Field(ge=1, le=5)]


class Hypothesis(StrictModel):
    hypothesis_id: Identifier
    family_id: Identifier
    premise: Annotated[str, Field(min_length=10, max_length=2000)]
    tests: Annotated[list[str], Field(min_length=1, max_length=20)]


class PriorExposure(StrictModel):
    exposure_id: Identifier
    start: str
    end: str
    evidence_sha256: Hash
    description: Annotated[str, Field(min_length=10, max_length=500)]
    kind: Literal['previously_observed_legacy_evaluation', 'previously_observed_development_diagnostics']

    _dates = field_validator('start', 'end')(iso_day)

    @model_validator(mode='after')
    def ordered(self):
        if self.start > self.end:
            raise ValueError('Exposure period is reversed')
        return self


class SignalRun(StrictModel):
    schema_version: Literal[2]
    attempt_id: Identifier
    trial_id: Hash
    spec_sha256: Hash
    bundle_id: Hash
    hypothesis_id: Identifier
    status: Literal['running', 'failed', 'completed']
    stage: Literal['validating', 'evaluating', 'publishing', 'finished']
    started_at: str
    updated_at: str
    finished_at: str | None
    failure: Literal['input_validation', 'execution_failure', 'timeout', 'output_limit', 'memory_limit',
                     'interrupted', 'publication_failure'] | None
    peak_rss_mb: Annotated[float, Field(ge=0)]
    worker_pid: int | None
    retry_of: Identifier | None
    artifacts: dict[str, Artifact]
    llm_calls: Literal[0]
    scientific_disposition: Literal['provisional_development_signal_diagnostics_not_portfolio_pnl']

    @model_validator(mode='after')
    def terminal(self):
        from datetime import datetime
        stamps = [datetime.fromisoformat(s) for s in (self.started_at, self.updated_at, self.finished_at) if s]
        if any(s.tzinfo is None or s.utcoffset() is None for s in stamps) or stamps != sorted(stamps):
            raise ValueError('Invalid run timestamps')
        if (self.status == 'running') != (self.finished_at is None):
            raise ValueError('Inconsistent terminal timestamp')
        if self.status != 'running' and self.stage != 'finished':
            raise ValueError('Terminal stage must be finished')
        if (self.status == 'failed') != (self.failure is not None):
            raise ValueError('Inconsistent failure reason')
        if self.status == 'completed' and not self.artifacts:
            raise ValueError('Completion requires artifact evidence')
        return self


def baseline_recipes():
    return [
        {'name': 'value', 'operator': 'linear', 'factors': {'book_yield': 1., 'earnings_yield': 1.},
         'transform': 'cross_sectional_rank'},
        {'name': 'quality', 'operator': 'linear', 'factors': {'roa': 1., 'gross_profitability': 1., 'leverage': -1.},
         'transform': 'cross_sectional_rank'},
        {'name': 'momentum', 'operator': 'linear', 'factors': {'momentum_12_1': 1.},
         'transform': 'cross_sectional_rank'},
        {'name': 'quality_scaled_momentum', 'operator': 'quality_scaled_momentum',
         'factors': {'momentum_12_1': 1., 'roa': 1., 'gross_profitability': 1., 'leverage': -1.},
         'transform': 'cross_sectional_rank'},
    ]
