"""Action-aware replay of frozen factor/model tapes.

This module deliberately does not construct signals, models, expected returns or
risk estimates.  It replays immutable ``month-*.npz`` tapes from a completed
factor study through a raw-share corporate-action bridge.  All three terminal
action assumptions are run together so that the nonstandard cases are reported
as sensitivities rather than silently becoming part of the base result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import json
import math

import duckdb
import numpy as np
import pandas as pd

from .contracts import ResearchError, file_hash, read_json
from .factor_construction import annual_volatility
from .large_portfolio import optimize


SCENARIO_IDS = (
    "strict_standard_only",
    "imputed_zero_insolvency_last_mark_other",
    "imputed_last_mark_all_unresolved",
)

SCENARIO_METADATA = {
    "strict_standard_only": {
        "label": "standard_actions_only_unknown_nonstandard_censored",
        "imputed": False,
        "nonstandard_policy": "retain the unresolved parent claim and censor observed NAV",
    },
    "imputed_zero_insolvency_last_mark_other": {
        "label": "imputed_sensitivity_zero_insolvency_last_mark_other",
        "imputed": True,
        "nonstandard_policy": "zero insolvency claims; settle every other nonstandard claim at its last raw mark",
    },
    "imputed_last_mark_all_unresolved": {
        "label": "imputed_sensitivity_last_mark_all_unresolved",
        "imputed": True,
        "nonstandard_policy": "settle every nonstandard claim at its last raw mark",
    },
}

STANDARD_CLASSIFICATIONS = frozenset({"cash", "stock", "mixed"})
EPS = 1e-12


def scenario_metadata(scenario_id: str) -> dict[str, Any]:
    """Return an independent copy of a preregistered scenario description."""
    if scenario_id not in SCENARIO_METADATA:
        raise ResearchError("Unknown replay scenario")
    return dict(SCENARIO_METADATA[scenario_id])


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(_json_value(value), target, indent=2, allow_nan=False)


@dataclass(frozen=True)
class SparseTargets:
    """One next-close order represented only by names whose targets changed."""

    decision_index: int
    fill_index: int
    kind: str
    targets: dict[str, float]

    @classmethod
    def changed(
        cls,
        *,
        decision_index: int,
        fill_index: int,
        kind: str,
        current: dict[str, float],
        targets: dict[str, float],
        tolerance: float = EPS,
    ) -> "SparseTargets":
        sparse: dict[str, float] = {}
        for security_id, target in targets.items():
            target_value = _finite(target)
            if target_value is None:
                raise ResearchError("Sparse target is nonfinite")
            if abs(target_value - float(current.get(str(security_id), 0.0))) > tolerance:
                sparse[str(security_id)] = target_value
        return cls(int(decision_index), int(fill_index), str(kind), sparse)


@dataclass
class _ActionFunctions:
    analytical_to_raw: Callable[[float, float, float], float]
    resolve_package: Callable[[dict[str, Any], float, float, float], dict[str, Any]]
    claim_to_analytical: Callable[[float, float, float], float]


def _analytical_to_raw(units: float, adjusted_price: float, raw_price: float) -> float:
    if not all(math.isfinite(x) for x in (units, adjusted_price, raw_price)) or adjusted_price <= 0 or raw_price <= 0:
        raise ResearchError("Cannot bridge analytical units to raw shares")
    return float(units * adjusted_price / raw_price)


def _claim_to_analytical(raw_shares: float, adjusted_price: float, raw_price: float) -> float:
    if not all(math.isfinite(x) for x in (raw_shares, adjusted_price, raw_price)) or adjusted_price <= 0 or raw_price <= 0:
        raise ResearchError("Cannot bridge raw recipient shares to analytical units")
    return float(raw_shares * raw_price / adjusted_price)


def _resolve_package(
    package: dict[str, Any], parent_units: float, parent_adjusted: float, parent_raw: float
) -> dict[str, Any]:
    raw_parent = _analytical_to_raw(parent_units, parent_adjusted, parent_raw)
    classification = str(package.get("classification", "unresolved"))
    if classification not in STANDARD_CLASSIFICATIONS:
        return {
            "resolved": False,
            "parent_retained": True,
            "cash_usd": 0.0,
            "stock_claims": [],
            "raw_parent_shares": raw_parent,
            "package_id": str(package.get("package_id", "")),
        }
    cash_rate = _finite(package.get("cash_usd_per_raw_share")) or 0.0
    claims = []
    for leg in package.get("stock_legs", []):
        ratio = _finite(leg.get("ratio"))
        if ratio is None or ratio < 0 or not leg.get("recipient_id"):
            raise ResearchError("Invalid normalized stock-action leg")
        claims.append(
            {
                "recipient_id": str(leg["recipient_id"]),
                "recipient_ticker": str(leg.get("recipient_ticker", "")),
                "raw_shares": float(raw_parent * ratio),
            }
        )
    return {
        "resolved": True,
        "parent_retained": False,
        "cash_usd": float(raw_parent * cash_rate),
        "stock_claims": claims,
        "raw_parent_shares": raw_parent,
        "package_id": str(package.get("package_id", "")),
    }


def _resolve_raw_terms(package: dict[str, Any], raw_parent_shares: float) -> tuple[float, list[dict[str, Any]]]:
    """Apply normalized standard terms to an already-raw transitive claim."""
    if str(package.get("classification", "")) not in STANDARD_CLASSIFICATIONS or not math.isfinite(raw_parent_shares):
        raise ResearchError("Cannot apply nonstandard terms to a raw claim")
    cash_rate = _finite(package.get("cash_usd_per_raw_share")) or 0.0
    claims: list[dict[str, Any]] = []
    for leg in package.get("stock_legs", []):
        ratio = _finite(leg.get("ratio"))
        recipient_id = str(leg.get("recipient_id", ""))
        if ratio is None or ratio < 0 or not recipient_id:
            raise ResearchError("Invalid normalized raw-claim stock leg")
        claims.append(
            {
                "recipient_id": recipient_id,
                "recipient_ticker": str(leg.get("recipient_ticker", "")),
                "raw_shares": float(raw_parent_shares * ratio),
            }
        )
    return float(raw_parent_shares * cash_rate), claims


DEFAULT_ACTION_FUNCTIONS = _ActionFunctions(
    analytical_to_raw=_analytical_to_raw,
    resolve_package=_resolve_package,
    claim_to_analytical=_claim_to_analytical,
)


class QuoteBook:
    """Read-only base matrices plus normalized quotes for outside-axis recipients."""

    def __init__(
        self,
        dates: Iterable[str],
        security_ids: Iterable[str],
        adjusted: np.ndarray,
        raw: np.ndarray,
        dollar_volume: np.ndarray | None,
        supplemental: pd.DataFrame | None = None,
    ):
        self.dates = tuple(str(x) for x in dates)
        self.date_index = {day: i for i, day in enumerate(self.dates)}
        self.security_ids = tuple(str(x) for x in security_ids)
        self.base_index = {security_id: i for i, security_id in enumerate(self.security_ids)}
        if len(self.base_index) != len(self.security_ids):
            raise ResearchError("Repeated quote-book security identifier")
        self.adjusted = adjusted
        self.raw = raw
        self.dollar_volume = dollar_volume
        expected = (len(self.dates), len(self.security_ids))
        if adjusted.shape != expected or raw.shape != expected or (dollar_volume is not None and dollar_volume.shape != expected):
            raise ResearchError("Quote-book array shape mismatch")
        self.supplemental: dict[tuple[int, str], tuple[float, float, float | None]] = {}
        if supplemental is not None and len(supplemental):
            required = {"date", "security_id", "raw_close_usd", "adjusted_close_usd"}
            if not required.issubset(supplemental.columns):
                raise ResearchError("Supplemental action quotes are incomplete")
            frame = supplemental.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
            frame["security_id"] = frame["security_id"].astype(str)
            if frame.duplicated(["date", "security_id"]).any():
                raise ResearchError("Repeated supplemental action quote")
            for row in frame.itertuples(index=False):
                index = self.date_index.get(str(row.date))
                if index is None:
                    continue
                raw_value = _finite(row.raw_close_usd)
                adjusted_value = _finite(row.adjusted_close_usd)
                dv_value = _finite(getattr(row, "dollar_volume_usd", None))
                if raw_value is None or adjusted_value is None or raw_value <= 0 or adjusted_value <= 0:
                    raise ResearchError("Invalid supplemental action quote")
                if dv_value is not None and dv_value < 0:
                    raise ResearchError("Invalid supplemental dollar volume")
                self.supplemental[(index, str(row.security_id))] = (adjusted_value, raw_value, dv_value)

    def quote(self, index: int, security_id: str) -> tuple[float | None, float | None, float | None]:
        security_id = str(security_id)
        column = self.base_index.get(security_id)
        if column is not None:
            adjusted = _finite(self.adjusted[index, column])
            raw = _finite(self.raw[index, column])
            dv = None if self.dollar_volume is None else _finite(self.dollar_volume[index, column])
            return (
                adjusted if adjusted is not None and adjusted > 0 else None,
                raw if raw is not None and raw > 0 else None,
                dv if dv is not None and dv >= 0 else None,
            )
        return self.supplemental.get((index, security_id), (None, None, None))


def _is_insolvency(package: dict[str, Any]) -> bool:
    reasons = package.get("unresolved_reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    action_types = package.get("source_action_types", [])
    if isinstance(action_types, str):
        action_types = [action_types]
    text = " ".join(
        [str(package.get("classification", "")), *map(str, reasons), *map(str, action_types)]
    ).lower()
    return any(token in text for token in ("bankrupt", "insolv", "liquidat"))


@dataclass
class ReplayState:
    scenario_id: str
    cash: float
    units: dict[str, float] = field(default_factory=dict)
    raw_claims: dict[str, float] = field(default_factory=dict)
    adjusted_marks: dict[str, float] = field(default_factory=dict)
    raw_marks: dict[str, float] = field(default_factory=dict)
    adjusted_mark_dates: dict[str, str] = field(default_factory=dict)
    raw_mark_dates: dict[str, str] = field(default_factory=dict)
    restricted_ids: set[str] = field(default_factory=set)
    unknown_terminals: set[str] = field(default_factory=set)
    exit_after: dict[str, int] = field(default_factory=dict)
    risk_exit_after: dict[str, int] = field(default_factory=dict)
    pending: SparseTargets | None = None
    processed_packages: set[str] = field(default_factory=set)
    daily: list[dict[str, Any]] = field(default_factory=list)
    journal: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    last_day: str | None = None
    last_observed_nav: float | None = None
    ever_censored: bool = False
    imputed_events: int = 0

    def __post_init__(self) -> None:
        if self.scenario_id not in SCENARIO_IDS or not math.isfinite(self.cash):
            raise ResearchError("Invalid replay state")
        self.units = {str(k): float(v) for k, v in self.units.items() if abs(float(v)) > EPS}

    def _held_ids(self) -> set[str]:
        return {security_id for security_id, units in self.units.items() if abs(units) > EPS}

    def update_marks(self, index: int, quotes: QuoteBook) -> set[str]:
        observed: set[str] = set()
        day = quotes.dates[index]
        for security_id in self._held_ids() | {x for x, value in self.raw_claims.items() if abs(value) > EPS}:
            adjusted, raw, _ = quotes.quote(index, security_id)
            if adjusted is not None:
                self.adjusted_marks[security_id] = adjusted
                self.adjusted_mark_dates[security_id] = day
            if raw is not None:
                self.raw_marks[security_id] = raw
                self.raw_mark_dates[security_id] = day
            if (security_id in self.units and adjusted is not None) or (security_id in self.raw_claims and raw is not None):
                observed.add(security_id)
        return observed

    def accrue(self, day: str, policy: dict[str, Any]) -> tuple[float, float]:
        if self.last_day is None:
            return 0.0, 0.0
        elapsed = (date.fromisoformat(day) - date.fromisoformat(self.last_day)).days
        if elapsed < 0:
            raise ResearchError("Replay dates are not increasing")
        short_notional = 0.0
        for security_id, units in self.units.items():
            if units < 0 and security_id in self.adjusted_marks:
                short_notional += -units * self.adjusted_marks[security_id]
        for security_id, raw_shares in self.raw_claims.items():
            if raw_shares < 0 and security_id in self.raw_marks:
                short_notional += -raw_shares * self.raw_marks[security_id]
        execution = policy["execution"]
        borrow = short_notional * float(execution["annual_hypothetical_borrow"]) * elapsed / 365.0
        funding = max(-self.cash, 0.0) * float(execution["cash_debit"]) * elapsed / 365.0
        self.cash -= borrow + funding
        return float(borrow), float(funding)

    def valuation(self, index: int, quotes: QuoteBook) -> tuple[float, float | None, int]:
        estimate = float(self.cash)
        missing = 0
        for security_id, units in self.units.items():
            if abs(units) <= EPS:
                continue
            adjusted, _, _ = quotes.quote(index, security_id)
            mark = self.adjusted_marks.get(security_id)
            if mark is not None:
                estimate += units * mark
            if adjusted is None:
                missing += 1
        for security_id, raw_shares in self.raw_claims.items():
            if abs(raw_shares) <= EPS:
                continue
            _, raw, _ = quotes.quote(index, security_id)
            mark = self.raw_marks.get(security_id)
            if mark is not None:
                estimate += raw_shares * mark
            if raw is None:
                missing += 1
        unresolved = any(
            abs(self.units.get(security_id, 0.0)) > EPS or abs(self.raw_claims.get(security_id, 0.0)) > EPS
            for security_id in self.unknown_terminals
        )
        observed = None if missing or unresolved else float(estimate)
        if observed is None:
            self.ever_censored = True
        return float(estimate), observed, int(missing + unresolved)

    def schedule(self, order: SparseTargets) -> None:
        self.pending = order if order.targets else None

    def schedule_exit(self, security_id: str, first_index: int) -> None:
        security_id = str(security_id)
        self.restricted_ids.add(security_id)
        prior = self.exit_after.get(security_id)
        self.exit_after[security_id] = int(first_index if prior is None else min(prior, first_index))
        self.risk_exit_after.pop(security_id, None)

    def schedule_risk_exit(self, security_id: str, first_index: int) -> None:
        """Queue a temporary reduce-only exit without permanently restricting alpha."""
        security_id = str(security_id)
        if (
            security_id in self.exit_after
            or security_id in self.restricted_ids
            or security_id in self.unknown_terminals
        ):
            return
        prior = self.risk_exit_after.get(security_id)
        self.risk_exit_after[security_id] = int(first_index if prior is None else min(prior, first_index))

    def register_terminal_parents(self, packages: Iterable[dict[str, Any]]) -> None:
        """Give current-session terminal events priority over queued trades.

        Action packages become effective at the current close.  Their parents
        must therefore be removed from every pre-existing execution queue before
        that close is processed, and remain unavailable to all later alpha
        optimizations even when the frozen universe or price tape lags the event.
        """
        for package in packages:
            parent_id = str(package.get("parent_security_id", ""))
            if not parent_id:
                raise ResearchError("Terminal action package has no parent identity")
            self.restricted_ids.add(parent_id)
            self.exit_after.pop(parent_id, None)
            self.risk_exit_after.pop(parent_id, None)

    def _preview_delta(
        self,
        *,
        index: int,
        security_id: str,
        requested: float,
        quotes: QuoteBook,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        adjusted, _, dollar_volume = quotes.quote(index, security_id)
        if adjusted is None or dollar_volume is None or dollar_volume <= 0 or abs(requested) <= EPS:
            filled = 0.0
        else:
            cap_units = dollar_volume * float(policy["execution"]["participation"]) / adjusted
            filled = float(math.copysign(min(abs(requested), cap_units), requested))
        commission_rate = float(policy["execution"]["one_way_commission_bps"]) / 10_000.0
        slippage_rate = float(policy["execution"]["one_way_slippage_bps"]) / 10_000.0
        if abs(filled) > EPS:
            execution_price = adjusted * (1.0 + math.copysign(slippage_rate, filled))
            commission = abs(filled) * execution_price * commission_rate
            trade_cost = abs(filled) * adjusted * slippage_rate + commission
        else:
            execution_price = 0.0
            commission = 0.0
            trade_cost = 0.0
        return {
            "security_id": security_id,
            "requested": float(requested),
            "filled": float(filled),
            "adjusted": adjusted,
            "dollar_volume": dollar_volume,
            "execution_price": float(execution_price),
            "commission": float(commission),
            "trade_cost": float(trade_cost),
            "executable": bool(adjusted is not None and dollar_volume is not None and dollar_volume > 0),
        }

    def _commit_preview(
        self,
        preview: dict[str, Any],
        *,
        index: int,
        day: str,
        kind: str,
        decision_index: int,
        batch_status: str,
        batch_violations: Iterable[str] = (),
        batch_turnover: float | None = None,
    ) -> float:
        security_id = str(preview["security_id"])
        filled = float(preview["filled"])
        if abs(filled) > EPS:
            self.cash -= filled * float(preview["execution_price"]) + float(preview["commission"])
            new_units = self.units.get(security_id, 0.0) + filled
            if abs(new_units) <= EPS:
                self.units.pop(security_id, None)
                self.unknown_terminals.discard(security_id)
            else:
                self.units[security_id] = float(new_units)
        self.trades.append(
            {
                "scenario_id": self.scenario_id,
                "session_index": int(index),
                "date": day,
                "security_id": security_id,
                "kind": kind,
                "decision_index": int(decision_index),
                "requested_delta": float(preview["requested"]),
                "filled_delta": float(filled),
                "trade_cost_usd": float(preview["trade_cost"]),
                "executable": bool(preview["executable"]),
                "batch_status": batch_status,
                "batch_violations": ",".join(sorted(set(map(str, batch_violations)))),
                "batch_turnover": batch_turnover,
            }
        )
        return filled

    def _fill_delta(
        self,
        *,
        index: int,
        day: str,
        security_id: str,
        requested: float,
        quotes: QuoteBook,
        policy: dict[str, Any],
        kind: str,
        decision_index: int,
        batch_status: str = "reduce_only_action_exit",
    ) -> float:
        preview = self._preview_delta(
            index=index,
            security_id=security_id,
            requested=requested,
            quotes=quotes,
            policy=policy,
        )
        return self._commit_preview(
            preview,
            index=index,
            day=day,
            kind=kind,
            decision_index=decision_index,
            batch_status=batch_status,
        )

    def execute_orders(
        self,
        index: int,
        day: str,
        quotes: QuoteBook,
        policy: dict[str, Any],
        context: _TapeContext | None = None,
    ) -> float:
        trade_cost = 0.0
        # Snapshot every risk exit that is due at this close before any queue is
        # mutated.  The ordinary batch must not trade one of these names again
        # after a partial fill, or re-enter it after a complete fill.
        due_risk_ids = {
            security_id
            for security_id, first_index in self.risk_exit_after.items()
            if index >= first_index
        }
        # Action recipients remain reduce-only.  Missing quote/volume leaves the
        # exit outstanding for a later observed session.
        for security_id in sorted(list(self.exit_after)):
            if security_id in self.unknown_terminals:
                self.exit_after.pop(security_id, None)
                continue
            if index < self.exit_after[security_id]:
                continue
            units = self.units.get(security_id, 0.0)
            if abs(units) <= EPS:
                self.exit_after.pop(security_id, None)
                continue
            filled = self._fill_delta(
                index=index,
                day=day,
                security_id=security_id,
                requested=-units,
                quotes=quotes,
                policy=policy,
                kind="action_recipient_exit",
                decision_index=self.exit_after[security_id] - 1,
            )
            trade_cost += float(self.trades[-1]["trade_cost_usd"])
            if abs(self.units.get(security_id, 0.0)) <= EPS:
                self.exit_after.pop(security_id, None)
            elif abs(filled) <= EPS:
                # Explicitly keep trying; there is no stale-price execution.
                self.exit_after[security_id] = index + 1
        # A risk-orphan exit is also independent of the ordinary batch.  It is
        # temporary, always reduce-only and runs after action-recipient exits.
        for security_id in sorted(list(self.risk_exit_after)):
            if security_id in self.exit_after or security_id in self.unknown_terminals:
                self.risk_exit_after.pop(security_id, None)
                continue
            if index < self.risk_exit_after[security_id]:
                continue
            units = self.units.get(security_id, 0.0)
            if abs(units) <= EPS:
                self.risk_exit_after.pop(security_id, None)
                continue
            filled = self._fill_delta(
                index=index,
                day=day,
                security_id=security_id,
                requested=-units,
                quotes=quotes,
                policy=policy,
                kind="unsupported_risk_exit",
                decision_index=self.risk_exit_after[security_id] - 1,
                batch_status="reduce_only_risk_orphan_exit",
            )
            trade_cost += float(self.trades[-1]["trade_cost_usd"])
            if abs(self.units.get(security_id, 0.0)) <= EPS:
                self.risk_exit_after.pop(security_id, None)
            elif abs(filled) <= EPS:
                self.risk_exit_after[security_id] = index + 1
        order = self.pending
        if order is not None and order.fill_index == index:
            previews: list[dict[str, Any]] = []
            for security_id, target in sorted(order.targets.items()):
                if (
                    security_id in self.exit_after
                    or security_id in self.restricted_ids
                    or security_id in self.unknown_terminals
                    or security_id in due_risk_ids
                ):
                    continue
                requested = target - self.units.get(security_id, 0.0)
                previews.append(self._preview_delta(
                    index=index,
                    security_id=security_id,
                    requested=requested,
                    quotes=quotes,
                    policy=policy,
                ))

            # Preserve the parent's batch rule: if any held quote is absent,
            # only component-wise reductions may proceed on this session.
            missing_held = any(
                abs(units) > EPS and quotes.quote(index, security_id)[0] is None
                for security_id, units in self.units.items()
            )
            if missing_held:
                for preview in previews:
                    old = self.units.get(str(preview["security_id"]), 0.0)
                    proposed = old + float(preview["filled"])
                    reducing = abs(proposed) <= abs(old) + 1e-10 and proposed * old >= -1e-12
                    if not reducing:
                        preview.update(filled=0.0, execution_price=0.0, commission=0.0, trade_cost=0.0)

            proposed_units = dict(self.units)
            trial_cash = float(self.cash)
            notional = 0.0
            only_reduce = True
            for preview in previews:
                security_id = str(preview["security_id"])
                old = proposed_units.get(security_id, 0.0)
                filled = float(preview["filled"])
                proposed = old + filled
                if abs(proposed) <= EPS:
                    proposed_units.pop(security_id, None)
                else:
                    proposed_units[security_id] = proposed
                trial_cash -= filled * float(preview["execution_price"]) + float(preview["commission"])
                if preview["adjusted"] is not None:
                    notional += abs(filled) * float(preview["adjusted"])
                only_reduce = only_reduce and abs(proposed) <= abs(old) + 1e-10 and proposed * old >= -1e-12

            trial = ReplayState(
                self.scenario_id,
                trial_cash,
                units=proposed_units,
                raw_claims=dict(self.raw_claims),
            )
            trial.adjusted_marks = dict(self.adjusted_marks)
            trial.raw_marks = dict(self.raw_marks)
            trial.unknown_terminals = set(self.unknown_terminals)
            for security_id in proposed_units:
                adjusted, raw, _ = quotes.quote(index, security_id)
                if adjusted is not None:
                    trial.adjusted_marks[security_id] = adjusted
                if raw is not None:
                    trial.raw_marks[security_id] = raw
            _, observed, _ = trial.valuation(index, quotes)
            metrics = _position_metrics(trial, context, observed)
            violations = _breaches(metrics, policy["hard_limits"])
            turnover = notional / observed if observed is not None and observed > 0 else float("inf")
            if turnover > float(policy["hard_limits"]["turnover"]) + float(policy["hard_limits"].get("tolerance", 1e-6)):
                violations.append("turnover")
            rejected = bool(violations and not only_reduce)
            batch_status = (
                "rejected_batch_constraints"
                if rejected
                else "filled_derisk_residual_violation"
                if violations
                else "filled"
            )
            for preview in previews:
                if rejected:
                    preview.update(filled=0.0, execution_price=0.0, commission=0.0, trade_cost=0.0)
                self._commit_preview(
                    preview,
                    index=index,
                    day=day,
                    kind=order.kind,
                    decision_index=order.decision_index,
                    batch_status=batch_status,
                    batch_violations=violations,
                    batch_turnover=turnover,
                )
                trade_cost += float(self.trades[-1]["trade_cost_usd"])
            self.pending = None
        return float(trade_cost)

    def _record_action(self, day: str, package: dict[str, Any], **extra: Any) -> None:
        self.journal.append(
            {
                "scenario_id": self.scenario_id,
                "date": day,
                "package_id": str(package.get("package_id", "")),
                "parent_security_id": str(package.get("parent_security_id", "")),
                "classification": str(package.get("classification", "unresolved")),
                **extra,
            }
        )

    def process_packages(
        self,
        *,
        index: int,
        day: str,
        packages: Iterable[dict[str, Any]],
        quotes: QuoteBook,
        actions: _ActionFunctions,
        finalize_not_held: bool = False,
    ) -> int:
        processed = 0
        for package in packages:
            package_id = str(package.get("package_id", ""))
            if not package_id or package_id in self.processed_packages:
                continue
            parent_id = str(package.get("parent_security_id", ""))
            # A terminal package makes its parent permanently unavailable to
            # alpha from this session onward, whether or not the account held
            # it when the event became effective.  Frozen universe/tape data
            # can lag the terminal event, so relying on later membership alone
            # could otherwise re-enter an acquired or delisted security.
            self.restricted_ids.add(parent_id)
            self.exit_after.pop(parent_id, None)
            self.risk_exit_after.pop(parent_id, None)
            parent_units = float(self.units.get(parent_id, 0.0))
            parent_raw_claim = float(self.raw_claims.get(parent_id, 0.0))
            classification = str(package.get("classification", "unresolved"))
            if abs(parent_units) <= EPS and abs(parent_raw_claim) <= EPS:
                if finalize_not_held:
                    self.processed_packages.add(package_id)
                    self._record_action(day, package, status="not_held", imputed=False)
                continue
            self.processed_packages.add(package_id)
            processed += 1
            adjusted = self.adjusted_marks.get(parent_id)
            raw = self.raw_marks.get(parent_id)
            if classification in STANDARD_CLASSIFICATIONS:
                if abs(parent_units) > EPS and (adjusted is None or raw is None or adjusted <= 0 or raw <= 0):
                    self.unknown_terminals.add(parent_id)
                    self._record_action(
                        day,
                        package,
                        status="standard_missing_parent_conversion_basis",
                        imputed=False,
                        parent_adjusted_basis_date=self.adjusted_mark_dates.get(parent_id),
                        parent_raw_basis_date=self.raw_mark_dates.get(parent_id),
                    )
                    continue
                cash = 0.0
                claims: list[dict[str, Any]] = []
                raw_parent = parent_raw_claim
                if abs(parent_units) > EPS:
                    resolution = actions.resolve_package(package, parent_units, adjusted, raw)
                    if not resolution.get("resolved"):
                        self.unknown_terminals.add(parent_id)
                        self._record_action(
                            day,
                            package,
                            status="standard_unresolved",
                            imputed=False,
                            parent_adjusted_basis_date=self.adjusted_mark_dates.get(parent_id),
                            parent_raw_basis_date=self.raw_mark_dates.get(parent_id),
                        )
                        continue
                    if resolution.get("parent_retained", False):
                        raise ResearchError("Resolved terminal package unexpectedly retained its parent")
                    resolved_cash = _finite(resolution.get("cash_usd"))
                    resolved_raw = _finite(resolution.get("raw_parent_shares"))
                    if resolved_cash is None or resolved_raw is None:
                        raise ResearchError("Action resolver returned a nonfinite standard bridge")
                    cash += resolved_cash
                    raw_parent += resolved_raw
                    claims.extend(resolution.get("stock_claims", []))
                if abs(parent_raw_claim) > EPS:
                    claim_cash, claim_legs = _resolve_raw_terms(package, parent_raw_claim)
                    cash += claim_cash
                    claims.extend(claim_legs)
                if abs(parent_units) > EPS:
                    self.units.pop(parent_id, None)
                self.raw_claims.pop(parent_id, None)
                self.exit_after.pop(parent_id, None)
                self.unknown_terminals.discard(parent_id)
                self.cash += cash
                claim_count = 0
                for claim in claims:
                    recipient_id = str(claim.get("recipient_id", ""))
                    raw_shares = _finite(claim.get("raw_shares"))
                    if not recipient_id or raw_shares is None:
                        raise ResearchError("Action resolver returned an invalid stock claim")
                    self.raw_claims[recipient_id] = self.raw_claims.get(recipient_id, 0.0) + raw_shares
                    self.restricted_ids.add(recipient_id)
                    claim_count += 1
                self._record_action(
                    day,
                    package,
                    status="standard_resolved",
                    imputed=False,
                    parent_units=parent_units,
                    parent_raw_claim=parent_raw_claim,
                    raw_parent_shares=raw_parent,
                    parent_adjusted_basis_date=self.adjusted_mark_dates.get(parent_id),
                    parent_raw_basis_date=self.raw_mark_dates.get(parent_id),
                    cash_usd=cash,
                    stock_claims=claim_count,
                )
                continue

            insolvency = _is_insolvency(package)
            if self.scenario_id == "strict_standard_only":
                self.unknown_terminals.add(parent_id)
                self._record_action(
                    day,
                    package,
                    status="nonstandard_censored",
                    imputed=False,
                    insolvency=insolvency,
                    parent_adjusted_basis_date=self.adjusted_mark_dates.get(parent_id),
                    parent_raw_basis_date=self.raw_mark_dates.get(parent_id),
                )
                continue
            if self.scenario_id == "imputed_zero_insolvency_last_mark_other" and insolvency:
                self.units.pop(parent_id, None)
                self.raw_claims.pop(parent_id, None)
                self.exit_after.pop(parent_id, None)
                self.unknown_terminals.discard(parent_id)
                self.imputed_events += 1
                self._record_action(
                    day,
                    package,
                    status="nonstandard_imputed_zero",
                    imputed=True,
                    imputation="zero_insolvency",
                    parent_units=parent_units,
                    parent_raw_claim=parent_raw_claim,
                )
                continue
            missing_unit_basis = abs(parent_units) > EPS and (
                adjusted is None or raw is None or adjusted <= 0 or raw <= 0
            )
            missing_claim_basis = abs(parent_raw_claim) > EPS and (raw is None or raw <= 0)
            if missing_unit_basis or missing_claim_basis:
                self.unknown_terminals.add(parent_id)
                self._record_action(
                    day,
                    package,
                    status="nonstandard_missing_last_mark",
                    imputed=True,
                    insolvency=insolvency,
                    parent_adjusted_basis_date=self.adjusted_mark_dates.get(parent_id),
                    parent_raw_basis_date=self.raw_mark_dates.get(parent_id),
                )
                continue
            raw_parent = parent_raw_claim
            if abs(parent_units) > EPS:
                raw_parent += actions.analytical_to_raw(parent_units, adjusted, raw)
            settlement = float(raw_parent * raw)
            self.cash += settlement
            self.units.pop(parent_id, None)
            self.raw_claims.pop(parent_id, None)
            self.exit_after.pop(parent_id, None)
            self.unknown_terminals.discard(parent_id)
            self.imputed_events += 1
            self._record_action(
                day,
                package,
                status="nonstandard_imputed_last_mark",
                imputed=True,
                imputation="last_raw_mark",
                insolvency=insolvency,
                parent_units=parent_units,
                parent_raw_claim=parent_raw_claim,
                raw_parent_shares=raw_parent,
                parent_adjusted_basis_date=self.adjusted_mark_dates.get(parent_id),
                parent_raw_basis_date=self.raw_mark_dates.get(parent_id),
                settlement_usd=settlement,
            )
        return processed

    def deliver_claims(
        self, *, index: int, day: str, quotes: QuoteBook, actions: _ActionFunctions
    ) -> int:
        delivered = 0
        for security_id in sorted(list(self.raw_claims)):
            raw_shares = self.raw_claims[security_id]
            if abs(raw_shares) <= EPS:
                self.raw_claims.pop(security_id, None)
                continue
            adjusted, raw, _ = quotes.quote(index, security_id)
            if adjusted is None or raw is None:
                continue
            analytical = actions.claim_to_analytical(raw_shares, adjusted, raw)
            self.units[security_id] = self.units.get(security_id, 0.0) + analytical
            self.raw_claims.pop(security_id, None)
            self.schedule_exit(security_id, index + 1)
            delivered += 1
            self.journal.append(
                {
                    "scenario_id": self.scenario_id,
                    "date": day,
                    "package_id": None,
                    "parent_security_id": None,
                    "classification": "stock_claim_delivery",
                    "status": "recipient_claim_converted",
                    "recipient_security_id": security_id,
                    "raw_shares": float(raw_shares),
                    "analytical_units": float(analytical),
                    "exit_first_index": int(index + 1),
                    "imputed": False,
                }
            )
        return delivered


def _process_action_day(
    state: ReplayState,
    *,
    index: int,
    day: str,
    packages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    quotes: QuoteBook,
    actions: _ActionFunctions,
) -> None:
    """Resolve same-day recipient chains before declaring packages not held."""
    # One package transition per package, one aggregate delayed-claim
    # delivery pass, and one final no-progress pass proving quiescence.
    for _ in range(len(packages) + 2):
        processed = state.process_packages(
            index=index,
            day=day,
            packages=packages,
            quotes=quotes,
            actions=actions,
            finalize_not_held=False,
        )
        delivered = state.deliver_claims(index=index, day=day, quotes=quotes, actions=actions)
        if processed == 0 and delivered == 0:
            break
    else:
        raise ResearchError("Same-day action conversion chain did not terminate")
    state.process_packages(
        index=index,
        day=day,
        packages=packages,
        quotes=quotes,
        actions=actions,
        finalize_not_held=True,
    )


@dataclass(frozen=True)
class _TapeContext:
    date: str
    ids: tuple[str, ...]
    lookup: dict[str, int]
    industry: np.ndarray
    beta: np.ndarray
    supported: np.ndarray
    b: np.ndarray
    d: np.ndarray


def _position_metrics(state: ReplayState, context: _TapeContext | None, nav: float | None) -> dict[str, Any] | None:
    if context is None or nav is None or nav <= 0 or state.raw_claims:
        return None
    held = [security_id for security_id, units in state.units.items() if abs(units) > EPS]
    if any(
        security_id not in context.lookup
        or security_id not in state.adjusted_marks
        or not context.supported[context.lookup[security_id]]
        for security_id in held
    ):
        return None
    weights = np.zeros(len(context.ids))
    for security_id in held:
        j = context.lookup[security_id]
        weights[j] = state.units[security_id] * state.adjusted_marks[security_id] / nav
    gross = float(np.abs(weights).sum())
    result: dict[str, Any] = {
        "gross": gross,
        "name": float(np.abs(weights).max(initial=0.0)),
        "net": float(weights.sum()),
        "beta": float(weights @ context.beta),
        "industries": {},
        "annual_ex_ante_vol": float(annual_volatility(context.b, context.d, weights)),
    }
    for group in sorted(set(context.industry.astype(str))):
        mask = context.industry == group
        result["industries"][str(group)] = {
            "net": float(weights[mask].sum()),
            "beta": float(weights[mask] @ context.beta[mask]),
        }
    return result


def _unsupported_risk_holdings(state: ReplayState, context: _TapeContext | None) -> list[str]:
    """Return held names that cannot be represented by the active frozen risk tape.

    This is deliberately narrower than a missing NAV.  A name is a risk orphan
    only when it is outside the active cross-section, lacks an adjusted mark, or
    has no supported beta in that cross-section.  Those positions can be reduced
    without inventing risk inputs; all other unavailable-risk cases remain
    blocked.
    """
    if context is None:
        return []
    return sorted(
        security_id
        for security_id, units in state.units.items()
        if abs(units) > EPS
        and security_id not in state.unknown_terminals
        and security_id not in state.exit_after
        and (
            security_id not in context.lookup
            or security_id not in state.adjusted_marks
            or not context.supported[context.lookup[security_id]]
        )
    )


def _breaches(metrics: dict[str, Any] | None, limits: dict[str, Any]) -> list[str]:
    if metrics is None:
        return ["unavailable_held_risk"]
    tolerance = float(limits.get("tolerance", 1e-6))
    checks = {
        "gross": (metrics["gross"], float(limits["gross"])),
        "name": (metrics["name"], float(limits["name"])),
        "net": (abs(metrics["net"]), float(limits["abs_net"])),
        "beta": (abs(metrics["beta"]), float(limits["abs_beta"])),
        "forecast_risk": (metrics["annual_ex_ante_vol"], float(limits["annual_vol"])),
    }
    result = [name for name, (value, bound) in checks.items() if value > bound + tolerance]
    if any(abs(x["net"]) > float(limits["industry_abs_net"]) + tolerance for x in metrics["industries"].values()):
        result.append("industry_net")
    if any(abs(x["beta"]) > float(limits["industry_abs_beta"]) + tolerance for x in metrics["industries"].values()):
        result.append("industry_beta")
    return result


def _daily_control(
    state: ReplayState,
    *,
    index: int,
    final_index: int,
    day: str,
    metrics: dict[str, Any] | None,
    violations: list[str],
    policy: dict[str, Any],
    context: _TapeContext | None = None,
) -> None:
    """Schedule the parent's buffered next-close reduction without new risk inputs."""
    if not violations:
        return
    unsupported = _unsupported_risk_holdings(state, context) if metrics is None else []
    if metrics is None and unsupported and index < final_index:
        for security_id in unsupported:
            state.schedule_risk_exit(security_id, index + 1)
        state.journal.append(
            {
                "scenario_id": state.scenario_id,
                "date": day,
                "package_id": None,
                "parent_security_id": None,
                "classification": "daily_control",
                "status": "scheduled_unsupported_risk_exit",
                "violations": list(violations),
                "target_names": len(unsupported),
                "target_security_ids": list(unsupported),
                "imputed": False,
            }
        )
        return
    if metrics is None:
        state.journal.append(
            {
                "scenario_id": state.scenario_id,
                "date": day,
                "package_id": None,
                "parent_security_id": None,
                "classification": "daily_control",
                "status": "blocked_unavailable_held_risk_or_nav",
                "violations": list(violations),
                "imputed": False,
            }
        )
        return
    if index >= final_index:
        state.journal.append(
            {
                "scenario_id": state.scenario_id,
                "date": day,
                "package_id": None,
                "parent_security_id": None,
                "classification": "daily_control",
                "status": "right_censored",
                "violations": list(violations),
                "imputed": False,
            }
        )
        return
    optimization = policy["optimization"]
    hard = policy["hard_limits"]
    scale = 1.0
    buffered = (
        (metrics["gross"], float(optimization["inner_gross"])),
        (metrics["name"], float(optimization["inner_name"])),
        (abs(metrics["net"]), min(0.005, float(hard["abs_net"]))),
        (abs(metrics["beta"]), min(0.005, float(hard["abs_beta"]))),
        (metrics["annual_ex_ante_vol"], min(0.09, float(optimization["annual_vol_cap"]))),
    )
    for value, bound in buffered:
        if value > bound and value > 0:
            scale = min(scale, bound / value)
    industry_net = min(0.0025, float(hard["industry_abs_net"]))
    industry_beta = min(0.0025, float(hard["industry_abs_beta"]))
    for group in metrics["industries"].values():
        if abs(group["net"]) > industry_net:
            scale = min(scale, industry_net / abs(group["net"]))
        if abs(group["beta"]) > industry_beta:
            scale = min(scale, industry_beta / abs(group["beta"]))
    if not 0 <= scale < 1.0:
        # A numerical tolerance breach that requires no material reduction is
        # evidence, not a reason to manufacture an order.
        return
    targets = {security_id: units * scale for security_id, units in state.units.items()}
    order = SparseTargets.changed(
        decision_index=index,
        fill_index=index + 1,
        kind="daily_reduction",
        current=state.units,
        targets=targets,
    )
    state.schedule(order)
    state.journal.append(
        {
            "scenario_id": state.scenario_id,
            "date": day,
            "package_id": None,
            "parent_security_id": None,
            "classification": "daily_control",
            "status": "scheduled_next_close",
            "violations": list(violations),
            "uniform_scale": float(scale),
            "target_names": len(order.targets),
            "imputed": False,
        }
    )


def _action_groups(packages: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for package in packages:
        package_id = str(package.get("package_id", ""))
        day = str(package.get("action_date", ""))
        if not package_id or package_id in seen:
            raise ResearchError("Action packages need unique nonempty identifiers")
        try:
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError
        except ValueError:
            raise ResearchError("Action package has an invalid date") from None
        seen.add(package_id)
        grouped.setdefault(day, []).append(package)
    for values in grouped.values():
        values.sort(key=lambda item: (str(item.get("parent_security_id", "")), str(item.get("package_id", ""))))
    return grouped


def replay_arrays(
    *,
    dates: Iterable[str],
    security_ids: Iterable[str],
    adjusted: np.ndarray,
    raw: np.ndarray,
    dollar_volume: np.ndarray,
    packages: Iterable[dict[str, Any]] = (),
    scenario_id: str = "strict_standard_only",
    policy: dict[str, Any] | None = None,
    initial_units: dict[str, float] | None = None,
    starting_cash: float = 1_000_000.0,
    supplemental_quotes: pd.DataFrame | None = None,
    scheduled_orders: dict[int, SparseTargets] | None = None,
    actions: _ActionFunctions = DEFAULT_ACTION_FUNCTIONS,
) -> dict[str, Any]:
    """Small-array replay used by unit tests and independent arithmetic checks."""
    dates = tuple(str(x) for x in dates)
    if policy is None:
        policy = {
            "execution": {
                "annual_hypothetical_borrow": 0.03,
                "cash_debit": 0.05,
                "participation": 0.01,
                "one_way_commission_bps": 1.0,
                "one_way_slippage_bps": 5.0,
            }
        }
    quotes = QuoteBook(dates, security_ids, adjusted, raw, dollar_volume, supplemental_quotes)
    grouped = _action_groups(packages)
    state = ReplayState(scenario_id, float(starting_cash), units=initial_units or {})
    scheduled_orders = scheduled_orders or {}
    for index, day in enumerate(dates):
        borrow, funding = state.accrue(day, policy)
        state.update_marks(index, quotes)
        day_packages = grouped.get(day, ())
        state.register_terminal_parents(day_packages)
        if index in scheduled_orders:
            state.schedule(scheduled_orders[index])
        trade_cost = state.execute_orders(index, day, quotes, policy)
        state.update_marks(index, quotes)
        _process_action_day(
            state,
            index=index,
            day=day,
            packages=day_packages,
            quotes=quotes,
            actions=actions,
        )
        state.update_marks(index, quotes)
        estimate, observed, missing = state.valuation(index, quotes)
        daily_return = (
            observed / state.last_observed_nav - 1.0
            if observed is not None and state.last_observed_nav is not None and state.last_observed_nav > 0
            else None
        )
        state.daily.append(
            {
                "scenario_id": scenario_id,
                "date": day,
                "nav_observed_usd": observed,
                "nav_estimate_usd": estimate,
                "valuation_resolved": observed is not None,
                "missing_held_marks_or_claims": missing,
                "borrow_usd": borrow,
                "funding_usd": funding,
                "trade_cost_usd": trade_cost,
                "return_observed": daily_return,
                "cash_usd": state.cash,
                "held_names": len(state._held_ids()),
                "raw_claims": sum(abs(value) > EPS for value in state.raw_claims.values()),
            }
        )
        state.last_observed_nav = observed
        state.last_day = day
    return {
        "daily": state.daily,
        "journal": state.journal,
        "trades": state.trades,
        "final_units": dict(state.units),
        "final_raw_claims": dict(state.raw_claims),
        "final_cash": float(state.cash),
        "imputed_events": int(state.imputed_events),
        "ever_censored": bool(state.ever_censored),
        "scenario": scenario_metadata(scenario_id),
    }


def _forward_basis(
    *,
    security_id: str,
    decision_index: int,
    entry_index: int,
    exit_index: int,
    quotes: QuoteBook,
    packages_by_day: dict[str, list[dict[str, Any]]],
    scenario_id: str,
    actions: _ActionFunctions,
) -> tuple[float, bool, int]:
    decision_adjusted, _, _ = quotes.quote(decision_index, security_id)
    entry_adjusted, _, _ = quotes.quote(entry_index, security_id)
    if decision_adjusted is None or entry_adjusted is None:
        return 0.0, False, 1
    holdings = {security_id: 1.0 / decision_adjusted}
    raw_claims: dict[str, float] = {}
    cash = 0.0
    unknown: set[str] = set()
    adjusted_marks: dict[str, float] = {}
    raw_marks: dict[str, float] = {}
    entry_value = holdings[security_id] * entry_adjusted
    # Factor intervals are half-open in action time: the portfolio is formed at
    # the entry close and valued immediately before exit-close actions.  The
    # next interval owns actions effective on that exit session.
    for index in range(entry_index, exit_index):
        day = quotes.dates[index]
        for held_id in set(holdings) | set(raw_claims):
            adjusted, raw, _ = quotes.quote(index, held_id)
            if adjusted is not None:
                adjusted_marks[held_id] = adjusted
            if raw is not None:
                raw_marks[held_id] = raw
        day_packages = packages_by_day.get(day, ())
        processed_day: set[str] = set()
        # Match the account path's explicit final quiescence pass.  A claim can
        # first become quotable on an ordinary day with no action package.
        for _ in range(len(day_packages) + 2):
            progressed = False
            for package in day_packages:
                package_id = str(package.get("package_id", ""))
                if package_id in processed_day:
                    continue
                parent_id = str(package.get("parent_security_id", ""))
                parent_units = holdings.get(parent_id, 0.0)
                parent_raw_claim = raw_claims.get(parent_id, 0.0)
                if abs(parent_units) <= EPS and abs(parent_raw_claim) <= EPS:
                    continue
                processed_day.add(package_id)
                progressed = True
                classification = str(package.get("classification", "unresolved"))
                adjusted = adjusted_marks.get(parent_id)
                raw = raw_marks.get(parent_id)
                if classification in STANDARD_CLASSIFICATIONS:
                    if abs(parent_units) > EPS and (adjusted is None or raw is None):
                        unknown.add(parent_id)
                        continue
                    amount = 0.0
                    claims: list[dict[str, Any]] = []
                    if abs(parent_units) > EPS:
                        resolved = actions.resolve_package(package, parent_units, adjusted, raw)
                        if not resolved.get("resolved"):
                            unknown.add(parent_id)
                            continue
                        if resolved.get("parent_retained", False):
                            raise ResearchError("Resolved forward package unexpectedly retained its parent")
                        resolved_amount = _finite(resolved.get("cash_usd"))
                        if resolved_amount is None:
                            raise ResearchError("Action resolver returned nonfinite forward cash")
                        amount += resolved_amount
                        claims.extend(resolved.get("stock_claims", []))
                        holdings.pop(parent_id, None)
                    if abs(parent_raw_claim) > EPS:
                        claim_cash, claim_legs = _resolve_raw_terms(package, parent_raw_claim)
                        amount += claim_cash
                        claims.extend(claim_legs)
                        raw_claims.pop(parent_id, None)
                    unknown.discard(parent_id)
                    cash += amount
                    for claim in claims:
                        recipient_id = str(claim.get("recipient_id", ""))
                        raw_shares = _finite(claim.get("raw_shares"))
                        if not recipient_id or raw_shares is None:
                            raise ResearchError("Action resolver returned invalid forward stock claim")
                        raw_claims[recipient_id] = raw_claims.get(recipient_id, 0.0) + raw_shares
                    continue
                insolvency = _is_insolvency(package)
                if scenario_id == "strict_standard_only":
                    unknown.add(parent_id)
                elif scenario_id == "imputed_zero_insolvency_last_mark_other" and insolvency:
                    holdings.pop(parent_id, None)
                    raw_claims.pop(parent_id, None)
                    unknown.discard(parent_id)
                elif raw is not None and (abs(parent_units) <= EPS or adjusted is not None):
                    raw_parent = parent_raw_claim
                    if abs(parent_units) > EPS:
                        raw_parent += actions.analytical_to_raw(parent_units, adjusted, raw)
                    cash += raw_parent * raw
                    holdings.pop(parent_id, None)
                    raw_claims.pop(parent_id, None)
                    unknown.discard(parent_id)
                else:
                    unknown.add(parent_id)
            for recipient_id in list(raw_claims):
                adjusted, raw, _ = quotes.quote(index, recipient_id)
                if adjusted is None or raw is None:
                    continue
                units = actions.claim_to_analytical(raw_claims.pop(recipient_id), adjusted, raw)
                holdings[recipient_id] = holdings.get(recipient_id, 0.0) + units
                progressed = True
            if not progressed:
                break
        else:
            raise ResearchError("Forward action conversion chain did not terminate")
    value = cash
    missing = 0
    for held_id, units in holdings.items():
        adjusted, _, _ = quotes.quote(exit_index, held_id)
        if adjusted is None:
            missing += 1
        else:
            value += units * adjusted
    for recipient_id, raw_shares in raw_claims.items():
        _, raw, _ = quotes.quote(exit_index, recipient_id)
        if raw is None:
            missing += 1
        else:
            value += raw_shares * raw
    missing += sum(
        abs(holdings.get(security_id, 0.0)) > EPS or abs(raw_claims.get(security_id, 0.0)) > EPS
        for security_id in unknown
    )
    return float(value - entry_value), missing == 0, int(missing)


def action_aware_forward_interval(
    weights: dict[str, float] | np.ndarray,
    *,
    dates: Iterable[str],
    security_ids: Iterable[str],
    adjusted: np.ndarray,
    raw: np.ndarray,
    decision_index: int,
    entry_index: int,
    exit_index: int,
    packages: Iterable[dict[str, Any]] = (),
    scenario_id: str = "strict_standard_only",
    supplemental_quotes: pd.DataFrame | None = None,
    actions: _ActionFunctions = DEFAULT_ACTION_FUNCTIONS,
) -> dict[str, Any]:
    """Evaluate one frozen ideal portfolio with endpoint action conversion.

    The result has the same dollar-P&L-per-one-dollar-weight convention as the
    original study's ``forward_interval``.  Incomplete components are retained
    in ``known_component_pnl`` and never filled with zero in ``return_complete``.
    """
    dates = tuple(str(x) for x in dates)
    ids = tuple(str(x) for x in security_ids)
    if not (0 <= decision_index < entry_index <= exit_index < len(dates)):
        raise ResearchError("Invalid forward interval")
    quotes = QuoteBook(dates, ids, adjusted, raw, None, supplemental_quotes)
    grouped = _action_groups(packages)
    if isinstance(weights, dict):
        vector = {str(k): float(v) for k, v in weights.items() if abs(float(v)) > EPS}
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ResearchError("Invalid forward weights")
        vector = {security_id: float(value) for security_id, value in zip(ids, values) if abs(value) > EPS}
    known = 0.0
    missing = 0
    for security_id, weight in vector.items():
        pnl, complete, missing_count = _forward_basis(
            security_id=security_id,
            decision_index=decision_index,
            entry_index=entry_index,
            exit_index=exit_index,
            quotes=quotes,
            packages_by_day=grouped,
            scenario_id=scenario_id,
            actions=actions,
        )
        if complete:
            known += weight * pnl
        else:
            # Preserve the parent forward diagnostic's definition: one missing
            # starting security is one unavailable component even when its
            # consideration contains several unmarked stock legs.
            missing += 1
    return {
        "return_complete": float(known) if missing == 0 else None,
        "missing_names": int(missing),
        "known_component_pnl": float(known),
    }


def _settings(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(value, "model_dump"):
        data = value.model_dump()
    elif isinstance(value, (str, Path)):
        data = read_json(value)
    elif isinstance(value, dict):
        data = dict(value)
    else:
        raise ResearchError("Invalid replay settings")
    policy = data.get("policy", data.get("pinned_policy"))
    if not isinstance(policy, dict):
        raise ResearchError("Replay settings omit the pinned parent policy")
    scenarios = data.get("scenario_ids", data.get("scenarios", list(SCENARIO_IDS)))
    if isinstance(scenarios, dict):
        scenario_ids = tuple(scenarios)
    else:
        scenario_ids = tuple(x.get("scenario_id") if isinstance(x, dict) else x for x in scenarios)
    if set(scenario_ids) != set(SCENARIO_IDS) or len(scenario_ids) != len(SCENARIO_IDS):
        raise ResearchError("Replay must execute exactly the three preregistered scenarios")
    return data, policy


def _load_action_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, _ActionFunctions]:
    try:
        from . import factor_actions
    except ImportError as error:
        raise ResearchError("Normalized factor-action bundle support is unavailable") from error
    # The registered wrapper pins the normalized bundle files and their source
    # identities.  Replay consumes that closed bundle rather than reopening the
    # licensed upstream tables from manifest paths.
    loaded = factor_actions.load_action_bundle(path, verify_sources=False)
    manifest = loaded.get("manifest", loaded) if isinstance(loaded, dict) else {}
    packages = loaded.get("packages") if isinstance(loaded, dict) else None
    if not isinstance(packages, list):
        packages = read_json(path / "packages.json")
    supplemental = loaded.get("quotes", loaded.get("supplemental_quotes")) if isinstance(loaded, dict) else None
    if not isinstance(supplemental, pd.DataFrame):
        supplemental = pd.read_parquet(path / "supplemental-quotes.parquet")
    actions = _ActionFunctions(
        analytical_to_raw=factor_actions.analytical_to_raw,
        resolve_package=factor_actions.resolve_package,
        claim_to_analytical=factor_actions.claim_to_analytical,
    )
    return manifest, packages, supplemental, actions


def _verify_parent(parent: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    marker = read_json(parent / "COMPLETE.json")
    record = read_json(parent / "run.json")
    spec = read_json(parent / "spec.json")
    if marker.get("kind") != "large_cap_factor_study_v1" or record.get("status") != "completed":
        raise ResearchError("Parent factor study is not complete")
    if marker.get("trial_id") != record.get("trial_id") or marker.get("trial_id") != spec.get("trial_id", marker.get("trial_id")):
        # Schema-5 specs do not store trial_id; the COMPLETE/run identity is authoritative.
        if marker.get("trial_id") != record.get("trial_id"):
            raise ResearchError("Parent factor study identity mismatch")
    files = marker.get("files")
    if not isinstance(files, dict):
        raise ResearchError("Parent completion inventory is absent")
    for required in ("months.csv", "coverage.json", "outcome.json"):
        item = files.get(required)
        target = parent / required
        if not isinstance(item, dict) or not target.is_file() or target.stat().st_size != item.get("bytes") or file_hash(target) != item.get("sha256"):
            raise ResearchError("Parent factor artifact changed")
    return marker, record, spec


def _tape(parent: Path, filename: str, inventory: dict[str, Any]) -> dict[str, np.ndarray]:
    path = parent / filename
    item = inventory.get(filename)
    if not isinstance(item, dict) or not path.is_file() or path.stat().st_size != item.get("bytes") or file_hash(path) != item.get("sha256"):
        raise ResearchError("Frozen monthly tape changed")
    with np.load(path, allow_pickle=False) as source:
        required = {
            "security_ids",
            "factor_weights",
            "model_weights",
            "expected_returns",
            "beta",
            "risk_diagonal",
            "risk_loadings",
        }
        if not required.issubset(source.files):
            raise ResearchError("Frozen monthly tape is incomplete")
        return {name: np.array(source[name], copy=True) for name in required}


def _context(day: str, tape: dict[str, np.ndarray], industries: dict[str, str]) -> _TapeContext:
    ids = tuple(str(x) for x in tape["security_ids"].tolist())
    n = len(ids)
    beta_observed = np.asarray(tape["beta"], dtype=float)
    d = np.asarray(tape["risk_diagonal"], dtype=float)
    b = np.asarray(tape["risk_loadings"], dtype=float)
    if beta_observed.shape != (n,) or d.shape != (n,) or b.ndim != 2 or b.shape[1] != n:
        raise ResearchError("Frozen risk tape has invalid dimensions")
    supported = np.isfinite(beta_observed)
    if not np.isfinite(d).all() or not np.isfinite(b).all() or np.any(d <= 0):
        raise ResearchError("Frozen risk tape is nonfinite")
    beta = np.where(supported, beta_observed, 0.0)
    return _TapeContext(
        day,
        ids,
        {security_id: j for j, security_id in enumerate(ids)},
        np.array([industries.get(security_id, "UNKNOWN") for security_id in ids]),
        beta,
        supported,
        b,
        d,
    )


def _optimize_frozen(
    state: ReplayState,
    context: _TapeContext,
    tape: dict[str, np.ndarray],
    members: set[str],
    observed_nav: float | None,
    policy: dict[str, Any],
    *,
    index: int,
    quotes: QuoteBook,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    if observed_nav is None or observed_nav <= 0 or state.raw_claims:
        return {"status": "blocked_unresolved_account"}, None
    held = {security_id for security_id, value in state.units.items() if abs(value) > EPS}
    risk_orphans = set(_unsupported_risk_holdings(state, context))
    if any(security_id in state.unknown_terminals for security_id in held):
        return {"status": "blocked_unresolved_account"}, None
    action_exits = held & set(state.exit_after)
    queued_risk_exits = held & set(state.risk_exit_after)
    projected_risk_exits = risk_orphans | queued_risk_exits
    projected_exits = projected_risk_exits | action_exits
    current = np.zeros(len(context.ids))
    for security_id in held - projected_exits:
        j = context.lookup[security_id]
        current[j] = state.units[security_id] * state.adjusted_marks[security_id] / observed_nav
    formation_prices = {
        security_id: quotes.quote(index, security_id)[0]
        for security_id in context.ids
    }
    eligible = np.array(
        [
            security_id in members
            and security_id not in state.restricted_ids
            and security_id not in projected_exits
            and context.industry[j] != "UNKNOWN"
            and context.supported[j]
            and formation_prices[security_id] is not None
            for j, security_id in enumerate(context.ids)
        ],
        dtype=bool,
    )
    mu = np.asarray(tape["expected_returns"], dtype=float)
    model = np.asarray(tape["model_weights"], dtype=float)
    if mu.shape != current.shape or model.shape != current.shape or not np.isfinite(mu).all() or not np.isfinite(model).all():
        raise ResearchError("Frozen expected-return/model tape is invalid")
    limits = policy["optimization"]
    result = optimize(
        mu,
        current,
        context.industry,
        context.beta,
        context.b,
        context.d,
        eligible,
        max_gross=float(limits["inner_gross"]),
        max_name=float(limits["inner_name"]),
        max_turnover=float(limits["inner_turnover"]),
        target_vol=float(limits["annual_vol_cap"]),
        trading_cost=(float(policy["execution"]["one_way_commission_bps"]) + float(policy["execution"]["one_way_slippage_bps"])) / 10_000.0,
        annual_borrow=float(policy["execution"]["annual_hypothetical_borrow"]),
        horizon_calendar_days=float(limits["horizon_borrow_days"]),
        time_limit=float(limits["solver_time_limit_seconds"]),
    )
    weights = result.pop("weights", None)
    if weights is None:
        return result, None
    result["risk_orphan_exit_names"] = len(projected_risk_exits)
    result["action_recipient_exit_names"] = len(action_exits)
    if projected_exits and result.get("status") == "feasible":
        result["status"] = "feasible_conditional_reduce_only_exits"
    weights = np.asarray(weights, dtype=float)
    weights[~eligible & (np.abs(current) < EPS)] = 0.0
    targets: dict[str, float] = {}
    for j, security_id in enumerate(context.ids):
        price = formation_prices[security_id]
        if price is None or price <= 0:
            if abs(current[j]) > EPS:
                targets[security_id] = state.units.get(security_id, 0.0)
            continue
        target = weights[j] * observed_nav / price
        if abs(target - state.units.get(security_id, 0.0)) > EPS:
            targets[security_id] = float(target)
    return result, targets


def _optimization_cache_key(
    state: ReplayState,
    context: _TapeContext,
    members: set[str],
    observed_nav: float | None,
    *,
    index: int,
    quotes: QuoteBook,
) -> str:
    """Hash every scenario-state input that can change frozen optimization.

    The cache is shared across the three action scenarios at one formation.
    Queued reduce-only exits must therefore be part of the identity even when
    current units and NAV happen to match across scenarios.
    """
    held = state._held_ids()
    risk_orphans = set(_unsupported_risk_holdings(state, context))
    action_exits = held & set(state.exit_after)
    queued_risk_exits = held & set(state.risk_exit_after)
    projected_exits = risk_orphans | action_exits | queued_risk_exits
    current = np.array(
        [
            state.units.get(security_id, 0.0)
            * state.adjusted_marks.get(security_id, 0.0)
            / observed_nav
            if observed_nav not in (None, 0)
            else 0.0
            for security_id in context.ids
        ],
        dtype=float,
    )
    eligible_signature = np.array(
        [
            security_id in members
            and security_id not in state.restricted_ids
            and security_id not in projected_exits
            and context.industry[j] != "UNKNOWN"
            and context.supported[j]
            and quotes.quote(index, security_id)[0] is not None
            for j, security_id in enumerate(context.ids)
        ],
        dtype=np.uint8,
    )
    state_signature = json.dumps(
        {
            "projected_exit_ids": sorted(projected_exits),
            "unknown_held_ids": sorted(held & state.unknown_terminals),
            "has_raw_claims": bool(state.raw_claims),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    nav_signature = np.array(
        [np.nan if observed_nav is None else observed_nav],
        dtype=float,
    )
    return hashlib.sha256(
        current.tobytes()
        + eligible_signature.tobytes()
        + nav_signature.tobytes()
        + state_signature
    ).hexdigest()


def _load_factor_names(parent: Path, factor_count: int) -> list[str]:
    coverage = read_json(parent / "coverage.json")
    names = [str(row["jkp_id"]) for row in coverage if row.get("implementation_status") == "provider_reconstruction"]
    if len(names) != factor_count or len(set(names)) != len(names):
        raise ResearchError("Frozen factor-name inventory does not match monthly tapes")
    return names


def _forward_rows(
    *,
    tape_day: str,
    tape: dict[str, np.ndarray],
    next_day: str | None,
    calendar_index: dict[str, int],
    final_index: int,
    factor_names: list[str],
    quotes: QuoteBook,
    grouped_actions: dict[str, list[dict[str, Any]]],
    actions: _ActionFunctions,
) -> list[dict[str, Any]]:
    if next_day is None:
        return []
    decision = calendar_index[tape_day]
    entry = decision + 1
    exit_index = min(calendar_index[next_day] + 1, final_index)
    if entry > exit_index:
        return []
    ids = tuple(str(x) for x in tape["security_ids"].tolist())
    weights = np.vstack([np.asarray(tape["factor_weights"], dtype=float), np.asarray(tape["model_weights"], dtype=float)[None, :]])
    if weights.shape != (len(factor_names) + 1, len(ids)) or not np.isfinite(weights).all():
        raise ResearchError("Frozen forward weight tape is invalid")
    rows: list[dict[str, Any]] = []
    for scenario_id in SCENARIO_IDS:
        basis = np.zeros(len(ids))
        complete = np.zeros(len(ids), dtype=bool)
        missing_counts = np.zeros(len(ids), dtype=int)
        for j, security_id in enumerate(ids):
            basis[j], complete[j], missing_counts[j] = _forward_basis(
                security_id=security_id,
                decision_index=decision,
                entry_index=entry,
                exit_index=exit_index,
                quotes=quotes,
                packages_by_day=grouped_actions,
                scenario_id=scenario_id,
                actions=actions,
            )
        for name, vector in zip([*factor_names, "MODEL"], weights):
            active = np.abs(vector) > 1e-10
            missing = active & ~complete
            known = float(vector[active & complete] @ basis[active & complete])
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_label": SCENARIO_METADATA[scenario_id]["label"],
                    "imputed_sensitivity": SCENARIO_METADATA[scenario_id]["imputed"],
                    "date": tape_day,
                    "factor": name,
                    "entry_date": quotes.dates[entry],
                    "exit_date": quotes.dates[exit_index],
                    "return_complete": known if not missing.any() else None,
                    "known_component_pnl": known,
                    "missing_names": int(missing_counts[missing].sum()),
                }
            )
    return rows


def _resolved_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start: str | None = None
    end: str | None = None
    sessions = 0
    for row in rows:
        if row["valuation_resolved"]:
            if start is None:
                start = row["date"]
                sessions = 0
            end = row["date"]
            sessions += 1
        elif start is not None:
            segments.append({"start": start, "end": end, "sessions": sessions})
            start = end = None
            sessions = 0
    if start is not None:
        segments.append({"start": start, "end": end, "sessions": sessions})
    return segments


def run_replay(inputs: Path, frozen_run: Path, action_bundle: Path, output: Path, settings: Any) -> dict[str, Any]:
    """Replay one completed frozen factor study under all action scenarios."""
    inputs = Path(inputs).resolve()
    parent = Path(frozen_run).resolve()
    action_bundle = Path(action_bundle).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ResearchError("Replay output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    setting_data, policy = _settings(settings)

    from .factor_inputs import load_inputs

    input_manifest = load_inputs(inputs)
    marker, parent_record, parent_spec = _verify_parent(parent)
    if parent_spec.get("bundle_id") != input_manifest.get("bundle_id"):
        raise ResearchError("Parent run and replay inputs differ")
    parent_hash = file_hash(parent / "COMPLETE.json")
    requested_parent_hash = setting_data.get("parent_result_hash")
    if requested_parent_hash is not None and requested_parent_hash != parent_hash:
        raise ResearchError("Pinned parent result hash changed")

    action_manifest, packages, supplemental, actions = _load_action_bundle(action_bundle)
    action_bundle_id = action_manifest.get("action_bundle_id", action_manifest.get("bundle_id"))
    if not isinstance(action_bundle_id, str) or not action_bundle_id:
        raise ResearchError("Action bundle identity is absent")
    requested_action_id = setting_data.get("action_bundle_id")
    if requested_action_id is not None and requested_action_id != action_bundle_id:
        raise ResearchError("Pinned action bundle identity changed")

    identities = read_json(inputs / "identities.json")
    base_ids = tuple(str(row["security_id"]) for row in identities)
    industries = {str(row["security_id"]): str(row.get("industry", "UNKNOWN")) for row in identities}
    calendar = pd.read_csv(inputs / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.strftime("%Y-%m-%d")
    dates = tuple(calendar["date"].astype(str))
    calendar_index = {day: i for i, day in enumerate(dates)}
    start = str(input_manifest["start"])
    end = str(input_manifest["end"])
    first_index = calendar_index[start]
    final_index = calendar_index[end]
    adjusted = np.load(inputs / "adjusted.npy", mmap_mode="r", allow_pickle=False)
    raw = np.load(inputs / "raw.npy", mmap_mode="r", allow_pickle=False)
    dollar_volume = np.load(inputs / "dollar_volume.npy", mmap_mode="r", allow_pickle=False)
    quotes = QuoteBook(dates, base_ids, adjusted, raw, dollar_volume, supplemental)
    grouped_actions = _action_groups(packages)

    db = duckdb.connect(config={"threads": "1", "memory_limit": "2GB"})
    universe = db.execute(
        "SELECT date,security_id FROM read_parquet(?) ORDER BY date,security_id", [str(inputs / "universe.parquet")]
    ).df()
    db.close()
    universe["date"] = pd.to_datetime(universe["date"], errors="raise").dt.strftime("%Y-%m-%d")
    universe["security_id"] = universe["security_id"].astype(str)
    members = {day: set(group.security_id) for day, group in universe.groupby("date", sort=True)}

    parent_months = pd.read_csv(parent / "months.csv")
    parent_months["date"] = pd.to_datetime(parent_months["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if parent_months["date"].duplicated().any():
        raise ResearchError("Parent monthly formation dates are repeated")
    tape_days = parent_months["date"].tolist()
    tape_by_day = {day: f"month-{day[:7]}.npz" for day in tape_days}
    if len(set(tape_by_day.values())) != len(tape_by_day):
        raise ResearchError("Parent monthly tape names are ambiguous")
    boundary = {day: tape_days[j + 1] if j + 1 < len(tape_days) else None for j, day in enumerate(tape_days)}

    states = {
        scenario_id: ReplayState(scenario_id, float(policy["execution"]["starting_nav_usd"]))
        for scenario_id in SCENARIO_IDS
    }
    daily_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    context_by_scenario: dict[str, _TapeContext | None] = {scenario_id: None for scenario_id in SCENARIO_IDS}
    factor_names: list[str] | None = None

    for index in range(first_index, final_index + 1):
        day = dates[index]
        day_packages = grouped_actions.get(day, ())
        metrics_today: dict[str, dict[str, Any] | None] = {}
        violations_today: dict[str, list[str]] = {}
        for state in states.values():
            borrow, funding = state.accrue(day, policy)
            state.update_marks(index, quotes)
            state.register_terminal_parents(day_packages)
            trade_cost = state.execute_orders(
                index,
                day,
                quotes,
                policy,
                context_by_scenario[state.scenario_id],
            )
            state.update_marks(index, quotes)
            _process_action_day(
                state,
                index=index,
                day=day,
                packages=day_packages,
                quotes=quotes,
                actions=actions,
            )
            state.update_marks(index, quotes)
            estimate, observed, missing = state.valuation(index, quotes)
            metrics = _position_metrics(state, context_by_scenario[state.scenario_id], observed)
            violations = _breaches(metrics, policy["hard_limits"])
            metrics_today[state.scenario_id] = metrics
            violations_today[state.scenario_id] = violations
            daily_return = (
                observed / state.last_observed_nav - 1.0
                if observed is not None and state.last_observed_nav is not None and state.last_observed_nav > 0
                else None
            )
            row = {
                "scenario_id": state.scenario_id,
                "scenario_label": SCENARIO_METADATA[state.scenario_id]["label"],
                "imputed_sensitivity": SCENARIO_METADATA[state.scenario_id]["imputed"],
                "date": day,
                "nav_observed_usd": observed,
                "nav_estimate_usd": estimate,
                "valuation_resolved": observed is not None,
                "missing_held_marks_or_claims": missing,
                "borrow_usd": borrow,
                "funding_usd": funding,
                "trade_cost_usd": trade_cost,
                "return_observed": daily_return,
                "cash_usd": state.cash,
                "held_names": len(state._held_ids()),
                "raw_claims": sum(abs(value) > EPS for value in state.raw_claims.values()),
                "gross": None if metrics is None else metrics["gross"],
                "net": None if metrics is None else metrics["net"],
                "beta": None if metrics is None else metrics["beta"],
                "name_max": None if metrics is None else metrics["name"],
                "annual_ex_ante_vol": None if metrics is None else metrics["annual_ex_ante_vol"],
                "hard_violations": ",".join(violations),
            }
            state.daily.append(row)
            daily_rows.append(row)
            state.last_observed_nav = observed
            state.last_day = day

        if day not in tape_by_day:
            for scenario_id, state in states.items():
                _daily_control(
                    state,
                    index=index,
                    final_index=final_index,
                    day=day,
                    metrics=metrics_today[scenario_id],
                    violations=violations_today[scenario_id],
                    policy=policy,
                    context=context_by_scenario[scenario_id],
                )
            if index == final_index - 1:
                for state in states.values():
                    state.schedule(
                        SparseTargets.changed(
                            decision_index=index,
                            fill_index=index + 1,
                            kind="predeclared_final_close",
                            current=state.units,
                            targets={security_id: 0.0 for security_id in state.units},
                        )
                    )
            continue
        tape = _tape(parent, tape_by_day[day], marker["files"])
        context = _context(day, tape, industries)
        if factor_names is None:
            factor_names = _load_factor_names(parent, int(np.asarray(tape["factor_weights"]).shape[0]))
        elif len(factor_names) != int(np.asarray(tape["factor_weights"]).shape[0]):
            raise ResearchError("Frozen factor count changed between monthly tapes")
        if day != tape_days[-1]:
            forward_rows.extend(
                _forward_rows(
                    tape_day=day,
                    tape=tape,
                    next_day=boundary[day],
                    calendar_index=calendar_index,
                    final_index=final_index,
                    factor_names=factor_names,
                    quotes=quotes,
                    grouped_actions=grouped_actions,
                    actions=actions,
                )
            )
        optimization_cache: dict[str, tuple[dict[str, Any], dict[str, float] | None]] = {}
        for scenario_id, state in states.items():
            context_by_scenario[scenario_id] = context
            latest = state.daily[-1]
            # The formation-close tape replaces the prior month's risk context
            # for both the close diagnostic and any next-close reduction.
            metrics = _position_metrics(state, context, latest["nav_observed_usd"])
            violations = _breaches(metrics, policy["hard_limits"])
            metrics_today[scenario_id] = metrics
            violations_today[scenario_id] = violations
            latest.update(
                gross=None if metrics is None else metrics["gross"],
                net=None if metrics is None else metrics["net"],
                beta=None if metrics is None else metrics["beta"],
                name_max=None if metrics is None else metrics["name"],
                annual_ex_ante_vol=None if metrics is None else metrics["annual_ex_ante_vol"],
                hard_violations=",".join(violations),
            )
            if day == tape_days[-1]:
                result, targets = {"status": "right_censored"}, None
            else:
                key = _optimization_cache_key(
                    state,
                    context,
                    members.get(day, set()),
                    latest["nav_observed_usd"],
                    index=index,
                    quotes=quotes,
                )
                if key not in optimization_cache:
                    result, targets = _optimize_frozen(
                        state,
                        context,
                        tape,
                        members.get(day, set()),
                        latest["nav_observed_usd"],
                        policy,
                        index=index,
                        quotes=quotes,
                    )
                    optimization_cache[key] = (dict(result), None if targets is None else dict(targets))
                else:
                    cached_result, cached_targets = optimization_cache[key]
                    result, targets = dict(cached_result), None if cached_targets is None else dict(cached_targets)
                if targets is not None:
                    state.schedule(
                        SparseTargets.changed(
                            decision_index=index,
                            fill_index=index + 1,
                            kind="frozen_monthly_alpha",
                            current=state.units,
                            targets=targets,
                        )
                    )
            month_rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_label": SCENARIO_METADATA[scenario_id]["label"],
                    "imputed_sensitivity": SCENARIO_METADATA[scenario_id]["imputed"],
                    "date": day,
                    "universe_names": len(members.get(day, set())),
                    "held_names": len(state._held_ids()),
                    "restricted_held_names": sum(
                        abs(state.units.get(security_id, 0.0)) > EPS for security_id in state.restricted_ids
                    ),
                    "optimizer_status": result.get("status"),
                    "optimizer_gross": result.get("gross"),
                    "optimizer_turnover": result.get("turnover"),
                    "optimizer_annual_vol": result.get("annual_ex_ante_vol"),
                    "solver_seconds": result.get("solve_seconds"),
                    "scheduled_target_names": 0 if targets is None else len(targets),
                    "risk_orphan_exit_names": result.get("risk_orphan_exit_names", 0),
                    "action_recipient_exit_names": result.get("action_recipient_exit_names", 0),
                }
            )
            # Buffered reductions have priority over the monthly alpha batch,
            # exactly as in the frozen parent execution policy.
            _daily_control(
                state,
                index=index,
                final_index=final_index,
                day=day,
                metrics=metrics,
                violations=violations,
                policy=policy,
                context=context,
            )

        if index == final_index - 1:
            for state in states.values():
                state.schedule(
                    SparseTargets.changed(
                        decision_index=index,
                        fill_index=index + 1,
                        kind="predeclared_final_close",
                        current=state.units,
                        targets={security_id: 0.0 for security_id in state.units},
                    )
                )

    pd.DataFrame(daily_rows).to_csv(output / "daily.csv", index=False)
    pd.DataFrame(month_rows).to_csv(output / "months.csv", index=False)
    pd.DataFrame(forward_rows).to_csv(output / "forward.csv", index=False)
    journal = [entry for state in states.values() for entry in state.journal]
    journal.sort(key=lambda row: (str(row.get("date")), str(row.get("scenario_id")), str(row.get("package_id"))))
    _write_json(
        output / "action-journal.json",
        {
            "scenario_ids": list(SCENARIO_IDS),
            "standard_action_accounting_common": True,
            "entries": journal,
        },
    )

    all_trades = [trade for state in states.values() for trade in state.trades]
    trade_ids = sorted({str(trade["security_id"]) for trade in all_trades})
    id_lookup = {security_id: j for j, security_id in enumerate(trade_ids)}
    scenario_lookup = {scenario_id: j for j, scenario_id in enumerate(SCENARIO_IDS)}
    np.savez_compressed(
        output / "trade-vectors.npz",
        scenario_ids=np.array(SCENARIO_IDS),
        security_ids=np.array(trade_ids),
        scenario_index=np.array([scenario_lookup[x["scenario_id"]] for x in all_trades], dtype=np.int16),
        session_index=np.array([x["session_index"] for x in all_trades], dtype=np.int32),
        security_index=np.array([id_lookup[x["security_id"]] for x in all_trades], dtype=np.int32),
        requested_delta=np.array([x["requested_delta"] for x in all_trades], dtype=float),
        filled_delta=np.array([x["filled_delta"] for x in all_trades], dtype=float),
        kind=np.array([x["kind"] for x in all_trades]),
        batch_status=np.array([x["batch_status"] for x in all_trades]),
        batch_violations=np.array([x["batch_violations"] for x in all_trades]),
        batch_turnover=np.array(
            [np.nan if x["batch_turnover"] is None else x["batch_turnover"] for x in all_trades],
            dtype=float,
        ),
    )

    summaries: dict[str, Any] = {}
    for scenario_id, state in states.items():
        rows = state.daily
        complete_path = bool(rows) and all(row["valuation_resolved"] for row in rows)
        final = rows[-1]
        resolved_rows = [row for row in rows if row["valuation_resolved"]]
        unresolved_terminal_exposure = float(
            sum(
                state.units.get(security_id, 0.0) * state.adjusted_marks.get(security_id, 0.0)
                for security_id in state.unknown_terminals
            )
        )
        unresolved_claim_exposure = float(
            sum(
                shares * state.raw_marks.get(security_id, 0.0)
                for security_id, shares in state.raw_claims.items()
            )
        )
        unresolved_action_events = sum(
            row.get("status")
            in {
                "standard_missing_parent_conversion_basis",
                "standard_unresolved",
                "nonstandard_censored",
                "nonstandard_missing_last_mark",
            }
            for row in state.journal
        )
        summaries[scenario_id] = {
            **scenario_metadata(scenario_id),
            "sessions": len(rows),
            "formations": sum(row["scenario_id"] == scenario_id for row in month_rows),
            "imputed_events": state.imputed_events,
            "unresolved_sessions": sum(not row["valuation_resolved"] for row in rows),
            "first_unresolved_date": next((row["date"] for row in rows if not row["valuation_resolved"]), None),
            "last_fully_reconstructed_date": resolved_rows[-1]["date"] if resolved_rows else None,
            "last_fully_reconstructed_nav_usd": resolved_rows[-1]["nav_observed_usd"] if resolved_rows else None,
            "continuous_resolved_segments": _resolved_segments(rows),
            "unresolved_action_events": int(unresolved_action_events),
            "unresolved_terminal_names": int(
                sum(
                    abs(state.units.get(security_id, 0.0)) > EPS
                    or abs(state.raw_claims.get(security_id, 0.0)) > EPS
                    for security_id in state.unknown_terminals
                )
            ),
            "unresolved_terminal_signed_marked_exposure_usd": unresolved_terminal_exposure,
            "unresolved_stock_claim_names": int(sum(abs(value) > EPS for value in state.raw_claims.values())),
            "unresolved_stock_claim_signed_marked_exposure_usd": unresolved_claim_exposure,
            "final_nav_observed_usd": final["nav_observed_usd"],
            "final_nav_estimate_usd": final["nav_estimate_usd"],
            "complete_path_return": (
                final["nav_observed_usd"] / float(policy["execution"]["starting_nav_usd"]) - 1.0
                if complete_path and final["nav_observed_usd"] is not None
                else None
            ),
            "borrow_usd": sum(row["borrow_usd"] for row in rows),
            "funding_usd": sum(row["funding_usd"] for row in rows),
            "trade_cost_usd": sum(row["trade_cost_usd"] for row in rows),
            "final_held_names": final["held_names"],
            "final_raw_claims": final["raw_claims"],
        }
    parent_trial_id = str(setting_data.get("parent_trial_id", marker["trial_id"]))
    if parent_trial_id != marker["trial_id"]:
        raise ResearchError("Pinned parent trial identity changed")
    outcomes = {
        "methodology": "frozen_factor_tape_action_aware_replay_v1",
        "start": start,
        "end": end,
        "parent_trial_id": parent_trial_id,
        "parent_result_hash": parent_hash,
        "action_bundle_id": action_bundle_id,
        "scenario_ids": list(SCENARIO_IDS),
        "scenarios": summaries,
        "standard_action_accounting_common": True,
        "recipient_execution": "next observed close, reduce-only, 1% dollar-volume participation; missing quote/volume retained",
        "point_in_time_attested": False,
        "investment_ready": False,
        "holdout_evaluations": 0,
        "provider_calls": 0,
        "llm_calls": 0,
        "known_limitations": [
            "Signals, factor/theme/model weights, expected returns and risk arrays are frozen parent artifacts, not recomputed",
            "Standard cash/stock/mixed actions use signed raw-share conversion; nonstandard scenarios are explicitly labeled sensitivities",
            "Strict NAV is censored for unresolved action economics or unmarked held recipient claims",
            "Action recipients and nonmembers are reduce-only; missing quote or volume never creates an executable fill",
            "Current identity/industry metadata and the parent study's point-in-time limitations remain unchanged",
            "No stock-loan availability or realized borrow fee evidence is introduced",
        ],
    }
    _write_json(output / "outcomes.json", outcomes)
    return outcomes


__all__ = [
    "SCENARIO_IDS",
    "SCENARIO_METADATA",
    "SparseTargets",
    "ReplayState",
    "action_aware_forward_interval",
    "replay_arrays",
    "run_replay",
    "scenario_metadata",
]
