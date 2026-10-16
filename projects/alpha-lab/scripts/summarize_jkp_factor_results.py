#!/usr/bin/env python3
"""Build the immutable, descriptive V4 JKP factor-results bundle.

The source run is outcome-known development evidence.  This generator therefore
reports arithmetic statistics only on fully reconstructed forward intervals.  It
does not manufacture missing returns or report path-dependent statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
BUNDLE_KIND = "jkp_action_accounting_v4_derived_factor_summary_v2"
DEFAULT_OUTPUT = Path(
    "data/market/research/analysis/jkp-action-accounting-v4-summary-v2"
)
SUPERSEDED_BUNDLE = {
    "path": "data/market/research/analysis/jkp-action-accounting-v4-summary-v1",
    "bundle_id": "fbd9d03295637bd67f7df1b54b932442270acc99692b908b998ea943e335e3c3",
    "manifest_sha256": "faf9a6602570aa54f1095452fe9f4bf82918d810fade2343b61ce23933b60b41",
    "reason": (
        "V1 used an overclaimed label for the conditional annual-mean-to-volatility "
        "ratio even though no cash-return convention, costs, or executable account "
        "was defined. Its numeric values remain unchanged historical artifacts."
    ),
}

V4_ROOT = Path("data/market/research/jkp-action-accounting-v4/run")
PARENT_ROOT = Path("data/market/research/jkp-framework-v1/run")

# These are byte hashes of the accepted, immutable local artifacts.  The
# COMPLETE manifests independently bind forward.csv, protocol.json and
# coverage.json, and the generator verifies both layers.
PINNED_SOURCE_HASHES: dict[str, str] = {
    (V4_ROOT / "COMPLETE.json").as_posix(): (
        "5768bed8f730820ea92e879fe26cb2792fbd0c3e1ce38b394ac1b33fc0b0caff"
    ),
    (V4_ROOT / "forward.csv").as_posix(): (
        "be61f981499ffcf6da3be17daf6ba7be5dc1d008c971fa6a3f428fa5daafd923"
    ),
    (V4_ROOT / "protocol.json").as_posix(): (
        "c9059b639266db94c2da36726a9b283ee45b9c1636abc1239d4ad203b5bb94d1"
    ),
    (PARENT_ROOT / "COMPLETE.json").as_posix(): (
        "0335453b939c80be4e535bb17ba2b7c150b807d8d43b6792a1bbbbe3928a38f8"
    ),
    (PARENT_ROOT / "coverage.json").as_posix(): (
        "1f2f96228e35012c3282983fddc8207ae2c39351f516f537cbe07fda9bba8d8e"
    ),
}

EXPECTED_SCENARIOS = (
    "strict_standard_only",
    "imputed_zero_insolvency_last_mark_other",
    "imputed_last_mark_all_unresolved",
)
EXPECTED_PRODUCTION_SHAPE = {
    "catalog_factors": 153,
    "supported_factors": 117,
    "blocked_factors": 36,
    "forward_factors": 118,
    "intervals_per_factor_scenario": 285,
    "forward_rows": 100_890,
}

FACTOR_METRIC_FIELDS = (
    "scenario_id",
    "scenario_type",
    "factor",
    "factor_type",
    "theme",
    "description",
    "data_family",
    "jkp_sign",
    "exact_jkp_replication",
    "complete_intervals",
    "total_intervals",
    "incomplete_intervals",
    "coverage_rate",
    "first_complete_entry_date",
    "last_complete_exit_date",
    "arithmetic_monthly_mean",
    "monthly_sample_volatility",
    "annualized_arithmetic_mean",
    "annualized_sample_volatility",
    "conditional_ann_mean_to_vol",
    "positive_interval_rate",
    "newey_west_lag4_t_stat",
    "statistical_status",
)

SENSITIVITY_FIELDS = (
    "factor",
    "all_scenarios_common_intervals",
    "strict_complete_intervals",
    "zero_insolvency_complete_intervals",
    "last_mark_complete_intervals",
    "strict_annualized_arithmetic_mean_on_common",
    "zero_insolvency_annualized_arithmetic_mean_on_common",
    "last_mark_annualized_arithmetic_mean_on_common",
    "zero_minus_strict_annualized_arithmetic_mean",
    "last_mark_minus_strict_annualized_arithmetic_mean",
    "zero_minus_strict_max_abs_interval_difference",
    "last_mark_minus_strict_max_abs_interval_difference",
    "statistical_status",
)

COVERAGE_FIELDS = (
    "scenario_id",
    "formation_year",
    "complete_factor_intervals",
    "total_factor_intervals",
    "coverage_rate",
    "model_complete_intervals",
    "model_total_intervals",
    "model_coverage_rate",
    "factors_with_any_complete_interval",
    "factors_fully_complete",
    "factors_total",
)

BLOCKED_FIELDS = (
    "jkp_id",
    "theme",
    "description",
    "data_family",
    "blocked_reason",
    "reference",
    "implementation_status",
    "exact_jkp_replication",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def newey_west_mean_t(values: Sequence[float], lag: int = 4) -> float | None:
    """Return the Bartlett-kernel HAC t statistic for the arithmetic mean.

    Autocovariances use the common ``1/n`` denominator.  The requested lag is
    truncated to ``n - 1`` for small samples.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        return None
    centered = array - float(array.mean())
    n = int(array.size)
    used_lag = min(int(lag), n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for offset in range(1, used_lag + 1):
        weight = 1.0 - offset / (used_lag + 1.0)
        autocovariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run_variance += 2.0 * weight * autocovariance
    if long_run_variance <= 0.0 or not math.isfinite(long_run_variance):
        return None
    return _finite_or_none(float(array.mean()) / math.sqrt(long_run_variance / n))


def summarize_returns(values: Sequence[float]) -> dict[str, float | int | None]:
    """Compute non-path-dependent monthly diagnostics for complete intervals."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("return sample must be a finite one-dimensional array")
    n = int(array.size)
    if n == 0:
        return {
            "observations": 0,
            "arithmetic_monthly_mean": None,
            "monthly_sample_volatility": None,
            "annualized_arithmetic_mean": None,
            "annualized_sample_volatility": None,
            "conditional_ann_mean_to_vol": None,
            "positive_interval_rate": None,
            "newey_west_lag4_t_stat": None,
        }
    monthly_mean = float(array.mean())
    monthly_vol = float(array.std(ddof=1)) if n >= 2 else None
    annual_mean = 12.0 * monthly_mean
    annual_vol = math.sqrt(12.0) * monthly_vol if monthly_vol is not None else None
    mean_to_vol = (
        annual_mean / annual_vol
        if annual_vol is not None and annual_vol > 0.0
        else None
    )
    return {
        "observations": n,
        "arithmetic_monthly_mean": _finite_or_none(monthly_mean),
        "monthly_sample_volatility": _finite_or_none(monthly_vol) if monthly_vol is not None else None,
        "annualized_arithmetic_mean": _finite_or_none(annual_mean),
        "annualized_sample_volatility": _finite_or_none(annual_vol) if annual_vol is not None else None,
        "conditional_ann_mean_to_vol": (
            _finite_or_none(mean_to_vol) if mean_to_vol is not None else None
        ),
        "positive_interval_rate": _finite_or_none(float(np.mean(array > 0.0))),
        "newey_west_lag4_t_stat": newey_west_mean_t(array, lag=4),
    }


def _verify_sources(
    repo_root: Path,
    expected_hashes: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(expected_hashes.items()):
        path = repo_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"source hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        evidence[relative] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return evidence


def _validate_manifests(repo_root: Path, source_evidence: Mapping[str, Any]) -> tuple[dict, dict, dict, list[dict]]:
    v4_complete = _json(repo_root / V4_ROOT / "COMPLETE.json")
    protocol = _json(repo_root / V4_ROOT / "protocol.json")
    parent_complete = _json(repo_root / PARENT_ROOT / "COMPLETE.json")
    coverage = _json(repo_root / PARENT_ROOT / "coverage.json")

    if v4_complete.get("kind") != "factor_action_accounting_continuation_v1":
        raise ValueError("unexpected V4 COMPLETE kind")
    if protocol.get("trial_id") is not None and v4_complete.get("trial_id") != protocol.get("trial_id"):
        raise ValueError("V4 COMPLETE/protocol trial mismatch")
    for name in ("forward.csv", "protocol.json"):
        relative = (V4_ROOT / name).as_posix()
        recorded = v4_complete.get("files", {}).get(name, {}).get("sha256")
        actual = source_evidence[relative]["sha256"]
        if recorded != actual:
            raise ValueError(f"V4 COMPLETE does not bind {name} to its pinned hash")

    parent_hash = source_evidence[(PARENT_ROOT / "COMPLETE.json").as_posix()]["sha256"]
    if protocol.get("parent_result_hash") != parent_hash:
        raise ValueError("V4 protocol does not bind the accepted parent result")
    if protocol.get("parent_trial_id") != parent_complete.get("trial_id"):
        raise ValueError("V4 protocol/parent COMPLETE trial mismatch")
    recorded_coverage = parent_complete.get("files", {}).get("coverage.json", {}).get("sha256")
    actual_coverage = source_evidence[(PARENT_ROOT / "coverage.json").as_posix()]["sha256"]
    if recorded_coverage != actual_coverage:
        raise ValueError("parent COMPLETE does not bind coverage.json to its pinned hash")

    outcome_known = protocol.get("outcome_known_status", {})
    if outcome_known.get("fresh_or_independent_evidence") is not False:
        raise ValueError("V4 protocol must label results as non-fresh evidence")
    if outcome_known.get("performance_exposed_locally") is not True:
        raise ValueError("V4 protocol must disclose locally exposed performance")
    reporting = protocol.get("reporting", {})
    if reporting.get("investment_ready") is not False:
        raise ValueError("V4 protocol must remain non-investment-ready")
    if reporting.get("point_in_time_attested") is not False:
        raise ValueError("V4 protocol must remain point-in-time unattested")
    window = protocol.get("window", {})
    if window != {
        "start": "1999-03-31",
        "end": "2022-12-30",
        "holdout_excluded": "2023-2025",
    }:
        raise ValueError(f"unexpected V4 research window: {window!r}")
    if not isinstance(coverage, list):
        raise ValueError("parent coverage.json must be a list")
    return v4_complete, protocol, parent_complete, coverage


def _catalog_maps(coverage: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    ids = [str(item.get("jkp_id", "")) for item in coverage]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("coverage catalog contains missing or duplicate JKP ids")
    supported = {
        str(item["jkp_id"]): item
        for item in coverage
        if item.get("implementation_status") == "provider_reconstruction"
    }
    blocked = [item for item in coverage if item.get("implementation_status") == "blocked"]
    if len(supported) + len(blocked) != len(coverage):
        statuses = sorted({str(item.get("implementation_status")) for item in coverage})
        raise ValueError(f"unexpected coverage implementation statuses: {statuses}")
    return supported, blocked


def _load_and_validate_forward(
    path: Path,
    protocol: Mapping[str, Any],
    supported: Mapping[str, dict],
    expected_shape: Mapping[str, int] | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = {
        "scenario_id",
        "scenario_label",
        "imputed_sensitivity",
        "date",
        "factor",
        "entry_date",
        "exit_date",
        "return_complete",
        "known_component_pnl",
        "missing_names",
    }
    if set(frame.columns) != expected_columns:
        raise ValueError(f"unexpected forward.csv columns: {list(frame.columns)!r}")
    if frame.empty:
        raise ValueError("forward.csv is empty")
    if set(frame["scenario_id"].astype(str)) != set(EXPECTED_SCENARIOS):
        raise ValueError("forward.csv scenario set differs from the fixed protocol")
    protocol_scenarios = {
        str(item["id"]) for item in protocol.get("scenarios_fixed_before_execution", [])
    }
    if protocol_scenarios != set(EXPECTED_SCENARIOS):
        raise ValueError("protocol scenario set differs from the fixed summary contract")
    forward_factors = set(frame["factor"].astype(str))
    if forward_factors != set(supported) | {"MODEL"}:
        missing = sorted((set(supported) | {"MODEL"}) - forward_factors)
        extra = sorted(forward_factors - (set(supported) | {"MODEL"}))
        raise ValueError(f"forward factor mismatch; missing={missing}, extra={extra}")
    key = ["scenario_id", "factor", "date", "entry_date", "exit_date"]
    if frame.duplicated(key).any():
        raise ValueError("forward.csv contains duplicate scenario/factor intervals")
    frame["missing_names"] = pd.to_numeric(frame["missing_names"], errors="raise")
    if (frame["missing_names"] < 0).any():
        raise ValueError("forward.csv has negative missing_names")
    finite_return = np.isfinite(pd.to_numeric(frame["return_complete"], errors="coerce"))
    complete_from_names = frame["missing_names"].eq(0).to_numpy()
    if not np.array_equal(finite_return, complete_from_names):
        raise ValueError("return_complete must be finite exactly when missing_names is zero")
    frame["is_complete"] = finite_return
    frame["return_complete"] = pd.to_numeric(frame["return_complete"], errors="coerce")

    start = str(protocol["window"]["start"])
    end = str(protocol["window"]["end"])
    if str(frame["date"].min()) < start or str(frame["exit_date"].max()) > end:
        raise ValueError("forward.csv reaches outside the pre-holdout protocol window")
    grouped_sizes = frame.groupby(["scenario_id", "factor"], sort=False).size()
    if grouped_sizes.nunique() != 1:
        raise ValueError("scenario/factor series have unequal interval counts")

    if expected_shape is not None:
        observed = {
            "catalog_factors": len(supported),  # adjusted below by caller
            "supported_factors": len(supported),
            "forward_factors": len(forward_factors),
            "intervals_per_factor_scenario": int(grouped_sizes.iloc[0]),
            "forward_rows": int(len(frame)),
        }
        for field in (
            "supported_factors",
            "forward_factors",
            "intervals_per_factor_scenario",
            "forward_rows",
        ):
            if observed[field] != int(expected_shape[field]):
                raise ValueError(
                    f"production shape mismatch for {field}: "
                    f"expected {expected_shape[field]}, got {observed[field]}"
                )
    return frame


def _factor_metadata(factor: str, supported: Mapping[str, dict]) -> dict[str, Any]:
    if factor == "MODEL":
        return {
            "factor_type": "equal_theme_model",
            "theme": "All active themes",
            "description": "Frozen equal-active-theme ideal model",
            "data_family": "model",
            "sign": None,
            "exact_JKP_replication": False,
        }
    item = supported[factor]
    return {
        "factor_type": "provider_reconstruction",
        "theme": item.get("theme"),
        "description": item.get("description"),
        "data_family": item.get("data_family"),
        "sign": item.get("sign"),
        "exact_JKP_replication": bool(item.get("exact_JKP_replication", False)),
    }


def _factor_metrics(frame: pd.DataFrame, supported: Mapping[str, dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenario_type = {
        EXPECTED_SCENARIOS[0]: "primary_ex_post_reconstruction",
        EXPECTED_SCENARIOS[1]: "imputation_sensitivity",
        EXPECTED_SCENARIOS[2]: "imputation_sensitivity",
    }
    for (scenario, factor), group in frame.groupby(["scenario_id", "factor"], sort=True):
        complete = group.loc[group["is_complete"]].sort_values(["entry_date", "exit_date"])
        stats = summarize_returns(complete["return_complete"].to_numpy(dtype=float))
        metadata = _factor_metadata(str(factor), supported)
        rows.append(
            {
                "scenario_id": str(scenario),
                "scenario_type": scenario_type[str(scenario)],
                "factor": str(factor),
                "factor_type": metadata["factor_type"],
                "theme": metadata["theme"],
                "description": metadata["description"],
                "data_family": metadata["data_family"],
                "jkp_sign": metadata["sign"],
                "exact_jkp_replication": metadata["exact_JKP_replication"],
                "complete_intervals": int(len(complete)),
                "total_intervals": int(len(group)),
                "incomplete_intervals": int(len(group) - len(complete)),
                "coverage_rate": float(len(complete) / len(group)),
                "first_complete_entry_date": (
                    str(complete["entry_date"].min()) if len(complete) else None
                ),
                "last_complete_exit_date": (
                    str(complete["exit_date"].max()) if len(complete) else None
                ),
                **{key: value for key, value in stats.items() if key != "observations"},
                "statistical_status": "outcome_known_conditional_non_executable",
            }
        )
    return rows


def _scenario_sensitivity(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    interval_key = ["date", "entry_date", "exit_date"]
    strict_id, zero_id, last_id = EXPECTED_SCENARIOS
    for factor, factor_frame in frame.groupby("factor", sort=True):
        series: dict[str, pd.Series] = {}
        complete_counts: dict[str, int] = {}
        for scenario in EXPECTED_SCENARIOS:
            part = factor_frame.loc[
                (factor_frame["scenario_id"] == scenario) & factor_frame["is_complete"],
                interval_key + ["return_complete"],
            ].copy()
            index = pd.MultiIndex.from_frame(part[interval_key])
            series[scenario] = pd.Series(
                part["return_complete"].to_numpy(dtype=float), index=index, dtype=float
            )
            complete_counts[scenario] = int(len(part))
        common = series[strict_id].index
        for scenario in (zero_id, last_id):
            common = common.intersection(series[scenario].index, sort=False)
        aligned = {scenario: series[scenario].loc[common].to_numpy() for scenario in EXPECTED_SCENARIOS}
        common_n = int(len(common))

        def annual_mean(values: np.ndarray) -> float | None:
            return _finite_or_none(12.0 * float(values.mean())) if len(values) else None

        strict_mean = annual_mean(aligned[strict_id])
        zero_mean = annual_mean(aligned[zero_id])
        last_mean = annual_mean(aligned[last_id])

        def difference(left: float | None, right: float | None) -> float | None:
            return _finite_or_none(left - right) if left is not None and right is not None else None

        def max_abs(left: np.ndarray, right: np.ndarray) -> float | None:
            return _finite_or_none(float(np.max(np.abs(left - right)))) if len(left) else None

        result.append(
            {
                "factor": str(factor),
                "all_scenarios_common_intervals": common_n,
                "strict_complete_intervals": complete_counts[strict_id],
                "zero_insolvency_complete_intervals": complete_counts[zero_id],
                "last_mark_complete_intervals": complete_counts[last_id],
                "strict_annualized_arithmetic_mean_on_common": strict_mean,
                "zero_insolvency_annualized_arithmetic_mean_on_common": zero_mean,
                "last_mark_annualized_arithmetic_mean_on_common": last_mean,
                "zero_minus_strict_annualized_arithmetic_mean": difference(zero_mean, strict_mean),
                "last_mark_minus_strict_annualized_arithmetic_mean": difference(last_mean, strict_mean),
                "zero_minus_strict_max_abs_interval_difference": max_abs(
                    aligned[zero_id], aligned[strict_id]
                ),
                "last_mark_minus_strict_max_abs_interval_difference": max_abs(
                    aligned[last_id], aligned[strict_id]
                ),
                "statistical_status": "outcome_known_common_complete_intervals_only",
            }
        )
    return result


def _coverage_by_year(frame: pd.DataFrame) -> list[dict[str, Any]]:
    working = frame.copy()
    working["formation_year"] = working["date"].astype(str).str.slice(0, 4).astype(int)
    rows: list[dict[str, Any]] = []
    for (scenario, year), group in working.groupby(["scenario_id", "formation_year"], sort=True):
        total = int(len(group))
        complete = int(group["is_complete"].sum())
        model = group.loc[group["factor"] == "MODEL"]
        factor_status = group.groupby("factor")["is_complete"].agg(["any", "all"])
        rows.append(
            {
                "scenario_id": str(scenario),
                "formation_year": int(year),
                "complete_factor_intervals": complete,
                "total_factor_intervals": total,
                "coverage_rate": float(complete / total),
                "model_complete_intervals": int(model["is_complete"].sum()),
                "model_total_intervals": int(len(model)),
                "model_coverage_rate": float(model["is_complete"].mean()),
                "factors_with_any_complete_interval": int(factor_status["any"].sum()),
                "factors_fully_complete": int(factor_status["all"].sum()),
                "factors_total": int(len(factor_status)),
            }
        )
    return rows


def _blocked_rows(blocked: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "jkp_id": str(item["jkp_id"]),
            "theme": item.get("theme"),
            "description": item.get("description"),
            "data_family": item.get("data_family"),
            "blocked_reason": item.get("blocked_reason"),
            "reference": item.get("reference"),
            "implementation_status": item.get("implementation_status"),
            "exact_jkp_replication": bool(item.get("exact_JKP_replication", False)),
        }
        for item in sorted(blocked, key=lambda value: str(value["jkp_id"]))
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if value is None:
                rendered[field] = ""
            elif isinstance(value, bool):
                rendered[field] = "true" if value else "false"
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite output value in {field}")
                rendered[field] = format(value, ".17g")
            else:
                rendered[field] = value
        writer.writerow(rendered)
    return stream.getvalue().encode("utf-8")


def _model_summaries(factor_metrics: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = (
        "complete_intervals",
        "total_intervals",
        "coverage_rate",
        "annualized_arithmetic_mean",
        "annualized_sample_volatility",
        "conditional_ann_mean_to_vol",
        "positive_interval_rate",
        "newey_west_lag4_t_stat",
    )
    return {
        str(row["scenario_id"]): {field: row[field] for field in fields}
        for row in factor_metrics
        if row["factor"] == "MODEL"
    }


def _readme(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    models = summary["model_diagnostics"]
    lines = [
        "# JKP action-accounting V4 factor summary (derived V2)",
        "",
        "This immutable local bundle describes the accepted V4 `forward.csv`. It is outcome-known development evidence, not fresh or independent validation. The statistics are conditional diagnostics for uncapped ideal factor/model portfolios; they are not an executable account backtest, investment-ready evidence, or point-in-time-attested performance.",
        "",
        "## Scope and rules",
        "",
        "- Window: 1999-03-31 through 2022-12-30. The 2023-2025 holdout remains untouched.",
        "- Complete interval: `return_complete` is finite and `missing_names == 0`.",
        "- Missing intervals are omitted, never filled with zero.",
        "- Annual arithmetic mean is 12 times the complete-interval monthly mean; annual volatility is sample monthly volatility times sqrt(12).",
        "- `Annual mean/vol` is the annual arithmetic mean divided by annual sample volatility. It is a conditional descriptive ratio; no cash-return convention, costs, or executable account is defined.",
        "- The mean t statistic uses a Bartlett Newey-West estimator with lag 4 and a `1/n` autocovariance denominator.",
        "- No compounded return, CAGR, or drawdown is reported because no factor or MODEL series has complete active-interval coverage.",
        "- Sensitivity differences use only intervals complete in all three fixed scenarios.",
        "",
        "## Coverage",
        "",
        f"The parent catalog has {summary['factor_catalog']['catalog_count']} JKP characteristics: {summary['factor_catalog']['supported_provider_reconstructions']} provisional provider reconstructions and {summary['factor_catalog']['blocked_count']} blocked characteristics. `MODEL` is reported separately, for {summary['factor_catalog']['forward_series_count']} forward series total.",
        "",
        "| Scenario | Complete factor intervals | Total | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scenario in EXPECTED_SCENARIOS:
        item = coverage[scenario]
        lines.append(
            f"| `{scenario}` | {item['complete_rows']:,} | {item['total_rows']:,} | {item['coverage_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen MODEL diagnostics on complete intervals",
            "",
            "| Scenario | n / 285 | Annual arithmetic mean | Annual sample vol | Annual mean/vol | Positive intervals | NW(4) t |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scenario in EXPECTED_SCENARIOS:
        item = models[scenario]
        lines.append(
            "| `{}` | {} / {} | {:.3%} | {:.3%} | {:.3f} | {:.2%} | {:.3f} |".format(
                scenario,
                item["complete_intervals"],
                item["total_intervals"],
                item["annualized_arithmetic_mean"],
                item["annualized_sample_volatility"],
                item["conditional_ann_mean_to_vol"],
                item["positive_interval_rate"],
                item["newey_west_lag4_t_stat"],
            )
        )
    lines.extend(
        [
            "",
            "## Superseded artifact",
            "",
            f"This V2 bundle supersedes `{SUPERSEDED_BUNDLE['path']}` ({SUPERSEDED_BUNDLE['bundle_id']}). V1 is retained byte-for-byte as a historical derived artifact. {SUPERSEDED_BUNDLE['reason']}",
            "",
            "## Files",
            "",
            "- `factor_metrics.csv`: one row per scenario and supported factor/MODEL, using complete intervals only.",
            "- `scenario_sensitivity.csv`: fixed-scenario comparisons on their common complete intervals.",
            "- `coverage_by_year.csv`: formation-year completeness counts; it contains no return aggregation.",
            "- `blocked_factors.csv`: all catalog characteristics unavailable from the existing providers.",
            "- `summary.json`: machine-readable scope, methods, counts, and frozen MODEL diagnostics.",
            "- `manifest.json`: pinned source evidence and byte hashes for every other file in this bundle.",
            "",
            "Provider reconstructions are approximations rather than exact JKP/WRDS replications. Current identity and industry metadata are backcast, action timing is ex-post and point-in-time unattested, and historical stock-loan availability and realized borrow fees are unavailable. Descriptive differences in this bundle must not be used to tune the untouched holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_no_path_metrics(
    factor_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
) -> None:
    forbidden = ("compound", "cagr", "drawdown", "cumulative")
    for field in set(FACTOR_METRIC_FIELDS) | set(SENSITIVITY_FIELDS):
        if any(token in field.lower() for token in forbidden):
            raise ValueError(f"path-dependent output field is forbidden: {field}")
    for collection in (factor_rows, sensitivity_rows):
        for row in collection:
            unexpected = set(row) - (set(FACTOR_METRIC_FIELDS) | set(SENSITIVITY_FIELDS))
            if unexpected:
                raise ValueError(f"unexpected output fields: {sorted(unexpected)}")


def _write_immutable(output_dir: Path, files: Mapping[str, bytes]) -> str:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {output_dir}")
        existing = {path.name for path in output_dir.iterdir() if path.is_file()}
        if existing != set(files):
            raise FileExistsError(
                f"immutable output has a different file set: {sorted(existing)}"
            )
        differences = [
            name for name, payload in files.items() if (output_dir / name).read_bytes() != payload
        ]
        if differences:
            raise FileExistsError(
                "immutable output differs from regenerated bytes: " + ", ".join(differences)
            )
        return "already_exists_identical"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        staging.replace(output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return "created"


def generate_bundle(
    repo_root: Path,
    output_dir: Path | None = None,
    *,
    expected_hashes: Mapping[str, str] = PINNED_SOURCE_HASHES,
    expected_shape: Mapping[str, int] | None = EXPECTED_PRODUCTION_SHAPE,
) -> dict[str, Any]:
    """Validate accepted sources, derive diagnostics, and write once."""

    repo_root = repo_root.resolve()
    destination = (output_dir or (repo_root / DEFAULT_OUTPUT)).resolve()
    source_evidence = _verify_sources(repo_root, expected_hashes)
    v4_complete, protocol, parent_complete, coverage = _validate_manifests(
        repo_root, source_evidence
    )
    supported, blocked = _catalog_maps(coverage)
    if expected_shape is not None:
        observed_catalog = {
            "catalog_factors": len(coverage),
            "supported_factors": len(supported),
            "blocked_factors": len(blocked),
        }
        for field, observed in observed_catalog.items():
            if observed != int(expected_shape[field]):
                raise ValueError(
                    f"production shape mismatch for {field}: "
                    f"expected {expected_shape[field]}, got {observed}"
                )

    frame = _load_and_validate_forward(
        repo_root / V4_ROOT / "forward.csv",
        protocol,
        supported,
        expected_shape,
    )
    factor_rows = _factor_metrics(frame, supported)
    sensitivity_rows = _scenario_sensitivity(frame)
    coverage_rows = _coverage_by_year(frame)
    blocked_rows = _blocked_rows(blocked)
    _validate_no_path_metrics(factor_rows, sensitivity_rows)

    bundle_identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "source_sha256": {
            path: item["sha256"] for path, item in sorted(source_evidence.items())
        },
        "method": {
            "complete_interval": "finite return_complete and missing_names == 0",
            "missing_returns": "omit; never zero-fill",
            "annual_mean": "12 * arithmetic monthly mean",
            "annual_volatility": "sqrt(12) * monthly sample standard deviation",
            "conditional_ann_mean_to_vol": (
                "annual arithmetic mean / annual sample volatility; conditional descriptive "
                "ratio only because no cash-return convention, costs, or executable account is defined"
            ),
            "mean_t_statistic": "Bartlett Newey-West lag 4; autocovariance denominator n",
            "sensitivity": "common complete intervals across all three fixed scenarios",
            "path_statistics": "omitted because every series has incomplete active intervals",
        },
    }
    bundle_id = _sha256_bytes(_canonical_json(bundle_identity))
    scenario_coverage = {
        scenario: {
            "complete_rows": int(group["is_complete"].sum()),
            "total_rows": int(len(group)),
            "coverage_rate": float(group["is_complete"].mean()),
        }
        for scenario, group in frame.groupby("scenario_id", sort=True)
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "bundle_id": bundle_id,
        "scientific_status": {
            "outcome_known_development": True,
            "fresh_or_independent_evidence": False,
            "conditional_non_executable_statistics": True,
            "investment_ready": False,
            "point_in_time_attested": False,
            "selection_or_tuning_permitted": False,
        },
        "window": {
            "start": protocol["window"]["start"],
            "end": protocol["window"]["end"],
            "holdout": "2023-2025",
            "holdout_evaluations": 0,
            "holdout_status": "untouched",
        },
        "source": {
            "v4_trial_id": v4_complete["trial_id"],
            "v4_result_sha256": source_evidence[(V4_ROOT / "COMPLETE.json").as_posix()]["sha256"],
            "parent_trial_id": parent_complete["trial_id"],
            "parent_result_sha256": source_evidence[(PARENT_ROOT / "COMPLETE.json").as_posix()]["sha256"],
            "forward_sha256": source_evidence[(V4_ROOT / "forward.csv").as_posix()]["sha256"],
        },
        "supersedes": SUPERSEDED_BUNDLE,
        "factor_catalog": {
            "catalog_count": len(coverage),
            "supported_provider_reconstructions": len(supported),
            "blocked_count": len(blocked),
            "forward_series_count": len(supported) + 1,
            "exact_jkp_replications": 0,
        },
        "method": bundle_identity["method"],
        "omitted_path_metrics": [
            "compounded return",
            "CAGR",
            "drawdown",
        ],
        "omission_reason": (
            "No supported factor or MODEL has complete active-interval coverage; "
            "path statistics would require an unstated missing-return assumption."
        ),
        "coverage": scenario_coverage,
        "model_diagnostics": _model_summaries(factor_rows),
        "limitations": [
            "Every statistic conditions on fully reconstructed intervals and is not a full-history track record.",
            "Ideal factor and MODEL portfolios are uncapped analytical directions before executable constraints, costs and realized borrow availability.",
            "Provider reconstructions are provisional approximations rather than exact JKP/WRDS replications.",
            "Current identity and industry metadata are backcast and action timing is ex-post point-in-time unattested.",
            "The outcome-known V4 reproduction cannot support strategy selection or holdout tuning.",
        ],
    }

    files: dict[str, bytes] = {
        "README.md": _readme(summary).encode("utf-8"),
        "summary.json": _pretty_json(summary),
        "factor_metrics.csv": _csv_bytes(factor_rows, FACTOR_METRIC_FIELDS),
        "scenario_sensitivity.csv": _csv_bytes(sensitivity_rows, SENSITIVITY_FIELDS),
        "coverage_by_year.csv": _csv_bytes(coverage_rows, COVERAGE_FIELDS),
        "blocked_factors.csv": _csv_bytes(blocked_rows, BLOCKED_FIELDS),
    }
    output_evidence = {
        name: {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
        for name, payload in sorted(files.items())
    }
    generator_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{BUNDLE_KIND}_manifest",
        "bundle_id": bundle_id,
        "immutable": True,
        "generator": {
            "path": "scripts/summarize_jkp_factor_results.py",
            "sha256": _sha256_file(generator_path),
        },
        "sources": source_evidence,
        "files": output_evidence,
        "validation": {
            "source_hashes_verified": True,
            "complete_interval_rule_verified": True,
            "catalog_forward_alignment_verified": True,
            "holdout_evaluations": 0,
            "path_dependent_metrics_emitted": False,
        },
    }
    files["manifest.json"] = _pretty_json(manifest)
    disposition = _write_immutable(destination, files)
    return {
        "status": disposition,
        "output_dir": str(destination),
        "bundle_id": bundle_id,
        "files": sorted(files),
        "model_diagnostics": summary["model_diagnostics"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="AlphaLab repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Destination (default: {DEFAULT_OUTPUT.as_posix()}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output
    if output is not None and not output.is_absolute():
        output = args.repo_root / output
    result = generate_bundle(args.repo_root, output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
