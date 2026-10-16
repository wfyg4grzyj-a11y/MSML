from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_jkp_factor_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_jkp_factor_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fixture_repo(root: Path) -> dict[str, str]:
    v4 = root / summary_module.V4_ROOT
    parent = root / summary_module.PARENT_ROOT
    v4.mkdir(parents=True)
    parent.mkdir(parents=True)

    protocol = {
        "trial_id": "v4-trial",
        "parent_trial_id": "parent-trial",
        "parent_result_hash": "PARENT_HASH_PLACEHOLDER",
        "window": {
            "start": "1999-03-31",
            "end": "2022-12-30",
            "holdout_excluded": "2023-2025",
        },
        "scenarios_fixed_before_execution": [
            {"id": value} for value in summary_module.EXPECTED_SCENARIOS
        ],
        "outcome_known_status": {
            "fresh_or_independent_evidence": False,
            "performance_exposed_locally": True,
        },
        "reporting": {"investment_ready": False, "point_in_time_attested": False},
    }
    coverage = [
        {
            "jkp_id": "factor_a",
            "signal_id": "provider_a",
            "theme": "Quality",
            "sign": 1,
            "description": "Synthetic supported factor",
            "reference": "Test",
            "implementation_status": "provider_reconstruction",
            "data_family": "fundamental",
            "blocked_reason": None,
            "exact_JKP_replication": False,
        },
        {
            "jkp_id": "factor_b",
            "signal_id": "blocked_b",
            "theme": "Investment",
            "sign": -1,
            "description": "Synthetic blocked factor",
            "reference": "Test",
            "implementation_status": "blocked",
            "data_family": "unavailable",
            "blocked_reason": "missing source field",
            "exact_JKP_replication": False,
        },
    ]
    _write_json(parent / "coverage.json", coverage)
    parent_complete = {
        "kind": "large_cap_factor_study_v1",
        "trial_id": "parent-trial",
        "files": {
            "coverage.json": {
                "sha256": _hash(parent / "coverage.json"),
                "bytes": (parent / "coverage.json").stat().st_size,
            }
        },
    }
    _write_json(parent / "COMPLETE.json", parent_complete)
    protocol["parent_result_hash"] = _hash(parent / "COMPLETE.json")
    _write_json(v4 / "protocol.json", protocol)

    dates = [
        ("1999-03-31", "1999-04-01", "1999-05-03"),
        ("1999-04-30", "1999-05-03", "1999-06-01"),
        ("2000-01-31", "2000-02-01", "2000-03-01"),
    ]
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(summary_module.EXPECTED_SCENARIOS):
        for factor_index, factor in enumerate(("factor_a", "MODEL")):
            for interval, (date, entry, exit_) in enumerate(dates):
                complete = not (scenario_index == 0 and interval == 1)
                rows.append(
                    {
                        "scenario_id": scenario,
                        "scenario_label": scenario,
                        "imputed_sensitivity": scenario_index > 0,
                        "date": date,
                        "factor": factor,
                        "entry_date": entry,
                        "exit_date": exit_,
                        "return_complete": (
                            0.01 * (interval + 1 + factor_index + scenario_index)
                            if complete
                            else ""
                        ),
                        "known_component_pnl": 0.0,
                        "missing_names": 0 if complete else 1,
                    }
                )
    with (v4 / "forward.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    v4_complete = {
        "kind": "factor_action_accounting_continuation_v1",
        "trial_id": "v4-trial",
        "files": {
            "forward.csv": {
                "sha256": _hash(v4 / "forward.csv"),
                "bytes": (v4 / "forward.csv").stat().st_size,
            },
            "protocol.json": {
                "sha256": _hash(v4 / "protocol.json"),
                "bytes": (v4 / "protocol.json").stat().st_size,
            },
        },
    }
    _write_json(v4 / "COMPLETE.json", v4_complete)
    return {
        relative: _hash(root / relative)
        for relative in (
            (summary_module.V4_ROOT / "COMPLETE.json").as_posix(),
            (summary_module.V4_ROOT / "forward.csv").as_posix(),
            (summary_module.V4_ROOT / "protocol.json").as_posix(),
            (summary_module.PARENT_ROOT / "COMPLETE.json").as_posix(),
            (summary_module.PARENT_ROOT / "coverage.json").as_posix(),
        )
    }


def test_arithmetic_diagnostics_and_newey_west_formula():
    values = np.array([0.02, -0.01, 0.03, 0.0, 0.01, 0.04])
    result = summary_module.summarize_returns(values)
    assert result["observations"] == 6
    assert result["annualized_arithmetic_mean"] == pytest.approx(12 * values.mean())
    assert result["annualized_sample_volatility"] == pytest.approx(
        math.sqrt(12) * values.std(ddof=1)
    )
    assert result["conditional_ann_mean_to_vol"] == pytest.approx(
        math.sqrt(12) * values.mean() / values.std(ddof=1)
    )

    centered = values - values.mean()
    long_run_variance = np.dot(centered, centered) / len(values)
    for lag in range(1, 5):
        long_run_variance += (
            2 * (1 - lag / 5) * np.dot(centered[lag:], centered[:-lag]) / len(values)
        )
    expected_t = values.mean() / math.sqrt(long_run_variance / len(values))
    assert result["newey_west_lag4_t_stat"] == pytest.approx(expected_t)


def test_bundle_is_hash_bound_non_path_dependent_and_immutable(tmp_path: Path):
    hashes = _fixture_repo(tmp_path)
    output = tmp_path / "derived"
    first = summary_module.generate_bundle(
        tmp_path,
        output,
        expected_hashes=hashes,
        expected_shape=None,
    )
    assert first["status"] == "created"
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "summary.json",
        "factor_metrics.csv",
        "scenario_sensitivity.csv",
        "coverage_by_year.csv",
        "blocked_factors.csv",
        "manifest.json",
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["scientific_status"] == {
        "conditional_non_executable_statistics": True,
        "fresh_or_independent_evidence": False,
        "investment_ready": False,
        "outcome_known_development": True,
        "point_in_time_attested": False,
        "selection_or_tuning_permitted": False,
    }
    assert summary["window"]["holdout_status"] == "untouched"
    assert summary["window"]["holdout_evaluations"] == 0
    assert summary["schema_version"] == 2
    assert summary["supersedes"]["bundle_id"] == summary_module.SUPERSEDED_BUNDLE["bundle_id"]
    assert "conditional_ann_mean_to_vol" in summary["model_diagnostics"]["strict_standard_only"]
    assert not any("sharpe" in key.lower() for key in summary["model_diagnostics"]["strict_standard_only"])
    assert all(
        b"sharpe" not in path.read_bytes().lower()
        for path in output.iterdir()
        if path.is_file()
    )
    with (output / "factor_metrics.csv").open(encoding="utf-8", newline="") as handle:
        factor_rows = list(csv.DictReader(handle))
    assert len(factor_rows) == 3 * 2
    assert {row["complete_intervals"] for row in factor_rows if row["scenario_id"] == "strict_standard_only"} == {"2"}
    forbidden = ("compound", "cagr", "drawdown", "cumulative")
    assert not any(
        token in column.lower()
        for column in factor_rows[0]
        for token in forbidden
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, evidence in manifest["files"].items():
        assert _hash(output / name) == evidence["sha256"]
        assert (output / name).stat().st_size == evidence["bytes"]

    second = summary_module.generate_bundle(
        tmp_path,
        output,
        expected_hashes=hashes,
        expected_shape=None,
    )
    assert second["status"] == "already_exists_identical"
    (output / "README.md").write_text("mutated", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable output differs"):
        summary_module.generate_bundle(
            tmp_path,
            output,
            expected_hashes=hashes,
            expected_shape=None,
        )


def test_source_hash_mismatch_fails_before_writing(tmp_path: Path):
    hashes = _fixture_repo(tmp_path)
    forward_key = (summary_module.V4_ROOT / "forward.csv").as_posix()
    hashes[forward_key] = "0" * 64
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="source hash mismatch"):
        summary_module.generate_bundle(
            tmp_path,
            output,
            expected_hashes=hashes,
            expected_shape=None,
        )
    assert not output.exists()
