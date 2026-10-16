from pathlib import Path
from types import SimpleNamespace
from datetime import date, timedelta
import csv
import copy
import os
import numpy as np

import pytest
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.bundles import artifact
from alpha_lab.research.contracts import Artifact, ResearchError, file_hash, parse, read_json
from alpha_lab.research.factor_workflow import PlatformSpec, numerical_environment
from alpha_lab.research.registry import Registry, initialize_registry
from alpha_lab.research import registry as registry_module
from alpha_lab.research.runtime import implementation_identity
from alpha_lab.research.signal_contracts import Family, Hypothesis
from alpha_lab.research import factor_replay_workflow as replay


def parent_policy():
    return {
        "requested_window": {"start": "1999-03-31", "end": "2022-12-30"},
        "hard_limits": {
            "gross": 1.0,
            "name": 0.05,
            "turnover": 2.0,
            "abs_net": 0.01,
            "abs_beta": 0.01,
            "tolerance": 1e-6,
            "industry_abs_net": 0.005,
            "industry_abs_beta": 0.005,
            "annual_vol": 0.1,
        },
        "execution": {
            "starting_nav_usd": 1_000_000.0,
            "one_way_commission_bps": 1.0,
            "one_way_slippage_bps": 5.0,
            "annual_hypothetical_borrow": 0.03,
            "day_count": "ACT365F",
            "cash_credit": 0.0,
            "cash_debit": 0.05,
            "participation": 0.01,
        },
        "risk": {
            "lookback_sessions": 252,
            "minimum_observations": 126,
            "annualization": 252,
            "diagonal_shrinkage": 0.3,
            "variance_floor": 1e-8,
        },
        "construction": {
            "minimum_industry_names": 20,
            "ideal_factor_annual_volatility": 0.1,
            "ideal_theme_annual_volatility": 0.1,
            "ideal_model_annual_volatility": 0.1,
        },
        "expected_returns": {"risk_aversion_gamma": 5.0, "horizon_sessions": 21},
        "optimization": {
            "inner_gross": 0.8,
            "inner_name": 0.04,
            "inner_turnover": 1.8,
            "annual_vol_cap": 0.1,
            "horizon_borrow_days": 30,
            "solver_time_limit_seconds": 30,
        },
    }


def spaced_dates(start, end, count):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    span = (last - first).days
    return [(first + timedelta(days=index * span // (count - 1))).isoformat() for index in range(count)]


def parent_spec():
    return PlatformSpec(
        schema_version=5,
        family_id="frozen-parent-family",
        hypothesis_id="frozen-parent-hypothesis",
        title="Synthetic frozen parent",
        bundle_id="1" * 64,
        fundamental_sha256="2" * 64,
        catalog_sha256="3" * 64,
        policy=parent_policy(),
        implementation=implementation_identity(),
        environment=numerical_environment(),
        seed=42,
        threads=1,
        timeout_seconds=60,
        memory_limit_mb=2048,
        max_output_bytes=10_000_000,
    )


def completed_parent(registry_path, parent_dir, spec, result_hash):
    initialize_registry(registry_path)
    family = Family(
        family_id=spec.family_id,
        premise="Synthetic completed parent factor study for replay workflow tests",
        max_unique_trials=1,
        max_attempts_per_trial=2,
    )
    hypothesis = Hypothesis(
        hypothesis_id=spec.hypothesis_id,
        family_id=spec.family_id,
        premise="A frozen synthetic parent result exists before the accounting continuation",
        tests=["completed immutable evidence"],
    )
    with Registry(registry_path) as registry:
        registry.register(family, hypothesis, spec)
        attempt = registry.begin(spec, parent_dir, access_bounds=("1999-03-31", "2022-12-30"))
        registry.finish(
            attempt,
            complete={
                "directory": str(parent_dir.resolve()),
                "trial_id": spec.semantic_id(),
                "attempt_id": attempt,
                "hypothesis_id": spec.hypothesis_id,
                "result_hash": result_hash,
                "evidence_kind": "development_signal_diagnostics_published",
            },
        )
    return attempt


def resolved_fixture(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    parent = tmp_path / "parent"
    actions = tmp_path / "actions"
    for path in (inputs, parent, actions):
        path.mkdir()
    real_uuid4 = registry_module.uuid.uuid4
    parent_issued = False

    def fixed_parent_uuid():
        nonlocal parent_issued
        if not parent_issued:
            parent_issued = True
            return SimpleNamespace(hex=replay.PARENT_ATTEMPT_ID)
        return real_uuid4()

    monkeypatch.setattr(registry_module.uuid, "uuid4", fixed_parent_uuid)
    monkeypatch.setattr(PlatformSpec, "semantic_id", lambda self: replay.PARENT_TRIAL_ID)
    source = parent_spec()
    result_hash = replay.PARENT_RESULT_HASH
    registry = tmp_path / "registry.sqlite"
    attempt = completed_parent(registry, parent, source, result_hash)
    parent_artifacts = {
        "spec.json": Artifact(sha256="5" * 64, bytes=101),
        "run.json": Artifact(sha256="6" * 64, bytes=202),
        "outcome.json": Artifact(sha256="7" * 64, bytes=303),
    }
    parent_binding = {
        "directory": str(parent.resolve()),
        "trial_id": source.semantic_id(),
        "attempt_id": attempt,
        "result_hash": result_hash,
        "complete_sha256": result_hash,
        "spec_sha256": "8" * 64,
        "artifacts": parent_artifacts,
        "spec": source.model_dump(),
    }
    parent_binding["complete"] = {
        "kind": "large_cap_factor_study_v1",
        "trial_id": source.semantic_id(),
        "files": {name: value.model_dump() for name, value in parent_artifacts.items()},
    }
    parent_binding["formation_dates"] = spaced_dates("1999-03-31", "2022-12-30", 286)
    evaluation_dates = spaced_dates("1999-03-31", "2022-12-30", 5979)
    parent_binding["formation_dates"] = [
        evaluation_dates[index * (len(evaluation_dates) - 1) // 285]
        for index in range(286)
    ]
    parent_binding["factor_ids"] = [f"factor_{index:03d}" for index in range(117)] + ["MODEL"]
    action_files = {
        "packages.json": Artifact(sha256="9" * 64, bytes=404),
        "supplemental-quotes.parquet": Artifact(sha256="a" * 64, bytes=505),
    }
    action_binding = {
        "directory": str(actions.resolve()),
        "bundle_id": "b" * 64,
        "parent_bundle_id": source.bundle_id,
        "start": "1999-03-31",
        "end": "2022-12-30",
        "manifest_sha256": "c" * 64,
        "files": action_files,
    }
    action_binding["manifest"] = {
        "schema_version": 1,
        "kind": "factor_action_bundle",
        "methodology": "sharadar_terminal_action_normalization_v1",
        "bundle_id": action_binding["bundle_id"],
        "parent_bundle_id": source.bundle_id,
        "start": action_binding["start"],
        "end": action_binding["end"],
        "point_in_time_attested": False,
        "settlement_timing_attested": False,
        "counts": {"packages": 2},
        "files": {name: value.model_dump() for name, value in action_files.items()},
    }
    action_binding["bundle_id"] = replay.digest(
        {key: value for key, value in action_binding["manifest"].items() if key != "bundle_id"}
    )
    action_binding["manifest"]["bundle_id"] = action_binding["bundle_id"]
    action_binding["packages"] = [
        replay.ActionPackagePin(
            package_id="e" * 64,
            parent_security_id="1001",
            action_date=evaluation_dates[1000],
            classification="cash",
            insolvency=False,
        ),
        replay.ActionPackagePin(
            package_id="f" * 64,
            parent_security_id="1002",
            action_date=evaluation_dates[2000],
            classification="unresolved",
            insolvency=False,
        ),
    ]
    input_binding = {
        "directory": str(inputs.resolve()),
        "bundle_id": source.bundle_id,
        "manifest_sha256": "d" * 64,
        "evaluation_dates": evaluation_dates,
        "evaluation_indices": list(range(313, 313 + len(evaluation_dates))),
    }
    protocol = {
        "family_id": replay.FAMILY_ID,
        "hypothesis_id": replay.HYPOTHESIS_ID,
        "parent_trial_id": replay.PARENT_TRIAL_ID,
        "parent_attempt_id": replay.PARENT_ATTEMPT_ID,
        "parent_result_hash": replay.PARENT_RESULT_HASH,
        "window": {"start": "1999-03-31", "end": "2022-12-30"},
        "scenarios_fixed_before_execution": [{"id": value} for value in replay.SCENARIO_IDS],
        "budget": {
            "unique_trials": 1,
            "attempts_max": 2,
            "scenarios_in_one_attempt": 3,
            "LLM_calls": 0,
            "provider_calls": 0,
        },
        "publication_union_schema_correction": {
            "expected_reproduction_counts": {
                "stock_claims_created_by_scenario": {value: 0 for value in replay.SCENARIO_IDS},
                "stock_claim_deliveries_by_scenario": {value: 0 for value in replay.SCENARIO_IDS},
                "daily_controls_by_scenario": {value: 0 for value in replay.SCENARIO_IDS},
                "reduce_only_action_exit_rows_by_scenario": {value: 0 for value in replay.SCENARIO_IDS},
            }
        },
    }
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, protocol)
    monkeypatch.setattr(replay, "PROTOCOL_SHA256", file_hash(protocol_path))
    monkeypatch.setattr(replay, "_parent_binding", lambda path: parent_binding)
    monkeypatch.setattr(replay, "_action_binding", lambda path: action_binding)
    monkeypatch.setattr(replay, "_input_binding", lambda path: input_binding)
    spec = replay.resolve_replay(
        inputs,
        parent,
        actions,
        {
            "family_id": replay.FAMILY_ID,
            "hypothesis_id": replay.HYPOTHESIS_ID,
            "title": "Synthetic frozen accounting continuation",
            "seed": 42,
            "threads": 1,
            "timeout_seconds": 5400,
            "memory_limit_mb": 12000,
            "max_output_bytes": 2_500_000_000,
        },
    )
    spec_path = tmp_path / "replay-spec.json"
    atomic_json(spec_path, spec.model_dump())
    assert replay.register_replay(registry, spec) == spec.semantic_id()
    return SimpleNamespace(
        inputs=inputs,
        parent=parent,
        actions=actions,
        registry=registry,
        spec=spec,
        spec_path=spec_path,
        parent_binding=parent_binding,
        action_binding=action_binding,
        input_binding=input_binding,
        protocol_path=protocol_path,
    )


def worker_artifacts(output, spec):
    output.mkdir(exist_ok=True)
    daily_rows = [
        "date,scenario_id,valuation_resolved,return_observed,nav_observed_usd,nav_estimate_usd,"
        "missing_held_marks_or_claims,borrow_usd,funding_usd,trade_cost_usd,held_names,raw_claims,hard_violations"
    ]
    month_rows = ["date,scenario_id,optimizer_status"]
    forward_rows = [
        "date,scenario_id,factor,entry_date,exit_date,return_complete,known_component_pnl,missing_names"
    ]
    for scenario_id in spec.scenarios:
        for index, day in enumerate(spec.evaluation_dates):
            observed_return = "" if index == 0 else "0.0"
            daily_rows.append(
                f"{day},{scenario_id},True,{observed_return},1000000.0,1000000.0,0,0.0,0.0,0.0,0,0,"
            )
        for day in spec.formation_dates:
            month_rows.append(f"{day},{scenario_id},optimal")
        for interval in spec.forward_intervals:
            for factor in spec.forward_factor_ids:
                forward_rows.append(
                    f"{interval.formation_date},{scenario_id},{factor},{interval.entry_date},"
                    f"{interval.exit_date},0.0,0.0,0"
                )
    (output / "daily.csv").write_text("\n".join(daily_rows) + "\n", encoding="utf-8")
    (output / "months.csv").write_text("\n".join(month_rows) + "\n", encoding="utf-8")
    (output / "forward.csv").write_text("\n".join(forward_rows) + "\n", encoding="utf-8")
    atomic_json(
        output / "action-journal.json",
        {
            "scenario_ids": list(spec.scenarios),
            "standard_action_accounting_common": True,
            "entries": [
                {
                    "scenario_id": scenario_id,
                    "date": package.action_date,
                    "package_id": package.package_id,
                    "parent_security_id": package.parent_security_id,
                    "classification": package.classification,
                    "status": "not_held",
                    "imputed": False,
                }
                for scenario_id in spec.scenarios
                for package in spec.action_packages
            ],
        },
    )
    atomic_json(
        output / "outcomes.json",
        {
            "start": spec.start,
            "end": spec.end,
            "parent_trial_id": spec.parent_trial_id,
            "parent_result_hash": spec.parent_result_hash,
            "action_bundle_id": spec.action_bundle_id,
            "scenario_ids": list(spec.scenarios),
            "methodology": "frozen_factor_tape_action_aware_replay_v1",
            "standard_action_accounting_common": True,
            "scenarios": {
                scenario_id: {
                    "imputed": scenario_id != "strict_standard_only",
                    "sessions": len(spec.evaluation_dates),
                    "formations": len(spec.formation_dates),
                    "imputed_events": 0,
                    "unresolved_sessions": 0,
                    "first_unresolved_date": None,
                    "unresolved_action_events": 0,
                    "unresolved_terminal_names": 0,
                    "unresolved_terminal_signed_marked_exposure_usd": 0.0,
                    "unresolved_stock_claim_names": 0,
                    "unresolved_stock_claim_signed_marked_exposure_usd": 0.0,
                    "last_fully_reconstructed_date": spec.end,
                    "last_fully_reconstructed_nav_usd": 1000000.0,
                    "continuous_resolved_segments": [
                        {"start": spec.start, "end": spec.end, "sessions": len(spec.evaluation_dates)}
                    ],
                    "final_nav_observed_usd": 1000000.0,
                    "final_nav_estimate_usd": 1000000.0,
                    "complete_path_return": 0.0,
                    "borrow_usd": 0.0,
                    "funding_usd": 0.0,
                    "trade_cost_usd": 0.0,
                    "final_held_names": 0,
                    "final_raw_claims": 0,
                }
                for scenario_id in spec.scenarios
            },
            "point_in_time_attested": False,
            "investment_ready": False,
            "holdout_evaluations": 0,
            "provider_calls": 0,
            "llm_calls": 0,
        },
    )
    np.savez_compressed(
        output / "trade-vectors.npz",
        scenario_ids=np.array(spec.scenarios),
        security_ids=np.array(["1"]),
        scenario_index=np.arange(len(spec.scenarios), dtype=np.int16),
        session_index=np.full(len(spec.scenarios), spec.evaluation_indices[1], dtype=np.int32),
        security_index=np.zeros(len(spec.scenarios), dtype=np.int32),
        requested_delta=np.ones(len(spec.scenarios), dtype=float),
        filled_delta=np.ones(len(spec.scenarios), dtype=float),
        kind=np.array(["frozen_monthly_alpha"] * len(spec.scenarios)),
        batch_status=np.array(["filled"] * len(spec.scenarios)),
        batch_violations=np.array([""] * len(spec.scenarios)),
        batch_turnover=np.full(len(spec.scenarios), 0.001, dtype=float),
    )
    atomic_json(
        output / "runtime.json",
        {
            "implementation": spec.implementation,
            "environment": spec.environment,
            "seed": spec.seed,
            "threads": spec.threads,
            "provider_keys_present": False,
            "proxy_present": False,
            "isolation": "owned_credential_filtered_process_not_filesystem_or_network_sandbox",
        },
    )


def set_daily_hard_violations(output, updates):
    path = output / "daily.csv"
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    matched = set()
    for row in rows:
        key = (row["scenario_id"], row["date"])
        if key in updates:
            row["hard_violations"] = updates[key]
            matched.add(key)
    assert matched == set(updates)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_trade_vectors_with_risk_rows(output, spec, risk_rows):
    security_ids = ["1"]
    for row in risk_rows:
        if row["security_id"] not in security_ids:
            security_ids.append(row["security_id"])
    scenario_index = list(range(len(spec.scenarios)))
    session_index = [spec.evaluation_indices[1]] * len(spec.scenarios)
    security_index = [0] * len(spec.scenarios)
    requested_delta = [1.0] * len(spec.scenarios)
    filled_delta = [1.0] * len(spec.scenarios)
    kind = ["frozen_monthly_alpha"] * len(spec.scenarios)
    batch_status = ["filled"] * len(spec.scenarios)
    batch_violations = [""] * len(spec.scenarios)
    batch_turnover = [0.001] * len(spec.scenarios)
    setup_holdings = {}
    for row in risk_rows:
        key = (row["scenario_id"], row["security_id"])
        setup_holdings.setdefault(key, row.get("pre_trade_units", -row["requested_delta"]))
    for (scenario_id, security_id), units in setup_holdings.items():
        if abs(units) <= 1e-12:
            continue
        scenario_index.append(spec.scenarios.index(scenario_id))
        session_index.append(spec.evaluation_indices[0])
        security_index.append(security_ids.index(security_id))
        requested_delta.append(units)
        filled_delta.append(units)
        kind.append("frozen_monthly_alpha")
        batch_status.append("filled")
        batch_violations.append("")
        batch_turnover.append(0.001)
    for row in risk_rows:
        scenario_index.append(spec.scenarios.index(row["scenario_id"]))
        session_index.append(row["session_index"])
        security_index.append(security_ids.index(row["security_id"]))
        requested_delta.append(row["requested_delta"])
        filled_delta.append(row["filled_delta"])
        kind.append(row.get("kind", "unsupported_risk_exit"))
        batch_status.append(row.get("batch_status", "reduce_only_risk_orphan_exit"))
        batch_violations.append("")
        batch_turnover.append(float("nan"))
    np.savez_compressed(
        output / "trade-vectors.npz",
        scenario_ids=np.array(spec.scenarios),
        security_ids=np.array(security_ids),
        scenario_index=np.array(scenario_index, dtype=np.int16),
        session_index=np.array(session_index, dtype=np.int32),
        security_index=np.array(security_index, dtype=np.int32),
        requested_delta=np.array(requested_delta, dtype=float),
        filled_delta=np.array(filled_delta, dtype=float),
        kind=np.array(kind),
        batch_status=np.array(batch_status),
        batch_violations=np.array(batch_violations),
        batch_turnover=np.array(batch_turnover, dtype=float),
    )


def risk_exit_control(scenario_id, day, security_ids):
    return {
        "scenario_id": scenario_id,
        "date": day,
        "package_id": None,
        "parent_security_id": None,
        "classification": "daily_control",
        "status": "scheduled_unsupported_risk_exit",
        "violations": ["unavailable_held_risk"],
        "target_names": len(security_ids),
        "target_security_ids": sorted(security_ids),
        "imputed": False,
    }


def test_schema6_registration_owned_publication_inspection_and_duplicate_guard(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    captured = {}

    class CompleteProcess:
        pid = os.getpid()

        def poll(self):
            return 0

        def wait(self):
            return 0

        def close(self):
            pass

    def start(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        output = Path(command[command.index("--output") + 1])
        worker_artifacts(output, fixture.spec)
        return CompleteProcess()

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://synthetic.invalid")
    monkeypatch.setattr(replay, "start_process", start)
    destination = tmp_path / "run"
    result = replay.run_replay(
        fixture.inputs,
        fixture.parent,
        fixture.actions,
        fixture.spec_path,
        destination,
        fixture.registry,
    )
    assert result["artifacts_verified"] and result["effective_status"] == "completed"
    assert result["parent_trial_id"] == fixture.spec.parent_trial_id
    assert "OPENAI_API_KEY" not in captured["env"] and "HTTPS_PROXY" not in captured["env"]
    assert captured["command"][0:4] == [
        os.sys.executable,
        "-s",
        "-P",
        "-m",
    ]
    before = {path.name: (file_hash(path), path.stat().st_mtime_ns) for path in destination.iterdir() if path.is_file()}
    inspected = replay.inspect_replay(destination)
    assert inspected["result_hash"] == result["result_hash"]
    assert read_json(destination / "protocol.json") == fixture.spec.protocol
    validation = read_json(destination / "validation.json")
    assert validation["protocol_sha256"] == fixture.spec.protocol_artifact.sha256
    assert validation["expected"]["forward_factors_per_interval"] == 118
    assert validation["observed"]["terminal_dispositions"] == len(fixture.spec.action_packages) * 3
    assert before == {path.name: (file_hash(path), path.stat().st_mtime_ns) for path in destination.iterdir() if path.is_file()}
    with Registry(fixture.registry, readonly=True) as registry:
        snapshot = registry.snapshot()
    assert len(snapshot["attempts"]) == 2
    assert snapshot["attempts"][-1]["status"] == "completed"
    registration = next(row for row in snapshot["registrations"] if row["trial"] == fixture.spec.semantic_id())
    assert registration["parent"] == fixture.spec.parent_trial_id
    assert snapshot["exposures"][-1]["kind"] == replay.EVIDENCE_KIND
    with pytest.raises(ResearchError, match="active|completed"):
        replay.run_replay(
            fixture.inputs,
            fixture.parent,
            fixture.actions,
            fixture.spec_path,
            tmp_path / "duplicate",
            fixture.registry,
        )
    assert not (tmp_path / "duplicate").exists()
    marker = read_json(destination / "COMPLETE.json")
    marker["files"].pop("daily.csv")
    atomic_json(destination / "COMPLETE.json", marker)
    with pytest.raises(ResearchError, match="inventory"):
        replay.inspect_replay(destination)


def test_replay_spec_freezes_all_scenarios_parent_costs_and_action_bundle(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    assert fixture.spec.schema_version == 6
    assert fixture.spec.parent_artifacts == fixture.parent_binding["artifacts"]
    assert fixture.spec.action_files == fixture.action_binding["files"]
    assert fixture.spec.action_manifest == fixture.action_binding["manifest"]
    assert fixture.spec.protocol_artifact.sha256 == file_hash(fixture.protocol_path)
    assert len(fixture.spec.forward_factor_ids) == 118
    assert len(fixture.spec.forward_intervals) == 285
    assert len(fixture.spec.action_packages) == 2
    assert tuple(fixture.spec.scenarios) == replay.SCENARIO_IDS
    assert fixture.spec.hypothesis_id == "jkp-fixed-model-action-continuation-v4"
    assert fixture.spec.scenarios == [
        "strict_standard_only",
        "imputed_zero_insolvency_last_mark_other",
        "imputed_last_mark_all_unresolved",
    ]
    altered = fixture.spec.model_dump()
    altered["scenarios"] = altered["scenarios"][:2]
    with pytest.raises(ResearchError):
        parse(replay.ReplaySpec, altered)
    altered = fixture.spec.model_dump()
    altered["max_output_bytes"] -= 1
    with pytest.raises(ResearchError):
        parse(replay.ReplaySpec, altered)
    unexpected = tmp_path / "unexpected-worker-output"
    worker_artifacts(unexpected, fixture.spec)
    (unexpected / "nested").mkdir()
    with pytest.raises(ResearchError, match="unexpected artifact"):
        replay._validate_results(unexpected, fixture.spec, worker_only=True)
    corrupt = tmp_path / "corrupt-worker-output"
    worker_artifacts(corrupt, fixture.spec)
    (corrupt / "trade-vectors.npz").write_bytes(b"not-an-npz")
    with pytest.raises(ResearchError, match="trade-vector"):
        replay._validate_results(corrupt, fixture.spec, worker_only=True)
    changed = dict(fixture.action_binding)
    changed["manifest_sha256"] = "e" * 64
    monkeypatch.setattr(replay, "_action_binding", lambda path: changed)
    with pytest.raises(ResearchError, match="changed"):
        replay.validate_replay_spec(fixture.spec, fixture.inputs, fixture.parent, fixture.actions)


def test_v8_protocol_binding_and_zero_terminal_cutoff_are_frozen(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    protocol = copy.deepcopy(fixture.spec.protocol)
    protocol["family_id"] = replay.TERMINAL_PARENT_PUBLICATION_FAMILY_ID
    protocol["hypothesis_id"] = replay.TERMINAL_PARENT_PUBLICATION_HYPOTHESIS_ID
    counts = protocol["publication_union_schema_correction"]["expected_reproduction_counts"]
    counts["reduce_only_risk_orphan_exit_rows_by_scenario"] = {
        scenario_id: 0 for scenario_id in fixture.spec.scenarios
    }
    counts["terminal_parent_trade_rows_at_or_after_by_scenario"] = {
        scenario_id: 0 for scenario_id in fixture.spec.scenarios
    }
    atomic_json(fixture.protocol_path, protocol)
    monkeypatch.setattr(
        replay, "TERMINAL_PARENT_PUBLICATION_PROTOCOL_SHA256", file_hash(fixture.protocol_path)
    )

    v8 = replay.resolve_replay(
        fixture.inputs,
        fixture.parent,
        fixture.actions,
        {
            "family_id": replay.TERMINAL_PARENT_PUBLICATION_FAMILY_ID,
            "hypothesis_id": replay.TERMINAL_PARENT_PUBLICATION_HYPOTHESIS_ID,
            "title": "Synthetic terminal-parent publication correction",
            "seed": 42,
            "threads": 1,
            "timeout_seconds": 5400,
            "memory_limit_mb": 12000,
            "max_output_bytes": 2_500_000_000,
        },
    )
    assert v8.protocol_artifact.sha256 == file_hash(fixture.protocol_path)
    assert replay._expected_reproduction_counts(v8)[
        "terminal_parent_trade_rows_at_or_after_by_scenario"
    ] == {scenario_id: 0 for scenario_id in fixture.spec.scenarios}

    changed = v8.model_dump()
    changed["protocol"]["publication_union_schema_correction"]["expected_reproduction_counts"][
        "terminal_parent_trade_rows_at_or_after_by_scenario"
    ][fixture.spec.scenarios[0]] = 1
    with pytest.raises(ResearchError, match="Invalid ReplaySpec"):
        parse(replay.ReplaySpec, changed)


def test_protocol_and_exact_result_reconciliation_reject_truncation_and_noop_artifacts(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    original_protocol = fixture.protocol_path.read_bytes()
    changed_protocol = read_json(fixture.protocol_path)
    changed_protocol["reporting"] = {"mutated": True}
    atomic_json(fixture.protocol_path, changed_protocol)
    with pytest.raises(ResearchError, match="protocol"):
        replay.validate_replay_spec(fixture.spec, fixture.inputs, fixture.parent, fixture.actions)
    fixture.protocol_path.write_bytes(original_protocol)

    output = tmp_path / "exact-output"
    worker_artifacts(output, fixture.spec)
    forward_path = output / "forward.csv"
    complete_forward = forward_path.read_text(encoding="utf-8")
    forward_path.write_text("\n".join(complete_forward.splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ResearchError, match="Cartesian"):
        replay._validate_results(output, fixture.spec, worker_only=True)
    forward_path.write_text(complete_forward, encoding="utf-8")

    journal_path = output / "action-journal.json"
    journal = read_json(journal_path)
    removed = journal["entries"].pop()
    atomic_json(journal_path, journal)
    with pytest.raises(ResearchError, match="omitted"):
        replay._validate_results(output, fixture.spec, worker_only=True)
    journal["entries"].append(removed)
    atomic_json(journal_path, journal)

    outcomes_path = output / "outcomes.json"
    outcomes = read_json(outcomes_path)
    outcomes["scenarios"][fixture.spec.scenarios[1]]["imputed_events"] = 1
    atomic_json(outcomes_path, outcomes)
    with pytest.raises(ResearchError, match="reconcile"):
        replay._validate_results(output, fixture.spec, worker_only=True)
    outcomes["scenarios"][fixture.spec.scenarios[1]]["imputed_events"] = 0
    atomic_json(outcomes_path, outcomes)

    vectors_path = output / "trade-vectors.npz"
    np.savez_compressed(
        vectors_path,
        scenario_ids=np.array(fixture.spec.scenarios),
        security_ids=np.array([], dtype="U1"),
        scenario_index=np.array([], dtype=np.int16),
        session_index=np.array([], dtype=np.int32),
        security_index=np.array([], dtype=np.int32),
        requested_delta=np.array([], dtype=float),
        filled_delta=np.array([], dtype=float),
        kind=np.array([], dtype="U1"),
        batch_status=np.array([], dtype="U1"),
        batch_violations=np.array([], dtype="U1"),
        batch_turnover=np.array([], dtype=float),
    )
    with pytest.raises(ResearchError, match="no-op"):
        replay._validate_results(output, fixture.spec, worker_only=True)


def test_publication_validates_daily_control_as_a_distinct_auxiliary_event(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / "daily-control-output"
    worker_artifacts(output, fixture.spec)
    first_day = fixture.spec.evaluation_dates[0]
    final_day = fixture.spec.evaluation_dates[-1]
    controls = [
        {
            "scenario_id": fixture.spec.scenarios[0],
            "date": first_day,
            "package_id": None,
            "parent_security_id": None,
            "classification": "daily_control",
            "status": "blocked_unavailable_held_risk_or_nav",
            "violations": ["unavailable_held_risk"],
            "imputed": False,
        },
        {
            "scenario_id": fixture.spec.scenarios[1],
            "date": first_day,
            "package_id": None,
            "parent_security_id": None,
            "classification": "daily_control",
            "status": "scheduled_next_close",
            "violations": ["industry_beta"],
            "uniform_scale": 0.5,
            "target_names": 2,
            "imputed": False,
        },
        {
            "scenario_id": fixture.spec.scenarios[2],
            "date": final_day,
            "package_id": None,
            "parent_security_id": None,
            "classification": "daily_control",
            "status": "right_censored",
            "violations": ["industry_beta"],
            "imputed": False,
        },
    ]
    updates = {
        (fixture.spec.scenarios[0], first_day): "unavailable_held_risk",
        (fixture.spec.scenarios[1], first_day): "industry_beta",
        (fixture.spec.scenarios[2], final_day): "industry_beta",
    }
    set_daily_hard_violations(output, updates)
    journal_path = output / "action-journal.json"
    journal = read_json(journal_path)
    journal["entries"].extend(controls)
    atomic_json(journal_path, journal)
    protocol = copy.deepcopy(fixture.spec.protocol)
    protocol["publication_union_schema_correction"]["expected_reproduction_counts"][
        "daily_controls_by_scenario"
    ] = {scenario_id: 1 for scenario_id in fixture.spec.scenarios}
    controlled_spec = fixture.spec.model_copy(update={"protocol": protocol})

    _, validation = replay._validate_results(output, controlled_spec, worker_only=True)
    assert validation["observed"]["daily_control_entries"] == 3
    assert validation["observed"]["delivery_entries"] == 0
    assert validation["observed"]["daily_controls_by_status"] == {
        "scheduled_next_close": 1,
        "blocked_unavailable_held_risk_or_nav": 1,
        "right_censored": 1,
    }

    daily_rows = {
        scenario_id: [
            {
                "date": day,
                "hard_violations": tuple(
                    updates.get((scenario_id, day), "").split(",")
                    if updates.get((scenario_id, day))
                    else ()
                ),
            }
            for day in fixture.spec.evaluation_dates
        ]
        for scenario_id in fixture.spec.scenarios
    }
    journal["entries"] = journal["entries"][:-1]
    atomic_json(journal_path, journal)
    with pytest.raises(ResearchError, match="omitted a daily-control"):
        replay._validate_action_journal(output, controlled_spec, daily_rows)

    journal["entries"].append(dict(controls[2]))
    journal["entries"].append(dict(controls[0]))
    atomic_json(journal_path, journal)
    with pytest.raises(ResearchError, match="invalid or repeated daily control"):
        replay._validate_action_journal(output, controlled_spec, daily_rows)

    journal["entries"] = journal["entries"][:-1]
    journal["entries"][-2]["uniform_scale"] = 1.0
    atomic_json(journal_path, journal)
    with pytest.raises(ResearchError, match="scheduled daily control"):
        replay._validate_action_journal(output, controlled_spec, daily_rows)

    journal["entries"][-2] = dict(controls[1])
    journal["entries"][-2]["classification"] = "unknown_auxiliary"
    atomic_json(journal_path, journal)
    with pytest.raises(ResearchError, match="unknown auxiliary"):
        replay._validate_action_journal(output, controlled_spec, daily_rows)


def test_publication_rejects_an_omitted_outcome_known_delivery(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / "missing-delivery-output"
    worker_artifacts(output, fixture.spec)
    protocol = copy.deepcopy(fixture.spec.protocol)
    counts = protocol["publication_union_schema_correction"]["expected_reproduction_counts"]
    counts["stock_claims_created_by_scenario"][fixture.spec.scenarios[0]] = 1
    counts["stock_claim_deliveries_by_scenario"][fixture.spec.scenarios[0]] = 1
    altered_spec = fixture.spec.model_copy(update={"protocol": protocol})
    daily_rows = {
        scenario_id: [
            {"date": day, "hard_violations": ()} for day in fixture.spec.evaluation_dates
        ]
        for scenario_id in fixture.spec.scenarios
    }
    with pytest.raises(ResearchError, match="outcome-known reproduction counts"):
        replay._validate_action_journal(output, altered_spec, daily_rows)


def test_v8_count_contract_requires_zero_terminal_parent_trade_rows(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    protocol = copy.deepcopy(fixture.spec.protocol)
    counts = protocol["publication_union_schema_correction"]["expected_reproduction_counts"]
    counts["reduce_only_risk_orphan_exit_rows_by_scenario"] = {
        scenario_id: 0 for scenario_id in fixture.spec.scenarios
    }
    counts["terminal_parent_trade_rows_at_or_after_by_scenario"] = {
        scenario_id: 0 for scenario_id in fixture.spec.scenarios
    }
    v8_spec = fixture.spec.model_copy(
        update={
            "family_id": replay.TERMINAL_PARENT_PUBLICATION_FAMILY_ID,
            "hypothesis_id": replay.TERMINAL_PARENT_PUBLICATION_HYPOTHESIS_ID,
            "protocol": protocol,
        }
    )
    assert replay._expected_reproduction_counts(v8_spec) == counts

    protocol = copy.deepcopy(protocol)
    protocol["publication_union_schema_correction"]["expected_reproduction_counts"].pop(
        "terminal_parent_trade_rows_at_or_after_by_scenario"
    )
    missing = v8_spec.model_copy(update={"protocol": protocol})
    with pytest.raises(ResearchError, match="invalid outcome-known reproduction counts"):
        replay._expected_reproduction_counts(missing)


def test_publication_accepts_derived_risk_orphan_control_and_validates_targets(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / "risk-orphan-control-output"
    worker_artifacts(output, fixture.spec)
    scenario_id = fixture.spec.scenarios[0]
    first_day = fixture.spec.evaluation_dates[0]
    set_daily_hard_violations(output, {(scenario_id, first_day): "unavailable_held_risk"})
    journal_path = output / "action-journal.json"
    journal = read_json(journal_path)
    journal["entries"].append(risk_exit_control(scenario_id, first_day, ["A", "B"]))
    atomic_json(journal_path, journal)
    write_trade_vectors_with_risk_rows(
        output,
        fixture.spec,
        [
            {
                "scenario_id": scenario_id,
                "session_index": fixture.spec.evaluation_indices[1],
                "security_id": security_id,
                "requested_delta": -10.0,
                "filled_delta": -5.0,
            }
            for security_id in ("A", "B")
        ],
    )
    protocol = copy.deepcopy(fixture.spec.protocol)
    expected_counts = protocol["publication_union_schema_correction"]["expected_reproduction_counts"]
    expected_counts["daily_controls_by_scenario"] = "derive_from_daily_hard_violations"
    expected_counts["reduce_only_risk_orphan_exit_rows_by_scenario"] = {
        value: 2 if value == scenario_id else 0 for value in fixture.spec.scenarios
    }
    controlled_spec = fixture.spec.model_copy(update={"protocol": protocol})
    _, validation = replay._validate_results(output, controlled_spec, worker_only=True)
    assert validation["observed"]["daily_controls_by_status"][
        "scheduled_unsupported_risk_exit"
    ] == 1
    assert validation["observed"]["trade_vectors"]["rows_by_batch_status_and_scenario"][scenario_id][
        "reduce_only_risk_orphan_exit"
    ] == 2

    journal["entries"][-1]["target_security_ids"] = ["B", "A"]
    atomic_json(journal_path, journal)
    daily_rows = {
        value: [
            {
                "date": day,
                "hard_violations": ("unavailable_held_risk",)
                if (value, day) == (scenario_id, first_day)
                else (),
            }
            for day in fixture.spec.evaluation_dates
        ]
        for value in fixture.spec.scenarios
    }
    with pytest.raises(ResearchError, match="unsupported-risk exit control"):
        replay._validate_action_journal(output, controlled_spec, daily_rows)


def test_publication_rejects_unbacked_duplicate_and_non_reduce_only_risk_trades(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / "invalid-risk-trades-output"
    worker_artifacts(output, fixture.spec)
    scenario_id = fixture.spec.scenarios[0]
    second_index = fixture.spec.evaluation_indices[1]
    first_day = fixture.spec.evaluation_dates[0]
    second_day = fixture.spec.evaluation_dates[1]
    daily_rows = {
        value: [{"valuation_resolved": True} for _ in fixture.spec.evaluation_dates]
        for value in fixture.spec.scenarios
    }
    journal_path = output / "action-journal.json"
    original_journal = read_json(journal_path)

    def set_schedules(*entries):
        journal = copy.deepcopy(original_journal)
        journal["entries"].extend(entries)
        atomic_json(journal_path, journal)

    def risk_row(**updates):
        row = {
            "scenario_id": scenario_id,
            "session_index": second_index,
            "security_id": "A",
            "requested_delta": -10.0,
            "filled_delta": -5.0,
        }
        row.update(updates)
        return row

    write_trade_vectors_with_risk_rows(output, fixture.spec, [risk_row()])
    with pytest.raises(ResearchError, match="prior scheduled control"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    set_schedules(risk_exit_control(scenario_id, second_day, ["A"]))
    with pytest.raises(ResearchError, match="prior scheduled control"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    set_schedules(risk_exit_control(scenario_id, first_day, ["A"]))
    replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    write_trade_vectors_with_risk_rows(output, fixture.spec, [risk_row(), risk_row()])
    with pytest.raises(ResearchError, match="repeat a security at one close"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    set_schedules(risk_exit_control(scenario_id, first_day, ["1"]))
    write_trade_vectors_with_risk_rows(
        output,
        fixture.spec,
        [risk_row(security_id="1")],
    )
    with pytest.raises(ResearchError, match="repeat a security at one close"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    set_schedules(risk_exit_control(scenario_id, first_day, ["A"]))

    for invalid_row in (
        risk_row(requested_delta=0.0, filled_delta=0.0),
        risk_row(filled_delta=1.0),
        risk_row(filled_delta=-11.0),
    ):
        write_trade_vectors_with_risk_rows(output, fixture.spec, [invalid_row])
        with pytest.raises(ResearchError, match="not reduce-only"):
            replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    write_trade_vectors_with_risk_rows(
        output,
        fixture.spec,
        [risk_row(requested_delta=-9.0, filled_delta=-5.0, pre_trade_units=10.0)],
    )
    with pytest.raises(ResearchError, match="does not flatten its pre-trade holding"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    write_trade_vectors_with_risk_rows(
        output,
        fixture.spec,
        [risk_row(kind="unsupported_risk_exit", batch_status="filled")],
    )
    with pytest.raises(ResearchError, match="inconsistent batch status"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)


def test_publication_requires_backed_reduce_only_action_recipient_trades(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / "invalid-action-recipient-trades-output"
    worker_artifacts(output, fixture.spec)
    scenario_id = fixture.spec.scenarios[0]
    first_day = fixture.spec.evaluation_dates[0]
    second_index = fixture.spec.evaluation_indices[1]
    third_index = fixture.spec.evaluation_indices[2]
    daily_rows = {
        value: [{"valuation_resolved": True} for _ in fixture.spec.evaluation_dates]
        for value in fixture.spec.scenarios
    }
    journal_path = output / "action-journal.json"
    original_journal = read_json(journal_path)

    def set_delivery(exit_first_index):
        journal = copy.deepcopy(original_journal)
        journal["entries"].append(
            {
                "scenario_id": scenario_id,
                "date": first_day,
                "package_id": None,
                "parent_security_id": None,
                "classification": "stock_claim_delivery",
                "status": "recipient_claim_converted",
                "recipient_security_id": "R",
                "raw_shares": 10.0,
                "analytical_units": 10.0,
                "exit_first_index": exit_first_index,
                "imputed": False,
            }
        )
        atomic_json(journal_path, journal)

    def action_row(**updates):
        row = {
            "scenario_id": scenario_id,
            "session_index": second_index,
            "security_id": "R",
            "requested_delta": -10.0,
            "filled_delta": -5.0,
            "pre_trade_units": 0.0,
            "kind": "action_recipient_exit",
            "batch_status": "reduce_only_action_exit",
        }
        row.update(updates)
        return row

    write_trade_vectors_with_risk_rows(output, fixture.spec, [action_row(pre_trade_units=10.0)])
    with pytest.raises(ResearchError, match="prior delivery"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    set_delivery(third_index)
    write_trade_vectors_with_risk_rows(output, fixture.spec, [action_row()])
    with pytest.raises(ResearchError, match="prior delivery"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    set_delivery(second_index)
    write_trade_vectors_with_risk_rows(output, fixture.spec, [action_row()])
    replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)
    write_trade_vectors_with_risk_rows(output, fixture.spec, [action_row(filled_delta=0.0)])
    replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    for invalid_row in (
        action_row(requested_delta=0.0, filled_delta=0.0),
        action_row(filled_delta=1.0),
        action_row(filled_delta=-11.0),
    ):
        write_trade_vectors_with_risk_rows(output, fixture.spec, [invalid_row])
        with pytest.raises(ResearchError, match="not reduce-only"):
            replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    write_trade_vectors_with_risk_rows(
        output,
        fixture.spec,
        [action_row(requested_delta=-9.0, filled_delta=-5.0)],
    )
    with pytest.raises(ResearchError, match="does not flatten its pre-trade holding"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)

    write_trade_vectors_with_risk_rows(
        output,
        fixture.spec,
        [action_row(batch_status="filled")],
    )
    with pytest.raises(ResearchError, match="inconsistent batch status"):
        replay._validate_trade_vectors(output / "trade-vectors.npz", fixture.spec, daily_rows)


@pytest.mark.parametrize("offset", [0, 1])
def test_v8_rejects_terminal_parent_trade_at_or_after_action_close(
    tmp_path, monkeypatch, offset
):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / f"terminal-parent-cutoff-{offset}"
    worker_artifacts(output, fixture.spec)
    path = output / "trade-vectors.npz"
    package = fixture.spec.action_packages[0]
    action_position = fixture.spec.evaluation_dates.index(package.action_date)
    action_index = fixture.spec.evaluation_indices[action_position]
    with np.load(path, allow_pickle=False) as vectors:
        arrays = {name: vectors[name].copy() for name in vectors.files}
    securities = arrays["security_ids"].tolist() + [package.parent_security_id]
    additions = {
        "scenario_index": 0,
        "session_index": action_index + offset,
        "security_index": len(securities) - 1,
        "requested_delta": 1.0,
        "filled_delta": 0.0,
        "kind": "frozen_monthly_alpha",
        "batch_status": "rejected_batch_constraints",
        "batch_violations": "gross",
        "batch_turnover": 0.001,
    }
    arrays["security_ids"] = np.array(securities)
    for name, value in additions.items():
        combined = np.append(arrays[name], value)
        arrays[name] = (
            combined.astype(arrays[name].dtype)
            if np.issubdtype(arrays[name].dtype, np.number)
            else combined
        )
    np.savez_compressed(path, **arrays)
    daily_rows = {
        scenario_id: [{"valuation_resolved": True} for _ in fixture.spec.evaluation_dates]
        for scenario_id in fixture.spec.scenarios
    }

    historical_protocol = copy.deepcopy(fixture.spec.protocol)
    historical_counts = historical_protocol["publication_union_schema_correction"][
        "expected_reproduction_counts"
    ]
    historical_counts["reduce_only_risk_orphan_exit_rows_by_scenario"] = {
        scenario_id: 0 for scenario_id in fixture.spec.scenarios
    }
    historical_counts.pop("terminal_parent_trade_rows_at_or_after_by_scenario", None)
    historical_v7 = fixture.spec.model_copy(
        update={
            "family_id": replay.FINAL_PUBLICATION_FAMILY_ID,
            "protocol": historical_protocol,
        }
    )
    validation = replay._validate_trade_vectors(path, historical_v7, daily_rows)
    assert validation["terminal_parent_trade_rows_at_or_after_by_scenario"] == {
        scenario_id: int(scenario_id == fixture.spec.scenarios[0])
        for scenario_id in fixture.spec.scenarios
    }
    _, legacy_publication_validation = replay._validate_results(
        output, historical_v7, worker_only=True
    )
    assert (
        "terminal_parent_trade_rows_at_or_after_by_scenario"
        not in legacy_publication_validation["observed"]["trade_vectors"]
    )

    corrected_v8 = fixture.spec.model_copy(
        update={"family_id": replay.TERMINAL_PARENT_PUBLICATION_FAMILY_ID}
    )
    with pytest.raises(ResearchError, match="terminal parent at or after"):
        replay._validate_trade_vectors(path, corrected_v8, daily_rows)


def test_publication_rejects_nonintegral_trade_indices(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    output = tmp_path / "nonintegral-trade-index-output"
    worker_artifacts(output, fixture.spec)
    path = output / "trade-vectors.npz"
    with np.load(path, allow_pickle=False) as vectors:
        arrays = {name: vectors[name].copy() for name in vectors.files}
    arrays["session_index"] = arrays["session_index"].astype(float)
    np.savez_compressed(path, **arrays)
    daily_rows = {
        scenario_id: [{"valuation_resolved": True} for _ in fixture.spec.evaluation_dates]
        for scenario_id in fixture.spec.scenarios
    }

    with pytest.raises(ResearchError, match="outside the registered run"):
        replay._validate_trade_vectors(path, fixture.spec, daily_rows)


def test_superseded_v6_spec_cannot_bypass_protocol_binding_at_registration(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    protocol = copy.deepcopy(fixture.spec.protocol)
    protocol["family_id"] = replay.RISK_ORPHAN_PUBLICATION_FAMILY_ID
    protocol["hypothesis_id"] = replay.RISK_ORPHAN_PUBLICATION_HYPOTHESIS_ID
    superseded = fixture.spec.model_copy(
        update={
            "family_id": replay.RISK_ORPHAN_PUBLICATION_FAMILY_ID,
            "hypothesis_id": replay.RISK_ORPHAN_PUBLICATION_HYPOTHESIS_ID,
            "protocol": protocol,
            "protocol_artifact": Artifact(
                sha256=replay.RISK_ORPHAN_PUBLICATION_PROTOCOL_SHA256,
                bytes=fixture.spec.protocol_artifact.bytes,
            ),
        }
    )

    with pytest.raises(ResearchError, match="Superseded V6 replay cannot be registered"):
        replay.register_replay(tmp_path / "never-created.sqlite", superseded)


def test_timeout_guard_terminates_owned_worker_and_records_failed_attempt(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)

    class WaitingProcess:
        pid = os.getpid()

        def __init__(self):
            self.stopped = False

        def poll(self):
            return 1 if self.stopped else None

        def terminate_tree(self):
            self.stopped = True

        def close(self):
            pass

    process = WaitingProcess()
    monkeypatch.setattr(replay, "start_process", lambda *args, **kwargs: process)
    clock = iter([0.0, 5401.0])
    monkeypatch.setattr(replay, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None))
    with pytest.raises(ResearchError, match="timeout"):
        replay.run_replay(
            fixture.inputs,
            fixture.parent,
            fixture.actions,
            fixture.spec_path,
            tmp_path / "timed-out",
            fixture.registry,
        )
    assert process.stopped
    with Registry(fixture.registry, readonly=True) as registry:
        attempt = registry.snapshot()["attempts"][-1]
    assert attempt["status"] == "failed" and attempt["failure"] == "timeout"


def test_registration_refuses_a_parent_not_matching_completed_result(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    changed = fixture.spec.model_copy(update={"parent_result_hash": "f" * 64, "parent_complete_sha256": "f" * 64})
    with pytest.raises(ResearchError):
        replay.register_replay(fixture.registry, changed)


def test_setup_failure_is_terminal_in_registry(tmp_path, monkeypatch):
    fixture = resolved_fixture(tmp_path, monkeypatch)
    original = replay.atomic_json

    def fail_run_record(path, value):
        if Path(path).name == "run.json":
            raise OSError("synthetic disk failure")
        return original(path, value)

    monkeypatch.setattr(replay, "atomic_json", fail_run_record)
    with pytest.raises(ResearchError, match="terminal run record"):
        replay.run_replay(
            fixture.inputs,
            fixture.parent,
            fixture.actions,
            fixture.spec_path,
            tmp_path / "setup-failure",
            fixture.registry,
        )
    with Registry(fixture.registry, readonly=True) as registry:
        attempt = registry.snapshot()["attempts"][-1]
    assert attempt["status"] == "failed" and attempt["failure"] == "setup_failure"
