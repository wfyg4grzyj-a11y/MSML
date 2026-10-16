"""Registered action-accounting continuation of a completed factor study.

The continuation is deliberately a new trial family.  It consumes the frozen
factor-study artifacts and a separately normalized action bundle; it never
reopens or retries the completed parent trial.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import psutil
import numpy as np
from pydantic import Field, model_validator

from alpha_lab.local_process import start_process
from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact, safe_file, verify_files
from .contracts import Artifact, Hash, Identifier, ResearchError, StrictModel, digest, file_hash, parse, read_json
from .factor_inputs import load_inputs
from .factor_workflow import PlatformSpec, inspect_platform, numerical_environment
from .registry import Registry, stamp
from .runtime import child_environment, implementation_identity
from .signal_contracts import Family, Hypothesis


FAMILY_ID = "jkp-large-cap-action-accounting-v4"
HYPOTHESIS_ID = "jkp-fixed-model-action-continuation-v4"
RISK_ORPHAN_FAMILY_ID = "jkp-large-cap-action-accounting-v5"
RISK_ORPHAN_HYPOTHESIS_ID = "jkp-fixed-model-risk-orphan-continuation-v5"
RISK_ORPHAN_PUBLICATION_FAMILY_ID = "jkp-large-cap-action-accounting-v6"
RISK_ORPHAN_PUBLICATION_HYPOTHESIS_ID = "jkp-fixed-model-risk-orphan-publication-v6"
FINAL_PUBLICATION_FAMILY_ID = "jkp-large-cap-action-accounting-v7"
FINAL_PUBLICATION_HYPOTHESIS_ID = "jkp-fixed-model-risk-orphan-publication-v7"
TERMINAL_PARENT_PUBLICATION_FAMILY_ID = "jkp-large-cap-action-accounting-v8"
TERMINAL_PARENT_PUBLICATION_HYPOTHESIS_ID = "jkp-fixed-model-terminal-parent-publication-v8"
PARENT_TRIAL_ID = "0475f5254a1977d1fbb1d924a04343236e3c64f8bfa3572888d435b4a7f232a3"
PARENT_ATTEMPT_ID = "23407743f43b45609d5d30d65d55276b"
PARENT_RESULT_HASH = "0335453b939c80be4e535bb17ba2b7c150b807d8d43b6792a1bbbbe3928a38f8"
PROTOCOL_SHA256 = "c9059b639266db94c2da36726a9b283ee45b9c1636abc1239d4ad203b5bb94d1"
RISK_ORPHAN_PROTOCOL_SHA256 = "de20e6ebef56e43872065eb8e5dab325da5908633645514afdd32ab9e4247d50"
RISK_ORPHAN_PUBLICATION_PROTOCOL_SHA256 = "0d19a13653bd4b817fc44b95c135d65b165d9f71d539c4414900f953ae4eafea"
FINAL_PUBLICATION_PROTOCOL_SHA256 = "cd274a236d18ac81b5df3cb5efbf2a3f0c1b625ae42c15440db70f2a10caa72b"
TERMINAL_PARENT_PUBLICATION_PROTOCOL_SHA256 = "c09e17c7b9171f4ccdbebcd8f7acb6412005d739d8e7fe11b63992b7e64d9c81"
SCENARIO_IDS = (
    "strict_standard_only",
    "imputed_zero_insolvency_last_mark_other",
    "imputed_last_mark_all_unresolved",
)
RESULT_FILES = {
    "daily.csv",
    "months.csv",
    "forward.csv",
    "action-journal.json",
    "outcomes.json",
    "trade-vectors.npz",
    "runtime.json",
}
PUBLICATION_FILES = RESULT_FILES | {
    "spec.json",
    "protocol.json",
    "validation.json",
    "parent-binding.json",
    "action-binding.json",
    "inputs-binding.json",
    "worker.out",
    "report.txt",
    "run.json",
    "COMPLETE.json",
}
MARKER_FILES = PUBLICATION_FILES - {"COMPLETE.json"}
RECORD_FILES = MARKER_FILES - {"run.json"}
EVIDENCE_KIND = "development_action_accounting_continuation_published"


class ParentModelInputs(StrictModel):
    bundle_id: Hash
    fundamental_sha256: Hash
    catalog_sha256: Hash


class ForwardIntervalPin(StrictModel):
    formation_date: str
    entry_date: str
    exit_date: str
    action_start_inclusive: Literal[True]
    action_end_exclusive: Literal[True]


class ActionPackagePin(StrictModel):
    package_id: Hash
    parent_security_id: Annotated[str, Field(min_length=1, max_length=128)]
    action_date: str
    classification: Literal["cash", "stock", "mixed", "unresolved"]
    insolvency: bool


class ReplaySpec(StrictModel):
    """Resolved schema-6 computation identity for one fixed replay."""

    schema_version: Literal[6]
    family_id: Literal[
        "jkp-large-cap-action-accounting-v4",
        "jkp-large-cap-action-accounting-v5",
        "jkp-large-cap-action-accounting-v6",
        "jkp-large-cap-action-accounting-v7",
        "jkp-large-cap-action-accounting-v8",
    ]
    hypothesis_id: Literal[
        "jkp-fixed-model-action-continuation-v4",
        "jkp-fixed-model-risk-orphan-continuation-v5",
        "jkp-fixed-model-risk-orphan-publication-v6",
        "jkp-fixed-model-risk-orphan-publication-v7",
        "jkp-fixed-model-terminal-parent-publication-v8",
    ]
    title: Annotated[str, Field(min_length=1, max_length=120)]
    start: Literal["1999-03-31"]
    end: Literal["2022-12-30"]
    parent_trial_id: Hash
    parent_attempt_id: Identifier
    parent_result_hash: Hash
    parent_complete_sha256: Hash
    parent_complete: dict
    parent_spec_sha256: Hash
    parent_artifacts: dict[str, Artifact]
    parent_spec: dict
    parent_model_inputs: ParentModelInputs
    protocol_artifact: Artifact
    protocol: dict
    inputs_manifest_sha256: Hash
    evaluation_dates: list[str]
    evaluation_indices: list[int]
    formation_dates: list[str]
    forward_intervals: list[ForwardIntervalPin]
    forward_factor_ids: list[str]
    action_bundle_id: Hash
    action_manifest: dict
    action_manifest_sha256: Hash
    action_files: dict[str, Artifact]
    action_packages: list[ActionPackagePin]
    scenarios: Annotated[
        list[
            Literal[
                "strict_standard_only",
                "imputed_zero_insolvency_last_mark_other",
                "imputed_last_mark_all_unresolved",
            ]
        ],
        Field(min_length=3, max_length=3),
    ]
    point_in_time_attested: Literal[False]
    investment_ready: Literal[False]
    holdout_evaluations: Literal[0]
    provider_calls: Literal[0]
    llm_calls: Literal[0]
    implementation: dict[str, Hash]
    environment: dict[str, str]
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    threads: Annotated[int, Field(ge=1, le=8)]
    timeout_seconds: Annotated[int, Field(ge=1, le=7200)]
    memory_limit_mb: Annotated[int, Field(ge=128, le=16000)]
    max_output_bytes: Annotated[int, Field(ge=1024, le=3_000_000_000)]

    @model_validator(mode="after")
    def fixed_parent_and_policy(self):
        parent = parse(PlatformSpec, self.parent_spec)
        if (
            self.parent_trial_id != PARENT_TRIAL_ID
            or self.parent_attempt_id != PARENT_ATTEMPT_ID
            or self.parent_result_hash != PARENT_RESULT_HASH
        ):
            raise ValueError("Replay parent differs from the preregistered completed JKP study")
        if parent.semantic_id() != self.parent_trial_id:
            raise ValueError("Parent trial identity differs from the pinned parent specification")
        if self.parent_result_hash != self.parent_complete_sha256:
            raise ValueError("The registry result must be the pinned parent completion marker")
        complete_files = {
            name: parse(Artifact, value)
            for name, value in self.parent_complete.get("files", {}).items()
        }
        if (
            self.parent_complete.get("kind") != "large_cap_factor_study_v1"
            or self.parent_complete.get("trial_id") != self.parent_trial_id
            or complete_files != self.parent_artifacts
        ):
            raise ValueError("Pinned parent completion marker differs from its artifact inventory")
        if self.parent_model_inputs != ParentModelInputs(
            bundle_id=parent.bundle_id,
            fundamental_sha256=parent.fundamental_sha256,
            catalog_sha256=parent.catalog_sha256,
        ):
            raise ValueError("Parent model inputs changed")
        protocol_scenarios = [row.get("id") for row in self.protocol.get("scenarios_fixed_before_execution", [])]
        protocol_budget = self.protocol.get("budget", {})
        protocol_counts = (
            self.protocol.get("publication_union_schema_correction", {})
            .get("expected_reproduction_counts", {})
        )
        expected_protocol = {
            (FAMILY_ID, HYPOTHESIS_ID): PROTOCOL_SHA256,
            (RISK_ORPHAN_FAMILY_ID, RISK_ORPHAN_HYPOTHESIS_ID): RISK_ORPHAN_PROTOCOL_SHA256,
            (
                RISK_ORPHAN_PUBLICATION_FAMILY_ID,
                RISK_ORPHAN_PUBLICATION_HYPOTHESIS_ID,
            ): RISK_ORPHAN_PUBLICATION_PROTOCOL_SHA256,
            (
                FINAL_PUBLICATION_FAMILY_ID,
                FINAL_PUBLICATION_HYPOTHESIS_ID,
            ): FINAL_PUBLICATION_PROTOCOL_SHA256,
            (
                TERMINAL_PARENT_PUBLICATION_FAMILY_ID,
                TERMINAL_PARENT_PUBLICATION_HYPOTHESIS_ID,
            ): TERMINAL_PARENT_PUBLICATION_PROTOCOL_SHA256,
        }.get((self.family_id, self.hypothesis_id))
        if (
            expected_protocol is None
            or self.protocol_artifact.sha256 != expected_protocol
            or self.protocol.get("family_id") != self.family_id
            or self.protocol.get("hypothesis_id") != self.hypothesis_id
            or self.protocol.get("parent_trial_id") != self.parent_trial_id
            or self.protocol.get("parent_attempt_id") != self.parent_attempt_id
            or self.protocol.get("parent_result_hash") != self.parent_result_hash
            or self.protocol.get("window", {}).get("start") != self.start
            or self.protocol.get("window", {}).get("end") != self.end
            or protocol_scenarios != list(SCENARIO_IDS)
            or protocol_budget.get("unique_trials") != 1
            or protocol_budget.get("attempts_max") != 2
            or protocol_budget.get("scenarios_in_one_attempt") != 3
            or protocol_budget.get("LLM_calls") != 0
            or protocol_budget.get("provider_calls") != 0
            or (
                self.family_id == TERMINAL_PARENT_PUBLICATION_FAMILY_ID
                and protocol_counts.get(
                    "terminal_parent_trade_rows_at_or_after_by_scenario"
                )
                != {scenario_id: 0 for scenario_id in SCENARIO_IDS}
            )
        ):
            raise ValueError("Continuation protocol differs from the amended preregistration")
        window = parent.policy.get("requested_window", {})
        if (window.get("start"), window.get("end")) != (self.start, self.end):
            raise ValueError("Replay must retain the parent's 1999-2022 window")
        for dates, expected_count, label in (
            (self.evaluation_dates, 5979, "evaluation"),
            (self.formation_dates, 286, "formation"),
        ):
            try:
                normalized = [date.fromisoformat(value).isoformat() for value in dates]
            except (TypeError, ValueError):
                raise ValueError(f"Replay {label} calendar contains an invalid date") from None
            if (
                len(dates) != expected_count
                or normalized != dates
                or dates != sorted(set(dates))
                or dates[0] != self.start
                or dates[-1] != self.end
            ):
                raise ValueError(f"Replay {label} calendar differs from the frozen study")
        if (
            len(self.evaluation_indices) != len(self.evaluation_dates)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.evaluation_indices)
            or any(right != left + 1 for left, right in zip(self.evaluation_indices, self.evaluation_indices[1:]))
        ):
            raise ValueError("Replay evaluation indices differ from the frozen input calendar")
        expected_intervals = []
        evaluation_lookup = {day: index for index, day in enumerate(self.evaluation_dates)}
        for formation, following in zip(self.formation_dates, self.formation_dates[1:]):
            decision = evaluation_lookup.get(formation)
            next_decision = evaluation_lookup.get(following)
            if decision is None or next_decision is None or decision + 1 >= len(self.evaluation_dates):
                raise ValueError("Replay formations are absent from the evaluation calendar")
            expected_intervals.append(
                ForwardIntervalPin(
                    formation_date=formation,
                    entry_date=self.evaluation_dates[decision + 1],
                    exit_date=self.evaluation_dates[min(next_decision + 1, len(self.evaluation_dates) - 1)],
                    action_start_inclusive=True,
                    action_end_exclusive=True,
                )
            )
        if self.forward_intervals != expected_intervals:
            raise ValueError("Replay forward intervals differ from the half-open frozen calendar")
        if (
            len(self.forward_factor_ids) != 118
            or self.forward_factor_ids[-1:] != ["MODEL"]
            or len(set(self.forward_factor_ids)) != len(self.forward_factor_ids)
            or any(not isinstance(value, str) or not value for value in self.forward_factor_ids)
        ):
            raise ValueError("Replay must retain the 117 frozen factors and MODEL")
        expected_limits = {
            "gross": 1.0,
            "name": 0.05,
            "turnover": 2.0,
            "abs_net": 0.01,
            "abs_beta": 0.01,
            "tolerance": 1e-6,
            "industry_abs_net": 0.005,
            "industry_abs_beta": 0.005,
            "annual_vol": 0.1,
        }
        expected_execution = {
            "starting_nav_usd": 1_000_000.0,
            "one_way_commission_bps": 1.0,
            "one_way_slippage_bps": 5.0,
            "annual_hypothetical_borrow": 0.03,
            "day_count": "ACT365F",
            "cash_credit": 0.0,
            "cash_debit": 0.05,
            "participation": 0.01,
        }
        if parent.policy.get("hard_limits") != expected_limits:
            raise ValueError("Replay cannot change the parent hard limits")
        if any(parent.policy.get("execution", {}).get(k) != v for k, v in expected_execution.items()):
            raise ValueError("Replay cannot change the parent execution costs or financing")
        if tuple(self.scenarios) != SCENARIO_IDS:
            raise ValueError("All three accounting scenarios must be preregistered in fixed order")
        if (
            self.seed != 42
            or self.threads != 1
            or self.timeout_seconds != 5400
            or self.memory_limit_mb != 12000
            or self.max_output_bytes != 2_500_000_000
        ):
            raise ValueError("Replay resource budget differs from the preregistered continuation")
        if not self.parent_artifacts or set(self.action_files) != {"packages.json", "supplemental-quotes.parquet"}:
            raise ValueError("Parent and normalized action artifacts must be completely pinned")
        action_files = {
            name: parse(Artifact, value)
            for name, value in self.action_manifest.get("files", {}).items()
        }
        if (
            self.action_manifest.get("schema_version") != 1
            or self.action_manifest.get("kind") != "factor_action_bundle"
            or self.action_manifest.get("methodology") != "sharadar_terminal_action_normalization_v1"
            or self.action_manifest.get("bundle_id") != self.action_bundle_id
            or self.action_manifest.get("parent_bundle_id") != self.parent_model_inputs.bundle_id
            or (self.action_manifest.get("start"), self.action_manifest.get("end")) != (self.start, self.end)
            or self.action_manifest.get("point_in_time_attested") is not False
            or self.action_manifest.get("settlement_timing_attested") is not False
            or action_files != self.action_files
            or self.action_manifest.get("bundle_id")
            != digest({key: value for key, value in self.action_manifest.items() if key != "bundle_id"})
        ):
            raise ValueError("Pinned action manifest differs from the replay identity")
        package_count = self.action_manifest.get("counts", {}).get("packages")
        package_keys = [(row.action_date, row.parent_security_id, row.package_id) for row in self.action_packages]
        evaluation_date_set = set(self.evaluation_dates)
        try:
            normalized_action_dates = [date.fromisoformat(row.action_date).isoformat() for row in self.action_packages]
        except (TypeError, ValueError):
            raise ValueError("Pinned action package contains an invalid date") from None
        if (
            isinstance(package_count, bool)
            or not isinstance(package_count, int)
            or package_count != len(self.action_packages)
            or not self.action_packages
            or len({row.package_id for row in self.action_packages}) != len(self.action_packages)
            or package_keys != sorted(package_keys)
            or any(
                normalized != row.action_date
                or row.action_date not in evaluation_date_set
                or not (self.start <= row.action_date <= self.end)
                for normalized, row in zip(normalized_action_dates, self.action_packages)
            )
        ):
            raise ValueError("Pinned action package inventory is incomplete or unordered")
        return self

    def computation(self):
        return self.model_dump(
            exclude={"family_id", "hypothesis_id", "title", "timeout_seconds", "memory_limit_mb", "max_output_bytes"}
        )

    def semantic_id(self):
        return digest(self.computation())

    def comparison_id(self):
        return digest(self.computation())


def replay_environment():
    return numerical_environment()


def _protocol_binding(action_bundle):
    root = Path(action_bundle).resolve().parent
    path = safe_file(root, "protocol.json")
    body = read_json(path)
    bound = parse(Artifact, artifact(path))
    allowed = {
        PROTOCOL_SHA256: (FAMILY_ID, HYPOTHESIS_ID),
        RISK_ORPHAN_PROTOCOL_SHA256: (RISK_ORPHAN_FAMILY_ID, RISK_ORPHAN_HYPOTHESIS_ID),
        FINAL_PUBLICATION_PROTOCOL_SHA256: (
            FINAL_PUBLICATION_FAMILY_ID,
            FINAL_PUBLICATION_HYPOTHESIS_ID,
        ),
        TERMINAL_PARENT_PUBLICATION_PROTOCOL_SHA256: (
            TERMINAL_PARENT_PUBLICATION_FAMILY_ID,
            TERMINAL_PARENT_PUBLICATION_HYPOTHESIS_ID,
        ),
    }
    superseded = {
        RISK_ORPHAN_PUBLICATION_PROTOCOL_SHA256: (
            RISK_ORPHAN_PUBLICATION_FAMILY_ID,
            RISK_ORPHAN_PUBLICATION_HYPOTHESIS_ID,
        )
    }
    if bound.sha256 in superseded:
        raise ResearchError("Continuation protocol was superseded before registration")
    if bound.sha256 not in allowed:
        raise ResearchError("Continuation protocol differs from the reviewed amendment")
    if not isinstance(body, dict):
        raise ResearchError("Continuation protocol is invalid")
    if (body.get("family_id"), body.get("hypothesis_id")) != allowed[bound.sha256]:
        raise ResearchError("Continuation protocol identity differs from its reviewed amendment")
    return {"path": str(path), "artifact": bound, "body": body}


def _input_binding(inputs):
    inputs = Path(inputs).resolve()
    bundle = load_inputs(inputs)
    manifest = safe_file(inputs, "manifest.json")
    with safe_file(inputs, "calendar.csv").open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if "date" not in set(reader.fieldnames or ()):
            raise ResearchError("Factor input calendar schema is incomplete")
        selected = [
            (index, row["date"])
            for index, row in enumerate(reader)
            if bundle["start"] <= row["date"] <= bundle["end"]
        ]
    return {
        "directory": str(inputs),
        "bundle_id": bundle["bundle_id"],
        "manifest_sha256": file_hash(manifest),
        "evaluation_dates": [day for _, day in selected],
        "evaluation_indices": [index for index, _ in selected],
    }


def _parent_binding(parent_run):
    parent_run = Path(parent_run).resolve()
    evidence = inspect_platform(parent_run)
    if not evidence["artifacts_verified"] or evidence["effective_status"] != "completed":
        raise ResearchError("Replay parent must be a completed, verified factor study")
    marker = read_json(safe_file(parent_run, "COMPLETE.json"))
    parent_spec_path = safe_file(parent_run, "spec.json")
    parent_spec = parse(PlatformSpec, read_json(parent_spec_path))
    artifacts = {name: parse(Artifact, value) for name, value in marker.get("files", {}).items()}
    if not artifacts or marker.get("trial_id") != parent_spec.semantic_id():
        raise ResearchError("Parent completion marker does not identify its factor specification")
    verify_files(parent_run, artifacts)
    with safe_file(parent_run, "months.csv").open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if "date" not in set(reader.fieldnames or ()):
            raise ResearchError("Parent formation calendar schema is incomplete")
        formation_dates = [row["date"] for row in reader]
    coverage = read_json(safe_file(parent_run, "coverage.json"))
    if not isinstance(coverage, list):
        raise ResearchError("Parent factor coverage inventory is invalid")
    factor_ids = [
        row.get("jkp_id")
        for row in coverage
        if isinstance(row, dict) and row.get("implementation_status") == "provider_reconstruction"
    ]
    if (
        len(factor_ids) != 117
        or len(set(factor_ids)) != 117
        or any(not isinstance(value, str) or not value for value in factor_ids)
    ):
        raise ResearchError("Parent factor coverage does not identify 117 frozen factors")
    return {
        "directory": str(parent_run),
        "trial_id": parent_spec.semantic_id(),
        "attempt_id": evidence["attempt_id"],
        "result_hash": evidence["result_hash"],
        "complete_sha256": file_hash(parent_run / "COMPLETE.json"),
        "complete": marker,
        "spec_sha256": file_hash(parent_spec_path),
        "artifacts": artifacts,
        "spec": parent_spec.model_dump(),
        "formation_dates": formation_dates,
        "factor_ids": factor_ids + ["MODEL"],
    }


def _action_binding(action_bundle):
    action_bundle = Path(action_bundle).resolve()
    # The canonical loader checks the semantic bundle ID, package ordering and
    # supplemental-quote schema.  Source paths are not reopened by the replay;
    # their identities are already committed inside the immutable manifest.
    from .factor_actions import load_action_bundle

    loaded = load_action_bundle(action_bundle, verify_sources=False)
    manifest_path = safe_file(action_bundle, "manifest.json")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "factor_action_bundle"
        or manifest.get("point_in_time_attested") is not False
        or manifest.get("settlement_timing_attested") is not False
        or set(manifest.get("files", {})) != {"packages.json", "supplemental-quotes.parquet"}
    ):
        raise ResearchError("Unsupported normalized factor-action bundle")
    files = {name: parse(Artifact, value) for name, value in manifest["files"].items()}
    verify_files(action_bundle, files)
    if manifest.get("bundle_id") != digest({k: v for k, v in manifest.items() if k != "bundle_id"}):
        raise ResearchError("Action bundle semantic identity changed")
    packages = loaded.get("packages") if isinstance(loaded, dict) else None
    if not isinstance(packages, list):
        raise ResearchError("Normalized factor-action package inventory is absent")
    package_pins = []
    for package in packages:
        reasons = package.get("unresolved_reasons", [])
        action_types = package.get("source_action_types", [])
        text = " ".join(
            [str(package.get("classification", "")), *map(str, reasons), *map(str, action_types)]
        ).lower()
        package_pins.append(
            ActionPackagePin(
                package_id=package["package_id"],
                parent_security_id=str(package["parent_security_id"]),
                action_date=package["action_date"],
                classification=package["classification"],
                insolvency=any(token in text for token in ("bankrupt", "insolv", "liquidat")),
            )
        )
    return {
        "directory": str(action_bundle),
        "bundle_id": manifest["bundle_id"],
        "manifest": manifest,
        "parent_bundle_id": manifest.get("parent_bundle_id"),
        "start": manifest.get("start"),
        "end": manifest.get("end"),
        "manifest_sha256": file_hash(manifest_path),
        "files": files,
        "packages": package_pins,
    }


def resolve_replay(inputs, parent_run, action_bundle, request):
    """Resolve every scientific input into one immutable schema-6 spec."""

    reserved = {
        "schema_version",
        "start",
        "end",
        "parent_trial_id",
        "parent_attempt_id",
        "parent_result_hash",
        "parent_complete_sha256",
        "parent_complete",
        "parent_spec_sha256",
        "parent_artifacts",
        "parent_spec",
        "parent_model_inputs",
        "protocol_artifact",
        "protocol",
        "inputs_manifest_sha256",
        "evaluation_dates",
        "evaluation_indices",
        "formation_dates",
        "forward_intervals",
        "forward_factor_ids",
        "action_bundle_id",
        "action_manifest",
        "action_manifest_sha256",
        "action_files",
        "action_packages",
        "scenarios",
        "point_in_time_attested",
        "investment_ready",
        "holdout_evaluations",
        "provider_calls",
        "llm_calls",
        "implementation",
        "environment",
    }
    if set(request) & reserved:
        raise ResearchError("Replay request contains reserved resolved fields")
    parent = _parent_binding(parent_run)
    action = _action_binding(action_bundle)
    source = _input_binding(inputs)
    protocol = _protocol_binding(action_bundle)
    parent_spec = parse(PlatformSpec, parent["spec"])
    if (
        parent["trial_id"] != PARENT_TRIAL_ID
        or parent["attempt_id"] != PARENT_ATTEMPT_ID
        or parent["result_hash"] != PARENT_RESULT_HASH
    ):
        raise ResearchError("Accounting continuation requires the preregistered completed JKP parent")
    start, end = parent_spec.policy["requested_window"]["start"], parent_spec.policy["requested_window"]["end"]
    if (start, end) != ("1999-03-31", "2022-12-30"):
        raise ResearchError("Accounting continuation requires the frozen 1999-2022 parent")
    if source["bundle_id"] != parent_spec.bundle_id or action["parent_bundle_id"] != parent_spec.bundle_id:
        raise ResearchError("Inputs, parent model and action bundle do not share one bundle identity")
    if (action["start"], action["end"]) != (start, end):
        raise ResearchError("Action bundle window differs from the frozen parent")
    date_lookup = {day: index for index, day in enumerate(source["evaluation_dates"])}
    forward_intervals = []
    for formation, following in zip(parent["formation_dates"], parent["formation_dates"][1:]):
        if formation not in date_lookup or following not in date_lookup:
            raise ResearchError("Parent formation is absent from the frozen input calendar")
        decision = date_lookup[formation]
        next_decision = date_lookup[following]
        forward_intervals.append(
            ForwardIntervalPin(
                formation_date=formation,
                entry_date=source["evaluation_dates"][decision + 1],
                exit_date=source["evaluation_dates"][min(next_decision + 1, len(source["evaluation_dates"]) - 1)],
                action_start_inclusive=True,
                action_end_exclusive=True,
            )
        )
    body = {
        "schema_version": 6,
        "start": start,
        "end": end,
        "parent_trial_id": parent["trial_id"],
        "parent_attempt_id": parent["attempt_id"],
        "parent_result_hash": parent["result_hash"],
        "parent_complete_sha256": parent["complete_sha256"],
        "parent_complete": parent["complete"],
        "parent_spec_sha256": parent["spec_sha256"],
        "parent_artifacts": parent["artifacts"],
        "parent_spec": parent_spec.model_dump(),
        "parent_model_inputs": {
            "bundle_id": parent_spec.bundle_id,
            "fundamental_sha256": parent_spec.fundamental_sha256,
            "catalog_sha256": parent_spec.catalog_sha256,
        },
        "protocol_artifact": protocol["artifact"],
        "protocol": protocol["body"],
        "inputs_manifest_sha256": source["manifest_sha256"],
        "evaluation_dates": source["evaluation_dates"],
        "evaluation_indices": source["evaluation_indices"],
        "formation_dates": parent["formation_dates"],
        "forward_intervals": [row.model_dump() for row in forward_intervals],
        "forward_factor_ids": parent["factor_ids"],
        "action_bundle_id": action["bundle_id"],
        "action_manifest": action["manifest"],
        "action_manifest_sha256": action["manifest_sha256"],
        "action_files": action["files"],
        "action_packages": [row.model_dump() for row in action["packages"]],
        "scenarios": list(SCENARIO_IDS),
        "point_in_time_attested": False,
        "investment_ready": False,
        "holdout_evaluations": 0,
        "provider_calls": 0,
        "llm_calls": 0,
        "implementation": implementation_identity(),
        "environment": replay_environment(),
    }
    return parse(ReplaySpec, request | body)


def validate_replay_spec(spec, inputs, parent_run, action_bundle):
    """Revalidate every pinned source immediately before and after execution."""

    spec = parse(ReplaySpec, spec.model_dump() if isinstance(spec, ReplaySpec) else spec)
    source = _input_binding(inputs)
    parent = _parent_binding(parent_run)
    action = _action_binding(action_bundle)
    protocol = _protocol_binding(action_bundle)
    expected_model = ParentModelInputs(
        bundle_id=parent["spec"]["bundle_id"],
        fundamental_sha256=parent["spec"]["fundamental_sha256"],
        catalog_sha256=parent["spec"]["catalog_sha256"],
    )
    if (
        spec.implementation != implementation_identity()
        or spec.environment != replay_environment()
        or source["bundle_id"] != spec.parent_model_inputs.bundle_id
        or source["manifest_sha256"] != spec.inputs_manifest_sha256
        or source["evaluation_dates"] != spec.evaluation_dates
        or source["evaluation_indices"] != spec.evaluation_indices
        or parent["trial_id"] != spec.parent_trial_id
        or parent["attempt_id"] != spec.parent_attempt_id
        or parent["result_hash"] != spec.parent_result_hash
        or parent["complete_sha256"] != spec.parent_complete_sha256
        or parent["complete"] != spec.parent_complete
        or parent["spec_sha256"] != spec.parent_spec_sha256
        or parent["spec"] != spec.parent_spec
        or parent["artifacts"] != spec.parent_artifacts
        or parent["formation_dates"] != spec.formation_dates
        or parent["factor_ids"] != spec.forward_factor_ids
        or expected_model != spec.parent_model_inputs
        or protocol["artifact"] != spec.protocol_artifact
        or protocol["body"] != spec.protocol
        or action["bundle_id"] != spec.action_bundle_id
        or action["manifest"] != spec.action_manifest
        or action["manifest_sha256"] != spec.action_manifest_sha256
        or action["files"] != spec.action_files
        or action["packages"] != spec.action_packages
        or action["parent_bundle_id"] != spec.parent_model_inputs.bundle_id
        or (action["start"], action["end"]) != (spec.start, spec.end)
    ):
        raise ResearchError("Replay source, parent, action bundle, code or runtime changed")
    return spec


def register_replay(registry_path, spec):
    """Register one new family/trial linked to the terminal completed parent."""

    spec = parse(ReplaySpec, spec.model_dump() if isinstance(spec, ReplaySpec) else spec)
    if (spec.family_id, spec.hypothesis_id) == (
        RISK_ORPHAN_PUBLICATION_FAMILY_ID,
        RISK_ORPHAN_PUBLICATION_HYPOTHESIS_ID,
    ):
        raise ResearchError("Superseded V6 replay cannot be registered")
    family = Family(
        family_id=spec.family_id,
        premise="One accounting-only continuation of a completed frozen JKP factor model; no signal or performance selection",
        max_unique_trials=1,
        max_attempts_per_trial=2,
    )
    hypothesis = Hypothesis(
        hypothesis_id=spec.hypothesis_id,
        family_id=spec.family_id,
        premise="Frozen model artifacts evaluated under preregistered action accounting and explicitly labelled imputation stresses",
        tests=[
            "Parent result and every upstream artifact remain immutable",
            "All three action scenarios execute together without outcome-driven selection",
            "Unknown economics stay censored in the strict primary path",
            "No holdout, provider, LLM or real-order access",
        ],
    )
    with Registry(registry_path) as registry:
        parent = registry.db.execute(
            "SELECT * FROM attempts WHERE id=? AND trial=?",
            (spec.parent_attempt_id, spec.parent_trial_id),
        ).fetchone()
        if (
            not parent
            or parent["status"] != "completed"
            or parent["result_hash"] != spec.parent_result_hash
            or parent["retry_of"] is not None
        ):
            raise ResearchError("Replay parent must be the pinned completed original attempt")
        return registry.register(family, hypothesis, spec, parent=spec.parent_trial_id)


def _settings(spec):
    return {
        "schema_version": 1,
        "protocol_sha256": spec.protocol_artifact.sha256,
        "scenario_ids": list(spec.scenarios),
        "parent_trial_id": spec.parent_trial_id,
        "parent_result_hash": spec.parent_result_hash,
        "action_bundle_id": spec.action_bundle_id,
        "start": spec.start,
        "end": spec.end,
        "evaluation_indices": list(spec.evaluation_indices),
        "forward_intervals": [row.model_dump() for row in spec.forward_intervals],
        "forward_factor_ids": list(spec.forward_factor_ids),
        "action_package_ids": [row.package_id for row in spec.action_packages],
        "policy": spec.parent_spec["policy"],
        "seed": spec.seed,
        "threads": spec.threads,
        "point_in_time_attested": False,
        "investment_ready": False,
        "holdout_evaluations": 0,
        "provider_calls": 0,
        "llm_calls": 0,
    }


def _runtime(spec):
    return {
        "implementation": spec.implementation,
        "environment": spec.environment,
        "seed": spec.seed,
        "threads": spec.threads,
        "provider_keys_present": any(k.upper().endswith("_API_KEY") for k in os.environ),
        "proxy_present": any(k.lower().endswith("_proxy") for k in os.environ),
        "isolation": "owned_credential_filtered_process_not_filesystem_or_network_sandbox",
    }


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _csv_inventory(directory, name, spec, required, unique_columns):
    by_scenario = {scenario_id: [] for scenario_id in spec.scenarios}
    seen = set()
    with safe_file(directory, name).open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not required <= set(reader.fieldnames or ()):
            raise ResearchError(f"Replay {name} schema is incomplete")
        for row in reader:
            scenario_id = row.get("scenario_id")
            day = row.get("date")
            if scenario_id not in by_scenario:
                raise ResearchError(f"Replay {name} contains an undeclared scenario")
            try:
                parsed_day = date.fromisoformat(str(day))
            except (TypeError, ValueError):
                raise ResearchError(f"Replay {name} contains an invalid date") from None
            if not (spec.start <= parsed_day.isoformat() <= spec.end):
                raise ResearchError(f"Replay {name} crossed the preregistered date boundary")
            key = tuple(row.get(column) for column in unique_columns)
            if any(value in (None, "") for value in key) or key in seen:
                raise ResearchError(f"Replay {name} contains an invalid or repeated result key")
            seen.add(key)
            by_scenario[scenario_id].append(dict(row))
    if not seen or any(not rows for rows in by_scenario.values()):
        raise ResearchError(f"Replay {name} omitted a preregistered scenario")
    return by_scenario


def _csv_bool(value, *, artifact_name, field):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ResearchError(f"Replay {artifact_name} contains an invalid {field}")


def _csv_int(value, *, artifact_name, field, minimum=0):
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        raise ResearchError(f"Replay {artifact_name} contains an invalid {field}") from None
    if str(parsed) != str(value).strip() or parsed < minimum:
        raise ResearchError(f"Replay {artifact_name} contains an invalid {field}")
    return parsed


def _csv_float(value, *, artifact_name, field, optional=False):
    if value in (None, ""):
        if optional:
            return None
        raise ResearchError(f"Replay {artifact_name} omitted {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ResearchError(f"Replay {artifact_name} contains an invalid {field}") from None
    if not math.isfinite(parsed):
        raise ResearchError(f"Replay {artifact_name} contains a nonfinite {field}")
    return parsed


def _same_number(actual, expected, *, optional=False):
    if actual is None or expected is None:
        return optional and actual is None and expected is None
    return _finite_number(actual) and _finite_number(expected) and math.isclose(
        float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-8
    )


def _resolved_segments(rows):
    result = []
    segment_start = segment_end = None
    sessions = 0
    for row in rows:
        if row["valuation_resolved"]:
            if segment_start is None:
                segment_start = row["date"]
                sessions = 0
            segment_end = row["date"]
            sessions += 1
        elif segment_start is not None:
            result.append({"start": segment_start, "end": segment_end, "sessions": sessions})
            segment_start = segment_end = None
            sessions = 0
    if segment_start is not None:
        result.append({"start": segment_start, "end": segment_end, "sessions": sessions})
    return result


def _scheduled_risk_exit_indices(path, spec):
    """Read risk-exit schedules after the action journal has passed validation."""
    journal = read_json(safe_file(Path(path).parent, "action-journal.json"))
    absolute_index = dict(zip(spec.evaluation_dates, spec.evaluation_indices))
    schedules = {}
    try:
        entries = journal["entries"]
        for entry in entries:
            if (
                entry.get("classification") != "daily_control"
                or entry.get("status") != "scheduled_unsupported_risk_exit"
            ):
                continue
            scenario_id = entry["scenario_id"]
            scheduled_index = absolute_index[entry["date"]]
            for security_id in entry["target_security_ids"]:
                schedules.setdefault((scenario_id, security_id), set()).add(scheduled_index)
    except (KeyError, TypeError):
        raise ResearchError("Replay risk-orphan schedule audit is invalid") from None
    return schedules


def _scheduled_action_exit_indices(path, spec):
    """Read recipient-exit schedules after the action journal has passed validation."""
    journal = read_json(safe_file(Path(path).parent, "action-journal.json"))
    schedules = {}
    try:
        for entry in journal["entries"]:
            if (
                entry.get("classification") != "stock_claim_delivery"
                or entry.get("status") != "recipient_claim_converted"
            ):
                continue
            key = (entry["scenario_id"], entry["recipient_security_id"])
            schedules.setdefault(key, set()).add(int(entry["exit_first_index"]))
    except (KeyError, TypeError, ValueError):
        raise ResearchError("Replay action-recipient schedule audit is invalid") from None
    return schedules


def _holding_events_from_journal(path, spec):
    """Return post-close unit additions and terminal removals for trade audit."""
    journal = read_json(safe_file(Path(path).parent, "action-journal.json"))
    absolute_index = dict(zip(spec.evaluation_dates, spec.evaluation_indices))
    additions = {}
    removals = {}
    removal_statuses = {
        "standard_resolved",
        "nonstandard_imputed_zero",
        "nonstandard_imputed_last_mark",
    }
    try:
        for entry in journal["entries"]:
            scenario_id = entry["scenario_id"]
            session_index = absolute_index[entry["date"]]
            key = (scenario_id, session_index)
            if entry.get("classification") == "stock_claim_delivery":
                security_id = entry["recipient_security_id"]
                units = float(entry["analytical_units"])
                if not security_id or not math.isfinite(units):
                    raise ValueError
                values = additions.setdefault(key, {})
                values[security_id] = values.get(security_id, 0.0) + units
            elif entry.get("status") in removal_statuses:
                security_id = entry["parent_security_id"]
                if not security_id:
                    raise ValueError
                removals.setdefault(key, set()).add(security_id)
    except (KeyError, TypeError, ValueError):
        raise ResearchError("Replay action journal cannot support the trade holding audit") from None
    return additions, removals


def _validate_trade_vectors(path, spec, daily_rows):
    expected = {
        "scenario_ids",
        "security_ids",
        "scenario_index",
        "session_index",
        "security_index",
        "requested_delta",
        "filled_delta",
        "kind",
        "batch_status",
        "batch_violations",
        "batch_turnover",
    }
    try:
        with np.load(path, allow_pickle=False) as vectors:
            if set(vectors.files) != expected or vectors["scenario_ids"].tolist() != list(spec.scenarios):
                raise ResearchError("Replay trade-vector identity is invalid")
            securities = vectors["security_ids"].tolist()
            if len(securities) != len(set(map(str, securities))):
                raise ResearchError("Replay trade-vector security identities are repeated")
            columns = [
                vectors[name]
                for name in (
                    "scenario_index",
                    "session_index",
                    "security_index",
                    "requested_delta",
                    "filled_delta",
                    "kind",
                    "batch_status",
                    "batch_violations",
                    "batch_turnover",
                )
            ]
            sizes = {int(column.size) for column in columns}
            if len(sizes) != 1 or any(column.ndim != 1 for column in columns):
                raise ResearchError("Replay trade-vector columns are misaligned")
            count = sizes.pop()
            if count == 0 or not securities:
                raise ResearchError("Replay trade vectors are an empty no-op artifact")
            scenario_index = vectors["scenario_index"]
            session_index = vectors["session_index"]
            security_index = vectors["security_index"]
            requested = vectors["requested_delta"]
            filled = vectors["filled_delta"]
            kinds = [str(value) for value in vectors["kind"].tolist()]
            statuses = [str(value) for value in vectors["batch_status"].tolist()]
            violations = [str(value) for value in vectors["batch_violations"].tolist()]
            turnover = vectors["batch_turnover"]
            if (
                not np.issubdtype(scenario_index.dtype, np.integer)
                or not np.issubdtype(session_index.dtype, np.integer)
                or not np.issubdtype(security_index.dtype, np.integer)
                or np.any(scenario_index < 0)
                or np.any(scenario_index >= len(spec.scenarios))
                or not set(map(int, scenario_index.tolist())) == set(range(len(spec.scenarios)))
                or not set(map(int, session_index.tolist())) <= set(spec.evaluation_indices)
                or np.any(security_index < 0)
                or np.any(security_index >= len(securities))
                or not np.isfinite(requested).all()
                or not np.isfinite(filled).all()
                or any(not str(value) for value in securities)
                or any(kind not in {"action_recipient_exit", "unsupported_risk_exit", "daily_reduction", "predeclared_final_close", "frozen_monthly_alpha"} for kind in kinds)
                or any(status not in {"reduce_only_action_exit", "reduce_only_risk_orphan_exit", "rejected_batch_constraints", "filled_derisk_residual_violation", "filled"} for status in statuses)
            ):
                raise ResearchError("Replay trade-vector values are outside the registered run")
            scenario_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
            filled_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
            status_counts = {status: 0 for status in set(statuses)}
            status_counts_by_scenario = {
                scenario_id: {status: 0 for status in set(statuses)} for scenario_id in spec.scenarios
            }
            risk_exit_schedules = _scheduled_risk_exit_indices(path, spec)
            action_exit_schedules = _scheduled_action_exit_indices(path, spec)
            holding_additions, holding_removals = _holding_events_from_journal(path, spec)
            terminal_action_indices = {}
            absolute_index = dict(zip(spec.evaluation_dates, spec.evaluation_indices))
            for package in spec.action_packages:
                action_index = absolute_index.get(package.action_date)
                if action_index is None:
                    raise ResearchError("Replay terminal package is outside the evaluation calendar")
                prior = terminal_action_indices.get(package.parent_security_id)
                terminal_action_indices[package.parent_security_id] = (
                    action_index if prior is None else min(prior, action_index)
                )
            observed_risk_exits = set()
            observed_trade_keys = set()
            terminal_trade_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
            holding_rows = {scenario_id: {} for scenario_id in spec.scenarios}
            daily_by_index = {
                scenario_id: {
                    absolute_index: row
                    for absolute_index, row in zip(spec.evaluation_indices, daily_rows[scenario_id])
                }
                for scenario_id in spec.scenarios
            }
            for row_index in range(count):
                scenario_id = spec.scenarios[int(scenario_index[row_index])]
                scenario_counts[scenario_id] += 1
                if abs(float(filled[row_index])) > 1e-12:
                    filled_counts[scenario_id] += 1
                status = statuses[row_index]
                status_counts[status] += 1
                status_counts_by_scenario[scenario_id][status] += 1
                batch_turnover = float(turnover[row_index])
                batch_violations = violations[row_index]
                kind = kinds[row_index]
                security_id = str(securities[int(security_index[row_index])])
                trade_index = int(session_index[row_index])
                terminal_index = terminal_action_indices.get(security_id)
                if terminal_index is not None and trade_index >= terminal_index:
                    terminal_trade_counts[scenario_id] += 1
                if (
                    spec.family_id == TERMINAL_PARENT_PUBLICATION_FAMILY_ID
                    and terminal_index is not None
                    and trade_index >= terminal_index
                ):
                    raise ResearchError(
                        "V8 replay trade vectors contain a terminal parent at or after its action close"
                    )
                trade_key = (scenario_id, trade_index, security_id)
                if trade_key in observed_trade_keys:
                    raise ResearchError("Replay trade vectors repeat a security at one close")
                observed_trade_keys.add(trade_key)
                holding_rows[scenario_id].setdefault(trade_index, []).append(
                    (security_id, float(requested[row_index]), float(filled[row_index]), status)
                )
                resolved = daily_by_index[scenario_id][int(session_index[row_index])]["valuation_resolved"]
                requested_delta = float(requested[row_index])
                filled_delta = float(filled[row_index])
                same_direction = (
                    abs(filled_delta) <= 1e-12
                    or math.copysign(1.0, filled_delta) == math.copysign(1.0, requested_delta)
                )
                fill_within_request = abs(filled_delta) <= abs(requested_delta) + max(
                    1e-12, abs(requested_delta) * 1e-12
                )
                if status == "reduce_only_action_exit":
                    valid = kind == "action_recipient_exit" and math.isnan(batch_turnover) and not batch_violations
                    if (
                        abs(requested_delta) <= 1e-12
                        or not same_direction
                        or not fill_within_request
                    ):
                        raise ResearchError("Replay action-recipient trade vector is not reduce-only")
                    if not any(
                        first_index <= trade_index
                        for first_index in action_exit_schedules.get((scenario_id, security_id), ())
                    ):
                        raise ResearchError(
                            "Replay action-recipient trade is not backed by a prior delivery"
                        )
                elif status == "reduce_only_risk_orphan_exit":
                    valid = kind == "unsupported_risk_exit" and math.isnan(batch_turnover) and not batch_violations
                    risk_key = trade_key
                    if (
                        risk_key in observed_risk_exits
                        or abs(requested_delta) <= 1e-12
                        or not same_direction
                        or not fill_within_request
                    ):
                        raise ResearchError("Replay risk-orphan trade vector is not reduce-only")
                    observed_risk_exits.add(risk_key)
                    if not any(
                        scheduled_index < trade_index
                        for scheduled_index in risk_exit_schedules.get((scenario_id, security_id), ())
                    ):
                        raise ResearchError(
                            "Replay risk-orphan trade is not backed by a prior scheduled control"
                        )
                else:
                    if kind in {"action_recipient_exit", "unsupported_risk_exit"}:
                        raise ResearchError("Replay reduce-only trade has an inconsistent batch status")
                    valid_turnover = (
                        math.isfinite(batch_turnover) and batch_turnover >= 0
                    ) or (math.isinf(batch_turnover) and batch_turnover > 0 and not resolved)
                    valid = valid_turnover and not math.isnan(batch_turnover)
                    if status == "filled":
                        valid = valid and not batch_violations
                    elif status == "rejected_batch_constraints":
                        valid = valid and bool(batch_violations) and abs(float(filled[row_index])) <= 1e-12
                    else:
                        valid = valid and bool(batch_violations)
                if not valid:
                    raise ResearchError("Replay trade-vector batch audit fields are inconsistent")
            # Reconstruct analytical holdings from zero.  Trades execute before
            # same-session action conversion; stock-claim deliveries add units,
            # then every successfully settled terminal parent is removed.  A
            # reduce-only row must request exactly the negative pre-trade
            # holding, rather than merely using a self-consistent fill sign.
            reduce_only_statuses = {
                "reduce_only_action_exit",
                "reduce_only_risk_orphan_exit",
            }
            for scenario_id in spec.scenarios:
                holdings = {}
                for trade_index in spec.evaluation_indices:
                    for security_id, requested_delta, filled_delta, status in holding_rows[
                        scenario_id
                    ].get(trade_index, ()):
                        current = float(holdings.get(security_id, 0.0))
                        if status in reduce_only_statuses and not math.isclose(
                            requested_delta,
                            -current,
                            rel_tol=1e-11,
                            abs_tol=1e-8,
                        ):
                            raise ResearchError(
                                "Replay reduce-only trade does not flatten its pre-trade holding"
                            )
                        updated = current + filled_delta
                        if abs(updated) <= 1e-12:
                            holdings.pop(security_id, None)
                        else:
                            holdings[security_id] = updated
                    event_key = (scenario_id, trade_index)
                    for security_id, units in holding_additions.get(event_key, {}).items():
                        holdings[security_id] = holdings.get(security_id, 0.0) + units
                        if abs(holdings[security_id]) <= 1e-12:
                            holdings.pop(security_id, None)
                    for security_id in holding_removals.get(event_key, ()):
                        holdings.pop(security_id, None)
            if any(not count for count in filled_counts.values()):
                raise ResearchError("Replay trade vectors contain no executed trade for a scenario")
            return {
                "rows": count,
                "rows_by_scenario": scenario_counts,
                "filled_rows_by_scenario": filled_counts,
                "rows_by_batch_status": status_counts,
                "rows_by_batch_status_and_scenario": status_counts_by_scenario,
                "terminal_parent_trade_rows_at_or_after_by_scenario": terminal_trade_counts,
            }
    except ResearchError:
        raise
    except Exception as error:
        raise ResearchError("Replay trade-vector artifact is invalid") from error


def _expected_reproduction_counts(spec):
    required_names = {
        "stock_claims_created_by_scenario",
        "stock_claim_deliveries_by_scenario",
        "daily_controls_by_scenario",
        "reduce_only_action_exit_rows_by_scenario",
    }
    risk_count_name = "reduce_only_risk_orphan_exit_rows_by_scenario"
    terminal_count_name = "terminal_parent_trade_rows_at_or_after_by_scenario"
    if spec.family_id == TERMINAL_PARENT_PUBLICATION_FAMILY_ID:
        required_names.update({risk_count_name, terminal_count_name})
        optional_names = set()
    elif spec.family_id == FINAL_PUBLICATION_FAMILY_ID:
        required_names.add(risk_count_name)
        optional_names = set()
    else:
        optional_names = {risk_count_name, terminal_count_name}
    try:
        counts = spec.protocol["publication_union_schema_correction"]["expected_reproduction_counts"]
    except (KeyError, TypeError):
        raise ResearchError("Replay protocol omitted outcome-known reproduction counts") from None
    if (
        not isinstance(counts, dict)
        or not required_names <= set(counts)
        or not set(counts) <= required_names | optional_names
    ):
        raise ResearchError("Replay protocol has invalid outcome-known reproduction counts")
    result = {}
    for name in counts:
        values = counts[name]
        if name == "daily_controls_by_scenario" and values == "derive_from_daily_hard_violations":
            result[name] = values
            continue
        if (
            not isinstance(values, dict)
            or set(values) != set(spec.scenarios)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values.values()
            )
        ):
            raise ResearchError("Replay protocol has invalid outcome-known reproduction counts")
        result[name] = dict(values)
    return result


def _validate_action_journal(directory, spec, daily_rows):
    journal = read_json(safe_file(directory, "action-journal.json"))
    if (
        not isinstance(journal, dict)
        or journal.get("scenario_ids") != list(spec.scenarios)
        or journal.get("standard_action_accounting_common") is not True
        or not isinstance(journal.get("entries"), list)
    ):
        raise ResearchError("Replay action journal must retain the common method and event-level records")
    packages = {row.package_id: row for row in spec.action_packages}
    dispositions = {}
    deliveries = set()
    controls = set()
    unresolved_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
    imputed_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
    disposition_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
    delivery_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
    control_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
    stock_claim_counts = {scenario_id: 0 for scenario_id in spec.scenarios}
    control_status_counts = {
        "scheduled_next_close": 0,
        "scheduled_unsupported_risk_exit": 0,
        "blocked_unavailable_held_risk_or_nav": 0,
        "right_censored": 0,
    }
    absolute_index = dict(zip(spec.evaluation_dates, spec.evaluation_indices))
    daily_violations = {
        (scenario_id, row["date"]): tuple(row["hard_violations"])
        for scenario_id, rows in daily_rows.items()
        for row in rows
    }
    allowed_violations = {
        "gross",
        "name",
        "net",
        "beta",
        "forecast_risk",
        "industry_net",
        "industry_beta",
        "unavailable_held_risk",
    }
    unresolved_statuses = {
        "standard_missing_parent_conversion_basis",
        "standard_unresolved",
        "nonstandard_censored",
        "nonstandard_missing_last_mark",
    }
    successful_imputations = {"nonstandard_imputed_zero", "nonstandard_imputed_last_mark"}
    for entry in journal["entries"]:
        if not isinstance(entry, dict) or entry.get("scenario_id") not in spec.scenarios:
            raise ResearchError("Replay action journal contains an undeclared scenario")
        scenario_id = entry["scenario_id"]
        try:
            day = date.fromisoformat(str(entry.get("date"))).isoformat()
        except (TypeError, ValueError):
            raise ResearchError("Replay action journal contains an invalid date") from None
        if day not in absolute_index:
            raise ResearchError("Replay action journal crossed the registered evaluation calendar")
        package_id = entry.get("package_id")
        if package_id is None:
            if entry.get("classification") == "daily_control":
                control_key = (scenario_id, day)
                status = entry.get("status")
                violations = entry.get("violations")
                common_keys = {
                    "scenario_id",
                    "date",
                    "package_id",
                    "parent_security_id",
                    "classification",
                    "status",
                    "violations",
                    "imputed",
                }
                expected_keys = common_keys
                if status == "scheduled_next_close":
                    expected_keys |= {"uniform_scale", "target_names"}
                elif status == "scheduled_unsupported_risk_exit":
                    expected_keys |= {"target_names", "target_security_ids"}
                if (
                    set(entry) != expected_keys
                    or entry.get("parent_security_id") is not None
                    or entry.get("imputed") is not False
                    or control_key in controls
                    or not isinstance(violations, list)
                    or not violations
                    or len(violations) != len(set(violations))
                    or any(not isinstance(value, str) or value not in allowed_violations for value in violations)
                    or tuple(violations) != daily_violations.get(control_key)
                    or status not in control_status_counts
                ):
                    raise ResearchError("Replay action journal contains an invalid or repeated daily control")
                if status == "scheduled_next_close":
                    scale = entry.get("uniform_scale")
                    target_names = entry.get("target_names")
                    if (
                        not _finite_number(scale)
                        or not 0 <= float(scale) < 1.0
                        or not isinstance(target_names, int)
                        or isinstance(target_names, bool)
                        or target_names <= 0
                        or "unavailable_held_risk" in violations
                    ):
                        raise ResearchError("Replay scheduled daily control is invalid")
                elif status == "scheduled_unsupported_risk_exit":
                    target_names = entry.get("target_names")
                    target_ids = entry.get("target_security_ids")
                    if (
                        violations != ["unavailable_held_risk"]
                        or not isinstance(target_names, int)
                        or isinstance(target_names, bool)
                        or target_names <= 0
                        or not isinstance(target_ids, list)
                        or len(target_ids) != target_names
                        or target_ids != sorted(set(target_ids))
                        or any(not isinstance(value, str) or not value for value in target_ids)
                    ):
                        raise ResearchError("Replay unsupported-risk exit control is invalid")
                elif status == "blocked_unavailable_held_risk_or_nav":
                    if violations != ["unavailable_held_risk"]:
                        raise ResearchError("Replay blocked daily control is invalid")
                elif day != spec.end or "unavailable_held_risk" in violations:
                    raise ResearchError("Replay right-censored daily control is invalid")
                controls.add(control_key)
                control_counts[scenario_id] += 1
                control_status_counts[status] += 1
                continue
            if entry.get("classification") != "stock_claim_delivery":
                raise ResearchError("Replay action journal contains an unknown auxiliary event")
            delivery_key = (scenario_id, day, entry.get("recipient_security_id"))
            if (
                entry.get("parent_security_id") is not None
                or entry.get("classification") != "stock_claim_delivery"
                or entry.get("status") != "recipient_claim_converted"
                or entry.get("imputed") is not False
                or not isinstance(entry.get("recipient_security_id"), str)
                or not entry["recipient_security_id"]
                or not _finite_number(entry.get("raw_shares"))
                or not _finite_number(entry.get("analytical_units"))
                or entry.get("exit_first_index") != absolute_index[day] + 1
                or delivery_key in deliveries
            ):
                raise ResearchError("Replay action journal contains an invalid recipient delivery")
            deliveries.add(delivery_key)
            delivery_counts[scenario_id] += 1
            continue
        package = packages.get(str(package_id))
        key = (scenario_id, str(package_id))
        if package is None or key in dispositions:
            raise ResearchError("Replay action journal contains an unknown or repeated terminal package")
        if (
            day != package.action_date
            or entry.get("parent_security_id") != package.parent_security_id
            or entry.get("classification") != package.classification
            or not isinstance(entry.get("imputed"), bool)
        ):
            raise ResearchError("Replay action journal changed a pinned terminal package")
        status = entry.get("status")
        if package.classification in {"cash", "stock", "mixed"}:
            allowed = {
                "not_held",
                "standard_resolved",
                "standard_missing_parent_conversion_basis",
                "standard_unresolved",
            }
            expected_imputed = False
        elif scenario_id == "strict_standard_only":
            allowed = {"not_held", "nonstandard_censored"}
            expected_imputed = False
        elif scenario_id == "imputed_zero_insolvency_last_mark_other" and package.insolvency:
            allowed = {"not_held", "nonstandard_imputed_zero"}
            expected_imputed = status == "nonstandard_imputed_zero"
        else:
            allowed = {"not_held", "nonstandard_imputed_last_mark", "nonstandard_missing_last_mark"}
            expected_imputed = status != "not_held"
        if status not in allowed or entry.get("imputed") is not expected_imputed:
            raise ResearchError("Replay action journal disposition conflicts with its preregistered scenario")
        if status == "standard_resolved":
            stock_claims = entry.get("stock_claims")
            if (
                not isinstance(stock_claims, int)
                or isinstance(stock_claims, bool)
                or stock_claims < 0
                or (package.classification == "cash" and stock_claims != 0)
            ):
                raise ResearchError("Replay standard action has invalid created stock-claim accounting")
            stock_claim_counts[scenario_id] += stock_claims
        elif "stock_claims" in entry:
            raise ResearchError("Replay non-resolved action reports created stock claims")
        dispositions[key] = status
        disposition_counts[scenario_id] += 1
        unresolved_counts[scenario_id] += int(status in unresolved_statuses)
        imputed_counts[scenario_id] += int(status in successful_imputations)
    expected_dispositions = {
        (scenario_id, package.package_id)
        for scenario_id in spec.scenarios
        for package in spec.action_packages
    }
    if set(dispositions) != expected_dispositions:
        raise ResearchError("Replay action journal omitted a terminal-package scenario disposition")
    expected_controls = {key for key, violations in daily_violations.items() if violations}
    if controls != expected_controls:
        raise ResearchError("Replay action journal omitted a daily-control audit record")
    expected_counts = _expected_reproduction_counts(spec)
    if (
        stock_claim_counts != expected_counts["stock_claims_created_by_scenario"]
        or delivery_counts != expected_counts["stock_claim_deliveries_by_scenario"]
        or (
            expected_counts["daily_controls_by_scenario"] != "derive_from_daily_hard_violations"
            and control_counts != expected_counts["daily_controls_by_scenario"]
        )
    ):
        raise ResearchError("Replay action journal differs from the pinned outcome-known reproduction counts")
    return {
        "terminal_dispositions": len(dispositions),
        "delivery_entries": len(deliveries),
        "daily_control_entries": len(controls),
        "dispositions_by_scenario": disposition_counts,
        "deliveries_by_scenario": delivery_counts,
        "daily_controls_by_scenario": control_counts,
        "daily_controls_by_status": {
            key: value
            for key, value in control_status_counts.items()
            if key != "scheduled_unsupported_risk_exit" or value
        },
        "stock_claims_created_by_scenario": stock_claim_counts,
        "unresolved_by_scenario": unresolved_counts,
        "imputed_by_scenario": imputed_counts,
    }


def _validate_results(directory, spec, *, worker_only=False):
    directory = Path(directory).resolve()
    children = list(directory.iterdir())
    found = {p.name for p in children if p.is_file()}
    if (not RESULT_FILES <= found or (worker_only and ({p.name for p in children} != RESULT_FILES
                                                        or any(not p.is_file() for p in children)))
            or any((directory / name).is_symlink() for name in RESULT_FILES)):
        raise ResearchError("Replay worker produced an unexpected artifact set")
    if any(safe_file(directory, name).stat().st_size == 0 for name in RESULT_FILES):
        raise ResearchError("Replay worker produced an empty artifact")
    outcome = read_json(safe_file(directory, "outcomes.json"))
    expected = {
        "start": spec.start,
        "end": spec.end,
        "parent_trial_id": spec.parent_trial_id,
        "parent_result_hash": spec.parent_result_hash,
        "action_bundle_id": spec.action_bundle_id,
        "scenario_ids": list(spec.scenarios),
        "point_in_time_attested": False,
        "investment_ready": False,
        "holdout_evaluations": 0,
        "provider_calls": 0,
        "llm_calls": 0,
    }
    if any(outcome.get(key) != value for key, value in expected.items()):
        raise ResearchError("Replay outcome identity, scenarios or access boundary changed")
    if (
        outcome.get("methodology") != "frozen_factor_tape_action_aware_replay_v1"
        or outcome.get("standard_action_accounting_common") is not True
        or set(outcome.get("scenarios", {})) != set(spec.scenarios)
    ):
        raise ResearchError("Replay outcome omitted the common method or a preregistered scenario")
    daily_csv = _csv_inventory(
        directory,
        "daily.csv",
        spec,
        {
            "scenario_id",
            "date",
            "valuation_resolved",
            "return_observed",
            "nav_observed_usd",
            "nav_estimate_usd",
            "missing_held_marks_or_claims",
            "borrow_usd",
            "funding_usd",
            "trade_cost_usd",
            "held_names",
            "raw_claims",
            "hard_violations",
        },
        ("scenario_id", "date"),
    )
    months = _csv_inventory(
        directory,
        "months.csv",
        spec,
        {"scenario_id", "date", "optimizer_status"},
        ("scenario_id", "date"),
    )
    forward = _csv_inventory(
        directory,
        "forward.csv",
        spec,
        {
            "scenario_id",
            "date",
            "factor",
            "entry_date",
            "exit_date",
            "return_complete",
            "known_component_pnl",
            "missing_names",
        },
        ("scenario_id", "date", "factor"),
    )
    daily = {scenario_id: [] for scenario_id in spec.scenarios}
    for scenario_id in spec.scenarios:
        if [row["date"] for row in daily_csv[scenario_id]] != spec.evaluation_dates:
            raise ResearchError("Replay daily coverage differs from the pinned evaluation calendar")
        previous_observed = None
        for row in daily_csv[scenario_id]:
            raw_violations = row["hard_violations"]
            hard_violations = tuple(raw_violations.split(",")) if raw_violations else ()
            allowed_violations = {
                "gross",
                "name",
                "net",
                "beta",
                "forecast_risk",
                "industry_net",
                "industry_beta",
                "unavailable_held_risk",
            }
            if (
                len(hard_violations) != len(set(hard_violations))
                or any(value not in allowed_violations for value in hard_violations)
            ):
                raise ResearchError("Replay daily.csv contains invalid hard_violations")
            resolved = _csv_bool(
                row["valuation_resolved"], artifact_name="daily.csv", field="valuation_resolved"
            )
            observed = _csv_float(
                row["nav_observed_usd"], artifact_name="daily.csv", field="nav_observed_usd", optional=True
            )
            estimate = _csv_float(
                row["nav_estimate_usd"], artifact_name="daily.csv", field="nav_estimate_usd"
            )
            missing = _csv_int(
                row["missing_held_marks_or_claims"],
                artifact_name="daily.csv",
                field="missing_held_marks_or_claims",
            )
            observed_return = _csv_float(
                row["return_observed"], artifact_name="daily.csv", field="return_observed", optional=True
            )
            expected_return = (
                observed / previous_observed - 1.0
                if observed is not None and previous_observed is not None and previous_observed > 0
                else None
            )
            parsed = {
                "date": row["date"],
                "valuation_resolved": resolved,
                "nav_observed_usd": observed,
                "nav_estimate_usd": estimate,
                "missing": missing,
                "borrow_usd": _csv_float(row["borrow_usd"], artifact_name="daily.csv", field="borrow_usd"),
                "funding_usd": _csv_float(row["funding_usd"], artifact_name="daily.csv", field="funding_usd"),
                "trade_cost_usd": _csv_float(
                    row["trade_cost_usd"], artifact_name="daily.csv", field="trade_cost_usd"
                ),
                "held_names": _csv_int(row["held_names"], artifact_name="daily.csv", field="held_names"),
                "raw_claims": _csv_int(row["raw_claims"], artifact_name="daily.csv", field="raw_claims"),
                "hard_violations": hard_violations,
            }
            if (
                resolved is not (observed is not None)
                or resolved is not (missing == 0)
                or not _same_number(observed_return, expected_return, optional=True)
                or any(parsed[field] < 0 for field in ("borrow_usd", "funding_usd", "trade_cost_usd"))
            ):
                raise ResearchError("Replay daily valuation, return or cost accounting is inconsistent")
            daily[scenario_id].append(parsed)
            previous_observed = observed
    if any([row["date"] for row in months[scenario_id]] != spec.formation_dates for scenario_id in spec.scenarios):
        raise ResearchError("Replay monthly coverage differs from the pinned parent formations")
    if any(not row.get("optimizer_status") for scenario_id in spec.scenarios for row in months[scenario_id]):
        raise ResearchError("Replay monthly optimizer audit is incomplete")
    interval_by_formation = {row.formation_date: row for row in spec.forward_intervals}
    expected_forward_keys = {
        (scenario_id, interval.formation_date, factor)
        for scenario_id in spec.scenarios
        for interval in spec.forward_intervals
        for factor in spec.forward_factor_ids
    }
    observed_forward_keys = set()
    incomplete_forward_rows = {scenario_id: 0 for scenario_id in spec.scenarios}
    for scenario_id in spec.scenarios:
        for row in forward[scenario_id]:
            key = (scenario_id, row["date"], row["factor"])
            interval = interval_by_formation.get(row["date"])
            missing_names = _csv_int(
                row["missing_names"], artifact_name="forward.csv", field="missing_names"
            )
            complete_return = _csv_float(
                row["return_complete"], artifact_name="forward.csv", field="return_complete", optional=True
            )
            _csv_float(
                row["known_component_pnl"], artifact_name="forward.csv", field="known_component_pnl"
            )
            if (
                interval is None
                or row["factor"] not in spec.forward_factor_ids
                or row["entry_date"] != interval.entry_date
                or row["exit_date"] != interval.exit_date
                or (complete_return is None) is not (missing_names > 0)
            ):
                raise ResearchError("Replay forward row differs from its pinned half-open factor interval")
            incomplete_forward_rows[scenario_id] += int(missing_names > 0)
            observed_forward_keys.add(key)
    if observed_forward_keys != expected_forward_keys:
        raise ResearchError("Replay forward coverage is not the exact 117-factor-plus-MODEL Cartesian result")
    journal_validation = _validate_action_journal(directory, spec, daily)
    trade_validation = _validate_trade_vectors(safe_file(directory, "trade-vectors.npz"), spec, daily)
    expected_reproduction = _expected_reproduction_counts(spec)
    observed_action_exits = {
        scenario_id: trade_validation["rows_by_batch_status_and_scenario"][scenario_id].get(
            "reduce_only_action_exit", 0
        )
        for scenario_id in spec.scenarios
    }
    if observed_action_exits != expected_reproduction["reduce_only_action_exit_rows_by_scenario"]:
        raise ResearchError("Replay trade vectors omitted a pinned recipient-exit row")
    expected_risk_exits = expected_reproduction.get("reduce_only_risk_orphan_exit_rows_by_scenario")
    if expected_risk_exits is not None:
        observed_risk_exits = {
            scenario_id: trade_validation["rows_by_batch_status_and_scenario"][scenario_id].get(
                "reduce_only_risk_orphan_exit", 0
            )
            for scenario_id in spec.scenarios
        }
        if observed_risk_exits != expected_risk_exits:
            raise ResearchError("Replay trade vectors omitted a pinned risk-orphan exit row")
    expected_terminal_trades = expected_reproduction.get(
        "terminal_parent_trade_rows_at_or_after_by_scenario"
    )
    if (
        expected_terminal_trades is not None
        and trade_validation["terminal_parent_trade_rows_at_or_after_by_scenario"]
        != expected_terminal_trades
    ):
        raise ResearchError("Replay trade vectors crossed the terminal-parent action cutoff")
    for scenario_id in spec.scenarios:
        scenario = outcome["scenarios"][scenario_id]
        if not isinstance(scenario, dict):
            raise ResearchError("Replay scenario summary is invalid")
        expected_imputed = scenario_id != "strict_standard_only"
        count_fields = (
            "sessions",
            "formations",
            "imputed_events",
            "unresolved_sessions",
            "unresolved_action_events",
            "unresolved_terminal_names",
            "unresolved_stock_claim_names",
        )
        if (
            scenario.get("imputed") is not expected_imputed
            or any(
                not isinstance(scenario.get(field), int)
                or isinstance(scenario[field], bool)
                or scenario[field] < 0
                for field in count_fields
            )
            or scenario.get("sessions", -1) < 1
            or scenario.get("sessions") != len(daily[scenario_id])
            or scenario.get("formations") != len(months[scenario_id])
            or scenario.get("unresolved_sessions", 0) > scenario.get("sessions", 0)
            or not isinstance(scenario.get("continuous_resolved_segments"), list)
            or not _finite_number(scenario.get("unresolved_terminal_signed_marked_exposure_usd"))
            or not _finite_number(scenario.get("unresolved_stock_claim_signed_marked_exposure_usd"))
            or (scenario.get("unresolved_sessions", 0) > 0 and scenario.get("complete_path_return") is not None)
            or (
                scenario.get("unresolved_sessions") == 0
                and not _finite_number(scenario.get("complete_path_return"))
            )
            or (scenario_id == "strict_standard_only" and scenario.get("imputed_events") != 0)
            or (
                scenario.get("unresolved_sessions", 0) == 0
                and scenario.get("first_unresolved_date") is not None
            )
            or (
                scenario.get("unresolved_sessions", 0) > 0
                and not isinstance(scenario.get("first_unresolved_date"), str)
            )
        ):
            raise ResearchError("Replay scenario coverage or sensitivity label is invalid")
        rows = daily[scenario_id]
        resolved_rows = [row for row in rows if row["valuation_resolved"]]
        first_unresolved = next((row["date"] for row in rows if not row["valuation_resolved"]), None)
        last_resolved = resolved_rows[-1] if resolved_rows else None
        final = rows[-1]
        complete_return = (
            final["nav_observed_usd"] / float(spec.parent_spec["policy"]["execution"]["starting_nav_usd"]) - 1.0
            if len(resolved_rows) == len(rows)
            else None
        )
        if (
            scenario.get("unresolved_sessions") != len(rows) - len(resolved_rows)
            or scenario.get("first_unresolved_date") != first_unresolved
            or scenario.get("last_fully_reconstructed_date")
            != (None if last_resolved is None else last_resolved["date"])
            or not _same_number(
                scenario.get("last_fully_reconstructed_nav_usd"),
                None if last_resolved is None else last_resolved["nav_observed_usd"],
                optional=True,
            )
            or scenario.get("continuous_resolved_segments") != _resolved_segments(rows)
            or scenario.get("unresolved_action_events")
            != journal_validation["unresolved_by_scenario"][scenario_id]
            or scenario.get("imputed_events") != journal_validation["imputed_by_scenario"][scenario_id]
            or not _same_number(scenario.get("final_nav_observed_usd"), final["nav_observed_usd"], optional=True)
            or not _same_number(scenario.get("final_nav_estimate_usd"), final["nav_estimate_usd"])
            or not _same_number(scenario.get("complete_path_return"), complete_return, optional=True)
            or not _same_number(scenario.get("borrow_usd"), sum(row["borrow_usd"] for row in rows))
            or not _same_number(scenario.get("funding_usd"), sum(row["funding_usd"] for row in rows))
            or not _same_number(scenario.get("trade_cost_usd"), sum(row["trade_cost_usd"] for row in rows))
            or scenario.get("final_held_names") != final["held_names"]
            or scenario.get("final_raw_claims") != final["raw_claims"]
            or scenario.get("unresolved_stock_claim_names") != final["raw_claims"]
            or scenario.get("unresolved_terminal_names", 0) > final["held_names"] + final["raw_claims"]
        ):
            raise ResearchError("Replay scenario summary does not reconcile to daily and event-level results")
    runtime = read_json(safe_file(directory, "runtime.json"))
    expected_runtime = {
        "implementation": spec.implementation,
        "environment": spec.environment,
        "seed": spec.seed,
        "threads": spec.threads,
        "provider_keys_present": False,
        "proxy_present": False,
        "isolation": "owned_credential_filtered_process_not_filesystem_or_network_sandbox",
    }
    if runtime != expected_runtime:
        raise ResearchError("Replay worker runtime identity or credential isolation failed")
    published_trade_validation = dict(trade_validation)
    if spec.family_id != TERMINAL_PARENT_PUBLICATION_FAMILY_ID:
        # Historical publications predate the V8 terminal-cutoff field.  Keep
        # their serialized validation records byte-reproducible while still
        # computing the count above for direct compatibility audits.
        published_trade_validation.pop(
            "terminal_parent_trade_rows_at_or_after_by_scenario", None
        )
    validation = {
        "schema_version": 1,
        "kind": "factor_action_accounting_continuation_validation",
        "trial_id": spec.semantic_id(),
        "protocol_sha256": spec.protocol_artifact.sha256,
        "expected": {
            "scenarios": len(spec.scenarios),
            "evaluation_sessions_per_scenario": len(spec.evaluation_dates),
            "formations_per_scenario": len(spec.formation_dates),
            "forward_intervals_per_scenario": len(spec.forward_intervals),
            "forward_factors_per_interval": len(spec.forward_factor_ids),
            "terminal_packages_per_scenario": len(spec.action_packages),
        },
        "observed": {
            "daily_rows": sum(len(rows) for rows in daily.values()),
            "monthly_rows": sum(len(rows) for rows in months.values()),
            "forward_rows": len(observed_forward_keys),
            "incomplete_forward_rows_by_scenario": incomplete_forward_rows,
            **journal_validation,
            "trade_vectors": published_trade_validation,
        },
        "result_artifacts": {name: artifact(safe_file(directory, name)) for name in sorted(RESULT_FILES)},
    }
    return outcome, validation


def _report(outcome):
    return (
        "Frozen JKP factor model: action-accounting continuation\n"
        f"Window: {outcome['start']}..{outcome['end']}\n"
        f"Parent trial: {outcome['parent_trial_id']}\n"
        f"Scenarios: {', '.join(outcome['scenario_ids'])}\n"
        "The strict path applies only documented standard terms; nonstandard stresses are imputed sensitivities, not observed bounds.\n"
        "This ex-post reconstruction is not point-in-time attested, investment ready, a parent retry, or a fresh holdout.\n"
        "No provider/LLM calls or real orders were allowed. See outcomes.json and action-journal.json for coverage and unresolved claims.\n"
    )


def inspect_replay(directory):
    directory = Path(directory).resolve()
    record = read_json(safe_file(directory, "run.json"))
    spec = parse(ReplaySpec, read_json(safe_file(directory, "spec.json")))
    base = {
        "directory": str(directory),
        "trial_id": spec.semantic_id(),
        "attempt_id": record["attempt_id"],
        "hypothesis_id": spec.hypothesis_id,
        "parent_trial_id": spec.parent_trial_id,
        "start": spec.start,
        "end": spec.end,
        "effective_status": record["status"],
        "artifacts_verified": False,
        "result_hash": None,
        "evidence_kind": EVIDENCE_KIND,
        "investment_ready": False,
    }
    if not (directory / "COMPLETE.json").exists():
        return base
    children = list(directory.iterdir())
    if {path.name for path in children} != PUBLICATION_FILES or any(
        not path.is_file() or path.is_symlink() for path in children
    ):
        raise ResearchError("Factor replay publication contains an unexpected or nonregular artifact")
    marker = read_json(safe_file(directory, "COMPLETE.json"))
    if (
        marker.get("kind") != "factor_action_accounting_continuation_v1"
        or marker.get("trial_id") != spec.semantic_id()
        or marker.get("parent_trial_id") != spec.parent_trial_id
        or record.get("status") != "completed"
    ):
        raise ResearchError("Invalid factor replay completion")
    marker_files = marker.get("files", {})
    record_files = record.get("artifacts", {})
    if (
        set(marker_files) != MARKER_FILES
        or set(record_files) != RECORD_FILES
        or any(marker_files[name] != record_files[name] for name in RECORD_FILES)
    ):
        raise ResearchError("Factor replay completion inventory is incomplete or inconsistent")
    verify_files(directory, marker_files)
    verify_files(directory, record_files)
    published_protocol = safe_file(directory, "protocol.json")
    if parse(Artifact, artifact(published_protocol)) != spec.protocol_artifact or read_json(published_protocol) != spec.protocol:
        raise ResearchError("Published continuation protocol differs from the registered identity")
    outcome, validation = _validate_results(directory, spec)
    if read_json(safe_file(directory, "validation.json")) != validation:
        raise ResearchError("Published replay validation record does not reproduce")
    if safe_file(directory, "report.txt").read_text(encoding="utf-8") != _report(outcome):
        raise ResearchError("Published replay report differs from the validated outcome")
    return base | {
        "artifacts_verified": True,
        "result_hash": file_hash(directory / "COMPLETE.json"),
        "report": str(directory / "report.txt"),
        "summary": outcome,
    }


def _preflight_attempt(registry_path, spec, retry_of):
    with Registry(registry_path, readonly=True) as registry:
        registration = registry.db.execute(
            "SELECT f.body,t.max_attempts FROM registrations r "
            "JOIN hypotheses h ON h.id=r.hypothesis JOIN families f ON f.id=h.family "
            "JOIN trials t ON t.id=r.trial "
            "WHERE r.hypothesis=? AND r.trial=? AND h.family=?",
            (spec.hypothesis_id, spec.semantic_id(), spec.family_id),
        ).fetchone()
        if not registration:
            raise ResearchError("Register this replay computation before executing")
        attempts = registry.db.execute(
            "SELECT * FROM attempts WHERE trial=? ORDER BY created,id", (spec.semantic_id(),)
        ).fetchall()
        if any(row["status"] in ("running", "completed") for row in attempts):
            raise ResearchError("Replay trial is active, unresolved or completed; inspect its evidence")
        limit = min(json.loads(registration["body"])["max_attempts_per_trial"], registration["max_attempts"])
        if len(attempts) >= limit:
            raise ResearchError("Replay trial attempt budget is exhausted")
        if (attempts and retry_of != attempts[-1]["id"]) or (not attempts and retry_of is not None):
            raise ResearchError("Replay retry must identify the last failed attempt")


def _remove_unstarted_destination(destination):
    for name in ("spec.json", "protocol.json", "parent-binding.json", "action-binding.json", "inputs-binding.json"):
        (destination / name).unlink(missing_ok=True)
    try:
        destination.rmdir()
    except OSError:
        pass


def run_replay(inputs, parent_run, action_bundle, spec_path, destination, registry_path, *, retry_of=None):
    spec = parse(ReplaySpec, read_json(spec_path))
    inputs, parent_run, action_bundle = map(lambda p: Path(p).resolve(), (inputs, parent_run, action_bundle))
    validate_replay_spec(spec, inputs, parent_run, action_bundle)
    destination = Path(destination).resolve()
    for source in (inputs, parent_run, action_bundle):
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ResearchError("Replay output must be separate from every immutable input")
    _preflight_attempt(registry_path, spec, retry_of)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(destination / "spec.json", spec.model_dump())
        protocol = _protocol_binding(action_bundle)
        shutil.copyfile(protocol["path"], destination / "protocol.json")
        if parse(Artifact, artifact(destination / "protocol.json")) != spec.protocol_artifact:
            raise ResearchError("Copied continuation protocol differs from the registered identity")
        atomic_json(
            destination / "parent-binding.json",
            {
                "directory": str(parent_run),
                "trial_id": spec.parent_trial_id,
                "attempt_id": spec.parent_attempt_id,
                "result_hash": spec.parent_result_hash,
                "complete_sha256": spec.parent_complete_sha256,
                "complete": spec.parent_complete,
                "spec_sha256": spec.parent_spec_sha256,
                "artifacts": {k: v.model_dump() for k, v in spec.parent_artifacts.items()},
                "formation_dates_sha256": digest(spec.formation_dates),
                "formations": len(spec.formation_dates),
                "forward_factor_ids_sha256": digest(spec.forward_factor_ids),
                "forward_factors": len(spec.forward_factor_ids),
            },
        )
        atomic_json(
            destination / "action-binding.json",
            {
                "directory": str(action_bundle),
                "bundle_id": spec.action_bundle_id,
                "manifest": spec.action_manifest,
                "manifest_sha256": spec.action_manifest_sha256,
                "files": {k: v.model_dump() for k, v in spec.action_files.items()},
                "action_packages_sha256": digest([row.model_dump() for row in spec.action_packages]),
                "terminal_packages": len(spec.action_packages),
            },
        )
        atomic_json(
            destination / "inputs-binding.json",
            {
                "directory": str(inputs),
                "bundle_id": spec.parent_model_inputs.bundle_id,
                "manifest_sha256": spec.inputs_manifest_sha256,
                "evaluation_dates_sha256": digest(spec.evaluation_dates),
                "evaluation_indices_sha256": digest(spec.evaluation_indices),
                "evaluation_sessions": len(spec.evaluation_dates),
            },
        )
    except BaseException:
        _remove_unstarted_destination(destination)
        raise
    with Registry(registry_path) as registry:
        try:
            attempt = registry.begin(spec, destination, retry_of, access_bounds=(spec.start, spec.end))
        except BaseException:
            _remove_unstarted_destination(destination)
            raise
        record = {
            "schema_version": 6,
            "kind": "factor_action_accounting_continuation",
            "attempt_id": attempt,
            "trial_id": spec.semantic_id(),
            "hypothesis_id": spec.hypothesis_id,
            "parent_trial_id": spec.parent_trial_id,
            "status": "running",
            "stage": "evaluating",
            "started_at": stamp(),
            "updated_at": stamp(),
            "finished_at": None,
            "failure": None,
            "worker_pid": None,
            "peak_rss_mb": 0.0,
            "artifacts": {},
            "llm_calls": 0,
            "provider_calls": 0,
            "holdout_evaluations": 0,
            "scientific_disposition": "ex_post_action_reconstruction_and_imputation_sensitivity_not_parent_retry",
        }
        work = destination / "worker-output"
        proc = None
        failure = "setup_failure"
        published = False
        try:
            atomic_json(destination / "run.json", record)
            work.mkdir()
            failure = "execution_failure"
            started = time.monotonic()
            with (destination / "worker.out").open("w", encoding="utf-8") as log:
                proc = start_process(
                    [
                        sys.executable,
                        "-s",
                        "-P",
                        "-m",
                        "alpha_lab.research.factor_replay_workflow",
                        "worker",
                        "--inputs",
                        str(inputs),
                        "--parent-run",
                        str(parent_run),
                        "--actions",
                        str(action_bundle),
                        "--spec",
                        str(destination / "spec.json"),
                        "--output",
                        str(work),
                    ],
                    cwd=str(destination),
                    env=child_environment(spec),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                record["worker_pid"] = proc.pid
                while proc.poll() is None:
                    try:
                        root = psutil.Process(proc.pid)
                        rss = sum(
                            p.memory_info().rss
                            for p in [root] + root.children(recursive=True)
                            if p.is_running()
                        ) / 2**20
                    except psutil.Error:
                        rss = record["peak_rss_mb"]
                    record["peak_rss_mb"] = max(record["peak_rss_mb"], rss)
                    record["updated_at"] = stamp()
                    atomic_json(destination / "run.json", record)
                    size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
                    failure = (
                        "timeout"
                        if time.monotonic() - started > spec.timeout_seconds
                        else "memory_limit"
                        if rss > spec.memory_limit_mb
                        else "output_limit"
                        if size > spec.max_output_bytes
                        else "execution_failure"
                    )
                    if failure != "execution_failure":
                        raise ResearchError("Factor replay resource guard reached")
                    time.sleep(1.0)
                if proc.wait() != 0:
                    raise ResearchError("Factor replay worker failed; worker.out contains diagnostics")
            failure = "publication_failure"
            validate_replay_spec(spec, inputs, parent_run, action_bundle)
            outcome, validation = _validate_results(work, spec, worker_only=True)
            if sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) > spec.max_output_bytes:
                failure = "output_limit"
                raise ResearchError("Factor replay output cap exceeded")
            for path in work.iterdir():
                path.replace(destination / path.name)
            work.rmdir()
            (destination / "report.txt").write_text(_report(outcome), encoding="utf-8")
            atomic_json(destination / "validation.json", validation)
            if sum(path.stat().st_size for path in destination.iterdir() if path.is_file()) > spec.max_output_bytes:
                failure = "output_limit"
                raise ResearchError("Factor replay publication cap exceeded")
            files = {
                path.name: artifact(path)
                for path in destination.iterdir()
                if path.is_file() and path.name not in {"run.json", "COMPLETE.json"}
            }
            record.update(
                status="completed",
                stage="finished",
                updated_at=stamp(),
                finished_at=stamp(),
                artifacts=files,
            )
            atomic_json(destination / "run.json", record)
            atomic_json(
                destination / "COMPLETE.json",
                {
                    "kind": "factor_action_accounting_continuation_v1",
                    "trial_id": spec.semantic_id(),
                    "parent_trial_id": spec.parent_trial_id,
                    "files": files | {"run.json": artifact(destination / "run.json")},
                },
            )
            evidence = inspect_replay(destination)
            published = True
            registry.finish(attempt, complete=evidence)
            return evidence
        except BaseException as error:
            if proc is not None and proc.poll() is None:
                proc.terminate_tree()
            if published:
                raise ResearchError("Published factor replay requires registry reconciliation") from None
            record.update(
                status="failed",
                stage="finished",
                updated_at=stamp(),
                finished_at=stamp(),
                failure=failure,
            )
            (destination / "COMPLETE.json").unlink(missing_ok=True)
            record_error = None
            try:
                atomic_json(destination / "run.json", record)
            except BaseException as write_error:
                record_error = write_error
            registry.finish(attempt, failure=failure)
            if record_error is not None:
                raise ResearchError("Factor replay failed and its terminal run record could not be saved") from record_error
            raise ResearchError("Factor replay failed: " + failure) from error
        finally:
            if proc is not None:
                proc.close()


def reconcile_replay(registry_path, directory):
    evidence = inspect_replay(directory)
    with Registry(registry_path) as registry:
        registry.verify_attempt(evidence)
        if evidence["artifacts_verified"]:
            registry.finish(evidence["attempt_id"], complete=evidence)
        elif evidence["effective_status"] == "failed":
            registry.finish(evidence["attempt_id"], failure=read_json(Path(directory) / "run.json")["failure"])
        else:
            raise ResearchError("Unfinished replay requires attention, not a duplicate run")
    return evidence


def _worker(inputs, parent_run, action_bundle, spec_path, output):
    from .factor_replay import run_replay as evaluate_replay

    spec = parse(ReplaySpec, read_json(spec_path))
    validate_replay_spec(spec, inputs, parent_run, action_bundle)
    result = evaluate_replay(inputs, parent_run, action_bundle, output, _settings(spec))
    atomic_json(Path(output) / "runtime.json", _runtime(spec))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("worker", "run"):
        item = commands.add_parser(command)
        item.add_argument("--inputs", type=Path, required=True)
        item.add_argument("--parent-run", type=Path, required=True)
        item.add_argument("--actions", type=Path, required=True)
        item.add_argument("--spec", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        if command == "run":
            item.add_argument("--registry", type=Path, required=True)
            item.add_argument("--retry-of")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory", type=Path)
    register = commands.add_parser("register")
    register.add_argument("--spec", type=Path, required=True)
    register.add_argument("--registry", type=Path, required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--directory", type=Path, required=True)
    reconcile.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_replay(args.directory)
    elif args.command == "register":
        result = {"trial_id": register_replay(args.registry, read_json(args.spec))}
    elif args.command == "reconcile":
        result = reconcile_replay(args.registry, args.directory)
    elif args.command == "run":
        result = run_replay(
            args.inputs,
            args.parent_run,
            args.actions,
            args.spec,
            args.output,
            args.registry,
            retry_of=args.retry_of,
        )
    else:
        result = _worker(args.inputs, args.parent_run, args.actions, args.spec, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
