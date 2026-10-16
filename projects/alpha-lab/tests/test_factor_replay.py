"""Independent arithmetic checks for the frozen action-aware factor replay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.research import factor_replay as replay_module
from alpha_lab.research.factor_replay import (
    SCENARIO_IDS,
    ReplayState,
    SparseTargets,
    action_aware_forward_interval,
    replay_arrays,
    scenario_metadata,
    _daily_control,
)


def execution_policy(participation=1.0):
    return {
        "execution": {
            "annual_hypothetical_borrow": 0.03,
            "cash_debit": 0.05,
            "participation": participation,
            "one_way_commission_bps": 1.0,
            "one_way_slippage_bps": 5.0,
        }
    }


def package(package_id, parent, day, classification, *, cash=None, legs=(), reasons=()):
    return {
        "package_id": package_id,
        "parent_security_id": parent,
        "parent_ticker": parent,
        "action_date": day,
        "classification": classification,
        "cash_usd_per_raw_share": cash,
        "stock_legs": list(legs),
        "source_rows_sha256": "0" * 64,
        "unresolved_reasons": list(reasons),
    }


def test_sparse_targets_store_only_changed_names_and_do_not_preorder_recipients():
    order = SparseTargets.changed(
        decision_index=4,
        fill_index=5,
        kind="frozen_monthly_alpha",
        current={"A": 10.0, "B": -2.0},
        targets={"A": 10.0, "B": 0.0, "C": 3.0},
    )
    assert order.targets == {"B": 0.0, "C": 3.0}
    state = ReplayState("strict_standard_only", 1_000.0, units={"A": 10.0})
    state.schedule(order)
    assert state.pending is order
    assert state.exit_after == {}
    state.schedule_exit("R", 9)
    assert state.exit_after == {"R": 9}
    assert "R" not in state.pending.targets


def test_initial_frozen_optimization_uses_formation_quotes_for_new_names(monkeypatch):
    dates = ("2020-01-31",)
    adjusted = np.array([[10.0, 20.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ("A", "B"),
        adjusted,
        adjusted.copy(),
        np.full_like(adjusted, 1_000_000.0),
    )
    context = replay_module._TapeContext(
        date=dates[0],
        ids=("A", "B"),
        lookup={"A": 0, "B": 1},
        industry=np.array(["I1", "I2"]),
        beta=np.array([1.0, 1.0]),
        supported=np.array([True, True]),
        b=np.zeros((2, 1)),
        d=np.ones(2),
    )
    tape = {
        "expected_returns": np.array([0.1, -0.1]),
        "model_weights": np.array([0.5, -0.5]),
    }
    monkeypatch.setattr(
        replay_module,
        "optimize",
        lambda *args, **kwargs: {
            "status": "optimal",
            "weights": np.array([0.5, -0.5]),
        },
    )
    policy = {
        "optimization": {
            "inner_gross": 1.0,
            "inner_name": 0.5,
            "inner_turnover": 2.0,
            "annual_vol_cap": 0.1,
            "horizon_borrow_days": 21.0,
            "solver_time_limit_seconds": 1.0,
        },
        "execution": {
            "one_way_commission_bps": 1.0,
            "one_way_slippage_bps": 5.0,
            "annual_hypothetical_borrow": 0.03,
        },
    }
    result, targets = replay_module._optimize_frozen(
        ReplayState("strict_standard_only", 1_000.0),
        context,
        tape,
        {"A", "B"},
        1_000.0,
        policy,
        index=0,
        quotes=quotes,
    )
    assert result["status"] == "optimal"
    assert targets == {"A": 50.0, "B": -25.0}


def test_standard_cash_recovery_then_unknown_terminal_censors_strict_nav():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    ids = ["A", "B"]
    adjusted = np.array([[10.0, 5.0], [10.0, 5.0], [np.nan, 5.0], [np.nan, np.nan]])
    raw = adjusted.copy()
    dv = np.full_like(adjusted, 1_000_000.0)
    actions = [
        package("cash-A", "A", dates[1], "cash", cash=20.0),
        package("unknown-B", "B", dates[2], "unresolved", reasons=["no_standard_consideration_terms"]),
    ]
    result = replay_arrays(
        dates=dates,
        security_ids=ids,
        adjusted=adjusted,
        raw=raw,
        dollar_volume=dv,
        packages=actions,
        scenario_id="strict_standard_only",
        policy=execution_policy(),
        initial_units={"A": 10.0, "B": 2.0},
        starting_cash=100.0,
    )
    standard = next(row for row in result["journal"] if row["package_id"] == "cash-A")
    assert standard["status"] == "standard_resolved"
    assert standard["cash_usd"] == 200.0
    assert result["daily"][1]["nav_observed_usd"] == 310.0
    assert result["daily"][2]["nav_observed_usd"] is None
    assert result["daily"][3]["nav_observed_usd"] is None
    assert result["final_units"] == {"B": 2.0}


def test_terminal_package_permanently_blocks_held_and_not_held_parent_reentry():
    dates = ["2020-01-02", "2020-01-03"]
    prices = np.array([[10.0], [10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ["A"],
        prices,
        prices.copy(),
        np.full_like(prices, 1_000_000.0),
    )
    context = replay_module._TapeContext(
        dates[1],
        ("A",),
        {"A": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.0001]),
    )
    policy = execution_policy()
    policy["hard_limits"] = {
        "gross": 10.0,
        "name": 10.0,
        "turnover": 10.0,
        "abs_net": 10.0,
        "abs_beta": 10.0,
        "industry_abs_net": 10.0,
        "industry_abs_beta": 10.0,
        "annual_vol": 10.0,
        "tolerance": 1e-6,
    }
    terminal = package("cash-A", "A", dates[0], "cash", cash=20.0)

    for initial_units, expected_status in (({}, "not_held"), ({"A": 1.0}, "standard_resolved")):
        state = ReplayState("strict_standard_only", 1_000.0, units=initial_units)
        state.update_marks(0, quotes)
        state.schedule_risk_exit("A", 1)
        state.process_packages(
            index=0,
            day=dates[0],
            packages=[terminal],
            quotes=quotes,
            actions=replay_module.DEFAULT_ACTION_FUNCTIONS,
            finalize_not_held=True,
        )

        assert state.journal[-1]["status"] == expected_status
        assert "A" in state.restricted_ids
        assert "A" not in state.risk_exit_after
        assert "A" not in state.units

        state.schedule(
            SparseTargets.changed(
                decision_index=0,
                fill_index=1,
                kind="frozen_monthly_alpha",
                current=state.units,
                targets={"A": 1.0},
            )
        )
        state.execute_orders(1, dates[1], quotes, policy, context)
        assert state.units == {}
        assert state.trades == []


def test_current_day_terminal_parent_preempts_pending_trade_and_later_reentry():
    dates = ["2020-01-02", "2020-01-03"]
    prices = np.array([[10.0], [10.0]])
    action = package("cash-A", "A", dates[0], "cash", cash=20.0)
    first = SparseTargets.changed(
        decision_index=-1,
        fill_index=0,
        kind="daily_reduction",
        current={"A": 10.0},
        targets={"A": 5.0},
    )
    second = SparseTargets.changed(
        decision_index=0,
        fill_index=1,
        kind="frozen_monthly_alpha",
        current={},
        targets={"A": 1.0},
    )
    policy = execution_policy()
    policy["hard_limits"] = {"turnover": 10.0}

    result = replay_arrays(
        dates=dates,
        security_ids=["A"],
        adjusted=prices,
        raw=prices.copy(),
        dollar_volume=np.full_like(prices, 1_000_000.0),
        packages=[action],
        policy=policy,
        initial_units={"A": 10.0},
        starting_cash=1_000.0,
        scheduled_orders={0: first, 1: second},
    )

    assert result["trades"] == []
    assert result["final_units"] == {}
    assert result["final_cash"] == 1_200.0
    assert result["journal"][0]["status"] == "standard_resolved"
    assert result["journal"][0]["parent_units"] == 10.0


def test_current_day_not_held_terminal_parent_blocks_pending_entry():
    dates = ["2020-01-02", "2020-01-03"]
    prices = np.array([[10.0], [10.0]])
    action = package("cash-A", "A", dates[0], "cash", cash=20.0)
    orders = {
        index: SparseTargets.changed(
            decision_index=index - 1,
            fill_index=index,
            kind="frozen_monthly_alpha",
            current={},
            targets={"A": 1.0},
        )
        for index in range(2)
    }
    policy = execution_policy()
    policy["hard_limits"] = {"turnover": 10.0}

    result = replay_arrays(
        dates=dates,
        security_ids=["A"],
        adjusted=prices,
        raw=prices.copy(),
        dollar_volume=np.full_like(prices, 1_000_000.0),
        packages=[action],
        policy=policy,
        starting_cash=1_000.0,
        scheduled_orders=orders,
    )

    assert result["trades"] == []
    assert result["final_units"] == {}
    assert result["final_cash"] == 1_000.0
    assert result["journal"][0]["status"] == "not_held"


def test_registering_terminal_parent_cancels_both_exit_queues():
    terminal = package("cash-A", "A", "2020-01-02", "cash", cash=20.0)
    risk_state = ReplayState("strict_standard_only", 1_000.0, units={"A": 1.0})
    risk_state.schedule_risk_exit("A", 1)
    recipient_state = ReplayState("strict_standard_only", 1_000.0, units={"A": 1.0})
    recipient_state.schedule_exit("A", 1)

    risk_state.register_terminal_parents([terminal])
    recipient_state.register_terminal_parents([terminal])

    assert risk_state.risk_exit_after == {}
    assert recipient_state.exit_after == {}
    assert risk_state.restricted_ids == {"A"}
    assert recipient_state.restricted_ids == {"A"}
    risk_state.schedule_risk_exit("A", 2)
    assert risk_state.risk_exit_after == {}


def test_no_action_replay_equals_direct_adjusted_unit_account():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    adjusted = np.array([[10.0], [11.0], [12.0]])
    result = replay_arrays(
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        dollar_volume=np.full_like(adjusted, 1_000_000.0),
        packages=[],
        scenario_id="strict_standard_only",
        policy=execution_policy(),
        initial_units={"A": 10.0},
        starting_cash=100.0,
    )
    assert [row["nav_observed_usd"] for row in result["daily"]] == [200.0, 210.0, 220.0]
    assert np.isclose(result["daily"][1]["return_observed"], 0.05)
    assert np.isclose(result["daily"][2]["return_observed"], 220.0 / 210.0 - 1.0)
    assert result["journal"] == []
    assert result["final_cash"] == 100.0


def test_preregistered_scenario_labels_disclose_imputation():
    assert SCENARIO_IDS == (
        "strict_standard_only",
        "imputed_zero_insolvency_last_mark_other",
        "imputed_last_mark_all_unresolved",
    )
    strict = scenario_metadata(SCENARIO_IDS[0])
    zero = scenario_metadata(SCENARIO_IDS[1])
    last = scenario_metadata(SCENARIO_IDS[2])
    assert strict["imputed"] is False
    assert zero["imputed"] is True and "imputed_sensitivity" in zero["label"]
    assert last["imputed"] is True and "imputed_sensitivity" in last["label"]


def test_stock_recipient_is_delivered_then_exited_at_next_observed_close():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    adjusted = np.array([[10.0], [10.0], [np.nan]])
    supplemental = pd.DataFrame(
        {
            "security_id": ["R", "R"],
            "ticker": ["R", "R"],
            "date": [dates[1], dates[2]],
            "raw_close_usd": [5.0, 6.0],
            "adjusted_close_usd": [5.0, 6.0],
            "dollar_volume_usd": [100_000.0, 100_000.0],
        }
    )
    action = package(
        "stock-A",
        "A",
        dates[1],
        "stock",
        legs=[{"recipient_id": "R", "recipient_ticker": "R", "ratio": 2.0}],
    )
    result = replay_arrays(
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        dollar_volume=np.full_like(adjusted, 1_000_000.0),
        packages=[action],
        scenario_id="strict_standard_only",
        policy=execution_policy(),
        initial_units={"A": 10.0},
        starting_cash=0.0,
        supplemental_quotes=supplemental,
    )
    delivery = next(row for row in result["journal"] if row["classification"] == "stock_claim_delivery")
    assert delivery["raw_shares"] == 20.0
    assert delivery["exit_first_index"] == 2
    assert result["final_units"] == {}
    exit_trade = next(row for row in result["trades"] if row["kind"] == "action_recipient_exit")
    assert exit_trade["filled_delta"] == -20.0


def test_delayed_stock_claim_delivery_on_day_without_action_reaches_quiescence():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    adjusted = np.array([[10.0], [10.0], [np.nan], [np.nan]])
    supplemental = pd.DataFrame(
        {
            "security_id": ["R", "R"],
            "ticker": ["R", "R"],
            "date": [dates[2], dates[3]],
            "raw_close_usd": [5.0, 6.0],
            "adjusted_close_usd": [5.0, 6.0],
            "dollar_volume_usd": [100_000.0, 100_000.0],
        }
    )
    action = package(
        "stock-A",
        "A",
        dates[1],
        "stock",
        legs=[{"recipient_id": "R", "recipient_ticker": "R", "ratio": 2.0}],
    )
    result = replay_arrays(
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        dollar_volume=np.full_like(adjusted, 1_000_000.0),
        packages=[action],
        scenario_id="strict_standard_only",
        policy=execution_policy(),
        initial_units={"A": 10.0},
        starting_cash=0.0,
        supplemental_quotes=supplemental,
    )
    delivery = next(row for row in result["journal"] if row["classification"] == "stock_claim_delivery")
    assert delivery["date"] == dates[2]
    assert delivery["raw_shares"] == 20.0
    assert result["final_raw_claims"] == {}
    assert result["final_units"] == {}


def test_action_aware_forward_interval_includes_entry_close_cash_consideration():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    adjusted = np.array([[10.0], [10.0], [10.0]])
    action = package("cash-A", "A", dates[1], "cash", cash=12.0)
    result = action_aware_forward_interval(
        np.array([1.0]),
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        decision_index=0,
        entry_index=1,
        exit_index=2,
        packages=[action],
        scenario_id="strict_standard_only",
    )
    assert result["missing_names"] == 0
    assert np.isclose(result["return_complete"], 0.2)
    assert np.isclose(result["known_component_pnl"], 0.2)


def test_forward_interval_delayed_stock_quote_on_day_without_action_reaches_quiescence():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    adjusted = np.array([[10.0], [10.0], [np.nan], [np.nan]])
    supplemental = pd.DataFrame(
        {
            "security_id": ["R", "R"],
            "ticker": ["R", "R"],
            "date": [dates[2], dates[3]],
            "raw_close_usd": [5.0, 6.0],
            "adjusted_close_usd": [5.0, 6.0],
            "dollar_volume_usd": [100_000.0, 100_000.0],
        }
    )
    action = package(
        "stock-A",
        "A",
        dates[1],
        "stock",
        legs=[{"recipient_id": "R", "recipient_ticker": "R", "ratio": 2.0}],
    )
    result = action_aware_forward_interval(
        np.array([1.0]),
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        decision_index=0,
        entry_index=1,
        exit_index=3,
        packages=[action],
        scenario_id="strict_standard_only",
        supplemental_quotes=supplemental,
    )
    assert result["missing_names"] == 0
    assert np.isclose(result["return_complete"], 0.2)
    assert np.isclose(result["known_component_pnl"], 0.2)


def test_action_aware_forward_interval_excludes_exit_close_action():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    adjusted = np.array([[10.0], [10.0], [10.0]])
    action = package("cash-A", "A", dates[2], "cash", cash=12.0)
    result = action_aware_forward_interval(
        np.array([1.0]),
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        decision_index=0,
        entry_index=1,
        exit_index=2,
        packages=[action],
        scenario_id="strict_standard_only",
    )
    assert result["missing_names"] == 0
    assert np.isclose(result["return_complete"], 0.0)


def test_increasing_batch_that_breaks_hard_limits_is_rejected():
    dates = ("2020-01-31",)
    adjusted = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ("A",),
        adjusted,
        adjusted.copy(),
        np.full_like(adjusted, 1_000_000.0),
    )
    context = replay_module._TapeContext(
        date=dates[0],
        ids=("A",),
        lookup={"A": 0},
        industry=np.array(["I"]),
        beta=np.array([0.0]),
        supported=np.array([True]),
        b=np.zeros((1, 1)),
        d=np.array([0.01]),
    )
    state = ReplayState("strict_standard_only", 1_000.0)
    state.schedule(
        SparseTargets.changed(
            decision_index=-1,
            fill_index=0,
            kind="frozen_monthly_alpha",
            current={},
            targets={"A": 10.0},
        )
    )
    policy = execution_policy()
    policy["hard_limits"] = {
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
    state.execute_orders(0, dates[0], quotes, policy, context)
    assert state.units == {}
    assert state.trades[-1]["filled_delta"] == 0.0
    assert state.trades[-1]["batch_status"] == "rejected_batch_constraints"
    assert "name" in state.trades[-1]["batch_violations"]


def test_frozen_context_uses_nan_beta_as_parent_risk_support_mask():
    tape = {
        "security_ids": np.array(["A", "B"]),
        "beta": np.array([1.2, np.nan]),
        "risk_diagonal": np.array([0.01, 0.02]),
        "risk_loadings": np.zeros((3, 2)),
    }
    context = replay_module._context("2020-01-31", tape, {"A": "I", "B": "I"})
    assert context.supported.tolist() == [True, False]
    assert context.beta.tolist() == [1.2, 0.0]

    state = ReplayState("strict_standard_only", 1_000.0, units={"B": 1.0})
    state.adjusted_marks["B"] = 10.0
    assert replay_module._position_metrics(state, context, 1_010.0) is None


def test_insolvency_source_type_drives_only_the_preregistered_zero_sensitivity():
    dates = ["2020-01-02", "2020-01-03"]
    adjusted = np.array([[10.0], [10.0]])
    action = package(
        "bankrupt-A",
        "A",
        dates[1],
        "unresolved",
        reasons=["no_standard_consideration_terms"],
    )
    action["source_action_types"] = ["bankruptcyliquidation"]
    common = dict(
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        dollar_volume=np.full_like(adjusted, 1_000_000.0),
        packages=[action],
        policy=execution_policy(),
        initial_units={"A": 10.0},
        starting_cash=100.0,
    )
    zero = replay_arrays(scenario_id="imputed_zero_insolvency_last_mark_other", **common)
    last = replay_arrays(scenario_id="imputed_last_mark_all_unresolved", **common)
    assert zero["final_units"] == {} and zero["final_cash"] == 100.0
    assert last["final_units"] == {} and last["final_cash"] == 200.0
    assert zero["journal"][-1]["imputation"] == "zero_insolvency"
    assert last["journal"][-1]["imputation"] == "last_raw_mark"


def test_daily_control_overrides_alpha_with_sparse_buffered_reduction():
    state = ReplayState("strict_standard_only", 1_000.0, units={"A": 10.0, "B": -10.0})
    state.schedule(
        SparseTargets.changed(
            decision_index=3,
            fill_index=4,
            kind="frozen_monthly_alpha",
            current=state.units,
            targets={"A": 12.0, "B": -12.0},
        )
    )
    policy = {
        "optimization": {"inner_gross": 0.8, "inner_name": 0.04, "annual_vol_cap": 0.1},
        "hard_limits": {
            "gross": 1.0,
            "name": 0.05,
            "abs_net": 0.01,
            "abs_beta": 0.01,
            "industry_abs_net": 0.005,
            "industry_abs_beta": 0.005,
            "annual_vol": 0.1,
        },
    }
    metrics = {
        "gross": 1.2,
        "name": 0.06,
        "net": 0.0,
        "beta": 0.0,
        "annual_ex_ante_vol": 0.08,
        "industries": {"I": {"net": 0.0, "beta": 0.0}},
    }
    _daily_control(
        state,
        index=3,
        final_index=10,
        day="2020-01-03",
        metrics=metrics,
        violations=["gross", "name"],
        policy=policy,
    )
    assert state.pending is not None and state.pending.kind == "daily_reduction"
    assert set(state.pending.targets) == {"A", "B"}
    assert np.isclose(state.pending.targets["A"], 10.0 * (2.0 / 3.0))
    assert np.isclose(state.pending.targets["B"], -10.0 * (2.0 / 3.0))
    assert state.journal[-1]["status"] == "scheduled_next_close"


def test_daily_control_queues_risk_exit_without_discarding_conditional_alpha():
    state = ReplayState("strict_standard_only", 1_000.0, units={"A": 10.0, "ORPHAN": -4.0})
    state.adjusted_marks.update({"A": 10.0, "ORPHAN": 7.0})
    state.schedule(
        SparseTargets.changed(
            decision_index=3,
            fill_index=4,
            kind="frozen_monthly_alpha",
            current=state.units,
            targets={"A": 12.0, "ORPHAN": -4.0},
        )
    )
    context = replay_module._TapeContext(
        "2020-01-03",
        ("A",),
        {"A": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    _daily_control(
        state,
        index=3,
        final_index=10,
        day="2020-01-03",
        metrics=None,
        violations=["unavailable_held_risk"],
        policy={"optimization": {}, "hard_limits": {}},
        context=context,
    )
    assert state.pending is not None
    assert state.pending.kind == "frozen_monthly_alpha"
    assert state.risk_exit_after == {"ORPHAN": 4}
    assert state.journal[-1] == {
        "scenario_id": "strict_standard_only",
        "date": "2020-01-03",
        "package_id": None,
        "parent_security_id": None,
        "classification": "daily_control",
        "status": "scheduled_unsupported_risk_exit",
        "violations": ["unavailable_held_risk"],
        "target_names": 1,
        "target_security_ids": ["ORPHAN"],
        "imputed": False,
    }


def test_daily_control_keeps_unavailable_nav_without_risk_orphan_blocked():
    state = ReplayState("strict_standard_only", 1_000.0, units={"A": 10.0})
    state.adjusted_marks["A"] = 10.0
    context = replay_module._TapeContext(
        "2020-01-03",
        ("A",),
        {"A": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    _daily_control(
        state,
        index=3,
        final_index=10,
        day="2020-01-03",
        metrics=None,
        violations=["unavailable_held_risk"],
        policy={"optimization": {}, "hard_limits": {}},
        context=context,
    )
    assert state.pending is None
    assert state.journal[-1]["status"] == "blocked_unavailable_held_risk_or_nav"


def test_unknown_terminal_is_never_reclassified_as_a_tradeable_risk_orphan():
    state = ReplayState("strict_standard_only", 1_000.0, units={"ORPHAN": 1.0})
    state.adjusted_marks["ORPHAN"] = 10.0
    state.unknown_terminals.add("ORPHAN")
    context = replay_module._TapeContext(
        "2020-01-03",
        ("A",),
        {"A": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    _daily_control(
        state,
        index=3,
        final_index=10,
        day="2020-01-03",
        metrics=None,
        violations=["unavailable_held_risk"],
        policy={"optimization": {}, "hard_limits": {}},
        context=context,
    )
    assert state.risk_exit_after == {}
    assert state.journal[-1]["status"] == "blocked_unavailable_held_risk_or_nav"


def test_risk_orphan_exit_executes_before_conditional_alpha_at_same_close():
    dates = ["2020-01-06"]
    ids = ["A", "B", "ORPHAN"]
    prices = np.array([[10.0, 10.0, 10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ids,
        prices,
        prices.copy(),
        np.full_like(prices, 1_000_000.0),
    )
    state = ReplayState(
        "strict_standard_only",
        1_000.0,
        units={"A": 1.0, "B": -1.0, "ORPHAN": -1.0},
    )
    state.update_marks(0, quotes)
    state.schedule_risk_exit("ORPHAN", 0)
    state.schedule(
        SparseTargets.changed(
            decision_index=-1,
            fill_index=0,
            kind="frozen_monthly_alpha",
            current=state.units,
            targets={"A": 2.0, "B": -2.0},
        )
    )
    context = replay_module._TapeContext(
        dates[0],
        ("A", "B"),
        {"A": 0, "B": 1},
        np.array(["I", "I"]),
        np.array([1.0, 1.0]),
        np.array([True, True]),
        np.zeros((1, 2)),
        np.array([0.0001, 0.0001]),
    )
    policy = execution_policy()
    policy["hard_limits"] = {
        "gross": 1.0,
        "name": 0.05,
        "turnover": 2.0,
        "abs_net": 0.01,
        "abs_beta": 0.01,
        "industry_abs_net": 0.005,
        "industry_abs_beta": 0.005,
        "annual_vol": 0.1,
        "tolerance": 1e-6,
    }
    state.execute_orders(0, dates[0], quotes, policy, context)
    assert "ORPHAN" not in state.units
    assert np.isclose(state.units["A"], 2.0)
    assert np.isclose(state.units["B"], -2.0)
    assert [row["kind"] for row in state.trades] == [
        "unsupported_risk_exit",
        "frozen_monthly_alpha",
        "frozen_monthly_alpha",
    ]
    assert state.trades[0]["batch_status"] == "reduce_only_risk_orphan_exit"
    assert all(row["batch_status"] == "filled" for row in state.trades[1:])


def test_queued_risk_exit_is_cancelled_if_name_becomes_unknown_terminal():
    dates = ["2020-01-06"]
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ["ORPHAN"],
        prices,
        prices.copy(),
        np.array([[1_000_000.0]]),
    )
    state = ReplayState("strict_standard_only", 1_000.0, units={"ORPHAN": 1.0})
    state.update_marks(0, quotes)
    state.schedule_risk_exit("ORPHAN", 0)
    state.unknown_terminals.add("ORPHAN")

    state.execute_orders(0, dates[0], quotes, execution_policy())

    assert state.units == {"ORPHAN": 1.0}
    assert state.unknown_terminals == {"ORPHAN"}
    assert state.risk_exit_after == {}
    assert state.trades == []


def test_queued_action_recipient_exit_is_cancelled_if_name_becomes_unknown_terminal():
    dates = ["2020-01-06"]
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ["RECIPIENT"],
        prices,
        prices.copy(),
        np.array([[1_000_000.0]]),
    )
    state = ReplayState("strict_standard_only", 1_000.0, units={"RECIPIENT": 1.0})
    state.update_marks(0, quotes)
    state.schedule_exit("RECIPIENT", 0)
    state.unknown_terminals.add("RECIPIENT")

    state.execute_orders(0, dates[0], quotes, execution_policy())

    assert state.units == {"RECIPIENT": 1.0}
    assert state.unknown_terminals == {"RECIPIENT"}
    assert state.exit_after == {}
    assert state.trades == []


def test_due_risk_exit_cannot_double_fill_through_pending_batch():
    dates = ["2020-01-06"]
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ["ORPHAN"],
        prices,
        prices.copy(),
        np.array([[100.0]]),
    )
    state = ReplayState("strict_standard_only", 1_000.0, units={"ORPHAN": 10.0})
    state.update_marks(0, quotes)
    state.schedule_risk_exit("ORPHAN", 0)
    state.schedule(
        SparseTargets.changed(
            decision_index=-1,
            fill_index=0,
            kind="daily_reduction",
            current=state.units,
            targets={"ORPHAN": 0.0},
        )
    )
    context = replay_module._TapeContext(
        dates[0],
        ("ORPHAN",),
        {"ORPHAN": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    policy = execution_policy(participation=0.1)
    policy["hard_limits"] = {
        "gross": 10.0,
        "name": 10.0,
        "turnover": 10.0,
        "abs_net": 10.0,
        "abs_beta": 10.0,
        "industry_abs_net": 10.0,
        "industry_abs_beta": 10.0,
        "annual_vol": 10.0,
        "tolerance": 1e-6,
    }

    state.execute_orders(0, dates[0], quotes, policy, context)

    assert np.isclose(state.units["ORPHAN"], 9.0)
    assert len(state.trades) == 1
    assert state.trades[0]["kind"] == "unsupported_risk_exit"
    assert np.isclose(state.trades[0]["filled_delta"], -1.0)


def test_completed_risk_exit_cannot_be_reentered_by_pending_batch():
    dates = ["2020-01-06"]
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ["ORPHAN"],
        prices,
        prices.copy(),
        np.array([[1_000_000.0]]),
    )
    state = ReplayState("strict_standard_only", 1_000.0, units={"ORPHAN": 1.0})
    state.update_marks(0, quotes)
    state.schedule_risk_exit("ORPHAN", 0)
    state.schedule(
        SparseTargets.changed(
            decision_index=-1,
            fill_index=0,
            kind="predeclared_final_close",
            current=state.units,
            targets={"ORPHAN": 0.5},
        )
    )
    context = replay_module._TapeContext(
        dates[0],
        ("ORPHAN",),
        {"ORPHAN": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    policy = execution_policy()
    policy["hard_limits"] = {
        "gross": 10.0,
        "name": 10.0,
        "turnover": 10.0,
        "abs_net": 10.0,
        "abs_beta": 10.0,
        "industry_abs_net": 10.0,
        "industry_abs_beta": 10.0,
        "annual_vol": 10.0,
        "tolerance": 1e-6,
    }

    state.execute_orders(0, dates[0], quotes, policy, context)

    assert state.units == {}
    assert len(state.trades) == 1
    assert state.trades[0]["kind"] == "unsupported_risk_exit"


def test_unknown_terminal_cannot_be_liquidated_by_predeclared_final_close():
    dates = ["2020-01-06"]
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ["UNKNOWN"],
        prices,
        prices.copy(),
        np.array([[1_000_000.0]]),
    )
    state = ReplayState("strict_standard_only", 1_000.0, units={"UNKNOWN": 1.0})
    state.update_marks(0, quotes)
    state.unknown_terminals.add("UNKNOWN")
    state.schedule(
        SparseTargets.changed(
            decision_index=-1,
            fill_index=0,
            kind="predeclared_final_close",
            current=state.units,
            targets={"UNKNOWN": 0.0},
        )
    )
    policy = execution_policy()
    policy["hard_limits"] = {"turnover": 10.0}

    state.execute_orders(0, dates[0], quotes, policy)

    assert state.units == {"UNKNOWN": 1.0}
    assert state.unknown_terminals == {"UNKNOWN"}
    assert state.trades == []


def test_frozen_optimizer_projects_already_queued_risk_exit_even_after_support_returns(monkeypatch):
    dates = ("2020-01-31",)
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ("ORPHAN",),
        prices,
        prices.copy(),
        np.array([[1_000_000.0]]),
    )
    context = replay_module._TapeContext(
        dates[0],
        ("ORPHAN",),
        {"ORPHAN": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    state = ReplayState("strict_standard_only", 900.0, units={"ORPHAN": 10.0})
    state.update_marks(0, quotes)
    state.schedule_risk_exit("ORPHAN", 1)
    captured = {}

    def fake_optimize(mu, current, industry, beta, b, d, eligible, **kwargs):
        captured["current"] = current.copy()
        captured["eligible"] = eligible.copy()
        return {"status": "feasible", "weights": np.zeros_like(current)}

    monkeypatch.setattr(replay_module, "optimize", fake_optimize)
    policy = {
        "optimization": {
            "inner_gross": 1.0,
            "inner_name": 0.05,
            "inner_turnover": 2.0,
            "annual_vol_cap": 0.1,
            "horizon_borrow_days": 21.0,
            "solver_time_limit_seconds": 1.0,
        },
        "execution": {
            "one_way_commission_bps": 1.0,
            "one_way_slippage_bps": 5.0,
            "annual_hypothetical_borrow": 0.03,
        },
    }
    tape = {
        "expected_returns": np.array([0.1]),
        "model_weights": np.array([1.0]),
    }

    result, targets = replay_module._optimize_frozen(
        state,
        context,
        tape,
        {"ORPHAN"},
        1_000.0,
        policy,
        index=0,
        quotes=quotes,
    )

    assert captured["current"].tolist() == [0.0]
    assert captured["eligible"].tolist() == [False]
    assert result["status"] == "feasible_conditional_reduce_only_exits"
    assert result["risk_orphan_exit_names"] == 1
    assert targets == {"ORPHAN": 0.0}


def test_optimization_cache_key_distinguishes_queued_reduce_only_exit():
    dates = ("2020-01-31",)
    prices = np.array([[10.0]])
    quotes = replay_module.QuoteBook(
        dates,
        ("A",),
        prices,
        prices.copy(),
        np.array([[1_000_000.0]]),
    )
    context = replay_module._TapeContext(
        dates[0],
        ("A",),
        {"A": 0},
        np.array(["I"]),
        np.array([1.0]),
        np.array([True]),
        np.zeros((1, 1)),
        np.array([0.01]),
    )
    unqueued = ReplayState("strict_standard_only", 900.0, units={"A": 10.0})
    queued = ReplayState("imputed_last_mark_all_unresolved", 900.0, units={"A": 10.0})
    for state in (unqueued, queued):
        state.update_marks(0, quotes)
    queued.schedule_risk_exit("A", 1)

    plain_key = replay_module._optimization_cache_key(
        unqueued,
        context,
        {"A"},
        1_000.0,
        index=0,
        quotes=quotes,
    )
    queued_key = replay_module._optimization_cache_key(
        queued,
        context,
        {"A"},
        1_000.0,
        index=0,
        quotes=quotes,
    )

    assert plain_key != queued_key


def test_same_day_transitive_raw_claim_chain_does_not_require_an_intermediate_quote():
    dates = ["2020-01-02", "2020-01-03"]
    adjusted = np.array([[10.0], [10.0]])
    first = package(
        "A-to-R",
        "A",
        dates[1],
        "stock",
        legs=[{"recipient_id": "R", "recipient_ticker": "R", "ratio": 2.0}],
    )
    second = package("R-to-cash", "R", dates[1], "cash", cash=3.0)
    result = replay_arrays(
        dates=dates,
        security_ids=["A"],
        adjusted=adjusted,
        raw=adjusted.copy(),
        dollar_volume=np.full_like(adjusted, 1_000_000.0),
        packages=[first, second],
        scenario_id="strict_standard_only",
        policy=execution_policy(),
        initial_units={"A": 10.0},
        starting_cash=0.0,
    )
    assert result["final_units"] == {}
    assert result["final_raw_claims"] == {}
    assert result["final_cash"] == 60.0
    assert [row["status"] for row in result["journal"] if row["package_id"]] == [
        "standard_resolved",
        "standard_resolved",
    ]
