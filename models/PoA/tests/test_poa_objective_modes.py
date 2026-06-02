import json
import math
import os
from pathlib import Path

import pytest
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_optimization import PoAOptimization


FEAS_TOL = 1e-5
OBJ_TOL = 1e-4
CASE = "base_test_case"
TEST_PROFILE = os.getenv("POA_OBJECTIVE_TEST_PROFILE", "full").strip().lower()
if TEST_PROFILE == "full":
    HORIZON = int(os.getenv("POA_OBJECTIVE_TEST_HORIZON", "6"))
    NN_MODEL_DIR = Path("models/neural_network/training/trained_models")
    NN_NORMALIZATION_STATS_PATH = Path(
        "models/neural_network/features/generated/normalized/min_max_stats.json"
    )
    NN_POLICY_GENERATORS = [1, 2]
    SOLVE_TIME_LIMIT = 400
else:
    HORIZON = int(os.getenv("POA_OBJECTIVE_TEST_HORIZON", "6"))
    NN_MODEL_DIR = None
    NN_NORMALIZATION_STATS_PATH = None
    NN_POLICY_GENERATORS = []
    SOLVE_TIME_LIMIT = 30
TIGHTENING_REPORT_PATH = Path("results/poa_tightening/final_tightening_report.json")
PHI_BOUNDS = (1.0, 5.0)
PIECEWISE_NUM_PIECES = 50
OPTIMAL_COST_BOUNDS_PATH = Path("test_outputs/optimal_cost_bounds_from_kkt.json")


def _require_gurobi_and_tightening_report(
    tightening_report_path=TIGHTENING_REPORT_PATH,
) -> None:
    if not pyo.SolverFactory("gurobi").available(exception_flag=False):
        pytest.skip("Gurobi solver is not available")
    if not Path(tightening_report_path).exists():
        pytest.skip(f"Tightening report not found: {tightening_report_path}")
    if NN_MODEL_DIR is not None and not NN_MODEL_DIR.exists():
        pytest.skip(f"NN model directory not found: {NN_MODEL_DIR}")
    if (
        NN_NORMALIZATION_STATS_PATH is not None
        and not NN_NORMALIZATION_STATS_PATH.exists()
    ):
        pytest.skip(
            f"NN normalization stats not found: {NN_NORMALIZATION_STATS_PATH}"
        )


@pytest.fixture(scope="module")
def poa_inputs():
    return _load_poa_inputs()


def _load_poa_inputs():
    scenario_manager = ScenarioManager(CASE)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set="PoA_analysis",
        seed=1,
    )
    ambiguity_set_config = PoAOptimization.load_ambiguity_set(
        config_path="config/ambiguity_set_config.yaml",
        config_name="base_test_case",
    )
    return scenarios, ambiguity_set_config


def _build_optimizer(
    poa_inputs,
    objective_mode="difference",
    ratio_bounds=None,
    tightening_report_path=TIGHTENING_REPORT_PATH,
):
    _require_gurobi_and_tightening_report(tightening_report_path)
    scenarios, ambiguity_set_config = poa_inputs
    optimizer = PoAOptimization(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=HORIZON,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=str(NN_MODEL_DIR) if NN_MODEL_DIR is not None else None,
        nn_normalization_stats_path=(
            str(NN_NORMALIZATION_STATS_PATH)
            if NN_NORMALIZATION_STATS_PATH is not None
            else None
        ),
        nn_policy_generators=list(NN_POLICY_GENERATORS),
        reference_case=CASE,
        objective_mode=objective_mode,
        ratio_bounds=ratio_bounds,
    )
    optimizer.load_tightening_report(tightening_report_path)
    optimizer.build_model()
    if NN_POLICY_GENERATORS:
        optimizer.apply_nn_relu_bounds_to_model()
    optimizer.apply_tightened_bounds_to_model()
    return optimizer


def _solve_or_skip(optimizer):
    try:
        results = optimizer.solve(time_limit=SOLVE_TIME_LIMIT)
    except Exception as exc:
        message = str(exc)
        if "Gurobi" in message or "gurobi" in message:
            pytest.skip(f"Gurobi solve failed in this environment: {message}")
        raise
    assert results.solver.termination_condition == TerminationCondition.optimal
    return optimizer.extract_objective_metrics()


class OptimalDispatchKKTBoundsComputer:
    OPTIMAL_KKT_BOUND_CONSTRAINTS = {
        # Demand support set.
        "demand_lower_bound_constraints",
        "demand_upper_bound_constraints",
        "demand_ramp_up_constraints",
        "demand_ramp_down_constraints",
        "demand_abs_deviation_pos_constraints",
        "demand_abs_deviation_neg_constraints",
        "demand_budget_constraint",
        "demand_lower_feasibility",
        # Wind/capacity support set.
        "conventional_capacity",
        "wind_total_lower_bound",
        "wind_total_upper_bound",
        "wind_even_block_split",
        "wind_ramp_up",
        "wind_ramp_down",
        "wind_abs_deviation_pos",
        "wind_abs_deviation_neg",
        "wind_budget_constraint",
        "wind_capacity_factor_lower_feasibility",
        "wind_capacity_factor_upper_feasibility",
        # True-cost optimal dispatch primal feasibility.
        "power_balance_opt",
        "generation_upper_opt",
        "generation_lower_opt",
        "ramp_up_opt",
        "ramp_up_initial_opt",
        "ramp_down_opt",
        "ramp_down_initial_opt",
        # True-cost optimal dispatch stationarity.
        "stationarity_opt",
        "final_ramp_up_dual_opt",
        "final_ramp_down_dual_opt",
        # True-cost optimal dispatch complementarity linearization.
        "upper_bound_complementarity_opt",
        "upper_bound_complementarity_dual_opt",
        "lower_bound_complementarity_opt",
        "lower_bound_complementarity_dual_opt",
        "ramp_up_complementarity_opt",
        "ramp_up_complementarity_dual_opt",
        "ramp_up_initial_complementarity_opt",
        "ramp_down_complementarity_opt",
        "ramp_down_complementarity_dual_opt",
        "ramp_down_initial_complementarity_opt",
        # Cost definition for the denominator.
        "cost_definition_opt",
    }

    def __init__(
        self,
        poa_inputs,
        tightening_report_path,
        phi_bounds=PHI_BOUNDS,
    ):
        self.poa_inputs = poa_inputs
        self.tightening_report_path = Path(tightening_report_path)
        self.phi_bounds = tuple(float(value) for value in phi_bounds)

    def _build_optimizer(self):
        optimizer = _build_optimizer(
            self.poa_inputs,
            objective_mode="difference",
            tightening_report_path=self.tightening_report_path,
        )
        self._extract_optimal_dispatch_kkt_block(optimizer.model)
        return optimizer

    def _extract_optimal_dispatch_kkt_block(self, model):
        deactivated = []
        kept = []
        for component in model.component_objects(pyo.Constraint, active=True):
            if component.local_name in self.OPTIMAL_KKT_BOUND_CONSTRAINTS:
                kept.append(component.local_name)
                continue
            component.deactivate()
            deactivated.append(component.local_name)
        self.last_constraint_extraction = {
            "kept_constraint_components": sorted(kept),
            "deactivated_constraint_components": sorted(deactivated),
            "kept_constraint_component_count": len(kept),
            "deactivated_constraint_component_count": len(deactivated),
        }

    def _get_c_opt_expression(self, optimizer):
        for method_name in (
            "_get_optimal_cost_expression",
            "get_optimal_cost_expression",
        ):
            method = getattr(optimizer, method_name, None)
            if method is None:
                continue
            try:
                return method()
            except TypeError:
                try:
                    return method(optimizer.model)
                except TypeError:
                    continue

        for attr_name in (
            "C_opt",
            "C_opt_expr",
            "optimal_cost",
            "optimal_cost_expr",
        ):
            if hasattr(optimizer.model, attr_name):
                return getattr(optimizer.model, attr_name)

        raise AttributeError(
            "PoA model must expose the optimal dispatch cost expression"
        )

    def _solve_bound(self, sense):
        if sense not in {"min", "max"}:
            raise ValueError("sense must be 'min' or 'max'")

        optimizer = self._build_optimizer()
        model = optimizer.model
        active_constraints_after_kkt_extraction = sum(
            1 for _ in model.component_data_objects(pyo.Constraint, active=True)
        )
        for objective in model.component_data_objects(pyo.Objective, active=True):
            objective.deactivate()

        c_opt_expression = self._get_c_opt_expression(optimizer)
        pyomo_sense = pyo.minimize if sense == "min" else pyo.maximize
        model.c_opt_bound_objective = pyo.Objective(
            expr=c_opt_expression,
            sense=pyomo_sense,
        )
        active_constraints_after = sum(
            1 for _ in model.component_data_objects(pyo.Constraint, active=True)
        )

        try:
            results = optimizer.solve(time_limit=SOLVE_TIME_LIMIT)
        except Exception as exc:
            message = str(exc)
            if "Gurobi" in message or "gurobi" in message:
                pytest.skip(f"Gurobi solve failed in this environment: {message}")
            raise

        termination_condition = results.solver.termination_condition
        if termination_condition != TerminationCondition.optimal:
            pytest.skip(
                f"C_opt {sense} bound solve did not terminate optimally: "
                f"{termination_condition}"
            )

        raw_bound = pyo.value(model.C_opt, exception=False)
        if raw_bound is None:
            raise ValueError(f"C_opt {sense} bound solve did not produce a value")
        return {
            "bound": float(raw_bound),
            "solver_status": str(results.solver.status),
            "termination_condition": str(termination_condition),
            "active_constraints_after_kkt_extraction": (
                active_constraints_after_kkt_extraction
            ),
            "active_constraints_after_objective_swap": active_constraints_after,
            "num_variables": int(model.nvariables()),
            "num_constraints": int(model.nconstraints()),
            "constraint_extraction": dict(self.last_constraint_extraction),
        }

    def compute_bounds(self):
        min_result = self._solve_bound("min")
        max_result = self._solve_bound("max")
        raw_lower = min_result["bound"]
        raw_upper = max_result["bound"]
        if not math.isfinite(raw_lower) or not math.isfinite(raw_upper):
            raise ValueError("Computed C_opt bounds must be finite")
        if raw_lower <= 0.0:
            raise ValueError("Computed C_opt lower bound must be strictly positive")
        if raw_upper < raw_lower:
            raise ValueError(
                "Computed C_opt upper bound must be greater than or equal to lower bound"
            )

        safe_lower = max(FEAS_TOL, raw_lower - abs(raw_lower) * 1e-6 - 1e-6)
        safe_upper = raw_upper + abs(raw_upper) * 1e-6 + 1e-6
        bounds = {
            "phi": self.phi_bounds,
            "C_opt": (safe_lower, safe_upper),
            "metadata": {
                "raw_C_opt_lower": raw_lower,
                "raw_C_opt_upper": raw_upper,
                "safe_C_opt_lower": safe_lower,
                "safe_C_opt_upper": safe_upper,
                "method": "poa_optimal_dispatch_kkt_bounds",
                "support_set_source": "regime_ambiguity_set",
                "test_profile": TEST_PROFILE,
                "horizon": HORIZON,
                "nn_policy_generators": list(NN_POLICY_GENERATORS),
                "active_constraints_min": min_result[
                    "active_constraints_after_objective_swap"
                ],
                "active_constraints_max": max_result[
                    "active_constraints_after_objective_swap"
                ],
                "num_variables_min": min_result["num_variables"],
                "num_variables_max": max_result["num_variables"],
                "num_constraints_min": min_result["num_constraints"],
                "num_constraints_max": max_result["num_constraints"],
                "constraint_extraction_min": min_result["constraint_extraction"],
                "constraint_extraction_max": max_result["constraint_extraction"],
                "solver_status_min": min_result["solver_status"],
                "termination_condition_min": min_result["termination_condition"],
                "solver_status_max": max_result["solver_status"],
                "termination_condition_max": max_result["termination_condition"],
            },
        }
        self.computed_bounds = bounds
        self.save_bounds(OPTIMAL_COST_BOUNDS_PATH)
        return bounds

    def save_bounds(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.computed_bounds, indent=2), encoding="utf-8")
        return path


@pytest.fixture(scope="module")
def difference_solution(poa_inputs):
    optimizer = _build_optimizer(poa_inputs, objective_mode="difference")
    return _solve_or_skip(optimizer)


@pytest.fixture(scope="module")
def computed_ratio_bounds(poa_inputs):
    computer = OptimalDispatchKKTBoundsComputer(
        poa_inputs=poa_inputs,
        tightening_report_path=TIGHTENING_REPORT_PATH,
        phi_bounds=PHI_BOUNDS,
    )
    return computer.compute_bounds()


@pytest.fixture(scope="module")
def ratio_solution(poa_inputs, computed_ratio_bounds):
    optimizer = _build_optimizer(
        poa_inputs,
        objective_mode="mccormick",
        ratio_bounds={
            "phi": computed_ratio_bounds["phi"],
            "C_opt": computed_ratio_bounds["C_opt"],
        },
    )
    return _solve_or_skip(optimizer)


@pytest.fixture(scope="module")
def ratio_piecewise_solution(poa_inputs, computed_ratio_bounds):
    optimizer = _build_optimizer(
        poa_inputs,
        objective_mode="piecewise_mccormick",
        ratio_bounds={
            "phi": computed_ratio_bounds["phi"],
            "C_opt": computed_ratio_bounds["C_opt"],
            "num_pieces": PIECEWISE_NUM_PIECES,
        },
    )
    return _solve_or_skip(optimizer)


def _assert_finite_metrics(metrics, keys):
    for key in keys:
        assert key in metrics
        assert metrics[key] is not None
        assert math.isfinite(metrics[key])


def _compact_objective(metrics):
    keys = [
        "C_eq",
        "C_opt",
        "difference_proxy",
        "ex_post_ratio",
        "objective_value",
        "phi",
        "z_ratio_product",
        "mccormick_product_gap",
        "ratio_gap",
        "active_piece",
        "active_piece_lower",
        "active_piece_upper",
        "num_pieces",
        "piecewise_product_gap",
        "piecewise_selected_delta_sum",
        "active_piece_delta_value",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _comparison_payload(
    difference_metrics,
    ratio_metrics,
    computed_ratio_bounds,
    piecewise_metrics=None,
):
    differences = {
        "delta_C_eq": ratio_metrics["C_eq"] - difference_metrics["C_eq"],
        "delta_C_opt": ratio_metrics["C_opt"] - difference_metrics["C_opt"],
        "delta_difference_proxy": (
            ratio_metrics["difference_proxy"]
            - difference_metrics["difference_proxy"]
        ),
        "delta_ex_post_ratio": (
            ratio_metrics["ex_post_ratio"] - difference_metrics["ex_post_ratio"]
        ),
        "phi_minus_ex_post_ratio": (
            ratio_metrics["phi"] - ratio_metrics["ex_post_ratio"]
        ),
    }
    payload = {
        "computed_ratio_bounds": {
            "phi": list(computed_ratio_bounds["phi"]),
            "C_opt": list(computed_ratio_bounds["C_opt"]),
            "metadata": computed_ratio_bounds["metadata"],
        },
        "difference_formulation": _compact_objective(difference_metrics),
        "mccormick_formulation": _compact_objective(ratio_metrics),
        "differences": differences,
    }
    if piecewise_metrics is not None:
        payload["piecewise_mccormick_formulation"] = _compact_objective(
            piecewise_metrics
        )
        payload["piecewise_gap_reduction"] = abs(ratio_metrics["ratio_gap"]) - abs(
            piecewise_metrics["ratio_gap"]
        )
    return payload


def _write_comparison_output(comparison, output_path=None):
    output_path = Path(output_path or "test_outputs/poa_objective_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return output_path


def _print_comparison_summary(
    difference_metrics,
    ratio_metrics,
    comparison_path,
    piecewise_metrics=None,
):
    print("\nPoA objective comparison")
    print(f"profile={TEST_PROFILE}, horizon={HORIZON}")
    print(f"comparison_json={comparison_path}")
    print("formulation,C_eq,C_opt,difference_proxy,ex_post_ratio,objective_value")
    rows = [
        ("difference", difference_metrics),
        ("mccormick", ratio_metrics),
    ]
    if piecewise_metrics is not None:
        rows.append(("piecewise_mccormick", piecewise_metrics))
    for name, metrics in rows:
        print(
            f"{name},{metrics['C_eq']:.6f},{metrics['C_opt']:.6f},"
            f"{metrics['difference_proxy']:.6f},"
            f"{metrics['ex_post_ratio']:.6f},{metrics['objective_value']:.6f}"
        )
    for name, metrics in rows:
        if "ratio_gap" in metrics and abs(metrics["ratio_gap"]) > 1e-3:
            print(
                f"WARNING: {name} ratio_gap={metrics['ratio_gap']:.6g}; "
                "check ex_post_ratio before interpreting phi."
            )


def test_difference_formulation_still_solves(difference_solution):
    metrics = difference_solution
    _assert_finite_metrics(
        metrics,
        ["C_eq", "C_opt", "difference_proxy", "ex_post_ratio"],
    )
    assert metrics["objective_mode"] == "difference"
    assert metrics["C_opt"] > 0
    assert metrics["difference_proxy"] == pytest.approx(
        metrics["C_eq"] - metrics["C_opt"],
        abs=OBJ_TOL,
    )
    assert metrics["ex_post_ratio"] == pytest.approx(
        metrics["C_eq"] / metrics["C_opt"],
        abs=FEAS_TOL,
    )


def test_computed_optimal_cost_bounds_are_valid(computed_ratio_bounds):
    C_opt_L, C_opt_U = computed_ratio_bounds["C_opt"]
    metadata = computed_ratio_bounds["metadata"]
    assert C_opt_L > 0
    assert C_opt_U >= C_opt_L
    assert OPTIMAL_COST_BOUNDS_PATH.exists()
    assert "raw_C_opt_lower" in metadata
    assert "raw_C_opt_upper" in metadata
    assert "safe_C_opt_lower" in metadata
    assert "safe_C_opt_upper" in metadata
    assert metadata["method"] == "poa_optimal_dispatch_kkt_bounds"
    assert metadata["active_constraints_min"] == metadata["num_constraints_min"]
    assert metadata["active_constraints_max"] == metadata["num_constraints_max"]
    assert metadata["horizon"] == HORIZON
    assert (
        metadata["constraint_extraction_min"]["deactivated_constraint_component_count"]
        > 0
    )
    assert (
        metadata["constraint_extraction_max"]["deactivated_constraint_component_count"]
        > 0
    )
    assert "power_balance_opt" in metadata["constraint_extraction_min"][
        "kept_constraint_components"
    ]
    assert "power_balance_eq" in metadata["constraint_extraction_min"][
        "deactivated_constraint_components"
    ]


def test_difference_solution_c_opt_within_computed_bounds(
    difference_solution,
    computed_ratio_bounds,
):
    C_opt_L, C_opt_U = computed_ratio_bounds["C_opt"]
    realized_C_opt = difference_solution["C_opt"]
    assert C_opt_L - FEAS_TOL <= realized_C_opt <= C_opt_U + FEAS_TOL


def test_mccormick_formulation_solves(ratio_solution, computed_ratio_bounds):
    metrics = ratio_solution
    phi_L, phi_U = computed_ratio_bounds["phi"]
    C_opt_L, C_opt_U = computed_ratio_bounds["C_opt"]
    _assert_finite_metrics(
        metrics,
        [
            "C_eq",
            "C_opt",
            "phi",
            "z_ratio_product",
            "ex_post_ratio",
            "mccormick_product_gap",
            "ratio_gap",
        ],
    )
    assert metrics["objective_mode"] == "mccormick"
    assert metrics["C_opt"] > 0
    assert phi_L - FEAS_TOL <= metrics["phi"] <= phi_U + FEAS_TOL
    assert C_opt_L - FEAS_TOL <= metrics["C_opt"] <= C_opt_U + FEAS_TOL
    assert metrics["z_ratio_product"] == pytest.approx(
        metrics["C_eq"],
        abs=FEAS_TOL,
    )
    assert metrics["ex_post_ratio"] == pytest.approx(
        metrics["C_eq"] / metrics["C_opt"],
        abs=FEAS_TOL,
    )
    assert math.isfinite(metrics["ratio_gap"])


def test_piecewise_mccormick_formulation_solves(
    ratio_piecewise_solution,
    computed_ratio_bounds,
):
    metrics = ratio_piecewise_solution
    phi_L, phi_U = computed_ratio_bounds["phi"]
    C_opt_L, C_opt_U = computed_ratio_bounds["C_opt"]
    active_slack_keys = [
        "active_mccormick_slack_lower_1",
        "active_mccormick_slack_lower_2",
        "active_mccormick_slack_upper_1",
        "active_mccormick_slack_upper_2",
    ]
    _assert_finite_metrics(
        metrics,
        [
            "C_eq",
            "C_opt",
            "phi",
            "z_ratio_product",
            "ex_post_ratio",
            "mccormick_product_gap",
            "ratio_gap",
            "active_piece",
            "active_piece_lower",
            "active_piece_upper",
            "num_pieces",
            "piecewise_product_gap",
            "piecewise_selected_delta_sum",
            "active_piece_delta_value",
            *active_slack_keys,
        ],
    )
    assert metrics["objective_mode"] == "piecewise_mccormick"
    assert metrics["C_opt"] > 0
    assert phi_L - FEAS_TOL <= metrics["phi"] <= phi_U + FEAS_TOL
    assert C_opt_L - FEAS_TOL <= metrics["C_opt"] <= C_opt_U + FEAS_TOL
    assert metrics["z_ratio_product"] == pytest.approx(
        metrics["C_eq"],
        abs=FEAS_TOL,
    )
    assert metrics["ex_post_ratio"] == pytest.approx(
        metrics["C_eq"] / metrics["C_opt"],
        abs=FEAS_TOL,
    )
    assert metrics["piecewise_selected_delta_sum"] == pytest.approx(1.0, abs=FEAS_TOL)
    assert metrics["active_piece_delta_value"] == pytest.approx(1.0, abs=FEAS_TOL)
    assert (
        metrics["active_piece_lower"] - FEAS_TOL
        <= metrics["C_opt"]
        <= metrics["active_piece_upper"] + FEAS_TOL
    )
    for key in active_slack_keys:
        assert metrics[key] >= -FEAS_TOL


def test_compare_difference_and_mccormick_formulations(
    difference_solution,
    ratio_solution,
    ratio_piecewise_solution,
    computed_ratio_bounds,
):
    _assert_finite_metrics(
        difference_solution,
        ["C_eq", "C_opt", "difference_proxy", "ex_post_ratio", "objective_value"],
    )
    _assert_finite_metrics(
        ratio_solution,
        [
            "C_eq",
            "C_opt",
            "difference_proxy",
            "ex_post_ratio",
            "phi",
            "z_ratio_product",
            "mccormick_product_gap",
            "ratio_gap",
            "objective_value",
        ],
    )
    _assert_finite_metrics(
        ratio_piecewise_solution,
        [
            "C_eq",
            "C_opt",
            "difference_proxy",
            "ex_post_ratio",
            "phi",
            "z_ratio_product",
            "mccormick_product_gap",
            "ratio_gap",
            "objective_value",
        ],
    )

    comparison = _comparison_payload(
        difference_solution,
        ratio_solution,
        computed_ratio_bounds,
        ratio_piecewise_solution,
    )
    differences = comparison["differences"]
    _assert_finite_metrics(differences, list(differences.keys()))
    assert math.isfinite(comparison["piecewise_gap_reduction"])

    output_path = _write_comparison_output(comparison)
    _print_comparison_summary(
        difference_solution,
        ratio_solution,
        output_path,
        ratio_piecewise_solution,
    )


@pytest.mark.parametrize(
    ("ratio_bounds", "message"),
    [
        (None, "ratio_bounds is required"),
        ({"phi": (0.0, 5.0), "C_opt": (0.0, 10000.0)}, "strictly positive"),
        (
            {"phi": (0.0, 5.0), "C_opt": (10000.0, 1.0)},
            "greater than or equal",
        ),
        ({"phi": (1.0, 1.0), "C_opt": (1.0, 10000.0)}, "greater than"),
        ({"phi": (-1.0, 5.0), "C_opt": (1.0, 10000.0)}, "nonnegative"),
    ],
)
def test_invalid_ratio_bounds_raise_clear_value_error(poa_inputs, ratio_bounds, message):
    scenarios, ambiguity_set_config = poa_inputs
    with pytest.raises(ValueError, match=message):
        PoAOptimization(
            scenarios_df=scenarios["scenarios_df"],
            costs_df=scenarios["costs_df"],
            ramps_df=scenarios["ramps_df"],
            num_time_steps=HORIZON,
            ambiguity_set_config=ambiguity_set_config,
            nn_model_dir=None,
            nn_policy_generators=[],
            reference_case=CASE,
            objective_mode="mccormick",
            ratio_bounds=ratio_bounds,
        )


@pytest.mark.parametrize(
    ("ratio_bounds", "message"),
    [
        (None, "ratio_bounds is required"),
        ({"C_opt": (1.0, 100.0), "num_pieces": 4}, "must contain bounds for"),
        ({"phi": PHI_BOUNDS, "num_pieces": 4}, "must contain bounds for"),
        (
            {"phi": PHI_BOUNDS, "C_opt": (0.0, 100.0), "num_pieces": 4},
            "strictly positive",
        ),
        (
            {"phi": PHI_BOUNDS, "C_opt": (100.0, 1.0), "num_pieces": 4},
            "greater than or equal",
        ),
        (
            {"phi": (2.0, 2.0), "C_opt": (1.0, 100.0), "num_pieces": 4},
            "greater than",
        ),
        (
            {"phi": PHI_BOUNDS, "C_opt": (1.0, 100.0), "num_pieces": 1},
            "at least 2",
        ),
        (
            {
                "phi": PHI_BOUNDS,
                "C_opt": (1.0, 100.0),
                "C_opt_breakpoints": [1.0, 50.0, 50.0, 100.0],
            },
            "strictly increasing",
        ),
        (
            {
                "phi": PHI_BOUNDS,
                "C_opt": (1.0, 100.0),
                "C_opt_breakpoints": [2.0, 50.0, 100.0],
            },
            "must match",
        ),
        (
            {
                "phi": PHI_BOUNDS,
                "C_opt": (1.0, 100.0),
                "C_opt_breakpoints": [1.0, 50.0, 99.0],
            },
            "must match",
        ),
    ],
)
def test_invalid_piecewise_ratio_bounds_raise_clear_value_error(
    poa_inputs,
    ratio_bounds,
    message,
):
    scenarios, ambiguity_set_config = poa_inputs
    with pytest.raises(ValueError, match=message):
        PoAOptimization(
            scenarios_df=scenarios["scenarios_df"],
            costs_df=scenarios["costs_df"],
            ramps_df=scenarios["ramps_df"],
            num_time_steps=HORIZON,
            ambiguity_set_config=ambiguity_set_config,
            nn_model_dir=None,
            nn_policy_generators=[],
            reference_case=CASE,
            objective_mode="piecewise_mccormick",
            ratio_bounds=ratio_bounds,
        )


def test_difference_mode_does_not_require_ratio_bounds(poa_inputs):
    scenarios, ambiguity_set_config = poa_inputs
    optimizer = PoAOptimization(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=HORIZON,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=None,
        nn_policy_generators=[],
        reference_case=CASE,
        objective_mode="difference",
    )
    assert optimizer.objective_mode == "difference"


def main():
    poa_inputs = _load_poa_inputs()
    computer = OptimalDispatchKKTBoundsComputer(
        poa_inputs=poa_inputs,
        tightening_report_path=TIGHTENING_REPORT_PATH,
        phi_bounds=PHI_BOUNDS,
    )
    computed_ratio_bounds = computer.compute_bounds()

    difference_optimizer = _build_optimizer(
        poa_inputs,
        objective_mode="difference",
    )
    difference_metrics = _solve_or_skip(difference_optimizer)

    ratio_optimizer = _build_optimizer(
        poa_inputs,
        objective_mode="mccormick",
        ratio_bounds={
            "phi": computed_ratio_bounds["phi"],
            "C_opt": computed_ratio_bounds["C_opt"],
        },
    )
    ratio_metrics = _solve_or_skip(ratio_optimizer)

    piecewise_optimizer = _build_optimizer(
        poa_inputs,
        objective_mode="piecewise_mccormick",
        ratio_bounds={
            "phi": computed_ratio_bounds["phi"],
            "C_opt": computed_ratio_bounds["C_opt"],
            "num_pieces": PIECEWISE_NUM_PIECES,
        },
    )
    piecewise_metrics = _solve_or_skip(piecewise_optimizer)

    comparison = _comparison_payload(
        difference_metrics,
        ratio_metrics,
        computed_ratio_bounds,
        piecewise_metrics,
    )
    comparison_path = _write_comparison_output(comparison)
    _print_comparison_summary(
        difference_metrics,
        ratio_metrics,
        comparison_path,
        piecewise_metrics,
    )
    print(f"bounds_json={OPTIMAL_COST_BOUNDS_PATH}")
    return comparison


if __name__ == "__main__":
    main()
