from pathlib import Path
import sys
from typing import Any, Optional
from pyomo.environ import *
import numpy as np
import time
import copy
import json
import statistics
from config.scenarios.scenario_generator import ScenarioManager

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _candidate in Path(__file__).resolve().parents:
    if (_candidate / "pyproject.toml").exists():
        _REPO_ROOT = _candidate
        break
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.PoA.PoA_optimization import PoAOptimization

DIAGNOSTIC_ABS_OBJ_TOL = 1e-5
DIAGNOSTIC_REL_OBJ_TOL = 1e-4
DIAGNOSTIC_VIOLATION_TOL = 1e-6
DIAGNOSTIC_LARGE_RESIDUAL_TOL = 1e-4
DEFAULT_PHI_BOUNDS = (1.0, 5.0)
DEFAULT_PIECEWISE_NUM_PIECES = 50
DEFAULT_USE_LAMBDA_BOUNDS = True
DEFAULT_USE_AGGREGATE_DUAL_BOUNDS = True
DEFAULT_RUN_POA_TIGHTENING = False
DEFAULT_POA_TIGHTENING_FLAGS = {
    "primal_big_m": True,
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
    "optimal_cost_bounds": True,
}
FACTORIAL_TIGHTENING_COMPONENTS = [
    "nn_relu_bounds_report",
    "alpha_bounds",
    "fixed_binaries",
    "tight_big_m",
    "lambda_bounds",
    "aggregate_dual_bounds",
]
FACTORIAL_POST_BUILD_COMPONENTS = {
    "alpha_bounds",
    "fixed_binaries",
    "tight_big_m",
    "lambda_bounds",
    "aggregate_dual_bounds",
}

def _objective_close(
    left: Optional[float],
    right: Optional[float],
    abs_tol: float = DIAGNOSTIC_ABS_OBJ_TOL,
    rel_tol: float = DIAGNOSTIC_REL_OBJ_TOL,
) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= max(
        float(abs_tol),
        float(rel_tol) * max(1.0, abs(float(left)), abs(float(right))),
    )


def _objective_materially_greater(
    left: Optional[float],
    right: Optional[float],
    abs_tol: float = DIAGNOSTIC_ABS_OBJ_TOL,
    rel_tol: float = DIAGNOSTIC_REL_OBJ_TOL,
) -> bool:
    if left is None or right is None:
        return False
    tolerance = max(
        float(abs_tol),
        float(rel_tol) * max(1.0, abs(float(left)), abs(float(right))),
    )
    return float(left) > float(right) + tolerance


def _is_successful_termination(variant_result: dict[str, Any]) -> bool:
    if variant_result.get("error"):
        return False
    termination = str(variant_result.get("termination_condition", "")).lower()
    status = str(variant_result.get("solver_status", "")).lower()
    return termination in {"optimal", "globallyoptimal", "locallyoptimal"} and (
        not status or status in {"ok", "warning"}
    )


def _json_ready_index(index: Any) -> str:
    if index is None:
        return ""
    if isinstance(index, str):
        return index
    if isinstance(index, tuple):
        return PoAOptimization._json_key(tuple(int(part) for part in index))
    return str(int(index))


def _model_size(model: ConcreteModel) -> dict[str, int]:
    variables = list(model.component_data_objects(Var, active=True))
    constraints = list(model.component_data_objects(Constraint, active=True))
    return {
        "num_variables": int(len(variables)),
        "num_binary_variables": int(sum(1 for var in variables if var.is_binary())),
        "num_constraints": int(len(constraints)),
        "active_constraints": int(len(constraints)),
    }


def _extract_nested_value(payload: Any, path: tuple[str, ...]) -> Any:
    current = payload
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        try:
            current = getattr(current, key)
            continue
        except Exception:
            pass
        try:
            current = current[key]
            continue
        except Exception:
            return None
    return current


def _optional_float_value(payload: Any) -> Optional[float]:
    if payload is None:
        return None
    try:
        value = float(payload)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _extract_solver_statistics(optimizer: PoAOptimization) -> dict[str, Any]:
    results = getattr(optimizer, "solver_results", None)
    solver = getattr(results, "solver", None) if results is not None else None
    statistics_payload = (
        getattr(results, "statistics", None) if results is not None else None
    )
    candidates: dict[str, tuple[tuple[str, ...], ...]] = {
        "mip_gap": (
            ("mip_gap",),
            ("gap",),
            ("MIPGap",),
            ("solver", "mip_gap"),
            ("solver", "gap"),
        ),
        "best_bound": (
            ("best_bound",),
            ("BestBd",),
            ("best_objective_bound",),
            ("solver", "best_bound"),
        ),
        "node_count": (
            ("node_count",),
            ("NodeCount",),
            ("branch_and_bound", "number_of_created_subproblems"),
            ("solver", "node_count"),
        ),
        "simplex_iterations": (
            ("simplex_iterations",),
            ("iteration_count",),
            ("IterationCount",),
            ("solver", "simplex_iterations"),
        ),
        "barrier_iterations": (
            ("barrier_iterations",),
            ("BarIterCount",),
            ("solver", "barrier_iterations"),
        ),
    }
    roots = tuple(root for root in (solver, statistics_payload, results) if root is not None)
    extracted: dict[str, Any] = {}
    for statistic_name, paths in candidates.items():
        extracted_value = None
        for root in roots:
            for path in paths:
                extracted_value = _optional_float_value(
                    _extract_nested_value(root, path)
                )
                if extracted_value is not None:
                    break
            if extracted_value is not None:
                break
        extracted[statistic_name] = extracted_value
    return extracted


def _variant_metrics(optimizer: PoAOptimization, runtime: float) -> dict[str, Any]:
    objective = optimizer.extract_objective_metrics()
    solver = getattr(optimizer, "solver_results", None)
    solver_status = str(solver.solver.status) if solver is not None else None
    termination = (
        str(solver.solver.termination_condition) if solver is not None else None
    )
    solver_statistics = _extract_solver_statistics(optimizer)
    m = optimizer.model
    return {
        "objective_value": objective.get("objective_value"),
        "C_eq": objective.get("C_eq"),
        "C_opt": objective.get("C_opt"),
        "PoA": objective.get("PoA_difference"),
        "PoA_ratio": objective.get("PoA_ratio"),
        "phi": objective.get("phi"),
        "mu_D": optimizer._safe_value(m.mu_D),
        "sigma_D": optimizer._safe_value(m.sigma_D),
        "mu_W": optimizer._safe_value(m.mu_W),
        "sigma_W": optimizer._safe_value(m.sigma_W),
        "solver_status": solver_status,
        "termination_condition": termination,
        "runtime": float(runtime),
        "model_size": _model_size(m),
        "mip_gap": solver_statistics.get("mip_gap"),
        "best_bound": solver_statistics.get("best_bound"),
        "node_count": solver_statistics.get("node_count"),
        "simplex_iterations": solver_statistics.get("simplex_iterations"),
        "barrier_iterations": solver_statistics.get("barrier_iterations"),
        "solver_statistics": solver_statistics,
        "objective_metrics": objective,
        "applied_nn_relu_stats": getattr(optimizer, "applied_nn_relu_stats", {}),
        "applied_tightening_stats": getattr(optimizer, "applied_tightening_stats", {}),
        "default_bounds_used": getattr(optimizer, "default_bounds_used", {}),
    }


def _clone_report_without_keys(report: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    cloned = copy.deepcopy(report)
    for key in keys:
        cloned.pop(key, None)
    return cloned


def _strip_aggregate_dual_bounds(report: dict[str, Any]) -> dict[str, Any]:
    cloned = _clone_report_without_keys(report, ("aggregate_dual_bounds",))
    aggregate_keys = {
        "mu_max_sum_ub",
        "mu_min_sum_ub",
        "mu_ramp_up_sum_ub",
        "mu_ramp_down_sum_ub",
    }
    tight_big_m = cloned.get("tight_big_m")
    if isinstance(tight_big_m, dict):
        for key in aggregate_keys:
            tight_big_m.pop(key, None)
    return cloned


def _load_ratio_bounds_from_tightening_report(
    tightening_report_path: str | Path,
    phi_bounds: tuple[float, float] = DEFAULT_PHI_BOUNDS,
    num_pieces: int = DEFAULT_PIECEWISE_NUM_PIECES,
) -> dict[str, Any]:
    with Path(tightening_report_path).open("r", encoding="utf-8") as file_handle:
        report = json.load(file_handle)

    raw_bounds = report.get("optimal_cost_bounds", {}) or {}
    if "C_opt" in raw_bounds and isinstance(raw_bounds.get("C_opt"), dict):
        c_opt_payload = raw_bounds.get("C_opt", {}) or {}
    else:
        c_opt_payload = raw_bounds

    lower = c_opt_payload.get("lower")
    upper = c_opt_payload.get("upper")
    if lower is None or upper is None:
        raise ValueError(
            "Piecewise McCormick diagnostics require optimal_cost_bounds with "
            f"finite lower/upper values in {tightening_report_path}."
        )

    lower = float(lower)
    upper = float(upper)
    if lower <= 0.0 or upper < lower:
        raise ValueError(
            "Invalid optimal_cost_bounds for ratio objective: "
            f"lower={lower}, upper={upper}."
        )

    return {
        "phi": tuple(float(value) for value in phi_bounds),
        "C_opt": (lower, upper),
        "num_pieces": int(num_pieces),
    }


def _base_tightening_report_for_diagnostic(
    report: dict[str, Any],
    use_lambda_bounds: bool = DEFAULT_USE_LAMBDA_BOUNDS,
    use_aggregate_dual_bounds: bool = DEFAULT_USE_AGGREGATE_DUAL_BOUNDS,
) -> dict[str, Any]:
    base_report = copy.deepcopy(report)
    if not use_lambda_bounds:
        base_report.pop("lambda_bounds", None)
    if not use_aggregate_dual_bounds:
        base_report = _strip_aggregate_dual_bounds(base_report)
    return base_report


def _generate_factorial_tightening_configurations(
    components: list[str] = FACTORIAL_TIGHTENING_COMPONENTS,
) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    num_components = len(components)
    full_mask = (1 << num_components) - 1
    for mask in range(1 << num_components):
        enabled_components = [
            component
            for component_idx, component in enumerate(components)
            if mask & (1 << component_idx)
        ]
        disabled_components = [
            component for component in components if component not in enabled_components
        ]
        suffix = "__".join(enabled_components)
        if not suffix:
            suffix = "none"
        if mask == full_mask:
            suffix = "full"
        configurations.append(
            {
                "variant": f"factorial_{mask:03d}_{suffix}",
                "enabled_components": enabled_components,
                "disabled_components": disabled_components,
                "component_flags": {
                    component: component in enabled_components
                    for component in components
                },
                "mask": int(mask),
            }
        )
    return configurations


def _filter_tightening_report_for_components(
    full_tightening_report: dict[str, Any],
    enabled_components: set[str],
) -> dict[str, Any]:
    # TODO(PoA cleanup): This is the old leave-one-out filter shape. Prefer
    # _construct_factorial_tightening_report(...) for new diagnostics so disabled
    # components mean "use PoAOptimization defaults", not "mutate a full report".
    filtered_report = copy.deepcopy(full_tightening_report)
    disabled_components = set(FACTORIAL_TIGHTENING_COMPONENTS) - set(enabled_components)

    if "nn_relu_bounds_report" in disabled_components:
        filtered_report.pop("nn_relu_bounds_report", None)
        filtered_report.pop("nn_relu_bounds", None)

    if "alpha_bounds" in disabled_components:
        filtered_report.pop("alpha_bounds", None)
        filtered_report.pop("alpha_optimization_results", None)

    if "fixed_binaries" in disabled_components:
        filtered_report.pop("fixed_binaries", None)
        filtered_report.pop("num_fixed_binaries", None)

    if "tight_big_m" in disabled_components:
        filtered_report.pop("tight_big_m", None)

    if "lambda_bounds" in disabled_components:
        filtered_report.pop("lambda_bounds", None)

    if "aggregate_dual_bounds" in disabled_components:
        filtered_report = _strip_aggregate_dual_bounds(filtered_report)

    return filtered_report


def _construct_factorial_tightening_report(
    full_tightening_report: dict[str, Any],
    enabled_components: set[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if full_tightening_report.get("metadata"):
        report["metadata"] = copy.deepcopy(full_tightening_report["metadata"])
    if full_tightening_report.get("primal_big_m"):
        report["primal_big_m"] = copy.deepcopy(full_tightening_report["primal_big_m"])
    if full_tightening_report.get("optimal_cost_bounds"):
        report["optimal_cost_bounds"] = copy.deepcopy(
            full_tightening_report["optimal_cost_bounds"]
        )
    if full_tightening_report.get("optimal_cost_bound_optimization_results"):
        report["optimal_cost_bound_optimization_results"] = copy.deepcopy(
            full_tightening_report["optimal_cost_bound_optimization_results"]
        )

    if "nn_relu_bounds_report" in enabled_components:
        if full_tightening_report.get("nn_relu_bounds_report"):
            report["nn_relu_bounds_report"] = copy.deepcopy(
                full_tightening_report["nn_relu_bounds_report"]
            )
        if full_tightening_report.get("nn_relu_bounds"):
            report["nn_relu_bounds"] = copy.deepcopy(
                full_tightening_report["nn_relu_bounds"]
            )

    if "alpha_bounds" in enabled_components:
        if full_tightening_report.get("alpha_bounds"):
            report["alpha_bounds"] = copy.deepcopy(full_tightening_report["alpha_bounds"])
        if full_tightening_report.get("alpha_optimization_results"):
            report["alpha_optimization_results"] = copy.deepcopy(
                full_tightening_report["alpha_optimization_results"]
            )

    if "fixed_binaries" in enabled_components and full_tightening_report.get(
        "fixed_binaries"
    ):
        report["fixed_binaries"] = copy.deepcopy(full_tightening_report["fixed_binaries"])

    if "tight_big_m" in enabled_components and full_tightening_report.get("tight_big_m"):
        report["tight_big_m"] = copy.deepcopy(full_tightening_report["tight_big_m"])
        if "aggregate_dual_bounds" not in enabled_components:
            report = _strip_aggregate_dual_bounds(report)

    if "lambda_bounds" in enabled_components and full_tightening_report.get(
        "lambda_bounds"
    ):
        report["lambda_bounds"] = copy.deepcopy(full_tightening_report["lambda_bounds"])

    if "aggregate_dual_bounds" in enabled_components:
        if full_tightening_report.get("aggregate_dual_bounds"):
            report["aggregate_dual_bounds"] = copy.deepcopy(
                full_tightening_report["aggregate_dual_bounds"]
            )
        if "tight_big_m" in enabled_components and full_tightening_report.get(
            "tight_big_m"
        ):
            report.setdefault(
                "tight_big_m",
                copy.deepcopy(full_tightening_report["tight_big_m"]),
            )

    return report


def _present_tightening_components(report: dict[str, Any]) -> list[str]:
    present_components = []
    for component in FACTORIAL_TIGHTENING_COMPONENTS:
        if component == "nn_relu_bounds_report":
            if report.get("nn_relu_bounds_report") or report.get("nn_relu_bounds"):
                present_components.append(component)
        elif report.get(component):
            present_components.append(component)
    return present_components


def _base_report_components(report: dict[str, Any]) -> list[str]:
    return [
        component
        for component in ("primal_big_m", "optimal_cost_bounds")
        if report.get(component)
    ]


def run_poa_tightening_for_tests(
    scenarios_df,
    costs_df,
    ramps_df,
    ambiguity_set_config: dict[str, Any],
    horizon: int,
    case: str,
    nn_model_dir: str | Path,
    nn_normalization_stats_path: str | Path,
    nn_policy_generators: list[int | str],
    tightening_flags: Optional[dict[str, bool]] = None,
    time_limit: Optional[float] = 400,
    parallel_workers: Optional[int] = 1,
    solver_threads: Optional[int] = None,
    tee: bool = False,
    use_default_stage_inputs: bool = True,
) -> Path:
    from models.PoA.PoA_tightening.tightening_main import (
        DEFAULT_TIGHTENING_OUTPUT_PATHS,
        PoATighteningMain,
    )

    flags = {**DEFAULT_POA_TIGHTENING_FLAGS, **(tightening_flags or {})}
    tightening = PoATighteningMain(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        p_init=None,
        num_time_steps=horizon,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=nn_model_dir,
        nn_normalization_stats_path=nn_normalization_stats_path,
        nn_policy_generators=nn_policy_generators,
        reference_case=case,
    )
    return tightening.run_all(
        run_primal_big_m=bool(flags["primal_big_m"]),
        run_relu_bounds=bool(flags["relu_bounds"]),
        run_alpha_bounds=bool(flags["alpha_bounds"]),
        run_slack_binary_fix=bool(flags["slack_binary_fix"]),
        run_dual_big_m=bool(flags["dual_big_m"]),
        run_optimal_cost_bounds=bool(flags["optimal_cost_bounds"]),
        output_paths=DEFAULT_TIGHTENING_OUTPUT_PATHS,
        previous_paths=DEFAULT_TIGHTENING_OUTPUT_PATHS,
        solver_name="gurobi",
        time_limit=time_limit,
        tee=tee,
        parallel_workers=parallel_workers,
        solver_threads=solver_threads,
        use_default_stage_inputs=use_default_stage_inputs,
    )


def _apply_diagnostic_variant_data(
    optimizer: PoAOptimization,
    variant_name: str,
    full_tightening_report: dict[str, Any],
    tightening_report_path: str | Path,
) -> dict[str, Any]:
    # TODO(PoA cleanup): Migrate this legacy leave-one-out diagnostic to the same
    # component-set API used by the factorial diagnostic. It still mutates
    # optimizer internals and computes primal defaults outside PoAOptimization.
    if variant_name == "untightened_baseline":
        from models.PoA.PoA_tightening.compute_primal_big_m import (
            compute_primal_big_m_bounds,
        )

        optimizer.primal_big_m = compute_primal_big_m_bounds(optimizer)
        optimizer.tightening_report = {
            "primal_big_m": optimizer.primal_big_m,
            "diagnostic_note": "analytic_primal_big_m_only",
        }
        optimizer._prepare_loaded_bounds()
        return {
            "loaded_tightening_components": ["primal_big_m"],
            "applies_post_build_tightening": False,
        }

    if variant_name == "tightened_full":
        variant_report = copy.deepcopy(full_tightening_report)
    elif variant_name == "no_fixed_binaries":
        variant_report = _clone_report_without_keys(
            full_tightening_report,
            ("fixed_binaries",),
        )
    elif variant_name == "loose_dual_big_m":
        variant_report = _clone_report_without_keys(
            full_tightening_report,
            ("tight_big_m",),
        )
    elif variant_name == "loose_lambda_bounds":
        variant_report = _clone_report_without_keys(
            full_tightening_report,
            ("lambda_bounds",),
        )
    elif variant_name == "no_aggregate_dual_bounds":
        variant_report = _strip_aggregate_dual_bounds(full_tightening_report)
    elif variant_name == "loose_alpha_bounds":
        variant_report = _clone_report_without_keys(
            full_tightening_report,
            ("alpha_bounds", "alpha_optimization_results"),
        )
    else:
        raise ValueError(f"Unknown diagnostic variant: {variant_name}")

    optimizer._set_tightening_report_data(variant_report, Path(tightening_report_path))
    optimizer._prepare_loaded_bounds()
    return {
        "loaded_tightening_components": sorted(
            key
            for key in (
                "primal_big_m",
                "nn_relu_bounds_report",
                "alpha_bounds",
                "fixed_binaries",
                "tight_big_m",
                "lambda_bounds",
                "aggregate_dual_bounds",
                "optimal_cost_bounds",
            )
            if variant_report.get(key)
        ),
        "applies_post_build_tightening": True,
        "tightening_report": variant_report,
    }


def _solve_diagnostic_variant(
    variant_name: str,
    optimizer_kwargs: dict[str, Any],
    full_tightening_report: dict[str, Any],
    nn_relu_report_path: str | Path,
    tightening_report_path: str | Path,
    time_limit: Optional[float],
) -> tuple[dict[str, Any], Optional[PoAOptimization]]:
    # TODO(PoA cleanup): Remove the separate ReLU-loading branch once the
    # leave-one-out diagnostic uses ensure_default_bounds_available(...) like the
    # factorial diagnostic.
    start = time.perf_counter()
    optimizer: Optional[PoAOptimization] = None
    try:
        optimizer = PoAOptimization(**optimizer_kwargs)
        if optimizer.nn_policy_generator_ids:
            if Path(nn_relu_report_path).exists():
                optimizer.load_nn_relu_bounds_report(nn_relu_report_path)
            else:
                nn_relu_report = (
                    full_tightening_report.get("nn_relu_bounds_report", {}) or {}
                )
                if not nn_relu_report and "nn_relu_bounds" in full_tightening_report:
                    nn_relu_report = full_tightening_report
                if not nn_relu_report:
                    raise FileNotFoundError(
                        f"NN ReLU bounds report not found: {nn_relu_report_path}"
                    )
                optimizer._set_nn_relu_bounds_from_report(nn_relu_report)

        variant_setup = _apply_diagnostic_variant_data(
            optimizer,
            variant_name,
            full_tightening_report,
            tightening_report_path,
        )
        build_start = time.perf_counter()
        optimizer.build_model()
        build_runtime = time.perf_counter() - build_start

        if optimizer.nn_policy_generator_ids:
            optimizer.apply_nn_relu_bounds_to_model()

        if variant_setup.get("applies_post_build_tightening"):
            optimizer.apply_tightened_bounds_to_model()

        solve_start = time.perf_counter()
        optimizer.solve(time_limit=time_limit)
        solve_runtime = time.perf_counter() - solve_start

        metrics = _variant_metrics(optimizer, time.perf_counter() - start)
        metrics.update(
            {
                "variant": variant_name,
                "build_runtime": float(build_runtime),
                "solve_runtime": float(solve_runtime),
                "loaded_tightening_components": variant_setup[
                    "loaded_tightening_components"
                ],
            }
        )
        return metrics, optimizer
    except Exception as exc:
        runtime = time.perf_counter() - start
        result = {
            "variant": variant_name,
            "objective_value": None,
            "C_eq": None,
            "C_opt": None,
            "PoA": None,
            "PoA_ratio": None,
            "phi": None,
            "mu_D": None,
            "sigma_D": None,
            "mu_W": None,
            "sigma_W": None,
            "solver_status": None,
            "termination_condition": None,
            "runtime": float(runtime),
            "model_size": _model_size(optimizer.model)
            if optimizer is not None and hasattr(optimizer, "model")
            else {},
            "error": f"{type(exc).__name__}: {exc}",
        }
        return result, optimizer


def _apply_factorial_variant_data(
    optimizer: PoAOptimization,
    variant_config: dict[str, Any],
    full_tightening_report: dict[str, Any],
    tightening_report_path: str | Path,
) -> dict[str, Any]:
    enabled_components = set(variant_config.get("enabled_components", []))
    tightened_variant_report = _construct_factorial_tightening_report(
        full_tightening_report,
        enabled_components,
    )

    if tightened_variant_report:
        optimizer._set_tightening_report_data(
            tightened_variant_report,
            Path(tightening_report_path),
        )
        if tightened_variant_report.get("primal_big_m"):
            optimizer._prepare_loaded_bounds()

    optimizer.ensure_default_bounds_available(
        template_report=full_tightening_report,
        include_nn_relu_bounds=bool(optimizer.nn_policy_generator_ids),
        include_tight_big_m=True,
        include_optimal_cost_bounds=(
            optimizer.objective_mode == "piecewise_mccormick"
        ),
        overwrite_existing=False,
    )

    enabled_present = _present_tightening_components(tightened_variant_report)
    return {
        "loaded_tightening_components": sorted(
            _base_report_components(tightened_variant_report) + enabled_present
        ),
        "enabled_components": list(variant_config.get("enabled_components", [])),
        "disabled_components": list(variant_config.get("disabled_components", [])),
        "component_flags": dict(variant_config.get("component_flags", {})),
        "applies_post_build_tightening": bool(
            enabled_components & FACTORIAL_POST_BUILD_COMPONENTS
        ),
        "tightening_report": tightened_variant_report,
    }


def _load_factorial_nn_relu_bounds_if_enabled(
    optimizer: PoAOptimization,
    enabled_components: set[str],
    full_tightening_report: dict[str, Any],
    nn_relu_report_path: str | Path,
) -> None:
    # TODO(PoA cleanup): Delete this compatibility helper after the factorial
    # diagnostic has been stable for a few runs. ReLU bounds now flow through
    # _construct_factorial_tightening_report(...) plus optimizer defaults.
    if "nn_relu_bounds_report" not in enabled_components:
        return
    if not optimizer.nn_policy_generator_ids:
        return
    if Path(nn_relu_report_path).exists():
        optimizer.load_nn_relu_bounds_report(nn_relu_report_path)
        return
    nn_relu_report = full_tightening_report.get("nn_relu_bounds_report", {}) or {}
    if not nn_relu_report and "nn_relu_bounds" in full_tightening_report:
        nn_relu_report = full_tightening_report
    if not nn_relu_report:
        raise FileNotFoundError(f"NN ReLU bounds report not found: {nn_relu_report_path}")
    optimizer._set_nn_relu_bounds_from_report(nn_relu_report)


def _solve_factorial_tightening_variant(
    variant_config: dict[str, Any],
    optimizer_kwargs: dict[str, Any],
    full_tightening_report: dict[str, Any],
    nn_relu_report_path: str | Path,
    tightening_report_path: str | Path,
    time_limit: Optional[float],
) -> tuple[dict[str, Any], Optional[PoAOptimization]]:
    start = time.perf_counter()
    optimizer: Optional[PoAOptimization] = None
    variant_name = str(variant_config["variant"])
    enabled_components = set(variant_config.get("enabled_components", []))
    try:
        optimizer = PoAOptimization(**{**optimizer_kwargs, "use_default_bounds": True})

        variant_setup = _apply_factorial_variant_data(
            optimizer,
            variant_config,
            full_tightening_report,
            tightening_report_path,
        )

        build_start = time.perf_counter()
        optimizer.build_model()
        build_runtime = time.perf_counter() - build_start

        if optimizer.nn_policy_generator_ids:
            optimizer.apply_nn_relu_bounds_to_model()

        if variant_setup.get("applies_post_build_tightening"):
            optimizer.apply_tightened_bounds_to_model(
                apply_alpha_bounds="alpha_bounds" in enabled_components,
                apply_fixed_binaries="fixed_binaries" in enabled_components,
                apply_dual_bounds=bool(
                    enabled_components
                    & {"tight_big_m", "lambda_bounds", "aggregate_dual_bounds"}
                ),
                apply_lambda_bounds="lambda_bounds" in enabled_components,
                apply_dual_big_m_bounds="tight_big_m" in enabled_components,
                apply_aggregate_dual_bounds=(
                    "aggregate_dual_bounds" in enabled_components
                ),
            )

        solve_start = time.perf_counter()
        optimizer.solve(time_limit=time_limit)
        solve_runtime = time.perf_counter() - solve_start

        metrics = _variant_metrics(optimizer, time.perf_counter() - start)
        metrics.update(
            {
                "variant": variant_name,
                "enabled_components": list(variant_config["enabled_components"]),
                "disabled_components": list(variant_config["disabled_components"]),
                "component_flags": dict(variant_config["component_flags"]),
                "mask": int(variant_config["mask"]),
                "build_runtime": float(build_runtime),
                "solve_runtime": float(solve_runtime),
                "loaded_tightening_components": variant_setup[
                    "loaded_tightening_components"
                ],
                "default_components_used": [
                    key
                    for key, used in getattr(
                        optimizer,
                        "default_bounds_used",
                        {},
                    ).items()
                    if key != "notes" and bool(used)
                ],
                "applies_post_build_tightening": bool(
                    variant_setup["applies_post_build_tightening"]
                ),
            }
        )
        return metrics, optimizer
    except Exception as exc:
        runtime = time.perf_counter() - start
        result = {
            "variant": variant_name,
            "enabled_components": list(variant_config.get("enabled_components", [])),
            "disabled_components": list(variant_config.get("disabled_components", [])),
            "component_flags": dict(variant_config.get("component_flags", {})),
            "mask": int(variant_config.get("mask", -1)),
            "objective_value": None,
            "C_eq": None,
            "C_opt": None,
            "PoA": None,
            "PoA_ratio": None,
            "phi": None,
            "mu_D": None,
            "sigma_D": None,
            "mu_W": None,
            "sigma_W": None,
            "solver_status": None,
            "termination_condition": None,
            "runtime": float(runtime),
            "build_runtime": None,
            "solve_runtime": None,
            "model_size": _model_size(optimizer.model)
            if optimizer is not None and hasattr(optimizer, "model")
            else {},
            "solver_statistics": _extract_solver_statistics(optimizer)
            if optimizer is not None and hasattr(optimizer, "solver_results")
            else {
                "mip_gap": None,
                "best_bound": None,
                "node_count": None,
                "simplex_iterations": None,
                "barrier_iterations": None,
            },
            "loaded_tightening_components": [],
            "default_components_used": [
                key
                for key, used in getattr(
                    optimizer,
                    "default_bounds_used",
                    {},
                ).items()
                if key != "notes" and bool(used)
            ]
            if optimizer is not None
            else [],
            "applies_post_build_tightening": bool(
                enabled_components & FACTORIAL_POST_BUILD_COMPONENTS
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
        result.update(
            {
                "mip_gap": result["solver_statistics"].get("mip_gap"),
                "best_bound": result["solver_statistics"].get("best_bound"),
                "node_count": result["solver_statistics"].get("node_count"),
                "simplex_iterations": result["solver_statistics"].get(
                    "simplex_iterations"
                ),
                "barrier_iterations": result["solver_statistics"].get(
                    "barrier_iterations"
                ),
            }
        )
        return result, optimizer


def _component_value_dict(optimizer: PoAOptimization, component_name: str) -> dict[str, Optional[float]]:
    component = getattr(optimizer.model, component_name, None)
    if component is None:
        return {}
    if not component.is_indexed():
        return {"": optimizer._safe_value(component)}
    return {
        _json_ready_index(index): optimizer._safe_value(component[index])
        for index in component
    }


def _collect_relevant_solution_values(optimizer: PoAOptimization) -> dict[str, Any]:
    component_names = (
        "alpha",
        "lambda_eq",
        "lambda_opt",
        "mu_upper_eq",
        "mu_lower_eq",
        "mu_ramp_up_eq",
        "mu_ramp_down_eq",
        "mu_upper_opt",
        "mu_lower_opt",
        "mu_ramp_up_opt",
        "mu_ramp_down_opt",
        "z_upper_eq",
        "z_lower_eq",
        "z_ramp_up_eq",
        "z_ramp_down_eq",
        "z_upper_opt",
        "z_lower_opt",
        "z_ramp_up_opt",
        "z_ramp_down_opt",
    )
    m = optimizer.model
    snapshot = {
        name: _component_value_dict(optimizer, name)
        for name in component_names
        if hasattr(m, name)
    }
    snapshot["scalars"] = {
        "objective": optimizer._safe_value(m.objective),
        "C_eq": optimizer._safe_value(m.C_eq),
        "C_opt": optimizer._safe_value(m.C_opt),
        "PoA": optimizer._safe_value(m.PoA),
        "phi": optimizer._safe_value(m.phi) if hasattr(m, "phi") else None,
        "mu_D": optimizer._safe_value(m.mu_D),
        "sigma_D": optimizer._safe_value(m.sigma_D),
        "mu_W": optimizer._safe_value(m.mu_W),
        "sigma_W": optimizer._safe_value(m.sigma_W),
    }
    return snapshot


def _bound_violation(
    value_to_check: Optional[float],
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    tolerance: float = DIAGNOSTIC_VIOLATION_TOL,
) -> float:
    if value_to_check is None:
        return 0.0
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(lower) - float(value_to_check))
    if upper is not None:
        violation = max(violation, float(value_to_check) - float(upper))
    return float(violation) if violation > tolerance else 0.0


def _record_violation(
    violations: list[dict[str, Any]],
    component: str,
    index: Any,
    value_to_check: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
    violation: float,
) -> None:
    if violation <= 0.0:
        return
    violations.append(
        {
            "component": component,
            "index": _json_ready_index(index),
            "value": value_to_check,
            "lower": lower,
            "upper": upper,
            "violation": float(violation),
        }
    )


def _summarize_violations(violations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": int(len(violations)),
        "max_violation": max(
            (float(item.get("violation", 0.0)) for item in violations),
            default=0.0,
        ),
        "violations": violations,
    }


def _check_alpha_bound_violations(
    optimizer: PoAOptimization,
    report: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    alpha_bounds = report.get("alpha_bounds", {}) or {}
    for key, bounds in alpha_bounds.items():
        index = optimizer._parse_json_index(key)
        if index not in optimizer.model.alpha:
            continue
        alpha_value = optimizer._safe_value(optimizer.model.alpha[index])
        lower = float(bounds["lower"])
        upper = float(bounds["upper"])
        _record_violation(
            violations,
            "alpha",
            index,
            alpha_value,
            lower,
            upper,
            _bound_violation(alpha_value, lower, upper),
        )
    return _summarize_violations(violations)


def _check_lambda_bound_violations(
    optimizer: PoAOptimization,
    report: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for lambda_name, entries in (report.get("lambda_bounds", {}) or {}).items():
        lambda_var = getattr(optimizer.model, lambda_name, None)
        if lambda_var is None:
            continue
        for raw_t, bounds in (entries or {}).items():
            t = int(raw_t)
            if t not in lambda_var:
                continue
            lambda_value = optimizer._safe_value(lambda_var[t])
            lower = float(bounds["lower"])
            upper = float(bounds["upper"])
            _record_violation(
                violations,
                lambda_name,
                t,
                lambda_value,
                lower,
                upper,
                _bound_violation(lambda_value, lower, upper),
            )
    return _summarize_violations(violations)


def _check_dual_big_m_violations(
    optimizer: PoAOptimization,
    report: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for dual_name, entries in (report.get("tight_big_m", {}) or {}).items():
        dual_var = getattr(optimizer.model, dual_name, None)
        if dual_var is None:
            continue
        for key, details in (entries or {}).items():
            index = optimizer._parse_json_index(key)
            if index not in dual_var:
                continue
            upper = optimizer._optional_numeric_bound(details)
            if upper is None:
                continue
            dual_value = optimizer._safe_value(dual_var[index])
            _record_violation(
                violations,
                dual_name,
                index,
                dual_value,
                0.0,
                float(upper),
                _bound_violation(dual_value, 0.0, float(upper)),
            )
    return _summarize_violations(violations)


def _check_fixed_binary_violations(
    optimizer: PoAOptimization,
    report: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for var_name, entries in (report.get("fixed_binaries", {}) or {}).items():
        binary_var = getattr(optimizer.model, var_name, None)
        if binary_var is None:
            continue
        for key, details in (entries or {}).items():
            index = optimizer._parse_json_index(key)
            if index not in binary_var:
                continue
            fixed_value = int(details.get("fixed_value", 0))
            binary_value = optimizer._safe_value(binary_var[index])
            violation = (
                abs(float(binary_value) - float(fixed_value))
                if binary_value is not None
                else 0.0
            )
            _record_violation(
                violations,
                var_name,
                index,
                binary_value,
                float(fixed_value),
                float(fixed_value),
                violation if violation > DIAGNOSTIC_VIOLATION_TOL else 0.0,
            )
    return _summarize_violations(violations)


def _check_aggregate_dual_bound_violations(
    optimizer: PoAOptimization,
    report: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    m = optimizer.model
    old_report = getattr(optimizer, "tightening_report", None)
    old_aggregate = getattr(optimizer, "aggregate_dual_bounds", {})
    old_tight_big_m = getattr(optimizer, "tight_big_m", {})
    optimizer.tightening_report = report
    optimizer.aggregate_dual_bounds = report.get("aggregate_dual_bounds", {}) or {}
    optimizer.tight_big_m = report.get("tight_big_m", {}) or {}
    try:
        aggregate_specs = (
            ("mu_max_sum_ub", "upper", "mu_upper"),
            ("mu_min_sum_ub", "lower", "mu_lower"),
            ("mu_ramp_up_sum_ub", "ramp_up", "mu_ramp_up"),
            ("mu_ramp_down_sum_ub", "ramp_down", "mu_ramp_down"),
        )
        for side in ("eq", "opt"):
            for t in range(optimizer.num_time_steps):
                for generic_key, constraint_type, dual_root in aggregate_specs:
                    dual_name = f"{dual_root}_{side}"
                    bound = optimizer._aggregate_dual_sum_upper_bound(
                        generic_key,
                        side,
                        int(t),
                        dual_name,
                    )
                    if bound is None:
                        continue
                    dual_var = getattr(m, dual_name, None)
                    if dual_var is None:
                        continue
                    if constraint_type in {"upper", "lower"}:
                        total = sum(
                            optimizer._safe_value(dual_var[i, b, t]) or 0.0
                            for i, b in m.generator_blocks
                        )
                    else:
                        total = sum(
                            optimizer._safe_value(dual_var[i, t]) or 0.0
                            for i in m.physical_generators
                        )
                    _record_violation(
                        violations,
                        f"{generic_key}_{side}",
                        int(t),
                        float(total),
                        0.0,
                        float(bound),
                        _bound_violation(float(total), 0.0, float(bound)),
                    )
    finally:
        optimizer.tightening_report = old_report
        optimizer.aggregate_dual_bounds = old_aggregate
        optimizer.tight_big_m = old_tight_big_m
    return _summarize_violations(violations)


def _check_optimal_cost_bound_violations(
    optimizer: PoAOptimization,
    report: dict[str, Any],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    raw_bounds = report.get("optimal_cost_bounds", {}) or {}
    if "C_opt" in raw_bounds and isinstance(raw_bounds.get("C_opt"), dict):
        bounds = raw_bounds.get("C_opt", {}) or {}
    else:
        bounds = raw_bounds
    if bounds:
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        C_opt = optimizer._safe_value(optimizer.model.C_opt)
        lower_value = float(lower) if lower is not None else None
        upper_value = float(upper) if upper is not None else None
        _record_violation(
            violations,
            "C_opt",
            "",
            C_opt,
            lower_value,
            upper_value,
            _bound_violation(C_opt, lower_value, upper_value),
        )
    return _summarize_violations(violations)


def _constraint_violation(constraint: Any) -> float:
    body = value(constraint.body, exception=False)
    if body is None:
        return 0.0
    violation = 0.0
    if constraint.lower is not None:
        lower = value(constraint.lower, exception=False)
        if lower is not None:
            violation = max(violation, float(lower) - float(body))
    if constraint.upper is not None:
        upper = value(constraint.upper, exception=False)
        if upper is not None:
            violation = max(violation, float(body) - float(upper))
    return max(0.0, float(violation))


def _max_kkt_complementarity_residual(optimizer: PoAOptimization) -> float:
    m = optimizer.model
    max_residual = 0.0

    def residual(mu_value: Optional[float], slack_value: float) -> None:
        nonlocal max_residual
        if mu_value is None:
            return
        max_residual = max(max_residual, abs(float(mu_value) * float(slack_value)))

    for side, dispatch_name in (("eq", "P_eq"), ("opt", "P_opt")):
        P = getattr(m, dispatch_name)
        mu_upper = getattr(m, f"mu_upper_{side}")
        mu_lower = getattr(m, f"mu_lower_{side}")
        mu_ramp_up = getattr(m, f"mu_ramp_up_{side}")
        mu_ramp_down = getattr(m, f"mu_ramp_down_{side}")
        for i, b in m.generator_blocks:
            for t in m.time_steps:
                P_value = optimizer._safe_value(P[i, b, t]) or 0.0
                P_max = optimizer._safe_value(m.P_max_block[i, b, t]) or 0.0
                residual(optimizer._safe_value(mu_upper[i, b, t]), P_max - P_value)
                residual(optimizer._safe_value(mu_lower[i, b, t]), P_value)
        for i in m.physical_generators:
            for t in m.time_steps:
                current = sum(
                    optimizer._safe_value(P[i, b, t]) or 0.0
                    for b in optimizer.local_blocks_by_generator[int(i)]
                )
                previous = (
                    float(optimizer.p_init[int(i)])
                    if int(t) == 0
                    else sum(
                        optimizer._safe_value(P[i, b, int(t) - 1]) or 0.0
                        for b in optimizer.local_blocks_by_generator[int(i)]
                    )
                )
                ramp_up_slack = float(optimizer.ramp_vector_up[int(i)]) - (
                    current - previous
                )
                ramp_down_slack = float(optimizer.ramp_vector_down[int(i)]) - (
                    previous - current
                )
                residual(optimizer._safe_value(mu_ramp_up[i, t]), ramp_up_slack)
                residual(optimizer._safe_value(mu_ramp_down[i, t]), ramp_down_slack)
    return float(max_residual)


def _compute_kkt_residuals(optimizer: PoAOptimization) -> dict[str, Any]:
    constraint_violations = [
        _constraint_violation(constraint)
        for constraint in optimizer.model.component_data_objects(Constraint, active=True)
    ]
    max_constraint_violation = max(constraint_violations, default=0.0)
    max_complementarity_residual = _max_kkt_complementarity_residual(optimizer)
    return {
        "max_constraint_violation": float(max_constraint_violation),
        "max_complementarity_residual": float(max_complementarity_residual),
        "large_residual": bool(
            max_constraint_violation > DIAGNOSTIC_LARGE_RESIDUAL_TOL
            or max_complementarity_residual > DIAGNOSTIC_LARGE_RESIDUAL_TOL
        ),
    }


def _run_feasibility_replay(
    untightened_optimizer: PoAOptimization,
    tightened_report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "alpha_bound_violations": _check_alpha_bound_violations(
            untightened_optimizer,
            tightened_report,
        ),
        "lambda_bound_violations": _check_lambda_bound_violations(
            untightened_optimizer,
            tightened_report,
        ),
        "dual_big_m_violations": _check_dual_big_m_violations(
            untightened_optimizer,
            tightened_report,
        ),
        "fixed_binary_violations": _check_fixed_binary_violations(
            untightened_optimizer,
            tightened_report,
        ),
        "aggregate_dual_bound_violations": _check_aggregate_dual_bound_violations(
            untightened_optimizer,
            tightened_report,
        ),
        "optimal_cost_bound_violations": _check_optimal_cost_bound_violations(
            untightened_optimizer,
            tightened_report,
        ),
    }
    total_count = sum(int(summary["count"]) for summary in checks.values())
    max_violation = max(
        (float(summary["max_violation"]) for summary in checks.values()),
        default=0.0,
    )
    return {
        **checks,
        "total_violation_count": int(total_count),
        "max_violation": float(max_violation),
        "untightened_solution_snapshot": _collect_relevant_solution_values(
            untightened_optimizer
        ),
    }


def _empty_replay_summary() -> dict[str, Any]:
    return {
        "checks": {},
        "total_violation_count": 0,
        "max_violation": 0.0,
    }


def _has_replay_relevant_tightening_entries(report: dict[str, Any]) -> bool:
    if report.get("alpha_bounds"):
        return True
    if report.get("lambda_bounds"):
        return True
    if report.get("tight_big_m"):
        return True
    if report.get("fixed_binaries"):
        return True
    if report.get("aggregate_dual_bounds"):
        return True
    return False


def _compact_replay_summary(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in replay.items()
        if key != "untightened_solution_snapshot"
    }


def _run_factorial_feasibility_replays(
    untightened_optimizer: PoAOptimization,
    full_tightening_report: dict[str, Any],
    variant_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    replay_results: dict[str, Any] = {}
    for variant_config in variant_configs:
        variant_name = str(variant_config["variant"])
        enabled_components = set(variant_config.get("enabled_components", []))
        filtered_report = _construct_factorial_tightening_report(
            full_tightening_report,
            enabled_components,
        )
        if not _has_replay_relevant_tightening_entries(filtered_report):
            replay_results[variant_name] = _empty_replay_summary()
            continue
        replay_results[variant_name] = _compact_replay_summary(
            _run_feasibility_replay(untightened_optimizer, filtered_report)
        )
    return replay_results


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def _median_or_none(values: list[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _min_or_none(values: list[float]) -> Optional[float]:
    return float(min(values)) if values else None


def _max_or_none(values: list[float]) -> Optional[float]:
    return float(max(values)) if values else None


def _share(values: list[bool]) -> Optional[float]:
    return float(sum(1 for value in values if value) / len(values)) if values else None


def _successful_factorial_results(
    variant_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        variant_name: result
        for variant_name, result in variant_results.items()
        if _is_successful_termination(result)
        and result.get("objective_value") is not None
        and result.get("runtime") is not None
    }


def _summarize_factorial_tightening_design(
    variant_results: dict[str, dict[str, Any]],
    baseline_variant: str,
    full_variant: str,
) -> dict[str, Any]:
    successful_results = _successful_factorial_results(variant_results)
    failed_variants = [
        variant_name
        for variant_name in variant_results
        if variant_name not in successful_results
    ]

    baseline_result = variant_results.get(baseline_variant, {})
    full_result = variant_results.get(full_variant, {})
    baseline_objective = baseline_result.get("objective_value")
    full_objective = full_result.get("objective_value")
    baseline_runtime = baseline_result.get("runtime")
    full_runtime = full_result.get("runtime")

    objective_delta = (
        float(full_objective) - float(baseline_objective)
        if baseline_objective is not None and full_objective is not None
        else None
    )
    runtime_reduction_absolute = (
        float(baseline_runtime) - float(full_runtime)
        if baseline_runtime is not None and full_runtime is not None
        else None
    )
    runtime_reduction_relative = (
        runtime_reduction_absolute / float(baseline_runtime)
        if runtime_reduction_absolute is not None
        and baseline_runtime not in (None, 0.0)
        else None
    )

    variants_with_objective_below_baseline = []
    variants_with_objective_above_baseline = []
    variants_matching_baseline_objective = []
    if baseline_objective is not None:
        for variant_name, result in successful_results.items():
            objective_value = result.get("objective_value")
            if _objective_close(objective_value, baseline_objective):
                variants_matching_baseline_objective.append(variant_name)
            elif _objective_materially_greater(baseline_objective, objective_value):
                variants_with_objective_below_baseline.append(variant_name)
            elif _objective_materially_greater(objective_value, baseline_objective):
                variants_with_objective_above_baseline.append(variant_name)

    best_runtime_variant = None
    if successful_results:
        best_runtime_variant = min(
            successful_results,
            key=lambda name: float(successful_results[name]["runtime"]),
        )

    best_objective_variant = None
    if successful_results:
        best_objective_variant = max(
            successful_results,
            key=lambda name: float(successful_results[name]["objective_value"]),
        )

    results_by_mask = {
        int(result.get("mask", -1)): result
        for result in successful_results.values()
        if result.get("mask") is not None
    }
    marginal_runtime_effects: dict[str, Any] = {}
    marginal_objective_effects: dict[str, Any] = {}
    for component_idx, component in enumerate(FACTORIAL_TIGHTENING_COMPONENTS):
        runtime_effects: list[float] = []
        objective_effects: list[float] = []
        objective_preserved: list[bool] = []
        component_bit = 1 << component_idx
        for mask, base_result in results_by_mask.items():
            if mask & component_bit:
                continue
            paired_result = results_by_mask.get(mask | component_bit)
            if paired_result is None:
                continue
            base_runtime = base_result.get("runtime")
            paired_runtime = paired_result.get("runtime")
            base_objective = base_result.get("objective_value")
            paired_objective = paired_result.get("objective_value")
            if base_runtime is not None and paired_runtime is not None:
                runtime_effects.append(float(base_runtime) - float(paired_runtime))
            if base_objective is not None and paired_objective is not None:
                objective_change = float(paired_objective) - float(base_objective)
                objective_effects.append(objective_change)
                objective_preserved.append(
                    _objective_close(paired_objective, base_objective)
                )

        marginal_runtime_effects[component] = {
            "count": int(len(runtime_effects)),
            "mean_marginal_runtime_reduction": _mean_or_none(runtime_effects),
            "median_marginal_runtime_reduction": _median_or_none(runtime_effects),
            "min_marginal_runtime_reduction": _min_or_none(runtime_effects),
            "max_marginal_runtime_reduction": _max_or_none(runtime_effects),
            "share_positive_runtime_reduction": _share(
                [value > 0.0 for value in runtime_effects]
            ),
        }
        marginal_objective_effects[component] = {
            "count": int(len(objective_effects)),
            "mean_marginal_objective_change": _mean_or_none(objective_effects),
            "max_abs_marginal_objective_change": _max_or_none(
                [abs(value) for value in objective_effects]
            ),
            "share_objective_preserved_using_close_tolerance": _share(
                objective_preserved
            ),
        }

    return {
        "baseline_variant": baseline_variant,
        "full_variant": full_variant,
        "baseline_objective": baseline_objective,
        "full_tightening_objective": full_objective,
        "objective_difference_full_minus_baseline": objective_delta,
        "objective_preserved_full": _objective_close(full_objective, baseline_objective),
        "baseline_runtime": baseline_runtime,
        "full_tightening_runtime": full_runtime,
        "runtime_reduction_absolute": runtime_reduction_absolute,
        "runtime_reduction_relative": runtime_reduction_relative,
        "successful_variants_count": int(len(successful_results)),
        "failed_variants_count": int(len(failed_variants)),
        "failed_variants": failed_variants,
        "best_runtime_variant": best_runtime_variant,
        "best_objective_variant": best_objective_variant,
        "variants_with_objective_below_baseline": variants_with_objective_below_baseline,
        "variants_with_objective_above_baseline": variants_with_objective_above_baseline,
        "variants_matching_baseline_objective": variants_matching_baseline_objective,
        "marginal_runtime_effects": marginal_runtime_effects,
        "marginal_objective_effects": marginal_objective_effects,
    }


def _diagnostic_conclusion(
    variant_results: dict[str, dict[str, Any]],
    feasibility_replay: Optional[dict[str, Any]],
    kkt_residuals: Optional[dict[str, Any]],
) -> str:
    baseline = variant_results.get("untightened_baseline", {})
    tightened = variant_results.get("tightened_full", {})
    if not _is_successful_termination(baseline) or not _is_successful_termination(
        tightened
    ):
        return "inconclusive"

    baseline_obj = baseline.get("objective_value")
    tightened_obj = tightened.get("objective_value")
    loosened_variants = [
        "no_fixed_binaries",
        "loose_dual_big_m",
        "loose_lambda_bounds",
        "loose_alpha_bounds",
        "no_aggregate_dual_bounds",
    ]
    successful_loosened = [
        name
        for name in loosened_variants
        if _is_successful_termination(variant_results.get(name, {}))
    ]
    all_loosened_match_baseline = (
        len(successful_loosened) == len(loosened_variants)
        and all(
            _objective_close(
                variant_results[name].get("objective_value"),
                baseline_obj,
            )
            for name in successful_loosened
        )
    )
    if _objective_close(tightened_obj, baseline_obj) and all_loosened_match_baseline:
        return "no_obvious_issue"

    tightened_is_lower = _objective_materially_greater(baseline_obj, tightened_obj)
    if tightened_is_lower and kkt_residuals and kkt_residuals.get("large_residual"):
        return "untightened_baseline_may_be_loose"

    suspicious_map = {
        "no_fixed_binaries": "suspicious_fixed_binaries",
        "loose_dual_big_m": "suspicious_dual_big_m",
        "loose_lambda_bounds": "suspicious_lambda_bounds",
        "loose_alpha_bounds": "suspicious_alpha_bounds",
        "no_aggregate_dual_bounds": "suspicious_aggregate_dual_bounds",
    }
    if tightened_is_lower:
        for variant_name, conclusion in suspicious_map.items():
            variant = variant_results.get(variant_name, {})
            if not _is_successful_termination(variant):
                continue
            variant_obj = variant.get("objective_value")
            if _objective_close(variant_obj, baseline_obj) and _objective_materially_greater(
                variant_obj,
                tightened_obj,
            ):
                return conclusion

    if (
        tightened_is_lower
        and feasibility_replay is not None
        and int(feasibility_replay.get("total_violation_count", 0)) > 0
    ):
        return "tightening_changes_feasible_region"

    if any(
        not _is_successful_termination(variant_results.get(name, {}))
        for name in loosened_variants
    ):
        return "inconclusive"

    return "inconclusive"


def run_tightening_comparison_diagnostic(
    optimizer_kwargs: dict[str, Any],
    nn_relu_report_path: str | Path,
    tightening_report_path: str | Path,
    output_path: str | Path = "results/poa_tightening/tightening_comparison_report.json",
    time_limit: Optional[float] = 400,
    use_lambda_bounds: bool = DEFAULT_USE_LAMBDA_BOUNDS,
    use_aggregate_dual_bounds: bool = DEFAULT_USE_AGGREGATE_DUAL_BOUNDS,
) -> Path:
    with Path(tightening_report_path).open("r", encoding="utf-8") as file_handle:
        raw_tightening_report = json.load(file_handle)
    full_tightening_report = _base_tightening_report_for_diagnostic(
        raw_tightening_report,
        use_lambda_bounds=use_lambda_bounds,
        use_aggregate_dual_bounds=use_aggregate_dual_bounds,
    )

    variant_names = [
        "untightened_baseline",
        "tightened_full",
        "no_fixed_binaries",
        "loose_dual_big_m",
        "loose_lambda_bounds",
        "no_aggregate_dual_bounds",
        "loose_alpha_bounds",
    ]
    variant_results: dict[str, dict[str, Any]] = {}
    optimizers: dict[str, PoAOptimization] = {}

    for variant_name in variant_names:
        print(f"\nRunning tightening diagnostic variant: {variant_name}")
        result, optimizer = _solve_diagnostic_variant(
            variant_name=variant_name,
            optimizer_kwargs=optimizer_kwargs,
            full_tightening_report=full_tightening_report,
            nn_relu_report_path=nn_relu_report_path,
            tightening_report_path=tightening_report_path,
            time_limit=time_limit,
        )
        variant_results[variant_name] = result
        if optimizer is not None and hasattr(optimizer, "model"):
            optimizers[variant_name] = optimizer
        print(
            f"  termination={result.get('termination_condition')} "
            f"objective={result.get('objective_value')} "
            f"runtime={result.get('runtime'):.2f}s"
        )

    feasibility_replay = None
    if (
        "untightened_baseline" in optimizers
        and _is_successful_termination(variant_results["untightened_baseline"])
        and _is_successful_termination(variant_results.get("tightened_full", {}))
    ):
        feasibility_replay = _run_feasibility_replay(
            optimizers["untightened_baseline"],
            full_tightening_report,
        )

    kkt_residuals = None
    if "untightened_baseline" in optimizers:
        kkt_residuals = _compute_kkt_residuals(optimizers["untightened_baseline"])

    report = {
        "metadata": {
            "reference_case": optimizer_kwargs.get("reference_case"),
            "num_time_steps": optimizer_kwargs.get("num_time_steps"),
            "nn_policy_generators": optimizer_kwargs.get("nn_policy_generators"),
            "objective_mode": optimizer_kwargs.get("objective_mode", "difference"),
            "ratio_bounds": optimizer_kwargs.get("ratio_bounds"),
            "nn_relu_report_path": str(nn_relu_report_path),
            "tightening_report_path": str(tightening_report_path),
            "time_limit": time_limit,
            "use_lambda_bounds": bool(use_lambda_bounds),
            "use_aggregate_dual_bounds": bool(use_aggregate_dual_bounds),
            "abs_obj_tol": DIAGNOSTIC_ABS_OBJ_TOL,
            "rel_obj_tol": DIAGNOSTIC_REL_OBJ_TOL,
            "violation_tol": DIAGNOSTIC_VIOLATION_TOL,
        },
        "variants": variant_results,
        "feasibility_replay": feasibility_replay,
        "untightened_baseline_kkt_residuals": kkt_residuals,
        "conclusion": _diagnostic_conclusion(
            variant_results,
            feasibility_replay,
            kkt_residuals,
        ),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    print(f"\nTightening diagnostic report saved to: {path}")
    print(f"Conclusion: {report['conclusion']}")
    return path


def run_full_factorial_tightening_design_diagnostic(
    optimizer_kwargs: dict[str, Any],
    nn_relu_report_path: str | Path,
    tightening_report_path: str | Path,
    output_path: str | Path = (
        "results/poa_tightening/full_factorial_tightening_design_report.json"
    ),
    time_limit: Optional[float] = 400,
    use_lambda_bounds: bool = DEFAULT_USE_LAMBDA_BOUNDS,
    use_aggregate_dual_bounds: bool = DEFAULT_USE_AGGREGATE_DUAL_BOUNDS,
    max_variants: Optional[int] = None,
    skip_failed_after: Optional[int] = None,
) -> Path:
    with Path(tightening_report_path).open("r", encoding="utf-8") as file_handle:
        raw_tightening_report = json.load(file_handle)
    full_tightening_report = _base_tightening_report_for_diagnostic(
        raw_tightening_report,
        use_lambda_bounds=use_lambda_bounds,
        use_aggregate_dual_bounds=use_aggregate_dual_bounds,
    )

    all_variant_configs = _generate_factorial_tightening_configurations()
    variant_configs = (
        all_variant_configs[: int(max_variants)]
        if max_variants is not None
        else list(all_variant_configs)
    )
    baseline_config = all_variant_configs[0]
    full_config = all_variant_configs[-1]
    baseline_variant = str(baseline_config["variant"])
    full_variant = str(full_config["variant"])

    variant_results: dict[str, dict[str, Any]] = {}
    optimizers: dict[str, PoAOptimization] = {}
    consecutive_failures = 0

    for variant_idx, variant_config in enumerate(variant_configs, start=1):
        variant_name = str(variant_config["variant"])
        print(
            f"\nRunning factorial tightening variant "
            f"{variant_idx}/{len(variant_configs)}: {variant_name}"
        )
        result, optimizer = _solve_factorial_tightening_variant(
            variant_config=variant_config,
            optimizer_kwargs=optimizer_kwargs,
            full_tightening_report=full_tightening_report,
            nn_relu_report_path=nn_relu_report_path,
            tightening_report_path=tightening_report_path,
            time_limit=time_limit,
        )
        variant_results[variant_name] = result
        if optimizer is not None and hasattr(optimizer, "model"):
            optimizers[variant_name] = optimizer

        if _is_successful_termination(result):
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        print(
            f"  termination={result.get('termination_condition')} "
            f"objective={result.get('objective_value')} "
            f"runtime={result.get('runtime'):.2f}s"
        )

        if (
            skip_failed_after is not None
            and consecutive_failures >= int(skip_failed_after)
        ):
            print(
                f"Stopping early after {consecutive_failures} consecutive "
                "factorial variant failures."
            )
            break

    solved_configs = [
        config
        for config in variant_configs
        if str(config["variant"]) in variant_results
    ]
    baseline_optimizer = optimizers.get(baseline_variant)
    feasibility_replays = {}
    untightened_solution_snapshot = None
    kkt_residuals = None
    if (
        baseline_optimizer is not None
        and _is_successful_termination(variant_results.get(baseline_variant, {}))
    ):
        untightened_solution_snapshot = _collect_relevant_solution_values(
            baseline_optimizer
        )
        feasibility_replays = _run_factorial_feasibility_replays(
            baseline_optimizer,
            full_tightening_report,
            solved_configs,
        )
        kkt_residuals = _compute_kkt_residuals(baseline_optimizer)

    report = {
        "metadata": {
            "design_type": "full_factorial",
            "reference_case": optimizer_kwargs.get("reference_case"),
            "num_time_steps": optimizer_kwargs.get("num_time_steps"),
            "nn_policy_generators": optimizer_kwargs.get("nn_policy_generators"),
            "objective_mode": optimizer_kwargs.get("objective_mode", "difference"),
            "ratio_bounds": optimizer_kwargs.get("ratio_bounds"),
            "nn_relu_report_path": str(nn_relu_report_path),
            "tightening_report_path": str(tightening_report_path),
            "time_limit": time_limit,
            "use_lambda_bounds": bool(use_lambda_bounds),
            "use_aggregate_dual_bounds": bool(use_aggregate_dual_bounds),
            "max_variants": max_variants,
            "skip_failed_after": skip_failed_after,
            "num_components": int(len(FACTORIAL_TIGHTENING_COMPONENTS)),
            "num_variants": int(len(all_variant_configs)),
            "num_variants_attempted": int(len(variant_results)),
            "components": list(FACTORIAL_TIGHTENING_COMPONENTS),
            "baseline_variant": baseline_variant,
            "full_variant": full_variant,
            "abs_obj_tol": DIAGNOSTIC_ABS_OBJ_TOL,
            "rel_obj_tol": DIAGNOSTIC_REL_OBJ_TOL,
            "violation_tol": DIAGNOSTIC_VIOLATION_TOL,
        },
        "variants": variant_results,
        "feasibility_replays": feasibility_replays,
        "untightened_baseline_solution_snapshot": untightened_solution_snapshot,
        "untightened_baseline_kkt_residuals": kkt_residuals,
        "summary": _summarize_factorial_tightening_design(
            variant_results,
            baseline_variant,
            full_variant,
        ),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)
    print(f"\nFull factorial tightening design report saved to: {path}")
    return path

if __name__ == "__main__":
   
    case = "base_test_case"
    regime_set = "PoA_analysis"
    seed = 1
    horizon = 6
    nn_relu_report_path = Path("results/poa_tightening/relu_bounds_report.json")
    tightening_report_path = Path("results/poa_tightening/final_tightening_report.json")

    run_poa_tightening = True
    poa_tightening_flags = {
        "primal_big_m": True,
        "relu_bounds": True,
        "alpha_bounds": True,
        "slack_binary_fix": True,
        "dual_big_m": True,
        "optimal_cost_bounds": True,
    }

    poa_tightening_time_limit = None
    poa_tightening_parallel_workers = 6
    poa_tightening_solver_threads = None
    poa_tightening_tee = False

    full_factorial_design_mode = True
    factorial_max_variants = None
    factorial_skip_failed_after = None
    use_lambda_bounds_in_diagnostics = True
    use_aggregate_dual_bounds_in_diagnostics = True
    diagnostic_mode = True
    objective_mode = "piecewise_mccormick"

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    ambiguity_set_config = PoAOptimization.load_ambiguity_set(
        config_path="config/ambiguity_set_config.yaml",
        config_name="base_test_case",
    )

    required_tightening_paths = [tightening_report_path]
    if nn_relu_report_path is not None:
        required_tightening_paths.append(nn_relu_report_path)
    missing_tightening_paths = [
        path for path in required_tightening_paths if not Path(path).exists()
    ]

    if run_poa_tightening or missing_tightening_paths:
        if missing_tightening_paths and not run_poa_tightening:
            missing_text = ", ".join(str(path) for path in missing_tightening_paths)
            print(
                "Tightening reports are missing; running PoA tightening first: "
                f"{missing_text}"
            )
        tightening_report_path = run_poa_tightening_for_tests(
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            ambiguity_set_config=ambiguity_set_config,
            horizon=horizon,
            case=case,
            nn_model_dir="models/neural_network/training/trained_models",
            nn_normalization_stats_path=(
                "models/neural_network/features/generated/normalized/min_max_stats.json"
            ),
            nn_policy_generators=[1, 2],
            tightening_flags=poa_tightening_flags,
            time_limit=poa_tightening_time_limit,
            parallel_workers=poa_tightening_parallel_workers,
            solver_threads=poa_tightening_solver_threads,
            tee=poa_tightening_tee,
        )
        nn_relu_report_path = Path("results/poa_tightening/relu_bounds_report.json")

    ratio_bounds = (
        _load_ratio_bounds_from_tightening_report(
            tightening_report_path,
            phi_bounds=DEFAULT_PHI_BOUNDS,
            num_pieces=DEFAULT_PIECEWISE_NUM_PIECES,
        )
        if objective_mode != "difference"
        else None
    )

    optimizer_kwargs = {
        "scenarios_df": scenarios_df,
        "costs_df": costs_df,
        "ramps_df": ramps_df,
        "p_init": None,
        "num_time_steps": horizon,
        "ambiguity_set_config": ambiguity_set_config,
        "nn_model_dir": "models/neural_network/training/trained_models",
        "nn_normalization_stats_path": (
            "models/neural_network/features/generated/normalized/min_max_stats.json"
        ),
        # None means all generators with available NN files. Use [] for true
        # costs only, or e.g. ["G2", "W1"] / [1, 2] for a selected subset.
        "nn_policy_generators": [1, 2],
        "reference_case": case,
        "objective_mode": objective_mode,
        "ratio_bounds": ratio_bounds,
    }

    if full_factorial_design_mode:
        run_full_factorial_tightening_design_diagnostic(
            optimizer_kwargs=optimizer_kwargs,
            nn_relu_report_path=nn_relu_report_path,
            tightening_report_path=tightening_report_path,
            output_path=(
                "results/poa_tightening/"
                "full_factorial_tightening_design_report.json"
            ),
            time_limit=None,
            use_lambda_bounds=use_lambda_bounds_in_diagnostics,
            use_aggregate_dual_bounds=use_aggregate_dual_bounds_in_diagnostics,
            max_variants=factorial_max_variants,
            skip_failed_after=factorial_skip_failed_after,
        )
        raise SystemExit(0)

    if diagnostic_mode:
        run_tightening_comparison_diagnostic(
            optimizer_kwargs=optimizer_kwargs,
            nn_relu_report_path=nn_relu_report_path,
            tightening_report_path=tightening_report_path,
            output_path="results/poa_tightening/tightening_comparison_report.json",
            time_limit=None,
            use_lambda_bounds=use_lambda_bounds_in_diagnostics,
            use_aggregate_dual_bounds=use_aggregate_dual_bounds_in_diagnostics,
        )
        raise SystemExit(0)

    optimizer = PoAOptimization(**optimizer_kwargs)

