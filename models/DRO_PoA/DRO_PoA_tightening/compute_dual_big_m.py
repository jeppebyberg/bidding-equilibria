from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    Objective,
    SolverFactory,
    TerminationCondition,
    maximize,
    minimize,
    value,
)

from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import compute_primal_big_m_bounds
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain


_PARALLEL_DUAL_COMPUTER: Optional["DRODualBigMComputer"] = None


def _bound_sense(bound_name: str) -> Any:
    if bound_name == "lower":
        return minimize
    if bound_name == "upper":
        return maximize
    raise ValueError(f"Unknown bound name: {bound_name}")


def _initialize_parallel_dual_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_DUAL_COMPUTER
    _PARALLEL_DUAL_COMPUTER = DRODualBigMComputer._from_parallel_stage_state(state)


def _get_parallel_dual_computer() -> "DRODualBigMComputer":
    if _PARALLEL_DUAL_COMPUTER is None:
        raise RuntimeError("Parallel DRO dual computer was not initialized")
    return _PARALLEL_DUAL_COMPUTER


def _solve_parallel_lambda_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        side,
        scenario_idx,
        time_idx,
        bound_name,
        alpha_bounds,
        fixed_binaries,
        solver_name,
        time_limit,
        tee,
        solver_options,
    ) = task
    computer = _get_parallel_dual_computer()
    lambda_name = computer._lambda_name(side)
    m = computer._build_side_kkt_model_for_dual_big_m(
        side=side,
        alpha_bounds=alpha_bounds,
        include_complementarity=True,
        fixed_binaries=fixed_binaries,
    )
    lambda_expr = getattr(m, lambda_name)[int(scenario_idx), int(time_idx)]
    m.tightening_objective = Objective(
        expr=lambda_expr,
        sense=_bound_sense(bound_name),
    )
    solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    return {
        "side": side,
        "lambda_name": lambda_name,
        "scenario_idx": int(scenario_idx),
        "time_idx": int(time_idx),
        "bound_name": bound_name,
        "value": computer._safe_value(lambda_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


def _solve_parallel_dual_big_m(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        side,
        constraint_type,
        scenario_index,
        regime_index,
        alpha_bounds,
        fixed_binaries,
        solver_name,
        time_limit,
        tee,
        solver_options,
    ) = task
    computer = _get_parallel_dual_computer()
    dual_name = computer._dual_name(side, constraint_type)
    binary_name = computer._binary_name(side, constraint_type)
    regime_key = computer._json_key(regime_index)
    scenario_key = computer._json_key(scenario_index)
    if (
        scenario_key in fixed_binaries.get(binary_name, {})
        or regime_key in fixed_binaries.get(binary_name, {})
    ):
        return {
            "side": side,
            "constraint_type": constraint_type,
            "dual_name": dual_name,
            "scenario_index": tuple(int(part) for part in scenario_index),
            "regime_index": tuple(int(part) for part in regime_index),
            "scenario_key": scenario_key,
            "regime_key": regime_key,
            "tight_big_m": 0.0,
            "fixed_by_slack": True,
            "termination_condition": "fixed_binary_zero",
        }
    m = computer._build_side_kkt_model_for_dual_big_m(
        side=side,
        alpha_bounds=alpha_bounds,
        include_complementarity=True,
        fixed_binaries=fixed_binaries,
    )
    dual_expr = getattr(m, dual_name)[tuple(scenario_index)]
    m.tightening_objective = Objective(expr=dual_expr, sense=maximize)
    solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    return {
        "side": side,
        "constraint_type": constraint_type,
        "dual_name": dual_name,
        "scenario_index": tuple(int(part) for part in scenario_index),
        "regime_index": tuple(int(part) for part in regime_index),
        "scenario_key": scenario_key,
        "regime_key": regime_key,
        "tight_big_m": computer._safe_value(dual_expr) if solved else None,
        "fixed_by_slack": False,
        "termination_condition": str(results.solver.termination_condition),
    }


def _solve_parallel_aggregate_dual_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        side,
        constraint_type,
        scenario_idx,
        time_idx,
        alpha_bounds,
        fixed_binaries,
        solver_name,
        time_limit,
        tee,
        solver_options,
    ) = task
    computer = _get_parallel_dual_computer()
    m = computer._build_side_kkt_model_for_dual_big_m(
        side=side,
        alpha_bounds=alpha_bounds,
        include_complementarity=True,
        fixed_binaries=fixed_binaries,
    )
    aggregate_expr = computer._aggregate_dual_expression(
        m,
        side,
        constraint_type,
        int(scenario_idx),
        int(time_idx),
    )
    m.tightening_objective = Objective(expr=aggregate_expr, sense=maximize)
    solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    return {
        "side": side,
        "constraint_type": constraint_type,
        "scenario_idx": int(scenario_idx),
        "time_idx": int(time_idx),
        "tight_big_m": computer._safe_value(aggregate_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


class DRODualBigMComputer(DROPoATighteningMain):
    def _ensure_primal_big_m_for_tightening(self) -> None:
        if getattr(self.poa, "primal_big_m", {}) or {}:
            if hasattr(self.poa, "_prepare_loaded_bounds"):
                self.poa._prepare_loaded_bounds()
            return
        self.poa.primal_big_m = compute_primal_big_m_bounds(self.poa)
        if hasattr(self.poa, "_prepare_loaded_bounds"):
            self.poa._prepare_loaded_bounds()

    def _build_side_kkt_model_for_dual_big_m(
        self,
        side: str,
        alpha_bounds: Optional[dict[tuple[int, ...], dict[str, float]]] = None,
        include_complementarity: bool = True,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> ConcreteModel:
        self._ensure_primal_big_m_for_tightening()
        self.model = ConcreteModel()
        from pyomo.environ import Set

        self.model.scenarios = Set(initialize=range(self.num_empirical_scenarios))
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))
        self.model.physical_generators = Set(initialize=range(self.num_physical_generators))
        self.model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)
        self.model.wind_physical_generators = Set(initialize=self.wind_physical_generator_ids)
        self.model.conventional_physical_generators = Set(
            initialize=self.conventional_physical_generator_ids
        )
        self.model.wind_blocks = Set(dimen=2, initialize=self.wind_block_pairs)
        self.model.conventional_blocks = Set(dimen=2, initialize=self.conventional_block_pairs)
        self._build_PoA_variables()

        if side == "eq":
            if alpha_bounds is None:
                raise ValueError("Equilibrium KKT dual tightening requires alpha_bounds")
            self._build_equilibrium_variables()
            self._build_complementarity_equilibrium_variables()
            self._build_support_set()
            self._build_transport_constraints()
            self._build_policy_constraints()
            self._apply_alpha_bounds(self.model, alpha_bounds)
            self._build_lower_level_equilibrium_constraints()
            self._build_KKT_stationarity_equilibrium_constraints()
            if include_complementarity:
                self._build_KKT_complementarity_equilibrium_constraints()
                self._apply_fixed_binaries(self.model, fixed_binaries)
        elif side == "opt":
            self._build_optimal_variables()
            self._build_complementarity_optimal_variables()
            self._build_support_set()
            self._build_transport_constraints()
            self._build_lower_level_optimal_constraints()
            self._build_KKT_stationarity_optimal_constraints()
            if include_complementarity:
                self._build_KKT_complementarity_optimal_constraints()
                self._apply_fixed_binaries(self.model, fixed_binaries)
        else:
            raise ValueError(f"Unknown KKT tightening side: {side}")
        return self.model

    def _apply_alpha_bounds(
        self,
        m: ConcreteModel,
        alpha_bounds: dict[tuple[int, ...], dict[str, float]],
    ) -> None:
        m.alpha_certified_bounds = ConstraintList()
        for k in m.scenarios:
            for i, b in m.generator_blocks:
                for t in m.time_steps:
                    scenario_key = (int(k), int(i), int(b), int(t))
                    regime_key = (int(i), int(b), int(t))
                    bounds = alpha_bounds.get(scenario_key, alpha_bounds.get(regime_key))
                    if bounds is None:
                        raise KeyError(
                            f"Missing alpha bounds for scenario key {scenario_key} "
                            f"or fallback regime key {regime_key}"
                        )
                    lower = float(bounds["lower"])
                    upper = float(bounds["upper"])
                    m.alpha_certified_bounds.add(m.alpha[k, i, b, t] >= lower)
                    m.alpha_certified_bounds.add(m.alpha[k, i, b, t] <= upper)
                    m.alpha[k, i, b, t].setlb(lower)
                    m.alpha[k, i, b, t].setub(upper)

    def _apply_fixed_binaries(
        self,
        m: ConcreteModel,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        for var_name, entries in (fixed_binaries or {}).items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, details in entries.items():
                index = self._parse_json_index(key)
                fixed_value = int((details or {}).get("fixed_value", 0))
                if len(index) == binary_var.dim():
                    if index in binary_var:
                        binary_var[index].fix(fixed_value)
                    continue
                for k in m.scenarios:
                    scenario_index = (int(k), *index)
                    if scenario_index in binary_var:
                        binary_var[scenario_index].fix(fixed_value)

    @staticmethod
    def _dual_name(side: str, constraint_type: str) -> str:
        return {
            ("eq", "upper"): "mu_upper_eq",
            ("eq", "lower"): "mu_lower_eq",
            ("eq", "ramp_up"): "mu_ramp_up_eq",
            ("eq", "ramp_down"): "mu_ramp_down_eq",
            ("opt", "upper"): "mu_upper_opt",
            ("opt", "lower"): "mu_lower_opt",
            ("opt", "ramp_up"): "mu_ramp_up_opt",
            ("opt", "ramp_down"): "mu_ramp_down_opt",
        }[(side, constraint_type)]

    @staticmethod
    def _binary_name(side: str, constraint_type: str) -> str:
        return {
            ("eq", "upper"): "z_upper_eq",
            ("eq", "lower"): "z_lower_eq",
            ("eq", "ramp_up"): "z_ramp_up_eq",
            ("eq", "ramp_down"): "z_ramp_down_eq",
            ("opt", "upper"): "z_upper_opt",
            ("opt", "lower"): "z_lower_opt",
            ("opt", "ramp_up"): "z_ramp_up_opt",
            ("opt", "ramp_down"): "z_ramp_down_opt",
        }[(side, constraint_type)]

    @staticmethod
    def _lambda_name(side: str) -> str:
        return {"eq": "lambda_eq", "opt": "lambda_opt"}[side]

    @staticmethod
    def _aggregate_dual_bound_key(constraint_type: str) -> str:
        return {
            "upper": "mu_max_sum_ub",
            "lower": "mu_min_sum_ub",
            "ramp_up": "mu_ramp_up_sum_ub",
            "ramp_down": "mu_ramp_down_sum_ub",
        }[constraint_type]

    def _aggregate_dual_expression(
        self,
        m: ConcreteModel,
        side: str,
        constraint_type: str,
        scenario_idx: int,
        time_idx: int,
    ) -> Any:
        dual_var = getattr(m, self._dual_name(side, constraint_type))
        if constraint_type in {"upper", "lower"}:
            return sum(
                dual_var[scenario_idx, i, b, int(time_idx)]
                for i, b in self.generator_block_pairs
            )
        return sum(
            dual_var[scenario_idx, i, int(time_idx)]
            for i in range(self.num_physical_generators)
        )

    @staticmethod
    def _safe_value(expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        return None if raw_value is None else float(raw_value)

    @staticmethod
    def _solve_tightening_model(
        m: ConcreteModel,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_options: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Any]:
        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        for option_name, option_value in (solver_options or {}).items():
            solver.options[option_name] = option_value
        results = solver.solve(m, tee=tee, load_solutions=False)
        termination = results.solver.termination_condition
        ok = termination in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
        if len(results.solution) > 0:
            m.solutions.load_from(results)
        return bool(ok), results

    def run_lambda_bound_tightening(
        self,
        alpha_bounds: Optional[dict[tuple[int, ...], dict[str, float]]] = None,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Alpha bounds must be computed before lambda tightening")
        fixed_binaries = fixed_binaries or getattr(self.poa, "fixed_binaries", {})
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)

        scenario_lambda_bounds: dict[str, dict[str, Any]] = {}
        lambda_bounds: dict[str, dict[str, Any]] = {}
        tasks = [
            (
                side,
                int(k),
                int(t),
                bound_name,
                alpha_bounds,
                fixed_binaries,
                solver_name,
                time_limit,
                tee,
                solver_options,
            )
            for side in ("eq", "opt")
            for k in range(self.num_empirical_scenarios)
            for t in range(self.num_time_steps)
            for bound_name in ("lower", "upper")
        ]
        workers = self._resolve_parallel_workers(parallel_workers, len(tasks))
        if workers > 1:
            print(
                f"Running DRO lambda-bound programs with {workers} worker processes",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_dual_computer,
                initargs=(self._parallel_stage_state(),),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_lambda_bound, task): task
                    for task in tasks
                }
                results_list = []
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    print(
                        f"[DRO Lambda {completed}/{len(tasks)}] "
                        f"{result['bound_name']} {result['lambda_name']}"
                        f"[{result['scenario_idx']},{result['time_idx']}] -> "
                        f"{result['value']} ({result['termination_condition']})",
                        flush=True,
                    )
                    results_list.append(result)
        else:
            global _PARALLEL_DUAL_COMPUTER
            _PARALLEL_DUAL_COMPUTER = self
            results_list = [_solve_parallel_lambda_bound(task) for task in tasks]

        grouped: dict[tuple[str, int, int], dict[str, Any]] = {}
        for result in results_list:
            key = (
                str(result["lambda_name"]),
                int(result["scenario_idx"]),
                int(result["time_idx"]),
            )
            entry = grouped.setdefault(key, {})
            bound_name = str(result["bound_name"])
            entry[bound_name] = result["value"]
            entry[f"{bound_name}_termination_condition"] = result["termination_condition"]

        for (lambda_name, k, t), entry in grouped.items():
            scenario_lambda_bounds.setdefault(lambda_name, {})[
                self._json_key((k, t))
            ] = entry
            regime_entry = lambda_bounds.setdefault(lambda_name, {}).setdefault(
                str(int(t)),
                {"lower": entry["lower"], "upper": entry["upper"]},
            )
            if entry["lower"] is not None:
                regime_entry["lower"] = (
                    float(entry["lower"])
                    if regime_entry.get("lower") is None
                    else min(float(regime_entry["lower"]), float(entry["lower"]))
                )
            if entry["upper"] is not None:
                regime_entry["upper"] = (
                    float(entry["upper"])
                    if regime_entry.get("upper") is None
                    else max(float(regime_entry["upper"]), float(entry["upper"]))
                )

        self.lambda_bounds = lambda_bounds
        return {
            "lambda_bounds": lambda_bounds,
            "scenario_lambda_bounds": scenario_lambda_bounds,
        }

    def run_dual_big_m_tightening(
        self,
        alpha_bounds: Optional[dict[tuple[int, ...], dict[str, float]]] = None,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Alpha bounds must be computed before dual Big-M tightening")
        fixed_binaries = fixed_binaries or getattr(self.poa, "fixed_binaries", {})
        lambda_report = self.run_lambda_bound_tightening(
            alpha_bounds=alpha_bounds,
            fixed_binaries=fixed_binaries,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)

        tight_big_m: dict[str, dict[str, Any]] = {}
        scenario_tight_big_m: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
        for side in ("eq", "opt"):
            for k in range(self.num_empirical_scenarios):
                for i, b in self.generator_block_pairs:
                    for t in range(self.num_time_steps):
                        tasks.append((side, "upper", (int(k), int(i), int(b), int(t)), (int(i), int(b), int(t))))
                        tasks.append((side, "lower", (int(k), int(i), int(b), int(t)), (int(i), int(b), int(t))))
                for i in range(self.num_physical_generators):
                    for t in range(self.num_time_steps):
                        tasks.append((side, "ramp_up", (int(k), int(i), int(t)), (int(i), int(t))))
                        tasks.append((side, "ramp_down", (int(k), int(i), int(t)), (int(i), int(t))))

        dual_parallel_tasks = [
            (
                side,
                constraint_type,
                scenario_index,
                regime_index,
                alpha_bounds,
                fixed_binaries,
                solver_name,
                time_limit,
                tee,
                solver_options,
            )
            for side, constraint_type, scenario_index, regime_index in tasks
        ]
        workers = self._resolve_parallel_workers(parallel_workers, len(dual_parallel_tasks))
        if workers > 1:
            print(
                f"Running DRO Dual Big-M programs with {workers} worker processes",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_dual_computer,
                initargs=(self._parallel_stage_state(),),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_dual_big_m, task): task
                    for task in dual_parallel_tasks
                }
                dual_results = []
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    print(
                        f"[DRO Dual Big-M {completed}/{len(dual_parallel_tasks)}] "
                        f"maximize {result['dual_name']}[{result['scenario_key']}] -> "
                        f"{result['tight_big_m']} ({result['termination_condition']})",
                        flush=True,
                    )
                    dual_results.append(result)
        else:
            global _PARALLEL_DUAL_COMPUTER
            _PARALLEL_DUAL_COMPUTER = self
            dual_results = [
                _solve_parallel_dual_big_m(task) for task in dual_parallel_tasks
            ]

        for result in dual_results:
            dual_name = str(result["dual_name"])
            regime_key = str(result["regime_key"])
            scenario_key = str(result["scenario_key"])
            details = {
                "tight_big_m": result["tight_big_m"],
                "fixed_by_slack": bool(result["fixed_by_slack"]),
                "termination_condition": result["termination_condition"],
            }
            scenario_tight_big_m.setdefault(dual_name, {})[scenario_key] = details
            current = tight_big_m.setdefault(dual_name, {}).get(regime_key)
            candidate_value = details["tight_big_m"]
            if current is None or (
                candidate_value is not None
                and float(candidate_value) > float(current.get("tight_big_m") or 0.0)
            ):
                tight_big_m.setdefault(dual_name, {})[regime_key] = {
                    **details,
                    "aggregation": "max over empirical scenarios k",
                }

        aggregate_dual_bounds: dict[str, dict[str, dict[str, Any]]] = {}
        aggregate_tasks = [
            (
                side,
                constraint_type,
                int(k),
                int(t),
                alpha_bounds,
                fixed_binaries,
                solver_name,
                time_limit,
                tee,
                solver_options,
            )
            for side in ("eq", "opt")
            for constraint_type in ("upper", "lower", "ramp_up", "ramp_down")
            for t in range(self.num_time_steps)
            for k in range(self.num_empirical_scenarios)
        ]
        aggregate_workers = self._resolve_parallel_workers(
            parallel_workers,
            len(aggregate_tasks),
        )
        if aggregate_workers > 1:
            print(
                f"Running DRO aggregate dual-bound programs with "
                f"{aggregate_workers} worker processes",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=aggregate_workers,
                initializer=_initialize_parallel_dual_computer,
                initargs=(self._parallel_stage_state(),),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_aggregate_dual_bound, task): task
                    for task in aggregate_tasks
                }
                aggregate_results = []
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    print(
                        f"[DRO Aggregate Dual {completed}/{len(aggregate_tasks)}] "
                        f"{result['side']}:{result['constraint_type']}:"
                        f"k={result['scenario_idx']},t={result['time_idx']} -> "
                        f"{result['tight_big_m']} ({result['termination_condition']})",
                        flush=True,
                    )
                    aggregate_results.append(result)
        else:
            _PARALLEL_DUAL_COMPUTER = self
            aggregate_results = [
                _solve_parallel_aggregate_dual_bound(task) for task in aggregate_tasks
            ]

        grouped_aggregate: dict[tuple[str, str, int], dict[str, Any]] = {}
        for result in aggregate_results:
            key = (
                str(result["side"]),
                str(result["constraint_type"]),
                int(result["time_idx"]),
            )
            candidate = {
                "tight_big_m": result["tight_big_m"],
                "side": result["side"],
                "constraint_type": result["constraint_type"],
                "termination_condition": result["termination_condition"],
            }
            best_details = grouped_aggregate.get(key)
            if best_details is None or (
                candidate["tight_big_m"] is not None
                and float(candidate["tight_big_m"])
                > float(best_details.get("tight_big_m") or 0.0)
            ):
                grouped_aggregate[key] = candidate

        for (side, constraint_type, t), best_details in grouped_aggregate.items():
            bound_key = self._aggregate_dual_bound_key(constraint_type)
            best_details["aggregation"] = "max over empirical scenarios k"
            aggregate_dual_bounds.setdefault(bound_key, {}).setdefault(side, {})[
                str(int(t))
            ] = best_details
            tight_big_m.setdefault(bound_key, {})[f"{side},{int(t)}"] = best_details

        self.tight_big_m = tight_big_m
        self.aggregate_dual_bounds = aggregate_dual_bounds
        return {
            "scenario_lambda_bounds": lambda_report["scenario_lambda_bounds"],
            "scenario_tight_big_m": scenario_tight_big_m,
            "regime_lambda_bounds": lambda_report["lambda_bounds"],
            "regime_tight_big_m": tight_big_m,
            "lambda_bounds": lambda_report["scenario_lambda_bounds"],
            "tight_big_m": scenario_tight_big_m,
            "aggregate_dual_bounds": aggregate_dual_bounds,
        }

    def run_dual_big_m(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or self._resolve_output_paths()["dual_big_m"]
        if not getattr(self.poa, "primal_big_m", None):
            raise ValueError("Primal Big-M values must be computed or loaded before dual Big-M.")
        if not getattr(self.poa, "alpha_bounds", None):
            raise ValueError("Alpha bounds must be computed or loaded before dual Big-M.")
        if "fixed_binaries" not in self.tightening_data:
            raise ValueError(
                "Slack-based complementarity binary fixing must be computed or loaded "
                "before dual Big-M."
            )

        start = time.perf_counter()
        dual_report = self.run_dual_big_m_tightening(
            alpha_bounds=self.poa.alpha_bounds,
            fixed_binaries=getattr(self.poa, "fixed_binaries", {}),
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        elapsed = time.perf_counter() - start
        report = {
            "metadata": {
                **self._metadata(),
                "description": (
                    "Scenario-indexed lambda and dual Big-M bounds. Regime-wide "
                    "aggregations are saved only as diagnostics/fallbacks."
                ),
                "tightening_scope": "scenario_wise",
                "runtime_seconds": elapsed,
            },
            "scenario_lambda_bounds": dual_report["scenario_lambda_bounds"],
            "regime_lambda_bounds": dual_report["regime_lambda_bounds"],
            "lambda_bounds": dual_report["lambda_bounds"],
            "scenario_tight_big_m": dual_report["scenario_tight_big_m"],
            "regime_tight_big_m": dual_report["regime_tight_big_m"],
            "tight_big_m": dual_report["tight_big_m"],
            "aggregate_dual_bounds": dual_report["aggregate_dual_bounds"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
            "fixed_binaries": self.tightening_data.get("fixed_binaries", {}),
        }
        self.tightening_data["scenario_lambda_bounds"] = report["scenario_lambda_bounds"]
        self.tightening_data["regime_lambda_bounds"] = report["regime_lambda_bounds"]
        self.tightening_data["lambda_bounds"] = report["lambda_bounds"]
        self.tightening_data["scenario_tight_big_m"] = report["scenario_tight_big_m"]
        self.tightening_data["regime_tight_big_m"] = report["regime_tight_big_m"]
        self.tightening_data["tight_big_m"] = report["tight_big_m"]
        self.tightening_data["aggregate_dual_bounds"] = report["aggregate_dual_bounds"]
        self.poa.lambda_bounds = report["lambda_bounds"]
        self.poa.tight_big_m = report["tight_big_m"]
        self.poa.aggregate_dual_bounds = report["aggregate_dual_bounds"]
        return self._save_stage_report("dual_big_m", report, output_path)
