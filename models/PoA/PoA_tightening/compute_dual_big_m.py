from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import *

from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.PoA_tightening.compute_primal_big_m import compute_primal_big_m_bounds
from models.PoA.PoA_tightening.tightening_main import (
    DEFAULT_TIGHTENING_OUTPUT_PATHS,
    PoATighteningMain,
)


_PARALLEL_DUAL_COMPUTER: Optional["DualBigMComputer"] = None


def _initialize_parallel_dual_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_DUAL_COMPUTER
    poa = PoAOptimization(
        scenarios_df=state["scenarios_df"],
        costs_df=state["costs_df"],
        ramps_df=state["ramps_df"],
        p_init=state["p_init"],
        num_time_steps=state["num_time_steps"],
        support_set_config=state["support_set_config"],
        nn_model_dir=state["nn_model_dir"],
        nn_normalization_stats_path=state["nn_normalization_stats_path"],
        nn_policy_generators=state["nn_policy_generators"],
        reference_case=state["reference_case"],
    )
    for attr_name in (
        "alpha_bounds",
        "fixed_binaries",
        "primal_big_m",
        "lambda_bounds",
        "tight_big_m",
        "aggregate_dual_bounds",
    ):
        if attr_name in state:
            setattr(poa, attr_name, state[attr_name])
    if state.get("nn_relu_bounds_report"):
        poa._set_nn_relu_bounds_from_report(state["nn_relu_bounds_report"])

    computer = DualBigMComputer.__new__(DualBigMComputer)
    computer.poa = poa
    computer.tightening_data = {}
    computer.stage_reports = {}
    _PARALLEL_DUAL_COMPUTER = computer


def _get_parallel_dual_computer() -> "DualBigMComputer":
    if _PARALLEL_DUAL_COMPUTER is None:
        raise RuntimeError("Parallel dual Big-M worker was not initialized")
    return _PARALLEL_DUAL_COMPUTER


def _bound_sense(bound_name: str) -> Any:
    if bound_name == "lower":
        return minimize
    if bound_name == "upper":
        return maximize
    raise ValueError(f"Unknown bound name: {bound_name}")


def _solve_parallel_lambda_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    side, time_idx, bound_name, solver_name, time_limit, tee, solver_options = task
    computer = _get_parallel_dual_computer()
    lambda_name = computer._lambda_name(side)
    m = computer._build_side_kkt_model_for_dual_big_m(
        side=side,
        alpha_bounds=computer.alpha_bounds,
        include_complementarity=True,
        fixed_binaries=computer.fixed_binaries,
    )
    lambda_expr = getattr(m, lambda_name)[int(time_idx)]
    m.tightening_objective = Objective(expr=lambda_expr, sense=_bound_sense(bound_name))
    solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    return {
        "side": side,
        "time_idx": int(time_idx),
        "lambda_name": lambda_name,
        "bound_name": bound_name,
        "value": computer._safe_value(lambda_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


def _solve_parallel_dual_big_m(task: tuple[Any, ...]) -> dict[str, Any]:
    side, constraint_type, index, solver_name, time_limit, tee, solver_options = task
    computer = _get_parallel_dual_computer()
    dual_name = computer._dual_name(side, constraint_type)
    m = computer._build_side_kkt_model_for_dual_big_m(
        side=side,
        alpha_bounds=computer.alpha_bounds,
        include_complementarity=True,
        fixed_binaries=computer.fixed_binaries,
    )
    dual_expr = getattr(m, dual_name)[index]
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
        "index": index,
        "dual_name": dual_name,
        "tight_big_m": computer._safe_value(dual_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


def _solve_parallel_aggregate_dual_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    side, constraint_type, time_idx, solver_name, time_limit, tee, solver_options = task
    computer = _get_parallel_dual_computer()
    bound_key = computer._aggregate_dual_bound_key(constraint_type)
    m = computer._build_side_kkt_model_for_dual_big_m(
        side=side,
        alpha_bounds=computer.alpha_bounds,
        include_complementarity=True,
        fixed_binaries=computer.fixed_binaries,
    )
    aggregate_expr = computer._aggregate_dual_expression(
        m,
        side,
        constraint_type,
        time_idx,
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
        "time_idx": int(time_idx),
        "bound_key": bound_key,
        "tight_big_m": computer._safe_value(aggregate_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


class DualBigMComputer(PoATighteningMain):
    def _ensure_primal_big_m_for_tightening(self) -> None:
        if getattr(self.poa, "primal_big_m", {}) or {}:
            return
        self.poa.primal_big_m = compute_primal_big_m_bounds(self.poa)
        self.poa._loaded_bounds_prepared = False

    def _parallel_optimizer_state_for_dual(
        self,
        **extra_state: Any,
    ) -> dict[str, Any]:
        state = {
            "scenarios_df": self.poa.scenarios_df,
            "costs_df": self.poa.costs_df,
            "ramps_df": self.poa.ramps_df,
            "p_init": self.poa.requested_p_init,
            "num_time_steps": self.poa.num_time_steps,
            "support_set_config": self.poa.support_set_config,
            "nn_model_dir": (
                str(self.poa.nn_model_dir) if self.poa.nn_model_dir is not None else None
            ),
            "nn_normalization_stats_path": (
                str(self.poa.nn_normalization_stats_path)
                if self.poa.nn_normalization_stats_path is not None
                else None
            ),
            "nn_policy_generators": list(self.poa.nn_policy_generator_ids),
            "reference_case": self.poa.reference_case,
            "nn_relu_bounds_report": getattr(self.poa, "nn_relu_bounds_report", {}) or {},
            "primal_big_m": getattr(self.poa, "primal_big_m", {}) or {},
        }
        state.update(extra_state)
        return state

    @staticmethod
    def _resolve_parallel_workers(
        parallel_workers: Optional[int],
        total_tasks: int,
    ) -> int:
        if total_tasks <= 1 or parallel_workers is None:
            return 1
        return min(max(1, int(parallel_workers)), int(total_tasks))

    @staticmethod
    def _solver_options_with_threads(
        solver_name: str,
        solver_threads: Optional[int],
        extra_options: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        options = dict(extra_options or {})
        if solver_threads is not None and solver_name == "gurobi":
            options["Threads"] = int(solver_threads)
        return options or None

    def _build_tightening_sets(self) -> None:
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

    def _build_side_kkt_model_for_dual_big_m(
        self,
        side: str,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        include_complementarity: bool = True,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> ConcreteModel:
        """
        Build one KKT side for lambda, componentwise dual, or aggregate dual bounds.
        """
        self._ensure_primal_big_m_for_tightening()
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()

        if side == "eq":
            if alpha_bounds is None:
                raise ValueError("Equilibrium KKT dual tightening requires alpha_bounds")
            self._build_equilibrium_variables()
            self._build_complementarity_equilibrium_variables()
            self._build_support_set()
            self._apply_alpha_bounds(self.model, alpha_bounds)
            self._build_lower_level_equilibrium_constraints()
            self._build_KKT_stationarity_equilibrium_constraints()
            if include_complementarity:
                self._build_KKT_complementarity_equilibrium_constraints()
                self._apply_fixed_binaries(self.model, fixed_binaries)
            else:
                for var_name in self._binary_components_for_side(side):
                    binary_var = getattr(self.model, var_name, None)
                    if binary_var is not None:
                        for index in binary_var:
                            binary_var[index].fix(0)
        elif side == "opt":
            self._build_optimal_variables()
            self._build_complementarity_optimal_variables()
            self._build_support_set()
            self._build_lower_level_optimal_constraints()
            self._build_KKT_stationarity_optimal_constraints()
            if include_complementarity:
                self._build_KKT_complementarity_optimal_constraints()
                self._apply_fixed_binaries(self.model, fixed_binaries)
            else:
                for var_name in self._binary_components_for_side(side):
                    binary_var = getattr(self.model, var_name, None)
                    if binary_var is not None:
                        for index in binary_var:
                            binary_var[index].fix(0)
        else:
            raise ValueError(f"Unknown KKT tightening side: {side}")

        return self.model

    def _apply_alpha_bounds(
        self,
        m: ConcreteModel,
        alpha_bounds: dict[tuple[int, int, int], dict[str, float]],
    ) -> None:
        m.alpha_certified_bounds = ConstraintList()
        for i, b in m.generator_blocks:
            for t in m.time_steps:
                bounds = alpha_bounds[(int(i), int(b), int(t))]
                lower = float(bounds["lower"])
                upper = float(bounds["upper"])
                m.alpha_certified_bounds.add(m.alpha[i, b, t] >= lower)
                m.alpha_certified_bounds.add(m.alpha[i, b, t] <= upper)
                m.alpha[i, b, t].setlb(lower)
                m.alpha[i, b, t].setub(upper)

    def _apply_fixed_binaries(
        self,
        m: ConcreteModel,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        fixed_binaries = fixed_binaries or getattr(self.poa, "fixed_binaries", {})
        for var_name, entries in fixed_binaries.items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, _details in entries.items():
                index = tuple(int(part) for part in str(key).split(",") if part != "")
                binary_var[index].fix(0)

    def _binary_components_for_side(self, side: str) -> tuple[str, ...]:
        if side == "eq":
            return ("z_upper_eq", "z_lower_eq", "z_ramp_up_eq", "z_ramp_down_eq")
        if side == "opt":
            return ("z_upper_opt", "z_lower_opt", "z_ramp_up_opt", "z_ramp_down_opt")
        raise ValueError(f"Unknown tightening side: {side}")

    def _binary_name(self, side: str, constraint_type: str) -> str:
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

    def _dual_name(self, side: str, constraint_type: str) -> str:
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

    def _lambda_name(self, side: str) -> str:
        return {
            "eq": "lambda_eq",
            "opt": "lambda_opt",
        }[side]

    def _aggregate_dual_bound_key(self, constraint_type: str) -> str:
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
        time_idx: int,
    ) -> Any:
        dual_var = getattr(m, self._dual_name(side, constraint_type))
        if constraint_type in {"upper", "lower"}:
            return sum(
                dual_var[i, b, int(time_idx)]
                for i, b in self.generator_block_pairs
            )
        return sum(
            dual_var[i, int(time_idx)]
            for i in range(self.num_physical_generators)
        )

    def _solve_tightening_model(
        self,
        m: ConcreteModel,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_options: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Any]:
        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if solver_options:
            for option_name, option_value in solver_options.items():
                solver.options[option_name] = option_value
        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        ok = termination in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
        return bool(ok), results

    def run_lambda_bound_tightening(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: Optional[int] = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Maximize and minimize each price dual before the dual Big-M programs.
        """
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before lambda tightening")
        fixed_binaries = fixed_binaries or getattr(self.poa, "fixed_binaries", {})

        self.lambda_bounds = {}

        lambda_bounds: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[str, int]] = [
            (side, t)
            for side in ("eq", "opt")
            for t in range(self.num_time_steps)
        ]
        print(
            f"\nLambda-bound optimization programs: {2 * len(tasks)}",
            flush=True,
        )
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)
        total_programs = 2 * len(tasks)
        workers = self._resolve_parallel_workers(parallel_workers, total_programs)
        if workers > 1:
            print(f"Running lambda-bound programs with {workers} worker processes", flush=True)
            parallel_tasks = [
                (side, time_idx, bound_name, solver_name, time_limit, tee, solver_options)
                for side, time_idx in tasks
                for bound_name in ("lower", "upper")
            ]
            result_by_task: dict[tuple[str, int, str], dict[str, Any]] = {}
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_dual_computer,
                initargs=(
                    self._parallel_optimizer_state_for_dual(
                        alpha_bounds=alpha_bounds,
                        fixed_binaries=fixed_binaries,
                        lambda_bounds={},
                    ),
                ),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_lambda_bound, task): task
                    for task in parallel_tasks
                }
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    key = (
                        str(result["side"]),
                        int(result["time_idx"]),
                        str(result["bound_name"]),
                    )
                    result_by_task[key] = result
                    action = "minimize" if result["bound_name"] == "lower" else "maximize"
                    print(
                        f"[Lambda done {completed}/{total_programs}] {action} "
                        f"{result['lambda_name']}[{result['time_idx']}] -> "
                        f"{result['value']} ({result['termination_condition']})",
                        flush=True,
                    )

            for side, time_idx in tasks:
                lambda_name = self._lambda_name(side)
                entry: dict[str, Any] = {}
                for bound_name in ("lower", "upper"):
                    result = result_by_task[(side, int(time_idx), bound_name)]
                    entry[bound_name] = result["value"]
                    entry[f"{bound_name}_termination_condition"] = result[
                        "termination_condition"
                    ]
                lambda_bounds.setdefault(lambda_name, {})[str(int(time_idx))] = entry

            self.lambda_bounds = lambda_bounds
            return {"lambda_bounds": lambda_bounds}

        for program_idx, (side, time_idx) in enumerate(tasks, start=1):
            lambda_name = self._lambda_name(side)
            entry: dict[str, Any] = {}
            for bound_name, sense in (("lower", minimize), ("upper", maximize)):
                print(
                    f"[Lambda {2 * (program_idx - 1) + (1 if bound_name == 'lower' else 2)}/"
                    f"{2 * len(tasks)}] "
                    f"{'minimize' if bound_name == 'lower' else 'maximize'} "
                    f"{lambda_name}[{time_idx}]",
                    flush=True,
                )
                m = self._build_side_kkt_model_for_dual_big_m(
                    side=side,
                    alpha_bounds=alpha_bounds,
                    include_complementarity=True,
                    fixed_binaries=fixed_binaries,
                )
                lambda_expr = getattr(m, lambda_name)[int(time_idx)]
                m.tightening_objective = Objective(expr=lambda_expr, sense=sense)
                solved, results = self._solve_tightening_model(
                    m,
                    solver_name=solver_name,
                    time_limit=time_limit,
                    tee=tee,
                    solver_options=solver_options,
                )
                entry[bound_name] = self._safe_value(lambda_expr) if solved else None
                entry[f"{bound_name}_termination_condition"] = str(
                    results.solver.termination_condition
                )

            lambda_bounds.setdefault(lambda_name, {})[str(int(time_idx))] = entry

        self.lambda_bounds = lambda_bounds
        return {"lambda_bounds": lambda_bounds}

    def run_dual_big_m_tightening(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: Optional[int] = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Maximize each dual variable and aggregate dual sum after slack fixing.
        """
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before Big-M tightening")
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
        tasks: list[tuple[str, str, tuple[int, ...]]] = []
        for side in ("eq", "opt"):
            for i, b in self.generator_block_pairs:
                for t in range(self.num_time_steps):
                    tasks.append((side, "upper", (int(i), int(b), int(t))))
                    tasks.append((side, "lower", (int(i), int(b), int(t))))
            for i in range(self.num_physical_generators):
                for t in range(self.num_time_steps):
                    tasks.append((side, "ramp_up", (int(i), int(t))))
                    tasks.append((side, "ramp_down", (int(i), int(t))))

        total_candidates = len(tasks)
        skipped_programs = sum(
            1
            for side, constraint_type, index in tasks
            if self._json_key(index)
            in fixed_binaries.get(self._binary_name(side, constraint_type), {})
        )
        total_programs = total_candidates - skipped_programs
        program_number = 0
        print(
            f"\nDual Big-M optimization programs: {total_programs} "
            f"({skipped_programs} skipped because slack fixed the binary)",
            flush=True,
        )

        pending_dual_tasks: list[tuple[str, str, tuple[int, ...]]] = []
        for side, constraint_type, index in tasks:
            dual_name = self._dual_name(side, constraint_type)
            binary_name = self._binary_name(side, constraint_type)
            key = self._json_key(index)
            if key in fixed_binaries.get(binary_name, {}):
                print(
                    f"[Dual Big-M skip] side={side}, constraint={constraint_type}, "
                    f"index={index}, binary={binary_name} fixed to 0",
                    flush=True,
                )
                tight_big_m.setdefault(dual_name, {})[key] = {
                    "tight_big_m": 0.0,
                    "fixed_by_slack": True,
                    "termination_condition": "fixed_binary_zero",
                }
                continue
            pending_dual_tasks.append((side, constraint_type, index))

        workers = self._resolve_parallel_workers(parallel_workers, total_programs)
        if workers > 1:
            print(f"Running Dual Big-M programs with {workers} worker processes", flush=True)
            parallel_tasks = [
                (side, constraint_type, index, solver_name, time_limit, tee, solver_options)
                for side, constraint_type, index in pending_dual_tasks
            ]
            result_by_task: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_dual_computer,
                initargs=(
                    self._parallel_optimizer_state_for_dual(
                        alpha_bounds=alpha_bounds,
                        fixed_binaries=fixed_binaries,
                        lambda_bounds=self.lambda_bounds,
                        tight_big_m={},
                    ),
                ),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_dual_big_m, task): task
                    for task in parallel_tasks
                }
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    record_key = (
                        str(result["side"]),
                        str(result["constraint_type"]),
                        tuple(result["index"]),
                    )
                    result_by_task[record_key] = result
                    print(
                        f"[Dual Big-M done {completed}/{total_programs}] "
                        f"maximize {result['dual_name']}{result['index']} -> "
                        f"{result['tight_big_m']} ({result['termination_condition']})",
                        flush=True,
                    )

            for side, constraint_type, index in pending_dual_tasks:
                result = result_by_task[(side, constraint_type, index)]
                dual_name = self._dual_name(side, constraint_type)
                key = self._json_key(index)
                tight_big_m.setdefault(dual_name, {})[key] = {
                    "tight_big_m": result["tight_big_m"],
                    "fixed_by_slack": False,
                    "termination_condition": result["termination_condition"],
                }
        else:
            for side, constraint_type, index in pending_dual_tasks:
                dual_name = self._dual_name(side, constraint_type)

                program_number += 1
                print(
                    f"[Dual Big-M {program_number}/{total_programs}] "
                    f"maximize {dual_name}{index} for side={side}, "
                    f"constraint={constraint_type}",
                    flush=True,
                )
                m = self._build_side_kkt_model_for_dual_big_m(
                    side=side,
                    alpha_bounds=alpha_bounds,
                    include_complementarity=True,
                    fixed_binaries=fixed_binaries,
                )
                dual_var = getattr(m, dual_name)
                dual_expr = dual_var[index]
                m.tightening_objective = Objective(expr=dual_expr, sense=maximize)
                solved, results = self._solve_tightening_model(
                    m,
                    solver_name,
                    time_limit,
                    tee,
                    solver_options=solver_options,
                )
                dual_bound = self._safe_value(dual_expr) if solved else None
                tight_big_m.setdefault(dual_name, {})[self._json_key(index)] = {
                    "tight_big_m": dual_bound,
                    "fixed_by_slack": False,
                    "termination_condition": str(results.solver.termination_condition),
                }

        self.tight_big_m = tight_big_m
        aggregate_dual_bounds: dict[str, dict[str, dict[str, Any]]] = {}
        aggregate_tasks: list[tuple[str, str, int]] = [
            (side, constraint_type, t)
            for side in ("eq", "opt")
            for constraint_type in ("upper", "lower", "ramp_up", "ramp_down")
            for t in range(self.num_time_steps)
        ]
        print(
            f"\nAggregate dual-bound optimization programs: {len(aggregate_tasks)}",
            flush=True,
        )
        aggregate_workers = self._resolve_parallel_workers(
            parallel_workers,
            len(aggregate_tasks),
        )
        if aggregate_workers > 1:
            print(
                f"Running aggregate dual-bound programs with {aggregate_workers} "
                "worker processes",
                flush=True,
            )
            parallel_tasks = [
                (side, constraint_type, time_idx, solver_name, time_limit, tee, solver_options)
                for side, constraint_type, time_idx in aggregate_tasks
            ]
            result_by_task: dict[tuple[str, str, int], dict[str, Any]] = {}
            with ProcessPoolExecutor(
                max_workers=aggregate_workers,
                initializer=_initialize_parallel_dual_computer,
                initargs=(
                    self._parallel_optimizer_state_for_dual(
                        alpha_bounds=alpha_bounds,
                        fixed_binaries=fixed_binaries,
                        lambda_bounds=self.lambda_bounds,
                        tight_big_m=tight_big_m,
                    ),
                ),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_aggregate_dual_bound, task): task
                    for task in parallel_tasks
                }
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    result_by_task[
                        (
                            str(result["side"]),
                            str(result["constraint_type"]),
                            int(result["time_idx"]),
                        )
                    ] = result
                    print(
                        f"[Aggregate dual bound done {completed}/{len(aggregate_tasks)}] "
                        f"maximize {result['bound_key']}[{result['side']},"
                        f"{result['time_idx']}] -> {result['tight_big_m']} "
                        f"({result['termination_condition']})",
                        flush=True,
                    )

            for side, constraint_type, time_idx in aggregate_tasks:
                result = result_by_task[(side, constraint_type, int(time_idx))]
                bound_key = self._aggregate_dual_bound_key(constraint_type)
                details = {
                    "tight_big_m": result["tight_big_m"],
                    "side": side,
                    "constraint_type": constraint_type,
                    "termination_condition": result["termination_condition"],
                }
                aggregate_dual_bounds.setdefault(bound_key, {}).setdefault(side, {})[
                    str(int(time_idx))
                ] = details
                tight_big_m.setdefault(bound_key, {})[f"{side},{int(time_idx)}"] = details
        else:
            for program_number, (side, constraint_type, time_idx) in enumerate(
                aggregate_tasks,
                start=1,
            ):
                bound_key = self._aggregate_dual_bound_key(constraint_type)
                print(
                    f"[Aggregate dual bound {program_number}/{len(aggregate_tasks)}] "
                    f"maximize {bound_key}[{side},{time_idx}]",
                    flush=True,
                )
                m = self._build_side_kkt_model_for_dual_big_m(
                    side=side,
                    alpha_bounds=alpha_bounds,
                    include_complementarity=True,
                    fixed_binaries=fixed_binaries,
                )
                aggregate_expr = self._aggregate_dual_expression(
                    m,
                    side,
                    constraint_type,
                    time_idx,
                )
                m.tightening_objective = Objective(expr=aggregate_expr, sense=maximize)
                solved, results = self._solve_tightening_model(
                    m,
                    solver_name,
                    time_limit,
                    tee,
                    solver_options=solver_options,
                )
                aggregate_bound = self._safe_value(aggregate_expr) if solved else None
                details = {
                    "tight_big_m": aggregate_bound,
                    "side": side,
                    "constraint_type": constraint_type,
                    "termination_condition": str(results.solver.termination_condition),
                }
                aggregate_dual_bounds.setdefault(bound_key, {}).setdefault(side, {})[
                    str(int(time_idx))
                ] = details
                tight_big_m.setdefault(bound_key, {})[
                    f"{side},{int(time_idx)}"
                ] = details

        self.tight_big_m = tight_big_m
        self.aggregate_dual_bounds = aggregate_dual_bounds
        return {
            "lambda_bounds": lambda_report["lambda_bounds"],
            "tight_big_m": tight_big_m,
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
    ) -> dict[str, object]:
        output_path = output_path or DEFAULT_TIGHTENING_OUTPUT_PATHS["dual_big_m"]

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
            fixed_binaries=self.poa.fixed_binaries,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        elapsed = time.perf_counter() - start

        report = {
            "metadata": {
                "description": (
                    "Lambda bounds, componentwise dual Big-M values, and aggregate "
                    "dual bounds for the final PoA KKT model."
                ),
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "runtime_seconds": elapsed,
            },
            "lambda_bounds": dual_report["lambda_bounds"],
            "tight_big_m": dual_report["tight_big_m"],
            "aggregate_dual_bounds": dual_report["aggregate_dual_bounds"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
            "alpha_optimization_results": self.tightening_data.get(
                "alpha_optimization_results",
                {},
            ),
            "slack_bounds": self.tightening_data.get("slack_bounds", {}),
            "fixed_binaries": self.tightening_data.get("fixed_binaries", {}),
        }
        self.tightening_data["lambda_bounds"] = report["lambda_bounds"]
        self.tightening_data["tight_big_m"] = report["tight_big_m"]
        self.tightening_data["aggregate_dual_bounds"] = report["aggregate_dual_bounds"]
        self.poa.lambda_bounds = report["lambda_bounds"]
        self.poa.tight_big_m = report["tight_big_m"]
        self.poa.aggregate_dual_bounds = report["aggregate_dual_bounds"]
        return self._save_stage_report("dual_big_m", report, output_path)
