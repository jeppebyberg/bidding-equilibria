from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Objective,
    Reals,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
    minimize,
    value,
)

from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import summarize_primal_big_m
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain


_PARALLEL_ALPHA_COMPUTER: Optional["DROAlphaBoundsComputer"] = None


def _initialize_parallel_alpha_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_ALPHA_COMPUTER
    _PARALLEL_ALPHA_COMPUTER = DROAlphaBoundsComputer._from_parallel_stage_state(state)


def _get_parallel_alpha_computer() -> "DROAlphaBoundsComputer":
    if _PARALLEL_ALPHA_COMPUTER is None:
        raise RuntimeError("Parallel DRO alpha computer was not initialized")
    return _PARALLEL_ALPHA_COMPUTER


def _solve_parallel_alpha_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    index, bound_name, solver_name, time_limit, tee, solver_options = task
    computer = _get_parallel_alpha_computer()
    m = computer._build_alpha_bound_model()
    alpha_expr = m.alpha[tuple(index)]
    m.tightening_objective = Objective(
        expr=alpha_expr,
        sense=minimize if bound_name == "lower" else maximize,
    )
    solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    return {
        "index": tuple(int(part) for part in index),
        "bound_name": str(bound_name),
        "value": computer._safe_value(alpha_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


class DROAlphaBoundsComputer(DROPoATighteningMain):
    def _build_tightening_sets(self) -> None:
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

    def _build_alpha_bound_model(self) -> ConcreteModel:
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()
        self.model.alpha = Var(
            self.model.scenarios,
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
        )
        self._build_support_set()
        self._build_transport_constraints()
        self._build_policy_constraints()
        return self.model

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

    @staticmethod
    def _safe_value(expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        return None if raw_value is None else float(raw_value)

    def _true_cost_alpha_bounds(self) -> dict[str, Any]:
        scenario_bounds: dict[tuple[int, int, int, int], dict[str, float]] = {}
        optimization_results: dict[str, dict[str, Any]] = {}
        for k in range(self.num_empirical_scenarios):
            for i, b in self.generator_block_pairs:
                i = int(i)
                b = int(b)
                global_block = self.local_to_global_block[(i, b)]
                bid_value = float(self.block_cost_vector[global_block])
                for t in range(self.num_time_steps):
                    scenario_index = (int(k), i, b, int(t))
                    scenario_bounds[scenario_index] = {
                        "lower": bid_value,
                        "upper": bid_value,
                    }
                    optimization_results[f"{self._json_key(scenario_index)}:analytic"] = {
                        "value": bid_value,
                        "termination_condition": "true_cost_policy_fixed",
                    }

        alpha_bounds = self.aggregate_lower_upper_bounds_over_k(scenario_bounds)
        return {
            "alpha_bounds": alpha_bounds,
            "scenario_alpha_bounds": self._jsonify_indexed_dict(scenario_bounds),
            "optimization_results": optimization_results,
            "num_optimization_programs": 0,
        }

    def compute_nn_certified_bid_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        if getattr(self.poa, "nn_policy_generator_ids", []):
            if not getattr(self.poa, "nn_relu_bounds", {}):
                raise ValueError(
                    "DRO ReLU bounds are required before alpha-bound tightening "
                    "for NN policy generators."
                )
            if not getattr(self.poa, "nn_policies", {}):
                self.poa._load_nn_policies()
                self.poa._load_nn_normalization_stats()

            scenario_alpha_bounds: dict[tuple[int, int, int, int], dict[str, float]] = {}
            optimization_results: dict[str, dict[str, Any]] = {}
            targets = [
                (int(k), int(i), int(b), int(t))
                for k in range(self.num_empirical_scenarios)
                for i, b in self.generator_block_pairs
                for t in range(self.num_time_steps)
            ]
            total_programs = 2 * len(targets)
            solver_options = self._solver_options_with_threads(solver_name, solver_threads)
            workers = self._resolve_parallel_workers(parallel_workers, total_programs)
            if workers > 1:
                print(
                    f"Running DRO alpha-bound programs with {workers} worker processes",
                    flush=True,
                )
                parallel_tasks = [
                    (index, bound_name, solver_name, time_limit, tee, solver_options)
                    for index in targets
                    for bound_name in ("lower", "upper")
                ]
                result_by_task: dict[tuple[tuple[int, int, int, int], str], dict[str, Any]] = {}
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_parallel_alpha_computer,
                    initargs=(self._parallel_stage_state(),),
                ) as executor:
                    future_to_task = {
                        executor.submit(_solve_parallel_alpha_bound, task): task
                        for task in parallel_tasks
                    }
                    for completed, future in enumerate(as_completed(future_to_task), start=1):
                        result = future.result()
                        index = tuple(result["index"])
                        bound_name = str(result["bound_name"])
                        result_by_task[(index, bound_name)] = result
                        action = "minimize" if bound_name == "lower" else "maximize"
                        print(
                            f"[DRO Alpha {completed}/{total_programs}] "
                            f"{action} alpha{index} -> {result['value']} "
                            f"({result['termination_condition']})",
                            flush=True,
                        )

                for index in targets:
                    lower = result_by_task[(index, "lower")]["value"]
                    upper = result_by_task[(index, "upper")]["value"]
                    if lower is None or upper is None:
                        raise RuntimeError(f"Could not compute alpha bounds for index {index}")
                    scenario_alpha_bounds[index] = {
                        "lower": float(lower),
                        "upper": float(upper),
                    }
                    for bound_name in ("lower", "upper"):
                        result = result_by_task[(index, bound_name)]
                        optimization_results[f"{self._json_key(index)}:{bound_name}"] = {
                            "value": result["value"],
                            "termination_condition": result["termination_condition"],
                        }
            else:
                for program_number, index in enumerate(targets, start=1):
                    lower_upper: dict[str, Optional[float]] = {"lower": None, "upper": None}
                    for bound_name, sense in (("lower", minimize), ("upper", maximize)):
                        print(
                            f"[DRO Alpha {2 * (program_number - 1) + (1 if bound_name == 'lower' else 2)}/"
                            f"{total_programs}] "
                            f"{'minimize' if bound_name == 'lower' else 'maximize'} "
                            f"alpha{index}",
                            flush=True,
                        )
                        m = self._build_alpha_bound_model()
                        alpha_expr = m.alpha[index]
                        m.tightening_objective = Objective(expr=alpha_expr, sense=sense)
                        solved, results = self._solve_tightening_model(
                            m,
                            solver_name=solver_name,
                            time_limit=time_limit,
                            tee=tee,
                            solver_options=solver_options,
                        )
                        bound_value = self._safe_value(alpha_expr) if solved else None
                        lower_upper[bound_name] = bound_value
                        optimization_results[f"{self._json_key(index)}:{bound_name}"] = {
                            "value": bound_value,
                            "termination_condition": str(results.solver.termination_condition),
                        }
                    if lower_upper["lower"] is None or lower_upper["upper"] is None:
                        raise RuntimeError(f"Could not compute alpha bounds for index {index}")
                    scenario_alpha_bounds[index] = {
                        "lower": float(lower_upper["lower"]),
                        "upper": float(lower_upper["upper"]),
                    }

            alpha_bounds = self.aggregate_lower_upper_bounds_over_k(scenario_alpha_bounds)
            return {
                "alpha_bounds": alpha_bounds,
                "scenario_alpha_bounds": self._jsonify_indexed_dict(scenario_alpha_bounds),
                "optimization_results": optimization_results,
                "num_optimization_programs": total_programs,
            }

        # The current DRO optimizer's policy constraints fix alpha to true costs,
        # so the exact certified alpha bounds are available without MILP OBBT.
        del solver_name, time_limit, tee, solver_threads
        return self._true_cost_alpha_bounds()

    def run_alpha_bounds(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or self._resolve_output_paths()["alpha_bounds"]

        if getattr(self.poa, "nn_policy_generator_ids", []) and not getattr(
            self.poa,
            "nn_relu_bounds",
            {},
        ):
            raise ValueError("ReLU bounds must be computed or loaded before alpha-bound tightening.")

        start = time.perf_counter()
        alpha_report = self.compute_nn_certified_bid_bounds(
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
                    "Certified regime-wide alpha bounds for DRO PoA. Internal "
                    "scenario-indexed bounds are aggregated to i,b,t keys."
                ),
                "num_optimization_programs": alpha_report["num_optimization_programs"],
                "primal_big_m_summary": summarize_primal_big_m(
                    self.tightening_data.get("primal_big_m", {})
                ),
                "runtime_seconds": elapsed,
            },
            "alpha_bounds": alpha_report["alpha_bounds"],
            "alpha_optimization_results": alpha_report["optimization_results"],
            "num_optimization_programs": alpha_report["num_optimization_programs"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
        }
        self.tightening_data["alpha_bounds"] = report["alpha_bounds"]
        self.tightening_data["alpha_optimization_results"] = report[
            "alpha_optimization_results"
        ]
        self.poa.alpha_bounds = self._parse_alpha_bounds(report["alpha_bounds"])
        self.poa.alpha_bound_optimization_results = report["alpha_optimization_results"]
        return self._save_stage_report("alpha_bounds", report, output_path)
