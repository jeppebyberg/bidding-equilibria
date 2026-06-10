from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import *

from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.PoA_tightening.compute_primal_big_m import summarize_primal_big_m
from models.PoA.PoA_tightening.tightening_main import (
    DEFAULT_TIGHTENING_OUTPUT_PATHS,
    PoATighteningMain,
)


_PARALLEL_ALPHA_COMPUTER: Optional["AlphaBoundsComputer"] = None


def _initialize_parallel_alpha_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_ALPHA_COMPUTER
    poa = PoAOptimization(
        scenarios_df=state["scenarios_df"],
        costs_df=state["costs_df"],
        ramps_df=state["ramps_df"],
        num_time_steps=state["num_time_steps"],
        ambiguity_set_config=state["ambiguity_set_config"],
        nn_model_dir=state["nn_model_dir"],
        nn_normalization_stats_path=state["nn_normalization_stats_path"],
        nn_policy_generators=state["nn_policy_generators"],
        reference_case=state["reference_case"],
        objective_mode=state.get("objective_mode", "difference"),
        mccormick_bounds=state.get("mccormick_bounds"),
        use_default_bounds=bool(state.get("use_default_bounds", False)),
    )
    if "p_init" in state:
        poa.p_init = state["p_init"]
    if state.get("primal_big_m"):
        poa.primal_big_m = state["primal_big_m"]
    if state.get("nn_relu_bounds_report"):
        poa._set_nn_relu_bounds_from_report(state["nn_relu_bounds_report"])

    computer = AlphaBoundsComputer.__new__(AlphaBoundsComputer)
    computer.poa = poa
    computer.tightening_data = {}
    computer.stage_reports = {}
    _PARALLEL_ALPHA_COMPUTER = computer


def _get_parallel_alpha_computer() -> "AlphaBoundsComputer":
    if _PARALLEL_ALPHA_COMPUTER is None:
        raise RuntimeError("Parallel alpha-bound worker was not initialized")
    return _PARALLEL_ALPHA_COMPUTER


def _bound_sense(bound_name: str) -> Any:
    if bound_name == "lower":
        return minimize
    if bound_name == "upper":
        return maximize
    raise ValueError(f"Unknown bound name: {bound_name}")


def _solve_parallel_alpha_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    index, bound_name, solver_name, time_limit, tee, solver_options = task
    computer = _get_parallel_alpha_computer()
    m = computer._build_alpha_bound_model()
    alpha_expr = m.alpha[index]
    m.tightening_objective = Objective(expr=alpha_expr, sense=_bound_sense(bound_name))
    solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    return {
        "index": index,
        "bound_name": bound_name,
        "value": computer._safe_value(alpha_expr) if solved else None,
        "termination_condition": str(results.solver.termination_condition),
    }


class AlphaBoundsComputer(PoATighteningMain):
    def _parallel_optimizer_state_for_alpha(self) -> dict[str, Any]:
        return {
            "scenarios_df": self.poa.scenarios_df,
            "costs_df": self.poa.costs_df,
            "ramps_df": self.poa.ramps_df,
            "p_init": list(self.poa.p_init),
            "num_time_steps": self.poa.num_time_steps,
            "ambiguity_set_config": self.poa.ambiguity_set_config,
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
            "objective_mode": self.poa.objective_mode,
            "mccormick_bounds": self.poa.mccormick_bounds,
            "use_default_bounds": self.poa.use_default_bounds,
            "nn_relu_bounds_report": getattr(self.poa, "nn_relu_bounds_report", {}) or {},
            "primal_big_m": getattr(self.poa, "primal_big_m", {}) or {},
        }

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
        """
        Build the common index sets used by the explicit alpha-bound model.
        """
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
        """
        Ambiguity-induced support set plus policy/ReLU constraints for certified alpha bounds.

        No lower-level dispatch, KKT stationarity, complementarity, PoA
        objective, or slack minimization is present. For NN-controlled
        generators, alpha is linked to the embedded ReLU policy. For
        non-NN generators, alpha is fixed by the same policy builder used in
        the final PoA model.
        """
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()
        self.model.alpha = Var(self.model.generator_blocks, self.model.time_steps, domain=Reals)
        self._build_support_set()
        self._build_policy_constraints()
        return self.model

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

    def compute_nn_certified_bid_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: Optional[int] = 1,
        solver_threads: Optional[int] = None,
        nn_relu_bounds_report_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        Compute exact ambiguity-set bid bounds.

        For each bidding block and time step, solve:

            min alpha[i,b,t]
            max alpha[i,b,t]

        subject only to the support set and policy constraints. For
        NN-controlled generators this includes the ReLU MILP embedding.
        """
        if self.nn_policy_generator_ids and not self.nn_policies:
            self._load_nn_policies()
            self._load_nn_normalization_stats()
        if nn_relu_bounds_report_path is not None:
            self.load_nn_relu_bounds_report(nn_relu_bounds_report_path)
        if self.nn_policy_generator_ids and not self.nn_relu_bounds:
            raise ValueError(
                "NN ReLU bounds are required before computing certified alpha bounds. "
                "Run compute_relu_bounds.py first or pass nn_relu_bounds_report_path."
            )

        alpha_bounds: dict[tuple[int, int, int], dict[str, float]] = {}
        optimization_results: dict[str, dict[str, Any]] = {}
        targets = [
            (int(i), int(b), int(t))
            for i, b in self.generator_block_pairs
            for t in range(self.num_time_steps)
        ]
        total_programs = 2 * len(targets)
        program_number = 0
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)
        print(f"\nAlpha-bound optimization programs: {total_programs}", flush=True)

        workers = self._resolve_parallel_workers(parallel_workers, total_programs)
        if workers > 1:
            print(f"Running alpha-bound programs with {workers} worker processes", flush=True)
            parallel_tasks = [
                (index, bound_name, solver_name, time_limit, tee, solver_options)
                for index in targets
                for bound_name in ("lower", "upper")
            ]
            result_by_task: dict[tuple[tuple[int, int, int], str], dict[str, Any]] = {}
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_alpha_computer,
                initargs=(self._parallel_optimizer_state_for_alpha(),),
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
                    _tc = result["termination_condition"]
                    if completed % 50 == 0 or completed == total_programs or "infeasible" in str(_tc).lower():
                        action = "minimize" if bound_name == "lower" else "maximize"
                        print(
                            f"[Alpha done {completed}/{total_programs}] {action} "
                            f"alpha{index} -> {result['value']} ({_tc})",
                            flush=True,
                        )

            for index in targets:
                lower_upper: dict[str, Optional[float]] = {"lower": None, "upper": None}
                for bound_name in ("lower", "upper"):
                    result = result_by_task[(index, bound_name)]
                    lower_upper[bound_name] = result["value"]
                    optimization_results[f"{self._json_key(index)}:{bound_name}"] = {
                        "value": result["value"],
                        "termination_condition": result["termination_condition"],
                    }
                if lower_upper["lower"] is None or lower_upper["upper"] is None:
                    raise RuntimeError(f"Could not compute alpha bounds for index {index}")
                alpha_bounds[index] = {
                    "lower": float(lower_upper["lower"]),
                    "upper": float(lower_upper["upper"]),
                }

            self.alpha_bounds = alpha_bounds
            self.alpha_bound_optimization_results = optimization_results
            return {
                "alpha_bounds": self._jsonify_indexed_dict(alpha_bounds),
                "optimization_results": optimization_results,
                "num_optimization_programs": total_programs,
            }

        for index in targets:
            lower_upper: dict[str, Optional[float]] = {"lower": None, "upper": None}
            for bound_name, sense in (("lower", minimize), ("upper", maximize)):
                program_number += 1
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
                _tc = str(results.solver.termination_condition)
                if program_number % 50 == 0 or program_number == total_programs or "infeasible" in _tc.lower():
                    action = "minimize" if bound_name == "lower" else "maximize"
                    print(f"[Alpha {program_number}/{total_programs}] {action} alpha{index} -> {bound_value} ({_tc})", flush=True)
                lower_upper[bound_name] = bound_value
                optimization_results[f"{self._json_key(index)}:{bound_name}"] = {
                    "value": bound_value,
                    "termination_condition": str(results.solver.termination_condition),
                }

            if lower_upper["lower"] is None or lower_upper["upper"] is None:
                raise RuntimeError(f"Could not compute alpha bounds for index {index}")
            alpha_bounds[index] = {
                "lower": float(lower_upper["lower"]),
                "upper": float(lower_upper["upper"]),
            }

        self.alpha_bounds = alpha_bounds
        self.alpha_bound_optimization_results = optimization_results
        return {
            "alpha_bounds": self._jsonify_indexed_dict(alpha_bounds),
            "optimization_results": optimization_results,
            "num_optimization_programs": total_programs,
        }

    def run_alpha_bounds(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or DEFAULT_TIGHTENING_OUTPUT_PATHS["alpha_bounds"]

        if getattr(self.poa, "nn_policy_generator_ids", []) and not self.poa.nn_relu_bounds:
            raise ValueError(
                "ReLU bounds must be computed or loaded before alpha-bound tightening."
            )

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
                "description": (
                    "Exact alpha bounds from ambiguity-set optimization with embedded "
                    "ReLU policy constraints."
                ),
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "ambiguity_set": self._ambiguity_metadata(),
                "nn_policy_generators": list(self.poa.nn_policy_generator_names),
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
