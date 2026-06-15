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
        num_time_steps=state["num_time_steps"],
        ambiguity_set_config=state["ambiguity_set_config"],
        nn_model_dir=state["nn_model_dir"],
        nn_normalization_stats_path=state["nn_normalization_stats_path"],
        nn_policy_generators=state["nn_policy_generators"],
        reference_case=state["reference_case"],
        objective_mode=state.get("objective_mode", "difference"),
        mccormick_bounds=state.get("mccormick_bounds"),
        use_default_bounds=bool(state.get("use_default_bounds", False)),
        alpha_ordering_epsilon=state.get(
            "alpha_ordering_epsilon",
            PoAOptimization.DEFAULT_ALPHA_ORDERING_EPSILON,
        ),
    )
    if "p_init" in state:
        poa.p_init = state["p_init"]
    for attr_name in (
        "alpha_bounds",
        "fixed_binaries",
        "primal_big_m",
        "tight_big_m",
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
    dual_value = computer._safe_value(dual_expr) if solved else None
    return {
        "side": side,
        "constraint_type": constraint_type,
        "index": index,
        "dual_name": dual_name,
        "tight_big_m": dual_value,
        "termination_condition": str(results.solver.termination_condition),
        **computer._dual_big_m_diagnostics(
            m=m,
            side=side,
            constraint_type=constraint_type,
            dual_value=dual_value,
            solved=solved,
        ),
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
            "alpha_ordering_epsilon": self.poa.alpha_ordering_epsilon,
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
        Build one KKT side for componentwise dual Big-M bounds.
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

    def _dual_big_m_diagnostics(
        self,
        m: ConcreteModel,
        side: str,
        constraint_type: str,
        dual_value: Optional[float],
        solved: bool,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        tolerance = max(1e-5, 1e-7 * float(self.default_dual_big_m))
        if not solved or dual_value is None:
            reasons.append("auxiliary dual Big-M problem did not return a usable solution")
        elif dual_value >= float(self.default_dual_big_m) - tolerance:
            reasons.append("dual value reached the default dual Big-M cap")

        lambda_var = getattr(m, self._lambda_name(side), None)
        if lambda_var is not None:
            lambda_tolerance = max(1e-5, 1e-7 * max(
                abs(float(self.default_lambda_lower)),
                abs(float(self.default_lambda_upper)),
                1.0,
            ))
            for t in m.time_steps:
                lambda_value = value(lambda_var[t], exception=False)
                if lambda_value is None:
                    continue
                if lambda_value <= float(self.default_lambda_lower) + lambda_tolerance:
                    reasons.append(
                        f"{self._lambda_name(side)}[{int(t)}] reached the default lower lambda bound"
                    )
                    break
                if lambda_value >= float(self.default_lambda_upper) - lambda_tolerance:
                    reasons.append(
                        f"{self._lambda_name(side)}[{int(t)}] reached the default upper lambda bound"
                    )
                    break

        cap_limited = bool(reasons)
        return {
            "certified": not cap_limited,
            "cap_limited": cap_limited,
            "cap_limit_reason": "; ".join(reasons) if reasons else None,
        }

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
        Maximize each componentwise dual variable after slack fixing.
        """
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before Big-M tightening")
        fixed_binaries = fixed_binaries or getattr(self.poa, "fixed_binaries", {})

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
                tight_big_m.setdefault(dual_name, {})[key] = {
                    "tight_big_m": 0.0,
                    "fixed_by_slack": True,
                    "termination_condition": "fixed_binary_zero",
                    "certified": True,
                    "cap_limited": False,
                    "cap_limit_reason": None,
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
                    _tc = result["termination_condition"]
                    if completed % 50 == 0 or completed == total_programs or "infeasible" in str(_tc).lower():
                        print(
                            f"[Dual Big-M done {completed}/{total_programs}] "
                            f"maximize {result['dual_name']}{result['index']} -> "
                            f"{result['tight_big_m']} ({_tc})",
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
                    "certified": result["certified"],
                    "cap_limited": result["cap_limited"],
                    "cap_limit_reason": result["cap_limit_reason"],
                }
        else:
            for side, constraint_type, index in pending_dual_tasks:
                dual_name = self._dual_name(side, constraint_type)

                program_number += 1
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
                diagnostics = self._dual_big_m_diagnostics(
                    m=m,
                    side=side,
                    constraint_type=constraint_type,
                    dual_value=dual_bound,
                    solved=solved,
                )
                _tc = str(results.solver.termination_condition)
                tight_big_m.setdefault(dual_name, {})[self._json_key(index)] = {
                    "tight_big_m": dual_bound,
                    "fixed_by_slack": False,
                    "termination_condition": _tc,
                    **diagnostics,
                }
                if program_number % 50 == 0 or program_number == total_programs or "infeasible" in _tc.lower():
                    print(f"[Dual Big-M {program_number}/{total_programs}] maximize {dual_name}{index} -> {dual_bound} ({_tc})", flush=True)

        self.tight_big_m = tight_big_m
        return {
            "tight_big_m": tight_big_m,
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
                    "Componentwise dual Big-M values for the final PoA KKT model."
                ),
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "ambiguity_set": self._ambiguity_metadata(),
                "runtime_seconds": elapsed,
            },
            "tight_big_m": dual_report["tight_big_m"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
            "alpha_optimization_results": self.tightening_data.get(
                "alpha_optimization_results",
                {},
            ),
            "slack_bounds": self.tightening_data.get("slack_bounds", {}),
            "fixed_binaries": self.tightening_data.get("fixed_binaries", {}),
        }
        self.tightening_data["tight_big_m"] = report["tight_big_m"]
        self.poa.tight_big_m = report["tight_big_m"]
        return self._save_stage_report("dual_big_m", report, output_path)
