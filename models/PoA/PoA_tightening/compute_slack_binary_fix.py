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


_PARALLEL_SLACK_COMPUTER: Optional["SlackBinaryFixComputer"] = None


def _initialize_parallel_slack_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_SLACK_COMPUTER
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
    for attr_name in ("alpha_bounds", "fixed_binaries", "primal_big_m"):
        if attr_name in state:
            setattr(poa, attr_name, state[attr_name])
    if state.get("nn_relu_bounds_report"):
        poa._set_nn_relu_bounds_from_report(state["nn_relu_bounds_report"])

    computer = SlackBinaryFixComputer.__new__(SlackBinaryFixComputer)
    computer.poa = poa
    computer.tightening_data = {}
    computer.stage_reports = {}
    _PARALLEL_SLACK_COMPUTER = computer


def _get_parallel_slack_computer() -> "SlackBinaryFixComputer":
    if _PARALLEL_SLACK_COMPUTER is None:
        raise RuntimeError("Parallel slack OBBT worker was not initialized")
    return _PARALLEL_SLACK_COMPUTER


def _classify_slack_result(
    termination: Any,
    incumbent_slack: Optional[float],
    epsilon: float,
    zero_slack_tol: float,
) -> tuple[str, Optional[float], bool]:
    is_optimal = termination in {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }
    if incumbent_slack is not None and incumbent_slack <= zero_slack_tol:
        minimum_slack = 0.0 if incumbent_slack < 0.0 else float(incumbent_slack)
        return "zero_slack_feasible", minimum_slack, False
    if is_optimal and incumbent_slack is not None:
        minimum_slack = float(incumbent_slack)
        return "positive_slack_optimal", minimum_slack, minimum_slack >= float(epsilon)
    return "undetermined", None, False


def _solve_parallel_slack_obbt(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        side,
        constraint_type,
        index,
        epsilon,
        zero_slack_tol,
        early_stop_enabled,
        relax_complementarity,
        solver_name,
        time_limit,
        tee,
        solver_options,
    ) = task
    computer = _get_parallel_slack_computer()
    m = computer._build_side_kkt_model_for_slack_fixing(
        side=side,
        alpha_bounds=computer.alpha_bounds,
        include_complementarity=not relax_complementarity,
        fixed_binaries=computer.fixed_binaries,
    )
    slack_expr = computer._slack_expression(m, side, constraint_type, index)
    m.target_slack = Var(domain=NonNegativeReals)
    m.target_slack_definition = Constraint(expr=m.target_slack == slack_expr)
    m.tightening_objective = Objective(expr=m.target_slack, sense=minimize)

    task_solver_options = dict(solver_options or {})
    if early_stop_enabled:
        task_solver_options["BestObjStop"] = zero_slack_tol

    _solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=task_solver_options or None,
    )
    termination = results.solver.termination_condition
    incumbent_slack = computer._safe_value(m.target_slack)
    result_classification, minimum_slack, is_inactive = _classify_slack_result(
        termination=termination,
        incumbent_slack=incumbent_slack,
        epsilon=epsilon,
        zero_slack_tol=zero_slack_tol,
    )
    return {
        "side": side,
        "constraint_type": constraint_type,
        "index": index,
        "minimum_slack": minimum_slack,
        "incumbent_slack_objective": incumbent_slack,
        "robustly_inactive": bool(is_inactive),
        "early_stop_enabled": early_stop_enabled,
        "slack_stop_tolerance": zero_slack_tol,
        "termination_condition": str(termination),
        "result_classification": result_classification,
    }


class SlackBinaryFixComputer(PoATighteningMain):
    def _ensure_primal_big_m_for_tightening(self) -> None:
        if getattr(self.poa, "primal_big_m", {}) or {}:
            return
        self.poa.primal_big_m = compute_primal_big_m_bounds(self.poa)
        self.poa._loaded_bounds_prepared = False

    def _parallel_optimizer_state_for_slack(
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

    def _build_side_kkt_model_for_slack_fixing(
        self,
        side: str,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        include_complementarity: bool = True,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> ConcreteModel:
        """
        Build one KKT side for slack-minimization binary fixing.
        """
        self._ensure_primal_big_m_for_tightening()
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()

        if side == "eq":
            if alpha_bounds is None:
                raise ValueError("Equilibrium KKT slack fixing requires alpha_bounds")
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

    def _dispatch_var(self, m: ConcreteModel, side: str) -> Any:
        return m.P_eq if side == "eq" else m.P_opt

    def _slack_expression(
        self,
        m: ConcreteModel,
        side: str,
        constraint_type: str,
        index: tuple[int, ...],
    ) -> Any:
        """
        Return the nonnegative slack for one lower-level inequality.
        """
        P = self._dispatch_var(m, side)
        if constraint_type == "upper":
            i, b, t = index
            return m.P_max_block[i, b, t] - P[i, b, t]
        if constraint_type == "lower":
            i, b, t = index
            return P[i, b, t]
        if constraint_type == "ramp_up":
            i, t = index
            current = sum(P[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            previous = (
                self.p_init[int(i)]
                if int(t) == 0
                else sum(P[i, b, int(t) - 1] for b in self.local_blocks_by_generator[int(i)])
            )
            return self.ramp_vector_up[int(i)] - (current - previous)
        if constraint_type == "ramp_down":
            i, t = index
            current = sum(P[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            previous = (
                self.p_init[int(i)]
                if int(t) == 0
                else sum(P[i, b, int(t) - 1] for b in self.local_blocks_by_generator[int(i)])
            )
            return self.ramp_vector_down[int(i)] - (previous - current)
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

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

    def run_slack_based_obbt(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        epsilon: float = 1e-6,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        relax_complementarity: bool = False,
        stop_at_zero_slack: bool = True,
        slack_stop_tol: Optional[float] = None,
        parallel_workers: Optional[int] = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Minimize every lower-level inequality slack over each KKT side.
        """
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before slack OBBT")

        zero_slack_tol = float(slack_stop_tol if slack_stop_tol is not None else epsilon)
        early_stop_enabled = bool(stop_at_zero_slack and solver_name == "gurobi")

        slack_bounds: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
        fixed_binaries: dict[str, dict[str, Any]] = {}

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

        total_programs = len(tasks)
        mode = "relaxed LP" if relax_complementarity else "KKT MILP"
        print(f"\nSlack OBBT optimization programs: {total_programs} ({mode})", flush=True)
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)

        workers = self._resolve_parallel_workers(parallel_workers, total_programs)
        if workers > 1:
            print(
                f"Running slack OBBT programs with {workers} worker processes",
                flush=True,
            )
            parallel_tasks = [
                (
                    side,
                    constraint_type,
                    index,
                    float(epsilon),
                    zero_slack_tol,
                    early_stop_enabled,
                    relax_complementarity,
                    solver_name,
                    time_limit,
                    tee,
                    solver_options,
                )
                for side, constraint_type, index in tasks
            ]
            result_by_task: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_slack_computer,
                initargs=(
                    self._parallel_optimizer_state_for_slack(
                        alpha_bounds=alpha_bounds,
                        fixed_binaries={},
                    ),
                ),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_slack_obbt, task): task
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
                            f"[Slack OBBT done {completed}/{total_programs}] "
                            f"side={result['side']}, constraint={result['constraint_type']}, "
                            f"index={result['index']} -> {result['minimum_slack']} ({_tc})",
                            flush=True,
                        )

            for side, constraint_type, index in tasks:
                record_key = (side, constraint_type, index)
                result = result_by_task[record_key]
                slack_bounds[record_key] = {
                    "minimum_slack": result["minimum_slack"],
                    "incumbent_slack_objective": result["incumbent_slack_objective"],
                    "robustly_inactive": bool(result["robustly_inactive"]),
                    "early_stop_enabled": result["early_stop_enabled"],
                    "slack_stop_tolerance": result["slack_stop_tolerance"],
                    "termination_condition": result["termination_condition"],
                    "result_classification": result["result_classification"],
                }
                if result["robustly_inactive"]:
                    var_name = self._binary_name(side, constraint_type)
                    minimum_slack = result["minimum_slack"]
                    fixed_binaries.setdefault(var_name, {})[self._json_key(index)] = {
                        "fixed_value": 0,
                        "minimum_slack": float(minimum_slack),
                        "side": side,
                        "constraint_type": constraint_type,
                        "result_classification": result["result_classification"],
                    }

            self.slack_bounds = slack_bounds
            self.fixed_binaries = fixed_binaries
            return {
                "epsilon": float(epsilon),
                "stop_at_zero_slack": bool(stop_at_zero_slack),
                "early_stop_enabled": early_stop_enabled,
                "slack_stop_tolerance": zero_slack_tol,
                "slack_bounds": {
                    f"{side}:{constraint_type}:{self._json_key(index)}": value
                    for (side, constraint_type, index), value in slack_bounds.items()
                },
                "fixed_binaries": fixed_binaries,
                "num_fixed_binaries": int(
                    sum(len(entries) for entries in fixed_binaries.values())
                ),
            }

        for program_number, (side, constraint_type, index) in enumerate(tasks, start=1):
            m = self._build_side_kkt_model_for_slack_fixing(
                side=side,
                alpha_bounds=alpha_bounds,
                include_complementarity=not relax_complementarity,
                fixed_binaries=fixed_binaries,
            )
            slack_expr = self._slack_expression(m, side, constraint_type, index)
            m.target_slack = Var(domain=NonNegativeReals)
            m.target_slack_definition = Constraint(expr=m.target_slack == slack_expr)
            m.tightening_objective = Objective(expr=m.target_slack, sense=minimize)
            task_solver_options = self._solver_options_with_threads(
                solver_name,
                solver_threads,
                {"BestObjStop": zero_slack_tol} if early_stop_enabled else None,
            )
            _solved, results = self._solve_tightening_model(
                m,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                solver_options=task_solver_options,
            )
            termination = results.solver.termination_condition
            incumbent_slack = self._safe_value(m.target_slack)
            result_classification, minimum_slack, is_inactive = _classify_slack_result(
                termination=termination,
                incumbent_slack=incumbent_slack,
                epsilon=float(epsilon),
                zero_slack_tol=zero_slack_tol,
            )

            record_key = (side, constraint_type, index)
            slack_bounds[record_key] = {
                "minimum_slack": minimum_slack,
                "incumbent_slack_objective": incumbent_slack,
                "robustly_inactive": bool(is_inactive),
                "early_stop_enabled": early_stop_enabled,
                "slack_stop_tolerance": zero_slack_tol,
                "termination_condition": str(termination),
                "result_classification": result_classification,
            }
            _tc = str(termination)
            if program_number % 50 == 0 or program_number == total_programs or "infeasible" in _tc.lower():
                print(f"[Slack OBBT {program_number}/{total_programs}] side={side}, constraint={constraint_type}, index={index} -> {minimum_slack} ({_tc})", flush=True)
            if is_inactive:
                var_name = self._binary_name(side, constraint_type)
                fixed_binaries.setdefault(var_name, {})[self._json_key(index)] = {
                    "fixed_value": 0,
                    "minimum_slack": float(minimum_slack),
                    "side": side,
                    "constraint_type": constraint_type,
                    "result_classification": result_classification,
                }

        self.slack_bounds = slack_bounds
        self.fixed_binaries = fixed_binaries
        return {
            "epsilon": float(epsilon),
            "stop_at_zero_slack": bool(stop_at_zero_slack),
            "early_stop_enabled": early_stop_enabled,
            "slack_stop_tolerance": zero_slack_tol,
            "slack_bounds": {
                f"{side}:{constraint_type}:{self._json_key(index)}": value
                for (side, constraint_type, index), value in slack_bounds.items()
            },
            "fixed_binaries": fixed_binaries,
            "num_fixed_binaries": int(sum(len(entries) for entries in fixed_binaries.values())),
        }

    def run_slack_binary_fix(
        self,
        output_path: str | Path | None = None,
        epsilon: float = 1e-6,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        relax_complementarity: bool = False,
        stop_at_zero_slack: bool = True,
        slack_stop_tol: Optional[float] = None,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or DEFAULT_TIGHTENING_OUTPUT_PATHS["slack_binary_fix"]

        if not getattr(self.poa, "alpha_bounds", None):
            raise ValueError("Alpha bounds must be computed or loaded before slack binary fixing.")

        start = time.perf_counter()
        slack_report = self.run_slack_based_obbt(
            alpha_bounds=self.poa.alpha_bounds,
            epsilon=epsilon,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            relax_complementarity=relax_complementarity,
            stop_at_zero_slack=stop_at_zero_slack,
            slack_stop_tol=slack_stop_tol,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        elapsed = time.perf_counter() - start

        report = {
            "metadata": {
                "description": (
                    "Slack minimization certificates for KKT complementarity "
                    "binary fixing."
                ),
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "ambiguity_set": self._ambiguity_metadata(),
                "runtime_seconds": elapsed,
            },
            "epsilon": slack_report["epsilon"],
            "stop_at_zero_slack": slack_report["stop_at_zero_slack"],
            "slack_stop_tolerance": slack_report["slack_stop_tolerance"],
            "early_stop_enabled": slack_report.get("early_stop_enabled"),
            "slack_bounds": slack_report["slack_bounds"],
            "fixed_binaries": slack_report["fixed_binaries"],
            "num_fixed_binaries": slack_report["num_fixed_binaries"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
            "alpha_optimization_results": self.tightening_data.get(
                "alpha_optimization_results",
                {},
            ),
            "tight_big_m": self.tightening_data.get("tight_big_m", {}),
        }
        self.tightening_data["slack_bounds"] = report["slack_bounds"]
        self.tightening_data["fixed_binaries"] = report["fixed_binaries"]
        self.poa.slack_bounds = getattr(self.poa, "slack_bounds", report["slack_bounds"])
        self.poa.fixed_binaries = report["fixed_binaries"]
        return self._save_stage_report("slack_binary_fix", report, output_path)
