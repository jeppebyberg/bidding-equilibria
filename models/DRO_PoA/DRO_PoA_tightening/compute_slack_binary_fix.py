from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Objective,
    SolverFactory,
    TerminationCondition,
    Var,
    minimize,
    value,
)

from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import compute_primal_big_m_bounds
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain


_PARALLEL_SLACK_COMPUTER: Optional["DROSlackBinaryFixComputer"] = None


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
        return "zero_slack_feasible", max(0.0, float(incumbent_slack)), False
    if is_optimal and incumbent_slack is not None:
        minimum_slack = float(incumbent_slack)
        return "positive_slack_optimal", minimum_slack, minimum_slack >= float(epsilon)
    return "undetermined", None, False


def _initialize_parallel_slack_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_SLACK_COMPUTER
    _PARALLEL_SLACK_COMPUTER = DROSlackBinaryFixComputer._from_parallel_stage_state(state)


def _get_parallel_slack_computer() -> "DROSlackBinaryFixComputer":
    if _PARALLEL_SLACK_COMPUTER is None:
        raise RuntimeError("Parallel DRO slack computer was not initialized")
    return _PARALLEL_SLACK_COMPUTER


def _solve_parallel_slack_obbt(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        side,
        constraint_type,
        index,
        alpha_bounds,
        epsilon,
        zero_slack_tol,
        relax_complementarity,
        solver_name,
        time_limit,
        tee,
        solver_options,
    ) = task
    computer = _get_parallel_slack_computer()
    m = computer._build_side_kkt_model_for_slack_fixing(
        side=side,
        alpha_bounds=alpha_bounds,
        include_complementarity=not relax_complementarity,
        fixed_binaries=None,
    )
    slack_expr = computer._slack_expression(m, side, constraint_type, tuple(index))
    m.target_slack = Var(domain=NonNegativeReals)
    m.target_slack_definition = Constraint(expr=m.target_slack == slack_expr)
    m.tightening_objective = Objective(expr=m.target_slack, sense=minimize)
    _solved, results = computer._solve_tightening_model(
        m,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options or None,
    )
    termination = results.solver.termination_condition
    incumbent_slack = computer._safe_value(m.target_slack)
    result_classification, minimum_slack, is_inactive = _classify_slack_result(
        termination=termination,
        incumbent_slack=incumbent_slack,
        epsilon=float(epsilon),
        zero_slack_tol=float(zero_slack_tol),
    )
    return {
        "side": side,
        "constraint_type": constraint_type,
        "index": tuple(int(part) for part in index),
        "minimum_slack": minimum_slack,
        "incumbent_slack_objective": incumbent_slack,
        "robustly_inactive": bool(is_inactive),
        "termination_condition": str(termination),
        "result_classification": result_classification,
    }


class DROSlackBinaryFixComputer(DROPoATighteningMain):
    def _ensure_primal_big_m_for_tightening(self) -> None:
        if getattr(self.poa, "primal_big_m", {}) or {}:
            if hasattr(self.poa, "_prepare_loaded_bounds"):
                self.poa._prepare_loaded_bounds()
            return
        self.poa.primal_big_m = compute_primal_big_m_bounds(self.poa)
        if hasattr(self.poa, "_prepare_loaded_bounds"):
            self.poa._prepare_loaded_bounds()

    def _build_side_kkt_model_for_slack_fixing(
        self,
        side: str,
        alpha_bounds: Optional[dict[tuple[int, ...], dict[str, float]]] = None,
        include_complementarity: bool = True,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> ConcreteModel:
        self._ensure_primal_big_m_for_tightening()
        self.model = ConcreteModel()
        from pyomo.environ import Set

        self.model.scenarios = Set(initialize=[0])  # single regime scenario
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

        # Use large epsilon so binaries/bounds are regime-wide (full support set),
        # not anchored to the Wasserstein neighbourhood of empirical scenario k=0.
        _saved_epsilon, self.epsilon = self.epsilon, 1e15
        if side == "eq":
            if alpha_bounds is None:
                raise ValueError("Equilibrium KKT slack fixing requires alpha_bounds")
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
            self.epsilon = _saved_epsilon
            raise ValueError(f"Unknown KKT tightening side: {side}")
        self.epsilon = _saved_epsilon
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
    def _dispatch_var(m: ConcreteModel, side: str) -> Any:
        return m.P_eq if side == "eq" else m.P_opt

    def _slack_expression(
        self,
        m: ConcreteModel,
        side: str,
        constraint_type: str,
        index: tuple[int, ...],
    ) -> Any:
        P = self._dispatch_var(m, side)
        if constraint_type == "upper":
            k, i, b, t = index
            return m.P_max_block[k, i, b, t] - P[k, i, b, t]
        if constraint_type == "lower":
            k, i, b, t = index
            return P[k, i, b, t]
        if constraint_type == "ramp_up":
            k, i, t = index
            current = sum(P[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            previous = (
                self.p_init[int(k)][int(i)]
                if int(t) == 0
                else sum(P[k, i, b, int(t) - 1] for b in self.local_blocks_by_generator[int(i)])
            )
            return self.ramp_vector_up[int(i)] - (current - previous)
        if constraint_type == "ramp_down":
            k, i, t = index
            current = sum(P[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            previous = (
                self.p_init[int(k)][int(i)]
                if int(t) == 0
                else sum(P[k, i, b, int(t) - 1] for b in self.local_blocks_by_generator[int(i)])
            )
            return self.ramp_vector_down[int(i)] - (previous - current)
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

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

    def run_slack_based_obbt(
        self,
        alpha_bounds: Optional[dict[tuple[int, ...], dict[str, float]]] = None,
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
        alpha_bounds = alpha_bounds or getattr(self.poa, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Alpha bounds must be computed before slack OBBT")

        zero_slack_tol = float(slack_stop_tol if slack_stop_tol is not None else epsilon)
        solver_options: dict[str, Any] = {}
        if solver_threads is not None and solver_name == "gurobi":
            solver_options["Threads"] = int(solver_threads)
        if stop_at_zero_slack and solver_name == "gurobi":
            solver_options["BestObjStop"] = zero_slack_tol

        tasks: list[tuple[str, str, tuple[int, ...]]] = []
        for side in ("eq", "opt"):
            for i, b in self.generator_block_pairs:
                for t in range(self.num_time_steps):
                    tasks.append((side, "upper", (0, int(i), int(b), int(t))))
                    tasks.append((side, "lower", (0, int(i), int(b), int(t))))
            for i in range(self.num_physical_generators):
                for t in range(self.num_time_steps):
                    tasks.append((side, "ramp_up", (0, int(i), int(t))))
                    tasks.append((side, "ramp_down", (0, int(i), int(t))))

        slack_bounds: dict[str, Any] = {}
        scenario_fixed_binaries: dict[str, dict[str, Any]] = {}
        workers = self._resolve_parallel_workers(parallel_workers, len(tasks))
        if workers > 1:
            print(
                f"Running DRO slack OBBT programs with {workers} worker processes",
                flush=True,
            )
            parallel_tasks = [
                (
                    side,
                    constraint_type,
                    index,
                    alpha_bounds,
                    float(epsilon),
                    zero_slack_tol,
                    relax_complementarity,
                    solver_name,
                    time_limit,
                    tee,
                    solver_options,
                )
                for side, constraint_type, index in tasks
            ]
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_parallel_slack_computer,
                initargs=(self._parallel_stage_state(),),
            ) as executor:
                future_to_task = {
                    executor.submit(_solve_parallel_slack_obbt, task): task
                    for task in parallel_tasks
                }
                result_iterable = []
                for completed, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    print(
                        f"[DRO Slack OBBT {completed}/{len(tasks)}] "
                        f"minimize side={result['side']}, "
                        f"constraint={result['constraint_type']}, "
                        f"index={result['index']} -> {result['minimum_slack']} "
                        f"({result['termination_condition']})",
                        flush=True,
                    )
                    result_iterable.append(result)
        else:
            global _PARALLEL_SLACK_COMPUTER
            _PARALLEL_SLACK_COMPUTER = self
            result_iterable = []
            for program_number, (side, constraint_type, index) in enumerate(tasks, start=1):
                print(
                    f"[DRO Slack OBBT {program_number}/{len(tasks)}] "
                    f"minimize side={side}, constraint={constraint_type}, index={index}",
                    flush=True,
                )
                result_iterable.append(
                    _solve_parallel_slack_obbt(
                        (
                            side,
                            constraint_type,
                            index,
                            alpha_bounds,
                            float(epsilon),
                            zero_slack_tol,
                            relax_complementarity,
                            solver_name,
                            time_limit,
                            tee,
                            solver_options,
                        )
                    )
                )

        regime_fixed_binaries: dict[str, dict[str, Any]] = {}
        for result in result_iterable:
            side = str(result["side"])
            constraint_type = str(result["constraint_type"])
            full_index = tuple(result["index"])
            regime_index = full_index[1:]  # strip k=0 → (i,b,t) or (i,t)
            minimum_slack = result["minimum_slack"]
            is_inactive = bool(result["robustly_inactive"])
            record_key = f"{side}:{constraint_type}:{self._json_key(regime_index)}"
            slack_bounds[record_key] = {
                "minimum_slack": result["minimum_slack"],
                "incumbent_slack_objective": result["incumbent_slack_objective"],
                "robustly_inactive": is_inactive,
                "termination_condition": result["termination_condition"],
                "result_classification": result["result_classification"],
            }
            if is_inactive:
                var_name = self._binary_name(side, constraint_type)
                regime_fixed_binaries.setdefault(var_name, {})[self._json_key(regime_index)] = {
                    "fixed_value": 0,
                    "minimum_slack": float(minimum_slack),
                    "side": side,
                    "constraint_type": constraint_type,
                    "result_classification": result["result_classification"],
                }

        return {
            "epsilon": float(epsilon),
            "stop_at_zero_slack": bool(stop_at_zero_slack),
            "slack_stop_tolerance": zero_slack_tol,
            "slack_bounds": slack_bounds,
            "scenario_fixed_binaries": regime_fixed_binaries,
            "regime_fixed_binaries": regime_fixed_binaries,
            "fixed_binaries": regime_fixed_binaries,
            "num_fixed_binaries": int(
                sum(len(entries) for entries in regime_fixed_binaries.values())
            ),
            "num_regime_fixed_binaries": int(
                sum(len(entries) for entries in regime_fixed_binaries.values())
            ),
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
        output_path = output_path or self._resolve_output_paths()["slack_binary_fix"]
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
                **self._metadata(),
                "description": (
                    "Regime-wide slack minimization certificates optimised over the "
                    "full support set. Keys are i,b,t or i,t (no scenario index)."
                ),
                "tightening_scope": "regime_wide",
                "runtime_seconds": elapsed,
            },
            **slack_report,
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
        }
        self.tightening_data["slack_bounds"] = report["slack_bounds"]
        self.tightening_data["scenario_fixed_binaries"] = report["scenario_fixed_binaries"]
        self.tightening_data["regime_fixed_binaries"] = report["regime_fixed_binaries"]
        self.tightening_data["fixed_binaries"] = report["fixed_binaries"]
        self.poa.slack_bounds = report["slack_bounds"]
        self.poa.fixed_binaries = report["fixed_binaries"]
        return self._save_stage_report("slack_binary_fix", report, output_path)
