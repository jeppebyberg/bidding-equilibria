from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
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

from models.DRO_PoA.DRO_PoA_tightening.tightening_main import (
    DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS,
    DROPoATighteningMain,
)


def _bound_sense(bound_name: str) -> Any:
    if bound_name == "lower":
        return minimize
    if bound_name == "upper":
        return maximize
    raise ValueError(f"Unknown bound name: {bound_name}")


class DROOptimalCostBoundsComputer(DROPoATighteningMain):
    """Compute valid denominator bounds from the DRO optimal-dispatch KKT block."""

    def _build_index_sets(self) -> None:
        dro = self.poa
        m = self.model
        m.scenarios = Set(initialize=range(dro.num_empirical_scenarios))
        m.time_steps = Set(initialize=range(dro.num_time_steps))
        m.time_steps_minus_1 = Set(initialize=range(1, dro.num_time_steps))
        m.time_steps_plus_1 = Set(initialize=range(dro.num_time_steps + 1))
        m.physical_generators = Set(initialize=range(dro.num_physical_generators))
        m.generator_blocks = Set(dimen=2, initialize=dro.generator_block_pairs)
        m.wind_physical_generators = Set(initialize=dro.wind_physical_generator_ids)
        m.conventional_physical_generators = Set(
            initialize=dro.conventional_physical_generator_ids
        )
        m.wind_blocks = Set(dimen=2, initialize=dro.wind_block_pairs)
        m.conventional_blocks = Set(dimen=2, initialize=dro.conventional_block_pairs)

    def _optimal_cost_expression(self, m: ConcreteModel, scenario_idx: int) -> Any:
        dro = self.poa
        k = int(scenario_idx)
        return sum(
            dro.block_cost_vector[dro.local_to_global_block[(int(i), int(b))]]
            * m.P_opt[k, i, b, t]
            for (i, b) in m.generator_blocks
            for t in m.time_steps
        )

    def _build_optimal_dispatch_kkt_bound_model(self) -> ConcreteModel:
        """
        Build only the DRO support set and true-cost optimal KKT block.

        This excludes equilibrium dispatch, policy constraints, transport
        variables, Wasserstein objective terms, PoA links, and ratio variables.
        """
        dro = self.poa
        if getattr(dro, "tightening_report", None) or getattr(dro, "primal_big_m", None):
            dro._prepare_loaded_bounds()

        self.model = ConcreteModel()
        self._build_index_sets()
        m = self.model

        dro._build_regime_variables()
        m.D = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_block = Var(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            domain=NonNegativeReals,
        )
        dro._build_optimal_variables()
        dro._build_complementarity_optimal_variables()
        dro._build_support_set()
        dro._build_lower_level_optimal_constraints()
        dro._build_KKT_stationarity_optimal_constraints()
        dro._build_KKT_complementarity_optimal_constraints()

        m.C_opt = Var(m.scenarios, domain=Reals)

        def cost_opt_rule(m, k):
            return m.C_opt[k] == self._optimal_cost_expression(m, int(k))

        m.cost_definition_opt = Constraint(m.scenarios, rule=cost_opt_rule)
        return m

    def _solve_bound(
        self,
        bound_name: str,
        scenario_idx: int,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_threads: Optional[int],
    ) -> dict[str, Any]:
        m = self._build_optimal_dispatch_kkt_bound_model()
        k = int(scenario_idx)
        m.tightening_objective = Objective(
            expr=m.C_opt[k],
            sense=_bound_sense(bound_name),
        )

        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if solver_threads is not None and solver_name == "gurobi":
            solver.options["Threads"] = int(solver_threads)

        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        solved = termination in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
        bound_value = value(m.C_opt[k], exception=False) if solved else None
        return {
            "bound_name": str(bound_name),
            "scenario_idx": int(k),
            "value": float(bound_value) if bound_value is not None else None,
            "solver_status": str(results.solver.status),
            "termination_condition": str(termination),
            "num_variables": int(m.nvariables()),
            "num_constraints": int(m.nconstraints()),
            "active_constraints": int(
                sum(1 for _ in m.component_data_objects(Constraint, active=True))
            ),
        }

    @staticmethod
    def _safe_bounds(raw_lower: float, raw_upper: float) -> dict[str, float]:
        if raw_lower <= 0.0:
            raise ValueError("Computed C_opt lower bound must be strictly positive")
        if raw_upper < raw_lower:
            raise ValueError("Computed C_opt upper bound is below lower bound")
        safe_lower = max(1e-5, raw_lower - abs(raw_lower) * 1e-6 - 1e-6)
        safe_upper = raw_upper + abs(raw_upper) * 1e-6 + 1e-6
        return {
            "lower": float(safe_lower),
            "upper": float(safe_upper),
            "raw_lower": float(raw_lower),
            "raw_upper": float(raw_upper),
        }

    def compute_optimal_cost_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        scenario_bounds: dict[str, Any] = {}
        optimization_results: dict[str, Any] = {}
        raw_lowers: list[float] = []
        raw_uppers: list[float] = []

        for k in range(self.poa.num_empirical_scenarios):
            lower_result = self._solve_bound(
                "lower",
                scenario_idx=k,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                solver_threads=solver_threads,
            )
            upper_result = self._solve_bound(
                "upper",
                scenario_idx=k,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                solver_threads=solver_threads,
            )
            if lower_result["value"] is None or upper_result["value"] is None:
                raise RuntimeError(f"Could not compute both C_opt bounds for scenario {k}")

            raw_lower = float(lower_result["value"])
            raw_upper = float(upper_result["value"])
            scenario_bounds[str(k)] = self._safe_bounds(raw_lower, raw_upper)
            optimization_results[str(k)] = {
                "lower": lower_result,
                "upper": upper_result,
            }
            raw_lowers.append(raw_lower)
            raw_uppers.append(raw_upper)

        global_raw_lower = min(raw_lowers)
        global_raw_upper = max(raw_uppers)
        global_bounds = self._safe_bounds(global_raw_lower, global_raw_upper)
        return {
            "C_opt": global_bounds,
            "scenario_C_opt": scenario_bounds,
            "optimization_results": optimization_results,
            "num_optimization_programs": 2 * int(self.poa.num_empirical_scenarios),
        }

    def run_optimal_cost_bounds(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS[
            "optimal_cost_bounds"
        ].format(regime_name=self.poa.regime_name)

        start = time.perf_counter()
        bound_report = self.compute_optimal_cost_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        elapsed = time.perf_counter() - start

        report = {
            "metadata": {
                "description": (
                    "DRO optimal dispatch cost bounds computed using the "
                    "scenario-indexed true-cost optimal KKT block."
                ),
                "method": "dro_optimal_dispatch_kkt_bounds",
                "model_type": "DRO_PoA",
                "tightening_type": "regime_wide",
                "tightening_scope": "scenario_wise",
                "reference_case": self.poa.reference_case,
                "regime_set": self.poa.regime_set,
                "regime_name": self.poa.regime_name,
                "num_time_steps": int(self.poa.num_time_steps),
                "num_empirical_scenarios": int(self.poa.num_empirical_scenarios),
                "eta": float(self.poa.eta),
                "epsilon": float(self.poa.epsilon),
                "runtime_seconds": float(elapsed),
            },
            "optimal_cost_bounds": bound_report["C_opt"],
            "scenario_optimal_cost_bounds": bound_report["scenario_C_opt"],
            "optimal_cost_bound_optimization_results": bound_report[
                "optimization_results"
            ],
            "num_optimization_programs": bound_report["num_optimization_programs"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
        }
        self.tightening_data["optimal_cost_bounds"] = report["optimal_cost_bounds"]
        self.tightening_data["scenario_optimal_cost_bounds"] = report[
            "scenario_optimal_cost_bounds"
        ]
        self.tightening_data["optimal_cost_bound_optimization_results"] = report[
            "optimal_cost_bound_optimization_results"
        ]
        return self._save_stage_report("optimal_cost_bounds", report, output_path)
