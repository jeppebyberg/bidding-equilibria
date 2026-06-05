from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    SolverFactory,
    TerminationCondition,
    maximize,
    minimize,
    value,
)

from models.PoA.PoA_tightening.tightening_main import (
    DEFAULT_TIGHTENING_OUTPUT_PATHS,
    PoATighteningMain,
)


def _bound_sense(bound_name: str) -> Any:
    if bound_name == "lower":
        return minimize
    if bound_name == "upper":
        return maximize
    raise ValueError(f"Unknown bound name: {bound_name}")


class OptimalCostBoundsComputer(PoATighteningMain):
    """Compute valid denominator bounds from the optimal-dispatch KKT block."""

    def _build_index_sets(self) -> None:
        from pyomo.environ import Set

        m = self.model
        m.time_steps = Set(initialize=range(self.poa.num_time_steps))
        m.time_steps_minus_1 = Set(initialize=range(1, self.poa.num_time_steps))
        m.time_steps_plus_1 = Set(initialize=range(self.poa.num_time_steps + 1))
        m.physical_generators = Set(initialize=range(self.poa.num_physical_generators))
        m.generator_blocks = Set(dimen=2, initialize=self.poa.generator_block_pairs)
        m.wind_physical_generators = Set(initialize=self.poa.wind_physical_generator_ids)
        m.conventional_physical_generators = Set(
            initialize=self.poa.conventional_physical_generator_ids
        )
        m.wind_blocks = Set(dimen=2, initialize=self.poa.wind_block_pairs)
        m.conventional_blocks = Set(dimen=2, initialize=self.poa.conventional_block_pairs)

    def _build_optimal_dispatch_kkt_bound_model(self) -> ConcreteModel:
        """
        Build only the ambiguity support set and true-cost optimal KKT block.

        This intentionally excludes equilibrium-side constraints, bidding-policy
        constraints, neural-network variables, PoA objective/linking constraints,
        and aggregate dual valid inequalities.

        The bound model is built as ``poa.tightening_model`` so the optimizer's
        main MPEC ``poa.model`` is never overwritten -- this matters when these
        bounds are solved from inside build_model() via
        ensure_default_bounds_available(). The shared poa ``_build_*`` helpers
        write to ``poa.model``, so it is temporarily pointed at the bound model
        and restored in the finally block.
        """
        self.poa._ensure_loaded_bounds_prepared()
        saved_model = self.poa.__dict__.get("model", None)
        self.poa.tightening_model = ConcreteModel()
        self.poa.model = self.poa.tightening_model
        try:
            self._build_index_sets()
            self._build_PoA_variables()
            self._build_optimal_variables()
            self._build_complementarity_optimal_variables()
            self._build_support_set()
            self._build_lower_level_optimal_constraints()
            self._build_KKT_stationarity_optimal_constraints()
            self._build_KKT_complementarity_optimal_constraints()
            self.model.cost_definition_opt = Constraint(
                expr=self.model.C_opt == self.poa._get_optimal_cost_expression(self.model)
            )
            tightening_model = self.poa.tightening_model
        finally:
            if saved_model is not None:
                self.poa.model = saved_model
            else:
                self.poa.__dict__.pop("model", None)
        return tightening_model

    def _solve_bound(
        self,
        bound_name: str,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_threads: Optional[int],
    ) -> dict[str, Any]:
        m = self._build_optimal_dispatch_kkt_bound_model()
        m.tightening_objective = Objective(
            expr=m.C_opt,
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
        bound_value = value(m.C_opt, exception=False) if solved else None
        return {
            "bound_name": bound_name,
            "value": float(bound_value) if bound_value is not None else None,
            "solver_status": str(results.solver.status),
            "termination_condition": str(termination),
            "num_variables": int(m.nvariables()),
            "num_constraints": int(m.nconstraints()),
            "active_constraints": sum(
                1 for _ in m.component_data_objects(Constraint, active=True)
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
            "raw_lower": float(raw_lower),
            "raw_upper": float(raw_upper),
            "safe_lower": float(safe_lower),
            "safe_upper": float(safe_upper),
        }

    def compute_optimal_cost_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        lower_result = self._solve_bound(
            "lower",
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        upper_result = self._solve_bound(
            "upper",
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        if lower_result["value"] is None or upper_result["value"] is None:
            raise RuntimeError("Could not compute both C_opt bounds")

        bounds = self._safe_bounds(lower_result["value"], upper_result["value"])
        return {
            "C_opt": {
                "lower": bounds["safe_lower"],
                "upper": bounds["safe_upper"],
                "raw_lower": bounds["raw_lower"],
                "raw_upper": bounds["raw_upper"],
            },
            "optimization_results": {
                "lower": lower_result,
                "upper": upper_result,
            },
            "num_optimization_programs": 2,
        }

    def run_optimal_cost_bounds(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or DEFAULT_TIGHTENING_OUTPUT_PATHS["optimal_cost_bounds"]
        if not getattr(self.poa, "primal_big_m", None):
            raise ValueError("Primal Big-M values must be loaded before C_opt bounds.")

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
                    "Optimal dispatch cost bounds computed over the regime-induced "
                    "support set using only the true-cost optimal KKT block."
                ),
                "method": "poa_optimal_dispatch_kkt_bounds",
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "ambiguity_set": self._ambiguity_metadata(),
                "runtime_seconds": elapsed,
            },
            "optimal_cost_bounds": bound_report["C_opt"],
            "optimal_cost_bound_optimization_results": bound_report[
                "optimization_results"
            ],
            "num_optimization_programs": bound_report["num_optimization_programs"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
        }
        self.tightening_data["optimal_cost_bounds"] = report["optimal_cost_bounds"]
        self.tightening_data["optimal_cost_bound_optimization_results"] = report[
            "optimal_cost_bound_optimization_results"
        ]
        return self._save_stage_report("optimal_cost_bounds", report, output_path)
