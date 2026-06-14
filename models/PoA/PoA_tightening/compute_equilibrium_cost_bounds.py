from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    SolverFactory,
    TerminationCondition,
    maximize,
    value,
)

from models.helper import derive_poa_upper_bound
from models.PoA.PoA_tightening.tightening_main import (
    DEFAULT_TIGHTENING_OUTPUT_PATHS,
    PoATighteningMain,
)

_OPTIMAL_TERMINATIONS = {
    TerminationCondition.optimal,
    TerminationCondition.locallyOptimal,
}


class EquilibriumCostBoundsComputer(PoATighteningMain):
    """Compute a valid upper bound on the equilibrium cost C_eq over the support set.

    Mirrors ``OptimalCostBoundsComputer`` but builds the *equilibrium* KKT block
    instead of the optimal-dispatch block, embeds the NN/true-cost bidding
    policy, and keeps the certified ``alpha_bounds`` from the tightening report
    as additional valid bounds. Combined with ``min C_opt`` from the
    optimal_cost_bounds stage it yields the McCormick PoA upper bound
    ``PoA_U = max C_eq / min C_opt``.
    """

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

    def _apply_alpha_bounds(
        self,
        m: ConcreteModel,
        alpha_bounds: dict[tuple[int, int, int], dict[str, float]],
    ) -> None:
        """Bracket the bids alpha within their certified [lower, upper]."""
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
        """Fix any complementarity binaries certified by the slack stage (valid, speeds the solve)."""
        for var_name, entries in (fixed_binaries or {}).items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, details in entries.items():
                index = tuple(int(part) for part in str(key).split(",") if part != "")
                if index in binary_var:
                    binary_var[index].fix(int((details or {}).get("fixed_value", 0)))

    def _build_equilibrium_kkt_bound_model(self) -> ConcreteModel:
        """
        Build the support set, bidding policy, and equilibrium KKT block.

        Excludes the optimal-dispatch block and the McCormick/PoA linking. The
        bound model is built as ``poa.tightening_model`` so the optimizer's main
        MPEC ``poa.model`` is never overwritten; the shared ``poa._build_*``
        helpers write to ``poa.model``, so it is temporarily pointed at the bound
        model and restored in the finally block (mirrors OptimalCostBoundsComputer).
        """
        self.poa._ensure_loaded_bounds_prepared()
        if not getattr(self.poa, "alpha_bounds", None):
            raise ValueError(
                "Equilibrium cost bounds require alpha bounds; compute or load the "
                "alpha_bounds tightening stage first."
            )
        saved_model = self.poa.__dict__.get("model", None)
        self.poa.tightening_model = ConcreteModel()
        self.poa.model = self.poa.tightening_model
        try:
            self._build_index_sets()
            self._build_PoA_variables()
            self._build_equilibrium_variables()
            self._build_complementarity_equilibrium_variables()
            self._build_support_set()
            self._build_policy_constraints()
            self._apply_alpha_bounds(self.model, self.poa.alpha_bounds)
            self._build_lower_level_equilibrium_constraints()
            self._build_KKT_stationarity_equilibrium_constraints()
            self._build_KKT_complementarity_equilibrium_constraints()
            self._apply_fixed_binaries(self.model, getattr(self.poa, "fixed_binaries", {}) or {})
            self.model.cost_definition_eq = Constraint(
                expr=self.model.C_eq == self.poa._get_equilibrium_cost_expression(self.model)
            )
            # Equilibrium cost is non-negative; an explicit lb keeps Gurobi presolve
            # from reporting INF_OR_UNBD for the otherwise free C_eq variable.
            if self.model.C_eq.lb is None or self.model.C_eq.lb < 0.0:
                self.model.C_eq.setlb(0.0)
            tightening_model = self.poa.tightening_model
        finally:
            if saved_model is not None:
                self.poa.model = saved_model
            else:
                self.poa.__dict__.pop("model", None)
        return tightening_model

    @staticmethod
    def _certified_objective_upper_bound(results: Any) -> Optional[float]:
        """Best dual (upper) bound on the maximize objective, or None.

        For a maximize problem the certified optimum lies in [incumbent,
        dual_bound]; Pyomo reports the two as problem.lower_bound and
        problem.upper_bound (naming is sense-dependent), so the larger finite value
        upper-bounds the true maximum -- valid even when the solve hits a time
        limit.
        """
        try:
            problem = results.problem[0]
        except (AttributeError, IndexError, TypeError):
            return None
        candidates = []
        for attr in ("upper_bound", "lower_bound"):
            raw = getattr(problem, attr, None)
            if raw is None:
                continue
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                candidates.append(numeric)
        return max(candidates) if candidates else None

    def _solve_max_equilibrium_cost(
        self,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_threads: Optional[int],
    ) -> dict[str, Any]:
        m = self._build_equilibrium_kkt_bound_model()
        m.tightening_objective = Objective(expr=m.C_eq, sense=maximize)

        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if solver_threads is not None and solver_name == "gurobi":
            solver.options["Threads"] = int(solver_threads)

        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        proven_optimal = termination in _OPTIMAL_TERMINATIONS

        incumbent = value(m.C_eq, exception=False)
        incumbent = float(incumbent) if incumbent is not None else None
        dual_bound = self._certified_objective_upper_bound(results)

        # A valid UPPER bound on max C_eq: the proven optimum when available, else
        # the solver's certified dual bound (which upper-bounds the true maximum),
        # so a time-limited solve still yields a valid (looser) bound -- never an
        # invalid (too-small) one from an unconverged incumbent.
        if proven_optimal and incumbent is not None:
            certified_max = incumbent
        elif dual_bound is not None:
            certified_max = dual_bound
        else:
            raise RuntimeError(
                "Equilibrium-cost maximization did not prove optimality and no "
                f"finite dual bound is available (termination={termination})."
            )

        return {
            "value": float(certified_max),
            "proven_optimal": bool(proven_optimal),
            "incumbent_C_eq": incumbent,
            "dual_bound_C_eq": dual_bound,
            "solver_status": str(results.solver.status),
            "termination_condition": str(termination),
            "num_variables": int(m.nvariables()),
            "num_constraints": int(m.nconstraints()),
            "active_constraints": sum(1 for _ in m.component_data_objects(Constraint, active=True)),
        }

    @staticmethod
    def _safe_max(raw_max: float) -> dict[str, float]:
        if raw_max < 0.0:
            raise ValueError("Computed C_eq upper bound must be non-negative")
        safe_max = raw_max + abs(raw_max) * 1e-6 + 1e-6
        return {"max": float(safe_max), "raw_max": float(raw_max)}

    def compute_equilibrium_cost_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        upper_result = self._solve_max_equilibrium_cost(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        bounds = self._safe_max(upper_result["value"])
        return {
            "C_eq": {
                "max": bounds["max"],
                "raw_max": bounds["raw_max"],
            },
            "optimization_results": {"upper": upper_result},
            "num_optimization_programs": 1,
        }

    def _resolve_derived_poa_bounds(
        self,
        c_eq_max: float,
        poa_bounds_lower: float,
        poa_bounds_margin: float,
    ) -> list[float] | None:
        """Combine this stage's max C_eq with min C_opt into the McCormick PoA box.

        min C_opt is read from the optimal_cost_bounds section, which is already
        merged into ``tightening_data`` because that always-on stage runs before
        this one. Returns ``[PoA_L, PoA_U]`` or None if min C_opt is unavailable.
        """
        optimal_cost_bounds = self.tightening_data.get("optimal_cost_bounds", {}) or {}
        if "C_opt" in optimal_cost_bounds and isinstance(optimal_cost_bounds.get("C_opt"), dict):
            optimal_cost_bounds = optimal_cost_bounds.get("C_opt", {}) or {}
        c_opt_min = optimal_cost_bounds.get("lower")
        if c_opt_min is None:
            return None
        poa_lower, poa_upper = derive_poa_upper_bound(
            c_eq_max,
            float(c_opt_min),
            lower=poa_bounds_lower,
            margin=poa_bounds_margin,
        )
        return [poa_lower, poa_upper]

    def run_equilibrium_cost_bounds(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
        poa_bounds_lower: float = 1.0,
        poa_bounds_margin: float = 1e-3,
    ) -> dict[str, Any]:
        output_path = output_path or DEFAULT_TIGHTENING_OUTPUT_PATHS.get("equilibrium_cost_bounds")
        if not getattr(self.poa, "primal_big_m", None):
            raise ValueError("Primal Big-M values must be loaded before C_eq bounds.")
        if not getattr(self.poa, "alpha_bounds", None):
            raise ValueError("Alpha bounds must be loaded before C_eq bounds.")

        start = time.perf_counter()
        bound_report = self.compute_equilibrium_cost_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        elapsed = time.perf_counter() - start

        derived_poa_bounds = self._resolve_derived_poa_bounds(
            bound_report["C_eq"]["max"],
            poa_bounds_lower=poa_bounds_lower,
            poa_bounds_margin=poa_bounds_margin,
        )
        cost_ratio_bound = self._compute_cost_ratio_bound()

        # Tighten derived_poa_bounds with the cost-ratio bound kappa: PoA <= kappa
        # is valid whenever a feasible displacement pair exists in the alpha bounds,
        # independently of C_eq and C_opt (no optimization required for kappa).
        if cost_ratio_bound is not None:
            kappa = cost_ratio_bound["kappa"]
            if derived_poa_bounds is not None:
                if kappa < derived_poa_bounds[1]:
                    derived_poa_bounds[1] = kappa
            else:
                # c_opt_min unavailable but kappa still gives a valid upper bound.
                derived_poa_bounds = [poa_bounds_lower, kappa]

        report = {
            "metadata": {
                "description": (
                    "Equilibrium cost upper bound computed over the regime-induced "
                    "support set using the embedded NN/true-cost bidding policy, "
                    "certified alpha bounds, and the equilibrium KKT block."
                ),
                "method": "poa_equilibrium_kkt_nn_policy_alpha_bounded_max",
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "ambiguity_set": self._ambiguity_metadata(),
                "runtime_seconds": elapsed,
            },
            "equilibrium_cost_bounds": bound_report["C_eq"],
            "equilibrium_cost_bound_optimization_results": bound_report["optimization_results"],
            "num_optimization_programs": bound_report["num_optimization_programs"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
        }
        if cost_ratio_bound is not None:
            report["cost_ratio_bound"] = cost_ratio_bound
        if derived_poa_bounds is not None:
            # PoA_U = min(max C_eq / min C_opt, kappa); the final solve reads this box.
            report["derived_poa_bounds"] = derived_poa_bounds
            report["metadata"]["derived_poa_bounds_margin"] = float(poa_bounds_margin)
            kappa_str = (
                f", kappa={cost_ratio_bound['kappa']:.6g}" if cost_ratio_bound is not None else ""
            )
            print(
                f"[equilibrium_cost_bounds] max C_eq={bound_report['C_eq']['max']:.6g}"
                f"{kappa_str} -> derived PoA box "
                f"({derived_poa_bounds[0]:.6g}, {derived_poa_bounds[1]:.6g})"
            )
        else:
            print(
                "[equilibrium_cost_bounds] WARNING: optimal_cost_bounds (min C_opt) "
                "unavailable; derived_poa_bounds not written."
            )
        self.tightening_data["equilibrium_cost_bounds"] = report["equilibrium_cost_bounds"]
        self.tightening_data["equilibrium_cost_bound_optimization_results"] = report[
            "equilibrium_cost_bound_optimization_results"
        ]
        if derived_poa_bounds is not None:
            self.tightening_data["derived_poa_bounds"] = derived_poa_bounds
        return self._save_stage_report("equilibrium_cost_bounds", report, output_path)
