from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
    value,
)

from models.DRO_PoA.DRO_PoA_tightening.tightening_main import (
    DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS,
    DROPoATighteningMain,
)
from models.helper import derive_poa_upper_bound

_OPTIMAL_TERMINATIONS = {
    TerminationCondition.optimal,
    TerminationCondition.locallyOptimal,
}


class DROEquilibriumCostBoundsComputer(DROPoATighteningMain):
    """Compute a valid upper bound on the DRO equilibrium cost C_eq over the support set.

    Mirrors ``DROOptimalCostBoundsComputer`` but builds the *equilibrium* KKT block
    instead of the optimal-dispatch block, embeds the NN/true-cost bidding policy,
    and keeps the certified ``alpha_bounds`` from the regime-wide tightening
    report as additional valid bounds. The resulting ``max C_eq`` over the
    regime-wide support set is a valid upper bound; combined with ``min C_opt``
    it gives the McCormick PoA upper bound ``PoA_U = max C_eq / min C_opt``.
    """

    def _build_index_sets(self) -> None:
        dro = self.poa
        m = self.model
        m.scenarios = Set(initialize=[0])  # single regime scenario
        m.time_steps = Set(initialize=range(dro.num_time_steps))
        m.time_steps_minus_1 = Set(initialize=range(1, dro.num_time_steps))
        m.time_steps_plus_1 = Set(initialize=range(dro.num_time_steps + 1))
        m.physical_generators = Set(initialize=range(dro.num_physical_generators))
        m.generator_blocks = Set(dimen=2, initialize=dro.generator_block_pairs)
        m.wind_physical_generators = Set(initialize=dro.wind_physical_generator_ids)
        m.conventional_physical_generators = Set(initialize=dro.conventional_physical_generator_ids)
        m.wind_blocks = Set(dimen=2, initialize=dro.wind_block_pairs)
        m.conventional_blocks = Set(dimen=2, initialize=dro.conventional_block_pairs)

    def _equilibrium_cost_expression(self, m: ConcreteModel, scenario_idx: int) -> Any:
        dro = self.poa
        k = int(scenario_idx)
        return sum(
            dro.block_cost_vector[dro.local_to_global_block[(int(i), int(b))]] * m.P_eq[k, i, b, t]
            for (i, b) in m.generator_blocks
            for t in m.time_steps
        )

    def _apply_alpha_bounds(
        self,
        m: ConcreteModel,
        alpha_bounds: dict[tuple[int, ...], dict[str, float]],
    ) -> None:
        """Bracket the bids alpha within their certified [lower, upper]."""
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
        """Fix any complementarity binaries certified by the slack stage (valid, speeds the solve)."""
        for var_name, entries in (fixed_binaries or {}).items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, details in (entries or {}).items():
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

    def _build_equilibrium_kkt_bound_model(self) -> ConcreteModel:
        """
        Build the DRO support set, bidding policy, and equilibrium KKT block.

        Mirrors ``DROOptimalCostBoundsComputer._build_optimal_dispatch_kkt_bound_model``
        but on the equilibrium side: excludes the optimal-dispatch block, transport
        objective terms, and PoA links. p_init is kept at its regime value (not
        freed); a denominator that frees p_init only makes ``max C_eq / min C_opt``
        more conservative, so the bound stays valid.
        """
        dro = self.poa
        if getattr(dro, "tightening_report", None) or getattr(dro, "primal_big_m", None):
            dro._prepare_loaded_bounds()
        if not getattr(dro, "alpha_bounds", None):
            raise ValueError(
                "DRO equilibrium cost bounds require alpha bounds; compute or load "
                "the alpha_bounds tightening stage first."
            )

        saved_model = dro.__dict__.get("model", None)
        dro.tightening_model = ConcreteModel()
        dro.model = dro.tightening_model
        try:
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
            dro._build_equilibrium_variables()
            dro._build_complementarity_equilibrium_variables()
            dro._build_support_set()
            dro._build_policy_constraints()
            self._apply_alpha_bounds(m, dro.alpha_bounds)
            dro._build_lower_level_equilibrium_constraints()
            dro._build_KKT_stationarity_equilibrium_constraints()
            dro._build_KKT_complementarity_equilibrium_constraints()
            self._apply_fixed_binaries(m, getattr(dro, "fixed_binaries", {}) or {})

            m.C_eq = Var(m.scenarios, domain=Reals)

            def cost_eq_rule(m, k):
                return m.C_eq[k] == self._equilibrium_cost_expression(m, int(k))

            m.cost_definition_eq = Constraint(m.scenarios, rule=cost_eq_rule)
            # Equilibrium cost is non-negative; an explicit lb keeps Gurobi presolve
            # from reporting INF_OR_UNBD for the otherwise free C_eq variables.
            for k in m.scenarios:
                if m.C_eq[k].lb is None or m.C_eq[k].lb < 0.0:
                    m.C_eq[k].setlb(0.0)
            tightening_model = dro.tightening_model
        finally:
            if saved_model is not None:
                dro.model = saved_model
            else:
                dro.__dict__.pop("model", None)
        return tightening_model

    @staticmethod
    def _certified_objective_upper_bound(results: Any) -> Optional[float]:
        """Best dual (upper) bound on the maximize objective, or None (valid under a time limit)."""
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
        scenario_idx: int,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_threads: Optional[int],
    ) -> dict[str, Any]:
        m = self._build_equilibrium_kkt_bound_model()
        k = int(scenario_idx)
        m.tightening_objective = Objective(expr=m.C_eq[k], sense=maximize)

        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if solver_threads is not None and solver_name == "gurobi":
            solver.options["Threads"] = int(solver_threads)

        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        proven_optimal = termination in _OPTIMAL_TERMINATIONS

        incumbent = value(m.C_eq[k], exception=False)
        incumbent = float(incumbent) if incumbent is not None else None
        dual_bound = self._certified_objective_upper_bound(results)

        if proven_optimal and incumbent is not None:
            certified_max = incumbent
        elif dual_bound is not None:
            certified_max = dual_bound
        else:
            raise RuntimeError(
                "DRO equilibrium-cost maximization did not prove optimality and no "
                f"finite dual bound is available (termination={termination})."
            )

        return {
            "scenario_idx": k,
            "value": float(certified_max),
            "proven_optimal": bool(proven_optimal),
            "incumbent_C_eq": incumbent,
            "dual_bound_C_eq": dual_bound,
            "solver_status": str(results.solver.status),
            "termination_condition": str(termination),
            "num_variables": int(m.nvariables()),
            "num_constraints": int(m.nconstraints()),
            "active_constraints": int(
                sum(1 for _ in m.component_data_objects(Constraint, active=True))
            ),
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
            scenario_idx=0,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        bounds = self._safe_max(upper_result["value"])
        return {
            "C_eq": {"max": bounds["max"], "raw_max": bounds["raw_max"]},
            "scenario_C_eq": {"0": bounds},
            "optimization_results": {"0": {"upper": upper_result}},
            "num_optimization_programs": 1,
        }

    def _resolve_derived_poa_bounds(
        self,
        c_eq_max: float,
        poa_bounds_lower: float,
        poa_bounds_margin: float,
    ) -> list[float] | None:
        """Combine this stage's max C_eq with min C_opt into the McCormick PoA box.

        min C_opt is read from the optimal_cost_bounds section, already merged into
        ``tightening_data`` because that stage runs before this one. Returns
        ``[PoA_L, PoA_U]`` or None if min C_opt is unavailable.
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
        output_path = output_path or DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS[
            "equilibrium_cost_bounds"
        ].format(regime_name=self.poa.regime_name)
        if not getattr(self.poa, "alpha_bounds", None):
            raise ValueError("Alpha bounds must be loaded before DRO C_eq bounds.")

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
                    "DRO equilibrium cost upper bound computed over the full regime "
                    "support set using the embedded NN/true-cost bidding policy, "
                    "certified alpha bounds, and the equilibrium KKT block."
                ),
                "method": "dro_equilibrium_kkt_nn_policy_alpha_bounded_max",
                "model_type": "DRO_PoA",
                "tightening_type": "regime_wide",
                "tightening_scope": "regime_wide",
                "reference_case": self.poa.reference_case,
                "regime_set": self.poa.regime_set,
                "regime_name": self.poa.regime_name,
                "num_time_steps": int(self.poa.num_time_steps),
                "num_empirical_scenarios": int(self.poa.num_empirical_scenarios),
                "eta": float(self.poa.eta),
                "epsilon": float(self.poa.epsilon),
                "runtime_seconds": float(elapsed),
            },
            "equilibrium_cost_bounds": bound_report["C_eq"],
            "scenario_equilibrium_cost_bounds": bound_report["scenario_C_eq"],
            "equilibrium_cost_bound_optimization_results": bound_report["optimization_results"],
            "num_optimization_programs": bound_report["num_optimization_programs"],
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
        }
        if cost_ratio_bound is not None:
            report["cost_ratio_bound"] = cost_ratio_bound
        if derived_poa_bounds is not None:
            # PoA_U = min(max C_eq / min C_opt, kappa); the final DRO solve reads this box.
            report["derived_poa_bounds"] = derived_poa_bounds
            report["metadata"]["derived_poa_bounds_margin"] = float(poa_bounds_margin)
            kappa_str = (
                f", kappa={cost_ratio_bound['kappa']:.6g}" if cost_ratio_bound is not None else ""
            )
            print(
                f"[equilibrium_cost_bounds][{self.poa.regime_name}] "
                f"max C_eq={bound_report['C_eq']['max']:.6g}{kappa_str} -> derived PoA box "
                f"({derived_poa_bounds[0]:.6g}, {derived_poa_bounds[1]:.6g})"
            )
        else:
            print(
                f"[equilibrium_cost_bounds][{self.poa.regime_name}] WARNING: "
                "optimal_cost_bounds (min C_opt) unavailable; derived_poa_bounds not written."
            )
        self.tightening_data["equilibrium_cost_bounds"] = report["equilibrium_cost_bounds"]
        self.tightening_data["scenario_equilibrium_cost_bounds"] = report[
            "scenario_equilibrium_cost_bounds"
        ]
        self.tightening_data["equilibrium_cost_bound_optimization_results"] = report[
            "equilibrium_cost_bound_optimization_results"
        ]
        if derived_poa_bounds is not None:
            self.tightening_data["derived_poa_bounds"] = derived_poa_bounds
        return self._save_stage_report("equilibrium_cost_bounds", report, output_path)
