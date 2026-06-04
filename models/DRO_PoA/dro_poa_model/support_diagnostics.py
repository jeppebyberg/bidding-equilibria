from __future__ import annotations

import numpy as np
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
    minimize,
    value,
)

from models.DRO_PoA.dro_poa_model.support_set import DROWassersteinSupportSet, _ar1_kappa


class DROPoASupportDiagnostics:
    """Diagnostics for the Wasserstein support set U.

    For each empirical scenario this computes the minimum achievable Wasserstein-1
    transport distance onto the support set and counts which constraint families
    (pointwise level bands, t=0 innovation band, AR(1) innovation bands) are
    violated.  The minimum distance reveals the W-floor that high-eta runs cannot
    cross: if the empirical demand/wind trajectory lies outside U the optimizer
    cannot match it and W[k] > 0 is unavoidable.

    The minimum distance is solved as a small Pyomo LP (one per channel) so the
    diagnostic uses the same solver stack as the rest of the model rather than a
    separate scipy backend.  A pointwise-only projection would miss AR(1)
    violations, so the box bounds and AR(1) bands are imposed jointly.
    """

    def diagnose_empirical_support_set_violations(
        self,
        solver_name: str = "gurobi",
    ) -> list[dict]:
        """Per-scenario minimum Wasserstein distance to U and per-family violations.

        Returns a list of dicts, one per empirical scenario.
        """
        T = self.num_time_steps
        D_ref = float(self.demand_D_ref)
        coverage = float(
            getattr(self, "ar1_coverage", None)
            or DROWassersteinSupportSet.AR1_JOINT_COVERAGE
        )
        kappa_ar1 = _ar1_kappa(T, coverage)
        kappa_level = kappa_ar1

        demand_shape = np.asarray(self.demand_shape, dtype=float)
        wind_shape = np.asarray(self.wind_shape, dtype=float)

        results = []
        for k in range(self.num_empirical_scenarios):
            D_emp = np.asarray(self.empirical_D[k], dtype=float)
            D_ref_vec = D_ref * self.mu_D_fixed * demand_shape
            stationary_std_D = self.sigma_D_fixed / np.sqrt(1.0 - self.demand_rho_fixed ** 2)
            margin_D = kappa_level * D_ref * stationary_std_D

            lb_D = np.maximum(D_ref_vec - margin_D, 0.0)
            ub_D = D_ref_vec + margin_D

            innov_margin_D = kappa_ar1 * D_ref * self.sigma_D_fixed
            t0_ref_D = D_ref * self.mu_D_fixed * demand_shape[0]
            ar1_ref_D = D_ref * self.mu_D_fixed * (
                demand_shape[1:] - self.demand_rho_fixed * demand_shape[:-1]
            )
            t0_violation_D = int(abs(D_emp[0] - t0_ref_D) > innov_margin_D + 1e-9)
            innov_D = D_emp[1:] - self.demand_rho_fixed * D_emp[:-1]
            ar1_violations_D = int(np.sum(np.abs(innov_D - ar1_ref_D) > innov_margin_D + 1e-9))

            min_W_demand = self._solve_min_wasserstein_projection(
                D_emp,
                lb_D,
                ub_D,
                ar1_ref_D,
                innov_margin_D,
                self.demand_rho_fixed,
                t0_ref_D,
                solver_name=solver_name,
            )

            min_W_wind = 0.0
            wind_ar1_violations = 0
            wind_t0_violations = 0
            wind_level_violations = 0
            stationary_std_W = self.sigma_W_fixed / np.sqrt(1.0 - self.wind_rho_fixed ** 2)

            for i in self.wind_physical_generator_ids:
                cap = float(self.static_physical_capacity[int(i)])
                P_emp = np.asarray(self.empirical_Pmax_phys[k][int(i)], dtype=float)
                P_ref_vec = cap * self.mu_W_fixed * wind_shape
                margin_W = kappa_level * cap * stationary_std_W
                lb_W = np.maximum(P_ref_vec - margin_W, 0.0)
                ub_W = np.minimum(P_ref_vec + margin_W, cap)

                innov_margin_W = kappa_ar1 * cap * self.sigma_W_fixed
                t0_ref_W = cap * self.mu_W_fixed * wind_shape[0]
                ar1_ref_W = cap * self.mu_W_fixed * (
                    wind_shape[1:] - self.wind_rho_fixed * wind_shape[:-1]
                )
                wind_t0_violations += int(abs(P_emp[0] - t0_ref_W) > innov_margin_W + 1e-9)
                innov_W = P_emp[1:] - self.wind_rho_fixed * P_emp[:-1]
                wind_ar1_violations += int(
                    np.sum(np.abs(innov_W - ar1_ref_W) > innov_margin_W + 1e-9)
                )
                wind_level_violations += int(
                    np.sum((P_emp < lb_W - 1e-9) | (P_emp > ub_W + 1e-9))
                )

                # Per-generator wind support sets are independent (no coupling
                # constraints across generators), so the joint minimum distance is
                # the sum of the per-generator minima.
                min_W_wind += self._solve_min_wasserstein_projection(
                    P_emp,
                    lb_W,
                    ub_W,
                    ar1_ref_W,
                    innov_margin_W,
                    self.wind_rho_fixed,
                    t0_ref_W,
                    solver_name=solver_name,
                )

            results.append({
                "scenario_k": k,
                "min_W_demand": min_W_demand,
                "min_W_wind": min_W_wind,
                "min_W_total": min_W_demand + min_W_wind,
                "coverage": coverage,
                "kappa": kappa_ar1,
                "demand_pointwise_violations": int(
                    np.sum(D_emp < lb_D - 1e-9) + np.sum(D_emp > ub_D + 1e-9)
                ),
                "demand_t0_violations": t0_violation_D,
                "demand_ar1_violations": ar1_violations_D,
                "wind_level_violations": wind_level_violations,
                "wind_t0_violations": wind_t0_violations,
                "wind_ar1_violations": wind_ar1_violations,
            })
        return results

    # ------------------------------------------------------------------
    # Minimum-distance projection LP
    #
    # min sum_t |x_emp[t] - x[t]|  s.t.  x in the AR(1) support tube.
    #
    # The tube for a single channel (demand, or one wind generator) is defined by
    # pointwise box bounds [lb[t], ub[t]] (level bands), a t=0 innovation band
    # around t0_ref, and AR(1) innovation bands around ar1_ref for t >= 1, all of
    # half-width innov_margin.  ar1_ref[s] is the reference innovation for step
    # s+1 (length T-1).  The optimal objective is the Wasserstein-1 transport cost
    # of moving x_emp onto U.  Method names are projection-scoped so they do not
    # collide with the main DRO model's _build_variables/_build_objective/
    # _build_constraints when this mixin is composed into DRO_PoAOptimization.
    # ------------------------------------------------------------------

    def _solve_min_wasserstein_projection(
        self,
        x_emp: "np.ndarray",
        lb: "np.ndarray",
        ub: "np.ndarray",
        ar1_ref: "np.ndarray",
        innov_margin: float,
        rho: float,
        t0_ref: float,
        solver_name: str = "gurobi",
    ) -> float:
        """Build and solve the projection LP for one channel; return min L1 distance."""
        self._build_projection_model(
            x_emp, lb, ub, ar1_ref, innov_margin, rho, t0_ref
        )
        solver = SolverFactory(solver_name)
        results = solver.solve(self.projection_model, tee=False)
        if results.solver.termination_condition == TerminationCondition.infeasible:
            return float("inf")
        return float(value(self.projection_model.projection_objective))

    def _build_projection_model(
        self,
        x_emp: "np.ndarray",
        lb: "np.ndarray",
        ub: "np.ndarray",
        ar1_ref: "np.ndarray",
        innov_margin: float,
        rho: float,
        t0_ref: float,
    ) -> None:
        """Declare the projection model and its sets, then build the LP."""
        # Channel data consumed by the _build_projection_* methods below.
        self._proj_x_emp = np.asarray(x_emp, dtype=float)
        self._proj_lb = np.asarray(lb, dtype=float)
        self._proj_ub = np.asarray(ub, dtype=float)
        self._proj_ar1_ref = np.asarray(ar1_ref, dtype=float)
        self._proj_innov_margin = float(innov_margin)
        self._proj_rho = float(rho)
        self._proj_t0_ref = float(t0_ref)

        T = len(self._proj_x_emp)
        self.projection_model = ConcreteModel()
        self.projection_model.time_steps = Set(initialize=range(T))
        self.projection_model.time_steps_minus_1 = Set(initialize=range(1, T))

        self._build_projection_variables()
        self._build_projection_objective()
        self._build_projection_constraints()

    def _build_projection_variables(self) -> None:
        m = self.projection_model
        lb = self._proj_lb
        ub = self._proj_ub

        def x_bounds(m, t):
            return (float(lb[t]), float(ub[t]))

        # x[t] is the projected trajectory (box-bounded by the level bands);
        # abs_dev[t] >= |x_emp[t] - x[t]| linearizes the L1 transport cost.
        m.x = Var(m.time_steps, domain=Reals, bounds=x_bounds)
        m.abs_dev = Var(m.time_steps, domain=NonNegativeReals)

    def _build_projection_objective(self) -> None:
        m = self.projection_model
        m.projection_objective = Objective(
            expr=sum(m.abs_dev[t] for t in m.time_steps),
            sense=minimize,
        )

    def _build_projection_constraints(self) -> None:
        m = self.projection_model
        rho = self._proj_rho
        innov_margin = self._proj_innov_margin
        t0_ref = self._proj_t0_ref
        ar1_ref = self._proj_ar1_ref
        x_emp = self._proj_x_emp

        # t=0 cold-start innovation band.
        m.projection_t0_up = Constraint(expr=m.x[0] <= t0_ref + innov_margin)
        m.projection_t0_down = Constraint(expr=m.x[0] >= t0_ref - innov_margin)

        # AR(1) innovation bands for t >= 1.
        def ar1_up_rule(m, t):
            return m.x[t] - rho * m.x[t - 1] <= float(ar1_ref[t - 1]) + innov_margin

        def ar1_down_rule(m, t):
            return m.x[t] - rho * m.x[t - 1] >= float(ar1_ref[t - 1]) - innov_margin

        m.projection_ar1_up = Constraint(m.time_steps_minus_1, rule=ar1_up_rule)
        m.projection_ar1_down = Constraint(m.time_steps_minus_1, rule=ar1_down_rule)

        # abs_dev[t] >= |x_emp[t] - x[t]|.
        def abs_pos_rule(m, t):
            return m.abs_dev[t] >= m.x[t] - float(x_emp[t])

        def abs_neg_rule(m, t):
            return m.abs_dev[t] >= float(x_emp[t]) - m.x[t]

        m.projection_abs_pos = Constraint(m.time_steps, rule=abs_pos_rule)
        m.projection_abs_neg = Constraint(m.time_steps, rule=abs_neg_rule)
