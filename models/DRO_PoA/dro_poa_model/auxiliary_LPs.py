from __future__ import annotations

from typing import Any

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    SolverFactory,
    Suffix,
    TerminationCondition,
    Var,
    minimize,
    value,
)

from models.DRO_PoA.dro_poa_model.support_set import (
    DROWassersteinSupportSet,
    support_tube_parameters,
)


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
        # Build the diagnostic tube from the SAME parameters as the model support
        # set (support_set.py), so "inside support per diagnostic" means W=0 is
        # feasible in the model. kappa_ar1 (innovation) and kappa_lvl (level) are
        # rho-independent; only the level scales differ between demand and wind.
        ar1_cov = getattr(self, "ar1_coverage", None)
        lvl_cov = getattr(self, "level_coverage", None)
        kappa_ar1, kappa_lvl, demand_scales = support_tube_parameters(
            T, self.demand_rho_fixed, ar1_cov, lvl_cov
        )
        _, _, wind_scales = support_tube_parameters(
            T, self.wind_rho_fixed, ar1_cov, lvl_cov
        )

        demand_shape = np.asarray(self.demand_shape, dtype=float)
        wind_shape = np.asarray(self.wind_shape, dtype=float)

        results = []
        for k in range(self.num_empirical_scenarios):
            D_emp = np.asarray(self.empirical_D[k], dtype=float)
            D_ref_vec = D_ref * self.mu_D_fixed * demand_shape
            # Time-varying level band: kappa_lvl * D_ref * sigma_D * _level_scale(rho, t),
            # matching support_set.py (scale 1.0 at t=0, growing to the stationary bound).
            margin_D = kappa_lvl * D_ref * self.sigma_D_fixed * np.asarray(demand_scales)

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

            for i in self.wind_physical_generator_ids:
                cap = float(self.static_physical_capacity[int(i)])
                P_emp = np.asarray(self.empirical_Pmax_phys[k][int(i)], dtype=float)
                P_ref_vec = cap * self.mu_W_fixed * wind_shape
                margin_W = kappa_lvl * cap * self.sigma_W_fixed * np.asarray(wind_scales)
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


class DROPoAMIPStart:
    """Empirical-trajectory MIP start for the DRO PoA MILP.

    When every empirical scenario lies inside the Wasserstein support set, the
    point D[k,t] = empirical_D[k][t], W[k] = 0 is feasible.  This mixin computes
    the full primal/dual/binary assignment that corresponds to that point and
    writes it into the main DRO model so Gurobi has an incumbent at the root node
    (its objective equals the average PoA of the empirical dispatch) and can prune
    rather than search for a first feasible solution.

    The economic dispatch needed for that assignment is solved as a small Pyomo LP
    (one per scenario per cost vector), mirroring the lower-level constraints of
    the DRO model.  The KKT duals are read straight from that LP's ``dual`` suffix,
    so by construction they satisfy the model's stationarity equations.  Method
    names are dispatch-scoped so they do not collide with the main model's
    _build_variables / _build_objective / _build_constraints under composition.

    Sign conventions follow the rest of the codebase (see economic_dispatch.py):
      - power balance is an equality, so lambda = -dual[power_balance];
      - every other lower-level row is written in '<= 0' form, so the nonnegative
        KKT multiplier is mu = max(0, -dual[row]).
    """

    def compute_empirical_mip_start(self, solver_name: str = "gurobi") -> None:
        """Build the empirical-trajectory point and load it as a Gurobi MIP start."""
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        m = self.model
        T = self.num_time_steps
        K = self.num_empirical_scenarios
        breakpoints = (
            list(self.mccormick_bounds["C_opt_breakpoints"])
            if self.objective_mode == "piecewise_mccormick" and self.mccormick_bounds
            else None
        )
        PoA_L = (
            self.mccormick_bounds["PoA"][0]
            if self.mccormick_bounds and "PoA" in self.mccormick_bounds
            else 1.0
        )

        for k in range(K):
            D_emp = np.asarray(self.empirical_D[k], dtype=float)
            Pmax_block_emp = self._empirical_block_capacities(k)
            alpha_ib = self._empirical_equilibrium_bids(k, D_emp, Pmax_block_emp)

            # True marginal costs for the optimal (social-planner) dispatch.
            costs_ib: dict[tuple[int, int], list[float]] = {}
            for i in range(self.num_physical_generators):
                for b in self.local_blocks_by_generator[i]:
                    global_b = self.local_to_global_block[(i, b)]
                    costs_ib[(i, b)] = [float(self.block_cost_vector[global_b])] * T

            p_init_k = self.p_init[k]
            eq_sol = self._solve_dispatch_lp(
                alpha_ib, Pmax_block_emp, D_emp.tolist(), p_init_k, solver_name
            )
            opt_sol = self._solve_dispatch_lp(
                costs_ib, Pmax_block_emp, D_emp.tolist(), p_init_k, solver_name
            )

            self._write_support_start(k, D_emp, Pmax_block_emp)
            self._write_regime_start()
            self._write_dispatch_start(
                k=k,
                solution=eq_sol,
                Pmax_block_emp=Pmax_block_emp,
                p_init_k=p_init_k,
                P_var=m.P_eq,
                alpha_var=m.alpha,
                lambda_var=m.lambda_eq,
                mu_upper_var=m.mu_upper_eq,
                mu_lower_var=m.mu_lower_eq,
                mu_ramp_up_var=m.mu_ramp_up_eq,
                mu_ramp_down_var=m.mu_ramp_down_eq,
                z_upper_var=m.z_upper_eq,
                z_lower_var=m.z_lower_eq,
                z_ramp_up_var=m.z_ramp_up_eq,
                z_ramp_down_var=m.z_ramp_down_eq,
                alpha_values=alpha_ib,
            )
            self._write_dispatch_start(
                k=k,
                solution=opt_sol,
                Pmax_block_emp=Pmax_block_emp,
                p_init_k=p_init_k,
                P_var=m.P_opt,
                alpha_var=None,
                lambda_var=m.lambda_opt,
                mu_upper_var=m.mu_upper_opt,
                mu_lower_var=m.mu_lower_opt,
                mu_ramp_up_var=m.mu_ramp_up_opt,
                mu_ramp_down_var=m.mu_ramp_down_opt,
                z_upper_var=m.z_upper_opt,
                z_lower_var=m.z_lower_opt,
                z_ramp_up_var=m.z_ramp_up_opt,
                z_ramp_down_var=m.z_ramp_down_opt,
                alpha_values=None,
            )
            self._write_cost_and_poa_start(k, eq_sol, opt_sol, breakpoints, PoA_L)

    # ------------------------------------------------------------------
    # Empirical-point helpers
    # ------------------------------------------------------------------

    def _empirical_block_capacities(self, k: int) -> dict[tuple[int, int], list[float]]:
        """Per-block capacity at the empirical point (W[k]=0 => empirical values).

        Wind blocks share the physical capacity equally; conventional blocks keep
        their static capacity.
        """
        T = self.num_time_steps
        Pmax_phys_emp = self.empirical_Pmax_phys[k]
        Pmax_block_emp: dict[tuple[int, int], list[float]] = {}
        for i in range(self.num_physical_generators):
            n_local = len(self.local_blocks_by_generator[i])
            for b in self.local_blocks_by_generator[i]:
                if i in self.wind_physical_generator_ids:
                    Pmax_block_emp[(i, b)] = [
                        float(Pmax_phys_emp[i][t]) / n_local for t in range(T)
                    ]
                else:
                    global_b = self.local_to_global_block[(i, b)]
                    Pmax_block_emp[(i, b)] = [
                        float(self.static_block_capacity[global_b])
                    ] * T
        return Pmax_block_emp

    def _empirical_equilibrium_bids(
        self,
        k: int,
        D_emp: "np.ndarray",
        Pmax_block_emp: dict[tuple[int, int], list[float]],
    ) -> dict[tuple[int, int], list[float]]:
        """Equilibrium bids alpha[i,b][t] at the empirical point.

        NN-policy generators bid the numpy forward pass of their stored network on
        the empirical features; all other generators bid their true marginal cost.
        """
        T = self.num_time_steps
        Pmax_phys_emp = self.empirical_Pmax_phys[k]
        alpha_ib: dict[tuple[int, int], list[float]] = {}

        total_wind = np.array([
            sum(float(Pmax_phys_emp[j][t]) for j in self.wind_physical_generator_ids)
            for t in range(T)
        ])

        for i in range(self.num_physical_generators):
            generator_name = self.physical_generator_names[i]
            if i not in self.nn_policy_generator_ids:
                for b in self.local_blocks_by_generator[i]:
                    global_b = self.local_to_global_block[(i, b)]
                    alpha_ib[(i, b)] = [float(self.block_cost_vector[global_b])] * T
                continue

            policy = self.nn_policies[generator_name]
            feature_columns = policy["feature_columns"]
            layers = policy["layers"]
            target_map = policy["target_map"]  # output index -> local block b

            for b in self.local_blocks_by_generator[i]:
                alpha_ib.setdefault((i, b), [0.0] * T)

            for t in range(T):
                previous_t = T - 1 if t == 0 else t - 1
                next_t = 0 if t == T - 1 else t + 1
                raw_features = {
                    "demand": float(D_emp[t]),
                    "previous_demand": float(D_emp[previous_t]),
                    "next_demand": float(D_emp[next_t]),
                    "total_wind_generation_capacity": float(total_wind[t]),
                    "previous_wind_generation_capacity": float(total_wind[previous_t]),
                    "next_wind_generation_capacity": float(total_wind[next_t]),
                    "residual_demand": float(D_emp[t]) - float(total_wind[t]),
                    "previous_residual_demand": float(D_emp[previous_t]) - float(total_wind[previous_t]),
                    "next_residual_demand": float(D_emp[next_t]) - float(total_wind[next_t]),
                    "total_demand_over_horizon": float(D_emp.sum()),
                    "total_wind_over_horizon": float(total_wind.sum()),
                    "total_residual_over_horizon": float((D_emp - total_wind).sum()),
                }

                # Normalize using the stored min/max stats (same as MILP embedding).
                features = np.array([
                    (raw_features[f] - self._nn_feature_bounds(generator_name, f)[0])
                    / max(
                        self._nn_feature_bounds(generator_name, f)[1]
                        - self._nn_feature_bounds(generator_name, f)[0],
                        self.normalization_epsilon,
                    )
                    for f in feature_columns
                ], dtype=float)

                # Forward pass: alternating Linear + ReLU layers.
                x = features
                for layer in layers:
                    layer_type = str(layer.get("type", "")).lower()
                    if layer_type == "linear":
                        weight = np.asarray(layer["weight"], dtype=float)
                        bias = np.asarray(layer["bias"], dtype=float)
                        x = weight @ x + bias
                    elif layer_type == "relu":
                        x = np.maximum(x, 0.0)

                for output_idx, local_block in target_map.items():
                    alpha_ib[(i, int(local_block))][t] = float(x[output_idx])

        return alpha_ib

    # ------------------------------------------------------------------
    # Dispatch LP: min sum bid*P s.t. balance, capacity, ramps
    # ------------------------------------------------------------------

    def _solve_dispatch_lp(
        self,
        bids_ib: dict[tuple[int, int], list[float]],
        Pmax_block: dict[tuple[int, int], list[float]],
        demand: list[float],
        p_init_i: list[float],
        solver_name: str = "gurobi",
    ) -> dict[str, Any]:
        """Build and solve one scenario's dispatch LP; return primal + KKT duals."""
        self._build_dispatch_model(bids_ib, Pmax_block, demand, p_init_i)
        m = self.dispatch_model
        solver = SolverFactory(solver_name)
        solver.solve(m, tee=False)

        T = self.num_time_steps
        pairs = list(self.generator_block_pairs)

        P = {
            (int(i), int(b), t): float(value(m.P[i, b, t]))
            for (i, b) in pairs for t in range(T)
        }
        # Equality dual negated; inequality (<= 0) duals made nonnegative.
        lambda_ = {t: -float(m.dual.get(m.power_balance[t], 0.0)) for t in range(T)}
        mu_upper = {
            (int(i), int(b), t): max(0.0, -float(m.dual.get(m.gen_upper[i, b, t], 0.0)))
            for (i, b) in pairs for t in range(T)
        }
        mu_lower = {
            (int(i), int(b), t): max(0.0, -float(m.dual.get(m.gen_lower[i, b, t], 0.0)))
            for (i, b) in pairs for t in range(T)
        }
        mu_ramp_up: dict[tuple[int, int], float] = {}
        mu_ramp_down: dict[tuple[int, int], float] = {}
        for i in range(self.num_physical_generators):
            mu_ramp_up[(i, 0)] = max(0.0, -float(m.dual.get(m.ramp_up_initial[i], 0.0)))
            mu_ramp_down[(i, 0)] = max(0.0, -float(m.dual.get(m.ramp_down_initial[i], 0.0)))
            for t in range(1, T):
                mu_ramp_up[(i, t)] = max(0.0, -float(m.dual.get(m.ramp_up[i, t], 0.0)))
                mu_ramp_down[(i, t)] = max(0.0, -float(m.dual.get(m.ramp_down[i, t], 0.0)))

        return {
            "P": P,
            "lambda": lambda_,
            "mu_upper": mu_upper,
            "mu_lower": mu_lower,
            "mu_ramp_up": mu_ramp_up,
            "mu_ramp_down": mu_ramp_down,
            "pairs": pairs,
        }

    def _build_dispatch_model(
        self,
        bids_ib: dict[tuple[int, int], list[float]],
        Pmax_block: dict[tuple[int, int], list[float]],
        demand: list[float],
        p_init_i: list[float],
    ) -> None:
        """Declare the dispatch model, its sets and dual suffix, then build the LP."""
        self._dispatch_bids = bids_ib
        self._dispatch_pmax = Pmax_block
        self._dispatch_demand = [float(d) for d in demand]
        self._dispatch_p_init = [float(p) for p in p_init_i]

        T = self.num_time_steps
        self.dispatch_model = ConcreteModel()
        self.dispatch_model.time_steps = Set(initialize=range(T))
        self.dispatch_model.time_steps_minus_1 = Set(initialize=range(1, T))
        self.dispatch_model.physical_generators = Set(
            initialize=range(self.num_physical_generators)
        )
        self.dispatch_model.generator_blocks = Set(
            dimen=2, initialize=list(self.generator_block_pairs)
        )
        self.dispatch_model.dual = Suffix(direction=Suffix.IMPORT)

        self._build_dispatch_variables()
        self._build_dispatch_objective()
        self._build_dispatch_constraints()

    def _build_dispatch_variables(self) -> None:
        m = self.dispatch_model
        # P[i,b,t] is free; nonnegativity is imposed by the gen_lower constraint so
        # its multiplier mu_lower is available from the LP dual.
        m.P = Var(m.generator_blocks, m.time_steps, domain=Reals)

    def _build_dispatch_objective(self) -> None:
        m = self.dispatch_model
        bids = self._dispatch_bids
        m.dispatch_cost = Objective(
            expr=sum(
                bids[(int(i), int(b))][t] * m.P[i, b, t]
                for (i, b) in m.generator_blocks
                for t in m.time_steps
            ),
            sense=minimize,
        )

    def _build_dispatch_constraints(self) -> None:
        m = self.dispatch_model
        pmax = self._dispatch_pmax
        demand = self._dispatch_demand
        p_init = self._dispatch_p_init

        def _p_total(m, i, t):
            return sum(m.P[i, b, t] for b in self.local_blocks_by_generator[int(i)])

        def power_balance_rule(m, t):
            # Written as demand - sum(P) == 0 (demand on the left) to match the
            # codebase dual convention, so lambda = -dual[power_balance] >= 0.
            return demand[t] - sum(m.P[i, b, t] for (i, b) in m.generator_blocks) == 0

        def gen_upper_rule(m, i, b, t):
            return m.P[i, b, t] - pmax[(int(i), int(b))][t] <= 0

        def gen_lower_rule(m, i, b, t):
            return -m.P[i, b, t] <= 0

        def ramp_up_rule(m, i, t):
            return _p_total(m, i, t) - _p_total(m, i, t - 1) - self.ramp_vector_up[int(i)] <= 0

        def ramp_up_initial_rule(m, i):
            return _p_total(m, i, 0) - p_init[int(i)] - self.ramp_vector_up[int(i)] <= 0

        def ramp_down_rule(m, i, t):
            return -_p_total(m, i, t) + _p_total(m, i, t - 1) - self.ramp_vector_down[int(i)] <= 0

        def ramp_down_initial_rule(m, i):
            return -_p_total(m, i, 0) + p_init[int(i)] - self.ramp_vector_down[int(i)] <= 0

        m.power_balance = Constraint(m.time_steps, rule=power_balance_rule)
        m.gen_upper = Constraint(m.generator_blocks, m.time_steps, rule=gen_upper_rule)
        m.gen_lower = Constraint(m.generator_blocks, m.time_steps, rule=gen_lower_rule)
        m.ramp_up = Constraint(m.physical_generators, m.time_steps_minus_1, rule=ramp_up_rule)
        m.ramp_up_initial = Constraint(m.physical_generators, rule=ramp_up_initial_rule)
        m.ramp_down = Constraint(m.physical_generators, m.time_steps_minus_1, rule=ramp_down_rule)
        m.ramp_down_initial = Constraint(m.physical_generators, rule=ramp_down_initial_rule)

    # ------------------------------------------------------------------
    # Writing the start values into the main DRO model
    # ------------------------------------------------------------------

    def _write_support_start(
        self,
        k: int,
        D_emp: "np.ndarray",
        Pmax_block_emp: dict[tuple[int, int], list[float]],
    ) -> None:
        """Transport/support variables: the empirical point has W[k]=0."""
        m = self.model
        T = self.num_time_steps
        m.wasserstein_distance[k].set_value(0.0)
        for t in range(T):
            m.D[k, t].set_value(float(D_emp[t]))
            m.D_transport_abs_deviation[k, t].set_value(0.0)
            m.D_abs_deviation[k, t].set_value(0.0)
        for i in range(self.num_physical_generators):
            for b in self.local_blocks_by_generator[i]:
                for t in range(T):
                    m.P_max_block[k, i, b, t].set_value(float(Pmax_block_emp[(i, b)][t]))
            for t in range(T):
                m.P_max_phys_transport_abs_deviation[k, i, t].set_value(0.0)
                if i in self.wind_physical_generator_ids:
                    m.P_max_phys_abs_deviation[k, i, t].set_value(0.0)

    def _write_regime_start(self) -> None:
        """Regime-fixing variables: pin to their fixed values (shared across k)."""
        m = self.model
        m.mu_D.set_value(float(self.mu_D_fixed))
        m.sigma_D.set_value(float(self.sigma_D_fixed))
        m.mu_W.set_value(float(self.mu_W_fixed))
        m.sigma_W.set_value(float(self.sigma_W_fixed))
        m.rho_D.set_value(float(self.demand_rho_fixed))
        m.rho_W.set_value(float(self.wind_rho_fixed))
        m.peak_W.set_value(float(self.peak_W_fixed))

    def _write_dispatch_start(
        self,
        k: int,
        solution: dict[str, Any],
        Pmax_block_emp: dict[tuple[int, int], list[float]],
        p_init_k: list[float],
        P_var: Any,
        alpha_var: Any,
        lambda_var: Any,
        mu_upper_var: Any,
        mu_lower_var: Any,
        mu_ramp_up_var: Any,
        mu_ramp_down_var: Any,
        z_upper_var: Any,
        z_lower_var: Any,
        z_ramp_up_var: Any,
        z_ramp_down_var: Any,
        alpha_values: dict[tuple[int, int], list[float]] | None,
    ) -> None:
        """Write one dispatch level's primal, duals, and complementarity binaries.

        Used for both the equilibrium level (bids = alpha) and the optimal level
        (bids = true cost).  Binaries come from the primal active set; duals come
        from the dispatch LP so the model's stationarity holds by construction.
        """
        T = self.num_time_steps
        P = solution["P"]
        lambda_ = solution["lambda"]

        for t in range(T):
            lambda_var[k, t].set_value(float(lambda_[t]))
        for i in range(self.num_physical_generators):
            mu_ramp_up_var[k, i, T].set_value(0.0)
            mu_ramp_down_var[k, i, T].set_value(0.0)

        for (i, b) in solution["pairs"]:
            for t in range(T):
                p_val = float(P[(int(i), int(b), t)])
                p_max = float(Pmax_block_emp[(i, b)][t])
                P_var[k, i, b, t].set_value(p_val)
                if alpha_var is not None and alpha_values is not None:
                    alpha_var[k, i, b, t].set_value(float(alpha_values[(i, b)][t]))
                mu_upper_var[k, i, b, t].set_value(solution["mu_upper"][(int(i), int(b), t)])
                mu_lower_var[k, i, b, t].set_value(solution["mu_lower"][(int(i), int(b), t)])
                # z_upper=1 iff the upper capacity bound is active at the solution.
                z_upper_var[k, i, b, t].set_value(1 if p_val >= p_max - 1e-8 else 0)
                z_lower_var[k, i, b, t].set_value(1 if p_val <= 1e-8 else 0)

        for i in range(self.num_physical_generators):
            for t in range(T):
                mu_ramp_up_var[k, i, t].set_value(solution["mu_ramp_up"][(i, t)])
                mu_ramp_down_var[k, i, t].set_value(solution["mu_ramp_down"][(i, t)])
                p_total_t = sum(
                    float(P[(i, b, t)]) for b in self.local_blocks_by_generator[i]
                )
                p_total_prev = (
                    float(p_init_k[i]) if t == 0
                    else sum(
                        float(P[(i, b, t - 1)]) for b in self.local_blocks_by_generator[i]
                    )
                )
                ramp_up_slack = float(self.ramp_vector_up[i]) - (p_total_t - p_total_prev)
                ramp_dn_slack = float(self.ramp_vector_down[i]) - (p_total_prev - p_total_t)
                z_ramp_up_var[k, i, t].set_value(1 if ramp_up_slack <= 1e-8 else 0)
                z_ramp_down_var[k, i, t].set_value(1 if ramp_dn_slack <= 1e-8 else 0)

    def _write_cost_and_poa_start(
        self,
        k: int,
        eq_sol: dict[str, Any],
        opt_sol: dict[str, Any],
        breakpoints: list[float] | None,
        PoA_L: float,
    ) -> None:
        """Costs, PoA, and the McCormick / piecewise-McCormick auxiliary variables."""
        m = self.model
        T = self.num_time_steps

        def _total_cost(P: dict[tuple[int, int, int], float]) -> float:
            return float(sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * float(P[(int(i), int(b), t)])
                for (i, b) in self.generator_block_pairs
                for t in range(T)
            ))

        c_eq_val = _total_cost(eq_sol["P"])
        c_opt_val = _total_cost(opt_sol["P"])
        poa_val = float(c_eq_val / c_opt_val) if c_opt_val > 1e-12 else PoA_L
        m.C_eq[k].set_value(c_eq_val)
        m.C_opt[k].set_value(c_opt_val)
        m.PoA[k].set_value(poa_val)

        if self.objective_mode == "mccormick":
            m.z_mccormick_product[k].set_value(c_eq_val)
        elif self.objective_mode == "piecewise_mccormick" and breakpoints is not None:
            active_piece = len(breakpoints) - 2  # default to the last piece
            for p_idx in range(len(breakpoints) - 1):
                if breakpoints[p_idx] <= c_opt_val <= breakpoints[p_idx + 1]:
                    active_piece = p_idx
                    break
            m.z_mccormick_product[k].set_value(c_eq_val)
            for p_idx in range(len(breakpoints) - 1):
                is_active = int(p_idx == active_piece)
                m.mccormick_piece_active[k, p_idx].set_value(is_active)
                m.C_opt_piece[k, p_idx].set_value(c_opt_val if is_active else 0.0)
                m.PoA_piece[k, p_idx].set_value(poa_val if is_active else 0.0)
                m.z_mccormick_piece[k, p_idx].set_value(
                    poa_val * c_opt_val if is_active else 0.0
                )
