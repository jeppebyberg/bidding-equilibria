# NOTE ON COVERAGE SCOPE: the 95% joint-coverage claim applies to the AR(1) innovation
# tube alone. A fresh trajectory drawn from regime r lands entirely within the tube with
# probability 0.95 (under i.i.d. N(0, scale^2) innovations). The level box
# (demand_lower/upper, wind_lower/upper) and the budget constraint are additional
# feasibility conditions that further restrict U(r) beyond the tube; they couple across
# timesteps and do not carry a closed-form coverage statement. The 95% claim is therefore
# scoped to: every whitened innovation |xi_t| <= kappa_t * scale, not to full U(r)
# membership.
import numpy as np
from pyomo.environ import Constraint, Expression
from scipy.stats import norm as _norm


AR1_JOINT_COVERAGE = 0.95  # target joint coverage across all innovation constraints


def _ar1_kappa(num_constraints: int, joint_coverage: float = AR1_JOINT_COVERAGE) -> float:
    """Return per-innovation kappa giving joint_coverage across num_constraints i.i.d. innovations.

    per-step coverage = joint_coverage^(1/num_constraints),
    kappa = Phi^-1((1 + per_step_coverage) / 2).
    Valid only under independent, identically-distributed innovations.

    num_constraints is the exact count of constraints imposed. For the AR(1) tube that
    runs over time_steps_minus_1 = range(1, T), pass T-1 (not T). Passing T would
    undercount the constraints by one, making the Sidak exponent too small and kappa
    too tight, so empirical joint coverage would fall below the target.
    """
    per_step = joint_coverage ** (1.0 / num_constraints)
    return float(_norm.ppf((1.0 + per_step) / 2.0))


class PoASupportSet:
    # ------------------------------------------------------------------
    # Support set
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_support_set_demand(self) -> None:
        m = self.model

        kappa = 1.96  # marginal 95% CI for point-wise level bounds

        # AR(1) tube: constraints are imposed over time_steps_minus_1 = range(1, T),
        # which contains exactly T-1 elements. Pass T-1 so the Sidak exponent matches
        # the constraint count.
        num_ar1_constraints = self.num_time_steps - 1
        kappa_ar1 = _ar1_kappa(num_ar1_constraints)  # homogeneous kappa for t >= 2

        # t=1 kappa (option a: explicit non-stationary variance).
        # The generator initializes residual[0] = sigma_D * z[0] (a single innovation,
        # not a draw from the stationary distribution). At t=1 the whitened increment is
        #   D_1 - rho*D_0 - ar1_ref_1
        #   = D_ref * (residual[1] - rho * residual[0])
        #   = D_ref * (rho*residual[0] + sigma_D*z[1] - rho*residual[0])
        #   = D_ref * sigma_D * z[1]
        # The rho*residual[0] terms cancel exactly regardless of how residual[0] was drawn,
        # so Var = (D_ref * sigma_D)^2 at t=1, identical to t >= 2. Hence kappa_ar1_t1 ==
        # kappa_ar1. The two-variable structure is kept so the logic is auditable.
        kappa_ar1_t1 = _ar1_kappa(num_ar1_constraints)  # equals kappa_ar1 for AR(1)

        # Level box uses stationary sigma = sigma_D / sqrt(1 - rho^2).
        m.stationary_std_dev_D = Expression(
            rule=lambda m: m.sigma_D / (np.sqrt(1 - (self.demand_rho_fixed)**2))
        )

        m.demand_reference = Expression(
            m.time_steps,
            rule=lambda m, t: self.demand_D_ref * m.mu_D * self.demand_shape[int(t)],
        )
        # Level box: stationary sigma (sigma_D / sqrt(1 - rho^2)), not the innovation sigma.
        m.demand_lower = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t] - kappa * self.demand_D_ref * m.stationary_std_dev_D
        )
        m.demand_upper = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t] + kappa * self.demand_D_ref * m.stationary_std_dev_D
        )

        m.demand_budget_expr = Expression(
            rule=lambda _: self.ambiguity_kappa * self.num_time_steps * self.demand_D_ref
        )

        def demand_lower_rule(m, t):
            return m.D[t] >= m.demand_lower[t]

        def demand_upper_rule(m, t):
            return m.D[t] <= m.demand_upper[t]

        # AR(1) tube: bounds D[t] - rho*D[t-1] - ar1_ref_t within +/- kappa_t * D_ref * sigma_D.
        # Uses innovation sigma (sigma_D), NOT the stationary sigma sigma_D/sqrt(1-rho^2).
        # Whitening identity: D[t] - rho*D[t-1] - ar1_ref_t = D_ref * sigma_D * z[t] exactly
        # when the max(..., 0) floor is inactive. For the configured regimes (mu_D ~ 0.73-0.87,
        # sigma_D ~ 0.012) the floor probability is < 1e-15 and has no practical effect.
        # t=1 uses kappa_ar1_t1, which equals kappa_ar1 (see derivation above).
        def demand_ar1_up_rule(m, t):
            kappa_t = kappa_ar1_t1 if t == 1 else kappa_ar1
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[t] - self.demand_rho_fixed * m.D[t - 1]
                <= ar1_ref + kappa_t * self.demand_D_ref * m.sigma_D  # innovation sigma
            )

        def demand_ar1_down_rule(m, t):
            kappa_t = kappa_ar1_t1 if t == 1 else kappa_ar1
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[t] - self.demand_rho_fixed * m.D[t - 1]
                >= ar1_ref - kappa_t * self.demand_D_ref * m.sigma_D  # innovation sigma
            )

        def demand_abs_deviation_pos_rule(m, t):
            return m.D_abs_deviation[t] >= m.D[t] - m.demand_reference[t]

        def demand_abs_deviation_neg_rule(m, t):
            return m.D_abs_deviation[t] >= m.demand_reference[t] - m.D[t]

        def demand_budget_rule(m):
            return sum(m.D_abs_deviation[t] for t in m.time_steps) <= m.demand_budget_expr

        def demand_feasibility_rule(m, t):
            return m.demand_lower[t] >= 0

        self.model.demand_lower_bound_constraints = Constraint(self.model.time_steps, rule=demand_lower_rule)
        self.model.demand_upper_bound_constraints = Constraint(self.model.time_steps, rule=demand_upper_rule)
        self.model.demand_ar1_up_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ar1_up_rule)
        self.model.demand_ar1_down_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ar1_down_rule)
        self.model.demand_abs_deviation_pos_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_pos_rule)
        self.model.demand_abs_deviation_neg_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_neg_rule)
        self.model.demand_budget_constraint = Constraint(rule=demand_budget_rule)
        self.model.demand_lower_feasibility = Constraint(self.model.time_steps, rule=demand_feasibility_rule)

    def _build_support_set_wind(self) -> None:
        m = self.model

        kappa = 1.96

        # AR(1) tube: T-1 constraints over time_steps_minus_1.
        num_ar1_constraints = self.num_time_steps - 1
        kappa_ar1 = _ar1_kappa(num_ar1_constraints)

        # t=1 kappa: same cancellation argument as demand — P_1 - rho*P_0 - ar1_ref_1
        # = cap * sigma_W * z[1] regardless of initialization, so kappa_ar1_t1 == kappa_ar1.
        kappa_ar1_t1 = _ar1_kappa(num_ar1_constraints)  # equals kappa_ar1 for AR(1)

        m.wind_reference = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: self.static_physical_capacity[int(i)]
            * m.mu_W
            * self.wind_shape[int(t)],
        )

        # Level box uses stationary sigma = sigma_W / sqrt(1 - rho_W^2).
        m.stationary_std_dev_W = Expression(
            rule=lambda m: m.sigma_W / (np.sqrt(1 - (self.wind_rho_fixed)**2))
        )

        # Level box: stationary sigma (sigma_W / sqrt(1 - rho_W^2)), not the innovation sigma.
        m.wind_lower = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t]
            - kappa * self.static_physical_capacity[int(i)] * m.stationary_std_dev_W,
        )

        m.wind_upper = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t]
            + kappa * self.static_physical_capacity[int(i)] * m.stationary_std_dev_W,
        )

        m.wind_budget_expr = Expression(
            m.wind_physical_generators,
            rule=lambda _, i: self.ambiguity_kappa * self.num_time_steps * self.static_physical_capacity[int(i)]
        )

        def conventional_capacity_rule(m, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.P_max_block[i, b, t] == self.static_block_capacity[global_block]

        def wind_total_lower_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                >= m.wind_lower[i, t]
            )

        def wind_total_upper_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= m.wind_upper[i, t]
            )

        def wind_even_block_split_rule(m, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return (
                len(local_blocks) * m.P_max_block[i, b, t]
                == sum(m.P_max_block[i, other_b, t] for other_b in local_blocks)
            )

        # AR(1) tube for wind: bounds P[t] - rho*P[t-1] - ar1_ref_t within
        # +/- kappa_t * cap * sigma_W.
        # Uses innovation sigma (sigma_W), NOT the stationary sigma.
        # Whitening identity: P[t] - rho*P[t-1] - ar1_ref_t = cap * sigma_W * z[t] exactly
        # when clip(..., 0, 1) is inactive. Unlike demand, the upper clip (wind_factor > 1)
        # can activate at peak timesteps with ~0.5% probability (mu_W*shape_max ~0.805,
        # stationary std ~0.076, so P(Z > 2.57) ~ 0.005). When it does, the whitened
        # residual is smaller in magnitude than the pure innovation would be, meaning the
        # tube constraint is more easily satisfied — the clipped trajectories do not escape
        # the tube. Empirical coverage will be >= 0.95, not below.
        # See validate_tube_coverage.py for empirical assessment.
        def wind_ar1_up_rule(m, i, t):
            kappa_t = kappa_ar1_t1 if t == 1 else kappa_ar1
            P_t = sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return (
                P_t - self.wind_rho_fixed * P_t1
                <= ar1_ref + kappa_t * self.static_physical_capacity[int(i)] * m.sigma_W  # innovation sigma
            )

        def wind_ar1_down_rule(m, i, t):
            kappa_t = kappa_ar1_t1 if t == 1 else kappa_ar1
            P_t = sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return (
                P_t - self.wind_rho_fixed * P_t1
                >= ar1_ref - kappa_t * self.static_physical_capacity[int(i)] * m.sigma_W  # innovation sigma
            )

        def wind_abs_deviation_pos_rule(m, i, t):
            return (
                m.P_max_phys_abs_deviation[i, t]
                >= sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - m.wind_reference[i, t]
            )

        def wind_abs_deviation_neg_rule(m, i, t):
            return (
                m.P_max_phys_abs_deviation[i, t]
                >= m.wind_reference[i, t]
                - sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            )

        def wind_budget_rule(m, i):
            return sum(
                m.P_max_phys_abs_deviation[i, t] for t in m.time_steps
            ) <= m.wind_budget_expr[i]

        def wind_capacity_factor_lower_feasibility_rule(m, i, t):
            return m.wind_reference[i, t] >= 0

        def wind_capacity_factor_upper_feasibility_rule(m, i, t):
            return m.wind_reference[i, t] <= self.static_physical_capacity[int(i)]

        self.model.conventional_capacity = Constraint(self.model.conventional_blocks, self.model.time_steps, rule=conventional_capacity_rule)
        self.model.wind_total_lower_bound = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_total_lower_rule)
        self.model.wind_total_upper_bound = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_total_upper_rule)
        self.model.wind_even_block_split = Constraint(self.model.wind_blocks, self.model.time_steps, rule=wind_even_block_split_rule)
        self.model.wind_ar1_up_constraints = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ar1_up_rule)
        self.model.wind_ar1_down_constraints = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ar1_down_rule)
        self.model.wind_abs_deviation_pos = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_pos_rule)
        self.model.wind_abs_deviation_neg = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_neg_rule)
        self.model.wind_budget_constraint = Constraint(self.model.wind_physical_generators, rule=wind_budget_rule)
        self.model.wind_capacity_factor_lower_feasibility = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_capacity_factor_lower_feasibility_rule)
        self.model.wind_capacity_factor_upper_feasibility = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_capacity_factor_upper_feasibility_rule)


PoASupportSet = PoASupportSet
