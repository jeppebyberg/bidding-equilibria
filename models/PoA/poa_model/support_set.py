# NOTE ON COVERAGE SCOPE: the 95% joint-coverage claim applies to all T AR(1) innovation
# constraints (t=0 through T-1). The same Sidak-corrected kappa is used for both the
# AR(1) tube and the level box so that a single coverage statement covers the full
# support set.
#
# Why 95% here vs 99% in DROWassersteinSupportSet:
# In the DRO inner problem, scenarios are explicitly drawn and must land within U(r);
# the 99% coverage is calibrated from out-of-sample analysis to ensure that.
# In the PoA upper level, no scenarios are drawn — the optimizer analytically chooses
# the worst-case trajectory inside U(r).  95% is therefore the right threshold for
# "plausible market state" without over-constraining the worst-case search.
import numpy as np
from pyomo.environ import Constraint, Expression
from scipy.stats import norm as _norm


AR1_JOINT_COVERAGE = 0.95  # joint coverage across all T innovation constraints


def _ar1_kappa(num_constraints: int, joint_coverage: float = AR1_JOINT_COVERAGE) -> float:
    """Return per-innovation kappa giving joint_coverage across num_constraints i.i.d. innovations.

    per-step coverage = joint_coverage^(1/num_constraints),
    kappa = Phi^-1((1 + per_step_coverage) / 2).
    Valid only under independent, identically-distributed innovations.

    Pass the exact number of innovation constraints imposed (T for both the t=0 cold-start
    and the t=1..T-1 tube, so that the Sidak exponent matches the constraint count).
    """
    per_step = joint_coverage ** (1.0 / num_constraints)
    return float(_norm.ppf((1.0 + per_step) / 2.0))


def _level_scale(rho: float, t: int) -> float:
    """Time-varying level-band scale factor (multiples of innovation sigma sigma).

    The cumulative AR(1) process starting from innovation sigma at t=0 can reach
    at most sum_{j=0}^{t} rho^j = (1 - rho^{t+1}) / (1 - rho) standard deviations
    from the reference. The stationary bound is 1/sqrt(1-rho^2). The effective
    level-band half-width at time t is the minimum of the two:

        scale(t) = min( (1-rho^{t+1})/(1-rho),  1/sqrt(1-rho^2) )

    Because rho is fixed in the PoA model, scale(t) is a plain Python float and
    keeps the level-band constraints linear in sigma_D / sigma_W.
    """
    cumulative = (1.0 - rho ** (t + 1)) / (1.0 - rho)
    stationary = 1.0 / np.sqrt(1.0 - rho ** 2)
    return min(cumulative, stationary)


class PoASupportSet:
    # ------------------------------------------------------------------
    # Support set
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_support_set_demand(self) -> None:
        m = self.model

        # One Sidak-corrected kappa for all T innovation constraints (t=0 cold-start +
        # t=1..T-1 AR(1) tube) and for the level box.  Matches DRO formulation exactly.
        kappa = _ar1_kappa(self.num_time_steps)  # joint 99 % over T constraints

        # Precompute time-varying scale factors (pure Python floats; rho is fixed).
        # scale(t) = min((1-rho^{t+1})/(1-rho), 1/sqrt(1-rho^2))
        # Grows from 1.0 at t=0 to the stationary value and stays there.
        demand_scales = [_level_scale(self.demand_rho_fixed, t) for t in range(self.num_time_steps)]

        m.demand_reference = Expression(
            m.time_steps,
            rule=lambda m, t: self.demand_D_ref * m.mu_D * self.demand_shape[int(t)],
        )
        # Time-varying level box: half-width = kappa * D_ref * sigma_D * scale(t).
        # At t=0 scale=1 (innovation sigma); grows to stationary sigma by t~2 for rho~0.75.
        m.demand_lower = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t]
                - kappa * self.demand_D_ref * m.sigma_D * demand_scales[int(t)]
        )
        m.demand_upper = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t]
                + kappa * self.demand_D_ref * m.sigma_D * demand_scales[int(t)]
        )

        m.demand_budget_expr = Expression(
            rule=lambda _: self.ambiguity_kappa * sum(m.demand_reference[t] for t in m.time_steps)
        )

        def demand_lower_rule(m, t):
            return m.D[t] >= m.demand_lower[t]

        def demand_upper_rule(m, t):
            return m.D[t] <= m.demand_upper[t]

        # AR(1) tube: bounds D[t] - rho*D[t-1] - ar1_ref_t within +/- kappa * D_ref * sigma_D.
        # Uses innovation sigma (sigma_D), NOT the stationary sigma sigma_D/sqrt(1-rho^2).
        # t=0 cold-start: D[0] ~ D_ref*mu_D*shape[0] + D_ref*sigma_D*z[0], so the whitened
        # deviation is simply D_ref*sigma_D*z[0] — same innovation sigma as t>=1.
        def demand_ar1_t0_up_rule(m):
            return (
                m.D[0] - self.demand_D_ref * m.mu_D * self.demand_shape[0]
                <= kappa * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_t0_down_rule(m):
            return (
                m.D[0] - self.demand_D_ref * m.mu_D * self.demand_shape[0]
                >= -kappa * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_up_rule(m, t):
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[t] - self.demand_rho_fixed * m.D[t - 1]
                <= ar1_ref + kappa * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_down_rule(m, t):
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[t] - self.demand_rho_fixed * m.D[t - 1]
                >= ar1_ref - kappa * self.demand_D_ref * m.sigma_D
            )

        def demand_abs_deviation_pos_rule(m, t):
            return m.D_abs_deviation[t] >= m.D[t] - m.demand_reference[t]

        def demand_abs_deviation_neg_rule(m, t):
            return m.D_abs_deviation[t] >= m.demand_reference[t] - m.D[t]

        def demand_budget_rule(m):
            return sum(m.D_abs_deviation[t] for t in m.time_steps) <= m.demand_budget_expr

        def demand_feasibility_rule(m, t):
            return m.demand_lower[t] >= 0

        self.model.demand_ar1_t0_up = Constraint(rule=demand_ar1_t0_up_rule)
        self.model.demand_ar1_t0_down = Constraint(rule=demand_ar1_t0_down_rule)
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

        # One Sidak-corrected kappa for all T innovation constraints and the level box.
        kappa = _ar1_kappa(self.num_time_steps)  # joint 99 % over T constraints

        m.wind_reference = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: self.static_physical_capacity[int(i)]
            * m.mu_W
            * self.wind_shape[int(t)],
        )

        # Precompute time-varying scale factors for wind (same formula, wind rho).
        wind_scales = [_level_scale(self.wind_rho_fixed, t) for t in range(self.num_time_steps)]

        # Time-varying level box for wind.
        m.wind_lower = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t]
                - kappa * self.static_physical_capacity[int(i)] * m.sigma_W * wind_scales[int(t)],
        )

        m.wind_upper = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t]
                + kappa * self.static_physical_capacity[int(i)] * m.sigma_W * wind_scales[int(t)],
        )

        m.wind_budget_expr = Expression(
            m.wind_physical_generators,
            rule=lambda _, i: self.ambiguity_kappa * sum(m.wind_reference[i, t] for t in m.time_steps)
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

        # Physical cap on the actual trajectory variable: P_max[i,t] <= cap_i.
        # This matches the DRO model's wind_physical_upper and ensures no realised
        # trajectory exceeds installed capacity, without restricting which regimes
        # (mu_W, sigma_W) the upper level may select.
        def wind_physical_upper_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= self.static_physical_capacity[int(i)]
            )

        def wind_even_block_split_rule(m, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return (
                len(local_blocks) * m.P_max_block[i, b, t]
                == sum(m.P_max_block[i, other_b, t] for other_b in local_blocks)
            )

        # AR(1) tube for wind: bounds P[t] - rho*P[t-1] - ar1_ref_t within +/- kappa * cap * sigma_W.
        # t=0 cold-start: P[0] ~ cap*mu_W*shape[0] + cap*sigma_W*z[0], innovation sigma only.
        # Uses innovation sigma (sigma_W), NOT the stationary sigma.
        def wind_ar1_t0_up_rule(m, i):
            ref0 = self.static_physical_capacity[int(i)] * m.mu_W * self.wind_shape[0]
            P_t0 = sum(m.P_max_block[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
            return P_t0 - ref0 <= kappa * self.static_physical_capacity[int(i)] * m.sigma_W

        def wind_ar1_t0_down_rule(m, i):
            ref0 = self.static_physical_capacity[int(i)] * m.mu_W * self.wind_shape[0]
            P_t0 = sum(m.P_max_block[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
            return P_t0 - ref0 >= -kappa * self.static_physical_capacity[int(i)] * m.sigma_W

        def wind_ar1_up_rule(m, i, t):
            P_t = sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return P_t - self.wind_rho_fixed * P_t1 <= ar1_ref + kappa * self.static_physical_capacity[int(i)] * m.sigma_W

        def wind_ar1_down_rule(m, i, t):
            P_t = sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return P_t - self.wind_rho_fixed * P_t1 >= ar1_ref - kappa * self.static_physical_capacity[int(i)] * m.sigma_W

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
        self.model.wind_ar1_t0_up = Constraint(self.model.wind_physical_generators, rule=wind_ar1_t0_up_rule)
        self.model.wind_ar1_t0_down = Constraint(self.model.wind_physical_generators, rule=wind_ar1_t0_down_rule)
        self.model.wind_ar1_up_constraints = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ar1_up_rule)
        self.model.wind_ar1_down_constraints = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ar1_down_rule)
        self.model.wind_abs_deviation_pos = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_pos_rule)
        self.model.wind_abs_deviation_neg = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_neg_rule)
        self.model.wind_budget_constraint = Constraint(self.model.wind_physical_generators, rule=wind_budget_rule)
        self.model.wind_capacity_factor_lower_feasibility = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_capacity_factor_lower_feasibility_rule)
        self.model.wind_capacity_factor_upper_feasibility = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_capacity_factor_upper_feasibility_rule)
        self.model.wind_physical_upper = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_physical_upper_rule)

PoASupportSet = PoASupportSet
