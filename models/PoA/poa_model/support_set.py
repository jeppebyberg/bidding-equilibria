import numpy as np
from pyomo.environ import Constraint, Expression
from scipy.stats import norm as _norm


AR1_JOINT_COVERAGE = 0.95  # target joint coverage across all T innovations


def _ar1_kappa(num_time_steps: int, joint_coverage: float = AR1_JOINT_COVERAGE) -> float:
    """Return the per-innovation kappa giving joint_coverage across num_time_steps i.i.d. innovations."""
    return float(_norm.ppf((1.0 + joint_coverage ** (1.0 / num_time_steps)) / 2.0))


class PoASupportSet:
    # ------------------------------------------------------------------
    # Support set
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_support_set_demand(self) -> None:
        m = self.model

        kappa = 1.96  # marginal 95 % CI for point-wise level bounds
        kappa_ar1 = _ar1_kappa(self.num_time_steps)  # joint coverage across T innovations

        m.stationary_std_dev_D = Expression(
            rule=lambda m: m.sigma_D / (np.sqrt(1 - (self.demand_rho_fixed)**2))
        )

        m.demand_reference = Expression(
            m.time_steps,
            rule=lambda m, t: self.demand_D_ref * m.mu_D * self.demand_shape[int(t)],
        )
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

        # AR(1) innovation constraints: bound D[t] - rho*D[t-1] around the
        # deterministic AR(1) trend by kappa_ar1 * sigma_D, ensuring the i.i.d.
        # innovation at each step satisfies the joint coverage target.
        def demand_ar1_up_rule(m, t):
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[t] - self.demand_rho_fixed * m.D[t - 1]
                <= ar1_ref + kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_down_rule(m, t):
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[t] - self.demand_rho_fixed * m.D[t - 1]
                >= ar1_ref - kappa_ar1 * self.demand_D_ref * m.sigma_D
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
        kappa_ar1 = _ar1_kappa(self.num_time_steps)

        m.wind_reference = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: self.static_physical_capacity[int(i)]
            * m.mu_W
            * self.wind_shape[int(t)],
        )

        m.stationary_std_dev_W = Expression(
            rule=lambda m: m.sigma_W / (np.sqrt(1 - (self.wind_rho_fixed)**2))
        )

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

        # AR(1) innovation constraints for wind capacity.
        def wind_ar1_up_rule(m, i, t):
            P_t = sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return (
                P_t - self.wind_rho_fixed * P_t1
                <= ar1_ref + kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        def wind_ar1_down_rule(m, i, t):
            P_t = sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return (
                P_t - self.wind_rho_fixed * P_t1
                >= ar1_ref - kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
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
