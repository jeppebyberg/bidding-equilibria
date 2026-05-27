import numpy as np
from pyomo.environ import Constraint, Expression


class PoASupportSet:
    # ------------------------------------------------------------------
    # Support set
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_support_set_demand(self) -> None:
        m = self.model
        m.demand_reference = Expression(
            m.time_steps,
            rule=lambda m, t: self.demand_D_ref * m.mu_D * self.demand_shape[int(t)],
        )

        kappa = 1.96 # Corresponds to a 95% confidence interval under normality assumptions

        m.stationary_std_dev_D = Expression(
            rule=lambda m: m.sigma_D / (np.sqrt(1 - (self.demand_rho_fixed)**2))
        )

        m.stationary_residual_D = Expression(
            rule=lambda m: m.sigma_D * np.sqrt(2 / (1 + self.demand_rho_fixed))
        )

        m.demand_lower = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t] - kappa * self.demand_D_ref * m.stationary_std_dev_D
        )
        m.demand_upper = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t] + kappa * self.demand_D_ref * m.stationary_std_dev_D
        )

        m.demand_ramp = Expression(
            m.time_steps_minus_1,
            rule=lambda m, t: 
                self.demand_D_ref * m.mu_D * self.demand_delta_shape[int(t)]
                + kappa * self.demand_D_ref * m.stationary_residual_D
            )

        m.demand_budget_expr = Expression(
            rule=lambda m: self.ambiguity_kappa * self.num_time_steps * kappa * self.demand_D_ref * m.stationary_std_dev_D
        )

        def demand_lower_rule(m, t):
            return m.D[t] >= m.demand_lower[t]

        def demand_upper_rule(m, t):
            return m.D[t] <= m.demand_upper[t]

        def demand_ramp_up_rule(m, t):
            return m.D[t] - m.D[t - 1] <= m.demand_ramp[t]

        def demand_ramp_down_rule(m, t):
            return m.D[t] - m.D[t - 1] >= -m.demand_ramp[t]
        
        # Budget constraints 
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
        self.model.demand_ramp_up_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_up_rule)
        self.model.demand_ramp_down_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_down_rule)
        self.model.demand_abs_deviation_pos_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_pos_rule)
        self.model.demand_abs_deviation_neg_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_neg_rule)
        self.model.demand_budget_constraint = Constraint(rule=demand_budget_rule)
        self.model.demand_lower_feasibility = Constraint(self.model.time_steps, rule=demand_feasibility_rule)

    def _build_support_set_wind(self) -> None:
        m = self.model
        m.wind_reference = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: self.static_physical_capacity[int(i)]
            * m.mu_W
            * self.wind_shape[int(t)],
        )

        kappa = 1.96 # Corresponds to a 95% confidence interval under normality assumptions

        m.stationary_std_dev_W = Expression(
            rule=lambda m: m.sigma_W / (np.sqrt(1 - (self.wind_rho_fixed)**2))
        )

        m.stationary_residual_W = Expression(
            rule=lambda m: m.sigma_W * np.sqrt(2 / (1 + self.wind_rho_fixed))
        )

        m.wind_lower = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t] - kappa * self.static_physical_capacity[int(i)] * m.stationary_std_dev_W,
        )

        m.wind_upper = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t] + kappa * self.static_physical_capacity[int(i)] * m.stationary_std_dev_W,
        )

        m.wind_ramp = Expression(
            m.wind_physical_generators,
            m.time_steps_minus_1,
            rule=lambda m, i, t: 
            self.static_physical_capacity[int(i)] * m.mu_W * self.wind_delta_shape[int(t)] + kappa * self.static_physical_capacity[int(i)] * m.stationary_residual_W
            )

        m.wind_budget_expr = Expression(
            m.wind_physical_generators,
            rule=lambda m, i: self.ambiguity_kappa * self.num_time_steps * kappa * self.static_physical_capacity[int(i)] * m.stationary_std_dev_W
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

        def wind_ramp_up_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)]) 
              - sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                <= m.wind_ramp[i, t])

        def wind_ramp_down_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= m.wind_ramp[i, t]
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
        self.model.wind_ramp_up = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ramp_up_rule)
        self.model.wind_ramp_down = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ramp_down_rule)
        self.model.wind_abs_deviation_pos = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_pos_rule)
        self.model.wind_abs_deviation_neg = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_neg_rule)
        self.model.wind_budget_constraint = Constraint(self.model.wind_physical_generators, rule=wind_budget_rule)
        self.model.wind_capacity_factor_lower_feasibility = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_capacity_factor_lower_feasibility_rule)
        self.model.wind_capacity_factor_upper_feasibility = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_capacity_factor_upper_feasibility_rule)


PoASupportSet = PoASupportSet
