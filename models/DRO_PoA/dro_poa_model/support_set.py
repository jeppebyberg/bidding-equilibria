from __future__ import annotations

import numpy as np
from pyomo.environ import Constraint, Expression
from scipy.stats import norm as _norm


AR1_JOINT_COVERAGE = 0.95  # target joint coverage across all T innovations


def _ar1_kappa(num_time_steps: int, joint_coverage: float = AR1_JOINT_COVERAGE) -> float:
    """Sidak-corrected kappa giving joint_coverage across num_time_steps i.i.d. innovations.

    Each per-step band is Phi^{-1}((1 + coverage^{1/T}) / 2) standard deviations wide,
    so T independent Gaussian innovations each fall inside with probability coverage^{1/T},
    giving joint coverage exactly coverage.
    """
    return float(_norm.ppf((1.0 + joint_coverage ** (1.0 / num_time_steps)) / 2.0))


class DROPoASupportSet:
    def _build_support_set(self) -> None:
        self._build_regime_fixing_constraints()
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_regime_fixing_constraints(self) -> None:
        m = self.model
        m.regime_mu_D_fixed = Constraint(expr=m.mu_D == self.mu_D_fixed)
        m.regime_sigma_D_fixed = Constraint(expr=m.sigma_D == self.sigma_D_fixed)
        m.regime_mu_W_fixed = Constraint(expr=m.mu_W == self.mu_W_fixed)
        m.regime_sigma_W_fixed = Constraint(expr=m.sigma_W == self.sigma_W_fixed)
        m.regime_rho_D_fixed = Constraint(expr=m.rho_D == self.demand_rho_fixed)
        m.regime_rho_W_fixed = Constraint(expr=m.rho_W == self.wind_rho_fixed)
        m.regime_peak_W_fixed = Constraint(expr=m.peak_W == self.peak_W_fixed)

    def _build_support_set_demand(self) -> None:
        m = self.model

        kappa = 1.96  # marginal 95 % CI for point-wise level bounds
        kappa_ar1 = _ar1_kappa(self.num_time_steps)  # joint coverage across T innovations

        m.demand_reference = Expression(
            m.time_steps,
            rule=lambda m, t: self.demand_D_ref * m.mu_D * self.demand_shape[int(t)],
        )
        m.stationary_std_dev_D = Expression(
            rule=lambda m: m.sigma_D / np.sqrt(1.0 - self.demand_rho_fixed ** 2)
        )
        m.demand_lower = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t]
            - kappa * self.demand_D_ref * m.stationary_std_dev_D,
        )
        m.demand_upper = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t]
            + kappa * self.demand_D_ref * m.stationary_std_dev_D,
        )
        m.demand_budget_expr = Expression(
            rule=lambda _: self.ambiguity_kappa * self.num_time_steps * self.demand_D_ref,
        )

        def demand_lower_rule(m, k, t):
            return m.D[k, t] >= m.demand_lower[t]

        def demand_upper_rule(m, k, t):
            return m.D[k, t] <= m.demand_upper[t]

        # AR(1) innovation constraints: bound D[k,t] - rho*D[k,t-1] around the
        # deterministic AR(1) trend by kappa_ar1 * sigma_D.
        def demand_ar1_up_rule(m, k, t):
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[k, t] - self.demand_rho_fixed * m.D[k, t - 1]
                <= ar1_ref + kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_down_rule(m, k, t):
            ar1_ref = (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[int(t)] - self.demand_rho_fixed * self.demand_shape[int(t) - 1])
            )
            return (
                m.D[k, t] - self.demand_rho_fixed * m.D[k, t - 1]
                >= ar1_ref - kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        def demand_abs_deviation_pos_rule(m, k, t):
            return m.D_abs_deviation[k, t] >= m.D[k, t] - m.demand_reference[t]

        def demand_abs_deviation_neg_rule(m, k, t):
            return m.D_abs_deviation[k, t] >= m.demand_reference[t] - m.D[k, t]

        def demand_budget_rule(m, k):
            return (
                sum(m.D_abs_deviation[k, t] for t in m.time_steps)
                <= m.demand_budget_expr
            )

        m.demand_lower_bound_constraints = Constraint(
            m.scenarios, m.time_steps, rule=demand_lower_rule
        )
        m.demand_upper_bound_constraints = Constraint(
            m.scenarios, m.time_steps, rule=demand_upper_rule
        )
        m.demand_ar1_up_constraints = Constraint(
            m.scenarios, m.time_steps_minus_1, rule=demand_ar1_up_rule
        )
        m.demand_ar1_down_constraints = Constraint(
            m.scenarios, m.time_steps_minus_1, rule=demand_ar1_down_rule
        )
        m.demand_abs_deviation_pos_constraints = Constraint(
            m.scenarios, m.time_steps, rule=demand_abs_deviation_pos_rule
        )
        m.demand_abs_deviation_neg_constraints = Constraint(
            m.scenarios, m.time_steps, rule=demand_abs_deviation_neg_rule
        )
        # m.demand_budget_constraint = Constraint(m.scenarios, rule=demand_budget_rule)

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
            rule=lambda m: m.sigma_W / np.sqrt(1.0 - self.wind_rho_fixed ** 2)
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
            rule=lambda _, i: self.ambiguity_kappa * self.num_time_steps * self.static_physical_capacity[int(i)],
        )

        def conventional_capacity_rule(m, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.P_max_block[k, i, b, t] == self.static_block_capacity[global_block]

        def wind_total_lower_rule(m, k, i, t):
            return (
                sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                >= m.wind_lower[i, t]
            )

        def wind_total_upper_rule(m, k, i, t):
            return (
                sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= m.wind_upper[i, t]
            )

        def wind_even_block_split_rule(m, k, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return len(local_blocks) * m.P_max_block[k, i, b, t] == sum(
                m.P_max_block[k, i, other_b, t] for other_b in local_blocks
            )

        # AR(1) innovation constraints for wind capacity.
        def wind_ar1_up_rule(m, k, i, t):
            P_t = sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return (
                P_t - self.wind_rho_fixed * P_t1
                <= ar1_ref + kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        def wind_ar1_down_rule(m, k, i, t):
            P_t = sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            P_t1 = sum(m.P_max_block[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
            ar1_ref = (
                self.static_physical_capacity[int(i)] * m.mu_W
                * (self.wind_shape[int(t)] - self.wind_rho_fixed * self.wind_shape[int(t) - 1])
            )
            return (
                P_t - self.wind_rho_fixed * P_t1
                >= ar1_ref - kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        def wind_abs_deviation_pos_rule(m, k, i, t):
            return m.P_max_phys_abs_deviation[k, i, t] >= sum(
                m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)]
            ) - m.wind_reference[i, t]

        def wind_abs_deviation_neg_rule(m, k, i, t):
            return m.P_max_phys_abs_deviation[k, i, t] >= m.wind_reference[i, t] - sum(
                m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)]
            )

        def wind_budget_rule(m, k, i):
            return (
                sum(m.P_max_phys_abs_deviation[k, i, t] for t in m.time_steps)
                <= m.wind_budget_expr[i]
            )

        def dispatch_capacity_feasibility_rule(m, k, t):
            return m.D[k, t] <= sum(
                m.P_max_block[k, i, b, t] for i, b in m.generator_blocks
            )

        m.conventional_capacity = Constraint(
            m.scenarios, m.conventional_blocks, m.time_steps, rule=conventional_capacity_rule
        )
        m.wind_total_lower_bound = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_total_lower_rule
        )
        m.wind_total_upper_bound = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_total_upper_rule
        )
        m.wind_even_block_split = Constraint(
            m.scenarios, m.wind_blocks, m.time_steps, rule=wind_even_block_split_rule
        )
        m.wind_ar1_up_constraints = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps_minus_1, rule=wind_ar1_up_rule
        )
        m.wind_ar1_down_constraints = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps_minus_1, rule=wind_ar1_down_rule
        )
        m.wind_abs_deviation_pos = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_abs_deviation_pos_rule
        )
        m.wind_abs_deviation_neg = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_abs_deviation_neg_rule
        )
        # m.wind_budget_constraint = Constraint(
        #     m.scenarios, m.wind_physical_generators, rule=wind_budget_rule
        # )
        m.dispatch_capacity_feasibility = Constraint(
            m.scenarios, m.time_steps, rule=dispatch_capacity_feasibility_rule
        )


class DROWassersteinSupportSet(DROPoASupportSet):
    """Support set U for the Wasserstein DRO inner problem.

    Retains only the constraints that belong to U — not the transport cost:
      - Regime fixing (parameters are given, not decision variables).
      - AR(1) innovation tube for t >= 1: bounds the whitened increment
          D[t] - rho_D * D[t-1] around the deterministic reference increment
          D_ref * mu_D * (shape[t] - rho_D * shape[t-1])
        within +/- kappa_ar1 * D_ref * sigma_D.  Analogue for wind.
      - t=0 innovation band: the generator initialises the residual at t=0
        with the innovation std (cold start, NOT stationary std), so the
        t=0 constraint is
          |D[0] - D_ref * mu_D * shape[0]| <= kappa_ar1 * D_ref * sigma_D
        with the same kappa_ar1 as interior steps.
      - Physical limits: wind capacity in [0, P_W_max], dispatch feasibility
          D[t] <= total available capacity.

    Dropped relative to DROPoASupportSet:
      - Pointwise level box (demand_lower/upper, wind_lower/upper): its job
        is now done by the transport cost term eta * c(s, s_hat).
      - Deviation budget (demand_budget, wind_budget): equivalent to an L1
        ball, which the Wasserstein cost already penalises.

    Coverage and kappa
    ------------------
    AR1_JOINT_COVERAGE is the target joint probability that ALL T innovations
    simultaneously fall inside their bands.  Set it high enough (>= 0.99 for
    most cases) so that zero generated draws are rejected.  The instance
    attribute ``ar1_coverage`` overrides the class default, e.g.::

        optimizer.ar1_coverage = 0.999
        optimizer._build_support_set()

    To use this class instead of DROPoASupportSet, replace DROPoASupportSet
    in DRO_PoAOptimization's base-class list.
    """

    AR1_JOINT_COVERAGE: float = 0.99
    LEVEL_JOINT_COVERAGE: float = 0.9999  # calibrated: first coverage giving 0% rejection on fresh draws
    FLEET_JOINT_COVERAGE: float = 0.999  # calibrated: first coverage giving 0% fleet rejection

    def _build_support_set(self) -> None:
        # _build_regime_fixing_constraints is inherited from DROPoASupportSet and is
        # accessible via self regardless of the concrete class.
        # _build_wasserstein_demand/_wind are called explicitly so this method works
        # correctly when invoked on a DRO_PoAOptimization instance (which does NOT
        # inherit from DROWassersteinSupportSet and would fail a self.method() lookup).
        self._build_regime_fixing_constraints()
        DROWassersteinSupportSet._build_wasserstein_demand(self)
        DROWassersteinSupportSet._build_wasserstein_wind(self)

    # ------------------------------------------------------------------
    # Demand
    # ------------------------------------------------------------------

    def _build_wasserstein_demand(self) -> None:
        m = self.model
        # ar1_coverage can be overridden on the instance (e.g. from DRO_PoAOptimization).
        # Fall back to the class constant, accessed via the class directly so that
        # this method works correctly when called on a non-DROWassersteinSupportSet instance.
        coverage = float(
            getattr(self, "ar1_coverage", None)
            or DROWassersteinSupportSet.AR1_JOINT_COVERAGE
        )
        kappa_ar1 = _ar1_kappa(self.num_time_steps, coverage)
        # level/fleet fallback to the single ar1_coverage so one calibrated κ drives all bands.
        level_coverage = float(getattr(self, "level_coverage", None) or coverage)
        kappa_lvl_D = _ar1_kappa(self.num_time_steps, level_coverage)
        # 1/sqrt(1-rho_D^2): multiply by m.sigma_D to get stationary std coefficient
        sigma_bar_D_scale = 1.0 / np.sqrt(1.0 - self.demand_rho_fixed ** 2)

        # Reference increment shared by both t=0 and t>=1 rules.
        # mu_D is a fixed Pyomo Var (pinned by regime_fixing); using it keeps
        # the expression symbolic so tightening can inspect it.
        def _ar1_ref_D(m, t: int) -> object:
            """D_ref * mu_D * (shape[t] - rho_D * shape[t-1])  for t >= 1."""
            return (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[t] - self.demand_rho_fixed * self.demand_shape[t - 1])
            )

        # t = 0: non-stationary cold start.
        # The generator draws residual[0] ~ N(0, sigma_D^2) (innovation std,
        # not stationary std), so the band is the same width as interior steps.
        def demand_ar1_t0_up_rule(m, k):
            return (
                m.D[k, 0] - self.demand_D_ref * m.mu_D * self.demand_shape[0]
                <= kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_t0_down_rule(m, k):
            return (
                m.D[k, 0] - self.demand_D_ref * m.mu_D * self.demand_shape[0]
                >= -kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        # t >= 1: stationary AR(1) innovation tube (identical to DROPoASupportSet).
        def demand_ar1_up_rule(m, k, t):
            return (
                m.D[k, t] - self.demand_rho_fixed * m.D[k, t - 1]
                <= _ar1_ref_D(m, t) + kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        def demand_ar1_down_rule(m, k, t):
            return (
                m.D[k, t] - self.demand_rho_fixed * m.D[k, t - 1]
                >= _ar1_ref_D(m, t) - kappa_ar1 * self.demand_D_ref * m.sigma_D
            )

        # D[k,t] >= 0 is enforced by the NonNegativeReals domain on m.D.
        # Dispatch feasibility: demand cannot exceed total available capacity.
        def dispatch_feasibility_rule(m, k, t):
            return m.D[k, t] <= sum(
                m.P_max_block[k, i, b, t] for i, b in m.generator_blocks
            )

        # Stationary level band: caps accumulated drift from stacked innovations.
        def demand_level_lower_rule(m, k, t):
            ref = self.demand_D_ref * m.mu_D * self.demand_shape[int(t)]
            return m.D[k, t] >= ref - kappa_lvl_D * self.demand_D_ref * m.sigma_D * sigma_bar_D_scale

        def demand_level_upper_rule(m, k, t):
            ref = self.demand_D_ref * m.mu_D * self.demand_shape[int(t)]
            return m.D[k, t] <= ref + kappa_lvl_D * self.demand_D_ref * m.sigma_D * sigma_bar_D_scale

        m.demand_ar1_t0_up = Constraint(m.scenarios, rule=demand_ar1_t0_up_rule)
        m.demand_ar1_t0_down = Constraint(m.scenarios, rule=demand_ar1_t0_down_rule)
        m.demand_ar1_up_constraints = Constraint(
            m.scenarios, m.time_steps_minus_1, rule=demand_ar1_up_rule
        )
        m.demand_ar1_down_constraints = Constraint(
            m.scenarios, m.time_steps_minus_1, rule=demand_ar1_down_rule
        )
        m.dispatch_capacity_feasibility = Constraint(
            m.scenarios, m.time_steps, rule=dispatch_feasibility_rule
        )
        m.demand_level_lower_constraints = Constraint(
            m.scenarios, m.time_steps, rule=demand_level_lower_rule
        )
        m.demand_level_upper_constraints = Constraint(
            m.scenarios, m.time_steps, rule=demand_level_upper_rule
        )

    # ------------------------------------------------------------------
    # Wind
    # ------------------------------------------------------------------

    def _build_wasserstein_wind(self) -> None:
        m = self.model
        coverage = float(
            getattr(self, "ar1_coverage", None)
            or DROWassersteinSupportSet.AR1_JOINT_COVERAGE
        )
        kappa_ar1 = _ar1_kappa(self.num_time_steps, coverage)
        # level/fleet fallback to the single ar1_coverage so one calibrated κ drives all bands.
        level_coverage = float(getattr(self, "level_coverage", None) or coverage)
        fleet_coverage = float(getattr(self, "fleet_coverage", None) or coverage)
        kappa_lvl_W = _ar1_kappa(self.num_time_steps, level_coverage)
        kappa_fleet = _ar1_kappa(self.num_time_steps, fleet_coverage)
        sigma_bar_W_scale = 1.0 / np.sqrt(1.0 - self.wind_rho_fixed ** 2)
        cap_rms = float(np.sqrt(sum(
            self.static_physical_capacity[int(i)] ** 2 for i in m.wind_physical_generators
        )))
        sum_caps = float(sum(self.static_physical_capacity[int(i)] for i in m.wind_physical_generators))

        def _ar1_ref_W(m, i: int, t: int) -> object:
            """P_W_max[i] * mu_W * (shape[t] - rho_W * shape[t-1])  for t >= 1."""
            return (
                self.static_physical_capacity[i] * m.mu_W
                * (self.wind_shape[t] - self.wind_rho_fixed * self.wind_shape[t - 1])
            )

        def _P_total(m, k, i, t):
            return sum(
                m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)]
            )

        # Physical upper bound: replaces the stationary-band upper from base class.
        def wind_physical_upper_rule(m, k, i, t):
            return _P_total(m, k, i, t) <= self.static_physical_capacity[int(i)]

        # t = 0: non-stationary cold start (wind_residual[0] ~ N(0, sigma_W^2)).
        def wind_ar1_t0_up_rule(m, k, i):
            ref0 = self.static_physical_capacity[int(i)] * m.mu_W * self.wind_shape[0]
            return (
                _P_total(m, k, i, 0) - ref0
                <= kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        def wind_ar1_t0_down_rule(m, k, i):
            ref0 = self.static_physical_capacity[int(i)] * m.mu_W * self.wind_shape[0]
            return (
                _P_total(m, k, i, 0) - ref0
                >= -kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        # t >= 1: stationary AR(1) innovation tube (identical to DROPoASupportSet).
        def wind_ar1_up_rule(m, k, i, t):
            return (
                _P_total(m, k, i, t) - self.wind_rho_fixed * _P_total(m, k, i, t - 1)
                <= _ar1_ref_W(m, int(i), t) + kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        def wind_ar1_down_rule(m, k, i, t):
            return (
                _P_total(m, k, i, t) - self.wind_rho_fixed * _P_total(m, k, i, t - 1)
                >= _ar1_ref_W(m, int(i), t) - kappa_ar1 * self.static_physical_capacity[int(i)] * m.sigma_W
            )

        # Conventional generator capacity is scenario-independent.
        def conventional_capacity_rule(m, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.P_max_block[k, i, b, t] == self.static_block_capacity[global_block]

        # Wind blocks are split evenly across a physical generator's local blocks.
        def wind_even_block_split_rule(m, k, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return len(local_blocks) * m.P_max_block[k, i, b, t] == sum(
                m.P_max_block[k, i, other_b, t] for other_b in local_blocks
            )

        m.conventional_capacity = Constraint(
            m.scenarios, m.conventional_blocks, m.time_steps, rule=conventional_capacity_rule
        )
        m.wind_physical_upper = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_physical_upper_rule
        )
        m.wind_ar1_t0_up = Constraint(
            m.scenarios, m.wind_physical_generators, rule=wind_ar1_t0_up_rule
        )
        m.wind_ar1_t0_down = Constraint(
            m.scenarios, m.wind_physical_generators, rule=wind_ar1_t0_down_rule
        )
        # Per-generator stationary level band: coexists with physical cap and innovation tube.
        def wind_level_lower_rule(m, k, i, t):
            cap_i = self.static_physical_capacity[int(i)]
            ref = cap_i * m.mu_W * self.wind_shape[int(t)]
            return _P_total(m, k, i, t) >= ref - kappa_lvl_W * cap_i * m.sigma_W * sigma_bar_W_scale

        def wind_level_upper_rule(m, k, i, t):
            cap_i = self.static_physical_capacity[int(i)]
            ref = cap_i * m.mu_W * self.wind_shape[int(t)]
            return _P_total(m, k, i, t) <= ref + kappa_lvl_W * cap_i * m.sigma_W * sigma_bar_W_scale

        # Fleet aggregate band: independence-scaled width cap_rms << sum_caps for N > 1.
        def wind_fleet_lower_rule(m, k, t):
            fleet_sum = sum(_P_total(m, k, i, t) for i in m.wind_physical_generators)
            ref_S = sum_caps * m.mu_W * self.wind_shape[int(t)]
            return fleet_sum >= ref_S - kappa_fleet * cap_rms * m.sigma_W * sigma_bar_W_scale

        def wind_fleet_upper_rule(m, k, t):
            fleet_sum = sum(_P_total(m, k, i, t) for i in m.wind_physical_generators)
            ref_S = sum_caps * m.mu_W * self.wind_shape[int(t)]
            return fleet_sum <= ref_S + kappa_fleet * cap_rms * m.sigma_W * sigma_bar_W_scale

        m.wind_ar1_up_constraints = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps_minus_1, rule=wind_ar1_up_rule
        )
        m.wind_ar1_down_constraints = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps_minus_1, rule=wind_ar1_down_rule
        )
        m.wind_even_block_split = Constraint(
            m.scenarios, m.wind_blocks, m.time_steps, rule=wind_even_block_split_rule
        )
        m.wind_level_lower_constraints = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_level_lower_rule
        )
        m.wind_level_upper_constraints = Constraint(
            m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_level_upper_rule
        )
        if bool(getattr(self, "enable_fleet_band", True)):
            m.wind_fleet_lower_constraint = Constraint(
                m.scenarios, m.time_steps, rule=wind_fleet_lower_rule
            )
            m.wind_fleet_upper_constraint = Constraint(
                m.scenarios, m.time_steps, rule=wind_fleet_upper_rule
            )
