from __future__ import annotations

import numpy as np
from pyomo.environ import Constraint
from scipy.stats import norm as _norm


def _ar1_kappa(num_time_steps: int, joint_coverage: float) -> float:
    """Sidak-corrected kappa giving joint_coverage across num_time_steps i.i.d. innovations.

    Each per-step band is Phi^{-1}((1 + coverage^{1/T}) / 2) standard deviations wide,
    so T independent Gaussian innovations each fall inside with probability coverage^{1/T},
    giving joint coverage exactly coverage.
    """
    return float(_norm.ppf((1.0 + joint_coverage ** (1.0 / num_time_steps)) / 2.0))


def _level_scale(rho: float, t: int) -> float:
    """Time-varying level-band scale factor (multiples of innovation sigma).

    scale(t) = min( (1-|rho|^{t+1})/(1-|rho|),  1/sqrt(1-rho^2) )

    Uses |rho| in the cumulative term because the worst-case level deviation
    satisfies |level_t| <= |rho|*|level_{t-1}| + |eps_t|, giving the geometric
    sum (1-|rho|^{t+1})/(1-|rho|) regardless of sign.  For positive rho this
    is identical to the previous formula; for negative rho the original formula
    oscillated (e.g. 0.25 at t=1 for rho=-0.75) making the band too tight.
    At t=0 this equals 1; it grows until it reaches 1/sqrt(1-rho^2) and stays flat.
    """
    abs_rho = abs(rho)
    cumulative = (1.0 - abs_rho ** (t + 1)) / (1.0 - abs_rho)
    stationary = 1.0 / np.sqrt(1.0 - rho ** 2)
    return min(cumulative, stationary)


def support_tube_parameters(
    num_time_steps: int,
    rho: float,
    ar1_coverage: float | None = None,
    level_coverage: float | None = None,
) -> tuple[float, float, list[float]]:
    """Resolve the AR(1) support-tube parameters shared by the model and diagnostic.

    Returns ``(kappa_ar1, kappa_lvl, level_scales)`` where ``kappa_ar1`` is the
    innovation-band multiplier, ``kappa_lvl`` the level-band multiplier, and
    ``level_scales[t] = _level_scale(rho, t)``. Resolving coverage here (rather than
    in two places) guarantees ``DROWassersteinSupportSet`` and
    ``DROPoASupportDiagnostics`` build the identical tube, so "inside support per
    diagnostic" means W=0 is feasible in the model.
    """
    coverage = float(ar1_coverage or DROWassersteinSupportSet.AR1_JOINT_COVERAGE)
    kappa_ar1 = _ar1_kappa(num_time_steps, coverage)
    resolved_level_coverage = float(level_coverage or coverage)
    kappa_lvl = _ar1_kappa(num_time_steps, resolved_level_coverage)
    level_scales = [_level_scale(rho, t) for t in range(num_time_steps)]
    return kappa_ar1, kappa_lvl, level_scales


class DROWassersteinSupportSet:
    """Support set U for the Wasserstein DRO inner problem.

    Constraints that define the feasible region for scenario variables
    (D[k,t], P_max_block[k,i,b,t]):

    - Regime fixing: all distribution parameters (mu_D, sigma_D, ...) are
      pinned to their fixed regime values.
    - AR(1) innovation tube for t >= 1: bounds the whitened increment
        D[t] - rho_D * D[t-1]  around the deterministic reference
        D_ref * mu_D * (shape[t] - rho_D * shape[t-1])
      within +/- kappa_ar1 * D_ref * sigma_D.  Analogue for wind.
    - t=0 innovation band: cold start with innovation std (not stationary std).
    - Stationary level band: caps accumulated drift from stacked innovations.
    - Physical limits: wind capacity in [0, cap_i], dispatch feasibility
        D[t] <= total available capacity.
    - Conventional generator capacity: fixed at static block capacity.
    - Wind block even-split: blocks within one physical generator share equally.

    Coverage and kappa
    ------------------
    ``ar1_coverage`` on the instance overrides the class default AR1_JOINT_COVERAGE.
    ``level_coverage`` overrides the level-band kappa independently.
    ``AR1_JOINT_COVERAGE`` is updated at runtime by the pipeline calibration step
    (``calibrate_ar1_coverage_from_scenarios``) to the minimum coverage that
    contains all empirical trajectories, so any remaining call sites that use
    the class default automatically pick up the calibrated value.
    """

    AR1_JOINT_COVERAGE: float = 0.99  # updated at runtime by calibration

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        self._build_regime_fixing_constraints()
        self._build_wasserstein_demand()
        self._build_wasserstein_wind()

    # ------------------------------------------------------------------
    # Regime fixing
    # ------------------------------------------------------------------

    def _build_regime_fixing_constraints(self) -> None:
        m = self.model
        m.regime_mu_D_fixed = Constraint(expr=m.mu_D == self.mu_D_fixed)
        m.regime_sigma_D_fixed = Constraint(expr=m.sigma_D == self.sigma_D_fixed)
        m.regime_mu_W_fixed = Constraint(expr=m.mu_W == self.mu_W_fixed)
        m.regime_sigma_W_fixed = Constraint(expr=m.sigma_W == self.sigma_W_fixed)
        m.regime_rho_D_fixed = Constraint(expr=m.rho_D == self.demand_rho_fixed)
        m.regime_rho_W_fixed = Constraint(expr=m.rho_W == self.wind_rho_fixed)
        m.regime_peak_W_fixed = Constraint(expr=m.peak_W == self.peak_W_fixed)

    # ------------------------------------------------------------------
    # Demand
    # ------------------------------------------------------------------

    def _build_wasserstein_demand(self) -> None:
        m = self.model
        kappa_ar1, kappa_lvl_D, demand_scales = support_tube_parameters(
            self.num_time_steps,
            self.demand_rho_fixed,
            getattr(self, "ar1_coverage", None),
            getattr(self, "level_coverage", None),
        )

        def _ar1_ref_D(m, t: int) -> object:
            return (
                self.demand_D_ref * m.mu_D
                * (self.demand_shape[t] - self.demand_rho_fixed * self.demand_shape[t - 1])
            )

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

        def dispatch_feasibility_rule(m, k, t):
            return m.D[k, t] <= sum(
                m.P_max_block[k, i, b, t] for i, b in m.generator_blocks
            )

        def demand_level_lower_rule(m, k, t):
            ref = self.demand_D_ref * m.mu_D * self.demand_shape[int(t)]
            return m.D[k, t] >= ref - kappa_lvl_D * self.demand_D_ref * m.sigma_D * demand_scales[int(t)]

        def demand_level_upper_rule(m, k, t):
            ref = self.demand_D_ref * m.mu_D * self.demand_shape[int(t)]
            return m.D[k, t] <= ref + kappa_lvl_D * self.demand_D_ref * m.sigma_D * demand_scales[int(t)]

        def demand_reference_feasibility_rule(m, t):
            return self.demand_D_ref * m.mu_D * self.demand_shape[int(t)] >= 0

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
        m.demand_reference_feasibility = Constraint(
            m.time_steps, rule=demand_reference_feasibility_rule
        )

    # ------------------------------------------------------------------
    # Wind
    # ------------------------------------------------------------------

    def _build_wasserstein_wind(self) -> None:
        m = self.model
        kappa_ar1, kappa_lvl_W, wind_scales = support_tube_parameters(
            self.num_time_steps,
            self.wind_rho_fixed,
            getattr(self, "ar1_coverage", None),
            getattr(self, "level_coverage", None),
        )

        def _ar1_ref_W(m, i: int, t: int) -> object:
            return (
                self.static_physical_capacity[i] * m.mu_W
                * (self.wind_shape[t] - self.wind_rho_fixed * self.wind_shape[t - 1])
            )

        def _P_total(m, k, i, t):
            return sum(
                m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)]
            )

        def wind_physical_upper_rule(m, k, i, t):
            return _P_total(m, k, i, t) <= self.static_physical_capacity[int(i)]

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

        def conventional_capacity_rule(m, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.P_max_block[k, i, b, t] == self.static_block_capacity[global_block]

        def wind_even_block_split_rule(m, k, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return len(local_blocks) * m.P_max_block[k, i, b, t] == sum(
                m.P_max_block[k, i, other_b, t] for other_b in local_blocks
            )

        def wind_level_lower_rule(m, k, i, t):
            cap_i = self.static_physical_capacity[int(i)]
            ref = cap_i * m.mu_W * self.wind_shape[int(t)]
            return _P_total(m, k, i, t) >= ref - kappa_lvl_W * cap_i * m.sigma_W * wind_scales[int(t)]

        def wind_level_upper_rule(m, k, i, t):
            cap_i = self.static_physical_capacity[int(i)]
            ref = cap_i * m.mu_W * self.wind_shape[int(t)]
            return _P_total(m, k, i, t) <= ref + kappa_lvl_W * cap_i * m.sigma_W * wind_scales[int(t)]

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
