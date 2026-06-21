# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-12
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Analytic upper bound on the PoA ratio over a DRO support set.

Certifies a valid McCormick PoA upper bound (``*_mccormick_PoA_bounds[1]``)
from case data instead of guessing it. For every scenario omega in the support
tube and every time step t:

  C_eq_t(omega)  <= C_anti(D_t)        (any feasible dispatch costs at most the
                                        anti-merit-order dispatch; wind unused)
  C_opt_t(omega) >= C_merit(D_t, W_t)  (ramp-relaxed merit order; relaxing
                                        constraints can only lower the minimum)

and by the mediant inequality the per-scenario ratio of sums is bounded by the
worst per-step ratio:

  PoA(omega) = sum_t C_eq_t / sum_t C_opt_t <= max_t C_anti(D_t)/C_merit(D_t, W_t).

Maximizing over the AR(1) level bands (|x_t - ref_t| <= kappa*sigma*scale*level_scale(rho,t))
gives a bound valid for every scenario the DRO can place mass on. It bounds ALL
feasible dispatches, not just NN-equilibrium-reachable ones, so it is
conservative -- achieved PoA values are typically far below it.

Edit STUDY_NAME / HORIZON below (regime params + calibrated kappa are read from
that run's artifacts), then run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.analytic_poa_bound
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.scenarios.scenario_generator import ScenarioManager
from driver.sensitivity.sensitivity_config import RESULT_ROOT
from models.DRO_PoA.dro_poa_model.support_set import _ar1_kappa, _level_scale

CASE = "base_test_case"
STUDY_NAME = "dro_horizon_timing_sweep_wind"
HORIZON = 6


def _run_dir() -> Path:
    return Path(RESULT_ROOT) / STUDY_NAME / f"T{HORIZON}"


def merit_cost(
    demand: float,
    wind_avail_per_gen: list[float],
    wind_blocks: list[tuple[float, float]],
    conv_blocks: list[tuple[float, float]],
) -> float:
    """Ramp-relaxed merit-order cost: valid lower bound on C_opt at this step."""
    cost, rem = 0.0, demand
    for (block_cost, block_cap), avail in zip(wind_blocks, wind_avail_per_gen):
        q = min(avail, block_cap, rem)
        cost += block_cost * q
        rem -= q
    for block_cost, block_cap in conv_blocks:
        q = min(block_cap, rem)
        cost += block_cost * q
        rem -= q
    return cost


def anti_cost(
    demand: float,
    wind_blocks: list[tuple[float, float]],
    conv_blocks: list[tuple[float, float]],
) -> float:
    """Anti-merit-order cost: valid upper bound on ANY feasible dispatch cost."""
    cost, rem = 0.0, demand
    for block_cost, block_cap in sorted(conv_blocks, reverse=True):
        q = min(block_cap, rem)
        cost += block_cost * q
        rem -= q
    for block_cost, block_cap in sorted(wind_blocks, reverse=True):
        q = min(block_cap, rem)
        cost += block_cost * q
        rem -= q
    return cost


def main() -> None:
    run_dir = _run_dir()
    regime_params = json.loads(
        (run_dir / "poa" / "poa_worst_case_regime_params.json").read_text(encoding="utf-8")
    )
    support_report = json.loads(
        (run_dir / "support_oos_report.json").read_text(encoding="utf-8")
    )
    coverage = float(support_report["chosen_coverage"])
    kappa = float(support_report.get("chosen_kappa", _ar1_kappa(HORIZON, coverage)))

    manager = ScenarioManager(CASE)
    conv_blocks = sorted(
        (float(b["cost"]), float(b["pmax"])) for b in manager.blocks if not b["is_wind"]
    )
    wind_blocks = sorted(
        (float(b["cost"]), float(b["pmax"])) for b in manager.blocks if b["is_wind"]
    )
    demand_ref_scalar = float(manager.base_case["demand"])
    demand_shape = ScenarioManager._build_demand_shape(HORIZON)
    wind_shape = ScenarioManager._build_wind_shape(HORIZON, float(regime_params["peak_W"]))

    thr_D = kappa * demand_ref_scalar * float(regime_params["sigma_D"])
    wind_cap = wind_blocks[0][1]  # per-generator capacity (uniform in this case)
    thr_W = kappa * wind_cap * float(regime_params["sigma_W"])

    print(f"Case {CASE}, study {STUDY_NAME}, T={HORIZON}")
    print(f"  regime: mu_D={regime_params['mu_D']:.4f} mu_W={regime_params['mu_W']:.4f}  "
          f"kappa={kappa:.4f} (coverage {coverage})")
    print(f"\n{'t':>2} {'D_ref':>7} {'D_min':>7} {'D_max':>7} {'W_max':>7} {'ratio':>8}")
    overall = 0.0
    for t in range(HORIZON):
        ls_D = _level_scale(float(regime_params["rho_D"]), t)
        ls_W = _level_scale(float(regime_params["rho_W"]), t)
        D_ref = demand_ref_scalar * float(regime_params["mu_D"]) * demand_shape[t]
        D_min = max(0.0, D_ref - thr_D * ls_D)
        D_max = D_ref + thr_D * ls_D
        wind_ref = wind_cap * float(regime_params["mu_W"]) * wind_shape[t]
        wind_max_per_gen = [
            min(cap, wind_ref + thr_W * ls_W) for _cost, cap in wind_blocks
        ]
        best = 0.0
        for demand in np.linspace(D_min, D_max, 200):
            if demand <= 0:
                continue
            ratio = anti_cost(demand, wind_blocks, conv_blocks) / merit_cost(
                demand, wind_max_per_gen, wind_blocks, conv_blocks
            )
            best = max(best, ratio)
        overall = max(overall, best)
        print(
            f"{t:>2} {D_ref:>7.1f} {D_min:>7.1f} {D_max:>7.1f} "
            f"{sum(wind_max_per_gen):>7.1f} {best:>8.2f}"
        )

    print(f"\nAnalytic PoA upper bound: {overall:.2f}")
    print("Any *_mccormick_PoA_bounds upper >= this value is certified non-clamping.")


if __name__ == "__main__":
    main()
