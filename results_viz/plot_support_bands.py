# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-03
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Visualise the PoA support set bands, matching the implementation in
models/PoA/poa_model/support_set.py exactly.

Three layers are shown for each variable:

  Cumulative AR(1) reachable set (uncapped)
      Half-width at t: kappa * sigma * (1 - rho^{t+1}) / (1 - rho)
      Grows from kappa*sigma at t=0 toward kappa*sigma/(1-rho) as t→inf.
      This is the set the AR(1) tube constraints alone allow.

  Time-varying level band  [what is actually implemented]
      Half-width at t: kappa * sigma * min((1-rho^{t+1})/(1-rho), 1/sqrt(1-rho^2))
      = min(cumulative AR(1), stationary bound)
      Grows until it hits the stationary bound and stays flat.
      This is demand_lower / demand_upper / wind_lower / wind_upper in the model.

  Stationary bound (constant reference line)
      Half-width: kappa * sigma / sqrt(1-rho^2)
      The asymptotic level-band cap.

  Effective feasible range  [wind only]
      Level band ∩ [0, cap_i]  (physical_upper + NonNegativeReals)

Run:
  .\.venv\\Scripts\\python.exe results_viz\\plot_support_bands.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
import numpy as np
from scipy.stats import norm as _norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager

OUTPUT_DIR = PROJECT_ROOT / "results_viz" / "figures" / "support_bands"

# ---------------------------------------------------------------------------
# Physical reference values
# ---------------------------------------------------------------------------

_MANAGER = ScenarioManager("base_test_case")
D_REF: float = float(_MANAGER.base_case["demand"])
_WIND_IDX = _MANAGER.wind_generator_indices[0]
CAP_W: float = float(_MANAGER.base_case["pmax_list"][_WIND_IDX])

POA_AR1_COVERAGE: float = 0.95  # matches AR1_JOINT_COVERAGE in support_set.py


def _ar1_kappa(num_constraints: int) -> float:
    """Sidak-corrected kappa: 95 % joint over num_constraints i.i.d. innovations."""
    per_step = POA_AR1_COVERAGE ** (1.0 / num_constraints)
    return float(_norm.ppf((1.0 + per_step) / 2.0))


def _level_scale(rho: float, t: int) -> float:
    """Time-varying level-band scale (multiples of innovation sigma).

    scale(t) = min( (1-rho^{t+1})/(1-rho),  1/sqrt(1-rho^2) )

    Matches _level_scale in models/PoA/poa_model/support_set.py exactly.
    """
    cumulative = (1.0 - rho ** (t + 1)) / (1.0 - rho)
    stationary = 1.0 / np.sqrt(1.0 - rho ** 2)
    return min(cumulative, stationary)


# ---------------------------------------------------------------------------
# Regime specification
# ---------------------------------------------------------------------------

@dataclass
class RegimeSpec:
    label: str
    mu_D: float
    sigma_D: float
    rho_D: float
    mu_W: float
    sigma_W: float
    rho_W: float
    peak_W: float = 14.0
    horizon: int = 24
    color: str = "steelblue"


REGIMES: list[RegimeSpec] = [
    RegimeSpec(
        label="Baseline  (μ_W=0.75, σ_W=0.025)  — within ambiguity set",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.75, sigma_W=0.025, rho_W=0.75,
        color="steelblue",
    ),
    RegimeSpec(
        label="High wind, high volatility  (μ_W=0.75, σ_W=0.08)  — upper > cap",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.75, sigma_W=0.08, rho_W=0.75,
        color="crimson",
    ),
    RegimeSpec(
        label="Low wind, high volatility   (μ_W=0.25, σ_W=0.10)  — lower < 0",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.25, sigma_W=0.10, rho_W=0.75,
        color="darkorange",
    ),
    RegimeSpec(
        label="Extreme  (μ_W=0.75, σ_W=0.15)  — upper > cap AND lower < 0",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.75, sigma_W=0.15, rho_W=0.75,
        color="purple",
    ),
]

# ---------------------------------------------------------------------------
# Band computation — mirrors support_set.py exactly
# ---------------------------------------------------------------------------

def compute_bands(spec: RegimeSpec) -> dict:
    T = spec.horizon
    demand_shape = ScenarioManager._build_demand_shape(T)
    wind_shape   = ScenarioManager._build_wind_shape(T, spec.peak_W)
    t_arr = np.arange(T)

    kappa = _ar1_kappa(T)  # Sidak over all T constraints, 95% joint

    # Precompute time-varying scale factors (pure Python floats, rho is fixed)
    d_scales = np.array([_level_scale(spec.rho_D, t) for t in range(T)])
    w_scales = np.array([_level_scale(spec.rho_W, t) for t in range(T)])

    # Cumulative AR(1) reachable half-width (uncapped, grows without bound)
    d_cumul = np.array([(1.0 - spec.rho_D**(t+1)) / (1.0 - spec.rho_D) for t in range(T)])
    w_cumul = np.array([(1.0 - spec.rho_W**(t+1)) / (1.0 - spec.rho_W) for t in range(T)])

    # Stationary bound (constant)
    d_stat = 1.0 / np.sqrt(1.0 - spec.rho_D ** 2)
    w_stat = 1.0 / np.sqrt(1.0 - spec.rho_W ** 2)

    # ---- Demand (MW) ----
    demand_ref    = D_REF * spec.mu_D * demand_shape
    # Level band: time-varying (implemented in support_set.py)
    demand_lower  = demand_ref - kappa * D_REF * spec.sigma_D * d_scales
    demand_upper  = demand_ref + kappa * D_REF * spec.sigma_D * d_scales
    # Cumulative AR(1) reachable set (uncapped, for comparison)
    demand_cumul_lo = demand_ref - kappa * D_REF * spec.sigma_D * d_cumul
    demand_cumul_hi = demand_ref + kappa * D_REF * spec.sigma_D * d_cumul
    # Stationary reference lines
    demand_stat_lo  = demand_ref - kappa * D_REF * spec.sigma_D * d_stat
    demand_stat_hi  = demand_ref + kappa * D_REF * spec.sigma_D * d_stat

    # ---- Wind (MW) ----
    wind_ref      = CAP_W * spec.mu_W * wind_shape
    wind_lower    = wind_ref - kappa * CAP_W * spec.sigma_W * w_scales
    wind_upper    = wind_ref + kappa * CAP_W * spec.sigma_W * w_scales
    wind_cumul_lo = wind_ref - kappa * CAP_W * spec.sigma_W * w_cumul
    wind_cumul_hi = wind_ref + kappa * CAP_W * spec.sigma_W * w_cumul
    wind_stat_lo  = wind_ref - kappa * CAP_W * spec.sigma_W * w_stat
    wind_stat_hi  = wind_ref + kappa * CAP_W * spec.sigma_W * w_stat

    # Effective feasible range: level band ∩ [0, cap_i]
    eff_lower = np.maximum(wind_lower, 0.0)
    eff_upper = np.minimum(wind_upper, CAP_W)

    return dict(
        t=t_arr, kappa=kappa,
        demand_ref=demand_ref,
        demand_lower=demand_lower, demand_upper=demand_upper,
        demand_cumul_lo=demand_cumul_lo, demand_cumul_hi=demand_cumul_hi,
        demand_stat_lo=demand_stat_lo, demand_stat_hi=demand_stat_hi,
        wind_ref=wind_ref,
        wind_lower=wind_lower, wind_upper=wind_upper,
        wind_cumul_lo=wind_cumul_lo, wind_cumul_hi=wind_cumul_hi,
        wind_stat_lo=wind_stat_lo, wind_stat_hi=wind_stat_hi,
        eff_lower=eff_lower, eff_upper=eff_upper,
        upper_clipped=bool(np.any(wind_upper > CAP_W)),
        lower_clipped=bool(np.any(wind_lower < 0.0)),
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_all(regimes: list[RegimeSpec], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(regimes)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n), sharex="col")
    if n == 1:
        axes = [axes]

    for row, (spec, ax_row) in enumerate(zip(regimes, axes)):
        ax_d, ax_w = ax_row
        b = compute_bands(spec)
        t = b["t"]
        c = spec.color

        # ---- Demand panel ----
        # Layer 1: cumulative AR(1) reachable set (uncapped, widest)
        ax_d.fill_between(t, b["demand_cumul_lo"], b["demand_cumul_hi"],
                          alpha=0.10, color=c,
                          label="Cumulative AR(1) reachable set  (uncapped)")
        # Layer 2: time-varying level band = min(cumulative, stationary)
        ax_d.fill_between(t, b["demand_lower"], b["demand_upper"],
                          alpha=0.35, color=c,
                          label="Level band  [D_lo, D_hi]  (implemented)")
        # Stationary asymptote
        ax_d.plot(t, b["demand_stat_lo"], color=c, linewidth=1.0, linestyle="--", alpha=0.6)
        ax_d.plot(t, b["demand_stat_hi"], color=c, linewidth=1.0, linestyle="--", alpha=0.6,
                  label="Stationary bound  ±κ·σ/√(1-ρ²)")
        ax_d.plot(t, b["demand_ref"], color=c, linewidth=2, label="Reference  μ_D·h_D·D_ref")
        ax_d.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax_d.set_ylabel("Demand (MW)")
        ax_d.set_title(f"{spec.label}\nDemand level band  (κ={b['kappa']:.2f}, Sidak 95% over T={spec.horizon})",
                       fontsize=9)
        ax_d.legend(fontsize=6.5, loc="lower right")
        ax_d.grid(True, alpha=0.2)

        # ---- Wind panel ----
        # Layer 1: cumulative AR(1) reachable set (uncapped)
        ax_w.fill_between(t, b["wind_cumul_lo"], b["wind_cumul_hi"],
                          alpha=0.10, color=c,
                          label="Cumulative AR(1) reachable set  (uncapped)")
        # Layer 2: time-varying level band (implemented)
        ax_w.fill_between(t, b["wind_lower"], b["wind_upper"],
                          alpha=0.25, color=c,
                          label="Level band  [W_lo, W_hi]  (implemented)")

        # Violations cut by physical constraints
        above_cap = np.maximum(b["wind_upper"] - CAP_W, 0.0)
        if np.any(above_cap > 0):
            ax_w.fill_between(t, CAP_W, CAP_W + above_cap,
                              alpha=0.4, color="red", hatch="///", linewidth=0,
                              label="Cut by wind_physical_upper  (P_max ≤ cap_i)")
        below_zero = np.maximum(-b["wind_lower"], 0.0)
        if np.any(below_zero > 0):
            ax_w.fill_between(t, -below_zero, 0.0,
                              alpha=0.4, color="purple", hatch="\\\\\\", linewidth=0,
                              label="Cut by NonNegativeReals  (P_max ≥ 0)")

        # Layer 3: effective range (intersection with physical constraints)
        ax_w.fill_between(t, b["eff_lower"], b["eff_upper"],
                          alpha=0.45, color=c,
                          label="Effective P_max range  (all constraints)")

        # Stationary asymptote
        ax_w.plot(t, b["wind_stat_lo"], color=c, linewidth=1.0, linestyle="--", alpha=0.6)
        ax_w.plot(t, b["wind_stat_hi"], color=c, linewidth=1.0, linestyle="--", alpha=0.6,
                  label="Stationary bound  ±κ·σ_W/√(1-ρ_W²)")

        ax_w.plot(t, b["wind_ref"], color=c, linewidth=2,
                  label="Reference  μ_W·h_W·cap_i  (MW)")
        ax_w.axhline(CAP_W, color="black", linewidth=1.5, linestyle="--",
                     label=f"Installed cap = {CAP_W:.0f} MW  ← wind_physical_upper")
        ax_w.axhline(0.0, color="black", linewidth=0.8, linestyle=":",
                     label="Zero  ← NonNegativeReals")

        flags = []
        if b["upper_clipped"]: flags.append("⚠ upper > cap_i")
        if b["lower_clipped"]: flags.append("⚠ lower < 0")
        clip_note = "  ".join(flags) if flags else "✓ level band within [0, cap_i]"
        ax_w.set_title(
            f"Wind capacity band  (κ={b['kappa']:.2f}, Sidak 95% over T={spec.horizon})\n{clip_note}",
            fontsize=9,
        )
        ax_w.set_ylabel("Wind capacity (MW)")
        ax_w.legend(fontsize=6.0, loc="upper right")
        ax_w.grid(True, alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Time step")

    kappa_example = _ar1_kappa(24)
    fig.suptitle(
        f"PoA support set bands  |  D_ref={D_REF:.0f} MW  |  cap_W={CAP_W:.0f} MW  |  "
        f"κ(T=24) = {kappa_example:.2f}  (Sidak, 95% joint over T constraints)\n"
        f"Level band = min(cumulative AR(1) reachable, stationary bound) — grows from t=0",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = output_dir / "support_bands.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Summary table
    print(f"\nκ(T=24) = {kappa_example:.4f}   |   stationary scale (ρ=0.75) = "
          f"{1/np.sqrt(1-0.75**2):.4f}   |   steady-state cumulative (ρ=0.75) = "
          f"{1/(1-0.75):.4f}")
    print(f"\nLevel-band scale(t) for ρ=0.75:")
    print("  " + "  ".join(f"t={t}: {_level_scale(0.75, t):.3f}" for t in range(8)))

    print(f"\n{'Regime':<60} {'W_lo_min':>10} {'W_hi_max':>10} {'Eff_lo':>8} {'Eff_hi':>8} {'>cap':>5} {'<0':>4}")
    print("-" * 115)
    for spec in regimes:
        b = compute_bands(spec)
        print(
            f"{spec.label:<60} "
            f"{b['wind_lower'].min():>10.2f} "
            f"{b['wind_upper'].max():>10.2f} "
            f"{b['eff_lower'].min():>8.2f} "
            f"{b['eff_upper'].max():>8.2f} "
            f"{'Y' if b['upper_clipped'] else 'n':>5} "
            f"{'Y' if b['lower_clipped'] else 'n':>4}"
        )


if __name__ == "__main__":
    plot_all(REGIMES, OUTPUT_DIR)
