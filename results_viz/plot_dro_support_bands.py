"""Visualise all constraints that bound the DRO support set variables D[k,t] and P_max[k,i,t].

The DRO support set (DROWassersteinSupportSet) imposes two types of bounds
on each scenario variable:

  AR(1) innovation tube  — bounds the whitened increment at each step:
      |D[k,t] - rho*D[k,t-1] - ar1_ref[t]| <= kappa_ar1 * D_ref * sigma_D
  (and analogously for wind)

  Level band  — caps accumulated drift from stacked innovations:
      ref[t] - kappa_lvl * D_ref * sigma_D * sigma_bar  <=  D[k,t]
                    <=  ref[t] + kappa_lvl * D_ref * sigma_D * sigma_bar
  where sigma_bar = sigma / sqrt(1 - rho^2)

  Physical cap  — wind_physical_upper:
      P_max[k,i,t] <= cap_i

  Non-negativity  — variable domain:
      D[k,t] >= 0,  P_max[k,i,t] >= 0

The effective feasible range for each variable is the intersection of the
level band, the physical cap, and non-negativity.

The AR(1) tube is harder to show as a static band (it constrains increments,
not levels), so it is visualised as a ±kappa_ar1*sigma corridor width around
the reference at each step — indicating how tightly each transition is
constrained independently of the level band.

Key difference from the PoA support set:
  kappa is Sidak-corrected for joint coverage over T steps
  (default 99 % joint), giving kappa >> 1.96 for T=24.

Run:
  .\.venv\\Scripts\\python.exe results_viz\\plot_dro_support_bands.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as _norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager

OUTPUT_DIR = PROJECT_ROOT / "results_viz" / "figures" / "dro_support_bands"

# ---------------------------------------------------------------------------
# Physical reference values
# ---------------------------------------------------------------------------

_MANAGER = ScenarioManager("base_test_case")
D_REF: float = float(_MANAGER.base_case["demand"])
_WIND_IDX = _MANAGER.wind_generator_indices[0]
CAP_W: float = float(_MANAGER.base_case["pmax_list"][_WIND_IDX])

# DRO default joint-coverage target (matches DROWassersteinSupportSet.AR1_JOINT_COVERAGE)
DRO_AR1_COVERAGE: float = 0.99


def _dro_kappa(horizon: int, coverage: float = DRO_AR1_COVERAGE) -> float:
    """Sidak-corrected kappa giving ``coverage`` jointly across ``horizon`` i.i.d. innovations."""
    return float(_norm.ppf((1.0 + coverage ** (1.0 / horizon)) / 2.0))


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
    # Baseline — within normal ambiguity-set range
    RegimeSpec(
        label="Baseline  (μ_W=0.75, σ_W=0.025)  — within ambiguity set",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.75, sigma_W=0.025, rho_W=0.75,
        color="steelblue",
    ),
    # High wind + high volatility: level band upper > cap_i at peak
    # kappa_lvl(T=24) ≈ 3.54, so need: μ_W·shape_peak + 3.54·σ_W/√(1-ρ²) > 1
    # With μ_W=0.75, shape_peak≈1.15, ρ=0.75: σ_W > 0.025 needed → use 0.06
    RegimeSpec(
        label="High wind, high volatility  (μ_W=0.75, σ_W=0.06)  — upper > cap",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.75, sigma_W=0.06, rho_W=0.75,
        color="crimson",
    ),
    # Low wind + high volatility: level band lower < 0 at trough
    # Need: μ_W·shape_trough - 3.54·σ_W/√(1-ρ²) < 0
    # With μ_W=0.25, shape_trough≈0.85, ρ=0.75: σ_W > 0.040 needed → use 0.06
    RegimeSpec(
        label="Low wind, high volatility   (μ_W=0.25, σ_W=0.06)  — lower < 0",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.25, sigma_W=0.06, rho_W=0.75,
        color="darkorange",
    ),
    # Both violations simultaneously
    RegimeSpec(
        label="Extreme  (μ_W=0.75, σ_W=0.12)  — upper > cap AND lower < 0",
        mu_D=0.80, sigma_D=0.012, rho_D=0.75,
        mu_W=0.75, sigma_W=0.12, rho_W=0.75,
        color="purple",
    ),
]

# ---------------------------------------------------------------------------
# Band computation
# ---------------------------------------------------------------------------

def compute_bands(spec: RegimeSpec) -> dict:
    demand_shape = ScenarioManager._build_demand_shape(spec.horizon)
    wind_shape   = ScenarioManager._build_wind_shape(spec.horizon, spec.peak_W)
    t = np.arange(spec.horizon)

    kappa = _dro_kappa(spec.horizon)          # same kappa for AR(1) tube and level band

    stat_std_D = spec.sigma_D / np.sqrt(1.0 - spec.rho_D ** 2)
    stat_std_W = spec.sigma_W / np.sqrt(1.0 - spec.rho_W ** 2)

    # ---- Demand (MW) ----
    demand_ref    = D_REF * spec.mu_D * demand_shape
    # Level band
    d_margin      = kappa * D_REF * stat_std_D
    demand_lower  = demand_ref - d_margin
    demand_upper  = demand_ref + d_margin
    # AR(1) innovation half-width (width of each single-step tube)
    d_ar1_half    = np.full(spec.horizon, kappa * D_REF * spec.sigma_D)

    # ---- Wind (MW) ----
    wind_ref      = CAP_W * spec.mu_W * wind_shape
    # Level band
    w_margin      = kappa * CAP_W * stat_std_W
    wind_lower    = wind_ref - w_margin
    wind_upper    = wind_ref + w_margin
    # AR(1) innovation half-width
    w_ar1_half    = np.full(spec.horizon, kappa * CAP_W * spec.sigma_W)

    # Effective feasible range: level band ∩ [0, cap_i]
    eff_w_lower = np.maximum(wind_lower, 0.0)
    eff_w_upper = np.minimum(wind_upper, CAP_W)

    return dict(
        t=t,
        kappa=kappa,
        demand_ref=demand_ref,
        demand_lower=demand_lower,
        demand_upper=demand_upper,
        d_ar1_half=d_ar1_half,
        wind_ref=wind_ref,
        wind_lower=wind_lower,
        wind_upper=wind_upper,
        w_ar1_half=w_ar1_half,
        eff_w_lower=eff_w_lower,
        eff_w_upper=eff_w_upper,
        upper_clipped=bool(np.any(wind_upper > CAP_W)),
        lower_clipped=bool(np.any(wind_lower < 0.0)),
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_all(regimes: list[RegimeSpec], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(regimes)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n), sharex="col")
    if n == 1:
        axes = [axes]

    for row, (spec, ax_row) in enumerate(zip(regimes, axes)):
        ax_d, ax_w = ax_row
        b = compute_bands(spec)
        t = b["t"]
        c = spec.color
        kappa = b["kappa"]

        # ---- Demand panel ----
        # Level band
        ax_d.fill_between(t, b["demand_lower"], b["demand_upper"],
                          alpha=0.20, color=c, label=f"Level band  [D_lo, D_hi]  (κ={kappa:.2f})")
        # AR(1) innovation corridor around reference
        ax_d.fill_between(t,
                          b["demand_ref"] - b["d_ar1_half"],
                          b["demand_ref"] + b["d_ar1_half"],
                          alpha=0.35, color=c, hatch="...",
                          label=f"AR(1) innovation corridor  ±κ·D_ref·σ_D")
        ax_d.plot(t, b["demand_ref"], color=c, linewidth=2,
                  label="Reference  μ_D·h_D·D_ref")
        ax_d.axhline(0, color="black", linewidth=0.7, linestyle=":",
                     label="Zero  ← NonNegativeReals")
        ax_d.set_ylabel("Demand (MW)")
        ax_d.set_title(f"{spec.label}\nDemand level band  (DRO, κ={kappa:.2f})", fontsize=9)
        ax_d.legend(fontsize=6.5, loc="lower right")
        ax_d.grid(True, alpha=0.2)

        # ---- Wind panel ----
        # Full level band
        ax_w.fill_between(t, b["wind_lower"], b["wind_upper"],
                          alpha=0.15, color=c,
                          label=f"Level band  [W_lo, W_hi]  (κ={kappa:.2f})")

        # Portions cut by physical constraints (shown as hatched overlays)
        above_cap  = np.maximum(b["wind_upper"] - CAP_W, 0.0)
        below_zero = np.maximum(-b["wind_lower"], 0.0)
        if np.any(above_cap > 0):
            ax_w.fill_between(t, CAP_W, CAP_W + above_cap,
                              alpha=0.35, color="red", hatch="///", linewidth=0,
                              label="Cut by wind_physical_upper  (P_max ≤ cap_i)")
        if np.any(below_zero > 0):
            ax_w.fill_between(t, -below_zero, 0.0,
                              alpha=0.35, color="purple", hatch="\\\\\\", linewidth=0,
                              label="Cut by NonNegativeReals  (P_max ≥ 0)")

        # Effective feasible range
        ax_w.fill_between(t, b["eff_w_lower"], b["eff_w_upper"],
                          alpha=0.40, color=c,
                          label="Effective P_max range  (all constraints)")

        # AR(1) innovation corridor around reference
        ax_w.fill_between(t,
                          b["wind_ref"] - b["w_ar1_half"],
                          b["wind_ref"] + b["w_ar1_half"],
                          alpha=0.30, color=c, hatch="...",
                          label="AR(1) innovation corridor  ±κ·cap_i·σ_W")

        ax_w.plot(t, b["wind_ref"], color=c, linewidth=2,
                  label="Reference  μ_W·h_W·cap_i  (MW)")
        ax_w.axhline(CAP_W, color="black", linewidth=1.5, linestyle="--",
                     label=f"Installed cap = {CAP_W:.0f} MW  ← wind_physical_upper")
        ax_w.axhline(0.0, color="black", linewidth=0.8, linestyle=":",
                     label="Zero  ← NonNegativeReals domain")

        flags = []
        if b["upper_clipped"]: flags.append("⚠ upper > cap_i")
        if b["lower_clipped"]: flags.append("⚠ lower < 0")
        clip_note = "  ".join(flags) if flags else "✓ level band within [0, cap_i]"
        ax_w.set_title(f"Wind capacity band  (DRO, κ={kappa:.2f})\n{clip_note}", fontsize=9)
        ax_w.set_ylabel("Wind capacity (MW)")
        ax_w.legend(fontsize=6.0, loc="upper right", ncol=1)
        ax_w.grid(True, alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Time step")

    fig.suptitle(
        f"DRO support set bands  |  D_ref={D_REF:.0f} MW  |  cap_W={CAP_W:.0f} MW  |  "
        f"AR(1) joint coverage={DRO_AR1_COVERAGE:.0%}  |  "
        f"κ(T=24)={_dro_kappa(24):.2f}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = output_dir / "dro_support_bands.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Summary table
    print(
        f"\n{'Regime':<62} {'κ':>5} "
        f"{'W_lo_min':>10} {'W_hi_max':>10} "
        f"{'Eff_lo':>8} {'Eff_hi':>8} "
        f"{'>cap':>5} {'<0':>4}"
    )
    print("-" * 118)
    for spec in regimes:
        b = compute_bands(spec)
        print(
            f"{spec.label:<62} {b['kappa']:>5.2f} "
            f"{b['wind_lower'].min():>10.2f} "
            f"{b['wind_upper'].max():>10.2f} "
            f"{b['eff_w_lower'].min():>8.2f} "
            f"{b['eff_w_upper'].max():>8.2f} "
            f"{'Y' if b['upper_clipped'] else 'n':>5} "
            f"{'Y' if b['lower_clipped'] else 'n':>4}"
        )

    # Kappa comparison table
    print("\nκ comparison: DRO (Sidak) vs PoA (marginal 1.96)")
    print(f"  {'T':>4}  {'DRO κ (99%)':>12}  {'PoA κ':>8}")
    print("  " + "-" * 28)
    for T in [4, 8, 12, 24]:
        print(f"  {T:>4}  {_dro_kappa(T):>12.4f}  {'1.9600':>8}")


if __name__ == "__main__":
    plot_all(REGIMES, OUTPUT_DIR)
