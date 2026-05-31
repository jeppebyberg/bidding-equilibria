"""Plot DRO empirical scenarios against support set bounds.

Run from the repository root:
    .venv/Scripts/python.exe driver/plot_dro_support_set_check.py

Reads scenarios from results/full_pipeline_DRO/dro_scenarios/scenarios.csv and
overlays the support set bounds derived from each scenario's stored regime
parameters.  Violated scenarios are drawn in red.
Figures are saved to results/full_pipeline_DRO/support_set_check/.

Toggle USE_WASSERSTEIN_SUPPORT_SET to switch between:
  False (default) -- DROPoASupportSet: pointwise level box (kappa=1.96,
                     stationary std) AND AR(1) innovation tube (joint 95%).
  True            -- DROWassersteinSupportSet: AR(1) innovation tube only
                     (joint 99%), including an explicit t=0 band; no level box.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm as _norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager

# ---------------------------------------------------------------------------
# Configuration — edit here
# ---------------------------------------------------------------------------
CASE = "base_test_case"
SCENARIO_CSV = PROJECT_ROOT / "results/full_pipeline_DRO/dro_scenarios/scenarios.csv"
OUT_DIR = PROJECT_ROOT / "results/full_pipeline_DRO/support_set_check"

# Set to True when the pipeline uses DROWassersteinSupportSet.
USE_WASSERSTEIN_SUPPORT_SET = True

# Parameters for the old support set (DROPoASupportSet)
KAPPA_PW = 1.96          # pointwise 95% CI
JOINT_COVERAGE_LEGACY = 0.95

# Parameters for the Wasserstein support set (DROWassersteinSupportSet)
JOINT_COVERAGE_WASSERSTEIN = 0.99

# Derived
JOINT_COVERAGE = JOINT_COVERAGE_WASSERSTEIN if USE_WASSERSTEIN_SUPPORT_SET else JOINT_COVERAGE_LEGACY
# ---------------------------------------------------------------------------


def _kappa_ar1(T: int, coverage: float) -> float:
    return float(_norm.ppf((1.0 + coverage ** (1.0 / T)) / 2.0))


# ---------------------------------------------------------------------------
# Bound helpers
# ---------------------------------------------------------------------------

def demand_pointwise_bounds(
    mu_D: float, sigma_D: float, rho_D: float,
    demand_shape: np.ndarray, D_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise ±kappa_pw * stationary_std level box (DROPoASupportSet only)."""
    ref = D_ref * mu_D * demand_shape
    margin = KAPPA_PW * D_ref * sigma_D / np.sqrt(1.0 - rho_D ** 2)
    return np.maximum(ref - margin, 0.0), ref + margin


def demand_ar1_tube(
    mu_D: float, sigma_D: float, rho_D: float,
    demand_shape: np.ndarray, D_ref: float, T: int,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) innovation tube half-width at each t>=1.

    Returns (ar1_ref, half_width) both of shape (T-1,).
    Whitened innovation = D[t] - rho*D[t-1] - ar1_ref[t] must lie in +-half_width.
    """
    ka = _kappa_ar1(T, coverage)
    ar1_ref = D_ref * mu_D * (demand_shape[1:] - rho_D * demand_shape[:-1])
    half_width = np.full(T - 1, ka * D_ref * sigma_D)
    return ar1_ref, half_width


def demand_t0_band(
    mu_D: float, sigma_D: float,
    demand_shape: np.ndarray, D_ref: float, T: int,
    coverage: float,
) -> tuple[float, float]:
    """t=0 band for DROWassersteinSupportSet (innovation std, not stationary std).

    Returns (half_width,) scalar.
    """
    ka = _kappa_ar1(T, coverage)
    return float(ka * D_ref * sigma_D)


def wind_pointwise_bounds(
    mu_W: float, sigma_W: float, rho_W: float,
    wind_shape: np.ndarray, capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise ±kappa_pw * stationary_std level box (DROPoASupportSet only)."""
    ref = capacity * mu_W * wind_shape
    margin = KAPPA_PW * capacity * sigma_W / np.sqrt(1.0 - rho_W ** 2)
    return np.maximum(ref - margin, 0.0), np.minimum(ref + margin, capacity)


def wind_ar1_tube(
    mu_W: float, sigma_W: float, rho_W: float,
    wind_shape: np.ndarray, capacity: float, T: int,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) innovation tube. Returns (ar1_ref, half_width) of shape (T-1,)."""
    ka = _kappa_ar1(T, coverage)
    ar1_ref = capacity * mu_W * (wind_shape[1:] - rho_W * wind_shape[:-1])
    half_width = np.full(T - 1, ka * capacity * sigma_W)
    return ar1_ref, half_width


# ---------------------------------------------------------------------------
# Violation detection
# ---------------------------------------------------------------------------

def demand_violated(
    D: np.ndarray,
    mu_D: float, sigma_D: float, rho_D: float,
    demand_shape: np.ndarray, D_ref: float, T: int,
    coverage: float,
    tol: float = 1e-9,
) -> bool:
    """True if D violates the active support set at any time step."""
    if not USE_WASSERSTEIN_SUPPORT_SET:
        lb, ub = demand_pointwise_bounds(mu_D, sigma_D, rho_D, demand_shape, D_ref)
        if np.any(D < lb - tol) or np.any(D > ub + tol):
            return True

    # t=0 band
    hw0 = demand_t0_band(mu_D, sigma_D, demand_shape, D_ref, T, coverage)
    ref0 = D_ref * mu_D * demand_shape[0]
    if abs(D[0] - ref0) > hw0 + tol:
        return True

    # t>=1 AR(1) tube
    if T > 1:
        ar1_ref, half_width = demand_ar1_tube(mu_D, sigma_D, rho_D, demand_shape, D_ref, T, coverage)
        whitened = D[1:] - rho_D * D[:-1] - ar1_ref
        if np.any(np.abs(whitened) > half_width + tol):
            return True

    return False


def wind_violated(
    P: np.ndarray,
    mu_W: float, sigma_W: float, rho_W: float,
    wind_shape: np.ndarray, capacity: float, T: int,
    coverage: float,
    tol: float = 1e-9,
) -> bool:
    """True if P violates the active support set at any time step."""
    if not USE_WASSERSTEIN_SUPPORT_SET:
        lb, ub = wind_pointwise_bounds(mu_W, sigma_W, rho_W, wind_shape, capacity)
        if np.any(P < lb - tol) or np.any(P > ub + tol):
            return True

    # Physical bounds (always active)
    if np.any(P < -tol) or np.any(P > capacity + tol):
        return True

    # t=0 band
    ka = _kappa_ar1(T, coverage)
    ref0 = capacity * mu_W * wind_shape[0]
    if abs(P[0] - ref0) > ka * capacity * sigma_W + tol:
        return True

    # t>=1 AR(1) tube
    if T > 1:
        ar1_ref, half_width = wind_ar1_tube(mu_W, sigma_W, rho_W, wind_shape, capacity, T, coverage)
        whitened = P[1:] - rho_W * P[:-1] - ar1_ref
        if np.any(np.abs(whitened) > half_width + tol):
            return True

    return False


# ---------------------------------------------------------------------------
# Profile parsing
# ---------------------------------------------------------------------------

def parse_profile(raw: Any) -> np.ndarray:
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return np.array([float(v) for v in raw], dtype=float)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

GOOD_COLOR = "#2196F3"
BAD_COLOR = "#F44336"
ALPHA_LINE = 0.45
BOUND_COLOR = "#333333"
BOUND_ALPHA = 0.15
T0_COLOR = "#FF9800"     # orange band for t=0 (Wasserstein only)
T0_ALPHA = 0.20


def _add_shading(
    ax: plt.Axes,
    t: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    label: str = "Support set",
    color: str = BOUND_COLOR,
    alpha: float = BOUND_ALPHA,
) -> None:
    ax.fill_between(t, lb, ub, color=color, alpha=alpha, label=label)
    ax.plot(t, lb, color=color, linewidth=1.0, linestyle="--")
    ax.plot(t, ub, color=color, linewidth=1.0, linestyle="--")


def _legend_handles(has_violations: bool, extra_handles: list[Any] | None = None) -> list[Any]:
    handles: list[Any] = [
        Patch(facecolor=BOUND_COLOR, alpha=BOUND_ALPHA, label="AR(1) tube"),
        Line2D([0], [0], color=GOOD_COLOR, linewidth=1.4, label="Within bounds"),
    ]
    if has_violations:
        handles.append(Line2D([0], [0], color=BAD_COLOR, linewidth=1.4, label="Violation"))
    if extra_handles:
        handles.extend(extra_handles)
    return handles


# ---------------------------------------------------------------------------
# Per-regime demand figure
# ---------------------------------------------------------------------------

def plot_demand_for_regime(
    regime_name: str,
    rows: list[dict[str, Any]],
    D_ref: float,
    demand_shape: np.ndarray,
    coverage: float,
    out_dir: Path,
) -> None:
    T = len(demand_shape)
    t = np.arange(T)
    t_innov = np.arange(1, T)  # AR(1) innovations are at t=1..T-1

    r = rows[0]
    mu_D, sigma_D, rho_D = float(r["mu_D"]), float(r["sigma_D"]), float(r["rho_D"])
    ar1_ref, half_width = demand_ar1_tube(mu_D, sigma_D, rho_D, demand_shape, D_ref, T, coverage)
    hw0 = demand_t0_band(mu_D, sigma_D, demand_shape, D_ref, T, coverage)
    ref0 = D_ref * mu_D * demand_shape[0]

    n_rows = 3 if USE_WASSERSTEIN_SUPPORT_SET else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=False)

    has_viol = False
    for row in rows:
        D = parse_profile(row["demand_profile"])
        viol = demand_violated(D, mu_D, sigma_D, rho_D, demand_shape, D_ref, T, coverage)
        color = BAD_COLOR if viol else GOOD_COLOR
        has_viol = has_viol or viol

        # Panel 0: raw demand profile
        axes[0].plot(t, D, color=color, alpha=ALPHA_LINE, linewidth=0.9)

        # Panel 1: whitened AR(1) innovations (centred at 0)
        if T > 1:
            whitened = D[1:] - rho_D * D[:-1] - ar1_ref
            axes[1].plot(t_innov, whitened, color=color, alpha=ALPHA_LINE, linewidth=0.9)

        # Panel 2: t=0 deviation from reference
        axes[2].scatter([0], [D[0] - ref0], color=color, alpha=0.5, s=12, zorder=3)

    # Bounds on panel 0 (level box for legacy, physical [0, inf) for Wasserstein)
    if not USE_WASSERSTEIN_SUPPORT_SET:
        lb_pw, ub_pw = demand_pointwise_bounds(mu_D, sigma_D, rho_D, demand_shape, D_ref)
        _add_shading(axes[0], t, lb_pw, ub_pw, label="Level box")
    axes[0].set_title(f"Demand profiles — regime: {regime_name}")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].grid(True, alpha=0.2)

    # Bounds on panel 1 (symmetric ±half_width around 0)
    if T > 1:
        _add_shading(
            axes[1], t_innov,
            -half_width, half_width,
            label=f"AR(1) tube (kappa={_kappa_ar1(T, coverage):.2f})",
        )
        axes[1].axhline(0.0, color=BOUND_COLOR, linewidth=1.2, linestyle="-")
    axes[1].set_title("Whitened AR(1) innovations: D[t] - rho*D[t-1] - reference (t>=1)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Whitened innovation (MW)")
    axes[1].grid(True, alpha=0.2)

    # Bounds on panel 2 (t=0 band)
    axes[2].axhline(0.0, color=BOUND_COLOR, linewidth=1.2, linestyle="-")
    axes[2].axhspan(-hw0, hw0, color=T0_COLOR, alpha=T0_ALPHA,
                    label=f"t=0 band (kappa={_kappa_ar1(T, coverage):.2f})")
    axes[2].set_title(f"t=0 deviation from reference (D[0] - D_ref*mu_D*shape[0])")
    axes[2].set_ylabel("Deviation (MW)")
    axes[2].set_xticks([0])
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(fontsize=8)

    axes[0].legend(handles=_legend_handles(has_viol), fontsize=8)
    axes[1].legend(handles=_legend_handles(has_viol), fontsize=8)

    support_label = "Wasserstein" if USE_WASSERSTEIN_SUPPORT_SET else "Legacy"
    fig.suptitle(
        f"Demand support-set check [{support_label}]  |  regime={regime_name}  "
        f"mu_D={mu_D:.3f}  sigma_D={sigma_D:.4f}  rho_D={rho_D:.3f}  n={len(rows)}",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"demand_{regime_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Per-regime wind figure
# ---------------------------------------------------------------------------

def plot_wind_for_regime(
    regime_name: str,
    rows: list[dict[str, Any]],
    wind_generators: list[dict[str, Any]],
    wind_shape_fn: Any,
    coverage: float,
    out_dir: Path,
) -> None:
    if not wind_generators:
        return

    r = rows[0]
    mu_W, sigma_W, rho_W, peak_W = (
        float(r["mu_W"]), float(r["sigma_W"]), float(r["rho_W"]), float(r["peak_W"])
    )
    T = len(parse_profile(r["demand_profile"]))
    t = np.arange(T)
    t_innov = np.arange(1, T)
    wind_shape = wind_shape_fn(T, peak_W)

    n_gen = len(wind_generators)
    fig, axes = plt.subplots(n_gen * 3, 1, figsize=(10, 3.5 * n_gen * 3), sharex=False)
    axes = np.asarray(axes).flatten()

    for g_idx, gen_info in enumerate(wind_generators):
        cap = gen_info["capacity"]
        block_cols = gen_info["block_profile_columns"]
        ax_pw = axes[g_idx * 3]
        ax_ar1 = axes[g_idx * 3 + 1]
        ax_t0 = axes[g_idx * 3 + 2]

        ar1_ref, half_width = wind_ar1_tube(mu_W, sigma_W, rho_W, wind_shape, cap, T, coverage)
        ka = _kappa_ar1(T, coverage)
        hw0 = ka * cap * sigma_W
        ref0 = cap * mu_W * wind_shape[0]

        has_viol = False
        for row in rows:
            P = sum(parse_profile(row[col]) for col in block_cols)
            viol = wind_violated(P, mu_W, sigma_W, rho_W, wind_shape, cap, T, coverage)
            color = BAD_COLOR if viol else GOOD_COLOR
            has_viol = has_viol or viol

            ax_pw.plot(t, P, color=color, alpha=ALPHA_LINE, linewidth=0.9)
            if T > 1:
                whitened = P[1:] - rho_W * P[:-1] - ar1_ref
                ax_ar1.plot(t_innov, whitened, color=color, alpha=ALPHA_LINE, linewidth=0.9)
            ax_t0.scatter([0], [P[0] - ref0], color=color, alpha=0.5, s=12, zorder=3)

        # Profile bounds
        if not USE_WASSERSTEIN_SUPPORT_SET:
            lb_pw, ub_pw = wind_pointwise_bounds(mu_W, sigma_W, rho_W, wind_shape, cap)
            _add_shading(ax_pw, t, lb_pw, ub_pw, label="Level box")
        else:
            ax_pw.axhline(0.0, color=BOUND_COLOR, linewidth=0.8, linestyle="--")
            ax_pw.axhline(cap, color=BOUND_COLOR, linewidth=0.8, linestyle="--",
                          label=f"Physical cap ({cap:.0f} MW)")
        ax_pw.set_title(f"{gen_info['name']} wind capacity  (cap={cap:.0f} MW)")
        ax_pw.set_ylabel("MW")
        ax_pw.grid(True, alpha=0.2)
        ax_pw.legend(handles=_legend_handles(has_viol), fontsize=8)

        # AR(1) innovation bounds
        if T > 1:
            _add_shading(ax_ar1, t_innov, -half_width, half_width,
                         label=f"AR(1) tube (kappa={ka:.2f})")
            ax_ar1.axhline(0.0, color=BOUND_COLOR, linewidth=1.2, linestyle="-")
        ax_ar1.set_title(f"{gen_info['name']} whitened AR(1) innovations (t>=1)")
        ax_ar1.set_ylabel("Whitened innovation (MW)")
        ax_ar1.grid(True, alpha=0.2)
        ax_ar1.legend(handles=_legend_handles(has_viol), fontsize=8)

        # t=0 band
        ax_t0.axhline(0.0, color=BOUND_COLOR, linewidth=1.2, linestyle="-")
        ax_t0.axhspan(-hw0, hw0, color=T0_COLOR, alpha=T0_ALPHA,
                      label=f"t=0 band (kappa={ka:.2f})")
        ax_t0.set_title(f"{gen_info['name']} t=0 deviation from reference")
        ax_t0.set_ylabel("Deviation (MW)")
        ax_t0.set_xticks([0])
        ax_t0.grid(True, alpha=0.2)
        ax_t0.legend(fontsize=8)

    axes[-1].set_xlabel("Time step")
    support_label = "Wasserstein" if USE_WASSERSTEIN_SUPPORT_SET else "Legacy"
    fig.suptitle(
        f"Wind support-set check [{support_label}]  |  regime={regime_name}  "
        f"mu_W={mu_W:.3f}  sigma_W={sigma_W:.4f}  rho_W={rho_W:.3f}  peak_W={peak_W:.1f}  "
        f"n={len(rows)}",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"wind_{regime_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_violation_summary(
    scenarios_df: pd.DataFrame,
    D_ref: float,
    wind_generators: list[dict[str, Any]],
    coverage: float,
) -> None:
    support_label = "Wasserstein AR(1)-only" if USE_WASSERSTEIN_SUPPORT_SET else "Legacy (level-box + AR(1))"
    print(f"\n=== Support-set violation summary [{support_label}, coverage={coverage}] ===")
    total = len(scenarios_df)
    demand_violated_n = 0
    wind_violated_n = 0

    for _, row in scenarios_df.iterrows():
        mu_D = float(row["mu_D"])
        sigma_D = float(row["sigma_D"])
        rho_D = float(row["rho_D"])
        mu_W = float(row["mu_W"])
        sigma_W = float(row["sigma_W"])
        rho_W = float(row["rho_W"])
        peak_W = float(row["peak_W"])

        D = parse_profile(row["demand_profile"])
        T = len(D)
        demand_shape = ScenarioManager._build_demand_shape(T)
        wind_shape = ScenarioManager._build_wind_shape(T, peak_W)

        if demand_violated(D, mu_D, sigma_D, rho_D, demand_shape, D_ref, T, coverage):
            demand_violated_n += 1

        for gen_info in wind_generators:
            P = sum(parse_profile(row[col]) for col in gen_info["block_profile_columns"])
            if wind_violated(P, mu_W, sigma_W, rho_W, wind_shape, gen_info["capacity"], T, coverage):
                wind_violated_n += 1
                break

    print(f"  Total scenarios : {total}")
    print(f"  Demand violated : {demand_violated_n} / {total}")
    print(f"  Wind violated   : {wind_violated_n} / {total}")
    if demand_violated_n == 0 and wind_violated_n == 0:
        print("  All scenarios within support set.")
    else:
        pct = 100.0 * max(demand_violated_n, wind_violated_n) / total
        print(f"  WARNING: {pct:.1f}% of scenarios outside support set.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCENARIO_CSV.exists():
        raise FileNotFoundError(
            f"Scenario CSV not found: {SCENARIO_CSV}\n"
            "Run the DRO pipeline with run_scenario_generation=True first."
        )

    scenarios_df = pd.read_csv(SCENARIO_CSV)
    print(f"Loaded {len(scenarios_df)} scenarios from {SCENARIO_CSV}")
    support_label = "Wasserstein" if USE_WASSERSTEIN_SUPPORT_SET else "Legacy"
    print(f"Support set mode: {support_label}  (coverage={JOINT_COVERAGE})")

    manager = ScenarioManager(CASE)
    D_ref = float(manager.base_case["demand"])

    wind_generators: list[dict[str, Any]] = []
    for gen in manager.physical_generators:
        if not bool(gen["is_wind"]):
            continue
        gen_name = gen["physical_name"]
        cap = float(gen["pmax"])
        block_cols = [
            f"{b['block_name']}_profile"
            for b in manager.blocks
            if b["physical_name"] == gen_name and bool(b["is_wind"])
            and f"{b['block_name']}_profile" in scenarios_df.columns
        ]
        if block_cols:
            wind_generators.append({"name": gen_name, "capacity": cap, "block_profile_columns": block_cols})

    print(f"Wind generators: {[g['name'] for g in wind_generators]}")

    required = ["mu_D", "sigma_D", "rho_D", "mu_W", "sigma_W", "rho_W", "peak_W", "demand_profile"]
    missing = [c for c in required if c not in scenarios_df.columns]
    if missing:
        raise ValueError(f"Scenarios CSV missing columns: {missing}. Re-generate scenarios.")

    print_violation_summary(scenarios_df, D_ref, wind_generators, JOINT_COVERAGE)

    regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())
    print(f"Regimes: {regimes}\n")

    for regime_name in regimes:
        regime_df = scenarios_df[scenarios_df["regime"].astype(str) == regime_name]
        rows = [row.to_dict() for _, row in regime_df.iterrows()]
        print(f"Plotting '{regime_name}' ({len(rows)} scenarios)...")

        T = len(parse_profile(rows[0]["demand_profile"]))
        demand_shape = ScenarioManager._build_demand_shape(T)

        plot_demand_for_regime(
            regime_name, rows, D_ref, demand_shape, JOINT_COVERAGE, OUT_DIR
        )
        plot_wind_for_regime(
            regime_name, rows, wind_generators, ScenarioManager._build_wind_shape,
            JOINT_COVERAGE, OUT_DIR,
        )

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
