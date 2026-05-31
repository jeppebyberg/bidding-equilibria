"""
Empirical validation of the AR(1) innovation tube coverage in support_set.py.

For each regime, draws N fresh trajectories (not used in any model construction),
whitens the innovations, and checks what fraction of whole trajectories land entirely
within the tube. The target is 0.95 per regime.

Usage:
    .venv\\Scripts\\python.exe driver\\validate_tube_coverage.py [--N 10000] [--horizon 8] [--seed 99]

NOTE: _is_within_support_set() in scenario_generator.py has the same Sidak Šidák bug
that was fixed here (it divides by T rather than T-1). Do not use that function to
generate "support-set enforced" scenarios (enforce_support_set=True) until it is
corrected to also pass T-1. This script bypasses it and checks coverage directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as _norm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.scenarios.scenario_generator import RegimeParameters, ScenarioManager
from models.PoA.poa_model.support_set import AR1_JOINT_COVERAGE, _ar1_kappa


# ---------------------------------------------------------------------------
# Core whitening helpers
# ---------------------------------------------------------------------------


def _whitened_demand_innovations(
    demand_profiles: np.ndarray,
    D_ref: float,
    mu_D: float,
    rho_D: float,
    demand_shape: np.ndarray,
) -> np.ndarray:
    """Return xi_hat[n, t] = D[n,t] - rho*D[n,t-1] - ar1_ref_t for interior t=1,...,T-1.

    Under unclipped AR(1): xi_hat[n, t] = D_ref * sigma_D * z[n, t], so the distribution
    should be N(0, (D_ref * sigma_D)^2).

    Args:
        demand_profiles: shape (N, T) in MW.
        D_ref: reference demand in MW.
        mu_D: demand mean multiplier (dimensionless).
        rho_D: AR(1) persistence.
        demand_shape: mean-one shape profile, shape (T,).

    Returns:
        xi_hat: shape (N, T-1), in MW.
    """
    # ar1_ref[t] = D_ref * mu_D * (shape[t] - rho * shape[t-1]) for t=1,...,T-1
    ar1_ref = D_ref * mu_D * (demand_shape[1:] - rho_D * demand_shape[:-1])  # (T-1,)
    xi_hat = demand_profiles[:, 1:] - rho_D * demand_profiles[:, :-1] - ar1_ref[np.newaxis, :]
    return xi_hat  # (N, T-1)


def _whitened_wind_innovations(
    wind_profiles: np.ndarray,
    capacity: float,
    mu_W: float,
    rho_W: float,
    wind_shape: np.ndarray,
) -> np.ndarray:
    """Return xi_hat[n, t] for wind. Under unclipped AR(1): N(0, (cap * sigma_W)^2).

    Args:
        wind_profiles: shape (N, T) in MW.
        capacity: nominal wind capacity in MW.
        mu_W: wind mean multiplier (dimensionless).
        rho_W: AR(1) persistence.
        wind_shape: mean-one shape profile, shape (T,).

    Returns:
        xi_hat: shape (N, T-1), in MW.
    """
    ar1_ref = capacity * mu_W * (wind_shape[1:] - rho_W * wind_shape[:-1])  # (T-1,)
    xi_hat = wind_profiles[:, 1:] - rho_W * wind_profiles[:, :-1] - ar1_ref[np.newaxis, :]
    return xi_hat  # (N, T-1)


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def _report_coverage(
    xi_hat: np.ndarray,
    scale: float,
    kappa_t1: float,
    kappa_homog: float,
    target: float,
    series_label: str,
    regime_name: str,
) -> Dict[str, object]:
    """Compute and print joint and per-step coverage, return results dict.

    Args:
        xi_hat: whitened innovations (N, T-1) in the same units as scale.
        scale: tube half-width unit (D_ref * sigma_D or cap * sigma_W).
        kappa_t1: kappa used for t=1 constraint.
        kappa_homog: kappa used for t>=2 constraints.
        target: target joint coverage (0.95).
        series_label: "Demand" or "Wind <gen_name>".
        regime_name: regime identifier for display.
    """
    N, num_constraints = xi_hat.shape
    T = num_constraints + 1  # total timesteps

    # Per-constraint kappas: t=1 (index 0) uses kappa_t1, rest use kappa_homog.
    kappas = np.full(num_constraints, kappa_homog)
    kappas[0] = kappa_t1  # t=1 maps to index 0 in the T-1 array

    bands = kappas * scale  # (T-1,) half-widths in MW (or same units as xi_hat)

    # Per-step containment: fraction of draws where |xi_hat| <= band at each t.
    contained_per_step = np.abs(xi_hat) <= bands[np.newaxis, :]  # (N, T-1) bool
    per_step_coverage = contained_per_step.mean(axis=0)  # (T-1,)

    # Joint containment: all T-1 innovations within band for a given draw.
    joint_contained = contained_per_step.all(axis=1)  # (N,) bool
    empirical_joint = joint_contained.mean()

    # 95% CI for empirical coverage under binomial.
    se = np.sqrt(empirical_joint * (1.0 - empirical_joint) / N)
    ci_lo = empirical_joint - 2.0 * se
    ci_hi = empirical_joint + 2.0 * se

    # Three-way verdict:
    #   PASS         -- target within 2-sigma CI
    #   CONSERVATIVE -- empirical significantly above target (fine; tube is tight)
    #   FAIL         -- empirical significantly below target (bad)
    if ci_lo <= target <= ci_hi:
        verdict = "PASS"
    elif ci_lo > target:
        verdict = "CONSERVATIVE (above target, tube is tight)"
    else:
        verdict = "FAIL (empirical below target)"

    print(f"\n  {series_label} | regime: {regime_name}")
    print(f"    T={T}, num_constraints={num_constraints}, N={N}")
    print(f"    kappa_t1 (t=1) : {kappa_t1:.6f}")
    print(f"    kappa_homog    : {kappa_homog:.6f}")
    print(f"    (kappa_t1 == kappa_homog: {np.isclose(kappa_t1, kappa_homog)}  -- AR(1) cancellation)")
    print(f"    target joint   : {target:.4f}")
    print(f"    empirical joint: {empirical_joint:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  {verdict}")
    print(f"    per-step coverage by t (t=1,...,T-1):")
    for idx, cov in enumerate(per_step_coverage):
        marker = " <-- t=1 (non-stationary init)" if idx == 0 else ""
        print(f"      t={idx + 1}: {cov:.4f}{marker}")

    passed = ci_hi >= target  # fails only when empirical is significantly below target
    return {
        "series": series_label,
        "regime": regime_name,
        "N": N,
        "num_constraints": num_constraints,
        "kappa_t1": kappa_t1,
        "kappa_homog": kappa_homog,
        "target_joint": target,
        "empirical_joint": empirical_joint,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "per_step_coverage": per_step_coverage,
        "verdict": verdict,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# Histogram plotting
# ---------------------------------------------------------------------------


def _plot_histogram(
    xi_hat: np.ndarray,
    scale: float,
    kappa: float,
    series_label: str,
    regime_name: str,
    save_dir: Optional[Path],
    show: bool = True,
) -> None:
    """Plot histogram of all whitened innovations and overlay N(0, scale^2).

    Also marks the +/- kappa * scale tube boundary.
    """
    flat = xi_hat.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))

    # Histogram of whitened innovations.
    ax.hist(flat, bins=80, density=True, alpha=0.65, color="steelblue", label="Whitened innovations")

    # Theoretical N(0, scale^2) overlay.
    x = np.linspace(flat.min(), flat.max(), 400)
    ax.plot(x, _norm.pdf(x, loc=0, scale=scale), "r-", linewidth=2.0,
            label=f"N(0, scale^2)  scale={scale:.3f}")

    # Tube boundaries.
    for sign, lbl in [(+1, f"+kappa={kappa:.3f}"), (-1, f"-kappa={kappa:.3f}")]:
        ax.axvline(sign * kappa * scale, color="orange", linestyle="--", linewidth=1.5, label=lbl)

    ax.set_title(f"Whitened innovations -- {series_label} | {regime_name}")
    ax.set_xlabel("xi_hat (MW)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"hist_{series_label.replace(' ', '_')}_{regime_name}.png"
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-regime validation
# ---------------------------------------------------------------------------


def _validate_regime(
    manager: ScenarioManager,
    regime: RegimeParameters,
    horizon: int,
    N: int,
    rng: np.random.Generator,
    save_dir: Optional[Path],
    target: float = AR1_JOINT_COVERAGE,
    show_plots: bool = True,
) -> List[Dict[str, object]]:
    """Draw N trajectories and validate coverage for demand and all wind generators."""
    demand_shape = ScenarioManager._build_demand_shape(horizon)
    wind_shape = ScenarioManager._build_wind_shape(horizon, regime.peak_W)
    D_ref = float(manager.base_case["demand"])

    # Draw N trajectories.
    all_demand: List[List[float]] = []
    all_wind: Dict[str, List[List[float]]] = {
        manager.physical_generators[i]["physical_name"]: []
        for i in manager.wind_generator_indices
    }

    print(f"\n  Drawing {N} trajectories for regime '{regime.name}' (T={horizon}) ...")
    for _ in range(N):
        d, w, _ = manager._simulate_single_scenario(regime=regime, rng=rng)
        all_demand.append(d)
        for gen_idx in manager.wind_generator_indices:
            gen_name = manager.physical_generators[gen_idx]["physical_name"]
            all_wind[gen_name].append(w[gen_name])

    demand_arr = np.array(all_demand, dtype=float)  # (N, T)

    num_ar1_constraints = horizon - 1
    kappa_homog = _ar1_kappa(num_ar1_constraints, target)
    # t=1 kappa: variance of D_1 - rho*D_0 - ar1_ref_1 equals (D_ref * sigma_D)^2
    # for AR(1) regardless of initialization. Computed explicitly here for auditability.
    # std_t1 = D_ref * sigma_D (same as homogeneous case).
    kappa_t1 = _ar1_kappa(num_ar1_constraints, target)  # equals kappa_homog

    results = []

    # --- Demand ---
    xi_demand = _whitened_demand_innovations(
        demand_arr, D_ref, regime.mu_D, regime.rho_D, demand_shape
    )
    scale_D = D_ref * regime.sigma_D  # innovation sigma in MW

    _plot_histogram(xi_demand, scale_D, kappa_homog, "Demand", regime.name, save_dir, show=show_plots)

    # Diagnose floor activations (should be essentially zero for demand).
    floor_active = (demand_arr < 1e-6).any(axis=1)
    if floor_active.any():
        print(f"  WARNING: {floor_active.sum()}/{N} demand draws hit the max(...,0) floor.")
        print("  The whitened innovations are not pure N(0, scale^2) for those draws.")

    r = _report_coverage(xi_demand, scale_D, kappa_t1, kappa_homog, target, "Demand", regime.name)
    results.append(r)

    # --- Wind generators ---
    for gen_idx in manager.wind_generator_indices:
        gen = manager.physical_generators[gen_idx]
        gen_name = gen["physical_name"]
        cap = float(gen["pmax"])
        wind_arr = np.array(all_wind[gen_name], dtype=float)  # (N, T)

        xi_wind = _whitened_wind_innovations(
            wind_arr, cap, regime.mu_W, regime.rho_W, wind_shape
        )
        scale_W = cap * regime.sigma_W  # innovation sigma in MW

        # Diagnose clip activations (upper and lower).
        raw_factor = (wind_arr / cap)  # (N, T)
        upper_clip_rate = (raw_factor > 1.0 - 1e-6).mean()
        lower_clip_rate = (raw_factor < 1e-6).mean()
        if upper_clip_rate > 1e-4 or lower_clip_rate > 1e-4:
            print(
                f"  INFO ({gen_name}): upper clip rate {upper_clip_rate:.4f}, "
                f"lower clip rate {lower_clip_rate:.4f}."
            )
            print(
                "  Clipped timesteps break exact N(0, scale^2) for xi_hat at those steps. "
                "Upper clip makes xi_hat smaller (easier to contain), so empirical coverage "
                ">= theoretical. Lower clip is negligible for these regimes."
            )

        _plot_histogram(xi_wind, scale_W, kappa_homog, f"Wind {gen_name}", regime.name, save_dir, show=show_plots)

        r = _report_coverage(
            xi_wind, scale_W, kappa_t1, kappa_homog, target, f"Wind {gen_name}", regime.name
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate AR(1) innovation tube coverage.")
    p.add_argument("--N", type=int, default=10_000, help="Trajectories per regime (default 10000)")
    p.add_argument("--horizon", type=int, default=None,
                   help="Override horizon (default: from base_case)")
    p.add_argument("--seed", type=int, default=99, help="RNG seed (default 99)")
    p.add_argument("--regime-set", type=str, default="policy_training",
                   help="Regime set from regime_definitions.yaml (default: policy_training)")
    p.add_argument("--regimes", nargs="+", default=None,
                   help="Subset of regime names to validate (default: all in set)")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Directory to save histogram plots (default: no save)")
    p.add_argument("--no-plots", action="store_true", help="Skip all plots")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    manager = ScenarioManager(base_case_reference="base_test_case")
    config = manager.load_regime_configuration(regime_set=args.regime_set)

    horizon = args.horizon if args.horizon is not None else manager.base_case["time_steps"]
    # Temporarily override time_steps for drawing purposes.
    manager.base_case["time_steps"] = horizon

    num_ar1_constraints = horizon - 1
    kappa_homog = _ar1_kappa(num_ar1_constraints, AR1_JOINT_COVERAGE)

    print("=" * 70)
    print("AR(1) INNOVATION TUBE COVERAGE VALIDATION")
    print("=" * 70)
    print(f"  Regime set       : {config['name']}")
    print(f"  Horizon T        : {horizon}")
    print(f"  num_constraints  : {num_ar1_constraints}  (= T - 1, matching time_steps_minus_1)")
    print(f"  Target coverage  : {AR1_JOINT_COVERAGE:.4f}")
    print(f"  kappa_homog      : {kappa_homog:.6f}  (all t, since kappa_t1 == kappa_homog)")
    print(f"  N per regime     : {args.N}")

    rng = np.random.default_rng(args.seed)
    save_dir = Path(args.save_dir) if args.save_dir else None

    show_plots = not args.no_plots

    regimes_to_run = config["regimes"]
    if args.regimes is not None:
        regimes_to_run = [r for r in regimes_to_run if r.name in args.regimes]
        if not regimes_to_run:
            print(f"ERROR: none of {args.regimes} found in regime set '{config['name']}'")
            sys.exit(1)

    all_results: List[Dict[str, object]] = []
    for regime in regimes_to_run:
        results = _validate_regime(manager, regime, horizon, args.N, rng, save_dir, show_plots=show_plots)
        all_results.extend(results)

    # Summary table.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Series':<25} {'Regime':<30} {'Empirical':>10} {'Target':>8} {'Pass':>6}"
    print(header)
    print("-" * len(header))
    all_pass = True
    for r in all_results:
        empirical = float(r["empirical_joint"])
        target = float(r["target_joint"])
        ci_lo = float(r["ci_lo"])
        ci_hi = float(r["ci_hi"])
        passed = bool(r["pass"])
        all_pass = all_pass and passed
        mark = str(r.get("verdict", "OK" if passed else "FAIL")).split(" ")[0]
        print(
            f"  {str(r['series']):<23} {str(r['regime']):<30} "
            f"{empirical:.4f} [{ci_lo:.3f},{ci_hi:.3f}]  {target:.4f}  {mark}"
        )

    print("-" * len(header))
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print()
    print("Interpretation:")
    print("  PASS = empirical joint coverage within 2-sigma of 0.95 (normal for iid N innovations).")
    print("  If demand fails: check max(...,0) floor activations (printed above).")
    print("  If wind fails above 0.95: expected -- upper clip makes containment easier (conservative).")
    print("  If wind fails below 0.95: unexpected -- check clip rate and sigma_W magnitude.")
    print()
    print("Coverage claim scope: tube constraints only (not level box or budget).")
    print("See support_set.py module docstring for the full scoping note.")


if __name__ == "__main__":
    main()
