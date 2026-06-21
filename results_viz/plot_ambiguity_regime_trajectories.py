# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-04
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Plot the ambiguity regime, optimized state trajectories, and fresh scenario coverage.

Coverage scope note: the 95% joint-coverage claim applies to the AR(1) innovation tube
alone. A fresh trajectory from regime r lands entirely within the tube with probability
0.95. The level box (demand_lower/upper) and budget are additional conditions on U(r)
and are shown for context but do not carry a closed-form coverage statement.

The AR(1) tube is visualized in level space as reference +/- kappa * scale. This is
a reference-linearized approximation: the true tube is conditional on D[t-1], so the
center shifts with D[t-1]. The width (2 * kappa * scale) is exact; only the centering
is approximate. See plot_fresh_scenario_innovations for the exact containment view.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.PoA.poa_model.support_set import AR1_JOINT_COVERAGE, _ar1_kappa

DEFAULT_OUTPUT_DIR = Path("results_viz/figures/ambiguity_regime_trajectories")

# Nominal kappa for level box (1.96 = 95% marginal CI, stationary sigma).
_LEVEL_BOX_KAPPA = 1.96


# ---------------------------------------------------------------------------
# Basic I/O helpers
# ---------------------------------------------------------------------------


def _series(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values])


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_default_result_path(results_dir: Path) -> Path:
    candidates = _find_ambiguity_result_paths(results_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No PoA result with an 'ambiguity_set' block found in {results_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_ambiguity_result_paths(results_dir: Path) -> list[Path]:
    candidates = sorted(results_dir.glob("poa_optimization*.json"))
    out: list[Path] = []
    for c in candidates:
        try:
            payload = _load_json(c)
        except Exception:
            continue
        if isinstance(payload.get("ambiguity_set"), dict):
            out.append(c)
    return out


def _resolve_result_paths(
    result_paths: list[Path],
    results_dir: Path,
    plot_all_matching_results: bool,
) -> list[Path]:
    explicit = [Path(p) for p in result_paths]
    if explicit:
        missing = [p for p in explicit if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Result path(s) not found: {', '.join(str(p) for p in missing)}")
        return explicit
    if plot_all_matching_results:
        paths = _find_ambiguity_result_paths(results_dir)
        if not paths:
            raise FileNotFoundError(
                f"No PoA result with an 'ambiguity_set' block found in {results_dir}"
            )
        return paths
    return [_find_default_result_path(results_dir)]


def _wind_generator_names(results: dict[str, Any]) -> list[str]:
    generators = results.get("generators", {}) or {}
    return [
        str(name)
        for name, gen in generators.items()
        if bool(gen.get("is_wind"))
    ]


def _format_regime_label(selected_regime: dict[str, Any]) -> str:
    parts = []
    for key in ("mu_D", "sigma_D", "mu_W", "sigma_W"):
        v = selected_regime.get(key)
        if v is not None:
            parts.append(f"{key}={float(v):.3g}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# AR(1) tube helpers
# ---------------------------------------------------------------------------


def _get_ambiguity_params(
    results: dict[str, Any],
) -> Optional[dict[str, float]]:
    """Extract fixed + selected regime params needed for AR(1) tube, or None."""
    amb = results.get("ambiguity_set")
    if not isinstance(amb, dict):
        return None
    fixed = amb.get("fixed_parameters") or {}
    sel = amb.get("selected_regime") or {}
    needed = {
        "D_ref": fixed.get("D_ref"),
        "rho_D": fixed.get("rho_D"),
        "rho_W": fixed.get("rho_W"),
        "tau_W": fixed.get("tau_W"),
        "mu_D": sel.get("mu_D"),
        "sigma_D": sel.get("sigma_D"),
        "mu_W": sel.get("mu_W"),
        "sigma_W": sel.get("sigma_W"),
    }
    if any(v is None for v in needed.values()):
        return None
    return {k: float(v) for k, v in needed.items()}  # type: ignore[arg-type]


def _ar1_tube_demand_bands(
    reference: np.ndarray,
    D_ref: float,
    sigma_D: float,
    kappa_ar1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) tube in level space, centered on the reference trajectory.

    Width = 2 * kappa_ar1 * D_ref * sigma_D (exact innovation-sigma scale).
    Centering is approximate: assumes D[t-1] is on the reference.
    """
    hw = kappa_ar1 * D_ref * sigma_D
    return reference - hw, reference + hw


def _ar1_tube_wind_bands(
    reference: np.ndarray,
    capacity: float,
    sigma_W: float,
    kappa_ar1: float,
) -> tuple[np.ndarray, np.ndarray]:
    hw = kappa_ar1 * capacity * sigma_W
    return reference - hw, reference + hw


def _cumulative_demand_bands(
    reference: np.ndarray,
    D_ref: float,
    sigma_D: float,
    rho_D: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative AR(1) band: half-width = kappa * D_ref * sigma_D * (1 - rho^{t+1}) / (1 - rho)."""
    T = len(reference)
    hw = np.array([kappa * D_ref * sigma_D * (1.0 - rho_D**(t+1)) / (1.0 - rho_D) for t in range(T)])
    return reference - hw, reference + hw


def _stationary_demand_bands(
    reference: np.ndarray,
    D_ref: float,
    sigma_D: float,
    rho_D: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary AR(1) band: half-width = kappa * D_ref * sigma_D / sqrt(1 - rho^2)."""
    hw = kappa * D_ref * sigma_D / np.sqrt(1.0 - rho_D**2)
    return reference - hw, reference + hw


def _cumulative_wind_bands(
    reference: np.ndarray,
    capacity: float,
    sigma_W: float,
    rho_W: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative AR(1) band for wind."""
    T = len(reference)
    hw = np.array([kappa * capacity * sigma_W * (1.0 - rho_W**(t+1)) / (1.0 - rho_W) for t in range(T)])
    return reference - hw, reference + hw


def _stationary_wind_bands(
    reference: np.ndarray,
    capacity: float,
    sigma_W: float,
    rho_W: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary AR(1) band for wind."""
    hw = kappa * capacity * sigma_W / np.sqrt(1.0 - rho_W**2)
    return reference - hw, reference + hw


def _demand_ar1_contained(
    demand: np.ndarray,
    D_ref: float,
    mu_D: float,
    rho_D: float,
    sigma_D: float,
    demand_shape: np.ndarray,
    kappa_ar1: float,
) -> bool:
    """True if every whitened demand innovation is within the tube."""
    if len(demand) < 2:
        return True
    ar1_ref = D_ref * mu_D * (demand_shape[1:] - rho_D * demand_shape[:-1])
    xi = demand[1:] - rho_D * demand[:-1] - ar1_ref
    return bool(np.all(np.abs(xi) <= kappa_ar1 * D_ref * sigma_D))


def _wind_ar1_contained(
    wind: np.ndarray,
    capacity: float,
    mu_W: float,
    rho_W: float,
    sigma_W: float,
    wind_shape: np.ndarray,
    kappa_ar1: float,
) -> bool:
    if len(wind) < 2:
        return True
    ar1_ref = capacity * mu_W * (wind_shape[1:] - rho_W * wind_shape[:-1])
    xi = wind[1:] - rho_W * wind[:-1] - ar1_ref
    return bool(np.all(np.abs(xi) <= kappa_ar1 * capacity * sigma_W))


# ---------------------------------------------------------------------------
# Enhanced existing plot: optimized trajectory with support set bounds
# ---------------------------------------------------------------------------


def plot_ambiguity_regime_trajectories(
    result_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
    extra_title_line: str | None = None,  # kept for backward compatibility; unused
) -> tuple[Path, Path]:
    """Plot the constraining support set and optimized trajectories as two figures.

    Figure 1 (demand): support band, reference, and the optimized demand trajectory.
    Figure 2 (wind):   a single shared support band (the wind support set is the
                       same for every wind generator), reference, and the optimized
                       trajectory of each wind generator overlaid.

    No title is drawn.
    """
    results = _load_json(result_path)
    ambiguity_set = results.get("ambiguity_set")
    if not isinstance(ambiguity_set, dict):
        raise ValueError(
            f"{result_path} does not contain an 'ambiguity_set' block. "
            "Run this script on a PoA result generated after the ambiguity-set refactor."
        )

    demand_block = ambiguity_set.get("demand", {}) or {}
    horizon = int(results["num_time_steps"])
    time = np.arange(horizon)

    demand = _series(results["demand_profile"])
    demand_reference = _series(demand_block["reference"])
    demand_lower = _series(demand_block["lower"])
    demand_upper = _series(demand_block["upper"])

    wind_names = _wind_generator_names(results)
    if not wind_names:
        raise ValueError("No wind generators found in result['generators']")
    wind_block = ambiguity_set.get("wind", {}) or {}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared figure size so the demand and wind figures have identical dimensions.
    # Sized to the A4 text width (~16 cm) so labels stay readable when included ~1:1.
    fig_size = (6.3, 3.9)

    # --- Figure 1: Demand support set + optimized trajectory ---
    fig_d, ax_d = plt.subplots(figsize=fig_size)
    ax_d.fill_between(
        time, demand_lower, demand_upper,
        color="tab:blue", alpha=0.20, label="Support set",
    )
    ax_d.plot(
        time, demand_reference,
        color="tab:blue", linestyle="--", linewidth=1.8, label="Reference",
    )
    ax_d.plot(
        time, demand,
        color="black", marker="o", linewidth=2.2, label="Optimized demand",
    )
    ax_d.set_xlabel("Time step", fontsize=14)
    ax_d.set_ylabel("MW", fontsize=14)
    ax_d.tick_params(labelsize=12)
    ax_d.grid(True, alpha=0.25)
    ax_d.legend(loc="best", fontsize=11)
    fig_d.tight_layout()
    demand_path = output_dir / f"{result_path.stem}_demand_support_set.png"
    fig_d.savefig(demand_path)
    if show:
        plt.show()
    plt.close(fig_d)

    # --- Figure 2: Shared wind support set + each generator's trajectory ---
    # The wind support set (reference/lower/upper) is identical across wind
    # generators, so a single band is drawn and the trajectories overlaid.
    missing = [g for g in wind_names if g not in wind_block]
    if missing:
        raise KeyError(f"ambiguity_set['wind'] is missing wind generator(s): {missing}")

    shared_amb = wind_block[wind_names[0]]
    wind_ref = _series(shared_amb["reference"])
    wind_lower = _series(shared_amb["lower"])
    wind_upper = _series(shared_amb["upper"])

    fig_w, ax_w = plt.subplots(figsize=fig_size)
    ax_w.fill_between(
        time, wind_lower, wind_upper,
        color="tab:green", alpha=0.20, label="Support set",
    )
    ax_w.plot(
        time, wind_ref,
        color="tab:green", linestyle="--", linewidth=1.8, label="Reference",
    )
    traj_colors = ["tab:orange", "tab:red", "tab:purple", "tab:brown", "tab:cyan"]
    for idx, gen_name in enumerate(wind_names):
        gen_profile = _series(results["generators"][gen_name]["physical_capacity_profile"])
        ax_w.plot(
            time, gen_profile,
            color=traj_colors[idx % len(traj_colors)],
            marker="o", linewidth=2.0, label=f"Optimized {gen_name}",
        )
    ax_w.set_xlabel("Time step", fontsize=14)
    ax_w.set_ylabel("MW", fontsize=14)
    ax_w.tick_params(labelsize=12)
    ax_w.grid(True, alpha=0.25)
    ax_w.legend(loc="best", fontsize=11)
    fig_w.tight_layout()
    wind_path = output_dir / f"{result_path.stem}_wind_support_set.png"
    fig_w.savefig(wind_path)
    if show:
        plt.show()
    plt.close(fig_w)

    return demand_path, wind_path


# ---------------------------------------------------------------------------
# New plot: fresh scenario coverage in level space
# ---------------------------------------------------------------------------


def plot_fresh_scenario_coverage(
    result_path: Path,
    N: int = 300,
    seed: int = 0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """Draw N fresh trajectories and show which lie inside the AR(1) innovation tube.

    Color code:
      - Blue (light): trajectory is inside the AR(1) tube for ALL interior time steps.
      - Red (brighter): trajectory escapes the tube at least once.
    The empirical joint coverage (blue fraction) should be ~0.95.

    The AR(1) tube band is shown as reference +/- kappa * scale (reference-linearized).
    The level box is shown for context (it is a separate, wider constraint).
    The optimized trajectory from the result is overlaid in black for comparison.
    """
    from config.scenarios.scenario_generator import RegimeParameters, ScenarioManager

    results = _load_json(result_path)
    ambiguity_set = results.get("ambiguity_set")
    if not isinstance(ambiguity_set, dict):
        raise ValueError(f"{result_path} does not contain an 'ambiguity_set' block.")

    params = _get_ambiguity_params(results)
    if params is None:
        raise ValueError(
            f"{result_path} is missing 'ambiguity_set.fixed_parameters'. "
            "Re-run the PoA pipeline to regenerate results with the current code."
        )

    horizon = int(results["num_time_steps"])
    kappa_ar1 = _ar1_kappa(horizon - 1)
    time = np.arange(horizon)

    demand_block = ambiguity_set["demand"]
    demand_shape = np.asarray(demand_block["shape"], dtype=float)
    demand_reference = _series(demand_block["reference"])
    demand_lower = _series(demand_block["lower"])
    demand_upper = _series(demand_block["upper"])
    demand_opt = _series(results["demand_profile"])

    wind_names = _wind_generator_names(results)
    wind_block = ambiguity_set.get("wind", {}) or {}

    # Build RegimeParameters from the JSON's selected regime + fixed params.
    regime = RegimeParameters(
        name="selected",
        n_scenarios=N,
        mu_D=params["mu_D"],
        rho_D=params["rho_D"],
        sigma_D=params["sigma_D"],
        mu_W=params["mu_W"],
        peak_W=params["tau_W"],
        rho_W=params["rho_W"],
        sigma_W=params["sigma_W"],
    )

    reference_case = str(results.get("reference_case", "base_test_case"))
    manager = ScenarioManager(base_case_reference=reference_case)
    manager.base_case["time_steps"] = horizon
    rng = np.random.default_rng(seed)

    # Draw N fresh trajectories.
    demand_trajectories: list[np.ndarray] = []
    wind_trajectories: dict[str, list[np.ndarray]] = {g: [] for g in wind_names}
    for _ in range(N):
        d, w, _ = manager._simulate_single_scenario(regime=regime, rng=rng)
        demand_trajectories.append(np.asarray(d, dtype=float))
        for g in wind_names:
            if g in w:
                wind_trajectories[g].append(np.asarray(w[g], dtype=float))

    # Check AR(1) tube containment for each trajectory.
    # A trajectory is "in the tube" if ALL of demand AND all wind generators are contained.
    def _demand_in(traj: np.ndarray) -> bool:
        return _demand_ar1_contained(
            traj, params["D_ref"], params["mu_D"], params["rho_D"],
            params["sigma_D"], demand_shape, kappa_ar1,
        )

    wind_shapes: dict[str, np.ndarray] = {}
    wind_caps: dict[str, float] = {}
    for g in wind_names:
        wind_shapes[g] = np.asarray(wind_block[g]["shape"], dtype=float)
        wind_caps[g] = float(wind_block[g].get("installed_capacity", 1.0))

    def _wind_in(traj: np.ndarray, g: str) -> bool:
        return _wind_ar1_contained(
            traj, wind_caps[g], params["mu_W"], params["rho_W"],
            params["sigma_W"], wind_shapes[g], kappa_ar1,
        )

    # Joint containment: all processes in tube.
    demand_contained = [_demand_in(d) for d in demand_trajectories]
    wind_contained: dict[str, list[bool]] = {
        g: [_wind_in(w, g) for w in wind_trajectories[g]]
        for g in wind_names
    }
    joint_contained = [
        demand_contained[n] and all(wind_contained[g][n] for g in wind_names)
        for n in range(N)
    ]
    # Per-process coverage (the 95% claim is per generator, not joint across all processes).
    demand_cov = float(np.mean(demand_contained))
    wind_cov_per: dict[str, float] = {g: float(np.mean(wind_contained[g])) for g in wind_names}
    # Joint across all processes (informational only; not the 95% claim).
    all_joint_cov = float(np.mean(joint_contained))

    # AR(1) tube level-space bands (reference-centered, exact width).
    d_tube_lo, d_tube_hi = _ar1_tube_demand_bands(
        demand_reference, params["D_ref"], params["sigma_D"], kappa_ar1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 1 + len(wind_names)
    fig, _axes = plt.subplots(n_rows, 1, figsize=(11, 3.4 * n_rows), sharex=True)
    axes = np.atleast_1d(_axes)

    # Build per-process summary string for the figure title.
    # The 95% claim is per generator; per-process coverages should each be ~0.95.
    per_proc_parts = [f"demand={demand_cov:.3f}"] + [f"{g}={wind_cov_per[g]:.3f}" for g in wind_names]
    per_proc_str = ",  ".join(per_proc_parts)
    def _cov_ci(cov: float) -> str:
        se_v = float(np.sqrt(cov * (1 - cov) / N))
        return f"{cov:.3f} [{cov-2*se_v:.3f}, {cov+2*se_v:.3f}]"

    fig.suptitle(
        f"Fresh Scenario Coverage -- AR(1) Tube per generator (kappa={kappa_ar1:.3f}, N={N})\n"
        f"{_format_regime_label(ambiguity_set.get('selected_regime', {}))}\n"
        f"Per-generator coverage (target {AR1_JOINT_COVERAGE:.2f}):  {per_proc_str}\n"
        f"(Joint across all generators: {all_joint_cov:.3f} -- not the 95% claim)",
        fontsize=9,
    )

    def _draw_process_ax(
        ax: Any,
        trajectories: list[np.ndarray],
        contained: list[bool],
        reference: np.ndarray,
        level_lower: np.ndarray,
        level_upper: np.ndarray,
        tube_lower: np.ndarray,
        tube_upper: np.ndarray,
        opt_trajectory: Optional[np.ndarray],
        process_color: str,
        process_label: str,
        process_cov: float,
    ) -> None:
        from matplotlib.lines import Line2D

        n_in = int(np.sum(contained))
        n_out = len(trajectories) - n_in
        proc_se = float(np.sqrt(process_cov * (1 - process_cov) / N))

        # Level box (wider, stationary sigma).
        ax.fill_between(
            time, level_lower, level_upper,
            color=process_color, alpha=0.13,
            label=f"Level box (stationary sigma, {_LEVEL_BOX_KAPPA:.2f}sigma)",
        )
        # AR(1) tube (narrower, innovation sigma).
        ax.fill_between(
            time, tube_lower, tube_upper,
            color="tab:orange", alpha=0.30,
            label=f"AR(1) tube (innovation sigma, 95% joint, kappa={kappa_ar1:.3f})",
        )
        # Draw trajectories: contained first (so out-of-tube are visible on top).
        for n, traj in enumerate(trajectories):
            if contained[n]:
                ax.plot(time, traj, color=process_color, alpha=0.08, linewidth=0.6)
        for n, traj in enumerate(trajectories):
            if not contained[n]:
                ax.plot(time, traj, color="tab:red", alpha=0.45, linewidth=0.9)

        ax.plot(
            time, reference,
            color=process_color, linestyle="--", linewidth=1.6, label="Reference",
        )
        if opt_trajectory is not None:
            ax.plot(
                time, opt_trajectory,
                color="black", marker="o", linewidth=2.2, markersize=5,
                label="Optimized trajectory",
            )

        # Dummy handles for in/out counts.
        extra_handles = [
            Line2D([0], [0], color=process_color, alpha=0.5, linewidth=1.5,
                   label=f"In tube: {n_in}/{N} ({n_in/N:.1%})"),
            Line2D([0], [0], color="tab:red", alpha=0.7, linewidth=1.5,
                   label=f"Out of tube: {n_out}/{N} ({n_out/N:.1%})"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + extra_handles, fontsize=7, loc="best")
        ax.set_title(
            f"{process_label}   "
            f"coverage = {process_cov:.3f} [{process_cov - 2*proc_se:.3f}, {process_cov + 2*proc_se:.3f}]"
            f"  (target {AR1_JOINT_COVERAGE:.2f})"
        )
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)

    # Demand row.
    _draw_process_ax(
        axes[0],
        demand_trajectories, demand_contained,
        demand_reference, demand_lower, demand_upper,
        d_tube_lo, d_tube_hi,
        demand_opt,
        "tab:blue", "Demand",
        demand_cov,
    )

    # Wind rows.
    for ax, g in zip(axes[1:], wind_names):
        w_ref = _series(wind_block[g]["reference"])
        w_lower = _series(wind_block[g]["lower"])
        w_upper = _series(wind_block[g]["upper"])
        w_tube_lo, w_tube_hi = _ar1_tube_wind_bands(
            w_ref, wind_caps[g], params["sigma_W"], kappa_ar1
        )
        w_opt = _series(results["generators"][g]["physical_capacity_profile"])
        _draw_process_ax(
            ax,
            wind_trajectories[g], wind_contained[g],
            w_ref, w_lower, w_upper,
            w_tube_lo, w_tube_hi,
            w_opt,
            "tab:green", f"Wind {g}",
            wind_cov_per[g],
        )

    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    output_path = output_dir / f"{result_path.stem}_fresh_scenario_coverage.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(
        f"\nCoverage (AR(1) tube, per generator, target {AR1_JOINT_COVERAGE:.2f}):\n"
        f"  kappa={kappa_ar1:.6f},  T={horizon},  num_constraints={horizon-1},  N={N}\n"
        f"  {per_proc_str}\n"
        f"  Joint across all generators (not the 95% claim): {all_joint_cov:.3f}"
    )
    return output_path


# ---------------------------------------------------------------------------
# New plot: whitened innovations to confirm N(0, scale^2) and exact 95%
# ---------------------------------------------------------------------------


def plot_fresh_scenario_innovations(
    result_path: Path,
    N: int = 300,
    seed: int = 0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Path:
    """Show the whitened innovations xi_t for fresh trajectories with +/- kappa bounds.

    For each process (demand, each wind generator):
      - Left panel: xi_t vs time, colored by per-trajectory containment.
      - Right panel: histogram of all xi_t values with N(0, scale^2) overlay.

    This is the exact view of the 95% claim: xi_t = D_t - rho*D_{t-1} - ar1_ref_t
    should look like N(0, (D_ref * sigma_D)^2) and lie within kappa * D_ref * sigma_D
    with joint probability 0.95.
    """
    from scipy.stats import norm as _norm_dist

    from config.scenarios.scenario_generator import RegimeParameters, ScenarioManager

    results = _load_json(result_path)
    ambiguity_set = results.get("ambiguity_set")
    if not isinstance(ambiguity_set, dict):
        raise ValueError(f"{result_path} does not contain an 'ambiguity_set' block.")

    params = _get_ambiguity_params(results)
    if params is None:
        raise ValueError(
            f"{result_path} is missing 'ambiguity_set.fixed_parameters'."
        )

    horizon = int(results["num_time_steps"])
    kappa_ar1 = _ar1_kappa(horizon - 1)
    t_interior = np.arange(1, horizon)  # t=1,...,T-1

    demand_shape = np.asarray(ambiguity_set["demand"]["shape"], dtype=float)
    wind_names = _wind_generator_names(results)
    wind_block = ambiguity_set.get("wind", {}) or {}

    regime = RegimeParameters(
        name="selected",
        n_scenarios=N,
        mu_D=params["mu_D"],
        rho_D=params["rho_D"],
        sigma_D=params["sigma_D"],
        mu_W=params["mu_W"],
        peak_W=params["tau_W"],
        rho_W=params["rho_W"],
        sigma_W=params["sigma_W"],
    )
    reference_case = str(results.get("reference_case", "base_test_case"))
    manager = ScenarioManager(base_case_reference=reference_case)
    manager.base_case["time_steps"] = horizon
    rng = np.random.default_rng(seed)

    demand_trajs: list[np.ndarray] = []
    wind_trajs: dict[str, list[np.ndarray]] = {g: [] for g in wind_names}
    for _ in range(N):
        d, w, _ = manager._simulate_single_scenario(regime=regime, rng=rng)
        demand_trajs.append(np.asarray(d, dtype=float))
        for g in wind_names:
            if g in w:
                wind_trajs[g].append(np.asarray(w[g], dtype=float))

    wind_shapes: dict[str, np.ndarray] = {}
    wind_caps: dict[str, float] = {}
    for g in wind_names:
        wind_shapes[g] = np.asarray(wind_block[g]["shape"], dtype=float)
        wind_caps[g] = float(wind_block[g].get("installed_capacity", 1.0))

    def _demand_xi(traj: np.ndarray) -> np.ndarray:
        ar1_ref = params["D_ref"] * params["mu_D"] * (demand_shape[1:] - params["rho_D"] * demand_shape[:-1])
        return traj[1:] - params["rho_D"] * traj[:-1] - ar1_ref

    def _wind_xi(traj: np.ndarray, g: str) -> np.ndarray:
        ws = wind_shapes[g]
        cap = wind_caps[g]
        ar1_ref = cap * params["mu_W"] * (ws[1:] - params["rho_W"] * ws[:-1])
        return traj[1:] - params["rho_W"] * traj[:-1] - ar1_ref

    # Containment for coloring.
    demand_scale = params["D_ref"] * params["sigma_D"]
    demand_xi_all = np.stack([_demand_xi(d) for d in demand_trajs])  # (N, T-1)
    demand_traj_in = (np.abs(demand_xi_all) <= kappa_ar1 * demand_scale).all(axis=1)  # (N,)

    wind_xi_all: dict[str, np.ndarray] = {}
    wind_traj_in: dict[str, np.ndarray] = {}
    for g in wind_names:
        scale_w = wind_caps[g] * params["sigma_W"]
        xi_w = np.stack([_wind_xi(w, g) for w in wind_trajs[g]])
        wind_xi_all[g] = xi_w
        wind_traj_in[g] = (np.abs(xi_w) <= kappa_ar1 * scale_w).all(axis=1)

    n_processes = 1 + len(wind_names)
    fig, axes = plt.subplots(
        n_processes, 2,
        figsize=(13, 3.4 * n_processes),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    if n_processes == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Whitened Innovations -- AR(1) Tube Containment\n"
        f"{_format_regime_label(ambiguity_set.get('selected_regime', {}))}\n"
        f"kappa={kappa_ar1:.4f},  T={horizon},  N={N}",
        fontsize=10,
    )

    def _draw_innovation_row(
        ax_series: Any,
        ax_hist: Any,
        xi_all: np.ndarray,
        traj_in: np.ndarray,
        scale: float,
        process_color: str,
        process_label: str,
    ) -> None:
        bound = kappa_ar1 * scale
        emp_joint = float(traj_in.mean())

        # Series plot: xi_t for each trajectory vs time.
        for n in range(N):
            color = process_color if traj_in[n] else "tab:red"
            alpha = 0.07 if traj_in[n] else 0.35
            ax_series.plot(t_interior, xi_all[n], color=color, alpha=alpha, linewidth=0.7)
        ax_series.axhline(bound, color="tab:orange", linewidth=1.8, linestyle="--",
                          label=f"+kappa*scale = {bound:.2f}")
        ax_series.axhline(-bound, color="tab:orange", linewidth=1.8, linestyle="--",
                          label=f"-kappa*scale = {-bound:.2f}")
        ax_series.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_series.set_title(
            f"{process_label}   joint cov = {emp_joint:.3f}  (target {AR1_JOINT_COVERAGE:.2f})"
        )
        ax_series.set_ylabel("xi_t (MW)")
        ax_series.set_xlabel("t")
        ax_series.legend(fontsize=7, loc="upper right")
        ax_series.grid(True, alpha=0.2)

        # Histogram plot: all xi values pooled, overlay N(0, scale^2).
        flat_xi = xi_all.flatten()
        ax_hist.hist(flat_xi, bins=50, density=True, alpha=0.65,
                     color=process_color, label="xi_t (all draws, all t)")
        x_range = np.linspace(flat_xi.min(), flat_xi.max(), 300)
        ax_hist.plot(x_range, _norm_dist.pdf(x_range, loc=0, scale=scale),
                     "r-", linewidth=2.0, label=f"N(0, {scale:.2f}^2)")
        ax_hist.axvline(bound, color="tab:orange", linestyle="--", linewidth=1.5)
        ax_hist.axvline(-bound, color="tab:orange", linestyle="--", linewidth=1.5)
        ax_hist.set_ylabel("Density")
        ax_hist.set_xlabel("xi_t (MW)")
        ax_hist.legend(fontsize=7, loc="best")
        ax_hist.grid(True, alpha=0.2)

    # Demand row.
    _draw_innovation_row(
        axes[0, 0], axes[0, 1],
        demand_xi_all, demand_traj_in,
        demand_scale, "tab:blue", "Demand",
    )

    # Wind rows.
    for row, g in enumerate(wind_names, start=1):
        scale_w = wind_caps[g] * params["sigma_W"]
        _draw_innovation_row(
            axes[row, 0], axes[row, 1],
            wind_xi_all[g], wind_traj_in[g],
            scale_w, "tab:green", f"Wind {g}",
        )

    fig.tight_layout()
    output_path = output_dir / f"{result_path.stem}_fresh_scenario_innovations.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Edit these paths/settings directly when running this script.
    result_paths: list[Path] = [
        Path("results/base_case/poa/poa_optimization_T6_piecewise_mccormick.json"),
    ]
    results_dir = Path("results/base_case/poa")
    plot_all_matching_results = False
    output_dir = DEFAULT_OUTPUT_DIR
    show = False

    # Number of fresh scenarios for the coverage and innovations plots.
    N_fresh = 300
    fresh_seed = 42

    selected_result_paths = _resolve_result_paths(
        result_paths=result_paths,
        results_dir=results_dir,
        plot_all_matching_results=plot_all_matching_results,
    )

    for result_path in selected_result_paths:
        # 1. Support set plots: demand figure and wind figure.
        p_demand, p_wind = plot_ambiguity_regime_trajectories(
            result_path=result_path,
            output_dir=output_dir,
            show=show,
        )
        print(f"Saved demand support set plot:    {p_demand}")
        print(f"Saved wind support set plot:      {p_wind}")

        # 2. Fresh scenario coverage in level space.
        try:
            p2 = plot_fresh_scenario_coverage(
                result_path=result_path,
                N=N_fresh,
                seed=fresh_seed,
                output_dir=output_dir,
                show=show,
            )
            print(f"Saved coverage plot:              {p2}")
        except ValueError as exc:
            print(f"Skipping coverage plot: {exc}")

        # 3. Whitened innovations to confirm 95% and N(0, scale^2).
        try:
            p3 = plot_fresh_scenario_innovations(
                result_path=result_path,
                N=N_fresh,
                seed=fresh_seed,
                output_dir=output_dir,
                show=show,
            )
            print(f"Saved innovations plot:           {p3}")
        except ValueError as exc:
            print(f"Skipping innovations plot: {exc}")


if __name__ == "__main__":
    main()
