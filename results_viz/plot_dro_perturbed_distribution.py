"""Visualize the induced worst-case perturbed distribution from DRO eta sweep results.

For each value of eta the DRO problem produces a discrete distribution
    P_hat^(r)_eta = (1/K) sum_k delta_{s_tilde^(k,*)}
over the K optimized perturbations of the empirical trajectories.

Three figure types are produced per regime:

1. distribution_fan_{regime}_{eta_label}.png
   For a selected subset of eta values: a multi-subplot fan of all K perturbed
   trajectories (coloured by scenario k) against the grey empirical reference.
   One subplot per market variable (demand + each wind generator).

2. distribution_summary_{regime}.png
   Mean +/- IQR band of the induced distribution at each time step, stacked
   across eta values on a single shared x-axis.  Shows how the distribution
   shifts and contracts as eta increases.

3. transport_scatter_{regime}.png
   For a subset of eta values: scatter of (empirical value, perturbed value)
   for all (k, t) pairs.  Points above the 45-degree line indicate the
   optimizer pushed the trajectory above its empirical level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_DRO_RESULT_DIR = Path("results/base_case/dro/poa_worst_case")
DEFAULT_OUTPUT_DIR = Path("results_viz/figures/dro_perturbed_distribution")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _series(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values])


def _wind_names_from_result(result: dict[str, Any]) -> list[str]:
    phys = result.get("physical_generator_names", [])
    return [str(n) for n in phys if str(n).startswith("W")]


def load_sorted_results(regime_dir: Path) -> list[tuple[float, dict[str, Any]]]:
    """Return (eta, payload) pairs sorted by eta, skipping unreadable files."""
    pairs: list[tuple[float, dict[str, Any]]] = []
    for path in regime_dir.glob("dro_poa_eta_*.json"):
        try:
            payload = _load_json(path)
            pairs.append((float(payload["eta"]), payload))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _extract_profiles(result: dict[str, Any], wind_names: list[str]) -> dict[str, np.ndarray]:
    """Return {variable_name: array(K, T)} for demand and each wind generator."""
    scenarios = result.get("scenarios", [])
    K = len(scenarios)
    T = int(result.get("num_time_steps", 1))

    emp_D = np.full((K, T), np.nan)
    opt_D = np.full((K, T), np.nan)
    emp_W: dict[str, np.ndarray] = {g: np.full((K, T), np.nan) for g in wind_names}
    opt_W: dict[str, np.ndarray] = {g: np.full((K, T), np.nan) for g in wind_names}

    for s in scenarios:
        k = int(s["k"])
        emp_D[k] = _series(s["empirical_demand_profile"])
        opt_D[k] = _series(s["optimized_demand_profile"])
        emp_phys = s.get("empirical_physical_capacity_profiles", {})
        opt_phys = s.get("optimized_physical_capacity_profiles", {})
        for g in wind_names:
            if g in emp_phys:
                emp_W[g][k] = _series(emp_phys[g])
            if g in opt_phys:
                opt_W[g][k] = _series(opt_phys[g])

    out: dict[str, np.ndarray] = {"emp_demand": emp_D, "opt_demand": opt_D}
    for g in wind_names:
        out[f"emp_{g}"] = emp_W[g]
        out[f"opt_{g}"] = opt_W[g]
    return out


# ---------------------------------------------------------------------------
# Figure 1: distribution fan (one figure per selected eta)
# ---------------------------------------------------------------------------

def _eta_label(eta: float) -> str:
    return f"{eta:.4g}".replace("-", "m").replace(".", "p")


def plot_distribution_fan(
    results: list[tuple[float, dict[str, Any]]],
    selected_etas: list[float] | None,
    output_dir: Path,
    regime_name: str,
    show: bool = False,
) -> list[Path]:
    """One fan figure per selected eta, empirical (grey) vs perturbed (tab10)."""
    if not results:
        return []

    wind_names = _wind_names_from_result(results[0][1])
    T = int(results[0][1].get("num_time_steps", 1))
    time = np.arange(T)
    n_vars = 1 + len(wind_names)
    var_labels = ["Demand D[t]"] + [f"{g} capacity [t]" for g in wind_names]

    # Pick the etas to show
    all_etas = [eta for eta, _ in results]
    if selected_etas is None:
        # Automatic: ~5 evenly-spaced across the range
        n_show = min(5, len(results))
        idx = np.round(np.linspace(0, len(results) - 1, n_show)).astype(int)
        selected_etas = [all_etas[i] for i in idx]

    # Colormap for the K scenarios
    scenario_cmap = plt.get_cmap("tab10")

    saved: list[Path] = []
    for eta_target in selected_etas:
        # Pick nearest available eta
        nearest_idx = int(np.argmin([abs(e - eta_target) for e in all_etas]))
        eta, result = results[nearest_idx]
        profiles = _extract_profiles(result, wind_names)
        K = profiles["emp_demand"].shape[0]

        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3.2 * n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]

        for ax, label, emp_key, opt_key in zip(
            axes,
            var_labels,
            ["emp_demand"] + [f"emp_{g}" for g in wind_names],
            ["opt_demand"] + [f"opt_{g}" for g in wind_names],
        ):
            emp = profiles[emp_key]
            opt = profiles[opt_key]

            # Empirical trajectories — light grey dashed
            for k in range(K):
                ax.plot(time, emp[k], color="0.60", linewidth=1.2, linestyle="--", alpha=0.7,
                        zorder=2)

            # Perturbed trajectories — coloured by scenario k
            for k in range(K):
                ax.plot(time, opt[k], color=scenario_cmap(k % 10), linewidth=1.6,
                        alpha=0.75, zorder=3)

            # Empirical mean as thick grey reference
            emp_mean = np.nanmean(emp, axis=0)
            ax.plot(time, emp_mean, color="0.25", linewidth=2.5, linestyle="--",
                    zorder=4, label="Empirical mean")

            # Perturbed mean
            opt_mean = np.nanmean(opt, axis=0)
            ax.plot(time, opt_mean, color="black", linewidth=2.5, linestyle="-",
                    zorder=5, label="Perturbed mean")

            ax.set_title(label, fontsize=10)
            ax.set_ylabel("MW")
            ax.grid(True, alpha=0.25)
            ax.set_xticks(time)

        axes[-1].set_xlabel("Time step t")

        legend_handles = [
            Line2D([0], [0], color="0.60", linewidth=1.2, linestyle="--",
                   label="Empirical trajectories"),
            Line2D([0], [0], color="0.25", linewidth=2.5, linestyle="--",
                   label="Empirical mean"),
            Line2D([0], [0], color=scenario_cmap(0), linewidth=1.6,
                   label="Perturbed trajectories (each k)"),
            Line2D([0], [0], color="black", linewidth=2.5, linestyle="-",
                   label="Perturbed mean"),
        ]
        axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=2)

        fig.suptitle(
            f"Induced distribution $\\hat{{P}}^{{(r)}}_{{\\eta}}$ — regime: {regime_name},"
            f" $\\eta = {eta:.4g}$\n"
            f"(K={K} scenarios, T={T} time steps)",
            fontsize=11,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

        fname = f"distribution_fan_{regime_name}_eta{_eta_label(eta)}"
        for ext in (".pdf", ".png"):
            p = output_dir / (fname + ext)
            fig.savefig(p, dpi=180, bbox_inches="tight")
            if ext == ".png":
                saved.append(p)
        if show:
            plt.show()
        plt.close(fig)

    return saved


# ---------------------------------------------------------------------------
# Figure 2: distribution summary (mean +/- IQR strip per eta)
# ---------------------------------------------------------------------------

def plot_distribution_summary(
    results: list[tuple[float, dict[str, Any]]],
    output_dir: Path,
    regime_name: str,
    show: bool = False,
) -> list[Path]:
    """Mean trajectory of every second eta value, coloured by eta via viridis.

    The empirical mean is overlaid as a thick grey dashed reference.
    """
    if not results:
        return []

    wind_names = _wind_names_from_result(results[0][1])
    T = int(results[0][1].get("num_time_steps", 1))
    time = np.arange(T)
    n_vars = 1 + len(wind_names)
    var_labels = ["Demand D[t]"] + [f"{g} capacity" for g in wind_names]
    emp_keys = ["emp_demand"] + [f"emp_{g}" for g in wind_names]
    opt_keys = ["opt_demand"] + [f"opt_{g}" for g in wind_names]

    # Every second eta (index 0, 2, 4, ...)
    selected = results[::2]
    all_etas = [eta for eta, _ in results]
    selected_etas = [eta for eta, _ in selected]

    eta_cmap = plt.get_cmap("viridis")
    eta_norm = mcolors.Normalize(vmin=min(all_etas), vmax=max(all_etas) or 1.0)

    fig, axes = plt.subplots(n_vars, 1, figsize=(11, 3.5 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    # Empirical mean (from first result — same for all eta)
    emp_profiles_first = _extract_profiles(results[0][1], wind_names)

    for ax, label, emp_key, opt_key in zip(axes, var_labels, emp_keys, opt_keys):
        # Empirical mean — thick grey dashed reference
        emp = emp_profiles_first[emp_key]
        emp_mean = np.nanmean(emp, axis=0)
        ax.plot(time, emp_mean, color="0.40", linewidth=2.2, linestyle="--", zorder=2,
                label="Empirical mean")

        # Mean trajectory for every second eta
        for eta, result in selected:
            profiles = _extract_profiles(result, wind_names)
            opt = profiles[opt_key]
            opt_mean = np.nanmean(opt, axis=0)
            color = eta_cmap(eta_norm(eta))
            ax.plot(time, opt_mean, color=color, linewidth=1.8, alpha=0.9, zorder=3)

        ax.set_title(label, fontsize=10)
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(time)

    axes[-1].set_xlabel("Time step t")

    # Colorbar for eta
    sm = cm.ScalarMappable(cmap=eta_cmap, norm=eta_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("$\\eta$")

    emp_handles = [
        Line2D([0], [0], color="0.40", linewidth=2.2, linestyle="--",
               label="Empirical mean"),
        Line2D([0], [0], color=eta_cmap(eta_norm(selected_etas[0])), linewidth=1.8,
               label=f"Perturbed mean (every 2nd $\\eta$, n={len(selected)})"),
    ]
    axes[0].legend(handles=emp_handles, loc="upper right", fontsize=8)

    fig.suptitle(
        f"Induced distribution mean trajectories — regime: {regime_name}\n"
        f"Every 2nd $\\eta$ value ({len(selected)} shown)  |  colour = $\\eta$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.93, 0.93])

    fname = f"distribution_summary_{regime_name}"
    saved: list[Path] = []
    for ext in (".pdf", ".png"):
        p = output_dir / (fname + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        if ext == ".png":
            saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Figure 3: transport scatter (empirical vs perturbed)
# ---------------------------------------------------------------------------

def plot_transport_scatter(
    results: list[tuple[float, dict[str, Any]]],
    selected_etas: list[float] | None,
    output_dir: Path,
    regime_name: str,
    show: bool = False,
) -> list[Path]:
    """Scatter of (s^(k)_t, s~^(k,*)_t) per (k, t), grouped by selected eta values.

    Points above the 45-degree line indicate the optimizer pushed the value
    upward relative to its empirical reference.
    """
    if not results:
        return []

    wind_names = _wind_names_from_result(results[0][1])
    all_etas = [eta for eta, _ in results]

    if selected_etas is None:
        n_show = min(4, len(results))
        idx = np.round(np.linspace(0, len(results) - 1, n_show)).astype(int)
        selected_etas = [all_etas[i] for i in idx]

    var_labels = ["Demand D"] + [f"{g}" for g in wind_names]
    emp_keys = ["emp_demand"] + [f"emp_{g}" for g in wind_names]
    opt_keys = ["opt_demand"] + [f"opt_{g}" for g in wind_names]
    n_vars = len(var_labels)

    # Resolve nearest available eta for each target
    resolved: list[tuple[float, dict[str, Any]]] = []
    for eta_target in selected_etas:
        nearest_idx = int(np.argmin([abs(e - eta_target) for e in all_etas]))
        resolved.append(results[nearest_idx])

    n_cols = len(resolved)
    fig, axes = plt.subplots(
        n_vars, n_cols,
        figsize=(4.0 * n_cols, 3.8 * n_vars),
        squeeze=False,
    )

    scenario_cmap = plt.get_cmap("tab10")

    for col_idx, (eta, result) in enumerate(resolved):
        profiles = _extract_profiles(result, wind_names)
        K = profiles["emp_demand"].shape[0]

        for row_idx, (label, emp_key, opt_key) in enumerate(
            zip(var_labels, emp_keys, opt_keys)
        ):
            ax = axes[row_idx][col_idx]
            emp = profiles[emp_key]  # (K, T)
            opt = profiles[opt_key]  # (K, T)

            for k in range(K):
                x = emp[k]
                y = opt[k]
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.any():
                    ax.scatter(x[valid], y[valid], color=scenario_cmap(k % 10),
                               s=35, alpha=0.8, zorder=3, label=f"k={k}")

            # 45-degree reference line
            all_vals = np.concatenate([emp.ravel(), opt.ravel()])
            all_vals = all_vals[~np.isnan(all_vals)]
            if len(all_vals):
                lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
                margin = 0.05 * (hi - lo) if hi > lo else 1.0
                lims = (lo - margin, hi + margin)
                ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.5, zorder=2)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)

            if row_idx == 0:
                ax.set_title(f"$\\eta = {eta:.4g}$", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Perturbed {label}", fontsize=9)
            if row_idx == n_vars - 1:
                ax.set_xlabel(f"Empirical {label}", fontsize=9)

    # Single legend for scenario colours
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=scenario_cmap(k % 10), markersize=7, label=f"k={k}")
        for k in range(profiles["emp_demand"].shape[0])
    ]
    fig.legend(handles=legend_handles, title="Scenario k", loc="lower center",
               ncol=min(K, 5), fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Transport scatter — regime: {regime_name}\n"
        "Each point = one (k, t) pair.  Above diagonal: optimizer raised value.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])

    fname = f"transport_scatter_{regime_name}"
    saved: list[Path] = []
    for ext in (".pdf", ".png"):
        p = output_dir / (fname + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        if ext == ".png":
            saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def plot_dro_perturbed_distribution(
    regime_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    selected_fan_etas: list[float] | None = None,
    selected_transport_etas: list[float] | None = None,
    show: bool = False,
) -> dict[str, list[Path]]:
    """Produce all three figure types for a single regime directory.

    Args:
        regime_dir: directory containing dro_poa_eta_*.json result files.
        output_dir: root output directory; a sub-folder per regime is created.
        selected_fan_etas: eta values to show in the distribution-fan figure
            (nearest available is used).  Defaults to ~5 evenly-spaced values.
        selected_transport_etas: eta values for the transport scatter.
            Defaults to ~4 evenly-spaced values.
        show: if True, call plt.show() after each figure.
    """
    results = load_sorted_results(regime_dir)
    if not results:
        raise FileNotFoundError(f"No dro_poa_eta_*.json files found in {regime_dir}")

    regime_name = regime_dir.name
    out = output_dir / regime_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"  {regime_name}: {len(results)} eta values, "
          f"eta in [{results[0][0]:.4g}, {results[-1][0]:.4g}]")

    saved: dict[str, list[Path]] = {}

    fan_paths = plot_distribution_fan(
        results=results,
        selected_etas=selected_fan_etas,
        output_dir=out,
        regime_name=regime_name,
        show=show,
    )
    saved["fan"] = fan_paths
    print(f"    Fan figures:     {len(fan_paths)}")

    summary_paths = plot_distribution_summary(
        results=results,
        output_dir=out,
        regime_name=regime_name,
        show=show,
    )
    saved["summary"] = summary_paths
    print(f"    Summary figure:  {len(summary_paths)}")

    scatter_paths = plot_transport_scatter(
        results=results,
        selected_etas=selected_transport_etas,
        output_dir=out,
        regime_name=regime_name,
        show=show,
    )
    saved["scatter"] = scatter_paths
    print(f"    Scatter figure:  {len(scatter_paths)}")

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(
        description="Visualize DRO worst-case perturbed distribution from eta sweep results"
    )
    parser.add_argument(
        "--regime-dir",
        type=Path,
        default=DEFAULT_DRO_RESULT_DIR,
        help="Directory containing dro_poa_eta_*.json files "
             f"(default: {DEFAULT_DRO_RESULT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--fan-etas",
        type=float,
        nargs="+",
        default=None,
        metavar="ETA",
        help="Eta values to show in distribution-fan figures (default: auto ~5 values)",
    )
    parser.add_argument(
        "--scatter-etas",
        type=float,
        nargs="+",
        default=None,
        metavar="ETA",
        help="Eta values to show in transport scatter (default: auto ~4 values)",
    )
    parser.add_argument("--show", action="store_true", help="Call plt.show() after each figure")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg" if not args.show else "TkAgg")

    print(f"DRO perturbed distribution visualizer")
    print(f"  Regime dir:  {args.regime_dir}")
    print(f"  Output dir:  {args.output_dir}")

    saved = plot_dro_perturbed_distribution(
        regime_dir=args.regime_dir,
        output_dir=args.output_dir,
        selected_fan_etas=args.fan_etas,
        selected_transport_etas=args.scatter_etas,
        show=args.show,
    )
    total = sum(len(v) for v in saved.values())
    print(f"\nDone. {total} PNG files saved under {args.output_dir}.")


if __name__ == "__main__":
    main()
