# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-22
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Standalone setup figure: deterministic demand and wind shape functions.

Reproduces setup_viz/shape_functions_by_regime with a single wind peak, no
super-title, and no legends, saved as a vector PDF (plus PNG via the thesis
style hook). The shapes are the deterministic mean-one multipliers that the
ScenarioManager uses before stochastic perturbation.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.plot_shape_functions_setup
"""

from __future__ import annotations

from pathlib import Path

import results_viz._thesis_style  # noqa: F401  (installs PDF+PNG savefig)
from config.scenarios.scenario_generator import ScenarioManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Targets: the standalone setup_viz illustration and the base-case setup figure.
CASE = "base_test_case"
ILLUSTRATION_HORIZON = 24  # smooth full-day illustration for the setup_viz figure
BASE_CASE_HORIZON = 6  # the base-case runs at T=6 (matches results/base_case/poa T6)
WIND_PEAK_HOUR = 14.0  # single wind peak to display (base-case peak_W)

SETUP_VIZ_PATH = PROJECT_ROOT / "setup_viz" / "shape_functions_by_regime.pdf"
BASE_CASE_PATH = (
    PROJECT_ROOT / "results" / "base_case" / "figures" / "setup" / "shape_functions.pdf"
)
HORIZON_SWEEP_ROOT = PROJECT_ROOT / "results" / "sensitivity_studies" / "horizon_sweep"
OUTPUT_PATH = SETUP_VIZ_PATH


COMBINED_PATH = HORIZON_SWEEP_ROOT / "figures" / "shape_functions_by_horizon.pdf"

# A4 0.85\textwidth sizing + document-point fonts (same convention as the other
# thesis figures), so the combined plot is legible when included at that width.
TEXTWIDTH_IN = 6.3
WIDTH_FRACTION = 0.85
FIG_WIDTH_IN = TEXTWIDTH_IN * WIDTH_FRACTION
FONT_SIZES = {
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
}


def _horizon_sweep_targets() -> list[tuple[Path, int, str]]:
    """One (path, T, marker) target per T<N> run dir in the horizon sweep."""
    targets = []
    for run_dir in sorted(HORIZON_SWEEP_ROOT.glob("T*")):
        if not run_dir.is_dir() or not run_dir.name[1:].isdigit():
            continue
        horizon = int(run_dir.name[1:])
        save_path = run_dir / "figures" / "setup" / "shape_functions.pdf"
        targets.append((save_path, horizon, "o"))
    return targets


def make_combined_figure(horizons: list[int], save_path: Path | None = COMBINED_PATH):
    """Overlay the shape functions for several horizons on a shared hours axis.

    Each horizon discretizes the same continuous day, so step indices are not
    comparable; mapping each step to its time-of-day (hours in [0, 24)) puts every
    horizon on a common x-axis and shows the finer horizons sampling more densely.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    manager = ScenarioManager(CASE)
    cmap = plt.get_cmap("tab10")

    with plt.rc_context(FONT_SIZES):
        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.0), sharex=True
        )
        sorted_horizons = sorted(horizons)
        n = len(sorted_horizons)
        for idx, horizon in enumerate(sorted_horizons):
            color = cmap(idx % cmap.N)
            demand_hours = np.linspace(0.0, 24.0, horizon)
            wind_hours = np.linspace(0.0, 24.0, horizon, endpoint=False)
            # Lower horizon -> higher zorder, so the coarsest (fewest points) sits
            # on top and stays visible over the denser, finer horizons.
            zorder = n - idx + 3
            axes[0].plot(
                demand_hours, manager._build_demand_shape(horizon),
                marker="o", markersize=5, linewidth=1.8, color=color,
                label=f"T = {horizon}", zorder=zorder,
            )
            axes[1].plot(
                wind_hours, manager._build_wind_shape(horizon, WIND_PEAK_HOUR),
                marker="o", markersize=5, linewidth=1.8, color=color,
                label=f"T = {horizon}", zorder=zorder,
            )

        for ax in axes:
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
            ax.set_ylabel("Multiplier")
            ax.grid(True, alpha=0.25)
        axes[0].set_title("Demand Shape")
        axes[1].set_title("Wind Shape")
        axes[1].set_xlabel("Time (h)")

        # One shared legend in a single row beneath the plot (both panels use the
        # same horizon colors), so it never overlaps the curves.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Horizon",
            loc="upper center", bbox_to_anchor=(0.5, 0.0),
            ncol=len(labels), frameon=False,
        )
        # Reserve room at the bottom for the legend row.
        fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def make_figure(save_path: Path | None, horizon: int, marker: str | None = None):
    import matplotlib.pyplot as plt

    manager = ScenarioManager(CASE)
    demand_shape = manager._build_demand_shape(horizon)
    wind_shape = manager._build_wind_shape(horizon, WIND_PEAK_HOUR)
    time_index = range(horizon)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.0, 5.0), sharex=True)

    axes[0].plot(time_index, demand_shape, linewidth=2.2, marker=marker)
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    axes[0].set_title("Demand Shape")
    axes[0].set_ylabel("Multiplier")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_index, wind_shape, linewidth=2.2, color="tab:orange", marker=marker)
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    axes[1].set_title("Wind Shape")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Multiplier")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def main() -> None:
    # The base-case figure uses the actual model horizon (T=6) so it matches the
    # discretization the optimization runs at; markers make the few steps clear.
    targets = [
        (SETUP_VIZ_PATH, ILLUSTRATION_HORIZON, None),
        (BASE_CASE_PATH, BASE_CASE_HORIZON, "o"),
        *_horizon_sweep_targets(),
    ]
    for path, horizon, marker in targets:
        make_figure(path, horizon, marker)
        print(f"Wrote {path}  (T={horizon}, + .png)")

    # Combined overlay of every horizon-sweep T on a shared hours axis.
    sweep_horizons = [h for _, h, _ in _horizon_sweep_targets()]
    if sweep_horizons:
        make_combined_figure(sweep_horizons, COMBINED_PATH)
        print(f"Wrote {COMBINED_PATH}  (T={sorted(sweep_horizons)}, + .png)")


if __name__ == "__main__":
    main()
