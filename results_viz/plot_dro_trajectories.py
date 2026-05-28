"""Plot optimized demand and wind trajectories from DRO results.

One figure is produced per scenario per regime.  Each figure shows how the
optimizer's worst-case perturbation for that scenario evolves as eta increases,
with the fixed empirical (reference) trajectory overlaid in grey.

Output structure:
    output_dir/
        {regime_name}/
            scenario_k{k}.png
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


DEFAULT_OUTPUT_DIR = Path("results_viz/figures/dro_trajectories")
DEFAULT_DRO_RESULT_DIR = Path("results/dro_poa")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _series(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values])


def _wind_generator_names(physical_names: list[str]) -> list[str]:
    return [name for name in physical_names if name.startswith("W")]


def _load_sorted_result_files(regime_dir: Path) -> list[tuple[float, Path]]:
    """Return (eta, path) pairs sorted by eta value."""
    pairs: list[tuple[float, Path]] = []
    for path in regime_dir.glob("dro_poa_eta_*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            pairs.append((float(payload["eta"]), path))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _find_regime_dirs(dro_result_dir: Path) -> list[Path]:
    """Return subdirectories of dro_result_dir that contain result files."""
    return sorted(
        d for d in dro_result_dir.iterdir()
        if d.is_dir() and any(d.glob("dro_poa_eta_*.json"))
    )


# ---------------------------------------------------------------------------
# Per-scenario figure
# ---------------------------------------------------------------------------

def _plot_scenario(
    axes: list[Any],
    k: int,
    wind_names: list[str],
    time: np.ndarray,
    empirical_scenario: dict[str, Any],
    selected_results: list[dict[str, Any]],
    etas: list[float],
    cmap: Any,
    norm: mcolors.Normalize,
) -> None:
    # Empirical reference for this scenario.
    emp_kw: dict[str, Any] = dict(
        color="0.55", linewidth=1.5, linestyle="--", alpha=0.9, zorder=2,
        label="Empirical",
    )
    axes[0].plot(time, _series(empirical_scenario["empirical_demand_profile"]), **emp_kw)
    for ax, gen_name in zip(axes[1:], wind_names):
        ax.plot(
            time,
            _series(empirical_scenario["empirical_physical_capacity_profiles"][gen_name]),
            **emp_kw,
        )

    # Optimized trajectory for this scenario at each selected eta.
    for result, eta in zip(selected_results, etas):
        scenario = next((s for s in result.get("scenarios", []) if s["k"] == k), None)
        if scenario is None:
            continue
        opt_kw: dict[str, Any] = dict(
            color=cmap(norm(eta)), linewidth=1.8, alpha=0.8, zorder=3,
        )
        axes[0].plot(time, _series(scenario["optimized_demand_profile"]), **opt_kw)
        for ax, gen_name in zip(axes[1:], wind_names):
            ax.plot(
                time,
                _series(scenario["optimized_physical_capacity_profiles"][gen_name]),
                **opt_kw,
            )


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_dro_trajectories_for_regime(
    regime_dir: Path,
    every_nth: int = 5,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> list[Path]:
    """Produce one figure per scenario for a single regime directory."""
    all_files = _load_sorted_result_files(regime_dir)
    if not all_files:
        raise FileNotFoundError(f"No DRO result files found in {regime_dir}")

    selected = all_files[::every_nth]
    results_list = [_load_json(path) for _, path in selected]
    etas = [float(r["eta"]) for r in results_list]

    first = results_list[0]
    physical_names = [str(n) for n in first.get("physical_generator_names", [])]
    wind_names = _wind_generator_names(physical_names)
    if not wind_names:
        raise ValueError("No wind generators (names starting with 'W') found in result")

    horizon = int(first["num_time_steps"])
    time = np.arange(horizon)
    regime_name = str(first.get("regime_name", "unknown"))

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(etas), vmax=max(etas))

    regime_output_dir = output_dir / regime_name
    regime_output_dir.mkdir(parents=True, exist_ok=True)

    empirical_handle = Line2D(
        [0], [0], color="0.55", linewidth=1.5, linestyle="--", label="Empirical"
    )

    n_rows = 1 + len(wind_names)
    output_paths: list[Path] = []

    for empirical_scenario in first["scenarios"]:
        k = int(empirical_scenario["k"])

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(11, 3.1 * n_rows),
            sharex=True,
        )
        if n_rows == 1:
            axes = [axes]

        fig.suptitle(
            f"DRO Optimized Trajectories — regime: {regime_name}, scenario k={k}\n"
            f"(every {every_nth}th η, {len(etas)} values shown)",
            fontsize=12,
        )

        _plot_scenario(
            axes=axes,
            k=k,
            wind_names=wind_names,
            time=time,
            empirical_scenario=empirical_scenario,
            selected_results=results_list,
            etas=etas,
            cmap=cmap,
            norm=norm,
        )

        axes[0].set_title("Demand D[t]")
        for ax, gen_name in zip(axes[1:], wind_names):
            ax.set_title(f"{gen_name} Wind Availability")
        for ax in axes:
            ax.set_ylabel("MW")
            ax.grid(True, alpha=0.25)
            ax.legend(handles=[empirical_handle], loc="upper left", fontsize=8)
        axes[-1].set_xlabel("Time step")

        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="η")

        output_path = regime_output_dir / f"scenario_k{k}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_dro_trajectories(
    dro_result_dir: Path = DEFAULT_DRO_RESULT_DIR,
    every_nth: int = 5,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> dict[str, list[Path]]:
    """Plot trajectories for every regime found under dro_result_dir."""
    regime_dirs = _find_regime_dirs(dro_result_dir)
    if not regime_dirs:
        raise FileNotFoundError(
            f"No regime subdirectories with DRO results found in {dro_result_dir}"
        )

    results: dict[str, list[Path]] = {}
    for regime_dir in regime_dirs:
        paths = plot_dro_trajectories_for_regime(
            regime_dir=regime_dir,
            every_nth=every_nth,
            output_dir=output_dir,
            show=show,
        )
        results[regime_dir.name] = paths
        print(f"  {regime_dir.name}: {len(paths)} scenario plots saved to {output_dir / regime_dir.name}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Edit these paths/settings directly when running this script.
    dro_result_dir = DEFAULT_DRO_RESULT_DIR
    every_nth = 1
    output_dir = DEFAULT_OUTPUT_DIR
    show = False

    print(f"Plotting DRO trajectories from {dro_result_dir}")
    results = plot_dro_trajectories(
        dro_result_dir=dro_result_dir,
        every_nth=every_nth,
        output_dir=output_dir,
        show=show,
    )
    total = sum(len(v) for v in results.values())
    print(f"Done. {total} figures saved across {len(results)} regime(s).")


if __name__ == "__main__":
    main()
