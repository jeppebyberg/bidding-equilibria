"""Plot the ambiguity regime and optimized state trajectories from PoA results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = Path("results_viz/figures/ambiguity_regime_trajectories")


def _series(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values])


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _find_default_result_path(results_dir: Path) -> Path:
    ambiguity_candidates = _find_ambiguity_result_paths(results_dir)
    if not ambiguity_candidates:
        raise FileNotFoundError(
            f"No PoA result with an 'ambiguity_set' block found in {results_dir}"
        )
    return max(ambiguity_candidates, key=lambda path: path.stat().st_mtime)


def _find_ambiguity_result_paths(results_dir: Path) -> list[Path]:
    candidates = sorted(results_dir.glob("poa_optimization*.json"))
    ambiguity_candidates: list[Path] = []
    for candidate in candidates:
        try:
            payload = _load_json(candidate)
        except Exception:
            continue
        if isinstance(payload.get("ambiguity_set"), dict):
            ambiguity_candidates.append(candidate)
    return ambiguity_candidates


def _resolve_result_paths(
    result_paths: list[Path],
    results_dir: Path,
    plot_all_matching_results: bool,
) -> list[Path]:
    explicit_paths = [Path(path) for path in result_paths]
    if explicit_paths:
        missing_paths = [path for path in explicit_paths if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Result path(s) not found: {missing}")
        return explicit_paths

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
        str(generator_name)
        for generator_name, generator in generators.items()
        if bool(generator.get("is_wind"))
    ]


def _format_regime_label(selected_regime: dict[str, Any]) -> str:
    parts = []
    for key in ("mu_D", "sigma_D", "mu_W", "sigma_W"):
        value = selected_regime.get(key)
        if value is not None:
            parts.append(f"{key}={float(value):.3g}")
    return ", ".join(parts)


def plot_ambiguity_regime_trajectories(
    result_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Path:
    results = _load_json(result_path)
    ambiguity_set = results.get("ambiguity_set")
    if not isinstance(ambiguity_set, dict):
        raise ValueError(
            f"{result_path} does not contain an 'ambiguity_set' block. "
            "Run this script on a PoA result generated after the ambiguity-set refactor."
        )

    demand_block = ambiguity_set.get("demand", {}) or {}
    selected_regime = ambiguity_set.get("selected_regime", {}) or {}
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
    fig, axes = plt.subplots(
        1 + len(wind_names),
        1,
        figsize=(11, 3.1 * (1 + len(wind_names))),
        sharex=True,
    )
    regime_label = _format_regime_label(selected_regime)
    fig.suptitle(
        f"Optimized Ambiguity Regime and Induced Support Bounds\n{regime_label}",
        fontsize=12,
    )

    axes[0].fill_between(
        time,
        demand_lower,
        demand_upper,
        color="tab:blue",
        alpha=0.16,
        label="Induced demand support",
    )
    axes[0].plot(
        time,
        demand_reference,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label="Demand reference",
    )
    axes[0].plot(
        time,
        demand,
        color="black",
        marker="o",
        linewidth=2.2,
        label="Optimized demand D[t]",
    )
    axes[0].set_title("Demand Trajectory")
    axes[0].set_ylabel("MW")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    for axis, generator_name in zip(axes[1:], wind_names):
        if generator_name not in wind_block:
            raise KeyError(
                f"ambiguity_set['wind'] is missing wind generator '{generator_name}'"
            )
        generator_ambiguity = wind_block[generator_name]
        generator_profile = _series(
            results["generators"][generator_name]["physical_capacity_profile"]
        )
        axis.fill_between(
            time,
            _series(generator_ambiguity["lower"]),
            _series(generator_ambiguity["upper"]),
            color="tab:green",
            alpha=0.18,
            label="Induced wind support",
        )
        axis.plot(
            time,
            _series(generator_ambiguity["reference"]),
            color="tab:green",
            linestyle="--",
            linewidth=1.8,
            label="Wind reference",
        )
        axis.plot(
            time,
            generator_profile,
            color="black",
            marker="o",
            linewidth=2.0,
            label="Optimized wind availability",
        )
        axis.set_title(f"{generator_name} Wind Availability")
        axis.set_ylabel("MW")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Time step")

    fig.tight_layout()
    output_path = output_dir / f"{result_path.stem}_ambiguity_regime_trajectories.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    # Edit these paths/settings directly when running this script.
    # Leave result_paths empty to use either the newest or all matching PoA results.
    result_paths = [
        # Path("results/temporary_poa_results/poa_optimization_T8.json"),
        # Path("results/temporary_poa_results/poa_optimization_T8_ratio_piecewise_mccormick.json"),
    ]
    results_dir = Path("results/temporary_poa_results")
    plot_all_matching_results = True
    output_dir = DEFAULT_OUTPUT_DIR
    show = False

    selected_result_paths = _resolve_result_paths(
        result_paths=result_paths,
        results_dir=results_dir,
        plot_all_matching_results=plot_all_matching_results,
    )
    for result_path in selected_result_paths:
        output_path = plot_ambiguity_regime_trajectories(
            result_path=result_path,
            output_dir=output_dir,
            show=show,
        )
        print(f"Saved ambiguity-regime trajectory plot: {output_path}")


if __name__ == "__main__":
    main()
