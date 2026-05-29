"""Plot DRO PoA values as a function of eta for one regime."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = Path("results/dro_poa")
DEFAULT_OUTPUT_ROOT = Path("results_viz/figures/dro_poa_eta_sweep")

METRIC_LABELS = {
    "average_poa_difference": "Average PoA difference",
    "average_poa_ratio": "Average PoA ratio",
    "inner_objective": "DRO inner objective",
    "dro_objective_with_epsilon": "DRO objective + eta * epsilon",
    "average_wasserstein_distance": "Average Wasserstein distance",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _average_poa_ratio(result: dict[str, Any]) -> float | None:
    top_level_ratio = _as_optional_float(result.get("average_poa_ratio"))
    if top_level_ratio is not None:
        return top_level_ratio

    ratios = []
    for scenario in result.get("scenarios", []) or []:
        c_eq = _as_optional_float(scenario.get("C_eq"))
        c_opt = _as_optional_float(scenario.get("C_opt"))
        if c_eq is None or c_opt in (None, 0.0):
            continue
        ratios.append(c_eq / c_opt)
    if not ratios:
        return None
    return float(np.mean(ratios))


def _average_poa_difference(result: dict[str, Any]) -> float | None:
    top_level_difference = _as_optional_float(result.get("average_poa_difference"))
    if top_level_difference is not None:
        return top_level_difference

    differences = []
    for scenario in result.get("scenarios", []) or []:
        difference = _as_optional_float(scenario.get("PoA_difference"))
        if difference is not None:
            differences.append(difference)
    if not differences:
        return None
    return float(np.mean(differences))


def _summary_record(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "result_path": str(path),
        "source_dir": str(path.parent),
        "source_mtime": path.stat().st_mtime,
        "reference_case": result.get("reference_case"),
        "regime_set": result.get("regime_set"),
        "regime_name": result.get("regime_name"),
        "num_time_steps": result.get("num_time_steps"),
        "num_empirical_scenarios": result.get("num_empirical_scenarios"),
        "eta": _as_optional_float(result.get("eta")),
        "epsilon": _as_optional_float(result.get("epsilon")),
        "inner_objective": _as_optional_float(result.get("inner_objective")),
        "dro_objective_with_epsilon": _as_optional_float(
            result.get("dro_objective_with_epsilon")
        ),
        "average_poa_difference": _average_poa_difference(result),
        "average_poa_ratio": _average_poa_ratio(result),
        "average_wasserstein_distance": _as_optional_float(
            result.get("average_wasserstein_distance")
        ),
    }
    solver = result.get("solver", {}) or {}
    record["solver_status"] = solver.get("status")
    record["solver_termination_condition"] = solver.get("termination_condition")
    return record


def discover_regime_names(results_dir: Path) -> list[str]:
    """Return all regime names that have at least one eta-sweep JSON file."""
    if not results_dir.exists():
        return []
    skip = {"old_results"}
    names = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in skip:
            continue
        if any(subdir.glob("dro_poa_eta_*_T*.json")):
            names.append(subdir.name)
    return names


def _candidate_result_paths(
    results_dir: Path,
    regime_name: str,
    include_archives: bool,
) -> list[Path]:
    regime_dir = results_dir / regime_name
    paths = []
    if regime_dir.exists():
        paths.extend(regime_dir.glob("dro_poa_eta_*_T*.json"))

    archive_root = results_dir / "old_results"
    if include_archives and archive_root.exists():
        for archived_regime_dir in archive_root.glob(f"{regime_name}_*"):
            if archived_regime_dir.is_dir():
                paths.extend(archived_regime_dir.glob("dro_poa_eta_*_T*.json"))

    if not paths:
        available = sorted(
            path.name for path in results_dir.iterdir() if path.is_dir()
        ) if results_dir.exists() else []
        raise FileNotFoundError(
            f"No DRO PoA eta-sweep JSON files found for regime '{regime_name}' "
            f"under {results_dir}. "
            f"Available regimes: {', '.join(available) or '<none>'}"
        )
    return sorted(paths)


def load_eta_sweep_records(
    results_dir: Path,
    regime_name: str,
    include_archives: bool = True,
) -> list[dict[str, Any]]:
    records = []
    for path in _candidate_result_paths(results_dir, regime_name, include_archives):
        result = _load_json(path)
        if result.get("regime_name") not in (None, regime_name):
            continue
        record = _summary_record(path, result)
        if record["eta"] is not None and record["epsilon"] is not None:
            records.append(record)

    if not records:
        raise FileNotFoundError(
            f"No DRO PoA eta-sweep records with eta and epsilon found for "
            f"regime '{regime_name}'"
        )
    return select_latest_sweep_per_epsilon(records)


def select_latest_sweep_per_epsilon(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_mtime_by_key: dict[tuple[float, str], float] = {}
    for record in records:
        key = (float(record["epsilon"]), str(record["source_dir"]))
        source_mtime_by_key[key] = max(
            source_mtime_by_key.get(key, float("-inf")),
            float(record["source_mtime"]),
        )

    latest_source_by_epsilon: dict[float, str] = {}
    latest_mtime_by_epsilon: dict[float, float] = {}
    for (epsilon, source_dir), source_mtime in source_mtime_by_key.items():
        if source_mtime > latest_mtime_by_epsilon.get(epsilon, float("-inf")):
            latest_mtime_by_epsilon[epsilon] = source_mtime
            latest_source_by_epsilon[epsilon] = source_dir

    selected = [
        record
        for record in records
        if str(record["source_dir"])
        == latest_source_by_epsilon.get(float(record["epsilon"]))
    ]

    latest_by_epsilon_eta: dict[tuple[float, float], dict[str, Any]] = {}
    for record in selected:
        key = (float(record["epsilon"]), float(record["eta"]))
        previous = latest_by_epsilon_eta.get(key)
        if previous is None or float(record["source_mtime"]) > float(
            previous["source_mtime"]
        ):
            latest_by_epsilon_eta[key] = record

    records = list(latest_by_epsilon_eta.values())
    return sorted(records, key=lambda record: float(record["eta"]))


def write_summary_csv(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eta",
        "average_poa_difference",
        "average_poa_ratio",
        "inner_objective",
        "dro_objective_with_epsilon",
        "average_wasserstein_distance",
        "epsilon",
        "num_time_steps",
        "num_empirical_scenarios",
        "solver_status",
        "solver_termination_condition",
        "source_dir",
        "result_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    return output_path


def _records_by_epsilon(records: list[dict[str, Any]]) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(float(record["epsilon"]), []).append(record)
    return {
        epsilon: sorted(epsilon_records, key=lambda record: float(record["eta"]))
        for epsilon, epsilon_records in sorted(grouped.items())
    }


def _plot_metric_by_epsilon(
    axis: plt.Axes,
    grouped_records: dict[float, list[dict[str, Any]]],
    metric: str,
) -> bool:
    plotted_any = False
    for epsilon, epsilon_records in grouped_records.items():
        etas = np.asarray(
            [float(record["eta"]) for record in epsilon_records],
            dtype=float,
        )
        values = np.asarray(
            [
                np.nan
                if record.get(metric) is None
                else float(record[metric])
                for record in epsilon_records
            ],
            dtype=float,
        )
        if np.all(np.isnan(values)):
            continue
        plotted_any = True
        axis.plot(
            etas,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=f"epsilon={epsilon:.4g}",
        )

    axis.set_ylabel(METRIC_LABELS[metric])
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(True, alpha=0.25)
    return plotted_any


def plot_poa_eta_sweep(
    records: list[dict[str, Any]],
    output_dir: Path,
    regime_name: str,
    show: bool = False,
) -> Path:
    grouped_records = _records_by_epsilon(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)

    plotted_ratio = _plot_metric_by_epsilon(
        axes[0],
        grouped_records,
        "average_poa_ratio",
    )
    plotted_difference = _plot_metric_by_epsilon(
        axes[1],
        grouped_records,
        "average_poa_difference",
    )
    if not plotted_ratio and not plotted_difference:
        raise ValueError("PoA ratio and PoA difference are missing for all epsilon curves.")

    axes[0].set_title(f"DRO PoA eta sweep: {regime_name}")
    axes[0].legend(loc="best", frameon=True)
    axes[1].set_xlabel("eta")

    horizon = records[0].get("num_time_steps")
    n_scenarios = records[0].get("num_empirical_scenarios")
    context = []
    if horizon is not None:
        context.append(f"T={horizon}")
    if n_scenarios is not None:
        context.append(f"N={n_scenarios}")
    n_epsilons = len(grouped_records)
    context.append(f"{n_epsilons} epsilon curve{'s' if n_epsilons != 1 else ''}")
    if context:
        axes[1].text(
            0.01,
            0.02,
            ", ".join(context),
            transform=axes[1].transAxes,
            fontsize=9,
            color="0.35",
        )

    fig.tight_layout()
    output_path = output_dir / f"{regime_name}_poa_by_eta_by_epsilon.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_poa_epsilon_frontier(
    records: list[dict[str, Any]],
    output_dir: Path,
    regime_name: str,
    poa_metric: str = "inner_objective",
    show: bool = False,
) -> Path:
    """Plot worst-case PoA vs achieved Wasserstein distance, tangent slope = eta at each point.

    Each (eta, epsilon_cap) pair in the sweep traces one point on the PoA–epsilon
    efficient frontier.  The envelope theorem guarantees that dv/d(epsilon) = eta at
    the optimum, so the tangent line drawn at every point has slope exactly eta.
    Large eta pins the solution near epsilon=0 (steep tangent); eta->0 reaches the
    robust plateau (flat tangent).
    """
    grouped = _records_by_epsilon(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for color_idx, (eps_cap, eps_records) in enumerate(grouped.items()):
        color = colors[color_idx % len(colors)]
        points: list[tuple[float, float, float]] = []
        for r in eps_records:
            x = _as_optional_float(r.get("average_wasserstein_distance"))
            y = _as_optional_float(r.get(poa_metric))
            eta = _as_optional_float(r.get("eta"))
            if x is not None and y is not None and eta is not None:
                points.append((x, y, eta))

        if not points:
            continue

        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        etas = [p[2] for p in points]

        label = f"ε_cap={eps_cap:.4g}" if eps_cap != 0.0 else "ε_cap=0"
        ax.plot(xs, ys, "o-", color=color, linewidth=2.0, markersize=6, label=label, zorder=3)

        x_span = max(xs) - min(xs) if len(xs) > 1 else (xs[0] if xs[0] > 0 else 1.0)
        half_len = x_span * 0.10

        for x0, y0, eta in zip(xs, ys, etas):
            ax.plot(
                [x0 - half_len, x0 + half_len],
                [y0 - eta * half_len, y0 + eta * half_len],
                color=color,
                linewidth=1.0,
                alpha=0.55,
                zorder=2,
            )
            ax.annotate(
                f"η={eta:.3g}",
                xy=(x0, y0),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Achieved Wasserstein distance (ε)")
    ax.set_ylabel(METRIC_LABELS.get(poa_metric, poa_metric))
    ax.set_title(
        f"PoA–ε frontier: {regime_name}\n"
        "tangent slope = η at each point  (envelope theorem)"
    )
    ax.legend(loc="best", frameon=True)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{regime_name}_poa_epsilon_frontier.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("*.png", "eta_sweep_summary.csv"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    # Edit these paths/settings directly when running this script.
    results_dir = DEFAULT_RESULTS_DIR
    regime: str | None = None  # set to a regime name to plot only that one, or None for all
    output_root = DEFAULT_OUTPUT_ROOT
    include_archives = True
    show = False

    regimes = discover_regime_names(results_dir) if regime is None else [regime]
    if not regimes:
        print(f"No eta-sweep results found under {results_dir}")
        return

    for regime_name in regimes:
        try:
            records = load_eta_sweep_records(
                results_dir,
                regime_name,
                include_archives=include_archives,
            )
        except FileNotFoundError as exc:
            print(f"Skipping '{regime_name}': {exc}")
            continue
        output_dir = output_root / regime_name
        clean_output_dir(output_dir)
        csv_path = write_summary_csv(records, output_dir / "eta_sweep_summary.csv")
        figure_path = plot_poa_eta_sweep(
            records=records,
            output_dir=output_dir,
            regime_name=regime_name,
            show=show,
        )
        frontier_path = plot_poa_epsilon_frontier(
            records=records,
            output_dir=output_dir,
            regime_name=regime_name,
            poa_metric="inner_objective",
            show=show,
        )
        print(f"Saved DRO PoA eta-sweep figure:   {figure_path}")
        print(f"Saved DRO PoA ε-frontier figure:  {frontier_path}")
        print(f"Saved DRO PoA eta-sweep summary:  {csv_path}")


if __name__ == "__main__":
    main()
