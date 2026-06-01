"""Plot DRO PoA values as a function of eta for one regime."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, NamedTuple

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


def _epsilon_label(epsilon: float, labels: dict[float, str] | None) -> str:
    """Return a display label for epsilon, using a custom mapping if one is provided.

    Lookup uses a relative tolerance of 1e-6 so that float-serialised epsilon
    values (e.g. 2000.0000000001) still match a key of 2000.0.
    Falls back to a plain numeric string when no match is found.
    """
    if labels:
        for key, text in labels.items():
            tol = 1e-6 * max(1.0, abs(float(key)))
            if abs(float(key) - epsilon) <= tol:
                return text
    return f"eps_cap = {epsilon:.4g}" if epsilon != 0.0 else "SAA (eps_cap = 0)"


def _plot_metric_by_epsilon(
    axis: plt.Axes,
    grouped_records: dict[float, list[dict[str, Any]]],
    metric: str,
    epsilon_labels: dict[float, str] | None = None,
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
            label=_epsilon_label(epsilon, epsilon_labels),
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
    epsilon_labels: dict[float, str] | None = None,
) -> Path:
    grouped_records = _records_by_epsilon(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)

    plotted_ratio = _plot_metric_by_epsilon(axes[0], grouped_records, "average_poa_ratio",
                                            epsilon_labels=epsilon_labels)
    plotted_difference = _plot_metric_by_epsilon(axes[1], grouped_records, "average_poa_difference",
                                                 epsilon_labels=epsilon_labels)
    if not plotted_ratio and not plotted_difference:
        raise ValueError("PoA ratio and PoA difference are missing for all epsilon curves.")

    # Log scale when eta spans more than one order of magnitude.
    all_etas = [
        float(r["eta"])
        for r in records
        if r.get("eta") is not None and float(r["eta"]) > 0
    ]
    if all_etas and max(all_etas) / min(all_etas) > 20:
        axes[1].set_xscale("log")

    # Reference line: competitive PoA ratio = 1.
    if plotted_ratio:
        axes[0].axhline(1.0, color="0.4", linewidth=1.2, linestyle="--", alpha=0.6,
                        zorder=1, label="PoA ratio = 1 (competitive)")

    axes[0].set_title(f"DRO PoA η sweep  —  {regime_name}", fontsize=12)
    axes[0].legend(loc="best", frameon=True, fontsize=9)
    axes[0].set_ylabel(METRIC_LABELS["average_poa_ratio"])
    axes[1].set_xlabel("η  (Wasserstein penalty)", fontsize=10)

    horizon = records[0].get("num_time_steps")
    n_scenarios = records[0].get("num_empirical_scenarios")
    context_parts = []
    if horizon is not None:
        context_parts.append(f"T = {horizon}")
    if n_scenarios is not None:
        context_parts.append(f"N = {n_scenarios}")
    n_epsilons = len(grouped_records)
    context_parts.append(f"{n_epsilons} ε curve{'s' if n_epsilons != 1 else ''}")
    if context_parts:
        axes[1].text(
            0.01, 0.97, "  ".join(context_parts),
            transform=axes[1].transAxes,
            fontsize=8.5, color="0.40", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.85),
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
    oos_mean_poa: float | None = None,
    oos_max_poa: float | None = None,
    show: bool = False,
    epsilon_labels: dict[float, str] | None = None,
) -> Path:
    """Plot worst-case PoA vs achieved Wasserstein distance.

    Each (η, ε_cap) pair traces one point on the PoA–ε efficient frontier.
    The envelope theorem guarantees dv/dε = η at each optimally solved point, so
    η labels are shown sparsely (at most 5 per curve) to avoid clutter.

    If oos_mean_poa is provided, a horizontal dashed line marks that level and
    the calibrated ε* crossing is marked with a star.
    """
    import math

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
        n = len(points)

        ax.plot(xs, ys, "o-", color=color, linewidth=2.0, markersize=5, zorder=3,
                label=_epsilon_label(eps_cap, epsilon_labels))

        # Annotate at most 5 points per curve, always including first and last.
        step = max(1, math.ceil(n / 5))
        annotated_indices = sorted(set(range(0, n, step)) | {n - 1})
        for i in annotated_indices:
            ax.annotate(
                f"η={etas[i]:.2g}",
                xy=(xs[i], ys[i]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.85,
            )

    # OOS overlays.
    for label_name, poa_value, oos_color, line_style in [
        ("mean", oos_mean_poa, "tab:orange", "--"),
        ("max", oos_max_poa, "tab:red", "-."),
    ]:
        if poa_value is None or not np.isfinite(poa_value):
            continue
        ax.axhline(
            poa_value,
            color=oos_color,
            linestyle=line_style,
            linewidth=1.5,
            zorder=4,
            label=f"OOS {label_name} PoA = {poa_value:.4g}",
        )
        try:
            ce = calibrated_epsilon(records, poa_value, poa_metric)
            if ce["in_range"] and ce["eps_star"] is not None:
                eps_star = float(ce["eps_star"])
                ax.axvline(eps_star, color=oos_color, linestyle=":", linewidth=1.5, zorder=4)
                ax.plot(
                    eps_star, poa_value,
                    marker="*", markersize=13, color=oos_color,
                    zorder=5, label=f"ε* = {eps_star:.4g}", linestyle="none",
                )
        except (ValueError, StopIteration):
            pass  # frontier too sparse for calibration — overlay line still shown

    ax.set_xlabel("Achieved Wasserstein distance (ε)", fontsize=10)
    ax.set_ylabel(METRIC_LABELS.get(poa_metric, poa_metric), fontsize=10)
    ax.set_title(
        f"PoA–ε frontier  —  {regime_name}\n"
        "η labels show marginal cost of robustness  (dPoA/dε = η, envelope theorem)",
        fontsize=11,
    )
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{regime_name}_poa_epsilon_frontier.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


class _FrontierPoint(NamedTuple):
    eps: float  # average_wasserstein_distance — achieved Wasserstein radius ε(η)
    poa: float  # worst-case expected PoA at this η
    eta: float  # Lagrangian multiplier; by the envelope theorem η = dPoA/dε exactly


def _extract_frontier(
    records: list[dict[str, Any]],
    poa_metric: str,
) -> list[_FrontierPoint]:
    """Return valid (eps, poa, eta) triples sorted by achieved ε ascending."""
    points = []
    for r in records:
        eps = _as_optional_float(r.get("average_wasserstein_distance"))
        poa = _as_optional_float(r.get(poa_metric))
        eta = _as_optional_float(r.get("eta"))
        if eps is not None and poa is not None and eta is not None:
            points.append(_FrontierPoint(eps=eps, poa=poa, eta=eta))
    return sorted(points, key=lambda p: p.eps)


def _check_frontier_shape(points: list[_FrontierPoint], poa_metric: str) -> None:
    """Warn if the frontier violates non-decreasing PoA or non-increasing η in ε.

    The PoA-ε frontier should be non-decreasing (more robustness → higher worst-case PoA)
    and concave (slopes η non-increasing as ε grows). Violations indicate a coarse or
    non-monotone η grid. The data is not corrected — only a warning is emitted.
    """
    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]
        if curr.poa < prev.poa - 1e-9:
            warnings.warn(
                f"Frontier non-monotone at eps={curr.eps:.6g}: {poa_metric} fell from "
                f"{prev.poa:.6g} to {curr.poa:.6g}. Re-gridding eta may resolve this.",
                stacklevel=3,
            )
        if curr.eta > prev.eta + 1e-9:
            warnings.warn(
                f"Frontier non-concave at eps={curr.eps:.6g}: eta rose from "
                f"{prev.eta:.6g} to {curr.eta:.6g}. Re-gridding eta may resolve this.",
                stacklevel=3,
            )


def query_frontier(
    records: list[dict[str, Any]],
    eps_query: float,
    poa_metric: str = "inner_objective",
) -> dict[str, Any]:
    """Query the PoA-ε frontier at a requested Wasserstein trust radius.

    Treats the sweep output as a lookup table and linearly interpolates between
    the two bracketing sweep points in ε. Emits warnings if the frontier violates
    the expected monotonicity or concavity before interpolating.

    Args:
        records:    Sweep records as returned by load_eta_sweep_records().
        eps_query:  Requested achieved Wasserstein radius ε to query.
        poa_metric: PoA metric to use as the y-axis (key in METRIC_LABELS);
                    defaults to "inner_objective".

    Returns:
        dict with keys:
            eps_query  – the queried ε.
            poa        – interpolated worst-case expected PoA, or None if out of range.
            eta        – interpolated marginal cost dPoA/dε at eps_query, or None.
            in_range   – True iff eps_query lies within [min ε, max ε] of the sweep.
            message    – one-line interpretation string.
    """
    points = _extract_frontier(records, poa_metric)
    if len(points) < 2:
        raise ValueError(
            f"Need at least 2 valid frontier points for '{poa_metric}'; got {len(points)}."
        )
    _check_frontier_shape(points, poa_metric)

    eps_arr = np.array([p.eps for p in points])
    poa_arr = np.array([p.poa for p in points])
    # Source: stored η from sweep records. The envelope theorem guarantees η = dPoA/dε
    # exactly at each optimally solved point, so we read η directly rather than
    # finite-differencing the PoA values.
    eta_arr = np.array([p.eta for p in points])

    eps_min, eps_max = float(eps_arr[0]), float(eps_arr[-1])

    if eps_query < eps_min:
        return {
            "eps_query": eps_query,
            "poa": None,
            "eta": None,
            "in_range": False,
            "message": (
                f"eps={eps_query:.6g} is below the swept range [{eps_min:.6g}, {eps_max:.6g}] "
                "(below the SAA point). Extrapolation is not supported."
            ),
        }
    if eps_query > eps_max:
        return {
            "eps_query": eps_query,
            "poa": None,
            "eta": None,
            "in_range": False,
            "message": (
                f"eps={eps_query:.6g} is above the swept range [{eps_min:.6g}, {eps_max:.6g}] "
                "(beyond the robust plateau). Extrapolation is not supported."
            ),
        }

    poa_interp = float(np.interp(eps_query, eps_arr, poa_arr))
    eta_interp = float(np.interp(eps_query, eps_arr, eta_arr))

    message = (
        f"eps={eps_query:.6g} -> plan for worst-case PoA {poa_interp:.4g}; "
        f"marginal cost of robustness eta={eta_interp:.4g} (PoA per unit eps)."
    )
    return {
        "eps_query": eps_query,
        "poa": poa_interp,
        "eta": eta_interp,
        "in_range": True,
        "message": message,
    }


def calibrated_epsilon(
    records: list[dict[str, Any]],
    oos_mean_poa: float,
    poa_metric: str = "inner_objective",
) -> dict[str, Any]:
    """Return the smallest ε* on the frontier whose worst-case PoA meets or exceeds oos_mean_poa.

    Locates the calibration crossing: the smallest Wasserstein trust radius at
    which the worst-case plan is at least as conservative as the realized OOS PoA.
    Interpolates linearly between the two bracketing frontier points.

    Use driver/run_oos_poa_evaluation.py to compute oos_mean_poa from fresh
    scenarios before calling this function.

    Args:
        records:      Sweep records from load_eta_sweep_records().
        oos_mean_poa: Mean OOS realized PoA ratio from fresh unconstrained draws.
        poa_metric:   PoA metric key (same as query_frontier).

    Returns:
        dict with keys:
            eps_star        – calibrated ε* (or None if OOS PoA exceeds frontier).
            poa_at_eps_star – oos_mean_poa (the crossing value), or None.
            in_range        – True if a valid crossing exists within the sweep.
            message         – one-line interpretation string.
    """
    points = _extract_frontier(records, poa_metric)
    if len(points) < 2:
        raise ValueError(
            f"Need at least 2 valid frontier points for '{poa_metric}'; got {len(points)}."
        )
    _check_frontier_shape(points, poa_metric)

    poa_min = points[0].poa
    poa_max = points[-1].poa
    eps_min = points[0].eps
    eps_max = points[-1].eps

    # Even the tightest (SAA) point already covers oos_mean_poa
    if oos_mean_poa <= poa_min:
        return {
            "eps_star": eps_min,
            "poa_at_eps_star": poa_min,
            "in_range": True,
            "message": (
                f"eps*={eps_min:.6g}: even the SAA point (eps={eps_min:.6g}, "
                f"worst-case PoA={poa_min:.4g}) covers OOS PoA {oos_mean_poa:.4g}. "
                "No additional robustness required."
            ),
        }

    # OOS PoA exceeds the most robust frontier point — sweep range is insufficient
    if oos_mean_poa > poa_max:
        return {
            "eps_star": None,
            "poa_at_eps_star": None,
            "in_range": False,
            "message": (
                f"OOS PoA {oos_mean_poa:.4g} exceeds the maximum frontier PoA "
                f"{poa_max:.4g} at eps={eps_max:.6g}. "
                "Extend the eta sweep to larger eps to find a covering plan."
            ),
        }

    # Find first frontier point whose PoA >= oos_mean_poa
    cross_idx = next(i for i, p in enumerate(points) if p.poa >= oos_mean_poa)
    lo, hi = points[cross_idx - 1], points[cross_idx]

    # Linear interpolation in PoA space to find ε*
    t = (oos_mean_poa - lo.poa) / (hi.poa - lo.poa)
    eps_star = lo.eps + t * (hi.eps - lo.eps)

    return {
        "eps_star": eps_star,
        "poa_at_eps_star": oos_mean_poa,
        "in_range": True,
        "message": (
            f"eps*={eps_star:.6g}: smallest trust radius whose worst-case PoA "
            f"covers the OOS realized PoA of {oos_mean_poa:.4g}."
        ),
    }


def _demo_query_frontier(records: list[dict[str, Any]], regime_name: str) -> None:
    """Print query_frontier results at the 25th, 50th, and 75th percentile of swept ε."""
    eps_values = np.asarray(
        [r["average_wasserstein_distance"] for r in records
         if r.get("average_wasserstein_distance") is not None],
        dtype=float,
    )
    if len(eps_values) < 2:
        return
    eps_min, eps_max = float(eps_values.min()), float(eps_values.max())
    queries = [
        eps_min + 0.25 * (eps_max - eps_min),
        eps_min + 0.50 * (eps_max - eps_min),
        eps_min + 0.75 * (eps_max - eps_min),
    ]
    print(f"\nFrontier queries for regime '{regime_name}' (inner_objective):")
    for eps_q in queries:
        result = query_frontier(records, eps_q)
        print(f"  {result['message']}")


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("*.png", "eta_sweep_summary.csv"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _load_oos_poa_by_regime(
    oos_results_path: Path,
) -> dict[str, dict[str, float]]:
    """Load oos_poa_results.json and return OOS mean/max PoA by regime."""
    if not oos_results_path.exists():
        return {}
    with oos_results_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    result: dict[str, dict[str, float]] = {}
    for entry in entries if isinstance(entries, list) else []:
        reg = entry.get("regime_name")
        if not reg:
            continue
        stats: dict[str, float] = {}
        for key, stat_name in (
            ("oos_mean_poa_ratio", "mean"),
            ("oos_max_poa_ratio", "max"),
        ):
            val = entry.get(key)
            if val is None and key == "oos_max_poa_ratio":
                ratios = entry.get("poa_ratios") or []
                try:
                    val = max(float(ratio) for ratio in ratios)
                except (TypeError, ValueError):
                    val = None
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fval):
                stats[stat_name] = fval
        if stats:
            result[str(reg)] = stats
    return result


def main() -> None:
    # Edit these paths/settings directly when running this script.
    results_dir = DEFAULT_RESULTS_DIR
    regime: str | None = None  # set to a regime name to plot only that one, or None for all
    output_root = DEFAULT_OUTPUT_ROOT
    include_archives = True
    show = False
    # Path to OOS evaluation results (produced by driver/run_oos_poa_evaluation.py).
    # Set to None to skip the OOS overlay entirely.
    oos_results_path: Path | None = Path("results/oos_poa/oos_poa_results.json")

    # Human-readable labels for epsilon_cap values shown in the plots.
    # Keys are the numeric epsilon_cap values stored in the result JSON.
    # If a value is not listed here, the plot falls back to "eps_cap = X".
    # Example for a multi-cap experiment:
    #   EPSILON_LABELS = {0.0: "SAA", 0.5: "tight", 5.0: "moderate", 2000.0: "unconstrained"}
    EPSILON_LABELS: dict[float, str] = {
        0.0: "SAA (no robustness)",
        2000.0: "DRO — Wasserstein cap (no binding constraint)",
    }

    oos_poa_by_regime = _load_oos_poa_by_regime(oos_results_path) if oos_results_path else {}
    if oos_poa_by_regime:
        print(f"Loaded OOS PoA mean/max for regimes: {sorted(oos_poa_by_regime)}")

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
            epsilon_labels=EPSILON_LABELS,
        )
        frontier_path = plot_poa_epsilon_frontier(
            records=records,
            output_dir=output_dir,
            regime_name=regime_name,
            poa_metric="inner_objective",
            oos_mean_poa=oos_poa_by_regime.get(regime_name, {}).get("mean"),
            oos_max_poa=oos_poa_by_regime.get(regime_name, {}).get("max"),
            show=show,
            epsilon_labels=EPSILON_LABELS,
        )
        print(f"Saved DRO PoA eta-sweep figure:   {figure_path}")
        print(f"Saved DRO PoA eps-frontier figure: {frontier_path}")
        print(f"Saved DRO PoA eta-sweep summary:  {csv_path}")
        _demo_query_frontier(records, regime_name)


if __name__ == "__main__":
    main()
