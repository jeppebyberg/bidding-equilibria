"""Summarize base-PoA compute time and binary counts across the horizon sweep.

Reads each run's ``poa/poa_optimization_T*.json`` (written by block3) and pulls
the solve wall time and binary-variable counts out of its ``solver`` block,
then writes a cross-horizon comparison as CSV + JSON and two plots.

Run (after horizon_poa_compute_sweep):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.horizon_poa_compute_summary
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.horizon_poa_compute_sweep import (  # noqa: E402
    HIDDEN_LAYERS,
    HORIZONS,
    STUDY_NAME,
    run_label,
    run_name,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "sensitivity_studies" / STUDY_NAME


def _find_poa_result(run_dir: Path) -> Path | None:
    """Return the run's PoA result JSON, or None if it was not produced."""
    poa_dir = run_dir / "poa"
    if not poa_dir.exists():
        return None
    matches = sorted(poa_dir.glob("poa_optimization_T*.json"))
    return matches[0] if matches else None


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    """Pull compute time and binary counts from one run's PoA result."""
    result_path = _find_poa_result(run_dir)
    if result_path is None:
        return None
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    solver = result.get("solver", {}) or {}
    counts = solver.get("variable_counts", {}) or {}
    objective = result.get("objective", {}) or {}
    return {
        "num_time_steps": result.get("num_time_steps"),
        "wall_time_seconds": solver.get("wall_time_seconds"),
        "solver_reported_time_seconds": solver.get("solver_reported_time_seconds"),
        "termination_condition": solver.get("termination_condition"),
        "mip_gap": solver.get("mip_gap"),
        "num_binary_variables": counts.get("num_binary_variables"),
        "num_binary_variables_free": counts.get("num_binary_variables_free"),
        "num_binary_variables_fixed": counts.get("num_binary_variables_fixed"),
        "num_continuous_variables": counts.get("num_continuous_variables"),
        "PoA": objective.get("PoA"),
        "result_path": str(result_path),
    }


def collect_summary() -> dict[str, Any]:
    """Build the cross-horizon summary, preserving HORIZONS order."""
    runs: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        name = run_name(horizon)
        run_dir = RESULT_ROOT / name
        loaded = _load_run(run_dir)
        if loaded is None:
            print(f"  [skip] {name}: no PoA result found under {run_dir / 'poa'}")
            continue
        runs.append({"run_name": name, "horizon": horizon, **loaded})
    return {
        "study": STUDY_NAME,
        "hidden_layers": list(HIDDEN_LAYERS),
        "runs": runs,
    }


_FIELDNAMES = [
    "run_name",
    "horizon",
    "wall_time_seconds",
    "solver_reported_time_seconds",
    "num_binary_variables",
    "num_binary_variables_free",
    "num_binary_variables_fixed",
    "num_continuous_variables",
    "termination_condition",
    "mip_gap",
    "PoA",
]


def write_csv(summary: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["runs"])


def _fmt(value: Any, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def print_table(summary: dict[str, Any]) -> None:
    header = (
        f"{'horizon':>8}{'wall (s)':>12}{'solver (s)':>12}"
        f"{'binaries':>12}{'free bin':>12}{'fixed bin':>12}{'PoA':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for run in summary["runs"]:
        print(
            f"{run['horizon']:>8}"
            f"{_fmt(run['wall_time_seconds'], '.2f'):>12}"
            f"{_fmt(run['solver_reported_time_seconds'], '.2f'):>12}"
            f"{_fmt(run['num_binary_variables'], 'd'):>12}"
            f"{_fmt(run['num_binary_variables_free'], 'd'):>12}"
            f"{_fmt(run['num_binary_variables_fixed'], 'd'):>12}"
            f"{_fmt(run['PoA'], '.4f'):>10}"
        )
    print()


def plot_summary(
    summary: dict[str, Any],
    poa_path: Path,
    time_path: Path,
    binary_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = summary["runs"]
    if not runs:
        return
    horizons = [r["horizon"] for r in runs]
    arch = "[" + ", ".join(str(n) for n in summary["hidden_layers"]) + "]"

    poa = [r["PoA"] for r in runs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(horizons, poa, marker="o")
    ax.set_xlabel("horizon (time steps)")
    ax.set_ylabel("Price of Anarchy")
    ax.set_title(f"Base-PoA vs horizon (NN {arch})")
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(poa_path, dpi=150)
    plt.close(fig)

    wall = [r["wall_time_seconds"] for r in runs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(horizons, wall, marker="o")
    ax.set_xlabel("horizon (time steps)")
    ax.set_ylabel("PoA solve wall time (s)")
    ax.set_title(f"Base-PoA solve time vs horizon (NN {arch})")
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(time_path, dpi=150)
    plt.close(fig)

    total = [r["num_binary_variables"] for r in runs]
    free = [r["num_binary_variables_free"] for r in runs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(horizons, total, marker="o", label="total binaries")
    ax.plot(horizons, free, marker="s", label="free binaries (post-tightening)")
    ax.set_xlabel("horizon (time steps)")
    ax.set_ylabel("binary variables")
    ax.set_title(f"Base-PoA binary count vs horizon (NN {arch})")
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(binary_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Base-PoA compute summary for study '{STUDY_NAME}'")
    summary = collect_summary()
    if not summary["runs"]:
        print("No PoA results found. Run the sweep first.")
        return

    print_table(summary)

    json_path = RESULT_ROOT / "compute_summary.json"
    csv_path = RESULT_ROOT / "compute_summary.csv"
    poa_plot_path = RESULT_ROOT / "poa_vs_horizon.png"
    time_plot_path = RESULT_ROOT / "compute_time_vs_horizon.png"
    binary_plot_path = RESULT_ROOT / "binaries_vs_horizon.png"

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_csv(summary, csv_path)
    try:
        plot_summary(summary, poa_plot_path, time_plot_path, binary_plot_path)
        plotted = f"{poa_plot_path}\n  {time_plot_path}\n  {binary_plot_path}"
    except Exception as exc:  # plotting is optional
        plotted = f"(skipped: {exc})"

    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {plotted}")


if __name__ == "__main__":
    main()
