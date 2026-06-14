"""Summarize tightening time vs. PoA compute time across the tightening progression.

Reads, per cumulative case:
  - poa/tightening/final_tightening_report.json -> ``stage_timings`` (per-stage
    tightening wall time) = time spent on tightening
  - poa/poa_optimization_T*.json                -> ``solver.wall_time_seconds``
    (PoA solve/compute time), variable_counts, objective.PoA

and reports, for each case in cumulative order, the tightening time (total and
per stage), the PoA solve time, their sum, and the resulting MILP size / PoA.
As stages are added the tightening time grows but the MILP should get smaller /
faster to solve -- the trade-off this study is meant to expose.

Writes the cross-case comparison as CSV + JSON and two plots:
  - tightening vs. compute time per case (stacked = total wall time)
  - per-stage tightening time per case (stacked breakdown)

Run (after bound_tightening_progression):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.bound_tightening_progression_summary
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

from driver.sensitivity.bound_tightening_progression import (  # noqa: E402
    HORIZON,
    RESULT_ROOT,
    STUDY_NAME,
    TIGHTENING_CASES,
    run_dir,
)

# All tightening stages, in execution order, for the per-stage breakdown.
STAGE_ORDER = (
    "primal_big_m",
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
    "optimal_cost_bounds",
    "equilibrium_cost_bounds",
)

STAGE_COLORS = {
    "primal_big_m": "tab:gray",
    "relu_bounds": "tab:blue",
    "alpha_bounds": "tab:orange",
    "slack_binary_fix": "tab:green",
    "dual_big_m": "tab:red",
    "optimal_cost_bounds": "tab:purple",
    "equilibrium_cost_bounds": "tab:brown",
}


def _find_poa_result(run_dir_path: Path) -> Path | None:
    poa_dir = run_dir_path / "poa"
    if not poa_dir.exists():
        return None
    matches = sorted(poa_dir.glob("poa_optimization_T*.json"))
    return matches[0] if matches else None


def _load_stage_timings(run_dir_path: Path) -> dict[str, dict[str, Any]]:
    report_path = run_dir_path / "poa" / "tightening" / "final_tightening_report.json"
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    return report.get("stage_timings", {}) or {}


def _load_run(case_name: str, label: str) -> dict[str, Any] | None:
    """Pull tightening timings + PoA compute metrics for one case."""
    run_dir_path = run_dir(case_name)
    result_path = _find_poa_result(run_dir_path)
    if result_path is None:
        return None
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    solver = result.get("solver", {}) or {}
    counts = solver.get("variable_counts", {}) or {}
    objective = result.get("objective", {}) or {}

    timings = _load_stage_timings(run_dir_path)
    # Per-stage tightening wall time (seconds); stages not present default to 0.
    stage_seconds = {
        stage: float(timings.get(stage, {}).get("computation_time_seconds", 0.0) or 0.0)
        for stage in STAGE_ORDER
    }
    # Stages that actually ran this case (mode == "run"), for the table.
    ran_stages = [
        stage for stage in STAGE_ORDER if timings.get(stage, {}).get("mode") == "run"
    ]
    tightening_seconds = sum(stage_seconds.values())
    compute_seconds = solver.get("wall_time_seconds")

    total_seconds = None
    if isinstance(compute_seconds, (int, float)):
        total_seconds = tightening_seconds + float(compute_seconds)

    return {
        "case_name": case_name,
        "label": label,
        "ran_stages": ran_stages,
        "stage_seconds": stage_seconds,
        "tightening_seconds": tightening_seconds,
        "compute_seconds": compute_seconds,
        "total_seconds": total_seconds,
        "termination_condition": solver.get("termination_condition"),
        "mip_gap": solver.get("mip_gap"),
        "solver_threads": solver.get("solver_threads"),
        "solver_seed": solver.get("solver_seed"),
        "num_binary_variables": counts.get("num_binary_variables"),
        "num_binary_variables_free": counts.get("num_binary_variables_free"),
        "num_binary_variables_fixed": counts.get("num_binary_variables_fixed"),
        "PoA": objective.get("PoA"),
        "ex_post_ratio": objective.get("ex_post_ratio"),
        "result_path": str(result_path),
    }


def collect_summary() -> dict[str, Any]:
    """Build the cross-case summary, preserving the cumulative case order."""
    runs: list[dict[str, Any]] = []
    for case_name, label, _stages in TIGHTENING_CASES:
        loaded = _load_run(case_name, label)
        if loaded is None:
            print(f"  [skip] {case_name}: no PoA result found under {run_dir(case_name) / 'poa'}")
            continue
        runs.append(loaded)
    return {"study": STUDY_NAME, "runs": runs}


_FIELDNAMES = [
    "case_name",
    "label",
    "tightening_seconds",
    "compute_seconds",
    "total_seconds",
    "num_binary_variables",
    "num_binary_variables_free",
    "num_binary_variables_fixed",
    "PoA",
    "ex_post_ratio",
    "termination_condition",
    "mip_gap",
]


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for run in summary["runs"]:
        row = {key: run.get(key) for key in _FIELDNAMES}
        # Flatten per-stage tightening seconds into their own columns.
        for stage in STAGE_ORDER:
            row[f"tighten_{stage}_s"] = run["stage_seconds"].get(stage)
        rows.append(row)
    fieldnames = _FIELDNAMES + [f"tighten_{stage}_s" for stage in STAGE_ORDER]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def print_table(summary: dict[str, Any]) -> None:
    header = (
        f"{'case':>16}{'tighten(s)':>12}{'compute(s)':>12}{'total(s)':>11}"
        f"{'binaries':>10}{'free':>8}{'PoA':>10}{'ex-post':>10}{'term':>10}"
    )
    print(header)
    print("-" * len(header))
    for run in summary["runs"]:
        print(
            f"{run['case_name']:>16}"
            f"{_fmt(run['tightening_seconds'], '.2f'):>12}"
            f"{_fmt(run['compute_seconds'], '.2f'):>12}"
            f"{_fmt(run['total_seconds'], '.2f'):>11}"
            f"{_fmt(run['num_binary_variables'], 'd'):>10}"
            f"{_fmt(run['num_binary_variables_free'], 'd'):>8}"
            f"{_fmt(run['PoA'], '.4f'):>10}"
            f"{_fmt(run['ex_post_ratio'], '.4f'):>10}"
            f"{str(run['termination_condition'] or '-'):>10}"
        )
    print()

    # Solve times are only 1:1 comparable when every case used identical Gurobi
    # Threads/Seed (multi-threaded/unseeded solves have a nondeterministic search
    # path). Surface the settings and warn loudly if they are not uniform.
    settings = {(r.get("solver_threads"), r.get("solver_seed")) for r in summary["runs"]}
    threads, seed = next(iter(settings))
    if len(settings) == 1:
        if threads is None or seed is None:
            print(
                f"  [warn] solver Threads={threads}, Seed={seed} (Gurobi default for "
                f"None) -- solve times are NOT deterministic/comparable 1:1.\n"
                f"         Pin POA_SOLVER_THREADS/POA_SOLVER_SEED and re-run.\n"
            )
        else:
            print(f"  solver settings (all cases): Threads={threads}, Seed={seed}\n")
    else:
        print(
            f"  [warn] cases used DIFFERENT solver settings {sorted(settings)} -- "
            f"solve times are NOT comparable 1:1.\n"
        )


def plot_summary(summary: dict[str, Any], time_path: Path, stage_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = summary["runs"]
    if not runs:
        return
    labels = [r["case_name"] for r in runs]
    x = np.arange(len(runs))

    # Plot 1: tightening vs. compute, stacked so the bar height is total wall time.
    tighten = [r["tightening_seconds"] or 0.0 for r in runs]
    compute = [r["compute_seconds"] if isinstance(r["compute_seconds"], (int, float)) else 0.0 for r in runs]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, tighten, color="tab:orange", label="tightening time")
    ax.bar(x, compute, bottom=tighten, color="tab:blue", label="PoA solve (compute) time")
    for xi, (t, c) in enumerate(zip(tighten, compute)):
        ax.text(xi, t + c, f"{t + c:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"Tightening vs. PoA compute time per cumulative stage (T={HORIZON})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(time_path, dpi=150)
    plt.close(fig)

    # Plot 2: per-stage tightening time, stacked, to show where tightening goes.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottoms = np.zeros(len(runs))
    for stage in STAGE_ORDER:
        heights = np.array([r["stage_seconds"].get(stage, 0.0) for r in runs])
        if not heights.any():
            continue
        ax.bar(x, heights, bottom=bottoms, color=STAGE_COLORS.get(stage), label=stage)
        bottoms += heights
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("tightening wall time (s)")
    ax.set_title(f"Per-stage tightening time per cumulative case (T={HORIZON})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(stage_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Tightening/compute-time summary for study '{STUDY_NAME}'")
    summary = collect_summary()
    if not summary["runs"]:
        print("No PoA results found. Run the progression first.")
        return

    print_table(summary)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_ROOT / "tightening_progression_summary.json"
    csv_path = RESULT_ROOT / "tightening_progression_summary.csv"
    time_plot_path = RESULT_ROOT / "tightening_vs_compute_time.png"
    stage_plot_path = RESULT_ROOT / "tightening_time_by_stage.png"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_csv(summary, csv_path)
    try:
        plot_summary(summary, time_plot_path, stage_plot_path)
        plotted = f"{time_plot_path}\n  {stage_plot_path}"
    except Exception as exc:  # plotting is optional
        plotted = f"(skipped: {exc})"

    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {plotted}")


if __name__ == "__main__":
    main()
