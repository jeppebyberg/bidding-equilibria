"""Summarize PoA solve cost and ex-post relaxation gap across the pieces sweep.

Reads each run's ``poa/poa_optimization_T*.json`` (written by block3) and pulls,
per horizon and McCormick piece count:

  - solver.wall_time_seconds          -> compute time
  - solver.variable_counts.*          -> binary-variable counts
  - objective.PoA                     -> relaxed (upper-bound) PoA
  - objective.ex_post_ratio           -> realized C_eq / C_opt at the solution
  - objective.mccormick_gap           -> PoA - ex_post_ratio (relaxation gap)

and derives the ex-post fraction ``ex_post_ratio / PoA`` -- the share of the
reported (relaxed) PoA that is actually realized. As pieces increase the
relaxation tightens, so the gap shrinks and the fraction approaches 1, while the
binary count and solve time grow: the trade-off that justifies the chosen count.
Each plot shows one line per horizon (T=6).

Writes the cross-run comparison as CSV + JSON and three plots.

Run (after mccormick_pieces_sweep):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.mccormick_pieces_summary
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

from driver.sensitivity.mccormick_pieces_sweep import (  # noqa: E402
    HORIZONS,
    RESULT_ROOT,
    STUDY_NAME,
    horizon_name,
    run_dir,
    run_name,
)

# Number of pieces in the production configuration, highlighted in the plots.
CHOSEN_PIECES = 100


def _find_poa_result(run_dir_path: Path) -> Path | None:
    """Return the run's PoA result JSON, or None if it was not produced."""
    poa_dir = run_dir_path / "poa"
    if not poa_dir.exists():
        return None
    matches = sorted(poa_dir.glob("poa_optimization_T*.json"))
    return matches[0] if matches else None


def _safe_fraction(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)):
        return None
    if not isinstance(denominator, (int, float)) or denominator == 0.0:
        return None
    return numerator / denominator


def _load_run(run_dir_path: Path) -> dict[str, Any] | None:
    """Pull compute, binary, and ex-post metrics from one run's PoA result."""
    result_path = _find_poa_result(run_dir_path)
    if result_path is None:
        return None
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    solver = result.get("solver", {}) or {}
    counts = solver.get("variable_counts", {}) or {}
    objective = result.get("objective", {}) or {}

    relaxed_poa = objective.get("PoA")
    ex_post_ratio = objective.get("ex_post_ratio")
    mccormick_gap = objective.get("mccormick_gap")
    if mccormick_gap is None and isinstance(relaxed_poa, (int, float)) and isinstance(
        ex_post_ratio, (int, float)
    ):
        mccormick_gap = relaxed_poa - ex_post_ratio

    return {
        "num_pieces": objective.get("num_pieces"),
        "wall_time_seconds": solver.get("wall_time_seconds"),
        "solver_reported_time_seconds": solver.get("solver_reported_time_seconds"),
        "termination_condition": solver.get("termination_condition"),
        "mip_gap": solver.get("mip_gap"),
        "num_binary_variables": counts.get("num_binary_variables"),
        "num_binary_variables_free": counts.get("num_binary_variables_free"),
        "num_binary_variables_fixed": counts.get("num_binary_variables_fixed"),
        "num_continuous_variables": counts.get("num_continuous_variables"),
        "PoA_relaxed": relaxed_poa,
        "ex_post_ratio": ex_post_ratio,
        "mccormick_gap": mccormick_gap,
        "ex_post_fraction": _safe_fraction(ex_post_ratio, relaxed_poa),
        "result_path": str(result_path),
    }


def _discover_pieces(horizon: int) -> list[int]:
    """Return every piece count with a P<n> dir on disk for this horizon, sorted.

    Discovering from disk (rather than the sweep's PIECES list) means the summary
    stitches together every run present -- e.g. an early 5..100 sweep plus a later
    200/500 sweep -- regardless of what the most recent sweep run targeted.
    """
    horizon_dir = RESULT_ROOT / horizon_name(horizon)
    if not horizon_dir.exists():
        return []
    pieces: list[int] = []
    for child in horizon_dir.glob("P*"):
        if child.is_dir() and child.name[1:].isdigit():
            pieces.append(int(child.name[1:]))
    return sorted(pieces)


def collect_summary() -> dict[str, Any]:
    """Build the cross-pieces summary from every run on disk, in piece order."""
    horizons = [h for h in HORIZONS if _discover_pieces(h)]
    runs: list[dict[str, Any]] = []
    for horizon in horizons:
        for num_pieces in _discover_pieces(horizon):
            name = run_name(num_pieces)
            run_dir_path = run_dir(horizon, num_pieces)
            loaded = _load_run(run_dir_path)
            if loaded is None:
                print(
                    f"  [skip] T{horizon}/{name}: no PoA result found under "
                    f"{run_dir_path / 'poa'}"
                )
                continue
            runs.append(
                {"run_name": name, "horizon": horizon, "pieces": num_pieces, **loaded}
            )
    return {
        "study": STUDY_NAME,
        "chosen_pieces": CHOSEN_PIECES,
        "horizons": horizons,
        "runs": runs,
    }


def _runs_for_horizon(summary: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    return [r for r in summary["runs"] if r["horizon"] == horizon]


_FIELDNAMES = [
    "run_name",
    "horizon",
    "pieces",
    "wall_time_seconds",
    "solver_reported_time_seconds",
    "num_binary_variables",
    "num_binary_variables_free",
    "num_binary_variables_fixed",
    "num_continuous_variables",
    "PoA_relaxed",
    "ex_post_ratio",
    "mccormick_gap",
    "ex_post_fraction",
    "termination_condition",
    "mip_gap",
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
        f"{'pieces':>8}{'wall (s)':>12}{'binaries':>12}{'free bin':>12}"
        f"{'PoA (relax)':>14}{'ex-post':>10}{'gap':>10}{'ex-post %':>11}"
    )
    for horizon in summary["horizons"]:
        runs = _runs_for_horizon(summary, horizon)
        if not runs:
            continue
        print(f"\nT={horizon}")
        print(header)
        print("-" * len(header))
        for run in runs:
            frac = run["ex_post_fraction"]
            frac_pct = frac * 100.0 if isinstance(frac, (int, float)) else None
            print(
                f"{run['pieces']:>8}"
                f"{_fmt(run['wall_time_seconds'], '.2f'):>12}"
                f"{_fmt(run['num_binary_variables'], 'd'):>12}"
                f"{_fmt(run['num_binary_variables_free'], 'd'):>12}"
                f"{_fmt(run['PoA_relaxed'], '.4f'):>14}"
                f"{_fmt(run['ex_post_ratio'], '.4f'):>10}"
                f"{_fmt(run['mccormick_gap'], '.4f'):>10}"
                f"{_fmt(frac_pct, '.2f'):>11}"
            )
    print()


_HORIZON_COLORS = {6: "tab:blue", 8: "tab:orange"}


def _apply_log_x(ax: Any, all_pieces: list[int]) -> None:
    """Log x-axis with integer piece-count tick labels (5..500 spans too wide for linear)."""
    from matplotlib.ticker import FixedFormatter, FixedLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(all_pieces))
    ax.xaxis.set_major_formatter(FixedFormatter([str(p) for p in all_pieces]))
    ax.xaxis.set_minor_locator(FixedLocator([]))


def _is_capped(run: dict[str, Any]) -> bool:
    """True if the run hit the solve time limit (not a proven optimum)."""
    return str(run.get("termination_condition")) not in {"optimal", "None", "none"}


def plot_summary(
    summary: dict[str, Any],
    time_path: Path,
    binary_path: Path,
    gap_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [h for h in summary["horizons"] if _runs_for_horizon(summary, h)]
    if not horizons:
        return
    chosen = summary.get("chosen_pieces")
    all_pieces = sorted(
        {r["pieces"] for h in horizons for r in _runs_for_horizon(summary, h)}
    )

    def _mark_chosen(ax: Any) -> None:
        if chosen in all_pieces:
            ax.axvline(chosen, color="tab:red", ls="--", alpha=0.6,
                       label=f"chosen = {chosen}")

    fig, ax = plt.subplots(figsize=(7, 5))
    capped_labelled = False
    for horizon in horizons:
        runs = _runs_for_horizon(summary, horizon)
        ax.plot(
            [r["pieces"] for r in runs],
            [r["wall_time_seconds"] for r in runs],
            marker="o",
            color=_HORIZON_COLORS.get(horizon),
            label=f"T={horizon}",
        )
        # Overlay hollow red markers on runs that hit the time limit (incumbent,
        # not a proven optimum) so the capped points are not read as solved.
        capped = [r for r in runs if _is_capped(r)]
        if capped:
            ax.scatter(
                [r["pieces"] for r in capped],
                [r["wall_time_seconds"] for r in capped],
                facecolors="none", edgecolors="tab:red", s=120, linewidths=1.8,
                zorder=5, label=None if capped_labelled else "hit time limit",
            )
            capped_labelled = True
    ax.set_xlabel("McCormick pieces")
    ax.set_ylabel("PoA solve wall time (s)")
    ax.set_title("PoA solve time vs. McCormick pieces")
    _apply_log_x(ax, all_pieces)
    ax.grid(alpha=0.3)
    _mark_chosen(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(time_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for horizon in horizons:
        runs = _runs_for_horizon(summary, horizon)
        pieces = [r["pieces"] for r in runs]
        color = _HORIZON_COLORS.get(horizon)
        ax.plot(
            pieces,
            [r["num_binary_variables"] for r in runs],
            marker="o",
            color=color,
            label=f"T={horizon} total binaries",
        )
        ax.plot(
            pieces,
            [r["num_binary_variables_free"] for r in runs],
            marker="s",
            ls=":",
            color=color,
            label=f"T={horizon} free binaries",
        )
    ax.set_xlabel("McCormick pieces")
    ax.set_ylabel("binary variables")
    ax.set_title("PoA binary count vs. McCormick pieces")
    _apply_log_x(ax, all_pieces)
    ax.grid(alpha=0.3)
    _mark_chosen(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(binary_path, dpi=150)
    plt.close(fig)

    # Ex-post relaxation gap and the realized fraction of the relaxed PoA.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    for horizon in horizons:
        runs = _runs_for_horizon(summary, horizon)
        pieces = [r["pieces"] for r in runs]
        color = _HORIZON_COLORS.get(horizon)
        ax.plot(
            pieces,
            [r["mccormick_gap"] for r in runs],
            marker="o",
            color=color,
            label=f"T={horizon} gap (PoA - ex-post)",
        )
        frac_pct = [
            (r["ex_post_fraction"] * 100.0)
            if isinstance(r["ex_post_fraction"], (int, float))
            else None
            for r in runs
        ]
        ax2.plot(
            pieces,
            frac_pct,
            marker="s",
            ls=":",
            color=color,
            label=f"T={horizon} ex-post fraction (%)",
        )
    ax.set_xlabel("McCormick pieces")
    ax.set_ylabel("relaxation gap")
    ax2.set_ylabel("ex-post fraction (%)")
    ax.set_title("Ex-post relaxation gap vs. McCormick pieces")
    _apply_log_x(ax, all_pieces)
    ax.grid(alpha=0.3)
    _mark_chosen(ax)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(gap_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"McCormick-pieces compute/ex-post summary for study '{STUDY_NAME}'")
    summary = collect_summary()
    if not summary["runs"]:
        print("No PoA results found. Run the sweep first.")
        return

    print_table(summary)

    json_path = RESULT_ROOT / "pieces_summary.json"
    csv_path = RESULT_ROOT / "pieces_summary.csv"
    time_plot_path = RESULT_ROOT / "compute_time_vs_pieces.png"
    binary_plot_path = RESULT_ROOT / "binaries_vs_pieces.png"
    gap_plot_path = RESULT_ROOT / "ex_post_gap_vs_pieces.png"

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_csv(summary, csv_path)
    try:
        plot_summary(summary, time_plot_path, binary_plot_path, gap_plot_path)
        plotted = f"{time_plot_path}\n  {binary_plot_path}\n  {gap_plot_path}"
    except Exception as exc:  # plotting is optional
        plotted = f"(skipped: {exc})"

    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {plotted}")


if __name__ == "__main__":
    main()
