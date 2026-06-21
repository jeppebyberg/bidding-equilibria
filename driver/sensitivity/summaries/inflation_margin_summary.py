# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-15
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Summarize PoA across the inflation-margin sweep and plot PoA vs. margin.

Reads each run's ``poa/poa_optimization_T*.json`` (written by block3) and pulls,
per inflation-margin value:

  - objective.PoA            -> relaxed (upper-bound) PoA
  - objective.ex_post_ratio  -> realized C_eq / C_opt at the solution
  - objective.mccormick_gap  -> PoA - ex_post_ratio (relaxation gap)
  - solver.termination_condition / mip_gap

The plot shows both the relaxed PoA (the reported worst-case bound) and the
ex-post realized ratio as a function of inflation_margin, with the base margin
(0.25) marked. The margin axis is log-scaled because the values span 0.01..5.

Run (after inflation_margin_sweep):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.summaries.inflation_margin_summary
"""

from __future__ import annotations

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.inflation_margin_sweep import (  # noqa: E402
    BASE_MARGIN,
    MARGIN_VALUES,
    STUDY_NAME,
    run_name,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "sensitivity_studies" / STUDY_NAME


def _find_poa_result(run_dir_path: Path) -> Path | None:
    """Return the run's PoA result JSON, or None if it was not produced."""
    poa_dir = run_dir_path / "poa"
    if not poa_dir.exists():
        return None
    matches = sorted(poa_dir.glob("poa_optimization_T*.json"))
    return matches[0] if matches else None


def _load_run(run_dir_path: Path) -> dict[str, Any] | None:
    """Pull PoA metrics from one run's PoA result."""
    result_path = _find_poa_result(run_dir_path)
    if result_path is None:
        return None
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    solver = result.get("solver", {}) or {}
    objective = result.get("objective", {}) or {}

    relaxed_poa = objective.get("PoA")
    ex_post_ratio = objective.get("ex_post_ratio")
    mccormick_gap = objective.get("mccormick_gap")
    if mccormick_gap is None and isinstance(relaxed_poa, (int, float)) and isinstance(
        ex_post_ratio, (int, float)
    ):
        mccormick_gap = relaxed_poa - ex_post_ratio

    return {
        "PoA_relaxed": relaxed_poa,
        "ex_post_ratio": ex_post_ratio,
        "mccormick_gap": mccormick_gap,
        "num_pieces": objective.get("num_pieces"),
        "termination_condition": solver.get("termination_condition"),
        "mip_gap": solver.get("mip_gap"),
        "wall_time_seconds": solver.get("wall_time_seconds"),
        "result_path": str(result_path),
    }


def collect_summary() -> dict[str, Any]:
    """Build the cross-margin summary from every run on disk, in margin order."""
    runs: list[dict[str, Any]] = []
    for margin in MARGIN_VALUES:
        name = run_name(margin)
        run_dir_path = RESULT_ROOT / name
        loaded = _load_run(run_dir_path)
        if loaded is None:
            print(f"  [skip] {name}: no PoA result found under {run_dir_path / 'poa'}")
            continue
        runs.append({"run_name": name, "inflation_margin": float(margin), **loaded})
    return {"study": STUDY_NAME, "base_margin": BASE_MARGIN, "runs": runs}


_FIELDNAMES = [
    "run_name",
    "inflation_margin",
    "PoA_relaxed",
    "ex_post_ratio",
    "mccormick_gap",
    "num_pieces",
    "termination_condition",
    "mip_gap",
    "wall_time_seconds",
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
        f"{'margin':>10}{'PoA (relax)':>14}{'ex-post':>11}{'gap':>10}{'term':>11}"
    )
    print(header)
    print("-" * len(header))
    for run in summary["runs"]:
        base_tag = "  (base)" if run["inflation_margin"] == BASE_MARGIN else ""
        print(
            f"{_fmt(run['inflation_margin'], 'g'):>10}"
            f"{_fmt(run['PoA_relaxed'], '.4f'):>14}"
            f"{_fmt(run['ex_post_ratio'], '.4f'):>11}"
            f"{_fmt(run['mccormick_gap'], '.4f'):>10}"
            f"{str(run['termination_condition']):>11}"
            f"{base_tag}"
        )
    print()


def plot_summary(summary: dict[str, Any], poa_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    runs = summary["runs"]
    if not runs:
        return
    margins = [r["inflation_margin"] for r in runs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        margins,
        [r["PoA_relaxed"] for r in runs],
        marker="o",
        color="tab:blue",
        label="PoA (relaxed upper bound)",
    )
    ax.plot(
        margins,
        [r["ex_post_ratio"] for r in runs],
        marker="s",
        ls=":",
        color="tab:green",
        label="ex-post realized ratio",
    )
    base = summary.get("base_margin")
    if base in margins:
        ax.axvline(base, color="tab:red", ls="--", alpha=0.6, label=f"base = {base:g}")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(margins))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{m:g}" for m in margins]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("inflation_margin")
    ax.set_ylabel("Price of Anarchy")
    ax.set_title("PoA vs. heuristic inflation margin")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(poa_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Inflation-margin PoA summary for study '{STUDY_NAME}'")
    summary = collect_summary()
    if not summary["runs"]:
        print("No PoA results found. Run the sweep first.")
        return

    print_table(summary)

    json_path = RESULT_ROOT / "inflation_margin_summary.json"
    csv_path = RESULT_ROOT / "inflation_margin_summary.csv"
    poa_plot_path = RESULT_ROOT / "poa_vs_inflation_margin.png"

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_csv(summary, csv_path)
    try:
        plot_summary(summary, poa_plot_path)
        plotted = str(poa_plot_path)
    except Exception as exc:  # plotting is optional
        plotted = f"(skipped: {exc})"

    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {plotted}")


if __name__ == "__main__":
    main()
