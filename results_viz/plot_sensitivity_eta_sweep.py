"""Plot the eta-sweep results from the sensitivity pipeline.

Reads from results/sensitivity_studies/{sensitivity_case}/dro/ and writes
figures to results_viz/figures/sensitivity_eta_sweep/{sensitivity_case}/.

All plotting logic lives in plot_dro_poa_eta_sweep.py — this script only
sets the paths and calls the shared functions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_viz.plot_dro_poa_eta_sweep import (
    clean_output_dir,
    discover_regime_names,
    load_eta_sweep_records,
    plot_poa_epsilon_frontier,
    plot_poa_eta_sweep,
    write_summary_csv,
)

DEFAULT_SENSITIVITY_CASE = "base_test_case"
SENSITIVITY_RESULTS_ROOT = Path("results/sensitivity_studies")
OUTPUT_ROOT = Path("results_viz/figures/sensitivity_eta_sweep")

EPSILON_LABELS: dict[float, str] = {
    0.0: "SAA (no robustness)",
    2000.0: "DRO (Wasserstein cap = 2000)",
}


def resolve_results_dir(sensitivity_case: str) -> Path:
    """Return the DRO eta-sweep folder for a named sensitivity case."""
    case_dir = SENSITIVITY_RESULTS_ROOT / sensitivity_case
    preferred = case_dir / "dro"
    if discover_regime_names(preferred):
        return preferred

    nested_legacy = case_dir / sensitivity_case / "dro"
    if discover_regime_names(nested_legacy):
        return nested_legacy

    if preferred.exists():
        return preferred

    return preferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sensitivity eta-sweep results for one sensitivity case."
    )
    parser.add_argument(
        "sensitivity_case",
        nargs="?",
        default=DEFAULT_SENSITIVITY_CASE,
        help=(
            f"Sensitivity case folder under {SENSITIVITY_RESULTS_ROOT} "
            f"(default: {DEFAULT_SENSITIVITY_CASE})."
        ),
    )
    parser.add_argument(
        "--dro-dir",
        default=None,
        help=(
            "Direct path to a DRO results directory (e.g. "
            "results/sensitivity_pipeline/dro). "
            "When given, skips sensitivity-case resolution and reads from "
            "this directory instead."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dro_dir is not None:
        results_dir = Path(args.dro_dir)
        sensitivity_case = results_dir.parent.name
    else:
        sensitivity_case = str(args.sensitivity_case)
        results_dir = resolve_results_dir(sensitivity_case)

    regimes = discover_regime_names(results_dir)
    if not regimes:
        print(f"No eta-sweep results found under {results_dir}")
        return

    for regime_name in regimes:
        try:
            records = load_eta_sweep_records(results_dir, regime_name)
        except FileNotFoundError as exc:
            print(f"Skipping '{regime_name}': {exc}")
            continue

        output_dir = OUTPUT_ROOT / sensitivity_case / regime_name
        clean_output_dir(output_dir)

        csv_path = write_summary_csv(records, output_dir / "eta_sweep_summary.csv")
        figure_path = plot_poa_eta_sweep(
            records=records,
            output_dir=output_dir,
            regime_name=regime_name,
            epsilon_labels=EPSILON_LABELS,
        )
        frontier_path = plot_poa_epsilon_frontier(
            records=records,
            output_dir=output_dir,
            regime_name=regime_name,
            poa_metric="inner_objective",
            epsilon_labels=EPSILON_LABELS,
        )
        print(f"Saved eta-sweep figure:    {figure_path}")
        print(f"Saved eps-frontier figure: {frontier_path}")
        print(f"Saved summary CSV:         {csv_path}")


if __name__ == "__main__":
    main()
