"""Plot the eta-sweep results from the sensitivity pipeline.

Reads from results/sensitivity_pipeline/dro/ and writes figures to
results_viz/figures/sensitivity_eta_sweep/.

All plotting logic lives in plot_dro_poa_eta_sweep.py — this script only
sets the paths and calls the shared functions.
"""
from __future__ import annotations

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

RESULTS_DIR = Path("results/sensitivity_pipeline/dro")
OUTPUT_ROOT = Path("results_viz/figures/sensitivity_eta_sweep")

EPSILON_LABELS: dict[float, str] = {
    0.0: "SAA (no robustness)",
    2000.0: "DRO (Wasserstein cap = 2000)",
}


def main() -> None:
    regimes = discover_regime_names(RESULTS_DIR)
    if not regimes:
        print(f"No eta-sweep results found under {RESULTS_DIR}")
        return

    for regime_name in regimes:
        try:
            records = load_eta_sweep_records(RESULTS_DIR, regime_name)
        except FileNotFoundError as exc:
            print(f"Skipping '{regime_name}': {exc}")
            continue

        output_dir = OUTPUT_ROOT / regime_name
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
