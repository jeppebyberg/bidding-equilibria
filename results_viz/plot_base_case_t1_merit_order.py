# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-21
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Base-case-specific merit order curve at t = 1 (thesis figure).

This is intentionally a one-off, case-specific figure and does NOT change the
general merit-order visualization in visualize_poa_trajectory.py. It zooms into
a fixed window, drops the title / clearing-price / lambda annotations, relabels
the policy bids, and uses euro units.
"""

from __future__ import annotations

from pathlib import Path

import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402

from results_viz.visualize_poa_trajectory import generate_single_merit_order_figure  # noqa: E402

RESULT_PATH = Path("results/base_case/poa/poa_optimization_T6_piecewise_mccormick.json")
OUTPUT_DIR = Path("results_viz/figures/base_case_merit_order")
TIME_STEP = 1

# Zoom window (base case, t = 1).
XLIM = (30.0, 160.0)   # cumulative capacity, MW
YLIM = (-22.5, 50.0)   # bid / cost, EUR/MWh
ANNOTATION_FONTSIZE = 12.0  # bid/cost annotation boxes

# Sized to the A4 text width (~16 cm) with larger fonts so labels stay readable
# when the figure is placed on an A4 page.
FIGSIZE = (6.3, 4.7)
LABEL_FONTSIZE = 14.0
TICK_FONTSIZE = 12.0
LEGEND_FONTSIZE = 12.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"base_case_merit_order_t{TIME_STEP}.png"
    generate_single_merit_order_figure(
        RESULT_PATH,
        out,
        t=TIME_STEP,
        show_title=False,
        show_clearing_price=False,
        show_lambda_lines=False,
        xlim=XLIM,
        ylim=YLIM,
        currency="€",  # euro sign
        eq_legend_label="Pol. bid",
        annotation_fontsize=ANNOTATION_FONTSIZE,
        figsize=FIGSIZE,
        label_fontsize=LABEL_FONTSIZE,
        tick_fontsize=TICK_FONTSIZE,
        legend_fontsize=LEGEND_FONTSIZE,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
