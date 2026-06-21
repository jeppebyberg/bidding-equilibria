# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-01
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Plot the PoA ambiguity regime trajectories for the sensitivity pipeline result.

Delegates to plot_ambiguity_regime_trajectories.py for all plotting logic.
Figures are saved to results_viz/figures/sensitivity_poa_trajectories/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_viz.plot_ambiguity_regime_trajectories import (
    plot_ambiguity_regime_trajectories,
    plot_fresh_scenario_coverage,
    plot_fresh_scenario_innovations,
)

RESULT_PATH = Path(
    "results/sensitivity_pipeline/poa/poa_optimization_T8_piecewise_mccormick.json"
)
OUTPUT_DIR = Path("results_viz/figures/sensitivity_poa_trajectories")

N_FRESH = 300
FRESH_SEED = 42


def _poa_title_line(result_path: Path) -> str | None:
    """Extract PoA metrics from the result JSON and format a title line."""
    try:
        with result_path.open("r", encoding="utf-8") as fh:
            result: dict[str, Any] = json.load(fh)
    except Exception:
        return None
    obj = result.get("objective") or {}
    poa_ratio = obj.get("PoA_ratio")
    poa_relaxed = obj.get("PoA")
    parts = []
    if poa_ratio is not None:
        parts.append(f"ex-post PoA = {float(poa_ratio):.4f}")
    if poa_relaxed is not None:
        parts.append(f"relaxed PoA = {float(poa_relaxed):.4f}")
    return "  |  ".join(parts) if parts else None


def main() -> None:
    if not RESULT_PATH.exists():
        print(f"Result not found: {RESULT_PATH}")
        return

    p1 = plot_ambiguity_regime_trajectories(
        result_path=RESULT_PATH,
        output_dir=OUTPUT_DIR,
        extra_title_line=_poa_title_line(RESULT_PATH),
    )
    print(f"Saved trajectory plot:   {p1}")

    try:
        p2 = plot_fresh_scenario_coverage(
            result_path=RESULT_PATH,
            N=N_FRESH,
            seed=FRESH_SEED,
            output_dir=OUTPUT_DIR,
        )
        print(f"Saved coverage plot:     {p2}")
    except ValueError as exc:
        print(f"Skipping coverage plot: {exc}")

    try:
        p3 = plot_fresh_scenario_innovations(
            result_path=RESULT_PATH,
            N=N_FRESH,
            seed=FRESH_SEED,
            output_dir=OUTPUT_DIR,
        )
        print(f"Saved innovations plot:  {p3}")
    except ValueError as exc:
        print(f"Skipping innovations plot: {exc}")


if __name__ == "__main__":
    main()
