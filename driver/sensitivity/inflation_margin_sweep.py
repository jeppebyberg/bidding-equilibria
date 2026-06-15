"""Sensitivity study: heuristic inflation margin in [0.01, 0.1, 0.25, 0.5, 1, 5].

``inflation_margin`` is the economic gap by which a marginal block is bid below
its nearest higher opponent when the merit-order heuristic writes a best-response
label (see MeritOrderHeuristic). It is decoupled from ``bid_tolerance`` (a purely
numerical comparison guard); only the strategic undercut is swept here, so every
PoA difference is attributable to how aggressively the heuristic labels inflate.

Because the margin changes the synthetic labels, each run must regenerate labels,
features, and trained policies before re-solving the PoA -- nothing upstream can
be reused across margins. The only field that changes across runs is
``inflation_margin``; composition, costs, capacities, ramps, scenarios (seed) and
all tightening settings match base_test_case.

The margin = 0.25 run is identical to base_test_case and reuses the base-case
results (results/base_case) rather than recomputing them.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.inflation_margin_sweep
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "inflation_margin_sweep"

# Inflation-margin values under study. Base case is 0.25.
MARGIN_VALUES = [0.01, 0.1, 0.25, 0.5, 1.0, 5.0]
BASE_MARGIN = 0.25

# Run the full pipeline per case: the margin changes the labels, so policies and
# the PoA must be recomputed. DRO half stays off (non-DRO is the default pipeline).
_BASE_OVERRIDES: dict[str, Any] = {
    "run_scenario_generation": True,
    "run_heuristic_labels": True,
    "run_feature_building": True,
    "run_nn_training": True,
    "run_poa_tightening": True,
    "run_poa_optimization": True,
    "run_dro_tightening": False,
    "run_dro_optimization": False,
}


def run_name(margin: float) -> str:
    # Filesystem-clean, sortable: 0.01 -> m0p01, 0.5 -> m0p5, 5.0 -> m5.
    return "m" + format(margin, "g").replace(".", "p")


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=_BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=run_name(margin),
                overrides={"inflation_margin": float(margin)},
                label=(
                    f"inflation_margin = {margin:g}"
                    + (" (base)" if margin == BASE_MARGIN else "")
                ),
            )
            for margin in MARGIN_VALUES
        ],
    )


def run() -> dict[str, Any]:
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
