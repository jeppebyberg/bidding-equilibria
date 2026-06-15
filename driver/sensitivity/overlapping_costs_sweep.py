"""Sensitivity study: overlapping conventional cost ranges.

Varies the cost spread of G1 and G2 so their bid ranges overlap:
  G1 = [10, 40]  (block 1: 25 MW @ 10, block 2: 25 MW @ 40)
  G2 = [20, 30]  (block 1: 25 MW @ 20, block 2: 25 MW @ 30)
  G3 = [50, 60]  unchanged from base case

This places G1's second block (cost=40) above G2's entire range [20, 30],
creating a mixed merit order: G1_B1 < G2_B1 < G2_B2 < G1_B2 < G3_B1 < G3_B2.
In the base case the ranges are non-overlapping (G1=[10,20], G2=[30,40]).

Wind generators (W1, W2, W3) and all capacities and ramp rates are unchanged.

This is a two-run study: the base case is copied from existing results, and the
overlapping case runs the full pipeline from scratch.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.overlapping_costs_sweep
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from driver.sensitivity.sensitivity_config import (
    SENSITIVITY_CONFIG_DIR,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "overlapping_costs_sweep"

CASE_NAME_BASE = "base_test_case"
CASE_NAME_OVERLAP = "overlapping_costs"

CASE_LABELS: dict[str, str] = {
    CASE_NAME_BASE: "Base [G1:10-20, G2:30-40]",
    CASE_NAME_OVERLAP: "Overlap [G1:10-40, G2:20-30]",
}

_BASE_OVERRIDES: dict[str, Any] = {
    "run_scenario_generation": True,
    "run_heuristic_labels": True,
    "run_feature_building": True,
    "run_nn_training": True,
    "run_poa_tightening": True,
    "run_poa_optimization": True,
    "run_dro_tightening": True,
    "run_dro_optimization": True,
}


def _generate_overlapping_case() -> dict[str, Any]:
    """3C+3W case with G1=[10,40] and G2=[20,30] (overlapping cost ranges)."""
    generators: list[dict[str, Any]] = [
        {
            "id": 0,
            "name": "G3",
            "type": "conventional",
            "pmin": 0.0,
            "R_rate_up": 20.0,
            "R_rate_down": 20.0,
            "bidding_blocks": [
                {"block_id": 0, "name": "G3_B1", "pmax": 25.0, "cost": 50.0},
                {"block_id": 1, "name": "G3_B2", "pmax": 25.0, "cost": 60.0},
            ],
        },
        {
            "id": 1,
            "name": "G2",
            "type": "conventional",
            "pmin": 0.0,
            "R_rate_up": 20.0,
            "R_rate_down": 20.0,
            "bidding_blocks": [
                {"block_id": 0, "name": "G2_B1", "pmax": 25.0, "cost": 20.0},
                {"block_id": 1, "name": "G2_B2", "pmax": 25.0, "cost": 30.0},
            ],
        },
        {
            "id": 2,
            "name": "G1",
            "type": "conventional",
            "pmin": 0.0,
            "R_rate_up": 20.0,
            "R_rate_down": 20.0,
            "bidding_blocks": [
                {"block_id": 0, "name": "G1_B1", "pmax": 25.0, "cost": 10.0},
                {"block_id": 1, "name": "G1_B2", "pmax": 25.0, "cost": 40.0},
            ],
        },
        {
            "id": 3,
            "name": "W3",
            "type": "wind",
            "pmin": 0.0,
            "R_rate_up": 50.0,
            "R_rate_down": 50.0,
            "bidding_blocks": [{"block_id": 0, "name": "W3_B1", "pmax": 50.0, "cost": 0.3}],
        },
        {
            "id": 4,
            "name": "W2",
            "type": "wind",
            "pmin": 0.0,
            "R_rate_up": 50.0,
            "R_rate_down": 50.0,
            "bidding_blocks": [{"block_id": 0, "name": "W2_B1", "pmax": 50.0, "cost": 0.2}],
        },
        {
            "id": 5,
            "name": "W1",
            "type": "wind",
            "pmin": 0.0,
            "R_rate_up": 50.0,
            "R_rate_down": 50.0,
            "bidding_blocks": [{"block_id": 0, "name": "W1_B1", "pmax": 50.0, "cost": 0.1}],
        },
    ]
    return {
        "demand": [100.0],
        "generators": generators,
        "players": [{"id": i, "controlled_generators": [i]} for i in range(6)],
    }


def _write_reference_cases() -> Path:
    payload = {CASE_NAME_OVERLAP: _generate_overlapping_case()}
    path = SENSITIVITY_CONFIG_DIR / f"{STUDY_NAME}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    print(f"Wrote reference case: {path}")
    return path


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=_BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name="base",
                overrides={"case": CASE_NAME_BASE},
                label=CASE_LABELS[CASE_NAME_BASE],
            ),
            SensitivityRun(
                name="overlapping",
                overrides={"case": CASE_NAME_OVERLAP},
                label=CASE_LABELS[CASE_NAME_OVERLAP],
            ),
        ],
    )


def run() -> dict[str, Any]:
    _write_reference_cases()
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
