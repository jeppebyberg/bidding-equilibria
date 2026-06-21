# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-15
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Sensitivity study: number of bidding blocks per conventional generator [1, 2, 3, 4].

Each conventional generator (G1, G2, G3) has the same total capacity (50 MW)
split equally across B blocks, with costs linearly spaced within the generator's
cost range:
  block_cap = 50 / B  MW
  block_costs = linspace(gen_min_cost, gen_max_cost, B)

At B=1 each generator submits a single block at its lowest cost:
  G1: 50 MW at cost 10,  G2: 50 MW at cost 30,  G3: 50 MW at cost 50
At B=2 (base case):
  G1: [25 MW @ 10, 25 MW @ 20],  G2: [25 MW @ 30, 25 MW @ 40],  G3: [25 MW @ 50, 25 MW @ 60]

Wind generators always have one block (50 MW) with their base-case costs and
ramp rates unchanged.

The B=2 run is physically identical to base_test_case and reuses existing results.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.bidding_blocks_sweep
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from driver.sensitivity.sensitivity_config import (
    SENSITIVITY_CONFIG_DIR,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "bidding_blocks_sweep"

NUM_BLOCKS_VALUES = [1, 2, 3, 4]
# Subset to actually run this session; edit to skip already-completed runs.
NUM_BLOCKS_TO_RUN = list(NUM_BLOCKS_VALUES)

# 3C+3W composition matching base_test_case (highest-cost generator listed first).
_CONV_GENS = [
    ("G3", 50.0, 60.0),  # (name, min_cost, max_cost)
    ("G2", 30.0, 40.0),
    ("G1", 10.0, 20.0),
]
_WIND_GENS = [
    ("W3", 50.0, 0.3),  # (name, cap_mw, cost)
    ("W2", 50.0, 0.2),
    ("W1", 50.0, 0.1),
]
_CONV_CAP_MW = 50.0  # total per conventional generator
_DEMAND_MW = 100.0


def _case_name(n_blocks: int) -> str:
    return f"blocks_B{n_blocks}"


def _generate_case(n_blocks: int) -> dict[str, Any]:
    """Build one reference case with n_blocks bidding blocks per conv generator."""
    block_cap = _CONV_CAP_MW / n_blocks
    generators: list[dict[str, Any]] = []
    gen_id = 0

    for name, min_cost, max_cost in _CONV_GENS:
        costs = np.linspace(min_cost, max_cost, n_blocks).tolist()
        generators.append(
            {
                "id": gen_id,
                "name": name,
                "type": "conventional",
                "pmin": 0.0,
                "R_rate_up": 20.0,
                "R_rate_down": 20.0,
                "bidding_blocks": [
                    {
                        "block_id": j,
                        "name": f"{name}_B{j + 1}",
                        "pmax": float(block_cap),
                        "cost": float(costs[j]),
                    }
                    for j in range(n_blocks)
                ],
            }
        )
        gen_id += 1

    for name, cap, cost in _WIND_GENS:
        generators.append(
            {
                "id": gen_id,
                "name": name,
                "type": "wind",
                "pmin": 0.0,
                "R_rate_up": 50.0,
                "R_rate_down": 50.0,
                "bidding_blocks": [
                    {"block_id": 0, "name": f"{name}_B1", "pmax": float(cap), "cost": float(cost)}
                ],
            }
        )
        gen_id += 1

    return {
        "demand": [float(_DEMAND_MW)],
        "generators": generators,
        "players": [{"id": i, "controlled_generators": [i]} for i in range(gen_id)],
    }


def _write_reference_cases(blocks_to_write: list[int]) -> Path:
    """Write per-block-count reference cases to config/sensitivity_studies/."""
    payload = {_case_name(b): _generate_case(b) for b in blocks_to_write}
    path = SENSITIVITY_CONFIG_DIR / f"{STUDY_NAME}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    print(f"Wrote {len(payload)} reference case(s): {path}")
    return path


CASE_LABELS: dict[str, str] = {_case_name(b): f"B = {b} blocks" for b in NUM_BLOCKS_VALUES}
CASE_LABELS["base_test_case"] = "B = 2 (base)"

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


def _case_override(n_blocks: int) -> str:
    # B=2 is physically identical to base_test_case; reuse existing results.
    return "base_test_case" if n_blocks == 2 else _case_name(n_blocks)


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=_BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=_case_name(b),
                overrides={"case": _case_override(b)},
                label=f"B = {b}",
            )
            for b in NUM_BLOCKS_VALUES
        if b in NUM_BLOCKS_TO_RUN
        ],
    )


def run() -> dict[str, Any]:
    blocks_to_write = [b for b in NUM_BLOCKS_TO_RUN if _case_override(b) != "base_test_case"]
    _write_reference_cases(blocks_to_write)
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
