"""Sensitivity study: number of market players (symmetric wind + conventional).

Total demand and total installed capacity are held fixed.  As the number of
players N increases, each player's block capacities shrink proportionally:
  conv_block_cap = 75 / N   (each conv player: 2 blocks x conv_block_cap MW)
  wind_block_cap = 150 / N  (each wind player: 1 block x wind_block_cap MW)

Ramp rates scale with per-player capacity to keep the ramp/capacity ratio constant
at the base-case values (conv: 40 %, wind: 100 %).

N ranges from 2 to 8 (16 participants at maximum), with costs linearly
interpolated within the base-case range so the total merit order is fixed.

Run:
  python -m driver.sensitivity.num_players_sweep

DRO-only (assumes trained policies and PoA regimes exist):
  python -m driver.sensitivity.num_players_sweep --dro-only
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.sensitivity_config import (  # noqa: E402
    BaseCompositionSpec,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_reference_case_sweep_config,
)

# Cost range anchored to the N=3 base case so the total merit order is
# the same for all player counts (only the number of slices changes).
_CONV_B1_MIN, _CONV_B1_MAX = 10.0, 50.0
_CONV_B2_MIN, _CONV_B2_MAX = 20.0, 60.0
_WIND_MIN, _WIND_MAX = 0.01, 0.50

STUDY_NAME = "num_players_sweep"

TOTAL_CONV_CAP_MW = 150.0
TOTAL_WIND_CAP_MW = 150.0
DEMAND_MW = 100.0

_BASE_N = 3
_BASE_CONV_RAMP = 20.0
_BASE_WIND_RAMP = 50.0

N_VALUES = [2, 3, 4, 5, 6, 7, 8]


@dataclass
class NumPlayersSpec(BaseCompositionSpec):
    """Composition with N_conv = N_wind = N and fixed total capacities."""

    @property
    def conv_block_cap(self) -> float:
        return TOTAL_CONV_CAP_MW / (2 * self.n_conv)

    @property
    def wind_block_cap(self) -> float:
        return TOTAL_WIND_CAP_MW / self.n_wind

    @property
    def conv_ramp(self) -> float:
        return _BASE_CONV_RAMP * _BASE_N / self.n_conv

    @property
    def wind_ramp(self) -> float:
        return _BASE_WIND_RAMP * _BASE_N / self.n_wind

    @property
    def case_name(self) -> str:
        return f"num_players_{self.n_conv}"


def _make_spec(n: int) -> NumPlayersSpec:
    return NumPlayersSpec(
        n_wind=n,
        n_conv=n,
        demand=DEMAND_MW,
        conv_b1_costs=np.linspace(_CONV_B1_MIN, _CONV_B1_MAX, n).tolist(),
        conv_b2_costs=np.linspace(_CONV_B2_MIN, _CONV_B2_MAX, n).tolist(),
        wind_costs=np.linspace(_WIND_MIN, _WIND_MAX, n).tolist(),
    )


def build_study() -> SensitivityStudy:
    specs = [_make_spec(n) for n in N_VALUES]
    write_reference_case_sweep_config(STUDY_NAME, specs)
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={"case": spec.case_name},
                label=f"N={spec.n_conv}",
            )
            for spec in specs
        ],
    )


def run() -> dict[str, Any]:
    return run_sensitivity_study(build_study())


def run_dro_only(
    n_values: list[int] | None = None,
    run_tightening: bool = True,
) -> dict[str, Any]:
    """Run only the DRO analysis (block4) for the given player counts.

    Assumes trained policies and PoA runtime regimes already exist in the
    sensitivity run directories.
    """
    targets = n_values if n_values is not None else N_VALUES
    specs = [_make_spec(n) for n in targets]
    write_reference_case_sweep_config(STUDY_NAME, specs)
    study = SensitivityStudy(
        name=STUDY_NAME,
        blocks=("block4",),
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={
                    "case": spec.case_name,
                    "run_dro_tightening": run_tightening,
                    "run_dro_optimization": True,
                    "archive_existing_dro_results": True,
                },
                label=f"N={spec.n_conv}",
            )
            for spec in specs
        ],
        reuse_base_case_results=False,
    )
    return run_sensitivity_study(study)


if __name__ == "__main__":
    if "--dro-only" in sys.argv:
        run_dro_only()
    else:
        run()
