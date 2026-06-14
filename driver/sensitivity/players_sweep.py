"""Sensitivity study: market player count N = [2, 3, 4, 5].

N conventional generators and N wind generators, where N is the study parameter.
Total installed capacity is fixed at 150 MW conventional and 150 MW wind, so
individual capacities shrink as N grows:
  conv_block_cap = 75 / N  MW  (two blocks per conventional generator)
  wind_block_cap = 150 / N MW  (one block per wind generator)

Ramp rates scale with per-player capacity to hold the base-case ramp/capacity
ratios fixed: conv 40% (= 20 MW/h at N=3) and wind 100% (= 50 MW/h at N=3).

Generator costs are linearly interpolated so that the aggregate merit order
spans the same range regardless of N:
  conv_b1_costs = linspace(10, 50, N)   -> [10, 30, 50] at N=3 (base case)
  conv_b2_costs = linspace(20, 60, N)   -> [20, 40, 60] at N=3 (base case)
  wind_costs    = linspace(0.1, 0.3, N) -> [0.1, 0.2, 0.3] at N=3 (base case)

The N=3 run resolves to base_test_case and reuses existing results.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.players_sweep
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from driver.sensitivity.sensitivity_config import (
    BaseCompositionSpec,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_reference_case_sweep_config,
)

STUDY_NAME = "players_sweep"

TOTAL_CONV_CAP_MW = 150.0
TOTAL_WIND_CAP_MW = 150.0
DEMAND_MW = 100.0

_BASE_N = 3
_BASE_CONV_RAMP = 20.0  # MW/h at N=3
_BASE_WIND_RAMP = 50.0  # MW/h at N=3

_CONV_B1_MIN, _CONV_B1_MAX = 10.0, 50.0
_CONV_B2_MIN, _CONV_B2_MAX = 20.0, 60.0
_WIND_MIN, _WIND_MAX = 0.1, 0.3

N_VALUES = [2, 3, 4, 5]


@dataclass
class PlayersSpec(BaseCompositionSpec):
    """N-symmetric composition with fixed total capacity and scaled ramp rates."""

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
        return f"players_N{self.n_conv}"


def _make_spec(n: int) -> PlayersSpec:
    return PlayersSpec(
        n_conv=n,
        n_wind=n,
        demand=DEMAND_MW,
        conv_b1_costs=np.linspace(_CONV_B1_MIN, _CONV_B1_MAX, n).tolist(),
        conv_b2_costs=np.linspace(_CONV_B2_MIN, _CONV_B2_MAX, n).tolist(),
        wind_costs=np.linspace(_WIND_MIN, _WIND_MAX, n).tolist(),
    )


specs = [_make_spec(n) for n in N_VALUES]

CASE_LABELS: dict[str, str] = {spec.case_name: f"N = {spec.n_conv}" for spec in specs}
CASE_LABELS["base_test_case"] = "N = 3 (base)"

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


def _case_override(spec: PlayersSpec) -> str:
    # N=3 is physically identical to base_test_case; reuse existing results.
    return "base_test_case" if spec.n_conv == _BASE_N else spec.case_name


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=_BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={"case": _case_override(spec)},
                label=f"N = {spec.n_conv}",
            )
            for spec in specs
        ],
    )


def run() -> dict[str, Any]:
    sweep_specs = [s for s in specs if _case_override(s) != "base_test_case"]
    write_reference_case_sweep_config(STUDY_NAME, sweep_specs)
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
