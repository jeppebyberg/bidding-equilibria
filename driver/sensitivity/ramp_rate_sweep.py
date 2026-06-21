# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-14
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Sensitivity study: conventional generator ramp rate in [5, 10, 20, 50] MW/h.

R_rate_up = R_rate_down is swept for all conventional generators (G1, G2, G3).
Wind ramp rates are held fixed at 50 MW/h. Composition (3C+3W), costs, and
per-generator capacities all match base_test_case, so all PoA differences are
attributable to the ramp constraint alone.

The R=20 run is physically identical to base_test_case and reuses existing
results rather than recomputing them.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.ramp_rate_sweep
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from driver.sensitivity.sensitivity_config import (
    BaseCompositionSpec,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_reference_case_sweep_config,
)

STUDY_NAME = "ramp_rate_sweep"

# Costs match base_test_case exactly (G1=10/20, G2=30/40, G3=50/60, W1-W3=0.1-0.3).
_CONV_B1_COSTS = [10.0, 30.0, 50.0]
_CONV_B2_COSTS = [20.0, 40.0, 60.0]
_WIND_COSTS = [0.1, 0.2, 0.3]

# R_rate_up = R_rate_down values under study (MW/h). Base case is R=20.
RAMP_VALUES = [5, 10, 20, 50]


@dataclass
class RampRateSpec(BaseCompositionSpec):
    """3C+3W base-case composition with variable conventional ramp rate."""

    conv_block_cap: float = 25.0
    wind_block_cap: float = 50.0
    conv_ramp: float = 20.0
    wind_ramp: float = 50.0
    # Override parent default to match base_test_case wind cost spacing.
    wind_costs: list = field(default_factory=lambda: list(_WIND_COSTS))

    @property
    def case_name(self) -> str:
        return f"ramp_R{int(self.conv_ramp):03d}"


specs = [
    RampRateSpec(
        n_conv=3,
        n_wind=3,
        conv_b1_costs=list(_CONV_B1_COSTS),
        conv_b2_costs=list(_CONV_B2_COSTS),
        wind_costs=list(_WIND_COSTS),
        conv_ramp=float(r),
    )
    for r in RAMP_VALUES
]

CASE_LABELS: dict[str, str] = {spec.case_name: f"R = {int(spec.conv_ramp)} MW/h" for spec in specs}
CASE_LABELS["base_test_case"] = "R = 20 MW/h (base)"

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


def _case_override(spec: RampRateSpec) -> str:
    # R=20 is physically identical to base_test_case; reuse existing results.
    return "base_test_case" if spec.conv_ramp == 20.0 else spec.case_name


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
                label=f"R = {int(spec.conv_ramp)} MW/h",
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
