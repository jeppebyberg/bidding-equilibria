"""Composition sensitivity sweep: varying conventional/wind player count.

Sweeps five market compositions from wind-heavy (1C+5W) to conventional-heavy
(5C+1W).  Physical unit parameters (block capacities, ramp rates, demand) mirror
base_test_case so PoA differences are attributable to composition alone.  All
generators -- conventional and wind -- carry learned NN bidding policies
(allow_wind_to_play=True), so strategic behaviour is symmetric across types.

The 3C+3W run resolves directly to ``base_test_case`` so that existing base-case
results are reused rather than recomputed.  Wind costs for all other compositions
follow the base_test_case spacing (0.1 per generator: W1=0.1, W2=0.2, ...).

Note: the (1C+5W) and (2C+4W) compositions have the tightest firm-capacity
margin -- 1C * 50 MW = 50 MW and 2C * 50 MW = 100 MW respectively -- the latter
exactly covering demand=100 in the worst-case (wind=0) scenario.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.composition_sweep
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


STUDY_NAME = "composition_sweep"


@dataclass
class CompositionSpec(BaseCompositionSpec):
    """Compositions matching base_test_case unit parameters.

    Wind costs follow the base_test_case spacing (0.1 per generator): W1=0.1,
    W2=0.2, ..., W5=0.5.  The 3C+3W run uses case="base_test_case" directly
    so its wind costs are identical to the base case by construction.
    """

    conv_block_cap: float = 25.0
    wind_block_cap: float = 50.0
    conv_ramp: float = 20.0
    wind_ramp: float = 50.0
    demand: float = 100.0
    wind_costs: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    label: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.label:
            self.label = f"{self.n_conv}C + {self.n_wind}W"

    @property
    def case_name(self) -> str:
        return f"comp_{self.n_conv}C_{self.n_wind}W"


compositions = [
    CompositionSpec(n_conv=1, n_wind=5, label="Wind-heavy (1C + 5W)"),
    CompositionSpec(n_conv=2, n_wind=4, label="Wind-heavy (2C + 4W)"),
    CompositionSpec(n_conv=3, n_wind=3, label="Balanced (3C + 3W)"),
    CompositionSpec(n_conv=4, n_wind=2, label="Conv-heavy (4C + 2W)"),
    CompositionSpec(n_conv=5, n_wind=1, label="Conv-heavy (5C + 1W)"),
]

CASE_LABELS: dict[str, str] = {spec.case_name: spec.label for spec in compositions}


BASE_OVERRIDES: dict[str, Any] = {
    "run_scenario_generation": True,
    "run_heuristic_labels": True,
    "run_feature_building": True,
    "run_nn_training": True,
    "run_poa_tightening": True,
    "run_poa_optimization": True,
    "run_dro_tightening": True,
    "run_dro_optimization": True,
}


def _case_override(spec: CompositionSpec) -> str:
    # 3C+3W is physically identical to base_test_case; using its name directly
    # lets run_matches_base_case reuse existing results.
    if spec.n_conv == 3 and spec.n_wind == 3:
        return "base_test_case"
    return spec.case_name


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={"case": _case_override(spec)},
                label=spec.label,
            )
            for spec in compositions
        ],
    )


def run() -> dict[str, Any]:
    # Write reference cases for compositions that use their own case YAML entry;
    # 3C+3W resolves to base_test_case (already in reference_cases.yaml).
    sweep_specs = [s for s in compositions if _case_override(s) != "base_test_case"]
    write_reference_case_sweep_config(STUDY_NAME, sweep_specs)
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
