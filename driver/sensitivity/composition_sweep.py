"""Composition sensitivity sweep: 6 generators, varying conventional/wind split.

Sweeps three market compositions while holding total player count at 6 (matching
base_test_case): wind-heavy (2C+4W), balanced (3C+3W), and conventional-heavy
(4C+2W).  Physical unit parameters (block capacities, ramp rates, demand) mirror
base_test_case so PoA differences are attributable to composition alone.

Note: the (2C+4W) composition has the tightest firm-capacity margin -- 2C * 50 MW
= 100 MW exactly covers demand=100 in the worst-case (wind=0) scenario.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.composition_sweep
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """6-generator compositions matching base_test_case unit parameters."""

    conv_block_cap: float = 25.0
    wind_block_cap: float = 50.0
    conv_ramp: float = 20.0
    wind_ramp: float = 50.0
    demand: float = 100.0
    label: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.label:
            self.label = f"{self.n_conv}C + {self.n_wind}W"

    @property
    def case_name(self) -> str:
        return f"comp_{self.n_conv}C_{self.n_wind}W"


compositions = [
    CompositionSpec(n_conv=2, n_wind=4, label="Wind-heavy (2C + 4W)"),
    CompositionSpec(n_conv=3, n_wind=3, label="Balanced (3C + 3W)"),
    CompositionSpec(n_conv=4, n_wind=2, label="Conv-heavy (4C + 2W)"),
]

CASE_LABELS: dict[str, str] = {spec.case_name: spec.label for spec in compositions}


BASE_OVERRIDES: dict[str, Any] = {}


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={"case": spec.case_name},
                label=spec.label,
            )
            for spec in compositions
        ],
    )


def run() -> dict[str, Any]:
    write_reference_case_sweep_config(STUDY_NAME, compositions)
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
