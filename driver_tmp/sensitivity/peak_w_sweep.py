"""Example sensitivity study: vary the wind-profile peak hour.

This mirrors the legacy driver/sensitivity/peak_w_sweep.py pattern, but uses
PROJECT_CONFIG as the base setup and writes study-local ambiguity-set entries.

Run:
  .\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.peak_w_sweep
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver_tmp.project_config import load_project_config  # noqa: E402
from driver_tmp.sensitivity.sensitivity_config import (  # noqa: E402
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_ambiguity_sweep_config,
)


STUDY_NAME = "peak_w_sweep"
BASE_AMBIGUITY_CONFIG = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"


@dataclass(frozen=True)
class PeakWindSpec:
    """One peak-wind sensitivity run."""

    peak_w: float

    @property
    def run_name(self) -> str:
        return f"peak_w_{int(self.peak_w)}"

    @property
    def ambiguity_set_name(self) -> str:
        return self.run_name


def write_peak_w_ambiguity_configs(specs: list[PeakWindSpec]) -> Path:
    base_config = load_project_config()
    with BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as file_handle:
        raw: dict[str, Any] = yaml.safe_load(file_handle)

    base_entry = raw["ambiguity_sets"][base_config.ambiguity_set_config_name]
    ambiguity_sets: dict[str, Any] = {}
    for spec in specs:
        entry = copy.deepcopy(base_entry)
        entry["wind"]["tau_fixed"] = float(spec.peak_w)
        entry["description"] = f"Peak wind sensitivity: tau_fixed = {spec.peak_w:.1f} h."
        ambiguity_sets[spec.ambiguity_set_name] = entry

    return write_ambiguity_sweep_config(
        study_name=STUDY_NAME,
        ambiguity_sets=ambiguity_sets,
        default_name=specs[0].ambiguity_set_name,
    )


def build_study(specs: list[PeakWindSpec]) -> SensitivityStudy:
    ambiguity_config_path = write_peak_w_ambiguity_configs(specs)
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        runs=[
            SensitivityRun(
                spec.run_name,
                {
                    "ambiguity_set_config_path": str(ambiguity_config_path),
                    "ambiguity_set_config_name": spec.ambiguity_set_name,
                },
            )
            for spec in specs
        ],
    )


specs = [PeakWindSpec(peak_w=value) for value in (2, 8, 14, 20)]


def run() -> dict[str, Any]:
    study = build_study(specs)
    return run_sensitivity_study(study)


if __name__ == "__main__":
    run()
