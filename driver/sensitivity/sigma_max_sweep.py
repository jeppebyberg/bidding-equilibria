"""Sensitivity study: max sigma in the DRO ambiguity set.

Shrinks the upper bound on the regime standard deviation, ``demand.sigma.max``
and ``wind.sigma.max`` jointly, across [0.025 (base), 0.02, 0.015, 0.01]. The
``sigma.min`` floors and every other ambiguity-set field are inherited from
base_test_case; only the sigma ceiling is tightened, so each PoA difference is
attributable to the narrower spread the uncertainty set is allowed to take.

Run:
  python -m driver.sensitivity.sigma_max_sweep
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

from driver.project_config import load_project_config  # noqa: E402
from driver.sensitivity.sensitivity_config import (  # noqa: E402
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_ambiguity_sweep_config,
)

STUDY_NAME = "sigma_max_sweep"
BASE_AMBIGUITY_CONFIG = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"

# Max sigma values under study. Base case is 0.025.
SIGMA_MAX_VALUES = [0.025, 0.02, 0.015, 0.01]
BASE_SIGMA_MAX = 0.025


@dataclass(frozen=True)
class SigmaMaxSpec:
    """One max-sigma sensitivity run."""

    sigma_max: float

    @property
    def run_name(self) -> str:
        # Filesystem-clean, sortable: 0.025 -> sigma_0p025.
        return "sigma_" + format(self.sigma_max, "g").replace(".", "p")

    @property
    def ambiguity_set_name(self) -> str:
        return self.run_name


def write_sigma_max_ambiguity_configs(specs: list[SigmaMaxSpec]) -> Path:
    base_config = load_project_config()
    with BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    base_entry = raw["ambiguity_sets"][base_config.ambiguity_set_config_name]
    ambiguity_sets: dict[str, Any] = {}
    for spec in specs:
        entry = copy.deepcopy(base_entry)
        entry["demand"]["sigma"]["max"] = float(spec.sigma_max)
        entry["wind"]["sigma"]["max"] = float(spec.sigma_max)
        entry["description"] = f"Max-sigma sensitivity: sigma.max = {spec.sigma_max:g}."
        ambiguity_sets[spec.ambiguity_set_name] = entry

    return write_ambiguity_sweep_config(
        study_name=STUDY_NAME,
        ambiguity_sets=ambiguity_sets,
        default_name=specs[0].ambiguity_set_name,
    )


def build_study(specs: list[SigmaMaxSpec]) -> SensitivityStudy:
    ambiguity_config_path = write_sigma_max_ambiguity_configs(specs)
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        runs=[
            SensitivityRun(
                name=spec.run_name,
                overrides={
                    "ambiguity_set_config_path": str(ambiguity_config_path),
                    "ambiguity_set_config_name": spec.ambiguity_set_name,
                },
                label=(
                    f"sigma.max = {spec.sigma_max:g}"
                    + (" (base)" if spec.sigma_max == BASE_SIGMA_MAX else "")
                ),
            )
            for spec in specs
        ],
    )


specs = [SigmaMaxSpec(sigma_max=v) for v in SIGMA_MAX_VALUES]


def run() -> dict[str, Any]:
    study = build_study(specs)
    return run_sensitivity_study(study)


if __name__ == "__main__":
    run()
