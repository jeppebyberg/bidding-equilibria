"""Sensitivity study: wind mean bounds in the DRO ambiguity set.

Widens / narrows the band the regime wind mean is allowed to take,
``wind.mu.min`` and ``wind.mu.max``, across a symmetric grid around 0.5:
[(0.0, 1.0), (0.1, 0.9), (0.25, 0.75) (base), (0.4, 0.6)]. Every other
ambiguity-set field (sigma bounds, rho, tau, demand block, kappa) is inherited
from base_test_case; only the wind mean band changes, so each PoA difference is
attributable to how much spread the wind mean is allowed.

Run:
  python -m driver.sensitivity.wind_bound_sweep
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

STUDY_NAME = "wind_bound_sweep"
BASE_AMBIGUITY_CONFIG = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"

# Wind mean bounds (mu.min, mu.max) under study. Base case is (0.25, 0.75).
WIND_MU_BOUNDS = [
    (0.0, 1.0),
    (0.1, 0.9),
    (0.25, 0.75),  # base, reuses base_case
    (0.4, 0.6),
]


def _fmt(value: float) -> str:
    # Filesystem-clean, sortable: 0.25 -> 0p25.
    return format(value, "g").replace(".", "p")


@dataclass(frozen=True)
class WindMuBoundSpec:
    """One wind-mean-bound sensitivity run."""

    mu_min: float
    mu_max: float

    @property
    def run_name(self) -> str:
        return f"wind_mu_{_fmt(self.mu_min)}_{_fmt(self.mu_max)}"

    @property
    def ambiguity_set_name(self) -> str:
        return self.run_name


def _base_wind_mu() -> tuple[float, float]:
    base_config = load_project_config()
    with BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        base_entry = yaml.safe_load(fh)["ambiguity_sets"][
            base_config.ambiguity_set_config_name
        ]
    return float(base_entry["wind"]["mu"]["min"]), float(base_entry["wind"]["mu"]["max"])


def write_wind_bound_ambiguity_configs(specs: list[WindMuBoundSpec]) -> Path:
    base_config = load_project_config()
    with BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    base_entry = raw["ambiguity_sets"][base_config.ambiguity_set_config_name]
    ambiguity_sets: dict[str, Any] = {}
    for spec in specs:
        entry = copy.deepcopy(base_entry)
        entry["wind"]["mu"]["min"] = float(spec.mu_min)
        entry["wind"]["mu"]["max"] = float(spec.mu_max)
        entry["description"] = (
            f"Wind-mean-bound sensitivity: wind.mu in "
            f"[{spec.mu_min:g}, {spec.mu_max:g}]."
        )
        ambiguity_sets[spec.ambiguity_set_name] = entry

    return write_ambiguity_sweep_config(
        study_name=STUDY_NAME,
        ambiguity_sets=ambiguity_sets,
        default_name=specs[0].ambiguity_set_name,
    )


def _run_overrides(spec: WindMuBoundSpec, study_config_path: Path) -> dict[str, Any]:
    """Ambiguity-set pointer for one run.

    The wind.mu = base bounds variant resolves to an ambiguity set identical to
    base_test_case, so it points at the base ambiguity config instead of the
    study-local copy. That makes its substantive config match the base case, so
    the framework reuses results/base_case rather than recomputing an identical
    solve. (Base-case reuse compares config-pointer fields literally, not the
    resolved YAML content, so a study-local pointer would always miss the match.)
    """
    base_min, base_max = _base_wind_mu()
    if (float(spec.mu_min), float(spec.mu_max)) == (base_min, base_max):
        base_config = load_project_config()
        return {
            "ambiguity_set_config_path": str(base_config.ambiguity_set_config_path),
            "ambiguity_set_config_name": base_config.ambiguity_set_config_name,
        }
    return {
        "ambiguity_set_config_path": str(study_config_path),
        "ambiguity_set_config_name": spec.ambiguity_set_name,
    }


def build_study(specs: list[WindMuBoundSpec]) -> SensitivityStudy:
    ambiguity_config_path = write_wind_bound_ambiguity_configs(specs)
    base_min, base_max = _base_wind_mu()
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        runs=[
            SensitivityRun(
                name=spec.run_name,
                overrides=_run_overrides(spec, ambiguity_config_path),
                label=(
                    f"wind.mu in [{spec.mu_min:g}, {spec.mu_max:g}]"
                    + (
                        " (base)"
                        if (spec.mu_min, spec.mu_max) == (base_min, base_max)
                        else ""
                    )
                ),
            )
            for spec in specs
        ],
    )


specs = [WindMuBoundSpec(mu_min=lo, mu_max=hi) for lo, hi in WIND_MU_BOUNDS]


def run() -> dict[str, Any]:
    study = build_study(specs)
    return run_sensitivity_study(study)


if __name__ == "__main__":
    run()
