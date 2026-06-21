# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-16
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Sensitivity study: autocorrelation rho in the DRO ambiguity set.

Varies ``demand.rho_fixed`` and ``wind.rho_fixed`` jointly across a grid that
includes negative values.  The regime-definition rho parameters (rho_D / rho_W
used for scenario generation) are unchanged; only the DRO ambiguity-set
constraint is swept.

Run:
  python -m driver.sensitivity.rho_sweep
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

STUDY_NAME = "rho_sweep"
BASE_AMBIGUITY_CONFIG = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"

RHO_VALUES = [-0.25, 0.0, 0.25, 0.50, 0.75, 0.99]


@dataclass(frozen=True)
class RhoSpec:
    """One rho sensitivity run."""

    rho: float

    @property
    def run_name(self) -> str:
        sign = "neg" if self.rho < 0 else "pos"
        mag = f"{abs(self.rho):.2f}".replace(".", "p")
        return f"rho_{sign}{mag}"

    @property
    def ambiguity_set_name(self) -> str:
        return self.run_name


def write_rho_ambiguity_configs(specs: list[RhoSpec]) -> Path:
    base_config = load_project_config()
    with BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    base_entry = raw["ambiguity_sets"][base_config.ambiguity_set_config_name]
    ambiguity_sets: dict[str, Any] = {}
    for spec in specs:
        entry = copy.deepcopy(base_entry)
        entry["demand"]["rho_fixed"] = float(spec.rho)
        entry["wind"]["rho_fixed"] = float(spec.rho)
        entry["description"] = f"Rho sensitivity: rho_fixed = {spec.rho:.2f}."
        ambiguity_sets[spec.ambiguity_set_name] = entry

    return write_ambiguity_sweep_config(
        study_name=STUDY_NAME,
        ambiguity_sets=ambiguity_sets,
        default_name=specs[0].ambiguity_set_name,
    )


def _run_overrides(spec: RhoSpec, study_config_path: Path) -> dict[str, Any]:
    """Ambiguity-set pointer for one run.

    The rho = base rho_fixed variant resolves to an ambiguity set identical to
    base_test_case, so it points at the base ambiguity config instead of the
    study-local copy. That makes its substantive config match the base case, so
    the framework reuses results/base_case rather than recomputing an identical
    solve. (Base-case reuse compares config-pointer fields literally, not the
    resolved YAML content, so a study-local pointer would always miss the match.)
    """
    base_config = load_project_config()
    with BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        base_entry = yaml.safe_load(fh)["ambiguity_sets"][base_config.ambiguity_set_config_name]
    base_rho = float(base_entry["demand"]["rho_fixed"])
    if float(spec.rho) == base_rho == float(base_entry["wind"]["rho_fixed"]):
        return {
            "ambiguity_set_config_path": str(base_config.ambiguity_set_config_path),
            "ambiguity_set_config_name": base_config.ambiguity_set_config_name,
        }
    return {
        "ambiguity_set_config_path": str(study_config_path),
        "ambiguity_set_config_name": spec.ambiguity_set_name,
    }


def build_study(specs: list[RhoSpec]) -> SensitivityStudy:
    ambiguity_config_path = write_rho_ambiguity_configs(specs)
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        runs=[
            SensitivityRun(
                name=spec.run_name,
                overrides=_run_overrides(spec, ambiguity_config_path),
            )
            for spec in specs
        ],
    )


specs = [RhoSpec(rho=v) for v in RHO_VALUES]


def run() -> dict[str, Any]:
    study = build_study(specs)
    return run_sensitivity_study(study)


if __name__ == "__main__":
    run()
