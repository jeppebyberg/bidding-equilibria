# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-17
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Sensitivity study: reference demand level used for scenario generation.

The regime parameters ``mu_D`` and ``sigma_D`` are expressed in
selected-reference-demand units, so the reference case's ``demand`` field is the
scale that turns them into absolute MW. Base_test_case uses demand = 100; this
study sweeps it across [80, 90, 100, 110, 120] while leaving every generator,
cost, capacity, ramp rate and regime parameter untouched. Each PoA difference is
therefore attributable to the reference demand level alone.

Each run deepcopies the literal ``base_test_case`` reference-case entry and
changes only ``demand``, so the swept cases differ from the base by exactly one
field. The demand = 100 run resolves directly to ``base_test_case`` so existing
base-case results are reused rather than recomputed.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.demand_ref_sweep
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
    SENSITIVITY_CONFIG_DIR,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "demand_ref_sweep"
BASE_REFERENCE_CASES = PROJECT_ROOT / "config" / "reference_cases.yaml"
BASE_CASE_NAME = "base_test_case"

# Reference demand levels under study. Base case is 100.
DEMAND_LEVELS = [80.0, 90.0, 100.0, 110.0, 120.0]


@dataclass(frozen=True)
class DemandRefSpec:
    """One reference-demand-level sensitivity run."""

    demand: float

    @property
    def is_base(self) -> bool:
        return float(self.demand) == 100.0

    @property
    def case_name(self) -> str:
        # demand = 100 reuses the existing base_test_case entry/results.
        return BASE_CASE_NAME if self.is_base else f"dref_{int(self.demand)}"

    @property
    def run_name(self) -> str:
        return f"dref_{int(self.demand)}"

    @property
    def label(self) -> str:
        return f"D_ref = {self.demand:g}" + (" (base)" if self.is_base else "")


def _base_case_entry() -> dict[str, Any]:
    with BASE_REFERENCE_CASES.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return raw[BASE_CASE_NAME]


def write_demand_ref_cases(specs: list[DemandRefSpec]) -> Path:
    """Write study-local reference cases that differ from base only in ``demand``.

    Written under config/sensitivity_studies/ so ``load_test_case`` discovers
    them automatically. The demand = 100 spec resolves to ``base_test_case`` and
    is therefore not emitted here.
    """
    base_entry = _base_case_entry()
    cases: dict[str, Any] = {}
    for spec in specs:
        if spec.is_base:
            continue
        entry = copy.deepcopy(base_entry)
        entry["demand"] = [float(spec.demand)]
        cases[spec.case_name] = entry

    path = SENSITIVITY_CONFIG_DIR / f"{STUDY_NAME}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cases, fh, sort_keys=False)
    print(f"Wrote {len(cases)} reference case(s): {path}")
    return path


def build_study(specs: list[DemandRefSpec]) -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        runs=[
            SensitivityRun(
                name=spec.run_name,
                overrides={"case": spec.case_name},
                label=spec.label,
            )
            for spec in specs
        ],
    )


specs = [DemandRefSpec(demand=value) for value in DEMAND_LEVELS]


def run() -> dict[str, Any]:
    # The base (demand = 100) run resolves to base_test_case, already present in
    # reference_cases.yaml, so only the off-base levels get study-local entries.
    write_demand_ref_cases(specs)
    return run_sensitivity_study(build_study(specs))


if __name__ == "__main__":
    run()
