"""DRO eta-sweep for the horizon study, reusing the existing PoA artifacts.

The non-DRO ``horizon_sweep`` already produced, per horizon, an isolated run
directory under ``results/sensitivity_studies/horizon_sweep/T{4,6,8,10}`` with
trained policies, the runtime regime config, and the PoA solve. This driver runs
ONLY block4 on those directories, so it adds the DRO eta-sweep (``<case>/dro/``)
without regenerating scenarios, retraining, or re-solving the PoA.

block4 rebuilds the DRO scenarios from each case's ``runtime_regime_definitions``
and reference case, then solves the full default eta grid keeping the inherited
``dro_time_limit`` (1000s) cap per solve. T6 == base_test_case already carries a
DRO sweep (results/base_case/dro), so it is intentionally skipped here.

Run (all remaining horizons):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.horizon_sweep_dro

Run a subset (e.g. only T=10):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.horizon_sweep_dro 10
"""

from __future__ import annotations

import sys
from typing import Any

from driver.sensitivity.horizon_sweep import _horizon_overrides
from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

# T6 is the base case (already has DRO); only the remaining horizons need a sweep.
HORIZONS_TO_RUN = [4, 8, 10]

_DRO_BASE_OVERRIDES: dict[str, Any] = {
    "allow_wind_to_play": True,
    # block4-only: every non-DRO stage is reused from disk, nothing recomputed.
    "run_dro_tightening": True,
    "run_dro_optimization": True,
}


def build_study(horizons: list[int]) -> SensitivityStudy:
    """DRO eta-sweep study over the given horizons (each reuses its PoA artifacts)."""
    return SensitivityStudy(
        name="horizon_sweep",
        blocks=("block4",),
        # Only non-base horizons are listed, so base-case copy must not short-circuit.
        reuse_base_case_results=False,
        base_overrides=_DRO_BASE_OVERRIDES,
        runs=[SensitivityRun(f"T{h}", _horizon_overrides(h)) for h in horizons],
    )


def _parse_horizons(argv: list[str]) -> list[int]:
    """Horizons from CLI args (e.g. '10' or '4 10'); default = all remaining."""
    return [int(a) for a in argv] if argv else list(HORIZONS_TO_RUN)


# Default study over all remaining horizons (kept for importers).
study = build_study(HORIZONS_TO_RUN)


if __name__ == "__main__":
    run_sensitivity_study(build_study(_parse_horizons(sys.argv[1:])))
