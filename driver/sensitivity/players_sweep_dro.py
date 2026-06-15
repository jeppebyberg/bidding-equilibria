"""DRO eta-sweep for the players study, reusing the existing PoA artifacts.

The non-DRO ``players_sweep`` already produced, per player count, an isolated run
directory under ``results/sensitivity_studies/players_sweep/players_N{2..5}`` with
trained policies, the runtime regime config, and the PoA solve. This driver runs
ONLY block4 on those directories, so it adds the DRO eta-sweep (``<case>/dro/``)
without regenerating scenarios, retraining, or re-solving the PoA.

block4 rebuilds the DRO scenarios from each case's ``runtime_regime_definitions``
and its ``players_N*`` reference case (re-emitted here so the lookup resolves),
then solves the full default eta grid keeping the inherited ``dro_time_limit``
(1000s) cap per solve. N=3 == base_test_case already carries a DRO sweep
(results/base_case/dro), so it is intentionally skipped here.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.players_sweep_dro
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driver.sensitivity.players_sweep import (
    STUDY_NAME,
    _BASE_N,
    _case_override,
    specs,
)
from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_reference_case_sweep_config,
)

# N=3 is the base case (already has DRO); only the off-base counts need a sweep.
_DRO_BASE_OVERRIDES: dict[str, Any] = {
    # block4-only: every non-DRO stage is reused from disk, nothing recomputed.
    "run_dro_tightening": True,
    "run_dro_optimization": True,
}


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("block4",),
        # Only non-base counts are listed, so base-case copy must not short-circuit.
        reuse_base_case_results=False,
        base_overrides=_DRO_BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={"case": _case_override(spec)},
                label=f"N = {spec.n_conv}",
            )
            for spec in specs
            if spec.n_conv != _BASE_N
        ],
    )


def run() -> dict[str, Any]:
    # Re-emit the players_N* reference cases so ScenarioManager(case) resolves them.
    sweep_specs = [s for s in specs if _case_override(s) != "base_test_case"]
    write_reference_case_sweep_config(STUDY_NAME, sweep_specs)
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
