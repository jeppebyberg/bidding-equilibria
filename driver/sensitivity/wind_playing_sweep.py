"""Sensitivity study: effect of letting WIND bid strategically.

Two runs on the otherwise-identical ``base_test_case`` system (G1/G2/G3
conventional + W1/W2/W3 wind):

  * ``no_wind`` -- baseline: ``allow_wind_to_play=False`` so only the three
    conventional units carry NN bidding policies (wind is a price-taker).
  * ``wind``    -- ``allow_wind_to_play=True`` so all six generators carry NN
    policies and wind bids strategically.

Everything else is held fixed, so any change in the worst-case PoA / DRO bounds
is attributable to wind being allowed to play. Both runs cover the full PoA and
DRO stack (block1 -> block4) so the conventional-only and wind-playing cases are
compared apples-to-apples, including the eta-indexed DRO curves.

The McCormick PoA ceiling is NOT hand-set here: with ``equilibrium_cost_bounds``
on in the tightening flags (the base config default), each run derives its own
PoA box (PoA_U = max C_eq / min C_opt) from the equilibrium/optimal cost bounds
tightening stage. The static ``*_mccormick_PoA_bounds`` only supplies the lower
anchor (1.0) and a fallback, so strategic wind's higher worst-case cost ratio is
captured automatically per run -- no manual widening needed or wanted.

Outputs are isolated under
``results/sensitivity_studies/wind_playing_sweep/{no_wind,wind}`` and do not
touch results/base_case. Cross-run comparison plots are written at the end when
``plot_results_along_the_way`` is on (the base config default).

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.wind_playing_sweep

DRO-only (assumes block1-3 artifacts already exist in the run dirs):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.wind_playing_sweep --dro-only
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.sensitivity_config import (  # noqa: E402
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "wind_playing_sweep"

# Full PoA + DRO stack for both runs.
FULL_BLOCKS = ("block1", "block2", "block3", "block4")
# Only the DRO leg, for re-running block4 on top of existing block1-3 artifacts.
DRO_BLOCKS = ("block4",)


def _wind_overrides(allow_wind: bool) -> dict[str, Any]:
    # Clear nn_policy_generators so ensure_requested_policy_generators repopulates
    # it from the physical generators under the current allow_wind_to_play flag,
    # instead of reusing whatever conventional-only list the loaded PROJECT_CONFIG
    # baked in. With allow_wind=True all six units get policies; with False the
    # wind units are stripped back out.
    return {
        "allow_wind_to_play": allow_wind,
        "nn_policy_generators": [],
    }


def build_study(blocks: tuple[str, ...] = FULL_BLOCKS) -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=blocks,
        base_overrides={
            # Build every stage fresh for both runs so the baseline and the
            # wind-playing case are computed identically end-to-end.
            "run_scenario_generation": True,
            "run_heuristic_labels": True,
            "run_feature_building": True,
            "run_nn_training": True,
            "run_poa_tightening": True,
            "run_poa_optimization": True,
            "run_dro_tightening": True,
            "run_dro_optimization": True,
        },
        runs=[
            SensitivityRun(
                name="no_wind",
                overrides=_wind_overrides(allow_wind=False),
                label="wind price-taker",
            ),
            SensitivityRun(
                name="wind",
                overrides=_wind_overrides(allow_wind=True),
                label="wind strategic",
            ),
        ],
        # Use the existing results/base_case as the no_wind baseline: the no_wind
        # run's config (allow_wind_to_play=False) matches the base case, so its
        # results are copied in rather than recomputed. Only the wind run is
        # actually solved. Requires results/base_case to be fully generated
        # (block1->block4, incl. DRO) with the current PROJECT_CONFIG.
        reuse_base_case_results=True,
    )


def run() -> dict[str, Any]:
    return run_sensitivity_study(build_study())


def run_dro_only() -> dict[str, Any]:
    """Re-run only the DRO leg (block4) on top of existing block1-3 artifacts."""
    study = build_study(blocks=DRO_BLOCKS)
    for run_spec in study.runs:
        run_spec.overrides.update(
            {
                "run_dro_tightening": True,
                "run_dro_optimization": True,
                "archive_existing_dro_results": True,
            }
        )
    return run_sensitivity_study(study)


if __name__ == "__main__":
    if "--dro-only" in sys.argv:
        run_dro_only()
    else:
        run()
