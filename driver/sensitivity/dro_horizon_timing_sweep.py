"""Sensitivity study: DRO PoA compute time vs horizon (T=6 vs T=8).

Full retrain per horizon -- scenarios -> labels -> features -> NN training ->
PoA tightening + solve -> DRO tightening -> DRO eta sweep -- so each horizon is
self-contained and the comparison is apples-to-apples.

The DRO solve runs with ``dro_time_limit=None`` so no Gurobi ``TimeLimit`` is
imposed: every eta is solved to global optimality (or genuine termination) and
the recorded ``solver.wall_time_seconds`` is the true compute time, not a capped
incumbent. The full default eta grid is used (inherited from PROJECT_CONFIG).

Each run records, per eta, in ``dro/<regime>/dro_poa_eta_*_T*.json``:
  - ``solver.wall_time_seconds``   -> per-eta DRO compute time
  - ``solver.termination_condition``
and the aggregate ``dro/eta_sweep_summary.json``.

Companion summary (table + CSV):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_timing_summary

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_timing_sweep
"""

from __future__ import annotations

import math
from typing import Any

from driver.project_config import load_project_config
from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "dro_horizon_timing_sweep"

# Horizons (time steps) under study.
HORIZONS_TO_RUN = [6, 8]

# Keep the total number of training labels roughly constant across horizons,
# matching the base config. ProjectConfig.__post_init__ derives
# synthetic_num_scenarios from this target, but only at construction time, so a
# later ``horizon`` override does NOT re-derive it -- we must do it per run here.
_SYNTHETIC_LABELS_TARGET = load_project_config().synthetic_labels_target


def run_name(horizon: int) -> str:
    return f"T{horizon}"


def _horizon_overrides(horizon: int) -> dict[str, Any]:
    """Field overrides that make ``horizon`` actually propagate to training data.

    Without explicitly setting ``synthetic_time_steps`` (and the derived
    ``synthetic_num_scenarios``), every run would generate scenarios at the base
    config's horizon, training every NN on identical data.
    """
    overrides: dict[str, Any] = {
        "horizon": horizon,
        "synthetic_time_steps": horizon,
    }
    if _SYNTHETIC_LABELS_TARGET is not None:
        overrides["synthetic_num_scenarios"] = math.ceil(
            _SYNTHETIC_LABELS_TARGET / horizon
        )
    return overrides


study = SensitivityStudy(
    name=STUDY_NAME,
    # Full chain through the DRO solve: data/labels -> policy training ->
    # PoA tightening + solve (defines the worst-case regime) -> DRO tightening
    # -> DRO eta sweep. block35 (OOS support) and block45 (OOS PoA) are
    # post-/side-diagnostics that do not affect the DRO solve timing, so they
    # are omitted.
    blocks=("block1", "block2", "block3", "block4"),
    base_overrides={
        # THE point of this study: no solver time limit on the DRO solves, so
        # every eta runs to genuine termination and reports true compute time.
        "dro_time_limit": None,
        # Build everything fresh per horizon (scenarios/features/labels are
        # horizon-dependent), so each run is self-contained.
        "run_scenario_generation": True,
        "run_heuristic_labels": True,
        "run_feature_building": True,
        "run_nn_training": True,
        "run_poa_tightening": True,
        "run_poa_optimization": True,
        "run_dro_tightening": True,
        "run_dro_optimization": True,
        # Batch run: skip per-stage plotting to keep the run clean and fast.
        "plot_results_along_the_way": True,
    },
    runs=[SensitivityRun(run_name(h), _horizon_overrides(h)) for h in HORIZONS_TO_RUN],
    # Always recompute -- this is a timing study, never copy a stale base case.
    reuse_base_case_results=False,
)


dro_only_study = SensitivityStudy(
    name=STUDY_NAME,
    blocks=("block4",),
    base_overrides={
        "dro_time_limit": None,
        "run_scenario_generation": False,
        "run_heuristic_labels": False,
        "run_feature_building": False,
        "run_nn_training": False,
        "run_poa_tightening": False,
        "run_poa_optimization": False,
        "run_dro_tightening": False,
        "run_dro_optimization": True,
        "plot_results_along_the_way": True,
    },
    runs=[SensitivityRun(run_name(h), _horizon_overrides(h)) for h in HORIZONS_TO_RUN],
    reuse_base_case_results=True,
)


if __name__ == "__main__":
    run_sensitivity_study(dro_only_study)
