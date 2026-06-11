"""Sensitivity study: base-PoA compute time and binary count vs horizon.

Holds the NN architecture fixed at ``[8, 8]`` and solves the base PoA (no DRO)
for an increasing number of time steps. Each PoA MILP solve is capped at
``POA_TIME_LIMIT`` seconds; runs that hit the cap report the incumbent and MIP
gap rather than a proven optimum. Each run records, in its
``poa/poa_optimization_T*.json``:

  - ``solver.wall_time_seconds``            -> compute time
  - ``solver.variable_counts.*``            -> binary-variable counts

Companion summary (table + CSV/JSON + plots):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.horizon_poa_compute_summary

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.horizon_poa_compute_sweep
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

STUDY_NAME = "horizon_poa_compute_sweep"

# Fixed NN architecture for the whole sweep.
HIDDEN_LAYERS = [8, 8]

# Horizons (time steps) under study. Used by the companion summary to report
# every horizon that has a result on disk.
HORIZONS = [4, 6, 8, 10, 12, 24]

# Horizons this sweep actually (re)computes. Subset of HORIZONS so a run can
# fill in missing horizons without overwriting already-completed ones.
HORIZONS_TO_RUN = [12]

# Horizons whose upstream artifacts (scenarios, labels, features, trained
# policies, tightening reports) already exist on disk and should be reused: the
# run jumps straight to the PoA solve instead of rebuilding and retraining.
# T12 completed everything except the interrupted solve, so only it re-solves.
REUSE_UPSTREAM = {12}

# Per-run PoA solve time limit (seconds). Runs that hit this limit report the
# incumbent and MIP gap instead of running to global optimality.
POA_TIME_LIMIT = 5000

# Keep the total number of training labels roughly constant across horizons,
# matching the base config. ProjectConfig.__post_init__ derives
# synthetic_num_scenarios from this target, but only at construction time, so a
# later ``horizon`` override does NOT re-derive it -- we must do it per run here.
_SYNTHETIC_LABELS_TARGET = load_project_config().synthetic_labels_target


def run_name(horizon: int) -> str:
    return f"T{horizon}"


def run_label(horizon: int) -> str:
    return f"T={horizon}"


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
    if horizon in REUSE_UPSTREAM:
        # Reuse the existing scenarios/labels/features/policies/tightening for
        # this horizon and only run the (time-limited) PoA solve.
        overrides.update(
            {
                "run_scenario_generation": False,
                "run_heuristic_labels": False,
                "run_feature_building": False,
                "run_nn_training": False,
                "run_poa_tightening": False,
                "run_poa_optimization": True,
            }
        )
    return overrides


study = SensitivityStudy(
    name=STUDY_NAME,
    # Base-PoA pipeline only: data/labels -> policy training -> PoA solve.
    # No block35/block4/block45 (those are the DRO bridge and DRO solve).
    blocks=("block1", "block2", "block3"),
    base_overrides={
        "hidden_layers": list(HIDDEN_LAYERS),
        # Cap each PoA MILP solve; runs hitting the cap report incumbent + gap.
        "poa_time_limit": POA_TIME_LIMIT,
        # Build everything fresh per horizon (scenarios/features/labels are
        # horizon-dependent), so each run is self-contained.
        "run_scenario_generation": True,
        "run_heuristic_labels": True,
        "run_feature_building": True,
        "run_nn_training": True,
        "run_poa_tightening": True,
        "run_poa_optimization": True,
        # Base PoA only -- keep the DRO half of the pipeline off.
        "run_dro_tightening": False,
        "run_dro_optimization": False,
    },
    runs=[SensitivityRun(run_name(h), _horizon_overrides(h)) for h in HORIZONS_TO_RUN],
    # Always recompute -- this is a timing study, never copy a stale base case.
    reuse_base_case_results=False,
)


if __name__ == "__main__":
    run_sensitivity_study(study)
