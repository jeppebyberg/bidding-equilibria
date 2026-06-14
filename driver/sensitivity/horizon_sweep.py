"""Sensitivity study: vary the PoA/DRO horizon over [4, 6, 8, 10].

Starts from ``driver.project_config.PROJECT_CONFIG`` (wind allowed to play) and
changes only the horizon per run.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.horizon_sweep
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

# Keep the total number of training labels roughly constant across horizons,
# matching the base config. ProjectConfig.__post_init__ derives
# synthetic_num_scenarios from this target, but only at construction time, so a
# later ``horizon`` override does NOT re-derive it -- we must do it per run here.
_SYNTHETIC_LABELS_TARGET = load_project_config().synthetic_labels_target


def _horizon_overrides(horizon: int) -> dict[str, Any]:
    """Field overrides that make ``horizon`` actually propagate to training data.

    Without explicitly setting ``synthetic_time_steps`` (and the derived
    ``synthetic_num_scenarios``), every run would reuse the base config's
    horizon-6 scenarios, training every NN on identical data. Each horizon must
    generate its own scenarios at its own length.
    """
    overrides: dict[str, Any] = {
        "horizon": horizon,
        "synthetic_time_steps": horizon,
    }
    if _SYNTHETIC_LABELS_TARGET is not None:
        overrides["synthetic_num_scenarios"] = math.ceil(_SYNTHETIC_LABELS_TARGET / horizon)
    return overrides


study = SensitivityStudy(
    name="horizon_sweep",
    blocks=("full",),
    # The base case is horizon=6 (project_config). The T6 run is identical to it,
    # so reuse the existing base-case results instead of recomputing them. Every
    # other horizon runs the same full pipeline the base case ran (its own
    # training, features, and solve) in a fully isolated directory -- a true 1:1
    # comparison. Do NOT share model_dir/feature dirs: with run_nn_training=True
    # the shared dirs would overwrite results/base_case before T6 can reuse it.
    reuse_base_case_results=True,
    base_overrides={
        "allow_wind_to_play": True,
        "run_scenario_generation": True,
        "run_heuristic_labels": True,
        "run_feature_building": True,
        "run_nn_training": True,
        "run_poa_optimization": True,
        "run_dro_tightening": False,
        "run_dro_optimization": False,
    },
    runs=[
        SensitivityRun("T4", _horizon_overrides(4)),
        SensitivityRun("T6", _horizon_overrides(6)),
        SensitivityRun("T8", _horizon_overrides(8)),
        SensitivityRun("T10", _horizon_overrides(10)),
    ],
)


if __name__ == "__main__":
    run_sensitivity_study(study)
