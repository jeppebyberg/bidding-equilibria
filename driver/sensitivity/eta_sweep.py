"""Example sensitivity study: compare different DRO eta grids.

Run:
  .\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.eta_sweep
"""

from __future__ import annotations

import numpy as np

from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)


study = SensitivityStudy(
    name="eta_grid_sweep",
    blocks=("block3", "block35", "block4", "block45"),
    shared_artifact_fields=(
        "model_dir",
        "normalized_feature_dir",
    ),
    base_overrides={
        "run_scenario_generation": False,
        "run_heuristic_labels": False,
        "run_feature_building": False,
        "run_nn_training": False,
        "run_poa_optimization": True,
        "run_dro_optimization": True,
    },
    runs=[
        SensitivityRun("coarse_eta", {"etas": [0.0, 0.1, 1.0, 10.0]}),
        SensitivityRun("default_eta", {}),
        SensitivityRun("dense_eta", {"etas": [0.0] + np.logspace(-3, 1, 21).tolist()}),
    ],
)


if __name__ == "__main__":
    run_sensitivity_study(study)
