"""Example sensitivity study: vary the PoA/DRO horizon.

Run:
  .\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.horizon_sweep
"""

from __future__ import annotations

from driver_tmp.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)


study = SensitivityStudy(
    name="horizon_sweep",
    blocks=("full",),
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
        SensitivityRun("T4", {"horizon": 4}),
        SensitivityRun("T8", {"horizon": 8}),
        SensitivityRun("T12", {"horizon": 12}),
    ],
)


if __name__ == "__main__":
    run_sensitivity_study(study)
