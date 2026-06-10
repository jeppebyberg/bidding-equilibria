# driver_tmp sensitivity studies

Sensitivity scripts start from `driver_tmp.project_config.PROJECT_CONFIG`,
override only the fields under study, and write outputs to:

```text
results/sensitivity_studies/<study_name>/<run_name>/
```

Run an existing study with:

```powershell
.\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.eta_sweep
.\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.horizon_sweep
.\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.peak_w_sweep
.\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.test_sweep
```

Create a new study by declaring the changed fields:

```python
from driver_tmp.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

study = SensitivityStudy(
    name="my_study",
    blocks=("block3", "block35", "block4", "block45"),
    shared_artifact_fields=("model_dir", "normalized_feature_dir"),
    runs=[
        SensitivityRun("eta_small", {"etas": [0.0, 0.1, 1.0]}),
        SensitivityRun("eta_large", {"etas": [0.0, 10.0, 100.0]}),
    ],
)

if __name__ == "__main__":
    run_sensitivity_study(study)
```

Use `shared_artifact_fields` when a study should reuse existing trained policy
artifacts from `PROJECT_CONFIG` while writing new PoA/DRO results to the
sensitivity folder. Leave it empty for fully isolated runs that regenerate data,
labels, features, and policies per case.
