"""Sensitivity study: the ambiguity-budget multiplier ``ambiguity_kappa``.

``ambiguity_kappa`` (base 0.25) scales the support-set budget used both by the
PoA support set / tightening big-Ms (block3) and by the DRO solve (block4); it
also sets the OOS-coverage budget (block35). It does NOT affect scenario
generation, heuristic labels, feature building, or NN training, so this sweep
fixes one trained policy/feature set from ``results/base_case`` and recomputes
only the kappa-dependent legs (PoA + support-OOS + DRO) per value. Reusing the
same policies makes every PoA/DRO difference attributable to kappa alone rather
than to per-run NN-training noise, and cuts the compute roughly 5x (no retrain).

Sweep: kappa in [0.0, 0.25 (base), 0.5, 0.75, 1.0]. ``kappa = 0`` collapses the
support set to the reference point (zero budget); ``kappa < 0`` is rejected by
the PoA model, so 0 is the valid floor.

Requires an existing ``results/base_case`` (run driver/run_full_pipeline.py
first). Each value solves into its own isolated run directory under
``results/sensitivity_studies/ambiguity_kappa_sweep/<run>/``.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.ambiguity_kappa_sweep
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.sensitivity_config import (  # noqa: E402
    SensitivityRun,
    SensitivityStudy,
    base_case_result_root,
    run_sensitivity_study,
)

STUDY_NAME = "ambiguity_kappa_sweep"

# Ambiguity-budget multipliers under study. Base case is 0.25.
KAPPA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE_KAPPA = 0.25

# kappa changes only the PoA/support/DRO legs, so reuse one fixed policy/feature
# set from results/base_case and skip the data + NN stages for every run.
REUSE_DATA_NN_OVERRIDES: dict[str, Any] = {
    "run_scenario_generation": False,
    "run_heuristic_labels": False,
    "run_feature_building": False,
    "run_nn_training": False,
}

# Output-location fields repointed back at results/base_case so the reused data
# and trained policies are read in place (read-only); the kappa-dependent legs
# still write into each run's own isolated directories.
SHARED_BASE_CASE_FIELDS: tuple[str, ...] = (
    "synthetic_scenario_dir",
    "heuristic_results_path",
    "raw_feature_dir",
    "normalized_feature_dir",
    "model_dir",
    "training_result_dir",
)


@dataclass(frozen=True)
class KappaSpec:
    """One ambiguity-kappa sensitivity run."""

    kappa: float

    @property
    def run_name(self) -> str:
        # Filesystem-clean, fixed-width, sortable: 0.25 -> kappa_0p25.
        return "kappa_" + f"{self.kappa:.2f}".replace(".", "p")

    @property
    def label(self) -> str:
        base = f"kappa = {self.kappa:g}"
        return base + " (base)" if self.kappa == BASE_KAPPA else base


def build_study(specs: list[KappaSpec]) -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("full",),
        base_overrides=dict(REUSE_DATA_NN_OVERRIDES),
        shared_artifact_fields=SHARED_BASE_CASE_FIELDS,
        # Every run recomputes PoA + DRO with the reused policies, so none is a
        # literal base-case match (the skip-regen flags differ); disable copying.
        reuse_base_case_results=False,
        runs=[
            SensitivityRun(
                name=spec.run_name,
                overrides={"ambiguity_kappa": float(spec.kappa)},
                label=spec.label,
            )
            for spec in specs
        ],
    )


specs = [KappaSpec(kappa=v) for v in KAPPA_VALUES]


def run() -> dict[str, Any]:
    policies_dir = base_case_result_root() / "trained_models"
    if not policies_dir.exists():
        raise FileNotFoundError(
            f"No trained policies under {policies_dir}. Generate the base case "
            "first (driver/run_full_pipeline.py) before sweeping ambiguity_kappa."
        )
    study = build_study(specs)
    return run_sensitivity_study(study)


if __name__ == "__main__":
    run()
