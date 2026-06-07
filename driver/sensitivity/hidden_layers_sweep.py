"""Sensitivity study: number of neurons in the two hidden layers.

Varies ``ProjectConfig.hidden_layers`` over [4, 4], [8, 4], [4, 8], [8, 8];
every other setting comes from PROJECT_CONFIG. A run whose result-affecting
setup is identical to the base case (i.e. the run whose hidden_layers equal the
base-case default) copies the base-case results instead of recomputing them.

Run full study:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.hidden_layers_sweep

Run DRO only (for specific configs, assumes trained policies and PoA regimes exist):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.hidden_layers_sweep --dro-only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    base_case_setup_diff,
    build_sensitivity_config,
    run_matches_base_case,
    run_sensitivity_study,
)

STUDY_NAME = "hidden_layers_sweep"

HIDDEN_LAYER_SETUPS: list[list[int]] = [[4, 4], [8, 4], [4, 8], [8, 8], [4, 4, 4], [8, 8, 8]]


def run_name(hidden_layers: list[int]) -> str:
    return "hidden_" + "_".join(str(n) for n in hidden_layers)


def run_label(hidden_layers: list[int]) -> str:
    return "[" + ", ".join(str(n) for n in hidden_layers) + "]"


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        runs=[
            SensitivityRun(
                name=run_name(layers),
                overrides={"hidden_layers": layers},
                label=run_label(layers),
            )
            for layers in HIDDEN_LAYER_SETUPS
        ],
    )


def preview() -> None:
    """Print, per run, whether it matches the base case (and would be copied)."""
    study = build_study()
    print(f"Base-case reuse preview for study '{STUDY_NAME}':")
    for run in study.runs:
        cfg = build_sensitivity_config(study, run)
        if run_matches_base_case(cfg):
            print(f"  {run.name}: matches base case -> will copy base_case results")
        else:
            diff = base_case_setup_diff(cfg)
            reason = ", ".join(sorted(diff)) or "base-case results missing on disk"
            print(f"  {run.name}: recompute (differs in: {reason})")


def run() -> dict[str, Any]:
    return run_sensitivity_study(build_study())


def run_dro_only(
    layer_configs: list[list[int]] | None = None,
    run_tightening: bool = True,
) -> dict[str, Any]:
    """Run only the DRO analysis (block4) for the given hidden-layer configs.

    Assumes trained policies and PoA runtime regimes already exist in the
    sensitivity run directories.  Tightening is re-run by default so the
    calibrated support-set coverage is picked up; set run_tightening=False to
    reuse existing tightening reports.

    Args:
        layer_configs: list of hidden-layer specs to process.
                       Defaults to [[4, 4, 4], [8, 8, 8]].
        run_tightening: whether to redo the DRO tightening stages.
    """
    targets = layer_configs if layer_configs is not None else [[4, 4, 4], [8, 8, 8]]
    study = SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("block4",),
        runs=[
            SensitivityRun(
                name=run_name(layers),
                overrides={
                    "hidden_layers": layers,
                    "run_dro_tightening": run_tightening,
                    "run_dro_optimization": True,
                    "archive_existing_dro_results": True,
                },
                label=run_label(layers),
            )
            for layers in targets
        ],
        reuse_base_case_results=False,
    )
    return run_sensitivity_study(study)


if __name__ == "__main__":
    import sys
    if "--dro-only" in sys.argv:
        run_dro_only()
    else:
        preview()
        run()
