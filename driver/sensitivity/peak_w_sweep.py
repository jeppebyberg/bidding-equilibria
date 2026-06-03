"""Peak wind sensitivity sweep: vary peak_W (wind peak hour) at fixed composition.

Sweeps peak_W in [2, 8, 14, 20] against the base_test_case physical system.
For each value, a custom ambiguity set config is written with wind.tau_fixed
overridden; all other PoA / DRO / NN parameters come from BaseConfig.
Results land in results/sensitivity_studies/peak_w_sweep/{case_name}/.

Run:
  .\\.venv\\Scripts\\python.exe driver\\sensitivity\\peak_w_sweep.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.sensitivity_config import BaseConfig
from driver.full_pipeline import FullPipelineConfig, main as run_pipeline

STUDY_NAME = "peak_w_sweep"
RESULT_ROOT = Path("results/sensitivity_studies")
_BASE_AMBIGUITY_CONFIG = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"

# ---------------------------------------------------------------------------
# Sweep spec
# ---------------------------------------------------------------------------

@dataclass
class PeakWindSpec:
    """One peak_W sensitivity run.

    peak_W [0, 24]: hour of peak wind generation in the AR(1) wind profile.
    All other ambiguity-set and physical parameters inherit from the base case.
    """

    peak_W: float

    @property
    def case_name(self) -> str:
        return f"peak_w_{int(self.peak_W)}"

    @property
    def ambiguity_set_name(self) -> str:
        return self.case_name


# ---------------------------------------------------------------------------
# Ambiguity set config writer
# ---------------------------------------------------------------------------

def write_peak_w_ambiguity_config(spec: PeakWindSpec, config_dir: Path) -> Path:
    """Write a copy of the base ambiguity set config with tau_fixed = spec.peak_W."""
    with _BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    base_entry: dict[str, Any] = dict(raw["ambiguity_sets"]["base_test_case"])
    base_entry["wind"] = dict(base_entry["wind"])
    base_entry["wind"]["tau_fixed"] = float(spec.peak_W)
    base_entry["description"] = (
        f"Peak wind sensitivity: tau_fixed = {spec.peak_W:.1f} h."
    )

    out = {
        "default_ambiguity_set": spec.ambiguity_set_name,
        "ambiguity_sets": {spec.ambiguity_set_name: base_entry},
    }

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "ambiguity_set_config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return config_path


# ---------------------------------------------------------------------------
# Per-run pipeline config builder
# ---------------------------------------------------------------------------

def build_peak_w_pipeline_config(
    spec: PeakWindSpec,
    base_config: BaseConfig,
) -> FullPipelineConfig:
    """Build a FullPipelineConfig for one peak_W run against base_test_case."""
    comp_dir = RESULT_ROOT / STUDY_NAME / spec.case_name

    ambiguity_config_path = write_peak_w_ambiguity_config(spec, comp_dir / "config")

    config = base_config.to_pipeline_config(case="base_test_case")
    config.ambiguity_set_config_path = str(ambiguity_config_path)
    config.ambiguity_set_config_name = spec.ambiguity_set_name

    config.synthetic_scenario_dir = comp_dir / "synthetic_scenarios"
    config.poa_scenario_dir = comp_dir / "poa_scenarios"
    config.dro_scenario_dir = comp_dir / "dro_scenarios"
    config.heuristic_results_path = comp_dir / "merit_order_results.json"
    config.raw_feature_dir = comp_dir / "features" / "raw"
    config.normalized_feature_dir = comp_dir / "features" / "normalized"
    config.model_dir = comp_dir / "trained_models"
    config.training_result_dir = comp_dir / "training_results"
    config.poa_result_dir = comp_dir / "poa"
    config.dro_result_dir = comp_dir / "dro"
    config.dro_result_archive_dir = comp_dir / "dro" / "old_results"
    config.runtime_config_path = comp_dir / "runtime_regime_definitions.yaml"
    config.support_calibration_report_path = comp_dir / "support_calibration.json"

    return config


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_peak_w_sweep(specs: list[PeakWindSpec], base_config: BaseConfig) -> None:
    n = len(specs)
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Peak wind sweep  |  study='{STUDY_NAME}'  ({n} run(s))")
    print(f"  result_root: {RESULT_ROOT / STUDY_NAME}")
    print(f"  peak_W values: {[s.peak_W for s in specs]}")
    print(f"{sep}")

    for idx, spec in enumerate(specs):
        print(f"\n{sep}")
        print(f"  [{idx + 1}/{n}] {spec.case_name}  (peak_W = {spec.peak_W:.1f} h)")
        print(f"  result_dir : {RESULT_ROOT / STUDY_NAME / spec.case_name}")
        print(f"{sep}")

        config = build_peak_w_pipeline_config(spec, base_config)
        run_pipeline(config)

    print(f"\n{sep}")
    print(f"  Sweep complete.  Results: {RESULT_ROOT / STUDY_NAME}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Sweep specification
# ---------------------------------------------------------------------------

specs = [PeakWindSpec(peak_W=v) for v in [2, 8, 14, 20]]

base_config = BaseConfig(
    synthetic_time_steps=24,
    synthetic_seed=1,
    poa_seed=1,
    synthetic_num_scenarios=1000,
    ambiguity_set_config_path="config/ambiguity_set_config.yaml",  # overridden per run
    ambiguity_set_config_name="base_test_case",                    # overridden per run
    bid_tolerance=1e-2,

    hidden_layers=[8, 8],
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=500,
    patience=50,
    min_delta=1e-6,
    nn_final_activation="linear",

    horizon=8,
    solver_name="gurobi",
    preprocessing_time_limit=200,
    epsilon=1e-6,
    poa_parallel_workers=6,
    poa_solver_threads_per_worker=1,

    poa_context_num_scenarios=1,
    poa_objective_mode="piecewise_mccormick",
    poa_mccormick_PoA_bounds=(1.0, 10.0),
    poa_mccormick_num_pieces=100,

    poa_worst_case_n_scenarios=10,
    etas=[0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0],
    dro_wasserstein_epsilon=2000.0,
    ambiguity_kappa=0.3,
    dro_tightening_eta=0.0,
    dro_objective_mode="piecewise_mccormick",
    dro_mccormick_PoA_bounds=(1.0, 10.0),
    dro_mccormick_num_pieces=100,

    run_scenario_generation=True,
    run_heuristic_labels=True,
    run_feature_building=True,
    run_nn_training=True,
    run_poa_tightening=True,
    run_poa_optimization=True,
    run_dro_tightening=True,
    run_dro_optimization=True,
    archive_existing_dro_results=True,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_peak_w_sweep(specs, base_config)
