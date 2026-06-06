"""Peak wind sensitivity sweep: vary peak_W (wind peak hour) at fixed composition.

Sweeps peak_W in [2, 8, 14, 20] against the base_test_case physical system.
For each value, a dedicated ambiguity set is written to
  config/sensitivity_studies/peak_w_sweep_ambiguity_sets.yaml
with wind.tau_fixed overridden.  The original ambiguity_set_config.yaml is
never modified.  All PoA / DRO / NN parameters come from BaseConfig defaults.
Results land in results/sensitivity_studies/peak_w_sweep/{case_name}/.

Run:
  .venv/Scripts/python.exe driver/sensitivity/peak_w_sweep.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import copy

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xXgraveyard.driver.sensitivity.sensitivity_config import BaseConfig
from xXgraveyard.driver.full_pipeline import FullPipelineConfig, main as run_pipeline

STUDY_NAME = "peak_w_sweep"
RESULT_ROOT = Path("results/sensitivity_studies")
_BASE_AMBIGUITY_CONFIG = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"
AMBIGUITY_CONFIG_PATH = PROJECT_ROOT / "config" / "sensitivity_studies" / "peak_w_sweep_ambiguity_sets.yaml"


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
# Shared ambiguity set config writer
# ---------------------------------------------------------------------------

def write_peak_w_ambiguity_configs(specs: list[PeakWindSpec]) -> Path:
    """Write all peak_W ambiguity sets to a single shared config file.

    Reads base_test_case from ambiguity_set_config.yaml, overrides tau_fixed
    for each spec, and writes all entries to
    config/sensitivity_studies/peak_w_sweep_ambiguity_sets.yaml.
    The original ambiguity_set_config.yaml is never modified.
    """
    with _BASE_AMBIGUITY_CONFIG.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    base_entry: dict[str, Any] = dict(raw["ambiguity_sets"]["base_test_case"])

    ambiguity_sets: dict[str, Any] = {}
    for spec in specs:
        entry = copy.deepcopy(base_entry)
        entry["wind"]["tau_fixed"] = float(spec.peak_W)
        entry["description"] = f"Peak wind sensitivity: tau_fixed = {spec.peak_W:.1f} h."
        ambiguity_sets[spec.ambiguity_set_name] = entry

    out = {
        "default_ambiguity_set": specs[0].ambiguity_set_name,
        "ambiguity_sets": ambiguity_sets,
    }

    AMBIGUITY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AMBIGUITY_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Wrote {len(specs)} ambiguity set(s) to: {AMBIGUITY_CONFIG_PATH}")
    return AMBIGUITY_CONFIG_PATH


# ---------------------------------------------------------------------------
# Per-run pipeline config builder
# ---------------------------------------------------------------------------

def build_peak_w_pipeline_config(
    spec: PeakWindSpec,
    base_config: BaseConfig,
) -> FullPipelineConfig:
    """Build a FullPipelineConfig for one peak_W run against base_test_case."""
    comp_dir = RESULT_ROOT / STUDY_NAME / spec.case_name

    config = base_config.to_pipeline_config(case="base_test_case")
    config.ambiguity_set_config_path = str(AMBIGUITY_CONFIG_PATH)
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

    write_peak_w_ambiguity_configs(specs)

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

base_config = BaseConfig()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_peak_w_sweep(specs, base_config)
