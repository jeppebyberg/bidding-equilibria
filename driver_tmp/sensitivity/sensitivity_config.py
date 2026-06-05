"""Shared sensitivity-study infrastructure for ``driver_tmp``.

Each study starts from ``driver_tmp.project_config.PROJECT_CONFIG``, changes
only the fields under study, writes outputs to an isolated run directory, and
then runs the requested block sequence.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver_tmp import (  # noqa: E402
    block1_data_labels_pipeline,
    block2_policy_training_pipeline,
    block3_poa_pipeline,
    block35_support_oos_pipeline,
    block4_dro_poa_pipeline,
    block45_oos_poa_pipeline,
    full_pipeline,
)
from driver_tmp.project_config import load_project_config  # noqa: E402
from driver_tmp.core.block0_core import (  # noqa: E402
    ProjectConfig,
    ensure_requested_policy_generators,
    pipeline_manifest,
)
from driver_tmp.block0_system_setup import write_manifest  # noqa: E402
from driver_tmp.core.visualization_core import (  # noqa: E402
    plot_sensitivity_comparison_stage,
)


RESULT_ROOT = Path("results/sensitivity_studies")
SENSITIVITY_CONFIG_DIR = PROJECT_ROOT / "config" / "sensitivity_studies"

CONV_B1_COSTS = [10.0, 30.0, 50.0, 70.0, 90.0]
CONV_B2_COSTS = [20.0, 40.0, 60.0, 80.0, 100.0]
WIND_COSTS = [0.01, 0.25, 0.50, 0.75, 1.00]

PipelineRunner = Callable[[ProjectConfig], dict[str, Any]]


BLOCK_RUNNERS: dict[str, PipelineRunner] = {
    "full": full_pipeline.run,
    "block1": block1_data_labels_pipeline.run,
    "block2": block2_policy_training_pipeline.run,
    "block3": block3_poa_pipeline.run,
    "block35": block35_support_oos_pipeline.run,
    "block4": block4_dro_poa_pipeline.run,
    "block45": block45_oos_poa_pipeline.run,
}


@dataclass
class SensitivityRun:
    """One sensitivity run: a name plus field overrides on PROJECT_CONFIG."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass
class SensitivityStudy:
    """A collection of sensitivity runs sharing one base PROJECT_CONFIG."""

    name: str
    runs: list[SensitivityRun]
    result_root: Path = RESULT_ROOT
    blocks: tuple[str, ...] = ("full",)
    base_overrides: dict[str, Any] = field(default_factory=dict)
    shared_artifact_fields: tuple[str, ...] = field(default_factory=tuple)


def load_base_config(overrides: dict[str, Any] | None = None) -> ProjectConfig:
    cfg = load_project_config()
    apply_overrides(cfg, overrides or {})
    return cfg


def apply_overrides(config: ProjectConfig, overrides: dict[str, Any]) -> ProjectConfig:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"ProjectConfig has no field '{key}'")
        setattr(config, key, copy.deepcopy(value))
    return config


def isolate_run_outputs(
    config: ProjectConfig,
    study_name: str,
    run_name: str,
    result_root: Path = RESULT_ROOT,
) -> ProjectConfig:
    """Point every generated artifact for this run into one isolated folder."""
    run_dir = Path(result_root) / study_name / run_name
    config.case_label = f"{study_name}/{run_name}"
    config.figures_dir = run_dir / "figures"
    config.poa_result_dir = run_dir / "poa"
    config.synthetic_scenario_dir = run_dir / "synthetic_scenarios"
    config.poa_scenario_dir = run_dir / "poa_scenarios"
    config.dro_scenario_dir = run_dir / "dro_scenarios"
    config.heuristic_results_path = run_dir / "merit_order_results.json"
    config.raw_feature_dir = run_dir / "features" / "raw"
    config.normalized_feature_dir = run_dir / "features" / "normalized"
    config.model_dir = run_dir / "trained_models"
    config.training_result_dir = run_dir / "training_results"
    config.dro_result_dir = run_dir / "dro"
    config.dro_result_archive_dir = run_dir / "dro" / "old_results"
    config.runtime_config_path = run_dir / "runtime_regime_definitions.yaml"
    config.support_calibration_report_path = run_dir / "support_oos_report.json"
    return config


def build_sensitivity_config(
    study: SensitivityStudy,
    run: SensitivityRun,
) -> ProjectConfig:
    base_cfg = load_base_config(study.base_overrides)
    cfg = copy.deepcopy(base_cfg)
    isolate_run_outputs(cfg, study.name, run.name, study.result_root)
    for field_name in study.shared_artifact_fields:
        if not hasattr(cfg, field_name):
            raise AttributeError(f"ProjectConfig has no field '{field_name}'")
        setattr(cfg, field_name, copy.deepcopy(getattr(base_cfg, field_name)))
    if "case" in run.overrides and "nn_policy_generators" not in run.overrides:
        cfg.nn_policy_generators = []
    apply_overrides(cfg, run.overrides)
    ensure_requested_policy_generators(cfg)
    if run.label:
        cfg.case_label = run.label
    return cfg


def run_blocks(config: ProjectConfig, blocks: Iterable[str]) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for block_name in blocks:
        if block_name not in BLOCK_RUNNERS:
            available = ", ".join(sorted(BLOCK_RUNNERS))
            raise ValueError(f"Unknown block '{block_name}'. Available: {available}")
        manifests[block_name] = BLOCK_RUNNERS[block_name](config)
    return manifests


def run_sensitivity_study(study: SensitivityStudy) -> dict[str, Any]:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Sensitivity study: {study.name} ({len(study.runs)} run(s))")
    print(f"  result_root: {Path(study.result_root) / study.name}")
    print(f"  blocks: {', '.join(study.blocks)}")
    print(f"{sep}")

    study_manifest: dict[str, Any] = {
        "study": study.name,
        "result_root": str(Path(study.result_root) / study.name),
        "blocks": list(study.blocks),
        "shared_artifact_fields": list(study.shared_artifact_fields),
        "runs": {},
    }

    for idx, run in enumerate(study.runs, start=1):
        print(f"\n{sep}")
        print(f"  [{idx}/{len(study.runs)}] {run.name}")
        if run.overrides:
            for key, value in run.overrides.items():
                print(f"    {key}: {value}")
        print(f"{sep}")

        cfg = build_sensitivity_config(study, run)
        run_manifest = {
            "config": pipeline_manifest(cfg),
            "overrides": copy.deepcopy(run.overrides),
            "block_manifests": run_blocks(cfg, study.blocks),
        }
        write_manifest(f"sensitivity_{study.name}_{run.name}", run_manifest, cfg)
        study_manifest["runs"][run.name] = run_manifest["config"]

    study_manifest_path = Path(study.result_root) / study.name / "study_manifest.yaml"
    study_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with study_manifest_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(study_manifest, file_handle, sort_keys=False)
    print(f"\nSensitivity study complete: {study_manifest_path}")

    base_cfg = load_base_config(study.base_overrides)
    if getattr(base_cfg, "plot_results_along_the_way", False):
        print(f"\n{sep}")
        print("  Building cross-run sensitivity comparison plots")
        print(f"{sep}")
        plot_sensitivity_comparison_stage(study.name, study.result_root)

    return study_manifest


def write_ambiguity_sweep_config(
    study_name: str,
    ambiguity_sets: dict[str, Any],
    default_name: str,
) -> Path:
    """Write a study-local ambiguity-set YAML under config/sensitivity_studies."""
    path = SENSITIVITY_CONFIG_DIR / f"{study_name}_ambiguity_sets.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_ambiguity_set": default_name,
        "ambiguity_sets": ambiguity_sets,
    }
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)
    print(f"Wrote ambiguity-set sweep config: {path}")
    return path


@dataclass
class BaseCompositionSpec:
    """Shared composition fields for reference-case sensitivity studies."""

    n_wind: int
    n_conv: int
    demand: float = 100.0
    conv_b1_costs: list[float] = field(default_factory=lambda: list(CONV_B1_COSTS))
    conv_b2_costs: list[float] = field(default_factory=lambda: list(CONV_B2_COSTS))
    wind_costs: list[float] = field(default_factory=lambda: list(WIND_COSTS))

    def __post_init__(self) -> None:
        if self.n_wind + self.n_conv < 1:
            raise ValueError("Composition must have at least one generator.")
        if self.n_conv > len(self.conv_b1_costs):
            raise ValueError(
                f"n_conv={self.n_conv} exceeds conv cost template length "
                f"{len(self.conv_b1_costs)}."
            )
        if self.n_wind > len(self.wind_costs):
            raise ValueError(
                f"n_wind={self.n_wind} exceeds wind cost template length "
                f"{len(self.wind_costs)}."
            )

    @property
    def wind_names(self) -> list[str]:
        return [f"W{i + 1}" for i in range(self.n_wind)]

    @property
    def conv_names(self) -> list[str]:
        return [f"G{i + 1}" for i in range(self.n_conv)]

    @property
    def case_name(self) -> str:
        return f"{self.n_wind}W_{self.n_conv}C"

    @property
    def total_conv_capacity_mw(self) -> float:
        return self.n_conv * 2 * float(getattr(self, "conv_block_cap"))

    @property
    def total_wind_capacity_mw(self) -> float:
        return self.n_wind * float(getattr(self, "wind_block_cap"))


def generate_reference_case(spec: BaseCompositionSpec) -> dict[str, Any]:
    """Build a reference-case YAML entry from a composition spec."""
    generators: list[dict[str, Any]] = []
    gen_id = 0

    for i in range(spec.n_conv - 1, -1, -1):
        name = f"G{i + 1}"
        generators.append(
            {
                "id": gen_id,
                "name": name,
                "type": "conventional",
                "pmin": 0.0,
                "R_rate_up": float(spec.conv_ramp),
                "R_rate_down": float(spec.conv_ramp),
                "bidding_blocks": [
                    {
                        "block_id": 0,
                        "name": f"{name}_B1",
                        "pmax": float(spec.conv_block_cap),
                        "cost": float(spec.conv_b1_costs[i]),
                    },
                    {
                        "block_id": 1,
                        "name": f"{name}_B2",
                        "pmax": float(spec.conv_block_cap),
                        "cost": float(spec.conv_b2_costs[i]),
                    },
                ],
            }
        )
        gen_id += 1

    for i in range(spec.n_wind - 1, -1, -1):
        name = f"W{i + 1}"
        generators.append(
            {
                "id": gen_id,
                "name": name,
                "type": "wind",
                "pmin": 0.0,
                "R_rate_up": float(spec.wind_ramp),
                "R_rate_down": float(spec.wind_ramp),
                "bidding_blocks": [
                    {
                        "block_id": 0,
                        "name": f"{name}_B1",
                        "pmax": float(spec.wind_block_cap),
                        "cost": float(spec.wind_costs[i]),
                    }
                ],
            }
        )
        gen_id += 1

    return {
        "demand": [float(spec.demand)],
        "time_steps": [24],
        "generators": generators,
        "players": [{"id": i, "controlled_generators": [i]} for i in range(gen_id)],
    }


def write_reference_case_sweep_config(
    study_name: str,
    specs: Iterable[BaseCompositionSpec],
) -> Path:
    """Write study-local reference cases under config/sensitivity_studies."""
    specs = list(specs)
    path = SENSITIVITY_CONFIG_DIR / f"{study_name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {spec.case_name: generate_reference_case(spec) for spec in specs}
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)
    print(f"Wrote {len(payload)} reference case(s): {path}")
    return path
