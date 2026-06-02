"""Shared configuration and infrastructure for all sensitivity analyses.

Defines:
  - BaseConfig: project-standard hyperparameters (NN, solver, DRO) shared across
    all sensitivity studies.  Modify specific fields per study; pass to
    CompositionSweepConfig as the ``base_config``.
  - _BaseCompositionSpec: shared base for all composition spec classes.
    Concrete spec classes (e.g. CompositionSpec, CapacityCompositionSpec) live
    in the individual sensitivity scripts that use them.
  - Reference case YAML helpers.
  - CompositionSweepConfig and run_composition_sweep: sweep infrastructure used
    by individual sensitivity scripts.

Individual sensitivity scripts import from here and define only what varies.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.full_pipeline import (
    FullPipelineConfig,
    main as run_pipeline,
)

REFERENCE_CASES_PATH = PROJECT_ROOT / "config" / "reference_cases.yaml"
SENSITIVITY_STUDIES_DIR = PROJECT_ROOT / "config" / "sensitivity_studies"

_TIGHTENING_STAGES = [
    "primal_big_m",
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
    "optimal_cost_bounds",
]

# ---------------------------------------------------------------------------
# Cost templates — index i → generator Gi / Wi (cheapest = index 0)
# ---------------------------------------------------------------------------

_CONV_B1_COSTS = [10.0, 30.0, 50.0, 70.0, 90.0]
_CONV_B2_COSTS = [20.0, 40.0, 60.0, 80.0, 100.0]
_WIND_COSTS    = [0.01, 0.25, 0.50, 0.75, 1.00]


# ---------------------------------------------------------------------------
# BaseConfig — project-standard hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """Project-standard hyperparameters shared across all sensitivity analyses.

    Covers all fields in SensitivityPipelineConfig except the case name,
    output paths, and nn_policy_generators — those are set per-composition
    by the sweep infrastructure.

    Usage::

        cfg = BaseConfig(hidden_layers=[8, 8], num_epochs=200)
        # modify any field, then pass to CompositionSweepConfig:
        sweep = CompositionSweepConfig(compositions=..., base_config=cfg, ...)
    """

    # Scenario generation
    synthetic_time_steps: int | None = 24
    synthetic_seed: int = 1
    poa_seed: int = 1
    synthetic_num_scenarios: int = 1000
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"
    bid_tolerance: float = 1e-2

    # NN features
    nn_feature_columns: list[str] = field(default_factory=lambda: [
        "demand",
        "total_wind_generation_capacity",
        "total_generation_capacity",
        "residual_demand",
        "previous_generation_capacity",
        "previous_demand",
        "next_generation_capacity",
        "next_demand",
        "own_generation_capacity",
        "previous_own_generation_capacity",
        "next_own_generation_capacity",
    ])
    per_generator_normalization: bool = True

    # NN architecture
    hidden_layers: list[int] = field(default_factory=lambda: [4, 8])
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 500
    weight_decay: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    patience: int | None = 50
    min_delta: float = 1e-6
    device: str | None = None
    nn_final_activation: str = "linear"

    # NN training gate: only train generators with more accepted label changes.
    nn_training_min_label_changes: int | None = 100

    # Solver
    horizon: int = 8
    solver_name: str = "gurobi"
    preprocessing_time_limit: int = 200
    epsilon: float = 1e-6
    poa_parallel_workers: int = 6
    poa_solver_threads_per_worker: int | None = 1

    # PoA
    poa_context_num_scenarios: int = 1
    poa_objective_mode: str = "piecewise_mccormick"
    poa_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    poa_mccormick_num_pieces: int = 50
    poa_time_limit: int | None = None

    # DRO
    poa_worst_case_regime_name: str = "poa_worst_case"
    poa_worst_case_n_scenarios: int = 10
    poa_regime_set: str = "sensitivity_runtime"
    etas: list[float] = field(
        default_factory=lambda: [0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0]
    )
    dro_wasserstein_epsilon: float = 2000.0
    ambiguity_kappa: float = 0.3
    dro_tightening_eta: float = 0.0
    dro_objective_mode: str = "piecewise_mccormick"
    dro_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    dro_mccormick_num_pieces: int = 50
    dro_time_limit: int | None = None

    # Support calibration
    calibrate_support_coverage: bool = True
    support_verify_seed: int = 77777
    support_verify_num_draws: int = 2000
    support_coverage_grid: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.999, 0.9999]
    )
    ar1_coverage: float | None = None

    # Stage toggles
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True
    run_poa_tightening: bool = True
    poa_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {s: True for s in _TIGHTENING_STAGES}
    )
    run_poa_optimization: bool = True
    run_dro_tightening: bool = True
    dro_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {s: True for s in _TIGHTENING_STAGES}
    )
    run_dro_optimization: bool = True
    archive_existing_dro_results: bool = True

    def to_pipeline_config(self, case: str = "base_test_case") -> FullPipelineConfig:
        """Build a SensitivityPipelineConfig from this base config.

        Fields present in both BaseConfig and SensitivityPipelineConfig are
        forwarded directly.  Case name and output paths use SensitivityPipelineConfig
        defaults and are overridden by the sweep infrastructure.
        """
        pipeline_fields = {f.name for f in dc_fields(FullPipelineConfig)}
        kwargs = {
            f.name: getattr(self, f.name)
            for f in dc_fields(self)
            if f.name in pipeline_fields
        }
        return FullPipelineConfig(case=case, **kwargs)


# ---------------------------------------------------------------------------
# Composition spec classes
# ---------------------------------------------------------------------------

@dataclass
class _BaseCompositionSpec:
    """Shared fields and validation for all composition sensitivity specifications."""

    n_wind: int
    n_conv: int

    demand: float = 100.0
    conv_b1_costs: list[float] = field(default_factory=lambda: list(_CONV_B1_COSTS))
    conv_b2_costs: list[float] = field(default_factory=lambda: list(_CONV_B2_COSTS))
    wind_costs: list[float] = field(default_factory=lambda: list(_WIND_COSTS))

    def __post_init__(self) -> None:
        if self.n_wind + self.n_conv < 1:
            raise ValueError("Composition must have at least one generator.")
        if self.n_conv > len(self.conv_b1_costs):
            raise ValueError(
                f"n_conv={self.n_conv} exceeds conv cost template length {len(self.conv_b1_costs)}."
            )
        if self.n_wind > len(self.wind_costs):
            raise ValueError(
                f"n_wind={self.n_wind} exceeds wind cost template length {len(self.wind_costs)}."
            )

    @property
    def wind_names(self) -> list[str]:
        return [f"W{i + 1}" for i in range(self.n_wind)]

    @property
    def conv_names(self) -> list[str]:
        return [f"G{i + 1}" for i in range(self.n_conv)]

    @property
    def total_generators(self) -> int:
        return self.n_wind + self.n_conv


# ---------------------------------------------------------------------------
# Reference case YAML helpers
# ---------------------------------------------------------------------------

def generate_reference_case(spec: _BaseCompositionSpec) -> dict[str, Any]:
    """Build the YAML-compatible dict for a reference case from a composition spec.

    Generator ordering: conventional G{n_conv}...G1 (most expensive first),
    then wind W{n_wind}...W1, matching the base case convention.
    """
    generators: list[dict[str, Any]] = []
    gen_id = 0

    for i in range(spec.n_conv - 1, -1, -1):
        name = f"G{i + 1}"
        generators.append({
            "id": gen_id,
            "name": name,
            "type": "conventional",
            "pmin": 0.0,
            "R_rate_up": spec.conv_ramp,
            "R_rate_down": spec.conv_ramp,
            "bidding_blocks": [
                {"block_id": 0, "name": f"{name}_B1",
                 "pmax": float(spec.conv_block_cap), "cost": float(spec.conv_b1_costs[i])},
                {"block_id": 1, "name": f"{name}_B2",
                 "pmax": float(spec.conv_block_cap), "cost": float(spec.conv_b2_costs[i])},
            ],
        })
        gen_id += 1

    for i in range(spec.n_wind - 1, -1, -1):
        name = f"W{i + 1}"
        generators.append({
            "id": gen_id,
            "name": name,
            "type": "wind",
            "pmin": 0.0,
            "R_rate_up": spec.wind_ramp,
            "R_rate_down": spec.wind_ramp,
            "bidding_blocks": [
                {"block_id": 0, "name": f"{name}_B1",
                 "pmax": float(spec.wind_block_cap), "cost": float(spec.wind_costs[i])},
            ],
        })
        gen_id += 1

    players = [{"id": i, "controlled_generators": [i]} for i in range(gen_id)]
    return {
        "demand": [float(spec.demand)],
        "time_steps": [24],
        "generators": generators,
        "players": players,
    }


def register_composition_in_yaml(
    spec: _BaseCompositionSpec,
    path: Path = REFERENCE_CASES_PATH,
) -> bool:
    """Append the composition's reference case to the YAML file if not already present.

    Returns True if newly added, False if already existed.
    """
    with path.open("r", encoding="utf-8") as fh:
        existing: dict[str, Any] = yaml.safe_load(fh) or {}

    if spec.case_name in existing:
        return False

    new_block = yaml.dump(
        {spec.case_name: generate_reference_case(spec)},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{new_block}")
    return True


def default_nn_policy_generators(spec: _BaseCompositionSpec) -> list[str]:
    """Return which generators should receive strategic NN policies.

    Convention: G1 (cheapest conv, usual price setter) + all wind except W1
    (zero-cost wind has no incentive to withhold).  With only one wind
    generator, include it regardless.
    """
    result: list[str] = []
    if spec.n_conv > 0:
        result.append("G1")
    if spec.n_wind >= 2:
        result.extend(f"W{i}" for i in range(2, spec.n_wind + 1))
    elif spec.n_wind == 1:
        result.append("W1")
    return result


def write_study_yamls(
    sweep: CompositionSweepConfig,
    output_dir: Path = SENSITIVITY_STUDIES_DIR,
) -> Path:
    """Write one YAML per composition to config/sensitivity_studies/{study_name}/.

    Returns the study directory.  Used by sensitivity_pipeline.run_sensitivity_study
    to re-run a study without recomputing specs.
    """
    study_dir = output_dir / sweep.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    for spec in sweep.compositions:
        yaml_path = study_dir / f"{spec.case_name}.yaml"
        yaml_path.write_text(
            yaml.dump(
                {spec.case_name: generate_reference_case(spec)},
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {len(sweep.compositions)} case YAML(s) to: {study_dir}")
    return study_dir


# ---------------------------------------------------------------------------
# Composition sweep infrastructure
# ---------------------------------------------------------------------------

@dataclass
class CompositionSweepConfig:
    """Sweep over physical compositions sharing a common pipeline configuration.

    ``base_config`` holds all NN/solver/DRO hyperparameters.  Per-composition,
    only the case name, output paths, and nn_policy_generators are set
    automatically.

    To patch individual pipeline fields for specific compositions (e.g., vary
    dro_wasserstein_epsilon per run), use ``composition_overrides``::

        composition_overrides={
            "cap_rho3p0_omega0p20_3W_3C": {"dro_wasserstein_epsilon": 1000.0},
        }
    """

    compositions: list[_BaseCompositionSpec]
    base_config: BaseConfig

    study_name: str = "composition_sweep"
    result_root: Path = Path("results/sensitivity_studies")
    reference_cases_path: Path = REFERENCE_CASES_PATH

    # Maps case_name -> {SensitivityPipelineConfig field: value}.
    composition_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_composition_sensitivity_config(
    spec: _BaseCompositionSpec,
    sweep: CompositionSweepConfig,
    nn_policy_generators: list[str] | None = None,
) -> FullPipelineConfig:
    """Build a per-composition SensitivityPipelineConfig from the sweep's base config.

    Calls base_config.to_pipeline_config(), sets the case name and all output
    paths to an isolated subdirectory, then applies nn_policy_generators and
    any entries in sweep.composition_overrides.
    """
    config = sweep.base_config.to_pipeline_config(spec.case_name)

    comp_dir = sweep.result_root / sweep.study_name / spec.case_name
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

    if nn_policy_generators is not None:
        config.nn_policy_generators = nn_policy_generators

    for key, value in sweep.composition_overrides.get(spec.case_name, {}).items():
        setattr(config, key, value)

    return config


def run_composition_sweep(sweep: CompositionSweepConfig) -> None:
    """Write study YAMLs, register all compositions, and run the pipeline for each."""
    n = len(sweep.compositions)
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Composition sweep  |  study='{sweep.study_name}'  ({n} composition(s))")
    print(f"  result_root: {sweep.result_root / sweep.study_name}")
    print(f"{sep}")

    write_study_yamls(sweep)

    print("\nRegistering compositions in reference_cases.yaml:")
    for spec in sweep.compositions:
        added = register_composition_in_yaml(spec, sweep.reference_cases_path)
        tag = "added  " if added else "exists "
        print(f"  [{tag}] {spec.case_name}  "
              f"({spec.n_wind}W + {spec.n_conv}C, "
              f"total cap = {spec.total_conv_capacity_mw:.0f} MW conv + "
              f"{spec.total_wind_capacity_mw:.0f} MW wind)")

    for idx, spec in enumerate(sweep.compositions):
        nn_gens = default_nn_policy_generators(spec)
        print(f"\n{sep}")
        print(f"  [{idx + 1}/{n}] {spec.case_name}")
        print(f"    generators : {spec.n_conv} conv ({', '.join(spec.conv_names)})"
              f"  +  {spec.n_wind} wind ({', '.join(spec.wind_names)})")
        print(f"    nn_policy  : {nn_gens}")
        print(f"    result_dir : {sweep.result_root / sweep.study_name / spec.case_name}")
        print(f"{sep}")

        config = build_composition_sensitivity_config(spec, sweep, nn_policy_generators=nn_gens)
        run_pipeline(config)

    print(f"\n{sep}")
    print(f"  Sweep complete.  Results: {sweep.result_root / sweep.study_name}")
    print(f"{sep}\n")
