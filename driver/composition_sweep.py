"""Composition sweep: run the sensitivity pipeline for different renewable/conventional mixes.

For each (n_wind, n_conv) composition, generates a reference case entry, registers
it in config/reference_cases.yaml (append-only — existing content is untouched),
then runs the full sensitivity pipeline (base PoA + DRO eta sweep) in an isolated
result subdirectory.

Typical usage: edit the ``SensitivityCompositionSweep`` block at the bottom and run:
  .\\.venv\\Scripts\\python.exe driver\\composition_sweep.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.run_full_pipeline import TIGHTENING_FLAGS as POA_TIGHTENING_FLAGS
from driver.run_full_pipeline_DRO import DRO_TIGHTENING_FLAGS
from driver.sensitivity_pipeline import (
    SensitivityPipelineConfig,
    main as run_sensitivity_pipeline,
)

REFERENCE_CASES_PATH = PROJECT_ROOT / "config" / "reference_cases.yaml"

# ---------------------------------------------------------------------------
# Composition specification
# ---------------------------------------------------------------------------

# Cost templates — index i → generator Gi / Wi (cheapest = index 0)
_CONV_B1_COSTS = [10.0, 30.0, 50.0, 70.0, 90.0]   # first block per conv generator
_CONV_B2_COSTS = [20.0, 40.0, 60.0, 80.0, 100.0]  # second block per conv generator
_WIND_COSTS    = [0.01, 0.25, 0.50, 0.75, 1.00]    # single block per wind generator

MAX_GENERATORS = min(len(_CONV_B1_COSTS), len(_WIND_COSTS))  # 5 of each type supported


@dataclass
class CompositionSpec:
    """Physical system composition for one sensitivity scenario."""

    n_wind: int
    n_conv: int

    # Per-generator physical parameters
    demand: float = 100.0
    conv_block_cap: float = 25.0   # MW per conv block; each conv generator has 2 blocks
    wind_block_cap: float = 50.0   # MW per wind block; each wind generator has 1 block
    conv_ramp: float = 20.0        # MW/step ramp rate (up and down)
    wind_ramp: float = 50.0        # MW/step ramp rate (up and down)

    # Cost overrides — defaults drawn from module-level templates
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
    def case_name(self) -> str:
        return f"comp_{self.n_wind}W_{self.n_conv}C"

    @property
    def wind_names(self) -> list[str]:
        return [f"W{i + 1}" for i in range(self.n_wind)]

    @property
    def conv_names(self) -> list[str]:
        return [f"G{i + 1}" for i in range(self.n_conv)]

    @property
    def total_generators(self) -> int:
        return self.n_wind + self.n_conv

    @property
    def total_conv_capacity_mw(self) -> float:
        return self.n_conv * 2 * self.conv_block_cap

    @property
    def total_wind_capacity_mw(self) -> float:
        return self.n_wind * self.wind_block_cap


# ---------------------------------------------------------------------------
# Reference case YAML generation
# ---------------------------------------------------------------------------

def generate_reference_case(spec: CompositionSpec) -> dict[str, Any]:
    """Build the YAML-compatible dict for a reference case with this composition.

    Generator ordering mirrors the base case convention: conventional generators
    are listed G{n_conv}...G1 (descending cost), then wind W{n_wind}...W1.
    This means cheaper generators appear later in the list, matching the base
    case where G3 (most expensive) has physical id 0.
    """
    generators: list[dict[str, Any]] = []
    gen_id = 0

    # Conventional generators — most expensive first (G{n_conv}, ..., G1)
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
        })
        gen_id += 1

    # Wind generators — most expensive first (W{n_wind}, ..., W1)
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
                {
                    "block_id": 0,
                    "name": f"{name}_B1",
                    "pmax": float(spec.wind_block_cap),
                    "cost": float(spec.wind_costs[i]),
                },
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
    spec: CompositionSpec,
    path: Path = REFERENCE_CASES_PATH,
) -> bool:
    """Append the composition's reference case to the YAML file if not already present.

    Existing entries are never modified — the new case is appended at the end.
    Returns True if the case was newly added, False if it already existed.
    """
    with path.open("r", encoding="utf-8") as fh:
        existing: dict[str, Any] = yaml.safe_load(fh) or {}

    if spec.case_name in existing:
        return False

    case_dict = generate_reference_case(spec)
    new_block = yaml.dump(
        {spec.case_name: case_dict},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{new_block}")

    return True


# ---------------------------------------------------------------------------
# Neural-network policy generator selection
# ---------------------------------------------------------------------------

def default_nn_policy_generators(spec: CompositionSpec) -> list[str]:
    """Choose which generators receive strategic NN policies.

    Convention (matches base case):
    - Cheapest conventional generator (G1) — the usual marginal price setter.
    - All wind generators except W1 (the zero-cost wind has no incentive to withhold).
    - If there is only one wind generator, include it regardless.
    - If there are no conventional generators, include all wind generators.
    """
    result: list[str] = []
    if spec.n_conv > 0:
        result.append("G1")
    if spec.n_wind >= 2:
        result.extend(f"W{i}" for i in range(2, spec.n_wind + 1))
    elif spec.n_wind == 1:
        result.append("W1")
    return result


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

@dataclass
class CompositionSweepConfig:
    """Configuration for a sweep over renewable/conventional generator mixes."""

    compositions: list[CompositionSpec]

    # NN and training parameters shared across all compositions
    horizon: int = 8
    synthetic_time_steps: int | None = 24
    synthetic_num_scenarios: int = 500
    synthetic_seed: int = 1
    poa_context_num_scenarios: int = 1
    poa_seed: int = 1
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"
    bid_tolerance: float = 1e-2

    nn_feature_columns: list[str] = field(
        default_factory=lambda: [
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
        ]
    )
    per_generator_normalization: bool = True
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

    # Solver
    solver_name: str = "gurobi"
    preprocessing_time_limit: int = 200
    epsilon: float = 1e-6
    poa_parallel_workers: int = 6
    poa_solver_threads_per_worker: int | None = 1
    poa_time_limit: int | None = None
    dro_time_limit: int | None = None

    # PoA objective
    poa_objective_mode: str = "piecewise_mccormick"
    poa_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    poa_mccormick_num_pieces: int = 50

    # DRO eta sweep
    etas: list[float] = field(
        default_factory=lambda: [0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0]
    )
    dro_wasserstein_epsilon: float = 2000.0
    ambiguity_kappa: float = 0.3
    dro_tightening_eta: float = 0.0
    dro_objective_mode: str = "piecewise_mccormick"
    dro_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    dro_mccormick_num_pieces: int = 50
    use_wasserstein_support_set: bool = True
    calibrate_support_coverage: bool = True
    support_verify_seed: int = 77777
    support_verify_num_draws: int = 2000
    support_coverage_grid: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.999, 0.9999]
    )
    support_include_fleet_band: bool = True
    ar1_coverage: float | None = None
    poa_worst_case_n_scenarios: int = 10
    use_poa_optimal_as_dro_scenario: bool = True

    # Stage toggles
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True
    run_poa_tightening: bool = True
    poa_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {k: True for k in POA_TIGHTENING_FLAGS}
    )
    run_poa_optimization: bool = True
    run_dro_tightening: bool = True
    dro_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {k: True for k in DRO_TIGHTENING_FLAGS}
    )
    run_dro_optimization: bool = True
    archive_existing_dro_results: bool = True

    # Root directory; each composition writes to {result_root}/{case_name}/
    result_root: Path = Path("results/composition_sweep")

    # Path to the shared reference cases YAML that compositions are registered in
    reference_cases_path: Path = REFERENCE_CASES_PATH


# ---------------------------------------------------------------------------
# Config builder for one composition
# ---------------------------------------------------------------------------

def build_composition_sensitivity_config(
    spec: CompositionSpec,
    sweep: CompositionSweepConfig,
    nn_policy_generators: list[str] | None = None,
) -> SensitivityPipelineConfig:
    """Build a SensitivityPipelineConfig for one composition.

    All I/O paths are isolated under {result_root}/{case_name}/ so that
    compositions never overwrite each other's artifacts.  NN artifacts are also
    per-composition because each has different generators and feature sets.
    """
    comp_dir = sweep.result_root / spec.case_name
    nn_gens = nn_policy_generators if nn_policy_generators is not None else default_nn_policy_generators(spec)

    return SensitivityPipelineConfig(
        case=spec.case_name,
        synthetic_time_steps=sweep.synthetic_time_steps,
        synthetic_seed=sweep.synthetic_seed,
        poa_seed=sweep.poa_seed,
        synthetic_num_scenarios=sweep.synthetic_num_scenarios,
        ambiguity_set_config_path=sweep.ambiguity_set_config_path,
        ambiguity_set_config_name=sweep.ambiguity_set_config_name,
        bid_tolerance=sweep.bid_tolerance,
        nn_feature_columns=list(sweep.nn_feature_columns),
        per_generator_normalization=sweep.per_generator_normalization,
        hidden_layers=list(sweep.hidden_layers),
        learning_rate=sweep.learning_rate,
        batch_size=sweep.batch_size,
        num_epochs=sweep.num_epochs,
        weight_decay=sweep.weight_decay,
        test_size=sweep.test_size,
        random_state=sweep.random_state,
        patience=sweep.patience,
        min_delta=sweep.min_delta,
        device=sweep.device,
        nn_final_activation=sweep.nn_final_activation,
        horizon=sweep.horizon,
        nn_policy_generators=nn_gens,
        solver_name=sweep.solver_name,
        preprocessing_time_limit=sweep.preprocessing_time_limit,
        epsilon=sweep.epsilon,
        poa_parallel_workers=sweep.poa_parallel_workers,
        poa_solver_threads_per_worker=sweep.poa_solver_threads_per_worker,
        poa_context_num_scenarios=sweep.poa_context_num_scenarios,
        poa_objective_mode=sweep.poa_objective_mode,
        poa_mccormick_PoA_bounds=sweep.poa_mccormick_PoA_bounds,
        poa_mccormick_num_pieces=sweep.poa_mccormick_num_pieces,
        poa_time_limit=sweep.poa_time_limit,
        poa_worst_case_n_scenarios=sweep.poa_worst_case_n_scenarios,
        use_poa_optimal_as_dro_scenario=sweep.use_poa_optimal_as_dro_scenario,
        etas=list(sweep.etas),
        dro_wasserstein_epsilon=sweep.dro_wasserstein_epsilon,
        ambiguity_kappa=sweep.ambiguity_kappa,
        dro_tightening_eta=sweep.dro_tightening_eta,
        dro_objective_mode=sweep.dro_objective_mode,
        dro_mccormick_PoA_bounds=sweep.dro_mccormick_PoA_bounds,
        dro_mccormick_num_pieces=sweep.dro_mccormick_num_pieces,
        dro_time_limit=sweep.dro_time_limit,
        use_wasserstein_support_set=sweep.use_wasserstein_support_set,
        calibrate_support_coverage=sweep.calibrate_support_coverage,
        support_verify_seed=sweep.support_verify_seed,
        support_verify_num_draws=sweep.support_verify_num_draws,
        support_coverage_grid=list(sweep.support_coverage_grid),
        support_include_fleet_band=sweep.support_include_fleet_band,
        ar1_coverage=sweep.ar1_coverage,
        run_scenario_generation=sweep.run_scenario_generation,
        run_heuristic_labels=sweep.run_heuristic_labels,
        run_feature_building=sweep.run_feature_building,
        run_nn_training=sweep.run_nn_training,
        run_poa_tightening=sweep.run_poa_tightening,
        poa_tightening_flags=dict(sweep.poa_tightening_flags),
        run_poa_optimization=sweep.run_poa_optimization,
        run_dro_tightening=sweep.run_dro_tightening,
        dro_tightening_flags=dict(sweep.dro_tightening_flags),
        run_dro_optimization=sweep.run_dro_optimization,
        archive_existing_dro_results=sweep.archive_existing_dro_results,
        # All I/O paths isolated to this composition's directory
        synthetic_scenario_dir=comp_dir / "synthetic_scenarios",
        poa_scenario_dir=comp_dir / "poa_scenarios",
        dro_scenario_dir=comp_dir / "dro_scenarios",
        heuristic_results_path=comp_dir / "merit_order_results.json",
        raw_feature_dir=comp_dir / "features" / "raw",
        normalized_feature_dir=comp_dir / "features" / "normalized",
        model_dir=comp_dir / "trained_models",
        training_result_dir=comp_dir / "training_results",
        poa_result_dir=comp_dir / "poa",
        dro_result_dir=comp_dir / "dro",
        dro_result_archive_dir=comp_dir / "dro" / "old_results",
        runtime_config_path=comp_dir / "runtime_regime_definitions.yaml",
        support_calibration_report_path=comp_dir / "support_calibration.json",
    )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_composition_sweep(sweep: CompositionSweepConfig) -> None:
    """Register all compositions and run the sensitivity pipeline for each."""
    n = len(sweep.compositions)
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Composition sweep  |  {n} composition(s)")
    print(f"  result_root: {sweep.result_root}")
    print(f"{sep}")

    # Register all compositions in reference_cases.yaml before running any pipeline
    print("\nRegistering compositions in reference_cases.yaml:")
    for spec in sweep.compositions:
        added = register_composition_in_yaml(spec, sweep.reference_cases_path)
        tag = "added  " if added else "exists "
        print(f"  [{tag}] {spec.case_name}  "
              f"({spec.n_wind}W + {spec.n_conv}C, "
              f"total cap = {spec.total_conv_capacity_mw:.0f} MW conv + "
              f"{spec.total_wind_capacity_mw:.0f} MW wind)")

    # Run each composition
    for idx, spec in enumerate(sweep.compositions):
        nn_gens = default_nn_policy_generators(spec)
        print(f"\n{sep}")
        print(f"  [{idx + 1}/{n}] {spec.case_name}")
        print(f"    generators : {spec.n_conv} conv ({', '.join(spec.conv_names)})  "
              f"+  {spec.n_wind} wind ({', '.join(spec.wind_names)})")
        print(f"    nn_policy  : {nn_gens}")
        print(f"    result_dir : {sweep.result_root / spec.case_name}")
        print(f"{sep}")

        config = build_composition_sensitivity_config(spec, sweep, nn_policy_generators=nn_gens)
        run_sensitivity_pipeline(config)

    print(f"\n{sep}")
    print(f"  Composition sweep complete.")
    print(f"  Results saved under: {sweep.result_root}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def standard_6gen_compositions() -> list[CompositionSpec]:
    """Return the five standard 6-generator compositions (total W+C = 6)."""
    return [
        CompositionSpec(n_wind=1, n_conv=5),
        CompositionSpec(n_wind=2, n_conv=4),
        CompositionSpec(n_wind=3, n_conv=3),
        CompositionSpec(n_wind=4, n_conv=2),
        CompositionSpec(n_wind=5, n_conv=1),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sweep_config = CompositionSweepConfig(
        compositions=standard_6gen_compositions(),

        # Model dimensions
        horizon=8,
        synthetic_time_steps=24,
        synthetic_num_scenarios=500,
        synthetic_seed=1,
        poa_context_num_scenarios=1,
        poa_seed=1,
        ambiguity_set_config_path="config/ambiguity_set_config.yaml",
        ambiguity_set_config_name="base_test_case",
        bid_tolerance=1e-2,

        # NN architecture
        nn_feature_columns=[
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
        ],
        per_generator_normalization=True,
        hidden_layers=[4, 8],
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=500,
        weight_decay=0.0,
        test_size=0.2,
        random_state=42,
        patience=50,
        min_delta=1e-6,
        device=None,
        nn_final_activation="linear",

        # Solver
        solver_name="gurobi",
        preprocessing_time_limit=200,
        epsilon=1e-6,
        poa_parallel_workers=6,
        poa_solver_threads_per_worker=1,
        poa_time_limit=None,
        dro_time_limit=None,

        # PoA objective
        poa_objective_mode="piecewise_mccormick",
        poa_mccormick_PoA_bounds=(1.0, 10.0),
        poa_mccormick_num_pieces=50,

        # DRO eta sweep
        etas=[0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0],
        dro_wasserstein_epsilon=2000.0,
        ambiguity_kappa=0.3,
        dro_tightening_eta=0.0,
        dro_objective_mode="piecewise_mccormick",
        dro_mccormick_PoA_bounds=(1.0, 10.0),
        dro_mccormick_num_pieces=50,
        use_wasserstein_support_set=True,
        poa_worst_case_n_scenarios=10,
        use_poa_optimal_as_dro_scenario=True,

        # Stage toggles — all on by default; flip to False to reuse artifacts
        run_scenario_generation=True,
        run_heuristic_labels=True,
        run_feature_building=True,
        run_nn_training=True,
        run_poa_tightening=True,
        poa_tightening_flags={k: True for k in POA_TIGHTENING_FLAGS},
        run_poa_optimization=True,
        run_dro_tightening=True,
        dro_tightening_flags={k: True for k in DRO_TIGHTENING_FLAGS},
        run_dro_optimization=True,
        archive_existing_dro_results=True,

        result_root=Path("results/composition_sweep"),
        reference_cases_path=REFERENCE_CASES_PATH,
    )

    run_composition_sweep(sweep_config)
