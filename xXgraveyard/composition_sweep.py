"""Composition sweep: run the sensitivity pipeline for different renewable/conventional mixes.

For each composition spec, registers the case in config/reference_cases.yaml (append-only),
then runs the full sensitivity pipeline in an isolated result subdirectory.

All NN, solver, training, and DRO hyperparameters are defined once in a base
SensitivityPipelineConfig.  The sweep overrides only the case name, output paths,
and nn_policy_generators per composition.

To vary a single pipeline parameter across compositions (e.g., dro_wasserstein_epsilon),
pass a ``composition_overrides`` dict mapping case_name -> {field: value} in
CompositionSweepConfig.

Typical usage: edit the entry point block at the bottom and run:
  .\\.venv\\Scripts\\python.exe driver\\composition_sweep.py
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xXgraveyard.driver.full_pipeline import (
    FullPipelineConfig,
    main as run_pipeline,
)

REFERENCE_CASES_PATH = PROJECT_ROOT / "config" / "reference_cases.yaml"
SENSITIVITY_STUDIES_DIR = PROJECT_ROOT / "config" / "sensitivity_studies"

# ---------------------------------------------------------------------------
# Cost templates — index i → generator Gi / Wi (cheapest = index 0)
# ---------------------------------------------------------------------------

_CONV_B1_COSTS = [10.0, 30.0, 50.0, 70.0, 90.0]   # first block per conv generator
_CONV_B2_COSTS = [20.0, 40.0, 60.0, 80.0, 100.0]  # second block per conv generator
_WIND_COSTS    = [0.01, 0.25, 0.50, 0.75, 1.00]    # single block per wind generator

MAX_GENERATORS = min(len(_CONV_B1_COSTS), len(_WIND_COSTS))


# ---------------------------------------------------------------------------
# Shared base — fields common to all composition specifications
# ---------------------------------------------------------------------------

@dataclass
class _BaseCompositionSpec:
    """Shared fields and validation for all composition sensitivity specifications.

    Subclasses must define ``case_name``, ``conv_block_cap``, ``wind_block_cap``,
    ``conv_ramp``, and ``wind_ramp`` (either as regular fields or computed in
    ``__post_init__``).
    """

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
# Sensitivity 1 — Count composition: vary n_wind and n_conv, fixed capacities
# ---------------------------------------------------------------------------

@dataclass
class CompositionSpec(_BaseCompositionSpec):
    """Count sensitivity: vary the number of renewable vs conventional generators.

    Per-generator capacities are fixed at the base-case values (25 MW per conv
    block, 2 blocks per generator; 50 MW per wind block, 1 block per generator)
    regardless of n_wind / n_conv.

    Case name: ``comp_{n_wind}W_{n_conv}C``  e.g. ``comp_3W_3C``
    """

    conv_block_cap: float = 25.0   # MW per conv block; each conv generator has 2 blocks
    wind_block_cap: float = 50.0   # MW per wind block; each wind generator has 1 block
    conv_ramp: float = 20.0        # MW/step ramp rate (up and down)
    wind_ramp: float = 50.0        # MW/step ramp rate (up and down)

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def case_name(self) -> str:
        return f"comp_{self.n_wind}W_{self.n_conv}C"

    @property
    def total_conv_capacity_mw(self) -> float:
        return self.n_conv * 2 * self.conv_block_cap

    @property
    def total_wind_capacity_mw(self) -> float:
        return self.n_wind * self.wind_block_cap


# ---------------------------------------------------------------------------
# Sensitivity 2 — Capacity composition: vary omega and rho, fixed generator counts
# ---------------------------------------------------------------------------

@dataclass
class CapacityCompositionSpec(_BaseCompositionSpec):
    """Capacity sensitivity: vary the wind capacity share (omega) and total capacity ratio (rho).

    The number of generators (n_wind, n_conv) is fixed; per-generator capacities
    are derived from two economy-wide parameters:

        wind total  = omega       * rho * demand      [MW]
        conv total  = (1 - omega) * rho * demand      [MW]

    Each wind generator receives one block of size  wind_total / n_wind.
    Each conv generator receives two equal blocks of size  conv_total / (n_conv * 2).

    Ramp rates scale with per-generator capacity so ramp / capacity is constant:

        conv_ramp = conv_ramp_fraction * (conv_total / n_conv)
        wind_ramp = wind_ramp_fraction * (wind_total / n_wind)

    Defaults from the base case: conv 20/50 = 0.40, wind 50/50 = 1.00.
    Marginal costs (block costs) are independent of installed capacity.

    Case name: ``cap_rho{rho}_omega{omega}_{n_wind}W_{n_conv}C``
    e.g.       ``cap_rho3p0_omega0p40_3W_3C``
    """

    omega: float = 0.5   # wind fraction of total installed capacity [0, 1]
    rho: float = 3.0     # total installed capacity / demand [dimensionless]

    conv_ramp_fraction: float = 0.40  # ramp / per-generator capacity (base case: 20/50)
    wind_ramp_fraction: float = 1.00  # ramp / per-generator capacity (base case: 50/50)

    # Derived in __post_init__ — not constructor arguments.
    conv_block_cap: float = field(init=False)
    wind_block_cap: float = field(init=False)
    conv_ramp: float = field(init=False)
    wind_ramp: float = field(init=False)

    def __post_init__(self) -> None:
        total_wind_cap = self.omega * self.rho * self.demand
        total_conv_cap = (1.0 - self.omega) * self.rho * self.demand
        self.wind_block_cap = total_wind_cap / self.n_wind if self.n_wind > 0 else 0.0
        self.conv_block_cap = total_conv_cap / (self.n_conv * 2) if self.n_conv > 0 else 0.0
        self.conv_ramp = self.conv_ramp_fraction * self.conv_block_cap * 2 if self.n_conv > 0 else 0.0
        self.wind_ramp = self.wind_ramp_fraction * self.wind_block_cap if self.n_wind > 0 else 0.0
        super().__post_init__()

    @property
    def case_name(self) -> str:
        omega_str = f"{self.omega:.2f}".replace(".", "p")
        rho_str = f"{self.rho:.1f}".replace(".", "p")
        return f"cap_rho{rho_str}_omega{omega_str}_{self.n_wind}W_{self.n_conv}C"

    @property
    def total_wind_capacity_mw(self) -> float:
        return self.omega * self.rho * self.demand

    @property
    def total_conv_capacity_mw(self) -> float:
        return (1.0 - self.omega) * self.rho * self.demand


# ---------------------------------------------------------------------------
# Reference case YAML generation
# ---------------------------------------------------------------------------

def generate_reference_case(spec: _BaseCompositionSpec) -> dict[str, Any]:
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
    spec: _BaseCompositionSpec,
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

def default_nn_policy_generators(spec: _BaseCompositionSpec) -> list[str]:
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
    """Sweep over physical compositions sharing a base pipeline configuration.

    All NN, solver, training, and DRO hyperparameters live in ``base_config``.
    Per-composition, only the case name, output paths, and nn_policy_generators
    are overridden automatically.

    To vary a single pipeline parameter across compositions (e.g., to compare
    DRO results at different dro_wasserstein_epsilon values), pass
    ``composition_overrides`` mapping case_name -> {field_name: value}.
    These patches are applied on top of the path overrides.
    """

    compositions: list[_BaseCompositionSpec]
    base_config: FullPipelineConfig

    study_name: str = "capacity_composition"
    result_root: Path = Path("results/composition_sweep")
    reference_cases_path: Path = REFERENCE_CASES_PATH

    # Optional per-composition SensitivityPipelineConfig field overrides.
    # Maps case_name -> {field_name: new_value}.
    composition_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config builder for one composition
# ---------------------------------------------------------------------------

def build_composition_sensitivity_config(
    spec: _BaseCompositionSpec,
    sweep: CompositionSweepConfig,
    nn_policy_generators: list[str] | None = None,
) -> FullPipelineConfig:
    """Deep-copy the base config and apply per-composition overrides.

    Sets the case name and all output paths to an isolated subdirectory under
    {result_root}/{study_name}/{case_name}/.  Applies nn_policy_generators and
    any entries in sweep.composition_overrides[spec.case_name] on top.
    """
    config = copy.deepcopy(sweep.base_config)
    config.case = spec.case_name

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


# ---------------------------------------------------------------------------
# Study YAML writer
# ---------------------------------------------------------------------------

def write_study_yamls(
    sweep: CompositionSweepConfig,
    output_dir: Path = SENSITIVITY_STUDIES_DIR,
) -> Path:
    """Write one YAML file per composition to config/sensitivity_studies/{study_name}/.

    Each file contains a single reference case entry (same structure as
    reference_cases.yaml) so that sensitivity_pipeline.py can read them
    individually with run_sensitivity_study().

    Returns the study directory path.
    """
    study_dir = output_dir / sweep.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    for spec in sweep.compositions:
        case_dict = generate_reference_case(spec)
        yaml_path = study_dir / f"{spec.case_name}.yaml"
        yaml_path.write_text(
            yaml.dump(
                {spec.case_name: case_dict},
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {len(sweep.compositions)} case YAML(s) to: {study_dir}")
    return study_dir


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_composition_sweep(sweep: CompositionSweepConfig) -> None:
    """Write study YAMLs, register all compositions, and run the pipeline for each."""
    n = len(sweep.compositions)
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  Composition sweep  |  study='{sweep.study_name}'  ({n} composition(s))")
    print(f"  result_root: {sweep.result_root / sweep.study_name}")
    print(f"{sep}")

    # Write one YAML per composition to the study folder so that
    # sensitivity_pipeline.py can re-run the study without recomputing.
    write_study_yamls(sweep)

    # Register all compositions in reference_cases.yaml before running any pipeline.
    print("\nRegistering compositions in reference_cases.yaml:")
    for spec in sweep.compositions:
        added = register_composition_in_yaml(spec, sweep.reference_cases_path)
        tag = "added  " if added else "exists "
        print(f"  [{tag}] {spec.case_name}  "
              f"({spec.n_wind}W + {spec.n_conv}C, "
              f"total cap = {spec.total_conv_capacity_mw:.0f} MW conv + "
              f"{spec.total_wind_capacity_mw:.0f} MW wind)")

    # Run the sensitivity pipeline for each composition.
    for idx, spec in enumerate(sweep.compositions):
        nn_gens = default_nn_policy_generators(spec)
        print(f"\n{sep}")
        print(f"  [{idx + 1}/{n}] {spec.case_name}")
        print(f"    generators : {spec.n_conv} conv ({', '.join(spec.conv_names)})  "
              f"+  {spec.n_wind} wind ({', '.join(spec.wind_names)})")
        print(f"    nn_policy  : {nn_gens}")
        print(f"    result_dir : {sweep.result_root / sweep.study_name / spec.case_name}")
        print(f"{sep}")

        config = build_composition_sensitivity_config(spec, sweep, nn_policy_generators=nn_gens)
        run_pipeline(config)

    print(f"\n{sep}")
    print(f"  Composition sweep complete.")
    print(f"  Results saved under: {sweep.result_root / sweep.study_name}")
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

def omega_sweep_compositions(
    n_wind: int,
    n_conv: int,
    rho: float,
    omegas: list[float],
    demand: float = 100.0,
    conv_ramp_fraction: float = 0.40,
    wind_ramp_fraction: float = 1.00,
) -> list[CapacityCompositionSpec]:
    """Return one CapacityCompositionSpec per omega value, all with the same rho.

    Parameters
    ----------
    n_wind, n_conv : int
        Number of wind and conventional generators (fixed across sweep).
    rho : float
        Total installed capacity / demand ratio (fixed across sweep).
    omegas : list[float]
        Wind capacity fractions to sweep over, e.g. [0.2, 0.4, 0.6, 0.8].
    demand : float
        Reference demand level (MW); matches the ``demand`` field in the case YAML.
    conv_ramp_fraction, wind_ramp_fraction : float
        Ramp rate / per-generator capacity ratio (constant across omegas).
    """
    return [
        CapacityCompositionSpec(
            n_wind=n_wind,
            n_conv=n_conv,
            omega=omega,
            rho=rho,
            demand=demand,
            conv_ramp_fraction=conv_ramp_fraction,
            wind_ramp_fraction=wind_ramp_fraction,
        )
        for omega in omegas
    ]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Omega sweep: vary wind share (omega) with fixed rho and generator counts.
    #   omega = 0.2 → 20% wind / 80% conv of total installed capacity
    #   rho   = 3.0 → total installed capacity = 3 × demand = 300 MW
    #   n_wind=3, n_conv=3: 3 generators per type (6 total)
    #
    # To run a discrete-count sweep instead, replace compositions with
    # standard_6gen_compositions() which varies (n_wind, n_conv) at fixed
    # 25/50 MW blocks.
    # -----------------------------------------------------------------------
    compositions = omega_sweep_compositions(
        n_wind=3,
        n_conv=3,
        rho=3.0,
        omegas=[0.2, 0.4, 0.6, 0.8],
        demand=100.0,
    )

    # All shared hyperparameters live here.  Only case, output paths, and
    # nn_policy_generators are overridden per composition.
    base_config = FullPipelineConfig(
        # Placeholder — overridden per composition by the sweep runner.
        case="base_test_case",

        synthetic_time_steps=24,
        synthetic_seed=1,
        poa_seed=1,
        synthetic_num_scenarios=500,
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
        horizon=8,
        solver_name="gurobi",
        preprocessing_time_limit=200,
        epsilon=1e-6,
        poa_parallel_workers=6,
        poa_solver_threads_per_worker=1,

        # PoA
        poa_context_num_scenarios=1,
        poa_objective_mode="piecewise_mccormick",
        poa_mccormick_PoA_bounds=(1.0, 10.0),
        poa_mccormick_num_pieces=50,
        poa_time_limit=None,

        # DRO eta sweep
        poa_worst_case_n_scenarios=10,
        etas=[0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0],
        dro_wasserstein_epsilon=2000.0,
        ambiguity_kappa=0.3,
        dro_tightening_eta=0.0,
        dro_objective_mode="piecewise_mccormick",
        dro_mccormick_PoA_bounds=(1.0, 10.0),
        dro_mccormick_num_pieces=50,
        dro_time_limit=None,

        # Stage toggles — flip to False to reuse existing artifacts
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

    sweep_config = CompositionSweepConfig(
        compositions=compositions,
        base_config=base_config,
        study_name="capacity_composition",
        result_root=Path("results/composition_sweep"),

        # Example: vary one DRO parameter across compositions.
        # composition_overrides={
        #     "cap_rho3p0_omega0p20_3W_3C": {"dro_wasserstein_epsilon": 1000.0},
        #     "cap_rho3p0_omega0p80_3W_3C": {"dro_wasserstein_epsilon": 3000.0},
        # },
    )

    run_composition_sweep(sweep_config)
