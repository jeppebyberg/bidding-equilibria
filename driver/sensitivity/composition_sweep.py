"""Composition sweep: vary generator count mix (n_wind, n_conv) at fixed per-generator capacities.

Sweeps five 6-generator compositions from 1W+5C to 5W+1C.
All NN, solver, and DRO settings come from BaseConfig and are shared across runs.
Results land in results/sensitivity_studies/composition_sweep/{case_name}/.

Run:
  .\\.venv\\Scripts\\python.exe driver\\sensitivity\\composition_sweep.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.sensitivity_config import (
    BaseConfig,
    CompositionSweepConfig,
    _BaseCompositionSpec,
    run_composition_sweep,
)


# ---------------------------------------------------------------------------
# Composition spec — count sensitivity (fixed per-generator block capacities)
# ---------------------------------------------------------------------------

@dataclass
class CompositionSpec(_BaseCompositionSpec):
    """Count sensitivity: vary n_wind and n_conv at fixed per-generator capacities.

    Case name: ``comp_{n_wind}W_{n_conv}C``  e.g. ``comp_3W_3C``
    """

    conv_block_cap: float = 25.0   # MW per conv block; each conv generator has 2 blocks
    wind_block_cap: float = 50.0   # MW per wind block; each wind generator has 1 block
    conv_ramp: float = 20.0
    wind_ramp: float = 50.0

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


def standard_6gen_compositions() -> list[CompositionSpec]:
    """Five standard 6-generator compositions varying n_wind and n_conv (total = 6)."""
    return [
        CompositionSpec(n_wind=1, n_conv=5),
        CompositionSpec(n_wind=2, n_conv=4),
        CompositionSpec(n_wind=3, n_conv=3),
        CompositionSpec(n_wind=4, n_conv=2),
        CompositionSpec(n_wind=5, n_conv=1),
    ]


# ---------------------------------------------------------------------------
# Base config — everything shared across all composition runs
# ---------------------------------------------------------------------------

base_config = BaseConfig(
    synthetic_time_steps=24,
    synthetic_seed=1,
    poa_seed=1,
    synthetic_num_scenarios=500,
    ambiguity_set_config_path="config/ambiguity_set_config.yaml",
    ambiguity_set_config_name="base_test_case",
    bid_tolerance=1e-2,

    hidden_layers=[4, 8],
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
    poa_mccormick_num_pieces=50,

    poa_worst_case_n_scenarios=10,
    etas=[0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0],
    dro_wasserstein_epsilon=2000.0,
    ambiguity_kappa=0.3,
    dro_tightening_eta=0.0,
    dro_objective_mode="piecewise_mccormick",
    dro_mccormick_PoA_bounds=(1.0, 10.0),
    dro_mccormick_num_pieces=50,

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
# Sweep config
# ---------------------------------------------------------------------------

sweep_config = CompositionSweepConfig(
    compositions=standard_6gen_compositions(),
    base_config=base_config,
    study_name="composition_sweep",
    result_root=Path("results/sensitivity_studies"),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_composition_sweep(sweep_config)
