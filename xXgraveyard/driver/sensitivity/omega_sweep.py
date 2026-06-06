"""Omega sweep: vary wind capacity share (omega) at fixed rho and generator counts.

Sweeps omega in [0.2, 0.4, 0.6, 0.8] with rho=3.0 and 3W+3C generators.
All NN, solver, and DRO settings come from BaseConfig and are shared across runs.
Results land in results/sensitivity_studies/omega_sweep/{case_name}/.

Run:
  .\\.venv\\Scripts\\python.exe driver\\sensitivity\\omega_sweep.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xXgraveyard.driver.sensitivity.sensitivity_config import (
    BaseConfig,
    CompositionSweepConfig,
    _BaseCompositionSpec,
    run_composition_sweep,
)


# ---------------------------------------------------------------------------
# Composition spec — capacity sensitivity (omega / rho parameterisation)
# ---------------------------------------------------------------------------

@dataclass
class CapacityCompositionSpec(_BaseCompositionSpec):
    """Capacity sensitivity: vary wind share omega and total capacity ratio rho.

    Per-generator capacities are derived from:
        wind total  = omega       * rho * demand   [MW]
        conv total  = (1 - omega) * rho * demand   [MW]

    Case name: ``cap_rho{rho}_omega{omega}_{n_wind}W_{n_conv}C``
    """

    omega: float = 0.5   # wind fraction of total installed capacity [0, 1]
    rho: float = 3.0     # total installed capacity / demand

    conv_ramp_fraction: float = 0.40   # ramp / per-generator capacity
    wind_ramp_fraction: float = 1.00

    conv_block_cap: float = field(init=False)
    wind_block_cap: float = field(init=False)
    conv_ramp: float = field(init=False)
    wind_ramp: float = field(init=False)

    def __post_init__(self) -> None:
        total_wind = self.omega * self.rho * self.demand
        total_conv = (1.0 - self.omega) * self.rho * self.demand
        self.wind_block_cap = total_wind / self.n_wind if self.n_wind > 0 else 0.0
        self.conv_block_cap = total_conv / (self.n_conv * 2) if self.n_conv > 0 else 0.0
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


def omega_sweep_compositions(
    n_wind: int,
    n_conv: int,
    rho: float,
    omegas: list[float],
    demand: float = 100.0,
    conv_ramp_fraction: float = 0.40,
    wind_ramp_fraction: float = 1.00,
) -> list[CapacityCompositionSpec]:
    """One CapacityCompositionSpec per omega value at fixed rho and generator counts."""
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
# Compositions — what changes across runs
# ---------------------------------------------------------------------------

compositions = omega_sweep_compositions(
    n_wind=3,
    n_conv=3,
    rho=3.0,
    omegas=[0.2, 0.4, 0.6, 0.8],
    demand=100.0,
)

# ---------------------------------------------------------------------------
# Base config — everything shared across all omega runs
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
    compositions=compositions,
    base_config=base_config,
    study_name="omega_sweep",
    result_root=Path("results/sensitivity_studies"),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_composition_sweep(sweep_config)
