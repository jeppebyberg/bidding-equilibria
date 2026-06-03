"""Minimal test sweep for validating the sensitivity infrastructure end-to-end.

Uses a single small composition (2W + 2C) with all pipeline settings cut to the
bone so the full sweep completes in minutes rather than hours.  Not intended to
produce meaningful PoA estimates — existence and correctness of outputs is the goal.

Run:
  .\\.venv\\Scripts\\python.exe driver\\sensitivity\\test_sweep.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

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
# Composition spec — reuse count-sensitivity shape with small fixed capacities
# ---------------------------------------------------------------------------

@dataclass
class CompositionSpec(_BaseCompositionSpec):
    conv_block_cap: float = 25.0
    wind_block_cap: float = 50.0
    conv_ramp: float = 20.0
    wind_ramp: float = 50.0
    label: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.label:
            self.label = f"{self.n_wind}W + {self.n_conv}C"

    @property
    def case_name(self) -> str:
        return f"test_{self.n_wind}W_{self.n_conv}C"

    @property
    def total_conv_capacity_mw(self) -> float:
        return self.n_conv * 2 * self.conv_block_cap

    @property
    def total_wind_capacity_mw(self) -> float:
        return self.n_wind * self.wind_block_cap


# ---------------------------------------------------------------------------
# Base config — everything turned down for speed
# ---------------------------------------------------------------------------

base_config = BaseConfig(
    # Very few scenarios — just enough for the pipeline stages to run
    synthetic_num_scenarios=50,
    poa_context_num_scenarios=1,

    # Short horizon keeps the MILP small
    horizon=4,

    # Tiny NN — fast to train, fast to embed
    hidden_layers=[4, 4],
    num_epochs=20,
    patience=5,
    min_delta=1e-4,
    batch_size=16,

    # Coarse McCormick relaxation — fewer binary variables
    poa_mccormick_num_pieces=5,
    dro_mccormick_num_pieces=5,

    # Short solver budgets
    preprocessing_time_limit=30,
    poa_time_limit=60,
    dro_time_limit=60,

    # Keep thread usage low for a laptop
    poa_parallel_workers=6,
    poa_solver_threads_per_worker=1,

    # DRO: minimal scenario set
    poa_worst_case_n_scenarios=3,

    # Run the full pipeline so every stage is exercised
    run_scenario_generation=True,
    run_heuristic_labels=True,
    run_feature_building=True,
    run_nn_training=True,
    run_poa_tightening=True,
    run_poa_optimization=True,
    run_dro_tightening=True,
    run_dro_optimization=True,
    archive_existing_dro_results=False,
)

# ---------------------------------------------------------------------------
# Compositions
# ---------------------------------------------------------------------------

compositions = [
    CompositionSpec(n_wind=1, n_conv=3, label="Low wind (1W + 3C)"),
    CompositionSpec(n_wind=3, n_conv=1, label="High wind (3W + 1C)"),
]

# Maps case_name -> human-readable label; import this in plotting scripts.
CASE_LABELS: dict[str, str] = {spec.case_name: spec.label for spec in compositions}

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

sweep_config = CompositionSweepConfig(
    compositions=compositions,
    base_config=base_config,
    study_name="test_sweep",
    result_root=Path("results/sensitivity_studies"),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_composition_sweep(sweep_config)
