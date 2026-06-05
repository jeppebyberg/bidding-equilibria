"""Minimal composition sensitivity test for the driver_tmp pipeline.

This ports the legacy ``driver/sensitivity/test_sweep.py`` into the new
PROJECT_CONFIG-based sensitivity setup.  It writes two small study-local
reference cases, then runs the full block pipeline for each case with reduced
scenario counts, NN size, McCormick pieces, and solver budgets.

Run:
  .\\.venv\\Scripts\\python.exe -m driver_tmp.sensitivity.test_sweep
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driver_tmp.sensitivity.sensitivity_config import (
    BaseCompositionSpec,
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
    write_reference_case_sweep_config,
)


STUDY_NAME = "test_sweep"


@dataclass
class CompositionSpec(BaseCompositionSpec):
    """Small fixed-capacity composition used for smoke testing."""

    conv_block_cap: float = 25.0
    wind_block_cap: float = 50.0
    conv_ramp: float = 20.0
    wind_ramp: float = 50.0
    # Demand is held below the base 100 MW so that wind-heavy compositions stay
    # N-1 feasible: in 3W+1C the single 50 MW conventional unit carries all firm
    # capacity, so losing it leaves only stochastic wind. At demand=100 the
    # poa_worst_case regime cannot draw any feasible scenario. The binding step
    # is the morning shape peak, where feasibility needs ~demand <= 127 * mu_W;
    # with observed worst-case mu_W ~ 0.47, demand=40 leaves ample headroom.
    demand: float = 40.0
    label: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.label:
            self.label = f"{self.n_wind}W + {self.n_conv}C"

    @property
    def case_name(self) -> str:
        return f"test_{self.n_wind}W_{self.n_conv}C"


compositions = [
    CompositionSpec(n_wind=1, n_conv=3, label="Low wind (1W + 3C)"),
    CompositionSpec(n_wind=3, n_conv=1, label="High wind (3W + 1C)"),
    CompositionSpec(n_wind=2, n_conv=2, label="Balanced (2W + 2C)"),
]

CASE_LABELS: dict[str, str] = {
    spec.case_name: spec.label
    for spec in compositions
}


BASE_OVERRIDES: dict[str, Any] = {
    # Very few scenarios: just enough for the pipeline stages to run.
    "synthetic_num_scenarios": 500,
    "poa_context_num_scenarios": 1,
    # Short horizon keeps the MILPs small.
    "horizon": 4,
    # Tiny NN: fast to train and embed.
    "hidden_layers": [4, 4],
    "num_epochs": 500,
    "patience": 5,
    "min_delta": 1e-4,
    "batch_size": 16,
    # Coarse McCormick relaxation.
    "poa_mccormick_num_pieces": 25,
    "dro_mccormick_num_pieces": 25,

    # Keep thread usage laptop-friendly.
    "poa_parallel_workers": 6,
    "poa_solver_threads_per_worker": 1,
    # DRO: minimal empirical scenario set.
    "poa_worst_case_n_scenarios": 10,
    # Run the full pipeline so every stage is exercised.
    "run_scenario_generation": True,
    "run_heuristic_labels": True,
    "run_feature_building": True,
    "run_nn_training": True,
    "run_poa_tightening": True,
    "run_poa_optimization": True,
    "run_dro_tightening": True,
    "run_dro_optimization": True,
    "plot_results_along_the_way": True,
}


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        result_root=Path("results/sensitivity_studies"),
        blocks=("full",),
        base_overrides=BASE_OVERRIDES,
        runs=[
            SensitivityRun(
                name=spec.case_name,
                overrides={"case": spec.case_name},
                label=spec.label,
            )
            for spec in compositions
        ],
    )


def run() -> dict[str, Any]:
    write_reference_case_sweep_config(STUDY_NAME, compositions)
    return run_sensitivity_study(build_study())


if __name__ == "__main__":
    run()
