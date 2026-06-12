"""Sensitivity study: DRO PoA compute time vs horizon (T=6 vs T=8), WIND PLAYING.

Same setup as ``dro_horizon_timing_sweep`` but with wind allowed to bid
strategically (``allow_wind_to_play=True``), so all six generators
(G1/G2/G3 + W1/W2/W3) carry NN policies instead of only the three
conventional units. The McCormick PoA envelope is widened to [1, 50] because
strategic near-zero-cost wind can push the worst-case cost ratio well above the
no-wind [1, 10] range.

Everything else matches the no-wind study: full retrain per horizon, full
default eta grid, and ``dro_time_limit=None`` (no Gurobi TimeLimit) so each eta
runs to genuine termination and reports true compute time.

Outputs are isolated under
``results/sensitivity_studies/dro_horizon_timing_sweep_wind/T{6,8}`` and do NOT
touch the no-wind study or results/base_case.

Companion summary / plots (pass this study name as the first arg):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_timing_summary dro_horizon_timing_sweep_wind
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_eta_sweep_plot dro_horizon_timing_sweep_wind

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_timing_sweep_wind
"""

from __future__ import annotations

from driver.sensitivity.dro_horizon_timing_sweep import (
    HORIZONS_TO_RUN,
    _horizon_overrides,
    run_name,
)
from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    run_sensitivity_study,
)

STUDY_NAME = "dro_horizon_timing_sweep_wind"

# Widened McCormick PoA envelope for the wind-playing case.
POA_BOUNDS = (1.0, 50.0)


study = SensitivityStudy(
    name=STUDY_NAME,
    blocks=("block1", "block2", "block3", "block4"),
    base_overrides={
        # No solver time limit on the DRO solves -- true compute time per eta.
        "dro_time_limit": None,
        # Let wind bid strategically. Clearing nn_policy_generators forces
        # ensure_requested_policy_generators to repopulate it from all physical
        # generators (wind included) instead of reusing the conventional-only
        # list baked into the loaded PROJECT_CONFIG.
        "allow_wind_to_play": True,
        "nn_policy_generators": [],
        # Wider PoA envelope: strategic zero-cost wind can inflate the cost ratio
        # past the no-wind [1, 10] range.
        "poa_mccormick_PoA_bounds": POA_BOUNDS,
        "dro_mccormick_PoA_bounds": POA_BOUNDS,
        # Build everything fresh per horizon.
        "run_scenario_generation": True,
        "run_heuristic_labels": True,
        "run_feature_building": True,
        "run_nn_training": True,
        "run_poa_tightening": True,
        "run_poa_optimization": True,
        "run_dro_tightening": True,
        "run_dro_optimization": True,
        # Batch run: skip per-stage plotting.
        "plot_results_along_the_way": False,
    },
    runs=[SensitivityRun(run_name(h), _horizon_overrides(h)) for h in HORIZONS_TO_RUN],
    reuse_base_case_results=False,
)


if __name__ == "__main__":
    run_sensitivity_study(study)
