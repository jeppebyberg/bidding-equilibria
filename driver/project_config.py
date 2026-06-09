"""Active project setup for the block-oriented workflow.

Keep most settings at their defaults in ``ProjectConfig`` and override only the
fields you want to change for the current run. All block scripts load their
configuration from this file through ``block0_core.build_config()``.
"""

from __future__ import annotations
from copy import deepcopy
from driver.core.block0_core import ProjectConfig


# Edit this object when changing experiments. For example, to do a quick data smoke run, change only:
#     synthetic_num_scenarios=50,
# and leave the rest to ProjectConfig defaults.
PROJECT_CONFIG = ProjectConfig(
    case_label = "base_case",
    synthetic_labels_target=25000,
    horizon = 6,

    allow_wind_to_play = False,

    batch_size = 16,
    lr_scheduler_patience = 20,

    plot_results_along_the_way = True,
    run_scenario_generation = True,
    run_heuristic_labels = True,
    run_feature_building = True,
    run_nn_training = True,
    run_poa_tightening = True,

    ambiguity_kappa = 0.25,

    poa_tightening_flags = {
        "relu_bounds": True,
        "alpha_bounds": True,
        "slack_binary_fix": True,
        "dual_big_m": True,
    },

    run_poa_optimization = True,

    run_dro_tightening = True,

    dro_tightening_flags = {
        "relu_bounds": True,
        "alpha_bounds": True,
        "slack_binary_fix": True,
        "dual_big_m": True,
    },

    run_dro_optimization = True,

    poa_mccormick_num_pieces = 50,
    poa_mccormick_PoA_bounds = (1.0, 10.0),

    dro_mccormick_num_pieces = 50,
    dro_mccormick_PoA_bounds = (1.0, 10.0),
)

def load_project_config() -> ProjectConfig:
    """Return a fresh copy so pipeline blocks may mutate runtime fields safely."""
    return deepcopy(PROJECT_CONFIG)
