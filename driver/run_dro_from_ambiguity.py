"""Draw empirical scenarios directly from the ambiguity set R (regime parameters
and trajectories), run the full DRO tightening over R + the support set, and solve
the DRO-PoA program for a single eta.

This does NOT use regime_definitions.yaml.  Per-scenario regime parameters
(mu_D, sigma_D, mu_W, sigma_W) are drawn uniformly over the ambiguity-set box,
rho/peak are fixed from the ambiguity config, and a trajectory is simulated for
each draw.  A single r^DRO is then optimized over that box together with the
perturbed trajectories under the Wasserstein penalty eta.

The PoA McCormick box is NOT set by hand: the full tightening estimates the alpha
bounds, then the equilibrium_cost_bounds stage derives PoA_U = max C_eq / min C_opt
(further tightened by the analytical cost-ratio kappa).  The final solve reads that
derived box from the tightening report.

Run:
    .venv/Scripts/python.exe driver/run_dro_from_ambiguity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config.scenarios.scenario_generator import ScenarioManager
from models.DRO_PoA.DRO_PoA_optimization_new import DRO_PoAOptimization
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain

# ----------------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------------
CASE = "base_test_case"
AMBIGUITY_CONFIG_PATH = "config/ambiguity_set_config.yaml"
AMBIGUITY_SET_NAME = "base_test_case"
HORIZON = 6
NUM_SCENARIOS = 8
SEED = 1
ETA = 0.5

NN_MODEL_DIR = "models/neural_network/training/trained_models"
NN_NORMALIZATION_STATS = (
    "models/neural_network/features/generated/normalized/min_max_stats.json"
)
NN_POLICY_GENERATORS = ["G1", "W2", "W3"]

OBJECTIVE_MODE = "piecewise_mccormick"
NUM_PIECES = 25
POA_BOUNDS_LOWER = 1.0          # PoA_L; PoA_U is derived from the cost bounds
POA_BOUNDS_MARGIN = 1e-3        # safety margin on the derived PoA_U

PREPROCESSING_TIME_LIMIT = 200.0   # per-program time limit during tightening
SOLVE_TIME_LIMIT = 600.0           # final DRO MILP time limit
SOLVER_THREADS = 1
# Parallel tightening now rebuilds each worker's optimizer from the same R-based
# regime context (DROPoATighteningMain carries regime_parameters + the ambiguity
# config through _parallel_stage_state), so workers bound r^DRO by R -- not by
# regime_definitions.  Raise this to use multiple worker processes.
PARALLEL_WORKERS = 1


def draw_scenarios() -> dict[str, Any]:
    manager = ScenarioManager(base_case_reference=CASE)
    manager.base_case["time_steps"] = HORIZON
    scenario_set = manager.create_scenario_set_from_ambiguity_set(
        ambiguity_config_path=AMBIGUITY_CONFIG_PATH,
        ambiguity_set=AMBIGUITY_SET_NAME,
        n_scenarios=NUM_SCENARIOS,
        seed=SEED,
        enforce_support_set=True,
    )
    print(scenario_set["description_text"])
    print("\nDrawn empirical regimes r^(k) (mu_D, sigma_D, mu_W, sigma_W):")
    for _, row in scenario_set["scenarios_df"].iterrows():
        print(
            f"  k={int(row['scenario_id']) - 1}: "
            f"mu_D={row['mu_D']:.4f}, sigma_D={row['sigma_D']:.4f}, "
            f"mu_W={row['mu_W']:.4f}, sigma_W={row['sigma_W']:.4f}"
        )
    return scenario_set


def run_tightening(scenario_set: dict[str, Any], regime_context: dict[str, Any]) -> Path:
    tightening = DROPoATighteningMain(
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        num_time_steps=HORIZON,
        regime_parameters=regime_context,
        ambiguity_set_config_path=AMBIGUITY_CONFIG_PATH,
        ambiguity_set_config_name=AMBIGUITY_SET_NAME,
        eta=ETA,
        epsilon=0.0,
        nn_model_dir=NN_MODEL_DIR,
        nn_normalization_stats_path=NN_NORMALIZATION_STATS,
        nn_policy_generators=NN_POLICY_GENERATORS,
        reference_case=CASE,
        objective_mode=OBJECTIVE_MODE,
        mccormick_bounds={"num_pieces": NUM_PIECES},
        defer_mccormick_bound_validation=True,
        ambiguity_kappa=0.3,
        use_default_bounds=True,
        ar1_coverage=None,
    )
    print("\n=== Running full DRO tightening over R + support set ===")
    final_report_path = tightening.run_all(
        run_primal_big_m=True,
        run_relu_bounds=True,
        run_alpha_bounds=True,
        run_slack_binary_fix=True,
        run_dual_big_m=True,
        run_optimal_cost_bounds=True,
        run_equilibrium_cost_bounds=True,
        poa_bounds_lower=POA_BOUNDS_LOWER,
        poa_bounds_margin=POA_BOUNDS_MARGIN,
        solver_name="gurobi",
        time_limit=PREPROCESSING_TIME_LIMIT,
        parallel_workers=PARALLEL_WORKERS,
        solver_threads=SOLVER_THREADS,
    )
    return Path(final_report_path)


def read_derived_poa_box(report_path: Path) -> tuple[float, float]:
    with report_path.open("r", encoding="utf-8") as file_handle:
        report = json.load(file_handle)
    box = report.get("derived_poa_bounds")
    if not (isinstance(box, (list, tuple)) and len(box) == 2):
        raise ValueError(
            "Tightening report has no derived_poa_bounds; the equilibrium_cost_bounds "
            "stage (which needs alpha_bounds) must run to derive the PoA box."
        )
    return (float(box[0]), float(box[1]))


def main() -> None:
    scenario_set = draw_scenarios()
    regime_context = DRO_PoAOptimization.regime_context_from_ambiguity_set(
        ambiguity_set_config_path=AMBIGUITY_CONFIG_PATH,
        ambiguity_set_config_name=AMBIGUITY_SET_NAME,
    )

    start = time.perf_counter()

    # 1) Full tightening; the PoA box is derived (not hand-set) from the cost bounds.
    report_path = run_tightening(scenario_set, regime_context)
    derived_poa_box = read_derived_poa_box(report_path)
    print(f"\nDerived PoA box (max C_eq / min C_opt): {derived_poa_box}")

    # 2) Final solve: load the tightening report (ReLU/big-M/alpha/cost bounds) and
    #    the derived PoA box, then solve for the chosen eta.
    optimizer = DRO_PoAOptimization(
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        num_time_steps=HORIZON,
        regime_parameters=regime_context,
        ambiguity_set_config_path=AMBIGUITY_CONFIG_PATH,
        ambiguity_set_config_name=AMBIGUITY_SET_NAME,
        eta=ETA,
        nn_model_dir=NN_MODEL_DIR,
        nn_normalization_stats_path=NN_NORMALIZATION_STATS,
        nn_policy_generators=NN_POLICY_GENERATORS,
        reference_case=CASE,
        objective_mode=OBJECTIVE_MODE,
        mccormick_bounds={"PoA": derived_poa_box, "num_pieces": NUM_PIECES},
        defer_mccormick_bound_validation=True,
    )
    optimizer.load_regime_wide_tightening_report(report_path)
    optimizer.build_model()
    optimizer.apply_regime_wide_tightening_to_model()
    optimizer.solve(time_limit=SOLVE_TIME_LIMIT)
    elapsed = time.perf_counter() - start

    summary = optimizer.extract_results()
    out_path = optimizer.save_results("results/dro_poa/dro_from_ambiguity.json")

    print("\n=== DRO solve complete ===")
    print(f"  eta:                       {ETA}")
    print(f"  scenarios (K):             {summary['num_empirical_scenarios']}")
    print(f"  derived PoA box:           {derived_poa_box}")
    print(f"  objective (avg PoA-eta*W): {summary['inner_objective']}")
    print(f"  avg PoA ratio:             {summary['average_poa_ratio']}")
    print(f"  avg relaxed PoA:           {summary['average_relaxed_PoA']}")
    print(f"  avg Wasserstein W:         {summary['average_wasserstein_distance']}")
    print(f"  avg regime transport cost: {summary['average_regime_transport_cost']}")
    print(f"  avg traj. transport cost:  {summary['average_trajectory_transport_cost']}")
    print(f"  r^DRO:                     {summary['dro_regime']}")
    print(f"  tightening report:         {report_path}")
    print(f"  results JSON:              {Path(out_path)}")
    print(f"  runtime:                   {elapsed:.1f}s")


if __name__ == "__main__":
    main()

