from pathlib import Path
import json
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_tightening.bidding_blocks_tightening import (
    BiddingBlocksTighteningOptimizer,
)


def build_optimizer() -> BiddingBlocksTighteningOptimizer:
    """
    Create the bidding-block PoA optimizer used for alpha-bound computation.

    Keep this setup aligned with the final PoA run: same case, support set,
    horizon, NN policy directory, and selected NN-controlled generators.
    """
    case = "test_case_bidding_blocks"
    regime_set = "PoA_analysis"
    seed = 1
    horizon = 4

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    support_set_config = BiddingBlocksTighteningOptimizer.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case_bidding_blocks_base",
    )

    optimizer = BiddingBlocksTighteningOptimizer(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=horizon,
        support_set_config=support_set_config,
        nn_model_dir="models/neural_network/training/trained_models",
        nn_normalization_stats_path=(
            "models/neural_network/features/generated/normalized/min_max_stats.json"
        ),
        nn_policy_generators=[1, 2],
        reference_case=case,
    )
    return optimizer


def main() -> None:
    output_path = Path("results/poa_bidding_blocks_alpha_bounds.json")
    nn_relu_bounds_report_path = Path("results/poa_nn_relu_bounds_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = build_optimizer()

    start = time.perf_counter()
    alpha_report = optimizer.compute_nn_certified_bid_bounds(
        solver_name="gurobi",
        time_limit=200,
        tee=False,
        nn_relu_bounds_report_path=nn_relu_bounds_report_path,
    )
    elapsed = time.perf_counter() - start

    payload = {
        "metadata": {
            "description": (
                "Exact alpha bounds from support-set optimization with embedded "
                "ReLU policy constraints."
            ),
            "reference_case": optimizer.reference_case,
            "num_time_steps": optimizer.num_time_steps,
            "nn_policy_generators": list(optimizer.nn_policy_generator_names),
            "num_optimization_programs": alpha_report["num_optimization_programs"],
        },
        "alpha_bounds": alpha_report["alpha_bounds"],
        "alpha_optimization_results": alpha_report["optimization_results"],
        # Empty placeholders keep this file compatible with load_tightening_report().
        "fixed_binaries": {},
        "slack_bounds": {},
        "lambda_bounds": {},
        "tight_big_m": {},
        "aggregate_dual_bounds": {},
    }
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    alpha_values = list(alpha_report["alpha_bounds"].values())
    lower_values = [float(item["lower"]) for item in alpha_values]
    upper_values = [float(item["upper"]) for item in alpha_values]

    print("\nAlpha-bound computation complete")
    print(f"  NN ReLU bounds: {nn_relu_bounds_report_path}")
    print(f"  Report: {output_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Alpha entries: {len(alpha_values)}")
    print(f"  Min alpha lower bound: {min(lower_values):.6g}")
    print(f"  Max alpha upper bound: {max(upper_values):.6g}")


if __name__ == "__main__":
    main()
