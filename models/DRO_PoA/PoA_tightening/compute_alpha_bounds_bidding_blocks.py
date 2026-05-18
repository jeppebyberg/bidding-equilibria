from pathlib import Path
import json
import time

from config.scenarios.scenario_generator import ScenarioManager
from models.DRO_PoA.PoA_tightening.bidding_blocks_tightening import (
    BiddingBlocksTighteningOptimizer,
)


def build_optimizer() -> BiddingBlocksTighteningOptimizer:
    """
    Create the DRO bidding-block optimizer used for alpha-bound computation.

    Keep this setup aligned with the final DRO-PoA run: same case, support set,
    horizon, and empirical scenarios.
    """
    case = "test_case_bidding_blocks"
    regime_set = "PoA_analysis"
    seed = 1
    max_scenarios = 10

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )
    scenarios_df = scenarios["scenarios_df"].head(max_scenarios).copy()

    support_set_config = BiddingBlocksTighteningOptimizer.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case_bidding_blocks_base",
    )

    optimizer = BiddingBlocksTighteningOptimizer(
        P_init=None,
        num_time_steps=int(scenarios_df.iloc[0]["time_steps"]),
        reference_case=case,
        support_set_config=support_set_config,
        eta=0.0,
        empirical_scenario=scenarios_df,
    )
    return optimizer


def main() -> None:
    output_path = Path("results/dro_poa_bidding_blocks_alpha_bounds.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = build_optimizer()

    start = time.perf_counter()
    alpha_report = optimizer.compute_nn_certified_bid_bounds(
        solver_name="gurobi",
        time_limit=200,
        tee=False,
    )
    elapsed = time.perf_counter() - start

    payload = {
        "metadata": {
            "description": (
                "Exact DRO alpha bounds from scenario-indexed support-set optimization."
            ),
            "reference_case": optimizer.reference_case,
            "num_time_steps": optimizer.num_time_steps,
            "num_empirical_scenarios": optimizer.num_empirical_scenarios,
            "scenario_ids": list(optimizer.empirical_scenario_ids),
            "regime": optimizer.empirical_regime,
            "num_optimization_programs": alpha_report["num_optimization_programs"],
        },
        "alpha_bounds": alpha_report["alpha_bounds"],
        "alpha_optimization_results": alpha_report["optimization_results"],
        # Empty placeholders keep this file compatible with load_tightening_report().
        "fixed_binaries": {},
        "slack_bounds": {},
        "tight_big_m": {},
    }
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)

    alpha_values = list(alpha_report["alpha_bounds"].values())
    lower_values = [float(item["lower"]) for item in alpha_values]
    upper_values = [float(item["upper"]) for item in alpha_values]

    print("\nAlpha-bound computation complete")
    print(f"  Report: {output_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Alpha entries: {len(alpha_values)}")
    print(f"  Min alpha lower bound: {min(lower_values):.6g}")
    print(f"  Max alpha upper bound: {max(upper_values):.6g}")


if __name__ == "__main__":
    main()
