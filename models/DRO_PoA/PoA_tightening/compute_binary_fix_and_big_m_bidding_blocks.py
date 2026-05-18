from pathlib import Path
import time

from config.scenarios.scenario_generator import ScenarioManager
from models.DRO_PoA.PoA_tightening.bidding_blocks_tightening import (
    BiddingBlocksTighteningOptimizer,
)


def build_optimizer() -> BiddingBlocksTighteningOptimizer:
    """
    Create the same optimizer configuration used for alpha-bound computation.

    The slack OBBT and dual maximization models must use the same support set,
    horizon, empirical scenarios, and dual-bound defaults as the model that
    produced the scenario-indexed alpha-bound JSON.
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

    return BiddingBlocksTighteningOptimizer(
        P_init=None,
        num_time_steps=int(scenarios_df.iloc[0]["time_steps"]),
        reference_case=case,
        support_set_config=support_set_config,
        eta=0.0,
        empirical_scenario=scenarios_df,
    )


def main() -> None:
    alpha_bounds_path = Path("results/dro_poa_bidding_blocks_alpha_bounds.json")
    output_path = Path("results/dro_poa_bidding_blocks_tightening_report.json")
    epsilon = 1e-6
    solver_name = "gurobi"
    time_limit = 200

    optimizer = build_optimizer()
    optimizer.load_tightening_report(alpha_bounds_path)

    start = time.perf_counter()
    slack_report = optimizer.run_slack_based_obbt(
        alpha_bounds=optimizer.alpha_bounds,
        epsilon=epsilon,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=False,
        stop_at_zero_slack=True,
        slack_stop_tol=epsilon,
    )
    big_m_report = optimizer.run_dual_big_m_tightening(
        alpha_bounds=optimizer.alpha_bounds,
        fixed_binaries=optimizer.fixed_binaries,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=False,
    )
    report_path = optimizer.save_tightening_report(output_path)
    elapsed = time.perf_counter() - start

    fixed_binaries = slack_report["fixed_binaries"]
    tight_big_m = big_m_report["tight_big_m"]
    tightened_values = [
        value["tight_big_m"]
        for component in tight_big_m.values()
        for value in component.values()
        if value["tight_big_m"] is not None
    ]

    print("\nBinary-fix and Big-M tightening complete")
    print(f"  Alpha bounds: {alpha_bounds_path}")
    print(f"  Report: {report_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Fixed complementarity binaries: {slack_report['num_fixed_binaries']}")
    print(f"  Tightened dual Big-M values: {len(tightened_values)}")
    if tightened_values:
        print(f"  Smallest tight Big-M: {min(tightened_values):.6g}")
        print(f"  Largest tight Big-M: {max(tightened_values):.6g}")

    print("\nFixed binaries by component")
    for component_name, entries in fixed_binaries.items():
        print(f"  {component_name}: {len(entries)}")


if __name__ == "__main__":
    main()
