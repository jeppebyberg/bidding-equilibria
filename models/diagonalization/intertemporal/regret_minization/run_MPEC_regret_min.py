"""
Test / validation script for the Regret-Minimization MPEC model.

Validates that the MPEC can:
  1. Build and solve for each strategic player
  2. Produce sensible bids, theta, and profits
  3. Update bids with optimal values

This script does NOT run the full best-response algorithm — for that,
run ``python -m drivers.best_response_algo_regret_min``.
"""
import sys
import os

# Add project root so imports work when running this file directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from config.intertemporal.scenarios.scenario_generator import ScenarioManager
from models.diagonalization.intertemporal.regret_minization.MPEC_regret_min import MPECModel


if __name__ == "__main__":
    from config.default_loader import load_test_case_config

    TEST_CASE  = "test_case1"

    test_config = load_test_case_config(TEST_CASE)

    demand_cfg = test_config["demand"]
    capacity_cfg = test_config["capacity"]

    # Scenario generation
    scenario_manager = ScenarioManager(TEST_CASE)
    players_config   = scenario_manager.get_players_config()

    # Demand
    demand_scenarios = scenario_manager.generate_demand_scenarios(
        demand_cfg["type"],
        num_scenarios=demand_cfg["num_scenarios"],
        min_factor=demand_cfg["min_factor"],
        max_factor=demand_cfg["max_factor"],
    )

    # Capacity
    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        capacity_cfg["type"],
        num_scenarios=capacity_cfg["num_scenarios"],
        min_factor=capacity_cfg["min_factor"],
        max_factor=capacity_cfg["max_factor"],
    )
    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )
    scenarios_df = scenarios["scenarios_df"]
    costs_df     = scenarios["costs_df"]
    ramps_df     = scenarios["ramps_df"]

    print(f"Scenarios shape: {scenarios_df.shape}")
    print(f"Costs: {costs_df.iloc[0].to_dict()}")

    # ── 2. Create MPEC model ───────────────────────────────────────
    mpec = MPECModel(
      reference_case=TEST_CASE,
      scenarios_df=scenarios_df,
      costs_df=costs_df,
      ramps_df=ramps_df,
      players_config=players_config,
    )

    # ── 3. Solve for each strategic player ─────────────────────────
    for player in players_config:
        pid = player['id']
        print(f"\n{'='*50}")
        print(f"  Setting strategic player to {pid}...")
        print(f"{'='*50}")

        mpec.build_model(pid)

        print("  Solving MPEC...")
        mpec.solve()
        print("  Solved successfully!")

        theta = mpec.get_optimal_theta()
        print(f"  Optimal theta: {theta}")

        profits = mpec.get_scenario_profits()
        print(f"  Scenario profits: {profits}")

        # Update scenarios_df with optimal bids for this player
        scenarios_df = mpec.update_bids_with_optimal_values(scenarios_df)

        stop = True

    # ── 4. Print full summary ──────────────────────────────────────
    mpec.print_players_summary()
