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

from config.base_case.scenarios.scenario_generator import ScenarioManager
from models.diagonalization.regret_minization.MPEC_regret_min import MPECModel


# ─── Configuration ─────────────────────────────────────────────────
BASE_CASE   = "test_case"
NUM_DEMAND  = 2
DEMAND_MIN  = 0.9
DEMAND_MAX  = 1.1


if __name__ == "__main__":

    # ── 1. Scenario generation ─────────────────────────────────────
    mgr = ScenarioManager(BASE_CASE)
    players_config = mgr.players_config

    demand_scenarios   = mgr.generate_demand_scenarios("linear",
                                                        num_scenarios=NUM_DEMAND,
                                                        min_factor=DEMAND_MIN,
                                                        max_factor=DEMAND_MAX)
    capacity_scenarios = mgr.generate_capacity_scenarios("linear", num_scenarios=1)
    scenario_set       = mgr.create_scenario_set(demand_scenarios=demand_scenarios,
                                                  capacity_scenarios=capacity_scenarios)
    scenarios_df = scenario_set['scenarios_df']
    costs_df     = scenario_set['costs_df']

    print(f"Scenarios shape: {scenarios_df.shape}")
    print(f"Costs: {costs_df.iloc[0].to_dict()}")

    # ── 2. Create MPEC model ───────────────────────────────────────
    mpec = MPECModel(scenarios_df, costs_df, players_config)

    # ── 3. Solve for each strategic player ─────────────────────────
    for player in players_config:
        pid = player['id']
        print(f"\n{'='*50}")
        print(f"  Setting strategic player to {pid}...")
        print(f"{'='*50}")

        mpec.update_strategic_player(pid)

        print("  Solving MPEC...")
        mpec.solve()
        print("  Solved successfully!")

        theta = mpec.get_optimal_theta()
        print(f"  Optimal theta: {theta}")

        profits = mpec.get_scenario_profits()
        print(f"  Scenario profits: {profits}")

        # Update scenarios_df with optimal bids for this player
        scenarios_df = mpec.update_bids_with_optimal_values(scenarios_df)

    # ── 4. Print full summary ──────────────────────────────────────
    mpec.print_players_summary()
