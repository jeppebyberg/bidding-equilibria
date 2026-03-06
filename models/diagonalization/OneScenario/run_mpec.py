"""
Script to run MPEC model examples and tests
"""
import numpy as np
from models.diagonalization.OneScenario.MPEC import MPECModel
from models.diagonalization.OneScenario.economic_dispatch import EconomicDispatchModel
from config.base_case.scenarios.scenario_generator import ScenarioManager

if __name__ == "__main__":
    print("=== Testing OneScenario MPEC Model ===")
    
    # Load case data using ScenarioManager (same pattern as MultipleScenarios)
    manager = ScenarioManager("test_case")
    base_case = manager.base_case
    players_config = manager.get_players_config()
    
    # Generate a single scenario using ScenarioManager
    demand_scenarios = manager.generate_demand_scenarios(
        "linear", num_scenarios=1, min_factor=1.0, max_factor=1.0,
    )
    capacity_scenarios = manager.generate_capacity_scenarios(
        "linear", num_scenarios=1, min_factor=1.0, max_factor=1.0,
    )
    scenarios = manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )
    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    
    cost_vector = base_case['cost_vector']
    initial_bids = cost_vector.copy()
    
    # Create MPEC model with DataFrame API (same interface as MPEC_MS)
    mpec_model = MPECModel(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        players_config=players_config,
        strategic_player_id=0,  # Start with Player 0
    )
    
    print(f"\n=== Testing Strategic Player 0 ===")
    print(f"Player 0 controls generators: {mpec_model.strategic_generators}")
    
    # Ensure model is built by calling update_strategic_player (same pattern as MPEC_MS)
    mpec_model.update_strategic_player(strategic_player_id=0)
    
    mpec_model.solve()

    print("\n=== MPEC Solution ===")
    print(f"Objective value: {mpec_model.get_scenario_profits()[0]}")

    # Get optimal bids
    optimal_bids = mpec_model.get_optimal_bids()
    print(f"Initial bids: {initial_bids}")
    print(f"Optimal bids: {[round(b, 2) for b in optimal_bids]}")

    # Get model data for comparison
    ed = EconomicDispatchModel(scenarios_df, costs_df)
    ed.solve()
    dispatch = ed.get_dispatch()
    clearing_price = ed.get_clearing_price()
    print(f"\nED dispatch with initial bids: {[round(d, 2) for d in dispatch]}")
    print(f"ED clearing price: {clearing_price:.2f}")

    # Update bids for next player
    updated_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    strategic_player_id = 1
    print(f"\n=== Testing Strategic Player {strategic_player_id} ===")

    # Create new model with updated bids (same pattern as MPEC_MS usage)
    mpec_model_2 = MPECModel(
        scenarios_df=updated_df,
        costs_df=costs_df,
        players_config=players_config,
    )
    mpec_model_2.update_strategic_player(strategic_player_id=strategic_player_id)
    print(f"Player {strategic_player_id} controls generators: {mpec_model_2.strategic_generators}")
    mpec_model_2.solve()
    print("\n=== MPEC Solution ===")
    print(f"Objective value: {mpec_model_2.get_scenario_profits()[0]}")
    print(f"Optimal bids: {[round(b, 2) for b in mpec_model_2.get_optimal_bids()]}")

