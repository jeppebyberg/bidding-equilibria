"""
Test script for MPEC model with multiple scenarios
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from models.diagonalization.MultipleScenarios.MPEC_MS import MPECModel
import config.base_case as config

if __name__ == "__main__":
     # Load base case data (only for costs/bids)
    # num_generators, _, _, bid_list, _, generators, players = config.load_setup_data("test_case_multiple_owners")
    
    # Generate multiple demand scenarios using ScenarioManager from test_case_multiple_owners
    scenario_manager = config.ScenarioManager("test_case_multiple_owners")
    
    # Extract players configuration for MPEC model
    players_config = scenario_manager.players_config

    # Generate desired scenarios
    demand_linear = scenario_manager.generate_demand_scenarios("linear", num_scenarios=10, min_factor=0.8, max_factor=1.2)
    capacity_linear = scenario_manager.generate_capacity_scenarios("linear", num_scenarios=3, min_factor=0.7, max_factor=1.0)

    # Create scenario set with separate DataFrames (now default behavior)
    print("=== Creating Scenario Set ===")
    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_linear,
        capacity_scenarios=capacity_linear
    )
    
    print(scenarios['description_text'])
    
    # Extract the separate DataFrames
    scenarios_df = scenarios['scenarios_df']
    costs_df = scenarios['costs_df']

    print(f"Costs values: {costs_df.iloc[0].to_dict()}")

    mpec_model = MPECModel(scenarios_df, costs_df, players_config)
    
    # Set strategic player (this also builds the model)
    print("Setting strategic player to 0...")
    mpec_model.update_strategic_player(0)
    
    # Try to solve
    print("\nAttempting to solve MPEC model...")
    mpec_model.solve()
    print("MPEC model solved successfully!")
    
    # Set strategic player (this also builds the model)
    print("Setting strategic player to 1...")
    mpec_model.update_strategic_player(1)

    print("\nAttempting to solve MPEC model...")
    mpec_model.solve()
    print("MPEC model solved successfully!")

    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)
    
    mpec_model.print_players_summary()

    stop = True