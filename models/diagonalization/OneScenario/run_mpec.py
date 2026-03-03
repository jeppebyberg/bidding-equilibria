"""
Script to run MPEC model examples and tests
"""
import numpy as np
from .MPEC import MPECModel
from .economic_dispatch import EconomicDispatchModel
from config.base_case.scenarios.scenario_generator import ScenarioManager

if __name__ == "__main__":
    print("=== Testing OneScenario MPEC Model ===")
    
    # Load case data using ScenarioManager (same pattern as MultipleScenarios)
    manager = ScenarioManager("test_case_multiple_owners")
    base_case = manager.base_case
    players_config = manager.get_players_config()
    
    # Extract data for OneScenario
    demand = base_case['demand']
    pmax_list = base_case['pmax_list']
    pmin_list = base_case['pmin_list']
    cost_vector = base_case['cost_vector']
    num_generators = base_case['num_generators']
    generators = [gen.get('name', f'G{i+1}') for i, gen in enumerate(base_case.get('generators', []))]
    
    initial_bids = cost_vector.copy()
    
    print(f"Loaded {len(players_config)} players for {num_generators} generators")
    
    # Create MPEC model with new player configuration API (same as MultipleScenarios pattern)
    mpec_model = MPECModel(
        demand=demand,
        pmax_list=pmax_list,
        pmin_list=pmin_list,
        num_generators=num_generators,
        generators=generators,
        players_config=players_config,
        strategic_player_id=0,  # Start with Player 0
        bid_vector=initial_bids,
        cost_vector=cost_vector
    )
    
    print(f"\n=== Testing Strategic Player 0 ===")
    print(f"Player 0 controls generators: {mpec_model.strategic_generators}")
    
    # Ensure model is built by calling update_strategic_player (same pattern as MultipleScenarios)
    mpec_model.update_strategic_player(0, initial_bids, cost_vector)
    
    mpec_model.solve()

    print("\n=== MPEC Solution ===")
    print(f"Objective value: {-mpec_model.model.objective.expr()}")

    # Get model data for comparison
    model = EconomicDispatchModel()
    dispatch, clearing_price = model.economic_dispatch(
        num_generators=mpec_model.num_generators, 
        demand=mpec_model.demand, 
        Pmax=mpec_model.Pmax, 
        Pmin=mpec_model.Pmin, 
        bid_list=initial_bids
    )

    print(f"Strategic player: 0")
    print(f"Initial bids: {initial_bids}")

    alpha = []  
    for i in mpec_model.model.n_gen:
        if i in mpec_model.model.strategic_index:
            alpha_val = mpec_model.model.alpha[i].value
            if alpha_val is None:
                print(f"Warning: No solution for strategic player {i}, using initial bid")
                alpha_val = initial_bids[i]
            alpha.append(alpha_val)
        else:
            alpha.append(mpec_model.bid_vector[i])  # Non-strategic players have fixed bids
    print(f"Alpha:        {np.array(alpha).round(2)}")

    strategic_player_id = 1
    print(f"\n=== Testing Strategic Player {strategic_player_id} ===")

    # Use the new API for switching strategic players (same as MultipleScenarios)
    mpec_model.update_strategic_player(strategic_player_id, alpha, cost_vector)
    print(f"Player {strategic_player_id} controls generators: {mpec_model.strategic_generators}")
    mpec_model.solve()
    print(f"Strategic player: {strategic_player_id}")
    print("\n=== MPEC Solution ===")
    print(f"Objective value: {-mpec_model.model.objective.expr()}")

