"""
Script to run MPEC model examples and tests
"""
import numpy as np
from .MPEC import MPECModel
from .economic_dispatch import EconomicDispatchModel
import config.base_case as config

if __name__ == "__main__":
    print("=== Testing MPEC Model ===")
    
    # Load case data using config
    num_generators, pmax_list, pmin_list, cost_vector, demand, generators = config.load_setup_data("test_case")
    initial_bids = cost_vector.copy()
    
    # Create MPEC model directly with case data
    mpec_model = MPECModel(
        demand=demand,
        pmax_list=pmax_list,
        pmin_list=pmin_list,
        num_generators=num_generators,
        generators=generators,
        strategic_player=0,
        bid_vector=initial_bids,
        cost_vector=cost_vector
    )
    
    print(f"\n=== Testing Strategic Player 0 ===")
    
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

    strategic_player = 1
    print(f"\n=== Testing Strategic Player {strategic_player} ===")

    mpec_model.update_strategic_player(strategic_player=strategic_player, bid_vector=alpha, cost_vector=cost_vector)
    mpec_model.solve()

    print(f"Strategic player: {strategic_player}")
    print("\n=== MPEC Solution ===")
    print(f"Objective value: {-mpec_model.model.objective.expr()}")

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

    print(f"\nProfit of each agent")
    for i in mpec_model.model.n_gen:
        profit = mpec_model.model.lambda_var.value * mpec_model.model.P[i].value - cost_vector[i] * mpec_model.model.P[i].value
        profit_ED = clearing_price * dispatch[i] - cost_vector[i] * dispatch[i]
        print(f"Profit agent {i}: {profit} (ED: {profit_ED})")