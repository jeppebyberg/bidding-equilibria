"""
Script to run Economic Dispatch model for multiple scenarios using ScenarioManager
"""

import time

from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel
from config.scenarios.scenario_generator import ScenarioManager
import numpy as np


def compute_p_init_from_unconstrained_initial_ed(scenarios_df, costs_df, ramps_df):
    """Solve ED without initial ramp constraints and use first-step physical dispatch."""
    ed_for_p_init = EconomicDispatchQuadraticModel(
        scenarios_df,
        costs_df,
        ramps_df,
        p_init=None,
        beta_coeff=0.001,
    )
    ed_for_p_init.solve()
    dispatches = ed_for_p_init.get_dispatches()
    if dispatches is None:
        raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
    return [list(dispatches[s][0]) for s in range(len(dispatches))]

if __name__ == "__main__":

    manager = ScenarioManager("test_case_bidding_blocks")
    scenarios = manager.create_scenario_set_from_regimes(regime_set="policy_training")

    print(scenarios['description_text'])

    scenarios_df = scenarios['scenarios_df']
    costs_df = scenarios['costs_df']
    ramps_df = scenarios['ramps_df']

    P_init = compute_p_init_from_unconstrained_initial_ed(scenarios_df, costs_df, ramps_df)

    # Run economic dispatch 
    print("\n" + "="*60)
    print("=== RUNNING ECONOMIC DISPATCH ===")
    print("="*60)   

    start = time.perf_counter()

    ed = EconomicDispatchQuadraticModel(scenarios_df, costs_df, ramps_df, p_init=P_init, beta_coeff=0.00001)
    ed.solve()
    
    physical_generator_names = ed.get_physical_generator_names()
    block_names = ed.get_block_names()
    block_to_physical = ed.get_block_to_physical_mapping()
    all_physical_dispatches = ed.get_dispatches()
    all_block_dispatches = ed.get_block_dispatches()
    clearing_prices = ed.get_clearing_prices()
    all_profits = ed.get_generator_profits()
    
    stop = time.perf_counter()
    print(f"Time quadratic: {stop - start:.4f} sek")

    print("\nPhysical generators:", physical_generator_names)
    print("Bidding blocks:", block_names)
    print("Block-to-physical mapping:", block_to_physical)

    # Calculate and print average profits per physical generator
    print("\n" + "="*60)
    print("=== AVERAGE PROFITS PER PHYSICAL GENERATOR (ACROSS SCENARIOS) ===")
    print("="*60)

    for g, generator_name in enumerate(physical_generator_names):
        avg_profit = np.mean([all_profits[s][g] for s in range(len(all_profits))])
        print(f"{generator_name}: ${avg_profit:.2f}")

    if all_physical_dispatches is not None and all_block_dispatches is not None and clearing_prices is not None:
        print("\nFirst scenario, first time step:")
        print("Physical dispatch:", dict(zip(physical_generator_names, all_physical_dispatches[0][0])))
        print("Block dispatch:", dict(zip(block_names, all_block_dispatches[0][0])))
        print(f"Clearing price: {clearing_prices[0][0]:.2f}")

    stop = True