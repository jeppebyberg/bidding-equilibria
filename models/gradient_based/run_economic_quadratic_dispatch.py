"""
Script to run Economic Dispatch model for multiple scenarios using ScenarioManager
"""

from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel
from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from config.intertemporal.scenarios.scenario_generator_2 import ScenarioManagerV2
import numpy as np

import time

def compute_p_init_from_ed(scenarios_df, costs_df, ramps_df):
    """Solve ED and extract first time-step dispatch as [scenario][generator]."""
    # Use neutral initial conditions: 50% of scenario capacity for every generator, because all generators can ramp more than 50% of their capacity.
    initial_dispatch = []
    for _, row in scenarios_df.iterrows():
        initial_dispatch.append([
            0.5 * float(row[f"{gen}_cap"])
            for gen in generator_names
        ])

    ed_for_p_init = EconomicDispatchQuadraticModel(
        scenarios_df,
        costs_df,
        ramps_df,
        p_init=initial_dispatch,
        beta_coeff=0.001
    )
    ed_for_p_init.solve()
    dispatches = ed_for_p_init.get_dispatches()
    if dispatches is None:
        raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
    return [list(dispatches[s][0]) for s in range(len(dispatches))]

def compute_p_init_from_ed_linear(scenarios_df, costs_df, ramps_df):
    """Solve ED and extract first time-step dispatch as [scenario][generator]."""
    # Use neutral initial conditions: 50% of scenario capacity for every generator, because all generators can ramp more than 50% of their capacity.
    initial_dispatch = []
    for _, row in scenarios_df.iterrows():
        initial_dispatch.append([
            0.5 * float(row[f"{gen}_cap"])
            for gen in generator_names
        ])

    ed_for_p_init = EconomicDispatchModel(
        scenarios_df,
        costs_df,
        ramps_df,
        p_init=initial_dispatch,
    )
    ed_for_p_init.solve()
    dispatches = ed_for_p_init.get_dispatches()
    if dispatches is None:
        raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
    return [list(dispatches[s][0]) for s in range(len(dispatches))]

if __name__ == "__main__":

    manager = ScenarioManagerV2("test_case1")
    scenarios = manager.create_scenario_set_from_regimes(regime_set="policy_training")

    print(scenarios['description_text'])

    scenarios_df = scenarios['scenarios_df']
    costs_df = scenarios['costs_df']
    ramps_df = scenarios['ramps_df']

    generator_names = [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    P_init = compute_p_init_from_ed(scenarios_df, costs_df, ramps_df)
    P_init_linear = compute_p_init_from_ed_linear(scenarios_df, costs_df, ramps_df)

    # Run economic dispatch 
    print("\n" + "="*60)
    print("=== RUNNING ECONOMIC DISPATCH ===")
    print("="*60)   

    start = time.perf_counter()

    # Run economic dispatch using the new DataFrame constructor
    ed = EconomicDispatchQuadraticModel(scenarios_df, costs_df, ramps_df, p_init=P_init, beta_coeff=0.00001)
    ed.solve()
    
    all_dispatches = ed.get_dispatches()
    clearing_prices = ed.get_clearing_prices()
    all_profits = ed.get_generator_profits()
    
    stop = time.perf_counter()
    print(f"Time quadratic: {stop - start:.4f} sek")

    # Calculate and print average profits per generator
    print("\n" + "="*60)
    print("=== AVERAGE PROFITS PER PLAYER (ACROSS SCENARIOS) ===")
    print("="*60)

    # Quadratic dispatch average profits
    print("\n--- QUADRATIC ECONOMIC DISPATCH ---")
    num_generators = len(generator_names)
    for g in range(num_generators):
        avg_profit = np.mean([all_profits[s][g] for s in range(len(all_profits))])
        print(f"{generator_names[g]}: ${avg_profit:.2f}")

    # # Linear dispatch average profits
    # start = time.perf_counter()

    # # Run linear economic dispatch for comparison
    # ed_linear = EconomicDispatchModel(scenarios_df, costs_df, ramps_df, p_init=P_init_linear)
    # ed_linear.solve()

    # all_dispatches_linear = ed_linear.get_dispatches()
    # clearing_prices_linear = ed_linear.get_clearing_prices()
    # all_profits_linear = ed_linear.get_generator_profits()

    # stop = time.perf_counter()
    # print(f"Time linear: {stop - start:.4f} sek")

    # print("\n--- LINEAR ECONOMIC DISPATCH ---")
    # for g in range(num_generators):
    #     avg_profit = np.mean([all_profits_linear[s][g] for s in range(len(all_profits_linear))])
    #     print(f"{generator_names[g]}: ${avg_profit:.2f}")

    # stop = True