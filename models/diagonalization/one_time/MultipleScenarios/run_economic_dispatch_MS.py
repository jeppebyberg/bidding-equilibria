"""
Script to run Economic Dispatch model for multiple scenarios using ScenarioManager
"""

from models.diagonalization.one_time.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from config.base_case.scenarios.scenario_generator import ScenarioManager
import numpy as np

if __name__ == "__main__":
    # Generate multiple demand scenarios using ScenarioManager
    manager = ScenarioManager("test_case")
    
    demand_linear = manager.generate_demand_scenarios("linear", num_scenarios=10, min_factor=0.8, max_factor=1.2)
    capacity_linear = manager.generate_capacity_scenarios("linear", num_scenarios=3, min_factor=0.7, max_factor=1.0)

    # Create scenario set with separate DataFrames
    print("=== Creating Scenario Set ===")
    scenarios = manager.create_scenario_set(
        demand_scenarios=demand_linear,
        capacity_scenarios=capacity_linear
    )
    
    print(scenarios['description_text'])
    
    scenarios_df = scenarios['scenarios_df']
    costs_df = scenarios['costs_df']
    
    print(f"\nDataFrames created:")
    print(f"Scenarios DataFrame columns: {list(scenarios_df.columns)}")
    print(f"Costs values: {costs_df.iloc[0].to_dict()}")

    # Run economic dispatch 
    print("\n" + "="*60)
    print("=== RUNNING ECONOMIC DISPATCH ===")
    print("="*60)
    
    # Extract basic info for display
    demand_col = [col for col in scenarios_df.columns if any(kw in col.lower() for kw in ['demand', 'load'])][0]
    demand_list = scenarios_df[demand_col].tolist()
    capacity_columns = [col for col in scenarios_df.columns if col.endswith('_cap')]
    generator_names = [col.replace('_cap', '') for col in capacity_columns]

    print(f"Demand range: {min(demand_list):.1f} - {max(demand_list):.1f} MW")
    
    # Show costs from the costs DataFrame
    print("Generator costs (from costs_df):")
    for gen_name in generator_names:
        cost_col = f"{gen_name}_cost"
        if cost_col in costs_df.columns:
            cost = costs_df[cost_col].iloc[0]
            print(f"  {gen_name}: ${cost:.2f}/MWh")

    # Run economic dispatch using the new DataFrame constructor
    ed = EconomicDispatchModel(scenarios_df, costs_df)
    ed.solve()
    
    all_dispatches = ed.get_dispatches()
    clearing_prices = ed.get_clearing_prices()
    all_profits = ed.get_generator_profits()
    
    # Summary statistics
    print("=== Summary Statistics ===\n")
    avg_price = np.mean(clearing_prices)
    min_price = np.min(clearing_prices)
    max_price = np.max(clearing_prices)
    
    print(f"Average clearing price: ${avg_price:.2f}/MWh")
    print(f"Price range: ${min_price:.2f} - ${max_price:.2f}/MWh")
