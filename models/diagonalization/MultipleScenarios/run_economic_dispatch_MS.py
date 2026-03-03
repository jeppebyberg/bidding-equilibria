"""
Script to run Economic Dispatch model for multiple scenarios using ScenarioManager
"""
import sys
import os
# Add project root to Python path for debugging
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from models.diagonalization.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
import config.base_case as config
import numpy as np

if __name__ == "__main__":
    # Load base case data (only for costs/bids)
    num_generators, _, _, bid_list, _, generators = config.load_setup_data("test_case1")
    
    # Generate multiple demand scenarios using ScenarioManager from test_case1
    scenario_manager = config.ScenarioManager("test_case")
    
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
    
    print(f"\nDataFrames created:")
    print(f"Scenarios DataFrame columns: {list(scenarios_df.columns)}")
    print(f"Costs values: {costs_df.iloc[0].to_dict()}")

    # Run economic dispatch 
    print("\n" + "="*60)
    print("=== RUNNING ECONOMIC DISPATCH ===")
    print("="*60)
    
    # Extract basic info for display
    demand_scenarios_list = scenarios_df['demand'].tolist()
    # Auto-detect generator capacity columns
    capacity_columns = [col for col in scenarios_df.columns if col.endswith('_cap')]
    generator_names = [col.replace('_cap', '') for col in capacity_columns]

    print(f"Demand range: {min(demand_scenarios_list):.1f} - {max(demand_scenarios_list):.1f} MW")
    
    # Show costs from the separate costs DataFrame
    print("Generator costs (from costs_df):")
    for gen_name in generator_names:
        cost_col = f"{gen_name}_cost"
        if cost_col in costs_df.columns:
            cost = costs_df[cost_col].iloc[0]
            print(f"  {gen_name}: ${cost:.2f}/MWh")

    # Run economic dispatch using the separate DataFrames method
    model = EconomicDispatchModel()
    all_dispatches, clearing_prices, all_profits = model.economic_dispatch_from_dataframe(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        pmin_default=0.0
    )
    
    # Summary statistics
    print("=== Summary Statistics ===\n")
    avg_price = np.mean(clearing_prices)
    min_price = np.min(clearing_prices)
    max_price = np.max(clearing_prices)
    
    print(f"Average clearing price: ${avg_price:.2f}/MWh")
    print(f"Price range: ${min_price:.2f} - ${max_price:.2f}/MWh")
