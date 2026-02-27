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

from models.diagonalization.MultipleScenarios.economic_dispatch import EconomicDispatchModel
import config.base_case as config
import numpy as np

if __name__ == "__main__":
    # Load base case data (only for costs/bids)
    num_generators, _, _, bid_list, _, generators = config.load_setup_data("test_case1")
    
    # Generate multiple demand scenarios using ScenarioManager
    scenario_manager = config.ScenarioManager("test_case1")
    # Initialize with base case reference
    
    demand_linear = scenario_manager.generate_demand_scenarios("linear", num_scenarios=10, min_factor=0.8, max_factor=1.2)
    capacity_linear = scenario_manager.generate_capacity_scenarios("linear", num_scenarios=3, min_factor=0.7, max_factor=1.0)

    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_linear,
        capacity_scenarios=capacity_linear
    )
    
    print(scenarios['description_text'])
    
    scenarios_df = scenarios['scenarios_table']

    # Extract basic info for display
    demand_scenarios = scenarios_df['demand'].tolist()
    generator_columns = ['G1', 'G2', 'G3', 'W4', 'W5', 'W6', 'W7']

    print(f"=== Economic Dispatch for Multiple Scenarios ===\n")
    print(f"Number of generators: {num_generators}")
    print(f"Number of scenarios: {len(scenarios_df)}")
    print(f"Demand range: {min(demand_scenarios):.1f} - {max(demand_scenarios):.1f} MW")
    print(f"Generator bids: {[f'${b:.2f}' for b in bid_list]}")
    print()
    
    # Run economic dispatch using the generic DataFrame method
    model = EconomicDispatchModel()
    all_dispatches, clearing_prices = model.economic_dispatch_from_dataframe(
        scenarios_df=scenarios_df,
        bid_list=bid_list,
        demand_col='demand',
        generator_cols=generator_columns,
        pmin_default=0.0
    )
    
    # Display results for each scenario
    print("=== Results by Scenario ===\n")
    for i, (dispatch, price) in enumerate(zip(all_dispatches, clearing_prices)):
        scenario_row = scenarios_df.iloc[i]
        demand = scenario_row['demand']
        pmax_scenario = [scenario_row[col] for col in generator_columns]
        pmin_scenario = [0.0] * num_generators  # Using default Pmin = 0
        total_dispatch = sum(dispatch)
        print(f"Scenario {i+1}: Demand = {demand:.1f} MW")
        print(f"  Clearing Price: ${price:.2f}/MWh")
        print(f"  Dispatch: {[f'{d:.1f}' for d in dispatch]} MW")
        print(f"  Pmax: {[f'{p:.1f}' for p in pmax_scenario]} MW")
        print(f"  Pmin: {[f'{p:.1f}' for p in pmin_scenario]} MW")
        print(f"  Total Dispatch: {total_dispatch:.1f} MW")
        print(f"  Supply-Demand Balance: {abs(total_dispatch - demand):.3f} MW\n")
    
    # Summary statistics
    print("=== Summary Statistics ===\n")
    avg_price = np.mean(clearing_prices)
    min_price = np.min(clearing_prices)
    max_price = np.max(clearing_prices)
    price_volatility = np.std(clearing_prices)
    
    print(f"Average clearing price: ${avg_price:.2f}/MWh")
    print(f"Price range: ${min_price:.2f} - ${max_price:.2f}/MWh")
    print(f"Price volatility (std): ${price_volatility:.2f}/MWh")
    
    # Generator utilization across scenarios
    print(f"\n=== Generator Utilization Across Scenarios ===\n")
    for gen_id, gen_col in enumerate(generator_columns):
        gen_dispatches = [dispatch[gen_id] for dispatch in all_dispatches]
        avg_dispatch = np.mean(gen_dispatches)
        max_dispatch = np.max(gen_dispatches)
        
        # Get average Pmax for this generator across scenarios
        gen_pmax_values = scenarios_df[gen_col].tolist()
        avg_pmax = np.mean(gen_pmax_values)
        utilization = (avg_dispatch / avg_pmax) * 100 if avg_pmax > 0 else 0
        
        print(f"Generator {gen_col}: Avg = {avg_dispatch:.1f} MW, "
              f"Max = {max_dispatch:.1f} MW, "
              f"Avg Pmax = {avg_pmax:.1f} MW, "
              f"Utilization = {utilization:.1f}%, "
              f"Bid = ${bid_list[gen_id]:.2f}/MWh")