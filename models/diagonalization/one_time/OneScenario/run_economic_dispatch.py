"""
Script to run Economic Dispatch model examples and tests
"""
from .economic_dispatch import EconomicDispatchModel
from config.base_case.scenarios.scenario_generator import ScenarioManager

if __name__ == "__main__":
    # Use ScenarioManager to generate a single scenario (same pattern as run_mpec.py)
    manager = ScenarioManager("test_case")

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

    ed = EconomicDispatchModel(scenarios_df, costs_df)
    ed.solve()

    print("Optimal Dispatch:", ed.get_dispatch())
    print("Market Clearing Price:", ed.get_clearing_price())
    print("Generator Profits:", [round(p, 2) for p in ed.get_generator_profits()])