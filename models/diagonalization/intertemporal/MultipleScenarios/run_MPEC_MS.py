"""
Test script for MPEC model with multiple scenarios
"""

if __name__ == "__main__":
    from models.diagonalization.intertemporal.MultipleScenarios.MPEC_MS import MPECModel
    from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
    from config.intertemporal.scenarios.scenario_generator import ScenarioManager
    from config.default_loader import load_test_case_config
    from models.diagonalization.features.feature_setup import FeatureBuilder, DEFAULT_FEATURES

    TEST_CASE  = "test_case1"

    test_config = load_test_case_config(TEST_CASE)

    demand_cfg = test_config["demand"]
    capacity_cfg = test_config["capacity"]

    # Scenario generation
    scenario_manager = ScenarioManager(TEST_CASE)
    players_config   = scenario_manager.get_players_config()

    # Demand
    demand_scenarios = scenario_manager.generate_demand_scenarios(
        demand_cfg["type"],
        num_scenarios=demand_cfg["num_scenarios"],
        min_factor=demand_cfg["min_factor"],
        max_factor=demand_cfg["max_factor"],
    )

    # Capacity
    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        capacity_cfg["type"],
        num_scenarios=capacity_cfg["num_scenarios"],
        min_factor=capacity_cfg["min_factor"],
        max_factor=capacity_cfg["max_factor"],
    )

    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )
    scenarios_df = scenarios["scenarios_df"]
    costs_df     = scenarios["costs_df"]
    ramps_df     = scenarios["ramps_df"]

    # Generator names from the DataFrame columns
    generator_names = [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    print(f"Costs values: {costs_df.iloc[0].to_dict()}")

    # Solve ED once and use first time-step dispatch as initial condition for MPEC ramps.
    print("\n=== Solving Economic Dispatch For P_init ===")
    ed_model = EconomicDispatchModel(scenarios_df, costs_df, ramps_df)
    ed_model.solve()
    dispatches = ed_model.get_dispatches()
    if dispatches is None:
        raise RuntimeError("Economic dispatch did not return dispatch values. Cannot build P_init.")

    p_init = [list(dispatches[s][0]) for s in range(len(dispatches))]
    print(f"P_init built with shape: ({len(p_init)}, {len(p_init[0]) if p_init else 0})")

    feature_builder = FeatureBuilder(TEST_CASE, DEFAULT_FEATURES)
    feature_matrix_by_player = MPECModel.precompute_feature_matrix_by_player(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        feature_builder=feature_builder,
        p_init=p_init,
    )

    mpec_model = MPECModel(
        scenarios_df,
        costs_df,
        ramps_df,
        players_config,
        p_init=p_init,
        feature_builder=feature_builder,
    )
    
    # Set strategic player (this also builds the model)
    print("Setting strategic player to 0...")
    mpec_model.update_strategic_player(0)
    
    # Try to solve
    print("\nAttempting to solve MPEC model...")
    mpec_model.solve()
    print("MPEC model solved successfully!")
    
    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

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