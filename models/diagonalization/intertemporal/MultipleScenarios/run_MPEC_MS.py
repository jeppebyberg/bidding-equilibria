"""
Test script for MPEC model with multiple scenarios
"""
def compute_p_init_from_ed(scenarios_df, costs_df, ramps_df):
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

    feature_builder = FeatureBuilder(TEST_CASE, DEFAULT_FEATURES)
    feature_matrix_by_player = feature_builder.build_intertemporal_feature_matrix_by_player_from_frames(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        generator_names=generator_names,
        players_config=players_config,
        fit_normalizer=True,
    )

    P_init = compute_p_init_from_ed(scenarios_df, costs_df, ramps_df)

    mpec_model = MPECModel(
        scenarios_df,
        costs_df,
        ramps_df,
        players_config,
        p_init=P_init,
        feature_matrix_by_player=feature_matrix_by_player,
        NN_nodes=4,  # Must be an even number to have equal positive and negative parts
    )

    # Build model for strategic player 0
    print("Building model for strategic player 0...")
    mpec_model.build_model(0)
    
    # Try to solve
    print("\nAttempting to solve MPEC model...")
    mpec_model.solve()
    print("MPEC model solved successfully!")
    
    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    # Build model for strategic player 1
    print("Building model for strategic player 1...")
    mpec_model.build_model(1)

    print("\nAttempting to solve MPEC model...")
    mpec_model.solve()
    print("MPEC model solved successfully!")

    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    stop = True