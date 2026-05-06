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
    from models.synthetic_data_generation.MPEC_MS import MPECModel
    from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
    from config.intertemporal.scenarios.scenario_generator_2 import ScenarioManagerV2
    from config.default_loader import load_test_case_config

    import time

    TEST_CASE  = "test_case1"

    # Scenario generation
    scenario_manager = ScenarioManagerV2(TEST_CASE)
    players_config   = scenario_manager.get_players_config()

    scenarios = scenario_manager.create_scenario_set_from_regimes(regime_set="policy_training")

    print(scenarios['description_text'])

    scenarios_df = scenarios['scenarios_df']
    costs_df = scenarios['costs_df']
    ramps_df = scenarios['ramps_df']

    # Generator names from the DataFrame columns
    generator_names = [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    print(f"Costs values: {costs_df.iloc[0].to_dict()}")

    # Solve ED once and use first time-step dispatch as initial condition for MPEC ramps.

    P_init = compute_p_init_from_ed(scenarios_df, costs_df, ramps_df)

    mpec_model = MPECModel(
        scenarios_df,
        costs_df,
        ramps_df,
        players_config,
        p_init=P_init,
    )

    # Build model for strategic player 0
    print("Building model for strategic player 0...")
    mpec_model.build_model(0)
    
    # Try to solve

    start = time.perf_counter()
    print("\nAttempting to solve MPEC model...")
    mpec_model.solve(tee = False, parallel=True, max_workers=8)
    print("MPEC model solved successfully!")
    end = time.perf_counter()
    print(f"MPEC solve time: {end - start:.2f} seconds")
    

    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    # Build model for strategic player 1
    print("Building model for strategic player 1...")
    mpec_model.build_model(1)

    print("\nAttempting to solve MPEC model...")
    mpec_model.solve(parallel=True, max_workers=8)
    print("MPEC model solved successfully!")

    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    stop = True