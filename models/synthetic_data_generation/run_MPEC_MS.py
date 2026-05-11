"""
Test script for MPEC model with multiple scenarios
"""
def compute_p_init_from_ed(scenarios_df, costs_df, ramps_df, generator_names):
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
    from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel
    from config.scenarios.scenario_generator import ScenarioManager

    import time

    TEST_CASE  = "test_case_bidding_blocks"
    SOLVER_TEE = False

    # i7-10510U: 4 physical cores / 8 logical processors.
    # Use a balanced layout instead of letting several Gurobi solves each grab
    # all 8 logical threads. Good alternatives to benchmark are:
    #   "throughput": 6 workers x 1 Gurobi thread
    #   "balanced":   4 workers x 2 Gurobi threads
    #   "focused":    2 workers x 4 Gurobi threads
    PERFORMANCE_PRESET = "throughput"
    PERFORMANCE_PRESETS = {
        "throughput": {"max_workers": 6, "solver_threads": 1},
        "balanced": {"max_workers": 4, "solver_threads": 2},
        "focused": {"max_workers": 2, "solver_threads": 4},
    }
    MPEC_MAX_WORKERS = PERFORMANCE_PRESETS[PERFORMANCE_PRESET]["max_workers"]
    GUROBI_THREADS_PER_WORKER = PERFORMANCE_PRESETS[PERFORMANCE_PRESET]["solver_threads"]
    GUROBI_OPTIONS = {
        "Presolve": 2,
    }
    MPEC_CONFIG_OVERRIDES = {
        "big_m_method": "fbbt",
        "print_big_m_summary": False,
    }

    # Scenario generation
    scenario_manager = ScenarioManager(TEST_CASE)
    players_config   = scenario_manager.get_players_config()

    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set="policy_training", 
        seed = 1
    )

    print(scenarios["description_text"])

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    scenario_manager.plot_demand_profiles_by_regime(
		scenarios["scenarios_df"],
		title="Demand Profiles by Regime",
		show_regime_mean=True,
		alpha=0.22,
		save_path="models/synthetic_data_generation/plots/demand_profiles_by_regime.png",
        show = False,
	)

    scenario_manager.plot_wind_profiles_by_regime(
		scenarios["scenarios_df"],
		title="Wind Profiles by Regime",
		show_regime_mean=True,
		alpha=0.22,
		save_path="models/synthetic_data_generation/plots/wind_profiles_by_regime.png",
        show = False,
	)

    generator_names = [
        c.replace("_cap", "")
        for c in scenarios_df.columns
        if c.endswith("_cap")
    ]
	
    # Solve ED once and use first time-step dispatch as initial condition for MPEC ramps.
    P_init = compute_p_init_from_ed(scenarios_df, costs_df, ramps_df, generator_names)

    mpec_model = MPECModel(
        scenarios_df,
        costs_df,
        ramps_df,
        players_config,
        p_init=P_init,
        config_overrides=MPEC_CONFIG_OVERRIDES,
    )

    # Build model for strategic player 0
    print("Building model for strategic player 0...")
    mpec_model.build_model(3)
    
    # Try to solve

    start = time.perf_counter()
    print(
        "\nAttempting to solve MPEC model "
        f"({MPEC_MAX_WORKERS} scenario workers x {GUROBI_THREADS_PER_WORKER} Gurobi threads)..."
    )
    mpec_model.solve(
        tee=SOLVER_TEE,
        parallel=True,
        max_workers=MPEC_MAX_WORKERS,
        solver_threads=GUROBI_THREADS_PER_WORKER,
        solver_options=GUROBI_OPTIONS,
    )
    print("MPEC model solved successfully!")
    end = time.perf_counter()
    print(f"MPEC solve time: {end - start:.2f} seconds")
    
    stop = True

    # Update bid scenarios with optimal values
    print("\n=== Updating Bid Scenarios ===")
    # scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    # # Build model for strategic player 1
    # print("Building model for strategic player 1...")
    # mpec_model.build_model(1)

    # print("\nAttempting to solve MPEC model...")
    # mpec_model.solve(
    #     tee=SOLVER_TEE,
    #     parallel=True,
    #     max_workers=MPEC_MAX_WORKERS,
    #     solver_threads=GUROBI_THREADS_PER_WORKER,
    #     solver_options=GUROBI_OPTIONS,
    # )
    # print("MPEC model solved successfully!")


    # # Update bid scenarios with optimal values
    # print("\n=== Updating Bid Scenarios ===")
    # scenarios_df = mpec_model.update_bids_with_optimal_values(scenarios_df)

    # stop = True
