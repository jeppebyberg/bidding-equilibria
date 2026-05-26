from config.scenarios.scenario_generator import ScenarioManager


def test_regime_and_ambiguity_scenario_generation_smoke() -> None:
	manager = ScenarioManager(base_case_reference="test_case_bidding_blocks")

	fixed = manager.create_scenario_set_from_regimes(
		regime_config_path="regime_definitions.yaml",
		regime_set="PoA_analysis",
		seed=42,
	)
	assert not fixed["scenarios_df"].empty
	assert "regime_config" in fixed

	n_scenarios = 5
	ambiguity = manager.create_scenario_set_from_ambiguity_set(
		ambiguity_config_path="ambiguity_set_config.yaml",
		ambiguity_set="test_case_bidding_blocks_base",
		n_scenarios=n_scenarios,
		seed=42,
	)

	scenarios_df = ambiguity["scenarios_df"]
	ambiguity_config = ambiguity["ambiguity_config"]
	assert len(scenarios_df) == n_scenarios

	required_columns = {
		"scenario_id",
		"regime",
		"scenario_source",
		"ambiguity_set",
		"demand_profile",
		"mu_D",
		"rho_D",
		"sigma_D",
		"mu_W",
		"rho_W",
		"sigma_W",
		"peak_W",
	}
	assert required_columns.issubset(scenarios_df.columns)

	assert scenarios_df["scenario_source"].eq("ambiguity_set").all()
	assert scenarios_df["ambiguity_set"].eq("test_case_bidding_blocks_base").all()
	assert scenarios_df["mu_D"].between(
		ambiguity_config["demand_mu_min"],
		ambiguity_config["demand_mu_max"],
	).all()
	assert scenarios_df["sigma_D"].between(
		ambiguity_config["demand_sigma_min"],
		ambiguity_config["demand_sigma_max"],
	).all()
	assert scenarios_df["rho_D"].eq(ambiguity_config["rho_D"]).all()
	assert scenarios_df["mu_W"].between(
		ambiguity_config["wind_mu_min"],
		ambiguity_config["wind_mu_max"],
	).all()
	assert scenarios_df["sigma_W"].between(
		ambiguity_config["wind_sigma_min"],
		ambiguity_config["wind_sigma_max"],
	).all()
	assert scenarios_df["rho_W"].eq(ambiguity_config["rho_W"]).all()
	assert scenarios_df["peak_W"].eq(ambiguity_config["peak_W"]).all()

	horizon = int(scenarios_df["time_steps"].iloc[0])
	for demand_profile in scenarios_df["demand_profile"]:
		assert len(demand_profile) == horizon

	wind_profile_columns = [column for column in scenarios_df.columns if column.endswith("_profile") and column != "demand_profile"]
	assert wind_profile_columns
	for column in wind_profile_columns:
		for wind_profile in scenarios_df[column]:
			assert len(wind_profile) == horizon
