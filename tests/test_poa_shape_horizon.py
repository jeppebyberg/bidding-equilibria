import numpy as np

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_optimization import PoAOptimization


def test_poa_shape_profiles_are_built_at_requested_horizon() -> None:
	horizon = 8
	manager = ScenarioManager(base_case_reference="base_test_case")
	manager.base_case["time_steps"] = horizon
	scenario_set = manager.create_scenario_set_from_ambiguity_set(
		ambiguity_config_path="config/ambiguity_set_config.yaml",
		ambiguity_set="base_test_case",
		n_scenarios=1,
		seed=1,
	)
	ambiguity_config = PoAOptimization.load_ambiguity_set(
		"config/ambiguity_set_config.yaml",
		"base_test_case",
	)

	optimizer = PoAOptimization(
		scenarios_df=scenario_set["scenarios_df"],
		costs_df=scenario_set["costs_df"],
		ramps_df=scenario_set["ramps_df"],
		num_time_steps=horizon,
		ambiguity_set_config=ambiguity_config,
		reference_case="base_test_case",
	)

	reference_manager = ScenarioManager(base_case_reference="base_test_case")
	expected_demand_shape = reference_manager._build_demand_shape(horizon)
	expected_wind_shape = reference_manager._build_wind_shape(
		horizon,
		ambiguity_config["wind"]["tau_fixed"],
	)
	truncated_demand_shape = reference_manager._build_demand_shape(24)[:horizon]
	truncated_wind_shape = reference_manager._build_wind_shape(
		24,
		ambiguity_config["wind"]["tau_fixed"],
	)[:horizon]

	assert np.allclose(optimizer.demand_shape, expected_demand_shape)
	assert np.allclose(optimizer.wind_shape, expected_wind_shape)
	assert not np.allclose(optimizer.demand_shape, truncated_demand_shape)
	assert not np.allclose(optimizer.wind_shape, truncated_wind_shape)
