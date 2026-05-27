from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.neural_network.features import NeuralNetworkFeatureBuilder


def main() -> None:
    case = "base_test_case"
    ambiguity_set = "base_test_case"
    num_scenarios = 500
    seed = 42
    results_path = "results/merit_order_best_response_results.json"
    raw_output_dir = "models/neural_network/features/generated/raw"
    normalized_output_dir = "models/neural_network/features/generated/normalized"

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_ambiguity_set(
        ambiguity_config_path=str(ScenarioManager.AMBIGUITY_CONFIG_PATH),
        ambiguity_set=ambiguity_set,
        n_scenarios=num_scenarios,
        seed=seed,
    )

    builder = NeuralNetworkFeatureBuilder(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        results_path=results_path,
    )
    raw_datasets = builder.build_all_generator_datasets(normalize=False)
    raw_paths = builder.save_datasets(output_dir=raw_output_dir, normalize=False)
    normalized_paths = builder.save_datasets(
        output_dir=normalized_output_dir,
        normalize=True,
        per_generator_normalization=True,
        save_stats=True,
    )

    for generator_name, dataframe in raw_datasets.items():
        target_columns = [
            column for column in dataframe.columns if column.startswith("target_bid_")
        ]
        print(
            f"{generator_name}: shape={dataframe.shape}, "
            f"targets={target_columns}, raw={raw_paths[generator_name]}, "
            f"normalized={normalized_paths[generator_name]}"
        )


if __name__ == "__main__":
    main()
