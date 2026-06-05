from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from models.neural_network.features import NeuralNetworkFeatureBuilder
from models.neural_network.training.trainer import (
    BiddingPolicyTrainingConfig,
    train_generator_policy,
)

from driver_tmp.core.block0_core import ProjectConfig
from driver_tmp.core.block1_core import write_json


def filter_nn_policy_generators_by_activity(
    heuristic_results_path: Path,
    candidate_generators: list[str],
    min_label_changes: int,
) -> tuple[list[str], dict[str, int]]:
    with heuristic_results_path.open("r", encoding="utf-8") as fh:
        results: dict[str, Any] = json.load(fh)

    block_to_physical: dict[str, str] = results.get("block_to_physical", {})
    history: list[dict[str, Any]] = results.get("history", [])

    changes: dict[str, int] = {gen: 0 for gen in candidate_generators}
    for entry in history:
        if not entry.get("accepted"):
            continue
        block_name = str(entry.get("block_name", ""))
        physical = block_to_physical.get(block_name, "")
        if physical in changes:
            changes[physical] += 1

    filtered = [
        gen for gen in candidate_generators
        if changes.get(gen, 0) > min_label_changes
    ]
    return filtered, changes


def discover_trained_policy_generators(model_dir: Path) -> list[str]:
    return sorted(p.stem.replace("_policy", "") for p in model_dir.glob("*_policy.pt"))


def build_features(config: ProjectConfig, scenarios: dict[str, Any]) -> dict[str, Path]:
    start = time.perf_counter()
    builder = NeuralNetworkFeatureBuilder(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        results_path=config.heuristic_results_path,
        feature_columns=config.nn_feature_columns,
    )
    raw_paths = builder.save_datasets(
        output_dir=config.raw_feature_dir,
        normalize=False,
    )
    normalized_paths = builder.save_datasets(
        output_dir=config.normalized_feature_dir,
        normalize=True,
        per_generator_normalization=config.per_generator_normalization,
        save_stats=True,
    )
    elapsed = time.perf_counter() - start
    print("\nBuilt NN feature datasets")
    for generator_name in sorted(raw_paths):
        print(
            f"  {generator_name}: raw={raw_paths[generator_name]}, "
            f"normalized={normalized_paths[generator_name]}"
        )
    print(f"Feature-building runtime: {elapsed:.2f} seconds")
    return normalized_paths


def find_generator_feature_files(feature_dir: Path) -> list[Path]:
    if not feature_dir.exists():
        raise ValueError(f"Feature directory does not exist: {feature_dir}")
    return sorted(
        path
        for path in feature_dir.glob("*_features_normalized.csv")
        if path.is_file()
    )


def train_policies(config: ProjectConfig) -> Path:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.training_result_dir.mkdir(parents=True, exist_ok=True)

    training_config = BiddingPolicyTrainingConfig(
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
        patience=config.patience,
        min_delta=config.min_delta,
        device=config.device,
        final_activation=config.nn_final_activation,
        use_lr_scheduler=config.use_lr_scheduler,
        lr_scheduler_factor=config.lr_scheduler_factor,
        lr_scheduler_patience=config.lr_scheduler_patience,
        lr_scheduler_min_lr=config.lr_scheduler_min_lr,
    )

    all_csv_paths = find_generator_feature_files(config.normalized_feature_dir)
    if config.nn_policy_generators:
        allowed = set(config.nn_policy_generators)
        csv_paths = [
            p for p in all_csv_paths
            if p.name.replace("_features_normalized.csv", "") in allowed
        ]
    else:
        csv_paths = all_csv_paths
    if not csv_paths:
        raise ValueError(
            f"No normalized generator feature CSVs found in {config.normalized_feature_dir}"
        )

    start = time.perf_counter()
    summary_entries = []
    for csv_path in csv_paths:
        result = train_generator_policy(
            csv_path=csv_path,
            model_dir=config.model_dir,
            result_dir=config.training_result_dir,
            config=training_config,
        )
        policy_data = result["policy_data"]
        history = result["history"]
        summary_entries.append(result["summary"])
        print(
            f"{policy_data.generator_name}: rows={policy_data.num_rows}, "
            f"features={policy_data.input_dim}, targets={policy_data.output_dim}, "
            f"best_val_loss={history['best_val_loss']:.8g}, "
            f"final_test_loss={history['final_test_loss']:.8g}, model={result['model_path']}"
        )

    summary_path = config.training_result_dir / "training_summary.json"
    write_json(summary_path, summary_entries)
    elapsed = time.perf_counter() - start
    print(f"\nSaved training summary to {summary_path}")
    print(f"NN training runtime: {elapsed:.2f} seconds")

    if getattr(config, "plot_results_along_the_way", False):
        plot_training_results(config)

    return summary_path


def plot_training_results(config: ProjectConfig) -> Path:
    from models.neural_network.training.visualize_training import (
        main as visualize_training_main,
    )

    figures_dir = config.figures_dir or Path("results") / config.case / "figures"
    plot_dir = Path(figures_dir) / "training_results" / "plots"
    visualize_training_main(
        model_dir=config.model_dir,
        result_dir=config.training_result_dir,
        plot_dir=plot_dir,
    )
    print(f"Saved training plots to {plot_dir}")
    return plot_dir

