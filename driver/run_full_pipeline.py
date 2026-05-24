from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.neural_network.features import NeuralNetworkFeatureBuilder
from models.neural_network.training.trainer import (
    BiddingPolicyTrainingConfig,
    train_generator_policy,
)
from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.PoA_tightening.compute_primal_big_m import (
    PrimalBigMComputer,
    ambiguity_set_summary,
    compute_primal_big_m_bounds,
    summarize_primal_big_m,
)
from models.PoA.PoA_tightening.tightening_main import (
    DEFAULT_TIGHTENING_OUTPUT_PATHS,
    PoATighteningMain,
)
from models.PoA.PoA_tightening.compute_alpha_bounds import AlphaBoundsComputer
from models.PoA.PoA_tightening.compute_dual_big_m import DualBigMComputer
from models.PoA.PoA_tightening.compute_relu_bounds import ReLUBoundsComputer
from models.PoA.PoA_tightening.compute_slack_binary_fix import SlackBinaryFixComputer
from models.synthetic_data_generation.merit_order_best_response import MeritOrderHeuristic


RUN_TIGHTENING = True

TIGHTENING_FLAGS = {
    "primal_big_m": True,
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
}

TIGHTENING_PREVIOUS_PATHS = {
    "primal_big_m": "results/poa_tightening/primal_big_m_report.json",
    "relu_bounds": "results/poa_tightening/relu_bounds_report.json",
    "alpha_bounds": "results/poa_tightening/alpha_bounds_report.json",
    "slack_binary_fix": "results/poa_tightening/slack_binary_fix_report.json",
    "dual_big_m": "results/poa_tightening/dual_big_m_report.json",
}

TIGHTENING_OUTPUT_PATHS = {
    "primal_big_m": "results/poa_tightening/primal_big_m_report.json",
    "relu_bounds": "results/poa_tightening/relu_bounds_report.json",
    "alpha_bounds": "results/poa_tightening/alpha_bounds_report.json",
    "slack_binary_fix": "results/poa_tightening/slack_binary_fix_report.json",
    "dual_big_m": "results/poa_tightening/dual_big_m_report.json",
    "final": "results/poa_tightening/final_tightening_report.json",
}

TIGHTENING_STAGE_ORDER = (
    "primal_big_m",
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
)

TIGHTENING_STAGE_LABELS = {
    "primal_big_m": "Primal Big-M",
    "relu_bounds": "NN ReLU bounds",
    "alpha_bounds": "Alpha bounds",
    "slack_binary_fix": "Slack binary fix",
    "dual_big_m": "Dual Big-M",
}


@dataclass
class FullPipelineConfig:
    # Case and random seeds.
    case: str = "test_case_bidding_blocks"
    synthetic_time_steps: int | None = 24
    poa_time_steps: int | None = None
    synthetic_regime_set: str = "policy_training_runtime"
    poa_regime_set: str = "PoA_analysis_runtime"
    source_synthetic_regime_set: str = "policy_training"
    source_poa_regime_set: str = "PoA_analysis"
    synthetic_seed: int = 1
    poa_seed: int = 1

    # Scenario counts. Set a regime name to the desired number of scenarios.
    # Use {} to keep the counts from config/regime_definitions.yaml.
    synthetic_scenarios_per_regime: dict[str, int] = field(
        default_factory=lambda: {
            "normal": 100,
            "high_demand": 100,
            "normal_peak_shift_wind": 100,
            "high_demand_peak_shift_wind": 100,
        }
    )
    poa_context_scenarios_per_regime: dict[str, int] = field(
        default_factory=lambda: {
            "normal": 2,
            "high_demand": 2,
            "normal_peak_shift_wind": 5,
            "high_demand_peak_shift_wind": 5,
        }
    )

    # Heuristic synthetic-label generation.
    bid_tolerance: float = 1e-2

    # Neural-network feature and training parameters.
    # These names must be supported by both NeuralNetworkFeatureBuilder and
    # PoAOptimizationBiddingBlocks._raw_nn_feature_expression.
    nn_feature_columns: list[str] = field(
        default_factory=lambda: [
            "demand",
            "total_generation_capacity",
            "residual_demand",
            "next_generation_capacity",
            "next_demand",
            "own_generation_capacity",
            "next_own_generation_capacity",
        ]
    )
    per_generator_normalization: bool = True
    hidden_layers: list[int] = field(default_factory=lambda: [7, 7])
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 500
    weight_decay: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    patience: int | None = 50
    min_delta: float = 1e-6
    device: str | None = None

    # PoA parameters. The uncertainty set is induced by ambiguity_set_config_name
    # in models/PoA/ambiguity_set_config.yaml. The generated PoA context
    # scenarios only provide model dimensions/static case data.
    horizon: int = 8
    ambiguity_set_config_path: str = "models/PoA/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "test_case_bidding_blocks_base"
    nn_policy_generators: list[int] = field(default_factory=lambda: [1, 2])
    solver_name: str = "gurobi"
    preprocessing_time_limit: int = 200
    poa_time_limit: int | None = 400
    epsilon: float = 1e-6
    # Parallelizes the independent tightening submodels inside the three PoA
    # preprocessing stages. Keep Gurobi threads low when workers > 1.
    poa_parallel_workers: int = 1
    poa_solver_threads_per_worker: int | None = None

    # Step toggles. Turn expensive stages off when reusing previous outputs.
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True
    run_tightening: bool = RUN_TIGHTENING
    tightening_flags: dict[str, bool] = field(
        default_factory=lambda: dict(TIGHTENING_FLAGS)
    )
    tightening_previous_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(TIGHTENING_PREVIOUS_PATHS)
    )
    tightening_output_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(TIGHTENING_OUTPUT_PATHS)
    )
    # Legacy per-stage toggles retained for direct calls to the compatibility
    # helper functions below. The main pipeline uses tightening_flags.
    run_poa_nn_relu_bounds: bool = True
    run_poa_alpha_bounds: bool = True
    run_poa_slack_binary_fix: bool = True
    run_poa_dual_big_m: bool = True
    run_poa_optimization: bool = True

    # Outputs.
    runtime_config_path: Path = Path("results/full_pipeline/runtime_regime_definitions.yaml")
    synthetic_scenario_dir: Path = Path("results/full_pipeline/synthetic_scenarios")
    poa_scenario_dir: Path = Path("results/full_pipeline/poa_scenarios")
    heuristic_results_path: Path = Path("results/merit_order_best_response_results.json")
    raw_feature_dir: Path = Path("models/neural_network/features/generated/raw")
    normalized_feature_dir: Path = Path("models/neural_network/features/generated/normalized")
    model_dir: Path = Path("models/neural_network/training/trained_models")
    training_result_dir: Path = Path("models/neural_network/training/training_results")
    poa_result_dir: Path = Path("results")

    @property
    def nn_normalization_stats_path(self) -> Path:
        return self.normalized_feature_dir / "min_max_stats.json"

    @property
    def alpha_bounds_path(self) -> Path:
        return Path(self.tightening_output_paths["alpha_bounds"])

    @property
    def nn_relu_bounds_path(self) -> Path:
        return Path(self.tightening_output_paths["relu_bounds"])

    @property
    def primal_big_m_path(self) -> Path:
        return Path(self.tightening_output_paths["primal_big_m"])

    @property
    def slack_report_path(self) -> Path:
        return Path(self.tightening_output_paths["slack_binary_fix"])

    @property
    def tightening_report_path(self) -> Path:
        return Path(self.tightening_output_paths["final"])

    @property
    def dual_big_m_path(self) -> Path:
        return Path(self.tightening_output_paths["dual_big_m"])

    @property
    def poa_results_path(self) -> Path:
        return self.poa_result_dir / (
            f"poa_optimization_T{self.horizon}.json"
        )


def main(config: FullPipelineConfig) -> None:
    print_pipeline_header(config)

    write_runtime_regime_config(config)

    synthetic_manager = ScenarioManager(config.case)
    synthetic_scenarios = load_or_generate_scenarios(
        config=config,
        manager=synthetic_manager,
        regime_set=config.synthetic_regime_set,
        seed=config.synthetic_seed,
        output_dir=config.synthetic_scenario_dir,
        should_generate=config.run_scenario_generation,
        time_steps=config.synthetic_time_steps,
    )

    if config.run_heuristic_labels:
        run_heuristic(config, synthetic_scenarios, synthetic_manager)

    if config.run_feature_building:
        build_features(config, synthetic_scenarios)

    if config.run_nn_training:
        train_policies(config)

    if any(
        [
            config.run_tightening,
            config.run_poa_optimization,
        ]
    ):
        load_or_generate_scenarios(
            config=config,
            manager=ScenarioManager(config.case),
            regime_set=config.poa_regime_set,
            seed=config.poa_seed,
            output_dir=config.poa_scenario_dir,
            should_generate=True,
            time_steps=config.poa_time_steps,
        )

    if config.run_tightening:
        run_tightening_pipeline(config)
    if config.run_poa_optimization:
        run_final_poa(config)

    print("\nFull pipeline complete.")


def print_pipeline_header(config: FullPipelineConfig) -> None:
    print(
        "\nFull pipeline configuration\n"
        f"  case={config.case}\n"
        f"  synthetic_time_steps={config.synthetic_time_steps or 'case default'}\n"
        f"  poa_time_steps={config.poa_time_steps or 'case default'}, poa_horizon={config.horizon}\n"
        f"  synthetic_regime_set={config.synthetic_regime_set}, seed={config.synthetic_seed}\n"
        f"  poa_regime_set={config.poa_regime_set}, seed={config.poa_seed}\n"
        f"  hidden_layers={config.hidden_layers}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}\n"
        f"  solver={config.solver_name}, poa_parallel_workers={config.poa_parallel_workers}"
    )


def write_runtime_regime_config(config: FullPipelineConfig) -> Path:
    source_path = ScenarioManager.DEFAULT_REGIME_CONFIG_PATH
    with source_path.open("r", encoding="utf-8") as file_handle:
        raw_config = yaml.safe_load(file_handle) or {}

    regime_sets = raw_config.get("regime_sets")
    if not isinstance(regime_sets, dict):
        raise ValueError(f"Expected 'regime_sets' in {source_path}")

    synthetic_set = _copy_regime_set_with_counts(
        regime_sets=regime_sets,
        source_name=config.source_synthetic_regime_set,
        scenario_counts=config.synthetic_scenarios_per_regime,
    )
    poa_set = _copy_regime_set_with_counts(
        regime_sets=regime_sets,
        source_name=config.source_poa_regime_set,
        scenario_counts=config.poa_context_scenarios_per_regime,
    )

    runtime_config = copy.deepcopy(raw_config)
    runtime_config["default_regime_set"] = config.synthetic_regime_set
    runtime_config["regime_sets"] = {
        config.synthetic_regime_set: synthetic_set,
        config.poa_regime_set: poa_set,
    }

    config.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    with config.runtime_config_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(runtime_config, file_handle, sort_keys=False)
    print(f"\nWrote runtime regime config: {config.runtime_config_path}")
    return config.runtime_config_path


def _copy_regime_set_with_counts(
    regime_sets: dict[str, Any],
    source_name: str,
    scenario_counts: dict[str, int],
) -> dict[str, Any]:
    if source_name not in regime_sets:
        available = ", ".join(sorted(regime_sets))
        raise ValueError(f"Unknown source regime set '{source_name}'. Available: {available}")

    regime_set = copy.deepcopy(regime_sets[source_name])
    for regime in regime_set.get("regimes", []):
        regime_name = str(regime.get("name"))
        if regime_name in scenario_counts:
            regime["n_scenarios"] = int(scenario_counts[regime_name])
    return regime_set

def load_or_generate_scenarios(
    config: FullPipelineConfig,
    manager: ScenarioManager,
    regime_set: str,
    seed: int,
    output_dir: Path,
    should_generate: bool,
    time_steps: int | None,
) -> dict[str, Any]:
    # The scenario generator is deterministic for a fixed runtime YAML + seed, so
    # regenerating is usually safer than deserializing list-valued CSV columns.
    apply_time_steps_override(manager, time_steps)
    scenarios = manager.create_scenario_set_from_regimes(
        regime_config_path=str(config.runtime_config_path),
        regime_set=regime_set,
        seed=seed,
    )
    if should_generate:
        save_scenario_tables(scenarios, output_dir)
    print(scenarios["description_text"])
    return scenarios

def save_scenario_tables(scenarios: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios["scenarios_df"].to_csv(output_dir / "scenarios.csv", index=False)
    scenarios["costs_df"].to_csv(output_dir / "costs.csv", index=False)
    scenarios["ramps_df"].to_csv(output_dir / "ramps.csv", index=False)

def apply_time_steps_override(manager: ScenarioManager, time_steps: int | None) -> None:
    if time_steps is None:
        return
    if time_steps <= 0:
        raise ValueError(f"time_steps must be positive, got {time_steps}")
    manager.base_case["time_steps"] = int(time_steps)

def run_heuristic(
    config: FullPipelineConfig,
    scenarios: dict[str, Any],
    scenario_manager: ScenarioManager,
) -> Path:
    start = time.perf_counter()
    heuristic = MeritOrderHeuristic(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        players_config=scenario_manager.get_players_config(),
        bid_tolerance=config.bid_tolerance,
    )
    heuristic.run()
    output_path = heuristic.save_results(config.heuristic_results_path)
    elapsed = time.perf_counter() - start
    print(f"\nSaved heuristic synthetic labels to {output_path}")
    print(f"Heuristic runtime: {elapsed:.2f} seconds")
    return output_path

def build_features(config: FullPipelineConfig, scenarios: dict[str, Any]) -> dict[str, Path]:
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

def train_policies(config: FullPipelineConfig) -> Path:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.training_result_dir.mkdir(parents=True, exist_ok=True)

    training_config = BiddingPolicyTrainingConfig(
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        test_size=config.test_size,
        random_state=config.random_state,
        patience=config.patience,
        min_delta=config.min_delta,
        device=config.device,
    )

    csv_paths = find_generator_feature_files(config.normalized_feature_dir)
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
            f"best_test_loss={history['best_test_loss']:.8g}, model={result['model_path']}"
        )

    summary_path = config.training_result_dir / "training_summary.json"
    write_json(summary_path, summary_entries)
    elapsed = time.perf_counter() - start
    print(f"\nSaved training summary to {summary_path}")
    print(f"NN training runtime: {elapsed:.2f} seconds")
    return summary_path

def find_generator_feature_files(feature_dir: Path) -> list[Path]:
    if not feature_dir.exists():
        raise ValueError(f"Feature directory does not exist: {feature_dir}")
    return sorted(
        path
        for path in feature_dir.glob("*_features_normalized.csv")
        if path.is_file()
    )

def load_poa_scenario_data(config: FullPipelineConfig) -> dict[str, Any]:
    if not config.runtime_config_path.exists():
        write_runtime_regime_config(config)
    scenario_manager = ScenarioManager(config.case)
    apply_time_steps_override(scenario_manager, config.poa_time_steps)
    return scenario_manager.create_scenario_set_from_regimes(
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        seed=config.poa_seed,
    )


def load_ambiguity_set_config(config: FullPipelineConfig) -> dict[str, Any]:
    return PoAOptimization.load_ambiguity_set_config(
        config_path=config.ambiguity_set_config_path,
        config_name=config.ambiguity_set_config_name,
    )


def build_poa_optimizer(
    config: FullPipelineConfig,
    optimizer_cls: type[PoAOptimization] = PoAOptimization,
) -> PoAOptimization:
    scenarios = load_poa_scenario_data(config)
    ambiguity_set_config = load_ambiguity_set_config(config)
    return optimizer_cls(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=config.horizon,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=str(config.model_dir),
        nn_normalization_stats_path=str(config.nn_normalization_stats_path),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
    )


def build_poa_tightening(
    config: FullPipelineConfig,
    tightening_cls: type[PoATighteningMain] = PoATighteningMain,
) -> PoATighteningMain:
    scenarios = load_poa_scenario_data(config)
    ambiguity_set_config = load_ambiguity_set_config(config)
    return tightening_cls(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=config.horizon,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=str(config.model_dir),
        nn_normalization_stats_path=str(config.nn_normalization_stats_path),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
    )


def run_tightening_pipeline(config: FullPipelineConfig) -> Path:
    flags = {**TIGHTENING_FLAGS, **dict(config.tightening_flags)}
    previous_paths = {
        **DEFAULT_TIGHTENING_OUTPUT_PATHS,
        **dict(config.tightening_previous_paths),
    }
    output_paths = {
        **DEFAULT_TIGHTENING_OUTPUT_PATHS,
        **dict(config.tightening_output_paths),
    }

    print_tightening_plan(config, flags, previous_paths, output_paths)

    start = time.perf_counter()
    tightening = build_poa_tightening(config)
    final_report_path = tightening.run_all(
        run_primal_big_m=bool(flags["primal_big_m"]),
        run_relu_bounds=bool(flags["relu_bounds"]),
        run_alpha_bounds=bool(flags["alpha_bounds"]),
        run_slack_binary_fix=bool(flags["slack_binary_fix"]),
        run_dual_big_m=bool(flags["dual_big_m"]),
        previous_paths=previous_paths,
        output_paths=output_paths,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        epsilon=config.epsilon,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    elapsed = time.perf_counter() - start
    print(f"\nStaged PoA tightening complete: {final_report_path}")
    print(f"Tightening runtime: {elapsed:.2f} seconds")
    return final_report_path


def print_tightening_plan(
    config: FullPipelineConfig,
    flags: dict[str, bool],
    previous_paths: dict[str, str | Path],
    output_paths: dict[str, str | Path],
) -> None:
    time_limit = (
        "none"
        if config.preprocessing_time_limit is None
        else f"{config.preprocessing_time_limit} seconds"
    )
    threads = (
        "solver default"
        if config.poa_solver_threads_per_worker is None
        else str(config.poa_solver_threads_per_worker)
    )

    print("\nStarting staged PoA tightening")
    print("  Runtime")
    print(f"    solver: {config.solver_name}")
    print(f"    time limit: {time_limit}")
    print(f"    parallel workers: {config.poa_parallel_workers}")
    print(f"    solver threads per worker: {threads}")
    print("  Stages")
    for stage_name in TIGHTENING_STAGE_ORDER:
        label = TIGHTENING_STAGE_LABELS[stage_name]
        should_run = bool(flags[stage_name])
        action = "run" if should_run else "reuse"
        path_label = "output" if should_run else "input"
        path = output_paths[stage_name] if should_run else previous_paths[stage_name]
        print(f"    {label:<17} {action:<5} {path_label}: {path}")
    print(f"  Final report: {output_paths['final']}")


def run_nn_relu_bounds(config: FullPipelineConfig) -> Path:
    print("\nStarting PoA NN ReLU bound tightening")
    print(f"  output={config.nn_relu_bounds_path}")
    print(f"  horizon={config.horizon}")
    print(f"  policy_generators={list(config.nn_policy_generators)}")
    print(f"  model_dir={config.model_dir}")
    print(f"  normalization_stats={config.nn_normalization_stats_path}")
    print(
        f"  solver={config.solver_name}, time_limit={config.preprocessing_time_limit}, "
        f"parallel_workers={config.poa_parallel_workers}"
    )
    stage = build_poa_tightening(config, ReLUBoundsComputer)
    optimizer = stage.poa
    print(
        "  resolved_policy_generators="
        f"{list(optimizer.nn_policy_generator_names)}"
    )
    print(
        f"  physical_generators={optimizer.num_physical_generators}, "
        f"generator_blocks={len(optimizer.generator_block_pairs)}, "
        f"time_steps={optimizer.num_time_steps}"
    )
    start = time.perf_counter()
    report = stage.run_relu_bounds(
        output_path=config.nn_relu_bounds_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
    )
    output_path = config.nn_relu_bounds_path
    elapsed = time.perf_counter() - start

    summary = report.get("summary", {})
    total_fixed = sum(
        int(details.get("num_active", 0)) + int(details.get("num_inactive", 0))
        for details in summary.values()
    )

    print(f"\nNN ReLU preactivation-bound report complete: {output_path}")
    print(f"NN ReLU bound runtime: {elapsed:.2f} seconds")
    if optimizer.nn_bound_warnings:
        print("NN ReLU bound warnings:")
        for warning in optimizer.nn_bound_warnings:
            print(f"  - {warning}")
    for generator_name in optimizer.nn_policy_generator_names:
        details = summary.get(generator_name, {})
        print(
            f"  {generator_name}: "
            f"active={int(details.get('num_active', 0))}, "
            f"inactive={int(details.get('num_inactive', 0))}, "
            f"ambiguous={int(details.get('num_ambiguous', 0))}, "
            f"min_L={details.get('min_L')}, "
            f"max_U={details.get('max_U')}"
        )
    print(f"Total fixed NN ReLU binaries: {total_fixed}")
    return output_path


def run_alpha_bounds(config: FullPipelineConfig) -> Path:
    stage = build_poa_tightening(config, AlphaBoundsComputer)
    if config.primal_big_m_path.exists():
        stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    else:
        stage._as_stage(PrimalBigMComputer).run_primal_big_m(
            output_path=config.primal_big_m_path
        )
    stage._load_previous_stage("relu_bounds", config.nn_relu_bounds_path)

    start = time.perf_counter()
    report = stage.run_alpha_bounds(
        output_path=config.alpha_bounds_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    elapsed = time.perf_counter() - start

    output_path = config.alpha_bounds_path
    print(f"\nAlpha-bound computation complete: {output_path}")
    print(f"Alpha entries: {len(report.get('alpha_bounds', {}))}")
    print(f"Alpha-bound runtime: {elapsed:.2f} seconds")
    return output_path


def run_primal_big_m(
    config: FullPipelineConfig,
    optimizer: PoAOptimization | None = None,
) -> dict[str, dict[str, Any]]:
    if optimizer is None:
        optimizer = build_poa_optimizer(config, PoAOptimization)

    start = time.perf_counter()
    primal_big_m = compute_primal_big_m_bounds(optimizer)
    summary = summarize_primal_big_m(primal_big_m)
    payload = {
        "metadata": {
            "description": (
                "Analytic primal slack Big-M values used by PoAOptimization "
                "KKT complementarity constraints."
            ),
            "reference_case": optimizer.reference_case,
            "num_time_steps": optimizer.num_time_steps,
            "physical_generator_names": list(optimizer.physical_generator_names),
            "block_names": list(optimizer.block_names),
            "ambiguity_set": ambiguity_set_summary(optimizer),
            "summary": summary,
        },
        "primal_big_m": primal_big_m,
        "fixed_binaries": {},
        "slack_bounds": {},
        "lambda_bounds": {},
        "tight_big_m": {},
        "aggregate_dual_bounds": {},
    }
    output_path = write_json(config.primal_big_m_path, payload)
    elapsed = time.perf_counter() - start

    optimizer.primal_big_m = primal_big_m
    print(f"\nPrimal Big-M report complete: {output_path}")
    for component_name, details in summary.items():
        print(
            f"  {component_name}: entries={details['entries']}, "
            f"min={details['min_big_m']}, max={details['max_big_m']}"
        )
    print(f"Primal Big-M runtime: {elapsed:.2f} seconds")
    return primal_big_m


def run_slack_binary_fix(config: FullPipelineConfig) -> Path:
    stage = build_poa_tightening(config, SlackBinaryFixComputer)
    stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    stage._load_previous_stage("alpha_bounds", config.alpha_bounds_path)

    start = time.perf_counter()
    slack_report = stage.run_slack_binary_fix(
        output_path=config.slack_report_path,
        epsilon=config.epsilon,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    output_path = config.slack_report_path
    elapsed = time.perf_counter() - start

    print(f"\nSlack minimization and binary fixing complete: {output_path}")
    print(f"Fixed complementarity binaries: {slack_report['num_fixed_binaries']}")
    print(f"Slack/binary runtime: {elapsed:.2f} seconds")
    return output_path


def run_dual_big_m(config: FullPipelineConfig) -> Path:
    stage = build_poa_tightening(config, DualBigMComputer)
    stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    stage._load_previous_stage("alpha_bounds", config.alpha_bounds_path)
    stage._load_previous_stage("slack_binary_fix", config.slack_report_path)

    start = time.perf_counter()
    stage.run_dual_big_m(
        output_path=config.dual_big_m_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    output_path = config.dual_big_m_path
    elapsed = time.perf_counter() - start

    print(f"\nDual Big-M tightening complete: {output_path}")
    print(f"Dual Big-M runtime: {elapsed:.2f} seconds")
    return output_path


def run_final_poa(config: FullPipelineConfig) -> Path:
    optimizer = build_poa_optimizer(config, PoAOptimization)
    start = time.perf_counter()
    optimizer.load_tightening_report(config.tightening_report_path)
    optimizer.build_model()
    if optimizer.nn_policy_generator_ids:
        applied_nn_relu_stats = optimizer.apply_nn_relu_bounds_to_model()
    else:
        applied_nn_relu_stats = {
            "delta_fixed_active": 0,
            "delta_fixed_inactive": 0,
            "delta_left_ambiguous": 0,
        }
    applied_stats = optimizer.apply_tightened_bounds_to_model()
    optimizer.solve(time_limit=config.poa_time_limit)
    output_path = optimizer.save_results(config.poa_results_path)
    elapsed = time.perf_counter() - start

    print(f"\nPoA optimization complete: {output_path}")
    print(f"Applied active NN ReLU fixes: {applied_nn_relu_stats['delta_fixed_active']}")
    print(f"Applied inactive NN ReLU fixes: {applied_nn_relu_stats['delta_fixed_inactive']}")
    print(f"Ambiguous NN ReLUs left binary: {applied_nn_relu_stats['delta_left_ambiguous']}")
    print(f"Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"Applied lambda bounds: {applied_stats['lambda_bounds']}")
    print(f"Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"Applied aggregate dual bounds: {applied_stats['aggregate_dual_bounds']}")
    print(f"Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"PoA optimization runtime: {elapsed:.2f} seconds")
    return output_path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return path


def ensure_primal_big_m_in_report(path: Path, optimizer: PoAOptimization) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"Expected PoA tightening-stage report not found: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if payload.get("primal_big_m"):
        return False

    primal_big_m = compute_primal_big_m_bounds(optimizer)
    payload["primal_big_m"] = primal_big_m
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["primal_big_m_summary"] = summarize_primal_big_m(primal_big_m)
    write_json(path, payload)
    print(f"\nAdded missing primal Big-M constants to existing report: {path}")
    for component_name, details in summarize_primal_big_m(primal_big_m).items():
        print(
            f"  {component_name}: entries={details['entries']}, "
            f"min={details['min_big_m']}, max={details['max_big_m']}"
        )
    return True


if __name__ == "__main__":
    run_config = FullPipelineConfig(
        # Case and random seeds.
        case="test_case_bidding_blocks",
        synthetic_time_steps=24,
        poa_time_steps=None,
        synthetic_regime_set="policy_training_runtime",
        poa_regime_set="PoA_analysis_runtime",
        source_synthetic_regime_set="policy_training",
        source_poa_regime_set="PoA_analysis",
        synthetic_seed=1,
        poa_seed=1,

        # Scenario counts. Set a regime name to the desired number of scenarios.
        # Use {} to keep the counts from config/regime_definitions.yaml.
        synthetic_scenarios_per_regime={
            "normal": 400,
            "high_demand": 400,
            "normal_peak_shift_wind": 400,
            "high_demand_peak_shift_wind": 400,
        },
        poa_context_scenarios_per_regime={
            "normal": 1,
        },

        # Heuristic synthetic-label generation.
        bid_tolerance=1e-2,

        # Neural-network feature and training parameters.
        nn_feature_columns=[
            "demand",
            "total_wind_generation_capacity",
            "total_generation_capacity",
            "residual_demand",
            "next_generation_capacity",
            "next_demand",
            "own_generation_capacity",
            "next_own_generation_capacity",
        ],

        per_generator_normalization=True,
        hidden_layers=[11, 11],
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=500,
        weight_decay=0.0,
        test_size=0.2,
        random_state=42,
        patience=50,
        min_delta=1e-6,
        device=None,

        # PoA parameters.
        horizon=24,
        ambiguity_set_config_path="models/PoA/ambiguity_set_config.yaml",
        ambiguity_set_config_name="test_case_bidding_blocks_base",
        nn_policy_generators=[1, 2],
        solver_name="gurobi",
        preprocessing_time_limit=200,
        poa_time_limit=None,
        epsilon=1e-6,
        poa_parallel_workers=6,
        poa_solver_threads_per_worker=1,

        # Step toggles. Turn expensive stages off when reusing previous outputs.
        run_scenario_generation=False,
        run_heuristic_labels=False,
        run_feature_building=False,
        run_nn_training=False,
        run_tightening=True,
        tightening_flags={
            "primal_big_m": True,
            "relu_bounds": True,
            "alpha_bounds": True,
            "slack_binary_fix": True,
            "dual_big_m": True,
        },
        tightening_previous_paths=dict(TIGHTENING_PREVIOUS_PATHS),
        tightening_output_paths=dict(TIGHTENING_OUTPUT_PATHS),
        
        run_poa_nn_relu_bounds=True,
        run_poa_alpha_bounds=True,
        run_poa_slack_binary_fix=True,
        run_poa_dual_big_m=True,
        run_poa_optimization=True,

        # Outputs
        runtime_config_path=Path("results/full_pipeline/runtime_regime_definitions.yaml"),
        synthetic_scenario_dir=Path("results/full_pipeline/synthetic_scenarios"),
        poa_scenario_dir=Path("results/full_pipeline/poa_scenarios"),
        heuristic_results_path=Path("results/merit_order_best_response_results.json"),
        raw_feature_dir=Path("models/neural_network/features/generated/raw"),
        normalized_feature_dir=Path("models/neural_network/features/generated/normalized"),
        model_dir=Path("models/neural_network/training/trained_models"),
        training_result_dir=Path("models/neural_network/training/training_results"),
        poa_result_dir=Path("results"),
    )
    main(run_config)
