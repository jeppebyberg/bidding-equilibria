from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.helper import sanitize_for_json
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
from models.PoA.PoA_tightening.compute_optimal_cost_bounds import (
    OptimalCostBoundsComputer,
)
from models.PoA.PoA_tightening.compute_relu_bounds import ReLUBoundsComputer
from models.PoA.PoA_tightening.compute_slack_binary_fix import SlackBinaryFixComputer

from driver.core.block0_core import (
    ALWAYS_ON_TIGHTENING_STAGES,
    ProjectConfig,
    poa_tightening_paths,
)
from driver.core.block1_core import apply_time_steps_override

RUN_TIGHTENING = True

TIGHTENING_FLAGS = {
    "primal_big_m": True,
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
    "optimal_cost_bounds": True,
}

TIGHTENING_PREVIOUS_PATHS = {
    "primal_big_m": "results/poa_tightening/primal_big_m_report.json",
    "relu_bounds": "results/poa_tightening/relu_bounds_report.json",
    "alpha_bounds": "results/poa_tightening/alpha_bounds_report.json",
    "slack_binary_fix": "results/poa_tightening/slack_binary_fix_report.json",
    "dual_big_m": "results/poa_tightening/dual_big_m_report.json",
    "optimal_cost_bounds": "results/poa_tightening/optimal_cost_bounds_report.json",
}

TIGHTENING_OUTPUT_PATHS = {
    "primal_big_m": "results/poa_tightening/primal_big_m_report.json",
    "relu_bounds": "results/poa_tightening/relu_bounds_report.json",
    "alpha_bounds": "results/poa_tightening/alpha_bounds_report.json",
    "slack_binary_fix": "results/poa_tightening/slack_binary_fix_report.json",
    "dual_big_m": "results/poa_tightening/dual_big_m_report.json",
    "optimal_cost_bounds": "results/poa_tightening/optimal_cost_bounds_report.json",
    "final": "results/poa_tightening/final_tightening_report.json",
}

TIGHTENING_STAGE_ORDER = (
    "primal_big_m",
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
    "optimal_cost_bounds",
)

TIGHTENING_STAGE_LABELS = {
    "primal_big_m": "Primal Big-M",
    "relu_bounds": "NN ReLU bounds",
    "alpha_bounds": "Alpha bounds",
    "slack_binary_fix": "Slack binary fix",
    "dual_big_m": "Dual Big-M",
    "optimal_cost_bounds": "C_opt bounds",
}

_REGIME_PARAM_COLUMNS = ("mu_D", "rho_D", "sigma_D", "mu_W", "rho_W", "sigma_W", "peak_W")


@dataclass
class PoAPipelineConfig:
    # Case and random seeds.
    case: str = "base_test_case"
    synthetic_time_steps: int | None = None
    synthetic_seed: int = 1
    poa_seed: int = 1

    # Scenario counts for ambiguity-set draws.
    synthetic_num_scenarios: int = 400
    poa_context_num_scenarios: int = 1

    # Heuristic synthetic-label generation.
    bid_tolerance: float = 1e-2

    # Neural-network feature and training parameters.
    # These names must be supported by both NeuralNetworkFeatureBuilder and PoAOptimization._raw_nn_feature_expression.
    nn_feature_columns: list[str] = field(
        default_factory=lambda: [
            "demand",
            "previous_demand",
            "next_demand",
            "total_wind_generation_capacity",
            "previous_wind_generation_capacity",
            "next_wind_generation_capacity",
            "residual_demand",
            "previous_residual_demand",
            "next_residual_demand",
            "total_demand_over_horizon",
            "total_wind_over_horizon",
            "total_residual_over_horizon",
        ]
    )
    per_generator_normalization: bool = True
    hidden_layers: list[int] = field(default_factory=lambda: [7, 7])
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 500
    weight_decay: float = 0.0
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    patience: int | None = 50
    min_delta: float = 1e-6
    device: str | None = None
    nn_final_activation: str = "linear"
    # ReduceLROnPlateau scheduler on validation loss (set use_lr_scheduler=False
    # to keep a constant learning rate).
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 20
    lr_scheduler_min_lr: float = 1e-6

    # PoA parameters. The uncertainty set is induced by ambiguity_set_config_name
    # in config/ambiguity_set_config.yaml. The generated PoA context
    # scenarios only provide model dimensions/static case data.
    horizon: int = 8
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"
    nn_policy_generators: list[int | str] = field(default_factory=lambda: ["G1", "W3"])

    # Objective modes: "difference", "mccormick", or
    # "piecewise_mccormick". Difference is the historical default.
    poa_objective_mode: str = "piecewise_mccormick"
    # You may pass a complete PoAOptimization mccormick_bounds dictionary directly.
    # If omitted for McCormick modes, set poa_mccormick_PoA_bounds and
    # poa_mccormick_c_opt_bounds below.
    poa_mccormick_bounds: dict[str, Any] | None = None
    poa_mccormick_PoA_bounds: tuple[float, float] | None = None
    poa_mccormick_c_opt_bounds: tuple[float, float] | None = None
    poa_mccormick_num_pieces: int = 4
    poa_mccormick_c_opt_breakpoints: list[float] | None = None

    solver_name: str = "gurobi"
    preprocessing_time_limit: int = 200
    poa_time_limit: int | None = 400
    epsilon: float = 1e-6
    # Parallelizes the independent tightening submodels inside the three PoA
    # preprocessing stages. Keep Gurobi threads low when workers > 1.
    poa_parallel_workers: int = 1
    poa_solver_threads_per_worker: int | None = None

    # When True, render training diagnostic plots after NN training completes.
    plot_results_along_the_way: bool = False

    # Step toggles. Turn expensive stages off when reusing previous outputs.
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True
    run_tightening: bool = RUN_TIGHTENING
    tightening_flags: dict[str, bool] = field(default_factory=lambda: dict(TIGHTENING_FLAGS))
    tightening_previous_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(TIGHTENING_PREVIOUS_PATHS)
    )
    tightening_output_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(TIGHTENING_OUTPUT_PATHS)
    )
    run_poa_optimization: bool = True

    # Outputs.
    synthetic_scenario_dir: Path = Path("results/full_pipeline/synthetic_scenarios")
    poa_scenario_dir: Path = Path("results/full_pipeline/poa_scenarios")
    heuristic_results_path: Path = Path("results/merit_order_best_response_results.json")
    raw_feature_dir: Path = Path("models/neural_network/features/generated/raw")
    normalized_feature_dir: Path = Path("models/neural_network/features/generated/normalized")
    model_dir: Path = Path("models/neural_network/training/trained_models")
    training_result_dir: Path = Path("models/neural_network/training/training_results")
    poa_result_dir: Path = Path("results")
    # Figures root; None falls back to results/<case>/figures (see plotting code).
    figures_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.synthetic_time_steps is None:
            self.synthetic_time_steps = self.horizon

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
    def optimal_cost_bounds_path(self) -> Path:
        return Path(self.tightening_output_paths["optimal_cost_bounds"])

    @property
    def poa_results_path(self) -> Path:
        objective_suffix = (
            "" if self.poa_objective_mode == "difference" else f"_{self.poa_objective_mode}"
        )
        return self.poa_result_dir / (f"poa_optimization_T{self.horizon}{objective_suffix}.json")


def build_poa_config(config: ProjectConfig) -> PoAPipelineConfig:
    return PoAPipelineConfig(
        case=config.case,
        synthetic_time_steps=config.synthetic_time_steps,
        synthetic_seed=config.synthetic_seed,
        poa_seed=config.poa_seed,
        synthetic_num_scenarios=config.synthetic_num_scenarios,
        poa_context_num_scenarios=config.poa_context_num_scenarios,
        bid_tolerance=config.bid_tolerance,
        nn_feature_columns=list(config.nn_feature_columns),
        per_generator_normalization=config.per_generator_normalization,
        hidden_layers=list(config.hidden_layers),
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
        nn_final_activation=config.nn_final_activation,
        use_lr_scheduler=config.use_lr_scheduler,
        lr_scheduler_factor=config.lr_scheduler_factor,
        lr_scheduler_patience=config.lr_scheduler_patience,
        lr_scheduler_min_lr=config.lr_scheduler_min_lr,
        horizon=config.horizon,
        ambiguity_set_config_path=config.ambiguity_set_config_path,
        ambiguity_set_config_name=config.ambiguity_set_config_name,
        nn_policy_generators=list(config.nn_policy_generators),
        poa_objective_mode=config.poa_objective_mode,
        poa_mccormick_bounds=config.poa_mccormick_bounds,
        poa_mccormick_PoA_bounds=config.poa_mccormick_PoA_bounds,
        poa_mccormick_c_opt_bounds=config.poa_mccormick_c_opt_bounds,
        poa_mccormick_num_pieces=config.poa_mccormick_num_pieces,
        poa_mccormick_c_opt_breakpoints=config.poa_mccormick_c_opt_breakpoints,
        solver_name=config.solver_name,
        preprocessing_time_limit=config.preprocessing_time_limit,
        poa_time_limit=config.poa_time_limit,
        epsilon=config.epsilon,
        poa_parallel_workers=config.poa_parallel_workers,
        poa_solver_threads_per_worker=config.poa_solver_threads_per_worker,
        plot_results_along_the_way=config.plot_results_along_the_way,
        run_scenario_generation=config.run_scenario_generation,
        run_heuristic_labels=config.run_heuristic_labels,
        run_feature_building=config.run_feature_building,
        run_nn_training=config.run_nn_training,
        run_tightening=config.run_poa_tightening,
        tightening_flags=dict(config.poa_tightening_flags),
        tightening_previous_paths=poa_tightening_paths(Path(config.poa_result_dir)),
        tightening_output_paths=poa_tightening_paths(Path(config.poa_result_dir)),
        run_poa_optimization=config.run_poa_optimization,
        synthetic_scenario_dir=Path(config.synthetic_scenario_dir),
        poa_scenario_dir=Path(config.poa_scenario_dir),
        heuristic_results_path=Path(config.heuristic_results_path),
        raw_feature_dir=Path(config.raw_feature_dir),
        normalized_feature_dir=Path(config.normalized_feature_dir),
        model_dir=Path(config.model_dir),
        training_result_dir=Path(config.training_result_dir),
        poa_result_dir=Path(config.poa_result_dir),
        figures_dir=Path(config.figures_dir),
    )


def load_poa_scenario_data(config: PoAPipelineConfig) -> dict[str, Any]:
    scenario_manager = ScenarioManager(config.case)
    apply_time_steps_override(scenario_manager, config.horizon)
    return scenario_manager.create_scenario_set_from_ambiguity_set(
        ambiguity_config_path=config.ambiguity_set_config_path,
        ambiguity_set=config.ambiguity_set_config_name,
        n_scenarios=config.poa_context_num_scenarios,
        seed=config.poa_seed,
    )


def load_ambiguity_set_config(config: PoAPipelineConfig) -> dict[str, Any]:
    return PoAOptimization.load_ambiguity_set(
        config_path=config.ambiguity_set_config_path,
        config_name=config.ambiguity_set_config_name,
    )


def build_poa_mccormick_bounds(config: PoAPipelineConfig) -> dict[str, Any] | None:
    mode = str(config.poa_objective_mode).strip().lower()
    if mode not in PoAOptimization.allowed_objective_modes:
        allowed = ", ".join(sorted(PoAOptimization.allowed_objective_modes))
        raise ValueError(
            f"poa_objective_mode must be one of {{{allowed}}}; got "
            f"{config.poa_objective_mode!r}"
        )
    if mode == "difference":
        return None

    if config.poa_mccormick_bounds is not None:
        return dict(config.poa_mccormick_bounds)

    c_opt_bounds = config.poa_mccormick_c_opt_bounds
    if c_opt_bounds is None:
        c_opt_bounds = load_poa_optimal_cost_bounds(config)
    if c_opt_bounds is None:
        c_opt_bounds = default_poa_mccormick_c_opt_bounds()
        print(
            "\nWARNING: no PoA optimal_cost_bounds report found; using loose "
            f"default C_opt bounds {c_opt_bounds} for McCormick final solve."
        )

    if config.poa_mccormick_PoA_bounds is None or c_opt_bounds is None:
        raise ValueError(
            "McCormick objective modes require bounds. Set either "
            "poa_mccormick_bounds={'PoA': (...), 'C_opt': (...), ...} or both "
            "poa_mccormick_PoA_bounds and poa_mccormick_c_opt_bounds in FullPipelineConfig. "
            "Alternatively, run the optimal_cost_bounds tightening stage and provide "
            "poa_mccormick_PoA_bounds."
        )

    mccormick_bounds: dict[str, Any] = {
        "PoA": tuple(config.poa_mccormick_PoA_bounds),
        "C_opt": tuple(c_opt_bounds),
    }
    if mode == "piecewise_mccormick":
        if config.poa_mccormick_c_opt_breakpoints is not None:
            mccormick_bounds["C_opt_breakpoints"] = list(config.poa_mccormick_c_opt_breakpoints)
        else:
            mccormick_bounds["num_pieces"] = int(config.poa_mccormick_num_pieces)
    return mccormick_bounds


def default_poa_mccormick_c_opt_bounds() -> tuple[float, float]:
    return (
        float(PoAOptimization.DEFAULT_LOOSE_C_OPT_LOWER),
        float(PoAOptimization.DEFAULT_LOOSE_C_OPT_UPPER),
    )


def build_poa_tightening_mccormick_bounds(
    config: PoAPipelineConfig,
) -> dict[str, Any] | None:
    mode = str(config.poa_objective_mode).strip().lower()
    if mode == "difference":
        return None
    if config.poa_mccormick_bounds is not None:
        return dict(config.poa_mccormick_bounds)

    mccormick_bounds: dict[str, Any] = {
        "C_opt": default_poa_mccormick_c_opt_bounds(),
    }
    if config.poa_mccormick_PoA_bounds is not None:
        mccormick_bounds["PoA"] = tuple(config.poa_mccormick_PoA_bounds)
    if mode == "piecewise_mccormick":
        mccormick_bounds["num_pieces"] = int(config.poa_mccormick_num_pieces)
    return mccormick_bounds


def load_poa_optimal_cost_bounds(config: PoAPipelineConfig) -> tuple[float, float] | None:
    for path in (config.tightening_report_path, config.optimal_cost_bounds_path):
        if not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        bounds = report.get("optimal_cost_bounds", {}) or {}
        if "C_opt" in bounds and isinstance(bounds.get("C_opt"), dict):
            bounds = bounds.get("C_opt", {}) or {}
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        if lower is not None and upper is not None:
            return (float(lower), float(upper))
    return None


def build_poa_optimizer(
    config: PoAPipelineConfig,
    optimizer_cls: type[PoAOptimization] = PoAOptimization,
) -> PoAOptimization:
    scenarios = load_poa_scenario_data(config)
    ambiguity_set_config = load_ambiguity_set_config(config)
    mccormick_bounds = build_poa_mccormick_bounds(config)
    optimizer = optimizer_cls(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=config.horizon,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=str(config.model_dir),
        nn_normalization_stats_path=str(config.nn_normalization_stats_path),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
        objective_mode=config.poa_objective_mode,
        mccormick_bounds=mccormick_bounds,
    )
    return optimizer


def build_poa_tightening(
    config: PoAPipelineConfig,
    tightening_cls: type[PoATighteningMain] = PoATighteningMain,
) -> PoATighteningMain:
    scenarios = load_poa_scenario_data(config)
    ambiguity_set_config = load_ambiguity_set_config(config)
    objective_mode = str(config.poa_objective_mode).strip().lower()
    tightening = tightening_cls(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=config.horizon,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=str(config.model_dir),
        nn_normalization_stats_path=str(config.nn_normalization_stats_path),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
        objective_mode=objective_mode,
        mccormick_bounds=build_poa_tightening_mccormick_bounds(config),
        use_default_bounds=(objective_mode != "difference"),
    )
    return tightening


def run_tightening_pipeline(
    config: PoAPipelineConfig,
    run_optional_stages: bool = True,
) -> Path:
    flags = {**TIGHTENING_FLAGS, **dict(config.tightening_flags)}
    if not run_optional_stages:
        # Tightening disabled: skip the optional speed-tightening stages but still
        # compute the always-on stages below.
        for stage_name in flags:
            if stage_name not in ALWAYS_ON_TIGHTENING_STAGES:
                flags[stage_name] = False
    # primal_big_m and optimal_cost_bounds always run, regardless of config.
    for stage_name in ALWAYS_ON_TIGHTENING_STAGES:
        flags[stage_name] = True
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
        run_optimal_cost_bounds=bool(flags["optimal_cost_bounds"]),
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
    config: PoAPipelineConfig,
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


def run_nn_relu_bounds(config: PoAPipelineConfig) -> Path:
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
    print("  resolved_policy_generators=" f"{list(optimizer.nn_policy_generator_names)}")
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


def run_alpha_bounds(config: PoAPipelineConfig) -> Path:
    stage = build_poa_tightening(config, AlphaBoundsComputer)
    if config.primal_big_m_path.exists():
        stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    else:
        stage._as_stage(PrimalBigMComputer).run_primal_big_m(output_path=config.primal_big_m_path)
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
    config: PoAPipelineConfig,
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
        "tight_big_m": {},
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


def run_slack_binary_fix(config: PoAPipelineConfig) -> Path:
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


def run_dual_big_m(config: PoAPipelineConfig) -> Path:
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


def run_optimal_cost_bounds(config: PoAPipelineConfig) -> Path:
    stage = build_poa_tightening(config, OptimalCostBoundsComputer)
    if config.primal_big_m_path.exists():
        stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    else:
        stage._as_stage(PrimalBigMComputer).run_primal_big_m(output_path=config.primal_big_m_path)

    start = time.perf_counter()
    report = stage.run_optimal_cost_bounds(
        output_path=config.optimal_cost_bounds_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    output_path = config.optimal_cost_bounds_path
    elapsed = time.perf_counter() - start
    bounds = report.get("optimal_cost_bounds", {}) or {}

    print(f"\nC_opt bound tightening complete: {output_path}")
    print(f"C_opt lower: {bounds.get('lower')}")
    print(f"C_opt upper: {bounds.get('upper')}")
    print(f"C_opt bound runtime: {elapsed:.2f} seconds")
    return output_path


def run_final_poa(config: PoAPipelineConfig) -> Path:
    optimizer = build_poa_optimizer(config, PoAOptimization)
    start = time.perf_counter()
    if config.tightening_report_path.exists():
        optimizer.load_tightening_report(config.tightening_report_path)
    else:
        print(
            "\nWARNING: no PoA final tightening report found; using loose "
            "default bounds for final PoA model construction."
        )
        optimizer.ensure_default_bounds_available(
            include_nn_relu_bounds=None,
            include_alpha_bounds=True,
            include_tight_big_m=True,
            include_lambda_bounds=True,
            include_optimal_cost_bounds=True,
            overwrite_existing=False,
        )
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
    save_poa_solve_log_artifacts(optimizer, Path(output_path))
    elapsed = time.perf_counter() - start

    print(f"\nPoA optimization complete: {output_path}")
    print(f"PoA objective mode: {optimizer.objective_mode}")
    if optimizer.objective_mode != "difference":
        objective_metrics = optimizer.extract_objective_metrics()
        print(f"  PoA: {objective_metrics.get('PoA')}")
        print(f"  ex-post ratio: {objective_metrics.get('ex_post_ratio')}")
        print(f"  ratio gap: {objective_metrics.get('ratio_gap')}")
        print(f"  McCormick product gap: {objective_metrics.get('mccormick_product_gap')}")
    print(f"Applied active NN ReLU fixes: {applied_nn_relu_stats['delta_fixed_active']}")
    print(f"Applied inactive NN ReLU fixes: {applied_nn_relu_stats['delta_fixed_inactive']}")
    print(f"Ambiguous NN ReLUs left binary: {applied_nn_relu_stats['delta_left_ambiguous']}")
    print(f"Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"Applied lambda bounds: {applied_stats['lambda_bounds']}")
    print(f"Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"PoA optimization runtime: {elapsed:.2f} seconds")
    return output_path


def save_poa_solve_log_artifacts(
    optimizer: PoAOptimization,
    result_path: Path,
) -> Path | None:
    """Persist the final PoA solve's Gurobi log and parsed bound progression.

    Writes two sidecar files under ``<poa_dir>/solve_logs/``: the raw Gurobi log
    (``<result>_gurobi.log``) and a JSON time series of incumbent/BestBd/gap over
    solve time (``<result>_progress.json``), so bound movement and compute time
    can be plotted without re-running. The subfolder keeps them out of the
    ``poa_optimization_T*.json`` result globs used by the summary.
    """
    log_text = getattr(optimizer, "last_solve_gurobi_log", None)
    progression = getattr(optimizer, "solve_bound_progression", None) or []
    if not log_text and not progression:
        return None

    def _opt_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    log_dir = result_path.parent / "solve_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = result_path.stem
    if log_text:
        (log_dir / f"{stem}_gurobi.log").write_text(log_text, encoding="utf-8")

    solver_results = getattr(optimizer, "solver_results", None)
    termination = (
        str(solver_results.solver.termination_condition) if solver_results is not None else None
    )
    payload = {
        "objective_mode": optimizer.objective_mode,
        "num_time_steps": int(optimizer.num_time_steps),
        "termination_condition": termination,
        "wall_time_seconds": _opt_float(getattr(optimizer, "solve_wall_time_seconds", None)),
        "best_objective_bound": _opt_float(getattr(optimizer, "best_objective_bound", None)),
        "mip_gap": _opt_float(getattr(optimizer, "mip_gap", None)),
        "bound_progression": progression,
    }
    progress_path = log_dir / f"{stem}_progress.json"
    with progress_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return progress_path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(sanitize_for_json(payload), file_handle, indent=2)
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


def extract_poa_regime_params(
    poa_scenario_dir: Path,
    poa_results_path: Path | None = None,
) -> dict[str, float]:
    if poa_results_path is not None and Path(poa_results_path).exists():
        with Path(poa_results_path).open("r", encoding="utf-8") as fh:
            poa_result: dict[str, Any] = json.load(fh)
        ambiguity_set = poa_result.get("ambiguity_set", {}) or {}
        selected_regime = ambiguity_set.get("selected_regime", {}) or {}
        fixed_parameters = ambiguity_set.get("fixed_parameters", {}) or {}
        required_selected = ("mu_D", "sigma_D", "mu_W", "sigma_W")
        missing_selected = [key for key in required_selected if selected_regime.get(key) is None]
        required_fixed = ("rho_D", "rho_W")
        missing_fixed = [key for key in required_fixed if fixed_parameters.get(key) is None]
        peak_value = fixed_parameters.get("peak_W", fixed_parameters.get("tau_W"))
        if not missing_selected and not missing_fixed and peak_value is not None:
            return {
                "mu_D": float(selected_regime["mu_D"]),
                "rho_D": float(fixed_parameters["rho_D"]),
                "sigma_D": float(selected_regime["sigma_D"]),
                "mu_W": float(selected_regime["mu_W"]),
                "rho_W": float(fixed_parameters["rho_W"]),
                "sigma_W": float(selected_regime["sigma_W"]),
                "peak_W": float(peak_value),
            }
        print(
            "\nWARNING: PoA result did not contain a complete optimized regime; "
            "falling back to PoA context scenario CSV."
        )

    scenarios_csv = Path(poa_scenario_dir) / "scenarios.csv"
    if not scenarios_csv.exists():
        raise FileNotFoundError(
            f"PoA context scenario not found: {scenarios_csv}\n"
            "Run the PoA stage first so that the context scenario is saved."
        )
    df = pd.read_csv(scenarios_csv, nrows=1)
    missing = [col for col in _REGIME_PARAM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"PoA context scenario CSV is missing regime parameter columns: {missing}")
    row = df.iloc[0]
    return {col: float(row[col]) for col in _REGIME_PARAM_COLUMNS}


def write_poa_regime_runtime_config(
    config: ProjectConfig,
    regime_params: dict[str, float],
) -> Path:
    runtime_config = {
        "regime_sets": {
            config.poa_regime_set: {
                "description": "Single regime extracted from the optimized PoA state.",
                "seed": config.poa_seed,
                "enforce_dispatch_feasibility": True,
                "n_minus_one_margin": 0.0,
                "max_draw_attempts": 500,
                "regimes": [
                    {
                        "name": config.poa_worst_case_regime_name,
                        "n_scenarios": config.poa_worst_case_n_scenarios,
                        **regime_params,
                    }
                ],
            }
        }
    }
    output_path = write_json(Path(config.runtime_config_path), runtime_config)
    print(f"\nWrote PoA regime runtime config: {output_path}")
    return output_path


def print_regime_bridge(config: ProjectConfig, regime_params: dict[str, float]) -> None:
    print(
        f"\nPoA -> DRO regime bridge: extracted optimized state as "
        f"DRO regime '{config.poa_worst_case_regime_name}'"
    )
    for key, value in regime_params.items():
        print(f"  {key}: {value:.6g}")
