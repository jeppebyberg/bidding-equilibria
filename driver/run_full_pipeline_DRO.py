from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from driver.run_full_pipeline import (
    apply_time_steps_override,
    build_features,
    load_or_generate_scenarios,
    run_heuristic,
    train_policies,
    write_json,
    write_runtime_regime_config,
)
from models.DRO_PoA.DRO_PoA_optimization import DRO_PoAOptimization
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import (
    DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS,
    DROPoATighteningMain,
)


DRO_TIGHTENING_FLAGS = {
    "primal_big_m": True,
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
    "optimal_cost_bounds": True,
}

DRO_TIGHTENING_STAGE_ORDER = (
    "primal_big_m",
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
    "optimal_cost_bounds",
)

DRO_TIGHTENING_STAGE_LABELS = {
    "primal_big_m": "Primal Big-M",
    "relu_bounds": "NN ReLU bounds",
    "alpha_bounds": "Alpha bounds",
    "slack_binary_fix": "Slack binary fix",
    "dual_big_m": "Dual Big-M",
    "optimal_cost_bounds": "C_opt bounds",
}


@dataclass
class DROFullPipelineConfig:
    # Case and random seeds.
    case: str = "test_case_bidding_blocks"
    synthetic_time_steps: int | None = 24
    dro_time_steps: int | None = None
    synthetic_regime_set: str = "policy_training_runtime"
    poa_regime_set: str = "PoA_analysis_runtime"
    source_synthetic_regime_set: str = "policy_training"
    source_poa_regime_set: str = "PoA_analysis"
    synthetic_seed: int = 1
    poa_seed: int = 1

    # Scenario counts. The poa_* names are retained so the shared PoA driver
    # helpers can write a runtime regime file; here they define DRO regimes.
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
    dro_regime_names: list[str] | None = None

    # Heuristic synthetic-label generation.
    bid_tolerance: float = 1e-2

    # Neural-network feature and training parameters.
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

    # DRO parameters.
    horizon: int = 8
    etas: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])
    dro_wasserstein_epsilon: float = 0.0
    # Tightening is regime-wide and support-driven, so it is run once per regime.
    # The value is kept in tightening metadata but the report is reused for all eta.
    dro_tightening_eta: float = 0.0
    nn_policy_generators: list[int] = field(default_factory=lambda: [1, 2])

    # Objective modes: "difference", "ratio_mccormick", or
    # "ratio_piecewise_mccormick". Difference is the historical default.
    dro_objective_mode: str = "difference"
    # You may pass a complete DRO_PoAOptimization ratio_bounds dictionary directly.
    # If omitted for ratio modes, set dro_ratio_phi_bounds and either
    # dro_ratio_c_opt_bounds or run/load the optimal_cost_bounds stage.
    dro_ratio_bounds: dict[str, Any] | None = None
    dro_ratio_phi_bounds: tuple[float, float] | None = None
    dro_ratio_c_opt_bounds: tuple[float, float] | None = None
    dro_ratio_num_pieces: int = 4
    dro_ratio_c_opt_breakpoints: list[float] | None = None

    solver_name: str = "gurobi"
    preprocessing_time_limit: int | None = 200
    dro_time_limit: int | None = 400
    slack_epsilon: float = 1e-6
    poa_parallel_workers: int = 1
    poa_solver_threads_per_worker: int | None = None

    # Step toggles.
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True
    run_dro_tightening: bool = True
    run_dro_optimization: bool = True
    tightening_flags: dict[str, bool] = field(
        default_factory=lambda: dict(DRO_TIGHTENING_FLAGS)
    )
    tightening_previous_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS)
    )
    tightening_output_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS)
    )

    # Outputs.
    runtime_config_path: Path = Path("results/full_pipeline_DRO/runtime_regime_definitions.yaml")
    synthetic_scenario_dir: Path = Path("results/full_pipeline_DRO/synthetic_scenarios")
    poa_scenario_dir: Path = Path("results/full_pipeline_DRO/dro_scenarios")
    heuristic_results_path: Path = Path("results/dro_merit_order_best_response_results.json")
    raw_feature_dir: Path = Path("models/neural_network/features/generated/raw")
    normalized_feature_dir: Path = Path("models/neural_network/features/generated/normalized")
    model_dir: Path = Path("models/neural_network/training/trained_models")
    training_result_dir: Path = Path("models/neural_network/training/training_results")
    dro_result_dir: Path = Path("results/dro_poa")
    archive_existing_dro_results: bool = True
    dro_result_archive_dir: Path = Path("results/dro_poa/old_results")

    @property
    def nn_normalization_stats_path(self) -> Path:
        return self.normalized_feature_dir / "min_max_stats.json"


def main(config: DROFullPipelineConfig) -> None:
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

    if config.run_dro_tightening or config.run_dro_optimization:
        dro_scenarios = load_dro_scenario_data(config)
        regime_names = resolve_dro_regime_names(config, dro_scenarios)
    else:
        dro_scenarios = {}
        regime_names = []

    if config.run_dro_tightening:
        for regime_name in regime_names:
            run_dro_tightening_for_regime(config, dro_scenarios, regime_name)

    if config.run_dro_optimization:
        if config.archive_existing_dro_results:
            archive_existing_dro_result_folders(config, regime_names)
        sweep_summary = run_dro_eta_sweep(config, dro_scenarios, regime_names)
        summary_path = config.dro_result_dir / "eta_sweep_summary.json"
        write_json(summary_path, sweep_summary)
        print(f"\nSaved DRO eta-sweep summary: {summary_path}")

    print("\nDRO full pipeline complete.")


def print_pipeline_header(config: DROFullPipelineConfig) -> None:
    eta_values = ", ".join(str(float(eta)) for eta in config.etas)
    print(
        "\nDRO full pipeline configuration\n"
        f"  case={config.case}\n"
        f"  synthetic_time_steps={config.synthetic_time_steps or 'case default'}\n"
        f"  dro_time_steps={config.dro_time_steps or 'case default'}, dro_horizon={config.horizon}\n"
        f"  synthetic_regime_set={config.synthetic_regime_set}, seed={config.synthetic_seed}\n"
        f"  dro_regime_set={config.poa_regime_set}, seed={config.poa_seed}\n"
        f"  regimes={config.dro_regime_names or 'all generated DRO regimes'}\n"
        f"  dro_objective_mode={config.dro_objective_mode}\n"
        f"  etas=[{eta_values}], wasserstein_epsilon={config.dro_wasserstein_epsilon}\n"
        f"  solver={config.solver_name}, parallel_workers={config.poa_parallel_workers}"
    )


def load_dro_scenario_data(config: DROFullPipelineConfig) -> dict[str, Any]:
    if not config.runtime_config_path.exists():
        write_runtime_regime_config(config)
    scenario_manager = ScenarioManager(config.case)
    apply_time_steps_override(scenario_manager, config.dro_time_steps)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        seed=config.poa_seed,
    )
    if config.run_scenario_generation:
        config.poa_scenario_dir.mkdir(parents=True, exist_ok=True)
        scenarios["scenarios_df"].to_csv(
            config.poa_scenario_dir / "scenarios.csv",
            index=False,
        )
        scenarios["costs_df"].to_csv(config.poa_scenario_dir / "costs.csv", index=False)
        scenarios["ramps_df"].to_csv(config.poa_scenario_dir / "ramps.csv", index=False)
    print(scenarios["description_text"])
    return scenarios


def resolve_dro_regime_names(
    config: DROFullPipelineConfig,
    scenarios: dict[str, Any],
) -> list[str]:
    if config.dro_regime_names is not None:
        return [str(regime_name) for regime_name in config.dro_regime_names]
    scenarios_df = scenarios["scenarios_df"]
    if "regime" not in scenarios_df.columns:
        raise ValueError(
            "dro_regime_names must be provided when scenarios_df has no 'regime' column"
        )
    return sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())


def build_dro_tightening(
    config: DROFullPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
) -> DROPoATighteningMain:
    return DROPoATighteningMain(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=config.horizon,
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        regime_name=regime_name,
        eta=config.dro_tightening_eta,
        epsilon=config.dro_wasserstein_epsilon,
        nn_model_dir=str(config.model_dir) if config.nn_policy_generators else None,
        nn_normalization_stats_path=(
            str(config.nn_normalization_stats_path)
            if config.nn_policy_generators
            else None
        ),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
    )


def run_dro_tightening_for_regime(
    config: DROFullPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
) -> Path:
    flags = {**DRO_TIGHTENING_FLAGS, **dict(config.tightening_flags)}
    previous_paths = resolved_stage_paths(config.tightening_previous_paths, regime_name)
    output_paths = resolved_stage_paths(config.tightening_output_paths, regime_name)

    print_tightening_plan(config, regime_name, flags, previous_paths, output_paths)

    start = time.perf_counter()
    tightening = build_dro_tightening(config, scenarios, regime_name)
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
        epsilon=config.slack_epsilon,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    elapsed = time.perf_counter() - start
    print(f"\nDRO regime-wide tightening complete: {final_report_path}")
    print(f"  Regime: {regime_name}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    return final_report_path


def print_tightening_plan(
    config: DROFullPipelineConfig,
    regime_name: str,
    flags: dict[str, bool],
    previous_paths: dict[str, str],
    output_paths: dict[str, str],
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

    print(f"\nStarting DRO staged tightening for regime '{regime_name}'")
    print("  Runtime")
    print(f"    solver: {config.solver_name}")
    print(f"    time limit: {time_limit}")
    print(f"    parallel workers: {config.poa_parallel_workers}")
    print(f"    solver threads per worker: {threads}")
    print("  Stages")
    for stage_name in DRO_TIGHTENING_STAGE_ORDER:
        label = DRO_TIGHTENING_STAGE_LABELS[stage_name]
        should_run = bool(flags[stage_name])
        action = "run" if should_run else "reuse"
        path_label = "output" if should_run else "input"
        path = output_paths[stage_name] if should_run else previous_paths[stage_name]
        print(f"    {label:<17} {action:<5} {path_label}: {path}")
    print(f"  Final report: {output_paths['final']}")
    print("  Note: this support-tightening report is reused for every eta in the regime.")


def run_dro_eta_sweep(
    config: DROFullPipelineConfig,
    scenarios: dict[str, Any],
    regime_names: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for regime_name in regime_names:
        stage_paths = resolved_stage_paths(config.tightening_output_paths, regime_name)
        tightening_report_path = Path(stage_paths["final"])
        if not tightening_report_path.exists():
            raise FileNotFoundError(
                "Expected DRO regime-wide tightening report not found for "
                f"regime '{regime_name}': {tightening_report_path}"
            )
        for eta in config.etas:
            result_path = run_final_dro_for_eta(
                config=config,
                scenarios=scenarios,
                regime_name=regime_name,
                eta=float(eta),
                tightening_report_path=tightening_report_path,
            )
            summaries.append(load_result_summary(result_path))
    return summaries


def archive_existing_dro_result_folders(
    config: DROFullPipelineConfig,
    regime_names: list[str],
) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_paths: list[Path] = []
    for regime_name in regime_names:
        regime_dir = config.dro_result_dir / regime_name
        if not regime_dir.exists():
            continue
        if not any(regime_dir.iterdir()):
            continue

        archive_dir = config.dro_result_archive_dir / f"{regime_name}_{timestamp}"
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        archive_dir = _available_archive_path(archive_dir)
        shutil.move(str(regime_dir), str(archive_dir))
        archived_paths.append(archive_dir)
        print(f"\nArchived existing DRO results for regime '{regime_name}': {archive_dir}")
    return archived_paths


def _available_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def run_final_dro_for_eta(
    config: DROFullPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
    eta: float,
    tightening_report_path: Path,
) -> Path:
    optimizer = build_dro_optimizer(config, scenarios, regime_name, eta)
    result_path = dro_result_path(config, regime_name, eta)

    start = time.perf_counter()
    optimizer.load_regime_wide_tightening_report(tightening_report_path)
    optimizer.build_model()
    applied_stats = optimizer.apply_regime_wide_tightening_to_model()
    optimizer.solve(time_limit=config.dro_time_limit)
    output_path = optimizer.save_results(result_path)
    elapsed = time.perf_counter() - start

    print(f"\nDRO PoA optimization complete: {output_path}")
    print(f"  Regime: {regime_name}")
    print(f"  Eta: {eta}")
    print(f"  Objective mode: {optimizer.objective_mode}")
    if optimizer.objective_mode != "difference":
        summary = optimizer.solution_summary()
        print(f"  Average ex-post ratio: {summary.get('average_poa_ratio')}")
        print(f"  Average relaxed phi: {summary.get('average_relaxed_phi')}")
    print(f"  Reused tightening report: {tightening_report_path}")
    print(f"  Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"  Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"  Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"  Applied aggregate dual bounds: {applied_stats['aggregate_dual_bounds']}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    return output_path


def build_dro_ratio_bounds(
    config: DROFullPipelineConfig,
    regime_name: str,
) -> dict[str, Any] | None:
    mode = str(config.dro_objective_mode).strip().lower()
    if mode not in DRO_PoAOptimization.allowed_objective_modes:
        allowed = ", ".join(sorted(DRO_PoAOptimization.allowed_objective_modes))
        raise ValueError(
            f"dro_objective_mode must be one of {{{allowed}}}; got "
            f"{config.dro_objective_mode!r}"
        )
    if mode == "difference":
        return None

    if config.dro_ratio_bounds is not None:
        return dict(config.dro_ratio_bounds)

    c_opt_bounds = config.dro_ratio_c_opt_bounds
    if c_opt_bounds is None:
        c_opt_bounds = load_dro_optimal_cost_bounds(config, regime_name)

    if config.dro_ratio_phi_bounds is None or c_opt_bounds is None:
        raise ValueError(
            "DRO ratio objective modes require ratio bounds. Set either "
            "dro_ratio_bounds={'phi': (...), 'C_opt': (...), ...} or both "
            "dro_ratio_phi_bounds and dro_ratio_c_opt_bounds in "
            "DROFullPipelineConfig. Alternatively, run the optimal_cost_bounds "
            "tightening stage and provide dro_ratio_phi_bounds."
        )

    ratio_bounds: dict[str, Any] = {
        "phi": tuple(config.dro_ratio_phi_bounds),
        "C_opt": tuple(c_opt_bounds),
    }
    if mode == "ratio_piecewise_mccormick":
        if config.dro_ratio_c_opt_breakpoints is not None:
            ratio_bounds["C_opt_breakpoints"] = list(
                config.dro_ratio_c_opt_breakpoints
            )
        else:
            ratio_bounds["num_pieces"] = int(config.dro_ratio_num_pieces)
    return ratio_bounds


def load_dro_optimal_cost_bounds(
    config: DROFullPipelineConfig,
    regime_name: str,
) -> tuple[float, float] | None:
    output_stage_paths = resolved_stage_paths(config.tightening_output_paths, regime_name)
    previous_stage_paths = resolved_stage_paths(config.tightening_previous_paths, regime_name)
    candidate_paths = tuple(
        dict.fromkeys(
            [
                output_stage_paths["final"],
                output_stage_paths["optimal_cost_bounds"],
                previous_stage_paths["final"],
                previous_stage_paths["optimal_cost_bounds"],
            ]
        )
    )
    for path in candidate_paths:
        report_path = Path(path)
        if not report_path.exists():
            continue
        with report_path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        bounds = report.get("optimal_cost_bounds", {}) or {}
        if "C_opt" in bounds and isinstance(bounds.get("C_opt"), dict):
            bounds = bounds.get("C_opt", {}) or {}
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        if lower is not None and upper is not None:
            return (float(lower), float(upper))
    return None


def build_dro_optimizer(
    config: DROFullPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
    eta: float,
) -> DRO_PoAOptimization:
    ratio_bounds = build_dro_ratio_bounds(config, regime_name)
    return DRO_PoAOptimization(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=config.horizon,
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        regime_name=regime_name,
        eta=float(eta),
        epsilon=float(config.dro_wasserstein_epsilon),
        nn_model_dir=str(config.model_dir) if config.nn_policy_generators else None,
        nn_normalization_stats_path=(
            str(config.nn_normalization_stats_path)
            if config.nn_policy_generators
            else None
        ),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
        objective_mode=config.dro_objective_mode,
        ratio_bounds=ratio_bounds,
    )


def resolved_stage_paths(
    paths: dict[str, str | Path],
    regime_name: str,
) -> dict[str, str]:
    merged = {**DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS, **dict(paths)}
    return {
        stage_name: str(path_template).format(regime_name=regime_name)
        for stage_name, path_template in merged.items()
    }


def eta_label(eta: float) -> str:
    return f"{float(eta):.8g}".replace("-", "m").replace(".", "p")


def dro_result_path(
    config: DROFullPipelineConfig,
    regime_name: str,
    eta: float,
) -> Path:
    objective_suffix = (
        ""
        if str(config.dro_objective_mode).strip().lower() == "difference"
        else f"_{str(config.dro_objective_mode).strip().lower()}"
    )
    return (
        config.dro_result_dir
        / regime_name
        / f"dro_poa_eta_{eta_label(eta)}_T{config.horizon}{objective_suffix}.json"
    )


def load_result_summary(result_path: Path) -> dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as file_handle:
        result = json.load(file_handle)
    return {
        "result_path": str(result_path),
        "reference_case": result.get("reference_case"),
        "regime_set": result.get("regime_set"),
        "regime_name": result.get("regime_name"),
        "num_time_steps": result.get("num_time_steps"),
        "num_empirical_scenarios": result.get("num_empirical_scenarios"),
        "eta": result.get("eta"),
        "epsilon": result.get("epsilon"),
        "inner_objective": result.get("inner_objective"),
        "dro_objective_with_epsilon": result.get("dro_objective_with_epsilon"),
        "objective_mode": result.get("objective_mode"),
        "average_poa_difference": result.get("average_poa_difference"),
        "average_poa_ratio": result.get("average_poa_ratio", average_poa_ratio(result)),
        "average_relaxed_phi": result.get("average_relaxed_phi"),
        "average_wasserstein_distance": result.get("average_wasserstein_distance"),
        "solver": result.get("solver", {}),
    }


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_poa_ratio(result: dict[str, Any]) -> float | None:
    ratios = []
    for scenario in result.get("scenarios", []) or []:
        c_eq = optional_float(scenario.get("C_eq"))
        c_opt = optional_float(scenario.get("C_opt"))
        if c_eq is None or c_opt in (None, 0.0):
            continue
        ratios.append(c_eq / c_opt)
    if not ratios:
        return None
    return float(np.mean(ratios))


if __name__ == "__main__":
    run_config = DROFullPipelineConfig(
        # Case and random seeds.
        case="test_case_bidding_blocks",
        synthetic_time_steps=24,
        dro_time_steps=None,
        synthetic_seed=1,
        poa_seed=1,
        horizon=6,

        synthetic_scenarios_per_regime={
        "normal": 100,
        "high_demand": 100,
        "normal_peak_shift_wind": 100,
        "high_demand_peak_shift_wind": 100,
        },
        
        poa_context_scenarios_per_regime={
        "normal": 6,
        "high_demand": 10,
        "normal_peak_shift_wind": 10,
        "high_demand_peak_shift_wind": 10,
        },

        # Set to None to solve all generated regimes, or list regime names here.
        dro_regime_names=["normal"],
        # dro_regime_names=['normal'],
        # Eta grid for the final DRO sweep. Tightening is still run once per regime.
        etas=np.linspace(0, 0.5, 20).tolist(),
        dro_wasserstein_epsilon=0.1,

        # Objective modes:
        #   "difference"
        #   "ratio_mccormick"
        #   "ratio_piecewise_mccormick"
        # For ratio modes, set dro_ratio_phi_bounds. If dro_ratio_c_opt_bounds
        # is omitted, the pipeline reads C_opt bounds from the DRO
        # optimal_cost_bounds tightening stage.
        dro_objective_mode="ratio_piecewise_mccormick",
        dro_ratio_phi_bounds=(1.0, 5.0),
        dro_ratio_num_pieces=50,

        # Neural policy controls.
        nn_policy_generators=[1, 2],
        hidden_layers=[7, 7],
        num_epochs=500,

        # Solver controls.
        solver_name="gurobi",

        preprocessing_time_limit=200,
        dro_time_limit=None,
        poa_parallel_workers=6,
        poa_solver_threads_per_worker=None,

        archive_existing_dro_results = True,

        # Expensive upstream stages can be turned on when regenerating artifacts.
        run_scenario_generation=True,
        run_heuristic_labels=False,
        run_feature_building=False,
        run_nn_training=False,
        run_dro_tightening=True,
        run_dro_optimization=True,
        tightening_flags={
            "primal_big_m": True,
            "relu_bounds": True,
            "alpha_bounds": True,
            "slack_binary_fix": True,
            "dual_big_m": True,
            "optimal_cost_bounds": True,
        },
    )
    main(run_config)
