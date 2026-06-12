from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config.scenarios.scenario_generator import ScenarioManager
from driver.core.block0_core import (
    ALWAYS_ON_TIGHTENING_STAGES,
    DRO_TIGHTENING_FLAGS,
    ProjectConfig,
    dro_tightening_paths,
)
from driver.core.block1_core import apply_time_steps_override, write_json
from models.DRO_PoA.DRO_PoA_optimization import DRO_PoAOptimization
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import (
    DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS,
    DROPoATighteningMain,
)
from models.DRO_PoA.dro_poa_model.support_set import (
    DROWassersteinSupportSet,
    _ar1_kappa as _dro_ar1_kappa,
    _level_scale as _dro_level_scale,
)

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
class DROPoAPipelineConfig:
    case: str = "base_test_case"
    synthetic_time_steps: int | None = None
    poa_regime_set: str = "PoA_analysis_runtime"
    source_poa_regime_set: str = "PoA_analysis"
    synthetic_seed: int = 1
    poa_seed: int = 1
    synthetic_num_scenarios: int = 400
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"
    poa_context_scenarios_per_regime: dict[str, int] = field(default_factory=dict)
    dro_regime_names: list[str] | None = None
    bid_tolerance: float = 1e-2
    nn_feature_columns: list[str] = field(default_factory=list)
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
    horizon: int = 8
    etas: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])
    dro_wasserstein_epsilon: float = 0.0
    ambiguity_kappa: float = 0.3
    dro_tightening_eta: float = 0.0
    nn_policy_generators: list[int] = field(default_factory=lambda: [1, 2])
    dro_objective_mode: str = "piecewise_mccormick"
    dro_mccormick_bounds: dict[str, Any] | None = None
    dro_mccormick_PoA_bounds: tuple[float, float] | None = None
    dro_mccormick_c_opt_bounds: tuple[float, float] | None = None
    dro_mccormick_num_pieces: int = 4
    dro_mccormick_c_opt_breakpoints: list[float] | None = None
    solver_name: str = "gurobi"
    preprocessing_time_limit: int | None = 200
    dro_time_limit: int | None = 400
    # Cushion on the derived McCormick PoA upper bound.
    poa_bounds_derivation_margin: float = 1e-3
    slack_epsilon: float = 1e-6
    poa_parallel_workers: int = 1
    poa_solver_threads_per_worker: int | None = None
    ar1_coverage: float | None = None
    calibrate_support_coverage: bool = True
    support_coverage_grid: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    )
    support_calibration_report_path: Path | None = None
    support_verify_num_draws: int = 2000
    support_verify_seed: int = 77777
    plot_results_along_the_way: bool = False
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True
    run_dro_tightening: bool = True
    run_dro_optimization: bool = True
    tightening_flags: dict[str, bool] = field(default_factory=lambda: dict(DRO_TIGHTENING_FLAGS))
    tightening_previous_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS)
    )
    tightening_output_paths: dict[str, str | Path] = field(
        default_factory=lambda: dict(DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS)
    )
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

    def __post_init__(self) -> None:
        if self.synthetic_time_steps is None:
            self.synthetic_time_steps = self.horizon

    @property
    def nn_normalization_stats_path(self) -> Path:
        return self.normalized_feature_dir / "min_max_stats.json"


def build_dro_config(config: ProjectConfig) -> DROPoAPipelineConfig:
    return DROPoAPipelineConfig(
        case=config.case,
        synthetic_time_steps=config.synthetic_time_steps,
        poa_regime_set=config.poa_regime_set,
        source_poa_regime_set=config.poa_regime_set,
        synthetic_seed=config.synthetic_seed,
        poa_seed=config.poa_seed,
        synthetic_num_scenarios=config.synthetic_num_scenarios,
        ambiguity_set_config_path=config.ambiguity_set_config_path,
        ambiguity_set_config_name=config.ambiguity_set_config_name,
        poa_context_scenarios_per_regime={
            config.poa_worst_case_regime_name: config.poa_worst_case_n_scenarios
        },
        dro_regime_names=[config.poa_worst_case_regime_name],
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
        horizon=config.horizon,
        etas=list(config.etas),
        dro_wasserstein_epsilon=config.dro_wasserstein_epsilon,
        ambiguity_kappa=config.ambiguity_kappa,
        dro_tightening_eta=config.dro_tightening_eta,
        nn_policy_generators=list(config.nn_policy_generators),
        dro_objective_mode=config.dro_objective_mode,
        dro_mccormick_bounds=config.dro_mccormick_bounds,
        dro_mccormick_PoA_bounds=config.dro_mccormick_PoA_bounds,
        dro_mccormick_c_opt_bounds=config.dro_mccormick_c_opt_bounds,
        dro_mccormick_num_pieces=config.dro_mccormick_num_pieces,
        dro_mccormick_c_opt_breakpoints=config.dro_mccormick_c_opt_breakpoints,
        solver_name=config.solver_name,
        preprocessing_time_limit=config.preprocessing_time_limit,
        dro_time_limit=config.dro_time_limit,
        poa_bounds_derivation_margin=config.poa_bounds_derivation_margin,
        slack_epsilon=config.epsilon,
        poa_parallel_workers=config.poa_parallel_workers,
        poa_solver_threads_per_worker=config.poa_solver_threads_per_worker,
        ar1_coverage=config.ar1_coverage,
        calibrate_support_coverage=config.calibrate_support_coverage,
        support_coverage_grid=list(config.support_coverage_grid),
        support_calibration_report_path=(
            Path(config.support_calibration_report_path)
            if config.support_calibration_report_path is not None
            else None
        ),
        support_verify_num_draws=config.support_verify_num_draws,
        support_verify_seed=config.support_verify_seed,
        run_scenario_generation=False,
        run_heuristic_labels=False,
        run_feature_building=False,
        run_nn_training=False,
        run_dro_tightening=config.run_dro_tightening,
        run_dro_optimization=config.run_dro_optimization,
        tightening_flags=dict(config.dro_tightening_flags),
        tightening_previous_paths=dro_tightening_paths(Path(config.dro_result_dir)),
        tightening_output_paths=dro_tightening_paths(Path(config.dro_result_dir)),
        runtime_config_path=Path(config.runtime_config_path),
        synthetic_scenario_dir=Path(config.synthetic_scenario_dir),
        poa_scenario_dir=Path(config.dro_scenario_dir),
        heuristic_results_path=Path(config.heuristic_results_path),
        raw_feature_dir=Path(config.raw_feature_dir),
        normalized_feature_dir=Path(config.normalized_feature_dir),
        model_dir=Path(config.model_dir),
        training_result_dir=Path(config.training_result_dir),
        dro_result_dir=Path(config.dro_result_dir),
        archive_existing_dro_results=config.archive_existing_dro_results,
        dro_result_archive_dir=Path(config.dro_result_archive_dir),
    )


def _profile_values(value: Any, horizon: int) -> list[float]:
    import ast

    if isinstance(value, (list, np.ndarray)):
        values = value
    else:
        values = ast.literal_eval(str(value))
    return [float(item) for item in values[:horizon]]


def validate_scenarios_within_wasserstein_support(
    scenarios: dict[str, Any],
    manager: ScenarioManager,
    horizon: int,
    ar1_coverage: float,
    _verbose: bool = True,
) -> dict[str, Any]:
    kappa = _dro_ar1_kappa(horizon, ar1_coverage)
    scenarios_df = scenarios["scenarios_df"].copy().reset_index(drop=True)
    demand_ref_scalar = float(manager.base_case["demand"])
    demand_shape = ScenarioManager._build_demand_shape(horizon)
    wind_generators = [g for g in manager.physical_generators if bool(g["is_wind"])]
    wind_blocks_by_generator = {
        str(g["physical_name"]): [
            block
            for block in manager.blocks
            if block["physical_name"] == g["physical_name"] and bool(block["is_wind"])
        ]
        for g in wind_generators
    }

    rejected_reasons: dict[int, str] = {}
    for scenario_idx, row in scenarios_df.iterrows():
        mu_D = float(row["mu_D"])
        sigma_D = float(row["sigma_D"])
        rho_D = float(row["rho_D"])
        mu_W = float(row["mu_W"])
        sigma_W = float(row["sigma_W"])
        rho_W = float(row["rho_W"])
        peak_W = float(row["peak_W"])
        wind_shape = ScenarioManager._build_wind_shape(horizon, peak_W)

        demand = np.asarray(_profile_values(row["demand_profile"], horizon), dtype=float)
        demand_ref = demand_ref_scalar * mu_D * demand_shape
        demand_threshold = kappa * demand_ref_scalar * sigma_D
        # Time-varying level margin: kappa * sigma * _level_scale(rho, t), matching the model.
        demand_level_margins = demand_threshold * np.array(
            [_dro_level_scale(rho_D, t) for t in range(horizon)]
        )
        if abs(demand[0] - demand_ref[0]) > demand_threshold + 1e-9:
            rejected_reasons[int(scenario_idx)] = "demand_innov_t0"
            continue
        if horizon > 1:
            ar1_ref = demand_ref_scalar * mu_D * (demand_shape[1:] - rho_D * demand_shape[:-1])
            innov = demand[1:] - rho_D * demand[:-1] - ar1_ref
            if bool(np.any(np.abs(innov) > demand_threshold + 1e-9)):
                rejected_reasons[int(scenario_idx)] = "demand_innov"
                continue
        if bool(np.any(np.abs(demand - demand_ref) > demand_level_margins + 1e-9)):
            rejected_reasons[int(scenario_idx)] = "demand_level"
            continue

        wind_level_scales = np.array([_dro_level_scale(rho_W, t) for t in range(horizon)])
        for generator in wind_generators:
            generator_name = str(generator["physical_name"])
            capacity = float(generator["pmax"])
            wind = np.zeros(horizon, dtype=float)
            for block in wind_blocks_by_generator[generator_name]:
                column = f"{block['block_name']}_profile"
                if column in scenarios_df.columns:
                    wind += np.asarray(_profile_values(row[column], horizon), dtype=float)

            wind_ref = capacity * mu_W * wind_shape
            wind_threshold = kappa * capacity * sigma_W
            wind_level_margins = wind_threshold * wind_level_scales
            if abs(wind[0] - wind_ref[0]) > wind_threshold + 1e-9:
                rejected_reasons[int(scenario_idx)] = f"wind_{generator_name}_innov_t0"
                break
            if horizon > 1:
                ar1_ref = capacity * mu_W * (wind_shape[1:] - rho_W * wind_shape[:-1])
                innov = wind[1:] - rho_W * wind[:-1] - ar1_ref
                if bool(np.any(np.abs(innov) > wind_threshold + 1e-9)):
                    rejected_reasons[int(scenario_idx)] = f"wind_{generator_name}_innov"
                    break
            if bool(np.any(np.abs(wind - wind_ref) > wind_level_margins + 1e-9)):
                rejected_reasons[int(scenario_idx)] = f"wind_{generator_name}_level"
                break

    if not rejected_reasons:
        if _verbose:
            print(
                "DRO support validation: 0 empirical scenarios dropped "
                f"(coverage={ar1_coverage}, kappa={kappa:.4f})."
            )
        return scenarios

    total = len(scenarios_df)
    rejected = sorted(rejected_reasons)
    reason_counts: dict[str, int] = {}
    for reason in rejected_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if _verbose:
        print(
            "DRO support validation: "
            f"{len(rejected)}/{total} empirical scenarios outside support bands "
            f"(coverage={ar1_coverage}, kappa={kappa:.4f}); dropping them."
        )
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")
    valid_idx = [idx for idx in range(total) if idx not in set(rejected)]
    if not valid_idx:
        raise RuntimeError(
            "All empirical scenarios were rejected by the DRO support bands. "
            "Increase ar1_coverage or loosen the support configuration."
        )
    return {**scenarios, "scenarios_df": scenarios_df.iloc[valid_idx].reset_index(drop=True)}


def calibrate_ar1_coverage_from_scenarios(
    scenarios: dict[str, Any],
    manager: ScenarioManager,
    horizon: int,
    coverage_grid: list[float],
    report_path: "Path | None" = None,
) -> tuple[float, dict[str, Any]]:
    """Find the smallest coverage in coverage_grid at which ALL empirical scenarios
    lie inside the AR(1) support tube (innovation + time-varying level bands).

    Returns (chosen_coverage, report_dict). Raises RuntimeError if no grid value
    achieves 0 outside scenarios. Saves a JSON report to report_path if given.
    """
    sorted_grid = sorted(coverage_grid)
    per_coverage_outside: dict[str, int] = {}
    chosen: float | None = None

    for coverage in sorted_grid:
        validated = validate_scenarios_within_wasserstein_support(
            scenarios=scenarios,
            manager=manager,
            horizon=horizon,
            ar1_coverage=coverage,
            _verbose=False,
        )
        n_outside = len(scenarios["scenarios_df"]) - len(validated["scenarios_df"])
        per_coverage_outside[str(coverage)] = n_outside
        if n_outside == 0:
            chosen = coverage
            break

    if chosen is None:
        raise RuntimeError(
            "No coverage in support_coverage_grid achieves 0 scenarios outside the "
            f"support tube. Grid: {sorted_grid}. Outside counts: {per_coverage_outside}. "
            "Extend support_coverage_grid or relax the ambiguity-set configuration."
        )

    report: dict[str, Any] = {
        "chosen_coverage": chosen,
        "chosen_kappa": _dro_ar1_kappa(horizon, chosen),
        "n_empirical_scenarios": len(scenarios["scenarios_df"]),
        "per_coverage_outside": per_coverage_outside,
        "coverage_grid": sorted_grid,
    }
    print(
        f"[support calibration] coverage={chosen}  "
        f"kappa={report['chosen_kappa']:.4f}  "
        f"(first grid level with 0/{report['n_empirical_scenarios']} scenarios outside)"
    )
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(report_path).open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return chosen, report


def load_dro_scenario_data(config: DROPoAPipelineConfig) -> dict[str, Any]:
    if not config.runtime_config_path.exists():
        raise FileNotFoundError(
            "DRO runtime regime config is missing. Run Block 3 first: "
            f"{config.runtime_config_path}"
        )
    scenario_manager = ScenarioManager(config.case)
    apply_time_steps_override(scenario_manager, config.horizon)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        seed=config.poa_seed,
        enforce_support_set=False,
    )

    if config.ar1_coverage is not None:
        ar1_coverage = float(config.ar1_coverage)
    elif config.calibrate_support_coverage:
        ar1_coverage, _ = calibrate_ar1_coverage_from_scenarios(
            scenarios=scenarios,
            manager=scenario_manager,
            horizon=config.horizon,
            coverage_grid=list(config.support_coverage_grid),
            report_path=config.support_calibration_report_path,
        )
        # Propagate the calibrated value so tightening and optimisation use it.
        # Also update the class-level default so any call site that reads
        # DROWassersteinSupportSet.AR1_JOINT_COVERAGE as a fallback gets the
        # calibrated value rather than the hard-coded 0.99.
        config.ar1_coverage = ar1_coverage
        DROWassersteinSupportSet.AR1_JOINT_COVERAGE = ar1_coverage
    else:
        raise ValueError(
            "ar1_coverage is not set and calibrate_support_coverage=False. "
            "Either provide ar1_coverage explicitly or set calibrate_support_coverage=True."
        )

    scenarios = validate_scenarios_within_wasserstein_support(
        scenarios=scenarios,
        manager=scenario_manager,
        horizon=config.horizon,
        ar1_coverage=ar1_coverage,
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
    config: DROPoAPipelineConfig,
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
    config: DROPoAPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
) -> DROPoATighteningMain:
    objective_mode = str(config.dro_objective_mode).strip().lower()
    return DROPoATighteningMain(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=config.horizon,
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        regime_name=regime_name,
        eta=config.dro_tightening_eta,
        epsilon=config.dro_wasserstein_epsilon,
        nn_model_dir=str(config.model_dir) if config.nn_policy_generators else None,
        nn_normalization_stats_path=(
            str(config.nn_normalization_stats_path) if config.nn_policy_generators else None
        ),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
        case_label=getattr(config, "case_label", ""),
        objective_mode=objective_mode,
        mccormick_bounds=build_dro_tightening_mccormick_bounds(config),
        ambiguity_kappa=config.ambiguity_kappa,
        use_default_bounds=(objective_mode != "difference"),
        ar1_coverage=config.ar1_coverage,
    )


def run_dro_tightening_for_regime(
    config: DROPoAPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
) -> Path:
    flags = {**DRO_TIGHTENING_FLAGS, **dict(config.tightening_flags)}
    # primal_big_m and optimal_cost_bounds always run, regardless of config.
    for stage_name in ALWAYS_ON_TIGHTENING_STAGES:
        flags[stage_name] = True
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
        run_equilibrium_cost_bounds=bool(flags.get("equilibrium_cost_bounds", False)),
        poa_bounds_lower=(
            config.dro_mccormick_PoA_bounds[0] if config.dro_mccormick_PoA_bounds else 1.0
        ),
        poa_bounds_margin=config.poa_bounds_derivation_margin,
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


def run_dro_essential_bounds_for_regime(
    config: DROPoAPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
) -> Path:
    """Compute only primal_big_m and optimal_cost_bounds for one regime.

    Used when full DRO tightening is disabled but the McCormick C_opt envelope
    still needs valid denominator bounds. The remaining stages reuse existing
    reports when present and otherwise fall back to the final solve's loose
    defaults.
    """
    previous_paths = resolved_stage_paths(config.tightening_previous_paths, regime_name)
    output_paths = resolved_stage_paths(config.tightening_output_paths, regime_name)

    print(
        f"\nComputing essential DRO bounds (primal_big_m, optimal_cost_bounds) "
        f"for regime '{regime_name}'"
    )

    start = time.perf_counter()
    tightening = build_dro_tightening(config, scenarios, regime_name)
    final_report_path = tightening.run_essential_bounds(
        previous_paths=previous_paths,
        output_paths=output_paths,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    elapsed = time.perf_counter() - start
    print(f"\nDRO essential-bound computation complete: {final_report_path}")
    print(f"  Regime: {regime_name}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    return final_report_path


def print_tightening_plan(
    config: DROPoAPipelineConfig,
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


def _print_wasserstein_floor_diagnostic(
    rows: list[dict[str, Any]],
    regime_name: str,
) -> None:
    """Print the minimum achievable Wasserstein distance for each empirical scenario.

    If all values are 0, the support set is properly enforced and Wâ†’0 is feasible
    at sufficiently high eta.  Non-zero values mean the empirical scenario lies
    outside the support set and W cannot reach 0 regardless of eta.

    ``rows`` is the (eta-independent) diagnostic from
    DROPoASupportDiagnostics.diagnose_empirical_support_set_violations, passed in
    so it can be computed once per regime and reused across the eta sweep.
    """
    any_violation = any(row["min_W_total"] > 1e-9 for row in rows)
    print(f"\n  Wasserstein floor diagnostic (regime '{regime_name}'):")
    for row in rows:
        flag = " <-- VIOLATION" if row["min_W_total"] > 1e-9 else ""
        print(
            f"    k={row['scenario_k']:2d}  min_W={row['min_W_total']:.4f}"
            f"  (demand: pw={row['demand_pointwise_violations']}"
            f" t0={row.get('demand_t0_violations', 0)}"
            f" ar1={row['demand_ar1_violations']}"
            f"  wind: level={row.get('wind_level_violations', 0)}"
            f" t0={row.get('wind_t0_violations', 0)}"
            f" ar1={row['wind_ar1_violations']}){flag}"
        )
    if any_violation:
        print(
            "  WARNING: some empirical scenarios are outside the support set. "
            "W cannot reach 0 at any eta.  Re-generate scenarios with "
            "enforce_support_set=True."
        )
    else:
        print(
            "  All empirical scenarios are within the support set. "
            "W=0 is feasible; increase eta if W is still large at solve time."
        )


def run_dro_eta_sweep(
    config: DROPoAPipelineConfig,
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
        summaries.extend(
            run_dro_eta_path_for_regime(
                config=config,
                scenarios=scenarios,
                regime_name=regime_name,
                tightening_report_path=tightening_report_path,
            )
        )
    return summaries


def _snapshot_solution(optimizer: DRO_PoAOptimization) -> list:
    """Save all current model variable values so they can be restored as a warm start."""
    from pyomo.environ import Var

    return [
        (var_data, var_data.value)
        for var_data in optimizer.model.component_data_objects(Var, active=True)
        if var_data.value is not None
    ]


def _restore_solution(optimizer: DRO_PoAOptimization, snapshot: list) -> None:
    """Write a saved snapshot back into the model variables for warm-start reuse."""
    for var_data, value in snapshot:
        var_data.set_value(value)


def _set_poa_upper_cut(optimizer: DRO_PoAOptimization, upper_bound: float) -> None:
    """Add/replace an average-PoA upper bound constraint in the live persistent model.

    After the eta_min=0 solve, calling this caps the LP relaxation at upper_bound,
    bringing Gurobi's BestBd down from the loose variable bound (10) to the actual
    optimum of the least-penalised problem for all subsequent higher-eta solves.
    """
    from pyomo.environ import Constraint as _Constraint

    m = optimizer.model
    solver = optimizer._persistent_solver
    K = optimizer.num_empirical_scenarios

    if hasattr(m, "poa_upper_cut"):
        for c in m.poa_upper_cut.values():
            solver.remove_constraint(c)
        m.del_component(m.poa_upper_cut)

    m.poa_upper_cut = _Constraint(expr=sum(m.PoA[k] for k in m.scenarios) / K <= upper_bound + 1e-4)
    for c in m.poa_upper_cut.values():
        solver.add_constraint(c)
    print(f"  PoA upper cut added: avg PoA <= {upper_bound + 1e-4:.6g}")


def _poa_scenario_caps_from_eta0(optimizer: DRO_PoAOptimization) -> dict[int, float]:
    """Per-scenario PoA upper bounds derived from the eta=0 solution.

    At eta=0 the scenario blocks are independent and the objective is the plain
    average PoA, so the optimum maximizes every PoA[k] individually. Each PoA[k]
    can exceed its incumbent value by at most K times the solve's absolute gap
    (otherwise holding the other scenarios at their incumbents would beat the
    proven bound), so PoA[k] <= value + K * gap is valid. The support floor
    PoA[k] - eta * W[k] >= 1 only shrinks the feasible set as eta grows, so the
    caps remain valid for every eta > 0.

    Returns an empty dict when the certified bound or a PoA value is missing.
    """
    from pyomo.environ import value as _value

    m = optimizer.model
    if not hasattr(m, "PoA"):
        return {}
    bound = optional_float(getattr(optimizer, "best_objective_bound", None))
    incumbent = optional_float(_value(m.objective, exception=False))
    if bound is None or incumbent is None:
        return {}
    K = optimizer.num_empirical_scenarios
    slack = max(0.0, bound - incumbent) * K + 1e-4
    caps: dict[int, float] = {}
    for k in m.scenarios:
        poa_k = optional_float(_value(m.PoA[k], exception=False))
        if poa_k is None:
            return {}
        caps[int(k)] = poa_k + slack
    return caps


def _set_poa_scenario_caps(
    optimizer: DRO_PoAOptimization,
    caps: dict[int, float],
) -> None:
    """Tighten each PoA[k] variable upper bound in the live persistent model.

    Unlike the average PoA cut, these pull the LP relaxation down per scenario,
    so BestBd no longer sits at the loose variable bound for scenarios whose
    individual worst-case PoA is far below the maximum.
    """
    m = optimizer.model
    solver = optimizer._persistent_solver
    applied = 0
    for k, cap in caps.items():
        var = m.PoA[k]
        if var.ub is None or cap < float(var.ub):
            var.setub(cap)
            solver.update_var(var)
            applied += 1
    print(f"  Per-scenario PoA caps tightened: {applied}/{len(caps)}")


def run_dro_eta_path_for_regime(
    config: DROPoAPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
    tightening_report_path: Path,
) -> list[dict[str, Any]]:
    """Solve the full eta path for one regime, reusing a single persistent model.

    Every eta shares the same model structure -- only the objective term
    -eta * W[k] and the support-floor constraints PoA[k] - eta * W[k] >= 1 depend
    on eta -- so the model is built and loaded into Gurobi once, then updated in
    place per eta (see DRO_PoAOptimization.update_eta).

    Solve order: eta_max first (cold), then eta_min (warm), then interior descending.

    eta_max is the most penalised/constrained problem and gives the lowest objective
    value -- a valid Gurobi Cutoff for every subsequent solve (the feasible set only
    grows as eta decreases, so every later optimum is >= the eta_max optimum). The
    Cutoff is then ratcheted up after every interior solve, since each optimum is a
    valid lower bound for all smaller etas.

    eta_min=0 is the least penalised and gives the highest average PoA -- a valid
    upper bound on m.PoA[k] for all higher-eta solves, which tightens Gurobi's LP
    relaxation BestBd from the loose variable bound (10) down to the actual optimum.
    Because the scenarios decouple at eta=0, the same solve also yields valid
    per-scenario PoA caps (see _poa_scenario_caps_from_eta0).

    Interior etas are then solved descending, each warm-starting from the previous.
    """
    unique_etas = sorted({float(eta) for eta in config.etas}, reverse=True)
    eta_max = unique_etas[0]
    eta_min = unique_etas[-1]

    if len(unique_etas) == 1 or eta_max == eta_min:
        # Single eta: no bounding strategy, just solve once.
        ordered_etas = unique_etas
    else:
        # Solve order: [eta_max, eta_min, interior descending]
        interior = unique_etas[1:-1]  # already descending, excluding endpoints
        ordered_etas = [eta_max, eta_min] + interior

    # Build the model once (at eta_max), apply tightening, load into persistent solver.
    optimizer = build_dro_optimizer(config, scenarios, regime_name, eta=eta_max)
    optimizer.load_regime_wide_tightening_report(tightening_report_path)
    optimizer.build_model()
    applied_stats = optimizer.apply_regime_wide_tightening_to_model()
    optimizer.attach_persistent_solver()

    _print_wasserstein_floor_diagnostic(optimizer.support_set_diagnostics(), regime_name)

    summaries: list[dict[str, Any]] = []
    eta_count = len(ordered_etas)
    eta_max_snapshot: dict = {}
    best_cutoff: float | None = None

    for position, eta in enumerate(ordered_etas):
        is_first_solve = position == 0  # eta_max
        is_second_solve = position == 1  # eta_min
        is_interior = position >= 2

        if not is_first_solve:
            optimizer.update_eta(eta)

        # The first interior solve restores the eta_max snapshot to begin the
        # descent from a feasible, empirically-grounded point (eta_min's solution
        # can have large W[k] that violate the support floor at eta > 0).
        # Subsequent interior solves chain from the previous one.
        if position == 2 and eta_max_snapshot:
            _restore_solution(optimizer, eta_max_snapshot)

        result_path = solve_and_save_dro_for_eta(
            optimizer=optimizer,
            config=config,
            regime_name=regime_name,
            eta=eta,
            applied_stats=applied_stats,
            tightening_report_path=tightening_report_path,
            from_empirical_start=is_first_solve,
            eta_index=position + 1,
            eta_count=eta_count,
        )
        summary = load_result_summary(result_path)
        summaries.append(summary)

        if is_first_solve:
            # Save eta_max solution for all interior warm starts.
            eta_max_snapshot = _snapshot_solution(optimizer)

        if is_first_solve or is_interior:
            # Descending etas: each incumbent objective is a valid lower bound
            # for every smaller eta (the feasible set only grows as eta
            # decreases), so ratchet the Cutoff up with the latest solve.
            # eta_min is excluded: its optimum is the path maximum, not a
            # lower bound for the interior etas solved after it.
            lb = optional_float(summary.get("inner_objective"))
            if lb is not None and (best_cutoff is None or lb > best_cutoff):
                best_cutoff = lb
                optimizer._persistent_solver.options["Cutoff"] = lb - 1e-4
                print(f"  Cutoff (lower bound) set to: {lb:.6g}")

        if is_second_solve:
            # eta=0 gives the highest reachable average PoA; cap m.PoA[k] so
            # the LP relaxation BestBd reflects this rather than the loose
            # variable bound (10).
            ub = optional_float(summary.get("average_relaxed_PoA"))
            if ub is None:
                ub = optional_float(summary.get("inner_objective"))
            if float(eta) == 0.0:
                # At eta=0 the objective is exactly the average PoA, so the
                # certified solver bound is a valid cap even if this solve hit
                # the time limit (where the incumbent understates the max).
                bound_ub = optional_float((summary.get("solver") or {}).get("best_objective_bound"))
                if bound_ub is not None:
                    ub = bound_ub
            if ub is not None:
                _set_poa_upper_cut(optimizer, ub)
            # At eta=0 the scenarios decouple, so the solution also yields
            # valid per-scenario PoA caps -- much stronger than the average
            # cut alone for the small-eta solves that follow.
            if float(eta) == 0.0:
                scenario_caps = _poa_scenario_caps_from_eta0(optimizer)
                if scenario_caps:
                    _set_poa_scenario_caps(optimizer, scenario_caps)

    # Report in ascending eta order so the summary reads naturally.
    summaries.sort(key=lambda s: float(s["eta"]) if s.get("eta") is not None else float("inf"))
    return summaries


def archive_existing_dro_result_folders(
    config: DROPoAPipelineConfig,
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


def save_dro_solve_log_artifacts(
    optimizer: DRO_PoAOptimization,
    result_path: Path,
) -> Path | None:
    """Persist the per-eta Gurobi log and parsed bound progression.

    Writes two sidecar files under ``<regime_dir>/solve_logs/``: the raw Gurobi
    log segment for this solve (``<result>_gurobi.log``) and a JSON time series
    of incumbent/BestBd/gap over solve time (``<result>_progress.json``), so
    bound movement and computation time can be plotted without re-running.
    The subfolder keeps them out of the ``dro_poa_eta_*_T*.json`` result globs
    used by the eta-sweep plotting.
    """
    log_text = getattr(optimizer, "last_solve_gurobi_log", None)
    progression = getattr(optimizer, "solve_bound_progression", None) or []
    if not log_text and not progression:
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
        "eta": float(optimizer.eta),
        "regime_name": optimizer.regime_name,
        "objective_mode": optimizer.objective_mode,
        "num_time_steps": int(optimizer.num_time_steps),
        "termination_condition": termination,
        "wall_time_seconds": optional_float(getattr(optimizer, "solve_wall_time_seconds", None)),
        "best_objective_bound": optional_float(getattr(optimizer, "best_objective_bound", None)),
        "mip_gap": optional_float(getattr(optimizer, "mip_gap", None)),
        "bound_progression": progression,
    }
    progress_path = log_dir / f"{stem}_progress.json"
    with progress_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return progress_path


def _print_dro_solve_banner(
    regime_name: str,
    eta: float,
    eta_index: int | None,
    eta_count: int | None,
    objective_mode: str,
    from_empirical_start: bool,
) -> None:
    """Print a clear header before the Gurobi log so each solve is identifiable.

    The Gurobi solver output (tee=True) is otherwise an undifferentiated wall of
    text; this banner labels which regime/eta/objective is being solved.
    """
    progress = (
        f" [{eta_index}/{eta_count}]" if eta_index is not None and eta_count is not None else ""
    )
    start_kind = "empirical MIP start" if from_empirical_start else "warm start (previous eta)"
    print("\n" + "=" * 78)
    print(f"SOLVING DRO PoA{progress}  |  regime: {regime_name}  |  eta: {float(eta):.6g}")
    print(f"  objective mode: {objective_mode}  |  start: {start_kind}")
    print("=" * 78)


def solve_and_save_dro_for_eta(
    optimizer: DRO_PoAOptimization,
    config: DROPoAPipelineConfig,
    regime_name: str,
    eta: float,
    applied_stats: dict[str, Any],
    tightening_report_path: Path,
    from_empirical_start: bool,
    eta_index: int | None = None,
    eta_count: int | None = None,
) -> Path:
    """Solve one eta on an already-built optimizer and save its results.

    The optimizer is built, tightened, and loaded into the persistent solver once
    per regime by run_dro_eta_path_for_regime; this function only runs the solve
    and writes the result.  The solve always warm starts: from the empirical MIP
    start on the first eta, or from the previous eta's optimum afterwards.
    """
    result_path = dro_result_path(config, regime_name, eta)

    _print_dro_solve_banner(
        regime_name=regime_name,
        eta=eta,
        eta_index=eta_index,
        eta_count=eta_count,
        objective_mode=optimizer.objective_mode,
        from_empirical_start=from_empirical_start,
    )

    start = time.perf_counter()
    optimizer.solve(time_limit=config.dro_time_limit, warm_start=True)
    output_path = optimizer.save_results(result_path)
    progress_path = save_dro_solve_log_artifacts(optimizer, output_path)
    elapsed = time.perf_counter() - start

    print(f"\nDRO PoA optimization complete: {output_path}")
    print(f"  Regime: {regime_name}")
    print(f"  Eta: {eta}")
    bound = optional_float(getattr(optimizer, "best_objective_bound", None))
    gap = optional_float(getattr(optimizer, "mip_gap", None))
    if bound is not None:
        gap_note = f"  (MIP gap {gap:.2%})" if gap is not None else ""
        print(f"  Best objective bound: {bound:.6g}{gap_note}")
    if progress_path is not None:
        print(f"  Solve log + bound progression: {progress_path.parent}")
    print(f"  Start: {'empirical MIP start' if from_empirical_start else 'previous eta'}")
    print(f"  Objective mode: {optimizer.objective_mode}")
    if optimizer.objective_mode != "difference":
        summary = optimizer.solution_summary()
        print(f"  Average ex-post ratio: {summary.get('average_poa_ratio')}")
        print(f"  Average relaxed PoA: {summary.get('average_relaxed_PoA')}")
    print(f"  Reused tightening report: {tightening_report_path}")
    print(f"  Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"  Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"  Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    return output_path


def build_dro_mccormick_bounds(
    config: DROPoAPipelineConfig,
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

    if config.dro_mccormick_bounds is not None:
        return dict(config.dro_mccormick_bounds)

    c_opt_bounds = config.dro_mccormick_c_opt_bounds
    if c_opt_bounds is None:
        c_opt_bounds = load_dro_optimal_cost_bounds(config, regime_name)

    if c_opt_bounds is None:
        raise ValueError(
            "DRO mccormick objective modes require C_opt bounds. Set "
            "dro_mccormick_bounds={'PoA': (...), 'C_opt': (...), ...} or "
            "dro_mccormick_c_opt_bounds in DROFullPipelineConfig, or run/load the "
            "optimal_cost_bounds tightening stage."
        )

    poa_bounds = resolve_dro_mccormick_poa_box(config, regime_name)

    mccormick_bounds: dict[str, Any] = {
        "PoA": poa_bounds,
        "C_opt": tuple(c_opt_bounds),
    }
    if mode == "piecewise_mccormick":
        if config.dro_mccormick_c_opt_breakpoints is not None:
            mccormick_bounds["C_opt_breakpoints"] = list(config.dro_mccormick_c_opt_breakpoints)
        else:
            mccormick_bounds["num_pieces"] = int(config.dro_mccormick_num_pieces)
    return mccormick_bounds


def default_dro_mccormick_c_opt_bounds() -> tuple[float, float]:
    return (
        float(DRO_PoAOptimization.DEFAULT_LOOSE_C_OPT_LOWER),
        float(DRO_PoAOptimization.DEFAULT_LOOSE_C_OPT_UPPER),
    )


def build_dro_tightening_mccormick_bounds(
    config: DROPoAPipelineConfig,
) -> dict[str, Any] | None:
    mode = str(config.dro_objective_mode).strip().lower()
    if mode == "difference":
        return None
    if config.dro_mccormick_bounds is not None:
        return dict(config.dro_mccormick_bounds)

    mccormick_bounds: dict[str, Any] = {
        "C_opt": default_dro_mccormick_c_opt_bounds(),
    }
    if config.dro_mccormick_PoA_bounds is not None:
        mccormick_bounds["PoA"] = tuple(config.dro_mccormick_PoA_bounds)
    if mode == "piecewise_mccormick":
        mccormick_bounds["num_pieces"] = int(config.dro_mccormick_num_pieces)
    return mccormick_bounds


def load_dro_optimal_cost_bounds(
    config: DROPoAPipelineConfig,
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


def load_dro_derived_poa_bounds(
    config: DROPoAPipelineConfig,
    regime_name: str,
) -> tuple[float, float] | None:
    """Read the derived PoA box [PoA_L, PoA_U] from the regime's tightening report, or None.

    The box is computed and stored by the equilibrium_cost_bounds stage
    (PoA_U = max C_eq / min C_opt), so the final solve only reads it here.
    """
    output_stage_paths = resolved_stage_paths(config.tightening_output_paths, regime_name)
    previous_stage_paths = resolved_stage_paths(config.tightening_previous_paths, regime_name)
    candidate_paths = tuple(
        dict.fromkeys(
            [
                output_stage_paths["final"],
                output_stage_paths["equilibrium_cost_bounds"],
                previous_stage_paths["final"],
                previous_stage_paths["equilibrium_cost_bounds"],
            ]
        )
    )
    for path in candidate_paths:
        report_path = Path(path)
        if not report_path.exists():
            continue
        with report_path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        box = report.get("derived_poa_bounds")
        if isinstance(box, (list, tuple)) and len(box) == 2:
            return (float(box[0]), float(box[1]))
    return None


def resolve_dro_mccormick_poa_box(
    config: DROPoAPipelineConfig,
    regime_name: str,
) -> tuple[float, float]:
    """Resolve the McCormick PoA box (PoA_L, PoA_U) for the DRO solve of one regime.

    If the equilibrium_cost_bounds stage stored a ``derived_poa_bounds`` box
    (PoA_U = max C_eq / min C_opt), use it. Otherwise fall back to the hand-set
    ``dro_mccormick_PoA_bounds`` (e.g. (1, 20)).
    """
    derived = load_dro_derived_poa_bounds(config, regime_name)
    if derived is not None:
        print(
            f"[block4][{regime_name}] using derived McCormick PoA box from tightening "
            f"report: ({derived[0]:.6g}, {derived[1]:.6g})"
        )
        return derived
    config_box = (
        tuple(config.dro_mccormick_PoA_bounds)
        if config.dro_mccormick_PoA_bounds is not None
        else None
    )
    if config_box is None:
        raise ValueError(
            "DRO mccormick objective modes require a PoA box. Set dro_mccormick_PoA_bounds "
            "(e.g. (1.0, 50.0)) or run the equilibrium_cost_bounds tightening stage."
        )
    return config_box


def build_dro_optimizer(
    config: DROPoAPipelineConfig,
    scenarios: dict[str, Any],
    regime_name: str,
    eta: float,
) -> DRO_PoAOptimization:
    mccormick_bounds = build_dro_mccormick_bounds(config, regime_name)
    optimizer = DRO_PoAOptimization(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=config.horizon,
        regime_config_path=str(config.runtime_config_path),
        regime_set=config.poa_regime_set,
        regime_name=regime_name,
        eta=float(eta),
        epsilon=float(config.dro_wasserstein_epsilon),
        nn_model_dir=str(config.model_dir) if config.nn_policy_generators else None,
        nn_normalization_stats_path=(
            str(config.nn_normalization_stats_path) if config.nn_policy_generators else None
        ),
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
        case_label=getattr(config, "case_label", ""),
        objective_mode=config.dro_objective_mode,
        mccormick_bounds=mccormick_bounds,
        ambiguity_kappa=config.ambiguity_kappa,
        ar1_coverage=config.ar1_coverage,
    )
    return optimizer


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
    config: DROPoAPipelineConfig,
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
        "average_relaxed_PoA": result.get("average_relaxed_PoA"),
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
