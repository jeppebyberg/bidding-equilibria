"""Sensitivity pipeline: base PoA → DRO eta sweep on the worst-case regime.

Runs shared pre-processing (scenario generation, heuristic labels, feature
building, NN training) once, then:

  1. Base PoA optimization over the ambiguity-set context scenario.
  2. Extracts the AR(1) regime parameters (mu_D, sigma_D, rho_D, mu_W, ...)
     from the PoA context scenario — i.e. the market state the PoA upper
     level chose as worst case.
  3. Writes a custom runtime regime definition with those parameters.
  4. DRO eta sweep over that extracted regime.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from driver.run_full_pipeline import (
    TIGHTENING_FLAGS as POA_TIGHTENING_FLAGS,
    TIGHTENING_OUTPUT_PATHS as POA_TIGHTENING_OUTPUT_PATHS,
    TIGHTENING_PREVIOUS_PATHS as POA_TIGHTENING_PREVIOUS_PATHS,
    FullPipelineConfig,
    build_features,
    load_or_generate_scenarios,
    run_final_poa,
    run_heuristic,
    run_tightening_pipeline,
    train_policies,
    write_json,
)
from driver.run_full_pipeline_DRO import (
    DRO_TIGHTENING_FLAGS,
    DROFullPipelineConfig,
    archive_existing_dro_result_folders,
    load_dro_scenario_data,
    resolve_dro_regime_names,
    run_dro_eta_sweep,
    run_dro_tightening_for_regime,
    run_support_calibration,
)
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import (
    DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS,
)

# AR(1) parameter columns present in the scenarios_df / scenarios.csv.
_REGIME_PARAM_COLUMNS = ("mu_D", "rho_D", "sigma_D", "mu_W", "rho_W", "sigma_W", "peak_W")


@dataclass
class SensitivityPipelineConfig:
    # -----------------------------------------------------------------------
    # Shared / common
    # -----------------------------------------------------------------------
    case: str = "base_test_case"
    synthetic_time_steps: int | None = 24
    synthetic_seed: int = 1
    poa_seed: int = 1

    synthetic_num_scenarios: int = 500
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"

    bid_tolerance: float = 1e-2

    # Neural-network feature and training parameters.
    nn_feature_columns: list[str] = field(
        default_factory=lambda: [
            "demand",
            "total_wind_generation_capacity",
            "total_generation_capacity",
            "residual_demand",
            "previous_generation_capacity",
            "previous_demand",
            "next_generation_capacity",
            "next_demand",
            "own_generation_capacity",
            "previous_own_generation_capacity",
            "next_own_generation_capacity",
        ]
    )
    per_generator_normalization: bool = True
    hidden_layers: list[int] = field(default_factory=lambda: [4, 8])
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 500
    weight_decay: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    patience: int | None = 50
    min_delta: float = 1e-6
    device: str | None = None
    nn_final_activation: str = "linear"

    horizon: int = 8
    nn_policy_generators: list[str] = field(
        default_factory=lambda: ["G1", "W2", "W3"]
    )

    solver_name: str = "gurobi"
    preprocessing_time_limit: int = 200
    epsilon: float = 1e-6
    poa_parallel_workers: int = 6
    poa_solver_threads_per_worker: int | None = None

    # -----------------------------------------------------------------------
    # PoA-specific
    # -----------------------------------------------------------------------
    poa_context_num_scenarios: int = 1
    poa_objective_mode: str = "piecewise_mccormick"
    poa_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    poa_mccormick_num_pieces: int = 50
    poa_time_limit: int | None = None
    run_poa_tightening: bool = True
    poa_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {k: True for k in POA_TIGHTENING_FLAGS}
    )
    run_poa_optimization: bool = True
    poa_result_dir: Path = Path("results/sensitivity_pipeline/poa")

    # -----------------------------------------------------------------------
    # DRO-specific
    # -----------------------------------------------------------------------
    # Name given to the custom regime built from the PoA context scenario.
    poa_worst_case_regime_name: str = "poa_worst_case"
    # Number of stochastic DRO scenarios drawn around that regime.
    poa_worst_case_n_scenarios: int = 10

    # Regime set name written to the runtime YAML.
    poa_regime_set: str = "sensitivity_runtime"

    etas: list[float] = field(
        default_factory=lambda: [0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0]
    )
    dro_wasserstein_epsilon: float = 2000.0
    ambiguity_kappa: float = 0.3
    dro_tightening_eta: float = 0.0

    dro_objective_mode: str = "piecewise_mccormick"
    dro_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    dro_mccormick_num_pieces: int = 50
    dro_time_limit: int | None = None

    use_wasserstein_support_set: bool = True
    calibrate_support_coverage: bool = True
    support_verify_seed: int = 77777
    support_verify_num_draws: int = 2000
    support_coverage_grid: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.999, 0.9999]
    )
    support_include_fleet_band: bool = True
    ar1_coverage: float | None = None

    run_dro_tightening: bool = True
    dro_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {k: True for k in DRO_TIGHTENING_FLAGS}
    )
    run_dro_optimization: bool = True
    archive_existing_dro_results: bool = True

    # When True (default), the DRO empirical distribution is the single
    # PoA-optimal scenario (D[t], P_W_max[t] from the PoA result JSON)
    # rather than fresh stochastic draws from the regime.  This ensures
    # DRO(eta=0) evaluates PoA at the same worst-case market state as the
    # base PoA, making the two analyses directly comparable.
    use_poa_optimal_as_dro_scenario: bool = True

    # -----------------------------------------------------------------------
    # Step toggles
    # -----------------------------------------------------------------------
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True

    # -----------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------
    synthetic_scenario_dir: Path = Path(
        "results/sensitivity_pipeline/synthetic_scenarios"
    )
    poa_scenario_dir: Path = Path("results/sensitivity_pipeline/poa_scenarios")
    dro_scenario_dir: Path = Path("results/sensitivity_pipeline/dro_scenarios")
    heuristic_results_path: Path = Path(
        "results/sensitivity_pipeline/merit_order_results.json"
    )
    raw_feature_dir: Path = Path("models/neural_network/features/generated/raw")
    normalized_feature_dir: Path = Path(
        "models/neural_network/features/generated/normalized"
    )
    model_dir: Path = Path("models/neural_network/training/trained_models")
    training_result_dir: Path = Path(
        "models/neural_network/training/training_results"
    )
    dro_result_dir: Path = Path("results/sensitivity_pipeline/dro")
    dro_result_archive_dir: Path = Path("results/sensitivity_pipeline/dro/old_results")
    runtime_config_path: Path = Path(
        "results/sensitivity_pipeline/runtime_regime_definitions.yaml"
    )
    support_calibration_report_path: Path = Path(
        "results/sensitivity_pipeline/support_calibration.json"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: SensitivityPipelineConfig) -> None:
    print_pipeline_header(config)

    poa_config = build_poa_config(config)

    # Stage 1: shared pre-processing ------------------------------------------
    synthetic_manager = ScenarioManager(config.case)
    synthetic_scenarios = load_or_generate_scenarios(
        config=poa_config,
        manager=synthetic_manager,
        n_scenarios=config.synthetic_num_scenarios,
        seed=config.synthetic_seed,
        output_dir=config.synthetic_scenario_dir,
        should_generate=config.run_scenario_generation,
        time_steps=config.synthetic_time_steps,
    )

    if config.run_heuristic_labels:
        run_heuristic(poa_config, synthetic_scenarios, synthetic_manager)

    if config.run_feature_building:
        build_features(poa_config, synthetic_scenarios)

    if config.run_nn_training:
        train_policies(poa_config)

    # Stage 2: base PoA analysis -----------------------------------------------
    if config.run_poa_tightening or config.run_poa_optimization:
        load_or_generate_scenarios(
            config=poa_config,
            manager=ScenarioManager(config.case),
            n_scenarios=config.poa_context_num_scenarios,
            seed=config.poa_seed,
            output_dir=config.poa_scenario_dir,
            should_generate=True,
            time_steps=config.horizon,
        )

    if config.run_poa_tightening:
        run_tightening_pipeline(poa_config)

    if config.run_poa_optimization:
        run_final_poa(poa_config)

    # Stage 3: bridge — extract regime from PoA context scenario ---------------
    regime_params = extract_poa_regime_params(config.poa_scenario_dir)
    print_regime_bridge(config, regime_params)

    # Stage 4: DRO pipeline ----------------------------------------------------
    # The DRO McCormick needs C_opt bounds that span the PoA worst-case scenario.
    # DRO tightening uses regime-drawn scenarios whose C_opt can differ widely
    # from the PoA optimal scenario's C_opt, corrupting the McCormick approximation.
    # Use the PoA tightening bounds instead, which are valid for the shared physical model.
    poa_c_opt_bounds = load_poa_tightening_c_opt_bounds(poa_config)
    if poa_c_opt_bounds is not None:
        print(
            f"\nSensitivity bridge: using PoA tightening C_opt bounds "
            f"[{poa_c_opt_bounds[0]:.4g}, {poa_c_opt_bounds[1]:.4g}] for DRO McCormick"
        )
    dro_config = build_dro_config(config, poa_c_opt_bounds=poa_c_opt_bounds)
    write_poa_regime_runtime_config(config, dro_config, regime_params)

    if config.run_dro_tightening or config.run_dro_optimization:
        if config.use_wasserstein_support_set and config.calibrate_support_coverage:
            run_support_calibration(dro_config)
        if config.use_poa_optimal_as_dro_scenario:
            dro_scenarios = load_poa_optimal_scenario_as_dro_scenarios(
                config, poa_config
            )
            print(dro_scenarios["description_text"])
        else:
            dro_scenarios = load_dro_scenario_data(dro_config)
        regime_names = resolve_dro_regime_names(dro_config, dro_scenarios)
    else:
        dro_scenarios = {}
        regime_names = []

    if config.run_dro_tightening:
        for regime_name in regime_names:
            run_dro_tightening_for_regime(dro_config, dro_scenarios, regime_name)

    if config.run_dro_optimization:
        if config.archive_existing_dro_results:
            archive_existing_dro_result_folders(dro_config, regime_names)
        sweep_summary = run_dro_eta_sweep(dro_config, dro_scenarios, regime_names)
        summary_path = dro_config.dro_result_dir / "eta_sweep_summary.json"
        write_json(summary_path, sweep_summary)
        print(f"\nSaved DRO eta-sweep summary: {summary_path}")

    print("\nSensitivity pipeline complete.")


# ---------------------------------------------------------------------------
# Sensitivity bridge
# ---------------------------------------------------------------------------

def extract_poa_regime_params(poa_scenario_dir: Path) -> dict[str, float]:
    """Read AR(1) regime parameters from the PoA context scenario CSV.

    The PoA optimization uses a single context scenario whose parameters
    (mu_D, rho_D, sigma_D, mu_W, rho_W, sigma_W, peak_W) define the market
    regime that maximised inefficiency.  Those parameters are exactly what
    DRO regime definitions require.
    """
    scenarios_csv = Path(poa_scenario_dir) / "scenarios.csv"
    if not scenarios_csv.exists():
        raise FileNotFoundError(
            f"PoA context scenario not found: {scenarios_csv}\n"
            "Run the PoA stage first (run_poa_tightening=True or "
            "run_poa_optimization=True) so that the context scenario is saved."
        )
    df = pd.read_csv(scenarios_csv, nrows=1)
    missing = [col for col in _REGIME_PARAM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"PoA context scenario CSV is missing regime parameter columns: {missing}"
        )
    row = df.iloc[0]
    return {col: float(row[col]) for col in _REGIME_PARAM_COLUMNS}


def write_poa_regime_runtime_config(
    config: SensitivityPipelineConfig,
    dro_config: DROFullPipelineConfig,
    regime_params: dict[str, float],
) -> Path:
    """Write a runtime regime YAML containing only the extracted PoA regime.

    The file is read by ScenarioManager.create_scenario_set_from_regimes via
    load_dro_scenario_data.  We bypass run_full_pipeline_DRO's
    write_runtime_regime_config entirely and write a self-contained definition.
    """
    runtime_config = {
        "regime_sets": {
            dro_config.poa_regime_set: {
                "description": (
                    "Single regime extracted from the PoA worst-case context scenario."
                ),
                "seed": dro_config.poa_seed,
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
    output_path = write_json(dro_config.runtime_config_path, runtime_config)
    print(f"\nWrote PoA regime runtime config: {output_path}")
    return output_path


def load_poa_optimal_scenario_as_dro_scenarios(
    config: SensitivityPipelineConfig,
    poa_config: FullPipelineConfig,
) -> dict[str, Any]:
    """Build the DRO empirical distribution from the PoA-optimal trajectory.

    The base PoA finds the worst-case market scenario (D[t], P_W_max[t]) by
    optimising over the ambiguity set.  Using that exact scenario as the DRO
    reference distribution (N=1) ensures DRO(eta=0) evaluates PoA at the same
    market state as the base PoA, making the two analyses directly comparable.
    At eta>0 the DRO then explores distributional uncertainty around that point.

    The regime parameters (mu_D, sigma_D, ...) and all static columns (block
    capacities, costs) are taken from the original PoA context scenario so the
    DRO support set and block structure remain consistent.
    """
    result_path = poa_config.poa_results_path
    if not result_path.exists():
        raise FileNotFoundError(
            f"PoA result not found: {result_path}\n"
            "Run run_poa_optimization=True before the DRO stage."
        )
    with result_path.open("r", encoding="utf-8") as fh:
        poa_result: dict[str, Any] = json.load(fh)

    context_csv = config.poa_scenario_dir / "scenarios.csv"
    if not context_csv.exists():
        raise FileNotFoundError(
            f"PoA context scenario not found: {context_csv}\n"
            "Run the PoA stage so the context scenario is saved first."
        )

    context_df = pd.read_csv(context_csv)
    costs_df = pd.read_csv(config.poa_scenario_dir / "costs.csv")
    ramps_df = pd.read_csv(config.poa_scenario_dir / "ramps.csv")

    # Start from the context scenario row; replace the stochastic profiles
    # with the PoA-optimal trajectories.
    row = context_df.iloc[0].copy()

    # Overwrite demand profile and scalar demand with PoA-optimal values.
    demand_profile: list[float] = poa_result.get("demand_profile") or []
    if demand_profile:
        row["demand_profile"] = str(demand_profile)
        row["demand"] = float(np.mean(demand_profile))

    # Overwrite each wind generator's capacity profile.
    generators: dict[str, Any] = poa_result.get("generators") or {}
    for gen_name, gen_data in generators.items():
        if not gen_data.get("is_wind"):
            continue
        cap_profile: list[float] = gen_data.get("physical_capacity_profile") or []
        col = f"{gen_name}_B1_profile"
        if col in context_df.columns and cap_profile:
            row[col] = str(cap_profile)

    # Ensure the regime column matches the name the DRO optimizer will filter on.
    row["regime"] = config.poa_worst_case_regime_name

    obj: dict[str, Any] = poa_result.get("objective") or {}
    poa_ratio = obj.get("PoA_ratio")
    ratio_str = f"{float(poa_ratio):.4f}" if poa_ratio is not None else "N/A"

    return {
        "scenarios_df": pd.DataFrame([row]),
        "costs_df": costs_df,
        "ramps_df": ramps_df,
        "description_text": (
            f"\nSensitivity bridge: DRO empirical scenario = PoA-optimal trajectory "
            f"(ex-post PoA={ratio_str}, T={poa_result.get('num_time_steps')}, N=1)"
        ),
    }


def load_poa_tightening_c_opt_bounds(
    poa_config: FullPipelineConfig,
) -> tuple[float, float] | None:
    """Read C_opt bounds from the PoA final tightening report.

    The PoA tightening runs on the PoA context scenario and produces C_opt
    bounds that span the actual worst-case cost range.  These bounds must be
    used for the DRO McCormick — DRO tightening runs on regime-drawn scenarios
    whose C_opt range can be entirely different, which would make the piecewise
    McCormick invalid over the PoA worst-case region.
    """
    report_path = poa_config.tightening_report_path
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as fh:
        report: dict[str, Any] = json.load(fh)
    bounds = report.get("optimal_cost_bounds") or {}
    if isinstance(bounds.get("C_opt"), dict):
        bounds = bounds["C_opt"]
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if lower is None or upper is None:
        return None
    return (float(lower), float(upper))


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_poa_config(config: SensitivityPipelineConfig) -> FullPipelineConfig:
    return FullPipelineConfig(
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
        test_size=config.test_size,
        random_state=config.random_state,
        patience=config.patience,
        min_delta=config.min_delta,
        device=config.device,
        nn_final_activation=config.nn_final_activation,
        horizon=config.horizon,
        ambiguity_set_config_path=config.ambiguity_set_config_path,
        ambiguity_set_config_name=config.ambiguity_set_config_name,
        nn_policy_generators=list(config.nn_policy_generators),
        poa_objective_mode=config.poa_objective_mode,
        poa_mccormick_PoA_bounds=config.poa_mccormick_PoA_bounds,
        poa_mccormick_num_pieces=config.poa_mccormick_num_pieces,
        solver_name=config.solver_name,
        preprocessing_time_limit=config.preprocessing_time_limit,
        poa_time_limit=config.poa_time_limit,
        epsilon=config.epsilon,
        poa_parallel_workers=config.poa_parallel_workers,
        poa_solver_threads_per_worker=config.poa_solver_threads_per_worker,
        run_scenario_generation=config.run_scenario_generation,
        run_heuristic_labels=config.run_heuristic_labels,
        run_feature_building=config.run_feature_building,
        run_nn_training=config.run_nn_training,
        run_tightening=config.run_poa_tightening,
        tightening_flags=dict(config.poa_tightening_flags),
        tightening_previous_paths=dict(POA_TIGHTENING_PREVIOUS_PATHS),
        tightening_output_paths=dict(POA_TIGHTENING_OUTPUT_PATHS),
        run_poa_optimization=config.run_poa_optimization,
        synthetic_scenario_dir=config.synthetic_scenario_dir,
        poa_scenario_dir=config.poa_scenario_dir,
        heuristic_results_path=config.heuristic_results_path,
        raw_feature_dir=config.raw_feature_dir,
        normalized_feature_dir=config.normalized_feature_dir,
        model_dir=config.model_dir,
        training_result_dir=config.training_result_dir,
        poa_result_dir=config.poa_result_dir,
    )


def build_dro_config(
    config: SensitivityPipelineConfig,
    poa_c_opt_bounds: tuple[float, float] | None = None,
) -> DROFullPipelineConfig:
    """Build the DRO config targeting only the extracted PoA regime."""
    return DROFullPipelineConfig(
        case=config.case,
        synthetic_time_steps=config.synthetic_time_steps,
        poa_regime_set=config.poa_regime_set,
        # source_poa_regime_set is unused since we write the runtime config
        # ourselves — set to a dummy value so the field is populated.
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
        dro_mccormick_PoA_bounds=config.dro_mccormick_PoA_bounds,
        dro_mccormick_c_opt_bounds=poa_c_opt_bounds,
        dro_mccormick_num_pieces=config.dro_mccormick_num_pieces,
        solver_name=config.solver_name,
        preprocessing_time_limit=config.preprocessing_time_limit,
        dro_time_limit=config.dro_time_limit,
        slack_epsilon=config.epsilon,
        poa_parallel_workers=config.poa_parallel_workers,
        poa_solver_threads_per_worker=config.poa_solver_threads_per_worker,
        use_wasserstein_support_set=config.use_wasserstein_support_set,
        calibrate_support_coverage=config.calibrate_support_coverage,
        support_verify_seed=config.support_verify_seed,
        support_verify_num_draws=config.support_verify_num_draws,
        support_coverage_grid=list(config.support_coverage_grid),
        support_include_fleet_band=config.support_include_fleet_band,
        ar1_coverage=config.ar1_coverage,
        support_calibration_report_path=config.support_calibration_report_path,
        # Pre-processing already completed in the PoA stage; skip in DRO.
        run_scenario_generation=False,
        run_heuristic_labels=False,
        run_feature_building=False,
        run_nn_training=False,
        run_dro_tightening=config.run_dro_tightening,
        run_dro_optimization=config.run_dro_optimization,
        tightening_flags=dict(config.dro_tightening_flags),
        tightening_previous_paths=dict(DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS),
        tightening_output_paths=dict(DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS),
        runtime_config_path=config.runtime_config_path,
        synthetic_scenario_dir=config.synthetic_scenario_dir,
        poa_scenario_dir=config.dro_scenario_dir,
        heuristic_results_path=config.heuristic_results_path,
        raw_feature_dir=config.raw_feature_dir,
        normalized_feature_dir=config.normalized_feature_dir,
        model_dir=config.model_dir,
        training_result_dir=config.training_result_dir,
        dro_result_dir=config.dro_result_dir,
        archive_existing_dro_results=config.archive_existing_dro_results,
        dro_result_archive_dir=config.dro_result_archive_dir,
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_pipeline_header(config: SensitivityPipelineConfig) -> None:
    eta_values = ", ".join(str(float(eta)) for eta in config.etas)
    print(
        "\nSensitivity pipeline configuration\n"
        f"  case={config.case}\n"
        f"  synthetic_time_steps={config.synthetic_time_steps or 'case default'}\n"
        f"  horizon={config.horizon}\n"
        f"  poa_objective_mode={config.poa_objective_mode}\n"
        f"  poa_context_num_scenarios={config.poa_context_num_scenarios}, "
        f"seed={config.poa_seed}\n"
        f"  bridge: PoA context scenario → DRO regime "
        f"'{config.poa_worst_case_regime_name}' "
        f"({config.poa_worst_case_n_scenarios} DRO scenarios)\n"
        f"  dro_objective_mode={config.dro_objective_mode}\n"
        f"  dro_mccormick_PoA_bounds={config.dro_mccormick_PoA_bounds}\n"
        f"  etas=[{eta_values}]\n"
        f"  wasserstein_epsilon={config.dro_wasserstein_epsilon}\n"
        f"  solver={config.solver_name}, "
        f"parallel_workers={config.poa_parallel_workers}"
    )


def print_regime_bridge(
    config: SensitivityPipelineConfig,
    regime_params: dict[str, float],
) -> None:
    print(
        f"\nSensitivity bridge: PoA context scenario → "
        f"DRO regime '{config.poa_worst_case_regime_name}'"
    )
    for key, value in regime_params.items():
        print(f"  {key}: {value:.6g}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_config = SensitivityPipelineConfig(
        # Case and random seeds.
        case="base_test_case",
        synthetic_time_steps=24,
        synthetic_seed=1,
        poa_seed=1,

        # Scenario counts.
        synthetic_num_scenarios=500,
        ambiguity_set_config_path="config/ambiguity_set_config.yaml",
        ambiguity_set_config_name="base_test_case",

        # Heuristic synthetic-label generation.
        bid_tolerance=1e-2,

        # Neural-network feature and training parameters.
        nn_feature_columns=[
            "demand",
            "total_wind_generation_capacity",
            "total_generation_capacity",
            "residual_demand",
            "previous_generation_capacity",
            "previous_demand",
            "next_generation_capacity",
            "next_demand",
            "own_generation_capacity",
            "previous_own_generation_capacity",
            "next_own_generation_capacity",
        ],
        per_generator_normalization=True,
        hidden_layers=[4, 8],
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=500,
        weight_decay=0.0,
        test_size=0.2,
        random_state=42,
        patience=50,
        min_delta=1e-6,
        device=None,
        nn_final_activation="linear",

        # Model dimensions.
        horizon=8,
        nn_policy_generators=["G1", "W2", "W3"],

        solver_name="gurobi",
        preprocessing_time_limit=200,
        epsilon=1e-6,
        poa_parallel_workers=6,
        poa_solver_threads_per_worker=1,

        # PoA analysis.
        poa_context_num_scenarios=1,
        poa_objective_mode="piecewise_mccormick",
        poa_mccormick_PoA_bounds=(1.0, 10.0),
        poa_mccormick_num_pieces=50,
        poa_time_limit=None,

        # DRO analysis — regime derived from PoA context scenario.
        poa_worst_case_regime_name="poa_worst_case",
        poa_worst_case_n_scenarios=10,
        etas=[0.0] + np.logspace(-2, 0.5, 10).tolist() + [10.0],
        dro_wasserstein_epsilon=2000,
        ambiguity_kappa=0.3,
        dro_tightening_eta=0.0,
        dro_objective_mode="piecewise_mccormick",
        dro_mccormick_PoA_bounds=(1.0, 10.0),
        dro_mccormick_num_pieces=50,
        dro_time_limit=None,
        use_wasserstein_support_set=True,

        # Step toggles. Turn expensive stages off when reusing outputs.
        run_scenario_generation=False,
        run_heuristic_labels=False,
        run_feature_building=False,
        run_nn_training=False,
        run_poa_tightening=True,
        
        poa_tightening_flags={
            "primal_big_m": True,
            "relu_bounds": True,
            "alpha_bounds": True,
            "slack_binary_fix": True,
            "dual_big_m": True,
            "optimal_cost_bounds": True,
        },

        run_poa_optimization=True,
        
        run_dro_tightening=True,
        dro_tightening_flags={
            "primal_big_m": True,
            "relu_bounds": True,
            "alpha_bounds": True,
            "slack_binary_fix": True,
            "dual_big_m": True,
            "optimal_cost_bounds": True,
        },
        run_dro_optimization=True,
        archive_existing_dro_results=True,

        # Output paths — reuse existing NN artifacts; write PoA/DRO to
        # sensitivity_pipeline/ so results don't overwrite the standalone runs.
        synthetic_scenario_dir=Path("results/full_pipeline/synthetic_scenarios"),
        poa_scenario_dir=Path("results/sensitivity_pipeline/poa_scenarios"),
        dro_scenario_dir=Path("results/sensitivity_pipeline/dro_scenarios"),
        heuristic_results_path=Path("results/merit_order_best_response_results.json"),
        raw_feature_dir=Path("models/neural_network/features/generated/raw"),
        normalized_feature_dir=Path("models/neural_network/features/generated/normalized"),
        model_dir=Path("models/neural_network/training/trained_models"),
        training_result_dir=Path("models/neural_network/training/training_results"),
        poa_result_dir=Path("results/sensitivity_pipeline/poa"),
        dro_result_dir=Path("results/sensitivity_pipeline/dro"),
        dro_result_archive_dir=Path("results/sensitivity_pipeline/dro/old_results"),
        runtime_config_path=Path(
            "results/sensitivity_pipeline/runtime_regime_definitions.yaml"
        ),
        support_calibration_report_path=Path(
            "results/sensitivity_pipeline/support_calibration.json"
        ),
    )
    main(run_config)
