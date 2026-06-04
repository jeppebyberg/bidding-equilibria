"""Full pipeline: base PoA → DRO eta sweep on the worst-case regime.

Runs shared pre-processing (scenario generation, heuristic labels, feature
building, NN training) once, then:

  1. Base PoA optimization over the ambiguity-set context scenario.
  2. Extracts the AR(1) regime parameters (mu_D, sigma_D, rho_D, mu_W, ...)
     from the PoA context scenario — the market state the PoA upper level
     chose as worst case.
  3. Writes a custom runtime regime definition with those parameters.
  4. DRO eta sweep over that extracted regime.

This script runs a single case.  For sensitivity analyses over multiple cases
or physical configurations, see driver/sensitivity/.
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
from driver.PoA_pipeline import (
    TIGHTENING_FLAGS as POA_TIGHTENING_FLAGS,
    PoAPipelineConfig,
    build_features,
    load_or_generate_scenarios,
    run_final_poa,
    run_heuristic,
    run_tightening_pipeline,
    train_policies,
    write_json,
)
from driver.DRO_PoA_pipeline import (
    DRO_TIGHTENING_FLAGS,
    DROPoAPipelineConfig,
    archive_existing_dro_result_folders,
    load_dro_scenario_data,
    resolve_dro_regime_names,
    run_dro_eta_sweep,
    run_dro_tightening_for_regime,
    run_support_calibration,
)
from driver.visualization_pipeline import (
    plot_base_poa_stage,
    plot_dro_stage,
    plot_nn_policy_stage,
    run_oos_evaluation_stage,
)

# AR(1) parameter columns present in the scenarios_df / scenarios.csv.
_REGIME_PARAM_COLUMNS = ("mu_D", "rho_D", "sigma_D", "mu_W", "rho_W", "sigma_W", "peak_W")


def default_eta_grid() -> list[float]:
    """Wasserstein-penalty (eta) grid for the DRO sweep — 12 points.

    Log-spaced across eta, which concentrates points at small eta where the
    achieved Wasserstein distance (epsilon) moves fastest, and thins them toward
    the SAA end (large eta) where epsilon saturates near zero:

      - 0.0                    : fully robust anchor (largest achieved epsilon).
      - logspace(-2.5, 0.5, 15): geometric sweep from eta=0.01 to ~3.16 through
                                 the knee; dense at the low-eta / steep end.
      - 10.0                   : SAA end anchor (epsilon -> 0).

    A previous attempt at a hand-tuned grid with a dense geometric core in
    [0.05, 1.0] starved the steep robust tail (long bare segment at high
    epsilon) and piled most points at epsilon ~ 0, because the knee actually
    lands below eta ~ 0.2.  Log spacing avoids both failure modes.

    The exact eta->epsilon map is case/horizon dependent; re-check the sweep
    once and widen the logspace exponents if the knee lands elsewhere.
    """
    return [0.0] + np.logspace(-2.5, 0.5, 15).tolist() + [10.0]


@dataclass
class FullPipelineConfig:
    # -----------------------------------------------------------------------
    # Shared / common
    # -----------------------------------------------------------------------
    case: str = "base_test_case"
    case_label: str = ""
    synthetic_time_steps: int | None = 24
    synthetic_seed: int = 1
    poa_seed: int = 2

    synthetic_num_scenarios: int = 500
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"

    bid_tolerance: float = 1e-2

    # Neural-network feature and training parameters.
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
    hidden_layers: list[int] = field(default_factory=lambda: [4, 8])
    learning_rate: float = 1e-3
    batch_size: int = 32
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

    horizon: int = 8
    nn_policy_generators: list[str] = field(default_factory=list)

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
    # Rooted under results/<case>/poa in __post_init__ when left as None.
    poa_result_dir: Path | None = None

    # -----------------------------------------------------------------------
    # DRO-specific
    # -----------------------------------------------------------------------
    # Name given to the custom regime built from the PoA context scenario.
    poa_worst_case_regime_name: str = "poa_worst_case"
    # Number of stochastic DRO scenarios drawn around that regime.
    poa_worst_case_n_scenarios: int = 10

    # Regime set name written to the runtime YAML.
    poa_regime_set: str = "sensitivity_runtime"

    etas: list[float] = field(default_factory=default_eta_grid)
    dro_wasserstein_epsilon: float = 2000.0
    ambiguity_kappa: float = 0.3
    dro_tightening_eta: float = 0.0

    dro_objective_mode: str = "piecewise_mccormick"
    dro_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 10.0)
    dro_mccormick_num_pieces: int = 50
    dro_time_limit: int | None = None

    calibrate_support_coverage: bool = True
    support_verify_seed: int = 77777
    support_verify_num_draws: int = 2000
    support_coverage_grid: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.999, 0.9999]
    )
    ar1_coverage: float | None = None

    run_dro_tightening: bool = True
    dro_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: {k: True for k in DRO_TIGHTENING_FLAGS}
    )
    run_dro_optimization: bool = True
    archive_existing_dro_results: bool = True

    # -----------------------------------------------------------------------
    # NN training gate: only generators with more than this many accepted
    # heuristic label changes receive an NN policy; the rest bid at their
    # true marginal cost.  Set to None to train all nn_policy_generators.
    # -----------------------------------------------------------------------
    nn_training_min_label_changes: int | None = 50

    # -----------------------------------------------------------------------
    # Step toggles
    # -----------------------------------------------------------------------
    # When True, render training diagnostic plots after NN training completes.
    plot_results_along_the_way: bool = False
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = True
    run_nn_training: bool = True

    # -----------------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------------
    # Left as None so __post_init__ can root every artifact under
    # results/<case>/...  Set any of these explicitly to override one path.
    synthetic_scenario_dir: Path | None = None
    poa_scenario_dir: Path | None = None
    dro_scenario_dir: Path | None = None
    heuristic_results_path: Path | None = None
    raw_feature_dir: Path | None = None
    normalized_feature_dir: Path | None = None
    model_dir: Path | None = None
    training_result_dir: Path | None = None
    dro_result_dir: Path | None = None
    dro_result_archive_dir: Path | None = None
    runtime_config_path: Path | None = None
    support_calibration_report_path: Path | None = None
    # Root for all pipeline-triggered figures (training, NN policy, base PoA,
    # DRO). Defaults to results/<case>/figures; sensitivity sweeps point it at
    # each composition's own folder.
    figures_dir: Path | None = None

    def __post_init__(self) -> None:
        # Root all unset output paths under results/<case>/ so every artifact for
        # a run (data, models, figures) lives together with that case.
        root = Path("results") / self.case
        path_defaults: dict[str, Path] = {
            "figures_dir": root / "figures",
            "poa_result_dir": root / "poa",
            "synthetic_scenario_dir": root / "synthetic_scenarios",
            "poa_scenario_dir": root / "poa_scenarios",
            "dro_scenario_dir": root / "dro_scenarios",
            "heuristic_results_path": root / "merit_order_results.json",
            "raw_feature_dir": root / "features" / "raw",
            "normalized_feature_dir": root / "features" / "normalized",
            "model_dir": root / "trained_models",
            "training_result_dir": root / "training_results",
            "dro_result_dir": root / "dro",
            "dro_result_archive_dir": root / "dro" / "old_results",
            "runtime_config_path": root / "runtime_regime_definitions.yaml",
            "support_calibration_report_path": root / "support_calibration.json",
        }
        for name, default_path in path_defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, default_path)


# ---------------------------------------------------------------------------
# NN training gate: label-change activity filter
# ---------------------------------------------------------------------------

def filter_nn_policy_generators_by_activity(
    heuristic_results_path: Path,
    candidate_generators: list[str],
    min_label_changes: int,
) -> tuple[list[str], dict[str, int]]:
    """Return generators whose accepted heuristic bid changes exceed the threshold.

    Reads the merit-order heuristic result JSON and counts, for each physical
    generator in ``candidate_generators``, how many of its block bids were
    accepted as updated labels.  Only generators with a count > min_label_changes
    are returned; the rest will bid at true marginal cost in all downstream models.

    Returns (filtered_generators, changes_per_generator).
    """
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

    filtered = [gen for gen in candidate_generators if changes.get(gen, 0) > min_label_changes]
    return filtered, changes


def _print_label_change_filter(
    threshold: int,
    label_counts: dict[str, int],
    filtered: list[str],
    original: list[str],
) -> None:
    dropped = [g for g in original if g not in filtered]
    print(f"\nNN policy generator filter (threshold > {threshold} accepted label changes):")
    for gen in original:
        count = label_counts.get(gen, 0)
        status = "TRAIN NN" if gen in filtered else "true cost (skipped)"
        print(f"  {gen:>6}: {count:>6} changes  →  {status}")
    if dropped:
        print(f"  Skipping NN training for: {', '.join(dropped)}")
    if not filtered:
        print("  WARNING: no generators exceeded the threshold — "
              "all will bid at true marginal cost.")


def discover_trained_policy_generators(model_dir: Path) -> list[str]:
    """Return generator names that have a trained policy file in model_dir.

    Scans for files matching ``{generator}_policy.pt`` and returns the
    generator names in sorted order.  Used to synchronise nn_policy_generators
    with what was actually trained, rather than relying on a hardcoded list.
    """
    return sorted(p.stem.replace("_policy", "") for p in model_dir.glob("*_policy.pt"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: FullPipelineConfig) -> None:
    print_pipeline_header(config)

    if not config.nn_policy_generators:
        _manager = ScenarioManager(config.case)
        config.nn_policy_generators = sorted(
            g["physical_name"] for g in _manager.physical_generators
        )

    pipeline_config = build_poa_config(config)

    # Stage 1: shared pre-processing ------------------------------------------
    synthetic_manager = ScenarioManager(config.case)
    synthetic_scenarios = load_or_generate_scenarios(
        config=pipeline_config,
        manager=synthetic_manager,
        n_scenarios=config.synthetic_num_scenarios,
        seed=config.synthetic_seed,
        output_dir=config.synthetic_scenario_dir,
        should_generate=config.run_scenario_generation,
        time_steps=config.synthetic_time_steps,
    )

    if config.run_heuristic_labels:
        run_heuristic(pipeline_config, synthetic_scenarios, synthetic_manager)

    # Gate NN training on label-change activity.  Applied whether or not the
    # heuristic was re-run this session, so existing results are also used.
    if (
        config.nn_training_min_label_changes is not None
        and pipeline_config.heuristic_results_path.exists()
    ):
        filtered_gens, label_counts = filter_nn_policy_generators_by_activity(
            pipeline_config.heuristic_results_path,
            pipeline_config.nn_policy_generators,
            config.nn_training_min_label_changes,
        )
        _print_label_change_filter(
            config.nn_training_min_label_changes,
            label_counts,
            filtered_gens,
            pipeline_config.nn_policy_generators,
        )
        # Propagate to both the PoA config and the top-level config so that
        # subsequent DRO config construction picks up the filtered list.
        pipeline_config.nn_policy_generators = filtered_gens
        config.nn_policy_generators = filtered_gens

    if config.run_feature_building:
        build_features(pipeline_config, synthetic_scenarios)

    if config.run_nn_training:
        train_policies(pipeline_config)

    # Sync nn_policy_generators to exactly the generators with trained model files.
    # This picks up the effect of the label-change filter and any skipped training,
    # and replaces any hardcoded list that may have been passed in.
    discovered = discover_trained_policy_generators(pipeline_config.model_dir)
    if discovered:
        if discovered != pipeline_config.nn_policy_generators:
            print(f"\nnn_policy_generators updated from trained model files: {discovered}")
        pipeline_config.nn_policy_generators = discovered
        config.nn_policy_generators = discovered
    else:
        print("\nWARNING: no trained model files found — nn_policy_generators unchanged.")

    if config.plot_results_along_the_way:
        plot_nn_policy_stage(pipeline_config)

    # Stage 2: base PoA analysis -----------------------------------------------
    if config.run_poa_tightening or config.run_poa_optimization:
        load_or_generate_scenarios(
            config=pipeline_config,
            manager=ScenarioManager(config.case),
            n_scenarios=config.poa_context_num_scenarios,
            seed=config.poa_seed,
            output_dir=config.poa_scenario_dir,
            should_generate=True,
            time_steps=config.horizon,
        )

    if config.run_poa_tightening:
        run_tightening_pipeline(pipeline_config)

    if config.run_poa_optimization:
        run_final_poa(pipeline_config)

    if config.plot_results_along_the_way:
        plot_base_poa_stage(pipeline_config)

    # Stage 3: bridge — extract regime from optimized PoA state ----------------
    regime_params = extract_poa_regime_params(
        config.poa_scenario_dir,
        poa_results_path=pipeline_config.poa_results_path,
    )
    print_regime_bridge(config, regime_params)

    # Stage 4: DRO pipeline ----------------------------------------------------
    dro_config = build_dro_config(config)
    write_poa_regime_runtime_config(config, dro_config, regime_params)

    if config.run_dro_tightening or config.run_dro_optimization:
        if config.calibrate_support_coverage:
            run_support_calibration(dro_config)
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

    if config.plot_results_along_the_way:
        # Auto-run OOS evaluation so the DRO frontier plots get OOS overlays,
        # then render the eta sweep / frontier with and without OOS.
        oos_results_path = run_oos_evaluation_stage(config, dro_config, regime_names)
        plot_dro_stage(config, dro_config, regime_names, oos_results_path)

    print("\nSensitivity pipeline complete.")


# ---------------------------------------------------------------------------
# Sensitivity bridge
# ---------------------------------------------------------------------------

def extract_poa_regime_params(
    poa_scenario_dir: Path,
    poa_results_path: Path | None = None,
) -> dict[str, float]:
    """Read AR(1) regime parameters from the optimized PoA result.

    The sensitivity bridge should center the DRO regime on the state selected
    by the base PoA upper level, not merely on the sampled context scenario
    used to instantiate the model.  If a PoA result JSON is unavailable, fall
    back to the context scenario CSV.
    """
    if poa_results_path is not None and Path(poa_results_path).exists():
        with Path(poa_results_path).open("r", encoding="utf-8") as fh:
            poa_result: dict[str, Any] = json.load(fh)
        ambiguity_set = poa_result.get("ambiguity_set", {}) or {}
        selected_regime = ambiguity_set.get("selected_regime", {}) or {}
        fixed_parameters = ambiguity_set.get("fixed_parameters", {}) or {}
        required_selected = ("mu_D", "sigma_D", "mu_W", "sigma_W")
        missing_selected = [
            key for key in required_selected if selected_regime.get(key) is None
        ]
        required_fixed = ("rho_D", "rho_W")
        missing_fixed = [
            key for key in required_fixed if fixed_parameters.get(key) is None
        ]
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
    config: FullPipelineConfig,
    dro_config: DROPoAPipelineConfig,
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
                    "Single regime extracted from the optimized PoA state."
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


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _poa_tightening_paths(base_dir: Path) -> dict[str, str]:
    base = str(base_dir / "tightening")
    stages = ["primal_big_m", "relu_bounds", "alpha_bounds", "slack_binary_fix",
              "dual_big_m", "optimal_cost_bounds"]
    paths = {s: f"{base}/{s}_report.json" for s in stages}
    paths["final"] = f"{base}/final_tightening_report.json"
    return paths


def _dro_tightening_paths(base_dir: Path) -> dict[str, str]:
    base = str(base_dir / "tightening" / "{regime_name}")
    stages = ["primal_big_m", "relu_bounds", "alpha_bounds", "slack_binary_fix",
              "dual_big_m", "optimal_cost_bounds"]
    paths = {s: f"{base}/{s}_report.json" for s in stages}
    paths["final"] = f"{base}/final_tightening_report.json"
    return paths


def build_poa_config(config: FullPipelineConfig) -> PoAPipelineConfig:
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
        poa_mccormick_PoA_bounds=config.poa_mccormick_PoA_bounds,
        poa_mccormick_num_pieces=config.poa_mccormick_num_pieces,
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
        tightening_previous_paths=_poa_tightening_paths(config.poa_result_dir),
        tightening_output_paths=_poa_tightening_paths(config.poa_result_dir),
        run_poa_optimization=config.run_poa_optimization,
        synthetic_scenario_dir=config.synthetic_scenario_dir,
        poa_scenario_dir=config.poa_scenario_dir,
        heuristic_results_path=config.heuristic_results_path,
        raw_feature_dir=config.raw_feature_dir,
        normalized_feature_dir=config.normalized_feature_dir,
        model_dir=config.model_dir,
        training_result_dir=config.training_result_dir,
        poa_result_dir=config.poa_result_dir,
        figures_dir=config.figures_dir,
    )


def build_dro_config(config: FullPipelineConfig) -> DROPoAPipelineConfig:
    """Build the DRO config targeting only the extracted PoA regime."""
    return DROPoAPipelineConfig(
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
        dro_mccormick_PoA_bounds=config.dro_mccormick_PoA_bounds,
        dro_mccormick_c_opt_bounds=None,
        dro_mccormick_num_pieces=config.dro_mccormick_num_pieces,
        solver_name=config.solver_name,
        preprocessing_time_limit=config.preprocessing_time_limit,
        dro_time_limit=config.dro_time_limit,
        slack_epsilon=config.epsilon,
        poa_parallel_workers=config.poa_parallel_workers,
        poa_solver_threads_per_worker=config.poa_solver_threads_per_worker,
        calibrate_support_coverage=config.calibrate_support_coverage,
        support_verify_seed=config.support_verify_seed,
        support_verify_num_draws=config.support_verify_num_draws,
        support_coverage_grid=list(config.support_coverage_grid),
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
        tightening_previous_paths=_dro_tightening_paths(config.dro_result_dir),
        tightening_output_paths=_dro_tightening_paths(config.dro_result_dir),
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

def print_pipeline_header(config: FullPipelineConfig) -> None:
    eta_values = ", ".join(str(float(eta)) for eta in config.etas)
    print(
        "\nConfiguration in pipeline: \n"
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
    config: FullPipelineConfig,
    regime_params: dict[str, float],
) -> None:
    print(
        f"\nConfiguration bridge: PoA context scenario → "
        f"DRO regime '{config.poa_worst_case_regime_name}'"
    )
    for key, value in regime_params.items():
        print(f"  {key}: {value:.6g}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = FullPipelineConfig(
        case="base_test_case",
        synthetic_time_steps=24,
        synthetic_seed=1,
        poa_seed=2,
        synthetic_num_scenarios=1000,
        ambiguity_set_config_path="config/ambiguity_set_config.yaml",
        ambiguity_set_config_name="base_test_case",
        bid_tolerance=1e-2,

        nn_feature_columns=[
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
        ],
        per_generator_normalization=True,
        hidden_layers=[8, 8],
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=500,
        weight_decay=0.01,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        patience=100,
        min_delta=1e-6,
        device=None,
        nn_final_activation="linear",
        use_lr_scheduler=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=20,
        lr_scheduler_min_lr=1e-6,

        horizon=8,
        solver_name="gurobi",
        preprocessing_time_limit=200,
        epsilon=1e-6,
        poa_parallel_workers=6,
        poa_solver_threads_per_worker=1,

        poa_context_num_scenarios=1,
        poa_objective_mode="piecewise_mccormick",
        poa_mccormick_PoA_bounds=(1.0, 20.0),
        poa_mccormick_num_pieces=50,
        poa_time_limit=None,

        poa_worst_case_regime_name="poa_worst_case",
        poa_worst_case_n_scenarios=10,

        etas=default_eta_grid(),
        dro_wasserstein_epsilon=2000,
        ambiguity_kappa=0.25,
        dro_tightening_eta=0.0,
        dro_objective_mode="piecewise_mccormick",
        dro_mccormick_PoA_bounds=(1.0, 20.0),
        dro_mccormick_num_pieces=50,
        dro_time_limit=None,

        # Stage toggles — flip to False to reuse existing artifacts.
        plot_results_along_the_way=True,
        run_scenario_generation=False,
        run_heuristic_labels=False,
        run_feature_building=False,
        run_nn_training=False,
        run_poa_tightening=False,
        poa_tightening_flags={
            "primal_big_m": True,
            "relu_bounds": True,
            "alpha_bounds": True,
            "slack_binary_fix": True,
            "dual_big_m": True,
            "optimal_cost_bounds": True,
        },
        run_poa_optimization=False,
        run_dro_tightening=False,
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
    )

    main(config)
