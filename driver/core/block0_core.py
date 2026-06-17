from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.scenarios.scenario_generator import ScenarioManager

POA_TIGHTENING_FLAGS = {
    "primal_big_m": True,
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
    "optimal_cost_bounds": True,
    "equilibrium_cost_bounds": False,
}

DRO_TIGHTENING_FLAGS = {
    "primal_big_m": True,
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
    "optimal_cost_bounds": True,
    "equilibrium_cost_bounds": False,
}

# primal_big_m and optimal_cost_bounds are correctness-critical for a valid
# McCormick C_opt envelope and are always computed, regardless of the
# run_*_tightening toggles or the user-facing *_tightening_flags. The flag dicts
# above only need to list the optional speed-tightening stages; these two are
# force-enabled in code.
ALWAYS_ON_TIGHTENING_STAGES = ("primal_big_m", "optimal_cost_bounds")


def default_eta_grid() -> list[float]:
    # One anchor in the flat left plateau (eta < ~0.005).
    low = [0.001, 0.005]
    # Five points across the ~2-decade transition where the curve actually moves.
    mid = np.logspace(-2.0, 0.0, 10).tolist()  # 0.01, 0.032, 0.1, 0.316, 1.0
    # One anchor to confirm the right plateau.
    tail = [5.0, 10.0]
    return [0.0] + low + mid + tail


def poa_tightening_paths(base_dir: Path) -> dict[str, str]:
    base = str(base_dir / "tightening")
    paths = {stage: f"{base}/{stage}_report.json" for stage in POA_TIGHTENING_FLAGS}
    paths["final"] = f"{base}/final_tightening_report.json"
    return paths


def dro_tightening_paths(base_dir: Path) -> dict[str, str]:
    base = str(base_dir / "tightening" / "{regime_name}")
    paths = {stage: f"{base}/{stage}_report.json" for stage in DRO_TIGHTENING_FLAGS}
    paths["final"] = f"{base}/final_tightening_report.json"
    return paths


@dataclass
class ProjectConfig:
    case: str = "base_test_case"
    case_label: str = ""
    synthetic_time_steps: int | None = None
    synthetic_seed: int = 1
    poa_seed: int = 2
    synthetic_num_scenarios: int = 1000
    synthetic_labels_target: int | None = None
    ambiguity_set_config_path: str = "config/ambiguity_set_config.yaml"
    ambiguity_set_config_name: str = "base_test_case"
    bid_tolerance: float = 1e-2
    # Undercut applied when a marginal block bids just below its nearest higher
    # competitor. Kept separate from bid_tolerance so the buffer below the next
    # generator's cost (e.g. wind vs the conventional fringe at 10 -> label ~9.75
    # with margin 0.25) can be widened without coarsening numerical comparisons.
    inflation_margin: float = 0.25

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
    hidden_layers: list[int] = field(default_factory=lambda: [8, 8])
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 500
    weight_decay: float = 0.01
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    patience: int | None = 75
    min_delta: float = 1e-6
    device: str | None = None
    nn_final_activation: str = "linear"
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 25
    lr_scheduler_min_lr: float = 1e-6
    nn_training_min_label_changes: int | None = 50

    horizon: int = 8
    nn_policy_generators: list[str] = field(default_factory=list)
    allow_wind_to_play: bool = True
    solver_name: str = "gurobi"
    preprocessing_time_limit: int | None = 200
    epsilon: float = 1e-6
    poa_parallel_workers: int = 6
    poa_solver_threads_per_worker: int | None = 1
    # Gurobi Threads/Seed for the FINAL PoA solve (not the tightening subproblems).
    # None = Gurobi default (all cores, default seed). Pin both (e.g. threads=1,
    # seed=0) to make solves deterministic and comparable 1:1 across runs.
    poa_solver_threads: int | None = None
    poa_solver_seed: int | None = None

    poa_context_num_scenarios: int = 1
    poa_objective_mode: str = "piecewise_mccormick"
    poa_mccormick_bounds: dict[str, Any] | None = None
    poa_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 100.0)
    poa_mccormick_c_opt_bounds: tuple[float, float] | None = None
    poa_mccormick_num_pieces: int = 50
    poa_mccormick_c_opt_breakpoints: list[float] | None = None
    poa_time_limit: int | None = 3600
    run_poa_tightening: bool = False
    poa_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: dict(POA_TIGHTENING_FLAGS)
    )
    run_poa_optimization: bool = False
    poa_result_dir: Path | None = None

    poa_worst_case_regime_name: str = "poa_worst_case"
    poa_worst_case_n_scenarios: int = 10
    poa_regime_set: str = "sensitivity_runtime"
    etas: list[float] = field(default_factory=default_eta_grid)
    dro_wasserstein_epsilon: float = 1.0
    ambiguity_kappa: float = 0.3
    dro_tightening_eta: float = 0.0
    dro_objective_mode: str = "piecewise_mccormick"
    dro_mccormick_bounds: dict[str, Any] | None = None
    dro_mccormick_PoA_bounds: tuple[float, float] | None = (1.0, 100.0)
    dro_mccormick_c_opt_bounds: tuple[float, float] | None = None
    dro_mccormick_num_pieces: int = 100
    dro_mccormick_c_opt_breakpoints: list[float] | None = None
    dro_time_limit: int = 1000

    # Relative cushion on the derived PoA upper bound (numerical safety).
    poa_bounds_derivation_margin: float = 1e-3

    calibrate_support_coverage: bool = True
    support_verify_seed: int = 77777
    support_verify_num_draws: int = 2000
    support_coverage_grid: list[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
    )
    ar1_coverage: float | None = None
    run_dro_tightening: bool = False
    dro_tightening_flags: dict[str, bool] = field(
        default_factory=lambda: dict(DRO_TIGHTENING_FLAGS)
    )
    run_dro_optimization: bool = False
    archive_existing_dro_results: bool = True

    plot_results_along_the_way: bool = True
    run_scenario_generation: bool = True
    run_heuristic_labels: bool = True
    run_feature_building: bool = False
    run_nn_training: bool = False

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
    figures_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.synthetic_time_steps is None:
            self.synthetic_time_steps = self.horizon
        if self.synthetic_labels_target is not None:
            import math

            self.synthetic_num_scenarios = math.ceil(self.synthetic_labels_target / self.horizon)
        root = Path("results") / self.case_label
        defaults = {
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
        for name, path in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, path)
        ensure_requested_policy_generators(self)

    @property
    def nn_normalization_stats_path(self) -> Path:
        return Path(self.normalized_feature_dir) / "min_max_stats.json"

    @property
    def poa_results_path(self) -> Path:
        suffix = "" if self.poa_objective_mode == "difference" else f"_{self.poa_objective_mode}"
        return Path(self.poa_result_dir) / f"poa_optimization_T{self.horizon}{suffix}.json"

    @property
    def poa_tightening_output_paths(self) -> dict[str, str]:
        return poa_tightening_paths(Path(self.poa_result_dir))

    @property
    def dro_tightening_output_paths(self) -> dict[str, str]:
        return dro_tightening_paths(Path(self.dro_result_dir))


def ensure_requested_policy_generators(config: ProjectConfig) -> ProjectConfig:
    needs_manager = not config.nn_policy_generators or not config.allow_wind_to_play
    if needs_manager:
        manager = ScenarioManager(config.case)
        if not config.nn_policy_generators:
            config.nn_policy_generators = sorted(
                gen["physical_name"] for gen in manager.physical_generators
            )
        if not config.allow_wind_to_play:
            wind_names = {
                gen["physical_name"] for gen in manager.physical_generators if bool(gen["is_wind"])
            }
            config.nn_policy_generators = [
                name for name in config.nn_policy_generators if name not in wind_names
            ]
    return config


def build_config() -> ProjectConfig:
    from driver.project_config import load_project_config

    return load_project_config()


def pipeline_manifest(config: ProjectConfig | None = None) -> dict[str, Any]:
    cfg = config or build_config()
    return {
        "case_label": cfg.case_label,
        "horizon": cfg.horizon,
        "synthetic": {
            "scenario_dir": str(cfg.synthetic_scenario_dir),
            "heuristic_results_path": str(cfg.heuristic_results_path),
            "raw_feature_dir": str(cfg.raw_feature_dir),
            "normalized_feature_dir": str(cfg.normalized_feature_dir),
            "normalization_stats_path": str(cfg.nn_normalization_stats_path),
        },
        "policies": {
            "model_dir": str(cfg.model_dir),
            "training_result_dir": str(cfg.training_result_dir),
            "requested_generators": list(cfg.nn_policy_generators),
        },
        "poa": {
            "scenario_dir": str(cfg.poa_scenario_dir),
            "result_dir": str(cfg.poa_result_dir),
            "result_path": str(cfg.poa_results_path),
            "tightening_report_path": cfg.poa_tightening_output_paths["final"],
        },
        "regime_bridge": {
            "regime_name": cfg.poa_worst_case_regime_name,
            "runtime_config_path": str(cfg.runtime_config_path),
            "dro_scenario_dir": str(cfg.dro_scenario_dir),
        },
        "support_oos": {
            "support_calibration_report_path": str(cfg.support_calibration_report_path),
        },
        "dro": {
            "result_dir": str(cfg.dro_result_dir),
            "archive_dir": str(cfg.dro_result_archive_dir),
            "eta_sweep_summary_path": str(Path(cfg.dro_result_dir) / "eta_sweep_summary.json"),
            "etas": list(cfg.etas),
        },
        "oos_poa": {
            "result_path": str(Path(cfg.figures_dir).parent / "oos_poa" / "oos_poa_results.json"),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_manifest(
    name: str,
    payload: dict[str, Any],
    config: ProjectConfig | None = None,
) -> Path:
    cfg = config or build_config()
    path = Path(cfg.figures_dir).parent / "pipeline_manifests" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    print(f"[manifest] {name}: {path}")
    return path


def print_setup(config: ProjectConfig | None = None) -> None:
    cfg = config or build_config()
    manifest = pipeline_manifest(cfg)
    print("\nBlock-oriented system setup")
    print(f"  case_label: {manifest['case_label']}")
    print(f"  horizon: {manifest['horizon']}")
    print(f"  PoA result: {manifest['poa']['result_path']}")
    print(f"  regime config: {manifest['regime_bridge']['runtime_config_path']}")
    print(f"  DRO result dir: {manifest['dro']['result_dir']}")
    print(f"  OOS PoA result: {manifest['oos_poa']['result_path']}")
    write_manifest("block0_system_setup", manifest, cfg)
