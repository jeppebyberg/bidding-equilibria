"""Sensitivity study: PoA and DRO bound-tightening progression.

Runs the base learned-policy configuration through five cumulative tightening
settings:

  1. loose optional stages
  2. ReLU bounds
  3. ReLU + alpha bounds
  4. ReLU + alpha + slack binary fixing
  5. base case: all optional stages, including dual Big-M tightening

Each run solves both the base PoA model and the DRO eta sweep. Upstream
scenarios, labels, features, and trained policies are reused by default; only the
PoA/DRO tightening reports and final optimization outputs are recomputed in
study-local result folders.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.bound_tightening_progression
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from driver import block3_poa_pipeline
from driver.block0_system_setup import write_manifest
from driver.core.block0_core import ProjectConfig, pipeline_manifest
from driver.core.block4_core import (
    archive_existing_dro_result_folders,
    build_dro_config,
    build_dro_tightening,
    load_dro_scenario_data,
    resolve_dro_regime_names,
    resolved_stage_paths,
    run_dro_eta_sweep,
    write_json,
)
from driver.core.block2_core import discover_trained_policy_generators
from driver.sensitivity.sensitivity_config import (
    SensitivityRun,
    SensitivityStudy,
    build_sensitivity_config,
)
from models.DRO_PoA.DRO_PoA_tightening.compute_alpha_bounds import (
    DROAlphaBoundsComputer,
)
from models.DRO_PoA.DRO_PoA_tightening.compute_dual_big_m import (
    DRODualBigMComputer,
)
from models.DRO_PoA.DRO_PoA_tightening.compute_optimal_cost_bounds import (
    DROOptimalCostBoundsComputer,
)
from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import (
    DROPrimalBigMComputer,
)
from models.DRO_PoA.DRO_PoA_tightening.compute_relu_bounds import (
    DROReLUBoundsComputer,
)
from models.DRO_PoA.DRO_PoA_tightening.compute_slack_binary_fix import (
    DROSlackBinaryFixComputer,
)
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain

STUDY_NAME = "bound_tightening_progression"

OPTIONAL_TIGHTENING_STAGES = (
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
)

LOOSE_FLAGS = {
    "relu_bounds": False,
    "alpha_bounds": False,
    "slack_binary_fix": False,
    "dual_big_m": False,
}

RELU_FLAGS = {
    "relu_bounds": True,
    "alpha_bounds": False,
    "slack_binary_fix": False,
    "dual_big_m": False,
}

ALPHA_FLAGS = {
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": False,
    "dual_big_m": False,
}

SLACK_FLAGS = {
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": False,
}

BASE_FLAGS = {
    "relu_bounds": True,
    "alpha_bounds": True,
    "slack_binary_fix": True,
    "dual_big_m": True,
}

TIGHTENING_CASES: tuple[tuple[str, str, dict[str, bool]], ...] = (
    ("x_tightening_flags_1", "loose optional stages", LOOSE_FLAGS),
    ("x_tightening_flags_2", "relu bounds", RELU_FLAGS),
    ("x_tightening_flags_3", "relu + alpha bounds", ALPHA_FLAGS),
    ("x_tightening_flags_4", "relu + alpha + slack fixes", SLACK_FLAGS),
    ("base_case", "base tightening", BASE_FLAGS),
)

BASE_OVERRIDES: dict[str, Any] = {
    # Reuse the base data/features/policies. The study recomputes only PoA/DRO
    # tightening and final solves for each flag configuration.
    "run_scenario_generation": False,
    "run_heuristic_labels": False,
    "run_feature_building": False,
    "run_nn_training": False,
    "run_poa_tightening": True,
    "run_poa_optimization": True,
    "run_dro_tightening": True,
    "run_dro_optimization": True,
    "archive_existing_dro_results": False,
    "plot_results_along_the_way": False,
}

SHARED_ARTIFACT_FIELDS = (
    "model_dir",
    "normalized_feature_dir",
    "training_result_dir",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _flags_with_required_stages(flags: dict[str, bool]) -> dict[str, bool]:
    """Return a full flag dict; always-on stages are handled by pipeline code."""
    return {
        "primal_big_m": True,
        **{stage: bool(flags.get(stage, False)) for stage in OPTIONAL_TIGHTENING_STAGES},
        "optimal_cost_bounds": True,
    }


def build_study() -> SensitivityStudy:
    return SensitivityStudy(
        name=STUDY_NAME,
        blocks=("block3", "block4"),
        base_overrides=BASE_OVERRIDES,
        shared_artifact_fields=SHARED_ARTIFACT_FIELDS,
        runs=[
            SensitivityRun(
                name=name,
                label=label,
                overrides={
                    "poa_tightening_flags": _flags_with_required_stages(flags),
                    "dro_tightening_flags": _flags_with_required_stages(flags),
                },
            )
            for name, label, flags in TIGHTENING_CASES
        ],
        reuse_base_case_results=False,
    )


def _validate_cumulative_flags(flags: dict[str, bool]) -> None:
    if flags.get("alpha_bounds") and not flags.get("relu_bounds"):
        raise ValueError("alpha_bounds=True requires relu_bounds=True")
    if flags.get("slack_binary_fix") and not flags.get("alpha_bounds"):
        raise ValueError("slack_binary_fix=True requires alpha_bounds=True")
    if flags.get("dual_big_m") and not flags.get("slack_binary_fix"):
        raise ValueError("dual_big_m=True requires slack_binary_fix=True")


def _validate_shared_artifacts(config: ProjectConfig) -> None:
    model_dir = Path(config.model_dir)
    stats_path = Path(config.normalized_feature_dir) / "min_max_stats.json"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Trained model directory not found: {model_dir}. "
            "Run the base pipeline through NN training, or point PROJECT_CONFIG.model_dir "
            "at existing trained policies before launching this study."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {stats_path}. "
            "Run feature building/training first, or point normalized_feature_dir "
            "at existing normalized features."
        )


def _record_stage_timing(
    tightening: DROPoATighteningMain,
    stage_name: str,
    output_path: str | Path,
    run_callable: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    start = time.perf_counter()
    report = run_callable()
    tightening._record_stage_timing(
        stage_name,
        time.perf_counter() - start,
        "run",
        output_path,
    )
    return report


def _print_dro_partial_tightening_plan(
    config: Any,
    regime_name: str,
    flags: dict[str, bool],
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
    labels = {
        "primal_big_m": "Primal Big-M",
        "relu_bounds": "NN ReLU bounds",
        "alpha_bounds": "Alpha bounds",
        "slack_binary_fix": "Slack binary fix",
        "dual_big_m": "Dual Big-M",
        "optimal_cost_bounds": "C_opt bounds",
    }
    for stage_name in (
        "primal_big_m",
        *OPTIONAL_TIGHTENING_STAGES,
        "optimal_cost_bounds",
    ):
        should_run = stage_name in {"primal_big_m", "optimal_cost_bounds"} or bool(
            flags.get(stage_name, False)
        )
        action = "run" if should_run else "loose"
        print(f"    {labels[stage_name]:<17} {action:<5} output: {output_paths[stage_name]}")
    print(f"  Final report: {output_paths['final']}")


def run_partial_dro_tightening_for_regime(
    config: Any,
    scenarios: dict[str, Any],
    regime_name: str,
) -> Path:
    """Run exactly the requested DRO stages and write a valid final report.

    The regular DRO tightening orchestrator requires skipped partial stages to
    already exist on disk. For this progression study, missing optional sections
    are intentional: the final optimizer will apply the stages present in the
    report and use its loose defaults for the omitted ones.
    """
    flags = {
        stage: bool(config.tightening_flags.get(stage, False))
        for stage in OPTIONAL_TIGHTENING_STAGES
    }
    _validate_cumulative_flags(flags)
    output_paths = resolved_stage_paths(config.tightening_output_paths, regime_name)
    _print_dro_partial_tightening_plan(config, regime_name, flags, output_paths)

    tightening = build_dro_tightening(config, scenarios, regime_name)

    primal_stage = tightening._as_stage(DROPrimalBigMComputer)
    relu_stage = tightening._as_stage(DROReLUBoundsComputer)
    alpha_stage = tightening._as_stage(DROAlphaBoundsComputer)
    slack_stage = tightening._as_stage(DROSlackBinaryFixComputer)
    dual_stage = tightening._as_stage(DRODualBigMComputer)
    optimal_cost_stage = tightening._as_stage(DROOptimalCostBoundsComputer)

    _record_stage_timing(
        tightening,
        "primal_big_m",
        output_paths["primal_big_m"],
        lambda: primal_stage.run_primal_big_m(output_path=output_paths["primal_big_m"]),
    )

    if flags["relu_bounds"]:
        _record_stage_timing(
            tightening,
            "relu_bounds",
            output_paths["relu_bounds"],
            lambda: relu_stage.run_relu_bounds(
                output_path=output_paths["relu_bounds"],
                solver_name=config.solver_name,
                time_limit=config.preprocessing_time_limit,
                tee=False,
                parallel_workers=config.poa_parallel_workers,
                solver_threads=config.poa_solver_threads_per_worker,
            ),
        )

    if flags["alpha_bounds"]:
        _record_stage_timing(
            tightening,
            "alpha_bounds",
            output_paths["alpha_bounds"],
            lambda: alpha_stage.run_alpha_bounds(
                output_path=output_paths["alpha_bounds"],
                solver_name=config.solver_name,
                time_limit=config.preprocessing_time_limit,
                tee=False,
                parallel_workers=config.poa_parallel_workers,
                solver_threads=config.poa_solver_threads_per_worker,
            ),
        )

    if flags["slack_binary_fix"]:
        _record_stage_timing(
            tightening,
            "slack_binary_fix",
            output_paths["slack_binary_fix"],
            lambda: slack_stage.run_slack_binary_fix(
                output_path=output_paths["slack_binary_fix"],
                epsilon=config.slack_epsilon,
                solver_name=config.solver_name,
                time_limit=config.preprocessing_time_limit,
                tee=False,
                parallel_workers=config.poa_parallel_workers,
                solver_threads=config.poa_solver_threads_per_worker,
            ),
        )

    if flags["dual_big_m"]:
        _record_stage_timing(
            tightening,
            "dual_big_m",
            output_paths["dual_big_m"],
            lambda: dual_stage.run_dual_big_m(
                output_path=output_paths["dual_big_m"],
                solver_name=config.solver_name,
                time_limit=config.preprocessing_time_limit,
                tee=False,
                parallel_workers=config.poa_parallel_workers,
                solver_threads=config.poa_solver_threads_per_worker,
            ),
        )

    _record_stage_timing(
        tightening,
        "optimal_cost_bounds",
        output_paths["optimal_cost_bounds"],
        lambda: optimal_cost_stage.run_optimal_cost_bounds(
            output_path=output_paths["optimal_cost_bounds"],
            solver_name=config.solver_name,
            time_limit=config.preprocessing_time_limit,
            tee=False,
            solver_threads=config.poa_solver_threads_per_worker,
        ),
    )

    final_report_path = tightening.save_final_report(output_paths["final"])
    print(f"\nDRO tightening report complete: {final_report_path}")
    print(f"  Regime: {regime_name}")
    return final_report_path


def run_dro_progression_block(config: ProjectConfig) -> dict[str, Any]:
    discovered = discover_trained_policy_generators(config.model_dir)
    if discovered:
        config.nn_policy_generators = discovered
        print(f"[bound-tightening] using trained policy generators: {discovered}")

    dcfg = build_dro_config(config)
    scenarios = load_dro_scenario_data(dcfg)
    regime_names = resolve_dro_regime_names(dcfg, scenarios)

    tightening_reports: dict[str, str] = {}
    if config.run_dro_tightening:
        for regime_name in regime_names:
            report_path = run_partial_dro_tightening_for_regime(
                dcfg,
                scenarios,
                regime_name,
            )
            tightening_reports[regime_name] = str(report_path)
    else:
        print("[bound-tightening] Reusing existing DRO tightening reports.")

    summary_path = dcfg.dro_result_dir / "eta_sweep_summary.json"
    sweep_summary: list[dict[str, Any]] = []
    if config.run_dro_optimization:
        if config.archive_existing_dro_results:
            archive_existing_dro_result_folders(dcfg, regime_names)
        sweep_summary = run_dro_eta_sweep(dcfg, scenarios, regime_names)
        write_json(summary_path, sweep_summary)
        print(f"\nSaved DRO eta-sweep summary: {summary_path}")
    else:
        print("[bound-tightening] Reusing existing DRO eta-sweep results.")

    return {
        "block": "bound_tightening_progression_dro",
        "runtime_config_path": dcfg.runtime_config_path,
        "dro_scenario_dir": dcfg.poa_scenario_dir,
        "dro_result_dir": dcfg.dro_result_dir,
        "eta_sweep_summary_path": summary_path,
        "regime_names": regime_names,
        "etas": list(dcfg.etas),
        "tightening_reports": tightening_reports,
        "derived_poa_bound": derived_poa_bound,
        "ran_dro_tightening": bool(config.run_dro_tightening),
        "ran_dro_optimization": bool(config.run_dro_optimization),
        "num_summary_records": len(sweep_summary),
    }


def run() -> dict[str, Any]:
    study = build_study()
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Sensitivity study: {study.name} ({len(study.runs)} run(s))")
    print(f"  result_root: {Path(study.result_root) / study.name}")
    print("  blocks: block3, custom DRO progression block")
    print(f"{sep}")

    study_manifest: dict[str, Any] = {
        "study": study.name,
        "result_root": str(Path(study.result_root) / study.name),
        "blocks": ["block3", "bound_tightening_progression_dro"],
        "shared_artifact_fields": list(study.shared_artifact_fields),
        "runs": {},
    }

    for idx, sensitivity_run in enumerate(study.runs, start=1):
        print(f"\n{sep}")
        print(f"  [{idx}/{len(study.runs)}] {sensitivity_run.name}")
        print(f"  {sensitivity_run.label or sensitivity_run.name}")
        print(f"{sep}")

        cfg = build_sensitivity_config(study, sensitivity_run)
        _validate_shared_artifacts(cfg)

        block3_manifest = block3_poa_pipeline.run(cfg)
        dro_manifest = run_dro_progression_block(cfg)

        run_manifest = {
            "config": pipeline_manifest(cfg),
            "overrides": copy.deepcopy(sensitivity_run.overrides),
            "block_manifests": {
                "block3": block3_manifest,
                "bound_tightening_progression_dro": dro_manifest,
            },
        }
        write_manifest(
            f"sensitivity_{study.name}_{sensitivity_run.name}",
            run_manifest,
            cfg,
        )
        study_manifest["runs"][sensitivity_run.name] = run_manifest

    study_manifest_path = Path(study.result_root) / study.name / "study_manifest.yaml"
    study_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with study_manifest_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(_jsonable(study_manifest), file_handle, sort_keys=False)
    print(f"\nSensitivity study complete: {study_manifest_path}")
    return study_manifest


if __name__ == "__main__":
    run()
