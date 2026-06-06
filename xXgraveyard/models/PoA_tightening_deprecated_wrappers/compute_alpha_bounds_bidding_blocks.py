"""Deprecated wrapper for the staged PoA alpha-bound tightening module."""

from __future__ import annotations

from models.PoA.PoA_tightening.compute_alpha_bounds import AlphaBoundsComputer


def main() -> None:
    from xXgraveyard.driver.PoA_pipeline import PoAPipelineConfig, build_poa_tightening

    config = PoAPipelineConfig(run_tightening=False)
    stage = build_poa_tightening(config, AlphaBoundsComputer)
    stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    stage._load_previous_stage("relu_bounds", config.nn_relu_bounds_path)
    report = stage.run_alpha_bounds(
        output_path=config.alpha_bounds_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    print(f"\nDeprecated wrapper complete: {config.alpha_bounds_path}")
    print(f"Alpha entries: {len(report.get('alpha_bounds', {}))}")


if __name__ == "__main__":
    main()
