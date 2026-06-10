"""Deprecated wrapper for the staged PoA dual Big-M tightening module."""

from __future__ import annotations

from models.PoA.PoA_tightening.compute_dual_big_m import DualBigMComputer


def main() -> None:
    from xXgraveyard.driver.PoA_pipeline import PoAPipelineConfig, build_poa_tightening

    config = PoAPipelineConfig(run_tightening=False)
    stage = build_poa_tightening(config, DualBigMComputer)
    stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    stage._load_previous_stage("alpha_bounds", config.alpha_bounds_path)
    stage._load_previous_stage("slack_binary_fix", config.slack_report_path)
    report = stage.run_dual_big_m(
        output_path=config.dual_big_m_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    print(f"\nDeprecated wrapper complete: {config.dual_big_m_path}")
    print(f"Dual components: {list(report.get('tight_big_m', {}))}")


if __name__ == "__main__":
    main()
