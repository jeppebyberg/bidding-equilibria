"""Deprecated wrapper for the staged PoA slack/binary-fix tightening module."""

from __future__ import annotations

from models.PoA.PoA_tightening.compute_slack_binary_fix import SlackBinaryFixComputer


def main() -> None:
    from xXgraveyard.driver.PoA_pipeline import PoAPipelineConfig, build_poa_tightening

    config = PoAPipelineConfig(run_tightening=False)
    stage = build_poa_tightening(config, SlackBinaryFixComputer)
    stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    stage._load_previous_stage("alpha_bounds", config.alpha_bounds_path)
    report = stage.run_slack_binary_fix(
        output_path=config.slack_report_path,
        epsilon=config.epsilon,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    print(f"\nDeprecated wrapper complete: {config.slack_report_path}")
    print(f"Fixed complementarity binaries: {report.get('num_fixed_binaries', 0)}")


if __name__ == "__main__":
    main()
