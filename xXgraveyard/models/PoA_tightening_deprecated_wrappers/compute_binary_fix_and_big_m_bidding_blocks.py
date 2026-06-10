"""Deprecated wrapper for the staged PoA slack-fix plus dual Big-M workflow."""

from __future__ import annotations

from models.PoA.PoA_tightening.compute_dual_big_m import DualBigMComputer
from models.PoA.PoA_tightening.compute_slack_binary_fix import SlackBinaryFixComputer


def main() -> None:
    from xXgraveyard.driver.PoA_pipeline import PoAPipelineConfig, build_poa_tightening

    config = PoAPipelineConfig(run_tightening=False)
    slack_stage = build_poa_tightening(config, SlackBinaryFixComputer)
    slack_stage._load_previous_stage("primal_big_m", config.primal_big_m_path)
    slack_stage._load_previous_stage("alpha_bounds", config.alpha_bounds_path)
    slack_stage.run_slack_binary_fix(
        output_path=config.slack_report_path,
        epsilon=config.epsilon,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        slack_stop_tol=config.epsilon,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )

    dual_stage = slack_stage._as_stage(DualBigMComputer)
    report = dual_stage.run_dual_big_m(
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
