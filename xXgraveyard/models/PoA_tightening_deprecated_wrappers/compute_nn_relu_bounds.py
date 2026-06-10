"""Deprecated wrapper for the staged PoA ReLU-bound tightening module."""

from __future__ import annotations

from models.PoA.PoA_tightening.compute_relu_bounds import ReLUBoundsComputer


def main() -> None:
    from xXgraveyard.driver.PoA_pipeline import PoAPipelineConfig, build_poa_tightening

    config = PoAPipelineConfig(run_tightening=False)
    stage = build_poa_tightening(config, ReLUBoundsComputer)
    report = stage.run_relu_bounds(
        output_path=config.nn_relu_bounds_path,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
    )
    print(f"\nDeprecated wrapper complete: {config.nn_relu_bounds_path}")
    print(f"ReLU summaries: {list(report.get('summary', {}))}")


if __name__ == "__main__":
    main()
