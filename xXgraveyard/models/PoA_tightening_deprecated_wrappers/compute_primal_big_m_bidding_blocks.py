"""Deprecated wrapper for the staged PoA primal Big-M tightening module."""

from __future__ import annotations

from models.PoA.PoA_tightening.compute_primal_big_m import (
    PrimalBigMComputer,
    compute_primal_big_m_bounds,
    summarize_primal_big_m,
    support_set_summary,
)


def main() -> None:
    from driver.run_full_pipeline import FullPipelineConfig, build_poa_tightening

    config = FullPipelineConfig(run_tightening=False)
    stage = build_poa_tightening(config, PrimalBigMComputer)
    report = stage.run_primal_big_m(output_path=config.primal_big_m_path)
    print(f"\nDeprecated wrapper complete: {config.primal_big_m_path}")
    print(f"Primal Big-M components: {list(report['primal_big_m'])}")


if __name__ == "__main__":
    main()
