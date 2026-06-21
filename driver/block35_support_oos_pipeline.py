# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-06
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Block 3.5: out-of-sample support-set diagnostics for the PoA regime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from driver.core.block35_core import (  # noqa: E402
    analyze_poa_support_band_coverage,
    draw_poa_regime_oos_samples,
    resolve_poa_runtime_regime_names,
)
from driver.core.visualization_core import plot_poa_support_oos_stage  # noqa: E402
from driver.block0_system_setup import (  # noqa: E402
    build_config,
    write_manifest,
)


def run(config=None) -> dict[str, Any]:
    cfg = config or build_config()

    if not Path(cfg.runtime_config_path).exists():
        raise FileNotFoundError(
            "PoA regime runtime config is missing. Run block3_poa_pipeline.py "
            f"first, or point runtime_config_path at an existing file: {cfg.runtime_config_path}"
        )

    scenarios = draw_poa_regime_oos_samples(cfg)
    support_report = analyze_poa_support_band_coverage(cfg, scenarios)
    regime_names = resolve_poa_runtime_regime_names(cfg)

    if cfg.plot_results_along_the_way:
        plot_poa_support_oos_stage(cfg)

    manifest = {
        "block": "block35_support_oos",
        "runtime_config_path": cfg.runtime_config_path,
        "support_oos_report_path": cfg.support_calibration_report_path,
        "support_report": support_report,
        "support_model": "base_poa",
        "regime_names": regime_names,
        "num_oos_draws": len(scenarios["scenarios_df"]),
        "num_violating_scenarios": support_report["num_violating_scenarios"],
    }
    write_manifest("block35_support_oos", manifest, cfg)
    return manifest


if __name__ == "__main__":
    run()
