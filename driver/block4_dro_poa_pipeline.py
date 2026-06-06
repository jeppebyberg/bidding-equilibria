"""Block 4: solve DRO PoA over the configured eta grid."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from driver.core.block2_core import discover_trained_policy_generators  # noqa: E402
from driver.core.block4_core import (  # noqa: E402
    archive_existing_dro_result_folders,
    build_dro_config,
    load_dro_scenario_data,
    resolve_dro_regime_names,
    run_dro_essential_bounds_for_regime,
    run_dro_eta_sweep,
    run_dro_tightening_for_regime,
    write_json,
)
from driver.core.visualization_core import plot_dro_stage  # noqa: E402
from driver.block0_system_setup import (  # noqa: E402
    build_config,
    write_manifest,
)

def run(config=None) -> dict[str, Any]:
    cfg = config or build_config()
    discovered = discover_trained_policy_generators(cfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        print(f"[block4] using trained policy generators: {discovered}")
    dcfg = build_dro_config(cfg)

    if not Path(dcfg.runtime_config_path).exists():
        raise FileNotFoundError(
            "PoA regime runtime config is missing. Run block3_poa_pipeline.py "
            f"first, or point runtime_config_path at an existing file: {dcfg.runtime_config_path}"
        )

    if cfg.run_dro_tightening or cfg.run_dro_optimization:
        scenarios = load_dro_scenario_data(dcfg)
        regime_names = resolve_dro_regime_names(dcfg, scenarios)
    else:
        scenarios = {}
        regime_names = list(dcfg.dro_regime_names)

    if cfg.run_dro_tightening:
        for regime_name in regime_names:
            run_dro_tightening_for_regime(dcfg, scenarios, regime_name)
    elif scenarios:
        print(
            "[block4] run_dro_tightening=False; computing only the essential "
            "primal_big_m and optimal_cost_bounds stages per regime (remaining "
            "stages reuse existing reports when present and loose defaults otherwise)."
        )
        for regime_name in regime_names:
            run_dro_essential_bounds_for_regime(dcfg, scenarios, regime_name)
    else:
        print("[block4] Reusing existing DRO tightening reports.")

    summary_path = dcfg.dro_result_dir / "eta_sweep_summary.json"
    sweep_summary: list[dict[str, Any]] = []
    if cfg.run_dro_optimization:
        if cfg.archive_existing_dro_results:
            archive_existing_dro_result_folders(dcfg, regime_names)
        sweep_summary = run_dro_eta_sweep(dcfg, scenarios, regime_names)
        write_json(summary_path, sweep_summary)
        print(f"\nSaved DRO eta-sweep summary: {summary_path}")
    else:
        print("[block4] Reusing existing DRO eta-sweep results.")

    if cfg.plot_results_along_the_way:
        plot_dro_stage(cfg, dcfg, regime_names, oos_results_path=None)

    manifest = {
        "block": "block4_dro_poa",
        "runtime_config_path": dcfg.runtime_config_path,
        "dro_scenario_dir": dcfg.poa_scenario_dir,
        "dro_result_dir": dcfg.dro_result_dir,
        "eta_sweep_summary_path": summary_path,
        "regime_names": regime_names,
        "etas": list(dcfg.etas),
        "ran_dro_tightening": bool(cfg.run_dro_tightening),
        "ran_dro_optimization": bool(cfg.run_dro_optimization),
        "num_summary_records": len(sweep_summary),
    }
    write_manifest("block4_dro_poa", manifest, cfg)
    return manifest


if __name__ == "__main__":
    run()
