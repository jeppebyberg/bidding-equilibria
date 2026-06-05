"""Block 3: solve the learned-policy PoA problem and export its regime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.scenarios.scenario_generator import ScenarioManager  # noqa: E402
from driver_tmp.core.block1_core import load_or_generate_scenarios  # noqa: E402
from driver_tmp.core.block2_core import discover_trained_policy_generators  # noqa: E402
from driver_tmp.core.block3_core import (  # noqa: E402
    build_poa_config,
    extract_poa_regime_params,
    print_regime_bridge,
    run_final_poa,
    run_tightening_pipeline,
    write_poa_regime_runtime_config,
)
from driver_tmp.core.visualization_core import plot_base_poa_stage  # noqa: E402
from driver_tmp.block0_system_setup import (  # noqa: E402
    build_config,
    write_manifest,
)

def run(config=None) -> dict[str, Any]:
    cfg = config or build_config()
    pcfg = build_poa_config(cfg)

    discovered = discover_trained_policy_generators(pcfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        pcfg.nn_policy_generators = discovered
        print(f"[block3] using trained policy generators: {discovered}")

    if cfg.run_poa_tightening or cfg.run_poa_optimization:
        load_or_generate_scenarios(
            config=pcfg,
            manager=ScenarioManager(cfg.case),
            n_scenarios=cfg.poa_context_num_scenarios,
            seed=cfg.poa_seed,
            output_dir=cfg.poa_scenario_dir,
            should_generate=True,
            time_steps=cfg.horizon,
        )

    if cfg.run_poa_tightening:
        run_tightening_pipeline(pcfg)
    else:
        print(
            "[block3] run_poa_tightening=False; computing only the always-on "
            "primal_big_m and optimal_cost_bounds stages (remaining stages reuse "
            "existing reports when present and loose defaults otherwise)."
        )
        run_tightening_pipeline(pcfg, run_optional_stages=False)

    if cfg.run_poa_optimization:
        run_final_poa(pcfg)
    else:
        print("[block3] Reusing existing PoA optimization result.")

    if cfg.plot_results_along_the_way:
        plot_base_poa_stage(pcfg)

    regime_params = extract_poa_regime_params(
        cfg.poa_scenario_dir,
        poa_results_path=pcfg.poa_results_path,
    )
    print_regime_bridge(cfg, regime_params)

    runtime_config_path = write_poa_regime_runtime_config(cfg, regime_params)
    regime_params_path = Path(cfg.poa_result_dir) / "poa_worst_case_regime_params.json"
    regime_params_path.parent.mkdir(parents=True, exist_ok=True)
    with regime_params_path.open("w", encoding="utf-8") as fh:
        json.dump(regime_params, fh, indent=2)

    manifest = {
        "block": "block3_poa",
        "poa_scenario_dir": cfg.poa_scenario_dir,
        "poa_result_path": pcfg.poa_results_path,
        "poa_tightening_report_path": pcfg.tightening_report_path,
        "regime_name": cfg.poa_worst_case_regime_name,
        "regime_params_path": regime_params_path,
        "runtime_config_path": runtime_config_path,
        "ran_poa_tightening": bool(cfg.run_poa_tightening),
        "ran_poa_optimization": bool(cfg.run_poa_optimization),
    }
    write_manifest("block3_poa", manifest, cfg)
    return manifest

if __name__ == "__main__":
    run()
