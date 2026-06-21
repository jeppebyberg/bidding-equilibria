# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-12
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Block 1: generate synthetic data and heuristic bid labels."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.scenarios.scenario_generator import ScenarioManager  # noqa: E402
from driver.core.block1_core import load_or_generate_scenarios, run_heuristic  # noqa: E402
from driver.core.visualization_core import plot_setup_stage  # noqa: E402
from driver.block0_system_setup import (  # noqa: E402
    build_config,
    write_manifest,
)

def run(config=None) -> dict[str, Any]:
    cfg = config or build_config()

    t0 = time.perf_counter()
    manager = ScenarioManager(cfg.case)
    scenarios = load_or_generate_scenarios(
        config=cfg,
        manager=manager,
        n_scenarios=cfg.synthetic_num_scenarios,
        seed=cfg.synthetic_seed,
        output_dir=cfg.synthetic_scenario_dir,
        should_generate=cfg.run_scenario_generation,
        time_steps=cfg.synthetic_time_steps,
    )

    if cfg.run_heuristic_labels:
        run_heuristic(cfg, scenarios, manager)
    else:
        print("[block1] Reusing existing heuristic labels.")
    wall_time = time.perf_counter() - t0

    if cfg.plot_results_along_the_way:
        plot_setup_stage(cfg, manager, scenarios)

    manifest = {
        "block": "block1_data_labels",
        "scenario_dir": cfg.synthetic_scenario_dir,
        "heuristic_results_path": cfg.heuristic_results_path,
        "num_scenarios": len(scenarios["scenarios_df"]),
        "saved_scenario_tables": bool(cfg.run_scenario_generation),
        "ran_heuristic_labels": bool(cfg.run_heuristic_labels),
        "wall_time_seconds": wall_time,
    }
    write_manifest("block1_data_labels", manifest, cfg)
    return manifest


if __name__ == "__main__":
    run()
