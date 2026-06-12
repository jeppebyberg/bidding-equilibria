"""Block 2: build policy features and train bidding policies."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.scenarios.scenario_generator import ScenarioManager  # noqa: E402
from driver.core.block1_core import load_or_generate_scenarios  # noqa: E402
from driver.core.block2_core import (  # noqa: E402
    build_features,
    discover_trained_policy_generators,
    filter_nn_policy_generators_by_activity,
    train_policies,
)
from driver.block0_system_setup import (  # noqa: E402
    build_config,
    write_manifest,
)

def _resolve_policy_generators(cfg, pcfg) -> list[str]:
    if (
        cfg.nn_training_min_label_changes is not None
        and pcfg.heuristic_results_path.exists()
    ):
        filtered, label_counts = filter_nn_policy_generators_by_activity(
            pcfg.heuristic_results_path,
            pcfg.nn_policy_generators,
            cfg.nn_training_min_label_changes,
        )
        print(
            "[block2] label-change activity filter: "
            f"min={cfg.nn_training_min_label_changes}, selected={filtered}"
        )
        col_width = max(len(f"{gen}: {count}") for gen, count in label_counts.items())
        for gen, count in sorted(label_counts.items()):
            entry = f"{gen}: {count}"
            tag = "  <- Train" if gen in filtered else ""
            print(f"  {entry:<{col_width}}{tag}")
        return filtered
    return list(pcfg.nn_policy_generators)

def run(config=None) -> dict[str, Any]:
    cfg = config or build_config()
    cfg.nn_policy_generators = _resolve_policy_generators(cfg, cfg)

    t0 = time.perf_counter()
    manager = ScenarioManager(cfg.case)
    scenarios = load_or_generate_scenarios(
        config=cfg,
        manager=manager,
        n_scenarios=cfg.synthetic_num_scenarios,
        seed=cfg.synthetic_seed,
        output_dir=cfg.synthetic_scenario_dir,
        should_generate=False,
        time_steps=cfg.synthetic_time_steps,
    )

    if cfg.run_feature_building:
        build_features(cfg, scenarios)
    else:
        print("[block2] Reusing existing feature tables.")

    if cfg.run_nn_training:
        train_policies(cfg)
    else:
        print("[block2] Reusing existing trained policies.")
    wall_time = time.perf_counter() - t0

    discovered = discover_trained_policy_generators(cfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        print(f"[block2] trained policy generators: {discovered}")
    else:
        print("[block2] WARNING: no trained policy files discovered.")

    manifest = {
        "block": "block2_policy_training",
        "raw_feature_dir": cfg.raw_feature_dir,
        "normalized_feature_dir": cfg.normalized_feature_dir,
        "normalization_stats_path": cfg.nn_normalization_stats_path,
        "model_dir": cfg.model_dir,
        "training_result_dir": cfg.training_result_dir,
        "nn_policy_generators": list(cfg.nn_policy_generators),
        "ran_feature_building": bool(cfg.run_feature_building),
        "ran_nn_training": bool(cfg.run_nn_training),
        "wall_time_seconds": wall_time,
    }
    write_manifest("block2_policy_training", manifest, cfg)
    return manifest

if __name__ == "__main__":
    run()
