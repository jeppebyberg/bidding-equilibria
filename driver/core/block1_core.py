from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config.scenarios.scenario_generator import ScenarioManager
from models.helper import sanitize_for_json
from models.synthetic_data_generation.merit_order_best_response import MeritOrderHeuristic

from driver.core.block0_core import ProjectConfig


def apply_time_steps_override(manager: ScenarioManager, time_steps: int | None) -> None:
    if time_steps is None:
        return
    if time_steps <= 0:
        raise ValueError(f"time_steps must be positive, got {time_steps}")
    manager.base_case["time_steps"] = int(time_steps)


def save_scenario_tables(scenarios: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios["scenarios_df"].to_csv(output_dir / "scenarios.csv", index=False)
    scenarios["costs_df"].to_csv(output_dir / "costs.csv", index=False)
    scenarios["ramps_df"].to_csv(output_dir / "ramps.csv", index=False)


def load_or_generate_scenarios(
    config: ProjectConfig,
    manager: ScenarioManager,
    n_scenarios: int,
    seed: int,
    output_dir: Path,
    should_generate: bool,
    time_steps: int | None,
) -> dict[str, Any]:
    apply_time_steps_override(manager, time_steps)
    scenarios = manager.create_scenario_set_from_ambiguity_set(
        ambiguity_config_path=config.ambiguity_set_config_path,
        ambiguity_set=config.ambiguity_set_config_name,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    if should_generate:
        save_scenario_tables(scenarios, output_dir)
    print(scenarios["description_text"])
    return scenarios


def run_heuristic(
    config: ProjectConfig,
    scenarios: dict[str, Any],
    scenario_manager: ScenarioManager,
) -> Path:
    start = time.perf_counter()
    heuristic = MeritOrderHeuristic(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        players_config=scenario_manager.get_players_config(),
        bid_tolerance=config.bid_tolerance,
    )
    heuristic.run()
    output_path = heuristic.save_results(config.heuristic_results_path)
    elapsed = time.perf_counter() - start
    print(f"\nSaved heuristic synthetic labels to {output_path}")
    print(f"Heuristic runtime: {elapsed:.2f} seconds")
    return output_path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(sanitize_for_json(payload), fh, indent=2)
    return path

