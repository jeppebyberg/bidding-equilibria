from __future__ import annotations

import time
from math import sqrt
from pathlib import Path
from typing import Any

from models.DRO_PoA.DRO_PoA_tightening.tightening_main import (
    DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS,
    DROPoATighteningMain,
)

_SUPPORT_SET_KAPPA = 1.96  # matches kappa in support_set.py _build_support_set_wind


def _wind_physical_capacity_ub(optimizer: Any, physical_generator_idx: int, time_idx: int) -> float:
    i = int(physical_generator_idx)
    t = int(time_idx)
    installed_capacity = float(optimizer.static_physical_capacity[i])
    if installed_capacity <= 0.0:
        return 0.0
    rho_W = float(optimizer.wind_rho_fixed)
    stationary_std_dev_W = float(optimizer.sigma_W_fixed) / sqrt(max(1e-12, 1.0 - rho_W ** 2))
    profile_upper = installed_capacity * (
        float(optimizer.mu_W_fixed) * float(optimizer.wind_shape[t])
        + _SUPPORT_SET_KAPPA * stationary_std_dev_W
    )
    return float(min(installed_capacity, max(0.0, profile_upper)))


def _physical_capacity_ub(optimizer: Any, physical_generator_idx: int, time_idx: int) -> float:
    i = int(physical_generator_idx)
    if i in optimizer.wind_physical_generator_ids:
        return _wind_physical_capacity_ub(optimizer, i, int(time_idx))
    return float(optimizer.static_physical_capacity[i])


def _block_capacity_ub(
    optimizer: Any,
    physical_generator_idx: int,
    local_block_idx: int,
    time_idx: int,
) -> float:
    i = int(physical_generator_idx)
    b = int(local_block_idx)
    global_block = optimizer.local_to_global_block[(i, b)]
    if i in optimizer.conventional_physical_generator_ids:
        return float(optimizer.static_block_capacity[global_block])

    local_blocks = optimizer.local_blocks_by_generator[i]
    physical_capacity_ub = _physical_capacity_ub(optimizer, i, int(time_idx))
    if len(local_blocks) == 1:
        return physical_capacity_ub

    static_total = sum(
        optimizer.static_block_capacity[optimizer.local_to_global_block[(i, local_b)]]
        for local_b in local_blocks
    )
    if static_total <= 0.0:
        return 0.0
    block_share = float(optimizer.static_block_capacity[global_block]) / float(static_total)
    return float(block_share * physical_capacity_ub)


def compute_primal_big_m_bounds(optimizer: Any) -> dict[str, dict[str, Any]]:
    block_capacity: dict[str, Any] = {}
    lower_generation: dict[str, Any] = {}
    for i, b in optimizer.generator_block_pairs:
        i = int(i)
        b = int(b)
        global_block = optimizer.local_to_global_block[(i, b)]
        for t in range(optimizer.num_time_steps):
            key = f"{i},{b},{t}"
            big_m = _block_capacity_ub(optimizer, i, b, t)
            common = {
                "big_m": big_m,
                "physical_generator_index": i,
                "physical_generator_name": optimizer.physical_generator_names[i],
                "local_block_index": b,
                "global_block_index": int(global_block),
                "block_name": optimizer.block_names[int(global_block)],
                "time_index": int(t),
            }
            block_capacity[key] = dict(common)
            lower_generation[key] = dict(common)

    physical_capacity: dict[str, Any] = {}
    ramp_up: dict[str, Any] = {}
    ramp_down: dict[str, Any] = {}
    ramp_up_initial: dict[str, Any] = {}
    ramp_down_initial: dict[str, Any] = {}
    for i in range(optimizer.num_physical_generators):
        common = {
            "physical_generator_index": int(i),
            "physical_generator_name": optimizer.physical_generator_names[i],
            "is_wind": i in optimizer.wind_physical_generator_ids,
        }
        max_physical_capacity = 0.0
        for t in range(optimizer.num_time_steps):
            capacity_ub = _physical_capacity_ub(optimizer, i, t)
            max_physical_capacity = max(max_physical_capacity, capacity_ub)
            physical_capacity[f"{i},{t}"] = {
                "big_m": capacity_ub,
                "time_index": int(t),
                **common,
            }
            ramp_up[f"{i},{t}"] = {
                "big_m": float(optimizer.ramp_vector_up[i] + capacity_ub),
                "ramp_limit": float(optimizer.ramp_vector_up[i]),
                "time_index": int(t),
                **common,
            }
            ramp_down[f"{i},{t}"] = {
                "big_m": float(optimizer.ramp_vector_down[i] + capacity_ub),
                "ramp_limit": float(optimizer.ramp_vector_down[i]),
                "time_index": int(t),
                **common,
            }

        p_init_values = [
            float(optimizer.p_init[k][i]) for k in range(optimizer.num_empirical_scenarios)
        ]
        ramp_up_initial[str(i)] = {
            "big_m": max(value + float(optimizer.ramp_vector_up[i]) for value in p_init_values),
            "aggregation": "max over empirical scenarios k",
            "ramp_limit": float(optimizer.ramp_vector_up[i]),
            **common,
        }
        ramp_down_initial[str(i)] = {
            "big_m": max(
                max(0.0, max_physical_capacity - value + float(optimizer.ramp_vector_down[i]))
                for value in p_init_values
            ),
            "aggregation": "max over empirical scenarios k",
            "ramp_limit": float(optimizer.ramp_vector_down[i]),
            **common,
        }

    return {
        "block_capacity": block_capacity,
        "lower_generation": lower_generation,
        "physical_capacity": physical_capacity,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down,
        "ramp_up_initial": ramp_up_initial,
        "ramp_down_initial": ramp_down_initial,
    }


def summarize_primal_big_m(primal_big_m: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for component_name, entries in primal_big_m.items():
        values = [
            float(details["big_m"])
            for details in (entries or {}).values()
            if isinstance(details, dict) and details.get("big_m") is not None
        ]
        summary[component_name] = {
            "entries": len(entries or {}),
            "min_big_m": min(values) if values else None,
            "max_big_m": max(values) if values else None,
        }
    return summary


class DROPrimalBigMComputer(DROPoATighteningMain):
    def compute_primal_big_m_bounds(self, optimizer: Any) -> dict[str, dict[str, Any]]:
        return compute_primal_big_m_bounds(optimizer)

    def run_primal_big_m(self, output_path: str | Path | None = None) -> dict[str, Any]:
        output_path = output_path or self._resolve_output_paths()["primal_big_m"]
        start = time.perf_counter()
        primal_big_m = self.compute_primal_big_m_bounds(self.poa)
        elapsed = time.perf_counter() - start

        report = {
            "metadata": {
                **self._metadata(),
                "description": (
                    "Regime-wide analytic primal slack Big-M values for the "
                    "scenario-indexed DRO PoA KKT complementarity constraints."
                ),
                "summary": summarize_primal_big_m(primal_big_m),
                "runtime_seconds": elapsed,
            },
            "primal_big_m": primal_big_m,
        }
        self.poa.primal_big_m = primal_big_m
        self.tightening_data["primal_big_m"] = primal_big_m
        if hasattr(self.poa, "_loaded_bounds_prepared"):
            self.poa._loaded_bounds_prepared = False
        return self._save_stage_report("primal_big_m", report, output_path)
