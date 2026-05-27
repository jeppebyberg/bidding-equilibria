from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from models.PoA.PoA_tightening.tightening_main import (
    DEFAULT_TIGHTENING_OUTPUT_PATHS,
    PoATighteningMain,
)


def _block_capacity_big_m(optimizer: Any, physical_generator_idx: int, local_block_idx: int) -> float:
    global_block = optimizer.local_to_global_block[
        (physical_generator_idx, local_block_idx)
    ]

    if physical_generator_idx in optimizer.conventional_physical_generator_ids:
        return float(optimizer.static_block_capacity[global_block])

    local_blocks = optimizer.local_blocks_by_generator[physical_generator_idx]
    if len(local_blocks) == 1:
        return float(optimizer.support_wind_max[physical_generator_idx])

    static_total = sum(
        optimizer.static_block_capacity[
            optimizer.local_to_global_block[(physical_generator_idx, b)]
        ]
        for b in local_blocks
    )
    if static_total <= 0:
        return 0.0

    block_share = optimizer.static_block_capacity[global_block] / static_total
    return float(block_share * optimizer.support_wind_max[physical_generator_idx])

def _computed_physical_capacity_big_m(
    optimizer: Any,
    physical_generator_idx: int,
) -> float:
    if physical_generator_idx in optimizer.wind_physical_generator_ids:
        return float(optimizer.support_wind_max[physical_generator_idx])
    return float(optimizer.static_physical_capacity[physical_generator_idx])

def _physical_capacity_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return _computed_physical_capacity_big_m(optimizer, physical_generator_idx)

def _computed_ramp_up_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return float(
        optimizer.ramp_vector_up[physical_generator_idx]
        + _computed_physical_capacity_big_m(optimizer, physical_generator_idx)
    )

def _ramp_up_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return _computed_ramp_up_big_m(optimizer, physical_generator_idx)

def _computed_ramp_down_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return float(
        optimizer.ramp_vector_down[physical_generator_idx]
        + _computed_physical_capacity_big_m(optimizer, physical_generator_idx)
    )

def _ramp_down_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return _computed_ramp_down_big_m(optimizer, physical_generator_idx)

def _computed_ramp_up_initial_big_m(
    optimizer: Any,
    physical_generator_idx: int,
) -> float:
    return float(
        optimizer.p_init[physical_generator_idx]
        + optimizer.ramp_vector_up[physical_generator_idx]
    )

def _ramp_up_initial_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return _computed_ramp_up_initial_big_m(optimizer, physical_generator_idx)

def _computed_ramp_down_initial_big_m(
    optimizer: Any,
    physical_generator_idx: int,
) -> float:
    return float(
        max(
            0.0,
            _computed_physical_capacity_big_m(optimizer, physical_generator_idx)
            - optimizer.p_init[physical_generator_idx]
            + optimizer.ramp_vector_down[physical_generator_idx],
        )
    )

def _ramp_down_initial_big_m(optimizer: Any, physical_generator_idx: int) -> float:
    return _computed_ramp_down_initial_big_m(optimizer, physical_generator_idx)

def compute_primal_big_m_bounds(optimizer: Any) -> dict[str, dict[str, Any]]:
    block_capacity: dict[str, Any] = {}
    for physical_generator_idx, local_block_idx in optimizer.generator_block_pairs:
        i = int(physical_generator_idx)
        b = int(local_block_idx)
        global_block = optimizer.local_to_global_block[(i, b)]
        block_capacity[optimizer._json_key((i, b))] = {
            "big_m": _block_capacity_big_m(optimizer, i, b),
            "physical_generator_index": i,
            "physical_generator_name": optimizer.physical_generator_names[i],
            "local_block_index": b,
            "global_block_index": int(global_block),
            "block_name": optimizer.block_names[int(global_block)],
        }

    physical_capacity: dict[str, Any] = {}
    ramp_up: dict[str, Any] = {}
    ramp_down: dict[str, Any] = {}
    ramp_up_initial: dict[str, Any] = {}
    ramp_down_initial: dict[str, Any] = {}
    for physical_generator_idx in range(optimizer.num_physical_generators):
        i = int(physical_generator_idx)
        common = {
            "physical_generator_index": i,
            "physical_generator_name": optimizer.physical_generator_names[i],
            "is_wind": i in optimizer.wind_physical_generator_ids,
        }
        physical_capacity[str(i)] = {
            "big_m": _physical_capacity_big_m(optimizer, i),
            **common,
        }
        ramp_up[str(i)] = {
            "big_m": _ramp_up_big_m(optimizer, i),
            "ramp_limit": float(optimizer.ramp_vector_up[i]),
            **common,
        }
        ramp_down[str(i)] = {
            "big_m": _ramp_down_big_m(optimizer, i),
            "ramp_limit": float(optimizer.ramp_vector_down[i]),
            **common,
        }
        ramp_up_initial[str(i)] = {
            "big_m": _ramp_up_initial_big_m(optimizer, i),
            "p_init": float(optimizer.p_init[i]),
            "ramp_limit": float(optimizer.ramp_vector_up[i]),
            **common,
        }
        ramp_down_initial[str(i)] = {
            "big_m": _ramp_down_initial_big_m(optimizer, i),
            "p_init": float(optimizer.p_init[i]),
            "ramp_limit": float(optimizer.ramp_vector_down[i]),
            **common,
        }

    return {
        "block_capacity": block_capacity,
        "physical_capacity": physical_capacity,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down,
        "ramp_up_initial": ramp_up_initial,
        "ramp_down_initial": ramp_down_initial,
    }

def support_set_summary(optimizer: Any) -> dict[str, Any]:
    return {
        "demand": {
            "reference": list(optimizer.support_demand_reference),
            "min": float(optimizer.support_demand_min),
            "max": float(optimizer.support_demand_max),
            "ramp": float(optimizer.support_demand_ramp),
            "budget": float(optimizer.support_demand_budget),
        },
        "wind": {
            optimizer.physical_generator_names[i]: {
                "reference": list(optimizer.support_wind_reference[i]),
                "min": float(optimizer.support_wind_min[i]),
                "max": float(optimizer.support_wind_max[i]),
            }
            for i in optimizer.wind_physical_generator_ids
        },
        "wind_ramp": float(optimizer.support_wind_ramp),
        "wind_budget": float(optimizer.support_wind_budget),
    }

def summarize_primal_big_m(primal_big_m: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for component_name, entries in primal_big_m.items():
        values = [
            float(details["big_m"])
            for details in entries.values()
            if isinstance(details, dict) and details.get("big_m") is not None
        ]
        summary[component_name] = {
            "entries": len(entries),
            "min_big_m": min(values) if values else None,
            "max_big_m": max(values) if values else None,
        }
    return summary
class PrimalBigMComputer(PoATighteningMain):
    def run_primal_big_m(self, output_path: str | Path | None = None) -> dict[str, Any]:
        output_path = output_path or DEFAULT_TIGHTENING_OUTPUT_PATHS["primal_big_m"]
        start = time.perf_counter()
        primal_big_m = compute_primal_big_m_bounds(self.poa)
        elapsed = time.perf_counter() - start
        report = {
            "metadata": {
                "description": (
                    "Analytic primal slack Big-M values used by PoAOptimization "
                    "KKT complementarity constraints."
                ),
                "reference_case": self.poa.reference_case,
                "num_time_steps": self.poa.num_time_steps,
                "physical_generator_names": list(self.poa.physical_generator_names),
                "block_names": list(self.poa.block_names),
                "support_set": support_set_summary(self.poa),
                "summary": summarize_primal_big_m(primal_big_m),
                "runtime_seconds": elapsed,
            },
            "primal_big_m": primal_big_m,
        }
        self.poa.primal_big_m = primal_big_m
        self.tightening_data["primal_big_m"] = primal_big_m
        print(f"Primal Big-M values obtained: {output_path}", flush=True)
        return self._save_stage_report("primal_big_m", report, output_path)
