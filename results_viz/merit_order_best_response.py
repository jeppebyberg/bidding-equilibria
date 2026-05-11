from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.scenarios.scenario_generator import ScenarioManager


RESULTS_PATH = Path("results/merit_order_best_response_results.json")
OUTPUT_DIR = Path("results_viz/figures/merit_order_best_response")

CASE = "test_case_bidding_blocks"
REGIME_SET = "policy_training"
SEED = 1

SCENARIO_IDX = 0
TIME_STEP = 0


def _as_profile(value: Any) -> List[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected a profile-like value, got {type(value).__name__}")
    return [float(v) for v in value]


def _load_scenarios() -> Optional[pd.DataFrame]:
    try:
        manager = ScenarioManager(CASE)
        scenarios = manager.create_scenario_set_from_regimes(
            regime_set=REGIME_SET,
            seed=SEED,
        )
        return scenarios["scenarios_df"].reset_index(drop=True)
    except Exception as exc:
        print(f"Could not rebuild scenarios for capacity widths: {exc}")
        print("Falling back to final dispatched quantities as merit-order widths.")
        return None


def _capacity_at(
    scenarios_df: Optional[pd.DataFrame],
    result: Dict[str, Any],
    scenario_idx: int,
    time_step: int,
) -> Tuple[np.ndarray, str]:
    if scenarios_df is None:
        return np.asarray(result["final_block_dispatches"][scenario_idx][time_step], dtype=np.float64), "dispatch"

    capacities = []
    for block_name in result["block_names"]:
        cap_profile_col = f"{block_name}_cap_profile"
        availability_col = f"{block_name}_profile"
        if cap_profile_col in scenarios_df.columns:
            capacities.append(_as_profile(scenarios_df.at[scenario_idx, cap_profile_col])[time_step])
        elif availability_col in scenarios_df.columns:
            capacities.append(_as_profile(scenarios_df.at[scenario_idx, availability_col])[time_step])
        else:
            capacities.append(float(scenarios_df.at[scenario_idx, f"{block_name}_cap"]))
    return np.asarray(capacities, dtype=np.float64), "available capacity"


def _demand_at(
    scenarios_df: Optional[pd.DataFrame],
    result: Dict[str, Any],
    scenario_idx: int,
    time_step: int,
) -> float:
    if scenarios_df is None or "demand_profile" not in scenarios_df.columns:
        return float(np.sum(result["final_block_dispatches"][scenario_idx][time_step]))
    return _as_profile(scenarios_df.at[scenario_idx, "demand_profile"])[time_step]


def _build_stack(values: np.ndarray, capacities: np.ndarray) -> Dict[str, Any]:
    order = np.lexsort((np.arange(len(values)), values))
    ordered_values = values[order]
    ordered_capacities = capacities[order]
    cumulative = np.cumsum(ordered_capacities)
    edges = np.r_[0.0, cumulative]
    step_values = np.r_[ordered_values, ordered_values[-1]]
    return {
        "order": order,
        "values": ordered_values,
        "capacities": ordered_capacities,
        "cumulative": cumulative,
        "edges": edges,
        "step_values": step_values,
    }


def _clearing_from_stack(stack: Dict[str, Any], demand: float) -> float:
    if demand <= 0:
        return float(stack["values"][0])
    idx = int(np.searchsorted(stack["cumulative"], demand, side="left"))
    idx = min(idx, len(stack["values"]) - 1)
    return float(stack["values"][idx])


def _annotate_blocks(ax: plt.Axes, stack: Dict[str, Any], block_names: List[str], y_offset: float) -> None:
    for local_idx, block_idx in enumerate(stack["order"]):
        width = float(stack["capacities"][local_idx])
        if width <= 1e-9:
            continue
        left = 0.0 if local_idx == 0 else float(stack["cumulative"][local_idx - 1])
        center = left + 0.5 * width
        ax.text(
            center,
            float(stack["values"][local_idx]) + y_offset,
            block_names[int(block_idx)],
            ha="center",
            va="bottom" if y_offset >= 0 else "top",
            rotation=35,
            fontsize=8,
        )


def plot_final_merit_order(
    result: Dict[str, Any],
    scenarios_df: Optional[pd.DataFrame],
    scenario_idx: int,
    time_step: int,
    output_dir: Path,
) -> Path:
    block_names = list(result["block_names"])
    final_bids = np.asarray(result["final_bids"][scenario_idx], dtype=np.float64)[:, time_step]
    block_costs = np.asarray(result["block_costs"], dtype=np.float64)
    capacities, width_source = _capacity_at(scenarios_df, result, scenario_idx, time_step)
    demand = _demand_at(scenarios_df, result, scenario_idx, time_step)
    clearing_price = float(result["final_clearing_prices"][scenario_idx][time_step])

    bid_stack = _build_stack(final_bids, capacities)
    cost_stack = _build_stack(block_costs, capacities)
    merit_price = _clearing_from_stack(bid_stack, demand)
    total_capacity = float(np.sum(capacities))

    y_values = np.r_[final_bids, block_costs, clearing_price, merit_price]
    y_min = min(0.0, float(np.min(y_values)))
    y_max = float(np.max(y_values))
    y_span = max(y_max - y_min, 1.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.step(
        bid_stack["edges"],
        bid_stack["step_values"],
        where="post",
        color="#d62728",
        linewidth=2.6,
        label="Final Merit Order (Bids)",
    )
    ax.step(
        cost_stack["edges"],
        cost_stack["step_values"],
        where="post",
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        label="Cost Reference",
    )
    ax.axvline(demand, color="#2ca02c", linewidth=2.2, label=f"Demand ({demand:.1f} MW)")
    ax.axhline(
        clearing_price,
        color="#111111",
        linewidth=1.9,
        linestyle=":",
        label=f"ED Clearing Price ({clearing_price:.2f})",
    )
    ax.scatter(
        [demand],
        [merit_price],
        s=120,
        color="#d62728",
        edgecolor="black",
        zorder=5,
        label=f"Merit-Order Price ({merit_price:.2f})",
    )
    ax.scatter(
        [demand],
        [clearing_price],
        s=110,
        marker="D",
        color="#111111",
        edgecolor="white",
        zorder=6,
    )

    _annotate_blocks(ax, bid_stack, block_names, y_offset=0.04 * y_span)

    ax.set_title(f"Final Merit Order Curve - Scenario {scenario_idx}, Time {time_step}")
    ax.set_xlabel(f"Cumulative {width_source.title()} (MW)")
    ax.set_ylabel("Price / Bid")
    ax.set_xlim(0.0, max(total_capacity, demand) * 1.05)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.22 * y_span)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"final_merit_order_s{scenario_idx:03d}_t{time_step:02d}.png"
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    result = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    scenarios_df = _load_scenarios()
    for scenario_idx in range(len(result["final_bids"])):
        saved = plot_final_merit_order(
            result=result,
            scenarios_df=scenarios_df,
            scenario_idx=scenario_idx,
            time_step=TIME_STEP,
            output_dir=OUTPUT_DIR,
        )
        print(f"Saved final merit-order curve to {saved}")
