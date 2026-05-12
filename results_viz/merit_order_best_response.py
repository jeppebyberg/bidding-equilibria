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
    valid = capacities > 1e-9
    order = [
        int(idx)
        for idx in np.lexsort((np.arange(len(values)), values))
        if bool(valid[int(idx)])
    ]
    if not order:
        raise ValueError("No positive-capacity blocks are available for this scenario/time")
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


def _history_updates_for_time(
    result: Dict[str, Any],
    scenario_idx: int,
    time_step: int,
) -> List[Dict[str, Any]]:
    return [
        row
        for row in result.get("history", [])
        if bool(row.get("accepted"))
        and int(row.get("scenario_id", -1)) == int(scenario_idx)
        and int(row.get("time_id", -1)) == int(time_step)
    ]


def _reconstruct_ed_bids(
    result: Dict[str, Any],
    final_bids: np.ndarray,
    scenario_idx: int,
    time_step: int,
) -> np.ndarray:
    ed_bids = final_bids.astype(np.float64).copy()
    for update in _history_updates_for_time(result, scenario_idx, time_step):
        block_id = int(update["block_id"])
        ed_bids[block_id] = float(update["old_bid"])
    return ed_bids


def _block_span(stack: Dict[str, Any], block_idx: int) -> Optional[Tuple[float, float, float]]:
    matches = np.where(np.asarray(stack["order"], dtype=int) == int(block_idx))[0]
    if len(matches) == 0:
        return None
    local_idx = int(matches[0])
    left = float(stack["edges"][local_idx])
    right = float(stack["edges"][local_idx + 1])
    value = float(stack["values"][local_idx])
    return left, right, value


def _clearing_from_stack(stack: Dict[str, Any], demand: float) -> float:
    if demand <= 0:
        return float(stack["values"][0])
    idx = int(np.searchsorted(stack["cumulative"], demand, side="left"))
    idx = min(idx, len(stack["values"]) - 1)
    return float(stack["values"][idx])


def _annotate_stack(
    ax: plt.Axes,
    stack: Dict[str, Any],
    block_names: List[str],
    label: str,
    color: str,
    y_offset: float,
) -> None:
    for local_idx, block_idx in enumerate(stack["order"]):
        width = float(stack["capacities"][local_idx])
        if width <= 1e-9:
            continue
        left = 0.0 if local_idx == 0 else float(stack["cumulative"][local_idx - 1])
        center = left + 0.5 * width
        value = float(stack["values"][local_idx])
        ax.annotate(
            f"{block_names[int(block_idx)]}\n{label}: {value:.2f}",
            xy=(center, value),
            xytext=(center, value + y_offset),
            ha="center",
            va="bottom" if y_offset >= 0 else "top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.75),
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
    ed_bids = _reconstruct_ed_bids(result, final_bids, scenario_idx, time_step)
    updates = _history_updates_for_time(result, scenario_idx, time_step)
    update = updates[-1] if updates else None
    block_costs = np.asarray(result["block_costs"], dtype=np.float64)
    capacities, width_source = _capacity_at(scenarios_df, result, scenario_idx, time_step)
    demand = _demand_at(scenarios_df, result, scenario_idx, time_step)
    original_clearing_price = float(result["final_clearing_prices"][scenario_idx][time_step])

    ed_stack = _build_stack(ed_bids, capacities)
    bid_stack = _build_stack(final_bids, capacities)
    cost_stack = _build_stack(block_costs, capacities)
    inflated_merit_price = _clearing_from_stack(bid_stack, demand)
    total_capacity = float(np.sum(capacities))

    threshold_bid = None if update is None else float(update["threshold_bid"])
    y_values = np.r_[
        final_bids,
        ed_bids,
        block_costs,
        original_clearing_price,
        inflated_merit_price,
        [] if threshold_bid is None else [threshold_bid],
    ]
    y_min = min(0.0, float(np.min(y_values)))
    y_max = float(np.max(y_values))
    y_span = max(y_max - y_min, 1.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.step(
        cost_stack["edges"],
        cost_stack["step_values"],
        where="post",
        color="#2636ff",
        linewidth=2.6,
        label="True Costs Merit Order",
    )
    ax.step(
        bid_stack["edges"],
        bid_stack["step_values"],
        where="post",
        color="#ff2b2b",
        linewidth=2.6,
        linestyle="--",
        label="Heuristic Merit Order",
    )
    ax.axvline(demand, color="#2ca02c", linewidth=2.4, label=f"Demand ({demand:.1f} MW)")
    ax.axhline(
        original_clearing_price,
        color="#111111",
        linewidth=2.0,
        linestyle=":",
        # label=f"Original Market Clearing Price ({original_clearing_price:.2f})",
    )
    ax.scatter(
        [demand],
        [inflated_merit_price],
        s=150,
        marker="s",
        color="#ff2b2b",
        edgecolor="black",
        zorder=5,
        label=f"Heuristic Clearing ({inflated_merit_price:.2f})",
    )
    if updates:
        highlighted_label_used = False
        for moved in updates:
            block_id = int(moved["block_id"])
            span = _block_span(bid_stack, block_id)
            if span is None:
                continue
            left, right, new_bid = span
            ax.axvspan(
                left,
                right,
                color="#ff2b2b",
                alpha=0.14,
                label=(
                    "Inflated Block"
                    if not highlighted_label_used
                    else None
                ),
            )
            highlighted_label_used = True
        ax.axhline(
            float(update["threshold_bid"]),
            color="#9467bd",
            linewidth=1.5,
            linestyle="-.",
        )

    _annotate_stack(ax, cost_stack, block_names, "Cost", "#2636ff", y_offset=-0.12 * y_span)
    _annotate_stack(ax, bid_stack, block_names, "Bid", "#ff2b2b", y_offset=0.08 * y_span)

    regime = "scenario"
    if scenarios_df is not None and "regime" in scenarios_df.columns:
        regime = str(scenarios_df.at[scenario_idx, "regime"])
    ax.set_title(
        f"Merit Order Comparison: Original Bids vs Inflated Bids\n"
        f"Regime {regime}, Scenario {scenario_idx}, Time {time_step}",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(f"Cumulative {width_source.title()} (MW)", fontsize=12)
    ax.set_ylabel("Price / Bid ($/MWh)", fontsize=12)
    ax.set_xlim(0.0, max(total_capacity, demand) * 1.05)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.22 * y_span)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"one_pass_merit_order_s{scenario_idx:03d}_t{time_step:02d}.png"
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
        print(f"Saved one-pass merit-order diagnostic to {saved}")
