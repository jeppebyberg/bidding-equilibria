"""Visualize trained NN bidding policies as merit order curves.

Compares three strategies on the held-out test scenarios:
  True costs (blue solid)  -- the competitive benchmark
  Label bids  (red dashed) -- heuristic best-response used as training targets
  NN policy   (green dot)  -- predictions of the trained bidding policy network

Produces two figure types:
  1. Merit order curves for a grid of (scenario, time_step) selections.
  2. Scatter plots of NN predictions vs. label bids across the full test set,
     one subplot per generator block.

Paths and settings are configured in main() at the bottom of the file.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.neural_network.training.model import BiddingPolicyNetwork


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _as_profile(value: Any, expected_len: int) -> list[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        profile = [float(v) for v in value]
        if len(profile) != expected_len:
            raise ValueError(f"Profile length {len(profile)} != expected {expected_len}")
        return profile
    raise ValueError(f"Expected list-like, got {type(value).__name__}")


def _load_result_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scenarios_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).reset_index(drop=True)


def _load_normalization_stats(stats_path: Path) -> dict[str, dict[str, float]]:
    """Return {generator_name: {'feature_min': ..., 'feature_max': ...}}."""
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    if raw.get("per_generator"):
        return {gen: raw["stats"][gen] for gen in raw["stats"]}
    return {"_shared": raw["stats"]}


# ---------------------------------------------------------------------------
# NN policy loading and inference
# ---------------------------------------------------------------------------


class NNPolicy:
    """Wraps a trained BiddingPolicyNetwork with its metadata for inference."""

    def __init__(self, model: BiddingPolicyNetwork, metadata: dict[str, Any]) -> None:
        self.model = model
        self.generator_name: str = metadata["generator_name"]
        self.feature_columns: list[str] = metadata["feature_columns"]
        self.target_columns: list[str] = metadata["target_columns"]
        # Maps target column name → block_name (e.g. "target_bid_G1_B1" → "G1_B1")
        self.block_names: list[str] = [
            col.replace("target_bid_", "") for col in self.target_columns
        ]


def load_nn_policy(model_dir: Path, generator_name: str) -> NNPolicy:
    """Load trained model and metadata for one generator."""
    meta_path = model_dir / f"{generator_name}_policy_metadata.json"
    model_path = model_dir / f"{generator_name}_policy.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model = BiddingPolicyNetwork(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_layers=metadata["hidden_layers"],
        final_activation=metadata.get("final_activation", "linear"),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return NNPolicy(model, metadata)


def predict_bids(policy: NNPolicy, feature_row: np.ndarray) -> np.ndarray:
    """Run one feature row through the model. Returns predicted bids in $/MWh."""
    x = torch.tensor(feature_row.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        return policy.model(x).numpy()[0]


def warn_if_label_artifacts_disagree(
    result: dict[str, Any],
    normalized_dfs: dict[str, pd.DataFrame],
    nn_policies: dict[str, NNPolicy],
    tolerance: float = 1e-8,
    max_examples: int = 8,
) -> None:
    """Print a short warning if feature targets differ from result JSON labels."""
    block_names: list[str] = result["block_names"]
    final_bids = result["final_bids"]
    mismatch_count = 0
    examples: list[str] = []

    for gen_name, policy in sorted(nn_policies.items()):
        ndf = normalized_dfs.get(gen_name)
        if ndf is None:
            continue
        for block_name, target_col in zip(policy.block_names, policy.target_columns):
            if block_name not in block_names or target_col not in ndf.columns:
                continue
            block_idx = block_names.index(block_name)
            cols = ["scenario_id", "time_id", target_col]
            for _, row in ndf[cols].iterrows():
                scenario_id = int(row["scenario_id"])
                time_id = int(row["time_id"])
                feature_value = float(row[target_col])
                result_value = float(final_bids[scenario_id][block_idx][time_id])
                if abs(feature_value - result_value) <= tolerance:
                    continue
                mismatch_count += 1
                if len(examples) < max_examples:
                    examples.append(
                        f"    s={scenario_id}, t={time_id}, {block_name}: "
                        f"feature target={feature_value:.6g}, "
                        f"result JSON={result_value:.6g}"
                    )

    if mismatch_count == 0:
        return

    print(
        "\nWARNING: feature target labels differ from "
        "results['final_bids'] for loaded NN generators."
    )
    print(
        "  The visualizer uses results['final_bids'] as the authoritative "
        "ramp-constrained heuristic labels."
    )
    print(
        "  Rebuild features and retrain policies before interpreting NN "
        "prediction errors against the current heuristic labels."
    )
    print(f"  Mismatched rows: {mismatch_count}")
    print("  Examples:")
    for example in examples:
        print(example)


# ---------------------------------------------------------------------------
# Test scenario identification
# ---------------------------------------------------------------------------


def get_test_scenarios(
    normalized_csv: Path,
    test_fraction: float,
    random_state: int,
) -> list[int]:
    """Reproduce GroupShuffleSplit to return test scenario_ids (0-indexed)."""
    df = pd.read_csv(normalized_csv)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=random_state
    )
    _, test_idx = next(splitter.split(df, groups=df["scenario_id"]))
    return sorted(df.iloc[test_idx]["scenario_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Capacity and demand helpers (mirrors merit_order_best_response.py logic)
# ---------------------------------------------------------------------------


def _block_capacity(
    scenarios_df: pd.DataFrame,
    result: dict[str, Any],
    scenario_idx: int,
    time_step: int,
    block_name: str,
    num_time_steps: int,
) -> float:
    for col in (f"{block_name}_cap_profile", f"{block_name}_profile"):
        if col in scenarios_df.columns:
            return _as_profile(scenarios_df.at[scenario_idx, col], num_time_steps)[time_step]
    cap_col = f"{block_name}_cap"
    if cap_col in scenarios_df.columns:
        return float(scenarios_df.at[scenario_idx, cap_col])
    # Last resort: fall back to dispatched quantity
    fbd = result.get("final_block_dispatches")
    if fbd:
        block_idx = result["block_names"].index(block_name)
        return float(fbd[scenario_idx][time_step][block_idx])
    raise KeyError(f"Cannot find capacity for block '{block_name}'")


def _demand_value(
    scenarios_df: pd.DataFrame,
    scenario_idx: int,
    time_step: int,
    num_time_steps: int,
) -> float:
    if "demand_profile" in scenarios_df.columns:
        return _as_profile(scenarios_df.at[scenario_idx, "demand_profile"], num_time_steps)[time_step]
    return float(scenarios_df.at[scenario_idx, "demand"])


# ---------------------------------------------------------------------------
# Bid vector assembly
# ---------------------------------------------------------------------------


def build_bid_vectors(
    scenario_id: int,
    time_step: int,
    result: dict[str, Any],
    scenarios_df: pd.DataFrame,
    normalized_dfs: dict[str, pd.DataFrame],
    nn_policies: dict[str, NNPolicy],
) -> dict[str, np.ndarray]:
    """Return true-cost, label, and NN bid vectors for all blocks.

    For NN-policy generators the NN vector uses model predictions.
    For other generators the NN vector falls back to true costs.
    """
    block_names: list[str] = result["block_names"]
    block_costs = np.asarray(result["block_costs"], dtype=float)
    label_bids = np.asarray(result["final_bids"][scenario_id], dtype=float)[:, time_step]

    nn_bids = block_costs.copy()
    for gen_name, policy in nn_policies.items():
        ndf = normalized_dfs[gen_name]
        row = ndf.loc[(ndf["scenario_id"] == scenario_id) & (ndf["time_id"] == time_step)]
        if row.empty:
            continue
        feat = row[policy.feature_columns].to_numpy(dtype=np.float32)[0]
        predicted = predict_bids(policy, feat)
        for i, block_name in enumerate(policy.block_names):
            if block_name in block_names:
                nn_bids[block_names.index(block_name)] = float(predicted[i])

    return {"true_cost": block_costs, "label": label_bids, "nn": nn_bids}


# ---------------------------------------------------------------------------
# Merit order plot helpers
# ---------------------------------------------------------------------------


def _build_stack(bids: np.ndarray, capacities: np.ndarray) -> dict[str, Any]:
    valid = capacities > 1e-9
    order = [
        int(i)
        for i in np.lexsort((np.arange(len(bids)), bids))
        if bool(valid[int(i)])
    ]
    if not order:
        raise ValueError("No positive-capacity blocks available")
    ordered_bids = bids[order]
    ordered_caps = capacities[order]
    cumulative = np.cumsum(ordered_caps)
    return {
        "order": order,
        "bids": ordered_bids,
        "caps": ordered_caps,
        "cumulative": cumulative,
        "edges": np.r_[0.0, cumulative],
        "step_values": np.r_[ordered_bids, ordered_bids[-1]],
    }


def _clearing_price(stack: dict[str, Any], demand: float) -> float:
    idx = min(
        int(np.searchsorted(stack["cumulative"], max(demand, 0.0), side="left")),
        len(stack["bids"]) - 1,
    )
    return float(stack["bids"][idx])


def _annotate_stack(
    ax: plt.Axes,
    stack: dict[str, Any],
    block_names: list[str],
    block_to_physical: dict[str, str],
    bid_label: str,
    color: str,
    y_offset: float,
) -> None:
    """Annotate each block in a merit order stack with its name and bid value.

    Each annotation box is placed at (center_x, bid_value + y_offset) and
    connected to the step with an arrow. y_offset > 0 places labels above the
    step; y_offset < 0 places them below.
    """
    for local_idx, block_idx in enumerate(stack["order"]):
        cap = float(stack["caps"][local_idx])
        if cap <= 1e-9:
            continue
        left = 0.0 if local_idx == 0 else float(stack["cumulative"][local_idx - 1])
        center_x = left + 0.5 * cap
        bid_val = float(stack["bids"][local_idx])
        block_name = block_names[block_idx]
        physical = block_to_physical.get(block_name, block_name)
        label_text = f"{physical}\n({block_name})\n{bid_label}: {bid_val:.1f}"
        ax.annotate(
            label_text,
            xy=(center_x, bid_val),
            xytext=(center_x, bid_val + y_offset),
            ha="center",
            va="bottom" if y_offset >= 0 else "top",
            fontsize=6.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.75),
        )


def plot_merit_order_comparison(
    time_step: int,
    bid_vectors: dict[str, np.ndarray],
    capacities: np.ndarray,
    demand: float,
    block_names: list[str],
    block_to_physical: dict[str, str],
    ax: plt.Axes,
) -> None:
    """Plot three merit order curves (true cost / label / NN) on a single Axes.

    Annotates the true-cost stack below each step and the NN-policy stack
    above each step so it is clear which generator owns which block and what
    it is bidding under each strategy.
    """
    tc_stack = _build_stack(bid_vectors["true_cost"], capacities)
    lb_stack = _build_stack(bid_vectors["label"], capacities)
    nn_stack = _build_stack(bid_vectors["nn"], capacities)

    tc_price = _clearing_price(tc_stack, demand)
    lb_price = _clearing_price(lb_stack, demand)
    nn_price = _clearing_price(nn_stack, demand)

    total_cap = float(np.sum(capacities))
    y_all = np.concatenate([tc_stack["bids"], lb_stack["bids"], nn_stack["bids"]])
    y_span = float(np.ptp(y_all)) or 1.0

    ax.step(tc_stack["edges"], tc_stack["step_values"], where="post",
            color="#2636ff", linewidth=2.2, label=f"True costs  (CP={tc_price:.1f})")
    ax.step(lb_stack["edges"], lb_stack["step_values"], where="post",
            color="#ff2b2b", linewidth=2.0, linestyle="--", label=f"Label bids  (CP={lb_price:.1f})")
    ax.step(nn_stack["edges"], nn_stack["step_values"], where="post",
            color="#2ca02c", linewidth=2.0, linestyle=":", label=f"NN policy   (CP={nn_price:.1f})")

    ax.axvline(demand, color="gray", linewidth=1.6, linestyle="-.",
               label=f"Demand {demand:.0f} MW")

    # Annotate true-cost stack below steps and NN stack above steps.
    _annotate_stack(ax, tc_stack, block_names, block_to_physical,
                    "cost", "#2636ff", -0.22 * y_span)
    _annotate_stack(ax, nn_stack, block_names, block_to_physical,
                    "NN", "#2ca02c", +0.18 * y_span)

    ax.set_xlim(0.0, total_cap * 1.05)
    ax.set_ylim(
        float(np.min(y_all)) - 0.55 * y_span,
        float(np.max(y_all)) + 0.45 * y_span,
    )
    ax.set_title(f"t = {time_step}", fontsize=10)
    ax.set_xlabel("Cumulative capacity (MW)", fontsize=8)
    ax.set_ylabel("Bid ($/MWh)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)


def make_scenario_figure(
    scenario_id: int,
    time_steps: list[int],
    result: dict[str, Any],
    scenarios_df: pd.DataFrame,
    normalized_dfs: dict[str, pd.DataFrame],
    nn_policies: dict[str, NNPolicy],
    output_path: Path,
    show: bool = False,
) -> Path:
    """One figure per scenario: each time step is a separate subplot column."""
    block_names: list[str] = result["block_names"]
    block_to_physical: dict[str, str] = result["block_to_physical"]
    nn_gen_names = list(nn_policies.keys())
    num_time_steps = int(result["num_time_steps"])

    ncols = len(time_steps)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 9.0), squeeze=False)
    fig.suptitle(
        f"Scenario {scenario_id}  —  Merit Order: True Costs vs. Label Bids vs. NN Policy\n"
        f"NN generators: {', '.join(nn_gen_names)}  |  others bid at true cost",
        fontsize=11,
    )

    for col, time_step in enumerate(time_steps):
        ax = axes[0][col]
        capacities = np.array([
            _block_capacity(scenarios_df, result, scenario_id, t, block_names[bi], num_time_steps)
            for bi, t in [(bi, time_step) for bi in range(len(block_names))]
        ], dtype=float)
        demand = _demand_value(scenarios_df, scenario_id, time_step, num_time_steps)
        bid_vectors = build_bid_vectors(
            scenario_id, time_step, result, scenarios_df, normalized_dfs, nn_policies
        )
        plot_merit_order_comparison(
            time_step, bid_vectors, capacities, demand,
            block_names, block_to_physical, ax,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Scatter plot: NN predictions vs. label bids (test set)
# ---------------------------------------------------------------------------


def make_scatter_figure(
    nn_policies: dict[str, NNPolicy],
    normalized_dfs: dict[str, pd.DataFrame],
    result: dict[str, Any],
    test_scenarios: list[int],
    output_path: Path,
    show: bool = False,
) -> Path:
    """Scatter plot of NN predictions vs. label bids on held-out test scenarios."""
    # Collect all (block_name, label, prediction) triplets from the test set.
    all_blocks: dict[str, tuple[list[float], list[float]]] = {}  # block -> (labels, preds)

    for gen_name, policy in nn_policies.items():
        ndf = normalized_dfs[gen_name]
        test_rows = ndf[ndf["scenario_id"].isin(test_scenarios)]
        if test_rows.empty:
            continue

        feat = test_rows[policy.feature_columns].to_numpy(dtype=np.float32)
        x = torch.tensor(feat, dtype=torch.float32)
        with torch.no_grad():
            preds = policy.model(x).numpy()  # (N, output_dim)

        for col_idx, (block, _target_col) in enumerate(
            zip(policy.block_names, policy.target_columns)
        ):
            if block not in result["block_names"]:
                continue
            block_idx = result["block_names"].index(block)
            labels = np.array(
                [
                    float(result["final_bids"][int(row.scenario_id)][block_idx][int(row.time_id)])
                    for row in test_rows.itertuples(index=False)
                ],
                dtype=float,
            )
            block_preds = preds[:, col_idx]
            if block not in all_blocks:
                all_blocks[block] = ([], [])
            all_blocks[block][0].extend(labels.tolist())
            all_blocks[block][1].extend(block_preds.tolist())

    if not all_blocks:
        raise ValueError("No test-set predictions found. Check test_scenarios and normalized_dfs.")

    n_blocks = len(all_blocks)
    ncols = min(n_blocks, 4)
    nrows = (n_blocks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(
        f"NN Policy: Predicted vs. Label Bids (test set, N={len(test_scenarios)} scenarios)",
        fontsize=11,
    )

    cmap = plt.get_cmap("tab10")
    for plot_idx, (block_name, (labels, preds)) in enumerate(sorted(all_blocks.items())):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]

        labels_arr = np.array(labels)
        preds_arr = np.array(preds)

        residuals = preds_arr - labels_arr
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        ax.scatter(labels_arr, preds_arr, s=8, alpha=0.35, color=cmap(plot_idx % cmap.N),
                   edgecolors="none")

        lo = min(labels_arr.min(), preds_arr.min())
        hi = max(labels_arr.max(), preds_arr.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")

        ax.set_title(f"{block_name}  |  RMSE={rmse:.3f}  MAE={mae:.3f}", fontsize=9)
        ax.set_xlabel("Label bid ($/MWh)", fontsize=8)
        ax.set_ylabel("NN predicted bid ($/MWh)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    for extra in range(n_blocks, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Bid error summary
# ---------------------------------------------------------------------------


def print_bid_error_summary(
    nn_policies: dict[str, NNPolicy],
    normalized_dfs: dict[str, pd.DataFrame],
    result: dict[str, Any],
    test_scenarios: list[int],
) -> None:
    """Print RMSE and MAE of NN predictions vs. label bids on the test set."""
    print("\nBid prediction errors on held-out test set:")
    print(f"  {'Block':<12} {'RMSE ($/MWh)':>14} {'MAE ($/MWh)':>13} {'Max err ($/MWh)':>16}")
    print("  " + "-" * 58)
    for gen_name, policy in sorted(nn_policies.items()):
        ndf = normalized_dfs[gen_name]
        test_rows = ndf[ndf["scenario_id"].isin(test_scenarios)]
        if test_rows.empty:
            continue
        feat = torch.tensor(test_rows[policy.feature_columns].to_numpy(dtype=np.float32))
        with torch.no_grad():
            preds = policy.model(feat).numpy()
        for col_idx, (block, _target_col) in enumerate(
            zip(policy.block_names, policy.target_columns)
        ):
            if block not in result["block_names"]:
                continue
            block_idx = result["block_names"].index(block)
            labels = np.array(
                [
                    float(result["final_bids"][int(row.scenario_id)][block_idx][int(row.time_id)])
                    for row in test_rows.itertuples(index=False)
                ],
                dtype=float,
            )
            err = preds[:, col_idx] - labels
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            max_err = float(np.max(np.abs(err)))
            print(f"  {block:<12} {rmse:>14.4f} {mae:>13.4f} {max_err:>16.4f}")
    print()


def find_max_error_point(
    nn_policies: dict[str, NNPolicy],
    normalized_dfs: dict[str, pd.DataFrame],
    result: dict[str, Any],
    test_scenarios: list[int],
) -> dict[str, Any] | None:
    """Locate the single (scenario, time, block) with the largest |NN - label| error.

    Returns a dict with scenario_id, time_id, block_name, generator_name, label,
    prediction and abs_error, or None if no test predictions are available.
    """
    best: dict[str, Any] | None = None

    for gen_name, policy in sorted(nn_policies.items()):
        ndf = normalized_dfs[gen_name]
        test_rows = ndf[ndf["scenario_id"].isin(test_scenarios)]
        if test_rows.empty:
            continue
        feat = torch.tensor(test_rows[policy.feature_columns].to_numpy(dtype=np.float32))
        with torch.no_grad():
            preds = policy.model(feat).numpy()

        scenario_ids = test_rows["scenario_id"].to_numpy()
        time_ids = test_rows["time_id"].to_numpy()

        for col_idx, block in enumerate(policy.block_names):
            if block not in result["block_names"]:
                continue
            block_idx = result["block_names"].index(block)
            labels = np.array(
                [
                    float(result["final_bids"][int(s)][block_idx][int(t)])
                    for s, t in zip(scenario_ids, time_ids)
                ],
                dtype=float,
            )
            abs_err = np.abs(preds[:, col_idx] - labels)
            row = int(np.argmax(abs_err))
            if best is None or abs_err[row] > best["abs_error"]:
                best = {
                    "scenario_id": int(scenario_ids[row]),
                    "time_id": int(time_ids[row]),
                    "block_name": block,
                    "generator_name": gen_name,
                    "label": float(labels[row]),
                    "prediction": float(preds[row, col_idx]),
                    "abs_error": float(abs_err[row]),
                }

    return best


def find_max_below_cost_point(
    nn_policies: dict[str, NNPolicy],
    normalized_dfs: dict[str, pd.DataFrame],
    result: dict[str, Any],
    test_scenarios: list[int],
) -> dict[str, Any] | None:
    """Locate the (scenario, time, block) where an NN bid undercuts true cost most.

    Mirrors find_max_error_point but ranks by (true_cost - NN bid), i.e. the
    largest amount by which the policy bids below the generator's marginal cost.
    Returns a dict with scenario_id, time_id, block_name, generator_name,
    true_cost, prediction and undercut (>0 means below cost), or None if no NN
    bid ever falls below cost on the test set.
    """
    block_costs = np.asarray(result["block_costs"], dtype=float)
    best: dict[str, Any] | None = None

    for gen_name, policy in sorted(nn_policies.items()):
        ndf = normalized_dfs[gen_name]
        test_rows = ndf[ndf["scenario_id"].isin(test_scenarios)]
        if test_rows.empty:
            continue
        feat = torch.tensor(test_rows[policy.feature_columns].to_numpy(dtype=np.float32))
        with torch.no_grad():
            preds = policy.model(feat).numpy()

        scenario_ids = test_rows["scenario_id"].to_numpy()
        time_ids = test_rows["time_id"].to_numpy()

        for col_idx, block in enumerate(policy.block_names):
            if block not in result["block_names"]:
                continue
            block_idx = result["block_names"].index(block)
            true_cost = float(block_costs[block_idx])
            undercut = true_cost - preds[:, col_idx]
            row = int(np.argmax(undercut))
            if best is None or undercut[row] > best["undercut"]:
                best = {
                    "scenario_id": int(scenario_ids[row]),
                    "time_id": int(time_ids[row]),
                    "block_name": block,
                    "generator_name": gen_name,
                    "true_cost": true_cost,
                    "prediction": float(preds[row, col_idx]),
                    "undercut": float(undercut[row]),
                }

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_merit_order_figures(
    result_json_path: Path,
    scenarios_csv_path: Path,
    normalized_csv_dir: Path,
    model_dir: Path,
    output_dir: Path,
    nn_generator_names: list[str] | None = None,
    n_merit_scenarios: int = 2,
    merit_time_steps: list[int] | None = None,
    sample_seed: int = 1,
    single_time_step_figures: bool = False,
    show: bool = False,
    include_max_error_figure: bool = True,
    include_below_cost_figure: bool = True,
) -> list[Path]:
    """Generate merit-order and prediction-scatter figures for trained NN policies.

    All paths are explicit so the pipeline can route outputs to a per-run folder.
    Returns the list of saved figure paths.

    When single_time_step_figures is True, each time step is rendered as its own
    figure (one merit-order curve per file) instead of being packed into the
    columns of a single per-scenario figure.
    """
    result_json_path = Path(result_json_path)
    scenarios_csv_path = Path(scenarios_csv_path)
    normalized_csv_dir = Path(normalized_csv_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if merit_time_steps is None:
        merit_time_steps = [0, 4, 12]

    # ---- Load data ------------------------------------------------------------
    result = _load_result_json(result_json_path)
    scenarios_df = _load_scenarios_df(scenarios_csv_path)
    num_time_steps = int(result["num_time_steps"])
    # Keep only time steps that exist in this result's horizon.
    merit_time_steps = sorted({t for t in merit_time_steps if 0 <= t < num_time_steps})
    if not merit_time_steps:
        merit_time_steps = list(range(min(3, num_time_steps)))

    saved_paths: list[Path] = []

    # Auto-detect NN generators from trained model files.
    if nn_generator_names is None:
        nn_generator_names = sorted(
            p.stem.replace("_policy", "")
            for p in model_dir.glob("*_policy.pt")
        )
    print(f"NN policy generators: {nn_generator_names}")

    nn_policies: dict[str, NNPolicy] = {}
    for gen_name in nn_generator_names:
        try:
            nn_policies[gen_name] = load_nn_policy(model_dir, gen_name)
            print(f"  Loaded {gen_name}: {nn_policies[gen_name].target_columns}")
        except FileNotFoundError as exc:
            print(f"  Skipping {gen_name}: {exc}")

    if not nn_policies:
        raise RuntimeError("No trained NN policies found. Run the NN training pipeline first.")

    normalized_dfs: dict[str, pd.DataFrame] = {}
    for gen_name in nn_policies:
        csv_path = normalized_csv_dir / f"{gen_name}_features_normalized.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Normalized features not found: {csv_path}")
        normalized_dfs[gen_name] = pd.read_csv(csv_path)

    warn_if_label_artifacts_disagree(result, normalized_dfs, nn_policies)

    # Identify test scenarios (using the same split as training).
    # Use the first NN-policy generator's metadata to get split params.
    first_gen = next(iter(nn_policies))
    first_meta = json.loads((model_dir / f"{first_gen}_policy_metadata.json").read_text())
    test_scenarios = get_test_scenarios(
        normalized_csv_dir / f"{first_gen}_features_normalized.csv",
        test_fraction=first_meta["test_fraction"],
        random_state=first_meta["random_state"],
    )
    print(f"Test scenarios: {len(test_scenarios)} / {result['num_scenarios']} total")

    # ---- Print bid error summary ----------------------------------------------
    print_bid_error_summary(nn_policies, normalized_dfs, result, test_scenarios)

    # ---- Figure for the single worst NN prediction error ----------------------
    if include_max_error_figure:
        worst = find_max_error_point(nn_policies, normalized_dfs, result, test_scenarios)
        if worst is None:
            print("No max-error point found (no test predictions available).")
        else:
            print(
                f"Max prediction error: block {worst['block_name']} "
                f"(scenario {worst['scenario_id']}, t={worst['time_id']}) "
                f"label={worst['label']:.3f}, NN={worst['prediction']:.3f}, "
                f"|error|={worst['abs_error']:.3f}"
            )
            p_err = make_scenario_figure(
                scenario_id=worst["scenario_id"],
                time_steps=[worst["time_id"]],
                result=result,
                scenarios_df=scenarios_df,
                normalized_dfs=normalized_dfs,
                nn_policies=nn_policies,
                output_path=output_dir
                / f"nn_policy_merit_order_max_error_s{worst['scenario_id']}"
                f"_t{worst['time_id']}_{worst['block_name']}.png",
                show=show,
            )
            print(f"Saved max-error figure:    {p_err}")
            saved_paths.append(p_err)

    # ---- Figure for the largest below-true-cost NN bid ------------------------
    if include_below_cost_figure:
        under = find_max_below_cost_point(
            nn_policies, normalized_dfs, result, test_scenarios
        )
        if under is None or under["undercut"] <= 0:
            print("No NN bid below true cost found on the test set.")
        else:
            print(
                f"Max below-cost bid: block {under['block_name']} "
                f"(scenario {under['scenario_id']}, t={under['time_id']}) "
                f"cost={under['true_cost']:.3f}, NN={under['prediction']:.3f}, "
                f"undercut={under['undercut']:.3f}"
            )
            p_under = make_scenario_figure(
                scenario_id=under["scenario_id"],
                time_steps=[under["time_id"]],
                result=result,
                scenarios_df=scenarios_df,
                normalized_dfs=normalized_dfs,
                nn_policies=nn_policies,
                output_path=output_dir
                / f"nn_policy_merit_order_below_cost_s{under['scenario_id']}"
                f"_t{under['time_id']}_{under['block_name']}.png",
                show=show,
            )
            print(f"Saved below-cost figure:   {p_under}")
            saved_paths.append(p_under)

    rng = np.random.default_rng(sample_seed)
    sampled_scenarios = sorted(
        rng.choice(test_scenarios, size=min(n_merit_scenarios, len(test_scenarios)), replace=False)
    )
    for s_id in sampled_scenarios:
        if single_time_step_figures:
            # One figure per time step (one merit-order curve per file).
            for t in merit_time_steps:
                p1 = make_scenario_figure(
                    scenario_id=int(s_id),
                    time_steps=[t],
                    result=result,
                    scenarios_df=scenarios_df,
                    normalized_dfs=normalized_dfs,
                    nn_policies=nn_policies,
                    output_path=output_dir
                    / f"nn_policy_merit_order_scenario_{s_id}_t{t}.png",
                    show=show,
                )
                print(f"Saved merit order figure:  {p1}")
                saved_paths.append(p1)
        else:
            p1 = make_scenario_figure(
                scenario_id=int(s_id),
                time_steps=merit_time_steps,
                result=result,
                scenarios_df=scenarios_df,
                normalized_dfs=normalized_dfs,
                nn_policies=nn_policies,
                output_path=output_dir / f"nn_policy_merit_order_scenario_{s_id}.png",
                show=show,
            )
            print(f"Saved merit order figure:  {p1}")
            saved_paths.append(p1)

    p2 = make_scatter_figure(
        nn_policies=nn_policies,
        normalized_dfs=normalized_dfs,
        result=result,
        test_scenarios=test_scenarios,
        output_path=output_dir / "nn_policy_prediction_scatter.png",
        show=show,
    )
    print(f"Saved scatter figure:      {p2}")
    saved_paths.append(p2)
    return saved_paths


def main() -> None:
    # ---- Paths ----------------------------------------------------------------
    result_json_path = ROOT / "results/merit_order_best_response_results.json"
    scenarios_csv_path = ROOT / "results/full_pipeline/synthetic_scenarios/scenarios.csv"
    normalized_csv_dir = ROOT / "models/neural_network/features/generated/normalized"
    model_dir = ROOT / "models/neural_network/training/trained_models"
    output_dir = ROOT / "models/neural_network/tests/figures"

    generate_merit_order_figures(
        result_json_path=result_json_path,
        scenarios_csv_path=scenarios_csv_path,
        normalized_csv_dir=normalized_csv_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        nn_generator_names=None,  # auto-detect all *.pt files
        n_merit_scenarios=2,
        merit_time_steps=[0, 4, 12],
        sample_seed=1,
        show=False,
    )


if __name__ == "__main__":
    main()
