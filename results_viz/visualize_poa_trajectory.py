"""Visualize the PoA-optimal bidding trajectory as merit order curves.

Reads a PoA result JSON produced by PoAResults.save_results() and writes one
PNG per time step to an output directory:

  poa_trajectory_merit_order_t0.png
  poa_trajectory_merit_order_t1.png
  ...

Each figure shows two merit order stacks:
  True cost  (blue solid)   -- competitive benchmark bids
  Eq. bid    (green dashed) -- bids set by the PoA-optimal policy (alpha*)

Horizontal dotted lines mark the equilibrium price (lambda_eq) and the
socially-optimal dispatch price (lambda_opt).  The demand level is a vertical
dash-dot line.  Each block is annotated with its name and bid value.
Shaded bars mark the dispatched quantities under each strategy.

Standalone usage
----------------
Edit the path inside main() and run:
    .venv/Scripts/python.exe results_viz/visualize_poa_trajectory.py

Pipeline usage
--------------
Call generate_poa_trajectory_figures(result_json_path, output_dir).
Returns a list of saved paths, one per time step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # noqa: E402


def _load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_stack(
    bids: np.ndarray,
    caps: np.ndarray,
    block_names: list[str],
) -> dict[str, Any] | None:
    valid = caps > 1e-9
    order = [int(i) for i in np.lexsort((np.arange(len(bids)), bids)) if bool(valid[int(i)])]
    if not order:
        return None
    obs = bids[order]
    ocs = caps[order]
    cumulative = np.cumsum(ocs)
    return {
        "order": order,
        "bids": obs,
        "caps": ocs,
        "cumulative": cumulative,
        "edges": np.r_[0.0, cumulative],
        "step_values": np.r_[obs, obs[-1]],
        "names": [block_names[i] for i in order],
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
    bid_label: str,
    color: str,
    y_offset: float,
    changed_blocks: set[int] | None = None,
) -> None:
    """Annotate each block in the merit order stack.

    Blocks whose global index appears in changed_blocks are drawn in red to
    signal that their rank shifted between the true-cost and equilibrium orders.
    """
    for local_idx in range(len(stack["order"])):
        cap = float(stack["caps"][local_idx])
        if cap <= 1e-9:
            continue
        global_idx = int(stack["order"][local_idx])
        left = 0.0 if local_idx == 0 else float(stack["cumulative"][local_idx - 1])
        center_x = left + 0.5 * cap
        bid_val = float(stack["bids"][local_idx])
        block_name = stack["names"][local_idx]
        ann_color = "#d62728" if (changed_blocks and global_idx in changed_blocks) else color
        ax.annotate(
            f"{block_name}\n{bid_label}: {bid_val:.1f}",
            xy=(center_x, bid_val),
            xytext=(center_x, bid_val + y_offset),
            ha="center",
            va="bottom" if y_offset >= 0 else "top",
            fontsize=6.5,
            color=ann_color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ann_color, alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=ann_color, lw=0.8, alpha=0.75),
        )


def _changed_order_blocks(
    tc_stack: dict[str, Any],
    eq_stack: dict[str, Any],
) -> set[int]:
    """Return global block indices whose merit-order rank differs between tc and eq stacks."""
    tc_rank = {int(gidx): rank for rank, gidx in enumerate(tc_stack["order"])}
    eq_rank = {int(gidx): rank for rank, gidx in enumerate(eq_stack["order"])}
    return {
        gidx
        for gidx in tc_rank
        if gidx in eq_rank and tc_rank[gidx] != eq_rank[gidx]
    }


def _extract_arrays(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_t = int(result["num_time_steps"])
    n_blocks = len(result["block_names"])
    true_costs = np.zeros(n_blocks)
    alpha = np.zeros((n_blocks, num_t))
    caps = np.zeros((n_blocks, num_t))

    for gen_data in result["generators"].values():
        for block_info in gen_data["blocks"]:
            g = int(block_info["global_block_index"])
            true_costs[g] = float(block_info["true_cost"])
            for t in range(num_t):
                alpha[g, t] = float(block_info["alpha_profile"][t])
                caps[g, t] = float(block_info["capacity_profile"][t])

    return true_costs, alpha, caps


def _plot_time_step(
    t: int,
    result: dict[str, Any],
    true_costs: np.ndarray,
    alpha: np.ndarray,
    caps: np.ndarray,
    output_path: Path,
    show: bool,
    show_dispatch_bars: bool,
) -> Path:
    demand = [float(d) for d in result["demand_profile"]]
    block_names: list[str] = result["block_names"]
    eq_prices = result.get("equilibrium_price_profile", [None] * len(demand))
    opt_prices = result.get("optimal_price_profile", [None] * len(demand))
    nn_generators = sorted(result.get("nn_policy_generators", []))

    obj = result.get("objective", {})
    header_parts = []
    poa_ratio = obj.get("PoA_ratio")
    c_eq = obj.get("C_eq")
    c_opt = obj.get("C_opt")
    if poa_ratio is not None:
        header_parts.append(f"PoA = {poa_ratio:.4f}")
    if c_eq is not None and c_opt is not None:
        header_parts.append(f"C_eq = {c_eq:.1f}  C_opt = {c_opt:.1f}")
    header = "  |  ".join(header_parts)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    fig.suptitle(
        f"PoA-optimal trajectory  —  t = {t}\n"
        f"NN generators: {', '.join(nn_generators) or 'none'}  |  {header}",
        fontsize=11,
    )

    tc_stack = _build_stack(true_costs, caps[:, t], block_names)
    eq_stack = _build_stack(alpha[:, t], caps[:, t], block_names)

    if tc_stack is None or eq_stack is None:
        ax.set_title("no active blocks")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output_path

    y_all = np.concatenate([tc_stack["bids"], eq_stack["bids"]])
    y_span = float(np.ptp(y_all)) or 1.0
    tc_cp = _clearing_price(tc_stack, demand[t])
    eq_cp = _clearing_price(eq_stack, demand[t])

    # Blocks whose rank changed between the two orderings are highlighted red.
    changed = _changed_order_blocks(tc_stack, eq_stack)

    ax.step(
        tc_stack["edges"], tc_stack["step_values"], where="post",
        color="#2636ff", linewidth=2.2, label=f"True cost  (CP={tc_cp:.1f})",
    )
    ax.step(
        eq_stack["edges"], eq_stack["step_values"], where="post",
        color="#2ca02c", linewidth=2.0, linestyle="--",
        label=f"Eq. bid  (CP={eq_cp:.1f})",
    )

    ax.axvline(
        demand[t], color="gray", linewidth=1.6, linestyle="-.",
        label=f"Demand = {demand[t]:.1f} MW",
    )

    if eq_prices[t] is not None:
        ax.axhline(
            float(eq_prices[t]), color="#2ca02c", linewidth=1.1, linestyle=":",
            alpha=0.8, label=f"$\\lambda_{{eq}}$ = {float(eq_prices[t]):.1f}",
        )
    if opt_prices[t] is not None:
        ax.axhline(
            float(opt_prices[t]), color="#2636ff", linewidth=1.1, linestyle=":",
            alpha=0.8, label=f"$\\lambda_{{opt}}$ = {float(opt_prices[t]):.1f}",
        )

    _annotate_stack(ax, tc_stack, "cost", "#2636ff", -0.22 * y_span, changed_blocks=changed)
    _annotate_stack(ax, eq_stack, "bid", "#2ca02c", +0.18 * y_span, changed_blocks=changed)

    total_cap = float(np.sum(caps[:, t]))
    ax.set_xlim(0.0, total_cap * 1.05)
    ax.set_ylim(
        float(np.min(y_all)) - 0.55 * y_span,
        float(np.max(y_all)) + 0.45 * y_span,
    )
    ax.set_xlabel("Cumulative capacity (MW)", fontsize=9)
    ax.set_ylabel("Bid / cost ($/MWh)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def generate_poa_trajectory_figures(
    result_json_path: str | Path,
    output_dir: str | Path,
    show: bool = False,
    show_dispatch_bars: bool = True,
) -> list[Path]:
    """Load a PoA result JSON and write one merit order figure per time step.

    Files are named poa_trajectory_merit_order_t{t}.png inside output_dir.
    Returns the list of saved paths.
    """
    result = _load_result(Path(result_json_path))
    num_t = int(result["num_time_steps"])
    true_costs, alpha, caps = _extract_arrays(result)
    output_dir = Path(output_dir)
    saved: list[Path] = []
    for t in range(num_t):
        out = output_dir / f"poa_trajectory_merit_order_t{t}.png"
        _plot_time_step(
            t=t,
            result=result,
            true_costs=true_costs,
            alpha=alpha,
            caps=caps,
            output_path=out,
            show=show,
            show_dispatch_bars=show_dispatch_bars,
        )
        saved.append(out)
    print(f"[poa_trajectory] Saved {len(saved)} figures to {output_dir}")
    return saved


def main() -> None:
    result_json_path = (
        ROOT / "results/test_tight_50_scenarios/poa/poa_optimization_T8_piecewise_mccormick.json"
    )
    output_dir = (
        ROOT / "results/test_tight_50_scenarios/figures/base_poa/merit_order_curves"
    )

    if not result_json_path.exists():
        candidates = sorted(ROOT.glob("results/**/poa_optimization_T*.json"))
        if not candidates:
            raise FileNotFoundError(
                "No poa_optimization_T*.json found under results/. "
                "Run the PoA optimization pipeline first."
            )
        result_json_path = candidates[-1]
        output_dir = (
            result_json_path.parent.parent / "figures" / "base_poa" / "merit_order_curves"
        )
        print(f"Using result: {result_json_path}")

    generate_poa_trajectory_figures(
        result_json_path=result_json_path,
        output_dir=output_dir,
        show=False,
    )


if __name__ == "__main__":
    main()
