# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-14
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Plot the inserted PoA upper bound (C_eq / C_opt) as a function of wind cost.

For a solved base PoA result, the worst-case state stores, per bidding block, the
true cost and both the equilibrium and optimal physical dispatch. The McCormick
PoA box that gets *inserted* into the model has upper bound

    PoA_U = max C_eq / min C_opt

(see models/PoA/PoA_tightening/compute_equilibrium_cost_bounds.py). The true wind
cost enters only the *cost objective*, not the dispatch feasibility, so we can
re-price the already-solved dispatch at a swept shared wind cost ``c`` without
re-solving any MILP:

    C_eq(c)  = C_eq_conv  + c * Q_eq_wind
    C_opt(c) = C_opt_conv + c * Q_opt_wind

where ``C_*_conv`` are the (fixed) conventional cost contributions and
``Q_*_wind`` are the total wind dispatch quantities in the equilibrium / optimal
solution. The plotted ratio C_eq(c) / C_opt(c) is exact at the as-solved wind
cost (it reproduces the stored ``objective.PoA_ratio``) and shows how the bound
responds to wind cost while holding the solved dispatch fixed.

This is most informative on a WIND-PLAYING result: when wind bids strategically,
its equilibrium dispatch Q_eq_wind drops well below the optimal Q_opt_wind, so
the equilibrium leans on expensive conventional generation and C_eq inflates
relative to C_opt. On the conventional-only base case Q_eq_wind == Q_opt_wind, so
the ratio merely decays toward 1 as the shared wind cost rises.

Note: this re-prices the single solved worst-case state. The exact inserted bound
at each wind cost (max C_eq / min C_opt over the whole support set) would require
re-solving the two tightening MILPs per cost; the as-solved derived bound from the
tightening report is drawn as a reference line when available.

Edit the paths in ``main()`` and run:
  .\\.venv\\Scripts\\python.exe -m results_viz.poa_upper_bound_vs_wind_cost
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@dataclass
class WindCostDecomposition:
    """Cost / dispatch split of a solved PoA state into conventional + wind parts."""

    c_eq_conv: float  # conventional true-cost contribution to C_eq (fixed in c)
    c_opt_conv: float  # conventional true-cost contribution to C_opt (fixed in c)
    q_eq_wind: float  # total wind dispatch in the equilibrium solution
    q_opt_wind: float  # total wind dispatch in the optimal solution
    base_wind_cost: float  # mean wind true cost as solved (x reference)
    min_conv_cost: float  # cheapest conventional block true cost (x-range default)
    realized_ratio: float  # stored objective.PoA_ratio (sanity reference)

    def c_eq(self, wind_cost: np.ndarray) -> np.ndarray:
        return self.c_eq_conv + wind_cost * self.q_eq_wind

    def c_opt(self, wind_cost: np.ndarray) -> np.ndarray:
        return self.c_opt_conv + wind_cost * self.q_opt_wind

    def ratio(self, wind_cost: np.ndarray) -> np.ndarray:
        return self.c_eq(wind_cost) / self.c_opt(wind_cost)


def decompose_result(results: dict[str, Any]) -> WindCostDecomposition:
    """Split a solved PoA result into conventional/wind cost and dispatch parts."""
    c_eq_conv = c_opt_conv = 0.0
    q_eq_wind = q_opt_wind = 0.0
    wind_costs: list[float] = []
    conv_costs: list[float] = []

    for generator in results["generators"].values():
        is_wind = bool(generator["is_wind"])
        for block in generator["blocks"]:
            true_cost = float(block["true_cost"])
            eq_dispatch = float(np.nansum(block["equilibrium_dispatch"]))
            opt_dispatch = float(np.nansum(block["optimal_dispatch"]))
            if is_wind:
                wind_costs.append(true_cost)
                q_eq_wind += eq_dispatch
                q_opt_wind += opt_dispatch
            else:
                conv_costs.append(true_cost)
                c_eq_conv += true_cost * eq_dispatch
                c_opt_conv += true_cost * opt_dispatch

    if not wind_costs:
        raise ValueError("Result has no wind blocks; nothing to sweep over wind cost.")

    return WindCostDecomposition(
        c_eq_conv=c_eq_conv,
        c_opt_conv=c_opt_conv,
        q_eq_wind=q_eq_wind,
        q_opt_wind=q_opt_wind,
        base_wind_cost=float(np.mean(wind_costs)),
        min_conv_cost=float(min(conv_costs)) if conv_costs else float(max(wind_costs)),
        realized_ratio=float(results.get("objective", {}).get("PoA_ratio", float("nan"))),
    )


def _inserted_bound_upper(tightening_report_path: Path | None) -> float | None:
    """Read the as-solved inserted McCormick PoA box upper bound, if available."""
    if tightening_report_path is None or not Path(tightening_report_path).exists():
        return None
    report = _load_json(tightening_report_path)
    derived = report.get("derived_poa_bounds")
    if isinstance(derived, (list, tuple)) and len(derived) == 2:
        return float(derived[1])
    return None


def plot_poa_upper_bound_vs_wind_cost(
    result_path: Path,
    output_dir: Path,
    tightening_report_path: Path | None = None,
    wind_cost_max: float | None = None,
    num_points: int = 300,
    show: bool = False,
) -> Path:
    results = _load_json(result_path)
    decomp = decompose_result(results)

    upper = wind_cost_max if wind_cost_max is not None else decomp.min_conv_cost
    wind_cost = np.linspace(0.0, float(upper), int(num_points))
    ratio = decomp.ratio(wind_cost)
    inserted_upper = _inserted_bound_upper(tightening_report_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    # Top: the PoA ratio (C_eq / C_opt) vs shared wind cost.
    ax = axes[0]
    ax.plot(wind_cost, ratio, color="tab:red", linewidth=2.2, label=r"$C_{eq}(c)\,/\,C_{opt}(c)$")
    ax.axvline(
        decomp.base_wind_cost,
        color="gray",
        linestyle=":",
        linewidth=1.4,
        label=f"as-solved wind cost = {decomp.base_wind_cost:.3g}",
    )
    ax.plot(
        [decomp.base_wind_cost],
        [decomp.realized_ratio],
        marker="o",
        color="black",
        markersize=6,
        label=f"realized PoA ratio = {decomp.realized_ratio:.3g}",
    )
    if inserted_upper is not None:
        ax.axhline(
            inserted_upper,
            color="tab:blue",
            linestyle="--",
            linewidth=1.4,
            label=f"inserted PoA box upper = {inserted_upper:.3g}",
        )
    ax.set_ylabel("PoA upper bound  (C_eq / C_opt)")
    ax.set_title(f"Inserted PoA upper bound vs shared wind cost\n{result_path}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Bottom: the cost levels that drive the ratio.
    ax = axes[1]
    ax.plot(wind_cost, decomp.c_eq(wind_cost), color="tab:red", linewidth=2.0, label=r"$C_{eq}(c)$")
    ax.plot(
        wind_cost,
        decomp.c_opt(wind_cost),
        color="tab:green",
        linewidth=2.0,
        label=r"$C_{opt}(c)$",
    )
    ax.axvline(decomp.base_wind_cost, color="gray", linestyle=":", linewidth=1.4)
    ax.set_xlabel("shared wind cost  c")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "poa_upper_bound_vs_wind_cost.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(
        f"[decomp] C_eq_conv={decomp.c_eq_conv:.2f} C_opt_conv={decomp.c_opt_conv:.2f} "
        f"Q_eq_wind={decomp.q_eq_wind:.2f} Q_opt_wind={decomp.q_opt_wind:.2f}"
    )
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    # Edit these paths when running. Point result_path at a WIND-PLAYING run (e.g.
    # results/sensitivity_studies/wind_playing_sweep/wind/poa/...) once it exists to
    # see C_eq diverge from C_opt; the conventional-only base case is the default.
    result_path = Path(
        "results/base_case/poa/poa_optimization_T6_piecewise_mccormick.json"
    )
    tightening_report_path = Path(
        "results/base_case/dro/tightening/poa_worst_case/final_tightening_report.json"
    )
    output_dir = Path("results_viz/figures/poa_upper_bound_vs_wind_cost") / result_path.stem
    # Sweep wind cost from 0 to the cheapest conventional block cost by default
    # (set to a number to override the range).
    wind_cost_max: float | None = None
    show = False

    plot_poa_upper_bound_vs_wind_cost(
        result_path=result_path,
        output_dir=output_dir,
        tightening_report_path=tightening_report_path,
        wind_cost_max=wind_cost_max,
        show=show,
    )


if __name__ == "__main__":
    main()
