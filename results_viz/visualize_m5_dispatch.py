# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-17
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Visualize the m5 inflation-margin run: equilibrium vs optimal dispatch.

Shows the ramp-trap / curtailment mechanism behind PoA > 1 at
inflation_margin = 5.0:
  - Equilibrium floods wind at the t5 peak, knocking the cheap ramp-limited
    unit G1 to zero, and the ramp limit then forces expensive G2 on at t4/t6.
  - The optimum curtails a little wind at t5 to keep G1 "warm" so it can ramp
    to meet t4/t6 residual demand on its own; G2 never runs.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.visualize_m5_dispatch
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
import numpy as np

RESULT = Path(
    "results/sensitivity_studies/inflation_margin_sweep/m5/poa/"
    "poa_optimization_T6_piecewise_mccormick.json"
)
OUT = Path("results/sensitivity_studies/inflation_margin_sweep/m5/m5_dispatch_story.png")

# Stacking order (bottom -> top), cheapest true cost first.
STACK = ["W1", "W2", "W3", "G1", "G2"]
COLORS = {
    "W1": "#bfe3b0",
    "W2": "#8ecd77",
    "W3": "#5ba745",
    "G1": "#f2a93b",
    "G2": "#d4453b",
}


def load():
    data = json.loads(RESULT.read_text())
    gens = data["generators"]
    T = data["num_time_steps"]
    eq = {g: np.array(gens[g]["equilibrium_physical_dispatch"]) for g in gens}
    opt = {g: np.array(gens[g]["optimal_physical_dispatch"]) for g in gens}
    cap = {g: np.array(gens[g]["physical_capacity_profile"]) for g in gens}
    return data, T, eq, opt, cap


def stacked(ax, T, disp, cap, demand, title, mark_curtail):
    x = np.arange(T)
    bottom = np.zeros(T)
    for g in STACK:
        ax.bar(x, disp[g], bottom=bottom, color=COLORS[g], label=g,
               edgecolor="white", linewidth=0.5, width=0.7)
        bottom += disp[g]

    # Wind curtailment (available - dispatched) hatched above the stack.
    if mark_curtail:
        wind_avail = sum(cap[g] for g in ["W1", "W2", "W3"])
        wind_disp = sum(disp[g] for g in ["W1", "W2", "W3"])
        curt = wind_avail - wind_disp
        ax.bar(x, curt, bottom=bottom, facecolor="none", hatch="////",
               edgecolor="#c0392b", linewidth=0.0, width=0.7,
               label="wind curtailed")
        for t in range(T):
            if curt[t] > 1e-3:
                ax.annotate(f"curtail\n{curt[t]:.1f}", (t, bottom[t] + curt[t]),
                            ha="center", va="bottom", fontsize=8, color="#c0392b")

    ax.plot(x, demand, "k--o", lw=1.5, ms=5, label="demand", zorder=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{t+1}" for t in range(T)])
    ax.set_ylabel("MW")
    ax.set_ylim(0, max(demand) * 1.18)


def main():
    data, T, eq, opt, cap = load()
    demand = data["demand_profile"]
    x = np.arange(T)
    R = 20.0  # G1 ramp limit (binding throughout the dispatch)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (axA, axB), (axC, axD) = axes

    stacked(axA, T, eq, cap, demand, "Equilibrium dispatch  (C_eq = %.0f)" %
            data["objective"]["C_eq"], mark_curtail=True)
    stacked(axB, T, opt, cap, demand, "Optimal dispatch  (C_opt = %.0f)" %
            data["objective"]["C_opt"], mark_curtail=True)
    axB.legend(loc="upper left", fontsize=8, ncol=2)

    # Panel C: G1 trajectory with ramp envelope.
    g1_eq, g1_opt = eq["G1"], opt["G1"]
    axC.plot(x, g1_eq, "o-", color="#b9770e", lw=2, label="G1 equilibrium")
    axC.plot(x, g1_opt, "s--", color="#1f77b4", lw=2, label="G1 optimal")
    # Ramp envelope around the optimal path.
    for t in range(T - 1):
        axC.fill_between([t, t + 1], [g1_opt[t] - R, g1_opt[t + 1] - R],
                         [g1_opt[t] + R, g1_opt[t + 1] + R],
                         color="#1f77b4", alpha=0.06)
    axC.annotate("wind floods t5 ->\nG1 forced to 0", (4, 0), xytext=(2.3, 9),
                 fontsize=8, color="#b9770e",
                 arrowprops=dict(arrowstyle="->", color="#b9770e"))
    axC.annotate("curtail keeps\nG1 warm (5.1)", (4, g1_opt[4]), xytext=(4.1, 13),
                 fontsize=8, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    axC.set_title("G1 (cheap, ramp-limited +/-%g)" % R, fontsize=11, fontweight="bold")
    axC.set_xticks(x); axC.set_xticklabels([f"t{t+1}" for t in range(T)])
    axC.set_ylabel("MW"); axC.legend(fontsize=8)

    # Panel D: where the cost gap lives -- expensive G2 use + price.
    width = 0.35
    axD.bar(x - width / 2, eq["G2"], width, color="#d4453b", label="G2 equilibrium")
    axD.bar(x + width / 2, opt["G2"], width, color="#f4b6b2", label="G2 optimal (=0)")
    axD.set_title("Inefficiency: expensive G2 forced on in equilibrium",
                  fontsize=11, fontweight="bold")
    axD.set_xticks(x); axD.set_xticklabels([f"t{t+1}" for t in range(T)])
    axD.set_ylabel("G2 MW (true cost 30)")
    axD.legend(loc="upper left", fontsize=8)
    ax2 = axD.twinx()
    ax2.plot(x, data["equilibrium_price_profile"], "k:^", ms=4,
             label="eq price")
    ax2.plot(x, data["optimal_price_profile"], "0.5", ls=":", marker="v", ms=4,
             label="opt price")
    ax2.set_ylabel("clearing price")
    ax2.legend(loc="upper right", fontsize=8)

    ratio = data["objective"]["ex_post_ratio"]
    fig.suptitle(
        f"inflation_margin = 5.0 (m5):  PoA = C_eq/C_opt = {ratio:.3f}   "
        f"(gap = {data['objective']['PoA_difference']:.0f})",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
