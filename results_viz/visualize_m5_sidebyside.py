# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-17
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Side-by-side equilibrium vs optimal dispatch for the m5 inflation-margin run.

Clean two-panel figure for the thesis: stacked generation against demand,
with the optimum's t5 wind curtailment marked. Shows that the equilibrium
floods all wind (forcing expensive G2 at t4/t6), while the optimum spills a
little wind at the t5 peak to keep the cheap ramp-limited G1 warm.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.visualize_m5_sidebyside
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
OUT = Path("results/sensitivity_studies/inflation_margin_sweep/m5/m5_dispatch_sidebyside.png")

STACK = ["W1", "W2", "W3", "G1", "G2"]
LABELS = {
    "W1": "$W_1$", "W2": "$W_2$", "W3": "$W_3$",
    "G1": "$G_1$ (cost 10)", "G2": "$G_2$ (cost 30)",
}
COLORS = {
    "W1": "#bfe3b0", "W2": "#8ecd77", "W3": "#5ba745",
    "G1": "#f2a93b", "G2": "#d4453b",
}


def panel(ax, T, disp, cap, demand, title):
    x = np.arange(T)
    bottom = np.zeros(T)
    for g in STACK:
        ax.bar(x, disp[g], bottom=bottom, color=COLORS[g], label=LABELS[g],
               edgecolor="white", linewidth=0.5, width=0.72)
        bottom += disp[g]

    wind_avail = sum(cap[g] for g in ["W1", "W2", "W3"])
    wind_disp = sum(disp[g] for g in ["W1", "W2", "W3"])
    curt = wind_avail - wind_disp
    first = True
    for t in range(T):
        if curt[t] > 1e-3:
            ax.bar(x[t], curt[t], bottom=bottom[t], facecolor="none", hatch="////",
                   edgecolor="#c0392b", width=0.72,
                   label="wind curtailed" if first else None)
            ax.annotate(f"spill\n{curt[t]:.1f}", (t, bottom[t] + curt[t]),
                        ha="center", va="bottom", fontsize=8.5, color="#c0392b")
            first = False

    ax.plot(x, demand, "k--o", lw=1.6, ms=5, label="demand $D_t$", zorder=5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$t_{t+1}$" for t in range(T)], fontsize=11)
    ax.set_xlabel("period", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.margins(x=0.02)


def main():
    data = json.loads(RESULT.read_text())
    gens = data["generators"]
    T = data["num_time_steps"]
    eq = {g: np.array(gens[g]["equilibrium_physical_dispatch"]) for g in gens}
    opt = {g: np.array(gens[g]["optimal_physical_dispatch"]) for g in gens}
    cap = {g: np.array(gens[g]["physical_capacity_profile"]) for g in gens}
    demand = data["demand_profile"]

    # Sized for the A4 text width so the text stays readable when included ~1:1.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 3.8), sharey=True)
    panel(axL, T, eq, cap, demand, "Policy bids")
    panel(axR, T, opt, cap, demand, "True costs")
    axL.set_ylabel("generation [MW]", fontsize=13)
    axL.set_ylim(0, max(demand) * 1.16)

    handles, labels = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
