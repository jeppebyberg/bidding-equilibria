"""Side-by-side equilibrium vs optimal dispatch for the m5 inflation-margin run.

Clean two-panel figure for the thesis: stacked generation against demand,
with the optimum's t5 wind curtailment marked. Shows that the equilibrium
floods all wind (forcing expensive G2 at t4/t6), while the optimum spills a
little wind at the t5 peak to keep the cheap ramp-limited G1 warm.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.visualize_m5_sidebyside
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$t_{t+1}$" for t in range(T)])
    ax.set_xlabel("period")
    ax.margins(x=0.02)


def main():
    data = json.loads(RESULT.read_text())
    gens = data["generators"]
    T = data["num_time_steps"]
    eq = {g: np.array(gens[g]["equilibrium_physical_dispatch"]) for g in gens}
    opt = {g: np.array(gens[g]["optimal_physical_dispatch"]) for g in gens}
    cap = {g: np.array(gens[g]["physical_capacity_profile"]) for g in gens}
    demand = data["demand_profile"]
    obj = data["objective"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panel(axL, T, eq, cap, demand,
          r"Equilibrium  ($C_{\mathrm{eq}} = %.0f$)" % obj["C_eq"])
    panel(axR, T, opt, cap, demand,
          r"Social optimum  ($C_{\mathrm{opt}} = %.0f$)" % obj["C_opt"])
    axL.set_ylabel("generation [MW]")
    axL.set_ylim(0, max(demand) * 1.16)

    handles, labels = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        r"$m_5$: $\mathrm{PoA} = C_{\mathrm{eq}}/C_{\mathrm{opt}} = %.3f$"
        % obj["ex_post_ratio"], fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
