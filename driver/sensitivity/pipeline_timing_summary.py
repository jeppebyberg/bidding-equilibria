"""Full-pipeline compute-time breakdown per block and horizon.

Reads ``pipeline_manifests/block*.json`` from each sensitivity-study run and
plots total wall time broken down by block (data/labels, NN training, PoA
tightening, PoA solve, DRO tightening, DRO solve). Also writes a CSV.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.pipeline_timing_summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from driver.sensitivity.dro_horizon_timing_sweep import (
    HORIZONS_TO_RUN,
    STUDY_NAME,
    run_name,
)
from driver.sensitivity.sensitivity_config import RESULT_ROOT

# ── colour palette (one per sub-block) ────────────────────────────────────────
COLOURS = {
    "Data & labels":    "#4e79a7",
    "NN training":      "#f28e2b",
    "PoA tightening":   "#76b7b2",
    "PoA solve":        "#59a14f",
    "DRO tightening":   "#b07aa1",
    "DRO solve":        "#e15759",
}


def _manifest(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / "pipeline_manifests" / f"{name}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _dro_solver_wall_time(run_dir: Path) -> float:
    """Sum of per-eta wall_time_seconds from the eta_sweep_summary."""
    summary_path = run_dir / "dro" / "eta_sweep_summary.json"
    if not summary_path.exists():
        return 0.0
    with summary_path.open(encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        return 0.0
    return sum(
        float(r.get("solver", {}).get("wall_time_seconds") or 0.0)
        for r in records
    )


def collect_timings(study_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for horizon in HORIZONS_TO_RUN:
        run_dir = study_dir / run_name(horizon)
        if not run_dir.exists():
            continue

        b1 = _manifest(run_dir, "block1_data_labels")
        b2 = _manifest(run_dir, "block2_policy_training")
        b3 = _manifest(run_dir, "block3_poa")
        b4 = _manifest(run_dir, "block4_dro_poa")

        # Block 3: tightening and solve are stored separately when instrumented;
        # fall back to total wall_time if sub-fields are absent.
        b3_tight = float(b3.get("tightening_wall_time_seconds") or 0.0)
        b3_solve = float(b3.get("solve_wall_time_seconds") or 0.0)
        if b3_tight == 0.0 and b3_solve == 0.0:
            # Legacy manifest: put everything in tightening bucket.
            b3_tight = float(b3.get("wall_time_seconds") or 0.0)

        # Block 4: prefer per-eta solver sum (precise); fall back to manifest total.
        dro_solve = _dro_solver_wall_time(run_dir)
        if dro_solve == 0.0:
            dro_solve = float(b4.get("solve_wall_time_seconds") or 0.0)
        b4_tight = float(b4.get("tightening_wall_time_seconds") or 0.0)

        rows.append({
            "horizon": horizon,
            "Data & labels":  float(b1.get("wall_time_seconds") or 0.0),
            "NN training":    float(b2.get("wall_time_seconds") or 0.0),
            "PoA tightening": b3_tight,
            "PoA solve":      b3_solve,
            "DRO tightening": b4_tight,
            "DRO solve":      dro_solve,
        })
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fields = ["horizon"] + list(COLOURS.keys()) + ["total_seconds"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            total = sum(row[k] for k in COLOURS)
            writer.writerow({**row, "total_seconds": round(total, 2)})
    print(f"Saved: {out_path}")


STAGE_LABELS = {
    "Data & labels":    "Scenario &\nlabel gen.",
    "NN training":      "NN\ntraining",
    "PoA tightening":   "R-PoA\ntightening",
    "PoA solve":        "R-PoA\nsolve",
    "DRO tightening":   "DRO\ntightening",
    "DRO solve":        "DRO\nsolve",
}

# Hatch patterns to distinguish horizons when bars overlap.
HATCHES = ["", "///", "xxx", "..."]


def plot_timing(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        print("No timing data found — run the full pipeline first.")
        return

    stages = list(COLOURS.keys())
    n_stages = len(stages)
    n_horizons = len(rows)

    bar_width = 0.25
    stage_gap = 0.9  # x-distance between stage centres
    x_centres = np.arange(n_stages, dtype=float) * stage_gap

    fig, ax = plt.subplots(figsize=(max(9, n_stages * 1.6), 5))

    for h_idx, row in enumerate(rows):
        values = [float(row[s]) for s in stages]
        bottoms = list(np.cumsum([0.0] + values[:-1]))  # running total before each stage

        offset = (h_idx - (n_horizons - 1) / 2.0) * bar_width * 1.1
        x_pos = x_centres + offset
        hatch = HATCHES[h_idx % len(HATCHES)]
        horizon_label = f"T={row['horizon']}"

        for s_idx, stage in enumerate(stages):
            val = values[s_idx]
            bot = bottoms[s_idx]
            colour = COLOURS[stage]
            first_segment = h_idx == 0 and s_idx == 0  # avoid duplicate legend entries

            ax.bar(
                x_pos[s_idx], val, bar_width,
                bottom=bot,
                color=colour,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
                label=stage if h_idx == 0 else "_nolegend_",
            )

            if val > 5:
                ax.text(
                    x_pos[s_idx],
                    bot + val / 2,
                    f"{val:.0f}s",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )

        # Waterfall connector: horizontal line across bar top, then vertical drop
        # to next bar's bottom (shows cumulative time flow).
        tops = [b + v for b, v in zip(bottoms, values)]
        for s_idx in range(n_stages - 1):
            ax.plot(
                [x_pos[s_idx] + bar_width / 2, x_pos[s_idx + 1] - bar_width / 2],
                [tops[s_idx], tops[s_idx]],
                color="black", lw=0.8, ls="--", alpha=0.5,
            )

        # Horizon label above the last bar.
        ax.text(
            x_pos[-1],
            tops[-1] + tops[-1] * 0.02,
            horizon_label,
            ha="center", va="bottom",
            fontsize=8, color="black",
        )

    ax.set_xticks(x_centres)
    ax.set_xticklabels([STAGE_LABELS[s] for s in stages], fontsize=9)
    ax.set_ylabel("Cumulative wall time (seconds)")
    ax.set_title(f"Pipeline compute-time waterfall — {STUDY_NAME}")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOURS[s], label=s) for s in stages
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "pipeline_timing_breakdown.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    study_dir = Path(RESULT_ROOT) / STUDY_NAME
    out_dir = study_dir / "figures"

    rows = collect_timings(study_dir)
    if not rows:
        print(f"No run directories found under {study_dir}")
    else:
        write_csv(rows, study_dir / "pipeline_timing_breakdown.csv")
        plot_timing(rows, out_dir)

        print("\nTiming summary (seconds):")
        header = f"{'Horizon':>8} " + " ".join(f"{k:>16}" for k in COLOURS) + f"  {'Total':>10}"
        print(header)
        for row in rows:
            total = sum(row[k] for k in COLOURS)
            vals = " ".join(f"{row[k]:>16.1f}" for k in COLOURS)
            print(f"  T={row['horizon']:>5} {vals}  {total:>10.1f}")
