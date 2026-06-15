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

from driver.sensitivity.horizon_sweep import study as HORIZON_STUDY
from driver.sensitivity.sensitivity_config import RESULT_ROOT

STUDY_NAME = HORIZON_STUDY.name


def _study_runs() -> list[tuple[str, int]]:
    """(run_dir_name, horizon) pairs for each run in the horizon sweep."""
    pairs: list[tuple[str, int]] = []
    for run in HORIZON_STUDY.runs:
        horizon = run.overrides.get("horizon")
        if horizon is None:
            continue
        pairs.append((run.name, int(horizon)))
    return pairs

# ── colour palette (one per sub-block) ────────────────────────────────────────
COLOURS = {
    "Data & labels": "#4e79a7",
    "NN training": "#f28e2b",
    "PoA tightening": "#76b7b2",
    "PoA solve": "#59a14f",
    "DRO tightening": "#b07aa1",
    "DRO solve": "#e15759",
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
    return sum(float(r.get("solver", {}).get("wall_time_seconds") or 0.0) for r in records)


def collect_timings(study_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir_name, horizon in _study_runs():
        run_dir = study_dir / run_dir_name
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

        rows.append(
            {
                "horizon": horizon,
                "Data & labels": float(b1.get("wall_time_seconds") or 0.0),
                "NN training": float(b2.get("wall_time_seconds") or 0.0),
                "PoA tightening": b3_tight,
                "PoA solve": b3_solve,
                "DRO tightening": b4_tight,
                "DRO solve": dro_solve,
            }
        )
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


def plot_timing(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """One stacked bar per horizon: bar height = total wall time, coloured
    segments = per-stage breakdown. Shows both how compute scales with the
    horizon and where the time is spent."""
    if not rows:
        print("No timing data found — run the full pipeline first.")
        return

    stages = list(COLOURS.keys())
    rows = sorted(rows, key=lambda r: r["horizon"])
    x_pos = np.arange(len(rows), dtype=float)
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.6), 6))

    bottoms = np.zeros(len(rows))
    for stage in stages:
        values = np.array([float(row[stage]) for row in rows])
        ax.bar(
            x_pos,
            values,
            bar_width,
            bottom=bottoms,
            color=COLOURS[stage],
            edgecolor="white",
            linewidth=0.6,
            label=stage,
        )
        # Label a segment only when it is tall enough to hold the text.
        for x, val, bot, total in zip(
            x_pos, values, bottoms, [sum(r[s] for s in stages) for r in rows]
        ):
            if val > 0.04 * total:
                ax.text(
                    x,
                    bot + val / 2,
                    f"{val:.0f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottoms += values

    # Total wall time above each bar.
    for x, total in zip(x_pos, bottoms):
        ax.text(
            x,
            total + 0.01 * bottoms.max(),
            f"{total:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"T={row['horizon']}" for row in rows], fontsize=10)
    ax.set_xlabel("Horizon (time steps)")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_ylim(0, bottoms.max() * 1.12)
    ax.set_title(f"Pipeline compute-time breakdown — {STUDY_NAME}")
    # Legend in stacking order, top segment first.
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[s], label=s) for s in stages]
    ax.legend(handles=handles[::-1], loc="upper left", fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "pipeline_timing_breakdown.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main() -> None:
    study_dir = Path(RESULT_ROOT) / STUDY_NAME
    out_dir = study_dir / "figures"

    rows = collect_timings(study_dir)
    if not rows:
        print(f"No run directories found under {study_dir}")
        return

    write_csv(rows, study_dir / "pipeline_timing_breakdown.csv")
    plot_timing(rows, out_dir)

    print("\nTiming summary (seconds):")
    header = f"{'Horizon':>8} " + " ".join(f"{k:>16}" for k in COLOURS) + f"  {'Total':>10}"
    print(header)
    for row in rows:
        total = sum(row[k] for k in COLOURS)
        vals = " ".join(f"{row[k]:>16.1f}" for k in COLOURS)
        print(f"  T={row['horizon']:>5} {vals}  {total:>10.1f}")


if __name__ == "__main__":
    main()
