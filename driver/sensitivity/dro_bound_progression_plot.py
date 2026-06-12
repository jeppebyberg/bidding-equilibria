"""Plot Gurobi incumbent vs best bound over solve time for one DRO solve.

Reads the ``*_progress.json`` sidecar written by
``save_dro_solve_log_artifacts`` (parsed from the Gurobi node log) for a single
horizon/eta and plots the incumbent (a lower bound on the maximization optimum)
and the best bound (ObjBound, an upper bound) as functions of wall-clock time,
shading the optimality gap between them.

Edit HORIZON / ETA_LABEL below to target a different solve, then run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_bound_progression_plot
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from driver.sensitivity.dro_horizon_timing_sweep import STUDY_NAME, run_name
from driver.sensitivity.sensitivity_config import RESULT_ROOT

REGIME_NAME = "poa_worst_case"

# Target solve. ETA_LABEL is the eta encoded as in the result filename
# (eta_label(): 8 significant figures, '.' -> 'p', '-' -> 'm').
HORIZON = 8
ETA_LABEL = "0p0064938163"  # eta = 0.0064938163 (shown as 0.006494 in the CSV)


def _progress_path(horizon: int, eta_label: str) -> Path:
    log_dir = (
        Path(RESULT_ROOT)
        / STUDY_NAME
        / run_name(horizon)
        / "dro"
        / REGIME_NAME
        / "solve_logs"
    )
    stem = f"dro_poa_eta_{eta_label}_T{horizon}_piecewise_mccormick"
    return log_dir / f"{stem}_progress.json"


def main() -> None:
    path = _progress_path(HORIZON, ETA_LABEL)
    if not path.exists():
        raise FileNotFoundError(f"Progress sidecar not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    progression = payload.get("bound_progression") or []
    if not progression:
        raise ValueError(f"No bound_progression points in {path}")

    eta = float(payload.get("eta"))
    times = [float(p["time_s"]) for p in progression]
    incumbents = [
        float(p["incumbent"]) if p.get("incumbent") is not None else None
        for p in progression
    ]
    bounds = [
        float(p["best_bound"]) if p.get("best_bound") is not None else None
        for p in progression
    ]

    # Drop leading points before the first incumbent so the fill is well-defined.
    inc_t = [t for t, v in zip(times, incumbents) if v is not None]
    inc_v = [v for v in incumbents if v is not None]
    bnd_t = [t for t, v in zip(times, bounds) if v is not None]
    bnd_v = [v for v in bounds if v is not None]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ax.plot(
        bnd_t, bnd_v, color="tab:red", linewidth=2.0, marker="",
        label="Best bound (ObjBound, upper)",
    )
    ax.step(
        inc_t, inc_v, where="post", color="tab:blue", linewidth=2.0,
        label="Incumbent (lower)",
    )
    # Shade the optimality gap where both series are defined.
    if inc_t and bnd_t:
        common_t = sorted(set(inc_t) & set(bnd_t))
        if common_t:
            inc_map = dict(zip(inc_t, inc_v))
            bnd_map = dict(zip(bnd_t, bnd_v))
            ax.fill_between(
                common_t,
                [inc_map[t] for t in common_t],
                [bnd_map[t] for t in common_t],
                color="0.7", alpha=0.30, zorder=0, label="Optimality gap",
            )

    final = progression[-1]
    wall = payload.get("wall_time_seconds")
    gap_pct = final.get("gap_percent")
    final_inc = final.get("incumbent")
    if final_inc is not None:
        ax.axhline(
            float(final_inc), color="tab:blue", linestyle=":", linewidth=1.2,
            alpha=0.7,
        )

    ax.set_xlabel("Solve time (s)", fontsize=10)
    ax.set_ylabel("DRO inner objective", fontsize=10)
    ax.set_title(
        f"DRO bound progression  -  T={HORIZON}, eta={eta:.6g}  ({REGIME_NAME})",
        fontsize=12,
    )
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)

    parts = []
    if wall is not None:
        parts.append(f"wall = {float(wall):.0f} s")
    if final_inc is not None:
        parts.append(f"opt = {float(final_inc):.4f}")
    if gap_pct is not None:
        parts.append(f"final gap = {float(gap_pct):.2g}%")
    parts.append(f"{len(progression)} log points")
    ax.text(
        0.99, 0.02, "   ".join(parts), transform=ax.transAxes,
        fontsize=8.5, color="0.35", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.85),
    )

    fig.tight_layout()
    out_dir = Path(RESULT_ROOT) / STUDY_NAME / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bound_progression_T{HORIZON}_eta_{ETA_LABEL}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
