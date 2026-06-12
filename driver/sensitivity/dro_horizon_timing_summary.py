"""Companion summary for ``dro_horizon_timing_sweep``.

Reads each horizon run's ``dro/eta_sweep_summary.json`` and reports the DRO
compute time per eta and aggregated per horizon, so T=6 and T=8 can be compared
directly. Writes a per-eta CSV and a per-horizon CSV next to the study results.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_timing_summary
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from driver.sensitivity.dro_horizon_timing_sweep import (
    HORIZONS_TO_RUN,
    run_name,
)
from driver.sensitivity.dro_horizon_timing_sweep import STUDY_NAME as _DEFAULT_STUDY
from driver.sensitivity.sensitivity_config import RESULT_ROOT

# Which study to summarize: first CLI arg, else the no-wind study. Lets the same
# command serve both dro_horizon_timing_sweep and ..._wind.
STUDY_NAME = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else _DEFAULT_STUDY
)


def _study_dir() -> Path:
    return Path(RESULT_ROOT) / STUDY_NAME


def _summary_path(horizon: int) -> Path:
    return _study_dir() / run_name(horizon) / "dro" / "eta_sweep_summary.json"


def _load_summary(horizon: int) -> list[dict[str, Any]] | None:
    path = _summary_path(horizon)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (per_eta_rows, per_horizon_rows)."""
    per_eta: list[dict[str, Any]] = []
    per_horizon: list[dict[str, Any]] = []

    for horizon in HORIZONS_TO_RUN:
        records = _load_summary(horizon)
        if records is None:
            print(f"[T{horizon}] no eta_sweep_summary.json yet -- run not complete.")
            continue

        wall_times: list[float] = []
        non_optimal = 0
        free_binaries: int | None = None
        for rec in records:
            solver = rec.get("solver") or {}
            counts = solver.get("variable_counts") or {}
            wall = _as_float(solver.get("wall_time_seconds"))
            term = solver.get("termination_condition")
            if term is not None and str(term) != "optimal":
                non_optimal += 1
            if wall is not None:
                wall_times.append(wall)
            if free_binaries is None:
                free_binaries = counts.get("num_binary_variables_free")
            per_eta.append(
                {
                    "horizon": horizon,
                    "eta": _as_float(rec.get("eta")),
                    "wall_time_seconds": wall,
                    "termination_condition": term,
                    "mip_gap": _as_float(solver.get("mip_gap")),
                    "num_binary_free": counts.get("num_binary_variables_free"),
                    "num_binary_total": counts.get("num_binary_variables"),
                    "num_continuous": counts.get("num_continuous_variables"),
                }
            )

        total = sum(wall_times) if wall_times else None
        per_horizon.append(
            {
                "horizon": horizon,
                "num_etas": len(records),
                "total_dro_wall_seconds": total,
                "mean_eta_wall_seconds": (total / len(wall_times)) if wall_times else None,
                "max_eta_wall_seconds": max(wall_times) if wall_times else None,
                "num_non_optimal_etas": non_optimal,
                "num_binary_free": free_binaries,
            }
        )

    return per_eta, per_horizon


def _fmt(value: Any, width: int, prec: int = 2) -> str:
    if value is None:
        return f"{'-':>{width}}"
    if isinstance(value, float):
        return f"{value:>{width}.{prec}f}"
    return f"{value:>{width}}"


def print_tables(per_eta: list[dict[str, Any]], per_horizon: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("  DRO compute time vs horizon (no solver time limit)")
    print("=" * 72)

    print("\nPer-horizon aggregate")
    header = (
        f"  {'T':>3} {'#eta':>5} {'total_s':>10} {'mean_s':>9} "
        f"{'max_s':>9} {'non_opt':>8} {'free_bins':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in per_horizon:
        print(
            f"  {_fmt(row['horizon'], 3)} {_fmt(row['num_etas'], 5)} "
            f"{_fmt(row['total_dro_wall_seconds'], 10)} "
            f"{_fmt(row['mean_eta_wall_seconds'], 9)} "
            f"{_fmt(row['max_eta_wall_seconds'], 9)} "
            f"{_fmt(row['num_non_optimal_etas'], 8)} "
            f"{_fmt(row['num_binary_free'], 10)}"
        )

    if len(per_horizon) == 2:
        a, b = per_horizon
        ta, tb = a["total_dro_wall_seconds"], b["total_dro_wall_seconds"]
        if ta and tb:
            print(
                f"\n  Total DRO wall: T{a['horizon']}={ta:.1f}s vs "
                f"T{b['horizon']}={tb:.1f}s  "
                f"(x{tb / ta:.2f} from T{a['horizon']} to T{b['horizon']})"
            )

    print("\nPer-eta detail")
    print(f"  {'T':>3} {'eta':>12} {'wall_s':>10} {'term':>10} {'gap':>8}")
    print("  " + "-" * 48)
    for row in per_eta:
        print(
            f"  {_fmt(row['horizon'], 3)} {_fmt(row['eta'], 12, 6)} "
            f"{_fmt(row['wall_time_seconds'], 10)} "
            f"{str(row['termination_condition']):>10} "
            f"{_fmt(row['mip_gap'], 8, 4)}"
        )


def write_csv(per_eta: list[dict[str, Any]], per_horizon: list[dict[str, Any]]) -> None:
    study_dir = _study_dir()
    study_dir.mkdir(parents=True, exist_ok=True)

    eta_path = study_dir / "dro_timing_per_eta.csv"
    with eta_path.open("w", newline="", encoding="utf-8") as fh:
        if per_eta:
            writer = csv.DictWriter(fh, fieldnames=list(per_eta[0].keys()))
            writer.writeheader()
            writer.writerows(per_eta)
    print(f"\nWrote {eta_path}")

    horizon_path = study_dir / "dro_timing_per_horizon.csv"
    with horizon_path.open("w", newline="", encoding="utf-8") as fh:
        if per_horizon:
            writer = csv.DictWriter(fh, fieldnames=list(per_horizon[0].keys()))
            writer.writeheader()
            writer.writerows(per_horizon)
    print(f"Wrote {horizon_path}")


def main() -> None:
    per_eta, per_horizon = collect()
    if not per_horizon:
        print("No completed horizon runs found. Run dro_horizon_timing_sweep first.")
        return
    print_tables(per_eta, per_horizon)
    write_csv(per_eta, per_horizon)


if __name__ == "__main__":
    main()
