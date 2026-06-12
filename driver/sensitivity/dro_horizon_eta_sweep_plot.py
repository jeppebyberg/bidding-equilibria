"""Overlay the DRO eta sweep for T=6 and T=8 on shared axes.

Reads each horizon's DRO results from the ``dro_horizon_timing_sweep`` study and
plots one curve per horizon vs eta, reusing
``results_viz.plot_dro_poa_eta_sweep.plot_eta_sweep_comparison``.

Produces one figure per metric under the study's ``figures`` dir.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_horizon_eta_sweep_plot
"""

from __future__ import annotations

import sys
from pathlib import Path

from driver.sensitivity.dro_horizon_timing_sweep import (
    HORIZONS_TO_RUN,
    run_name,
)
from driver.sensitivity.dro_horizon_timing_sweep import STUDY_NAME as _DEFAULT_STUDY
from driver.sensitivity.sensitivity_config import RESULT_ROOT

# Which study to plot: first CLI arg, else the no-wind study. Lets the same
# command serve both dro_horizon_timing_sweep and ..._wind.
STUDY_NAME = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else _DEFAULT_STUDY
)
from results_viz.plot_dro_poa_eta_sweep import (
    load_eta_sweep_records,
    plot_eta_sweep_comparison,
    plot_frontier_comparison,
)

REGIME_NAME = "poa_worst_case"

# (metric key, figure filename, title). Keys must exist in
# results_viz.plot_dro_poa_eta_sweep.METRIC_LABELS.
METRICS = [
    (
        "average_poa_ratio",
        "eta_sweep_poa_ratio_T6_vs_T8.png",
        "DRO average PoA ratio vs eta  -  T=6 vs T=8",
    ),
    (
        "worst_case_expected_poa",
        "eta_sweep_worst_case_poa_T6_vs_T8.png",
        "DRO worst-case expected PoA vs eta  -  T=6 vs T=8",
    ),
]


def _dro_dir(horizon: int) -> Path:
    return Path(RESULT_ROOT) / STUDY_NAME / run_name(horizon) / "dro"


def main() -> None:
    records_by_label: dict[str, list] = {}
    for horizon in HORIZONS_TO_RUN:
        dro_dir = _dro_dir(horizon)
        try:
            records = load_eta_sweep_records(dro_dir, REGIME_NAME, include_archives=False)
        except FileNotFoundError as exc:
            print(f"[T{horizon}] skipped: {exc}")
            continue
        records_by_label[f"T={horizon}"] = records
        print(f"[T{horizon}] loaded {len(records)} eta records from {dro_dir}")

    if len(records_by_label) < 2:
        print("Need both horizons' results to overlay. Run the sweep first.")
        return

    output_dir = Path(RESULT_ROOT) / STUDY_NAME / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, title in METRICS:
        try:
            path = plot_eta_sweep_comparison(
                records_by_label=records_by_label,
                output_path=output_dir / filename,
                metric=metric,
                title=title,
            )
            print(f"Saved {metric}: {path}")
        except ValueError as exc:
            print(f"Skipped {metric}: {exc}")

    # PoA-epsilon frontier: x = achieved Wasserstein distance, y = worst-case
    # expected PoA. One curve per horizon.
    try:
        frontier_path = plot_frontier_comparison(
            records_by_label=records_by_label,
            output_path=output_dir / "poa_epsilon_frontier_T6_vs_T8.png",
            poa_metric="worst_case_expected_poa",
            title="PoA-epsilon frontier  -  T=6 vs T=8",
        )
        print(f"Saved frontier: {frontier_path}")
    except ValueError as exc:
        print(f"Skipped frontier: {exc}")


if __name__ == "__main__":
    main()
