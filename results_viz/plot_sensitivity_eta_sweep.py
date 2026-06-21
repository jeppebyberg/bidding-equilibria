# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-03
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Plot the eta-sweep results from the sensitivity pipeline.

Reads from results/sensitivity_studies/{sensitivity_case}/dro/ and writes
figures to results_viz/figures/sensitivity_eta_sweep/{sensitivity_case}/.

All plotting logic lives in plot_dro_poa_eta_sweep.py — this script only
sets the paths and calls the shared functions.
"""
from __future__ import annotations

import argparse
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_viz.plot_dro_poa_eta_sweep import (
    _as_optional_float,
    clean_output_dir,
    discover_regime_names,
    load_eta_sweep_records,
    write_summary_csv,
)

DEFAULT_SENSITIVITY_CASE = "base_test_case"
SENSITIVITY_RESULTS_ROOT = Path("results/sensitivity_studies")
OUTPUT_ROOT = Path("results_viz/figures/sensitivity_eta_sweep")

# Fallback epsilon labels used only when records carry no case_label.
_FALLBACK_EPSILON_LABELS: dict[float, str] = {
    0.0: "SAA (no robustness)",
}


def _build_epsilon_labels(records: list[dict]) -> dict[float, str]:
    """Return epsilon-to-label mapping for the plot legend.

    If every record shares the same non-empty case_label, use that as the
    curve label (with an epsilon suffix when there are multiple epsilon values).
    Otherwise fall back to numeric epsilon labels.
    """
    case_labels = {r.get("case_label", "") for r in records}
    case_label = next(iter(case_labels)) if len(case_labels) == 1 else ""

    epsilons = sorted({float(r["epsilon"]) for r in records if r.get("epsilon") is not None})
    if case_label:
        if len(epsilons) == 1:
            return {epsilons[0]: case_label}
        return {eps: f"{case_label} (ε = {eps:.4g})" for eps in epsilons}

    labels = dict(_FALLBACK_EPSILON_LABELS)
    for eps in epsilons:
        if eps not in labels:
            labels[eps] = f"ε = {eps:.4g}"
    return labels


def _discover_cases(study_path: Path) -> list[tuple[str, Path]]:
    """Return (case_name, dro_dir) pairs for all cases under a study directory.

    Supports two layouts:
      study/{case_name}/dro/   ← sensitivity sweep (multiple cases)
      study/dro/               ← single-case run
    """
    direct_dro = study_path / "dro"
    if discover_regime_names(direct_dro):
        return [(study_path.name, direct_dro)]

    cases = []
    if study_path.exists():
        for subdir in sorted(study_path.iterdir()):
            if not subdir.is_dir():
                continue
            dro_dir = subdir / "dro"
            if discover_regime_names(dro_dir):
                cases.append((subdir.name, dro_dir))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sensitivity eta-sweep results. Pass a study name (e.g. test_sweep) "
            "to plot all cases within it, or study/case_name to plot a single case."
        )
    )
    parser.add_argument(
        "sensitivity_case",
        nargs="?",
        default=DEFAULT_SENSITIVITY_CASE,
        help=(
            f"Study or case folder under {SENSITIVITY_RESULTS_ROOT} "
            f"(default: {DEFAULT_SENSITIVITY_CASE})."
        ),
    )
    parser.add_argument(
        "--dro-dir",
        default=None,
        help="Direct path to a study root or DRO results directory. Skips auto-discovery.",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Path to a base-case DRO results directory "
            "(e.g. results/sensitivity_pipeline/dro). "
            "When given, its curves are prepended to every comparison plot."
        ),
    )
    return parser.parse_args()


def _case_label(records: list[dict[str, Any]], fallback: str) -> str:
    if not records:
        return fallback
    label = records[0].get("case_label", "")
    if label:
        return label
    ref = records[0].get("reference_case", "")
    return ref if ref else fallback


def _pick_records_for_comparison(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For a comparison plot, use only the records with the largest epsilon value.

    When a case was run with multiple Wasserstein caps, the highest cap is the
    least constraining and best represents the full frontier for that case.
    """
    epsilons = [float(r["epsilon"]) for r in records if r.get("epsilon") is not None]
    if not epsilons:
        return records
    max_eps = max(epsilons)
    return [r for r in records if _as_optional_float(r.get("epsilon")) == max_eps]


def plot_comparison_eta_sweep(
    cases: dict[str, list[dict[str, Any]]],
    regime_name: str,
    output_dir: Path,
    metric: str = "average_poa_ratio",
) -> Path:
    """One curve per case: PoA metric vs eta on shared axes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for idx, (case_name, records) in enumerate(cases.items()):
        color = colors[idx % len(colors)]
        selected = _pick_records_for_comparison(records)
        etas = np.array([float(r["eta"]) for r in selected if r.get("eta") is not None])
        values = np.array([
            np.nan if r.get(metric) is None else float(r[metric])
            for r in selected
        ])
        label = _case_label(records, case_name)
        ax.plot(etas, values, "o-", color=color, linewidth=2.0, markersize=5.5, label=label)

    all_etas = [
        float(r["eta"])
        for recs in cases.values()
        for r in recs
        if r.get("eta") is not None and float(r["eta"]) > 0
    ]
    if all_etas and max(all_etas) / min(all_etas) > 20:
        ax.set_xscale("log")

    if metric == "average_poa_ratio":
        ax.axhline(1.0, color="0.4", linewidth=1.2, linestyle="--", alpha=0.6,
                   zorder=1, label="PoA = 1 (competitive)")
        ax.set_ylabel("Average PoA ratio")
    else:
        ax.set_ylabel(metric)

    ax.set_xlabel("η  (Wasserstein penalty)", fontsize=10)
    ax.set_title(f"PoA η sweep — {regime_name}  (sensitivity comparison)", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{regime_name}_comparison_poa_by_eta.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_comparison_epsilon_frontier(
    cases: dict[str, list[dict[str, Any]]],
    regime_name: str,
    output_dir: Path,
    poa_metric: str = "inner_objective",
) -> Path:
    """One curve per case on the PoA–ε frontier."""
    import math

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for idx, (case_name, records) in enumerate(cases.items()):
        color = colors[idx % len(colors)]
        selected = _pick_records_for_comparison(records)
        points = []
        for r in selected:
            x = _as_optional_float(r.get("average_wasserstein_distance"))
            y = _as_optional_float(r.get(poa_metric))
            eta = _as_optional_float(r.get("eta"))
            if x is not None and y is not None and eta is not None:
                points.append((x, y, eta))
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs, ys, etas = zip(*points)
        label = _case_label(records, case_name)
        ax.plot(xs, ys, "o-", color=color, linewidth=2.0, markersize=5.5,
                zorder=3, label=label)

        n = len(points)
        step = max(1, math.ceil(n / 5))
        for i in sorted(set(range(0, n, step)) | {n - 1}):
            ax.annotate(
                f"η={etas[i]:.2g}",
                xy=(xs[i], ys[i]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.85,
            )

    ax.set_xlabel("Achieved Wasserstein distance (ε)", fontsize=10)
    ax.set_ylabel("DRO inner objective" if poa_metric == "inner_objective" else poa_metric,
                  fontsize=10)
    ax.set_title(
        f"PoA–ε frontier — {regime_name}  (sensitivity comparison)\n"
        "η labels show marginal cost of robustness",
        fontsize=11,
    )
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{regime_name}_comparison_poa_epsilon_frontier.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_study(
    cases: list[tuple[str, Path]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load all records, keyed by regime_name then case_name."""
    by_regime: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for case_name, dro_dir in cases:
        for regime_name in discover_regime_names(dro_dir):
            try:
                records = load_eta_sweep_records(dro_dir, regime_name)
            except FileNotFoundError:
                continue
            by_regime.setdefault(regime_name, {})[case_name] = records
    return by_regime


def main() -> None:
    args = parse_args()

    if args.dro_dir is not None:
        study_path = Path(args.dro_dir)
    else:
        study_path = SENSITIVITY_RESULTS_ROOT / args.sensitivity_case
    cases = _discover_cases(study_path)

    if not cases:
        print(f"No DRO eta-sweep results found under {study_path}")
        return

    study_name = study_path.name
    print(f"Found {len(cases)} case(s) under '{study_name}':")
    for case_name, _ in cases:
        print(f"  {case_name}")
    print()

    output_root = OUTPUT_ROOT / study_name
    by_regime = _load_study(cases)

    # Prepend base-case curves to every regime if --base-dir was given.
    if args.base_dir is not None:
        base_path = Path(args.base_dir)
        for regime_name in discover_regime_names(base_path):
            try:
                base_records = load_eta_sweep_records(base_path, regime_name)
            except FileNotFoundError:
                continue
            if regime_name in by_regime:
                label = _case_label(base_records, "base_test_case")
                # Insert at the front so the base case is the first curve.
                by_regime[regime_name] = {label: base_records, **by_regime[regime_name]}

    for regime_name, case_records in by_regime.items():
        output_dir = output_root / regime_name
        clean_output_dir(output_dir)

        eta_path = plot_comparison_eta_sweep(case_records, regime_name, output_dir)
        frontier_path = plot_comparison_epsilon_frontier(case_records, regime_name, output_dir)

        all_records = [r for recs in case_records.values() for r in recs]
        csv_path = write_summary_csv(all_records, output_dir / "eta_sweep_summary.csv")

        print(f"[{regime_name}]")
        print(f"  Saved comparison eta-sweep:      {eta_path}")
        print(f"  Saved comparison epsilon frontier: {frontier_path}")
        print(f"  Saved summary CSV:                {csv_path}")


if __name__ == "__main__":
    main()
