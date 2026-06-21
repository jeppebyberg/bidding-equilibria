# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-06
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Comparison plots of PoA curves across all cases in a sensitivity study.

Two figures are produced per study:

  1. **eta sweep**  — average PoA ratio vs η (Wasserstein penalty parameter),
     one curve per case.  eta=0 is shown as a square marker to the left of the
     log scale.  A dashed horizontal line marks the base PoA (from the PoA
     optimisation result) for each case.

  2. **epsilon frontier** — average PoA ratio vs achieved Wasserstein distance,
     one curve per case.  η values are annotated sparsely along each curve.
     This reveals the PoA-robustness trade-off across compositions.

Usage
-----
Edit STUDY_NAME at the bottom and run:
  .venv\\Scripts\\python.exe results_viz\\plot_composition_comparison.py
"""
from __future__ import annotations

import json
import math
import re
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

from results_viz.plot_dro_poa_eta_sweep import load_eta_sweep_records

RESULTS_ROOT = Path("results/sensitivity_studies")
OUTPUT_ROOT = Path("results_viz/figures/composition_comparison")
DEFAULT_REGIME_NAME = "poa_worst_case"


# ---------------------------------------------------------------------------
# Case-label helpers
# ---------------------------------------------------------------------------

def _extract_curve_label(case_name: str) -> str:
    """Convert a case name to a compact plot label.

    ``cap_rho3p0_omega0p40_3W_3C``   →  "ω = 0.40  (3W + 3C)"
    ``comp_3W_3C``                    →  "3W + 3C"
    ``peak_w_14``                     →  "peak hour = 14"
    ``base_test_case``                →  "base_test_case"
    """
    peak_match = re.match(r"peak_w_(?P<tau>\d+)", case_name)
    if peak_match:
        return f"peak hour = {peak_match.group('tau')}"

    cap_match = re.match(
        r"cap_rho(?P<rho>[\dp]+)_omega(?P<omega>[\dp]+)_(?P<nw>\d+)W_(?P<nc>\d+)C",
        case_name,
    )
    if cap_match:
        omega = float(cap_match.group("omega").replace("p", "."))
        nw, nc = cap_match.group("nw"), cap_match.group("nc")
        return f"ω = {omega:.2f}  ({nw}W + {nc}C)"

    cnt_match = re.match(r"comp_(?P<nw>\d+)W_(?P<nc>\d+)C", case_name)
    if cnt_match:
        return f"{cnt_match.group('nw')}W + {cnt_match.group('nc')}C"

    # Generic ``...NW_NC`` suffix (e.g. ``test_3W_1C``).
    suffix_match = re.search(r"(?P<nw>\d+)W_(?P<nc>\d+)C", case_name)
    if suffix_match:
        return f"{suffix_match.group('nw')}W + {suffix_match.group('nc')}C"

    return case_name


def _case_sort_key(case_name: str) -> tuple[float, str]:
    peak_match = re.match(r"peak_w_(?P<tau>\d+)", case_name)
    if peak_match:
        return (float(peak_match.group("tau")), case_name)
    cap_match = re.match(r"cap_rho[\dp]+_omega(?P<omega>[\dp]+)", case_name)
    if cap_match:
        return (float(cap_match.group("omega").replace("p", ".")), case_name)
    cnt_match = re.match(r"comp_(?P<nw>\d+)W", case_name)
    if cnt_match:
        return (float(cnt_match.group("nw")), case_name)
    suffix_match = re.search(r"(?P<nw>\d+)W_\d+C", case_name)
    if suffix_match:
        return (float(suffix_match.group("nw")), case_name)
    return (0.0, case_name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_poa_result(case_dir: Path) -> dict[str, Any] | None:
    poa_dir = case_dir / "poa"
    if not poa_dir.exists():
        return None
    candidates = sorted(poa_dir.glob("poa_optimization_T*_piecewise_mccormick.json"))
    if not candidates:
        return None
    with candidates[-1].open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _poa_reference_value(poa_result: dict[str, Any]) -> float | None:
    obj = poa_result.get("objective") or {}
    val = obj.get("PoA_ratio") or obj.get("PoA")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


CaseEntry = tuple[str, list[dict[str, Any]], dict[str, Any] | None]


def discover_study_cases(study_results_dir: Path, regime_name: str) -> list[CaseEntry]:
    """Return (case_name, eta_records, poa_result) for every case that has results."""
    if not study_results_dir.exists():
        return []
    entries: list[CaseEntry] = []
    for case_dir in sorted(study_results_dir.iterdir(), key=lambda d: _case_sort_key(d.name)):
        if not case_dir.is_dir():
            continue
        try:
            records = load_eta_sweep_records(case_dir / "dro", regime_name)
        except FileNotFoundError:
            continue
        entries.append((case_dir.name, records, _load_poa_result(case_dir)))
    return entries


# ---------------------------------------------------------------------------
# Figure 1 — eta sweep comparison (average PoA vs η)
# ---------------------------------------------------------------------------

def _split_eta_zero(
    records: list[dict[str, Any]], metric: str
) -> tuple[float | None, list[float], list[float]]:
    """Separate eta=0 from positive-eta records."""
    eta0_val: float | None = None
    etas, values = [], []
    for r in records:
        eta, val = r.get("eta"), r.get(metric)
        if eta is None or val is None:
            continue
        if float(eta) == 0.0:
            eta0_val = float(val)
        else:
            etas.append(float(eta))
            values.append(float(val))
    return eta0_val, etas, values


def plot_eta_sweep_comparison(
    study_name: str,
    regime_name: str = DEFAULT_REGIME_NAME,
    results_root: Path = RESULTS_ROOT,
    output_root: Path = OUTPUT_ROOT,
    show: bool = False,
) -> Path:
    """Single-panel average PoA ratio vs η, one curve per case."""
    cases = discover_study_cases(results_root / study_name, regime_name)
    if not cases:
        print(f"No eta-sweep results found under {results_root / study_name}.")
        return output_root / study_name / "no_results.txt"

    output_dir = output_root / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Position eta=0 marker left of the log range.
    pos_etas = [float(r["eta"]) for _, recs, _ in cases for r in recs
                if r.get("eta") is not None and float(r["eta"]) > 0]
    eta0_x = (min(pos_etas) * 0.15) if pos_etas else 1e-3

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (case_name, records, poa_result) in enumerate(cases):
        color = colors[idx % len(colors)]
        label = _extract_curve_label(case_name)

        eta0_val, etas, values = _split_eta_zero(records, "average_poa_ratio")

        if etas:
            ax.semilogx(etas, values, color=color, linewidth=2, marker="o",
                        markersize=5, label=label)

        if eta0_val is not None:
            ax.plot(eta0_x, eta0_val, color=color, marker="s", markersize=8,
                    linestyle="none", zorder=5)
            if etas:
                ax.plot([eta0_x, etas[0]], [eta0_val, values[0]],
                        color=color, linewidth=1, linestyle=":")

        # Horizontal dashed reference at base PoA.
        if poa_result is not None:
            ref_val = _poa_reference_value(poa_result)
            if ref_val is not None:
                ax.axhline(ref_val, color=color, linewidth=1,
                           linestyle="--", alpha=0.45)

    ax.axvline(eta0_x, color="0.7", linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(1.0, color="0.5", linewidth=1, linestyle=":", alpha=0.4, zorder=0)
    ax.set_xlim(left=eta0_x * 0.5)

    # Relabel the eta=0 tick.
    raw_ticks = [t for t in ax.get_xticks() if t > eta0_x * 2]
    ax.set_xticks([eta0_x] + raw_ticks)
    ax.set_xticklabels(["0"] + [f"{t:.3g}" for t in raw_ticks])

    ax.set_xlabel("η  (Wasserstein penalty parameter)")
    ax.set_ylabel("Average PoA ratio")
    ax.grid(True, which="both", alpha=0.2)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.legend(title=study_name.replace("_", " ").title(),
              loc="upper right", fontsize=9, title_fontsize=9)
    fig.suptitle(
        f"PoA vs η — {study_name.replace('_', ' ')}  (regime: {regime_name})",
        fontsize=12,
    )
    fig.tight_layout()

    out = output_dir / f"poa_eta_comparison_{regime_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Thesis-figure curve labels / ordering (match results_viz/plot_base_poa_overview)
# ---------------------------------------------------------------------------

# wind_playing_sweep was not in the bar/line figures, so its labels live here.
_WIND_PLAYING_LABELS = {
    "no_wind": "Wind Gen. true cost",
    "wind": "Wind Gen. act strat.",
}


def _thesis_curve_label(study_name: str, case_name: str) -> str:
    """Per-curve legend label matching the appendix bar/line plots."""
    from results_viz import plot_base_poa_overview as ov

    if study_name == "wind_playing_sweep":
        return _WIND_PLAYING_LABELS.get(case_name, case_name)
    label_fn = ov.THESIS_TICK_LABELS.get(study_name)
    if label_fn is not None:
        return label_fn(case_name)
    meta = ov.STUDY_META.get(study_name)
    if meta is not None:
        x = meta.x_of(case_name)
        if x is not None:
            return f"{int(x) if float(x).is_integer() else x}"
    return _extract_curve_label(case_name)


def _thesis_sort_key(study_name: str, case_name: str) -> tuple[int, float, str]:
    """Order curves by the swept parameter value (numeric where possible)."""
    from results_viz import plot_base_poa_overview as ov

    meta = ov.STUDY_META.get(study_name)
    if meta is not None:
        x = meta.x_of(case_name)
        if x is not None:
            return (0, float(x), case_name)
    return (1, 0.0, _thesis_curve_label(study_name, case_name))


def _thesis_legend_title(study_name: str) -> str:
    from results_viz import plot_base_poa_overview as ov

    if study_name == "wind_playing_sweep":
        return "Wind generators"
    meta = ov.STUDY_META.get(study_name)
    return meta.xlabel if meta is not None else study_name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Figure 2 — epsilon frontier comparison (average PoA vs Wasserstein distance)
# ---------------------------------------------------------------------------

def plot_epsilon_frontier_comparison(
    study_name: str,
    regime_name: str = DEFAULT_REGIME_NAME,
    results_root: Path = RESULTS_ROOT,
    output_root: Path = OUTPUT_ROOT,
    show: bool = False,
    thesis_style: bool = False,
) -> Path:
    """Average PoA vs achieved Wasserstein distance, one curve per case.

    The frontier traces the PoA-robustness trade-off as η varies.  A lower
    curve (lower PoA for the same Wasserstein budget) indicates a more
    benign composition from a market-power perspective.

    ``thesis_style`` switches to the appendix layout: no title, x-axis
    ``ε [MW]``, y-axis ``E PoA``, A4-friendly size, curve labels matching the
    bar/line figures, no η annotations, and a red ring on any point that did
    not solve to optimality.
    """
    cases = discover_study_cases(results_root / study_name, regime_name)
    if not cases:
        print(f"No eta-sweep results found under {results_root / study_name}.")
        return output_root / study_name / "no_results.txt"

    if thesis_style:
        cases = sorted(cases, key=lambda c: _thesis_sort_key(study_name, c[0]))

    output_dir = output_root / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0) if thesis_style else (9, 5.5))

    nonopt_legend_done = False
    for idx, (case_name, records, _) in enumerate(cases):
        color = colors[idx % len(colors)]
        label = (
            _thesis_curve_label(study_name, case_name)
            if thesis_style
            else _extract_curve_label(case_name)
        )

        # Collect (wasserstein_dist, avg_poa_ratio, eta, optimal) tuples.
        points: list[tuple[float, float, float, bool]] = []
        for r in records:
            x = r.get("average_wasserstein_distance")
            y = r.get("average_poa_ratio")
            eta = r.get("eta")
            if x is None or y is None or eta is None:
                continue
            term = str(r.get("solver_termination_condition") or "").lower()
            try:
                points.append((float(x), float(y), float(eta), term == "optimal"))
            except (TypeError, ValueError):
                continue

        if not points:
            continue

        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        etas = [p[2] for p in points]
        n = len(points)

        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=5,
                zorder=3, label=label)

        if thesis_style:
            # No η annotations; instead ring every point not solved to optimality.
            non_opt = [(x, y) for x, y, _, ok in points if not ok]
            if non_opt:
                nx, ny = zip(*non_opt)
                ax.scatter(
                    nx, ny, s=170, facecolors="none", edgecolors="#d62728",
                    linewidths=2.2, zorder=6,
                    label=None if nonopt_legend_done else "Not solved to optimality",
                )
                nonopt_legend_done = True
        else:
            # Annotate η at most 5 points per curve (always first and last).
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

    ax.axhline(1.0, color="0.5", linewidth=1, linestyle=":", alpha=0.4, zorder=0)
    ax.grid(True, which="both", alpha=0.2)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)

    if thesis_style:
        ax.set_xlabel(r"$\varepsilon$ [MW]")
        ax.set_ylabel(r"$\mathbb{E}$ PoA")
        # Push the "Not solved to optimality" entry to the bottom of the legend.
        handles, labels = ax.get_legend_handles_labels()
        order = [i for i, lab in enumerate(labels) if lab != "Not solved to optimality"]
        order += [i for i, lab in enumerate(labels) if lab == "Not solved to optimality"]
        ax.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            title=_thesis_legend_title(study_name), loc="best",
            fontsize=9, title_fontsize=9,
        )
    else:
        ax.set_xlabel("Achieved Wasserstein distance  (distributional deviation)")
        ax.set_ylabel("Average PoA ratio")
        ax.legend(title=study_name.replace("_", " ").title(),
                  loc="upper left", fontsize=9, title_fontsize=9)
        fig.suptitle(
            f"PoA–robustness frontier — {study_name.replace('_', ' ')}  "
            f"(regime: {regime_name})",
            fontsize=12,
        )
    fig.tight_layout()

    out = output_dir / f"poa_frontier_comparison_{regime_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    STUDY_NAME = "capacity_composition"
    REGIME_NAME = DEFAULT_REGIME_NAME

    p1 = plot_eta_sweep_comparison(STUDY_NAME, REGIME_NAME)
    print(f"Saved eta-sweep comparison:  {p1}")

    p2 = plot_epsilon_frontier_comparison(STUDY_NAME, REGIME_NAME)
    print(f"Saved frontier comparison:   {p2}")
