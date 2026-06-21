# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-16
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Compare the DRO PoA-eta sweep across the ``dro_regime_box`` study's regimes.

The study (``driver/sensitivity/dro_regime_box_sweep.py``) re-centers the base-case
DRO on 9 regimes: a ``mean`` regime at the ambiguity-box midpoint plus each of
mu_D / sigma_D / mu_W / sigma_W pushed to its lower and upper box bound (rho and
peak_W held fixed). This script reads each regime's already-solved eta-sweep
(nothing is re-solved) and produces:

  * ``dro_regime_box_overlay.png``   -- all 9 regimes overlaid: PoA vs eta and
                                        PoA vs achieved Wasserstein radius.
  * ``dro_regime_box_by_param.png``  -- a 2x2 grid, one panel per swept parameter,
                                        each showing lo / mean / hi (PoA vs eta).
  * ``dro_regime_box_by_param_epsilon.png`` -- the same 2x2 grid but PoA vs the
                                        achieved Wasserstein radius (epsilon).
  * ``dro_regime_box_comparison.csv``-- per-eta table across regimes.

The base-case DRO centered on the actual PoA-optimized worst-case regime is
overlaid on every figure (red) as the reference "worst case" the box-midpoint
regimes are compared against.

Each regime lives in its own isolated run dir and is stored internally under the
regime name ``poa_worst_case``; only the run directory differs.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.plot_dro_regime_box
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from driver.sensitivity.dro_regime_box_sweep import (  # noqa: E402
    STUDY_NAME,
    _SWEPT,
    build_specs,
)
from results_viz.compare_dro_regimes import (  # noqa: E402
    RegimeSeries,
    load_series,
    plot_comparison,
    write_comparison_csv,
)

# Each box regime is written internally under poa_worst_case; only the dir differs.
INTERNAL_REGIME_NAME = "poa_worst_case"
STUDY_ROOT = Path("results/sensitivity_studies") / STUDY_NAME
OUTPUT_DIR = STUDY_ROOT

# The base pipeline centers its DRO on the actual PoA-optimized worst-case regime
# (mu_D / sigma_D / mu_W / sigma_W at the upper level's worst-case state). Overlay
# it as the reference "worst case" against the box-midpoint regimes.
WORST_CASE_DIR = Path("results/base_case/dro")
WORST_CASE_LABEL = "worst_case"
WORST_CASE_COLOR = "red"


def _regime_dir(regime_name: str) -> Path:
    return STUDY_ROOT / regime_name / "dro"


def load_worst_case_series() -> RegimeSeries | None:
    """Load the base-case worst-case DRO eta-sweep, or None if it is absent."""
    try:
        return load_series(
            WORST_CASE_DIR, INTERNAL_REGIME_NAME, WORST_CASE_LABEL, WORST_CASE_COLOR
        )
    except FileNotFoundError:
        print(f"[skip] no base-case worst-case DRO eta-sweep at {WORST_CASE_DIR}")
        return None


def load_box_series() -> list[RegimeSeries]:
    """Load every regime that has a solved eta-sweep; warn-skip the rest."""
    specs = build_specs()
    cmap = plt.get_cmap("tab10")
    series: list[RegimeSeries] = []
    for idx, spec in enumerate(specs):
        results_dir = _regime_dir(spec.name)
        color = "black" if spec.name == "mean" else cmap(idx % 10)
        try:
            series.append(load_series(results_dir, INTERNAL_REGIME_NAME, spec.name, color))
        except FileNotFoundError:
            print(f"[skip] no solved DRO eta-sweep for regime '{spec.name}' ({results_dir})")
    if not series:
        raise FileNotFoundError(
            f"No solved regimes found under {STUDY_ROOT}. "
            "Run driver.sensitivity.dro_regime_box_sweep first."
        )
    return series


def _series_xy(s: RegimeSeries, x_axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, poa) for the requested x-axis.

    ``eta``: positive-eta points only (eta=0 has no log position).
    ``epsilon``: every point, ordered by the achieved Wasserstein radius so the
    frontier line reads left-to-right.
    """
    if x_axis == "epsilon":
        order = np.argsort(s.radius)
        return s.radius[order], s.poa[order]
    pos = s.eta > 0
    return s.eta[pos], s.poa[pos]


def plot_by_parameter(
    series: list[RegimeSeries],
    output_dir: Path,
    worst: RegimeSeries | None = None,
    x_axis: str = "eta",
) -> Path:
    """2x2 grid: one panel per swept parameter, lo / mean / hi as PoA vs x.

    ``x_axis`` selects the abscissa: ``eta`` (Wasserstein penalty, log scale) or
    ``epsilon`` (achieved Wasserstein radius, linear scale). If ``worst`` is
    given, the PoA-optimized worst-case regime is overlaid on every panel as a
    common reference (red dashed).
    """
    if x_axis not in ("eta", "epsilon"):
        raise ValueError(f"x_axis must be 'eta' or 'epsilon', got {x_axis!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    by_name = {s.label: s for s in series}
    mean = by_name.get("mean")
    use_eps = x_axis == "epsilon"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=use_eps)
    for ax, (param, tag) in zip(axes.flat, _SWEPT):
        for suffix, color, marker in (("lo", "tab:blue", "v-"), ("hi", "tab:orange", "^-")):
            s = by_name.get(f"{tag}_{suffix}")
            if s is None:
                continue
            x, y = _series_xy(s, x_axis)
            ax.plot(x, y, marker, color=color, linewidth=2.0,
                    markersize=5.5, label=f"{param} {suffix}")
        if mean is not None:
            x, y = _series_xy(mean, x_axis)
            ax.plot(x, y, "o-", color="black", linewidth=2.0,
                    markersize=4.5, label="mean")
        if worst is not None:
            x, y = _series_xy(worst, x_axis)
            ax.plot(x, y, "s--", color=worst.color,
                    linewidth=2.0, markersize=4.5, label="worst case", alpha=0.9)
        if not use_eps:
            ax.set_xscale("log")
        ax.set_title(f"Sweep {param}")
        ax.set_ylabel("worst-case expected PoA", fontsize=9)
        ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8)
    xlabel = (
        "achieved Wasserstein radius $\\epsilon$ (transport budget)"
        if use_eps
        else r"$\eta$ (Wasserstein penalty)"
    )
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=9)

    axis_word = "PoA frontier vs radius" if use_eps else "mean/outer sweep"
    fig.suptitle(f"DRO regime box: per-parameter {axis_word}", fontsize=13)
    fig.tight_layout()
    suffix = "_epsilon" if use_eps else ""
    out_path = output_dir / f"dro_regime_box_by_param{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    box_series = load_box_series()
    worst = load_worst_case_series()
    # Overlay the worst-case regime alongside the box regimes everywhere.
    series = box_series + ([worst] if worst is not None else [])

    write_comparison_csv(series, OUTPUT_DIR / "dro_regime_box_comparison.csv")
    # Reuse the tested overlay (PoA vs eta + PoA vs achieved radius), then rename.
    overlay = plot_comparison(series, OUTPUT_DIR)
    renamed = OUTPUT_DIR / "dro_regime_box_overlay.png"
    Path(overlay).replace(renamed)
    print(f"[saved] {renamed}")
    plot_by_parameter(box_series, OUTPUT_DIR, worst=worst, x_axis="eta")
    plot_by_parameter(box_series, OUTPUT_DIR, worst=worst, x_axis="epsilon")

    # Console summary: nominal (high-eta) and adversarial (eta->0) endpoints.
    print("\nendpoint summary (worst-case expected PoA):")
    for s in series:
        nominal = s.poa[np.argmax(s.eta)]
        adversarial = s.poa[np.argmin(s.eta)]
        print(f"  {s.label:8s}: nominal(eta_max)={nominal:.3f}  "
              f"adversarial(eta_min)={adversarial:.3f}  "
              f"max_radius={np.nanmax(s.radius):.1f}")


if __name__ == "__main__":
    main()
