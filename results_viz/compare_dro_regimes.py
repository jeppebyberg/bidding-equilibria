# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-14
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Overlay the DRO PoA-eta sweep of two (or more) center regimes.

Compares how the DRO PoA bound responds to the ambiguity penalty eta when the
ambiguity ball is centered on different regimes -- e.g. the realistic ``normal``
regime (driver/sensitivity/dro_normal_regime_sweep.py) versus the PoA-selected
``poa_worst_case`` regime (the base pipeline default). Each regime's sweep is
read with the same loader the frontier plots use; nothing is re-solved.

Two panels:
  * PoA vs eta            -- robustness penalty on the x-axis (log scale).
  * PoA vs achieved radius -- the DRO frontier (worst-case expected PoA against
                              the realized Wasserstein transport).

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.compare_dro_regimes
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
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

from results_viz.plot_dro_poa_eta_sweep import load_eta_sweep_records  # noqa: E402


@dataclass
class RegimeSeries:
    label: str
    color: str
    eta: np.ndarray
    poa: np.ndarray          # worst_case_expected_poa
    ratio: np.ndarray        # average_poa_ratio
    radius: np.ndarray       # achieved average Wasserstein distance


def load_series(results_dir: Path, regime_name: str, label: str, color: str) -> RegimeSeries:
    records = load_eta_sweep_records(Path(results_dir), regime_name, include_archives=False)
    records = [r for r in records if r.get("worst_case_expected_poa") is not None]
    records.sort(key=lambda r: float(r["eta"]))
    return RegimeSeries(
        label=label,
        color=color,
        eta=np.array([float(r["eta"]) for r in records]),
        poa=np.array([float(r["worst_case_expected_poa"]) for r in records]),
        ratio=np.array([float(r.get("average_poa_ratio") or np.nan) for r in records]),
        radius=np.array([float(r.get("average_wasserstein_distance") or np.nan) for r in records]),
    )


def write_comparison_csv(series: list[RegimeSeries], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_eta: dict[float, dict[str, Any]] = {}
    for s in series:
        for e, p, r, w in zip(s.eta, s.poa, s.ratio, s.radius):
            row = by_eta.setdefault(round(float(e), 8), {"eta": float(e)})
            row[f"{s.label}_poa"] = p
            row[f"{s.label}_ratio"] = r
            row[f"{s.label}_radius"] = w
    fieldnames = ["eta"] + [
        f"{s.label}_{k}" for s in series for k in ("poa", "ratio", "radius")
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for e in sorted(by_eta):
            writer.writerow(by_eta[e])
    return output_path


def plot_comparison(series: list[RegimeSeries], output_dir: Path, show: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for s in series:
        pos = s.eta > 0
        ax.plot(s.eta[pos], s.poa[pos], "o-", color=s.color, linewidth=2.2,
                markersize=5.5, label=s.label)
        # eta = 0 (max ambiguity) shown as a marker at the left edge.
        if np.any(~pos):
            ax.plot(s.eta[pos].min() if pos.any() else 1e-3, s.poa[~pos][0],
                    "*", color=s.color, markersize=12)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$  (Wasserstein penalty; higher = less robust)", fontsize=10)
    ax.set_ylabel("worst-case expected PoA", fontsize=10)
    ax.set_title("DRO PoA vs ambiguity penalty")
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.1, alpha=0.7)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=9)

    ax = axes[1]
    for s in series:
        order = np.argsort(s.radius)
        ax.plot(s.radius[order], s.poa[order], "o-", color=s.color, linewidth=2.2,
                markersize=5.5, label=s.label)
    ax.set_xlabel("achieved Wasserstein radius (transport budget)", fontsize=10)
    ax.set_ylabel("worst-case expected PoA", fontsize=10)
    ax.set_title("DRO frontier: PoA vs achieved radius")
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.1, alpha=0.7)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.suptitle("DRO center-regime comparison", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / "dro_regime_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    series = [
        load_series(
            Path("results/sensitivity_studies/dro_normal_regime/normal/dro"),
            "normal", "normal", "tab:green",
        ),
        load_series(
            Path("results/base_case/dro"),
            "poa_worst_case", "worst_case", "tab:red",
        ),
    ]
    out_dir = Path("results_viz/figures/dro_regime_comparison")
    write_comparison_csv(series, out_dir / "dro_regime_comparison.csv")
    plot_comparison(series, out_dir)

    # Console summary: nominal (high-eta) and most-adversarial (eta->0) endpoints.
    print("\nendpoint summary (worst-case expected PoA):")
    for s in series:
        nominal = s.poa[np.argmax(s.eta)]
        adversarial = s.poa[np.argmin(s.eta)]
        print(f"  {s.label:12s}: nominal(eta_max)={nominal:.3f}  "
              f"adversarial(eta_min)={adversarial:.3f}  "
              f"max_radius={np.nanmax(s.radius):.1f}")


if __name__ == "__main__":
    main()
