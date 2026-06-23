# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-22
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Show that bidding blocks approximate a quadratic generation cost curve.

The reference-case conventional generators are defined by two equal-width
bidding blocks whose marginal costs are midpoint samples of an underlying convex
(quadratic) cost C(p) = a*p^2 + b*p. This figure recovers that quadratic from a
generator's blocks and overlays the piecewise-linear total-cost curve obtained
with B = 1, 2, 4, 8, ... equal-width blocks, so the approximation visibly tightens
as the number of blocks grows. With the midpoint-marginal rule the block curve
passes through the quadratic exactly at every block boundary, so more blocks add
more contact points.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.generation_cost_curves_blocks
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

import results_viz._thesis_style  # noqa: F401  (installs PDF+PNG savefig)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = PROJECT_ROOT / "config" / "reference_cases.yaml"
OUTPUT_PATH = PROJECT_ROOT / "setup_viz" / "generation_cost_curves_blocks.png"

# Conventional generator to display and the block counts to overlay. The cost is
# deliberately exaggerated (strong curvature, pure quadratic) so the block
# approximation is easy to see; absolute magnitudes are illustrative, so the
# y-axis label and tick values are suppressed.
DEFAULT_GENERATOR = "G2"
BLOCK_COUNTS = [1, 2, 4]
CURVATURE_BOOST = 1.0  # multiplies the recovered quadratic coefficient a

# Size the figure so its native width equals the width it is shown at in LaTeX
# (0.85\textwidth). Including it at that width is then 1:1, so the fonts -- set in
# document point sizes below -- stay legible instead of being scaled down.
# TEXTWIDTH_IN is a typical A4 thesis \textwidth (~16 cm); adjust if yours differs
# (print \the\textwidth in LaTeX, divide by 72.27 for inches).
TEXTWIDTH_IN = 6.3
WIDTH_FRACTION = 0.85
FIG_WIDTH_IN = TEXTWIDTH_IN * WIDTH_FRACTION
FIG_HEIGHT_IN = FIG_WIDTH_IN * 0.72
# Document-matched font sizes (pt) so text reads like ~11pt body text on the page.
FONT_SIZES = {
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
}


def _load_generators(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    # reference_cases.yaml nests generators under a single top-level case mapping.
    case = next(iter(config.values())) if "generators" not in config else config
    return list(case["generators"])


def _generator_blocks(generators: list[dict], name: str) -> list[dict]:
    gen = next((g for g in generators if str(g.get("name")) == name), None)
    if gen is None:
        raise ValueError(f"Generator '{name}' not found in {REFERENCE_CASES}")
    blocks = sorted(gen["bidding_blocks"], key=lambda b: int(b["block_id"]))
    if not blocks:
        raise ValueError(f"Generator '{name}' has no bidding blocks")
    return blocks


def _fit_quadratic(blocks: list[dict]) -> tuple[float, float, float]:
    """Recover C(p) = a*p^2 + b*p through the blocks' cumulative cost points.

    Returns (a, b, p_max). Each block contributes width pmax at constant marginal
    cost; cumulative (production, total_cost) points lie on the quadratic, so a
    least-squares fit on [p^2, p] (origin-anchored) recovers it exactly for the
    two-block reference cases.
    """
    prod = np.cumsum([float(b["pmax"]) for b in blocks])
    cost = np.cumsum([float(b["pmax"]) * float(b["cost"]) for b in blocks])
    design = np.column_stack([prod**2, prod])
    (a, b), *_ = np.linalg.lstsq(design, cost, rcond=None)
    return float(a), float(b), float(prod[-1])


def _block_curve(a: float, b: float, p_max: float, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear total cost using n equal-width midpoint-marginal blocks."""
    width = p_max / n_blocks
    midpoints = (np.arange(n_blocks) + 0.5) * width
    marginal = 2.0 * a * midpoints + b  # derivative of a*p^2 + b*p at the midpoint
    production = np.concatenate([[0.0], np.cumsum(np.full(n_blocks, width))])
    total_cost = np.concatenate([[0.0], np.cumsum(marginal * width)])
    return production, total_cost


def make_figure(generator_name: str = DEFAULT_GENERATOR, save_path: Path | None = OUTPUT_PATH):
    import matplotlib.pyplot as plt

    generators = _load_generators(REFERENCE_CASES)
    blocks = _generator_blocks(generators, generator_name)
    a, _b, p_max = _fit_quadratic(blocks)
    # Exaggerate the curvature and drop the linear term so the convex shape (and
    # the looseness of a coarse block approximation) is unmistakable.
    a *= CURVATURE_BOOST
    b = 0.0

    with plt.rc_context(FONT_SIZES):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        cmap = plt.get_cmap("viridis")

        p_dense = np.linspace(0.0, p_max, 400)
        c_dense = a * p_dense**2 + b * p_dense
        ax.plot(p_dense, c_dense, color="black", linewidth=2.6, label="True quadratic cost")

        for idx, n_blocks in enumerate(BLOCK_COUNTS):
            production, total_cost = _block_curve(a, b, p_max, n_blocks)
            color = cmap(idx / max(len(BLOCK_COUNTS) - 1, 1))
            ax.plot(
                production,
                total_cost,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=color,
                alpha=0.95,
                label=f"{n_blocks} block" + ("s" if n_blocks != 1 else ""),
            )

        ax.set_xlabel("Production (MW)")
        ax.set_yticklabels([])  # magnitudes are illustrative; hide the y axis
        ax.grid(True, alpha=0.25)
        ax.legend(title="Approximation", loc="upper left", framealpha=0.9)
        fig.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def main(argv: list[str]) -> None:
    name = argv[0] if argv else DEFAULT_GENERATOR
    make_figure(name)
    print(f"Wrote {OUTPUT_PATH} (+ .pdf)")


if __name__ == "__main__":
    main(sys.argv[1:])
