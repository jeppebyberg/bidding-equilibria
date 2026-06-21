# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-11
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Explanatory figures for the PoA support set (generic process, no model data needed).

Figure 1 (level box): mean/reference trajectory with nested level-box bands at
several joint-confidence levels.  The band half-width at time t is

    kappa(confidence) * sigma * scale(t),
    scale(t) = min( (1 - rho^{t+1}) / (1 - rho),  1 / sqrt(1 - rho^2) )

so the figure overlays the cumulative envelope (growing) and the stationary
envelope (constant) and shows the effective box is the pointwise minimum,
together with an admissible trajectory inside the box.

Figure 2 (AR(1) tube): the AR(1) tube constrains the whitened innovation

    eps_t = x_t - rho * x_{t-1} - ar1_ref_t,   |eps_t| <= kappa * sigma

which forces step-to-step smoothness.  The figure shows an admissible
trajectory and, at a few time steps, the one-step conditional tube
rho * x_{t-1} + ar1_ref_t +/- kappa * sigma that the next value must lie in.

kappa and scale(t) are imported from models.PoA.poa_model.support_set so the
figures stay consistent with the optimization model.

Output:
    results_viz/figures/support_set_explainer/support_set_level_bounds.png
    results_viz/figures/support_set_explainer/support_set_ar1_tube.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
import numpy as np

from models.PoA.poa_model.support_set import _ar1_kappa, _level_scale

OUTPUT_DIR = Path("results_viz/figures/support_set_explainer")

# Generic AR(1) process parameters (illustrative, not tied to demand or wind).
# rho is chosen high enough that the cumulative-vs-stationary crossover is visible.
T = 24
RHO = 0.9
SIGMA = 0.06  # innovation sigma, in units of the reference level
CONFIDENCE_LEVELS = [0.80, 0.95, 0.99]
HIGHLIGHT_CONFIDENCE = 0.95
AMBIGUITY_KAPPA = 0.25  # budget fraction, matches driver/project_config.py

# Larger fonts so labels stay readable when the figure is placed on an A4 page.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Figure size matched to the A4 text width (~16 cm) so it is included ~1:1.
A4_FIGSIZE = (6.3, 3.9)


def _reference_shape(num_steps: int) -> np.ndarray:
    """Generic smooth daily-looking reference trajectory (mean of the process)."""
    t = np.arange(num_steps)
    return 1.0 + 0.18 * np.sin(2.0 * np.pi * (t - 5.0) / num_steps) + 0.07 * np.sin(
        4.0 * np.pi * (t - 2.0) / num_steps
    )


def _scales(num_steps: int, rho: float) -> np.ndarray:
    return np.asarray([_level_scale(rho, t) for t in range(num_steps)])


def _ar1_ref(reference: np.ndarray, rho: float) -> np.ndarray:
    """AR(1) reference increment ar1_ref_t = ref_t - rho * ref_{t-1} for t >= 1."""
    return reference[1:] - rho * reference[:-1]


# ---------------------------------------------------------------------------
# Figure 1: level box with cumulative and stationary bounds
# ---------------------------------------------------------------------------

def plot_level_bounds(output_path: Path) -> None:
    time = np.arange(T)
    reference = _reference_shape(T)
    scales = _scales(T, RHO)

    fig, ax = plt.subplots(figsize=A4_FIGSIZE)

    # Single 95% level box.
    kappa_h = _ar1_kappa(T, HIGHLIGHT_CONFIDENCE)
    half_h = kappa_h * SIGMA * scales
    ax.fill_between(
        time,
        reference - half_h,
        reference + half_h,
        color=plt.cm.Blues(0.35),
        label=f"level box, joint {HIGHLIGHT_CONFIDENCE:.0%}",
    )

    # Overlay the two envelopes whose minimum defines the effective box:
    # cumulative (growing) and stationary (constant).
    cumulative = (1.0 - RHO ** (time + 1.0)) / (1.0 - RHO)
    stationary = np.full(T, 1.0 / np.sqrt(1.0 - RHO**2))
    for sign in (+1, -1):
        ax.plot(
            time, reference + sign * kappa_h * SIGMA * cumulative,
            color="tab:red", ls="--", lw=1.4,
            label="cumulative bound (95%)" if sign > 0 else None,
        )
        ax.plot(
            time, reference + sign * kappa_h * SIGMA * stationary,
            color="tab:green", ls=":", lw=1.6,
            label="stationary bound (95%)" if sign > 0 else None,
        )

    ax.plot(time, reference, color="black", lw=2.0, label="mean trajectory (reference)")
    kappa_admissible = kappa_h
    ax.plot(
        time, _illustrative_trajectory(reference, kappa_admissible),
        color="tab:blue", lw=1.8, marker="o", ms=4, label="admissible trajectory",
    )

    # Clip y so the bands stay readable; the cumulative envelope runs off-plot,
    # illustrating that the stationary bound caps it.
    margin = 1.35 * kappa_h * SIGMA * stationary[0]
    ax.set_ylim(reference.min() - margin, reference.max() + margin)

    ax.set_xlabel("time step t")
    ax.set_ylabel(r"Level (rel. to $\widetilde{D}$)")
    # 5 entries over 3 columns -> 2 rows.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.26),
        ncol=3, fontsize=9, frameon=True,
        columnspacing=1.0, handletextpad=0.4,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: AR(1) tube
# ---------------------------------------------------------------------------

def _illustrative_trajectory(reference: np.ndarray, kappa: float, seed: int = 7) -> np.ndarray:
    """Admissible AR(1) path: small innovations plus a downward excursion.

    All innovations stay inside the tube (|z_t| <= kappa); the excursion around
    t=9..11 drives the path toward the lower band edge so one annotated point
    sits far from the mean while remaining admissible.  Explicit recovery
    innovations at t=12..13 bring the path back so the total absolute deviation
    also satisfies the budget constraint (checked in _assert_admissible).
    """
    rng = np.random.default_rng(seed)
    z = np.clip(0.3 * rng.standard_normal(T), -0.5 * kappa, 0.5 * kappa)
    z[7] = 0.55 * kappa
    z[9], z[10], z[11] = -0.75 * kappa, -0.75 * kappa, -0.45 * kappa
    z[12], z[13] = 0.8 * kappa, 0.6 * kappa
    ar1_ref = _ar1_ref(reference, RHO)
    x = np.empty(T)
    x[0] = reference[0] + SIGMA * z[0]
    for t in range(1, T):
        x[t] = RHO * x[t - 1] + ar1_ref[t - 1] + SIGMA * z[t]
    _assert_admissible(x, reference, kappa)
    return x


def _assert_admissible(x: np.ndarray, reference: np.ndarray, kappa: float) -> None:
    """Verify all support-set constraints (not shown in the figures).

    Level box:  |x_t - ref_t| <= kappa * sigma * scale(t)
    AR(1) tube: |x_t - rho * x_{t-1} - ar1_ref_t| <= kappa * sigma (incl. t=0 cold start)
    Budget:     sum_t |x_t - ref_t| <= ambiguity_kappa * sum_t kappa * sigma * scale(t)
    """
    tol = 1e-9
    half_width = kappa * SIGMA * _scales(T, RHO)
    deviation = x - reference
    assert np.all(np.abs(deviation) <= half_width + tol), "level box violated"
    eps = np.concatenate(([deviation[0]], x[1:] - RHO * x[:-1] - _ar1_ref(reference, RHO)))
    assert np.all(np.abs(eps) <= kappa * SIGMA + tol), "AR(1) tube violated"
    budget = AMBIGUITY_KAPPA * np.sum(half_width)
    assert np.sum(np.abs(deviation)) <= budget + tol, "budget constraint violated"


def plot_ar1_tube(output_path: Path) -> None:
    time = np.arange(T)
    reference = _reference_shape(T)
    kappa = _ar1_kappa(T, HIGHLIGHT_CONFIDENCE)
    half_width = kappa * SIGMA * _scales(T, RHO)

    admissible = _illustrative_trajectory(reference, kappa)
    ar1_ref = _ar1_ref(reference, RHO)

    # Zoom on a snippet of the horizon so the individual steps are readable.
    t_lo, t_hi = 5, 14
    window = slice(t_lo, t_hi + 1)

    fig, ax = plt.subplots(figsize=A4_FIGSIZE)

    ax.fill_between(
        time[window], (reference - half_width)[window], (reference + half_width)[window],
        color=plt.cm.Blues(0.25), label=f"level box (joint {HIGHLIGHT_CONFIDENCE:.0%})",
    )
    ax.plot(time[window], reference[window], color="black", lw=2.0,
            label="mean trajectory (reference)")
    ax.plot(time[window], admissible[window], color="tab:blue", lw=1.8, marker="o", ms=5,
            label="admissible trajectory")

    def draw_one_step_tube(t_star: int, with_legend: bool) -> float:
        """Highlight x_{t_star-1}, link it to the tube it generates at t_star."""
        x_prev = admissible[t_star - 1]
        center = RHO * x_prev + ar1_ref[t_star - 1]
        ax.plot(t_star - 1, x_prev, "o", ms=14, mfc="none", mec="tab:purple",
                mew=2.2, zorder=6)
        ax.annotate(
            "",
            xy=(t_star, center), xytext=(t_star - 1, x_prev),
            arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=1.8, ls="--",
                            shrinkA=10, shrinkB=4),
        )
        ax.errorbar(
            t_star, center, yerr=kappa * SIGMA, color="tab:purple", lw=2.8,
            capsize=7, capthick=2.4, zorder=6,
            label="one-step tube given $x_{t-1}$" if with_legend else None,
        )
        return center

    # Point near the mean: tube is roughly centred on the mean trajectory.
    draw_one_step_tube(8, with_legend=True)

    # Point far below the mean: |rho| < 1 pulls the tube centre back toward it.
    t_far = 12
    center_far = draw_one_step_tube(t_far, with_legend=False)
    ax.plot(t_far, center_far, "o", ms=5, color="tab:purple", zorder=7)

    ax.set_xlabel("time step t")
    ax.set_ylabel(r"Level (rel. to $\widetilde{D}$)")
    # 4 entries over 2 columns -> 2 rows.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.26),
        ncol=2, fontsize=9, frameon=True,
        columnspacing=1.0, handletextpad=0.4,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_level_bounds(OUTPUT_DIR / "support_set_level_bounds.png")
    plot_ar1_tube(OUTPUT_DIR / "support_set_ar1_tube.png")
    print(f"Saved figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
