# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-22
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Base-case sampled demand and wind profile figures (thesis-ready).

Reads the base-case synthetic scenarios and renders two untitled vector PDFs:

  * demand_profiles -- up to N sampled demand trajectories with the mean
    highlighted; no legend (single regime).
  * wind_profiles   -- the wind generators overlaid in one axes, each a distinct
    color, with per-generator means highlighted and a legend.

Both are sized so their native width equals 0.85\\textwidth on A4, so including
them at that width is 1:1 and the document-point fonts stay legible.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.plot_base_case_profiles
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import results_viz._thesis_style  # noqa: F401  (installs PDF+PNG savefig)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_CSV = PROJECT_ROOT / "results" / "base_case" / "synthetic_scenarios" / "scenarios.csv"
SETUP_DIR = PROJECT_ROOT / "results" / "base_case" / "figures" / "setup"

N_PROFILES = 1000  # trajectories to draw per series
WIND_COLUMNS = ["W1_B1_profile", "W2_B1_profile", "W3_B1_profile"]
WIND_LABELS = {"W1_B1_profile": "W1", "W2_B1_profile": "W2", "W3_B1_profile": "W3"}
RANDOM_STATE = 12345

# Size the figure so its native width equals 0.85\textwidth on A4; including it at
# that width is then 1:1, keeping the document-point fonts legible. TEXTWIDTH_IN
# is a typical A4 thesis \textwidth (~16 cm); adjust if yours differs.
TEXTWIDTH_IN = 6.3
WIDTH_FRACTION = 0.85
FIG_WIDTH_IN = TEXTWIDTH_IN * WIDTH_FRACTION
FIG_HEIGHT_IN = FIG_WIDTH_IN * 0.72
FONT_SIZES = {
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
}

PROFILE_ALPHA = 0.02  # per-trajectory opacity (faint cloud)
PROFILE_LW = 0.8
MEAN_LW = 3.0  # highlighted mean


def _parse_profiles(series: pd.Series) -> np.ndarray:
    """Parse a column of stringified float lists into a (n, horizon) array."""
    return np.array([json.loads(s) for s in series], dtype=float)


def _sample(array: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(array) <= n:
        return array
    idx = rng.choice(len(array), size=n, replace=False)
    return array[idx]


def make_demand_figure(df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_STATE)
    profiles = _parse_profiles(df["demand_profile"])
    horizon = profiles.shape[1]
    sampled = _sample(profiles, N_PROFILES, rng)
    mean = profiles.mean(axis=0)

    with plt.rc_context(FONT_SIZES):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        time = range(horizon)
        for row in sampled:
            ax.plot(time, row, color="tab:blue", alpha=PROFILE_ALPHA, linewidth=PROFILE_LW)
        ax.plot(time, mean, color="black", linewidth=MEAN_LW)  # highlighted mean
        ax.set_xlabel("Time")
        ax.set_ylabel("Demand (MW)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def make_wind_figure(df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(RANDOM_STATE)
    cmap = plt.get_cmap("tab10")

    with plt.rc_context(FONT_SIZES):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        handles = []
        for gen_idx, col in enumerate(WIND_COLUMNS):
            color = cmap(gen_idx % cmap.N)
            profiles = _parse_profiles(df[col])
            horizon = profiles.shape[1]
            time = range(horizon)
            for row in _sample(profiles, N_PROFILES, rng):
                ax.plot(time, row, color=color, alpha=PROFILE_ALPHA, linewidth=PROFILE_LW)
            ax.plot(time, profiles.mean(axis=0), color=color, linewidth=MEAN_LW)
            handles.append(Line2D([0], [0], color=color, lw=MEAN_LW, label=WIND_LABELS[col]))

        ax.set_xlabel("Time")
        ax.set_ylabel("Wind availability (MW)")
        ax.grid(True, alpha=0.25)
        ax.legend(handles=handles, title="Generator", loc="best")
        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    df = pd.read_csv(SCENARIOS_CSV)
    demand_path = SETUP_DIR / "demand_profiles.pdf"
    wind_path = SETUP_DIR / "wind_profiles.pdf"
    make_demand_figure(df, demand_path)
    print(f"Wrote {demand_path} (+ .png)")
    make_wind_figure(df, wind_path)
    print(f"Wrote {wind_path} (+ .png)")


if __name__ == "__main__":
    main()
