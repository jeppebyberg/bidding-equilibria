# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-15
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Overlay held-out NN test predictions on the residual-demand bid regions.

This is a separate view from visualize_training_labels.py: it does NOT show the
training labels. Instead it shades the high/low bid regions learned from the
data (split on residual demand = demand - wind) and overlays the trained
network's predictions on the held-out test scenarios. Each test point is split
by whether the prediction falls outside the observed label range
[min_label, max_label] (padded by a tolerance): ABOVE the max (triangle up),
BELOW the min (triangle down), or inside the range (gray dot).
"""

from __future__ import annotations

import json
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
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.neural_network.training.model import BiddingPolicyNetwork
from results_viz.visualize_training_labels import (
    DECISION_MODE,
    MAX_LEAF_NODES,
    X_FEATURE,
    Y_FEATURE,
    _classify_binary,
    _draw_regions,
    _fit_shallow_tree,
    _has_variation,
    _target_columns,
)

RAW_DATA_DIR = Path("models/neural_network/features/generated/raw")
NORM_DATA_DIR = Path("models/neural_network/features/generated/normalized")
MODEL_DIR = Path("models/neural_network/training/trained_models")
# Per-block PNGs are written here, one file per (generator, block).
PLOT_DIR = Path("models/neural_network/training/training_results/plots/test_predictions")

# Must match the held-out split used during training (see train_policies.py /
# dataset.load_generator_policy_data).
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Split test points by whether the prediction falls outside the observed label
# range [min_label, max_label], padded by RANGE_TOL:
#   pred > max_label + RANGE_TOL -> above range (triangle up, red)
#   pred < min_label - RANGE_TOL -> below range (triangle down, blue)
#   otherwise                     -> inside range (small gray dot)
COLOR_OVER = "#DC2626"   # predicts above the label range
COLOR_UNDER = "#2563EB"  # predicts below the label range
COLOR_ON = "#9CA3AF"     # prediction inside the label range (within tolerance)
RANGE_TOL = 0.25         # bid-unit padding on the label range
POINT_SIZE = 30
ON_SIZE = 14

# Legend labels (kept here so the per-block plots and the standalone legend match).
LABEL_ABOVE = r"Above range  $[\,c_{j,b}+\nu,\ \infty\,[$"
LABEL_INSIDE = r"Inside range  $[\,c_{i,b}-\nu,\ c_{j,b}+\nu\,]$"
LABEL_BELOW = r"Below range  $]\,{-}\infty,\ c_{i,b}-\nu\,]$"


def _load_model(
    generator_name: str, metadata: dict, model_dir: Path
) -> BiddingPolicyNetwork:
    model = BiddingPolicyNetwork(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_layers=metadata["hidden_layers"],
        final_activation=metadata["final_activation"],
    )
    state_dict = torch.load(
        model_dir / f"{generator_name}_policy.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _test_row_mask(
    norm_df: pd.DataFrame, test_size: float, random_state: int
) -> np.ndarray:
    """Reconstruct the held-out test rows exactly as dataset.py does."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, test_index = next(splitter.split(norm_df, groups=norm_df["scenario_id"]))
    mask = np.zeros(len(norm_df), dtype=bool)
    mask[test_index] = True
    return mask


def _predict_test_bids(
    generator_name: str,
    metadata: dict,
    norm_df: pd.DataFrame,
    test_mask: np.ndarray,
    model_dir: Path,
) -> np.ndarray:
    """Return model predictions (raw bid units) for the test rows: (n_test, n_targets)."""
    model = _load_model(generator_name, metadata, model_dir)
    features = norm_df.loc[test_mask, metadata["feature_columns"]].to_numpy(dtype=np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(features)).cpu().numpy()
    return preds


def _draw_block_panel(
    ax,
    gen: str,
    block_label: str,
    target_col: str,
    target_idx: int,
    raw_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    metadata: dict,
    model_dir: Path,
    x_feature: str,
    y_feature: str,
    max_leaf_nodes: int,
    decision_mode: str,
    test_size: float,
    random_state: int,
    show_legend: bool = True,
) -> dict[str, int]:
    """Draw the region shading + held-out test predictions for one block onto ax.

    Returns a small stats dict (counts of above / below / inside the label range).
    """
    x_all = raw_df[x_feature].to_numpy()
    y_all = raw_df[y_feature].to_numpy()

    # Region shading: fit the shallow residual tree on the FULL label set,
    # exactly as in visualize_training_labels.
    low_val, high_val = _classify_binary(raw_df[target_col])
    labels_all = (raw_df[target_col].to_numpy() > low_val + 1e-6).astype(int)
    clf = _fit_shallow_tree(x_all, y_all, labels_all, max_leaf_nodes, decision_mode)
    _draw_regions(ax, x_all, y_all, clf, decision_mode)

    # Held-out test rows and their predictions.
    test_mask = _test_row_mask(norm_df, test_size, random_state)
    preds = _predict_test_bids(gen, metadata, norm_df, test_mask, model_dir)
    pred_bid = preds[:, target_idx]

    x_test = x_all[test_mask]
    y_test = y_all[test_mask]

    # Split by whether the prediction falls outside the observed label range
    # [low_val, high_val], padded by RANGE_TOL.
    over = pred_bid > high_val + RANGE_TOL    # above the label range
    under = pred_bid < low_val - RANGE_TOL    # below the label range
    inside = ~over & ~under

    ax.scatter(x_test[over], y_test[over], s=POINT_SIZE, marker="^",
               c=COLOR_OVER, alpha=0.85, edgecolors="white", linewidths=0.4,
               zorder=5, label=LABEL_ABOVE)
    ax.scatter(x_test[inside], y_test[inside], s=ON_SIZE, c=COLOR_ON,
               alpha=0.6, linewidths=0, zorder=4,
               label=LABEL_INSIDE)
    ax.scatter(x_test[under], y_test[under], s=POINT_SIZE, marker="v",
               c=COLOR_UNDER, alpha=0.85, edgecolors="white", linewidths=0.4,
               zorder=5, label=LABEL_BELOW)

    n_test = int(test_mask.sum())
    n_over = int(over.sum())
    n_under = int(under.sum())
    n_inside = int(inside.sum())

    ax.set_xlabel(r"$D_t$ [MWh]", fontsize=12)
    ax.set_ylabel(r"$\sum_{i \in \mathcal{I}_W}\bar{P}_i$ [MWh]", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, zorder=2)
    if show_legend:
        # Legend below the axes so it does not cover data in the small two-up figure.
        ax.legend(
            fontsize=8, markerscale=1.1, framealpha=0.85,
            loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
        )
    return {"test": n_test, "above": n_over, "below": n_under, "inside": n_inside}


def _collect_panels(
    raw_dir: Path, norm_dir: Path, model_dir: Path
) -> list[tuple[str, str, str, int, pd.DataFrame, pd.DataFrame, dict]]:
    """One entry per (generator, varying target block) that has a usable model."""
    panels: list[tuple[str, str, str, int, pd.DataFrame, pd.DataFrame, dict]] = []
    for raw_path in sorted(raw_dir.glob("*_features.csv")):
        generator_name = raw_path.stem.removesuffix("_features")
        meta_path = model_dir / f"{generator_name}_policy_metadata.json"
        norm_path = norm_dir / f"{generator_name}_features_normalized.csv"
        if not meta_path.exists() or not norm_path.exists():
            continue  # generator was not trained as an NN policy

        raw_df = pd.read_csv(raw_path)
        norm_df = pd.read_csv(norm_path)
        metadata = json.loads(meta_path.read_text())

        missing = [c for c in metadata["feature_columns"] if c not in norm_df.columns]
        if missing:
            print(
                f"WARNING: skipping {generator_name} - stale model trained on features "
                f"not in current data: {missing}. Retrain to include it."
            )
            continue

        for target_col in _target_columns(raw_df):
            if not _has_variation(raw_df, target_col):
                continue
            target_idx = metadata["target_columns"].index(target_col)
            block_label = target_col.replace("target_bid_", "")
            panels.append(
                (generator_name, block_label, target_col, target_idx, raw_df, norm_df, metadata)
            )
    return panels


def plot_test_predictions(
    output_dir: Path = PLOT_DIR,
    raw_dir: Path = RAW_DATA_DIR,
    norm_dir: Path = NORM_DATA_DIR,
    model_dir: Path = MODEL_DIR,
    x_feature: str = X_FEATURE,
    y_feature: str = Y_FEATURE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    decision_mode: str = DECISION_MODE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    show_legend: bool = False,
) -> list[Path]:
    """Save one PNG per (generator, block) into output_dir. Returns saved paths.

    Legends are omitted by default; use the standalone legend from save_legend()
    placed once under a row of panels instead.
    """
    panels = _collect_panels(raw_dir, norm_dir, model_dir)
    if not panels:
        print("No trained NN generators with varying labels found.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for gen, block_label, target_col, target_idx, raw_df, norm_df, meta in panels:
        # Sized for ~half the A4 text width so two figures sit side by side and
        # the text stays readable when included at ~0.48\textwidth each.
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        stats = _draw_block_panel(
            ax, gen, block_label, target_col, target_idx, raw_df, norm_df, meta,
            model_dir, x_feature, y_feature, max_leaf_nodes, decision_mode,
            test_size, random_state, show_legend=show_legend,
        )
        fig.tight_layout()
        out_path = output_dir / f"{gen}_{block_label}_test_predictions.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        print(
            f"{gen} - {block_label}: test={stats['test']}, above-range={stats['above']}, "
            f"below-range={stats['below']}, inside-range={stats['inside']} -> {out_path}"
        )

    print(f"\nSaved {len(saved_paths)} per-block plots to {output_dir}")
    return saved_paths


def _legend_handles() -> list:
    """Proxy handles matching the test-prediction scatter markers/labels."""
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], linestyle="none", marker="^", markersize=8,
               markerfacecolor=COLOR_OVER, markeredgecolor="white", markeredgewidth=0.4,
               label=LABEL_ABOVE),
        Line2D([], [], linestyle="none", marker="o", markersize=7,
               markerfacecolor=COLOR_ON, markeredgecolor="none", label=LABEL_INSIDE),
        Line2D([], [], linestyle="none", marker="v", markersize=8,
               markerfacecolor=COLOR_UNDER, markeredgecolor="white", markeredgewidth=0.4,
               label=LABEL_BELOW),
    ]


def plot_panels_side_by_side(
    panel_keys: list[tuple[str, str]],
    output_path: Path,
    raw_dir: Path = RAW_DATA_DIR,
    norm_dir: Path = NORM_DATA_DIR,
    model_dir: Path = MODEL_DIR,
    x_feature: str = X_FEATURE,
    y_feature: str = Y_FEATURE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    decision_mode: str = DECISION_MODE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    show_legend: bool = True,
    titles: list[str] | None = None,
) -> Path:
    """Draw the given (generator, block) test-prediction panels in one figure.

    Panels are placed side by side sharing a single y-axis (only the left panel
    carries the y-axis label and tick labels). A single shared legend is drawn
    centred underneath both panels. Optional per-panel titles may be supplied.
    """
    panels = _collect_panels(raw_dir, norm_dir, model_dir)
    by_key = {(rec[0], rec[1]): rec for rec in panels}
    missing = [k for k in panel_keys if k not in by_key]
    if missing:
        raise KeyError(f"Panels not found: {missing}. Available: {sorted(by_key)}")
    selected = [by_key[k] for k in panel_keys]

    n = len(selected)
    fig_h = 3.9 if show_legend else 3.4
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, fig_h), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (gen, block_label, target_col, target_idx, raw_df, norm_df, meta) in zip(axes, selected):
        _draw_block_panel(
            ax, gen, block_label, target_col, target_idx, raw_df, norm_df, meta,
            model_dir, x_feature, y_feature, max_leaf_nodes, decision_mode,
            test_size, random_state, show_legend=False,
        )
    for ax in axes[1:]:
        ax.set_ylabel("")  # single shared y-axis label on the left only

    if titles is not None:
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=12)

    if show_legend:
        # Reserve space at the bottom and place one shared legend under both panels.
        fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
        fig.legend(
            handles=_legend_handles(), loc="lower center",
            bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=True,
            fontsize=9, handletextpad=0.4, columnspacing=1.4,
        )
    else:
        fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved side-by-side panels to {output_path}")
    return output_path


def save_legend(output_path: Path, ncol: int = 3, fontsize: int = 11) -> Path:
    """Save a standalone, tight-cropped legend (no axes) for the test-prediction plots.

    Intended to be placed once underneath a row of side-by-side panels in LaTeX,
    so the individual panels can omit their own legend.
    """
    handles = _legend_handles()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.0, 0.6))
    leg = fig.legend(
        handles=handles, loc="center", ncol=ncol, frameon=True,
        fontsize=fontsize, handletextpad=0.4, columnspacing=1.4,
    )
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output_path, bbox_inches=bbox)
    plt.close(fig)
    print(f"Saved standalone legend to {output_path}")
    return output_path


def main() -> None:
    plot_test_predictions(PLOT_DIR)


if __name__ == "__main__":
    main()
