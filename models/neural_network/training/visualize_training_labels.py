from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_DATA_DIR = Path("models/neural_network/features/generated/raw")
PLOT_DIR = Path("models/neural_network/training/training_results/plots")

X_FEATURE = "demand"
Y_FEATURE = "total_wind_generation_capacity"

# Complexity of the splitting rule. max_leaf_nodes = number of regions;
# with 3 leaves you get at most 2 decision boundaries (splits).
MAX_LEAF_NODES = 3

# How the tree splits the plane:
#   "residual" -> split on (demand - wind); boundaries are clean 45-deg diagonals
#                 so the high-bid region is a diagonal band.
#   "xy"       -> split on demand and wind separately; axis-aligned rectangles.
DECISION_MODE = "residual"

COLOR_LOW = "#2563EB"
COLOR_HIGH = "#DC2626"
REGION_LOW = "#DBEAFE"
REGION_HIGH = "#FEE2E2"
ALPHA_POINTS = 0.6
POINT_SIZE = 14
MESH_RESOLUTION = 400


def load_all_raw_datasets(raw_dir: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_dir.glob("*_features.csv")):
        generator_name = path.stem.removesuffix("_features")
        datasets[generator_name] = pd.read_csv(path)
    return datasets


def _target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("target_bid_")]


def _has_variation(df: pd.DataFrame, col: str, tol: float = 1e-6) -> bool:
    return float(df[col].std()) > tol


def _classify_binary(series: pd.Series) -> tuple[float, float]:
    vals = sorted(series.unique())
    if len(vals) != 2:
        raise ValueError(f"Expected exactly 2 unique label values, got {len(vals)}: {vals}")
    return float(vals[0]), float(vals[1])


def _decision_matrix(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    """Build the feature matrix the tree splits on, for the given mode."""
    if mode == "residual":
        return (x - y).reshape(-1, 1)
    if mode == "xy":
        return np.column_stack([x, y])
    raise ValueError(f"Unknown decision mode: {mode!r} (use 'residual' or 'xy')")


def _decision_feature_names(x_feature: str, y_feature: str, mode: str) -> list[str]:
    if mode == "residual":
        return ["residual_demand (demand - wind)"]
    return [x_feature, y_feature]


def _fit_shallow_tree(
    x: np.ndarray,
    y: np.ndarray,
    labels_binary: np.ndarray,
    max_leaf_nodes: int,
    mode: str,
) -> DecisionTreeClassifier:
    """Fit a shallow decision tree (few splits) on the chosen decision feature(s)."""
    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    clf.fit(_decision_matrix(x, y, mode), labels_binary)
    return clf


def _draw_regions(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    clf: DecisionTreeClassifier,
    mode: str,
) -> None:
    """Shade background regions predicted by the shallow tree."""
    pad_x = 0.05 * (x.max() - x.min())
    pad_y = 0.05 * (y.max() - y.min())
    x1, x2 = x.min() - pad_x, x.max() + pad_x
    y1, y2 = y.min() - pad_y, y.max() + pad_y

    xg, yg = np.meshgrid(
        np.linspace(x1, x2, MESH_RESOLUTION),
        np.linspace(y1, y2, MESH_RESOLUTION),
    )
    features = _decision_matrix(xg.ravel(), yg.ravel(), mode)
    zz = clf.predict(features).reshape(xg.shape)
    ax.contourf(xg, yg, zz, levels=[-0.5, 0.5, 1.5],
                colors=[REGION_LOW, REGION_HIGH], alpha=1.0)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)


def _collect_label_panels(
    datasets: dict[str, pd.DataFrame],
) -> list[tuple[str, str, pd.DataFrame, str]]:
    """One entry per (generator, varying target block)."""
    panels: list[tuple[str, str, pd.DataFrame, str]] = []
    for gen_name, df in sorted(datasets.items()):
        for col in _target_columns(df):
            if _has_variation(df, col):
                block_label = col.replace("target_bid_", "")
                panels.append((gen_name, block_label, df, col))
    return panels


def _draw_label_panel(
    ax,
    gen_name: str,
    block_label: str,
    df: pd.DataFrame,
    target_col: str,
    x_feature: str,
    y_feature: str,
    max_leaf_nodes: int,
    shade_regions: bool,
    decision_mode: str,
) -> float:
    """Draw one block's label scatter (and optional regions) onto ax.

    Returns the in-sample fit accuracy of the shallow region tree.
    """
    low_val, high_val = _classify_binary(df[target_col])
    mask_high = df[target_col].values > low_val + 1e-6
    labels_binary = mask_high.astype(int)

    x = df[x_feature].values
    y = df[y_feature].values

    clf = _fit_shallow_tree(x, y, labels_binary, max_leaf_nodes, decision_mode)

    # Report the learned rule and accuracy for this panel
    accuracy = float(
        (clf.predict(_decision_matrix(x, y, decision_mode)) == labels_binary).mean()
    )
    rules = export_text(
        clf, feature_names=_decision_feature_names(x_feature, y_feature, decision_mode)
    ).strip()
    print(f"\n{gen_name} - {block_label}  (train accuracy = {accuracy:.1%})")
    print(rules)

    if shade_regions:
        _draw_regions(ax, x, y, clf, decision_mode)
    else:
        pad_x = 0.05 * (x.max() - x.min())
        pad_y = 0.05 * (y.max() - y.min())
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    ax.scatter(
        x[mask_high], y[mask_high],
        color=COLOR_HIGH, s=POINT_SIZE, alpha=ALPHA_POINTS, linewidths=0,
        label=f"high bid ({high_val:.4g})", zorder=5,
    )
    ax.scatter(
        x[~mask_high], y[~mask_high],
        color=COLOR_LOW, s=POINT_SIZE, alpha=ALPHA_POINTS, linewidths=0,
        label=f"low bid ({low_val:.4g})", zorder=5,
    )

    n_high = int(mask_high.sum())
    n_low = int((~mask_high).sum())
    pct_high = 100.0 * n_high / (n_low + n_high)

    ax.set_xlabel(x_feature.replace("_", " ") + " (MWh)")
    ax.set_ylabel(y_feature.replace("_", " ") + " (MWh)")
    ax.set_title(
        f"{gen_name} - {block_label}   (fit acc {accuracy:.0%})\n"
        f"high: {n_high} ({pct_high:.0f}%)   low: {n_low} ({100 - pct_high:.0f}%)"
    )
    ax.grid(True, alpha=0.2, zorder=2)
    ax.legend(fontsize=8, markerscale=1.5, framealpha=0.8)
    return accuracy


def plot_label_scatter(
    datasets: dict[str, pd.DataFrame],
    output_path: Path,
    x_feature: str = X_FEATURE,
    y_feature: str = Y_FEATURE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    shade_regions: bool = True,
    decision_mode: str = DECISION_MODE,
) -> None:
    """Combined grid of demand vs wind label scatters with region shading.

    A small number of splits (max_leaf_nodes regions) carve the (demand, wind)
    plane into high-bid (red) and low-bid (blue) regions. With
    decision_mode="residual" the tree splits on (demand - wind), so the
    boundaries are 45-deg diagonals and the high-bid region is a diagonal band.
    Set shade_regions=False to plot only the points with no background shading.
    """
    panels = _collect_label_panels(datasets)
    if not panels:
        print("No generators with varying labels found.")
        return

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (gen_name, block_label, df, target_col) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        _draw_label_panel(
            ax, gen_name, block_label, df, target_col, x_feature, y_feature,
            max_leaf_nodes, shade_regions, decision_mode,
        )

    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Training labels - merit-order heuristic bids\n"
        f"x = {x_feature.replace('_', ' ')},  y = {y_feature.replace('_', ' ')}  "
        f"(<= {max_leaf_nodes} regions)",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {output_path}")


def plot_label_scatter_individual(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path,
    x_feature: str = X_FEATURE,
    y_feature: str = Y_FEATURE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    shade_regions: bool = True,
    decision_mode: str = DECISION_MODE,
) -> list[Path]:
    """Save one PNG per (generator, block) into output_dir. Returns saved paths."""
    panels = _collect_label_panels(datasets)
    if not panels:
        print("No generators with varying labels found.")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "labels" if shade_regions else "labels_no_regions"

    saved_paths: list[Path] = []
    for gen_name, block_label, df, target_col in panels:
        fig, ax = plt.subplots(figsize=(6.0, 5.4))
        _draw_label_panel(
            ax, gen_name, block_label, df, target_col, x_feature, y_feature,
            max_leaf_nodes, shade_regions, decision_mode,
        )
        fig.tight_layout()
        out_path = output_dir / f"{gen_name}_{block_label}_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"Saved {out_path}")

    print(f"\nSaved {len(saved_paths)} per-block label plots to {output_dir}")
    return saved_paths


def main(
    raw_dir: Path = RAW_DATA_DIR,
    plot_dir: Path = PLOT_DIR,
    x_feature: str = X_FEATURE,
    y_feature: str = Y_FEATURE,
    max_leaf_nodes: int = MAX_LEAF_NODES,
    decision_mode: str = DECISION_MODE,
) -> None:
    datasets = load_all_raw_datasets(raw_dir)
    if not datasets:
        raise FileNotFoundError(f"No *_features.csv files found in {raw_dir}")

    scatter_path = plot_dir / "training_labels_scatter.png"
    plot_label_scatter(
        datasets, scatter_path,
        x_feature=x_feature, y_feature=y_feature, max_leaf_nodes=max_leaf_nodes,
        shade_regions=True, decision_mode=decision_mode,
    )

    plain_path = plot_dir / "training_labels_scatter_no_regions.png"
    plot_label_scatter(
        datasets, plain_path,
        x_feature=x_feature, y_feature=y_feature, max_leaf_nodes=max_leaf_nodes,
        shade_regions=False, decision_mode=decision_mode,
    )

    # Per-block PNGs (with regions) under plot_dir/training_labels.
    plot_label_scatter_individual(
        datasets, plot_dir / "training_labels",
        x_feature=x_feature, y_feature=y_feature, max_leaf_nodes=max_leaf_nodes,
        shade_regions=True, decision_mode=decision_mode,
    )


if __name__ == "__main__":
    main()
