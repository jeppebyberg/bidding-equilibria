from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULT_DIR = Path("models/neural_network/training/policy_design_cv_results")
SUMMARY_JSON_PATH = RESULT_DIR / "policy_design_cv_summary.json"
BEST_JSON_PATH = RESULT_DIR / "best_policy_design_by_generator.json"
PLOT_DIR = RESULT_DIR / "plots"


def main() -> None:
    """Create plots from saved policy-design cross-validation results."""
    create_policy_design_cv_plots(
        summary_json_path=SUMMARY_JSON_PATH,
        best_json_path=BEST_JSON_PATH,
        plot_dir=PLOT_DIR,
    )


def create_policy_design_cv_plots(
    summary_json_path: Path,
    best_json_path: Path,
    plot_dir: Path,
) -> None:
    """Generate summary and per-generator policy-design CV plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = read_json(summary_json_path)
    best_by_generator = read_json(best_json_path)
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise ValueError(f"No policy-design CV rows found in {summary_json_path}")

    valid_df = dataframe[dataframe["status"] == "valid"].copy()
    if valid_df.empty:
        raise ValueError(f"No valid policy-design CV rows found in {summary_json_path}")
    valid_df["architecture"] = valid_df["hidden_layers"].apply(format_hidden_layers)

    plot_best_mean_validation_loss(
        best_by_generator=best_by_generator,
        output_path=plot_dir / "best_mean_validation_loss_by_generator.png",
    )
    plot_mean_validation_loss_by_architecture(
        dataframe=valid_df,
        output_path=plot_dir / "mean_validation_loss_by_architecture.png",
    )
    plot_mean_validation_loss_by_feature_set(
        dataframe=valid_df,
        output_path=plot_dir / "mean_validation_loss_by_feature_set.png",
    )
    for generator_name, generator_df in valid_df.groupby("generator_name"):
        plot_generator_policy_design_grid(
            generator_name=str(generator_name),
            dataframe=generator_df,
            output_path=plot_dir / f"{generator_name}_policy_design_grid.png",
        )


def plot_best_mean_validation_loss(
    best_by_generator: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot the selected best mean validation loss for each generator."""
    generator_names = sorted(best_by_generator)
    losses = [
        float(best_by_generator[generator_name]["mean_validation_loss"])
        for generator_name in generator_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(generator_names, losses, color="#3A6EA5")
    ax.set_title("Best mean validation loss by generator")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Mean validation MSE")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(losses):
        ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mean_validation_loss_by_architecture(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot average validation loss grouped by hidden-layer architecture."""
    grouped = (
        dataframe.groupby("architecture")["mean_validation_loss"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(grouped.index, grouped.values, color="#5B8E7D")
    ax.set_title("Mean validation loss by architecture")
    ax.set_xlabel("Hidden layers")
    ax.set_ylabel("Mean validation MSE")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mean_validation_loss_by_feature_set(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot average validation loss grouped by feature-set name."""
    grouped = (
        dataframe.groupby("feature_set_name")["mean_validation_loss"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index, grouped.values, color="#A66E5C")
    ax.set_title("Mean validation loss by feature set")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Mean validation MSE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_generator_policy_design_grid(
    generator_name: str,
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a heatmap-like architecture and feature-set comparison plot."""
    pivot = dataframe.pivot_table(
        index="feature_set_name",
        columns="architecture",
        values="mean_validation_loss",
        aggfunc="mean",
    )
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    values = pivot.to_numpy(dtype=float)

    fig_width = max(8, 0.8 * len(pivot.columns))
    fig_height = max(4, 0.6 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_title(f"{generator_name} validation loss by design")
    ax.set_xlabel("Hidden layers")
    ax.set_ylabel("Feature set")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                label = ""
            else:
                label = f"{value:.3g}"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean validation MSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def format_hidden_layers(hidden_layers: list[int]) -> str:
    """Format a hidden-layer list for compact plot labels."""
    return "x".join(str(width) for width in hidden_layers)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    if not path.exists():
        raise ValueError(f"Missing policy-design CV result file: {path}")
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


if __name__ == "__main__":
    main()
