from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_DIR = Path("models/neural_network/training/trained_models")
RESULT_DIR = Path("models/neural_network/training/training_results")
PLOT_DIR = RESULT_DIR / "plots"
LOG_SCALE = False

def main(
    model_dir: Path = MODEL_DIR,
    result_dir: Path = RESULT_DIR,
    plot_dir: Path = PLOT_DIR,
    log_scale: bool = False,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    histories = _load_histories(result_dir)
    metadata = _load_metadata(model_dir)
    if not histories:
        raise ValueError(f"No *_training_history.json files found in {result_dir}")

    for generator_name, history in histories.items():
        metadata_for_generator = metadata.get(generator_name, {})
        path = plot_dir / f"{generator_name}_loss_curve.png"
        plot_loss_curve(
            generator_name=generator_name,
            history=history,
            metadata=metadata_for_generator,
            output_path=path,
            log_scale=log_scale,
        )
        print(f"Saved {path}")

        val_vs_test_path = plot_dir / f"{generator_name}_val_vs_test.png"
        plot_val_vs_test_curve(
            generator_name=generator_name,
            history=history,
            metadata=metadata_for_generator,
            output_path=val_vs_test_path,
            log_scale=log_scale,
        )
        print(f"Saved {val_vs_test_path}")

    combined_loss_path = plot_dir / "all_generators_loss_curves.png"
    plot_all_loss_curves(histories, combined_loss_path, log_scale=log_scale)
    print(f"Saved {combined_loss_path}")

    combined_test_path = plot_dir / "all_generators_test_error.png"
    plot_all_test_errors(histories, combined_test_path, log_scale=log_scale)
    print(f"Saved {combined_test_path}")

    best_loss_path = plot_dir / "best_val_loss_by_generator.png"
    plot_best_test_loss_by_generator(histories, best_loss_path, log_scale=log_scale)
    print(f"Saved {best_loss_path}")

    summary_path = plot_dir / "model_summary.png"
    plot_model_summary(metadata, histories, summary_path)
    print(f"Saved {summary_path}")

def plot_loss_curve(
    generator_name: str,
    history: dict[str, Any],
    metadata: dict[str, Any],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = list(range(1, len(train_loss) + 1))
    best_epoch = int(history["best_epoch"])
    best_val_loss = float(history["best_val_loss"])
    final_test_loss = float(history.get("final_test_loss", float("nan")))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=1.8)
    ax.plot(epochs, val_loss, label="Val loss", linewidth=1.8)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.0)
    ax.scatter(
        [best_epoch],
        [best_val_loss],
        color="black",
        s=35,
        zorder=3,
        label=f"Best val: {best_val_loss:.4g} | Test: {final_test_loss:.4g}",
    )

    title = f"{generator_name} training loss"
    if metadata:
        title += (
            f" | {metadata.get('input_dim', '?')} inputs -> "
            f"{metadata.get('output_dim', '?')} bids"
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_val_vs_test_curve(
    generator_name: str,
    history: dict[str, Any],
    metadata: dict[str, Any],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    """Per-generator validation error vs held-out test error.

    The validation loss is a per-epoch curve (it drives early stopping); the
    held-out test loss is a single value evaluated once at the best-val
    checkpoint, drawn as a horizontal reference line.
    """
    val_loss = history["val_loss"]
    epochs = list(range(1, len(val_loss) + 1))
    best_epoch = int(history["best_epoch"])
    best_val_loss = float(history["best_val_loss"])
    final_test_loss = float(history.get("final_test_loss", float("nan")))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, val_loss, color="#C2410C", label="Validation loss", linewidth=1.8)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.0)
    ax.scatter(
        [best_epoch],
        [best_val_loss],
        color="black",
        s=35,
        zorder=3,
        label=f"Best val: {best_val_loss:.4g} (epoch {best_epoch})",
    )
    ax.axhline(
        final_test_loss,
        color="#1D4ED8",
        linestyle=":",
        linewidth=1.8,
        label=f"Held-out test: {final_test_loss:.4g}",
    )

    gap = best_val_loss - final_test_loss
    title = f"{generator_name} validation vs test error  (val - test = {gap:+.4g})"
    if metadata:
        title += (
            f"\n{metadata.get('input_dim', '?')} inputs -> "
            f"{metadata.get('output_dim', '?')} bids"
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_all_test_errors(
    histories: dict[str, dict[str, Any]],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    """Combined bar chart of every generator's held-out test error.

    Test error is a single value per generator (evaluated once at the best-val
    checkpoint), so the natural cross-generator comparison is a bar chart.
    """
    generator_names = sorted(histories)
    test_losses = [
        float(histories[generator_name].get("final_test_loss", float("nan")))
        for generator_name in generator_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(generator_names, test_losses, color="#1D4ED8")
    ax.set_title("Held-out test error by generator")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Final test MSE loss")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(test_losses):
        ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_all_loss_curves(
    histories: dict[str, dict[str, Any]],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for generator_name, history in sorted(histories.items()):
        val_loss = history["val_loss"]
        epochs = list(range(1, len(val_loss) + 1))
        ax.plot(epochs, val_loss, label=generator_name, linewidth=1.7)

    ax.set_title("Val loss by generator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_best_test_loss_by_generator(
    histories: dict[str, dict[str, Any]],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    generator_names = sorted(histories)
    best_losses = [
        float(histories[generator_name]["best_val_loss"])
        for generator_name in generator_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(generator_names, best_losses, color="#3A6EA5")
    ax.set_title("Best val loss by generator")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Best MSE loss (val)")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(best_losses):
        ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_model_summary(
    metadata: dict[str, dict[str, Any]],
    histories: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    generator_names = sorted(histories)
    rows = []
    for generator_name in generator_names:
        metadata_for_generator = metadata.get(generator_name, {})
        history = histories[generator_name]
        hidden_layers = metadata_for_generator.get("hidden_layers", [])
        rows.append(
            [
                generator_name,
                str(metadata_for_generator.get("input_dim", "?")),
                str(metadata_for_generator.get("output_dim", "?")),
                " x ".join(str(width) for width in hidden_layers) or "?",
                str(len(history["train_loss"])),
                str(history["best_epoch"]),
                f"{float(history['best_val_loss']):.4g}",
                f"{float(history.get('final_test_loss', float('nan'))):.4g}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.45 * max(len(rows), 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Generator",
            "Inputs",
            "Outputs",
            "Hidden layers",
            "Epochs",
            "Best epoch",
            "Best val loss",
            "Final test loss",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E9EEF5")
        else:
            cell.set_facecolor("#FFFFFF" if row_index % 2 else "#F6F8FA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def _load_histories(result_dir: Path) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for path in sorted(result_dir.glob("*_training_history.json")):
        generator_name = path.name.removesuffix("_training_history.json")
        histories[generator_name] = _read_json(path)
    return histories

def _load_metadata(model_dir: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.glob("*_policy_metadata.json")):
        generator_name = path.name.removesuffix("_policy_metadata.json")
        metadata[generator_name] = _read_json(path)
    return metadata

def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)

if __name__ == "__main__":
    main(
        model_dir=MODEL_DIR,
        result_dir=RESULT_DIR,
        plot_dir=PLOT_DIR,
        log_scale=LOG_SCALE,
    )