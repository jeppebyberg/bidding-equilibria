from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
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

    combined_loss_path = plot_dir / "all_generators_loss_curves.png"
    plot_all_loss_curves(histories, combined_loss_path, log_scale=log_scale)
    print(f"Saved {combined_loss_path}")

    best_loss_path = plot_dir / "best_test_loss_by_generator.png"
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
    test_loss = history["test_loss"]
    epochs = list(range(1, len(train_loss) + 1))
    best_epoch = int(history["best_epoch"])
    best_test_loss = float(history["best_test_loss"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=1.8)
    ax.plot(epochs, test_loss, label="Test loss", linewidth=1.8)
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1.0)
    ax.scatter(
        [best_epoch],
        [best_test_loss],
        color="black",
        s=35,
        zorder=3,
        label=f"Best test: {best_test_loss:.4g}",
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


def plot_all_loss_curves(
    histories: dict[str, dict[str, Any]],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for generator_name, history in sorted(histories.items()):
        test_loss = history["test_loss"]
        epochs = list(range(1, len(test_loss) + 1))
        ax.plot(epochs, test_loss, label=generator_name, linewidth=1.7)

    ax.set_title("Test loss by generator")
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
        float(histories[generator_name]["best_test_loss"])
        for generator_name in generator_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(generator_names, best_losses, color="#3A6EA5")
    ax.set_title("Best test loss by generator")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Best MSE loss")
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
                f"{float(history['best_test_loss']):.4g}",
            ]
        )

    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.45 * max(len(rows), 1)))
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
            "Best test loss",
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


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Visualize neural-network bidding policy training results."
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use a logarithmic y-axis for loss plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_dir=args.model_dir,
        result_dir=args.result_dir,
        plot_dir=args.plot_dir,
        log_scale=args.log_scale,
    )
