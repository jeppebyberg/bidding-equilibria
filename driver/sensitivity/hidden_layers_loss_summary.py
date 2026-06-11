"""Summarize validation and test losses across the hidden_layers_sweep models.

Reads each run's training_summary.json (per generator: best_val_loss and
final_test_loss) plus the policy metadata (hidden_layers), then writes a
cross-model comparison as CSV + JSON and a grouped bar chart.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.hidden_layers_loss_summary
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.sensitivity.hidden_layers_sweep import (  # noqa: E402
    HIDDEN_LAYER_SETUPS,
    STUDY_NAME,
    run_label,
    run_name,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "sensitivity_studies" / STUDY_NAME


def _arch_label(hidden_layers: list[int]) -> str:
    return "[" + ", ".join(str(n) for n in hidden_layers) + "]"


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    """Return one run's per-generator and aggregate losses, or None if missing."""
    summary_path = run_dir / "training_results" / "training_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    generators: list[dict[str, Any]] = []
    for rec in records:
        name = rec["generator_name"]
        hidden_layers = None
        meta_path = run_dir / "trained_models" / f"{name}_policy_metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                hidden_layers = json.load(fh).get("hidden_layers")
        generators.append(
            {
                "generator_name": name,
                "hidden_layers": hidden_layers,
                "best_val_loss": rec.get("best_val_loss"),
                "final_test_loss": rec.get("final_test_loss"),
                "best_epoch": rec.get("best_epoch"),
                "training_time_seconds": rec.get("training_time_seconds"),
            }
        )

    val_losses = [g["best_val_loss"] for g in generators if g["best_val_loss"] is not None]
    test_losses = [g["final_test_loss"] for g in generators if g["final_test_loss"] is not None]
    return {
        "generators": generators,
        "mean_best_val_loss": (sum(val_losses) / len(val_losses)) if val_losses else None,
        "mean_final_test_loss": (sum(test_losses) / len(test_losses)) if test_losses else None,
        "total_training_time_seconds": sum(
            g["training_time_seconds"] for g in generators if g["training_time_seconds"]
        ),
    }


def collect_summary() -> dict[str, Any]:
    """Build the cross-model summary, preserving HIDDEN_LAYER_SETUPS order."""
    models: list[dict[str, Any]] = []
    for layers in HIDDEN_LAYER_SETUPS:
        name = run_name(layers)
        run_dir = RESULT_ROOT / name
        loaded = _load_run(run_dir)
        if loaded is None:
            print(f"  [skip] {name}: no training_summary.json found")
            continue
        models.append(
            {
                "run_name": name,
                "architecture": run_label(layers),
                "hidden_layers": list(layers),
                **loaded,
            }
        )
    return {"study": STUDY_NAME, "models": models}


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in summary["models"]:
        for gen in model["generators"]:
            rows.append(
                {
                    "architecture": model["architecture"],
                    "run_name": model["run_name"],
                    "generator": gen["generator_name"],
                    "best_val_loss": gen["best_val_loss"],
                    "final_test_loss": gen["final_test_loss"],
                    "best_epoch": gen["best_epoch"],
                }
            )
        rows.append(
            {
                "architecture": model["architecture"],
                "run_name": model["run_name"],
                "generator": "MEAN",
                "best_val_loss": model["mean_best_val_loss"],
                "final_test_loss": model["mean_final_test_loss"],
                "best_epoch": "",
            }
        )
    fieldnames = ["architecture", "run_name", "generator", "best_val_loss", "final_test_loss", "best_epoch"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary: dict[str, Any]) -> None:
    generators = sorted({g["generator_name"] for m in summary["models"] for g in m["generators"]})
    header = f"{'architecture':<14}"
    for name in generators:
        header += f"{name + ' val':>12}{name + ' test':>12}"
    header += f"{'mean val':>12}{'mean test':>12}"
    print("\n" + header)
    print("-" * len(header))
    for model in summary["models"]:
        by_gen = {g["generator_name"]: g for g in model["generators"]}
        row = f"{model['architecture']:<14}"
        for name in generators:
            g = by_gen.get(name, {})
            v = g.get("best_val_loss")
            t = g.get("final_test_loss")
            row += f"{(f'{v:.4f}' if v is not None else '-'):>12}"
            row += f"{(f'{t:.4f}' if t is not None else '-'):>12}"
        mv = model["mean_best_val_loss"]
        mt = model["mean_final_test_loss"]
        row += f"{(f'{mv:.4f}' if mv is not None else '-'):>12}"
        row += f"{(f'{mt:.4f}' if mt is not None else '-'):>12}"
        print(row)
    print()


def plot_summary(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = summary["models"]
    if not models:
        return
    labels = [m["architecture"] for m in models]
    mean_val = [m["mean_best_val_loss"] for m in models]
    mean_test = [m["mean_final_test_loss"] for m in models]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(labels)), 5))
    ax.bar(x - width / 2, mean_val, width, label="mean best val loss")
    ax.bar(x + width / 2, mean_test, width, label="mean final test loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("hidden layers")
    ax.set_ylabel("loss (mean over generators)")
    ax.set_title("Validation vs test loss across hidden-layer architectures")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Hidden-layer loss summary for study '{STUDY_NAME}'")
    summary = collect_summary()
    if not summary["models"]:
        print("No trained models found. Run the sweep first.")
        return

    print_table(summary)

    out_dir = RESULT_ROOT
    json_path = out_dir / "loss_summary.json"
    csv_path = out_dir / "loss_summary.csv"
    plot_path = out_dir / "loss_summary.png"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_csv(summary, csv_path)
    try:
        plot_summary(summary, plot_path)
        plotted = str(plot_path)
    except Exception as exc:  # plotting is optional
        plotted = f"(skipped: {exc})"

    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {plotted}")


if __name__ == "__main__":
    main()
