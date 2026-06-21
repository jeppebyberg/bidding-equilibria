# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-04
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.neural_network.training.trainer import (
    BiddingPolicyTrainingConfig,
    train_generator_policy,
)

FEATURE_DIR = Path("models/neural_network/features/generated/normalized")
MODEL_DIR = Path("models/neural_network/training/trained_models")
RESULT_DIR = Path("models/neural_network/training/training_results")

HIDDEN_LAYERS = [11, 11]
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 500
WEIGHT_DECAY = 0.0
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42
PATIENCE = 50
MIN_DELTA = 1e-6
DEVICE: str | None = None
FINAL_ACTIVATION = "linear"
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 20
LR_SCHEDULER_MIN_LR = 1e-6

def main(device: str | None = DEVICE) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    config = BiddingPolicyTrainingConfig(
        hidden_layers=HIDDEN_LAYERS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        device=device,
        final_activation=FINAL_ACTIVATION,
        use_lr_scheduler=USE_LR_SCHEDULER,
        lr_scheduler_factor=LR_SCHEDULER_FACTOR,
        lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
        lr_scheduler_min_lr=LR_SCHEDULER_MIN_LR,
    )

    csv_paths = _find_generator_feature_files(FEATURE_DIR)
    if not csv_paths:
        raise ValueError(f"No normalized generator feature CSVs found in {FEATURE_DIR}")

    summary_entries = []
    for csv_path in csv_paths:
        result = train_generator_policy(
            csv_path=csv_path,
            model_dir=MODEL_DIR,
            result_dir=RESULT_DIR,
            config=config,
        )
        policy_data = result["policy_data"]
        history = result["history"]
        summary_entries.append(result["summary"])

        print(
            f"{policy_data.generator_name}: "
            f"rows={policy_data.num_rows}, "
            f"features={policy_data.input_dim}, "
            f"targets={policy_data.output_dim}, "
            f"train_scenarios={len(policy_data.train_scenarios)}, "
            f"val_scenarios={len(policy_data.val_scenarios)}, "
            f"test_scenarios={len(policy_data.test_scenarios)}, "
            f"best_val_loss={history['best_val_loss']:.8g}, "
            f"final_test_loss={history['final_test_loss']:.8g}, "
            f"model={result['model_path']}"
        )

    summary_path = RESULT_DIR / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary_entries, file_handle, indent=2)
    print(f"Saved training summary to {summary_path}")

def _find_generator_feature_files(feature_dir: Path) -> list[Path]:
    if not feature_dir.exists():
        raise ValueError(f"Feature directory does not exist: {feature_dir}")
    return sorted(
        path
        for path in feature_dir.glob("*_features_normalized.csv")
        if path.is_file()
    )


if __name__ == "__main__":
    main(device=DEVICE)
