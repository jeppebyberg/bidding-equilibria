from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.neural_network.training.dataset import (
    identify_feature_columns,
    identify_target_columns,
)
from models.neural_network.training.model import BiddingPolicyNetwork
from models.neural_network.training.visualize_policy_design_cv import (
    create_policy_design_cv_plots,
)


FEATURE_DIR = Path("models/neural_network/features/generated/normalized")
RESULT_DIR = Path("models/neural_network/training/policy_design_cv_results")

HIDDEN_LAYER_GRID = [
    [4, 4],
    [8, 8],
    [11, 11],
    [8, 4],
    [4, 8],
    [4, 4, 4],
]

SUPPORTED_FEATURE_COLUMNS = [
    "demand",
    "total_wind_generation_capacity",
    "total_generation_capacity",
    "residual_demand",
    "previous_generation_capacity",
    "previous_demand",
    "next_generation_capacity",
    "next_demand",
    "own_generation_capacity",
    "previous_own_generation_capacity",
    "next_own_generation_capacity",
]

FEATURE_SETS = {
    "temporal_system_state": [
        "demand",
        "residual_demand",
        "previous_generation_capacity",
        "previous_demand",
        "next_generation_capacity",
        "next_demand",
    ],
    "compact": [
        "demand",
        "total_wind_generation_capacity",
        "residual_demand",
        "next_generation_capacity",
        "next_demand",
    ],
    "full_supported": SUPPORTED_FEATURE_COLUMNS,
}

K_FOLDS = 3
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 500
WEIGHT_DECAY = 0.0
PATIENCE = 50
MIN_DELTA = 1e-6
DEVICE: str | None = None
RANDOM_STATE = 42
SAVE_FOLD_MODELS = False
CREATE_PLOTS = True

@dataclass(frozen=True)
class FoldResult:
    validation_loss: float
    best_epoch: int
    train_size: int
    validation_size: int

def main() -> None:
    """Run cross-validation over policy architecture and feature-set choices."""
    set_reproducible_seeds(RANDOM_STATE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = find_generator_feature_files(FEATURE_DIR)
    if not csv_paths:
        raise ValueError(f"No normalized generator feature CSVs found in {FEATURE_DIR}")

    all_rows: list[dict[str, Any]] = []
    best_by_generator: dict[str, dict[str, Any]] = {}

    for csv_path in csv_paths:
        print(f"Cross-validating {csv_path}...")
        generator_rows = evaluate_generator_csv(csv_path)
        generator_name = str(generator_rows[0]["generator_name"])
        valid_rows = [row for row in generator_rows if row["status"] == "valid"]
        if not valid_rows:
            write_json(
                RESULT_DIR / f"{generator_name}_policy_design_cv.json",
                {"generator_name": generator_name, "results": generator_rows},
            )
            raise ValueError(
                f"No valid cross-validation configurations were available for "
                f"{generator_name}."
            )

        best_row = min(valid_rows, key=lambda row: row["mean_validation_loss"])
        best_by_generator[generator_name] = build_best_configuration(best_row)
        write_json(
            RESULT_DIR / f"{generator_name}_policy_design_cv.json",
            {
                "generator_name": generator_name,
                "source_csv": str(csv_path),
                "k_folds": K_FOLDS,
                "results": generator_rows,
                "best_configuration": best_by_generator[generator_name],
            },
        )
        all_rows.extend(generator_rows)

    write_json(RESULT_DIR / "policy_design_cv_summary.json", all_rows)
    write_summary_csv(RESULT_DIR / "policy_design_cv_summary.csv", all_rows)
    write_json(RESULT_DIR / "best_policy_design_by_generator.json", best_by_generator)

    if CREATE_PLOTS:
        create_policy_design_cv_plots(
            summary_json_path=RESULT_DIR / "policy_design_cv_summary.json",
            best_json_path=RESULT_DIR / "best_policy_design_by_generator.json",
            plot_dir=RESULT_DIR / "plots",
        )

    print(f"Saved policy-design cross-validation results to {RESULT_DIR}")


def find_generator_feature_files(feature_dir: Path) -> list[Path]:
    """Return normalized generator feature CSVs in deterministic order."""
    if not feature_dir.exists():
        raise ValueError(f"Feature directory does not exist: {feature_dir}")
    return sorted(
        path
        for path in feature_dir.glob("*_features_normalized.csv")
        if path.is_file()
    )


def evaluate_generator_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Evaluate every feature-set and architecture configuration for one CSV."""
    dataframe = pd.read_csv(csv_path)
    generator_name = infer_generator_name(dataframe, csv_path)
    available_features = identify_feature_columns(dataframe)
    target_columns = identify_target_columns(dataframe)
    if not target_columns:
        raise ValueError(f"No target_bid_ columns found in {csv_path}")

    validate_numeric_columns(dataframe, available_features, csv_path, "feature")
    validate_numeric_columns(dataframe, target_columns, csv_path, "target")

    fold_splits, fold_warning = build_fold_splits(dataframe, csv_path)
    rows: list[dict[str, Any]] = []

    for feature_set_name, requested_columns in FEATURE_SETS.items():
        feature_columns, skip_warning = resolve_feature_columns(
            feature_set_name=feature_set_name,
            requested_columns=requested_columns,
            available_features=available_features,
            target_columns=target_columns,
        )
        for hidden_layers in HIDDEN_LAYER_GRID:
            if len(hidden_layers) < 2:
                raise ValueError(
                    f"Architecture grid must not include fewer than two hidden "
                    f"layers: {hidden_layers}"
                )

            base_row = build_base_result_row(
                generator_name=generator_name,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                hidden_layers=hidden_layers,
                warning=skip_warning or fold_warning,
            )
            if skip_warning is not None or feature_columns is None:
                rows.append({**base_row, "status": "skipped"})
                continue

            fold_results = run_configuration_cv(
                dataframe=dataframe,
                generator_name=generator_name,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                target_columns=target_columns,
                hidden_layers=hidden_layers,
                fold_splits=fold_splits,
            )
            validation_losses = [result.validation_loss for result in fold_results]
            best_epochs = [result.best_epoch for result in fold_results]
            rows.append(
                {
                    **base_row,
                    "status": "valid",
                    "fold_validation_losses": validation_losses,
                    "mean_validation_loss": float(np.mean(validation_losses)),
                    "std_validation_loss": float(np.std(validation_losses, ddof=0)),
                    "mean_best_epoch": float(np.mean(best_epochs)),
                    "fold_train_sizes": [result.train_size for result in fold_results],
                    "fold_validation_sizes": [
                        result.validation_size for result in fold_results
                    ],
                }
            )

            print(
                f"  {generator_name} | {feature_set_name} | {hidden_layers}: "
                f"mean_validation_loss={np.mean(validation_losses):.8g}"
            )

    return rows


def run_configuration_cv(
    dataframe: pd.DataFrame,
    generator_name: str,
    feature_set_name: str,
    feature_columns: list[str],
    target_columns: list[str],
    hidden_layers: list[int],
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[FoldResult]:
    """Train fresh networks across folds for one policy-design configuration."""
    results: list[FoldResult] = []
    device = torch.device(DEVICE if DEVICE is not None else default_device())

    for fold_index, (train_index, validation_index) in enumerate(fold_splits, start=1):
        set_reproducible_seeds(RANDOM_STATE + fold_index)
        train_loader = make_data_loader(
            dataframe=dataframe.iloc[train_index],
            feature_columns=feature_columns,
            target_columns=target_columns,
            shuffle=True,
            seed=RANDOM_STATE + fold_index,
        )
        validation_loader = make_data_loader(
            dataframe=dataframe.iloc[validation_index],
            feature_columns=feature_columns,
            target_columns=target_columns,
            shuffle=False,
            seed=RANDOM_STATE + fold_index,
        )
        model = BiddingPolicyNetwork(
            input_dim=len(feature_columns),
            output_dim=len(target_columns),
            hidden_layers=hidden_layers,
        ).to(device)
        fold_result, best_state_dict = fit_fold_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
        )
        if SAVE_FOLD_MODELS:
            save_fold_model(
                model=model,
                state_dict=best_state_dict,
                generator_name=generator_name,
                feature_set_name=feature_set_name,
                hidden_layers=hidden_layers,
                fold_index=fold_index,
            )
        results.append(fold_result)

    return results


def fit_fold_model(
    model: BiddingPolicyNetwork,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[FoldResult, dict[str, torch.Tensor]]:
    """Fit one fold and return the best validation result."""
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        run_training_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss = evaluate_loss(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )
        if validation_loss < best_validation_loss - MIN_DELTA:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if PATIENCE is not None and epochs_without_improvement >= PATIENCE:
            break

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    return (
        FoldResult(
            validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            train_size=len(train_loader.dataset),
            validation_size=len(validation_loader.dataset),
        ),
        best_state_dict,
    )


def run_training_epoch(
    model: BiddingPolicyNetwork,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one optimizer epoch and return the sample-weighted training MSE."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def evaluate_loss(
    model: BiddingPolicyNetwork,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Return sample-weighted MSE on a validation loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = loss_function(predictions, targets)
            batch_size = features.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def make_data_loader(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a DataLoader from selected feature and target columns."""
    features = torch.tensor(
        dataframe[feature_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    targets = torch.tensor(
        dataframe[target_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def build_fold_splits(
    dataframe: pd.DataFrame,
    csv_path: Path,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str | None]:
    """Create scenario-aware folds when possible, otherwise row-based folds."""
    indices = np.arange(len(dataframe))
    if "scenario_id" in dataframe.columns:
        num_scenarios = int(dataframe["scenario_id"].nunique())
        if num_scenarios < K_FOLDS:
            raise ValueError(
                f"{csv_path} has {num_scenarios} scenario_id groups, but "
                f"K_FOLDS={K_FOLDS}."
            )
        splitter = GroupKFold(n_splits=K_FOLDS)
        splits = list(splitter.split(indices, groups=dataframe["scenario_id"]))
        return [(train, validation) for train, validation in splits], None

    if len(dataframe) < K_FOLDS:
        raise ValueError(f"{csv_path} has fewer rows than K_FOLDS={K_FOLDS}.")
    warning = (
        "No scenario_id column found; using row-based K-fold validation, so rows "
        "from the same scenario may be split across folds."
    )
    splitter = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(indices))
    return [(train, validation) for train, validation in splits], warning


def resolve_feature_columns(
    feature_set_name: str,
    requested_columns: list[str] | None,
    available_features: list[str],
    target_columns: list[str],
) -> tuple[list[str] | None, str | None]:
    """Resolve a configured feature set against columns available in a CSV."""
    available = set(available_features)
    targets = set(target_columns)
    if requested_columns is None:
        if not available_features:
            return None, "No non-target feature columns are available."
        return list(available_features), None

    missing_columns = [
        column
        for column in requested_columns
        if column not in available or column in targets
    ]
    if missing_columns:
        return None, (
            f"Skipping feature set '{feature_set_name}' because columns are "
            f"missing or invalid as inputs: {missing_columns}"
        )
    return list(requested_columns), None


def build_base_result_row(
    generator_name: str,
    feature_set_name: str,
    feature_columns: list[str] | None,
    hidden_layers: list[int],
    warning: str | None,
) -> dict[str, Any]:
    """Build the shared result fields for valid and skipped configurations."""
    resolved_features = feature_columns or []
    return {
        "generator_name": generator_name,
        "feature_set_name": feature_set_name,
        "feature_columns": resolved_features,
        "num_features": len(resolved_features),
        "hidden_layers": list(hidden_layers),
        "num_hidden_layers": len(hidden_layers),
        "total_hidden_neurons": int(sum(hidden_layers)),
        "fold_validation_losses": [],
        "mean_validation_loss": None,
        "std_validation_loss": None,
        "mean_best_epoch": None,
        "k_folds": K_FOLDS,
        "status": "pending",
        "warning": warning,
    }


def build_best_configuration(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the requested best-configuration JSON payload from a result row."""
    return {
        "feature_set_name": row["feature_set_name"],
        "feature_columns": row["feature_columns"],
        "hidden_layers": row["hidden_layers"],
        "num_features": row["num_features"],
        "num_hidden_layers": row["num_hidden_layers"],
        "total_hidden_neurons": row["total_hidden_neurons"],
        "mean_validation_loss": row["mean_validation_loss"],
        "std_validation_loss": row["std_validation_loss"],
        "k_folds": row["k_folds"],
    }


def validate_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    csv_path: Path,
    column_kind: str,
) -> None:
    """Raise a clear error if selected columns are nonnumeric or contain NaNs."""
    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric:
        raise ValueError(f"{column_kind.title()} columns must be numeric: {non_numeric}")

    nan_columns = dataframe[columns].columns[dataframe[columns].isna().any()].tolist()
    if nan_columns:
        raise ValueError(
            f"{column_kind.title()} columns contain NaN values in {csv_path}: "
            f"{nan_columns}"
        )


def infer_generator_name(dataframe: pd.DataFrame, path: Path) -> str:
    """Infer a generator name from the CSV metadata or file name."""
    if "generator_name" in dataframe.columns:
        names = dataframe["generator_name"].dropna().astype(str).unique().tolist()
        if len(names) == 1:
            return names[0]
        if len(names) > 1:
            raise ValueError(
                f"Expected one generator_name in {path}, found: {sorted(names)}"
            )

    stem = path.stem
    for suffix in ("_features_normalized", "_features"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def save_fold_model(
    model: BiddingPolicyNetwork,
    state_dict: dict[str, torch.Tensor],
    generator_name: str,
    feature_set_name: str,
    hidden_layers: list[int],
    fold_index: int,
) -> None:
    """Save a fold model only when SAVE_FOLD_MODELS is enabled."""
    model.load_state_dict(state_dict)
    model_dir = RESULT_DIR / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    architecture_name = "x".join(str(width) for width in hidden_layers)
    path = (
        model_dir
        / f"{generator_name}_{feature_set_name}_{architecture_name}_fold{fold_index}.pt"
    )
    torch.save(model.state_dict(), path)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file with deterministic indentation."""
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write cross-validation summary rows to CSV."""
    fieldnames = [
        "generator_name",
        "feature_set_name",
        "feature_columns",
        "num_features",
        "hidden_layers",
        "num_hidden_layers",
        "total_hidden_neurons",
        "fold_validation_losses",
        "mean_validation_loss",
        "std_validation_loss",
        "mean_best_epoch",
        "k_folds",
        "status",
        "warning",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field])
                    if isinstance(row.get(field), list)
                    else row.get(field)
                    for field in fieldnames
                }
            )


def set_reproducible_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for repeatable diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    """Return the same default device convention as the main trainer."""
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
