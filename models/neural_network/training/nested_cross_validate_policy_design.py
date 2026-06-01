from __future__ import annotations

"""Nested, repeated, scenario-grouped cross-validation for bidding policies.

This module answers two distinct questions that the previous design-CV script
conflated into one:

1. *Which* policy design (feature set + architecture + regularization +
   output activation) should we deploy for each generator?
2. *How well* will that selection procedure generalize, with an estimate that
   is NOT optimistically biased by the fact that we searched a grid?

Question 1 is answered by a repeated, scenario-grouped K-fold ranking over the
whole dataset, with several random seeds per configuration and a paired
statistical test between the best and the runner-up.

Question 2 is answered by *nested* cross-validation: an outer scenario-grouped
K-fold whose test groups are never seen during model selection. Inside each
outer-train split, the same selection procedure runs on an inner grouped
K-fold, the winning config is refit on the full outer-train split, and it is
scored once on the untouched outer-test groups. Averaging those outer-test
scores gives an honest generalization estimate.

Two leakage controls that the previous script lacked:
  * Early stopping uses a grouped "stop split" carved out of the *training*
    groups, so the evaluation set is never used to choose when to stop.
  * Feature standardization statistics are fit on training rows only and
    applied to the evaluation rows, removing any normalization leakage even if
    the input CSV was globally normalized.

torch is imported lazily inside the training functions so that the
orchestration helpers (split construction, aggregation, statistics) can be
imported and unit-tested without a torch install.
"""

import csv
import json
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.neural_network.training.dataset import (  # noqa: E402
    identify_feature_columns,
    identify_target_columns,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FEATURE_DIR = Path("models/neural_network/features/generated/normalized")
RESULT_DIR = Path("models/neural_network/training/nested_policy_design_cv_results")

# Candidate feature sets (same names as the legacy script so plots stay
# comparable). "full_supported" is resolved per-CSV to every available feature.
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

FEATURE_SETS: dict[str, list[str] | None] = {
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
    "full_supported": None,  # None => use every available feature column.
}

# --- Search space: ONLY feature set, number of hidden layers, and neurons per
# layer. Learning rate, weight decay, and the final activation are held fixed
# at the values below and are NOT searched.
HIDDEN_WIDTHS: list[int] = [4, 8, 11, 16, 24]   # neurons per hidden layer
HIDDEN_DEPTHS: list[int] = [1, 2, 3]                # number of hidden layers
INCLUDE_LINEAR_BASELINE = True                      # also try 0 hidden layers


def build_architecture_grid() -> list[list[int]]:
    """Constant-width architectures spanning the width x depth grid.

    Constant width keeps your two axes cleanly separable: every architecture's
    neuron count is simply width * depth, so width and depth effects do not get
    tangled the way they would with tapered layers like [8, 4].
    """
    grid: list[list[int]] = []
    if INCLUDE_LINEAR_BASELINE:
        grid.append([])  # pure linear map: a 0-hidden-layer anchor
    for depth in HIDDEN_DEPTHS:
        for width in HIDDEN_WIDTHS:
            grid.append([width] * depth)
    return grid


ARCHITECTURE_GRID: list[list[int]] = build_architecture_grid()

# Held fixed (not part of the search), per your instruction.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
FINAL_ACTIVATION = "linear"

BATCH_SIZE = 128
MAX_EPOCHS = 500
PATIENCE = 30
MIN_DELTA = 1e-6

# Cross-validation structure.
OUTER_K_FOLDS = 5          # honest generalization estimate (Q2)
INNER_K_FOLDS = 3          # model selection inside each outer-train split (Q2)
RANKING_K_FOLDS = 5        # whole-data ranking used for the deployed choice (Q1)
N_REPEATS = 1              # repeat CV with reshuffled group->fold assignment
N_SEEDS_PER_FIT = 1        # independent inits/batch orders per (config, fold)
STOP_SPLIT_GROUP_FRACTION = 0.2  # share of train groups reserved for early stop

RANDOM_STATE = 42
STANDARDIZE_IN_FOLD = True  # fit feature scaler on train rows only
DEVICE: str | None = None
CREATE_PLOTS = False  # legacy plotter expects the old schema; off by default.


@dataclass(frozen=True)
class PolicyConfig:
    """One policy design: a feature set and a hidden-layer architecture."""

    feature_set_name: str
    hidden_layers: tuple[int, ...]

    @property
    def architecture_label(self) -> str:
        return "x".join(str(width) for width in self.hidden_layers) or "linear"

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_layers)

    @property
    def total_hidden_neurons(self) -> int:
        return int(sum(self.hidden_layers))

    def describe(self) -> str:
        return f"{self.feature_set_name} | {self.architecture_label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set_name": self.feature_set_name,
            "hidden_layers": list(self.hidden_layers),
            "architecture_label": self.architecture_label,
            "num_hidden_layers": self.num_hidden_layers,
            "total_hidden_neurons": self.total_hidden_neurons,
        }


@dataclass
class FitMetrics:
    """Metrics from evaluating one trained model on one held-out set."""

    mse: float
    rmse: float
    mae: float
    normalized_rmse: float  # RMSE divided by per-target std on the eval set
    r2: float
    per_target_mse: list[float]
    best_epoch: int


def build_config_grid() -> list[PolicyConfig]:
    """Every (feature set, architecture) pair. lr/wd/activation are fixed."""
    return [
        PolicyConfig(
            feature_set_name=feature_set_name,
            hidden_layers=tuple(hidden_layers),
        )
        for feature_set_name, hidden_layers in product(
            FEATURE_SETS, ARCHITECTURE_GRID
        )
    ]

# --------------------------------------------------------------------------- #
# Grouped split helpers (shuffleable so repeats use different partitions)
# --------------------------------------------------------------------------- #

def grouped_kfold_indices(
    groups: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition rows into n_folds with no group split across train/eval.

    Unlike sklearn's GroupKFold this shuffles the group->fold assignment using
    ``seed``, which is what lets repeated CV use genuinely different partitions.
    """
    unique_groups = np.array(sorted(pd.unique(groups)))
    if len(unique_groups) < n_folds:
        raise ValueError(
            f"Need at least n_folds={n_folds} groups, found {len(unique_groups)}."
        )
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    fold_of_group = {
        group: index % n_folds for index, group in enumerate(shuffled)
    }
    fold_assignment = np.array([fold_of_group[group] for group in groups])

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        eval_mask = fold_assignment == fold
        eval_index = np.flatnonzero(eval_mask)
        train_index = np.flatnonzero(~eval_mask)
        if len(eval_index) == 0 or len(train_index) == 0:
            continue
        splits.append((train_index, eval_index))
    return splits


def grouped_holdout(
    groups: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into (keep, holdout) by whole groups, for early-stop splits."""
    unique_groups = np.array(sorted(pd.unique(groups)))
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(holdout_fraction * len(unique_groups))))
    n_holdout = min(n_holdout, len(unique_groups) - 1)  # keep >=1 group to train
    holdout_groups = set(shuffled[:n_holdout].tolist())
    holdout_mask = np.array([group in holdout_groups for group in groups])
    return np.flatnonzero(~holdout_mask), np.flatnonzero(holdout_mask)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    best_epoch: int,
) -> FitMetrics:
    """Compute a panel of regression metrics for one evaluation set."""
    errors = predictions - targets
    per_target_mse = np.mean(errors**2, axis=0)
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    target_std = targets.std(axis=0)
    safe_std = np.where(target_std > 1e-12, target_std, np.nan)
    normalized_rmse = float(
        np.nanmean(np.sqrt(per_target_mse) / safe_std)
    )

    total_variance = float(np.sum((targets - targets.mean(axis=0)) ** 2))
    residual = float(np.sum(errors**2))
    r2 = float("nan") if total_variance <= 1e-12 else 1.0 - residual / total_variance

    return FitMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        normalized_rmse=normalized_rmse,
        r2=r2,
        per_target_mse=[float(value) for value in per_target_mse],
        best_epoch=best_epoch,
    )


# --------------------------------------------------------------------------- #
# Training: lazy torch import keeps the rest of the module torch-free
# --------------------------------------------------------------------------- #

def train_and_evaluate(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    groups_train: np.ndarray,
    features_eval: np.ndarray,
    targets_eval: np.ndarray,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
    final_activation: str,
    seed: int,
    device_name: str | None,
) -> FitMetrics:
    """Train one network and score it once on a held-out evaluation set.

    Early stopping uses a grouped stop-split carved from the training groups,
    so ``features_eval`` is never used to decide when to stop. Feature
    standardization (optional) is fit on the training rows only.
    """
    import torch
    from torch import nn

    from models.neural_network.training.model import BiddingPolicyNetwork

    _seed_everything(seed)
    device = torch.device(device_name if device_name is not None else _default_device())

    # Leakage-free standardization: stats from training rows only.
    if STANDARDIZE_IN_FOLD:
        mean = features_train.mean(axis=0, keepdims=True)
        std = features_train.std(axis=0, keepdims=True)
        std = np.where(std > 1e-12, std, 1.0)
        features_train = (features_train - mean) / std
        features_eval = (features_eval - mean) / std

    # Grouped early-stopping split out of the training data.
    keep_index, stop_index = grouped_holdout(
        groups_train, STOP_SPLIT_GROUP_FRACTION, seed
    )
    x_keep = torch.tensor(features_train[keep_index], dtype=torch.float32)
    y_keep = torch.tensor(targets_train[keep_index], dtype=torch.float32)
    x_stop = torch.tensor(features_train[stop_index], dtype=torch.float32, device=device)
    y_stop = torch.tensor(targets_train[stop_index], dtype=torch.float32, device=device)
    x_eval = torch.tensor(features_eval, dtype=torch.float32, device=device)
    y_eval = torch.tensor(targets_eval, dtype=torch.float32, device=device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_keep, y_keep),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = BiddingPolicyNetwork(
        input_dim=features_train.shape[1],
        output_dim=targets_train.shape[1],
        hidden_layers=list(hidden_layers),
        final_activation=final_activation,
    ).to(device)
    if final_activation == "relu":
        _initialize_final_relu_bias(model, y_keep.to(device))

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_stop_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            stop_loss = float(loss_function(model(x_stop), y_stop).item())

        if stop_loss < best_stop_loss - MIN_DELTA:
            best_stop_loss = stop_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if PATIENCE is not None and epochs_without_improvement >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predictions = model(x_eval).cpu().numpy()
    return compute_metrics(predictions, targets_eval, best_epoch)


def _initialize_final_relu_bias(model: Any, targets: Any) -> None:
    """Seed the final-layer bias to target means (mirrors the legacy trainer)."""
    import torch
    from torch import nn

    final_linear = None
    for layer in reversed(model.network):
        if isinstance(layer, nn.Linear):
            final_linear = layer
            break
    if final_linear is None:
        return
    target_means = targets.mean(dim=0).clamp_min(1e-6)
    with torch.no_grad():
        final_linear.bias.copy_(target_means)


# A pluggable trainer reference so the orchestration can be unit-tested with a
# numpy stand-in instead of torch.
TrainerFn = Callable[..., FitMetrics]
_TRAINER: TrainerFn = train_and_evaluate


def set_trainer(trainer: TrainerFn) -> None:
    """Override the training backend (used by tests)."""
    global _TRAINER
    _TRAINER = trainer


# --------------------------------------------------------------------------- #
# Cross-validation of a single configuration
# --------------------------------------------------------------------------- #

def evaluate_config_on_folds(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    config: PolicyConfig,
    feature_index: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    repeat_seed: int,
) -> list[FitMetrics]:
    """Return one averaged-over-seeds FitMetrics per fold for ``config``."""
    fold_metrics: list[FitMetrics] = []
    selected_features = features[:, feature_index]
    for fold_number, (train_index, eval_index) in enumerate(folds):
        seed_metrics: list[FitMetrics] = []
        for seed_offset in range(N_SEEDS_PER_FIT):
            seed = repeat_seed + 1000 * fold_number + seed_offset
            metrics = _TRAINER(
                features_train=selected_features[train_index],
                targets_train=targets[train_index],
                groups_train=groups[train_index],
                features_eval=selected_features[eval_index],
                targets_eval=targets[eval_index],
                hidden_layers=config.hidden_layers,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                final_activation=FINAL_ACTIVATION,
                seed=seed,
                device_name=DEVICE,
            )
            seed_metrics.append(metrics)
        fold_metrics.append(_average_metrics(seed_metrics))
    return fold_metrics


def _average_metrics(metrics_list: list[FitMetrics]) -> FitMetrics:
    """Average a list of FitMetrics (over seeds) into one FitMetrics."""
    n_targets = len(metrics_list[0].per_target_mse)
    per_target = [
        float(np.mean([m.per_target_mse[j] for m in metrics_list]))
        for j in range(n_targets)
    ]
    return FitMetrics(
        mse=float(np.mean([m.mse for m in metrics_list])),
        rmse=float(np.mean([m.rmse for m in metrics_list])),
        mae=float(np.mean([m.mae for m in metrics_list])),
        normalized_rmse=float(np.mean([m.normalized_rmse for m in metrics_list])),
        r2=float(np.mean([m.r2 for m in metrics_list])),
        per_target_mse=per_target,
        best_epoch=int(np.mean([m.best_epoch for m in metrics_list])),
    )


# --------------------------------------------------------------------------- #
# Baselines (scored under the same grouped folds for context)
# --------------------------------------------------------------------------- #

def baseline_losses_on_folds(
    targets: np.ndarray,
    features: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[float]]:
    """Per-fold MSE for predict-the-train-mean and a least-squares linear map."""
    mean_losses: list[float] = []
    linear_losses: list[float] = []
    for train_index, eval_index in folds:
        y_train = targets[train_index]
        y_eval = targets[eval_index]

        mean_prediction = y_train.mean(axis=0, keepdims=True)
        mean_losses.append(float(np.mean((y_eval - mean_prediction) ** 2)))

        x_train = np.hstack([features[train_index], np.ones((len(train_index), 1))])
        x_eval = np.hstack([features[eval_index], np.ones((len(eval_index), 1))])
        coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        linear_losses.append(float(np.mean((y_eval - x_eval @ coefficients) ** 2)))
    return {"predict_mean": mean_losses, "linear": linear_losses}


# --------------------------------------------------------------------------- #
# Whole-data ranking (Q1: which design to deploy) + significance
# --------------------------------------------------------------------------- #

def rank_configs_on_full_data(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    configs: list[PolicyConfig],
    feature_indices: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Repeated grouped K-fold over all data; rank configs by mean fold MSE.

    Each config gets a vector of per-(repeat, fold) MSEs. Configurations are
    ranked by the grand mean; the vectors are paired across (repeat, fold)
    units to support a paired test between the best and the runner-up.
    """
    repeat_fold_plans = [
        grouped_kfold_indices(groups, RANKING_K_FOLDS, RANDOM_STATE + 7 * repeat)
        for repeat in range(N_REPEATS)
    ]

    rows: list[dict[str, Any]] = []
    for config in configs:
        feature_index = feature_indices[config.feature_set_name]
        paired_losses: list[float] = []
        normalized_losses: list[float] = []
        epochs: list[int] = []
        for repeat, folds in enumerate(repeat_fold_plans):
            metrics = evaluate_config_on_folds(
                features=features,
                targets=targets,
                groups=groups,
                config=config,
                feature_index=feature_index,
                folds=folds,
                repeat_seed=RANDOM_STATE + 7 * repeat,
            )
            paired_losses.extend(m.mse for m in metrics)
            normalized_losses.extend(m.normalized_rmse for m in metrics)
            epochs.extend(m.best_epoch for m in metrics)
        rows.append(
            {
                "config": config,
                "paired_mse": paired_losses,
                "mean_mse": float(np.mean(paired_losses)),
                "std_mse": float(np.std(paired_losses, ddof=1)),
                "se_mse": float(np.std(paired_losses, ddof=1) / np.sqrt(len(paired_losses))),
                "mean_normalized_rmse": float(np.mean(normalized_losses)),
                "mean_best_epoch": float(np.mean(epochs)),
                "n_units": len(paired_losses),
            }
        )
    rows.sort(key=lambda row: row["mean_mse"])
    return rows


def compare_best_to_runner_up(ranking: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired Wilcoxon + t-test between the top two configs (matched folds)."""
    if len(ranking) < 2:
        return {"comparable": False, "reason": "fewer than two configurations"}
    best = np.array(ranking[0]["paired_mse"])
    runner_up = np.array(ranking[1]["paired_mse"])
    differences = runner_up - best  # positive => best is better on that unit

    result: dict[str, Any] = {
        "comparable": True,
        "best_config": ranking[0]["config"].describe(),
        "runner_up_config": ranking[1]["config"].describe(),
        "mean_difference": float(np.mean(differences)),
        "n_units": int(len(differences)),
        "best_wins_fraction": float(np.mean(differences > 0)),
    }
    if np.allclose(differences, 0.0):
        result["wilcoxon_p_value"] = 1.0
        result["t_test_p_value"] = 1.0
        return result
    try:
        result["wilcoxon_p_value"] = float(
            stats.wilcoxon(differences, alternative="greater").pvalue
        )
    except ValueError as error:
        result["wilcoxon_p_value"] = None
        result["wilcoxon_note"] = str(error)
    result["t_test_p_value"] = float(
        stats.ttest_rel(runner_up, best, alternative="greater").pvalue
    )
    return result


# --------------------------------------------------------------------------- #
# Nested CV (Q2: honest generalization estimate of the selection procedure)
# --------------------------------------------------------------------------- #

def nested_cross_validation(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    configs: list[PolicyConfig],
    feature_indices: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Outer test scores never seen during inner model selection."""
    outer_test_mse: list[float] = []
    outer_test_normalized_rmse: list[float] = []
    selected_descriptions: list[str] = []

    for repeat in range(N_REPEATS):
        outer_folds = grouped_kfold_indices(
            groups, OUTER_K_FOLDS, RANDOM_STATE + 13 * repeat
        )
        for outer_number, (outer_train_index, outer_test_index) in enumerate(outer_folds):
            inner_groups = groups[outer_train_index]
            inner_folds = grouped_kfold_indices(
                inner_groups,
                INNER_K_FOLDS,
                RANDOM_STATE + 13 * repeat + 101 * outer_number,
            )
            # Remap inner fold indices (which index into outer_train) to global.
            inner_folds_global = [
                (outer_train_index[train], outer_train_index[evl])
                for train, evl in inner_folds
            ]

            best_config: PolicyConfig | None = None
            best_inner_mse = float("inf")
            for config in configs:
                feature_index = feature_indices[config.feature_set_name]
                metrics = evaluate_config_on_folds(
                    features=features,
                    targets=targets,
                    groups=groups,
                    config=config,
                    feature_index=feature_index,
                    folds=inner_folds_global,
                    repeat_seed=RANDOM_STATE + 13 * repeat + 101 * outer_number,
                )
                inner_mse = float(np.mean([m.mse for m in metrics]))
                if inner_mse < best_inner_mse:
                    best_inner_mse = inner_mse
                    best_config = config

            assert best_config is not None
            selected_descriptions.append(best_config.describe())

            # Refit selected config on full outer-train, score on outer-test.
            feature_index = feature_indices[best_config.feature_set_name]
            selected_features = features[:, feature_index]
            seed_metrics = [
                _TRAINER(
                    features_train=selected_features[outer_train_index],
                    targets_train=targets[outer_train_index],
                    groups_train=groups[outer_train_index],
                    features_eval=selected_features[outer_test_index],
                    targets_eval=targets[outer_test_index],
                    hidden_layers=best_config.hidden_layers,
                    learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    final_activation=FINAL_ACTIVATION,
                    seed=RANDOM_STATE + 13 * repeat + 101 * outer_number + seed_offset,
                    device_name=DEVICE,
                )
                for seed_offset in range(N_SEEDS_PER_FIT)
            ]
            averaged = _average_metrics(seed_metrics)
            outer_test_mse.append(averaged.mse)
            outer_test_normalized_rmse.append(averaged.normalized_rmse)

    selection_counts: dict[str, int] = {}
    for description in selected_descriptions:
        selection_counts[description] = selection_counts.get(description, 0) + 1
    most_common = max(selection_counts.items(), key=lambda item: item[1])

    n = len(outer_test_mse)
    mean_mse = float(np.mean(outer_test_mse))
    se_mse = float(np.std(outer_test_mse, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "nested_mean_mse": mean_mse,
        "nested_se_mse": se_mse,
        "nested_mse_95ci": [mean_mse - 1.96 * se_mse, mean_mse + 1.96 * se_mse],
        "nested_mean_normalized_rmse": float(np.mean(outer_test_normalized_rmse)),
        "outer_fold_mse": outer_test_mse,
        "n_outer_evaluations": n,
        "selection_stability": selection_counts,
        "most_frequently_selected": {
            "config": most_common[0],
            "times_selected": most_common[1],
            "out_of": n,
        },
    }


# --------------------------------------------------------------------------- #
# Per-generator driver
# --------------------------------------------------------------------------- #

def evaluate_generator(csv_path: Path, configs: list[PolicyConfig]) -> dict[str, Any]:
    """Run ranking + nested CV + baselines for one generator CSV."""
    dataframe = pd.read_csv(csv_path)
    generator_name = _infer_generator_name(dataframe, csv_path)
    available_features = identify_feature_columns(dataframe)
    target_columns = identify_target_columns(dataframe)
    if not target_columns:
        raise ValueError(f"No target_bid_ columns found in {csv_path}")
    if "scenario_id" not in dataframe.columns:
        raise ValueError(
            f"{csv_path} has no scenario_id column; grouped CV requires it."
        )

    groups = dataframe["scenario_id"].to_numpy()
    targets = dataframe[target_columns].to_numpy(dtype=np.float64)

    # Resolve each feature set to column indices, skipping invalid ones.
    feature_indices, usable_feature_sets = _resolve_feature_indices(
        dataframe, available_features
    )
    usable_configs = [
        config for config in configs
        if config.feature_set_name in usable_feature_sets
    ]
    if not usable_configs:
        raise ValueError(f"No usable feature sets for {generator_name}.")

    feature_matrix = dataframe[available_features].to_numpy(dtype=np.float64)
    column_position = {name: i for i, name in enumerate(available_features)}
    index_by_set = {
        name: np.array([column_position[c] for c in cols])
        for name, cols in usable_feature_sets.items()
    }

    print(f"  ranking {len(usable_configs)} configurations on full data...")
    ranking = rank_configs_on_full_data(
        features=feature_matrix,
        targets=targets,
        groups=groups,
        configs=usable_configs,
        feature_indices=index_by_set,
    )
    significance = compare_best_to_runner_up(ranking)

    print("  running nested cross-validation...")
    nested = nested_cross_validation(
        features=feature_matrix,
        targets=targets,
        groups=groups,
        configs=usable_configs,
        feature_indices=index_by_set,
    )

    # Baselines on the first ranking fold plan for context.
    baseline_folds = grouped_kfold_indices(groups, RANKING_K_FOLDS, RANDOM_STATE)
    baselines = baseline_losses_on_folds(targets, feature_matrix, groups, baseline_folds)

    best = ranking[0]
    return {
        "generator_name": generator_name,
        "source_csv": str(csv_path),
        "num_rows": int(len(dataframe)),
        "num_scenarios": int(pd.Series(groups).nunique()),
        "target_columns": target_columns,
        "deployed_choice": {
            "config": best["config"].to_dict(),
            "description": best["config"].describe(),
            "mean_cv_mse": best["mean_mse"],
            "se_cv_mse": best["se_mse"],
            "mean_normalized_rmse": best["mean_normalized_rmse"],
            "mean_best_epoch": best["mean_best_epoch"],
        },
        "selection_significance": significance,
        "honest_generalization_estimate": {
            "mean_mse": nested["nested_mean_mse"],
            "se_mse": nested["nested_se_mse"],
            "mse_95ci": nested["nested_mse_95ci"],
            "mean_normalized_rmse": nested["nested_mean_normalized_rmse"],
            "n_outer_evaluations": nested["n_outer_evaluations"],
            "selection_stability": nested["selection_stability"],
            "most_frequently_selected": nested["most_frequently_selected"],
        },
        "baselines": {
            "predict_mean_mse": float(np.mean(baselines["predict_mean"])),
            "linear_mse": float(np.mean(baselines["linear"])),
        },
        "ranking": [
            {
                "rank": position + 1,
                "config": row["config"].to_dict(),
                "description": row["config"].describe(),
                "mean_mse": row["mean_mse"],
                "std_mse": row["std_mse"],
                "se_mse": row["se_mse"],
                "mean_normalized_rmse": row["mean_normalized_rmse"],
                "mean_best_epoch": row["mean_best_epoch"],
                "n_units": row["n_units"],
            }
            for position, row in enumerate(ranking)
        ],
    }


def _resolve_feature_indices(
    dataframe: pd.DataFrame,
    available_features: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return per-set resolved columns, skipping sets with missing columns."""
    available = set(available_features)
    resolved: dict[str, list[str]] = {}
    usable: dict[str, list[str]] = {}
    for name, requested in FEATURE_SETS.items():
        if requested is None:
            columns = list(available_features)
        else:
            missing = [c for c in requested if c not in available]
            if missing:
                print(f"  skipping feature set '{name}' (missing columns: {missing})")
                continue
            columns = list(requested)
        resolved[name] = columns
        usable[name] = columns
    return resolved, usable


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #

def main() -> None:
    """Cross-validate every generator and write per-generator + summary JSON."""
    _seed_everything(RANDOM_STATE)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    configs = build_config_grid()
    print(
        f"Search space: {len(configs)} configurations "
        f"({len(FEATURE_SETS)} feature sets x {len(ARCHITECTURE_GRID)} architectures; "
        f"lr={LEARNING_RATE:g}, weight_decay={WEIGHT_DECAY:g}, "
        f"final_activation={FINAL_ACTIVATION} held fixed)."
    )
    print(
        f"Per generator: ranking={N_REPEATS}x{RANKING_K_FOLDS} folds, "
        f"nested={N_REPEATS}x{OUTER_K_FOLDS} outer x {INNER_K_FOLDS} inner, "
        f"{N_SEEDS_PER_FIT} seeds/fit. This is intentionally heavy."
    )

    csv_paths = _find_generator_feature_files(FEATURE_DIR)
    if not csv_paths:
        raise ValueError(f"No normalized generator feature CSVs in {FEATURE_DIR}")

    all_results: list[dict[str, Any]] = []
    deployed_by_generator: dict[str, Any] = {}
    for csv_path in csv_paths:
        print(f"Cross-validating {csv_path}...")
        result = evaluate_generator(csv_path, configs)
        all_results.append(result)
        deployed_by_generator[result["generator_name"]] = {
            "deployed_choice": result["deployed_choice"],
            "honest_generalization_estimate": result["honest_generalization_estimate"],
            "selection_significance": result["selection_significance"],
            "baselines": result["baselines"],
        }
        _write_json(
            RESULT_DIR / f"{result['generator_name']}_nested_policy_design_cv.json",
            result,
        )
        _print_generator_summary(result)

    _write_json(RESULT_DIR / "nested_policy_design_cv_summary.json", all_results)
    _write_json(
        RESULT_DIR / "deployed_policy_design_by_generator.json", deployed_by_generator
    )
    _write_ranking_csv(RESULT_DIR / "policy_design_ranking.csv", all_results)
    print(f"Saved results to {RESULT_DIR}")


def _print_generator_summary(result: dict[str, Any]) -> None:
    deployed = result["deployed_choice"]
    honest = result["honest_generalization_estimate"]
    sig = result["selection_significance"]
    base = result["baselines"]
    print(f"  -> deploy: {deployed['description']}")
    print(
        f"     ranking CV MSE = {deployed['mean_cv_mse']:.6g} "
        f"(+/- {deployed['se_cv_mse']:.2g} SE)"
    )
    print(
        f"     honest (nested) MSE = {honest['mean_mse']:.6g} "
        f"(95% CI {honest['mse_95ci'][0]:.4g}..{honest['mse_95ci'][1]:.4g})"
    )
    print(
        f"     baselines: predict-mean={base['predict_mean_mse']:.6g}, "
        f"linear={base['linear_mse']:.6g}"
    )
    if sig.get("comparable"):
        print(
            f"     best vs runner-up: wins "
            f"{sig['best_wins_fraction'] * 100:.0f}% of folds, "
            f"Wilcoxon p={sig.get('wilcoxon_p_value')}"
        )


def _write_ranking_csv(path: Path, all_results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "generator_name", "rank", "description", "feature_set_name",
        "architecture_label", "num_hidden_layers", "total_hidden_neurons",
        "mean_mse", "std_mse", "se_mse", "mean_normalized_rmse",
        "mean_best_epoch", "n_units",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            for row in result["ranking"]:
                config = row["config"]
                writer.writerow(
                    {
                        "generator_name": result["generator_name"],
                        "rank": row["rank"],
                        "description": row["description"],
                        "feature_set_name": config["feature_set_name"],
                        "architecture_label": config["architecture_label"],
                        "num_hidden_layers": config["num_hidden_layers"],
                        "total_hidden_neurons": config["total_hidden_neurons"],
                        "mean_mse": row["mean_mse"],
                        "std_mse": row["std_mse"],
                        "se_mse": row["se_mse"],
                        "mean_normalized_rmse": row["mean_normalized_rmse"],
                        "mean_best_epoch": row["mean_best_epoch"],
                        "n_units": row["n_units"],
                    }
                )


def _find_generator_feature_files(feature_dir: Path) -> list[Path]:
    if not feature_dir.exists():
        raise ValueError(f"Feature directory does not exist: {feature_dir}")
    return sorted(
        path for path in feature_dir.glob("*_features_normalized.csv") if path.is_file()
    )


def _infer_generator_name(dataframe: pd.DataFrame, path: Path) -> str:
    if "generator_name" in dataframe.columns:
        names = dataframe["generator_name"].dropna().astype(str).unique().tolist()
        if len(names) == 1:
            return names[0]
        if len(names) > 1:
            raise ValueError(f"Expected one generator_name in {path}, found {names}")
    stem = path.stem
    for suffix in ("_features_normalized", "_features"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
