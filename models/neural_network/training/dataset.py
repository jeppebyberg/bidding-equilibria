from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


METADATA_COLUMNS = {"scenario_id", "time_id", "generator_name"}
TARGET_COLUMN_PREFIX = "target_bid_"


@dataclass(frozen=True)
class BiddingPolicyData:
    generator_name: str
    train_loader: DataLoader
    test_loader: DataLoader
    feature_columns: list[str]
    target_columns: list[str]
    input_dim: int
    output_dim: int
    train_size: int
    test_size: int
    train_scenarios: list[int | str]
    test_scenarios: list[int | str]
    num_rows: int


def load_generator_policy_data(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 64,
    shuffle_train: bool = True,
) -> BiddingPolicyData:
    """Load one generator CSV and return grouped train/test DataLoaders."""
    path = Path(csv_path)
    dataframe = pd.read_csv(path)
    generator_name = _infer_generator_name(dataframe, path)

    feature_columns = identify_feature_columns(dataframe)
    target_columns = identify_target_columns(dataframe)
    _validate_dataframe(dataframe, feature_columns, target_columns, path)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(
        splitter.split(dataframe, groups=dataframe["scenario_id"])
    )
    train_df = dataframe.iloc[train_index].copy()
    test_df = dataframe.iloc[test_index].copy()

    train_scenarios = _sorted_unique(train_df["scenario_id"])
    test_scenarios = _sorted_unique(test_df["scenario_id"])
    overlap = set(train_scenarios) & set(test_scenarios)
    if overlap:
        raise ValueError(
            f"Train/test split leaked scenario_id groups for {path}: {sorted(overlap)}"
        )

    train_dataset = _to_tensor_dataset(train_df, feature_columns, target_columns)
    test_dataset = _to_tensor_dataset(test_df, feature_columns, target_columns)

    return BiddingPolicyData(
        generator_name=generator_name,
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
        ),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        feature_columns=feature_columns,
        target_columns=target_columns,
        input_dim=len(feature_columns),
        output_dim=len(target_columns),
        train_size=len(train_df),
        test_size=len(test_df),
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        num_rows=len(dataframe),
    )


def identify_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    target_columns = set(identify_target_columns(dataframe))
    excluded_columns = METADATA_COLUMNS | target_columns
    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns
        and not column.startswith(TARGET_COLUMN_PREFIX)
    ]


def identify_target_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column.startswith(TARGET_COLUMN_PREFIX)
    ]


def _validate_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    path: Path,
) -> None:
    if "scenario_id" not in dataframe.columns:
        raise ValueError(f"Dataset must contain scenario_id: {path}")
    if not feature_columns:
        raise ValueError(f"Dataset must contain at least one feature column: {path}")
    if not target_columns:
        raise ValueError(f"Dataset must contain at least one target_bid_ column: {path}")

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric_features:
        raise ValueError(
            f"Feature columns must be numeric in {path}: {non_numeric_features}"
        )

    non_numeric_targets = [
        column
        for column in target_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric_targets:
        raise ValueError(
            f"Target columns must be numeric in {path}: {non_numeric_targets}"
        )

    nan_feature_columns = dataframe[feature_columns].columns[
        dataframe[feature_columns].isna().any()
    ].tolist()
    if nan_feature_columns:
        raise ValueError(
            f"Feature columns contain NaN values in {path}: {nan_feature_columns}"
        )

    nan_target_columns = dataframe[target_columns].columns[
        dataframe[target_columns].isna().any()
    ].tolist()
    if nan_target_columns:
        raise ValueError(
            f"Target columns contain NaN values in {path}: {nan_target_columns}"
        )


def _to_tensor_dataset(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
) -> TensorDataset:
    features = torch.tensor(
        dataframe[feature_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    targets = torch.tensor(
        dataframe[target_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    return TensorDataset(features, targets)


def _infer_generator_name(dataframe: pd.DataFrame, path: Path) -> str:
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


def _sorted_unique(series: pd.Series) -> list[int | str]:
    values = series.dropna().unique().tolist()
    return sorted(value.item() if hasattr(value, "item") else value for value in values)
