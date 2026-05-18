from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MinMaxNormalizationStats:
    feature_min: dict[str, float]
    feature_max: dict[str, float]


class NeuralNetworkFeatureBuilder:
    """Build one supervised policy-training dataset per physical generator."""

    METADATA_COLUMNS = {"scenario_id", "time_id", "generator_name"}
    TARGET_COLUMN_PREFIX = "target_bid_"
    NORMALIZATION_TOLERANCE = 1e-9

    BASE_FEATURE_COLUMNS = [
        "demand",
        "total_wind_generation_capacity",
        "total_generation_capacity",
        "residual_demand",
        "previous_generation_capacity",
        "previous_demand",
        "next_generation_capacity",
        "next_demand",
    ]
    OWN_FEATURE_COLUMNS = [
        "own_generation_capacity",
        "previous_own_generation_capacity",
        "next_own_generation_capacity",
    ]
    COST_FEATURE_COLUMNS = [
        "average_true_cost",
        "minimum_true_cost",
        "maximum_true_cost",
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
        "average_true_cost",
        "minimum_true_cost",
        "maximum_true_cost",
    ]

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        results_path: str | Path = "results/merit_order_best_response_results.json",
        feature_columns: list[str] | None = None,
    ) -> None:
        self.scenarios_df = scenarios_df.copy(deep=True).reset_index(drop=True)
        self.costs_df = costs_df
        self.results_path = Path(results_path)
        self.results = self._load_results(self.results_path)
        self.feature_columns = self._resolve_feature_columns(feature_columns)

        self.physical_generator_names = [
            str(name) for name in self.results["physical_generator_names"]
        ]
        self.block_names = [str(name) for name in self.results["block_names"]]
        self.block_to_physical = {
            str(block): str(physical)
            for block, physical in self.results["block_to_physical"].items()
        }
        self.final_bids = self.results["final_bids"]
        self.num_scenarios = int(self.results["num_scenarios"])
        self.num_time_steps = int(self.results["num_time_steps"])
        self.num_blocks = len(self.block_names)

        self.physical_to_block_indices = self._build_physical_to_block_indices()
        self._market_features_by_scenario = self._compute_market_features()
        self._datasets: dict[str, pd.DataFrame] | None = None
        self._normalization_stats: (
            dict[str, MinMaxNormalizationStats] | MinMaxNormalizationStats | None
        ) = None
        self._normalization_per_generator: bool | None = None

    def build_all_generator_datasets(
        self,
        normalize: bool = False,
        per_generator_normalization: bool = True,
    ) -> dict[str, pd.DataFrame]:
        datasets = {
            generator_name: self.build_generator_dataset(generator_name)
            for generator_name in self.physical_generator_names
        }
        if not normalize:
            self._datasets = datasets
            self._normalization_stats = None
            self._normalization_per_generator = None
            return datasets

        stats = self.fit_min_max_stats(
            datasets,
            per_generator=per_generator_normalization,
        )
        normalized_datasets = self.normalize_datasets(
            datasets,
            stats,
            per_generator=per_generator_normalization,
        )
        self._datasets = normalized_datasets
        self._normalization_stats = stats
        self._normalization_per_generator = per_generator_normalization
        return normalized_datasets

    def build_generator_dataset(self, generator_name: str) -> pd.DataFrame:
        if generator_name not in self.physical_to_block_indices:
            raise ValueError(
                f"Unknown generator_name '{generator_name}'. "
                f"Available generators: {self.physical_generator_names}"
            )

        block_indices = self.physical_to_block_indices[generator_name]
        if not block_indices:
            raise ValueError(f"Generator '{generator_name}' has no bidding blocks")

        rows: list[dict[str, Any]] = []
        own_capacity = self._compute_own_capacity(block_indices)

        for scenario_id in range(self.num_scenarios):
            market_features = self._market_features_by_scenario[scenario_id]
            for time_id in range(self.num_time_steps):
                previous_time_id = (time_id - 1) % self.num_time_steps
                next_time_id = (time_id + 1) % self.num_time_steps
                row = {
                    "scenario_id": int(scenario_id),
                    "time_id": int(time_id),
                    "generator_name": generator_name,
                    **market_features[time_id],
                    "own_generation_capacity": float(own_capacity[scenario_id, time_id]),
                    "previous_own_generation_capacity": float(
                        own_capacity[scenario_id, previous_time_id]
                    ),
                    "next_own_generation_capacity": float(
                        own_capacity[scenario_id, next_time_id]
                    ),
                    **self._compute_true_cost_features(block_indices),
                }
                for block_idx in block_indices:
                    block_name = self.block_names[block_idx]
                    row[f"target_bid_{block_name}"] = float(
                        self.final_bids[scenario_id][block_idx][time_id]
                    )
                rows.append(self._select_output_columns(row, block_indices))

        dataframe = pd.DataFrame(rows)
        self._validate_generator_dataset(generator_name, dataframe, block_indices)
        return dataframe

    def save_datasets(
        self,
        output_dir: str | Path = "models/neural_network/features/generated",
        file_format: str = "csv",
        normalize: bool = False,
        per_generator_normalization: bool = True,
        save_stats: bool = True,
    ) -> dict[str, Path]:
        file_format = file_format.lower().lstrip(".")
        if file_format not in {"csv", "parquet"}:
            raise ValueError("file_format must be either 'csv' or 'parquet'")

        datasets = self.build_all_generator_datasets(
            normalize=normalize,
            per_generator_normalization=per_generator_normalization,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        for generator_name, dataframe in datasets.items():
            suffix = "_features_normalized" if normalize else "_features"
            file_name = f"{self._sanitize_filename(generator_name)}{suffix}.{file_format}"
            path = output_path / file_name
            if file_format == "csv":
                dataframe.to_csv(path, index=False)
            else:
                dataframe.to_parquet(path, index=False)
            saved_paths[generator_name] = path

        if normalize and save_stats:
            if self._normalization_stats is None:
                raise ValueError("Normalization stats were not computed.")
            self.save_normalization_stats(
                self._normalization_stats,
                output_path / "min_max_stats.json",
            )
        return saved_paths

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [column for column in self.feature_columns if column in df.columns]

    def fit_min_max_stats(
        self,
        datasets: dict[str, pd.DataFrame],
        per_generator: bool = True,
    ) -> dict[str, MinMaxNormalizationStats] | MinMaxNormalizationStats:
        if per_generator:
            return {
                generator_name: self._fit_min_max_stats_for_dataframe(dataframe)
                for generator_name, dataframe in datasets.items()
            }

        feature_columns = sorted(
            {
                column
                for dataframe in datasets.values()
                for column in self.get_feature_columns(dataframe)
            }
        )
        feature_min: dict[str, float] = {}
        feature_max: dict[str, float] = {}
        for column in feature_columns:
            values = pd.concat(
                [
                    dataframe[column]
                    for dataframe in datasets.values()
                    if column in dataframe.columns
                ],
                ignore_index=True,
            )
            feature_min[column] = float(values.min())
            feature_max[column] = float(values.max())
        return MinMaxNormalizationStats(feature_min=feature_min, feature_max=feature_max)

    def normalize_datasets(
        self,
        datasets: dict[str, pd.DataFrame],
        stats: dict[str, MinMaxNormalizationStats] | MinMaxNormalizationStats,
        per_generator: bool = True,
    ) -> dict[str, pd.DataFrame]:
        if per_generator:
            if not isinstance(stats, dict):
                raise ValueError("Per-generator normalization requires a stats dictionary.")
            missing_generators = sorted(set(datasets) - set(stats))
            if missing_generators:
                raise ValueError(
                    "Missing normalization stats for generators: "
                    f"{missing_generators}"
                )
            normalized = {
                generator_name: self._normalize_dataframe(
                    dataframe,
                    stats[generator_name],
                )
                for generator_name, dataframe in datasets.items()
            }
        else:
            if isinstance(stats, dict):
                raise ValueError("Shared normalization requires shared stats.")
            normalized = {
                generator_name: self._normalize_dataframe(dataframe, stats)
                for generator_name, dataframe in datasets.items()
            }

        self._validate_normalized_datasets(datasets, normalized)
        return normalized

    def save_normalization_stats(
        self,
        stats: dict[str, MinMaxNormalizationStats] | MinMaxNormalizationStats,
        output_path: str | Path = (
            "models/neural_network/features/generated/min_max_stats.json"
        ),
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(stats, dict):
            payload = {
                "per_generator": True,
                "stats": {
                    generator_name: self._normalization_stats_to_json(stats_for_generator)
                    for generator_name, stats_for_generator in stats.items()
                },
            }
        else:
            payload = {
                "per_generator": False,
                "stats": self._normalization_stats_to_json(stats),
            }

        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
        return path

    @staticmethod
    def _load_results(results_path: Path) -> dict[str, Any]:
        if not results_path.exists():
            raise ValueError(f"Results file does not exist: {results_path}")
        with results_path.open("r", encoding="utf-8") as file_handle:
            results = json.load(file_handle)

        required_fields = {
            "physical_generator_names",
            "block_names",
            "block_to_physical",
            "final_bids",
            "num_scenarios",
            "num_time_steps",
        }
        missing = sorted(required_fields - set(results))
        if missing:
            raise ValueError(f"Results file is missing required fields: {missing}")
        return results

    def _build_physical_to_block_indices(self) -> dict[str, list[int]]:
        mapping = {name: [] for name in self.physical_generator_names}
        for block_idx, block_name in enumerate(self.block_names):
            physical_name = self.block_to_physical.get(block_name, block_name)
            if physical_name not in mapping:
                mapping[physical_name] = []
            mapping[physical_name].append(block_idx)
        return mapping

    def _compute_market_features(self) -> list[list[dict[str, float]]]:
        features_by_scenario: list[list[dict[str, float]]] = []
        for scenario_id in range(self.num_scenarios):
            demand_profile = self._profile_value(
                self.scenarios_df.at[scenario_id, "demand_profile"],
                "demand_profile",
                self.num_time_steps,
            )
            total_generation_capacity = [
                sum(
                    self.available_capacity(scenario_id, block_name, time_id)
                    for block_name in self.block_names
                )
                for time_id in range(self.num_time_steps)
            ]
            total_wind_generation_capacity = [
                sum(
                    self.available_capacity(scenario_id, block_name, time_id)
                    for block_name in self.block_names
                    if self._is_wind_block(block_name)
                )
                for time_id in range(self.num_time_steps)
            ]

            scenario_features: list[dict[str, float]] = []
            for time_id in range(self.num_time_steps):
                previous_time_id = (time_id - 1) % self.num_time_steps
                next_time_id = (time_id + 1) % self.num_time_steps
                demand = float(demand_profile[time_id])
                wind_capacity = float(total_wind_generation_capacity[time_id])
                generation_capacity = float(total_generation_capacity[time_id])
                scenario_features.append(
                    {
                        "demand": demand,
                        "total_wind_generation_capacity": wind_capacity,
                        "total_generation_capacity": generation_capacity,
                        "residual_demand": demand - wind_capacity,
                        "previous_generation_capacity": float(
                            total_generation_capacity[previous_time_id]
                        ),
                        "previous_demand": float(demand_profile[previous_time_id]),
                        "next_generation_capacity": float(
                            total_generation_capacity[next_time_id]
                        ),
                        "next_demand": float(demand_profile[next_time_id]),
                    }
                )
            features_by_scenario.append(scenario_features)
        return features_by_scenario

    def _compute_own_capacity(self, block_indices: list[int]) -> np.ndarray:
        own_capacity = np.zeros((self.num_scenarios, self.num_time_steps), dtype=np.float64)
        for scenario_id in range(self.num_scenarios):
            for time_id in range(self.num_time_steps):
                own_capacity[scenario_id, time_id] = sum(
                    self.available_capacity(scenario_id, self.block_names[block_idx], time_id)
                    for block_idx in block_indices
                )
        return own_capacity

    def _compute_true_cost_features(self, block_indices: list[int]) -> dict[str, float]:
        costs = [
            float(self.costs_df[f"{self.block_names[block_idx]}_cost"].iloc[0])
            for block_idx in block_indices
        ]
        return {
            "average_true_cost": float(np.mean(costs)),
            "minimum_true_cost": float(np.min(costs)),
            "maximum_true_cost": float(np.max(costs)),
        }

    def available_capacity(
        self,
        scenario_id: int,
        block_name: str,
        time_id: int,
    ) -> float:
        profile_col = f"{block_name}_cap_profile"
        availability_profile_col = f"{block_name}_profile"
        if profile_col in self.scenarios_df.columns:
            profile = self._profile_value(
                self.scenarios_df.at[scenario_id, profile_col],
                profile_col,
                self.num_time_steps,
            )
            return float(profile[time_id])
        if availability_profile_col in self.scenarios_df.columns:
            profile = self._profile_value(
                self.scenarios_df.at[scenario_id, availability_profile_col],
                availability_profile_col,
                self.num_time_steps,
            )
            return float(profile[time_id])

        static_col = f"{block_name}_cap"
        if static_col not in self.scenarios_df.columns:
            raise ValueError(
                f"No available-capacity column found for block '{block_name}'. "
                f"Expected one of '{profile_col}', '{availability_profile_col}', or '{static_col}'."
            )
        return float(self.scenarios_df.at[scenario_id, static_col])

    @staticmethod
    def _profile_value(value: Any, column_name: str, expected_len: int) -> list[float]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Column '{column_name}' must contain a list, tuple, numpy array, "
                "or string representation of a list."
            )
        profile = [float(v) for v in value]
        if len(profile) != expected_len:
            raise ValueError(
                f"Column '{column_name}' profile length must be {expected_len}, "
                f"got {len(profile)}."
            )
        return profile

    def _is_wind_block(self, block_name: str) -> bool:
        physical_name = self.block_to_physical.get(block_name, block_name)
        return (
            block_name.startswith("W")
            or physical_name.startswith("W")
            or "wind" in physical_name.lower()
        )

    def _validate_generator_dataset(
        self,
        generator_name: str,
        dataframe: pd.DataFrame,
        block_indices: list[int],
    ) -> None:
        expected_rows = self.num_scenarios * self.num_time_steps
        if len(dataframe) != expected_rows:
            raise ValueError(
                f"Generated dataframe for '{generator_name}' has {len(dataframe)} rows; "
                f"expected {expected_rows}."
            )

        feature_columns = self.feature_columns
        target_columns = [
            f"target_bid_{self.block_names[block_idx]}" for block_idx in block_indices
        ]
        missing_columns = [
            col for col in feature_columns + target_columns if col not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Generated dataframe for '{generator_name}' is missing columns: "
                f"{missing_columns}"
            )
        if dataframe[feature_columns + target_columns].isna().any().any():
            nan_columns = dataframe[feature_columns + target_columns].columns[
                dataframe[feature_columns + target_columns].isna().any()
            ].tolist()
            raise ValueError(
                f"Generated dataframe for '{generator_name}' contains NaN values "
                f"in columns: {nan_columns}"
            )

    def _resolve_feature_columns(self, feature_columns: list[str] | None) -> list[str]:
        selected = (
            self.BASE_FEATURE_COLUMNS + self.OWN_FEATURE_COLUMNS
            if feature_columns is None
            else list(feature_columns)
        )
        unknown = [
            column
            for column in selected
            if column not in self.SUPPORTED_FEATURE_COLUMNS
        ]
        if unknown:
            raise ValueError(
                "Unsupported NN feature columns: "
                f"{unknown}. Supported columns: {self.SUPPORTED_FEATURE_COLUMNS}"
            )
        duplicates = sorted({column for column in selected if selected.count(column) > 1})
        if duplicates:
            raise ValueError(f"Duplicate NN feature columns are not allowed: {duplicates}")
        if not selected:
            raise ValueError("At least one NN feature column must be selected")
        return selected

    def _select_output_columns(
        self,
        row: dict[str, Any],
        block_indices: list[int],
    ) -> dict[str, Any]:
        target_columns = [
            f"target_bid_{self.block_names[block_idx]}" for block_idx in block_indices
        ]
        ordered_columns = [
            "scenario_id",
            "time_id",
            "generator_name",
            *self.feature_columns,
            *target_columns,
        ]
        missing = [column for column in ordered_columns if column not in row]
        if missing:
            raise ValueError(f"Could not build selected NN feature columns: {missing}")
        return {column: row[column] for column in ordered_columns}

    def _fit_min_max_stats_for_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> MinMaxNormalizationStats:
        feature_columns = self.get_feature_columns(dataframe)
        return MinMaxNormalizationStats(
            feature_min={
                column: float(dataframe[column].min()) for column in feature_columns
            },
            feature_max={
                column: float(dataframe[column].max()) for column in feature_columns
            },
        )

    def _normalize_dataframe(
        self,
        dataframe: pd.DataFrame,
        stats: MinMaxNormalizationStats,
    ) -> pd.DataFrame:
        normalized = dataframe.copy(deep=True)
        for column in self.get_feature_columns(dataframe):
            if column not in stats.feature_min or column not in stats.feature_max:
                raise ValueError(f"Missing min-max stats for feature column '{column}'.")
            minimum = stats.feature_min[column]
            maximum = stats.feature_max[column]
            denominator = maximum - minimum
            if denominator == 0:
                normalized[column] = 0.0
            else:
                normalized[column] = (dataframe[column] - minimum) / denominator
        return normalized

    def _validate_normalized_datasets(
        self,
        raw_datasets: dict[str, pd.DataFrame],
        normalized_datasets: dict[str, pd.DataFrame],
    ) -> None:
        for generator_name, raw_dataframe in raw_datasets.items():
            normalized_dataframe = normalized_datasets[generator_name]
            feature_columns = self.get_feature_columns(raw_dataframe)
            target_columns = [
                column
                for column in raw_dataframe.columns
                if column.startswith(self.TARGET_COLUMN_PREFIX)
            ]
            unchanged_columns = [
                column
                for column in self.METADATA_COLUMNS
                if column in raw_dataframe.columns
            ] + target_columns

            for column in feature_columns:
                below_zero = normalized_dataframe[column] < -self.NORMALIZATION_TOLERANCE
                above_one = normalized_dataframe[column] > 1.0 + self.NORMALIZATION_TOLERANCE
                if below_zero.any() or above_one.any():
                    raise ValueError(
                        f"Normalized feature column '{column}' for '{generator_name}' "
                        "contains values outside [0, 1]."
                    )

            for column in unchanged_columns:
                if not normalized_dataframe[column].equals(raw_dataframe[column]):
                    raise ValueError(
                        f"Column '{column}' for '{generator_name}' was altered during "
                        "normalization."
                    )

    @staticmethod
    def _normalization_stats_to_json(stats: MinMaxNormalizationStats) -> dict[str, Any]:
        return {
            "feature_min": {
                column: float(value) for column, value in stats.feature_min.items()
            },
            "feature_max": {
                column: float(value) for column, value in stats.feature_max.items()
            },
        }

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
        return sanitized.strip("._") or "generator"
