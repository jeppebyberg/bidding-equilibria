from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pyomo.environ import (
    Binary,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Reals,
    Set,
    Var,
)

from models.helper import target_columns_to_local_blocks


class DROPoAPolicyEmbedding:
    def _configure_nn_policy_generators(self) -> None:
        if self.nn_model_dir is None:
            self.nn_policy_generator_ids = []
            self.nn_policy_generator_names = []
            return

        requested = self.requested_nn_policy_generators
        if requested is None:
            ids = list(range(self.num_physical_generators))
        else:
            ids = []
            for raw_generator in requested:
                if isinstance(raw_generator, str):
                    generator_name = raw_generator.strip()
                    if generator_name not in self.physical_generator_names:
                        raise ValueError(
                            f"Unknown NN policy generator '{raw_generator}'. "
                            f"Available: {self.physical_generator_names}"
                        )
                    generator_idx = self.physical_generator_names.index(generator_name)
                else:
                    generator_idx = int(raw_generator)
                    if not 0 <= generator_idx < self.num_physical_generators:
                        raise ValueError(
                            f"NN policy generator index {generator_idx} is outside "
                            f"0..{self.num_physical_generators - 1}"
                        )
                if generator_idx not in ids:
                    ids.append(generator_idx)

        self.nn_policy_generator_ids = ids
        self.nn_policy_generator_names = [
            self.physical_generator_names[i] for i in self.nn_policy_generator_ids
        ]

    def _load_nn_policies(self) -> None:
        if self.nn_model_dir is None or not self.nn_model_dir.exists():
            raise FileNotFoundError(f"NN model directory not found: {self.nn_model_dir}")
        self.nn_policies = {}
        for generator_name in self.nn_policy_generator_names:
            weights_path = self.nn_model_dir / f"{generator_name}_policy_weights.json"
            metadata_path = self.nn_model_dir / f"{generator_name}_policy_metadata.json"
            if not weights_path.exists():
                raise FileNotFoundError(f"Missing NN weights file: {weights_path}")
            with weights_path.open("r", encoding="utf-8") as file_handle:
                weights = json.load(file_handle)
            metadata = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as file_handle:
                    metadata = json.load(file_handle)

            feature_columns = list(
                weights.get("feature_columns") or metadata.get("feature_columns") or []
            )
            target_columns = list(
                weights.get("target_columns") or metadata.get("target_columns") or []
            )
            layers = list(weights.get("layers", []))
            if not feature_columns or not target_columns or not layers:
                raise ValueError(f"Invalid NN policy payload for {generator_name}")
            self._validate_nn_policy(generator_name, feature_columns, target_columns, layers)
            self.nn_policies[generator_name] = {
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "layers": layers,
                "metadata": metadata,
                "target_map": target_columns_to_local_blocks(
                    generator_name=generator_name,
                    target_columns=target_columns,
                    block_names=self.block_names,
                    physical_generator_names=self.physical_generator_names,
                    global_to_local_block=self.global_to_local_block,
                    local_blocks_by_generator=self.local_blocks_by_generator,
                ),
            }

    def _validate_nn_policy(
        self,
        generator_name: str,
        feature_columns: list[str],
        target_columns: list[str],
        layers: list[dict[str, Any]],
    ) -> None:
        expected_input = len(feature_columns)
        current_dim = expected_input
        previous_was_hidden_linear = False
        linear_count = 0
        for idx, layer in enumerate(layers):
            layer_type = str(layer.get("type", "")).lower()
            if layer_type == "linear":
                weight = np.asarray(layer.get("weight"), dtype=float)
                bias = np.asarray(layer.get("bias"), dtype=float)
                if weight.ndim != 2 or bias.ndim != 1:
                    raise ValueError(f"{generator_name}: linear layer {idx} has invalid dimensions")
                if weight.shape[1] != current_dim or weight.shape[0] != bias.shape[0]:
                    raise ValueError(f"{generator_name}: inconsistent dimensions in linear layer {idx}")
                current_dim = int(weight.shape[0])
                previous_was_hidden_linear = idx < len(layers) - 1
                linear_count += 1
            elif layer_type == "relu":
                if not previous_was_hidden_linear:
                    raise ValueError(
                        f"{generator_name}: ReLU layer {idx} must follow a hidden linear layer"
                    )
                previous_was_hidden_linear = False
            else:
                raise ValueError(f"{generator_name}: unsupported layer type '{layer_type}'")
        if str(layers[-1].get("type", "")).lower() != "linear":
            raise ValueError(f"{generator_name}: final NN layer must be linear")
        if current_dim != len(target_columns):
            raise ValueError(
                f"{generator_name}: output dimension {current_dim} does not match "
                f"{len(target_columns)} target columns"
            )
        if linear_count < 1:
            raise ValueError(f"{generator_name}: NN must contain at least one linear layer")

    def _load_nn_normalization_stats(self) -> None:
        if self.nn_normalization_stats_path is None:
            self.nn_stats = {}
            return
        if not self.nn_normalization_stats_path.exists():
            raise FileNotFoundError(
                f"NN normalization stats not found: {self.nn_normalization_stats_path}"
            )
        with self.nn_normalization_stats_path.open("r", encoding="utf-8") as file_handle:
            self.nn_stats = json.load(file_handle)

    def _nn_feature_bounds(self, generator_name: str, feature_name: str) -> tuple[float, float]:
        stats = self.nn_stats or {}
        if bool(stats.get("per_generator")):
            generator_stats = stats.get("stats", {}).get(generator_name, {})
            mins = generator_stats.get("feature_min", {})
            maxs = generator_stats.get("feature_max", {})
            if feature_name in mins and feature_name in maxs:
                return float(mins[feature_name]), float(maxs[feature_name])
        if "feature_min" in stats and "feature_max" in stats:
            mins = stats["feature_min"]
            maxs = stats["feature_max"]
            if isinstance(mins, dict) and feature_name in mins:
                return float(mins[feature_name]), float(maxs[feature_name])
        return 0.0, 1.0

    @staticmethod
    def _parse_nn_relu_bound_key(key: str) -> tuple[int, ...]:
        parts = tuple(int(part) for part in str(key).split(",") if part != "")
        if len(parts) not in {3, 4}:
            raise ValueError(
                f"NN ReLU bound key '{key}' must have the form "
                "'time_idx,linear_idx,node' or 'k,time_idx,linear_idx,node'"
            )
        return tuple(int(part) for part in parts)

    def _set_nn_relu_bounds_from_report(self, report: dict[str, Any]) -> None:
        self.nn_relu_bounds_report = report
        self.nn_feature_bounds = report.get("nn_feature_bounds", {}) or {}
        self.nn_bound_warnings = list(
            report.get("warnings", report.get("nn_bound_warnings", [])) or []
        )
        relu_entries = report.get("scenario_nn_relu_bounds", {}) or report.get(
            "nn_relu_bounds",
            {},
        )
        parsed_bounds: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {}
        for generator_name, entries in (relu_entries or {}).items():
            generator_bounds: dict[tuple[int, ...], dict[str, Any]] = {}
            for key, details in (entries or {}).items():
                parsed_key = self._parse_nn_relu_bound_key(key)
                time_idx, linear_idx, node = parsed_key[-3:]
                parsed_details = dict(details or {})
                for numeric_key in ("L", "U", "h_lower", "h_upper"):
                    if numeric_key not in parsed_details:
                        raise ValueError(
                            f"NN ReLU bound entry {generator_name}[{key}] is missing "
                            f"'{numeric_key}'"
                        )
                    parsed_details[numeric_key] = float(parsed_details[numeric_key])
                status = str(parsed_details.get("status", "ambiguous")).lower()
                if status not in {"active", "inactive", "ambiguous"}:
                    raise ValueError(
                        f"NN ReLU bound entry {generator_name}[{key}] has invalid "
                        f"status '{status}'"
                    )
                parsed_details["status"] = status
                if len(parsed_key) == 4:
                    parsed_details.setdefault("scenario_idx", parsed_key[0])
                parsed_details.setdefault("time_idx", time_idx)
                parsed_details.setdefault("linear_idx", linear_idx)
                parsed_details.setdefault("node", node)
                generator_bounds[parsed_key] = parsed_details
            parsed_bounds[str(generator_name)] = generator_bounds
        self.nn_relu_bounds = parsed_bounds

    def load_nn_relu_bounds_report(
        self,
        report_path: str | Path = "results/dro_poa_tightening/final_tightening_report.json",
    ) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"NN ReLU bounds report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        relu_report = report.get("nn_relu_bounds_report", {}) or report
        self.nn_relu_bounds_report_path = path
        self._set_nn_relu_bounds_from_report(relu_report)
        return relu_report

    def _build_policy_constraints(self) -> None:
        m = self.model

        def true_cost_alpha_rule(m, k, i, b, t):
            if int(i) in self.nn_policy_generator_ids:
                return Constraint.Skip
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.alpha[k, i, b, t] == self.block_cost_vector[global_block]

        m.true_cost_alpha = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=true_cost_alpha_rule,
        )

        if self.nn_policy_generator_ids:
            self._build_nn_policy_constraints()
        else:
            m.nn_constraints = ConstraintList()

    def _raw_nn_feature_expression(
        self,
        feature_name: str,
        k: int,
        t: int,
        physical_generator_idx: int,
    ):
        m = self.model
        k = int(k)
        t = int(t)
        physical_generator_idx = int(physical_generator_idx)
        previous_t = self.num_time_steps - 1 if t == 0 else t - 1
        next_t = 0 if t == self.num_time_steps - 1 else t + 1

        total_wind_capacity = lambda time_idx: sum(
            m.P_max_block[k, i, b, time_idx] for (i, b) in self.wind_block_pairs
        )
        total_capacity = lambda time_idx: sum(
            m.P_max_block[k, i, b, time_idx] for (i, b) in self.generator_block_pairs
        )
        own_capacity = lambda time_idx: sum(
            m.P_max_block[k, physical_generator_idx, b, time_idx]
            for b in self.local_blocks_by_generator[physical_generator_idx]
        )

        if feature_name == "demand":
            return m.D[k, t]
        if feature_name == "total_wind_generation_capacity":
            return total_wind_capacity(t)
        if feature_name == "total_generation_capacity":
            return total_capacity(t)
        if feature_name == "residual_demand":
            return m.D[k, t] - total_wind_capacity(t)
        if feature_name == "previous_generation_capacity":
            return total_capacity(previous_t)
        if feature_name == "previous_demand":
            return m.D[k, previous_t]
        if feature_name == "next_generation_capacity":
            return total_capacity(next_t)
        if feature_name == "next_demand":
            return m.D[k, next_t]
        if feature_name == "own_generation_capacity":
            return own_capacity(t)
        if feature_name == "previous_own_generation_capacity":
            return own_capacity(previous_t)
        if feature_name == "next_own_generation_capacity":
            return own_capacity(next_t)
        if feature_name == "average_true_cost":
            costs = [
                self.block_cost_vector[
                    self.local_to_global_block[(physical_generator_idx, b)]
                ]
                for b in self.local_blocks_by_generator[physical_generator_idx]
            ]
            return float(np.mean(costs))
        if feature_name == "minimum_true_cost":
            return float(
                min(
                    self.block_cost_vector[
                        self.local_to_global_block[(physical_generator_idx, b)]
                    ]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        if feature_name == "maximum_true_cost":
            return float(
                max(
                    self.block_cost_vector[
                        self.local_to_global_block[(physical_generator_idx, b)]
                    ]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        raise ValueError(f"Unsupported NN feature name: {feature_name}")

    def _normalized_nn_feature_expression(
        self,
        generator_name: str,
        feature_name: str,
        k: int,
        t: int,
        physical_generator_idx: int,
    ):
        raw = self._raw_nn_feature_expression(feature_name, k, t, physical_generator_idx)
        feature_min, feature_max = self._nn_feature_bounds(generator_name, feature_name)
        denominator = feature_max - feature_min
        if abs(denominator) <= self.normalization_epsilon:
            return 0.0
        return (raw - feature_min) / denominator

    def _build_nn_policy_constraints(self) -> None:
        m = self.model
        if not self.nn_relu_bounds:
            raise ValueError(
                "NN ReLU bounds are required before building NN policy constraints. "
                "Run the DRO ReLU-bound tightening stage or call "
                "load_nn_relu_bounds_report(...)."
            )
        nn_input_indices: list[tuple[int, int, int, int]] = []
        nn_z_indices: list[tuple[int, int, int, int, int]] = []
        nn_h_indices: list[tuple[int, int, int, int, int]] = []
        nn_output_indices: list[tuple[int, int, int, int]] = []

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(i)]
            if generator_name not in self.nn_policies:
                raise ValueError(f"NN policy for generator {generator_name} was not loaded.")
            if generator_name not in self.nn_relu_bounds:
                raise ValueError(
                    "NN ReLU bounds report is missing bounds for generator "
                    f"{generator_name}."
                )
            policy = self.nn_policies[generator_name]
            for k in range(self.num_empirical_scenarios):
                for t in range(self.num_time_steps):
                    for f_idx, _ in enumerate(policy["feature_columns"]):
                        nn_input_indices.append((int(k), int(i), int(t), int(f_idx)))

            linear_idx = 0
            layers = policy["layers"]
            for layer_pos, layer in enumerate(layers):
                if str(layer.get("type", "")).lower() != "linear":
                    continue
                output_dim = len(layer["bias"])
                is_final_linear = layer_pos == len(layers) - 1
                for k in range(self.num_empirical_scenarios):
                    for t in range(self.num_time_steps):
                        for node in range(output_dim):
                            if is_final_linear:
                                nn_output_indices.append((int(k), int(i), int(t), int(node)))
                            else:
                                nn_z_indices.append(
                                    (int(k), int(i), int(t), int(linear_idx), int(node))
                                )
                                nn_h_indices.append(
                                    (int(k), int(i), int(t), int(linear_idx), int(node))
                                )
                linear_idx += 1

        m.nn_input_index = Set(dimen=4, initialize=nn_input_indices)
        m.nn_z_index = Set(dimen=5, initialize=nn_z_indices)
        m.nn_h_index = Set(dimen=5, initialize=nn_h_indices)
        m.nn_delta_index = Set(dimen=5, initialize=nn_h_indices)
        m.nn_output_index = Set(dimen=4, initialize=nn_output_indices)
        m.nn_input = Var(m.nn_input_index, domain=Reals)
        m.nn_z = Var(m.nn_z_index, domain=Reals)
        m.nn_h = Var(m.nn_h_index, domain=NonNegativeReals)
        m.nn_delta = Var(m.nn_delta_index, domain=Binary)
        m.nn_output = Var(m.nn_output_index, domain=Reals)
        m.nn_constraints = ConstraintList()

        def relu_bound_for_node(
            relu_bounds: dict[tuple[int, ...], dict[str, Any]],
            k: int,
            t: int,
            linear_idx: int,
            node: int,
            generator_name: str,
        ) -> dict[str, Any]:
            scenario_key = (int(k), int(t), int(linear_idx), int(node))
            regime_key = (int(t), int(linear_idx), int(node))
            if scenario_key in relu_bounds:
                return relu_bounds[scenario_key]
            if regime_key in relu_bounds:
                return relu_bounds[regime_key]
            raise ValueError(
                f"NN ReLU bounds report is missing bounds for {generator_name} "
                f"at scenario {k}, time {t}, linear layer {linear_idx}, node {node}."
            )

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(i)]
            relu_bounds = self.nn_relu_bounds[generator_name]
            for raw_key, bounds in relu_bounds.items():
                key = tuple(int(part) for part in raw_key)
                scenario_keys = (
                    [key]
                    if len(key) == 4
                    else [(int(k), *key) for k in m.scenarios]
                )
                for scenario_key in scenario_keys:
                    k, time_idx, linear_idx, node = scenario_key
                    index = (int(k), int(i), int(time_idx), int(linear_idx), int(node))
                    if index not in m.nn_z:
                        continue
                    m.nn_z[index].setlb(float(bounds["L"]))
                    m.nn_z[index].setub(float(bounds["U"]))
                    m.nn_h[index].setlb(float(bounds["h_lower"]))
                    m.nn_h[index].setub(float(bounds["h_upper"]))

        for i in self.nn_policy_generator_ids:
            i = int(i)
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            relu_bounds = self.nn_relu_bounds[generator_name]
            feature_columns = policy["feature_columns"]
            for k in range(self.num_empirical_scenarios):
                for t in range(self.num_time_steps):
                    for f_idx, feature_name in enumerate(feature_columns):
                        m.nn_constraints.add(
                            m.nn_input[k, i, t, f_idx]
                            == self._normalized_nn_feature_expression(
                                generator_name,
                                feature_name,
                                k,
                                t,
                                i,
                            )
                        )

                    previous_values = [
                        m.nn_input[k, i, t, f_idx]
                        for f_idx in range(len(feature_columns))
                    ]
                    linear_idx = 0
                    for layer_pos, layer in enumerate(policy["layers"]):
                        if str(layer.get("type", "")).lower() == "relu":
                            continue
                        weights = np.asarray(layer["weight"], dtype=float)
                        bias = np.asarray(layer["bias"], dtype=float)
                        is_final_linear = layer_pos == len(policy["layers"]) - 1
                        current_values = []
                        for node in range(weights.shape[0]):
                            expr = float(bias[node]) + sum(
                                float(weights[node, prev_idx]) * previous_values[prev_idx]
                                for prev_idx in range(weights.shape[1])
                            )
                            if is_final_linear:
                                m.nn_constraints.add(m.nn_output[k, i, t, node] == expr)
                                current_values.append(m.nn_output[k, i, t, node])
                                continue

                            m.nn_constraints.add(
                                m.nn_z[k, i, t, linear_idx, node] == expr
                            )
                            h = m.nn_h[k, i, t, linear_idx, node]
                            z = m.nn_z[k, i, t, linear_idx, node]
                            delta = m.nn_delta[k, i, t, linear_idx, node]
                            bounds = relu_bound_for_node(
                                relu_bounds,
                                int(k),
                                int(t),
                                int(linear_idx),
                                int(node),
                                generator_name,
                            )
                            status = str(bounds["status"]).lower()
                            if status == "inactive":
                                m.nn_constraints.add(h == 0)
                                delta.fix(0)
                            elif status == "active":
                                m.nn_constraints.add(h == z)
                                delta.fix(1)
                            elif status == "ambiguous":
                                L = float(bounds["L"])
                                U = float(bounds["U"])
                                m.nn_constraints.add(h >= z)
                                m.nn_constraints.add(h >= 0)
                                m.nn_constraints.add(h <= z - L * (1 - delta))
                                m.nn_constraints.add(h <= U * delta)
                            else:
                                raise ValueError(
                                    f"{generator_name}: unknown ReLU bound status '{status}'"
                                )
                            current_values.append(h)
                        previous_values = current_values
                        linear_idx += 1

            for output_idx, local_block in policy["target_map"].items():
                for k in m.scenarios:
                    for t in m.time_steps:
                        m.nn_constraints.add(
                            m.alpha[int(k), i, int(local_block), int(t)]
                            == m.nn_output[int(k), i, int(t), int(output_idx)]
                        )

    # ------------------------------------------------------------------
    # Applying regime-wide tightening reports
    # ------------------------------------------------------------------

    def summarize_nn_feature_bounds(self) -> dict[str, Any]:
        return dict(self.nn_feature_bounds or {})

    def summarize_nn_relu_bounds(self) -> dict[str, Any]:
        report_summary = (self.nn_relu_bounds_report or {}).get("summary")
        if isinstance(report_summary, dict):
            return report_summary
        if not self.nn_relu_bounds:
            return {}

        summary: dict[str, Any] = {}
        for generator_name, bounds in self.nn_relu_bounds.items():
            values = list(bounds.values())
            if not values:
                continue
            L_values = [float(item["L"]) for item in values]
            U_values = [float(item["U"]) for item in values]
            summary[generator_name] = {
                "num_hidden_neurons_time_indexed": len(values),
                "num_active": sum(1 for item in values if item["status"] == "active"),
                "num_inactive": sum(1 for item in values if item["status"] == "inactive"),
                "num_ambiguous": sum(
                    1 for item in values if item["status"] == "ambiguous"
                ),
                "min_L": float(min(L_values)),
                "max_L": float(max(L_values)),
                "min_U": float(min(U_values)),
                "max_U": float(max(U_values)),
                "max_M_minus": float(max(max(0.0, -L) for L in L_values)),
                "max_M_plus": float(max(max(0.0, U) for U in U_values)),
            }
        return summary

    # ------------------------------------------------------------------
    # Solve and results
    # ------------------------------------------------------------------

