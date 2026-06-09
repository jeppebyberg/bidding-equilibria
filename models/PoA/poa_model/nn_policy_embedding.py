import copy
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pyomo.environ import Binary, Constraint, ConstraintList, NonNegativeReals, Reals, Set, Var

from models.helper import target_columns_to_local_blocks


class PoAPolicyEmbedding:
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

            feature_columns = list(weights.get("feature_columns") or metadata.get("feature_columns") or [])
            target_columns = list(weights.get("target_columns") or metadata.get("target_columns") or [])
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
        self, generator_name: str, feature_columns: list[str], target_columns: list[str], layers: list[dict[str, Any]]
    ) -> None:
        expected_input = len(feature_columns)
        current_dim = expected_input
        previous_was_linear = False
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
                previous_was_linear = True
                linear_count += 1
            elif layer_type == "relu":
                if not previous_was_linear:
                    raise ValueError(f"{generator_name}: ReLU layer {idx} must follow a linear layer")
                previous_was_linear = False
            else:
                raise ValueError(f"{generator_name}: unsupported layer type '{layer_type}'")
        if str(layers[-1].get("type", "")).lower() not in {"linear", "relu"}:
            raise ValueError(f"{generator_name}: final NN layer must be linear or ReLU")
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
    def _parse_nn_relu_bound_key(key: str) -> tuple[int, int, int]:
        parts = tuple(int(part) for part in str(key).split(",") if part != "")
        if len(parts) != 3:
            raise ValueError(
                f"NN ReLU bound key '{key}' must have the form 'time_idx,linear_idx,node'"
            )
        return int(parts[0]), int(parts[1]), int(parts[2])

    def _set_nn_relu_bounds_from_report(self, report: dict[str, Any]) -> None:
        self.nn_relu_bounds_report = report
        self.nn_feature_bounds = report.get("nn_feature_bounds", {}) or {}
        self.nn_bound_warnings = list(
            report.get("warnings", report.get("nn_bound_warnings", [])) or []
        )

        parsed_bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
        for generator_name, entries in (report.get("nn_relu_bounds", {}) or {}).items():
            generator_bounds: dict[tuple[int, int, int], dict[str, Any]] = {}
            for key, details in (entries or {}).items():
                if not isinstance(details, dict):
                    raise ValueError(
                        f"Invalid NN ReLU bound entry for {generator_name}, key {key}"
                    )
                time_idx, linear_idx, node = self._parse_nn_relu_bound_key(key)
                parsed_details = dict(details)
                for numeric_key in ("L", "U", "h_lower", "h_upper"):
                    if numeric_key not in parsed_details:
                        raise ValueError(
                            f"NN ReLU bound entry {generator_name}[{key}] is missing "
                            f"'{numeric_key}'"
                        )
                    parsed_details[numeric_key] = float(parsed_details[numeric_key])
                parsed_details["status"] = str(parsed_details.get("status", "")).lower()
                if parsed_details["status"] not in {"active", "inactive", "ambiguous"}:
                    raise ValueError(
                        f"NN ReLU bound entry {generator_name}[{key}] has invalid "
                        f"status '{parsed_details['status']}'"
                    )
                parsed_details.setdefault("time_idx", time_idx)
                parsed_details.setdefault("linear_idx", linear_idx)
                parsed_details.setdefault("node", node)
                generator_bounds[(time_idx, linear_idx, node)] = parsed_details
            parsed_bounds[str(generator_name)] = generator_bounds

        self.nn_relu_bounds = parsed_bounds

    def load_nn_relu_bounds_report(
        self,
        report_path: str | Path = "results/poa_nn_relu_bounds_report.json",
    ) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"NN ReLU bounds report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)

        self.nn_relu_bounds_report_path = path
        if hasattr(self, "load_tightening_report_data"):
            self.load_tightening_report_data(
                {
                    "nn_relu_bounds_report": report,
                    "nn_relu_bounds": report.get("nn_relu_bounds", {}),
                },
                report_path=getattr(self, "tightening_report_path", None),
                prepare_bounds=False,
                merge_existing=True,
            )
        else:
            self._set_nn_relu_bounds_from_report(report)
        return report

    def build_default_nn_relu_bounds_report(
        self,
        template_report: Optional[dict[str, Any]] = None,
        margin: Optional[float] = None,
    ) -> dict[str, Any]:
        margin = float(self.default_relu_bound if margin is None else margin)
        source = self.nn_relu_bounds_report if template_report is None else template_report
        if not source:
            if self.nn_policy_generator_ids:
                if not self.nn_policies:
                    self._load_nn_policies()
                if not self.nn_stats:
                    self._load_nn_normalization_stats()
                report = {
                    "metadata": {
                        "reference_case": self.reference_case,
                        "num_time_steps": self.num_time_steps,
                        "nn_policy_generators": list(self.nn_policy_generator_names),
                        "physical_generator_names": list(self.physical_generator_names),
                        "nn_model_dir": str(self.nn_model_dir),
                        "nn_normalization_stats_path": str(
                            self.nn_normalization_stats_path
                        ),
                        "bound_methods": {
                            "all_layers": "default_loose_bounds",
                        },
                    },
                    "nn_feature_bounds": self.summarize_nn_feature_bounds(),
                    "nn_relu_bounds": {},
                    "summary": {},
                    "warnings": [],
                    "optimization_results": {},
                    "source": "default_loose_bounds",
                }
                for physical_generator_idx in self.nn_policy_generator_ids:
                    generator_name = self.physical_generator_names[
                        int(physical_generator_idx)
                    ]
                    policy = self.nn_policies.get(generator_name)
                    if not policy:
                        continue
                    linear_layers = [
                        layer
                        for layer in policy.get("layers", [])
                        if str(layer.get("type", "")).lower() == "linear"
                    ]
                    hidden_layers = linear_layers[:-1]
                    generator_entries: dict[str, Any] = {}
                    for linear_idx, layer in enumerate(hidden_layers):
                        for time_idx in range(self.num_time_steps):
                            for node in range(len(layer.get("bias", []))):
                                key = self._json_key(
                                    (int(time_idx), int(linear_idx), int(node))
                                )
                                generator_entries[key] = {
                                    "physical_generator_index": int(
                                        physical_generator_idx
                                    ),
                                    "time_idx": int(time_idx),
                                    "linear_idx": int(linear_idx),
                                    "node": int(node),
                                    "L": -margin,
                                    "U": margin,
                                    "h_lower": 0.0,
                                    "h_upper": margin,
                                    "status": "ambiguous",
                                    "bound_method": "default_loose_bounds",
                                }
                    report["nn_relu_bounds"][generator_name] = generator_entries
                return report
            return {}

        report = copy.deepcopy(source)
        if "nn_relu_bounds_report" in report and isinstance(report["nn_relu_bounds_report"], dict):
            report = copy.deepcopy(report["nn_relu_bounds_report"])
        if "nn_relu_bounds" not in report:
            return {}

        for generator_name, entries in (report.get("nn_relu_bounds", {}) or {}).items():
            for key, details in (entries or {}).items():
                if not isinstance(details, dict):
                    continue
                details["L"] = -margin
                details["U"] = margin
                details["h_lower"] = 0.0
                details["h_upper"] = margin
                details["status"] = "ambiguous"
                details["bound_method"] = "default_loose_bounds"
        report.setdefault("summary", {})
        report.setdefault("warnings", [])
        report["source"] = "default_loose_bounds"
        return report

    def ensure_default_nn_relu_bounds_available(
        self,
        template_report: Optional[dict[str, Any]] = None,
        overwrite_existing: bool = False,
    ) -> dict[str, Any]:
        if not self.nn_policy_generator_ids:
            return {}
        if self.nn_relu_bounds and not overwrite_existing:
            return self.nn_relu_bounds_report or {
                "nn_relu_bounds": {
                    generator_name: {
                        self._json_key(index): dict(bounds)
                        for index, bounds in generator_bounds.items()
                    }
                    for generator_name, generator_bounds in self.nn_relu_bounds.items()
                }
            }

        default_report = self.build_default_nn_relu_bounds_report(
            template_report={} if template_report is None else template_report
        )
        if not default_report:
            return {}

        if hasattr(self, "load_tightening_report_data"):
            self.load_tightening_report_data(
                {
                    "nn_relu_bounds_report": default_report,
                    "nn_relu_bounds": default_report.get("nn_relu_bounds", {}),
                },
                report_path=getattr(self, "tightening_report_path", None),
                prepare_bounds=False,
                merge_existing=True,
            )
        else:
            self._set_nn_relu_bounds_from_report(default_report)

        if hasattr(self, "_mark_default_bound_used"):
            self._mark_default_bound_used(
                "nn_relu_bounds_report",
                "Default loose NN ReLU bounds were used.",
            )
        return default_report

    def _raw_nn_feature_expression(self, feature_name: str, t: int, physical_generator_idx: int):
        m = self.model
        previous_t = self.num_time_steps - 1 if int(t) == 0 else int(t) - 1
        next_t = 0 if int(t) == self.num_time_steps - 1 else int(t) + 1

        total_wind_capacity = lambda time_idx: sum(
            m.P_max_block[i, b, time_idx] for (i, b) in self.wind_block_pairs
        )
        total_capacity = lambda time_idx: sum(
            m.P_max_block[i, b, time_idx] for (i, b) in self.generator_block_pairs
        )
        own_capacity = lambda time_idx: sum(
            m.P_max_block[physical_generator_idx, b, time_idx]
            for b in self.local_blocks_by_generator[physical_generator_idx]
        )

        if feature_name == "demand":
            return m.D[t]
        if feature_name == "total_wind_generation_capacity":
            return total_wind_capacity(t)
        if feature_name == "total_generation_capacity":
            return total_capacity(t)
        if feature_name == "residual_demand":
            return m.D[t] - total_wind_capacity(t)
        if feature_name == "previous_demand":
            return m.D[previous_t]
        if feature_name == "previous_wind_generation_capacity":
            return total_wind_capacity(previous_t)
        if feature_name == "previous_residual_demand":
            return m.D[previous_t] - total_wind_capacity(previous_t)
        if feature_name == "previous_generation_capacity":  # legacy
            return total_capacity(previous_t)
        if feature_name == "next_demand":
            return m.D[next_t]
        if feature_name == "next_wind_generation_capacity":
            return total_wind_capacity(next_t)
        if feature_name == "next_residual_demand":
            return m.D[next_t] - total_wind_capacity(next_t)
        if feature_name == "next_generation_capacity":  # legacy
            return total_capacity(next_t)
        if feature_name == "total_demand_over_horizon":
            return sum(m.D[time_idx] for time_idx in range(self.num_time_steps))
        if feature_name == "total_wind_over_horizon":
            return sum(total_wind_capacity(time_idx) for time_idx in range(self.num_time_steps))
        if feature_name == "total_residual_over_horizon":
            return sum(
                m.D[time_idx] - total_wind_capacity(time_idx)
                for time_idx in range(self.num_time_steps)
            )
        if feature_name == "own_generation_capacity":
            return own_capacity(t)
        if feature_name == "previous_own_generation_capacity":
            return own_capacity(previous_t)
        if feature_name == "next_own_generation_capacity":
            return own_capacity(next_t)
        if feature_name == "average_true_cost":
            costs = [
                self.block_cost_vector[self.local_to_global_block[(physical_generator_idx, b)]]
                for b in self.local_blocks_by_generator[physical_generator_idx]
            ]
            return float(np.mean(costs))
        if feature_name == "minimum_true_cost":
            return float(
                min(
                    self.block_cost_vector[self.local_to_global_block[(physical_generator_idx, b)]]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        if feature_name == "maximum_true_cost":
            return float(
                max(
                    self.block_cost_vector[self.local_to_global_block[(physical_generator_idx, b)]]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        raise ValueError(f"Unsupported NN feature name: {feature_name}")

    def _normalized_nn_feature_expression(
        self, generator_name: str, feature_name: str, t: int, physical_generator_idx: int
    ):
        raw = self._raw_nn_feature_expression(feature_name, t, physical_generator_idx)
        feature_min, feature_max = self._nn_feature_bounds(generator_name, feature_name)
        denominator = feature_max - feature_min
        if abs(denominator) <= self.normalization_epsilon:
            return 0.0
        return (raw - feature_min) / denominator

    @staticmethod
    def _clamp_relu_bounds_to_status(
        status: str, L: float, U: float, h_lower: float, h_upper: float
    ) -> tuple[float, float, float, float]:
        """Clamp OBBT-derived ReLU bounds to be consistent with the classified status.

        OBBT can return tiny positive L/U for inactive nodes when network weights are
        near machine precision (e.g. ~1e-40), which would impose a positive floor on h
        and conflict with the h == 0 constraint added for inactive nodes.  The same
        fallback logic mirrors _DUAL_BIG_M_OBBT_MIN_FRACTION in _prepare_dual_big_m.
        """
        if status == "inactive":
            safe_u = min(U, 0.0)
            return min(L, safe_u), safe_u, 0.0, 0.0
        if status == "active":
            return max(L, 0.0), U, max(h_lower, 0.0), h_upper
        return L, U, h_lower, h_upper  # ambiguous: Big-M values, keep as-is

    def _build_policy_constraints(self) -> None:
        def true_cost_alpha_rule(m, i, b, t):
            if int(i) in self.nn_policy_generator_ids:
                return Constraint.Skip
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.alpha[i, b, t] == self.block_cost_vector[global_block]

        self.model.true_cost_alpha = Constraint(
            self.model.generator_blocks,
            self.model.time_steps,
            rule=true_cost_alpha_rule,
        )

        if self.nn_policy_generator_ids:
            self._build_nn_policy_constraints()

    def _build_nn_policy_constraints(self) -> None:
        m = self.model
        if not self.nn_relu_bounds:
            self.ensure_default_nn_relu_bounds_available()
        if not self.nn_relu_bounds:
            raise ValueError(
                "NN ReLU bounds are required before building NN policy constraints, "
                "and default loose bounds could not be constructed. Run "
                "compute_relu_bounds.py or call load_nn_relu_bounds_report(...)."
            )
        nn_input_indices: list[tuple[int, int, int]] = []
        nn_z_indices: list[tuple[int, int, int, int]] = []
        nn_h_indices: list[tuple[int, int, int, int]] = []
        nn_output_indices: list[tuple[int, int, int]] = []

        linear_layer_dims: dict[tuple[int, int], int] = {}
        relu_after_linear_layers: set[tuple[int, int]] = set()
        output_dims: dict[int, int] = {}
        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            if generator_name not in self.nn_relu_bounds:
                raise ValueError(
                    f"NN ReLU bounds report is missing bounds for generator "
                    f"{generator_name}."
                )
            for f_idx, _ in enumerate(policy["feature_columns"]):
                for t in range(self.num_time_steps):
                    nn_input_indices.append((i, t, f_idx))
            layers = policy["layers"]
            linear_positions = [
                layer_pos
                for layer_pos, layer in enumerate(layers)
                if str(layer.get("type", "")).lower() == "linear"
            ]
            final_linear_idx = len(linear_positions) - 1
            linear_idx = 0
            for layer_pos, layer in enumerate(layers):
                if str(layer.get("type", "")).lower() != "linear":
                    continue
                output_dim = len(layer["bias"])
                linear_layer_dims[(i, linear_idx)] = output_dim
                is_final_linear = linear_idx == final_linear_idx
                has_relu_after = (
                    layer_pos + 1 < len(layers)
                    and str(layers[layer_pos + 1].get("type", "")).lower() == "relu"
                )
                if is_final_linear:
                    output_dims[i] = output_dim
                    for k in range(output_dim):
                        for t in range(self.num_time_steps):
                            nn_output_indices.append((i, t, k))
                if has_relu_after:
                    for node in range(output_dim):
                        for t in range(self.num_time_steps):
                            nn_z_indices.append((i, t, linear_idx, node))
                            nn_h_indices.append((i, t, linear_idx, node))
                    relu_after_linear_layers.add((i, linear_idx))
                linear_idx += 1

        m.nn_input_index = Set(dimen=3, initialize=nn_input_indices)
        m.nn_z_index = Set(dimen=4, initialize=nn_z_indices)
        m.nn_h_index = Set(dimen=4, initialize=nn_h_indices)
        m.nn_output_index = Set(dimen=3, initialize=nn_output_indices)
        m.nn_delta_index = Set(dimen=4, initialize=nn_h_indices)
        m.nn_input = Var(m.nn_input_index, domain=Reals)
        m.nn_z = Var(m.nn_z_index, domain=Reals)
        m.nn_h = Var(m.nn_h_index, domain=NonNegativeReals)
        m.nn_delta = Var(m.nn_delta_index, domain=Binary)
        m.nn_output = Var(m.nn_output_index, domain=Reals)
        m.nn_constraints = ConstraintList()

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[i]
            relu_bounds = self.nn_relu_bounds[generator_name]
            for (t, linear_idx, node), bounds in relu_bounds.items():
                index = (i, int(t), int(linear_idx), int(node))
                if index not in m.nn_z:
                    continue
                status = str(bounds.get("status", "ambiguous")).lower()
                z_lb, z_ub, h_lb, h_ub = self._clamp_relu_bounds_to_status(
                    status,
                    float(bounds["L"]),
                    float(bounds["U"]),
                    float(bounds["h_lower"]),
                    float(bounds["h_upper"]),
                )
                m.nn_z[index].setlb(z_lb)
                m.nn_z[index].setub(z_ub)
                m.nn_h[index].setlb(h_lb)
                m.nn_h[index].setub(h_ub)

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            relu_bounds = self.nn_relu_bounds[generator_name]
            feature_columns = policy["feature_columns"]
            for t in range(self.num_time_steps):
                for f_idx, feature_name in enumerate(feature_columns):
                    m.nn_constraints.add(
                        m.nn_input[i, t, f_idx]
                        == self._normalized_nn_feature_expression(generator_name, feature_name, t, i)
                    )

                previous_values = [m.nn_input[i, t, f] for f in range(len(feature_columns))]
                linear_positions = [
                    layer_pos
                    for layer_pos, layer in enumerate(policy["layers"])
                    if str(layer.get("type", "")).lower() == "linear"
                ]
                final_linear_idx = len(linear_positions) - 1
                linear_idx = 0
                for layer_pos, layer in enumerate(policy["layers"]):
                    if str(layer.get("type", "")).lower() == "relu":
                        continue
                    weights = np.asarray(layer["weight"], dtype=float)
                    bias = np.asarray(layer["bias"], dtype=float)
                    is_final_linear = linear_idx == final_linear_idx
                    has_relu_after = (i, linear_idx) in relu_after_linear_layers
                    current_values = []
                    for node in range(weights.shape[0]):
                        expr = float(bias[node]) + sum(
                            float(weights[node, prev_idx]) * previous_values[prev_idx]
                            for prev_idx in range(weights.shape[1])
                        )
                        if is_final_linear and not has_relu_after:
                            m.nn_constraints.add(m.nn_output[i, t, node] == expr)
                            current_values.append(m.nn_output[i, t, node])
                        else:
                            m.nn_constraints.add(m.nn_z[i, t, linear_idx, node] == expr)
                            h = m.nn_h[i, t, linear_idx, node]
                            z = m.nn_z[i, t, linear_idx, node]
                            delta = m.nn_delta[i, t, linear_idx, node]
                            bound_key = (int(t), int(linear_idx), int(node))
                            if bound_key not in relu_bounds:
                                raise ValueError(
                                    f"NN ReLU bounds report is missing bounds for "
                                    f"{generator_name} at time {t}, linear layer "
                                    f"{linear_idx}, node {node}."
                                )
                            bounds = relu_bounds[bound_key]
                            status = bounds["status"]
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
                            if is_final_linear:
                                m.nn_constraints.add(m.nn_output[i, t, node] == h)
                                current_values.append(m.nn_output[i, t, node])
                            else:
                                current_values.append(h)
                    previous_values = current_values
                    linear_idx += 1

            for output_idx, local_block in policy["target_map"].items():
                for t in range(self.num_time_steps):
                    m.nn_constraints.add(
                        m.alpha[i, local_block, t] == m.nn_output[i, t, output_idx]
                    )

    def apply_nn_relu_bounds_to_model(
        self,
        report: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        if report is not None:
            self._set_nn_relu_bounds_from_report(report)
        if not self.nn_relu_bounds:
            raise ValueError(
                "No NN ReLU bounds loaded. Call load_nn_relu_bounds_report() first."
            )

        m = self.model
        stats = {
            "z_bounds_applied": 0,
            "h_bounds_applied": 0,
            "delta_fixed_active": 0,
            "delta_fixed_inactive": 0,
            "delta_left_ambiguous": 0,
        }

        for physical_generator_idx in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(physical_generator_idx)]
            for (time_idx, linear_idx, node), bounds in (
                self.nn_relu_bounds.get(generator_name, {}) or {}
            ).items():
                index = (
                    int(physical_generator_idx),
                    int(time_idx),
                    int(linear_idx),
                    int(node),
                )
                status = str(bounds.get("status", "")).lower()
                z_lb, z_ub, h_lb, h_ub = self._clamp_relu_bounds_to_status(
                    status,
                    float(bounds["L"]),
                    float(bounds["U"]),
                    float(bounds["h_lower"]),
                    float(bounds["h_upper"]),
                )
                if hasattr(m, "nn_z") and index in m.nn_z:
                    m.nn_z[index].setlb(z_lb)
                    m.nn_z[index].setub(z_ub)
                    stats["z_bounds_applied"] += 1
                if hasattr(m, "nn_h") and index in m.nn_h:
                    m.nn_h[index].setlb(h_lb)
                    m.nn_h[index].setub(h_ub)
                    stats["h_bounds_applied"] += 1
                if not hasattr(m, "nn_delta") or index not in m.nn_delta:
                    continue

                if status == "inactive":
                    m.nn_delta[index].fix(0)
                    stats["delta_fixed_inactive"] += 1
                elif status == "active":
                    m.nn_delta[index].fix(1)
                    stats["delta_fixed_active"] += 1
                elif status == "ambiguous":
                    if m.nn_delta[index].fixed:
                        m.nn_delta[index].unfix()
                    stats["delta_left_ambiguous"] += 1
                else:
                    raise ValueError(
                        f"{generator_name}: unknown ReLU bound status '{status}'"
                    )

        self.applied_nn_relu_stats = stats
        return stats

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
