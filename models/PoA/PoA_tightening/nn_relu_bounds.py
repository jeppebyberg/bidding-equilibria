from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pyomo.environ import *

from models.PoA.PoA_optimization import PoAOptimization


class NNReLUBoundsOptimizer(PoAOptimization):
    """Standalone preprocessing optimizer for NN ReLU preactivation bounds."""

    # ------------------------------------------------------------------
    # Support-set feature-bound diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_float(value: Any, label: str) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a finite numeric value") from exc
        if not np.isfinite(numeric_value):
            raise ValueError(f"{label} must be finite; got {numeric_value}")
        return numeric_value

    def _validate_nn_interval(
        self,
        lower: Any,
        upper: Any,
        label: str,
    ) -> tuple[float, float]:
        lower_value = self._finite_float(lower, f"{label} lower bound")
        upper_value = self._finite_float(upper, f"{label} upper bound")
        if lower_value > upper_value:
            raise ValueError(
                f"Invalid {label} bounds: lower {lower_value} exceeds upper {upper_value}"
            )
        return lower_value, upper_value

    def _nn_time_indices(self) -> range:
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive to compute NN feature bounds")
        return range(self.num_time_steps)

    def _optional_nonnegative_finite_budget(
        self,
        attr_name: str,
        label: str,
    ) -> float | None:
        if not hasattr(self, attr_name):
            return None
        try:
            budget = float(getattr(self, attr_name))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(budget):
            return None
        if budget < 0.0:
            raise ValueError(f"{label} must be non-negative; got {budget}")
        return budget

    def _reference_profile_or_none(self, reference: Any, label: str) -> list[float] | None:
        if reference is None:
            return None

        values: list[float] = []
        for t in self._nn_time_indices():
            try:
                if isinstance(reference, dict):
                    if t in reference:
                        raw_value = reference[t]
                    elif str(t) in reference:
                        raw_value = reference[str(t)]
                    else:
                        return None
                else:
                    raw_value = reference[t]
                value = float(raw_value)
            except (KeyError, IndexError, TypeError, ValueError):
                return None
            if not np.isfinite(value):
                return None
            values.append(value)
        return values

    def _global_box_demand_bounds(self) -> tuple[float, float]:
        return self._validate_nn_interval(
            self.support_demand_min,
            self.support_demand_max,
            "support-set demand",
        )

    def _global_box_wind_bounds(self, physical_generator_idx: int) -> tuple[float, float]:
        i = int(physical_generator_idx)
        if i not in self.support_wind_min or i not in self.support_wind_max:
            raise ValueError(f"Missing support-set wind min/max bounds for generator {i}")
        return self._validate_nn_interval(
            self.support_wind_min[i],
            self.support_wind_max[i],
            f"support-set wind generator {i}",
        )

    def _wind_reference_profile_or_none(
        self,
        physical_generator_idx: int,
    ) -> list[float] | None:
        i = int(physical_generator_idx)
        references = getattr(self, "support_wind_reference", None)
        if references is None:
            return None

        reference = None
        if isinstance(references, dict):
            if i in references:
                reference = references[i]
            elif str(i) in references:
                reference = references[str(i)]
            else:
                if 0 <= i < len(self.physical_generator_names):
                    reference = references.get(self.physical_generator_names[i])
        else:
            try:
                reference = references[i]
            except (IndexError, TypeError, KeyError):
                return None

        return self._reference_profile_or_none(reference, f"support_wind_reference[{i}]")

    def _demand_reference_budget_available(self) -> bool:
        budget = self._optional_nonnegative_finite_budget(
            "support_demand_budget",
            "support_demand_budget",
        )
        if budget is None:
            return False
        reference = self._reference_profile_or_none(
            getattr(self, "support_demand_reference", None),
            "support_demand_reference",
        )
        return reference is not None

    def _wind_reference_budget_available(self, physical_generator_idx: int) -> bool:
        budget = self._optional_nonnegative_finite_budget(
            "support_wind_budget",
            "support_wind_budget",
        )
        if budget is None:
            return False
        return self._wind_reference_profile_or_none(physical_generator_idx) is not None

    def _timewise_demand_bounds(self) -> dict[int, tuple[float, float]]:
        demand_lower, demand_upper = self._global_box_demand_bounds()
        demand_budget = self._optional_nonnegative_finite_budget(
            "support_demand_budget",
            "support_demand_budget",
        )
        demand_reference = self._reference_profile_or_none(
            getattr(self, "support_demand_reference", None),
            "support_demand_reference",
        )
        if demand_budget is None or demand_reference is None:
            return {int(t): (demand_lower, demand_upper) for t in self._nn_time_indices()}

        bounds: dict[int, tuple[float, float]] = {}
        for t, reference_value in enumerate(demand_reference):
            lower = max(demand_lower, reference_value - demand_budget)
            upper = min(demand_upper, reference_value + demand_budget)
            bounds[int(t)] = self._validate_nn_interval(
                lower,
                upper,
                f"reference-tight demand at t={t}",
            )
        return bounds

    def _timewise_wind_bounds(self) -> dict[int, dict[int, tuple[float, float]]]:
        wind_budget = self._optional_nonnegative_finite_budget(
            "support_wind_budget",
            "support_wind_budget",
        )
        bounds: dict[int, dict[int, tuple[float, float]]] = {}
        for raw_i in self.wind_physical_generator_ids:
            i = int(raw_i)
            wind_lower, wind_upper = self._global_box_wind_bounds(i)
            wind_reference = self._wind_reference_profile_or_none(i)
            if wind_budget is None or wind_reference is None:
                bounds[i] = {
                    int(t): (wind_lower, wind_upper) for t in self._nn_time_indices()
                }
                continue

            generator_bounds: dict[int, tuple[float, float]] = {}
            for t, reference_value in enumerate(wind_reference):
                lower = max(wind_lower, reference_value - wind_budget)
                upper = min(wind_upper, reference_value + wind_budget)
                generator_bounds[int(t)] = self._validate_nn_interval(
                    lower,
                    upper,
                    f"reference-tight wind generator {i} at t={t}",
                )
            bounds[i] = generator_bounds
        return bounds

    def _global_reference_tight_demand_bounds(self) -> tuple[float, float]:
        timewise_bounds = self._timewise_demand_bounds()
        if not timewise_bounds:
            raise ValueError("No demand time steps available for NN feature bounds")
        lower = min(lower for lower, _ in timewise_bounds.values())
        upper = max(upper for _, upper in timewise_bounds.values())
        return self._validate_nn_interval(lower, upper, "global reference-tight demand")

    def _global_reference_tight_wind_bounds(
        self,
        physical_generator_idx: int,
    ) -> tuple[float, float]:
        i = int(physical_generator_idx)
        timewise_bounds = self._timewise_wind_bounds()
        if i not in timewise_bounds:
            raise ValueError(f"No wind time steps available for generator {i}")
        generator_bounds = timewise_bounds[i]
        if not generator_bounds:
            raise ValueError(f"No wind time steps available for generator {i}")
        lower = min(lower for lower, _ in generator_bounds.values())
        upper = max(upper for _, upper in generator_bounds.values())
        return self._validate_nn_interval(
            lower,
            upper,
            f"global reference-tight wind generator {i}",
        )

    def _raw_nn_feature_bounds(
        self,
        feature_name: str,
        physical_generator_idx: int,
    ) -> tuple[float, float]:
        physical_generator_idx = int(physical_generator_idx)

        def total_wind_bounds() -> tuple[float, float]:
            lower = 0.0
            upper = 0.0
            for i in self.wind_physical_generator_ids:
                wind_lower, wind_upper = self._global_reference_tight_wind_bounds(i)
                lower += wind_lower
                upper += wind_upper
            return self._validate_nn_interval(lower, upper, "total wind generation capacity")

        def total_generation_bounds() -> tuple[float, float]:
            conventional_total = sum(
                self.static_physical_capacity[i]
                for i in self.conventional_physical_generator_ids
            )
            wind_lower, wind_upper = total_wind_bounds()
            return self._validate_nn_interval(
                conventional_total + wind_lower,
                conventional_total + wind_upper,
                "total generation capacity",
            )

        def own_generation_bounds() -> tuple[float, float]:
            if physical_generator_idx in self.wind_physical_generator_ids:
                return self._global_reference_tight_wind_bounds(physical_generator_idx)
            cap = self._finite_float(
                self.static_physical_capacity[physical_generator_idx],
                f"static capacity for generator {physical_generator_idx}",
            )
            return cap, cap

        def demand_bounds() -> tuple[float, float]:
            return self._global_reference_tight_demand_bounds()

        def residual_demand_bounds() -> tuple[float, float]:
            demand_lower, demand_upper = demand_bounds()
            total_wind_lower, total_wind_upper = total_wind_bounds()
            return self._validate_nn_interval(
                demand_lower - total_wind_upper,
                demand_upper - total_wind_lower,
                "residual demand",
            )

        if feature_name == "demand":
            return demand_bounds()
        if feature_name == "total_wind_generation_capacity":
            return total_wind_bounds()
        if feature_name == "total_generation_capacity":
            return total_generation_bounds()
        if feature_name == "residual_demand":
            return residual_demand_bounds()
        if feature_name in {"previous_generation_capacity", "next_generation_capacity"}:
            return total_generation_bounds()
        if feature_name in {"previous_demand", "next_demand"}:
            return demand_bounds()
        if feature_name == "own_generation_capacity":
            return own_generation_bounds()
        if feature_name in {
            "previous_own_generation_capacity",
            "next_own_generation_capacity",
        }:
            return own_generation_bounds()
        if feature_name == "average_true_cost":
            costs = [
                self.block_cost_vector[
                    self.local_to_global_block[(physical_generator_idx, b)]
                ]
                for b in self.local_blocks_by_generator[physical_generator_idx]
            ]
            value = float(np.mean(costs))
            return self._validate_nn_interval(value, value, "average true cost")
        if feature_name == "minimum_true_cost":
            value = float(
                min(
                    self.block_cost_vector[
                        self.local_to_global_block[(physical_generator_idx, b)]
                    ]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
            return self._validate_nn_interval(value, value, "minimum true cost")
        if feature_name == "maximum_true_cost":
            value = float(
                max(
                    self.block_cost_vector[
                        self.local_to_global_block[(physical_generator_idx, b)]
                    ]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
            return self._validate_nn_interval(value, value, "maximum true cost")
        raise ValueError(f"Unsupported NN feature name: {feature_name}")

    def _normalized_nn_feature_bounds(
        self,
        generator_name: str,
        feature_name: str,
        physical_generator_idx: int,
    ) -> tuple[float, float]:
        raw_lower, raw_upper = self._raw_nn_feature_bounds(
            feature_name,
            physical_generator_idx,
        )
        feature_min, feature_max = self._nn_feature_bounds(generator_name, feature_name)
        feature_min = self._finite_float(
            feature_min,
            f"{generator_name} normalization minimum for {feature_name}",
        )
        feature_max = self._finite_float(
            feature_max,
            f"{generator_name} normalization maximum for {feature_name}",
        )
        denominator = feature_max - feature_min
        if abs(denominator) <= self.normalization_epsilon:
            return 0.0, 0.0
        normalized_lower = (raw_lower - feature_min) / denominator
        normalized_upper = (raw_upper - feature_min) / denominator
        return self._validate_nn_interval(
            min(normalized_lower, normalized_upper),
            max(normalized_lower, normalized_upper),
            f"normalized NN feature {feature_name} for {generator_name}",
        )

    @staticmethod
    def _combine_feature_bound_sources(sources: list[str]) -> str:
        unique_sources = set(sources)
        if not unique_sources:
            return "fixed_constant"
        if len(unique_sources) == 1:
            return sources[0]
        if "reference_budget_intersection" in unique_sources:
            return "mixed_reference_budget_intersection"
        return "mixed"

    def _wind_feature_bound_source(self, physical_generator_idx: int) -> str:
        if self._wind_reference_budget_available(physical_generator_idx):
            return "reference_budget_intersection"
        return "global_support_box"

    def _total_wind_feature_bound_source(self) -> str:
        if not self.wind_physical_generator_ids:
            return "fixed_zero"
        return self._combine_feature_bound_sources(
            [self._wind_feature_bound_source(i) for i in self.wind_physical_generator_ids]
        )

    def _demand_feature_bound_source(self) -> str:
        if self._demand_reference_budget_available():
            return "reference_budget_intersection"
        return "global_support_box"

    def _nn_feature_bound_source(
        self,
        feature_name: str,
        physical_generator_idx: int,
    ) -> str:
        if feature_name in {"demand", "previous_demand", "next_demand"}:
            return self._demand_feature_bound_source()
        if feature_name == "total_wind_generation_capacity":
            return self._total_wind_feature_bound_source()
        if feature_name in {
            "total_generation_capacity",
            "previous_generation_capacity",
            "next_generation_capacity",
        }:
            wind_source = self._total_wind_feature_bound_source()
            if wind_source == "fixed_zero":
                return "fixed_static_capacity"
            return wind_source
        if feature_name == "residual_demand":
            return self._combine_feature_bound_sources(
                [
                    self._demand_feature_bound_source(),
                    self._total_wind_feature_bound_source(),
                ]
            )
        if feature_name in {
            "own_generation_capacity",
            "previous_own_generation_capacity",
            "next_own_generation_capacity",
        }:
            if int(physical_generator_idx) in self.wind_physical_generator_ids:
                return self._wind_feature_bound_source(physical_generator_idx)
            return "fixed_static_capacity"
        if feature_name in {
            "average_true_cost",
            "minimum_true_cost",
            "maximum_true_cost",
        }:
            return "fixed_constant"
        return "unknown"

    def summarize_nn_feature_bounds(self) -> dict[str, Any]:
        policies = self.nn_policies or {}
        if not policies or not self.nn_policy_generator_ids:
            return {}

        summary: dict[str, Any] = {}
        for physical_generator_idx in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(physical_generator_idx)]
            policy = policies.get(generator_name)
            if not policy:
                continue

            feature_summary: dict[str, Any] = {}
            sources: list[str] = []
            for feature_name in policy["feature_columns"]:
                raw_lower, raw_upper = self._raw_nn_feature_bounds(
                    feature_name,
                    int(physical_generator_idx),
                )
                normalized_lower, normalized_upper = self._normalized_nn_feature_bounds(
                    generator_name,
                    feature_name,
                    int(physical_generator_idx),
                )
                feature_min, feature_max = self._nn_feature_bounds(
                    generator_name,
                    feature_name,
                )
                source = self._nn_feature_bound_source(
                    feature_name,
                    int(physical_generator_idx),
                )
                sources.append(source)
                feature_summary[feature_name] = {
                    "source": source,
                    "raw_lower": raw_lower,
                    "raw_upper": raw_upper,
                    "normalized_lower": normalized_lower,
                    "normalized_upper": normalized_upper,
                    "normalization_min": float(feature_min),
                    "normalization_max": float(feature_max),
                }

            summary[generator_name] = {
                "physical_generator_index": int(physical_generator_idx),
                "feature_bound_source": self._combine_feature_bound_sources(sources),
                "features": feature_summary,
            }
        return summary

    # ------------------------------------------------------------------
    # Preactivation-bound optimization model builders
    # ------------------------------------------------------------------

    def _build_tightening_sets(self) -> None:
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))
        self.model.physical_generators = Set(initialize=range(self.num_physical_generators))
        self.model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)
        self.model.wind_physical_generators = Set(initialize=self.wind_physical_generator_ids)
        self.model.conventional_physical_generators = Set(
            initialize=self.conventional_physical_generator_ids
        )
        self.model.wind_blocks = Set(dimen=2, initialize=self.wind_block_pairs)
        self.model.conventional_blocks = Set(dimen=2, initialize=self.conventional_block_pairs)

    @staticmethod
    def _linear_layers(policy: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            layer
            for layer in policy.get("layers", [])
            if str(layer.get("type", "")).lower() == "linear"
        ]

    @classmethod
    def _hidden_linear_layers(cls, policy: dict[str, Any]) -> list[dict[str, Any]]:
        linear_layers = cls._linear_layers(policy)
        return linear_layers[:-1]

    @staticmethod
    def _objective_sense(sense: str) -> Any:
        if sense == "lower":
            return minimize
        if sense == "upper":
            return maximize
        raise ValueError(f"Unknown bound sense: {sense}")

    def _ensure_nn_inputs_loaded(self) -> None:
        if self.nn_policy_generator_ids and not self.nn_policies:
            self._load_nn_policies()
        if self.nn_policy_generator_ids and not self.nn_stats:
            self._load_nn_normalization_stats()

    def _build_first_layer_preactivation_bound_model(
        self,
        generator_name: str,
        physical_generator_idx: int,
        time_idx: int,
        neuron_idx: int,
        sense: str,
    ) -> tuple[ConcreteModel, Any]:
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()
        self._build_support_set()

        policy = self.nn_policies[generator_name]
        hidden_layers = self._hidden_linear_layers(policy)
        if not hidden_layers:
            raise ValueError(f"{generator_name}: NN policy has no hidden ReLU layers")
        first_layer = hidden_layers[0]
        weights = np.asarray(first_layer["weight"], dtype=float)
        bias = np.asarray(first_layer["bias"], dtype=float)
        neuron_idx = int(neuron_idx)
        if not 0 <= neuron_idx < weights.shape[0]:
            raise ValueError(f"{generator_name}: first-layer neuron {neuron_idx} missing")

        feature_exprs = [
            self._normalized_nn_feature_expression(
                generator_name,
                feature_name,
                int(time_idx),
                int(physical_generator_idx),
            )
            for feature_name in policy["feature_columns"]
        ]
        z_expr = float(bias[neuron_idx]) + sum(
            float(weights[neuron_idx, feature_idx]) * feature_exprs[feature_idx]
            for feature_idx in range(weights.shape[1])
        )
        self.model.nn_preactivation_bound_objective = Objective(
            expr=z_expr,
            sense=self._objective_sense(sense),
        )
        return self.model, z_expr

    def _build_later_layer_preactivation_bound_model(
        self,
        generator_name: str,
        physical_generator_idx: int,
        time_idx: int,
        linear_idx: int,
        neuron_idx: int,
        previous_activation_bounds: dict[int, dict[str, float]],
        sense: str,
    ) -> tuple[ConcreteModel, Any]:
        m = ConcreteModel()
        previous_nodes = sorted(int(node) for node in previous_activation_bounds)
        m.previous_neurons = Set(initialize=previous_nodes)

        def previous_activation_bounds_rule(model, node):
            bounds = previous_activation_bounds[int(node)]
            return float(bounds["h_lower"]), float(bounds["h_upper"])

        m.a_prev = Var(m.previous_neurons, domain=Reals, bounds=previous_activation_bounds_rule)

        policy = self.nn_policies[generator_name]
        hidden_layers = self._hidden_linear_layers(policy)
        linear_idx = int(linear_idx)
        neuron_idx = int(neuron_idx)
        if not 0 <= linear_idx < len(hidden_layers):
            raise ValueError(f"{generator_name}: hidden linear layer {linear_idx} missing")
        if linear_idx <= 0:
            raise ValueError("Later-layer bound models require linear_idx > 0")

        layer = hidden_layers[linear_idx]
        weights = np.asarray(layer["weight"], dtype=float)
        bias = np.asarray(layer["bias"], dtype=float)
        if not 0 <= neuron_idx < weights.shape[0]:
            raise ValueError(
                f"{generator_name}: hidden layer {linear_idx} neuron {neuron_idx} missing"
            )
        if weights.shape[1] != len(previous_nodes):
            raise ValueError(
                f"{generator_name}: hidden layer {linear_idx} expects {weights.shape[1]} "
                f"previous activations, got {len(previous_nodes)}"
            )

        z_expr = float(bias[neuron_idx]) + sum(
            float(weights[neuron_idx, previous_node]) * m.a_prev[previous_node]
            for previous_node in previous_nodes
        )
        m.nn_preactivation_bound_objective = Objective(
            expr=z_expr,
            sense=self._objective_sense(sense),
        )
        return m, z_expr

    def _solve_bound_model(
        self,
        m: ConcreteModel,
        expr: Any,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_options: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[float], str]:
        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if solver_options:
            for option_name, option_value in solver_options.items():
                solver.options[option_name] = option_value

        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        expr_value = value(expr, exception=False)
        acceptable = {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal,
            TerminationCondition.feasible,
        }
        if termination in acceptable and expr_value is not None:
            return float(expr_value), str(termination)
        if termination == TerminationCondition.maxTimeLimit and expr_value is not None:
            return float(expr_value), str(termination)
        return None, str(termination)

    # ------------------------------------------------------------------
    # Bound computation and report serialization
    # ------------------------------------------------------------------

    def _classify_relu_status(
        self,
        L: float,
        U: float,
        tolerance: float = 1e-9,
    ) -> str:
        if U <= tolerance:
            return "inactive"
        if L >= -tolerance:
            return "active"
        return "ambiguous"

    def compute_first_layer_preactivation_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
    ) -> dict[str, dict[tuple[int, int, int], dict[str, Any]]]:
        self._ensure_nn_inputs_loaded()
        tolerance = float(getattr(self, "relu_bound_tolerance", 1e-9))
        bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}

        for physical_generator_idx in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(physical_generator_idx)]
            policy = self.nn_policies[generator_name]
            hidden_layers = self._hidden_linear_layers(policy)
            if not hidden_layers:
                bounds[generator_name] = {}
                continue
            first_layer = hidden_layers[0]
            first_layer_width = len(first_layer["bias"])
            generator_bounds: dict[tuple[int, int, int], dict[str, Any]] = {}

            for time_idx in range(self.num_time_steps):
                for neuron_idx in range(first_layer_width):
                    lower_model, lower_expr = self._build_first_layer_preactivation_bound_model(
                        generator_name,
                        int(physical_generator_idx),
                        int(time_idx),
                        int(neuron_idx),
                        "lower",
                    )
                    lower_value, lower_termination = self._solve_bound_model(
                        lower_model,
                        lower_expr,
                        solver_name=solver_name,
                        time_limit=time_limit,
                        tee=tee,
                    )

                    upper_model, upper_expr = self._build_first_layer_preactivation_bound_model(
                        generator_name,
                        int(physical_generator_idx),
                        int(time_idx),
                        int(neuron_idx),
                        "upper",
                    )
                    upper_value, upper_termination = self._solve_bound_model(
                        upper_model,
                        upper_expr,
                        solver_name=solver_name,
                        time_limit=time_limit,
                        tee=tee,
                    )

                    if lower_value is None or upper_value is None:
                        raise RuntimeError(
                            f"Could not compute first-layer ReLU bounds for "
                            f"{generator_name}, t={time_idx}, neuron={neuron_idx}: "
                            f"lower={lower_termination}, upper={upper_termination}"
                        )
                    L = float(lower_value)
                    U = float(upper_value)
                    if L > U + tolerance:
                        raise ValueError(
                            f"{generator_name}: invalid first-layer bounds at "
                            f"t={time_idx}, node={neuron_idx}: L={L}, U={U}"
                        )
                    key = (int(time_idx), 0, int(neuron_idx))
                    generator_bounds[key] = {
                        "physical_generator_index": int(physical_generator_idx),
                        "time_idx": int(time_idx),
                        "linear_idx": 0,
                        "node": int(neuron_idx),
                        "L": L,
                        "U": U,
                        "h_lower": float(max(0.0, L)),
                        "h_upper": float(max(0.0, U)),
                        "status": self._classify_relu_status(L, U, tolerance),
                        "bound_method": "support_set_optimization",
                        "lower_termination_condition": lower_termination,
                        "upper_termination_condition": upper_termination,
                    }
            bounds[generator_name] = generator_bounds

        self.nn_relu_bounds = bounds
        return bounds

    def compute_later_layer_preactivation_bounds(
        self,
        existing_bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]],
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
    ) -> dict[str, dict[tuple[int, int, int], dict[str, Any]]]:
        self._ensure_nn_inputs_loaded()
        tolerance = float(getattr(self, "relu_bound_tolerance", 1e-9))

        for physical_generator_idx in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(physical_generator_idx)]
            policy = self.nn_policies[generator_name]
            hidden_layers = self._hidden_linear_layers(policy)
            generator_bounds = existing_bounds.setdefault(generator_name, {})

            for linear_idx in range(1, len(hidden_layers)):
                layer = hidden_layers[linear_idx]
                layer_width = len(layer["bias"])
                previous_width = len(hidden_layers[linear_idx - 1]["bias"])
                for time_idx in range(self.num_time_steps):
                    previous_activation_bounds = {
                        int(prev_node): {
                            "h_lower": float(
                                generator_bounds[(int(time_idx), linear_idx - 1, prev_node)][
                                    "h_lower"
                                ]
                            ),
                            "h_upper": float(
                                generator_bounds[(int(time_idx), linear_idx - 1, prev_node)][
                                    "h_upper"
                                ]
                            ),
                        }
                        for prev_node in range(previous_width)
                    }

                    for neuron_idx in range(layer_width):
                        lower_model, lower_expr = (
                            self._build_later_layer_preactivation_bound_model(
                                generator_name,
                                int(physical_generator_idx),
                                int(time_idx),
                                int(linear_idx),
                                int(neuron_idx),
                                previous_activation_bounds,
                                "lower",
                            )
                        )
                        lower_value, lower_termination = self._solve_bound_model(
                            lower_model,
                            lower_expr,
                            solver_name=solver_name,
                            time_limit=time_limit,
                            tee=tee,
                        )

                        upper_model, upper_expr = (
                            self._build_later_layer_preactivation_bound_model(
                                generator_name,
                                int(physical_generator_idx),
                                int(time_idx),
                                int(linear_idx),
                                int(neuron_idx),
                                previous_activation_bounds,
                                "upper",
                            )
                        )
                        upper_value, upper_termination = self._solve_bound_model(
                            upper_model,
                            upper_expr,
                            solver_name=solver_name,
                            time_limit=time_limit,
                            tee=tee,
                        )

                        if lower_value is None or upper_value is None:
                            raise RuntimeError(
                                f"Could not compute later-layer ReLU bounds for "
                                f"{generator_name}, t={time_idx}, layer={linear_idx}, "
                                f"node={neuron_idx}: lower={lower_termination}, "
                                f"upper={upper_termination}"
                            )
                        L = float(lower_value)
                        U = float(upper_value)
                        if L > U + tolerance:
                            raise ValueError(
                                f"{generator_name}: invalid later-layer bounds at "
                                f"t={time_idx}, layer={linear_idx}, node={neuron_idx}: "
                                f"L={L}, U={U}"
                            )
                        key = (int(time_idx), int(linear_idx), int(neuron_idx))
                        generator_bounds[key] = {
                            "physical_generator_index": int(physical_generator_idx),
                            "time_idx": int(time_idx),
                            "linear_idx": int(linear_idx),
                            "node": int(neuron_idx),
                            "L": L,
                            "U": U,
                            "h_lower": float(max(0.0, L)),
                            "h_upper": float(max(0.0, U)),
                            "status": self._classify_relu_status(L, U, tolerance),
                            "bound_method": "activation_bound_optimization",
                            "lower_termination_condition": lower_termination,
                            "upper_termination_condition": upper_termination,
                        }

        self.nn_relu_bounds = existing_bounds
        return existing_bounds

    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in str(key).split(",") if part != "")

    def _jsonify_relu_bounds(
        self,
        bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        return {
            generator_name: {
                self._json_key(tuple(index)): details
                for index, details in sorted(generator_bounds.items())
            }
            for generator_name, generator_bounds in bounds.items()
        }

    def _summarize_relu_bounds(
        self,
        bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for generator_name, generator_bounds in bounds.items():
            values = list(generator_bounds.values())
            if not values:
                summary[generator_name] = {
                    "num_hidden_neurons_time_indexed": 0,
                    "num_active": 0,
                    "num_inactive": 0,
                    "num_ambiguous": 0,
                    "min_L": None,
                    "max_L": None,
                    "min_U": None,
                    "max_U": None,
                    "max_M_minus": None,
                    "max_M_plus": None,
                }
                continue

            L_values = [float(item["L"]) for item in values]
            U_values = [float(item["U"]) for item in values]
            summary[generator_name] = {
                "num_hidden_neurons_time_indexed": len(values),
                "num_active": sum(1 for item in values if item["status"] == "active"),
                "num_inactive": sum(1 for item in values if item["status"] == "inactive"),
                "num_ambiguous": sum(1 for item in values if item["status"] == "ambiguous"),
                "min_L": float(min(L_values)),
                "max_L": float(max(L_values)),
                "min_U": float(min(U_values)),
                "max_U": float(max(U_values)),
                "max_M_minus": float(max(max(0.0, -L) for L in L_values)),
                "max_M_plus": float(max(max(0.0, U) for U in U_values)),
            }
        return summary

    def summarize_nn_relu_bounds(self) -> dict[str, Any]:
        return self._summarize_relu_bounds(self.nn_relu_bounds)

    def compute_nn_relu_bounds_report(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        tolerance: float = 1e-9,
    ) -> dict[str, Any]:
        self.relu_bound_tolerance = float(tolerance)
        self._ensure_nn_inputs_loaded()
        self.nn_bound_warnings = []

        first_layer_bounds = self.compute_first_layer_preactivation_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )
        all_bounds = self.compute_later_layer_preactivation_bounds(
            first_layer_bounds,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )
        nn_feature_bounds = self.summarize_nn_feature_bounds()
        summary = self._summarize_relu_bounds(all_bounds)

        report = {
            "metadata": {
                "reference_case": self.reference_case,
                "num_time_steps": self.num_time_steps,
                "nn_policy_generators": list(self.nn_policy_generator_names),
                "physical_generator_names": list(self.physical_generator_names),
                "nn_model_dir": str(self.nn_model_dir),
                "nn_normalization_stats_path": str(self.nn_normalization_stats_path),
                "bound_methods": {
                    "first_layer": "support_set_optimization",
                    "later_layers": "activation_bound_optimization",
                },
                "tolerance": float(tolerance),
            },
            "nn_feature_bounds": nn_feature_bounds,
            "nn_relu_bounds": self._jsonify_relu_bounds(all_bounds),
            "summary": summary,
            "warnings": list(self.nn_bound_warnings),
            "optimization_results": {
                generator_name: {
                    self._json_key(index): {
                        "lower_termination_condition": details.get(
                            "lower_termination_condition"
                        ),
                        "upper_termination_condition": details.get(
                            "upper_termination_condition"
                        ),
                        "bound_method": details.get("bound_method"),
                    }
                    for index, details in generator_bounds.items()
                }
                for generator_name, generator_bounds in all_bounds.items()
            },
        }
        self.nn_feature_bounds = nn_feature_bounds
        self.nn_relu_bounds_report = report
        return report

    def save_nn_relu_bounds_report(
        self,
        output_path: str | Path = "results/poa_nn_relu_bounds_report.json",
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
    ) -> Path:
        report = self.compute_nn_relu_bounds_report(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(report, file_handle, indent=2)
        return path
