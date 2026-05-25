from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Objective,
    Reals,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
    minimize,
    value,
)

from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain


_PARALLEL_RELU_COMPUTER: Optional["DROReLUBoundsComputer"] = None


def _initialize_parallel_relu_computer(state: dict[str, Any]) -> None:
    global _PARALLEL_RELU_COMPUTER
    _PARALLEL_RELU_COMPUTER = DROReLUBoundsComputer._from_parallel_stage_state(state)


def _get_parallel_relu_computer() -> "DROReLUBoundsComputer":
    if _PARALLEL_RELU_COMPUTER is None:
        raise RuntimeError("Parallel DRO ReLU computer was not initialized")
    return _PARALLEL_RELU_COMPUTER


def _solve_parallel_first_layer_relu_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        generator_name,
        physical_generator_idx,
        scenario_idx,
        time_idx,
        neuron_idx,
        solver_name,
        time_limit,
        tee,
        solver_options,
        tolerance,
    ) = task
    computer = _get_parallel_relu_computer()
    computer.relu_bound_tolerance = float(tolerance)
    lower_model, lower_expr = computer._build_first_layer_preactivation_bound_model(
        generator_name,
        int(physical_generator_idx),
        int(scenario_idx),
        int(time_idx),
        int(neuron_idx),
        "lower",
    )
    lower_value, lower_termination = computer._solve_bound_model(
        lower_model,
        lower_expr,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    upper_model, upper_expr = computer._build_first_layer_preactivation_bound_model(
        generator_name,
        int(physical_generator_idx),
        int(scenario_idx),
        int(time_idx),
        int(neuron_idx),
        "upper",
    )
    upper_value, upper_termination = computer._solve_bound_model(
        upper_model,
        upper_expr,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    if lower_value is None or upper_value is None:
        raise RuntimeError(
            f"Could not compute first-layer ReLU bounds for {generator_name}, "
            f"k={scenario_idx}, t={time_idx}, node={neuron_idx}: "
            f"lower={lower_termination}, upper={upper_termination}"
        )
    L = float(lower_value)
    U = float(upper_value)
    if L > U + float(tolerance):
        raise ValueError(
            f"{generator_name}: invalid first-layer bounds at k={scenario_idx}, "
            f"t={time_idx}, node={neuron_idx}: L={L}, U={U}"
        )
    key = (int(scenario_idx), int(time_idx), 0, int(neuron_idx))
    return {
        "generator_name": generator_name,
        "key": key,
        "details": {
            "physical_generator_index": int(physical_generator_idx),
            "scenario_idx": int(scenario_idx),
            "time_idx": int(time_idx),
            "linear_idx": 0,
            "node": int(neuron_idx),
            "L": L,
            "U": U,
            "h_lower": float(max(0.0, L)),
            "h_upper": float(max(0.0, U)),
            "status": computer._classify_relu_status(L, U, float(tolerance)),
            "bound_method": "scenario_indexed_dro_support_optimization",
            "lower_termination_condition": lower_termination,
            "upper_termination_condition": upper_termination,
        },
    }


def _solve_parallel_later_layer_relu_bound(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        generator_name,
        physical_generator_idx,
        scenario_idx,
        time_idx,
        linear_idx,
        neuron_idx,
        previous_activation_bounds,
        solver_name,
        time_limit,
        tee,
        solver_options,
        tolerance,
    ) = task
    computer = _get_parallel_relu_computer()
    computer.relu_bound_tolerance = float(tolerance)
    lower_model, lower_expr = computer._build_later_layer_preactivation_bound_model(
        generator_name,
        int(physical_generator_idx),
        int(scenario_idx),
        int(time_idx),
        int(linear_idx),
        int(neuron_idx),
        previous_activation_bounds,
        "lower",
    )
    lower_value, lower_termination = computer._solve_bound_model(
        lower_model,
        lower_expr,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    upper_model, upper_expr = computer._build_later_layer_preactivation_bound_model(
        generator_name,
        int(physical_generator_idx),
        int(scenario_idx),
        int(time_idx),
        int(linear_idx),
        int(neuron_idx),
        previous_activation_bounds,
        "upper",
    )
    upper_value, upper_termination = computer._solve_bound_model(
        upper_model,
        upper_expr,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=tee,
        solver_options=solver_options,
    )
    if lower_value is None or upper_value is None:
        raise RuntimeError(
            f"Could not compute later-layer ReLU bounds for {generator_name}, "
            f"k={scenario_idx}, t={time_idx}, layer={linear_idx}, node={neuron_idx}: "
            f"lower={lower_termination}, upper={upper_termination}"
        )
    L = float(lower_value)
    U = float(upper_value)
    if L > U + float(tolerance):
        raise ValueError(
            f"{generator_name}: invalid later-layer bounds at k={scenario_idx}, "
            f"t={time_idx}, layer={linear_idx}, node={neuron_idx}: L={L}, U={U}"
        )
    key = (int(scenario_idx), int(time_idx), int(linear_idx), int(neuron_idx))
    return {
        "generator_name": generator_name,
        "key": key,
        "details": {
            "physical_generator_index": int(physical_generator_idx),
            "scenario_idx": int(scenario_idx),
            "time_idx": int(time_idx),
            "linear_idx": int(linear_idx),
            "node": int(neuron_idx),
            "L": L,
            "U": U,
            "h_lower": float(max(0.0, L)),
            "h_upper": float(max(0.0, U)),
            "status": computer._classify_relu_status(L, U, float(tolerance)),
            "bound_method": "activation_bound_optimization",
            "lower_termination_condition": lower_termination,
            "upper_termination_condition": upper_termination,
        },
    }


class DROReLUBoundsComputer(DROPoATighteningMain):
    """Exact scenario-indexed DRO ReLU preactivation tightening."""

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

    @staticmethod
    def _linear_layers(policy: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            layer
            for layer in policy.get("layers", [])
            if str(layer.get("type", "")).lower() == "linear"
        ]

    @classmethod
    def _hidden_linear_layers(cls, policy: dict[str, Any]) -> list[dict[str, Any]]:
        return cls._linear_layers(policy)[:-1]

    @staticmethod
    def _objective_sense(sense: str) -> Any:
        if sense == "lower":
            return minimize
        if sense == "upper":
            return maximize
        raise ValueError(f"Unknown bound sense: {sense}")

    def _ensure_nn_inputs_loaded(self) -> None:
        if self.poa.nn_policy_generator_ids and not self.poa.nn_policies:
            self.poa._load_nn_policies()
        if self.poa.nn_policy_generator_ids and not self.poa.nn_stats:
            self.poa._load_nn_normalization_stats()

    def _build_tightening_sets(self) -> None:
        self.model.scenarios = Set(initialize=range(self.num_empirical_scenarios))
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

    def _build_first_layer_preactivation_bound_model(
        self,
        generator_name: str,
        physical_generator_idx: int,
        scenario_idx: int,
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
                int(scenario_idx),
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
        scenario_idx: int,
        time_idx: int,
        linear_idx: int,
        neuron_idx: int,
        previous_activation_bounds: dict[int, dict[str, float]],
        sense: str,
    ) -> tuple[ConcreteModel, Any]:
        del physical_generator_idx, scenario_idx, time_idx
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
        for option_name, option_value in (solver_options or {}).items():
            solver.options[option_name] = option_value
        results = solver.solve(m, tee=tee, load_solutions=False)
        termination = results.solver.termination_condition
        if len(results.solution) > 0:
            m.solutions.load_from(results)
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

    @staticmethod
    def _classify_relu_status(L: float, U: float, tolerance: float = 1e-9) -> str:
        if U <= tolerance:
            return "inactive"
        if L >= -tolerance:
            return "active"
        return "ambiguous"

    def _count_relu_bound_programs(self) -> int:
        total_programs = 0
        for physical_generator_idx in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(physical_generator_idx)]
            policy = self.nn_policies[generator_name]
            for layer in self._hidden_linear_layers(policy):
                total_programs += (
                    2
                    * self.num_empirical_scenarios
                    * self.num_time_steps
                    * len(layer["bias"])
                )
        return int(total_programs)

    def _log_relu_bound_result(
        self,
        bound_name: str,
        generator_name: str,
        scenario_idx: int,
        time_idx: int,
        linear_idx: int,
        neuron_idx: int,
        bound_value: Optional[float],
        termination_condition: str,
    ) -> None:
        current = int(getattr(self, "_relu_bound_log_count", 0)) + 1
        self._relu_bound_log_count = current
        total = getattr(self, "_relu_bound_log_total", "?")
        action = "minimize" if bound_name == "lower" else "maximize"
        print(
            f"[DRO ReLU done {current}/{total}] {action} "
            f"relu_z({generator_name}, k={int(scenario_idx)}, t={int(time_idx)}, "
            f"layer={int(linear_idx)}, node={int(neuron_idx)}) -> {bound_value} "
            f"({termination_condition})",
            flush=True,
        )

    def summarize_nn_feature_bounds(self) -> dict[str, Any]:
        self._ensure_nn_inputs_loaded()
        if not self.nn_policy_generator_ids:
            return {}
        summary: dict[str, Any] = {}
        for physical_generator_idx in self.nn_policy_generator_ids:
            i = int(physical_generator_idx)
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies.get(generator_name)
            if not policy:
                continue
            feature_summary: dict[str, Any] = {}
            for feature_name in policy["feature_columns"]:
                feature_min, feature_max = self._nn_feature_bounds(generator_name, feature_name)
                feature_summary[feature_name] = {
                    "source": "dro_regime_support",
                    "raw_bound_method": "embedded_in_first_layer_support_optimization",
                    "normalization_min": float(feature_min),
                    "normalization_max": float(feature_max),
                }
            summary[generator_name] = {
                "physical_generator_index": i,
                "feature_bound_source": "dro_regime_support",
                "features": feature_summary,
            }
        return summary

    def compute_first_layer_preactivation_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]:
        self._ensure_nn_inputs_loaded()
        tolerance = float(getattr(self, "relu_bound_tolerance", 1e-9))
        bounds: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]] = {}
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)

        for physical_generator_idx in self.nn_policy_generator_ids:
            i = int(physical_generator_idx)
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            hidden_layers = self._hidden_linear_layers(policy)
            if not hidden_layers:
                bounds[generator_name] = {}
                continue
            first_layer_width = len(hidden_layers[0]["bias"])
            generator_bounds: dict[tuple[int, int, int, int], dict[str, Any]] = {}
            tasks = [
                (
                    generator_name,
                    i,
                    int(k),
                    int(time_idx),
                    int(neuron_idx),
                    solver_name,
                    time_limit,
                    tee,
                    solver_options,
                    tolerance,
                )
                for k in range(self.num_empirical_scenarios)
                for time_idx in range(self.num_time_steps)
                for neuron_idx in range(first_layer_width)
            ]
            workers = self._resolve_parallel_workers(parallel_workers, len(tasks))
            if workers > 1:
                print(
                    f"Running DRO first-layer ReLU programs for {generator_name} "
                    f"with {workers} worker processes",
                    flush=True,
                )
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_parallel_relu_computer,
                    initargs=(self._parallel_stage_state(),),
                ) as executor:
                    future_to_task = {
                        executor.submit(_solve_parallel_first_layer_relu_bound, task): task
                        for task in tasks
                    }
                    for future in as_completed(future_to_task):
                        result = future.result()
                        key = tuple(result["key"])
                        details = result["details"]
                        generator_bounds[key] = details
                        self._log_relu_bound_result(
                            "lower",
                            generator_name,
                            details["scenario_idx"],
                            details["time_idx"],
                            0,
                            details["node"],
                            details["L"],
                            details["lower_termination_condition"],
                        )
                        self._log_relu_bound_result(
                            "upper",
                            generator_name,
                            details["scenario_idx"],
                            details["time_idx"],
                            0,
                            details["node"],
                            details["U"],
                            details["upper_termination_condition"],
                        )
            else:
                global _PARALLEL_RELU_COMPUTER
                _PARALLEL_RELU_COMPUTER = self
                for task in tasks:
                    result = _solve_parallel_first_layer_relu_bound(task)
                    key = tuple(result["key"])
                    details = result["details"]
                    generator_bounds[key] = details
                    self._log_relu_bound_result(
                        "lower",
                        generator_name,
                        details["scenario_idx"],
                        details["time_idx"],
                        0,
                        details["node"],
                        details["L"],
                        details["lower_termination_condition"],
                    )
                    self._log_relu_bound_result(
                        "upper",
                        generator_name,
                        details["scenario_idx"],
                        details["time_idx"],
                        0,
                        details["node"],
                        details["U"],
                        details["upper_termination_condition"],
                    )
            bounds[generator_name] = generator_bounds
        return bounds

    def compute_later_layer_preactivation_bounds(
        self,
        existing_scenario_bounds: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]],
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]:
        self._ensure_nn_inputs_loaded()
        tolerance = float(getattr(self, "relu_bound_tolerance", 1e-9))
        solver_options = self._solver_options_with_threads(solver_name, solver_threads)
        for physical_generator_idx in self.nn_policy_generator_ids:
            i = int(physical_generator_idx)
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            hidden_layers = self._hidden_linear_layers(policy)
            generator_bounds = existing_scenario_bounds.setdefault(generator_name, {})

            for linear_idx in range(1, len(hidden_layers)):
                layer_width = len(hidden_layers[linear_idx]["bias"])
                previous_width = len(hidden_layers[linear_idx - 1]["bias"])
                tasks: list[tuple[Any, ...]] = []
                for k in range(self.num_empirical_scenarios):
                    for time_idx in range(self.num_time_steps):
                        previous_activation_bounds = {
                            int(prev_node): {
                                "h_lower": float(
                                    generator_bounds[
                                        (int(k), int(time_idx), linear_idx - 1, prev_node)
                                    ]["h_lower"]
                                ),
                                "h_upper": float(
                                    generator_bounds[
                                        (int(k), int(time_idx), linear_idx - 1, prev_node)
                                    ]["h_upper"]
                                ),
                            }
                            for prev_node in range(previous_width)
                        }
                        for neuron_idx in range(layer_width):
                            tasks.append(
                                (
                                    generator_name,
                                    i,
                                    int(k),
                                    int(time_idx),
                                    int(linear_idx),
                                    int(neuron_idx),
                                    previous_activation_bounds,
                                    solver_name,
                                    time_limit,
                                    tee,
                                    solver_options,
                                    tolerance,
                                )
                            )

                workers = self._resolve_parallel_workers(parallel_workers, len(tasks))
                if workers > 1:
                    print(
                        f"Running DRO later-layer ReLU programs for {generator_name} "
                        f"layer {linear_idx} with {workers} worker processes",
                        flush=True,
                    )
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_initialize_parallel_relu_computer,
                        initargs=(self._parallel_stage_state(),),
                    ) as executor:
                        future_to_task = {
                            executor.submit(_solve_parallel_later_layer_relu_bound, task): task
                            for task in tasks
                        }
                        result_iterable = [future.result() for future in as_completed(future_to_task)]
                else:
                    global _PARALLEL_RELU_COMPUTER
                    _PARALLEL_RELU_COMPUTER = self
                    result_iterable = [
                        _solve_parallel_later_layer_relu_bound(task) for task in tasks
                    ]

                for result in result_iterable:
                    key = tuple(result["key"])
                    details = result["details"]
                    generator_bounds[key] = details
                    self._log_relu_bound_result(
                        "lower",
                        generator_name,
                        details["scenario_idx"],
                        details["time_idx"],
                        details["linear_idx"],
                        details["node"],
                        details["L"],
                        details["lower_termination_condition"],
                    )
                    self._log_relu_bound_result(
                        "upper",
                        generator_name,
                        details["scenario_idx"],
                        details["time_idx"],
                        details["linear_idx"],
                        details["node"],
                        details["U"],
                        details["upper_termination_condition"],
                    )
        return existing_scenario_bounds

    def aggregate_scenario_relu_bounds_to_regime(
        self,
        scenario_bounds: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]],
        tolerance: float = 1e-9,
    ) -> dict[str, dict[tuple[int, int, int], dict[str, Any]]]:
        aggregated: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
        for generator_name, generator_bounds in (scenario_bounds or {}).items():
            grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
            for key, details in generator_bounds.items():
                if len(tuple(key)) != 4:
                    raise ValueError(
                        f"Scenario ReLU bound key for {generator_name} must be k,t,layer,node"
                    )
                _k, time_idx, linear_idx, node = tuple(int(part) for part in key)
                grouped.setdefault((time_idx, linear_idx, node), []).append(dict(details))

            generator_payload: dict[tuple[int, int, int], dict[str, Any]] = {}
            for regime_key, values in grouped.items():
                time_idx, linear_idx, node = regime_key
                L_values = [float(item["L"]) for item in values]
                U_values = [float(item["U"]) for item in values]
                h_lower_values = [float(item["h_lower"]) for item in values]
                h_upper_values = [float(item["h_upper"]) for item in values]
                L = min(L_values)
                U = max(U_values)
                h_lower = min(h_lower_values)
                h_upper = max(h_upper_values)
                generator_payload[regime_key] = {
                    "physical_generator_index": int(values[0]["physical_generator_index"]),
                    "time_idx": int(time_idx),
                    "linear_idx": int(linear_idx),
                    "node": int(node),
                    "L": float(L),
                    "U": float(U),
                    "h_lower": float(h_lower),
                    "h_upper": float(h_upper),
                    "status": self._classify_relu_status(L, U, tolerance),
                    "bound_method": "scenario_indexed_dro_relu_obbt",
                    "aggregation_rule": "min L over k, max U over k",
                    "scenario_summary": {
                        "num_scenarios": len(values),
                        "min_L": float(min(L_values)),
                        "max_L": float(max(L_values)),
                        "min_U": float(min(U_values)),
                        "max_U": float(max(U_values)),
                    },
                    "lower_termination_condition_summary": {
                        str(item["scenario_idx"]): item.get("lower_termination_condition")
                        for item in values
                    },
                    "upper_termination_condition_summary": {
                        str(item["scenario_idx"]): item.get("upper_termination_condition")
                        for item in values
                    },
                }
            aggregated[generator_name] = generator_payload
        return aggregated

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

    def _jsonify_scenario_relu_bounds(
        self,
        bounds: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]],
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

    def _relu_fixed_binaries(self, report: dict[str, Any]) -> dict[str, Any]:
        fixed: dict[str, Any] = {}
        relu_bounds = (
            report.get("scenario_nn_relu_bounds", {})
            or report.get("nn_relu_bounds", {})
            or {}
        )
        for generator_name, entries in relu_bounds.items():
            for key, details in (entries or {}).items():
                status = str(details.get("status", "")).lower()
                if status not in {"active", "inactive"}:
                    continue
                parsed_key = self._parse_json_index(key)
                fixed.setdefault(generator_name, {})[key] = {
                    "fixed_value": 1 if status == "active" else 0,
                    "status": status,
                    "scenario_idx": (
                        int(details.get("scenario_idx", parsed_key[0]))
                        if len(parsed_key) == 4
                        else None
                    ),
                    "time_idx": int(details.get("time_idx", parsed_key[-3])),
                    "linear_idx": int(details.get("linear_idx", parsed_key[-2])),
                    "node": int(details.get("node", parsed_key[-1])),
                }
        return fixed

    def compute_nn_relu_bounds_report(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        tolerance: float = 1e-9,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        self.relu_bound_tolerance = float(tolerance)
        self._ensure_nn_inputs_loaded()
        if not getattr(self.poa, "nn_policy_generator_ids", []):
            return {
                "metadata": {
                    **self._metadata(),
                    "tightening_scope": "scenario_wise",
                    "bound_methods": {
                        "first_layer": "scenario_indexed_dro_support_optimization",
                        "later_layers": "activation_bound_optimization",
                        "aggregation": "diagnostic regime_* fields use min L over k, max U over k",
                    },
                },
                "nn_feature_bounds": {},
                "scenario_nn_relu_bounds": {},
                "regime_nn_relu_bounds": {},
                "nn_relu_bounds": {},
                "summary": {},
                "warnings": [],
                "optimization_results": {},
                "relu_fixed_binaries": {},
            }

        self.nn_bound_warnings = []
        self._relu_bound_log_total = self._count_relu_bound_programs()
        self._relu_bound_log_count = 0
        print(
            f"\nDRO ReLU preactivation-bound optimization programs: "
            f"{self._relu_bound_log_total}",
            flush=True,
        )
        first_layer_bounds = self.compute_first_layer_preactivation_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        scenario_bounds = self.compute_later_layer_preactivation_bounds(
            first_layer_bounds,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        regime_bounds = self.aggregate_scenario_relu_bounds_to_regime(
            scenario_bounds,
            tolerance=tolerance,
        )
        scenario_json_bounds = self._jsonify_scenario_relu_bounds(scenario_bounds)
        regime_json_bounds = self._jsonify_relu_bounds(regime_bounds)
        nn_feature_bounds = self.summarize_nn_feature_bounds()
        summary = self._summarize_relu_bounds(scenario_bounds)
        report = {
            "metadata": {
                **self._metadata(),
                "description": (
                    "Exact scenario-indexed DRO ReLU preactivation bounds. "
                    "Regime-wide aggregations are saved only as diagnostics/fallbacks."
                ),
                "tightening_scope": "scenario_wise",
                "bound_methods": {
                    "first_layer": "scenario_indexed_dro_support_optimization",
                    "later_layers": "activation_bound_optimization",
                    "aggregation": "diagnostic regime_* fields use min L over k, max U over k",
                },
                "tolerance": float(tolerance),
            },
            "nn_feature_bounds": nn_feature_bounds,
            "scenario_nn_relu_bounds": scenario_json_bounds,
            "regime_nn_relu_bounds": regime_json_bounds,
            "nn_relu_bounds": scenario_json_bounds,
            "summary": summary,
            "warnings": list(getattr(self, "nn_bound_warnings", [])),
            "optimization_results": {
                generator_name: {
                    self._json_key(index): {
                        "lower_termination_condition_summary": details.get(
                            "lower_termination_condition_summary"
                        ),
                        "upper_termination_condition_summary": details.get(
                            "upper_termination_condition_summary"
                        ),
                        "bound_method": details.get("bound_method"),
                    }
                    for index, details in generator_bounds.items()
                }
                for generator_name, generator_bounds in regime_bounds.items()
            },
        }
        report["relu_fixed_binaries"] = self._relu_fixed_binaries(report)
        self.nn_feature_bounds = nn_feature_bounds
        self.nn_relu_bounds_report = report
        self.poa._set_nn_relu_bounds_from_report(report)
        return report

    def compute_relu_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        tolerance: float = 1e-9,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        return self.compute_nn_relu_bounds_report(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            tolerance=tolerance,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )

    def run_relu_bounds(
        self,
        output_path: str | Path | None = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        tolerance: float = 1e-9,
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        output_path = output_path or self._resolve_output_paths()["relu_bounds"]
        start = time.perf_counter()
        report = self.compute_relu_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            tolerance=tolerance,
            parallel_workers=parallel_workers,
            solver_threads=solver_threads,
        )
        report.setdefault("metadata", {}).update(
            {
                **self._metadata(),
                "description": (
                    "Scenario-wise DRO ReLU preactivation bounds. Main keys are "
                    "generator_name -> k,t,linear_idx,node; regime_* keys are diagnostics."
                ),
                "tightening_scope": "scenario_wise",
                "relu_tolerance": float(tolerance),
                "runtime_seconds": time.perf_counter() - start,
            }
        )
        self.poa._set_nn_relu_bounds_from_report(report)
        self.tightening_data["nn_relu_bounds_report"] = report
        self.tightening_data["scenario_nn_relu_bounds"] = report.get(
            "scenario_nn_relu_bounds",
            {},
        )
        self.tightening_data["regime_nn_relu_bounds"] = report.get("regime_nn_relu_bounds", {})
        self.tightening_data["nn_relu_bounds"] = report.get("nn_relu_bounds", {})
        return self._save_stage_report("relu_bounds", report, output_path)
