from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pyomo.environ import value


class DROPoAResults:
    def _safe_value(self, expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        if raw_value is None:
            return None
        return float(raw_value)

    def _profile_values(self, var: Any, *leading_indices: int) -> list[Optional[float]]:
        return [
            self._safe_value(var[(*leading_indices, t)])
            for t in range(self.num_time_steps)
        ]

    def _physical_capacity_profile_values(self, k: int, i: int) -> list[Optional[float]]:
        m = self.model
        return [
            self._safe_value(
                sum(
                    m.P_max_block[k, i, b, t]
                    for b in self.local_blocks_by_generator[int(i)]
                )
            )
            for t in range(self.num_time_steps)
        ]

    def _physical_dispatch_profile_values(self, var: Any, k: int, i: int) -> list[Optional[float]]:
        return [
            self._safe_value(
                sum(var[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            )
            for t in range(self.num_time_steps)
        ]

    def _json_serializable_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {str(key): self._json_serializable_payload(value) for key, value in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [self._json_serializable_payload(value) for value in payload]
        if isinstance(payload, np.generic):
            return payload.item()
        if isinstance(payload, (str, int, float, bool)) or payload is None:
            return payload
        return str(payload)

    def _json_serializable_mccormick_bounds(self) -> Optional[dict[str, Any]]:
        if getattr(self, "mccormick_bounds", None) is None:
            return None
        return self._json_serializable_payload(self.mccormick_bounds)

    def extract_results(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        inner_objective = self._safe_value(m.objective)
        ratio_mode = self.objective_mode in {
            "mccormick",
            "piecewise_mccormick",
        }
        poa_difference_values = [
            (
                self._safe_value(m.C_eq[k]) - self._safe_value(m.C_opt[k])
                if self._safe_value(m.C_eq[k]) is not None
                and self._safe_value(m.C_opt[k]) is not None
                else None
            )
            for k in m.scenarios
        ]
        wasserstein_values = [self._safe_value(m.wasserstein_distance[k]) for k in m.scenarios]
        average_poa = (
            float(np.mean([v for v in poa_difference_values if v is not None]))
            if any(v is not None for v in poa_difference_values)
            else None
        )
        average_wasserstein = (
            float(np.mean([v for v in wasserstein_values if v is not None]))
            if any(v is not None for v in wasserstein_values)
            else None
        )
        scenario_ratio_metrics: dict[int, dict[str, Optional[float]]] = {}
        poa_ratio_values: list[float] = []
        relaxed_poa_values: list[float] = []
        for k in range(self.num_empirical_scenarios):
            C_eq = self._safe_value(m.C_eq[k])
            C_opt = self._safe_value(m.C_opt[k])
            poa_ratio = (
                C_eq / C_opt
                if C_eq is not None and C_opt not in (None, 0.0)
                else None
            )
            relaxed_PoA = self._safe_value(m.PoA[k]) if ratio_mode else None
            ratio_relaxation_gap = (
                relaxed_PoA - poa_ratio
                if relaxed_PoA is not None and poa_ratio is not None
                else None
            )
            if poa_ratio is not None:
                poa_ratio_values.append(float(poa_ratio))
            if relaxed_PoA is not None:
                relaxed_poa_values.append(float(relaxed_PoA))
            scenario_ratio_metrics[k] = {
                "C_eq": C_eq,
                "C_opt": C_opt,
                "PoA_ratio": poa_ratio,
                "relaxed_PoA": relaxed_PoA,
                "relaxed_phi": relaxed_PoA,
                "ratio_relaxation_gap": ratio_relaxation_gap,
            }
        average_poa_ratio = (
            float(np.mean(poa_ratio_values)) if poa_ratio_values else None
        )
        average_relaxed_PoA = (
            float(np.mean(relaxed_poa_values))
            if ratio_mode and relaxed_poa_values
            else None
        )

        scenario_results = []
        for k in range(self.num_empirical_scenarios):
            ratio_metrics = scenario_ratio_metrics[k]
            generators: dict[str, Any] = {}
            for i, generator_name in enumerate(self.physical_generator_names):
                block_results = []
                for b in self.local_blocks_by_generator[i]:
                    global_block = self.local_to_global_block[(i, b)]
                    block_results.append(
                        {
                            "local_block_index": int(b),
                            "global_block_index": int(global_block),
                            "block_name": self.block_names[global_block],
                            "capacity_profile": self._profile_values(m.P_max_block, k, i, b),
                            "alpha_profile": self._profile_values(m.alpha, k, i, b),
                            "equilibrium_dispatch": self._profile_values(m.P_eq, k, i, b),
                            "optimal_dispatch": self._profile_values(m.P_opt, k, i, b),
                            "true_cost": float(self.block_cost_vector[global_block]),
                        }
                    )
                generators[generator_name] = {
                    "physical_generator_index": int(i),
                    "is_wind": i in self.wind_physical_generator_ids,
                    "empirical_physical_capacity_profile": list(
                        self.empirical_Pmax_phys[k][i]
                    ),
                    "optimized_physical_capacity_profile": self._physical_capacity_profile_values(k, i),
                    "equilibrium_physical_dispatch": self._physical_dispatch_profile_values(
                        m.P_eq,
                        k,
                        i,
                    ),
                    "optimal_physical_dispatch": self._physical_dispatch_profile_values(
                        m.P_opt,
                        k,
                        i,
                    ),
                    "blocks": block_results,
                }

            scenario_results.append(
                {
                    "k": int(k),
                    "scenario_id": self.empirical_scenario_ids[k],
                    "empirical_demand_profile": list(self.empirical_D[k]),
                    "optimized_demand_profile": self._profile_values(m.D, k),
                    "empirical_physical_capacity_profiles": {
                        self.physical_generator_names[i]: list(self.empirical_Pmax_phys[k][i])
                        for i in range(self.num_physical_generators)
                    },
                    "optimized_physical_capacity_profiles": {
                        self.physical_generator_names[i]: self._physical_capacity_profile_values(k, i)
                        for i in range(self.num_physical_generators)
                    },
                    "wasserstein_distance": self._safe_value(m.wasserstein_distance[k]),
                    "C_eq": ratio_metrics["C_eq"],
                    "C_opt": ratio_metrics["C_opt"],
                    "PoA_difference": poa_difference_values[k],
                    "PoA_ratio": ratio_metrics["PoA_ratio"],
                    "relaxed_PoA": ratio_metrics["relaxed_PoA"],
                    "relaxed_phi": ratio_metrics["relaxed_phi"],
                    "ratio_relaxation_gap": ratio_metrics["ratio_relaxation_gap"],
                    "equilibrium_price_profile": self._profile_values(m.lambda_eq, k),
                    "optimal_price_profile": self._profile_values(m.lambda_opt, k),
                    "generators": generators,
                }
            )

        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(
                    self.solver_results.solver.termination_condition
                ),
            }

        dro_objective_with_epsilon = (
            inner_objective + self.eta * self.epsilon
            if inner_objective is not None
            else None
        )
        return {
            "reference_case": self.reference_case,
            "case_label": getattr(self, "case_label", ""),
            "regime_set": self.regime_set,
            "regime_name": self.regime_name,
            "num_time_steps": self.num_time_steps,
            "num_empirical_scenarios": self.num_empirical_scenarios,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "objective_mode": self.objective_mode,
            "mccormick_bounds": self._json_serializable_mccormick_bounds(),
            "ratio_bounds": self._json_serializable_mccormick_bounds(),
            "optimal_cost_bounds": self._json_serializable_payload(
                self.optimal_cost_bounds
            ),
            "scenario_optimal_cost_bounds": self._json_serializable_payload(
                self.scenario_optimal_cost_bounds
            ),
            "optimal_cost_bound_optimization_results": self._json_serializable_payload(
                self.optimal_cost_bound_optimization_results
            ),
            "inner_objective": inner_objective,
            "dro_objective_with_epsilon": dro_objective_with_epsilon,
            "average_poa_difference": average_poa,
            "average_poa_ratio": average_poa_ratio,
            "average_relaxed_PoA": average_relaxed_PoA,
            "average_relaxed_phi": average_relaxed_PoA,
            "average_wasserstein_distance": average_wasserstein,
            "solver": solver_summary,
            "selected_regime_parameters": dict(self.selected_regime_parameters),
            "demand_shape": list(self.demand_shape),
            "wind_shape": list(self.wind_shape),
            "block_names": list(self.block_names),
            "physical_generator_names": list(self.physical_generator_names),
            "block_to_physical": dict(self.block_to_physical),
            "physical_to_block_indices": {
                str(i): list(blocks)
                for i, blocks in enumerate(self.physical_to_block_indices)
            },
            "policy_type": (
                "nn_policy"
                if self.nn_policy_generator_ids
                else "true_cost_baseline"
            ),
            "scenarios": scenario_results,
        }

    def save_results(self, output_path: str | Path) -> Path:
        results = self.extract_results()
        path = Path(output_path)
        if not path.suffix:
            path = path.with_suffix(".json")
        if path.suffix.lower() != ".json":
            raise ValueError("output_path must end with .json or have no suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2)
        return path

    def solution_summary(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        inner_objective = self._safe_value(m.objective)
        poa_difference_values = []
        for k in m.scenarios:
            C_eq = self._safe_value(m.C_eq[k])
            C_opt = self._safe_value(m.C_opt[k])
            if C_eq is not None and C_opt is not None:
                poa_difference_values.append(C_eq - C_opt)
        average_poa = (
            float(np.mean(poa_difference_values)) if poa_difference_values else None
        )
        ratio_mode = self.objective_mode in {
            "mccormick",
            "piecewise_mccormick",
        }
        poa_ratio_values = []
        relaxed_poa_values = []
        for k in m.scenarios:
            C_eq = self._safe_value(m.C_eq[k])
            C_opt = self._safe_value(m.C_opt[k])
            if C_eq is not None and C_opt not in (None, 0.0):
                poa_ratio_values.append(C_eq / C_opt)
            if ratio_mode:
                relaxed_PoA = self._safe_value(m.PoA[k])
                if relaxed_PoA is not None:
                    relaxed_poa_values.append(relaxed_PoA)
        average_poa_ratio = (
            float(np.mean(poa_ratio_values)) if poa_ratio_values else None
        )
        average_relaxed_PoA = (
            float(np.mean(relaxed_poa_values))
            if ratio_mode and relaxed_poa_values
            else None
        )
        average_wasserstein = self._safe_value(
            sum(m.wasserstein_distance[k] for k in m.scenarios)
            / self.num_empirical_scenarios
        )
        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(
                    self.solver_results.solver.termination_condition
                ),
            }
        return {
            "reference_case": self.reference_case,
            "regime_set": self.regime_set,
            "regime_name": self.regime_name,
            "num_time_steps": self.num_time_steps,
            "num_empirical_scenarios": self.num_empirical_scenarios,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "objective_mode": self.objective_mode,
            "mccormick_bounds": self._json_serializable_mccormick_bounds(),
            "ratio_bounds": self._json_serializable_mccormick_bounds(),
            "optimal_cost_bounds": self._json_serializable_payload(
                self.optimal_cost_bounds
            ),
            "scenario_optimal_cost_bounds": self._json_serializable_payload(
                self.scenario_optimal_cost_bounds
            ),
            "optimal_cost_bound_optimization_results": self._json_serializable_payload(
                self.optimal_cost_bound_optimization_results
            ),
            "inner_objective": inner_objective,
            "dro_objective_with_epsilon": (
                inner_objective + self.eta * self.epsilon
                if inner_objective is not None
                else None
            ),
            "average_poa_difference": average_poa,
            "average_poa_ratio": average_poa_ratio,
            "average_relaxed_PoA": average_relaxed_PoA,
            "average_relaxed_phi": average_relaxed_PoA,
            "average_wasserstein_distance": average_wasserstein,
            "solver": solver_summary,
        }
