import json
from pathlib import Path
from typing import Any, Optional

from pyomo.environ import value


class PoAResults:
    def _safe_value(self, expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        if raw_value is None:
            return None
        return float(raw_value)

    def _profile_values(self, var: Any, *leading_indices: int) -> list[Optional[float]]:
        return [
            self._safe_value(var[(*leading_indices, t)] if leading_indices else var[t])
            for t in range(self.num_time_steps)
        ]

    def extract_objective_metrics(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        C_eq = self._safe_value(m.C_eq)
        C_opt = self._safe_value(m.C_opt)
        difference_proxy = None
        ex_post_ratio = None
        if C_eq is not None and C_opt is not None:
            difference_proxy = C_eq - C_opt
            if C_opt != 0.0:
                ex_post_ratio = C_eq / C_opt

        objective_value = self._safe_value(m.objective)
        metrics: dict[str, Any] = {
            "objective_mode": self.objective_mode,
            "model_objective_value": objective_value,
            "objective_value": objective_value,
            "PoA_difference": difference_proxy,
            "PoA_ratio": ex_post_ratio,
            "C_eq": C_eq,
            "C_opt": C_opt,
            "difference_proxy": difference_proxy,
            "ex_post_ratio": ex_post_ratio,
        }

        if self.objective_mode in {
            "mccormick",
            "piecewise_mccormick",
        }:
            relaxed_poa = self._safe_value(m.PoA)
            z_mccormick_product = self._safe_value(m.z_mccormick_product)
            product_gap = None
            mccormick_gap = None
            if (
                relaxed_poa is not None
                and C_opt is not None
                and z_mccormick_product is not None
            ):
                product_gap = z_mccormick_product - relaxed_poa * C_opt
            if relaxed_poa is not None and ex_post_ratio is not None:
                mccormick_gap = relaxed_poa - ex_post_ratio
            metrics.update(
                {
                    "PoA": relaxed_poa,
                    "phi": relaxed_poa,
                    "z_mccormick_product": z_mccormick_product,
                    "z_ratio_product": z_mccormick_product,
                    "mccormick_product_gap": product_gap,
                    "mccormick_gap": mccormick_gap,
                    "ratio_gap": mccormick_gap,
                }
            )
            if self.objective_mode == "piecewise_mccormick":
                metrics.update(self._extract_piecewise_mccormick_metrics())
        return metrics

    def _extract_piecewise_mccormick_metrics(self) -> dict[str, Any]:
        m = self.model
        breakpoints = list((self.mccormick_bounds or {}).get("C_opt_breakpoints", []))
        PoA_L, PoA_U = (self.mccormick_bounds or {}).get("PoA", (None, None))
        piece_indices = [int(k) for k in m.mccormick_piece_index]
        delta_values = {
            k: self._safe_value(m.mccormick_piece_active[k])
            for k in piece_indices
        }
        selected_delta_sum = sum(
            value for value in delta_values.values() if value is not None
        )
        active_piece = max(
            piece_indices,
            key=lambda k: (
                delta_values[k] if delta_values[k] is not None else float("-inf")
            ),
        )
        active_delta = delta_values[active_piece]
        active_lower = breakpoints[active_piece]
        active_upper = breakpoints[active_piece + 1]
        active_PoA_piece = self._safe_value(m.PoA_piece[active_piece])
        active_C_opt_piece = self._safe_value(m.C_opt_piece[active_piece])
        active_z_piece = self._safe_value(m.z_mccormick_piece[active_piece])

        slacks: dict[str, Optional[float]] = {
            "active_mccormick_slack_lower_1": None,
            "active_mccormick_slack_lower_2": None,
            "active_mccormick_slack_upper_1": None,
            "active_mccormick_slack_upper_2": None,
        }
        if (
            PoA_L is not None
            and PoA_U is not None
            and active_delta is not None
            and active_PoA_piece is not None
            and active_C_opt_piece is not None
            and active_z_piece is not None
        ):
            lower_1_rhs = (
                PoA_L * active_C_opt_piece
                + active_lower * active_PoA_piece
                - PoA_L * active_lower * active_delta
            )
            lower_2_rhs = (
                PoA_U * active_C_opt_piece
                + active_upper * active_PoA_piece
                - PoA_U * active_upper * active_delta
            )
            upper_1_rhs = (
                PoA_U * active_C_opt_piece
                + active_lower * active_PoA_piece
                - PoA_U * active_lower * active_delta
            )
            upper_2_rhs = (
                PoA_L * active_C_opt_piece
                + active_upper * active_PoA_piece
                - PoA_L * active_upper * active_delta
            )
            slacks = {
                "active_mccormick_slack_lower_1": active_z_piece - lower_1_rhs,
                "active_mccormick_slack_lower_2": active_z_piece - lower_2_rhs,
                "active_mccormick_slack_upper_1": upper_1_rhs - active_z_piece,
                "active_mccormick_slack_upper_2": upper_2_rhs - active_z_piece,
            }

        relaxed_poa = self._safe_value(m.PoA)
        C_opt = self._safe_value(m.C_opt)
        sum_z_piece = sum(
            self._safe_value(m.z_mccormick_piece[k]) or 0.0
            for k in piece_indices
        )
        piecewise_product_gap = None
        if relaxed_poa is not None and C_opt is not None:
            piecewise_product_gap = sum_z_piece - relaxed_poa * C_opt

        return {
            "active_piece": int(active_piece),
            "active_piece_lower": active_lower,
            "active_piece_upper": active_upper,
            "num_pieces": len(piece_indices),
            "C_opt_breakpoints": list(breakpoints),
            "piecewise_product_gap": piecewise_product_gap,
            "piecewise_selected_delta_sum": selected_delta_sum,
            "active_piece_delta_value": active_delta,
            **slacks,
        }

    def check_dual_bound_activity(self, tol: float = 1e-5) -> list[dict]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        self._ensure_loaded_bounds_prepared()
        m = self.model
        active_bounds: list[dict] = []

        def add_if_active(component_name: str, var: Any, indices: tuple, bound: float) -> None:
            var_value = self._safe_value(var[indices])
            if var_value is None or var_value < bound - tol:
                return
            active_bounds.append(
                {
                    "component": component_name,
                    "indices": [int(idx) for idx in indices],
                    "value": var_value,
                    "bound": float(bound),
                    "relative_to_bound": var_value / bound if bound != 0 else None,
                }
            )

        capacity_components = (
            ("mu_upper_eq", m.mu_upper_eq, self.M_mu_upper_eq),
            ("mu_lower_eq", m.mu_lower_eq, self.M_mu_lower_eq),
            ("mu_upper_opt", m.mu_upper_opt, self.M_mu_upper_opt),
            ("mu_lower_opt", m.mu_lower_opt, self.M_mu_lower_opt),
        )
        for component_name, var, bound_map in capacity_components:
            for i, b in m.generator_blocks:
                for t in m.time_steps:
                    index = (int(i), int(b), int(t))
                    add_if_active(
                        component_name,
                        var,
                        (i, b, t),
                        bound_map[index],
                    )

        ramp_components = (
            ("mu_ramp_up_eq", m.mu_ramp_up_eq, self.M_mu_ramp_up_eq),
            ("mu_ramp_down_eq", m.mu_ramp_down_eq, self.M_mu_ramp_down_eq),
            ("mu_ramp_up_opt", m.mu_ramp_up_opt, self.M_mu_ramp_up_opt),
            ("mu_ramp_down_opt", m.mu_ramp_down_opt, self.M_mu_ramp_down_opt),
        )
        for component_name, var, bound_map in ramp_components:
            for i in m.physical_generators:
                for t in m.time_steps:
                    index = (int(i), int(t))
                    add_if_active(
                        component_name,
                        var,
                        (i, t),
                        bound_map[index],
                    )

        return active_bounds

    def extract_results(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        objective = self.extract_objective_metrics()

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
                        "capacity_profile": self._profile_values(m.P_max_block, i, b),
                        "alpha_profile": self._profile_values(m.alpha, i, b),
                        "equilibrium_dispatch": self._profile_values(m.P_eq, i, b),
                        "optimal_dispatch": self._profile_values(m.P_opt, i, b),
                        "true_cost": float(self.block_cost_vector[global_block]),
                    }
                )
            generators[generator_name] = {
                "physical_generator_index": int(i),
                "is_wind": i in self.wind_physical_generator_ids,
                "physical_capacity_profile": [
                    sum(
                        self._safe_value(m.P_max_block[i, b, t]) or 0.0
                        for b in self.local_blocks_by_generator[i]
                    )
                    for t in range(self.num_time_steps)
                ],
                "equilibrium_physical_dispatch": [
                    sum(
                        self._safe_value(m.P_eq[i, b, t]) or 0.0
                        for b in self.local_blocks_by_generator[i]
                    )
                    for t in range(self.num_time_steps)
                ],
                "optimal_physical_dispatch": [
                    sum(
                        self._safe_value(m.P_opt[i, b, t]) or 0.0
                        for b in self.local_blocks_by_generator[i]
                    )
                    for t in range(self.num_time_steps)
                ],
                "blocks": block_results,
            }

        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(self.solver_results.solver.termination_condition),
            }

        try:
            dual_bound_activity = self.check_dual_bound_activity()
        except Exception:
            dual_bound_activity = []

        return {
            "reference_case": self.reference_case,
            "num_time_steps": self.num_time_steps,
            "objective": objective,
            "demand_profile": self._profile_values(m.D),
            "block_names": list(self.block_names),
            "physical_generator_names": list(self.physical_generator_names),
            "block_to_physical": dict(self.block_to_physical),
            "physical_to_block_indices": {
                str(i): list(blocks)
                for i, blocks in enumerate(self.physical_to_block_indices)
            },
            "generators": generators,
            "equilibrium_price_profile": self._profile_values(m.lambda_eq),
            "optimal_price_profile": self._profile_values(m.lambda_opt),
            "ambiguity_set": {
                "regime_bounds": {
                    "mu_D": list(self.mu_D_bounds),
                    "sigma_D": list(self.sigma_D_bounds),
                    "mu_W": list(self.mu_W_bounds),
                    "sigma_W": list(self.sigma_W_bounds),
                },
                "fixed_parameters": {
                    "rho_D": float(self.demand_rho_fixed),
                    "rho_W": float(self.wind_rho_fixed),
                    "tau_W": float(self.wind_tau_fixed),
                    "kappa": float(self.ambiguity_kappa),
                    "D_ref": float(self.demand_D_ref),
                },
                "selected_regime": {
                    "mu_D": self._safe_value(m.mu_D),
                    "sigma_D": self._safe_value(m.sigma_D),
                    "mu_W": self._safe_value(m.mu_W),
                    "sigma_W": self._safe_value(m.sigma_W),
                },
                "demand": {
                    "shape": list(self.demand_shape),
                    "reference": [
                        self._safe_value(m.demand_reference[t])
                        for t in range(self.num_time_steps)
                    ],
                    "lower": [
                        self._safe_value(m.demand_lower[t])
                        for t in range(self.num_time_steps)
                    ],
                    "upper": [
                        self._safe_value(m.demand_upper[t])
                        for t in range(self.num_time_steps)
                    ],
                    "budget": self._safe_value(m.demand_budget_expr),
                },
                "wind": {
                    self.physical_generator_names[i]: {
                        "installed_capacity": float(self.static_physical_capacity[i]),
                        "shape": list(self.wind_shape),
                        "reference": [
                            self._safe_value(m.wind_reference[i, t])
                            for t in range(self.num_time_steps)
                        ],
                        "lower": [
                            self._safe_value(m.wind_lower[i, t])
                            for t in range(self.num_time_steps)
                        ],
                        "upper": [
                            self._safe_value(m.wind_upper[i, t])
                            for t in range(self.num_time_steps)
                        ],
                        "budget": self._safe_value(m.wind_budget_expr[i]),
                    }
                    for i in self.wind_physical_generator_ids
                },
            },
            "solver": solver_summary,
            "policy_type": (
                "true_cost_baseline"
                if not self.nn_policy_generator_ids
                else (
                    "neural_network"
                    if len(self.nn_policy_generator_ids) == self.num_physical_generators
                    else "mixed_neural_network_true_cost"
                )
            ),
            "nn_policy_generators": list(self.nn_policy_generator_names),
            "true_cost_policy_generators": [
                generator_name
                for i, generator_name in enumerate(self.physical_generator_names)
                if i not in self.nn_policy_generator_ids
            ],
            "dual_bounds": {
                "lambda_bound": float(self.lambda_bound),
                "lambda_bounds": self.lambda_bounds,
                "capacity_dual_bound": float(self.capacity_dual_bound),
                "ramp_dual_bound": float(self.ramp_dual_bound),
            },
            "dual_bound_activity": dual_bound_activity,
            "nn_feature_bounds": self.summarize_nn_feature_bounds(),
            "nn_relu_bounds": self.summarize_nn_relu_bounds(),
            "nn_bound_warnings": list(self.nn_bound_warnings),
            "default_bounds_used": getattr(self, "default_bounds_used", {}),
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
