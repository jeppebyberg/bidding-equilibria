from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel
from models.gradient_based.sensitivities.kkt_sensitivity import (
    EDParameters,
    EDSolution,
    compute_market_sensitivities,
)
from models.gradient_based.sensitivities.profit_sensitivity import (
    ProfitParameters,
    compute_profit_sensitivities,
)

class GradientBidTrainingKKTMS:
    """
    Analytical KKT-gradient label generation by direct bid-trajectory ascent.

    This is the direct-bid counterpart of GradientPolicyTrainingKKTNNMS:

        alpha_j -> ED outcome (P*, lambda*) -> profit pi_j

    Each player's scenario-specific bid trajectories are updated directly, so
    gradients are not averaged across scenarios and there are no policy
    parameters or feature matrices.
    """

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        beta_smooth: float = 0.001,
        learning_rate: float = 0.25,
        learning_rate_decay: float = 0.0,
        min_learning_rate: float = 0.0,
        max_iterations: int = 20,
        conv_tolerance: float = 1e-4,
        gradient_clip_norm: Optional[float] = None,
        gradient_clip_mode: str = "per_block",
        gradient_update_mode: str = "current",
        gradient_history_window: int = 0,
        bid_order_step_fraction: Optional[float] = 0.95,
        bid_order_epsilon: Optional[float] = 1e-6,
        alpha_min: Optional[float] = 0.0,
        alpha_max: Optional[float] = None,
        kkt_regularization: float = 1e-8,
        condition_warning_threshold: float = 1e10,
    ) -> None:
        if beta_smooth <= 0:
            raise ValueError(f"beta_smooth must be positive, got {beta_smooth}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if learning_rate_decay < 0:
            raise ValueError(f"learning_rate_decay must be nonnegative, got {learning_rate_decay}")
        if min_learning_rate < 0:
            raise ValueError(f"min_learning_rate must be nonnegative, got {min_learning_rate}")
        if min_learning_rate > learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if max_iterations < 0:
            raise ValueError(f"max_iterations must be nonnegative, got {max_iterations}")
        if conv_tolerance < 0:
            raise ValueError(f"conv_tolerance must be nonnegative, got {conv_tolerance}")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if gradient_clip_mode not in {"per_block", "global"}:
            raise ValueError("gradient_clip_mode must be either 'per_block' or 'global'")
        if gradient_update_mode not in {"current", "history_average"}:
            raise ValueError("gradient_update_mode must be either 'current' or 'history_average'")
        if gradient_history_window < 0:
            raise ValueError(f"gradient_history_window must be nonnegative, got {gradient_history_window}")
        if bid_order_step_fraction is not None and not (0.0 < bid_order_step_fraction <= 1.0):
            raise ValueError("bid_order_step_fraction must be in (0, 1] when provided")
        if bid_order_epsilon is not None and bid_order_epsilon < 0.0:
            raise ValueError("bid_order_epsilon must be nonnegative when provided")
        if alpha_min is not None and alpha_max is not None and alpha_min > alpha_max:
            raise ValueError("alpha_min cannot exceed alpha_max")
        if kkt_regularization < 0:
            raise ValueError("kkt_regularization must be nonnegative")
        if condition_warning_threshold <= 0:
            raise ValueError("condition_warning_threshold must be positive")

        self.scenarios_df = scenarios_df.copy(deep=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.beta_smooth = float(beta_smooth)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.min_learning_rate = float(min_learning_rate)
        self.current_learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.conv_tolerance = float(conv_tolerance)
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_mode = gradient_clip_mode
        self.gradient_update_mode = gradient_update_mode
        self.gradient_history_window = int(gradient_history_window)
        self.bid_order_step_fraction = (
            None if bid_order_step_fraction is None else float(bid_order_step_fraction)
        )
        self.bid_order_epsilon = None if bid_order_epsilon is None else float(bid_order_epsilon)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.kkt_regularization = float(kkt_regularization)
        self.condition_warning_threshold = float(condition_warning_threshold)

        self.num_scenarios = len(self.scenarios_df)
        self.num_time_steps = self._infer_num_time_steps()
        self._initialize_block_mapping_from_ed()
        self.cost_vector = np.asarray(
            [float(self.costs_df[f"{block}_cost"].iloc[0]) for block in self.block_names],
            dtype=np.float64,
        )
        self.player_index_by_id = {
            int(player["id"]): idx for idx, player in enumerate(self.players_config)
        }

        self._initialize_missing_bid_profiles()
        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)

        self.bid_history: List[List[List[List[float]]]] = []
        self.profit_history_training: List[List[float]] = []
        self.profit_history_training_scenario: List[List[List[float]]] = []
        self.dispatch_history: List[List[List[List[float]]]] = []
        self.clearing_price_history: List[List[List[float]]] = []
        self.gradient_norm_history: List[Dict[int, Dict[int, float]]] = []
        self.gradient_diagnostics_history: List[Dict[int, Dict[str, Any]]] = []
        self.kkt_condition_history: List[Dict[int, Dict[str, float]]] = []
        self.step_norm_history: List[Dict[int, Dict[int, float]]] = []
        self.last_bid_order_step_diagnostics: Dict[str, Any] = {
            "bid_order_step_scale": 1.0,
            "bid_order_limited_entries": 0,
            "bid_order_epsilon": self.bid_order_epsilon,
        }
        self.gradient_history_snapshots: List[pd.DataFrame] = []
        self.learning_rate_history: List[float] = []
        self.iteration = 0
        self.results: Optional[Dict[str, Any]] = None

    def _initialize_block_mapping_from_ed(self) -> None:
        """
        Reuse the ED model's ordering so training, ED solves, and KKT
        sensitivities agree on flattened block and physical-generator indices.
        """
        mapping_model = EconomicDispatchQuadraticModel(
            self.scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=None,
            beta_coeff=self.beta_smooth,
        )
        self.block_names = mapping_model.get_block_names()
        self.num_blocks = len(self.block_names)
        self.physical_generator_names = mapping_model.get_physical_generator_names()
        self.num_physical_generators = len(self.physical_generator_names)
        self.block_to_physical = mapping_model.get_block_to_physical_mapping()
        self.block_to_physical_idx = list(mapping_model.block_to_physical_idx)
        self.physical_to_block_indices = [
            list(blocks) for blocks in mapping_model.physical_to_block_indices
        ]
        self.blocks_by_generator = mapping_model.get_blocks_by_generator()
        self.local_blocks_by_generator = {
            int(generator_idx): list(local_blocks)
            for generator_idx, local_blocks in mapping_model.local_blocks_by_generator.items()
        }
        self.local_to_global_block = mapping_model.get_local_to_global_block_mapping()
        self.global_to_local_block = dict(mapping_model.global_to_local_block)
        self.generator_block_pairs = list(mapping_model.generator_block_pairs)

        # Backwards-compatible aliases for callers that still expect
        # generator_names to mean physical units.
        self.generator_names = list(self.physical_generator_names)
        self.num_generators = self.num_physical_generators

        if any(len(blocks) == 0 for blocks in self.physical_to_block_indices):
            raise ValueError("Every physical generator must have at least one bidding block")
        for block_name in self.block_names:
            if f"{block_name}_cap" not in self.scenarios_df.columns:
                raise ValueError(f"Missing block capacity column '{block_name}_cap'")
            if f"{block_name}_cost" not in self.costs_df.columns:
                raise ValueError(f"Missing block cost column '{block_name}_cost'")

    def _infer_num_time_steps(self) -> int:
        if "time_steps" in self.scenarios_df.columns:
            return int(self.scenarios_df["time_steps"].iloc[0])
        return len(self._as_profile(self.scenarios_df["demand_profile"].iloc[0], "demand_profile"))

    @staticmethod
    def _as_profile(value: Any, column_name: str) -> List[float]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"Column '{column_name}' must contain a list/tuple profile")
        return [float(v) for v in value]

    def _initialize_missing_bid_profiles(self) -> None:
        for block_idx, block_name in enumerate(self.block_names):
            profile_col = f"{block_name}_bid_profile"
            bid_col = f"{block_name}_bid"
            if profile_col not in self.scenarios_df.columns:
                self.scenarios_df[profile_col] = [
                    [float(self.cost_vector[block_idx])] * self.num_time_steps
                    for _ in range(self.num_scenarios)
                ]
            for s in range(self.num_scenarios):
                profile = self._as_profile(self.scenarios_df.at[s, profile_col], profile_col)
                if len(profile) != self.num_time_steps:
                    raise ValueError(
                        f"{profile_col} must have length {self.num_time_steps}, got {len(profile)}"
                    )
                if not np.all(np.isfinite(profile)):
                    raise ValueError(f"{profile_col} contains non-finite values in scenario {s}")
                self.scenarios_df.at[s, profile_col] = [float(v) for v in profile]
                self.scenarios_df.at[s, bid_col] = float(profile[0])

    def _coerce_index_or_name(self, value: Any, names: List[str], label: str) -> int:
        if isinstance(value, str) and not value.strip().lstrip("-").isdigit():
            if value not in names:
                raise ValueError(f"Unknown {label} name '{value}'. Available: {names}")
            return names.index(value)
        idx = int(value)
        if idx < 0 or idx >= len(names):
            raise ValueError(f"{label} index {idx} is out of range [0, {len(names) - 1}]")
        return idx

    def _controlled_physical_generators(self, player_id: int) -> List[int]:
        player = self._get_player_config(player_id)
        if "controlled_generators" not in player:
            return []
        return [
            self._coerce_index_or_name(value, self.physical_generator_names, "physical generator")
            for value in player["controlled_generators"]
        ]

    def _controlled_blocks(self, player_id: int) -> List[int]:
        player = self._get_player_config(player_id)
        if "controlled_blocks" in player:
            return [
                self._coerce_index_or_name(value, self.block_names, "bidding block")
                for value in player["controlled_blocks"]
            ]

        controlled_physical = self._controlled_physical_generators(player_id)
        if not controlled_physical:
            raise ValueError(
                f"Player {player_id} must define either controlled_blocks or controlled_generators"
            )
        controlled_blocks: List[int] = []
        for physical_idx in controlled_physical:
            controlled_blocks.extend(self.physical_to_block_indices[int(physical_idx)])
        return controlled_blocks

    def _controlled_generators(self, player_id: int) -> List[int]:
        """Compatibility alias: bid variables are now controlled blocks."""
        return self._controlled_blocks(player_id)

    def _compute_p_init_from_ed(self, scenarios_df: pd.DataFrame) -> List[List[float]]:
        initial_dispatch = []
        for _, row in scenarios_df.iterrows():
            physical_initial = []
            for block_indices in self.physical_to_block_indices:
                physical_capacity = sum(float(row[f"{self.block_names[b]}_cap"]) for b in block_indices)
                physical_initial.append(0.5 * physical_capacity)
            initial_dispatch.append(physical_initial)

        ed_for_p_init = EconomicDispatchQuadraticModel(
            scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=initial_dispatch,
            beta_coeff=self.beta_smooth,
        )
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def _clip_alpha_array(self, alpha: np.ndarray) -> np.ndarray:
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        lower = -np.inf if self.alpha_min is None else float(self.alpha_min)
        upper = np.inf if self.alpha_max is None else float(self.alpha_max)
        return np.clip(alpha_arr, lower, upper)

    def _learning_rate_for_iteration(self, iteration: int) -> float:
        """
        Inverse-time learning-rate decay.

        iteration is one-based in run(), so iteration 1 uses the base learning
        rate. With learning_rate_decay=0.0 this reduces to the old constant
        learning-rate behavior.
        """
        k = max(1, int(iteration))
        decayed = self.learning_rate / (1.0 + self.learning_rate_decay * float(k - 1))
        return float(max(self.min_learning_rate, decayed))

    def _solve_training_ed_model(self) -> EconomicDispatchQuadraticModel:
        ed = EconomicDispatchQuadraticModel(
            self.scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=self.P_init,
            beta_coeff=self.beta_smooth,
        )
        ed.solve()
        if ed.get_dispatches() is None or ed.get_clearing_prices() is None:
            raise RuntimeError("Quadratic ED solve did not return dispatches and clearing_prices.")
        return ed

    def solve_training_ed(self) -> Tuple[List[List[List[float]]], List[List[float]]]:
        ed = self._solve_training_ed_model()
        dispatches = ed.get_dispatches()
        clearing_prices = ed.get_clearing_prices()
        if dispatches is None or clearing_prices is None:
            raise RuntimeError("Quadratic ED solve did not return dispatches and clearing_prices.")
        return dispatches, clearing_prices

    def _extract_duals_from_ed(self, ed: EconomicDispatchQuadraticModel) -> Dict[str, List[Any]]:
        if hasattr(ed, "get_dual_variables"):
            duals = ed.get_dual_variables()
            required = {"mu_max", "mu_min", "mu_up", "mu_down"}
            if duals is not None and required.issubset(duals):
                return duals

        getters = {
            "mu_max": "get_mu_max",
            "mu_min": "get_mu_min",
            "mu_up": "get_mu_up",
            "mu_down": "get_mu_down",
        }
        if all(hasattr(ed, getter) for getter in getters.values()):
            return {name: getattr(ed, getter)() for name, getter in getters.items()}

        raise RuntimeError(
            "Analytical KKT gradients require dual variables from "
            "EconomicDispatchQuadraticModel. Add get_mu_max/get_mu_min/"
            "get_mu_up/get_mu_down methods."
        )

    def compute_player_profit(
        self,
        player_id: int,
        dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
        block_dispatches: Optional[List[List[List[float]]]] = None,
    ) -> Tuple[float, List[float]]:
        controlled = self._controlled_blocks(player_id)
        profit_params = self._build_profit_params()
        c_linear = np.asarray(profit_params.c_linear, dtype=np.float64).reshape(-1)
        c_quadratic = np.asarray(profit_params.c_quadratic, dtype=np.float64).reshape(-1)
        dispatch_source = block_dispatches if block_dispatches is not None else dispatches
        first_width = len(dispatch_source[0][0]) if dispatch_source and dispatch_source[0] else 0
        if first_width != self.num_blocks:
            raise ValueError(
                "compute_player_profit requires block-level dispatches with "
                f"{self.num_blocks} entries per time step, got {first_width}"
            )
        scenario_profits = []
        for s in range(self.num_scenarios):
            profit_s = 0.0
            for t in range(self.num_time_steps):
                for block_idx in controlled:
                    p = float(dispatch_source[s][t][block_idx])
                    profit_s += (
                        float(clearing_prices[s][t]) * p
                        - float(c_linear[block_idx]) * p
                        - 0.5 * float(c_quadratic[block_idx]) * p**2
                    )
            scenario_profits.append(float(profit_s))
        return float(np.mean(scenario_profits)), scenario_profits

    def compute_player_bid_gradients(
        self,
        player_id: int,
    ) -> Tuple[Dict[int, np.ndarray], float, List[float], Dict[str, Any]]:
        return self._compute_player_bid_gradients_for_df(player_id, self.scenarios_df)

    def _compute_player_bid_gradients_for_df(
        self,
        player_id: int,
        scenarios_df: pd.DataFrame,
    ) -> Tuple[Dict[int, np.ndarray], float, List[float], Dict[str, Any]]:
        original_scenarios_df = self.scenarios_df
        self.scenarios_df = scenarios_df.copy(deep=True)
        try:
            return self._compute_player_bid_gradients_current_df(player_id)
        finally:
            self.scenarios_df = original_scenarios_df

    def _compute_player_bid_gradients_current_df(
        self,
        player_id: int,
    ) -> Tuple[Dict[int, np.ndarray], float, List[float], Dict[str, Any]]:
        ed = self._solve_training_ed_model()
        dispatches = ed.get_dispatches()
        block_dispatches = ed.get_block_dispatches()
        clearing_prices = ed.get_clearing_prices()
        duals = self._extract_duals_from_ed(ed)
        if dispatches is None or block_dispatches is None or clearing_prices is None:
            raise RuntimeError("ED solve did not return dispatches and clearing prices.")

        baseline_profit, baseline_scenario_profits = self.compute_player_profit(
            player_id,
            dispatches,
            clearing_prices,
            block_dispatches=block_dispatches,
        )

        controlled = self._controlled_blocks(player_id)
        scenario_gradients: Dict[int, np.ndarray] = {}
        scenario_diagnostics = []
        gradient_norms = []
        condition_numbers = []

        for s in range(self.num_scenarios):
            solution_s = self._build_ed_solution_for_scenario(s, ed)
            params_s = self._build_ed_params_for_scenario(s)

            market_sens = compute_market_sensitivities(
                solution=solution_s,
                params=params_s,
                player_generators=controlled,
                include_beta=False,
                regularization=self.kkt_regularization,
                condition_warning_threshold=self.condition_warning_threshold,
            )

            profit_sens = compute_profit_sensitivities(
                P=solution_s.P,
                lambda_=solution_s.lambda_,
                profit_params=self._build_profit_params(),
                player_generators=controlled,
                flatten_dispatch=True,
            )

            self._validate_bid_gradient_shapes(
                market_sens=market_sens,
                profit_sens=profit_sens,
                n_owned=len(controlled),
            )

            dpi_dalpha = (
                np.asarray(profit_sens["dpi_dP"], dtype=np.float64)
                @ np.asarray(market_sens["dP_dalpha"], dtype=np.float64)
                + np.asarray(profit_sens["dpi_dlambda"], dtype=np.float64)
                @ np.asarray(market_sens["dlambda_dalpha"], dtype=np.float64)
            )
            dpi_dalpha = np.asarray(dpi_dalpha, dtype=np.float64).reshape(-1)
            expected = len(controlled) * self.num_time_steps
            if dpi_dalpha.shape != (expected,):
                raise ValueError(f"dpi_dalpha must have shape {(expected,)}, got {dpi_dalpha.shape}")
            if not np.all(np.isfinite(dpi_dalpha)):
                raise ValueError(f"Non-finite bid gradient for player {player_id}, scenario {s}")

            grad_norm = float(np.linalg.norm(dpi_dalpha))
            condition_number = float(market_sens["condition_number"])
            scenario_gradients[int(s)] = dpi_dalpha
            gradient_norms.append(grad_norm)
            condition_numbers.append(condition_number)
            scenario_diagnostics.append({
                "scenario": int(s),
                "profit": float(profit_sens["profit"]),
                "condition_number": condition_number,
                "gradient_norm": grad_norm,
            })

        diagnostics = {
            "scenario_diagnostics": scenario_diagnostics,
            "max_condition_number": float(np.max(condition_numbers)) if condition_numbers else 0.0,
            "mean_condition_number": float(np.mean(condition_numbers)) if condition_numbers else 0.0,
            "max_gradient_norm": float(np.max(gradient_norms)) if gradient_norms else 0.0,
            "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        }
        return scenario_gradients, baseline_profit, baseline_scenario_profits, diagnostics

    def _historical_gradient_snapshots_for_update(self) -> List[pd.DataFrame]:
        snapshots = self.gradient_history_snapshots
        if self.gradient_history_window > 0 and len(snapshots) > self.gradient_history_window:
            snapshots = snapshots[-self.gradient_history_window:]
        if snapshots:
            return snapshots
        return [self.scenarios_df.copy(deep=True)]

    def _compute_player_historical_average_bid_gradients(
        self,
        player_id: int,
    ) -> Tuple[Dict[int, np.ndarray], float, List[float], Dict[str, Any]]:
        snapshots = self._historical_gradient_snapshots_for_update()
        accumulated_gradients: Dict[int, np.ndarray] = {}
        snapshot_diagnostics = []
        condition_numbers = []
        latest_baseline_profit = 0.0
        latest_scenario_profits: List[float] = []

        for snapshot_idx, snapshot_df in enumerate(snapshots):
            gradients, baseline_profit, scenario_profits, diagnostics = (
                self._compute_player_bid_gradients_for_df(player_id, snapshot_df)
            )
            latest_baseline_profit = baseline_profit
            latest_scenario_profits = scenario_profits
            snapshot_diagnostics.append({
                "snapshot": int(snapshot_idx),
                "baseline_profit": float(baseline_profit),
                "max_gradient_norm": float(diagnostics["max_gradient_norm"]),
                "mean_gradient_norm": float(diagnostics["mean_gradient_norm"]),
                "max_condition_number": float(diagnostics["max_condition_number"]),
                "mean_condition_number": float(diagnostics["mean_condition_number"]),
            })
            condition_numbers.append(float(diagnostics["max_condition_number"]))
            condition_numbers.append(float(diagnostics["mean_condition_number"]))
            for scenario_idx, gradient in gradients.items():
                if int(scenario_idx) not in accumulated_gradients:
                    accumulated_gradients[int(scenario_idx)] = np.zeros_like(gradient)
                accumulated_gradients[int(scenario_idx)] += np.asarray(gradient, dtype=np.float64)

        averaged_gradients = {
            int(scenario_idx): gradient / float(len(snapshots))
            for scenario_idx, gradient in accumulated_gradients.items()
        }
        scenario_gradient_norms = {
            int(scenario_idx): float(np.linalg.norm(gradient))
            for scenario_idx, gradient in averaged_gradients.items()
        }
        gradient_norms = list(scenario_gradient_norms.values())

        diagnostics = {
            "scenario_diagnostics": [],
            "scenario_gradient_norms": scenario_gradient_norms,
            "snapshot_diagnostics": snapshot_diagnostics,
            "num_history_snapshots": int(len(snapshots)),
            "max_condition_number": float(np.max(condition_numbers)) if condition_numbers else 0.0,
            "mean_condition_number": float(np.mean(condition_numbers)) if condition_numbers else 0.0,
            "max_gradient_norm": float(np.max(gradient_norms)) if gradient_norms else 0.0,
            "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        }
        return averaged_gradients, latest_baseline_profit, latest_scenario_profits, diagnostics

    def compute_and_update_player_bid_gradients(
        self,
        player_id: int,
    ) -> Tuple[float, List[float], Dict[str, Any]]:
        """
        Compute and apply independent scenario-level bid gradients for one player.

        In ``current`` mode, the update uses only the gradient at the player's
        pre-update bid profile. In ``history_average`` mode, the update uses
        the average gradient over recorded complete-iteration bid profiles.
        """
        if self.gradient_update_mode == "history_average":
            scenario_gradients, baseline_profit, baseline_scenario_profits, diagnostics = (
                self._compute_player_historical_average_bid_gradients(player_id)
            )
        else:
            scenario_gradients, baseline_profit, baseline_scenario_profits, diagnostics = (
                self._compute_player_bid_gradients_current_df(player_id)
            )

        step_norms = []
        scenario_step_norms: Dict[int, float] = {}
        scenario_bid_order_step_diagnostics: Dict[int, Dict[str, Any]] = {}

        for s, gradient in scenario_gradients.items():
            step_norm = self._update_player_scenario_bids(
                player_id=player_id,
                scenario_idx=int(s),
                gradient=gradient,
            )
            step_norms.append(step_norm)
            scenario_step_norms[int(s)] = step_norm
            scenario_bid_order_step_diagnostics[int(s)] = dict(self.last_bid_order_step_diagnostics)

        diagnostics["scenario_step_norms"] = scenario_step_norms
        diagnostics["scenario_bid_order_step_diagnostics"] = scenario_bid_order_step_diagnostics
        diagnostics["max_step_norm"] = float(np.max(step_norms)) if step_norms else 0.0
        diagnostics["mean_step_norm"] = float(np.mean(step_norms)) if step_norms else 0.0
        return baseline_profit, baseline_scenario_profits, diagnostics

    def _validate_bid_gradient_shapes(
        self,
        market_sens: Dict[str, Any],
        profit_sens: Dict[str, Any],
        n_owned: int,
    ) -> None:
        n_alpha = int(n_owned) * self.num_time_steps
        n_dispatch = self.num_blocks * self.num_time_steps
        if np.asarray(market_sens["dP_dalpha"]).shape != (n_dispatch, n_alpha):
            raise ValueError(
                "market_sens['dP_dalpha'] must have shape "
                f"{(n_dispatch, n_alpha)}, got {np.asarray(market_sens['dP_dalpha']).shape}"
            )
        if np.asarray(market_sens["dlambda_dalpha"]).shape != (self.num_time_steps, n_alpha):
            raise ValueError(
                "market_sens['dlambda_dalpha'] must have shape "
                f"{(self.num_time_steps, n_alpha)}, got {np.asarray(market_sens['dlambda_dalpha']).shape}"
            )
        if np.asarray(profit_sens["dpi_dP"]).shape != (n_dispatch,):
            raise ValueError(
                f"profit_sens['dpi_dP'] must have length {n_dispatch}, got {np.asarray(profit_sens['dpi_dP']).shape}"
            )
        if np.asarray(profit_sens["dpi_dlambda"]).shape != (self.num_time_steps,):
            raise ValueError(
                "profit_sens['dpi_dlambda'] must have shape "
                f"{(self.num_time_steps,)}, got {np.asarray(profit_sens['dpi_dlambda']).shape}"
            )

    def update_player_bids(
        self,
        player_id: int,
        scenario_gradients: Dict[int, np.ndarray],
    ) -> None:
        for scenario_idx, gradient in scenario_gradients.items():
            self._update_player_scenario_bids(
                player_id=player_id,
                scenario_idx=int(scenario_idx),
                gradient=gradient,
            )

    def _update_player_scenario_bids(
        self,
        player_id: int,
        scenario_idx: int,
        gradient: np.ndarray,
    ) -> float:
        controlled = self._controlled_blocks(player_id)
        n_owned = len(controlled)
        expected = n_owned * self.num_time_steps
        s = int(scenario_idx)
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if grad.shape != (expected,):
            raise ValueError(f"Gradient for scenario {s} must have shape {(expected,)}, got {grad.shape}")
        if not np.all(np.isfinite(grad)):
            raise ValueError(f"Non-finite gradient for scenario {s}")

        grad = self._clip_owned_bid_gradient(grad, n_owned)

        current = self._flatten_owned_bids_time_major(s, controlled)
        raw_step = self.current_learning_rate * grad
        limited_step = self._limit_bid_step_to_order_region(
            scenario_idx=s,
            controlled=controlled,
            current=current,
            raw_step=raw_step,
        )
        self.last_bid_order_step_diagnostics = self._bid_order_step_diagnostics(
            raw_step=raw_step,
            limited_step=limited_step,
        )
        self.last_bid_order_step_diagnostics["bid_order_epsilon"] = self.bid_order_epsilon
        updated = current + limited_step
        updated = self._clip_alpha_array(updated)
        step_norm = float(np.linalg.norm(updated - current))
        if not np.all(np.isfinite(updated)):
            raise ValueError(f"Updated bids contain non-finite values for scenario {s}")

        updated_matrix = self._unflatten_owned_bids_time_major(updated, n_owned)
        for local_idx, block_idx in enumerate(controlled):
            gen_name = self.block_names[int(block_idx)]
            profile = [float(v) for v in updated_matrix[local_idx, :]]
            if len(profile) != self.num_time_steps:
                raise ValueError(
                    f"Updated {gen_name}_bid_profile must have length "
                    f"{self.num_time_steps}, got {len(profile)}"
                )
            if not np.all(np.isfinite(profile)):
                raise ValueError(f"Updated {gen_name}_bid_profile contains non-finite values")
            self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = profile
            self.scenarios_df.at[s, f"{gen_name}_bid"] = profile[0]

        self._validate_owned_bid_writeback(s, controlled, updated_matrix)
        return step_norm

    def _limit_bid_step_to_order_region(
        self,
        scenario_idx: int,
        controlled: List[int],
        current: np.ndarray,
        raw_step: np.ndarray,
    ) -> np.ndarray:
        """
        Keep one KKT-gradient step inside the current local bid-order cell.

        The implicit KKT sensitivity is a local derivative: it assumes the
        active constraint/order structure remains the same over the step. If a
        strategic bid jumps across the next neighboring bid in the merit order,
        the update can land on the other side of the price setter using the old
        derivative. This limiter shortens only the entries that would cross the
        nearest neighboring bid at the same time step.
        """
        if self.bid_order_epsilon is None and self.bid_order_step_fraction is None:
            return np.asarray(raw_step, dtype=np.float64).reshape(-1).copy()

        current_vec = np.asarray(current, dtype=np.float64).reshape(-1)
        step_vec = np.asarray(raw_step, dtype=np.float64).reshape(-1).copy()
        if step_vec.shape != current_vec.shape:
            raise ValueError(f"raw_step shape {step_vec.shape} does not match current shape {current_vec.shape}")

        controlled_set = {int(block_idx) for block_idx in controlled}
        market_alpha = self._scenario_alpha_matrix(int(scenario_idx))
        epsilon = self.bid_order_epsilon
        fraction = self.bid_order_step_fraction
        tol = 1e-10

        for t in range(self.num_time_steps):
            other_bids = np.asarray(
                [
                    market_alpha[block_idx, t]
                    for block_idx in range(self.num_blocks)
                    if int(block_idx) not in controlled_set
                ],
                dtype=np.float64,
            )
            if other_bids.size == 0:
                continue

            for local_idx, _block_idx in enumerate(controlled):
                flat_idx = t * len(controlled) + local_idx
                step = float(step_vec[flat_idx])
                if abs(step) <= tol:
                    continue

                bid = float(current_vec[flat_idx])

                if step > 0.0:
                    upper_neighbors = other_bids[other_bids > bid + tol]
                    if upper_neighbors.size == 0:
                        continue
                    neighbor = float(np.min(upper_neighbors))
                    max_step = (
                        neighbor - float(epsilon) - bid
                        if epsilon is not None
                        else (neighbor - bid) * float(fraction)
                    )
                    if step > max_step:
                        step_vec[flat_idx] = max(0.0, max_step)
                else:
                    lower_neighbors = other_bids[other_bids < bid - tol]
                    if lower_neighbors.size == 0:
                        continue
                    neighbor = float(np.max(lower_neighbors))
                    min_step = (
                        neighbor + float(epsilon) - bid
                        if epsilon is not None
                        else (neighbor - bid) * float(fraction)
                    )
                    if step < min_step:
                        step_vec[flat_idx] = min(0.0, min_step)

        return step_vec

    @staticmethod
    def _bid_order_step_diagnostics(
        raw_step: np.ndarray,
        limited_step: np.ndarray,
    ) -> Dict[str, Any]:
        raw = np.asarray(raw_step, dtype=np.float64).reshape(-1)
        limited = np.asarray(limited_step, dtype=np.float64).reshape(-1)
        if raw.shape != limited.shape:
            raise ValueError(f"raw_step shape {raw.shape} does not match limited_step shape {limited.shape}")

        raw_norm = float(np.linalg.norm(raw))
        limited_norm = float(np.linalg.norm(limited))
        changed = np.abs(raw - limited) > 1e-12
        return {
            "bid_order_step_scale": (
                float(limited_norm / raw_norm)
                if raw_norm > 0.0
                else 1.0
            ),
            "bid_order_limited_entries": int(np.count_nonzero(changed)),
            "raw_step_norm_before_bid_order_limit": raw_norm,
            "step_norm_after_bid_order_limit": limited_norm,
            "raw_step_max_abs_before_bid_order_limit": (
                float(np.max(np.abs(raw))) if raw.size else 0.0
            ),
            "step_max_abs_after_bid_order_limit": (
                float(np.max(np.abs(limited))) if limited.size else 0.0
            ),
        }

    def _clip_owned_bid_gradient(self, gradient: np.ndarray, n_owned: int) -> np.ndarray:
        """
        Clip bid gradients without penalizing players that own many blocks.

        The default ``per_block`` mode clips each controlled block's time
        trajectory separately, so ``gradient_clip_norm`` has the same meaning
        for every block regardless of how many blocks the player controls.
        ``global`` preserves the old behavior: one norm over all owned
        block-time bid variables.
        """
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if self.gradient_clip_norm is None:
            return grad

        clip_norm = float(self.gradient_clip_norm)
        if self.gradient_clip_mode == "global":
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > clip_norm:
                return grad * (clip_norm / (grad_norm + 1e-12))
            return grad

        grad_matrix = self._unflatten_owned_bids_time_major(grad, n_owned)
        for block_local_idx in range(int(n_owned)):
            block_grad_norm = float(np.linalg.norm(grad_matrix[block_local_idx, :]))
            if block_grad_norm > clip_norm:
                grad_matrix[block_local_idx, :] *= clip_norm / (block_grad_norm + 1e-12)
        return grad_matrix.T.reshape(-1)

    def _flatten_owned_bids_time_major(self, scenario_idx: int, controlled: List[int]) -> np.ndarray:
        alpha = np.zeros((len(controlled), self.num_time_steps), dtype=np.float64)
        for local_idx, block_idx in enumerate(controlled):
            gen_name = self.block_names[int(block_idx)]
            profile = self._as_profile(
                self.scenarios_df.at[int(scenario_idx), f"{gen_name}_bid_profile"],
                f"{gen_name}_bid_profile",
            )
            if len(profile) != self.num_time_steps:
                raise ValueError(
                    f"{gen_name}_bid_profile must have length {self.num_time_steps}, got {len(profile)}"
                )
            alpha[local_idx, :] = profile
        return alpha.T.reshape(-1)

    def _unflatten_owned_bids_time_major(self, vector: np.ndarray, n_owned: int) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        expected = int(n_owned) * self.num_time_steps
        if vec.size != expected:
            raise ValueError(f"Expected vector of length {expected}, got {vec.size}")
        return vec.reshape(self.num_time_steps, int(n_owned)).T

    def _validate_owned_bid_writeback(
        self,
        scenario_idx: int,
        controlled: List[int],
        expected_matrix: np.ndarray,
    ) -> None:
        for local_idx, block_idx in enumerate(controlled):
            gen_name = self.block_names[int(block_idx)]
            profile = self._as_profile(
                self.scenarios_df.at[int(scenario_idx), f"{gen_name}_bid_profile"],
                f"{gen_name}_bid_profile",
            )
            if not np.allclose(profile, expected_matrix[local_idx, :], rtol=0.0, atol=0.0):
                raise RuntimeError(
                    f"Scenario DataFrame bid writeback mismatch for scenario {scenario_idx}, generator {gen_name}"
                )
            if float(self.scenarios_df.at[int(scenario_idx), f"{gen_name}_bid"]) != float(profile[0]):
                raise RuntimeError(
                    f"{gen_name}_bid is inconsistent with the first bid-profile entry in scenario {scenario_idx}"
                )

    def _build_ed_solution_for_scenario(
        self,
        scenario_idx: int,
        ed: EconomicDispatchQuadraticModel,
    ) -> EDSolution:
        data = ed.get_scenario_kkt_data(int(scenario_idx))
        self._validate_scenario_kkt_data(data, int(scenario_idx))
        return EDSolution(
            P=np.asarray(data["P_block"], dtype=np.float64),
            P_phys=np.asarray(data["P_phys"], dtype=np.float64),
            lambda_=np.asarray(data["lambda_"], dtype=np.float64),
            mu_max=np.asarray(data["mu_max"], dtype=np.float64),
            mu_min=np.asarray(data["mu_min"], dtype=np.float64),
            mu_up=np.asarray(data["mu_up"], dtype=np.float64),
            mu_down=np.asarray(data["mu_down"], dtype=np.float64),
        )

    def _build_ed_params_for_scenario(self, scenario_idx: int) -> EDParameters:
        s = int(scenario_idx)
        return EDParameters(
            alpha=self._scenario_alpha_matrix(s),
            beta=np.full(
                (self.num_blocks, self.num_time_steps),
                2.0 * self.beta_smooth,
                dtype=np.float64,
            ),
            demand=np.asarray(
                self._as_profile(self.scenarios_df.at[s, "demand_profile"], "demand_profile"),
                dtype=np.float64,
            ),
            pmax=self._scenario_pmax_matrix(s),
            pmin=self._scenario_pmin_matrix(s),
            ramp_up=np.asarray(
                [float(self.ramps_df[f"{gen}_ramp_up"].iloc[0]) for gen in self.physical_generator_names],
                dtype=np.float64,
            ),
            ramp_down=np.asarray(
                [float(self.ramps_df[f"{gen}_ramp_down"].iloc[0]) for gen in self.physical_generator_names],
                dtype=np.float64,
            ),
            p_initial=np.asarray(self.P_init[s], dtype=np.float64),
            physical_to_block_indices=[list(blocks) for blocks in self.physical_to_block_indices],
            block_to_physical_idx=list(self.block_to_physical_idx),
            num_physical_generators=self.num_physical_generators,
            num_blocks=self.num_blocks,
        )

    def _scenario_alpha_matrix(self, scenario_idx: int) -> np.ndarray:
        alpha = np.zeros((self.num_blocks, self.num_time_steps), dtype=np.float64)
        for i, gen_name in enumerate(self.block_names):
            profile = self._as_profile(
                self.scenarios_df.at[scenario_idx, f"{gen_name}_bid_profile"],
                f"{gen_name}_bid_profile",
            )
            if len(profile) != self.num_time_steps:
                raise ValueError(
                    f"{gen_name}_bid_profile must have length {self.num_time_steps}, got {len(profile)}"
                )
            alpha[i, :] = profile
        return alpha

    def _scenario_pmax_matrix(self, scenario_idx: int) -> np.ndarray:
        pmax = np.zeros((self.num_blocks, self.num_time_steps), dtype=np.float64)
        for i, gen_name in enumerate(self.block_names):
            profile_col = f"{gen_name}_cap_profile"
            availability_profile_col = f"{gen_name}_profile"
            if profile_col in self.scenarios_df.columns:
                profile = self._as_profile(self.scenarios_df.at[scenario_idx, profile_col], profile_col)
                pmax[i, :] = profile
            elif availability_profile_col in self.scenarios_df.columns:
                profile = self._as_profile(
                    self.scenarios_df.at[scenario_idx, availability_profile_col],
                    availability_profile_col,
                )
                pmax[i, :] = profile
            else:
                pmax[i, :] = float(self.scenarios_df.at[scenario_idx, f"{gen_name}_cap"])
        return pmax

    def _scenario_pmin_matrix(self, scenario_idx: int) -> np.ndarray:
        pmin = np.zeros((self.num_blocks, self.num_time_steps), dtype=np.float64)
        for i, block_name in enumerate(self.block_names):
            profile_col = f"{block_name}_pmin_profile"
            static_col = f"{block_name}_pmin"
            if profile_col in self.scenarios_df.columns:
                profile = self._as_profile(self.scenarios_df.at[scenario_idx, profile_col], profile_col)
                pmin[i, :] = profile
            elif static_col in self.scenarios_df.columns:
                pmin[i, :] = float(self.scenarios_df.at[scenario_idx, static_col])
        return pmin

    def _build_profit_params(self) -> ProfitParameters:
        c_linear = np.asarray(
            [float(self.costs_df[f"{block}_cost"].iloc[0]) for block in self.block_names],
            dtype=np.float64,
        )
        c_quadratic = np.zeros(self.num_blocks, dtype=np.float64)
        for i, block in enumerate(self.block_names):
            for col in (f"{block}_quadratic_cost", f"{block}_cost_quadratic", f"{block}_quad_cost"):
                if col in self.costs_df.columns:
                    c_quadratic[i] = float(self.costs_df[col].iloc[0])
                    break
        return ProfitParameters(c_linear=c_linear, c_quadratic=c_quadratic)

    def _validate_scenario_kkt_data(self, data: Dict[str, np.ndarray], scenario_idx: int) -> None:
        expected_block_shape = (self.num_blocks, self.num_time_steps)
        expected_phys_shape = (self.num_physical_generators, self.num_time_steps)
        for key in ("P_block", "mu_max", "mu_min"):
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.shape != expected_block_shape:
                raise ValueError(
                    f"Scenario {scenario_idx}: {key} must have shape "
                    f"{expected_block_shape}, got {arr.shape}"
                )
        for key in ("P_phys", "mu_up", "mu_down"):
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.shape != expected_phys_shape:
                raise ValueError(
                    f"Scenario {scenario_idx}: {key} must have shape "
                    f"{expected_phys_shape}, got {arr.shape}"
                )
        p_block = np.asarray(data["P_block"], dtype=np.float64)
        p_phys = np.asarray(data["P_phys"], dtype=np.float64)
        for physical_idx, block_indices in enumerate(self.physical_to_block_indices):
            if not np.allclose(np.sum(p_block[block_indices, :], axis=0), p_phys[physical_idx, :]):
                physical_name = self.physical_generator_names[physical_idx]
                raise ValueError(
                    f"Scenario {scenario_idx}: block dispatches do not sum to physical dispatch for {physical_name}"
                )

    def run(self) -> Dict[str, Any]:
        print("=== Starting KKT Analytical Direct Bid Gradient Training ===")
        print(f"beta_smooth       : {self.beta_smooth}")
        print(f"learning_rate     : {self.learning_rate}")
        print(f"lr_decay          : {self.learning_rate_decay}")
        print(f"min_learning_rate : {self.min_learning_rate}")
        print(f"max_iterations    : {self.max_iterations}")
        print(f"gradient_clip_norm: {self.gradient_clip_norm}")
        print(f"gradient_clip_mode: {self.gradient_clip_mode}")
        print(f"gradient_update_mode: {self.gradient_update_mode}")
        print(f"gradient_history_window: {self.gradient_history_window}")
        print(f"bid_order_epsilon: {self.bid_order_epsilon}")
        print(f"players           : {[p['id'] for p in self.players_config]}")
        print(f"physical generators: {self.physical_generator_names}")
        print(f"bidding blocks    : {self.block_names}")

        self._record_iteration_state()
        self._record_gradient_history_snapshot()

        for iteration in range(1, self.max_iterations + 1):
            self.current_learning_rate = self._learning_rate_for_iteration(iteration)
            self.learning_rate_history.append(self.current_learning_rate)
            print(f"\n--- Iteration {iteration} ---")
            print(f"  active learning rate: {self.current_learning_rate:.6g}")
            iteration_gradient_norms: Dict[int, Dict[int, float]] = {}
            iteration_step_norms: Dict[int, Dict[int, float]] = {}
            iteration_diagnostics: Dict[int, Dict[str, Any]] = {}
            iteration_conditions: Dict[int, Dict[str, float]] = {}

            for player in self.players_config:
                player_id = int(player["id"])
                print(f"  Player {player_id}")

                baseline_profit, _, diagnostics = self.compute_and_update_player_bid_gradients(player_id)
                iteration_gradient_norms[player_id] = {
                    int(s): float(norm)
                    for s, norm in diagnostics["scenario_gradient_norms"].items()
                }
                iteration_step_norms[player_id] = {
                    int(s): float(norm)
                    for s, norm in diagnostics["scenario_step_norms"].items()
                }
                iteration_diagnostics[player_id] = diagnostics
                iteration_conditions[player_id] = {
                    "max_condition_number": float(diagnostics["max_condition_number"]),
                    "mean_condition_number": float(diagnostics["mean_condition_number"]),
                }

                alpha_summary = self._player_alpha_summary(player_id)

                print(f"    pre-update profit          : {baseline_profit:.6f}")
                print(f"    max scenario gradient norm : {diagnostics['max_gradient_norm']:.6f}")
                print(f"    mean scenario gradient norm: {diagnostics['mean_gradient_norm']:.6f}")
                print(f"    max scenario step norm     : {diagnostics['max_step_norm']:.6f}")
                print(f"    mean scenario step norm    : {diagnostics['mean_step_norm']:.6f}")
                print(f"    max condition number       : {diagnostics['max_condition_number']:.3e}")
                print(f"    mean condition number      : {diagnostics['mean_condition_number']:.3e}")
                if self.gradient_update_mode == "history_average":
                    print(f"    history snapshots used     : {diagnostics['num_history_snapshots']}")
                print(f"    post-update alpha min/max  : {alpha_summary['min']:.6f} / {alpha_summary['max']:.6f}")

            self.gradient_norm_history.append(iteration_gradient_norms)
            self.step_norm_history.append(iteration_step_norms)
            self.gradient_diagnostics_history.append(iteration_diagnostics)
            self.kkt_condition_history.append(iteration_conditions)
            self._record_iteration_state()
            self._record_gradient_history_snapshot()

            max_gradient_norm = max(
                (
                    norm
                    for scenario_norms in iteration_gradient_norms.values()
                    for norm in scenario_norms.values()
                ),
                default=0.0,
            )
            max_step_norm = max(
                (
                    norm
                    for scenario_norms in iteration_step_norms.values()
                    for norm in scenario_norms.values()
                ),
                default=0.0,
            )
            cur_profit = self.profit_history_training[-1]
            profit_str = ", ".join(
                f"P{player['id']}={cur_profit[idx]:.3f}"
                for idx, player in enumerate(self.players_config)
            )
            print(f"  Training profits: {profit_str}")
            print(f"  Max gradient norm: {max_gradient_norm:.6f}")
            print(f"  Max step norm: {max_step_norm:.6f}")
            market_alpha_summary = self._market_alpha_summary()
            print(
                "  Market alpha trajectory min/max: "
                f"{market_alpha_summary['min']:.6f} / {market_alpha_summary['max']:.6f}"
            )

            self.iteration = iteration
            if max_gradient_norm <= self.conv_tolerance:
                print("  Convergence achieved.")
                self.results = self.get_results()
                return self.results

        print("\nMaximum iterations reached without convergence.")
        self.results = self.get_results()
        return self.results

    def _record_iteration_state(self) -> None:
        ed = self._solve_training_ed_model()
        dispatches = ed.get_dispatches()
        block_dispatches = ed.get_block_dispatches()
        prices = ed.get_clearing_prices()
        if dispatches is None or block_dispatches is None or prices is None:
            raise RuntimeError("Quadratic ED solve did not return complete state.")
        player_profits, scenario_player_profits = self._compute_all_player_profits(
            dispatches, prices, block_dispatches=block_dispatches
        )
        self.bid_history.append(self._snapshot_all_bids())
        self.profit_history_training.append(player_profits)
        self.profit_history_training_scenario.append(scenario_player_profits)
        self.dispatch_history.append(dispatches)
        self.clearing_price_history.append(prices)

    def _record_gradient_history_snapshot(self) -> None:
        self.gradient_history_snapshots.append(self.scenarios_df.copy(deep=True))

    def _compute_all_player_profits(
        self,
        dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
        block_dispatches: Optional[List[List[List[float]]]] = None,
    ) -> Tuple[List[float], List[List[float]]]:
        player_profits = []
        scenario_by_player = [[0.0 for _ in self.players_config] for _ in range(self.num_scenarios)]
        for player_idx, player in enumerate(self.players_config):
            avg_profit, scenario_profits = self.compute_player_profit(
                int(player["id"]),
                dispatches,
                clearing_prices,
                block_dispatches=block_dispatches,
            )
            player_profits.append(avg_profit)
            for s, profit_s in enumerate(scenario_profits):
                scenario_by_player[s][player_idx] = profit_s
        return player_profits, scenario_by_player

    def get_results(self) -> Dict[str, Any]:
        ed = self._solve_training_ed_model()
        dispatches = ed.get_dispatches()
        block_dispatches = ed.get_block_dispatches()
        prices = ed.get_clearing_prices()
        if dispatches is None or block_dispatches is None or prices is None:
            raise RuntimeError("Quadratic ED solve did not return complete results.")
        player_profits, scenario_player_profits = self._compute_all_player_profits(
            dispatches, prices, block_dispatches=block_dispatches
        )
        return {
            "iterations": self.iteration,
            "num_scenarios": self.num_scenarios,
            "num_time_steps": self.num_time_steps,
            "generator_names": self.generator_names,
            "physical_generator_names": self.physical_generator_names,
            "block_names": self.block_names,
            "block_to_physical": self.block_to_physical,
            "block_to_physical_idx": self.block_to_physical_idx,
            "blocks_by_generator": self.blocks_by_generator,
            "generator_costs": self.cost_vector.tolist(),
            "block_costs": self.cost_vector.tolist(),
            "beta_smooth": self.beta_smooth,
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "min_learning_rate": self.min_learning_rate,
            "learning_rate_schedule": "inverse_time",
            "learning_rate_history": self.learning_rate_history,
            "max_iterations": self.max_iterations,
            "conv_tolerance": self.conv_tolerance,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_clip_mode": self.gradient_clip_mode,
            "gradient_update_mode": self.gradient_update_mode,
            "gradient_history_window": self.gradient_history_window,
            "bid_order_step_fraction": self.bid_order_step_fraction,
            "bid_order_epsilon": self.bid_order_epsilon,
            "num_gradient_history_snapshots": len(self.gradient_history_snapshots),
            "kkt_regularization": self.kkt_regularization,
            "condition_warning_threshold": self.condition_warning_threshold,
            "alpha_bounds": {
                "min": self.alpha_min,
                "max": self.alpha_max,
            },
            "final_alpha_summary": self._market_alpha_summary(),
            "final_alpha_summary_by_player": {
                int(player["id"]): self._player_alpha_summary(int(player["id"]))
                for player in self.players_config
            },
            "bid_history": self.bid_history,
            "profit_history_training": self.profit_history_training,
            "profit_history_training_scenario": self.profit_history_training_scenario,
            "dispatch_history": self.dispatch_history,
            "clearing_price_history": self.clearing_price_history,
            "gradient_norm_history": self.gradient_norm_history,
            "step_norm_history": self.step_norm_history,
            "gradient_diagnostics_history": self.gradient_diagnostics_history,
            "kkt_condition_history": self.kkt_condition_history,
            "final_dispatches": dispatches,
            "final_block_dispatches": block_dispatches,
            "final_clearing_prices": prices,
            "final_player_profits": player_profits,
            "final_player_profits_scenario": scenario_player_profits,
        }

    def save_results(self, output_path: str) -> Path:
        results = self.results or self.get_results()
        path = Path(output_path)
        if path.suffix and path.suffix.lower() != ".json":
            raise ValueError("output_path must end with .json or have no extension")
        if not path.suffix:
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2, default=self._json_default_serializer)
        return path

    def _snapshot_all_bids(self) -> List[List[List[float]]]:
        return [
            [
                list(self.scenarios_df.at[s, f"{block_name}_bid_profile"])
                for block_name in self.block_names
            ]
            for s in range(self.num_scenarios)
        ]

    def _player_alpha_summary(self, player_id: int) -> Dict[str, float]:
        return self._alpha_summary_for_blocks(self._controlled_blocks(player_id))

    def _market_alpha_summary(self) -> Dict[str, float]:
        return self._alpha_summary_for_blocks(list(range(self.num_blocks)))

    def _alpha_summary_for_blocks(self, block_indices: List[int]) -> Dict[str, float]:
        values = []
        deltas = []
        for s in range(self.num_scenarios):
            for block_idx in block_indices:
                gen_name = self.block_names[int(block_idx)]
                cost = float(self.cost_vector[int(block_idx)])
                profile = self._as_profile(
                    self.scenarios_df.at[s, f"{gen_name}_bid_profile"],
                    f"{gen_name}_bid_profile",
                )
                values.extend(profile)
                deltas.extend(v - cost for v in profile)
        alpha_values = np.asarray(values, dtype=np.float64)
        alpha_deltas = np.asarray(deltas, dtype=np.float64)

        return {
            "min": float(np.min(alpha_values)),
            "max": float(np.max(alpha_values)),
            "mean_abs_delta_from_cost": float(np.mean(np.abs(alpha_deltas))),
            "max_abs_delta_from_cost": float(np.max(np.abs(alpha_deltas))),
        }

    def _get_player_config(self, player_id: int) -> Dict[str, Any]:
        for player in self.players_config:
            if int(player["id"]) == int(player_id):
                return player
        raise KeyError(f"Unknown player_id: {player_id}")

    @staticmethod
    def _json_default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(key): value for key, value in obj.items()}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    import time

    from config.scenarios.scenario_generator import ScenarioManager

    TEST_CASE = "test_case_bidding_blocks"

    scenario_manager = ScenarioManager(TEST_CASE)
    players_config = scenario_manager.get_players_config()
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set="policy_training",
        seed=1,
    )

    print(scenarios["description_text"])

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    algo = GradientBidTrainingKKTMS(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        beta_smooth=0.01,
        learning_rate=0.25,
        learning_rate_decay=0.1,
        min_learning_rate=0.01,
        max_iterations=200,
        conv_tolerance=1e-4,
        gradient_clip_norm=100.0,
        gradient_update_mode="history_average",
        gradient_history_window=0,
        kkt_regularization=1e-10,
        alpha_min=0.0,
        alpha_max=None,
    )

    start = time.perf_counter()
    results = algo.run()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")

    saved_path = algo.save_results("results/gradient_bid_training_results.json")
    print(saved_path)
