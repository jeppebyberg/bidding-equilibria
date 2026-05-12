from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.scenarios.scenario_generator import ScenarioManager
from xXgraveyard.models.gradient_based.gradient_bid_training import GradientBidTrainingKKTMS


class ScenarioExpandedGradientBidTraining:
    """
    Scenario-wise gradient bid training with iteration-expanded histories.

    Each base scenario keeps its own DataFrame history. The history starts with
    a true-cost reference row and an update-ready row. During one iteration, the
    active update-ready row is updated sequentially by players; after all players
    have moved, a new update-ready row is appended unless the scenario converged.
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
        gradient_stagnation_iterations: int = 3,
        gradient_stagnation_tolerance: Optional[float] = None,
        max_bid_step: Optional[float] = None,
        gradient_clip_norm: Optional[float] = None,
        gradient_clip_mode: str = "per_block",
        gradient_history_window: int = 0,
        bid_order_step_fraction: Optional[float] = 0.95,
        bid_order_epsilon: Optional[float] = 1e-6,
        skip_player_ids: Optional[List[int]] = None,
        alpha_min: Optional[float] = 0.0,
        alpha_max: Optional[float] = None,
        kkt_regularization: float = 1e-8,
        condition_warning_threshold: float = 1e10,
    ) -> None:
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.skip_player_ids = {
            int(player_id)
            for player_id in (skip_player_ids or [])
        }
        self.beta_smooth = float(beta_smooth)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.min_learning_rate = float(min_learning_rate)
        self.max_iterations = int(max_iterations)
        self.conv_tolerance = float(conv_tolerance)
        if gradient_stagnation_iterations < 0:
            raise ValueError("gradient_stagnation_iterations must be nonnegative")
        self.gradient_stagnation_iterations = int(gradient_stagnation_iterations)
        self.gradient_stagnation_tolerance = (
            float(conv_tolerance)
            if gradient_stagnation_tolerance is None
            else float(gradient_stagnation_tolerance)
        )
        if max_bid_step is not None and max_bid_step <= 0:
            raise ValueError("max_bid_step must be positive when provided")
        self.max_bid_step = None if max_bid_step is None else float(max_bid_step)
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_mode = gradient_clip_mode
        self.gradient_history_window = int(gradient_history_window)
        if bid_order_step_fraction is not None and not (0.0 < bid_order_step_fraction <= 1.0):
            raise ValueError("bid_order_step_fraction must be in (0, 1] when provided")
        self.bid_order_step_fraction = (
            None if bid_order_step_fraction is None else float(bid_order_step_fraction)
        )
        if bid_order_epsilon is not None and bid_order_epsilon < 0.0:
            raise ValueError("bid_order_epsilon must be nonnegative when provided")
        self.bid_order_epsilon = None if bid_order_epsilon is None else float(bid_order_epsilon)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.kkt_regularization = float(kkt_regularization)
        self.condition_warning_threshold = float(condition_warning_threshold)

        self.template = self._make_trainer(
            scenarios_df.reset_index(drop=True),
            max_iterations=0,
        )
        self.block_names = list(self.template.block_names)
        self.cost_vector = np.asarray(self.template.cost_vector, dtype=np.float64)
        self.num_base_scenarios = len(scenarios_df)

        prepared = self._prepare_initial_scenarios(self.template.scenarios_df)
        self.scenario_histories: Dict[int, pd.DataFrame] = {
            int(base_idx): self._initial_history_for_scenario(int(base_idx), prepared.iloc[[base_idx]])
            for base_idx in range(self.num_base_scenarios)
        }
        self.active_scenarios = set(range(self.num_base_scenarios))
        self.converged_scenarios: Dict[int, int] = {}
        self.inactive_players_by_scenario: Dict[int, set[int]] = {
            int(base_idx): set()
            for base_idx in range(self.num_base_scenarios)
        }
        self.final_scenario_rows: Dict[int, Dict[str, Any]] = {}

        self.iteration_history: List[Dict[str, Any]] = []
        self.results: Optional[Dict[str, Any]] = None

    def _make_trainer(
        self,
        scenarios_df: pd.DataFrame,
        max_iterations: Optional[int] = None,
    ) -> GradientBidTrainingKKTMS:
        return GradientBidTrainingKKTMS(
            scenarios_df=scenarios_df.reset_index(drop=True),
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
            players_config=self.players_config,
            beta_smooth=self.beta_smooth,
            learning_rate=self.learning_rate,
            learning_rate_decay=self.learning_rate_decay,
            min_learning_rate=self.min_learning_rate,
            max_iterations=self.max_iterations if max_iterations is None else max_iterations,
            conv_tolerance=self.conv_tolerance,
            gradient_clip_norm=self.gradient_clip_norm,
            gradient_clip_mode=self.gradient_clip_mode,
            gradient_update_mode="current",
            gradient_history_window=0,
            bid_order_step_fraction=self.bid_order_step_fraction,
            bid_order_epsilon=self.bid_order_epsilon,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            kkt_regularization=self.kkt_regularization,
            condition_warning_threshold=self.condition_warning_threshold,
        )

    def _prepare_initial_scenarios(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        prepared = scenarios_df.copy(deep=True).reset_index(drop=True)
        skipped_blocks = {
            int(block_idx)
            for player in self.players_config
            if int(player["id"]) in self.skip_player_ids
            for block_idx in self.template._controlled_blocks(int(player["id"]))
        }
        max_cost_by_physical: Dict[str, float] = {}
        for block_idx, block_name in enumerate(self.block_names):
            physical_name = self.template.block_to_physical.get(block_name, block_name)
            cost = float(self.cost_vector[block_idx])
            max_cost_by_physical[physical_name] = max(
                cost,
                max_cost_by_physical.get(physical_name, cost),
            )

        for block_idx, block_name in enumerate(self.block_names):
            cost = float(self.cost_vector[block_idx])
            physical_name = self.template.block_to_physical.get(block_name, block_name)
            initial_bid = cost if block_idx in skipped_blocks else float(max_cost_by_physical[physical_name])
            true_cost_col = f"{block_name}_true_cost"
            true_cost_profile_col = f"{block_name}_true_cost_profile"
            bid_col = f"{block_name}_bid"
            bid_profile_col = f"{block_name}_bid_profile"

            prepared[true_cost_col] = cost
            prepared[true_cost_profile_col] = [
                [cost] * self.template.num_time_steps
                for _ in range(len(prepared))
            ]
            prepared[bid_col] = initial_bid
            prepared[bid_profile_col] = [
                [initial_bid] * self.template.num_time_steps
                for _ in range(len(prepared))
            ]
        return prepared

    def _initial_history_for_scenario(
        self,
        base_scenario_idx: int,
        scenario_df: pd.DataFrame,
    ) -> pd.DataFrame:
        true_cost_row = scenario_df.iloc[0].copy(deep=True)
        update_row = scenario_df.iloc[0].copy(deep=True)

        true_cost_row["base_scenario_idx"] = int(base_scenario_idx)
        true_cost_row["training_iteration"] = -1
        true_cost_row["history_role"] = "true_cost"
        true_cost_row["is_update_ready"] = False
        true_cost_row["converged"] = False

        update_row["base_scenario_idx"] = int(base_scenario_idx)
        update_row["training_iteration"] = 0
        update_row["history_role"] = "update_ready"
        update_row["is_update_ready"] = True
        update_row["converged"] = False

        return pd.DataFrame([true_cost_row, update_row]).reset_index(drop=True)

    def _learning_rate_for_iteration(self, iteration: int) -> float:
        k = max(1, int(iteration))
        decayed = self.learning_rate / (1.0 + self.learning_rate_decay * float(k - 1))
        return float(max(self.min_learning_rate, decayed))

    def _active_row_index(self, history_df: pd.DataFrame) -> int:
        update_rows = history_df.index[history_df["is_update_ready"].astype(bool)].tolist()
        if not update_rows:
            raise RuntimeError("Scenario history has no update-ready row")
        return int(update_rows[-1])

    def _gradient_history_df(self, history_df: pd.DataFrame) -> pd.DataFrame:
        update_rows = history_df[history_df["history_role"] == "update_ready"]
        if self.gradient_history_window > 0:
            update_rows = update_rows.tail(self.gradient_history_window)
        if update_rows.empty:
            raise RuntimeError("Scenario history has no update_ready rows for gradient calculation")
        return update_rows.reset_index(drop=True)

    def _average_player_gradient(
        self,
        history_df: pd.DataFrame,
        player_id: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        trainer = self._make_trainer(self._gradient_history_df(history_df), max_iterations=0)
        gradients, baseline_profit, scenario_profits, diagnostics = trainer.compute_player_bid_gradients(player_id)
        if not gradients:
            raise RuntimeError(f"No gradients returned for player {player_id}")

        gradient_stack = np.vstack([
            np.asarray(gradient, dtype=np.float64).reshape(1, -1)
            for _, gradient in sorted(gradients.items())
        ])
        average_gradient = np.mean(gradient_stack, axis=0)
        row_gradient_norms = {
            int(row_idx): float(np.linalg.norm(gradient))
            for row_idx, gradient in gradients.items()
        }
        diagnostics = dict(diagnostics)
        diagnostics["baseline_profit"] = float(baseline_profit)
        diagnostics["scenario_profits"] = [float(v) for v in scenario_profits]
        diagnostics["row_gradient_norms"] = row_gradient_norms
        diagnostics["history_rows_used"] = int(len(gradient_stack))
        diagnostics["average_gradient_norm"] = float(np.linalg.norm(average_gradient))
        controlled = trainer._controlled_blocks(player_id)
        latest_row_idx = max(int(row_idx) for row_idx in gradients)
        diagnostics["average_gradient_by_time"] = self._signed_time_profile(
            trainer=trainer,
            vector=average_gradient,
            n_owned=len(controlled),
        )
        diagnostics["current_gradient_by_time"] = self._signed_time_profile(
            trainer=trainer,
            vector=np.asarray(gradients[latest_row_idx], dtype=np.float64),
            n_owned=len(controlled),
        )
        return average_gradient, diagnostics

    @staticmethod
    def _signed_time_profile(
        trainer: GradientBidTrainingKKTMS,
        vector: np.ndarray,
        n_owned: int,
    ) -> List[float]:
        matrix = trainer._unflatten_owned_bids_time_major(vector, n_owned)
        return [
            float(np.sign(np.sum(matrix[:, t])) * np.linalg.norm(matrix[:, t]))
            for t in range(trainer.num_time_steps)
        ]

    def _scale_gradient_to_max_bid_step(
        self,
        gradient: np.ndarray,
        learning_rate: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        raw_step = float(learning_rate) * grad
        raw_step_norm = float(np.linalg.norm(raw_step))
        raw_step_max_abs = float(np.max(np.abs(raw_step))) if raw_step.size else 0.0
        if self.max_bid_step is None or raw_step_max_abs <= self.max_bid_step:
            return grad, {
                "raw_step_norm_before_max_bid_step": raw_step_norm,
                "raw_step_max_abs_before_max_bid_step": raw_step_max_abs,
                "max_bid_step_scale": 1.0,
            }
        scale = float(self.max_bid_step) / (raw_step_max_abs + 1e-12)
        return grad * scale, {
            "raw_step_norm_before_max_bid_step": raw_step_norm,
            "raw_step_max_abs_before_max_bid_step": raw_step_max_abs,
            "max_bid_step_scale": scale,
        }

    def _update_active_row(
        self,
        history_df: pd.DataFrame,
        player_id: int,
        gradient: np.ndarray,
        learning_rate: float,
    ) -> Tuple[pd.DataFrame, float, List[float], Dict[str, float]]:
        active_idx = self._active_row_index(history_df)
        active_row = history_df.iloc[[active_idx]].copy(deep=True).reset_index(drop=True)
        trainer = self._make_trainer(active_row, max_iterations=0)
        controlled = trainer._controlled_blocks(player_id)
        current = trainer._flatten_owned_bids_time_major(0, controlled)
        trainer.current_learning_rate = float(learning_rate)
        scaled_gradient, step_cap_diagnostics = self._scale_gradient_to_max_bid_step(
            gradient,
            learning_rate,
        )
        step_norm = trainer._update_player_scenario_bids(
            player_id=player_id,
            scenario_idx=0,
            gradient=scaled_gradient,
        )
        bid_order_diagnostics = dict(trainer.last_bid_order_step_diagnostics)
        updated = trainer._flatten_owned_bids_time_major(0, controlled)
        delta_matrix = trainer._unflatten_owned_bids_time_major(
            updated - current,
            len(controlled),
        )
        step_by_time = [
            float(
                np.sign(np.sum(delta_matrix[:, t]))
                * np.linalg.norm(delta_matrix[:, t])
            )
            for t in range(trainer.num_time_steps)
        ]

        updated_history = history_df.copy(deep=True)
        updated_row = trainer.scenarios_df.iloc[0].copy(deep=True)
        for column in updated_history.columns:
            if column in updated_row.index:
                updated_history.at[active_idx, column] = updated_row[column]
        return (
            updated_history,
            float(step_norm),
            step_by_time,
            {**step_cap_diagnostics, **bid_order_diagnostics},
        )

    @staticmethod
    def _format_profile(values: List[float], precision: int = 4) -> str:
        threshold = 0.5 * 10.0 ** (-int(precision))
        cleaned = [
            0.0 if abs(float(value)) < threshold else float(value)
            for value in values
        ]
        return "[" + ", ".join(f"{value:.{precision}f}" for value in cleaned) + "]"

    def _recent_player_step_norms(self, base_scenario_idx: int, player_id: int) -> List[float]:
        norms = []
        for record in self.iteration_history:
            scenario_record = record["scenarios"].get(int(base_scenario_idx))
            if scenario_record is None:
                scenario_record = record["scenarios"].get(str(int(base_scenario_idx)))
            if scenario_record is None:
                continue
            players = scenario_record.get("players", {})
            player_record = players.get(int(player_id))
            if player_record is None:
                player_record = players.get(str(int(player_id)))
            if player_record is not None and "step_norm" in player_record:
                norms.append(float(player_record["step_norm"]))
        return norms

    def _player_has_stopped_updating(
        self,
        base_scenario_idx: int,
        player_id: int,
    ) -> Tuple[bool, List[float]]:
        window = self.gradient_stagnation_iterations
        if window <= 0:
            return False, []
        recent = self._recent_player_step_norms(base_scenario_idx, player_id)[-window:]
        if len(recent) < window:
            return False, recent
        return max(recent) <= self.gradient_stagnation_tolerance, recent

    def _resolve_block_idx(self, block: Any) -> int:
        if isinstance(block, str) and not block.strip().lstrip("-").isdigit():
            if block not in self.block_names:
                raise ValueError(f"Unknown block '{block}'. Available: {self.block_names}")
            return self.block_names.index(block)
        idx = int(block)
        if idx < 0 or idx >= len(self.block_names):
            raise ValueError(f"Block index {idx} is out of range [0, {len(self.block_names) - 1}]")
        return idx

    def _perturb_bid(
        self,
        scenarios_df: pd.DataFrame,
        block_idx: int,
        time_step: int,
        delta: float,
    ) -> pd.DataFrame:
        perturbed = scenarios_df.copy(deep=True).reset_index(drop=True)
        block_name = self.block_names[int(block_idx)]
        profile_col = f"{block_name}_bid_profile"
        profile = self.template._as_profile(perturbed.at[0, profile_col], profile_col)
        profile[int(time_step)] = float(profile[int(time_step)] + delta)
        perturbed.at[0, profile_col] = profile
        perturbed.at[0, f"{block_name}_bid"] = float(profile[0])
        return perturbed

    def _append_next_update_ready_row(
        self,
        history_df: pd.DataFrame,
        next_iteration: int,
    ) -> pd.DataFrame:
        active_idx = self._active_row_index(history_df)
        updated_history = history_df.copy(deep=True)
        updated_history.at[active_idx, "is_update_ready"] = False

        next_row = updated_history.iloc[active_idx].copy(deep=True)
        next_row["training_iteration"] = int(next_iteration)
        next_row["history_role"] = "update_ready"
        next_row["is_update_ready"] = True
        next_row["converged"] = False

        return pd.concat([updated_history, pd.DataFrame([next_row])], ignore_index=True)

    def run(self) -> Dict[str, Any]:
        print("=== Starting Scenario-Expanded Gradient Bid Training ===")
        print(f"base scenarios         : {self.num_base_scenarios}")
        print(f"max_iterations         : {self.max_iterations}")
        print(f"learning_rate          : {self.learning_rate}")
        print(f"learning_rate_decay    : {self.learning_rate_decay}")
        print(f"max_bid_step           : {self.max_bid_step}")
        print(f"gradient_clip_norm     : {self.gradient_clip_norm}")
        print(f"gradient_history_window: {self.gradient_history_window}")
        print(f"bid_order_epsilon      : {self.bid_order_epsilon}")
        print(f"skip_player_ids        : {sorted(self.skip_player_ids)}")
        print(f"gradient_stagnation_iterations: {self.gradient_stagnation_iterations}")
        print(f"gradient_stagnation_tolerance : {self.gradient_stagnation_tolerance}")

        for iteration in range(1, self.max_iterations + 1):
            if not self.active_scenarios:
                print("\nAll scenarios converged.")
                break

            learning_rate = self._learning_rate_for_iteration(iteration)
            print(f"\n--- Iteration {iteration} | active scenarios: {len(self.active_scenarios)} ---")
            print(f"  active learning rate: {learning_rate:.6g}")
            iteration_record: Dict[str, Any] = {
                "iteration": int(iteration),
                "learning_rate": float(learning_rate),
                "scenarios": {},
            }

            for base_idx in sorted(self.active_scenarios):
                history_df = self.scenario_histories[base_idx]
                player_records = {}
                max_step_norm = 0.0
                max_gradient_norm = 0.0

                print(f"  Scenario {base_idx}")
                for player in self.players_config:
                    player_id = int(player["id"])
                    if player_id in self.skip_player_ids:
                        self.inactive_players_by_scenario[base_idx].add(player_id)
                        player_records[player_id] = {
                            "active": False,
                            "convergence_reason": "configured_skip",
                            "recent_step_norms": [],
                            "history_rows_used": 0,
                            "average_gradient_norm": 0.0,
                            "max_row_gradient_norm": 0.0,
                            "mean_row_gradient_norm": 0.0,
                            "max_condition_number": 0.0,
                            "mean_condition_number": 0.0,
                            "step_norm": 0.0,
                            "step_by_time": [0.0] * self.template.num_time_steps,
                            "raw_step_norm_before_max_bid_step": 0.0,
                            "raw_step_max_abs_before_max_bid_step": 0.0,
                            "max_bid_step_scale": 1.0,
                            "bid_order_step_scale": 1.0,
                            "bid_order_limited_entries": 0,
                            "bid_order_epsilon": self.bid_order_epsilon,
                            "raw_step_norm_before_bid_order_limit": 0.0,
                            "step_norm_after_bid_order_limit": 0.0,
                            "raw_step_max_abs_before_bid_order_limit": 0.0,
                            "step_max_abs_after_bid_order_limit": 0.0,
                        }
                        print(f"    P{player_id}: skipped by configuration")
                        continue

                    player_stopped, recent_step_norms = self._player_has_stopped_updating(base_idx, player_id)
                    if player_stopped:
                        self.inactive_players_by_scenario[base_idx].add(player_id)

                    if player_id in self.inactive_players_by_scenario[base_idx]:
                        player_records[player_id] = {
                            "active": False,
                            "convergence_reason": "player_step_stagnation",
                            "recent_step_norms": recent_step_norms,
                            "history_rows_used": 0,
                            "average_gradient_norm": 0.0,
                            "max_row_gradient_norm": 0.0,
                            "mean_row_gradient_norm": 0.0,
                            "max_condition_number": 0.0,
                            "mean_condition_number": 0.0,
                            "step_norm": 0.0,
                            "step_by_time": [0.0] * self.template.num_time_steps,
                            "raw_step_norm_before_max_bid_step": 0.0,
                            "raw_step_max_abs_before_max_bid_step": 0.0,
                            "max_bid_step_scale": 1.0,
                            "bid_order_step_scale": 1.0,
                            "bid_order_limited_entries": 0,
                            "bid_order_epsilon": self.bid_order_epsilon,
                            "raw_step_norm_before_bid_order_limit": 0.0,
                            "step_norm_after_bid_order_limit": 0.0,
                            "raw_step_max_abs_before_bid_order_limit": 0.0,
                            "step_max_abs_after_bid_order_limit": 0.0,
                        }
                        print(
                            f"    P{player_id}: skipped, no bid change for "
                            f"{self.gradient_stagnation_iterations} iterations "
                            f"{self._format_profile(recent_step_norms, precision=6)}"
                        )
                        continue

                    avg_gradient, diagnostics = self._average_player_gradient(history_df, player_id)
                    history_df, step_norm, step_by_time, step_cap_diagnostics = self._update_active_row(
                        history_df=history_df,
                        player_id=player_id,
                        gradient=avg_gradient,
                        learning_rate=learning_rate,
                    )
                    max_step_norm = max(max_step_norm, step_norm)
                    max_gradient_norm = max(max_gradient_norm, diagnostics["average_gradient_norm"])
                    player_records[player_id] = {
                        "active": True,
                        "convergence_reason": "",
                        "history_rows_used": diagnostics["history_rows_used"],
                        "average_gradient_norm": diagnostics["average_gradient_norm"],
                        "max_row_gradient_norm": diagnostics["max_gradient_norm"],
                        "mean_row_gradient_norm": diagnostics["mean_gradient_norm"],
                        "max_condition_number": diagnostics["max_condition_number"],
                        "mean_condition_number": diagnostics["mean_condition_number"],
                        "step_norm": step_norm,
                        "step_by_time": step_by_time,
                        "raw_step_norm_before_max_bid_step": step_cap_diagnostics[
                            "raw_step_norm_before_max_bid_step"
                        ],
                        "raw_step_max_abs_before_max_bid_step": step_cap_diagnostics[
                            "raw_step_max_abs_before_max_bid_step"
                        ],
                        "max_bid_step_scale": step_cap_diagnostics["max_bid_step_scale"],
                        "bid_order_step_scale": step_cap_diagnostics["bid_order_step_scale"],
                        "bid_order_limited_entries": step_cap_diagnostics["bid_order_limited_entries"],
                        "bid_order_epsilon": step_cap_diagnostics["bid_order_epsilon"],
                        "raw_step_norm_before_bid_order_limit": step_cap_diagnostics[
                            "raw_step_norm_before_bid_order_limit"
                        ],
                        "step_norm_after_bid_order_limit": step_cap_diagnostics[
                            "step_norm_after_bid_order_limit"
                        ],
                        "raw_step_max_abs_before_bid_order_limit": step_cap_diagnostics[
                            "raw_step_max_abs_before_bid_order_limit"
                        ],
                        "step_max_abs_after_bid_order_limit": step_cap_diagnostics[
                            "step_max_abs_after_bid_order_limit"
                        ],
                        "average_gradient_by_time": diagnostics["average_gradient_by_time"],
                        "current_gradient_by_time": diagnostics["current_gradient_by_time"],
                    }
                    print(
                        f"    P{player_id}: avg_grad={diagnostics['average_gradient_norm']:.6f}, "
                        # f"avg_grad_t={self._format_profile(diagnostics['average_gradient_by_time'])}, "
                        # f"cur_grad_t={self._format_profile(diagnostics['current_gradient_by_time'])}, "
                        f"step={step_norm:.6f}, step_t={self._format_profile(step_by_time)}, "
                        f"raw_max_step={step_cap_diagnostics['raw_step_max_abs_before_max_bid_step']:.4f}, "
                        f"step_scale={step_cap_diagnostics['max_bid_step_scale']:.4f}, "
                        f"order_limited={step_cap_diagnostics['bid_order_limited_entries']}, "
                        f"rows={diagnostics['history_rows_used']}"
                    )

                all_players_inactive = all(
                    int(player["id"]) in self.inactive_players_by_scenario[base_idx]
                    for player in self.players_config
                )
                converged = all_players_inactive or max_step_norm <= self.conv_tolerance
                convergence_reason = (
                    "all_players_inactive"
                    if all_players_inactive
                    else "step_tolerance" if converged else ""
                )
                active_idx = self._active_row_index(history_df)
                history_df.at[active_idx, "converged"] = bool(converged)
                history_df.at[active_idx, "convergence_reason"] = convergence_reason
                history_df.at[active_idx, "max_step_norm"] = float(max_step_norm)
                history_df.at[active_idx, "max_gradient_norm"] = float(max_gradient_norm)

                if converged:
                    self.converged_scenarios[base_idx] = int(iteration)
                    self.final_scenario_rows[base_idx] = history_df.iloc[active_idx].to_dict()
                    self.scenario_histories[base_idx] = history_df
                    print(f"    converged: {convergence_reason}, max_step={max_step_norm:.6g}")
                else:
                    self.scenario_histories[base_idx] = self._append_next_update_ready_row(
                        history_df,
                        next_iteration=iteration,
                    )

                iteration_record["scenarios"][base_idx] = {
                    "converged": bool(converged),
                    "convergence_reason": convergence_reason,
                    "max_step_norm": float(max_step_norm),
                    "max_gradient_norm": float(max_gradient_norm),
                    "inactive_players": sorted(self.inactive_players_by_scenario[base_idx]),
                    "players": player_records,
                }

            self.active_scenarios = {
                idx for idx in self.active_scenarios
                if idx not in self.converged_scenarios
            }
            self.iteration_history.append(iteration_record)

        for base_idx in sorted(self.active_scenarios):
            history_df = self.scenario_histories[base_idx]
            active_idx = self._active_row_index(history_df)
            self.final_scenario_rows[base_idx] = history_df.iloc[active_idx].to_dict()

        self.results = self.get_results()
        return self.results

    def get_results(self) -> Dict[str, Any]:
        final_bid_history = self._build_final_bid_history()
        expanded_bid_history = self._build_expanded_bid_history()
        return {
            "num_base_scenarios": self.num_base_scenarios,
            "num_scenarios": self.num_base_scenarios,
            "num_time_steps": self.template.num_time_steps,
            "iterations_run": len(self.iteration_history),
            "converged_scenarios": {
                int(base_idx): int(iteration)
                for base_idx, iteration in self.converged_scenarios.items()
            },
            "active_scenarios_remaining": sorted(int(idx) for idx in self.active_scenarios),
            "inactive_players_by_scenario": {
                int(base_idx): sorted(int(player_id) for player_id in player_ids)
                for base_idx, player_ids in self.inactive_players_by_scenario.items()
            },
            "parameters": {
                "beta_smooth": self.beta_smooth,
                "learning_rate": self.learning_rate,
                "learning_rate_decay": self.learning_rate_decay,
                "min_learning_rate": self.min_learning_rate,
                "max_iterations": self.max_iterations,
                "conv_tolerance": self.conv_tolerance,
                "gradient_stagnation_iterations": self.gradient_stagnation_iterations,
                "gradient_stagnation_tolerance": self.gradient_stagnation_tolerance,
                "max_bid_step": self.max_bid_step,
                "gradient_clip_norm": self.gradient_clip_norm,
                "gradient_clip_mode": self.gradient_clip_mode,
                "gradient_history_window": self.gradient_history_window,
                "bid_order_step_fraction": self.bid_order_step_fraction,
                "bid_order_epsilon": self.bid_order_epsilon,
                "skip_player_ids": sorted(self.skip_player_ids),
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "kkt_regularization": self.kkt_regularization,
                "condition_warning_threshold": self.condition_warning_threshold,
                "initial_bid_rule": "physical_generator_max_block_cost_except_configured_skips",
            },
            "block_names": self.block_names,
            "physical_generator_names": self.template.physical_generator_names,
            "generator_names": self.template.generator_names,
            "block_to_physical": self.template.block_to_physical,
            "blocks_by_generator": self.template.blocks_by_generator,
            "generator_costs": self.cost_vector.tolist(),
            "block_costs": self.cost_vector.tolist(),
            "bid_history": final_bid_history,
            "expanded_bid_history": expanded_bid_history,
            "iteration_history": self.iteration_history,
            "final_scenario_rows": {
                int(base_idx): row
                for base_idx, row in self.final_scenario_rows.items()
            },
            "scenario_histories": {
                int(base_idx): history_df.to_dict(orient="records")
                for base_idx, history_df in self.scenario_histories.items()
            },
        }

    def _row_bid_matrix(self, row: pd.Series) -> List[List[float]]:
        bids = []
        for block_name in self.block_names:
            profile = self.template._as_profile(
                row[f"{block_name}_bid_profile"],
                f"{block_name}_bid_profile",
            )
            bids.append([float(v) for v in profile])
        return bids

    def _build_final_bid_history(self) -> List[List[List[List[float]]]]:
        final_bids = []
        for base_idx in range(self.num_base_scenarios):
            if base_idx in self.final_scenario_rows:
                row = pd.Series(self.final_scenario_rows[base_idx])
            else:
                history_df = self.scenario_histories[base_idx]
                row = history_df.iloc[self._active_row_index(history_df)]
            final_bids.append(self._row_bid_matrix(row))
        return [final_bids]

    def _build_expanded_bid_history(self) -> List[Dict[str, Any]]:
        rows = []
        for base_idx in range(self.num_base_scenarios):
            history_df = self.scenario_histories[base_idx]
            for row_idx, row in history_df.iterrows():
                rows.append({
                    "base_scenario_idx": int(row["base_scenario_idx"]),
                    "row_index": int(row_idx),
                    "training_iteration": int(row["training_iteration"]),
                    "history_role": str(row["history_role"]),
                    "is_update_ready": bool(row["is_update_ready"]),
                    "converged": bool(row["converged"]),
                    "bids": self._row_bid_matrix(row),
                })
        return rows

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

    @staticmethod
    def _json_default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    import time

    TEST_CASE = "test_case_bidding_blocks"

    scenario_manager = ScenarioManager(TEST_CASE)
    players_config = scenario_manager.get_players_config()
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set="policy_training",
        seed=1,
    )

    algo = ScenarioExpandedGradientBidTraining(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        players_config=players_config,
        beta_smooth=0.00001,
        learning_rate=1.00,
        learning_rate_decay=0.0,
        min_learning_rate=0.01,
        max_iterations=50,
        conv_tolerance=1e-4,
        gradient_stagnation_iterations=10,
        gradient_stagnation_tolerance=1e-4,
        max_bid_step=5.0,
        gradient_clip_norm=None,
        gradient_history_window=10,
        skip_player_ids=[0],
        kkt_regularization=1e-8,
        alpha_min=None,
        alpha_max=None,
    )

    start = time.perf_counter()
    algo.run()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")

    saved_path = algo.save_results("results/gradient_bid_training_expanded_results.json")
    print(saved_path)
