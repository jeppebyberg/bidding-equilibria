from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel


class GradientPolicyTrainingMS:
    """
    Gradient-based policy training for multi-scenario intertemporal bidding.

    This class follows the sequential player-update structure of the existing
    best-response algorithm, but replaces the MPEC solve with finite-difference
    gradient ascent on linear bid-policy parameters.
    """

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        feature_matrix_by_player: Dict[int, Dict[Tuple[int, int, int], List[float]]],
        features: List[str],
        beta_smooth: float = 0.01,
        learning_rate: float = 1e-3,
        max_iterations: int = 25,
        conv_tolerance: float = 1e-4,
        gradient_method: str = "finite_difference",
        finite_difference_eps: float = 1e-4,
        gradient_clip_norm: Optional[float] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        NN_nodes: Optional[int] = None,
    ) -> None:
        if NN_nodes is not None:
            raise NotImplementedError(
                "GradientPolicyTrainingMS currently supports only the linear alpha policy. "
                "Neural-network policies are not implemented in this first version."
            )
        if beta_smooth <= 0:
            raise ValueError(f"beta_smooth must be positive, got {beta_smooth}")
        if not feature_matrix_by_player:
            raise ValueError("feature_matrix_by_player must not be empty")
        if gradient_method != "finite_difference":
            raise NotImplementedError("Only gradient_method='finite_difference' is implemented.")
        if finite_difference_eps <= 0:
            raise ValueError("finite_difference_eps must be positive")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if alpha_min is not None and alpha_max is not None and alpha_min > alpha_max:
            raise ValueError("alpha_min cannot exceed alpha_max")

        self.scenarios_df = scenarios_df.copy(deep=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.feature_matrix_by_player = feature_matrix_by_player
        self.features = list(features)
        self.beta_smooth = float(beta_smooth)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.conv_tolerance = float(conv_tolerance)
        self.gradient_method = gradient_method
        self.finite_difference_eps = float(finite_difference_eps)
        self.gradient_clip_norm = gradient_clip_norm
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.NN_nodes = NN_nodes

        capacity_cols = [col for col in self.scenarios_df.columns if col.endswith("_cap")]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'.")
        self.generator_names = [col.replace("_cap", "") for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(self.scenarios_df)
        self.num_time_steps = self._infer_num_time_steps()
        self.cost_vector = [float(self.costs_df[f"{gen}_cost"].iloc[0]) for gen in self.generator_names]
        self.player_index_by_id = {
            int(player["id"]): idx for idx, player in enumerate(self.players_config)
        }

        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)
        self.theta = self._compute_cost_theta()
        self._assert_theta_finite()
        self._apply_policy_to_scenarios()

        self.bid_history: List[List[List[List[float]]]] = []
        self.theta_history: List[Dict[int, Dict[int, np.ndarray]]] = []
        self.profit_history_training: List[List[float]] = []
        self.profit_history_training_scenario: List[List[List[float]]] = []
        self.dispatch_history: List[List[List[List[float]]]] = []
        self.clearing_price_history: List[List[List[float]]] = []
        self.gradient_norm_history: List[Dict[int, float]] = []
        self.iteration = 0
        self.results: Optional[Dict[str, Any]] = None

    def _infer_num_time_steps(self) -> int:
        if "time_steps" in self.scenarios_df.columns:
            return int(self.scenarios_df["time_steps"].iloc[0])
        demand_profile = self.scenarios_df["demand_profile"].iloc[0]
        return len(demand_profile)

    def _compute_p_init_from_ed(self, scenarios_df: pd.DataFrame) -> List[List[float]]:
        """Solve quadratic ED and extract first time-step dispatch as [scenario][generator]."""
        initial_dispatch = []
        for _, row in scenarios_df.iterrows():
            initial_dispatch.append([
                0.5 * float(row[f"{gen}_cap"])
                for gen in self.generator_names
            ])

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

    def _compute_cost_theta(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Initialize theta so alpha equals marginal cost when the player_cost
        feature is present; all other coefficients start at zero.
        """
        cost_theta: Dict[int, Dict[int, np.ndarray]] = {}
        for player in self.players_config:
            pid = int(player["id"])
            controlled = list(player["controlled_generators"])
            player_features = self.feature_matrix_by_player.get(pid, {})
            if not player_features:
                raise ValueError(f"No feature rows found for player {pid}")

            dim = len(next(iter(player_features.values())))
            theta_by_generator: Dict[int, np.ndarray] = {}
            for gen_idx in controlled:
                theta = np.zeros(dim, dtype=np.float64)
                if "player_cost" in self.features:
                    feature_idx = self.features.index("player_cost")
                    if feature_idx < dim:
                        theta[feature_idx] = float(self.cost_vector[gen_idx])
                theta_by_generator[int(gen_idx)] = theta

            cost_theta[pid] = theta_by_generator
        return cost_theta

    def _assert_theta_finite(self) -> None:
        for pid, theta_by_gen in self.theta.items():
            for gen_idx, theta in theta_by_gen.items():
                if not np.all(np.isfinite(theta)):
                    raise ValueError(f"Theta contains non-finite values for player {pid}, generator {gen_idx}")

    def _clip_alpha(self, alpha: float) -> float:
        if self.alpha_min is None and self.alpha_max is None:
            return float(alpha)
        lower = -np.inf if self.alpha_min is None else self.alpha_min
        upper = np.inf if self.alpha_max is None else self.alpha_max
        return float(np.clip(alpha, lower, upper))

    def _theta_to_bid_profile(self, player_id: int, gen_idx: int, scenario_idx: int) -> List[float]:
        theta = self.theta[player_id][gen_idx]
        phi_by_player = self.feature_matrix_by_player[player_id]
        bid_profile = []
        for t in range(self.num_time_steps):
            phi = np.asarray(phi_by_player[(scenario_idx, t, gen_idx)], dtype=np.float64)
            alpha = self._clip_alpha(float(theta @ phi))
            if not np.isfinite(alpha):
                raise ValueError(
                    f"Generated alpha is non-finite for player {player_id}, "
                    f"generator {gen_idx}, scenario {scenario_idx}, time {t}"
                )
            bid_profile.append(alpha)
        return bid_profile

    def _apply_policy_to_scenarios(self) -> None:
        for player in self.players_config:
            self._apply_player_policy_to_scenarios(int(player["id"]))

    def _apply_player_policy_to_scenarios(self, player_id: int) -> None:
        player_config = self._get_player_config(player_id)
        for s in range(self.num_scenarios):
            for gen_idx in player_config["controlled_generators"]:
                gen_idx = int(gen_idx)
                gen_name = self.generator_names[gen_idx]
                bid_profile = self._theta_to_bid_profile(player_id, gen_idx, s)
                self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = bid_profile
                self.scenarios_df.at[s, f"{gen_name}_bid"] = bid_profile[0]

    def solve_training_ed(self) -> Tuple[List[List[List[float]]], List[List[float]]]:
        ed = EconomicDispatchQuadraticModel(
            self.scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=self.P_init,
            beta_coeff=self.beta_smooth,
        )
        ed.solve()
        dispatches = ed.get_dispatches()
        clearing_prices = ed.get_clearing_prices()
        if dispatches is None or clearing_prices is None:
            raise RuntimeError("Quadratic ED solve did not return dispatches and clearing_prices.")
        return dispatches, clearing_prices

    def compute_player_profit(
        self,
        player_id: int,
        dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
    ) -> Tuple[float, List[float]]:
        player_config = self._get_player_config(player_id)
        controlled = [int(g) for g in player_config["controlled_generators"]]
        scenario_profits = []
        for s in range(self.num_scenarios):
            profit_s = 0.0
            for t in range(self.num_time_steps):
                for gen_idx in controlled:
                    profit_s += (
                        float(clearing_prices[s][t]) - float(self.cost_vector[gen_idx])
                    ) * float(dispatches[s][t][gen_idx])
            scenario_profits.append(float(profit_s))
        return float(np.mean(scenario_profits)), scenario_profits

    def compute_player_gradient(self, player_id: int) -> Tuple[Dict[int, np.ndarray], float, List[float]]:
        dispatches, clearing_prices = self.solve_training_ed()
        baseline_profit, baseline_scenario_profits = self.compute_player_profit(
            player_id, dispatches, clearing_prices
        )

        original_theta = {
            gen_idx: theta.copy()
            for gen_idx, theta in self.theta[player_id].items()
        }
        original_bid_profiles = self._snapshot_player_bid_profiles(player_id)
        gradient = {
            gen_idx: np.zeros_like(theta, dtype=np.float64)
            for gen_idx, theta in self.theta[player_id].items()
        }

        try:
            for gen_idx, theta in self.theta[player_id].items():
                for k in range(len(theta)):
                    self.theta[player_id][gen_idx][k] += self.finite_difference_eps
                    self._apply_player_policy_to_scenarios(player_id)
                    perturbed_dispatches, perturbed_prices = self.solve_training_ed()
                    perturbed_profit, _ = self.compute_player_profit(
                        player_id, perturbed_dispatches, perturbed_prices
                    )
                    gradient[gen_idx][k] = (
                        perturbed_profit - baseline_profit
                    ) / self.finite_difference_eps

                    self.theta[player_id][gen_idx][k] = original_theta[gen_idx][k]
                    self._restore_player_bid_profiles(player_id, original_bid_profiles)
        finally:
            for gen_idx, theta in original_theta.items():
                self.theta[player_id][gen_idx] = theta.copy()
            self._restore_player_bid_profiles(player_id, original_bid_profiles)

        return gradient, baseline_profit, baseline_scenario_profits

    def update_player_theta(self, player_id: int, gradient: Dict[int, np.ndarray]) -> None:
        if self.gradient_clip_norm is not None:
            total_norm = self._gradient_norm(gradient)
            if total_norm > self.gradient_clip_norm:
                scale = self.gradient_clip_norm / (total_norm + 1e-12)
                gradient = {gen_idx: grad * scale for gen_idx, grad in gradient.items()}

        for gen_idx, grad in gradient.items():
            self.theta[player_id][gen_idx] = self.theta[player_id][gen_idx] + self.learning_rate * grad
        self._assert_theta_finite()

    def run(self) -> Dict[str, Any]:
        print("=== Starting Gradient Policy Training ===")
        print(f"beta_smooth       : {self.beta_smooth}")
        print(f"learning_rate     : {self.learning_rate}")
        print(f"max_iterations    : {self.max_iterations}")
        print(f"features          : {self.features}")
        print(f"players           : {[p['id'] for p in self.players_config]}")
        print(f"generators        : {self.generator_names}")

        self._record_iteration_state()
        prev_profit = self.profit_history_training[-1]

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration} ---")
            iteration_gradient_norms: Dict[int, float] = {}

            for player in self.players_config:
                player_id = int(player["id"])
                print(f"  Player {player_id}")
                theta_before = self._deepcopy_theta()

                gradient, baseline_profit, _ = self.compute_player_gradient(player_id)
                gradient_norm = self._gradient_norm(gradient)
                iteration_gradient_norms[player_id] = gradient_norm
                self.update_player_theta(player_id, gradient)
                self._assert_other_players_unchanged(player_id, theta_before)

                self._apply_player_policy_to_scenarios(player_id)
                dispatches, clearing_prices = self.solve_training_ed()
                updated_profit, _ = self.compute_player_profit(player_id, dispatches, clearing_prices)
                alpha_min, alpha_max = self._player_alpha_min_max(player_id)

                print(f"    baseline profit    : {baseline_profit:.6f}")
                print(f"    profit after update: {updated_profit:.6f}")
                print(f"    gradient norm      : {gradient_norm:.6f}")
                print(f"    alpha min/max      : {alpha_min:.6f} / {alpha_max:.6f}")

            self.gradient_norm_history.append(iteration_gradient_norms)
            self._record_iteration_state()
            cur_profit = self.profit_history_training[-1]

            profit_str = ", ".join(
                f"P{player['id']}={cur_profit[idx]:.3f}"
                for idx, player in enumerate(self.players_config)
            )
            print(f"  Training profits: {profit_str}")

            if self._profits_converged(cur_profit, prev_profit):
                print("  Convergence achieved.")
                self.iteration = iteration
                self.results = self.get_results()
                return self.results

            prev_profit = cur_profit
            self.iteration = iteration

        print("\nMaximum iterations reached without convergence.")
        self.results = self.get_results()
        return self.results

    def get_results(self) -> Dict[str, Any]:
        dispatches, prices = self.solve_training_ed()
        player_profits, scenario_player_profits = self._compute_all_player_profits(dispatches, prices)
        return {
            "iterations": self.iteration,
            "num_scenarios": self.num_scenarios,
            "num_time_steps": self.num_time_steps,
            "generator_names": self.generator_names,
            "generator_costs": self.cost_vector.copy(),
            "features": self.features.copy(),
            "beta_smooth": self.beta_smooth,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "conv_tolerance": self.conv_tolerance,
            "gradient_method": self.gradient_method,
            "finite_difference_eps": self.finite_difference_eps,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "bid_history": self.bid_history,
            "theta_history": self.theta_history,
            "final_thetas": self._deepcopy_theta(),
            "profit_history_training": self.profit_history_training,
            "profit_history_training_scenario": self.profit_history_training_scenario,
            "dispatch_history": self.dispatch_history,
            "clearing_price_history": self.clearing_price_history,
            "gradient_norm_history": self.gradient_norm_history,
            "final_dispatches": dispatches,
            "final_clearing_prices": prices,
            "final_player_profits": player_profits,
            "final_player_profits_scenario": scenario_player_profits,
        }

    @staticmethod
    def _json_default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(key): value for key, value in obj.items()}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

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

    def _record_iteration_state(self) -> None:
        dispatches, prices = self.solve_training_ed()
        player_profits, scenario_player_profits = self._compute_all_player_profits(dispatches, prices)
        self.bid_history.append(self._snapshot_all_bids())
        self.theta_history.append(self._deepcopy_theta())
        self.profit_history_training.append(player_profits)
        self.profit_history_training_scenario.append(scenario_player_profits)
        self.dispatch_history.append(dispatches)
        self.clearing_price_history.append(prices)

    def _compute_all_player_profits(
        self,
        dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
    ) -> Tuple[List[float], List[List[float]]]:
        player_profits = []
        scenario_by_player = [[0.0 for _ in self.players_config] for _ in range(self.num_scenarios)]
        for player_idx, player in enumerate(self.players_config):
            avg_profit, scenario_profits = self.compute_player_profit(int(player["id"]), dispatches, clearing_prices)
            player_profits.append(avg_profit)
            for s, profit_s in enumerate(scenario_profits):
                scenario_by_player[s][player_idx] = profit_s
        return player_profits, scenario_by_player

    def _snapshot_all_bids(self) -> List[List[List[float]]]:
        return [
            [
                list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
                for gen_name in self.generator_names
            ]
            for s in range(self.num_scenarios)
        ]

    def _snapshot_player_bid_profiles(self, player_id: int) -> Dict[Tuple[int, int], List[float]]:
        player_config = self._get_player_config(player_id)
        snapshot = {}
        for s in range(self.num_scenarios):
            for gen_idx in player_config["controlled_generators"]:
                gen_name = self.generator_names[int(gen_idx)]
                snapshot[(s, int(gen_idx))] = list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
        return snapshot

    def _restore_player_bid_profiles(
        self,
        player_id: int,
        snapshot: Dict[Tuple[int, int], List[float]],
    ) -> None:
        for (s, gen_idx), bid_profile in snapshot.items():
            gen_name = self.generator_names[gen_idx]
            self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = list(bid_profile)
            self.scenarios_df.at[s, f"{gen_name}_bid"] = float(bid_profile[0])

    def _deepcopy_theta(self) -> Dict[int, Dict[int, np.ndarray]]:
        return {
            pid: {gen_idx: theta.copy() for gen_idx, theta in theta_by_gen.items()}
            for pid, theta_by_gen in self.theta.items()
        }

    @staticmethod
    def _gradient_norm(gradient: Dict[int, np.ndarray]) -> float:
        return float(np.sqrt(sum(float(np.sum(grad ** 2)) for grad in gradient.values())))

    def _profits_converged(self, cur_profit: List[float], prev_profit: List[float]) -> bool:
        checks = []
        for cur, prev in zip(cur_profit, prev_profit):
            checks.append(
                abs(cur - prev) <= self.conv_tolerance * abs(prev) + self.conv_tolerance
            )
        return all(checks)

    def _player_alpha_min_max(self, player_id: int) -> Tuple[float, float]:
        values = []
        player_config = self._get_player_config(player_id)
        for s in range(self.num_scenarios):
            for gen_idx in player_config["controlled_generators"]:
                gen_name = self.generator_names[int(gen_idx)]
                values.extend(float(v) for v in self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
        alpha_values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(alpha_values)):
            raise ValueError(f"Non-finite alpha values found for player {player_id}")
        return float(np.min(alpha_values)), float(np.max(alpha_values))

    def _assert_other_players_unchanged(
        self,
        updated_player_id: int,
        theta_before: Dict[int, Dict[int, np.ndarray]],
    ) -> None:
        for pid, theta_by_gen in self.theta.items():
            if pid == updated_player_id:
                continue
            for gen_idx, theta in theta_by_gen.items():
                if not np.array_equal(theta, theta_before[pid][gen_idx]):
                    raise RuntimeError(
                        f"Theta for player {pid}, generator {gen_idx} changed during "
                        f"player {updated_player_id}'s update."
                    )

    def _get_player_config(self, player_id: int) -> Dict[str, Any]:
        for player in self.players_config:
            if int(player["id"]) == int(player_id):
                return player
        raise KeyError(f"Unknown player_id: {player_id}")

if __name__ == "__main__":
    import time

    from config.intertemporal.scenarios.scenario_generator_2 import ScenarioManagerV2
    from models.diagonalization.features.feature_setup import FeatureBuilder, DEFAULT_FEATURES

    TEST_CASE = "test_case1"

    scenario_manager_2 = ScenarioManagerV2(TEST_CASE)
    players_config_2 = scenario_manager_2.get_players_config()
    scenarios_2 = scenario_manager_2.create_scenario_set_from_regimes(regime_set="policy_training")

    print(scenarios_2["description_text"])

    scenarios_df_2 = scenarios_2["scenarios_df"]
    costs_df_2 = scenarios_2["costs_df"]
    ramps_df_2 = scenarios_2["ramps_df"]
    generator_names = [c.replace("_cap", "") for c in scenarios_df_2.columns if c.endswith("_cap")]

    fb = FeatureBuilder(TEST_CASE, DEFAULT_FEATURES)
    feature_matrix_by_player = fb.build_intertemporal_feature_matrix_by_player_from_frames(
        scenarios_df=scenarios_df_2,
        costs_df=costs_df_2,
        generator_names=generator_names,
        players_config=players_config_2,
        fit_normalizer=True,
    )
    fb.save_feature_normalizer_stats("results/feature_normalizer_stats.json")
    features = fb.features

    algo = GradientPolicyTrainingMS(
        scenarios_df_2,
        costs_df_2,
        ramps_df_2,
        players_config_2,
        feature_matrix_by_player,
        features,
        beta_smooth=0.01,
        learning_rate=1e-3,
        max_iterations=10,
        finite_difference_eps=1e-4,
        alpha_min=None,
        alpha_max=None,
    )

    start = time.perf_counter()
    results = algo.run()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")

    saved_path = algo.save_results("results/gradient_policy_training_results.json")
    print(saved_path)

    stop = True