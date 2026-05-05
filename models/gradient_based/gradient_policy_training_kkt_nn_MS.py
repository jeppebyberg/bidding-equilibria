from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel

from models.gradient_based.sensitivities.kkt_sensitivity import (
    EDSolution,
    EDParameters,
    compute_market_sensitivities,
)
from models.gradient_based.sensitivities.profit_sensitivity import (
    ProfitParameters,
    compute_profit_sensitivities,
)
from models.gradient_based.sensitivities.policy_sensitivity import (
    PolicyParameters,
    compute_policy_bids_and_jacobian,
    flatten_parameters,
    unflatten_parameters,
)

class GradientPolicyTrainingKKTNNMS:
    """
    Analytical KKT-gradient policy training for multi-scenario NN bidding.

    The algorithm follows the sequential player-update structure of
    GradientPolicyTrainingMS, but computes gradients by chaining market KKT
    sensitivities, profit sensitivities, and one-hidden-layer ReLU policy
    sensitivities.
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
        NN_nodes: int = 4,
        random_seed: int = 1,
        nn_weight_scale: float = 1e-3,
        gradient_clip_norm: Optional[float] = None,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
        kkt_regularization: float = 1e-8,
        condition_warning_threshold: float = 1e10,
        policy_alpha_tolerance: float = 1e-6,
    ) -> None:
        if beta_smooth <= 0:
            raise ValueError(f"beta_smooth must be positive, got {beta_smooth}")
        if not feature_matrix_by_player:
            raise ValueError("feature_matrix_by_player must not be empty")
        if NN_nodes <= 0:
            raise ValueError(f"NN_nodes must be positive, got {NN_nodes}")
        if nn_weight_scale < 0:
            raise ValueError(f"nn_weight_scale must be nonnegative, got {nn_weight_scale}")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if alpha_min is not None and alpha_max is not None and alpha_min > alpha_max:
            raise ValueError("alpha_min cannot exceed alpha_max")
        if kkt_regularization < 0:
            raise ValueError("kkt_regularization must be nonnegative")
        if condition_warning_threshold <= 0:
            raise ValueError("condition_warning_threshold must be positive")
        if policy_alpha_tolerance < 0:
            raise ValueError("policy_alpha_tolerance must be nonnegative")

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
        self.NN_nodes = int(NN_nodes)
        self.random_seed = int(random_seed)
        self.nn_weight_scale = float(nn_weight_scale)
        self.gradient_clip_norm = gradient_clip_norm
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.kkt_regularization = float(kkt_regularization)
        self.condition_warning_threshold = float(condition_warning_threshold)
        self.policy_alpha_tolerance = float(policy_alpha_tolerance)
        self.gradient_method = "kkt_analytical_nn"

        capacity_cols = [col for col in self.scenarios_df.columns if col.endswith("_cap")]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'.")
        self.generator_names = [col.replace("_cap", "") for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(self.scenarios_df)
        self.num_time_steps = self._infer_num_time_steps()
        self.cost_vector = np.asarray(
            [float(self.costs_df[f"{gen}_cost"].iloc[0]) for gen in self.generator_names],
            dtype=np.float64,
        )
        self.player_index_by_id = {
            int(player["id"]): idx for idx, player in enumerate(self.players_config)
        }

        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)
        self.policy_params: Dict[int, PolicyParameters] = self._initialize_policy_params()
        self._apply_policy_to_scenarios()

        self.bid_history: List[List[List[List[float]]]] = []
        self.policy_params_history: List[Dict[int, Dict[str, Any]]] = []
        self.profit_history_training: List[List[float]] = []
        self.profit_history_training_scenario: List[List[List[float]]] = []
        self.dispatch_history: List[List[List[List[float]]]] = []
        self.clearing_price_history: List[List[List[float]]] = []
        self.gradient_norm_history: List[Dict[int, float]] = []
        self.gradient_diagnostics_history: List[Dict[int, Dict[str, Any]]] = []
        self.kkt_condition_history: List[Dict[int, Dict[str, float]]] = []
        self.iteration = 0
        self.results: Optional[Dict[str, Any]] = None

    def _infer_num_time_steps(self) -> int:
        if "time_steps" in self.scenarios_df.columns:
            return int(self.scenarios_df["time_steps"].iloc[0])
        return len(self._as_profile(self.scenarios_df["demand_profile"].iloc[0], "demand_profile"))

    @staticmethod
    def _as_profile(value: Any, column_name: str) -> List[float]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Column '{column_name}' must contain a list/tuple profile")
        return [float(v) for v in value]

    def _controlled_generators(self, player_id: int) -> List[int]:
        return [int(g) for g in self._get_player_config(player_id)["controlled_generators"]]

    def _initialize_policy_params(self) -> Dict[int, PolicyParameters]:
        rng = np.random.default_rng(self.random_seed)
        policy_params: Dict[int, PolicyParameters] = {}
        n_features = len(self.features)
        for player in self.players_config:
            player_id = int(player["id"])
            controlled = self._controlled_generators(player_id)
            n_owned = len(controlled)
            Gamma = rng.normal(
                loc=0.0,
                scale=self.nn_weight_scale,
                size=(n_owned, self.NN_nodes, n_features),
            )
            gamma = np.zeros((n_owned, self.NN_nodes), dtype=np.float64)
            Theta = rng.normal(
                loc=0.0,
                scale=self.nn_weight_scale,
                size=(n_owned, self.NN_nodes),
            )
            rho = np.asarray([self.cost_vector[g] for g in controlled], dtype=np.float64)
            policy_params[player_id] = PolicyParameters(
                Gamma=Gamma,
                gamma=gamma,
                Theta=Theta,
                rho=rho,
            )
        return policy_params

    def _compute_p_init_from_ed(self, scenarios_df: pd.DataFrame) -> List[List[float]]:
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

    def _clip_alpha_array(self, alpha: np.ndarray) -> np.ndarray:
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        lower = -np.inf if self.alpha_min is None else float(self.alpha_min)
        upper = np.inf if self.alpha_max is None else float(self.alpha_max)
        return np.clip(alpha_arr, lower, upper)

    def _features_for_player_scenario(
        self,
        player_id: int,
        scenario_idx: int,
        controlled_generators: List[int],
    ) -> np.ndarray:
        phi_by_player = self.feature_matrix_by_player[player_id]
        rows = []
        for t in range(self.num_time_steps):
            gen_features = [
                np.asarray(phi_by_player[(scenario_idx, t, int(gen_idx))], dtype=np.float64)
                for gen_idx in controlled_generators
            ]
            first = gen_features[0]
            for gen_idx, phi in zip(controlled_generators[1:], gen_features[1:]):
                if phi.shape != first.shape or not np.allclose(phi, first, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        "Feature vectors differ by owned generator for "
                        f"player {player_id}, scenario {scenario_idx}, time {t}. "
                        "policy_sensitivity.py currently expects one common feature matrix "
                        "per owned-generator policy evaluation."
                    )
            rows.append(first)

        features_s = np.vstack(rows)
        expected = (self.num_time_steps, len(self.features))
        if features_s.shape != expected:
            raise ValueError(f"features_s must have shape {expected}, got {features_s.shape}")
        if not np.all(np.isfinite(features_s)):
            raise ValueError(f"Non-finite features for player {player_id}, scenario {scenario_idx}")
        return features_s

    def _apply_policy_to_scenarios(self) -> None:
        for player in self.players_config:
            self._apply_player_policy_to_scenarios(int(player["id"]))

    def _apply_player_policy_to_scenarios(self, player_id: int) -> None:
        controlled = self._controlled_generators(player_id)
        params = self.policy_params[player_id]
        for s in range(self.num_scenarios):
            features_s = self._features_for_player_scenario(player_id, s, controlled)
            alpha_bids, _ = compute_policy_bids_and_jacobian(features_s, params)
            expected = (len(controlled), self.num_time_steps)
            if alpha_bids.shape != expected:
                raise ValueError(f"alpha_bids must have shape {expected}, got {alpha_bids.shape}")
            alpha_bids = self._clip_alpha_array(alpha_bids)
            if not np.all(np.isfinite(alpha_bids)):
                raise ValueError(f"Generated non-finite alpha bids for player {player_id}, scenario {s}")

            for local_idx, gen_idx in enumerate(controlled):
                gen_name = self.generator_names[int(gen_idx)]
                bid_profile = [float(v) for v in alpha_bids[local_idx, :]]
                self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = bid_profile
                self.scenarios_df.at[s, f"{gen_name}_bid"] = bid_profile[0]

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
    ) -> Tuple[float, List[float]]:
        controlled = self._controlled_generators(player_id)
        profit_params = self._build_profit_params()
        c_linear = np.asarray(profit_params.c_linear, dtype=np.float64).reshape(-1)
        c_quadratic = np.asarray(profit_params.c_quadratic, dtype=np.float64).reshape(-1)
        scenario_profits = []
        for s in range(self.num_scenarios):
            profit_s = 0.0
            for t in range(self.num_time_steps):
                for gen_idx in controlled:
                    p = float(dispatches[s][t][gen_idx])
                    profit_s += (
                        float(clearing_prices[s][t]) * p
                        - float(c_linear[gen_idx]) * p
                        - 0.5 * float(c_quadratic[gen_idx]) * p**2
                    )
            scenario_profits.append(float(profit_s))
        return float(np.mean(scenario_profits)), scenario_profits

    def compute_player_gradient(
        self,
        player_id: int,
    ) -> Tuple[PolicyParameters, float, List[float], Dict[str, Any]]:
        time_start = time.perf_counter()
        ed = self._solve_training_ed_model()
        time_end = time.perf_counter()
        print(f"ED solve time for gradient computation: {time_end - time_start:.4f} seconds")
        dispatches = ed.get_dispatches()
        clearing_prices = ed.get_clearing_prices()
        duals = self._extract_duals_from_ed(ed)
        if dispatches is None or clearing_prices is None:
            raise RuntimeError("ED solve did not return dispatches and clearing prices.")

        baseline_profit, baseline_scenario_profits = self.compute_player_profit(
            player_id,
            dispatches,
            clearing_prices,
        )

        controlled = self._controlled_generators(player_id)
        policy_params_j = self.policy_params[player_id]
        flat_gradients = []
        scenario_diagnostics = []
        n_theta = flatten_parameters(policy_params_j).size

        time_start = time.perf_counter()
        for s in range(self.num_scenarios):
            solution_s = self._build_ed_solution_for_scenario(s, dispatches, clearing_prices, duals)
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

            features_s = self._features_for_player_scenario(
                player_id=player_id,
                scenario_idx=s,
                controlled_generators=controlled,
            )
            alpha_bids, dalpha_dtheta = compute_policy_bids_and_jacobian(
                features=features_s,
                params=policy_params_j,
            )
            self._check_policy_bids_match_scenario_alpha(
                player_id=player_id,
                scenario_idx=s,
                alpha_bids=alpha_bids,
                controlled_generators=controlled,
                tolerance=self.policy_alpha_tolerance,
            )

            self._validate_gradient_shapes(
                market_sens=market_sens,
                profit_sens=profit_sens,
                dalpha_dtheta=dalpha_dtheta,
                n_owned=len(controlled),
                n_theta=n_theta,
            )

            dpi_dalpha = (
                np.asarray(profit_sens["dpi_dP"], dtype=np.float64) @ np.asarray(market_sens["dP_dalpha"], dtype=np.float64)
                + np.asarray(profit_sens["dpi_dlambda"], dtype=np.float64) @ np.asarray(market_sens["dlambda_dalpha"], dtype=np.float64)
            )
            dpi_dtheta = dpi_dalpha @ dalpha_dtheta
            if dpi_dtheta.shape != (n_theta,):
                raise ValueError(f"dpi_dtheta must have shape {(n_theta,)}, got {dpi_dtheta.shape}")

            flat_gradients.append(dpi_dtheta)
            scenario_diagnostics.append({
                "scenario": int(s),
                "profit": float(profit_sens["profit"]),
                "condition_number": float(market_sens["condition_number"]),
                "gradient_norm": float(np.linalg.norm(dpi_dtheta)),
            })
        time_end = time.perf_counter()
        print(f"Total gradient computation time for player {player_id}: {time_end - time_start:.4f} seconds")

        avg_flat_gradient = np.mean(np.vstack(flat_gradients), axis=0)
        if avg_flat_gradient.size != n_theta:
            raise ValueError(
                f"Average flat gradient length must be {n_theta}, got {avg_flat_gradient.size}"
            )

        gradient_params = unflatten_parameters(
            avg_flat_gradient,
            n_owned=len(controlled),
            n_neurons=self.NN_nodes,
            n_features=len(self.features),
        )
        condition_numbers = [d["condition_number"] for d in scenario_diagnostics]
        diagnostics = {
            "scenario_diagnostics": scenario_diagnostics,
            "max_condition_number": float(np.max(condition_numbers)),
            "mean_condition_number": float(np.mean(condition_numbers)),
            "flat_gradient_norm": float(np.linalg.norm(avg_flat_gradient)),
        }
        return gradient_params, baseline_profit, baseline_scenario_profits, diagnostics

    def _validate_gradient_shapes(
        self,
        market_sens: Dict[str, Any],
        profit_sens: Dict[str, Any],
        dalpha_dtheta: np.ndarray,
        n_owned: int,
        n_theta: int,
    ) -> None:
        n_alpha = int(n_owned) * self.num_time_steps
        n_dispatch = self.num_generators * self.num_time_steps
        if np.asarray(dalpha_dtheta).shape != (n_alpha, n_theta):
            raise ValueError(
                f"dalpha_dtheta must have shape {(n_alpha, n_theta)}, got {np.asarray(dalpha_dtheta).shape}"
            )
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

    def update_player_policy_params(
        self,
        player_id: int,
        gradient_params: PolicyParameters,
    ) -> None:
        current_flat = flatten_parameters(self.policy_params[player_id])
        gradient_flat = flatten_parameters(gradient_params)
        if current_flat.shape != gradient_flat.shape:
            raise ValueError(
                f"Current params and gradient disagree: {current_flat.shape} != {gradient_flat.shape}"
            )
        if self.gradient_clip_norm is not None:
            grad_norm = float(np.linalg.norm(gradient_flat))
            if grad_norm > self.gradient_clip_norm:
                gradient_flat = gradient_flat * (float(self.gradient_clip_norm) / (grad_norm + 1e-12))
        updated_flat = current_flat + self.learning_rate * gradient_flat
        controlled = self._controlled_generators(player_id)
        self.policy_params[player_id] = unflatten_parameters(
            updated_flat,
            n_owned=len(controlled),
            n_neurons=self.NN_nodes,
            n_features=len(self.features),
        )

    @staticmethod
    def _policy_gradient_norm(gradient_params: PolicyParameters) -> float:
        return float(np.linalg.norm(flatten_parameters(gradient_params)))

    def _build_ed_solution_for_scenario(
        self,
        scenario_idx: int,
        dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
        duals: Dict[str, List[Any]],
    ) -> EDSolution:
        s = int(scenario_idx)
        return EDSolution(
            P=np.asarray(dispatches[s], dtype=np.float64).T,
            lambda_=np.asarray(clearing_prices[s], dtype=np.float64),
            mu_max=np.asarray(duals["mu_max"][s], dtype=np.float64).T,
            mu_min=np.asarray(duals["mu_min"][s], dtype=np.float64).T,
            mu_up=np.asarray(duals["mu_up"][s], dtype=np.float64).T,
            mu_down=np.asarray(duals["mu_down"][s], dtype=np.float64).T,
        )

    def _build_ed_params_for_scenario(self, scenario_idx: int) -> EDParameters:
        s = int(scenario_idx)
        return EDParameters(
            alpha=self._scenario_alpha_matrix(s),
            beta=np.full(
                (self.num_generators, self.num_time_steps),
                self.beta_smooth,
                dtype=np.float64,
            ),
            demand=np.asarray(
                self._as_profile(self.scenarios_df.at[s, "demand_profile"], "demand_profile"),
                dtype=np.float64,
            ),
            pmax=self._scenario_pmax_matrix(s),
            pmin=np.zeros((self.num_generators, self.num_time_steps), dtype=np.float64),
            ramp_up=np.asarray(
                [float(self.ramps_df[f"{gen}_ramp_up"].iloc[0]) for gen in self.generator_names],
                dtype=np.float64,
            ),
            ramp_down=np.asarray(
                [float(self.ramps_df[f"{gen}_ramp_down"].iloc[0]) for gen in self.generator_names],
                dtype=np.float64,
            ),
            p_initial=np.asarray(self.P_init[s], dtype=np.float64),
        )

    def _scenario_alpha_matrix(self, scenario_idx: int) -> np.ndarray:
        alpha = np.zeros((self.num_generators, self.num_time_steps), dtype=np.float64)
        for i, gen_name in enumerate(self.generator_names):
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
        pmax = np.zeros((self.num_generators, self.num_time_steps), dtype=np.float64)
        for i, gen_name in enumerate(self.generator_names):
            profile_col = f"{gen_name}_cap_profile"
            wind_profile_col = f"{gen_name}_profile"
            if profile_col in self.scenarios_df.columns:
                profile = self._as_profile(self.scenarios_df.at[scenario_idx, profile_col], profile_col)
                pmax[i, :] = profile
            elif gen_name.startswith("W") and wind_profile_col in self.scenarios_df.columns:
                profile = self._as_profile(self.scenarios_df.at[scenario_idx, wind_profile_col], wind_profile_col)
                pmax[i, :] = profile
            else:
                pmax[i, :] = float(self.scenarios_df.at[scenario_idx, f"{gen_name}_cap"])
        return pmax

    def _build_profit_params(self) -> ProfitParameters:
        c_linear = np.asarray(
            [float(self.costs_df[f"{gen}_cost"].iloc[0]) for gen in self.generator_names],
            dtype=np.float64,
        )
        c_quadratic = np.zeros(self.num_generators, dtype=np.float64)
        for i, gen in enumerate(self.generator_names):
            for col in (f"{gen}_quadratic_cost", f"{gen}_cost_quadratic", f"{gen}_quad_cost"):
                if col in self.costs_df.columns:
                    c_quadratic[i] = float(self.costs_df[col].iloc[0])
                    break
        return ProfitParameters(c_linear=c_linear, c_quadratic=c_quadratic)

    def _check_policy_bids_match_scenario_alpha(
        self,
        player_id: int,
        scenario_idx: int,
        alpha_bids: np.ndarray,
        controlled_generators: List[int],
        tolerance: float,
    ) -> None:
        expected = (len(controlled_generators), self.num_time_steps)
        if alpha_bids.shape != expected:
            raise ValueError(f"alpha_bids must have shape {expected}, got {alpha_bids.shape}")
        stored = np.zeros_like(alpha_bids, dtype=np.float64)
        for local_idx, gen_idx in enumerate(controlled_generators):
            gen_name = self.generator_names[int(gen_idx)]
            stored[local_idx, :] = self._as_profile(
                self.scenarios_df.at[scenario_idx, f"{gen_name}_bid_profile"],
                f"{gen_name}_bid_profile",
            )
        generated = self._clip_alpha_array(alpha_bids)
        max_abs_diff = float(np.max(np.abs(generated - stored))) if stored.size else 0.0
        if max_abs_diff > tolerance:
            warnings.warn(
                "Policy-generated alpha bids differ from scenario_df stored bids for "
                f"player {player_id}, scenario {scenario_idx}: max_abs_diff={max_abs_diff:.3e}. "
                "Analytical KKT gradients require all components to be evaluated at the same bid profile.",
                RuntimeWarning,
                stacklevel=2,
            )

    def run(self) -> Dict[str, Any]:
        print("=== Starting KKT Analytical NN Gradient Policy Training ===")
        print(f"beta_smooth       : {self.beta_smooth}")
        print(f"learning_rate     : {self.learning_rate}")
        print(f"max_iterations    : {self.max_iterations}")
        print(f"features          : {self.features}")
        print(f"NN_nodes          : {self.NN_nodes}")
        print(f"players           : {[p['id'] for p in self.players_config]}")
        print(f"generators        : {self.generator_names}")

        self._record_iteration_state()

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration} ---")
            iteration_gradient_norms: Dict[int, float] = {}
            iteration_diagnostics: Dict[int, Dict[str, Any]] = {}
            iteration_conditions: Dict[int, Dict[str, float]] = {}

            for player in self.players_config:
                player_id = int(player["id"])
                print(f"  Player {player_id}")
                params_before = self._deepcopy_policy_params(raw=True)

                gradient_params, baseline_profit, _, diagnostics = self.compute_player_gradient(player_id)
                gradient_norm = self._policy_gradient_norm(gradient_params)
                iteration_gradient_norms[player_id] = gradient_norm
                iteration_diagnostics[player_id] = diagnostics
                iteration_conditions[player_id] = {
                    "max_condition_number": float(diagnostics["max_condition_number"]),
                    "mean_condition_number": float(diagnostics["mean_condition_number"]),
                }

                self.update_player_policy_params(player_id, gradient_params)
                self._assert_other_players_unchanged(player_id, params_before)
                self._apply_player_policy_to_scenarios(player_id)
                alpha_min, alpha_max = self._player_alpha_min_max(player_id)

                print(f"    baseline profit       : {baseline_profit:.6f}")
                print(f"    gradient norm         : {gradient_norm:.6f}")
                print(f"    max condition number  : {diagnostics['max_condition_number']:.3e}")
                print(f"    mean condition number : {diagnostics['mean_condition_number']:.3e}")
                print(f"    alpha min/max         : {alpha_min:.6f} / {alpha_max:.6f}")

            self.gradient_norm_history.append(iteration_gradient_norms)
            self.gradient_diagnostics_history.append(iteration_diagnostics)
            self.kkt_condition_history.append(iteration_conditions)
            self._record_iteration_state()

            max_gradient_norm = max(iteration_gradient_norms.values())
            cur_profit = self.profit_history_training[-1]
            profit_str = ", ".join(
                f"P{player['id']}={cur_profit[idx]:.3f}"
                for idx, player in enumerate(self.players_config)
            )
            print(f"  Training profits: {profit_str}")
            print(f"  Max gradient norm: {max_gradient_norm:.6f}")

            self.iteration = iteration
            if max_gradient_norm <= self.conv_tolerance:
                print("  Convergence achieved.")
                self.results = self.get_results()
                return self.results

        print("\nMaximum iterations reached without convergence.")
        self.results = self.get_results()
        return self.results

    def _record_iteration_state(self) -> None:
        dispatches, prices = self.solve_training_ed()
        player_profits, scenario_player_profits = self._compute_all_player_profits(dispatches, prices)
        self.bid_history.append(self._snapshot_all_bids())
        self.policy_params_history.append(self._deepcopy_policy_params())
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

    def get_results(self) -> Dict[str, Any]:
        dispatches, prices = self.solve_training_ed()
        player_profits, scenario_player_profits = self._compute_all_player_profits(dispatches, prices)
        return {
            "iterations": self.iteration,
            "num_scenarios": self.num_scenarios,
            "num_time_steps": self.num_time_steps,
            "generator_names": self.generator_names,
            "generator_costs": self.cost_vector.tolist(),
            "features": self.features.copy(),
            "beta_smooth": self.beta_smooth,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "conv_tolerance": self.conv_tolerance,
            "policy_type": "one_hidden_layer_relu",
            "NN_nodes": self.NN_nodes,
            "random_seed": self.random_seed,
            "nn_weight_scale": self.nn_weight_scale,
            "gradient_method": self.gradient_method,
            "kkt_regularization": self.kkt_regularization,
            "condition_warning_threshold": self.condition_warning_threshold,
            "policy_alpha_tolerance": self.policy_alpha_tolerance,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "bid_history": self.bid_history,
            "policy_params_history": self.policy_params_history,
            "final_policy_params": self._deepcopy_policy_params(),
            "profit_history_training": self.profit_history_training,
            "profit_history_training_scenario": self.profit_history_training_scenario,
            "dispatch_history": self.dispatch_history,
            "clearing_price_history": self.clearing_price_history,
            "gradient_norm_history": self.gradient_norm_history,
            "gradient_diagnostics_history": self.gradient_diagnostics_history,
            "kkt_condition_history": self.kkt_condition_history,
            "final_dispatches": dispatches,
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
                list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
                for gen_name in self.generator_names
            ]
            for s in range(self.num_scenarios)
        ]

    def _deepcopy_policy_params(self, raw: bool = False) -> Dict[int, Any]:
        copied = {
            pid: PolicyParameters(
                Gamma=params.Gamma.copy(),
                gamma=params.gamma.copy(),
                Theta=params.Theta.copy(),
                rho=params.rho.copy(),
            )
            for pid, params in self.policy_params.items()
        }
        if raw:
            return copied
        return {pid: self._policy_params_to_serializable(params) for pid, params in copied.items()}

    @staticmethod
    def _policy_params_to_serializable(params: PolicyParameters) -> Dict[str, Any]:
        return {
            "Gamma": np.asarray(params.Gamma, dtype=np.float64).tolist(),
            "gamma": np.asarray(params.gamma, dtype=np.float64).tolist(),
            "Theta": np.asarray(params.Theta, dtype=np.float64).tolist(),
            "rho": np.asarray(params.rho, dtype=np.float64).tolist(),
        }

    def _assert_other_players_unchanged(
        self,
        updated_player_id: int,
        params_before: Dict[int, PolicyParameters],
    ) -> None:
        for pid, params in self.policy_params.items():
            if pid == updated_player_id:
                continue
            before = params_before[pid]
            if (
                not np.array_equal(params.Gamma, before.Gamma)
                or not np.array_equal(params.gamma, before.gamma)
                or not np.array_equal(params.Theta, before.Theta)
                or not np.array_equal(params.rho, before.rho)
            ):
                raise RuntimeError(
                    f"Policy parameters for player {pid} changed during player "
                    f"{updated_player_id}'s update."
                )

    def _player_alpha_min_max(self, player_id: int) -> Tuple[float, float]:
        values = []
        for s in range(self.num_scenarios):
            for gen_idx in self._controlled_generators(player_id):
                gen_name = self.generator_names[int(gen_idx)]
                values.extend(float(v) for v in self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
        alpha_values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(alpha_values)):
            raise ValueError(f"Non-finite alpha values found for player {player_id}")
        return float(np.min(alpha_values)), float(np.max(alpha_values))

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

    from config.intertemporal.scenarios.scenario_generator_2 import ScenarioManagerV2
    from models.diagonalization.features.feature_setup import FeatureBuilder, DEFAULT_FEATURES

    TEST_CASE = "test_case1"

    scenario_manager_2 = ScenarioManagerV2(TEST_CASE)
    players_config_2 = scenario_manager_2.get_players_config()
    scenarios_2 = scenario_manager_2.create_scenario_set_from_regimes(
        regime_set="policy_training"
    )

    print(scenarios_2["description_text"])

    scenarios_df_2 = scenarios_2["scenarios_df"]
    costs_df_2 = scenarios_2["costs_df"]
    ramps_df_2 = scenarios_2["ramps_df"]

    generator_names = [
        c.replace("_cap", "")
        for c in scenarios_df_2.columns
        if c.endswith("_cap")
    ]

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

    algo = GradientPolicyTrainingKKTNNMS(
        scenarios_df_2,
        costs_df_2,
        ramps_df_2,
        players_config_2,
        feature_matrix_by_player,
        features,
        beta_smooth=0.01,
        learning_rate=0.5,
        max_iterations=10,
        conv_tolerance=1e-4,
        NN_nodes=4,
        random_seed=1,
        nn_weight_scale=1e-3,
        gradient_clip_norm=10.0,
        kkt_regularization=1e-8,
        alpha_min=None,
        alpha_max=None,
    )

    start = time.perf_counter()
    results = algo.run()
    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.6f} seconds")

    saved_path = algo.save_results(
        "results/gradient_policy_training_kkt_nn_results.json"
    )
    print(saved_path)
