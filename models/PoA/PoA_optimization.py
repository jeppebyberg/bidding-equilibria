import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.intertemporal.utils.cases_utils import load_setup_data

class PoAOptimization:
    def __init__(
            self,
            P_init,
            num_time_steps: int = 24,
            reference_case: str = "test_case1",
            feature_normalizer_stats_path: str = "results/feature_normalizer_stats.json",
            big_m_complementarity: float = 1e6,
            policy_results_path: Optional[str] = None,
            policy_data: Optional[Dict[str, Any]] = None,
            support_set_config: Optional[Dict[str, Any]] = None,
    ):
        self.P_init = P_init
        self.num_time_steps = num_time_steps

        self.reference_case = reference_case
        self.feature_normalizer_stats_path = Path(feature_normalizer_stats_path)
        self.big_m_complementarity = float(big_m_complementarity)

        self.feature_names = []
        self.feature_min = None
        self.feature_max = None
        self.private_feature_names = []
        self.player_private_min_max: Dict[int, Dict[str, np.ndarray]] = {}
        self.normalization_epsilon = 1e-12

        # Policy payload loaded from BR/gradient-training results.
        self.policy_type: Optional[str] = None  # "linear" or "nn"
        self.policy_by_generator: Dict[int, Any] = {}
        self.policy_metadata: Dict[str, Any] = {}
        self.support_set_config = support_set_config or {}

        # Load setup directly from config/intertemporal/reference_cases.yaml.
        # This keeps PoA aligned with the same reference-case source as the BR scripts.
        self._load_reference_case_setup()

        self.get_feature_normalization_stats()
        self._configure_support_set_parameters()

        if policy_data is not None:
            self.get_policy_stats(policy_data=policy_data)
        elif policy_results_path is not None:
            self.get_policy_stats(policy_results_path=policy_results_path)

    @staticmethod
    def _is_wind_generator(generator: Dict[str, Any]) -> bool:
        """Classify wind generators from the name pattern used in reference cases (e.g., W3)."""
        name_value = str(generator.get("name", "")).strip()
        if not name_value:
            return False
        return name_value[0].upper() == "W"

    def _build_generator_type_sets(self) -> None:
        """Create disjoint index sets for wind and conventional generators."""
        self.wind_generator_ids = [
            gen_idx
            for gen_idx, generator in enumerate(self.generators)
            if self._is_wind_generator(generator)
        ]
        self.conventional_generator_ids = [
            gen_idx
            for gen_idx in range(self.num_generators)
            if gen_idx not in self.wind_generator_ids
        ]

    def _load_reference_case_setup(self) -> None:
        """Load generator/time-step setup from intertemporal reference cases."""
        try:
            (
                num_generators,
                pmax_list,
                pmin_list,
                cost_vector,
                r_rates_up_list,
                r_rates_down_list,
                demand,
                generators,
                players,
                time_steps,
            ) = load_setup_data(self.reference_case)
        except Exception as exc:
            raise ValueError(f"Failed to load reference case '{self.reference_case}': {exc}") from exc

        self.num_generators = int(num_generators)
        self.generators = generators
        self.players_config = players
        self.demand = float(demand)
        self.pmax_list = [float(v) for v in pmax_list]
        self.pmin_list = [float(v) for v in pmin_list]
        self.cost_vector = [float(v) for v in cost_vector]
        self.ramp_vector_up = [float(v) for v in r_rates_up_list]
        self.ramp_vector_down = [float(v) for v in r_rates_down_list]
        self._build_generator_type_sets()

        self.base_case = {
            "case_name": self.reference_case,
            "num_generators": self.num_generators,
            "generators": self.generators,
            "wind_generators": self.wind_generator_ids,
            "conventional_generators": self.conventional_generator_ids,
            "players": self.players_config,
            "demand": self.demand,
            "pmax_list": self.pmax_list,
            "pmin_list": self.pmin_list,
            "cost_vector": self.cost_vector,
            "ramp_vector_up": self.ramp_vector_up,
            "ramp_vector_down": self.ramp_vector_down,
        }

    def _build_model(self) -> None:
        """
        Build the complete MPEC model structure.
        This is called once, then update_strategic_player() updates the changing parts between each strategic player.
        """

        self.model = ConcreteModel()
        
        # Define sets
        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.n_gen_wind = Set(initialize=self.wind_generator_ids)
        self.model.n_gen_conventional = Set(initialize=self.conventional_generator_ids)
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1)) # For ramp constraints
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps)) 

        self.get_feature_normalization_stats()

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def get_feature_normalization_stats(self) -> None:
        """Load min-max normalization stats used when training the policy."""
        if not self.feature_normalizer_stats_path.exists():
            raise FileNotFoundError(
                f"Feature normalization stats not found: {self.feature_normalizer_stats_path}"
            )

        with self.feature_normalizer_stats_path.open("r", encoding="utf-8") as file_handle:
            stats = json.load(file_handle)

        self.feature_names = list(stats.get("feature_names", []))
        self.feature_min = np.asarray(stats.get("min", []), dtype=np.float64)
        self.feature_max = np.asarray(stats.get("max", []), dtype=np.float64)
        self.private_feature_names = list(stats.get("private_feature_names", []))

        if len(self.feature_names) != len(self.feature_min) or len(self.feature_names) != len(self.feature_max):
            raise ValueError(
                "Invalid feature normalizer stats: feature_names, min, and max must have the same length."
            )

        self.player_private_min_max = {}
        for pid_str, player_stats in stats.get("player_private_min_max", {}).items():
            pid = int(pid_str)
            self.player_private_min_max[pid] = {
                "min": np.asarray(player_stats.get("min", []), dtype=np.float64),
                "max": np.asarray(player_stats.get("max", []), dtype=np.float64),
            }

    def get_policy_stats(
        self,
        policy_results_path: Optional[str] = None,
        policy_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Load policy parameters and normalize them to:
        - self.policy_type: "linear" or "nn"
        - self.policy_by_generator: dict[int, Any]
        """
        if policy_data is None and policy_results_path is None:
            raise ValueError("Provide either policy_data or policy_results_path")

        if policy_data is None:
            path = Path(str(policy_results_path))
            if not path.exists():
                raise FileNotFoundError(f"Policy results file not found: {path}")
            with path.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        else:
            payload = policy_data

        def _canonical_policy_type(raw_policy_type: Any) -> str:
            local_policy_type = str(raw_policy_type or "").lower()
            if local_policy_type in {"linear", "affine"}:
                return "linear"
            if local_policy_type in {"nn", "one_hidden_layer_relu", "one-hidden-layer-relu"}:
                return "nn"
            return local_policy_type

        def _controlled_generators_for_player(player_id: int) -> List[int]:
            controlled = next(
                (
                    p.get("controlled_generators", [])
                    for p in self.players_config
                    if int(p.get("id")) == int(player_id)
                ),
                None,
            )
            if controlled is None:
                raise ValueError(f"No player config found for policy player {player_id}")
            return [int(g) for g in controlled]

        def _flatten_gradient_nn_policy_payload(local_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
            """
            Convert gradient_policy_training.py final_policy_params to PoA's per-generator NN schema.

            Training schema per player:
                Gamma: input weights, shape (n_owned, n_nodes, n_features)
                gamma: hidden bias, shape (n_owned, n_nodes)
                Theta: output weights, shape (n_owned, n_nodes)
                rho: output bias, shape (n_owned,)

            PoA schema per generator:
                gamma: input weights, shape (n_nodes, n_features)
                Theta: hidden bias, shape (n_nodes,)
                Gamma: output weights, shape (n_nodes,)
                output_bias: scalar
            """
            final_params = local_payload.get("final_policy_params", {})
            if not isinstance(final_params, dict) or not final_params:
                raise ValueError("final_policy_params must be a non-empty mapping")

            per_generator: Dict[int, Dict[str, Any]] = {}
            for player_key, raw_params in final_params.items():
                player_id = int(player_key)
                controlled = _controlled_generators_for_player(player_id)

                input_weights = np.asarray(raw_params.get("Gamma", []), dtype=np.float64)
                hidden_bias = np.asarray(raw_params.get("gamma", []), dtype=np.float64)
                output_weights = np.asarray(raw_params.get("Theta", []), dtype=np.float64)
                output_bias = np.asarray(raw_params.get("rho", []), dtype=np.float64)

                if input_weights.ndim != 3:
                    raise ValueError(
                        f"Player {player_id}: final_policy_params.Gamma must have shape "
                        f"(n_owned, n_nodes, n_features), got {input_weights.shape}"
                    )

                n_owned, n_nodes, n_features = input_weights.shape
                expected_hidden_shape = (n_owned, n_nodes)
                if hidden_bias.shape != expected_hidden_shape:
                    raise ValueError(
                        f"Player {player_id}: final_policy_params.gamma must have shape "
                        f"{expected_hidden_shape}, got {hidden_bias.shape}"
                    )
                if output_weights.shape != expected_hidden_shape:
                    raise ValueError(
                        f"Player {player_id}: final_policy_params.Theta must have shape "
                        f"{expected_hidden_shape}, got {output_weights.shape}"
                    )
                if output_bias.shape != (n_owned,):
                    raise ValueError(
                        f"Player {player_id}: final_policy_params.rho must have shape "
                        f"({n_owned},), got {output_bias.shape}"
                    )
                if len(controlled) != n_owned:
                    raise ValueError(
                        f"Player {player_id}: policy has {n_owned} owned-generator blocks, "
                        f"but player config controls {len(controlled)} generators"
                    )

                if n_features != len(self.feature_names):
                    raise ValueError(
                        f"Player {player_id}: NN feature dimension {n_features} does not match "
                        f"loaded normalizer feature dimension {len(self.feature_names)}"
                    )

                for local_idx, gen_idx in enumerate(controlled):
                    per_generator[int(gen_idx)] = {
                        "gamma": input_weights[local_idx].astype(float).tolist(),
                        "Theta": hidden_bias[local_idx].astype(float).tolist(),
                        "Gamma": output_weights[local_idx].astype(float).tolist(),
                        "output_bias": float(output_bias[local_idx]),
                    }

            return per_generator

        if (
            isinstance(payload, dict)
            and "policy_type" in payload
            and "policy_by_generator" in payload
        ):
            extracted = payload
        else:
            def _flatten_linear_policy_payload(local_payload: Dict[str, Any]) -> Dict[int, List[float]]:
                per_generator: Dict[int, List[float]] = {}
                for player_key, value in local_payload.items():
                    player_id = int(player_key)
                    if isinstance(value, dict):
                        for gen_key, theta_vec in value.items():
                            per_generator[int(gen_key)] = theta_vec
                    else:
                        controlled = next(
                            (
                                p.get("controlled_generators", [])
                                for p in self.players_config
                                if int(p.get("id")) == player_id
                            ),
                            [],
                        )
                        for gen_idx in controlled:
                            per_generator[int(gen_idx)] = value
                return per_generator

            if "nn_policy_weights" in payload and payload["nn_policy_weights"]:
                extracted = {
                    "policy_type": "nn",
                    "policy_by_generator": {
                        int(k): v for k, v in payload["nn_policy_weights"].items()
                    },
                }
            elif "final_policy_params" in payload and payload["final_policy_params"]:
                extracted = {
                    "policy_type": "nn",
                    "policy_by_generator": _flatten_gradient_nn_policy_payload(payload),
                }
            elif "nn_policy_weights_history" in payload and payload["nn_policy_weights_history"]:
                latest = payload["nn_policy_weights_history"][-1]
                per_generator: Dict[int, Any] = {}
                for _, player_payload in latest.items():
                    if not isinstance(player_payload, dict):
                        continue
                    for gk, gval in player_payload.items():
                        per_generator[int(gk)] = gval
                if not per_generator:
                    raise ValueError("No NN generator payload found in nn_policy_weights_history")
                extracted = {"policy_type": "nn", "policy_by_generator": per_generator}
            elif "final_thetas" in payload and payload["final_thetas"]:
                extracted = {
                    "policy_type": "linear",
                    "policy_by_generator": _flatten_linear_policy_payload(payload["final_thetas"]),
                }
            elif "theta_history" in payload and payload["theta_history"]:
                extracted = {
                    "policy_type": "linear",
                    "policy_by_generator": _flatten_linear_policy_payload(payload["theta_history"][-1]),
                }
            else:
                raise ValueError("No supported policy payload found in policy input")

        policy_type = _canonical_policy_type(extracted.get("policy_type", ""))
        if policy_type not in {"linear", "nn"}:
            raise ValueError(f"Unsupported policy_type '{policy_type}'. Expected 'linear' or 'nn'.")

        payload_features = payload.get("features", []) if isinstance(payload, dict) else []
        if payload_features and list(payload_features) != list(self.feature_names):
            raise ValueError(
                "Policy feature order does not match loaded feature normalizer stats. "
                f"Policy: {list(payload_features)}, normalizer: {self.feature_names}"
            )

        raw_map = extracted.get("policy_by_generator", {})
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("policy_by_generator must be a non-empty dict")

        self.policy_type = policy_type
        self.policy_by_generator = {int(k): v for k, v in raw_map.items()}
        if isinstance(payload, dict):
            self.policy_metadata = {
                "source_policy_type": payload.get("policy_type"),
                "features": list(payload.get("features", [])),
                "alpha_bounds": payload.get("alpha_bounds", {}),
                "num_time_steps": payload.get("num_time_steps"),
                "NN_nodes": payload.get("NN_nodes"),
            }

    @staticmethod
    def load_support_set_config(
        config_path: str = "models/PoA/support_set_config.yaml",
        config_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load one named support-set configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Support-set config not found: {path}")

        with path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle) or {}

        if "support_sets" not in raw_config:
            return raw_config

        support_sets = raw_config.get("support_sets")
        if not isinstance(support_sets, dict) or not support_sets:
            raise ValueError("'support_sets' must be a non-empty mapping")

        selected_name = config_name or raw_config.get("default_support_set") or next(iter(support_sets))
        if selected_name not in support_sets:
            available = ", ".join(support_sets.keys())
            raise ValueError(f"Unknown support-set config '{selected_name}'. Available: {available}")

        return support_sets[selected_name] or {}

    @staticmethod
    def _as_profile(value: Any, horizon: int, name: str) -> List[float]:
        """Expand a scalar/profile value to one float per time step."""
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            profile = [float(v) for v in value]
        else:
            profile = [float(value)] * horizon

        if len(profile) != horizon:
            raise ValueError(f"{name} must have length {horizon}, got {len(profile)}")

        return profile

    def _feature_bound(self, feature_name: str, bound: str, default: float) -> float:
        """Read min/max feature-normalizer bounds when available."""
        if feature_name not in self.feature_names:
            return float(default)

        feature_idx = self.feature_names.index(feature_name)
        values = self.feature_min if bound == "min" else self.feature_max
        if values is None or feature_idx >= len(values):
            return float(default)

        return float(values[feature_idx])

    def _generator_name(self, gen_idx: int) -> str:
        generator = self.generators[int(gen_idx)]
        if isinstance(generator, dict):
            return str(generator.get("name", f"G{gen_idx}"))
        return str(generator)

    def _per_generator_config_value(self, config_value: Any, gen_idx: int, default: Any) -> Any:
        """Read either a scalar/list config value or a per-generator mapping."""
        if isinstance(config_value, dict):
            gen_name = self._generator_name(gen_idx)
            lookup_keys = (int(gen_idx), str(gen_idx), gen_name, gen_name.upper(), gen_name.lower())
            for key in lookup_keys:
                if key in config_value:
                    return config_value[key]
            return default

        if config_value is None:
            return default

        return config_value

    def _wind_generator_config_value(self, cfg: Dict[str, Any], field_name: str, gen_idx: int, default: Any) -> Any:
        """Read wind support config from the grouped schema, falling back to legacy keys."""
        grouped_cfg = cfg.get("wind_generators")
        if isinstance(grouped_cfg, dict):
            gen_name = self._generator_name(gen_idx)
            lookup_keys = (int(gen_idx), str(gen_idx), gen_name, gen_name.upper(), gen_name.lower())
            for key in lookup_keys:
                if key not in grouped_cfg:
                    continue
                generator_cfg = grouped_cfg[key]
                if isinstance(generator_cfg, dict) and field_name in generator_cfg:
                    return generator_cfg[field_name]

        legacy_key = {
            "reference": "wind_reference",
            "min": "wind_min",
            "max": "wind_max",
        }[field_name]
        return self._per_generator_config_value(cfg.get(legacy_key), gen_idx, default)

    def _configure_support_set_parameters(self) -> None:
        """Set support-set parameters, using config overrides when provided."""
        cfg = self.support_set_config

        self.support_demand_reference = self._as_profile(
            cfg.get("demand_reference", self.demand),
            self.num_time_steps,
            "demand_reference",
        )
        self.support_wind_reference = {
            int(i): self._as_profile(
                self._wind_generator_config_value(cfg, "reference", int(i), self.pmax_list[int(i)]),
                self.num_time_steps,
                f"wind_generators[{self._generator_name(int(i))}].reference",
            )
            for i in self.wind_generator_ids
        }

        demand_min_default = self._feature_bound("demand", "min", 0.8 * self.demand)
        demand_max_default = self._feature_bound("demand", "max", 1.2 * self.demand)
        self.support_demand_min = float(cfg.get("demand_min", demand_min_default))
        self.support_demand_max = float(cfg.get("demand_max", demand_max_default))
        if self.support_demand_min > self.support_demand_max:
            raise ValueError("support_set_config demand_min cannot exceed demand_max")

        demand_range = self.support_demand_max - self.support_demand_min
        self.support_demand_ramp = float(cfg.get("demand_ramp", demand_range))
        self.support_demand_budget = float(cfg.get("demand_budget", self.num_time_steps * demand_range))

        wind_total_min_default = self._feature_bound("wind_forecast", "min", 0.5 * sum(self.pmax_list[i] for i in self.wind_generator_ids))
        wind_total_max_default = self._feature_bound("wind_forecast", "max", sum(self.pmax_list[i] for i in self.wind_generator_ids))
        wind_count = max(len(self.wind_generator_ids), 1)
        wind_min_default = wind_total_min_default / wind_count
        wind_max_default = wind_total_max_default / wind_count

        self.support_wind_min = {
            int(i): float(self._wind_generator_config_value(cfg, "min", int(i), wind_min_default))
            for i in self.wind_generator_ids
        }
        self.support_wind_max = {
            int(i): float(self._wind_generator_config_value(cfg, "max", int(i), min(self.pmax_list[int(i)], wind_max_default)))
            for i in self.wind_generator_ids
        }
        for i in self.wind_generator_ids:
            if self.support_wind_min[int(i)] > self.support_wind_max[int(i)]:
                raise ValueError(f"support_set_config wind_min cannot exceed wind_max for generator {int(i)}")

        total_wind_range = sum(
            self.support_wind_max[int(i)] - self.support_wind_min[int(i)]
            for i in self.wind_generator_ids
        )
        self.support_wind_ramp = float(cfg.get("wind_ramp", max(total_wind_range, 0.0)))
        self.support_wind_budget = float(cfg.get("wind_budget", self.num_time_steps * max(total_wind_range, 0.0)))

    def _build_variables(self) -> None:
        self._build_PoA_variables()
        self._build_equilibrium_variables()
        self._build_complementarity_equilibrium_variables()
        self._build_optimal_variables()
        self._build_complementarity_optimal_variables()      

    def _build_PoA_variables(self) -> None:
        self.model.D = Var(self.model.time_steps, within=NonNegativeReals)  # Demand at each time step
        self.model.P_max = Var(self.model.n_gen, self.model.time_steps, within=NonNegativeReals)  # Max production capacity for each generator and time step
        self.model.C_eq = Var(domain=Reals)
        self.model.C_opt = Var(domain=Reals)
        self.model.PoA = Var(domain=Reals)

        # Auxiliary variables for support set deviations and budgets
        self.model.D_abs_deviation = Var(self.model.time_steps, within=NonNegativeReals)
        self.model.P_max_abs_deviation = Var(self.model.n_gen, self.model.time_steps, within=NonNegativeReals)

    def _build_equilibrium_variables(self) -> None:
        self.model.P_eq = Var(self.model.n_gen, self.model.time_steps, within=NonNegativeReals)  # Production levels in equilibrium
        self.model.alpha = Var(self.model.n_gen, self.model.time_steps, domain=Reals)  # Policy bid variables
        self.model.lambda_var_eq = Var(self.model.time_steps, domain=Reals) # Market clearing price
        self.model.mu_upper_bound_eq = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)  # Upper bound duals
        self.model.mu_lower_bound_eq = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)  # Lower bound duals
        self.model.mu_ramp_up_eq = Var(self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)  # Ramp up duals
        self.model.mu_ramp_down_eq = Var(self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)  # Ramp down duals

    def _build_complementarity_equilibrium_variables(self) -> None:
        #Complementarity variables for the upper and lower bounds (one per generator per scenario)
        self.model.z_upper_bound_eq = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_lower_bound_eq = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_eq = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_eq = Var(self.model.n_gen, self.model.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        self.model.P_opt = Var(self.model.n_gen, self.model.time_steps, within=NonNegativeReals)  # Production levels in optimal solution
        self.model.lambda_var_opt = Var(self.model.time_steps, domain=Reals) # Market clearing price 
        self.model.mu_upper_bound_opt = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)  # Upper bound duals
        self.model.mu_lower_bound_opt = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)  # Lower bound duals
        self.model.mu_ramp_up_opt = Var(self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)  # Ramp up duals
        self.model.mu_ramp_down_opt = Var(self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)  # Ramp down duals

    def _build_complementarity_optimal_variables(self) -> None:
        #Complementarity variables for the upper and lower bounds (one per generator per scenario)
        self.model.z_upper_bound_opt = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_lower_bound_opt = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_opt = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_opt = Var(self.model.n_gen, self.model.time_steps, domain=Binary)

    def _build_objective(self) -> None:
        self.model.objective = Objective(expr = self.model.PoA, sense=maximize)

    def _build_constraints(self) -> None:
        self._build_support_set()
        self._build_policy_related_constraints()
        self._build_lower_level_equilibrium_constraints()
        self._build_lower_level_optimal_constraints()
        self._build_KKT_stationarity_equilibrium_constraints()
        self._build_KKT_stationarity_optimal_constraints()
        self._build_KKT_complementarity_equilibrium_constraints()
        self._build_KKT_complementarity_optimal_constraints()
        self._build_PoA_constraints()

    def _build_support_set(self) -> None:
        """DRO support set for demand and realized generator capacities."""
        if self.support_demand_ramp < 0:
            raise ValueError("support_set_config demand_ramp must be non-negative")
        if self.support_demand_budget < 0:
            raise ValueError("support_set_config demand_budget must be non-negative")
        if self.support_wind_ramp < 0:
            raise ValueError("support_set_config wind_ramp must be non-negative")
        if self.support_wind_budget < 0:
            raise ValueError("support_set_config wind_budget must be non-negative")

        def demand_lower_rule(m, t):
            return m.D[t] >= self.support_demand_min

        def demand_upper_rule(m, t):
            return m.D[t] <= self.support_demand_max

        def demand_ramp_up_rule(m, t):
            return m.D[t] - m.D[t - 1] <= self.support_demand_ramp

        def demand_ramp_down_rule(m, t):
            return m.D[t - 1] - m.D[t] <= self.support_demand_ramp

        # Budget constraints 
        def demand_abs_deviation_pos_rule(m, t):
            return m.D_abs_deviation[t] >= m.D[t] - self.support_demand_reference[int(t)]

        def demand_abs_deviation_neg_rule(m, t):
            return m.D_abs_deviation[t] >= self.support_demand_reference[int(t)] - m.D[t]

        def demand_budget_rule(m):
            return sum(m.D_abs_deviation[t] for t in m.time_steps) <= self.support_demand_budget

        self.model.demand_lower_bound_constraints = Constraint(self.model.time_steps, rule=demand_lower_rule)
        self.model.demand_upper_bound_constraints = Constraint(self.model.time_steps, rule=demand_upper_rule)
        self.model.demand_ramp_up_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_up_rule)
        self.model.demand_ramp_down_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_down_rule)
        self.model.demand_abs_deviation_pos_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_pos_rule)
        self.model.demand_abs_deviation_neg_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_neg_rule)
        self.model.demand_budget_constraint = Constraint(rule=demand_budget_rule)

        def conventional_capacity_rule(m, i, t):
            return m.P_max[i, t] == self.pmax_list[int(i)]

        def wind_capacity_lower_rule(m, i, t):
            return m.P_max[i, t] >= self.support_wind_min[int(i)]

        def wind_capacity_upper_rule(m, i, t):
            return m.P_max[i, t] <= self.support_wind_max[int(i)]

        def wind_ramp_up_rule(m, i, t):
            return m.P_max[i, t] - m.P_max[i, t - 1] <= self.support_wind_ramp

        def wind_ramp_down_rule(m, i, t):
            return m.P_max[i, t - 1] - m.P_max[i, t] <= self.support_wind_ramp

        def capacity_reference(i: int, t: int) -> float:
            if i in self.wind_generator_ids:
                return self.support_wind_reference[i][t]
            return self.pmax_list[i]

        def capacity_abs_deviation_pos_rule(m, i, t):
            return m.P_max_abs_deviation[i, t] >= m.P_max[i, t] - capacity_reference(int(i), int(t))

        def capacity_abs_deviation_neg_rule(m, i, t):
            return m.P_max_abs_deviation[i, t] >= capacity_reference(int(i), int(t)) - m.P_max[i, t]

        def wind_budget_rule(m):
            return (
                sum(m.P_max_abs_deviation[i, t] for i in m.n_gen for t in m.time_steps)
                <= self.support_wind_budget
            )

        self.model.conventional_capacity_constraints = Constraint(self.model.n_gen_conventional, self.model.time_steps, rule=conventional_capacity_rule)
        self.model.wind_capacity_lower_bound_constraints = Constraint(self.model.n_gen_wind, self.model.time_steps, rule=wind_capacity_lower_rule)
        self.model.wind_capacity_upper_bound_constraints = Constraint(self.model.n_gen_wind, self.model.time_steps, rule=wind_capacity_upper_rule)
        self.model.wind_ramp_up_constraints = Constraint(self.model.n_gen_wind, self.model.time_steps_minus_1, rule=wind_ramp_up_rule)
        self.model.wind_ramp_down_constraints = Constraint(self.model.n_gen_wind, self.model.time_steps_minus_1, rule=wind_ramp_down_rule)
        self.model.capacity_abs_deviation_pos_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=capacity_abs_deviation_pos_rule)
        self.model.capacity_abs_deviation_neg_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=capacity_abs_deviation_neg_rule)
        self.model.wind_budget_constraint = Constraint(rule=wind_budget_rule)

    def modify_decision_variables_from_normalization_stats(self, t: int, gen_idx: int) -> List[Any]:
        """
        Convert PoA decision variables into a normalized policy feature vector.

        This function applies the same min-max normalization used in policy training.
        """
        if self.feature_min is None or self.feature_max is None:
            self.get_feature_normalization_stats()

        if t not in self.model.time_steps:
            raise ValueError(f"Invalid time index {t}")
        if gen_idx not in self.model.n_gen:
            raise ValueError(f"Invalid generator index {gen_idx}")

        def _generator_to_player_id(local_gen_idx: int) -> Optional[int]:
            for player in self.players_config:
                if local_gen_idx in player.get("controlled_generators", []):
                    return int(player["id"])
            return None

        def _private_feature_value(feature_name: str, local_t: int, player_id: int):
            if feature_name == "player_cost":
                controlled = next(
                    p["controlled_generators"]
                    for p in self.players_config
                    if int(p["id"]) == int(player_id)
                )
                if not controlled:
                    return 0.0
                return float(sum(self.cost_vector[g] for g in controlled) / len(controlled))

            if feature_name == "player_capacity":
                controlled = next(
                    p["controlled_generators"]
                    for p in self.players_config
                    if int(p["id"]) == int(player_id)
                )
                return sum(self.model.P_max[g, local_t] for g in controlled)

            raise ValueError(f"Unsupported private feature '{feature_name}'")

        def _raw_feature_expression(feature_name: str, local_t: int, local_gen_idx: int):
            player_id = _generator_to_player_id(local_gen_idx)

            if feature_name == "bias":
                return 1.0
            if feature_name == "demand":
                return self.model.D[local_t]
            if feature_name == "demand_sq":
                return self.model.D[local_t] * self.model.D[local_t]
            if feature_name == "wind_forecast":
                return sum(self.model.P_max[i, local_t] for i in self.model.n_gen_wind)
            if feature_name == "total_capacity":
                return sum(self.model.P_max[i, local_t] for i in self.model.n_gen)

            if feature_name in {"player_cost", "player_capacity"}:
                if player_id is None:
                    return 0.0
                return _private_feature_value(feature_name, local_t, player_id)

            if feature_name == "scarcity_ratio":
                total_cap = sum(self.model.P_max[i, local_t] for i in self.model.n_gen)
                return self.model.D[local_t] / total_cap

            if feature_name == "residual_demand":
                wind = sum(self.model.P_max[i, local_t] for i in self.model.n_gen_wind)
                return self.model.D[local_t] - wind

            if feature_name in {
                "supply_intercept",
                "supply_slope",
                "supply_curve",
                "demand_tm1",
                "wind_tm1",
                "total_capacity_tm1",
            }:
                return 0.0

            raise ValueError(f"Unsupported feature '{feature_name}' in PoA feature conversion")

        player_id = _generator_to_player_id(gen_idx)
        player_private = self.player_private_min_max.get(player_id, None) if player_id is not None else None
        private_name_to_pos = {name: idx for idx, name in enumerate(self.private_feature_names)}

        normalized_features: List[Any] = []
        for f_idx, f_name in enumerate(self.feature_names):
            raw_expr = _raw_feature_expression(f_name, t, gen_idx)

            if (
                player_private is not None
                and f_name in private_name_to_pos
                and private_name_to_pos[f_name] < len(player_private["min"])
                and private_name_to_pos[f_name] < len(player_private["max"])
            ):
                pos = private_name_to_pos[f_name]
                f_min = float(player_private["min"][pos])
                f_max = float(player_private["max"][pos])
            else:
                f_min = float(self.feature_min[f_idx])
                f_max = float(self.feature_max[f_idx])

            denom = f_max - f_min
            if abs(denom) <= self.normalization_epsilon:
                normalized_features.append(1.0)
            else:
                normalized_features.append((raw_expr - f_min) / denom)

        return normalized_features

    def _build_policy_related_constraints(self) -> None:
        """Apply policy constraints in the same structure as the MPEC model."""
        if not self.policy_by_generator:
            # PoA can still build without policy constraints while under development.
            return

        if not hasattr(self.model, "alpha"):
            self.model.alpha = Var(self.model.n_gen, self.model.time_steps, domain=Reals)

        if self.policy_type == "linear":
            def policy_rule(m, i, t):
                i_int = int(i)
                if i_int not in self.policy_by_generator:
                    return Constraint.Skip

                phi = self.modify_decision_variables_from_normalization_stats(t=int(t), gen_idx=i_int)
                theta = self.policy_by_generator[i_int]
                if len(theta) != len(phi):
                    raise ValueError(
                        f"theta length {len(theta)} does not match feature length {len(phi)}"
                    )
                return m.alpha[i, t] == sum(float(theta[f]) * phi[f] for f in range(len(phi)))

            self.model.policy_constraint = Constraint(self.model.n_gen, self.model.time_steps, rule=policy_rule)
            return

        if self.policy_type != "nn":
            raise ValueError(f"Unsupported loaded policy type '{self.policy_type}'")

        nn_params: Dict[int, Dict[str, Any]] = {}
        n_nodes: Optional[int] = None
        n_features_expected = len(self.feature_names)

        for gen_idx, policy_entry in self.policy_by_generator.items():
            if "output_bias" not in policy_entry:
                raise ValueError(f"Missing NN weight key 'output_bias' for generator {gen_idx}")

            gamma_raw = np.asarray(policy_entry.get("gamma", []), dtype=np.float64)
            Theta_raw = np.asarray(policy_entry.get("Theta", []), dtype=np.float64)
            Gamma_raw = np.asarray(policy_entry.get("Gamma", []), dtype=np.float64)

            # Canonical NN schema: gamma (nodes, features), Theta (nodes,), Gamma (nodes,)
            if gamma_raw.ndim == 2 and Theta_raw.ndim == 1 and Gamma_raw.ndim == 1:
                gamma_arr = gamma_raw
                gamma_bias_arr = Theta_raw
                output_weight_arr = Gamma_raw
            # Alternate exported schema handled in PoA parser compatibility.
            elif Gamma_raw.ndim == 2 and gamma_raw.ndim == 1 and Theta_raw.ndim == 1:
                gamma_arr = Gamma_raw
                gamma_bias_arr = gamma_raw
                output_weight_arr = Theta_raw
            else:
                raise ValueError(
                    "Unsupported NN weight schema. Expected either canonical "
                    "(gamma 2D, Theta 1D, Gamma 1D) or alternate "
                    "(Gamma 2D, gamma 1D, Theta 1D)."
                )

            if gamma_arr.shape[1] != n_features_expected:
                raise ValueError(
                    f"Generator {gen_idx}: NN feature dimension {gamma_arr.shape[1]} does not match expected {n_features_expected}"
                )

            local_nodes = int(gamma_arr.shape[0])
            if gamma_bias_arr.shape[0] != local_nodes:
                raise ValueError(
                    f"Generator {gen_idx}: Theta length {gamma_bias_arr.shape[0]} does not match number of hidden nodes {local_nodes}"
                )
            if output_weight_arr.shape[0] != local_nodes:
                raise ValueError(
                    f"Generator {gen_idx}: Gamma length {output_weight_arr.shape[0]} does not match number of hidden nodes {local_nodes}"
                )

            if n_nodes is None:
                n_nodes = local_nodes
            elif local_nodes != n_nodes:
                raise ValueError(
                    f"Inconsistent NN hidden-node counts across generators: expected {n_nodes}, got {local_nodes} for generator {gen_idx}"
                )

            nn_params[int(gen_idx)] = {
                "gamma": gamma_arr,
                "Theta": gamma_bias_arr,
                "Gamma": output_weight_arr,
                "output_bias": float(policy_entry["output_bias"]),
            }

        if n_nodes is None:
            return

        self.model.NN_nodes = Set(initialize=range(n_nodes))
        self.model.z_NN = Var(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, domain=Reals)
        self.model.y_NN = Var(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, domain=NonNegativeReals)
        self.model.delta_NN = Var(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, domain=Binary)

        BigM = float(self.big_m_complementarity)

        def translation_rule(m, i, t, n):
            i_int = int(i)
            n_int = int(n)
            if i_int not in nn_params:
                return Constraint.Skip

            phi = self.modify_decision_variables_from_normalization_stats(t=int(t), gen_idx=i_int)
            gamma = nn_params[i_int]["gamma"]
            Theta = nn_params[i_int]["Theta"]
            return m.z_NN[i, t, n] == sum(float(gamma[n_int, f]) * phi[f] for f in range(len(phi))) + float(Theta[n_int])

        def relu_lb_rule(m, i, t, n):
            if int(i) not in nn_params:
                return Constraint.Skip
            return m.z_NN[i, t, n] <= m.y_NN[i, t, n]

        def relu_ub_rule_1(m, i, t, n):
            if int(i) not in nn_params:
                return Constraint.Skip
            return m.y_NN[i, t, n] <= m.z_NN[i, t, n] + BigM * (1 - m.delta_NN[i, t, n])

        def relu_ub_rule_2(m, i, t, n):
            if int(i) not in nn_params:
                return Constraint.Skip
            return m.y_NN[i, t, n] <= BigM * m.delta_NN[i, t, n]

        def alpha_rule(m, i, t):
            i_int = int(i)
            if i_int not in nn_params:
                return Constraint.Skip
            Gamma = nn_params[i_int]["Gamma"]
            output_bias = nn_params[i_int]["output_bias"]
            return m.alpha[i, t] == sum(float(Gamma[int(n)]) * m.y_NN[i, t, n] for n in m.NN_nodes) + output_bias

        self.model.translation_constraints = Constraint(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=translation_rule)
        self.model.relu_lb_constraints = Constraint(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=relu_lb_rule)
        self.model.relu_ub_constraints = Constraint(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=relu_ub_rule_1)
        self.model.relu_ub_constraints_2 = Constraint(self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=relu_ub_rule_2)
        self.model.alpha_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=alpha_rule)

    def _build_lower_level_equilibrium_constraints(self):
        def power_balance_eq_rule(m, t):
            return sum(m.P_eq[i, t] for i in m.n_gen) - m.D[t] == 0
        
        def generation_upper_eq_rule(m, i, t):
            return m.P_eq[i, t] - m.P_max[i, t] <= 0 
    
        def generation_lower_eq_rule(m, i, t):
            return -m.P_eq[i, t] + 0 <= 0
        
        def ramp_up_eq_rule(m, i, t):
            return m.P_eq[i, t] - m.P_eq[i, t-1] - self.ramp_vector_up[i] <= 0
        
        def ramp_up_initial_eq_rule(m, i):
            return m.P_eq[i, 0] - self.P_init[i] - self.ramp_vector_up[i] <= 0
        
        def ramp_down_eq_rule(m, i, t):
            return -m.P_eq[i, t] + m.P_eq[i, t-1] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_eq_rule(m, i):
            return -m.P_eq[i, 0] + self.P_init[i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint_eq = Constraint(self.model.time_steps, rule=power_balance_eq_rule)
        self.model.generation_upper_bound_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=generation_upper_eq_rule)
        self.model.generation_lower_bound_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=generation_lower_eq_rule)
        self.model.ramp_up_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_eq_rule)
        self.model.ramp_down_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_eq_rule)
        self.model.ramp_up_initial_feasibility_constraints_eq = Constraint(self.model.n_gen, rule=ramp_up_initial_eq_rule)
        self.model.ramp_down_initial_feasibility_constraints_eq = Constraint(self.model.n_gen, rule=ramp_down_initial_eq_rule)

    def _build_lower_level_optimal_constraints(self):
        def power_balance_opt_rule(m, t):
            return sum(m.P_opt[i, t] for i in m.n_gen) - m.D[t] == 0
        
        def generation_upper_opt_rule(m, i, t):
            return m.P_opt[i, t] - m.P_max[i, t] <= 0 
    
        def generation_lower_opt_rule(m, i, t):
            return -m.P_opt[i, t] + 0 <= 0
        
        def ramp_up_opt_rule(m, i, t):
            return m.P_opt[i, t] - m.P_opt[i, t-1] - self.ramp_vector_up[i] <= 0
        
        def ramp_up_initial_opt_rule(m, i):
            return m.P_opt[i, 0] - self.P_init[i] - self.ramp_vector_up[i] <= 0
        
        def ramp_down_opt_rule(m, i, t):
            return -m.P_opt[i, t] + m.P_opt[i, t-1] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_opt_rule(m, i):
            return -m.P_opt[i, 0] + self.P_init[i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint_opt = Constraint(self.model.time_steps, rule=power_balance_opt_rule)
        self.model.generation_upper_bound_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=generation_upper_opt_rule)
        self.model.generation_lower_bound_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=generation_lower_opt_rule)
        self.model.ramp_up_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_opt_rule)
        self.model.ramp_down_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_opt_rule)
        self.model.ramp_up_initial_feasibility_constraints_opt = Constraint(self.model.n_gen, rule=ramp_up_initial_opt_rule)
        self.model.ramp_down_initial_feasibility_constraints_opt = Constraint(self.model.n_gen, rule=ramp_down_initial_opt_rule)

    def _build_KKT_stationarity_equilibrium_constraints(self):
        def stationarity_eq_rule(m, i, t):
            return m.alpha[i, t] - m.lambda_var_eq[t] + m.mu_upper_bound_eq[i, t] - m.mu_lower_bound_eq[i, t] + m.mu_ramp_up_eq[i, t] - m.mu_ramp_up_eq[i, t+1] - m.mu_ramp_down_eq[i, t] + m.mu_ramp_down_eq[i, t+1] == 0

        def final_ramp_up_dual_eq_rule(m, i):
            return m.mu_ramp_up_eq[i, self.num_time_steps] == 0

        def final_ramp_down_dual_eq_rule(m, i):
            return m.mu_ramp_down_eq[i, self.num_time_steps] == 0

        self.model.stationarity_constraint_strategic_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=stationarity_eq_rule)
        self.model.final_ramp_up_dual_constraint_eq = Constraint(self.model.n_gen, rule=final_ramp_up_dual_eq_rule)
        self.model.final_ramp_down_dual_constraint_eq = Constraint(self.model.n_gen, rule=final_ramp_down_dual_eq_rule)
    
    def _build_KKT_stationarity_optimal_constraints(self):
        def stationarity_opt_rule(m, i, t):
            return self.cost_vector[i] - m.lambda_var_opt[t] + m.mu_upper_bound_opt[i, t] - m.mu_lower_bound_opt[i, t] + m.mu_ramp_up_opt[i, t] - m.mu_ramp_up_opt[i, t+1] - m.mu_ramp_down_opt[i, t] + m.mu_ramp_down_opt[i, t+1] == 0

        def final_ramp_up_dual_opt_rule(m, i):
            return m.mu_ramp_up_opt[i, self.num_time_steps] == 0

        def final_ramp_down_dual_opt_rule(m, i):
            return m.mu_ramp_down_opt[i, self.num_time_steps] == 0

        self.model.stationarity_constraint_strategic_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=stationarity_opt_rule)
        self.model.final_ramp_up_dual_constraint_opt = Constraint(self.model.n_gen, rule=final_ramp_up_dual_opt_rule)
        self.model.final_ramp_down_dual_constraint_opt = Constraint(self.model.n_gen, rule=final_ramp_down_dual_opt_rule)

    def _build_KKT_complementarity_equilibrium_constraints(self):
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_eq_rule(m, i, t):
            return -BigM * (1 - m.z_upper_bound_eq[i, t]) <= m.P_eq[i, t] - m.P_max[i, t]  

        def upper_bound_complementarity_dual_eq_rule(m, i, t):
            return m.mu_upper_bound_eq[i, t] <= BigM * m.z_upper_bound_eq[i, t] 
        
        def lower_bound_complementarity_eq_rule(m, i, t):
            return -BigM * (1 - m.z_lower_bound_eq[i, t]) <= -m.P_eq[i, t] + 0

        def lower_bound_complementarity_dual_eq_rule(m, i, t):
            return m.mu_lower_bound_eq[i, t] <= BigM * m.z_lower_bound_eq[i, t] 

        def ramp_up_complementarity_eq_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_up_eq[i, t]) <= m.P_eq[i, t] - m.P_eq[i, t-1] - self.ramp_vector_up[i]
        
        def ramp_up_complementarity_initial_eq_rule(m, i):
            return -BigM * (1 - m.z_ramp_up_eq[i, 0]) <= m.P_eq[i, 0] - self.P_init[i] - self.ramp_vector_up[i]

        def ramp_up_complementarity_dual_eq_rule(m, i, t):
            return m.mu_ramp_up_eq[i, t] <= BigM * m.z_ramp_up_eq[i, t]

        def ramp_down_complementarity_eq_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_down_eq[i, t]) <= - m.P_eq[i, t] + m.P_eq[i, t-1] - self.ramp_vector_down[i]

        def ramp_down_complementarity_initial_eq_rule(m, i):
            return -BigM * (1 - m.z_ramp_down_eq[i, 0]) <= - m.P_eq[i, 0] + self.P_init[i] - self.ramp_vector_down[i]

        def ramp_down_complementarity_dual_eq_rule(m, i, t):
            return m.mu_ramp_down_eq[i, t] <= BigM * m.z_ramp_down_eq[i, t]

        self.model.upper_bound_complementarity_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_eq_rule)
        self.model.upper_bound_complementarity_constraints_dual_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_dual_eq_rule)
        self.model.lower_bound_complementarity_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_eq_rule)
        self.model.lower_bound_complementarity_constraints_dual_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_dual_eq_rule)

        self.model.ramp_up_complementarity_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_complementarity_eq_rule)
        self.model.ramp_up_complementarity_initial_constraints_eq = Constraint(self.model.n_gen, rule=ramp_up_complementarity_initial_eq_rule)
        self.model.ramp_up_complementarity_constraints_dual_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=ramp_up_complementarity_dual_eq_rule)

        self.model.ramp_down_complementarity_constraints_eq = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_complementarity_eq_rule)
        self.model.ramp_down_complementarity_initial_constraints_eq = Constraint(self.model.n_gen, rule=ramp_down_complementarity_initial_eq_rule)
        self.model.ramp_down_complementarity_constraints_dual_eq = Constraint(self.model.n_gen, self.model.time_steps, rule=ramp_down_complementarity_dual_eq_rule)

    def _build_KKT_complementarity_optimal_constraints(self):
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_opt_rule(m, i, t):
            return -BigM * (1 - m.z_upper_bound_opt[i, t]) <= m.P_opt[i, t] - m.P_max[i, t]  

        def upper_bound_complementarity_dual_opt_rule(m, i, t):
            return m.mu_upper_bound_opt[i, t] <= BigM * m.z_upper_bound_opt[i, t] 
        
        def lower_bound_complementarity_opt_rule(m, i, t):
            return -BigM * (1 - m.z_lower_bound_opt[i, t]) <= -m.P_opt[i, t] + 0

        def lower_bound_complementarity_dual_opt_rule(m, i, t):
            return m.mu_lower_bound_opt[i, t] <= BigM * m.z_lower_bound_opt[i, t] 

        def ramp_up_complementarity_opt_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_up_opt[i, t]) <= m.P_opt[i, t] - m.P_opt[i, t-1] - self.ramp_vector_up[i]
        
        def ramp_up_complementarity_initial_opt_rule(m, i):
            return -BigM * (1 - m.z_ramp_up_opt[i, 0]) <= m.P_opt[i, 0] - self.P_init[i] - self.ramp_vector_up[i]

        def ramp_up_complementarity_dual_opt_rule(m, i, t):
            return m.mu_ramp_up_opt[i, t] <= BigM * m.z_ramp_up_opt[i, t]

        def ramp_down_complementarity_opt_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_down_opt[i, t]) <= - m.P_opt[i, t] + m.P_opt[i, t-1] - self.ramp_vector_down[i]

        def ramp_down_complementarity_initial_opt_rule(m, i):
            return -BigM * (1 - m.z_ramp_down_opt[i, 0]) <= - m.P_opt[i, 0] + self.P_init[i] - self.ramp_vector_down[i]

        def ramp_down_complementarity_dual_opt_rule(m, i, t):
            return m.mu_ramp_down_opt[i, t] <= BigM * m.z_ramp_down_opt[i, t]

        self.model.upper_bound_complementarity_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_opt_rule)
        self.model.upper_bound_complementarity_constraints_dual_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_dual_opt_rule)
        self.model.lower_bound_complementarity_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_opt_rule)
        self.model.lower_bound_complementarity_constraints_dual_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_dual_opt_rule)

        self.model.ramp_up_complementarity_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_complementarity_opt_rule)
        self.model.ramp_up_complementarity_initial_constraints_opt = Constraint(self.model.n_gen, rule=ramp_up_complementarity_initial_opt_rule)
        self.model.ramp_up_complementarity_constraints_dual_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=ramp_up_complementarity_dual_opt_rule)

        self.model.ramp_down_complementarity_constraints_opt = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_complementarity_opt_rule)
        self.model.ramp_down_complementarity_initial_constraints_opt = Constraint(self.model.n_gen, rule=ramp_down_complementarity_initial_opt_rule)
        self.model.ramp_down_complementarity_constraints_dual_opt = Constraint(self.model.n_gen, self.model.time_steps, rule=ramp_down_complementarity_dual_opt_rule)

    def _build_PoA_constraints(self) -> None:
        def cost_eq_rule(m):
            return m.C_eq == sum(self.cost_vector[i] * m.P_eq[i, t] for i in m.n_gen for t in m.time_steps)

        def cost_opt_rule(m):
            return m.C_opt == sum(self.cost_vector[i] * m.P_opt[i, t] for i in m.n_gen for t in m.time_steps)
        
        def PoA_rule(m):
            return m.C_eq - m.C_opt == m.PoA

        self.model.cost_definition_eq = Constraint(rule=cost_eq_rule)
        self.model.cost_definition_opt = Constraint(rule=cost_opt_rule)
        self.model.PoA_constraint = Constraint(rule=PoA_rule)

    def solve(self) -> None:
        """
        Solve the optimization model.
        """

        # Create solver
        solver = SolverFactory("gurobi")
         
        results = solver.solve(self.model, tee=True)
        self.solver_results = results

        # Check solver status
        if not (results.solver.status == 'ok') and not (results.solver.termination_condition == 'optimal'):
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)
            raise ValueError("Solver did not find an optimal solution")
        else:
            print("Solver found optimal solution with PoA =", self.model.C_eq.value / self.model.C_opt.value)

    def _safe_value(self, expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        if raw_value is None:
            return None
        return float(raw_value)

    def _series_values(self, var: Any, first_index: Optional[int] = None) -> List[Optional[float]]:
        values: List[Optional[float]] = []
        for t in self.model.time_steps:
            if first_index is None:
                values.append(self._safe_value(var[t]))
            else:
                values.append(self._safe_value(var[first_index, t]))
        return values

    def extract_results(self) -> Dict[str, Any]:
        """Extract solved PoA trajectories in a JSON-friendly structure."""
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call _build_model() first.")

        generator_names = [self._generator_name(i) for i in range(self.num_generators)]
        time_steps = [int(t) for t in self.model.time_steps]

        demand_profile = self._series_values(self.model.D)
        eq_price_profile = self._series_values(self.model.lambda_var_eq)
        opt_price_profile = self._series_values(self.model.lambda_var_opt)

        generators: Dict[str, Dict[str, Any]] = {}
        for i in self.model.n_gen:
            gen_idx = int(i)
            gen_name = self._generator_name(gen_idx)
            generators[gen_name] = {
                "index": gen_idx,
                "cost": float(self.cost_vector[gen_idx]),
                "is_wind": gen_idx in self.wind_generator_ids,
                "capacity": self._series_values(self.model.P_max, gen_idx),
                "policy_bid": self._series_values(self.model.alpha, gen_idx),
                "equilibrium_dispatch": self._series_values(self.model.P_eq, gen_idx),
                "optimal_dispatch": self._series_values(self.model.P_opt, gen_idx),
            }

        support_set = {
            "demand": {
                "reference": list(self.support_demand_reference),
                "min": float(self.support_demand_min),
                "max": float(self.support_demand_max),
                "ramp": float(self.support_demand_ramp),
                "budget": float(self.support_demand_budget),
            },
            "wind": {
                self._generator_name(i): {
                    "reference": list(self.support_wind_reference[int(i)]),
                    "min": float(self.support_wind_min[int(i)]),
                    "max": float(self.support_wind_max[int(i)]),
                }
                for i in self.wind_generator_ids
            },
            "wind_ramp": float(self.support_wind_ramp),
            "wind_budget": float(self.support_wind_budget),
        }

        objective = {
            "PoA": self._safe_value(self.model.PoA),
            "C_eq": self._safe_value(self.model.C_eq),
            "C_opt": self._safe_value(self.model.C_opt),
        }
        if objective["C_eq"] is not None and objective["C_opt"] not in (None, 0.0):
            objective["C_eq_over_C_opt"] = objective["C_eq"] / objective["C_opt"]

        solver_summary: Dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(self.solver_results.solver.termination_condition),
            }

        return {
            "reference_case": self.reference_case,
            "num_time_steps": self.num_time_steps,
            "time_steps": time_steps,
            "generator_names": generator_names,
            "generator_costs": [float(v) for v in self.cost_vector],
            "wind_generator_names": [self._generator_name(i) for i in self.wind_generator_ids],
            "conventional_generator_names": [self._generator_name(i) for i in self.conventional_generator_ids],
            "features": list(self.feature_names),
            "policy_type": self.policy_type,
            "policy_metadata": dict(self.policy_metadata),
            "objective": objective,
            "solver": solver_summary,
            "support_set": support_set,
            "demand_profile": demand_profile,
            "equilibrium_price_profile": eq_price_profile,
            "optimal_price_profile": opt_price_profile,
            "generators": generators,
        }

    def save_results(self, output_path: str = "results/poa_optimization_results.json") -> Path:
        """Save extracted PoA results to JSON."""
        results = self.extract_results()
        path = Path(output_path)
        if path.suffix and path.suffix.lower() != ".json":
            raise ValueError("output_path must end with .json or have no extension")
        if not path.suffix:
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2)
        return path

    def plot_generator_result_trajectories(
        self,
        output_dir: str = "results_viz/figures/poa_optimization",
        show: bool = False,
    ) -> None:
        """Plot PoA trajectories in a gradient-policy-style stacked figure."""
        results = self.extract_results()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        time_axis = np.asarray(results["time_steps"], dtype=int)
        demand = np.asarray(results["demand_profile"], dtype=float)
        eq_price = np.asarray(results["equilibrium_price_profile"], dtype=float)
        opt_price = np.asarray(results["optimal_price_profile"], dtype=float)

        print("\n=== PoA Optimization Visualization ===")
        print(f"Reference case: {results['reference_case']}")
        print(f"PoA: {results['objective'].get('PoA')}")
        print(f"Generators: {', '.join(results['generator_names'])}")

        for gen_name, gen_results in results["generators"].items():
            capacity = np.asarray(gen_results["capacity"], dtype=float)
            bid = np.asarray(gen_results["policy_bid"], dtype=float)
            cost_bid = np.full_like(time_axis, float(gen_results["cost"]), dtype=float)
            eq_dispatch = np.asarray(gen_results["equilibrium_dispatch"], dtype=float)
            opt_dispatch = np.asarray(gen_results["optimal_dispatch"], dtype=float)

            fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
            fig.suptitle(f"PoA Worst-Case Trajectory - {gen_name}", fontsize=14)

            axes[0].plot(time_axis, demand, color="tab:blue", marker="o", linewidth=2.0)
            axes[0].set_ylabel("Demand")
            axes[0].set_title("Demand Trajectory")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(time_axis, capacity, color="tab:green", marker="o", linewidth=2.0)
            axes[1].set_ylabel("MW")
            axes[1].set_title(f"{gen_name} Available Capacity")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(time_axis, bid, color="tab:orange", marker="o", linewidth=2.0, label="Policy bid")
            axes[2].plot(
                time_axis,
                eq_price,
                color="tab:red",
                marker="s",
                linewidth=1.8,
                linestyle="--",
                label="Equilibrium price",
            )
            axes[2].set_ylabel("Price / Bid")
            axes[2].set_title(f"{gen_name} Equilibrium Bid and Market Clearing Price")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="best", fontsize=9)

            axes[3].plot(
                time_axis,
                cost_bid,
                color="tab:orange",
                marker="o",
                linewidth=2.0,
                label="Optimal bid (cost)",
            )
            axes[3].plot(
                time_axis,
                opt_price,
                color="tab:red",
                marker="s",
                linewidth=1.8,
                linestyle="--",
                label="Optimal price",
            )
            axes[3].set_ylabel("Price / Bid")
            axes[3].set_title(f"{gen_name} Optimal Bid and Market Clearing Price")
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(loc="best", fontsize=9)

            axes[4].plot(
                time_axis,
                eq_dispatch,
                color="tab:purple",
                marker="o",
                linewidth=2.0,
                label="Equilibrium dispatch",
            )
            axes[4].plot(
                time_axis,
                opt_dispatch,
                color="tab:gray",
                marker="s",
                linewidth=1.8,
                linestyle="--",
                label="Optimal dispatch",
            )
            axes[4].set_ylabel("MW")
            axes[4].set_xlabel("Time step")
            axes[4].set_title(f"{gen_name} Dispatch")
            axes[4].grid(True, alpha=0.3)
            axes[4].legend(loc="best", fontsize=9)

            for ax in axes:
                ax.set_xticks(time_axis)

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            out = output_path / f"poa_{gen_name}.png"
            fig.savefig(out, dpi=160, bbox_inches="tight")
            print(f"[saved] {out}")
            if show:
                plt.show()
            plt.close(fig)

    def plot_demand_capacity_trajectory(
        self,
        save_path: str = "results/poa_demand_capacity_trajectory.png",
        show: bool = True,
    ) -> None:
        """Plot solved trajectories for demand and generator capacities over time."""
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call _build_model() first.")

        time_points = list(self.model.time_steps)
        if not time_points:
            raise ValueError("No time steps found in model.")

        demand_vals = [value(self.model.D[t], exception=False) for t in time_points]
        demand_vals = [np.nan if v is None else float(v) for v in demand_vals]

        capacity_by_gen: Dict[int, List[float]] = {}
        for i in self.model.n_gen:
            vals = [value(self.model.P_max[i, t], exception=False) for t in time_points]
            capacity_by_gen[int(i)] = [np.nan if v is None else float(v) for v in vals]

        total_capacity = []
        for t_idx in range(len(time_points)):
            total_capacity.append(
                float(
                    np.nansum(
                        [capacity_by_gen[int(i)][t_idx] for i in self.model.n_gen]
                    )
                )
            )

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(time_points, demand_vals, color="black", linewidth=2.5, label="Demand")
        ax.plot(
            time_points,
            total_capacity,
            color="#1f77b4",
            linewidth=2.2,
            linestyle="--",
            label="Total Capacity",
        )

        for i in self.model.n_gen:
            ax.plot(
                time_points,
                capacity_by_gen[int(i)],
                linewidth=1.4,
                alpha=0.9,
                label=f"Generator {int(i)} Capacity",
            )

        ax.set_title("Demand and Generator Capacity Trajectories")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2, fontsize=9)
        plt.tight_layout()

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    # Example usage
    example_reference_case = "test_case1"
    num_generators, *_ = load_setup_data(example_reference_case)
    P_init = np.ones(int(num_generators)) * 25

    policy_results_path = "results/gradient_policy_training_nn_results.json"
    with Path(policy_results_path).open("r", encoding="utf-8") as file_handle:
        policy_results = json.load(file_handle)

    num_time_steps = int(policy_results.get("num_time_steps", 8))

    num_time_steps = 24
    support_set_config = PoAOptimization.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case1_base",
    )

    poa_opt = PoAOptimization(
        P_init,
        num_time_steps=num_time_steps,
        reference_case="test_case1",
        feature_normalizer_stats_path="results/feature_normalizer_stats_gradient.json",
        policy_data=policy_results,
        support_set_config=support_set_config,
        big_m_complementarity=1e8,
    )
    poa_opt._build_model()
    poa_opt.solve()
    saved_results_path = poa_opt.save_results("results/poa_optimization_results.json")
    print(f"Saved PoA results to {saved_results_path}")
    poa_opt.plot_generator_result_trajectories(
        output_dir="results_viz/figures/poa_optimization",
        show=False,
    )
    # poa_opt.plot_demand_capacity_trajectory(show=True)

    stop = True
