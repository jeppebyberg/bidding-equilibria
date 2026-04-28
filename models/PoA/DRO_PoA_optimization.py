import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import pandas as pd
import json
import ast
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config.intertemporal.utils.cases_utils import load_setup_data
from config.intertemporal.scenarios.scenario_generator_2 import ScenarioManagerV2

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
            eta: float = 0.0,
            empirical_scenario: Optional[Any] = None,
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

        # Policy payload loaded from BR results.
        self.policy_type: Optional[str] = None  # "linear" or "nn"
        self.policy_by_generator: Dict[int, Any] = {}
        self.support_set_config = support_set_config or {}
        self.eta = float(eta)
        if self.eta < 0:
            raise ValueError("eta must be non-negative")

        # Load setup directly from config/intertemporal/reference_cases.yaml.
        # This keeps PoA aligned with the same reference-case source as the BR scripts.
        self._load_reference_case_setup()

        self.get_feature_normalization_stats()
        self._configure_support_set_parameters()
        self._configure_empirical_scenario(empirical_scenario)

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
        self.model.scenarios = Set(initialize=range(self.num_empirical_scenarios))

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

        policy_type = str(extracted.get("policy_type", "")).lower()
        if policy_type not in {"linear", "nn"}:
            raise ValueError(f"Unsupported policy_type '{policy_type}'. Expected 'linear' or 'nn'.")

        raw_map = extracted.get("policy_by_generator", {})
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("policy_by_generator must be a non-empty dict")

        self.policy_type = policy_type
        self.policy_by_generator = {int(k): v for k, v in raw_map.items()}

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
        if isinstance(value, str):
            value = ast.literal_eval(value)

        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            profile = [float(v) for v in value]
        else:
            profile = [float(value)] * horizon

        if len(profile) != horizon:
            raise ValueError(f"{name} must have length {horizon}, got {len(profile)}")

        return profile

    def _generator_name(self, gen_idx: int) -> str:
        generator = self.generators[int(gen_idx)]
        if isinstance(generator, dict):
            return str(generator.get("name", f"G{gen_idx}"))
        return str(generator)

    def _feature_bound(self, feature_name: str, bound: str, default: float) -> float:
        """Read min/max feature-normalizer bounds when available."""
        if feature_name not in self.feature_names:
            return float(default)

        feature_idx = self.feature_names.index(feature_name)
        values = self.feature_min if bound == "min" else self.feature_max
        if values is None or feature_idx >= len(values):
            return float(default)

        return float(values[feature_idx])

    @staticmethod
    def _per_generator_config_value(config_value: Any, gen_idx: int, default: Any) -> Any:
        """Read either a scalar/list config value or a per-generator mapping."""
        if isinstance(config_value, dict):
            return config_value.get(int(gen_idx), config_value.get(str(gen_idx), default))

        if config_value is None:
            return default

        return config_value

    def _configure_support_set_parameters(self) -> None:
        """Set support-set parameters, using config overrides when provided."""
        cfg = self.support_set_config

        self.support_demand_reference = self._as_profile(
            cfg.get("demand_reference", self.demand),
            self.num_time_steps,
            "demand_reference",
        )
        wind_reference_cfg = cfg.get("wind_reference")
        self.support_wind_reference = {
            int(i): self._as_profile(
                self._per_generator_config_value(wind_reference_cfg, int(i), self.pmax_list[int(i)]),
                self.num_time_steps,
                f"wind_reference[{int(i)}]",
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

        wind_min_cfg = cfg.get("wind_min")
        wind_max_cfg = cfg.get("wind_max")
        self.support_wind_min = {
            int(i): float(self._per_generator_config_value(wind_min_cfg, int(i), wind_min_default))
            for i in self.wind_generator_ids
        }
        self.support_wind_max = {
            int(i): float(self._per_generator_config_value(wind_max_cfg, int(i), min(self.pmax_list[int(i)], wind_max_default)))
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

    def _configure_empirical_scenario(self, empirical_scenario: Optional[Any]) -> None:
        """Store empirical states s^(k) used in the Wasserstein penalty."""
        if empirical_scenario is None:
            empirical_rows = [
                {
                    "regime": "reference",
                    "scenario_id": None,
                    "demand_profile": list(self.support_demand_reference),
                }
            ]
        elif isinstance(empirical_scenario, pd.DataFrame):
            empirical_rows = [row.to_dict() for _, row in empirical_scenario.iterrows()]
        elif isinstance(empirical_scenario, dict):
            empirical_rows = [empirical_scenario]
        else:
            empirical_rows = list(empirical_scenario)

        if not empirical_rows:
            raise ValueError("At least one empirical scenario is required")

        self.num_empirical_scenarios = len(empirical_rows)
        self.empirical_regime = str(empirical_rows[0].get("regime", "unknown"))
        self.empirical_scenario_ids = [
            None if row.get("scenario_id") is None else int(row.get("scenario_id"))
            for row in empirical_rows
        ]
        self.empirical_scenario_id = self.empirical_scenario_ids[0]

        self.empirical_demand_profiles: Dict[int, List[float]] = {}
        self.empirical_capacity_profiles: Dict[int, Dict[int, List[float]]] = {}

        for k, row in enumerate(empirical_rows):
            self.empirical_demand_profiles[k] = self._as_profile(
                row["demand_profile"],
                self.num_time_steps,
                f"empirical_scenario[{k}]['demand_profile']",
            )

            capacity_profiles: Dict[int, List[float]] = {}
            for i in range(self.num_generators):
                gen_name = self._generator_name(i)
                profile_key = f"{gen_name}_profile"
                if i in self.wind_generator_ids and profile_key in row:
                    capacity_profiles[int(i)] = self._as_profile(
                        row[profile_key],
                        self.num_time_steps,
                        f"empirical_scenario[{k}]['{profile_key}']",
                    )
                else:
                    capacity_profiles[int(i)] = [float(self.pmax_list[int(i)])] * self.num_time_steps

            self.empirical_capacity_profiles[k] = capacity_profiles

    def _build_variables(self) -> None:
        # Decision variables needed for PoA feature conversion.
        self._build_PoA_variables()
        # Additional blocks can be enabled as PoA formulation is completed.
        self._build_equilibrium_variables()
        self._build_complementarity_equilibrium_variables()
        self._build_optimal_variables()
        self._build_complementarity_optimal_variables()      

    def _build_PoA_variables(self) -> None:
        self.model.D = Var(self.model.scenarios, self.model.time_steps, within=NonNegativeReals)
        self.model.P_max = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, within=NonNegativeReals)
        self.model.C_eq = Var(self.model.scenarios, domain=Reals)
        self.model.C_opt = Var(self.model.scenarios, domain=Reals)
        self.model.PoA = Var(self.model.scenarios, domain=Reals)
        self.model.wasserstein_distance = Var(self.model.scenarios, within=NonNegativeReals)

        # Auxiliary variables for support set deviations and budgets
        self.model.D_abs_deviation = Var(self.model.scenarios, self.model.time_steps, within=NonNegativeReals)
        self.model.P_max_abs_deviation = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, within=NonNegativeReals)
        self.model.D_transport_abs_deviation = Var(self.model.scenarios, self.model.time_steps, within=NonNegativeReals)
        self.model.P_max_transport_abs_deviation = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, within=NonNegativeReals)

    def _build_equilibrium_variables(self) -> None:
        self.model.P_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, within=NonNegativeReals)
        self.model.alpha = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Reals)
        self.model.lambda_var_eq = Var(self.model.scenarios, self.model.time_steps, domain=Reals)
        self.model.mu_upper_bound_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)
        self.model.mu_lower_bound_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)
        self.model.mu_ramp_up_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)
        self.model.mu_ramp_down_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)

    def _build_complementarity_equilibrium_variables(self) -> None:
        #Complementarity variables for the upper and lower bounds (one per generator per scenario)
        self.model.z_upper_bound_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_lower_bound_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_eq = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        self.model.P_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, within=NonNegativeReals)
        self.model.lambda_var_opt = Var(self.model.scenarios, self.model.time_steps, domain=Reals)
        self.model.mu_upper_bound_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)
        self.model.mu_lower_bound_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)
        self.model.mu_ramp_up_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)
        self.model.mu_ramp_down_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)

    def _build_complementarity_optimal_variables(self) -> None:
        #Complementarity variables for the upper and lower bounds (one per generator per scenario)
        self.model.z_upper_bound_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_lower_bound_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_opt = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Binary)

    def _build_objective(self) -> None:
        self.model.objective = Objective(
            expr=sum(
                self.model.PoA[k] - self.eta * self.model.wasserstein_distance[k]
                for k in self.model.scenarios
            ) / self.num_empirical_scenarios,
            sense=maximize,
        )

    def _build_constraints(self) -> None:
        self._build_support_set()
        self._build_policy_related_constraints()
        self._build_lower_level_equilibrium_constraints()
        self._build_lower_level_optimal_constraints()
        self._build_KKT_stationarity_equilibrium_constraints()
        self._build_KKT_stationarity_optimal_constraints()
        self._build_KKT_complementarity_equilibrium_constraints()
        self._build_KKT_complementarity_optimal_constraints()
        self._build_wasserstein_distance_constraints()
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

        def demand_lower_rule(m, k, t):
            return m.D[k, t] >= self.support_demand_min

        def demand_upper_rule(m, k, t):
            return m.D[k, t] <= self.support_demand_max

        def demand_ramp_up_rule(m, k, t):
            return m.D[k, t] - m.D[k, t - 1] <= self.support_demand_ramp

        def demand_ramp_down_rule(m, k, t):
            return m.D[k, t - 1] - m.D[k, t] <= self.support_demand_ramp

        # Budget constraints 
        def demand_abs_deviation_pos_rule(m, k, t):
            return m.D_abs_deviation[k, t] >= m.D[k, t] - self.support_demand_reference[int(t)]

        def demand_abs_deviation_neg_rule(m, k, t):
            return m.D_abs_deviation[k, t] >= self.support_demand_reference[int(t)] - m.D[k, t]

        def demand_budget_rule(m, k):
            return sum(m.D_abs_deviation[k, t] for t in m.time_steps) <= self.support_demand_budget

        self.model.demand_lower_bound_constraints = Constraint(self.model.scenarios, self.model.time_steps, rule=demand_lower_rule)
        self.model.demand_upper_bound_constraints = Constraint(self.model.scenarios, self.model.time_steps, rule=demand_upper_rule)
        self.model.demand_ramp_up_constraints = Constraint(self.model.scenarios, self.model.time_steps_minus_1, rule=demand_ramp_up_rule)
        self.model.demand_ramp_down_constraints = Constraint(self.model.scenarios, self.model.time_steps_minus_1, rule=demand_ramp_down_rule)
        self.model.demand_abs_deviation_pos_constraints = Constraint(self.model.scenarios, self.model.time_steps, rule=demand_abs_deviation_pos_rule)
        self.model.demand_abs_deviation_neg_constraints = Constraint(self.model.scenarios, self.model.time_steps, rule=demand_abs_deviation_neg_rule)
        self.model.demand_budget_constraint = Constraint(self.model.scenarios, rule=demand_budget_rule)

        def conventional_capacity_rule(m, k, i, t):
            return m.P_max[k, i, t] == self.pmax_list[int(i)]

        def wind_capacity_lower_rule(m, k, i, t):
            return m.P_max[k, i, t] >= self.support_wind_min[int(i)]

        def wind_capacity_upper_rule(m, k, i, t):
            return m.P_max[k, i, t] <= self.support_wind_max[int(i)]

        def wind_ramp_up_rule(m, k, i, t):
            return m.P_max[k, i, t] - m.P_max[k, i, t - 1] <= self.support_wind_ramp

        def wind_ramp_down_rule(m, k, i, t):
            return m.P_max[k, i, t - 1] - m.P_max[k, i, t] <= self.support_wind_ramp

        def capacity_reference(i: int, t: int) -> float:
            if i in self.wind_generator_ids:
                return self.support_wind_reference[i][t]
            return self.pmax_list[i]

        def capacity_abs_deviation_pos_rule(m, k, i, t):
            return m.P_max_abs_deviation[k, i, t] >= m.P_max[k, i, t] - capacity_reference(int(i), int(t))

        def capacity_abs_deviation_neg_rule(m, k, i, t):
            return m.P_max_abs_deviation[k, i, t] >= capacity_reference(int(i), int(t)) - m.P_max[k, i, t]

        def wind_budget_rule(m, k):
            return (
                sum(m.P_max_abs_deviation[k, i, t] for i in m.n_gen for t in m.time_steps)
                <= self.support_wind_budget
            )

        self.model.conventional_capacity_constraints = Constraint(self.model.scenarios, self.model.n_gen_conventional, self.model.time_steps, rule=conventional_capacity_rule)
        self.model.wind_capacity_lower_bound_constraints = Constraint(self.model.scenarios, self.model.n_gen_wind, self.model.time_steps, rule=wind_capacity_lower_rule)
        self.model.wind_capacity_upper_bound_constraints = Constraint(self.model.scenarios, self.model.n_gen_wind, self.model.time_steps, rule=wind_capacity_upper_rule)
        self.model.wind_ramp_up_constraints = Constraint(self.model.scenarios, self.model.n_gen_wind, self.model.time_steps_minus_1, rule=wind_ramp_up_rule)
        self.model.wind_ramp_down_constraints = Constraint(self.model.scenarios, self.model.n_gen_wind, self.model.time_steps_minus_1, rule=wind_ramp_down_rule)
        self.model.capacity_abs_deviation_pos_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=capacity_abs_deviation_pos_rule)
        self.model.capacity_abs_deviation_neg_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=capacity_abs_deviation_neg_rule)
        self.model.wind_budget_constraint = Constraint(self.model.scenarios, rule=wind_budget_rule)

    def _build_wasserstein_distance_constraints(self) -> None:
        """L1 transport distance from perturbed state to empirical scenario s^(k)."""
        def demand_transport_pos_rule(m, k, t):
            return m.D_transport_abs_deviation[k, t] >= m.D[k, t] - self.empirical_demand_profiles[int(k)][int(t)]

        def demand_transport_neg_rule(m, k, t):
            return m.D_transport_abs_deviation[k, t] >= self.empirical_demand_profiles[int(k)][int(t)] - m.D[k, t]

        def capacity_transport_pos_rule(m, k, i, t):
            return m.P_max_transport_abs_deviation[k, i, t] >= m.P_max[k, i, t] - self.empirical_capacity_profiles[int(k)][int(i)][int(t)]

        def capacity_transport_neg_rule(m, k, i, t):
            return m.P_max_transport_abs_deviation[k, i, t] >= self.empirical_capacity_profiles[int(k)][int(i)][int(t)] - m.P_max[k, i, t]

        def wasserstein_distance_rule(m, k):
            return m.wasserstein_distance[k] == (
                sum(m.D_transport_abs_deviation[k, t] for t in m.time_steps)
                + sum(m.P_max_transport_abs_deviation[k, i, t] for i in m.n_gen for t in m.time_steps)
            )

        self.model.demand_transport_pos_constraints = Constraint(self.model.scenarios, self.model.time_steps, rule=demand_transport_pos_rule)
        self.model.demand_transport_neg_constraints = Constraint(self.model.scenarios, self.model.time_steps, rule=demand_transport_neg_rule)
        self.model.capacity_transport_pos_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=capacity_transport_pos_rule)
        self.model.capacity_transport_neg_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=capacity_transport_neg_rule)
        self.model.wasserstein_distance_constraint = Constraint(self.model.scenarios, rule=wasserstein_distance_rule)

    def _modify_decision_variables_from_normalization_stats(self, k: int, t: int, gen_idx: int) -> List[Any]:
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
        if k not in self.model.scenarios:
            raise ValueError(f"Invalid scenario index {k}")

        def _generator_to_player_id(local_gen_idx: int) -> Optional[int]:
            for player in self.players_config:
                if local_gen_idx in player.get("controlled_generators", []):
                    return int(player["id"])
            return None

        def _private_feature_value(feature_name: str, local_k: int, local_t: int, player_id: int):
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
                return sum(self.model.P_max[local_k, g, local_t] for g in controlled)

            raise ValueError(f"Unsupported private feature '{feature_name}'")

        def _raw_feature_expression(feature_name: str, local_k: int, local_t: int, local_gen_idx: int):
            player_id = _generator_to_player_id(local_gen_idx)

            if feature_name == "bias":
                return 1.0
            if feature_name == "demand":
                return self.model.D[local_k, local_t]
            if feature_name == "demand_sq":
                return self.model.D[local_k, local_t] * self.model.D[local_k, local_t]
            if feature_name == "wind_forecast":
                return sum(self.model.P_max[local_k, i, local_t] for i in self.model.n_gen_wind)
            if feature_name == "total_capacity":
                return sum(self.model.P_max[local_k, i, local_t] for i in self.model.n_gen)

            if feature_name in {"player_cost", "player_capacity"}:
                if player_id is None:
                    return 0.0
                return _private_feature_value(feature_name, local_k, local_t, player_id)

            if feature_name == "scarcity_ratio":
                total_cap = sum(self.model.P_max[local_k, i, local_t] for i in self.model.n_gen)
                return self.model.D[local_k, local_t] / total_cap

            if feature_name == "residual_demand":
                wind = sum(self.model.P_max[local_k, i, local_t] for i in self.model.n_gen_wind)
                return self.model.D[local_k, local_t] - wind

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
            raw_expr = _raw_feature_expression(f_name, k, t, gen_idx)

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
                stop = True
                normalized_features.append(1.0)
            else:
                normalized_features.append((raw_expr - f_min) / denom)
                stop = True

        return normalized_features

    def _build_policy_related_constraints(self) -> None:
        """Apply policy constraints in the same structure as the MPEC model."""
        if not self.policy_by_generator:
            # PoA can still build without policy constraints while under development.
            return

        if not hasattr(self.model, "alpha"):
            self.model.alpha = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, domain=Reals)

        if self.policy_type == "linear":
            def policy_rule(m, k, i, t):
                i_int = int(i)
                if i_int not in self.policy_by_generator:
                    return Constraint.Skip

                phi = self._modify_decision_variables_from_normalization_stats(k=int(k), t=int(t), gen_idx=i_int)
                theta = self.policy_by_generator[i_int]
                if len(theta) != len(phi):
                    raise ValueError(
                        f"theta length {len(theta)} does not match feature length {len(phi)}"
                    )
                return m.alpha[k, i, t] == sum(float(theta[f]) * phi[f] for f in range(len(phi)))

            self.model.policy_constraint = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=policy_rule)
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
        self.model.z_NN = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, domain=Reals)
        self.model.y_NN = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, domain=NonNegativeReals)
        self.model.delta_NN = Var(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, domain=Binary)

        BigM = float(self.big_m_complementarity)

        def translation_rule(m, k, i, t, n):
            i_int = int(i)
            n_int = int(n)
            if i_int not in nn_params:
                return Constraint.Skip

            phi = self._modify_decision_variables_from_normalization_stats(k=int(k), t=int(t), gen_idx=i_int)
            gamma = nn_params[i_int]["gamma"]
            Theta = nn_params[i_int]["Theta"]
            return m.z_NN[k, i, t, n] == sum(float(gamma[n_int, f]) * phi[f] for f in range(len(phi))) + float(Theta[n_int])

        def relu_lb_rule(m, k, i, t, n):
            if int(i) not in nn_params:
                return Constraint.Skip
            return m.z_NN[k, i, t, n] <= m.y_NN[k, i, t, n]

        def relu_ub_rule_1(m, k, i, t, n):
            if int(i) not in nn_params:
                return Constraint.Skip
            return m.y_NN[k, i, t, n] <= m.z_NN[k, i, t, n] + BigM * (1 - m.delta_NN[k, i, t, n])

        def relu_ub_rule_2(m, k, i, t, n):
            if int(i) not in nn_params:
                return Constraint.Skip
            return m.y_NN[k, i, t, n] <= BigM * m.delta_NN[k, i, t, n]

        def alpha_rule(m, k, i, t):
            i_int = int(i)
            if i_int not in nn_params:
                return Constraint.Skip
            Gamma = nn_params[i_int]["Gamma"]
            output_bias = nn_params[i_int]["output_bias"]
            return m.alpha[k, i, t] == sum(float(Gamma[int(n)]) * m.y_NN[k, i, t, n] for n in m.NN_nodes) + output_bias

        self.model.translation_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=translation_rule)
        self.model.relu_lb_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=relu_lb_rule)
        self.model.relu_ub_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=relu_ub_rule_1)
        self.model.relu_ub_constraints_2 = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, self.model.NN_nodes, rule=relu_ub_rule_2)
        self.model.alpha_constraints = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=alpha_rule)

    def _build_lower_level_equilibrium_constraints(self):
        def power_balance_eq_rule(m, k, t):
            return sum(m.P_eq[k, i, t] for i in m.n_gen) - m.D[k, t] == 0
        
        def generation_upper_eq_rule(m, k, i, t):
            return m.P_eq[k, i, t] - m.P_max[k, i, t] <= 0 
    
        def generation_lower_eq_rule(m, k, i, t):
            return -m.P_eq[k, i, t] + 0 <= 0
        
        def ramp_up_eq_rule(m, k, i, t):
            return m.P_eq[k, i, t] - m.P_eq[k, i, t-1] - self.ramp_vector_up[i] <= 0
        
        def ramp_up_initial_eq_rule(m, k, i):
            return m.P_eq[k, i, 0] - self.P_init[i] - self.ramp_vector_up[i] <= 0
        
        def ramp_down_eq_rule(m, k, i, t):
            return -m.P_eq[k, i, t] + m.P_eq[k, i, t-1] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_eq_rule(m, k, i):
            return -m.P_eq[k, i, 0] + self.P_init[i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint_eq = Constraint(self.model.scenarios, self.model.time_steps, rule=power_balance_eq_rule)
        self.model.generation_upper_bound_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=generation_upper_eq_rule)
        self.model.generation_lower_bound_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=generation_lower_eq_rule)
        self.model.ramp_up_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_eq_rule)
        self.model.ramp_down_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_eq_rule)
        self.model.ramp_up_initial_feasibility_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_up_initial_eq_rule)
        self.model.ramp_down_initial_feasibility_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_down_initial_eq_rule)

    def _build_lower_level_optimal_constraints(self):
        def power_balance_opt_rule(m, k, t):
            return sum(m.P_opt[k, i, t] for i in m.n_gen) - m.D[k, t] == 0
        
        def generation_upper_opt_rule(m, k, i, t):
            return m.P_opt[k, i, t] - m.P_max[k, i, t] <= 0 
    
        def generation_lower_opt_rule(m, k, i, t):
            return -m.P_opt[k, i, t] + 0 <= 0
        
        def ramp_up_opt_rule(m, k, i, t):
            return m.P_opt[k, i, t] - m.P_opt[k, i, t-1] - self.ramp_vector_up[i] <= 0
        
        def ramp_up_initial_opt_rule(m, k, i):
            return m.P_opt[k, i, 0] - self.P_init[i] - self.ramp_vector_up[i] <= 0
        
        def ramp_down_opt_rule(m, k, i, t):
            return -m.P_opt[k, i, t] + m.P_opt[k, i, t-1] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_opt_rule(m, k, i):
            return -m.P_opt[k, i, 0] + self.P_init[i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint_opt = Constraint(self.model.scenarios, self.model.time_steps, rule=power_balance_opt_rule)
        self.model.generation_upper_bound_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=generation_upper_opt_rule)
        self.model.generation_lower_bound_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=generation_lower_opt_rule)
        self.model.ramp_up_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_opt_rule)
        self.model.ramp_down_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_opt_rule)
        self.model.ramp_up_initial_feasibility_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_up_initial_opt_rule)
        self.model.ramp_down_initial_feasibility_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_down_initial_opt_rule)

    def _build_KKT_stationarity_equilibrium_constraints(self):
        def stationarity_eq_rule(m, k, i, t):
            return m.alpha[k, i, t] - m.lambda_var_eq[k, t] + m.mu_upper_bound_eq[k, i, t] - m.mu_lower_bound_eq[k, i, t] + m.mu_ramp_up_eq[k, i, t] - m.mu_ramp_up_eq[k, i, t+1] - m.mu_ramp_down_eq[k, i, t] + m.mu_ramp_down_eq[k, i, t+1] == 0

        def final_ramp_up_dual_eq_rule(m, k, i):
            return m.mu_ramp_up_eq[k, i, self.num_time_steps] == 0

        def final_ramp_down_dual_eq_rule(m, k, i):
            return m.mu_ramp_down_eq[k, i, self.num_time_steps] == 0

        self.model.stationarity_constraint_strategic_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=stationarity_eq_rule)
        self.model.final_ramp_up_dual_constraint_eq = Constraint(self.model.scenarios, self.model.n_gen, rule=final_ramp_up_dual_eq_rule)
        self.model.final_ramp_down_dual_constraint_eq = Constraint(self.model.scenarios, self.model.n_gen, rule=final_ramp_down_dual_eq_rule)
    
    def _build_KKT_stationarity_optimal_constraints(self):
        def stationarity_opt_rule(m, k, i, t):
            return self.cost_vector[i] - m.lambda_var_opt[k, t] + m.mu_upper_bound_opt[k, i, t] - m.mu_lower_bound_opt[k, i, t] + m.mu_ramp_up_opt[k, i, t] - m.mu_ramp_up_opt[k, i, t+1] - m.mu_ramp_down_opt[k, i, t] + m.mu_ramp_down_opt[k, i, t+1] == 0

        def final_ramp_up_dual_opt_rule(m, k, i):
            return m.mu_ramp_up_opt[k, i, self.num_time_steps] == 0

        def final_ramp_down_dual_opt_rule(m, k, i):
            return m.mu_ramp_down_opt[k, i, self.num_time_steps] == 0

        self.model.stationarity_constraint_strategic_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=stationarity_opt_rule)
        self.model.final_ramp_up_dual_constraint_opt = Constraint(self.model.scenarios, self.model.n_gen, rule=final_ramp_up_dual_opt_rule)
        self.model.final_ramp_down_dual_constraint_opt = Constraint(self.model.scenarios, self.model.n_gen, rule=final_ramp_down_dual_opt_rule)

    def _build_KKT_complementarity_equilibrium_constraints(self):
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_eq_rule(m, k, i, t):
            return -BigM * (1 - m.z_upper_bound_eq[k, i, t]) <= m.P_eq[k, i, t] - m.P_max[k, i, t]  

        def upper_bound_complementarity_dual_eq_rule(m, k, i, t):
            return m.mu_upper_bound_eq[k, i, t] <= BigM * m.z_upper_bound_eq[k, i, t] 
        
        def lower_bound_complementarity_eq_rule(m, k, i, t):
            return -BigM * (1 - m.z_lower_bound_eq[k, i, t]) <= -m.P_eq[k, i, t] + 0

        def lower_bound_complementarity_dual_eq_rule(m, k, i, t):
            return m.mu_lower_bound_eq[k, i, t] <= BigM * m.z_lower_bound_eq[k, i, t] 

        def ramp_up_complementarity_eq_rule(m, k, i, t):
            return -BigM * (1 - m.z_ramp_up_eq[k, i, t]) <= m.P_eq[k, i, t] - m.P_eq[k, i, t-1] - self.ramp_vector_up[i]
        
        def ramp_up_complementarity_initial_eq_rule(m, k, i):
            return -BigM * (1 - m.z_ramp_up_eq[k, i, 0]) <= m.P_eq[k, i, 0] - self.P_init[i] - self.ramp_vector_up[i]

        def ramp_up_complementarity_dual_eq_rule(m, k, i, t):
            return m.mu_ramp_up_eq[k, i, t] <= BigM * m.z_ramp_up_eq[k, i, t]

        def ramp_down_complementarity_eq_rule(m, k, i, t):
            return -BigM * (1 - m.z_ramp_down_eq[k, i, t]) <= - m.P_eq[k, i, t] + m.P_eq[k, i, t-1] - self.ramp_vector_down[i]

        def ramp_down_complementarity_initial_eq_rule(m, k, i):
            return -BigM * (1 - m.z_ramp_down_eq[k, i, 0]) <= - m.P_eq[k, i, 0] + self.P_init[i] - self.ramp_vector_down[i]

        def ramp_down_complementarity_dual_eq_rule(m, k, i, t):
            return m.mu_ramp_down_eq[k, i, t] <= BigM * m.z_ramp_down_eq[k, i, t]

        self.model.upper_bound_complementarity_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_eq_rule)
        self.model.upper_bound_complementarity_constraints_dual_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_dual_eq_rule)
        self.model.lower_bound_complementarity_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_eq_rule)
        self.model.lower_bound_complementarity_constraints_dual_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_dual_eq_rule)

        self.model.ramp_up_complementarity_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_complementarity_eq_rule)
        self.model.ramp_up_complementarity_initial_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_up_complementarity_initial_eq_rule)
        self.model.ramp_up_complementarity_constraints_dual_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=ramp_up_complementarity_dual_eq_rule)

        self.model.ramp_down_complementarity_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_complementarity_eq_rule)
        self.model.ramp_down_complementarity_initial_constraints_eq = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_down_complementarity_initial_eq_rule)
        self.model.ramp_down_complementarity_constraints_dual_eq = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=ramp_down_complementarity_dual_eq_rule)

    def _build_KKT_complementarity_optimal_constraints(self):
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_opt_rule(m, k, i, t):
            return -BigM * (1 - m.z_upper_bound_opt[k, i, t]) <= m.P_opt[k, i, t] - m.P_max[k, i, t]  

        def upper_bound_complementarity_dual_opt_rule(m, k, i, t):
            return m.mu_upper_bound_opt[k, i, t] <= BigM * m.z_upper_bound_opt[k, i, t] 
        
        def lower_bound_complementarity_opt_rule(m, k, i, t):
            return -BigM * (1 - m.z_lower_bound_opt[k, i, t]) <= -m.P_opt[k, i, t] + 0

        def lower_bound_complementarity_dual_opt_rule(m, k, i, t):
            return m.mu_lower_bound_opt[k, i, t] <= BigM * m.z_lower_bound_opt[k, i, t] 

        def ramp_up_complementarity_opt_rule(m, k, i, t):
            return -BigM * (1 - m.z_ramp_up_opt[k, i, t]) <= m.P_opt[k, i, t] - m.P_opt[k, i, t-1] - self.ramp_vector_up[i]
        
        def ramp_up_complementarity_initial_opt_rule(m, k, i):
            return -BigM * (1 - m.z_ramp_up_opt[k, i, 0]) <= m.P_opt[k, i, 0] - self.P_init[i] - self.ramp_vector_up[i]

        def ramp_up_complementarity_dual_opt_rule(m, k, i, t):
            return m.mu_ramp_up_opt[k, i, t] <= BigM * m.z_ramp_up_opt[k, i, t]

        def ramp_down_complementarity_opt_rule(m, k, i, t):
            return -BigM * (1 - m.z_ramp_down_opt[k, i, t]) <= - m.P_opt[k, i, t] + m.P_opt[k, i, t-1] - self.ramp_vector_down[i]

        def ramp_down_complementarity_initial_opt_rule(m, k, i):
            return -BigM * (1 - m.z_ramp_down_opt[k, i, 0]) <= - m.P_opt[k, i, 0] + self.P_init[i] - self.ramp_vector_down[i]

        def ramp_down_complementarity_dual_opt_rule(m, k, i, t):
            return m.mu_ramp_down_opt[k, i, t] <= BigM * m.z_ramp_down_opt[k, i, t]

        self.model.upper_bound_complementarity_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_opt_rule)
        self.model.upper_bound_complementarity_constraints_dual_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_dual_opt_rule)
        self.model.lower_bound_complementarity_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_opt_rule)
        self.model.lower_bound_complementarity_constraints_dual_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_dual_opt_rule)

        self.model.ramp_up_complementarity_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_complementarity_opt_rule)
        self.model.ramp_up_complementarity_initial_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_up_complementarity_initial_opt_rule)
        self.model.ramp_up_complementarity_constraints_dual_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=ramp_up_complementarity_dual_opt_rule)

        self.model.ramp_down_complementarity_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_complementarity_opt_rule)
        self.model.ramp_down_complementarity_initial_constraints_opt = Constraint(self.model.scenarios, self.model.n_gen, rule=ramp_down_complementarity_initial_opt_rule)
        self.model.ramp_down_complementarity_constraints_dual_opt = Constraint(self.model.scenarios, self.model.n_gen, self.model.time_steps, rule=ramp_down_complementarity_dual_opt_rule)

    def _build_PoA_constraints(self) -> None:
        def cost_eq_rule(m, k):
            return m.C_eq[k] == sum(self.cost_vector[i] * m.P_eq[k, i, t] for i in m.n_gen for t in m.time_steps)

        def cost_opt_rule(m, k):
            return m.C_opt[k] == sum(self.cost_vector[i] * m.P_opt[k, i, t] for i in m.n_gen for t in m.time_steps)
        
        def PoA_rule(m, k):
            return m.C_eq[k] - m.C_opt[k] == m.PoA[k]

        self.model.cost_definition_eq = Constraint(self.model.scenarios, rule=cost_eq_rule)
        self.model.cost_definition_opt = Constraint(self.model.scenarios, rule=cost_opt_rule)
        self.model.PoA_constraint = Constraint(self.model.scenarios, rule=PoA_rule)

    def solve(self, solver_name: str = "gurobi", tee: bool = True) -> Dict[str, Any]:
        """
        Solve the optimization model.
        """

        # Create solver
        solver = SolverFactory(solver_name)

        # Solve
        results = solver.solve(self.model, tee=tee)

        # Check solver status
        if (
            results.solver.status != SolverStatus.ok
            or results.solver.termination_condition != TerminationCondition.optimal
        ):
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)
            raise ValueError("Solver did not find an optimal solution")

        solution = {
            "regime": self.empirical_regime,
            "scenario_ids": self.empirical_scenario_ids,
            "eta": self.eta,
            "inner_value": float(value(self.model.objective)),
            "average_poa_proxy": float(np.mean([value(self.model.PoA[k]) for k in self.model.scenarios])),
            "average_wasserstein_distance": float(np.mean([value(self.model.wasserstein_distance[k]) for k in self.model.scenarios])),
            "scenario_poa_proxy": {int(k): float(value(self.model.PoA[k])) for k in self.model.scenarios},
            "scenario_poa": {int(k): float(value(self.model.C_eq[k] / self.model.C_opt[k]) if value(self.model.C_opt[k]) > 1e-6 else 0.0) for k in self.model.scenarios},
            "average_poa": float(np.mean([value(self.model.C_eq[k] / self.model.C_opt[k]) if value(self.model.C_opt[k]) > 1e-6 else 0.0 for k in self.model.scenarios])),
            "min_poa": float(np.min([value(self.model.C_eq[k] / self.model.C_opt[k]) if value(self.model.C_opt[k]) > 1e-6 else 0.0 for k in self.model.scenarios])),
            "max_poa": float(np.max([value(self.model.C_eq[k] / self.model.C_opt[k]) if value(self.model.C_opt[k]) > 1e-6 else 0.0 for k in self.model.scenarios])),
            "scenario_wasserstein_distance": {
                int(k): float(value(self.model.wasserstein_distance[k])) for k in self.model.scenarios
            },
            "scenario_C_eq": {int(k): float(value(self.model.C_eq[k])) for k in self.model.scenarios},
            "scenario_C_opt": {int(k): float(value(self.model.C_opt[k])) for k in self.model.scenarios},
            "termination_condition": str(results.solver.termination_condition),
        }
        if tee:
            print(
                "Solved DRO inner problem: "
                f"regime={solution['regime']}, "
                f"n_scenarios={len(solution['scenario_ids'])}, "
                f"eta={solution['eta']}, "
                f"average_PoA_proxy={solution['average_poa_proxy']:.6f}, "
                f"min_PoA={solution['min_poa']:.6f}, "
                f"max_PoA={solution['max_poa']:.6f}"
                f"average_PoA={solution['average_poa']:.6f}, "
                f"average_distance={solution['average_wasserstein_distance']:.6f}, "
            )
        return solution

    @classmethod
    def load_regime_scenarios(
        cls,
        reference_case: str = "test_case1",
        regime_config_path: str = "config/intertemporal/scenarios/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate empirical scenarios s^(k) from ScenarioManagerV2."""
        manager = ScenarioManagerV2(base_case_reference=reference_case)
        scenario_set = manager.create_scenario_set_from_regimes(
            regime_config_path=regime_config_path,
            regime_set=regime_set,
            seed=seed,
        )
        return scenario_set["scenarios_df"]

    @classmethod
    def build_support_set_config_from_scenarios(
        cls,
        scenarios_df: pd.DataFrame,
        reference_case: str = "test_case1",
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build support-set parameters from a regime's empirical scenarios."""
        if regime is not None:
            local_df = scenarios_df[scenarios_df["regime"].astype(str) == str(regime)].copy()
        else:
            local_df = scenarios_df.copy()

        if local_df.empty:
            raise ValueError(f"No scenarios available for regime '{regime}'")

        (
            num_generators,
            pmax_list,
            _,
            _,
            _,
            _,
            _,
            generators,
            _,
            time_steps,
        ) = load_setup_data(reference_case)

        horizon = int(time_steps)
        demand_profiles = np.vstack(
            [
                np.asarray(cls._as_profile(row["demand_profile"], horizon, "demand_profile"), dtype=float)
                for _, row in local_df.iterrows()
            ]
        )

        demand_reference = np.mean(demand_profiles, axis=0).tolist()
        demand_deviations = np.sum(np.abs(demand_profiles - np.asarray(demand_reference)), axis=1)
        demand_ramp = float(np.max(np.abs(np.diff(demand_profiles, axis=1)))) if horizon > 1 else 0.0

        wind_reference: Dict[int, List[float]] = {}
        wind_min: Dict[int, float] = {}
        wind_max: Dict[int, float] = {}
        wind_ramp = 0.0
        wind_deviations = np.zeros(len(local_df), dtype=float)

        for gen_idx, generator in enumerate(generators):
            gen_name = str(generator.get("name", generator)) if isinstance(generator, dict) else str(generator)
            if not gen_name.upper().startswith("W"):
                continue

            profile_col = f"{gen_name}_profile"
            if profile_col not in local_df.columns:
                raise ValueError(f"Missing wind profile column '{profile_col}'")

            wind_profiles = np.vstack(
                [
                    np.asarray(cls._as_profile(row[profile_col], horizon, profile_col), dtype=float)
                    for _, row in local_df.iterrows()
                ]
            )
            reference = np.mean(wind_profiles, axis=0)
            wind_reference[int(gen_idx)] = reference.tolist()
            wind_min[int(gen_idx)] = float(np.min(wind_profiles))
            wind_max[int(gen_idx)] = float(min(float(pmax_list[int(gen_idx)]), np.max(wind_profiles)))
            if horizon > 1:
                wind_ramp = max(wind_ramp, float(np.max(np.abs(np.diff(wind_profiles, axis=1)))))
            wind_deviations += np.sum(np.abs(wind_profiles - reference), axis=1)

        return {
            "demand_reference": demand_reference,
            "demand_min": float(np.min(demand_profiles)),
            "demand_max": float(np.max(demand_profiles)),
            "demand_ramp": demand_ramp,
            "demand_budget": float(np.max(demand_deviations)),
            "wind_reference": wind_reference,
            "wind_min": wind_min,
            "wind_max": wind_max,
            "wind_ramp": wind_ramp,
            "wind_budget": float(np.max(wind_deviations)),
        }

    @classmethod
    def run_eta_sweep_by_regime(
        cls,
        eta_values: Sequence[float],
        epsilon: float,
        P_init: Sequence[float],
        scenarios_df: Optional[pd.DataFrame] = None,
        regimes: Optional[Sequence[str]] = None,
        reference_case: str = "test_case1",
        regime_config_path: str = "config/intertemporal/scenarios/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        support_set_name: Optional[str] = None,
        feature_normalizer_stats_path: str = "results/feature_normalizer_stats.json",
        big_m_complementarity: float = 1e6,
        policy_results_path: Optional[str] = "results/best_response_results.json",
        policy_data: Optional[Dict[str, Any]] = None,
        solver_name: str = "gurobi",
        tee: bool = False,
        max_scenarios_per_regime: Optional[int] = None,
    ) -> pd.DataFrame:
        """Evaluate the DRO objective as a function of eta for each regime."""
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if scenarios_df is None:
            scenarios_df = cls.load_regime_scenarios(
                reference_case=reference_case,
                regime_config_path=regime_config_path,
                regime_set=regime_set,
            )

        if regimes is None:
            regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())

        results: List[Dict[str, Any]] = []
        for regime in regimes:
            regime_df = scenarios_df[scenarios_df["regime"].astype(str) == str(regime)].copy()
            if regime_df.empty:
                raise ValueError(f"No scenarios available for regime '{regime}'")
            if max_scenarios_per_regime is not None:
                regime_df = regime_df.head(int(max_scenarios_per_regime)).copy()

            if support_set_name is None:
                support_set_config = cls.build_support_set_config_from_scenarios(
                    regime_df,
                    reference_case=reference_case,
                    regime=str(regime),
                )
            else:
                support_set_config = cls.load_support_set_config(
                    config_name=support_set_name,
                )

            for eta in eta_values:
                eta_float = float(eta)
                optimizer = cls(
                    P_init=P_init,
                    num_time_steps=int(regime_df.iloc[0]["time_steps"]),
                    reference_case=reference_case,
                    feature_normalizer_stats_path=feature_normalizer_stats_path,
                    big_m_complementarity=big_m_complementarity,
                    policy_results_path=policy_results_path,
                    policy_data=policy_data,
                    support_set_config=support_set_config,
                    eta=eta_float,
                    empirical_scenario=regime_df,
                )
                optimizer._build_model()
                solution = optimizer.solve(solver_name=solver_name, tee=tee)

                results.append(
                    {
                    "regime": solution['regime'], 
                    "n_scenarios": len(solution['scenario_ids']), 
                    "eta": solution['eta'], 
                    "average_PoA_proxy": solution['average_poa_proxy'], 
                    "min_PoA": solution['min_poa'], 
                    "max_PoA": solution['max_poa'],
                    "average_PoA": solution['average_poa'], 
                    "average_distance": solution['average_wasserstein_distance'], 
                    }
                )

        return pd.DataFrame(results)

    def plot_demand_capacity_trajectory(
        self,
        save_path: str = "results/poa_demand_capacity_trajectory.png",
        show: bool = True,
        scenario_index: int = 0,
    ) -> None:
        """Plot solved trajectories for demand and generator capacities over time."""
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call _build_model() first.")
        if scenario_index not in self.model.scenarios:
            raise ValueError(f"Invalid scenario_index {scenario_index}")

        time_points = list(self.model.time_steps)
        if not time_points:
            raise ValueError("No time steps found in model.")

        demand_vals = [value(self.model.D[scenario_index, t], exception=False) for t in time_points]
        demand_vals = [np.nan if v is None else float(v) for v in demand_vals]

        capacity_by_gen: Dict[int, List[float]] = {}
        for i in self.model.n_gen:
            vals = [value(self.model.P_max[scenario_index, i, t], exception=False) for t in time_points]
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

        ax.set_title(f"Demand and Generator Capacity Trajectories (Scenario {scenario_index})")
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
    example_reference_case = "test_case1"
    num_generators, *_ = load_setup_data(example_reference_case)
    P_init = np.ones(int(num_generators)) * 25

    eta_grid = [4.0, 5.0]
    epsilon = 1.0

    # Use max_scenarios_per_regime while developing; remove it for the full PoA_analysis run.
    dro_results = PoAOptimization.run_eta_sweep_by_regime(
        eta_values=eta_grid,
        epsilon=epsilon,
        P_init=P_init,
        reference_case=example_reference_case,
        regime_set="PoA_analysis",
        support_set_name="test_case1_base",
        policy_results_path="results/best_response_results.json",
        max_scenarios_per_regime=2,
        tee=False,
    )
    print(dro_results.to_string(index=False))
