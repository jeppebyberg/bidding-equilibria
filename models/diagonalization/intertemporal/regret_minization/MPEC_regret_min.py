from pyomo.environ import *
import ast
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

from .utilities.MPEC_utils import get_mpec_parameters


class MPECModel:
    def __init__(
        self,
        reference_case: str,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        strategic_player_id: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        feature_builder: Optional[Any] = None,
        ramps_df: Optional[pd.DataFrame] = None,
        pmin_default: float = 0.0,
        p_init: Optional[Any] = None,
    ):
        """
        Intertemporal regret-minimization MPEC model.

        Notes
        -----
        - This version is rebuilt for each strategic player / iteration.
        - No incremental `_update_strategic_player` flow is used.
        - Feature handling is intentionally placeholder-based for now.
        """
        self.reference_case = reference_case
        self.feature_builder = feature_builder  # kept for compatibility with caller

        self.config = get_mpec_parameters()
        if config_overrides:
            self.config.update(config_overrides)

        self.alpha_min = self.config.get("alpha_min")
        self.alpha_max = self.config.get("alpha_max")
        self.big_m_complementarity = self.config.get("big_m_complementarity")
        self.big_m_bid_separation = self.config.get("big_m_bid_separation")
        self.bid_separation_epsilon = self.config.get("bid_separation_epsilon")

        self.players_config = players_config
        self.strategic_player_id = strategic_player_id
        self.strategic_generators: List[int] = []

        self.scenarios_df = scenarios_df.copy().reset_index(drop=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df

        if self.ramps_df is None:
            raise ValueError("ramps_df is required for intertemporal ramp constraints.")

        self._extract_scenario_data(self.scenarios_df, self.costs_df, self.ramps_df, pmin_default)
        self.P_init = self._initialize_p_init(p_init)

        self.num_policy_features = len(self.feature_builder.features) if self.feature_builder is not None else 4
        self.features: Dict[Tuple[int, int, int], List[float]] = {}
        self._build_feature_matrix()

        self.model = None

        if self.strategic_player_id is not None:
            self.build_model(self.strategic_player_id)

    @staticmethod
    def _convert_profile(value: Any, expected_len: int, column_name: str) -> List[float]:
        """Convert profile-like input into a numeric list with expected length."""
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception as exc:
                raise ValueError(f"Could not parse profile column '{column_name}': {exc}") from exc

        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Column '{column_name}' must contain a list/tuple of length {expected_len}")

        if len(value) != expected_len:
            raise ValueError(
                f"Profile length mismatch in column '{column_name}': expected {expected_len}, got {len(value)}"
            )

        try:
            return [float(v) for v in value]
        except Exception as exc:
            raise ValueError(f"Profile column '{column_name}' contains non-numeric values") from exc

    def _extract_scenario_data(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        pmin_default: float,
    ) -> None:
        """Extract intertemporal scenario data."""
        
        demand_profile_col = None
        for col in scenarios_df.columns:
            if "demand_profile" in col.lower():
                demand_profile_col = col
                break

        if demand_profile_col is None:
            raise ValueError("No demand profile column found. Expected column containing 'demand_profile'.")

        capacity_cols = [col for col in scenarios_df.columns if col.endswith("_cap")]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'.")

        self.generator_names = [col.replace("_cap", "") for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(scenarios_df)

        if "time_steps" in scenarios_df.columns:
            self.num_time_steps = int(scenarios_df["time_steps"].iloc[0])
        else:
            first_profile = scenarios_df[demand_profile_col].iloc[0]
            if isinstance(first_profile, str):
                first_profile = ast.literal_eval(first_profile)
            self.num_time_steps = len(first_profile)

        if self.num_time_steps <= 0:
            raise ValueError(f"Invalid time_steps value: {self.num_time_steps}")

        self.demand_scenarios: List[List[float]] = []
        self.pmax_scenarios: List[List[List[float]]] = []
        self.pmin_scenarios: List[List[List[float]]] = []
        self.bid_scenarios: List[List[List[float]]] = []

        for _, row in scenarios_df.iterrows():
            demand_profile = self._convert_profile(row[demand_profile_col], self.num_time_steps, demand_profile_col)
            self.demand_scenarios.append(demand_profile)

            pmax_s_by_t: List[List[float]] = []
            pmin_s_by_t: List[List[float]] = []
            bid_s_by_t: List[List[float]] = []

            for t in range(self.num_time_steps):
                pmax_t: List[float] = []
                pmin_t: List[float] = []
                bid_t: List[float] = []

                for gen in self.generator_names:
                    cap = float(row[f"{gen}_cap"])
                    if gen.startswith("W") and f"{gen}_profile" in scenarios_df.columns:
                        wind_profile = self._convert_profile(row[f"{gen}_profile"], self.num_time_steps, f"{gen}_profile")
                        cap = wind_profile[t]

                    if f"{gen}_bid_profile" in scenarios_df.columns:
                        bid_profile = self._convert_profile(
                            row[f"{gen}_bid_profile"], self.num_time_steps, f"{gen}_bid_profile"
                        )
                        bid_value = bid_profile[t]
                    elif f"{gen}_bid" in scenarios_df.columns:
                        bid_value = float(row[f"{gen}_bid"])
                    else:
                        bid_value = float(costs_df[f"{gen}_cost"].iloc[0])

                    pmax_t.append(cap)
                    pmin_t.append(float(pmin_default))
                    bid_t.append(float(bid_value))

                pmax_s_by_t.append(pmax_t)
                pmin_s_by_t.append(pmin_t)
                bid_s_by_t.append(bid_t)

            self.pmax_scenarios.append(pmax_s_by_t)
            self.pmin_scenarios.append(pmin_s_by_t)
            self.bid_scenarios.append(bid_s_by_t)

        self.cost_vector = [float(costs_df[f"{gen}_cost"].iloc[0]) for gen in self.generator_names]
        self.ramp_vector_up = [float(ramps_df[f"{gen}_ramp_up"].iloc[0]) for gen in self.generator_names]
        self.ramp_vector_down = [float(ramps_df[f"{gen}_ramp_down"].iloc[0]) for gen in self.generator_names]

    def _initialize_p_init(self, p_init: Optional[Any]) -> List[List[float]]:
        """Normalize initial production input to shape [num_scenarios][num_generators]."""
        if p_init is None:
            return [[0.0 for _ in range(self.num_generators)] for _ in range(self.num_scenarios)]

        if np.isscalar(p_init):
            value = float(p_init)
            return [[value for _ in range(self.num_generators)] for _ in range(self.num_scenarios)]

        if isinstance(p_init, np.ndarray):
            p_init = p_init.tolist()

        if isinstance(p_init, (list, tuple)):
            if len(p_init) == self.num_generators and all(np.isscalar(v) for v in p_init):
                row = [float(v) for v in p_init]
                return [list(row) for _ in range(self.num_scenarios)]

            if len(p_init) == self.num_scenarios:
                matrix: List[List[float]] = []
                for s, row in enumerate(p_init):
                    if not isinstance(row, (list, tuple, np.ndarray)):
                        raise ValueError(
                            f"Invalid p_init row at scenario {s}: expected a sequence of length {self.num_generators}"
                        )
                    if len(row) != self.num_generators:
                        raise ValueError(
                            f"Invalid p_init shape at scenario {s}: expected {self.num_generators} generators, got {len(row)}"
                        )
                    matrix.append([float(v) for v in row])
                return matrix

        raise ValueError(
            "Invalid p_init format. Expected None, scalar, 1D vector of generators, or "
            "2D matrix [num_scenarios][num_generators]."
        )

    def _build_feature_matrix(self) -> None:
        """
        Build policy features per (scenario, time, generator).

        When a shared ``FeatureBuilder`` is provided, this uses the configured
        feature list and includes intertemporal values from the previous time
        step. Otherwise, it falls back to the legacy 4-feature placeholder.
        """
        self.features = {}

        if self.feature_builder is None:
            self.num_policy_features = 4
            for s in range(self.num_scenarios):
                for t in range(self.num_time_steps):
                    for i in range(self.num_generators):
                        self.features[s, t, i] = [
                            1.0,
                            float(self.demand_scenarios[s][t]),
                            float(self.pmax_scenarios[s][t][i]),
                            float(self.cost_vector[i]),
                        ]
            return

        expected_dim: Optional[int] = None

        for s in range(self.num_scenarios):
            for t in range(self.num_time_steps):
                demand_t = float(self.demand_scenarios[s][t])
                total_capacity_t = float(sum(float(self.pmax_scenarios[s][t][g]) for g in range(self.num_generators)))
                wind_t = float(sum(
                    float(self.pmax_scenarios[s][t][g])
                    for g, name in enumerate(self.generator_names)
                    if name.startswith("W")
                ))

                if t > 0:
                    demand_tm1 = float(self.demand_scenarios[s][t - 1])
                    total_capacity_tm1 = float(sum(float(self.pmax_scenarios[s][t - 1][g]) for g in range(self.num_generators)))
                    wind_tm1 = float(sum(
                        float(self.pmax_scenarios[s][t - 1][g])
                        for g, name in enumerate(self.generator_names)
                        if name.startswith("W")
                    ))
                else:
                    demand_tm1 = 0.0
                    total_capacity_tm1 = 0.0
                    wind_tm1 = 0.0

                for i in range(self.num_generators):
                    phi = self.feature_builder.build_intertemporal_features(
                        demand=demand_t,
                        wind_forecast=wind_t,
                        total_capacity=total_capacity_t,
                        player_cost=[float(self.cost_vector[i])],
                        player_capacity=[float(self.pmax_scenarios[s][t][i])],
                        demand_tm1=demand_tm1,
                        wind_tm1=wind_tm1,
                        total_capacity_tm1=total_capacity_tm1,
                    )

                    phi_list = np.atleast_1d(phi).astype(np.float64).tolist()
                    if expected_dim is None:
                        expected_dim = len(phi_list)
                    elif len(phi_list) != expected_dim:
                        raise ValueError(
                            f"Inconsistent feature dimension at (s={s}, t={t}, i={i}). "
                            f"Expected {expected_dim}, got {len(phi_list)}."
                        )

                    self.features[s, t, i] = phi_list

        if expected_dim is not None:
            self.num_policy_features = expected_dim

    def build_model(self, strategic_player_id: int) -> None:
        """Build a fresh model for the specified strategic player."""
        strategic_player = next((p for p in self.players_config if p["id"] == strategic_player_id), None)
        if strategic_player is None:
            raise ValueError(f"Strategic player {strategic_player_id} not found in players_config")

        self.strategic_player_id = strategic_player_id
        self.strategic_generators = list(strategic_player["controlled_generators"])

        self._build_feature_matrix()
        self._build_model()

    def _build_model(self) -> None:
        if self.strategic_player_id is None:
            raise ValueError("Strategic player must be set before building model.")

        self.model = ConcreteModel()

        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.n_scenarios = Set(initialize=range(self.num_scenarios))
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        self.model.strategic_index = Set(initialize=self.strategic_generators)

        non_strategic_gens = [i for i in range(self.num_generators) if i not in self.strategic_generators]
        self.model.non_strategic_index = Set(initialize=non_strategic_gens)

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        self._build_upper_level_primal_variables()
        self._build_lower_level_dual_variables()
        self._build_lower_level_primal_variables()
        self._build_complementarity_variables()
        self._build_bid_separation_variables()
        self._build_policy_variables()

    def _build_upper_level_primal_variables(self) -> None:
        self.model.alpha = Var(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, domain=Reals)

    def _build_lower_level_dual_variables(self) -> None:
        self.model.lambda_var = Var(self.model.n_scenarios, self.model.time_steps, domain=Reals)
        self.model.mu_upper_bound = Var(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=NonNegativeReals
        )
        self.model.mu_lower_bound = Var(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=NonNegativeReals
        )
        self.model.mu_ramp_up = Var(
            self.model.n_scenarios, self.model.time_steps_plus_1, self.model.n_gen, domain=NonNegativeReals
        )
        self.model.mu_ramp_down = Var(
            self.model.n_scenarios, self.model.time_steps_plus_1, self.model.n_gen, domain=NonNegativeReals
        )

    def _build_lower_level_primal_variables(self) -> None:
        self.model.P = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=NonNegativeReals)

    def _build_complementarity_variables(self) -> None:
        self.model.z_upper_bound = Var(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary
        )
        self.model.z_lower_bound = Var(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary
        )
        self.model.z_ramp_up = Var(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary
        )
        self.model.z_ramp_down = Var(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary
        )

    def _build_policy_variables(self) -> None:
        self.model.n_features = Set(initialize=range(self.num_policy_features))
        self.model.theta = Var(self.model.n_features, domain=Reals)

    def _build_bid_separation_variables(self) -> None:
        self.model.tau = Var(
            self.model.n_scenarios,
            self.model.time_steps,
            self.model.strategic_index,
            self.model.non_strategic_index,
            domain=Binary,
        )

    def _build_objective(self) -> None:
        self.model.objective = Objective(
            expr=
                1 / self.num_scenarios
                * sum(
                    sum(
                        self.model.lambda_var[s, t] * self.demand_scenarios[s][t]
                        + sum(
                            -self.bid_scenarios[s][t][i] * self.model.P[s, t, i]
                            for i in self.model.non_strategic_index
                        )
                        + sum(
                            -self.model.mu_upper_bound[s, t, i] * self.pmax_scenarios[s][t][i]
                            + self.model.mu_upper_bound[s, t, i] * self.pmin_scenarios[s][t][i]
                            - self.model.mu_ramp_up[s, t, i] * self.ramp_vector_up[i]
                            - self.model.mu_ramp_down[s, t, i] * self.ramp_vector_down[i]
                            for i in self.model.n_gen
                        )
                        for t in self.model.time_steps
                    )
                    + sum(
                        -self.model.mu_ramp_up[s, 0, i] * self.P_init[s][i]
                        + self.model.mu_ramp_down[s, 0, i] * self.P_init[s][i]
                        for i in self.model.n_gen
                    )
                    + sum(
                        sum(
                            self.model.mu_upper_bound[s, t, i] * self.pmax_scenarios[s][t][i]
                            - self.model.mu_lower_bound[s, t, i] * self.pmin_scenarios[s][t][i]
                            for i in self.model.strategic_index
                        )
                        for t in self.model.time_steps
                    )
                    + sum(
                        self.model.mu_ramp_up[s, 0, i] * (self.P_init[s][i] + self.ramp_vector_up[i])
                        - self.model.mu_ramp_down[s, 0, i] * (self.P_init[s][i] - self.ramp_vector_down[i])
                        + sum(
                            self.model.mu_ramp_up[s, t, i] * self.ramp_vector_up[i]
                            + self.model.mu_ramp_down[s, t, i] * self.ramp_vector_down[i]
                            for t in range(1, self.num_time_steps)
                        )
                        for i in self.model.strategic_index
                    )
                    + sum(
                        sum(
                            self.cost_vector[i] * self.model.P[s, t, i]
                            for i in self.model.strategic_index
                        )
                        for t in self.model.time_steps
                    )
                    for s in self.model.n_scenarios
                ),
            sense=minimize,
        )

    def _build_constraints(self) -> None:
        self._build_upper_level_constraints()
        self._build_lower_level_constraints()
        self._build_kkt_stationarity_constraints()
        self._build_kkt_complementarity_constraints()
        self._build_bid_separation_constraints()
        self._build_policy_constraints()

    def _build_upper_level_constraints(self) -> None:
        def min_bid_rule(m, s, t, i):
            return m.alpha[s, t, i] >= self.alpha_min

        def max_bid_rule(m, s, t, i):
            return m.alpha[s, t, i] <= self.alpha_max

        # def min_bid_rule_2(m, s, t, i):
        #     return m.alpha[s, t, i] >= self.cost_vector[i]

        self.model.min_bid_constraint = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=min_bid_rule
        )
        self.model.max_bid_constraint = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=max_bid_rule
        )
        # self.model.min_bid_constraint_2 = Constraint(
        #     self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=min_bid_rule_2
        # )

    def _build_lower_level_constraints(self) -> None:
        def power_balance_rule(m, s, t):
            return sum(m.P[s, t, i] for i in m.n_gen) == self.demand_scenarios[s][t]

        def generation_upper_rule(m, s, t, i):
            return 0 <= self.pmax_scenarios[s][t][i] - m.P[s, t, i]

        def generation_lower_rule(m, s, t, i):
            return 0 <= m.P[s, t, i] - self.pmin_scenarios[s][t][i]

        def ramp_up_rule(m, s, t, i):
            return m.P[s, t, i] - m.P[s, t - 1, i] - self.ramp_vector_up[i] <= 0

        def ramp_up_initial_rule(m, s, i):
            return m.P[s, 0, i] - self.P_init[s][i] - self.ramp_vector_up[i] <= 0

        def ramp_down_rule(m, s, t, i):
            return -m.P[s, t, i] + m.P[s, t - 1, i] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_rule(m, s, i):
            return -m.P[s, 0, i] + self.P_init[s][i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint = Constraint(self.model.n_scenarios, self.model.time_steps, rule=power_balance_rule)
        self.model.generation_upper_bound_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=generation_upper_rule
        )
        self.model.generation_lower_bound_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=generation_lower_rule
        )
        self.model.ramp_up_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_up_rule
        )
        self.model.ramp_down_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_down_rule
        )
        self.model.ramp_up_initial_feasibility_constraints = Constraint(
            self.model.n_scenarios, self.model.n_gen, rule=ramp_up_initial_rule
        )
        self.model.ramp_down_initial_feasibility_constraints = Constraint(
            self.model.n_scenarios, self.model.n_gen, rule=ramp_down_initial_rule
        )

    def _build_kkt_stationarity_constraints(self) -> None:

        def stationarity_rule_strategic_agent(m, s, t, i):
            return m.alpha[s, t, i] - m.lambda_var[s, t] + m.mu_upper_bound[s, t, i] - m.mu_lower_bound[s, t, i] + m.mu_ramp_up[s, t, i] - m.mu_ramp_up[s, t+1, i] - m.mu_ramp_down[s, t, i] + m.mu_ramp_down[s, t+1, i] == 0

        def stationarity_rule_non_strategic_agents(m, s, t, i):
            return self.bid_scenarios[s][t][i] - m.lambda_var[s, t] + m.mu_upper_bound[s, t, i] - m.mu_lower_bound[s, t, i] + m.mu_ramp_up[s, t, i] - m.mu_ramp_up[s, t+1, i] - m.mu_ramp_down[s, t, i] + m.mu_ramp_down[s, t+1, i] == 0

        def final_ramp_up_dual_rule(m, s, i):
            return m.mu_ramp_up[s, self.num_time_steps, i] == 0

        def final_ramp_down_dual_rule(m, s, i):
            return m.mu_ramp_down[s, self.num_time_steps, i] == 0

        self.model.stationarity_constraint_strategic = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=stationarity_rule_strategic_agent
        )
        self.model.stationarity_constraint_non_strategic = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.non_strategic_index, rule=stationarity_rule_non_strategic_agents
        )
        self.model.final_ramp_up_dual_constraint = Constraint(
            self.model.n_scenarios, self.model.n_gen, rule=final_ramp_up_dual_rule
        )
        self.model.final_ramp_down_dual_constraint = Constraint(
            self.model.n_scenarios, self.model.n_gen, rule=final_ramp_down_dual_rule
        )

    def _build_kkt_complementarity_constraints(self) -> None:
        big_m = self.big_m_complementarity

        def upper_bound_comp_rule(m, s, t, i):
            return self.pmax_scenarios[s][t][i] - m.P[s, t, i] <= big_m * (1 - m.z_upper_bound[s, t, i])

        def upper_bound_comp_dual_rule(m, s, t, i):
            return m.mu_upper_bound[s, t, i] <= big_m * m.z_upper_bound[s, t, i]

        def lower_bound_comp_rule(m, s, t, i):
            return m.P[s, t, i] - self.pmin_scenarios[s][t][i] <= big_m * (1 - m.z_lower_bound[s, t, i])

        def lower_bound_comp_dual_rule(m, s, t, i):
            return m.mu_lower_bound[s, t, i] <= big_m * m.z_lower_bound[s, t, i]

        def ramp_up_comp_rule(m, s, t, i):
            return -big_m * (1 - m.z_ramp_up[s, t, i]) <= m.P[s, t, i] - m.P[s, t - 1, i] - self.ramp_vector_up[i]

        def ramp_up_comp_initial_rule(m, s, i):
            return -big_m * (1 - m.z_ramp_up[s, 0, i]) <= m.P[s, 0, i] - self.P_init[s][i] - self.ramp_vector_up[i]

        def ramp_up_comp_dual_rule(m, s, t, i):
            return m.mu_ramp_up[s, t, i] <= big_m * m.z_ramp_up[s, t, i]

        def ramp_down_comp_rule(m, s, t, i):
            return -big_m * (1 - m.z_ramp_down[s, t, i]) <= -m.P[s, t, i] + m.P[s, t - 1, i] - self.ramp_vector_down[i]

        def ramp_down_comp_initial_rule(m, s, i):
            return -big_m * (1 - m.z_ramp_down[s, 0, i]) <= -m.P[s, 0, i] + self.P_init[s][i] - self.ramp_vector_down[i]

        def ramp_down_comp_dual_rule(m, s, t, i):
            return m.mu_ramp_down[s, t, i] <= big_m * m.z_ramp_down[s, t, i]

        self.model.upper_bound_complementarity_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=upper_bound_comp_rule
        )
        self.model.upper_bound_complementarity_constraints_dual = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=upper_bound_comp_dual_rule
        )
        self.model.lower_bound_complementarity_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=lower_bound_comp_rule
        )
        self.model.lower_bound_complementarity_constraints_dual = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=lower_bound_comp_dual_rule
        )
        self.model.ramp_up_complementarity_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_up_comp_rule
        )
        self.model.ramp_up_complementarity_initial_constraints = Constraint(
            self.model.n_scenarios, self.model.n_gen, rule=ramp_up_comp_initial_rule
        )
        self.model.ramp_up_complementarity_constraints_dual = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=ramp_up_comp_dual_rule
        )

        self.model.ramp_down_complementarity_constraints = Constraint(
            self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_down_comp_rule
        )
        self.model.ramp_down_complementarity_initial_constraints = Constraint(
            self.model.n_scenarios, self.model.n_gen, rule=ramp_down_comp_initial_rule
        )
        self.model.ramp_down_complementarity_constraints_dual = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=ramp_down_comp_dual_rule
        )

    def _build_bid_separation_constraints(self) -> None:
        big_m = self.big_m_bid_separation
        epsilon = self.bid_separation_epsilon

        def alpha_upper_separation(m, s, t, i, k):
            return m.alpha[s, t, i] >= self.bid_scenarios[s][t][k] + epsilon - big_m * (1 - m.tau[s, t, i, k])

        def alpha_lower_separation(m, s, t, i, k):
            return m.alpha[s, t, i] <= self.bid_scenarios[s][t][k] - epsilon + big_m * m.tau[s, t, i, k]

        self.model.alpha_upper_separation_constraints = Constraint(
            self.model.n_scenarios,
            self.model.time_steps,
            self.model.strategic_index,
            self.model.non_strategic_index,
            rule=alpha_upper_separation,
        )
        self.model.alpha_lower_separation_constraints = Constraint(
            self.model.n_scenarios,
            self.model.time_steps,
            self.model.strategic_index,
            self.model.non_strategic_index,
            rule=alpha_lower_separation,
        )

    def _build_policy_constraints(self) -> None:
        def policy_rule(m, s, t, i):
            phi = self.features[s, t, i]
            return m.alpha[s, t, i] == sum(m.theta[k] * float(phi[k]) for k in m.n_features)

        self.model.policy_constraint = Constraint(
            self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=policy_rule
        )

    def solve(self) -> None:
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model(strategic_player_id) first.")

        solver = SolverFactory("gurobi")
        results = solver.solve(self.model, tee=True)

        if not (results.solver.status == "ok") and not (results.solver.termination_condition == "optimal"):
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    def get_optimal_bids(self) -> List[List[List[float]]]:
        """
        Return full intertemporal bid tensor [scenario][time][generator].
        """
        if self.model is None or not hasattr(self.model, "alpha"):
            raise ValueError("Model has not been built/solved.")

        strategic_indices = self.strategic_generators
        optimal_bid_scenarios = [
            [list(self.bid_scenarios[s][t]) for t in range(self.num_time_steps)]
            for s in range(self.num_scenarios)
        ]

        for s in range(self.num_scenarios):
            for t in range(self.num_time_steps):
                for i in strategic_indices:
                    alpha_value = self.model.alpha[s, t, i].value
                    if alpha_value is not None:
                        optimal_bid_scenarios[s][t][i] = float(alpha_value)

        return optimal_bid_scenarios

    def update_bids_with_optimal_values(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """
        Write strategic bids back to `{gen}_bid_profile` columns.
        """
        optimal_bid_scenarios = self.get_optimal_bids()
        strategic_indices = self.strategic_generators
        updated_df = scenarios_df.copy()

        for gen_idx in strategic_indices:
            gen_name = self.generator_names[gen_idx]
            bid_profile_col = f"{gen_name}_bid_profile"

            if bid_profile_col not in updated_df.columns:
                updated_df[bid_profile_col] = [None] * len(updated_df)

            for s in range(self.num_scenarios):
                profile = [float(optimal_bid_scenarios[s][t][gen_idx]) for t in range(self.num_time_steps)]
                updated_df.at[s, bid_profile_col] = profile

        return updated_df

    def get_scenario_profits(self) -> List[float]:
        """
        Strategic-player profit per scenario, summed across time.
        """
        if self.model is None or not hasattr(self.model, "lambda_var"):
            raise ValueError("Model has not been built/solved.")

        profits = []
        for s in range(self.num_scenarios):
            profit_scenario = 0.0
            for t in range(self.num_time_steps):
                lambda_value = self.model.lambda_var[s, t].value
                lambda_value = 0.0 if lambda_value is None else float(lambda_value)
                for i in self.strategic_generators:
                    dispatch = self.model.P[s, t, i].value
                    dispatch = 0.0 if dispatch is None else float(dispatch)
                    cost = float(self.cost_vector[i])
                    profit_scenario += lambda_value * dispatch - cost * dispatch
            profits.append(float(profit_scenario))

        return profits

    def update_scenarios(self, new_scenarios_df: pd.DataFrame) -> None:
        """
        Replace scenario data; model will need rebuilding with build_model(...).
        """
        self.scenarios_df = new_scenarios_df.copy().reset_index(drop=True)
        self._extract_scenario_data(self.scenarios_df, self.costs_df, self.ramps_df, pmin_default=0.0)
        self._build_feature_matrix()
        self.model = None

    def get_optimal_theta(self) -> np.ndarray:
        if self.model is None or not hasattr(self.model, "theta"):
            raise ValueError("Policy variables (theta) not available. Build and solve first.")

        return np.array([self.model.theta[k].value for k in self.model.n_features], dtype=np.float64)

    def get_policy_bids(self, theta: np.ndarray, scenarios_df: pd.DataFrame) -> List[List[List[float]]]:
        """
        Evaluate placeholder policy on scenarios and return [s][t][i].
        """
        temp = MPECModel(
            reference_case=self.reference_case,
            scenarios_df=scenarios_df,
            costs_df=self.costs_df,
            players_config=self.players_config,
            strategic_player_id=self.strategic_player_id,
            config_overrides=self.config,
            feature_builder=self.feature_builder,
            ramps_df=self.ramps_df,
            p_init=self.P_init,
        )

        bids = [
            [list(temp.bid_scenarios[s][t]) for t in range(temp.num_time_steps)]
            for s in range(temp.num_scenarios)
        ]

        controlled = self.strategic_generators
        for s in range(temp.num_scenarios):
            for t in range(temp.num_time_steps):
                for i in controlled:
                    phi = np.array(temp.features[s, t, i], dtype=np.float64)
                    bids[s][t][i] = float(theta @ phi)

        return bids

    def print_players_summary(self) -> None:
        if not self.players_config:
            print("No players configuration loaded.")
            return

        print("\n=== Players Configuration Summary ===")
        print(f"Total Players: {len(self.players_config)}")
        for player in self.players_config:
            player_name = player["id"]
            controlled_gens = player["controlled_generators"]
            gen_names = [
                self.generator_names[i] if i < len(self.generator_names) else f"Gen{i}"
                for i in controlled_gens
            ]
            print(f"Player {player_name}: Controls {len(controlled_gens)} generators - {gen_names}")
