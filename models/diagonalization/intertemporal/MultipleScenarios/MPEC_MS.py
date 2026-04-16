from pyomo.environ import *
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import ast

from .utilities.MPEC_utils import get_mpec_parameters


class MPECModel:
    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        p_init: Any,
        feature_matrix_by_player: Dict[int, Dict[Tuple[int, int, int], List[float]]],
        strategic_player_id: int = None,
        pmin_default: float = 0.0,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MPEC model with scenario data and configuration
        
        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data with demand, generator capacity, and bid columns
            Expected columns:
            - Demand column: should contain 'demand' or 'load' in name
            - Generator capacity columns: should end with '_cap' (e.g., 'G1_cap', 'G2_cap')
            - Generator bid columns: should end with '_bid' (e.g., 'G1_bid', 'G2_bid')
        costs_df : pd.DataFrame
            DataFrame containing static generator costs
            Expected columns: should end with '_cost' (e.g., 'G1_cost', 'G2_cost')
        players_config : List[Dict[str, Any]]
            List of player configurations from base case, each with 'name' and 'controlled_generators'
        strategic_player_id : int, optional
            ID of the player to optimize (must match a player name in players_config)
        pmin_default : float, optional
            Default minimum generation level for generators (if not specified in scenarios_df)
        p_init : Any, optional
            Initial production levels for ramp constraints (list of lists: scenarios x generators)
        feature_matrix_by_player : Dict[int, Dict[Tuple[int, int, int], List[float]]]
            Precomputed feature matrix dictionary keyed by player id.
        config_overrides : Dict[str, Any], optional
            Configuration overrides for MPEC parameters
        """

        self.config = get_mpec_parameters()
        if config_overrides:
            self.config.update(config_overrides)

        self.alpha_min = self.config.get("alpha_min")
        self.alpha_max = self.config.get("alpha_max")
        self.big_m_complementarity = self.config.get("big_m_complementarity")
        self.big_m_bid_separation = self.config.get("big_m_bid_separation")
        self.bid_separation_epsilon = self.config.get("bid_separation_epsilon")

        if feature_matrix_by_player is None:
            raise ValueError("feature_matrix_by_player must be provided")

        self.P_init = p_init
        self.players_config = players_config
        self.strategic_player_id = strategic_player_id
        self.strategic_generators: List[int] = []

        self._extract_scenario_data(scenarios_df, costs_df, ramps_df, pmin_default)

        self.feature_matrix_by_player = feature_matrix_by_player

        self.features: Dict[Tuple[int, int, int], List[float]] = {}
        self.num_policy_features = 0
        if self.feature_matrix_by_player:
            first_player = next(iter(self.feature_matrix_by_player.values()))
            if first_player:
                self.num_policy_features = len(next(iter(first_player.values())))
        if self.num_policy_features == 0:
            raise ValueError("feature_matrix_by_player is empty; cannot infer policy feature dimension")

        if self.strategic_player_id is not None and self.players_config:
            strategic_player = next((p for p in self.players_config if p["id"] == self.strategic_player_id), None)
            if strategic_player:
                self.strategic_generators = strategic_player["controlled_generators"]
            else:
                raise ValueError(f"Strategic player {self.strategic_player_id} not found in players_config")

        self.scenarios_df = scenarios_df
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.model = None

        if self.strategic_player_id is not None:
            self.build_model(self.strategic_player_id)

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

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

    def _extract_scenario_data(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame, ramps_df: pd.DataFrame,
                               pmin_default: float) -> None:
        """Extract scenario and generator data from DataFrames"""
        
        # Auto-detect demand column
        demand_profile_col = None
        for col in scenarios_df.columns:
            if 'demand_profile' in col.lower():
                demand_profile_col = col
                break

        if demand_profile_col is None:
            raise ValueError("No demand profile column found. Expected column name containing 'demand_profile'")
        
        # Auto-detect generator capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'")
        
        # Extract generator information
        self.generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(scenarios_df)
        
        # Infer horizon from explicit time_steps column or first demand profile.
        if 'time_steps' in scenarios_df.columns:
            self.num_time_steps = int(scenarios_df['time_steps'].iloc[0])
        else:
            first_profile = scenarios_df[demand_profile_col].iloc[0]
            if isinstance(first_profile, str):
                first_profile = ast.literal_eval(first_profile)
            self.num_time_steps = len(first_profile)

        if self.num_time_steps <= 0:
            raise ValueError(f"Invalid time_steps value: {self.num_time_steps}")

        # Extract per-scenario data
        self.demand_scenarios: List[List[float]] = []
        self.pmax_scenarios = []
        self.pmin_scenarios = []
        self.bid_scenarios = []

        for _, row in scenarios_df.iterrows():
            demand_profile = self._convert_profile(
                row[demand_profile_col],
                self.num_time_steps,
                demand_profile_col,
            )
            self.demand_scenarios.append(demand_profile)

            pmax_scenario_by_time = []
            pmin_scenario_by_time = []
            bid_scenario_by_time = []

            for t in range(self.num_time_steps):
                pmax_t = []
                pmin_t = []
                bid_t = []

                for gen in self.generator_names:
                    cap = float(row[f"{gen}_cap"])
                    if gen.startswith('W') and f"{gen}_profile" in scenarios_df.columns:
                        wind_profile = self._convert_profile(
                            row[f"{gen}_profile"],
                            self.num_time_steps,
                            f"{gen}_profile",
                        )
                        cap = wind_profile[t]

                    if f"{gen}_bid_profile" in scenarios_df.columns:
                        bid_profile = self._convert_profile(
                            row[f"{gen}_bid_profile"],
                            self.num_time_steps,
                            f"{gen}_bid_profile",
                        )
                        bid_value = bid_profile[t]
                    else:
                        bid_value = float(row[f"{gen}_bid"])

                    pmax_t.append(cap)
                    pmin_t.append(float(pmin_default))
                    bid_t.append(bid_value)

                pmax_scenario_by_time.append(pmax_t)
                pmin_scenario_by_time.append(pmin_t)
                bid_scenario_by_time.append(bid_t)

            self.pmax_scenarios.append(pmax_scenario_by_time)
            self.pmin_scenarios.append(pmin_scenario_by_time)
            self.bid_scenarios.append(bid_scenario_by_time)
        
        # Extract costs
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in self.generator_names]
        
        self.ramp_vector_up = [ramps_df[f"{gen}_ramp_up"].iloc[0] for gen in self.generator_names]
        self.ramp_vector_down = [ramps_df[f"{gen}_ramp_down"].iloc[0] for gen in self.generator_names]

    def build_model(self, strategic_player_id: int) -> None:
        """
        Build a model for the specified strategic player.

        Parameters
        ----------
        strategic_player_id : int
            ID of the player to optimize (must match a player id in players_config)
        """
        strategic_player = next((p for p in self.players_config if p['id'] == strategic_player_id), None)
        if not strategic_player:
            raise ValueError(f"Strategic player {strategic_player_id} not found in players_config")

        self.strategic_player_id = strategic_player_id
        self.strategic_generators = strategic_player['controlled_generators']

        if strategic_player_id not in self.feature_matrix_by_player:
            raise ValueError(
                f"feature_matrix_by_player is missing player {strategic_player_id}. "
                "Precompute and pass feature matrices for all players from the best response driver."
            )

        self.features = self.feature_matrix_by_player[strategic_player_id]
        if self.strategic_player_id is None:
            raise ValueError("Must call build_model(strategic_player_id) before solving model")

        self.model = ConcreteModel()
        
        # Define sets
        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.n_scenarios = Set(initialize=range(self.num_scenarios))
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1)) # For ramp constraints
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps)) 
        
        # Create set of strategic generators
        self.model.strategic_index = Set(initialize=self.strategic_generators)

        # Create set of non-strategic generators  
        non_strategic_gens = [i for i in range(self.num_generators) if i not in self.strategic_generators]
        self.model.non_strategic_index = Set(initialize=non_strategic_gens)

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        """
        Function to build the Pyomo variables for the MPEC model. 
        """
        self._build_upper_level_primal_variables()
        self._build_lower_level_primal_variables()
        self._build_lower_level_dual_variables()
        self._build_complementarity_variables()
        self._build_bid_seperation_variables()
        self._build_policy_variables()

    def _build_upper_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the upper-level problem. 
        """
        #Bid variable for the strategic player(s)
        self.model.alpha = Var(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, domain=Reals)

    def _build_lower_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the lower-level problem. 
        """
        #Production variable for each generator in each scenario
        self.model.P = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=NonNegativeReals)

    def _build_lower_level_dual_variables(self) -> None:
        """
        Function to build the dual variables for the lower-level problem. 
        """
        #Dual variables for the market clearing problem (one per scenario)
        self.model.lambda_var = Var(self.model.n_scenarios, self.model.time_steps, domain=Reals)
        self.model.mu_upper_bound = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=NonNegativeReals)  # Upper bound duals
        self.model.mu_lower_bound = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=NonNegativeReals)  # Lower bound duals
        self.model.mu_ramp_up = Var(self.model.n_scenarios, self.model.time_steps_plus_1, self.model.n_gen, domain=NonNegativeReals)  # Ramp up duals
        self.model.mu_ramp_down = Var(self.model.n_scenarios, self.model.time_steps_plus_1, self.model.n_gen, domain=NonNegativeReals)  # Ramp down duals

    def _build_complementarity_variables(self) -> None:
        """
        Function to build the complementarity variables for the MPEC model. 
        """
        #Complementarity variables for the upper and lower bounds (one per generator per scenario)
        self.model.z_upper_bound = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary)
        self.model.z_lower_bound = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary)
        self.model.z_ramp_up = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary)
        self.model.z_ramp_down = Var(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, domain=Binary)

    def _build_policy_variables(self) -> None:
        """
        Function to build the policy variables for the MPEC model. 
        """
        self.model.n_features = Set(initialize=range(self.num_policy_features))
        self.model.theta = Var(self.model.n_features, domain=Reals)

    def _build_bid_seperation_variables(self) -> None:
        """
        Function to build the bid separation variables for the MPEC model.
        """
        
        #Binary bid separation variables for strategic players vs competitors
        self.model.tau = Var(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, self.model.non_strategic_index, domain=Binary)

    def _build_objective(self) -> None:
        """
        Function to build the objective function for the MPEC model.
        Minimizes the negative profit of the strategic player across all scenarios
        (equivalent to maximizing total profit).
        """

        self.model.objective = Objective(expr =
                                         1 / self.num_scenarios * 
                                         sum(
                                            sum(
                                                self.model.lambda_var[s, t] * self.demand_scenarios[s][t]  
                                                    + sum(
                                                          -self.bid_scenarios[s][t][i] * self.model.P[s, t, i] 
                                                          for i in self.model.non_strategic_index) 
                                                    + sum(-self.model.mu_upper_bound[s, t, i] * self.pmax_scenarios[s][t][i]
                                                          +self.model.mu_upper_bound[s, t, i] * self.pmin_scenarios[s][t][i]
                                                          -self.model.mu_ramp_up[s, t, i]     * self.ramp_vector_up[i]
                                                          -self.model.mu_ramp_down[s, t, i]   * self.ramp_vector_down[i]
                                                          for i in self.model.n_gen)
                                                for t in self.model.time_steps)
                                                # Initial conditions for ramp constraints (t=0)
                                                + sum(
                                                    -self.model.mu_ramp_up[s, 0, i]   * self.P_init[s][i]
                                                    +self.model.mu_ramp_down[s, 0, i] * self.P_init[s][i]
                                                    for i in self.model.n_gen)
                                          + sum(
                                                sum(
                                                    self.model.mu_upper_bound[s, t, i] * self.pmax_scenarios[s][t][i]
                                                   -self.model.mu_lower_bound[s, t, i] * self.pmin_scenarios[s][t][i]
                                                    for i in self.model.strategic_index)
                                                for t in self.model.time_steps)
                                          + sum(self.model.mu_ramp_up[s, 0, i]   * (self.P_init[s][i] + self.ramp_vector_up[i])
                                               -self.model.mu_ramp_down[s, 0, i] * (self.P_init[s][i] - self.ramp_vector_down[i])
                                                +sum(self.model.mu_ramp_up[s, t, i]   * self.ramp_vector_up[i]
                                                    +self.model.mu_ramp_down[s, t, i] * self.ramp_vector_down[i]
                                                     for t in range(1, self.num_time_steps))
                                                for i in self.model.strategic_index)   
                                         + sum(
                                               sum(self.cost_vector[i] * self.model.P[s, t, i]
                                                   for i in self.model.strategic_index) 
                                               for t in self.model.time_steps)
                                             for s in self.model.n_scenarios)
 
                                            , sense=minimize
        )

    def _build_constraints(self) -> None:
        """
        Function to build all constraints for the MPEC model. 
        """
        self._build_upper_level_constraints()
        self._build_lower_level_constraints()
        self._build_KKT_stationarity_constraints()
        self._build_KKT_complementarity_constraints()
        self._build_bid_seperation_constraints()
        self._build_policy_constraints()

    def _build_upper_level_constraints(self) -> None:
        """
        Function to build the upper-level constraints for the MPEC model. 
        """
        def min_bid_rule(model, s, t, i): 
            return model.alpha[s, t, i] >= self.alpha_min
        
        def max_bid_rule(model, s, t, i):
            return model.alpha[s, t, i] <= self.alpha_max
        
        self.model.min_bid_constraint = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=min_bid_rule)
        self.model.max_bid_constraint = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=max_bid_rule)

    def _build_lower_level_constraints(self) -> None:
        """
        Function to build the lower-level constraints for the MPEC model. 
        """

        def power_balance_rule(m, s, t):
            return sum(m.P[s, t, i] for i in m.n_gen) - self.demand_scenarios[s][t] == 0
        
        def generation_upper_rule(m, s, t, i):
            return m.P[s, t, i] - self.pmax_scenarios[s][t][i] <= 0 

        def generation_lower_rule(m, s, t, i):
            return -m.P[s, t, i] + self.pmin_scenarios[s][t][i] <= 0
        
        def ramp_up_rule(m, s, t, i):
            return m.P[s, t, i] - m.P[s, t-1, i] - self.ramp_vector_up[i] <= 0
        
        def ramp_up_initial_rule(m, s, i):
            return m.P[s, 0, i] - self.P_init[s][i] - self.ramp_vector_up[i] <= 0
        
        def ramp_down_rule(m, s, t, i):
            return -m.P[s, t, i] + m.P[s, t-1, i] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_rule(m, s, i):
            return - m.P[s, 0, i] + self.P_init[s][i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint = Constraint(self.model.n_scenarios, self.model.time_steps, rule=power_balance_rule)
        self.model.generation_upper_bound_constraints = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=generation_upper_rule)
        self.model.generation_lower_bound_constraints = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=generation_lower_rule)
        self.model.ramp_up_constraints = Constraint(self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_up_rule)
        self.model.ramp_down_constraints = Constraint(self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_down_rule)
        self.model.ramp_up_initial_feasibility_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=ramp_up_initial_rule)
        self.model.ramp_down_initial_feasibility_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=ramp_down_initial_rule)

    def _build_KKT_stationarity_constraints(self) -> None:
        """
        Function to build the KKT stationarity constraints for the MPEC model. 
        """
        def stationarity_rule_strategic_agent(m, s, t, i):
            return m.alpha[s, t, i] - m.lambda_var[s, t] + m.mu_upper_bound[s, t, i] - m.mu_lower_bound[s, t, i] + m.mu_ramp_up[s, t, i] - m.mu_ramp_up[s, t+1, i] - m.mu_ramp_down[s, t, i] + m.mu_ramp_down[s, t+1, i] == 0

        def stationarity_rule_non_strategic_agents(m, s, t, i):
            return self.bid_scenarios[s][t][i] - m.lambda_var[s, t] + m.mu_upper_bound[s, t, i] - m.mu_lower_bound[s, t, i] + m.mu_ramp_up[s, t, i] - m.mu_ramp_up[s, t+1, i] - m.mu_ramp_down[s, t, i] + m.mu_ramp_down[s, t+1, i] == 0

        def final_ramp_up_dual_rule(m, s, i):
            return m.mu_ramp_up[s, self.num_time_steps, i] == 0

        def final_ramp_down_dual_rule(m, s, i):
            return m.mu_ramp_down[s, self.num_time_steps, i] == 0

        self.model.stationarity_constraint_strategic = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=stationarity_rule_strategic_agent)
        self.model.stationarity_constraint_non_strategic = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.non_strategic_index, rule=stationarity_rule_non_strategic_agents)
        self.model.final_ramp_up_dual_constraint = Constraint(self.model.n_scenarios, self.model.n_gen, rule=final_ramp_up_dual_rule)
        self.model.final_ramp_down_dual_constraint = Constraint(self.model.n_scenarios, self.model.n_gen, rule=final_ramp_down_dual_rule)

    def _build_KKT_complementarity_constraints(self) -> None:
        """
        Function to build the KKT complementarity constraints for the MPEC model. 
        """
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_rule(m, s, t, i):
            return -BigM * (1 - m.z_upper_bound[s, t, i]) <= m.P[s, t, i] - self.pmax_scenarios[s][t][i]  

        def upper_bound_complementarity_rule_dual(m, s, t, i):
            return m.mu_upper_bound[s, t, i] <= BigM * m.z_upper_bound[s, t, i] 
        
        def lower_bound_complementarity_rule(m, s, t, i):
            return -BigM * (1 - m.z_lower_bound[s, t, i]) <= -m.P[s, t, i] + self.pmin_scenarios[s][t][i]

        def lower_bound_complementarity_rule_dual(m, s, t, i):
            return m.mu_lower_bound[s, t, i] <= BigM * m.z_lower_bound[s, t, i] 

        def ramp_up_complementarity_rule(m, s, t, i):
            return -BigM * (1 - m.z_ramp_up[s, t, i]) <= m.P[s, t, i] - m.P[s, t-1, i] - self.ramp_vector_up[i]
        
        def ramp_up_complementarity_initial_rule(m, s, i):
            return -BigM * (1 - m.z_ramp_up[s, 0, i]) <= m.P[s, 0, i] - self.P_init[s][i] - self.ramp_vector_up[i]

        def ramp_up_complementarity_rule_dual(m, s, t, i):
            return m.mu_ramp_up[s, t, i] <= BigM * m.z_ramp_up[s, t, i]

        def ramp_down_complementarity_rule(m, s, t, i):
            return -BigM * (1 - m.z_ramp_down[s, t, i]) <= - m.P[s, t, i] + m.P[s, t-1, i] - self.ramp_vector_down[i]

        def ramp_down_complementarity_initial_rule(m, s, i):
            return -BigM * (1 - m.z_ramp_down[s, 0, i]) <= - m.P[s, 0, i] + self.P_init[s][i] - self.ramp_vector_down[i]

        def ramp_down_complementarity_rule_dual(m, s, t, i):
            return m.mu_ramp_down[s, t, i] <= BigM * m.z_ramp_down[s, t, i]

        self.model.upper_bound_complementarity_constraints = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=upper_bound_complementarity_rule)
        self.model.upper_bound_complementarity_constraints_dual = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=upper_bound_complementarity_rule_dual)
        self.model.lower_bound_complementarity_constraints = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=lower_bound_complementarity_rule)
        self.model.lower_bound_complementarity_constraints_dual = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=lower_bound_complementarity_rule_dual)

        self.model.ramp_up_complementarity_constraints = Constraint(self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_up_complementarity_rule)
        self.model.ramp_up_complementarity_initial_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=ramp_up_complementarity_initial_rule)
        self.model.ramp_up_complementarity_constraints_dual = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=ramp_up_complementarity_rule_dual)
        
        self.model.ramp_down_complementarity_constraints = Constraint(self.model.n_scenarios, self.model.time_steps_minus_1, self.model.n_gen, rule=ramp_down_complementarity_rule)
        self.model.ramp_down_complementarity_initial_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=ramp_down_complementarity_initial_rule)
        self.model.ramp_down_complementarity_constraints_dual = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.n_gen, rule=ramp_down_complementarity_rule_dual)

    def _build_bid_seperation_constraints(self) -> None:
        """
        Function to build the bid seperation constraints for the MPEC model. 
        """
        BigM = self.big_m_bid_separation
        epsilon = self.bid_separation_epsilon

        def alpha_upper_seperation(m, s, t, i, k):
            return m.alpha[s, t, i] >= self.bid_scenarios[s][t][k] + epsilon - BigM * (1 - m.tau[s, t, i, k])
        
        def alpha_lower_seperation(m, s, t, i, k):
            return m.alpha[s, t, i] <= self.bid_scenarios[s][t][k] - epsilon + BigM * m.tau[s, t, i, k]

        self.model.alpha_upper_seperation_constraints = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, self.model.non_strategic_index, rule=alpha_upper_seperation)
        self.model.alpha_lower_seperation_constraints = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, self.model.non_strategic_index, rule=alpha_lower_seperation)

    def _build_policy_constraints(self) -> None:
        """
        Function to build the policy constraints for the MPEC model. 
        """
        def policy_rule(m, s, t, i):
            phi = self.features[s, t, i]
            return m.alpha[s, t, i] == sum(m.theta[k] * float(phi[k]) for k in m.n_features)
        self.model.policy_constraint = Constraint(self.model.n_scenarios, self.model.time_steps, self.model.strategic_index, rule=policy_rule)

    def solve(self) -> None:
        """
        Solve the optimization model.
        """

        # Create solver
        solver = SolverFactory("gurobi")

        # Solve
        results = solver.solve(self.model, tee=False)

        # Check solver status
        if not (results.solver.status == 'ok') and not (results.solver.termination_condition == 'optimal'):
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)
        # else:
            # print("Model solved successfully!")
    
    def get_optimal_bids(self) -> List[List[List[float]]]:
        """
        Extract optimal strategic bids from the solved model.
        
        Returns
        -------
        List[List[List[float]]]
            Complete bid tensor where optimal_bids[s][t][i] is the bid for generator i
            in scenario s at time t. Strategic generators get optimal alpha values,
            non-strategic generators keep their original bids.
        """        
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been built yet. Call build_model(strategic_player_id) first.")
            
        if not hasattr(self.model, 'alpha'):
            raise ValueError("Strategic bid variables (alpha) not found. Model may not be properly built.")
            
        # Get strategic generator indices
        strategic_indices = self.strategic_generators
            
        # Initialize with current multi-period bids.
        optimal_bid_scenarios = [
            [list(self.bid_scenarios[s][t]) for t in range(self.num_time_steps)]
            for s in range(self.num_scenarios)
        ]
        
        # Extract optimal alpha values for strategic generators
        for s in range(self.num_scenarios):
            for t in range(self.num_time_steps):
                for i in strategic_indices:
                    try:
                        alpha_value = self.model.alpha[s, t, i].value
                        if alpha_value is not None:
                            optimal_bid_scenarios[s][t][i] = float(alpha_value)
                        else:
                            print(f"Warning: Alpha value for scenario {s}, time {t}, generator {i} is None")
                    except KeyError:
                        print(f"Warning: Alpha variable for scenario {s}, time {t}, generator {i} not found")
                    
        return optimal_bid_scenarios
    
    def get_optimal_theta(self) -> np.ndarray:
        if self.model is None or not hasattr(self.model, "theta"):
            raise ValueError("Policy variables (theta) not available. Build and solve first.")

        return np.array([self.model.theta[k].value for k in self.model.n_features], dtype=np.float64)

    def update_bids_with_optimal_values(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update bid scenarios with optimal strategic bids and return updated DataFrame.
        
        Parameters
        ----------
        scenarios_df : pd.DataFrame, optional
            Original scenarios DataFrame to update. If None, creates new DataFrame 
            with updated bids.
            
        Returns
        -------
        pd.DataFrame
            Updated scenarios DataFrame with optimal strategic bids
        """
        # Get optimal multi-period bids
        optimal_bid_scenarios = self.get_optimal_bids()
        
        # Get strategic indices
        strategic_indices = self.strategic_generators
        
        # Create update scenarios DataFrame
        updated_df = scenarios_df.copy()        
            
        # Update bid columns in the DataFrame
        for gen_idx in strategic_indices:
            gen_name = self.generator_names[gen_idx]
            bid_profile_col = f"{gen_name}_bid_profile"

            # if bid_profile_col not in updated_df.columns:
            #     updated_df[bid_profile_col] = [None] * len(updated_df)

            for s in range(self.num_scenarios):
                profile = [float(optimal_bid_scenarios[s][t][gen_idx]) for t in range(self.num_time_steps)]
                updated_df.at[s, bid_profile_col] = profile
                    
        return updated_df

    def get_scenario_profits(self) -> List[float]:
        """
        Calculate the profit for the strategic player in each scenario based on the optimal bids and dispatch.
        
        Returns
        -------
        List[float]
            List of profits for the strategic player in each scenario
        """
        if not hasattr(self.model, 'lambda_var'):
            raise ValueError("Market clearing price variable (lambda_var) not found. Model may not be properly built.")
        
        profits = []
        for s in range(self.num_scenarios):
            profit_scenario = 0.0
            for t in range(self.num_time_steps):
                lambda_value = self.model.lambda_var[s, t].value if self.model.lambda_var[s, t].value is not None else 0.0
                for i in self.strategic_generators:
                    dispatch = self.model.P[s, t, i].value if self.model.P[s, t, i].value is not None else 0.0
                    cost = self.cost_vector[i]
                    profit_scenario += (lambda_value * dispatch - cost * dispatch)
            profits.append(float(profit_scenario))
        
        return profits

    def print_players_summary(self) -> None:
        """
        Print a summary of all players and their controlled generators.
        """
        if not self.players_config:
            print("No players configuration loaded.")
            return
            
        print(f"\n=== Players Configuration Summary ===")
        print(f"Total Players: {len(self.players_config)}")
        
        for player in self.players_config:
            player_name = player['id']
            controlled_gens = player['controlled_generators']
            gen_names = [self.generator_names[i] if i < len(self.generator_names) else f"Gen{i}" 
                        for i in controlled_gens]
            print(f"Player {player_name}: Controls {len(controlled_gens)} generators - {gen_names}")

