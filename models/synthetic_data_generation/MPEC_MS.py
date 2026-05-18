from pyomo.environ import *
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
import pandas as pd
import numpy as np
import os
import math
import warnings
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.diagonalization.intertemporal.MultipleScenarios.utilities.MPEC_utils import get_mpec_parameters
from models.helper import find_demand_profile_column, infer_num_time_steps, parse_profile_exact_length


def run_fbbt_for_big_m(model: ConcreteModel, logger=None) -> Any:
    """
    Run Pyomo FBBT before Big-M construction.

    FBBT propagates current variable bounds through constraints. Here it is used
    primarily to tighten primal slack Big-M values. It does not guarantee
    globally minimal Big-M values, and it should not be treated as a substitute
    for economic/manual dual bounds.
    """
    try:
        result = fbbt(
            model,
            max_iter=20,
            feasibility_tol=1e-8,
            improvement_tol=1e-6,
        )
        if logger is not None:
            logger.info("FBBT completed successfully.")
        return result
    except Exception as exc:
        raise RuntimeError(f"FBBT failed before Big-M construction: {exc}") from exc


def safe_expr_upper_bound(
    expr: Any,
    name: str,
    min_value: float = 0.0,
    fallback: Optional[float] = None,
    stats: Optional[Dict[str, int]] = None,
) -> float:
    """
    Compute a safe finite upper bound for a Big-M slack expression.

    If FBBT/expression bounds cannot provide a finite upper bound, the configured
    global Big-M is used only as an explicit fallback and a warning is emitted.
    """
    lb, ub = compute_bounds_on_expr(expr)

    try:
        ub_value = None if ub is None else float(value(ub))
    except Exception:
        ub_value = None

    if ub_value is None or math.isnan(ub_value) or not math.isfinite(ub_value):
        if stats is not None:
            stats["missing_bounds"] = stats.get("missing_bounds", 0) + 1
        if fallback is None:
            raise ValueError(f"No finite upper bound for Big-M expression {name}")
        if stats is not None:
            stats["fallback_count"] = stats.get("fallback_count", 0) + 1
        warnings.warn(
            f"No finite upper bound for Big-M expression {name}; using fallback {fallback}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return max(min_value, float(fallback))

    if ub_value < 0.0:
        if ub_value > -1e-7:
            ub_value = 0.0
        else:
            raise ValueError(
                f"Computed negative upper bound {ub_value} for nonnegative slack expression {name}"
            )

    return max(min_value, ub_value)


class MPECModel:
    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        p_init: Any,
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
        NN_nodes : int, optional
            Number of nodes in the neural network policy (if using NN-based policy)
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
        self.big_m_activation = self.config.get("big_m_activation")
        if self.config.get("big_m_method") is not None:
            self.big_m_method = str(self.config.get("big_m_method")).lower()
        else:
            self.big_m_method = "fbbt" if self.config.get("use_fbbt_big_m", True) else "global"
        if self.big_m_method not in {"global", "fbbt", "manual"}:
            raise ValueError(
                f"Invalid big_m_method '{self.big_m_method}'. Expected one of: global, fbbt, manual."
            )
        self.manual_big_m_values = self.config.get("manual_big_m_values", {})
        self.big_m_floor = float(self.config.get("big_m_floor", 1e-8))
        self.print_big_m_summary = bool(self.config.get("print_big_m_summary", True))
        
        self.P_init = p_init
        self.players_config = players_config
        self.pmin_default = pmin_default

        self._extract_scenario_data(scenarios_df, costs_df, ramps_df, pmin_default)

        self.scenarios_df = scenarios_df
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.model = None
        self.scenario_index: Optional[int] = None
        self._parallel_optimal_bid_scenarios: Optional[List[List[List[float]]]] = None
        self._parallel_scenario_profits: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_profile(value: Any, expected_len: int, column_name: str) -> List[float]:
        """Convert profile-like input into a numeric list with expected length."""
        return parse_profile_exact_length(value, expected_len, column_name)

    def _extract_scenario_data(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame, ramps_df: pd.DataFrame,
                               pmin_default: float) -> None:
        """Extract scenario and generator data from DataFrames"""
        
        # Auto-detect demand column
        demand_profile_col = find_demand_profile_column(scenarios_df)
        
        # Auto-detect generator capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'")
        
        # Extract generator information
        self.generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(scenarios_df)
        
        # Infer horizon from explicit time_steps column or first demand profile.
        self.num_time_steps = infer_num_time_steps(scenarios_df)

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
        self.cost_vector = [float(costs_df[f"{gen}_cost"].iloc[0]) for gen in self.generator_names]
        self.ramp_vector_up = [float(ramps_df[f"{gen}_ramp_up"].iloc[0]) for gen in self.generator_names]
        self.ramp_vector_down = [float(ramps_df[f"{gen}_ramp_down"].iloc[0]) for gen in self.generator_names]

    def build_model(self, strategic_player_id: int, scenario_index: Optional[int] = None) -> None:
        """
        Build a model for the specified strategic player.

        Parameters
        ----------
        strategic_player_id : int
            ID of the player to optimize (must match a player id in players_config)
        scenario_index : int, optional
            Scenario to build. If omitted for multiple scenarios, solve() will build
            and solve one independent model per scenario in parallel.
        """
        strategic_player = next((p for p in self.players_config if p['id'] == strategic_player_id), None)
        if not strategic_player:
            raise ValueError(f"Strategic player {strategic_player_id} not found in players_config")

        self.strategic_player_id = strategic_player_id
        self.strategic_generators = strategic_player['controlled_generators']
        self.scenario_index = scenario_index
        self._parallel_optimal_bid_scenarios = None
        self._parallel_scenario_profits = None

        if self.strategic_player_id is None:
            raise ValueError("Must call build_model(strategic_player_id) before solving model")

        if self.scenario_index is None:
            if self.num_scenarios > 1:
                self.model = None
                return
            self.scenario_index = 0

        if self.scenario_index < 0 or self.scenario_index >= self.num_scenarios:
            raise ValueError(f"scenario_index {self.scenario_index} is out of range")

        self.model = ConcreteModel()
        
        # Define sets
        self.model.n_gen = Set(initialize=range(self.num_generators))
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
        # self._build_bid_seperation_variables()

    def _build_upper_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the upper-level problem. 
        """
        #Bid variable for the strategic player(s)
        self.model.alpha = Var(self.model.strategic_index, self.model.time_steps, domain=Reals)

    def _build_lower_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the lower-level problem. 
        """
        #Production variable for each generator in this scenario
        self.model.P = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)

    def _build_lower_level_dual_variables(self) -> None:
        """
        Function to build the dual variables for the lower-level problem. 
        """
        #Dual variables for the market clearing problem
        self.model.lambda_var = Var(self.model.time_steps, domain=Reals)
        self.model.mu_upper_bound = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)  # Upper bound duals
        self.model.mu_lower_bound = Var(self.model.n_gen, self.model.time_steps, domain=NonNegativeReals)  # Lower bound duals
        self.model.mu_ramp_up = Var(self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)  # Ramp up duals
        self.model.mu_ramp_down = Var(self.model.n_gen, self.model.time_steps_plus_1, domain=NonNegativeReals)  # Ramp down duals

    def _build_complementarity_variables(self) -> None:
        """
        Function to build the complementarity variables for the MPEC model. 
        """
        #Complementarity variables for the upper and lower bounds
        self.model.z_upper_bound = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_lower_bound = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up = Var(self.model.n_gen, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down = Var(self.model.n_gen, self.model.time_steps, domain=Binary)

    def _build_bid_seperation_variables(self) -> None:
        """
        Function to build the bid separation variables for the MPEC model.
        """
        
        #Binary bid separation variables for strategic players vs competitors
        self.model.tau = Var(self.model.strategic_index, self.model.non_strategic_index, self.model.time_steps, domain=Binary)

    def _build_objective(self) -> None:
        """
        Function to build the objective function for the MPEC model.
        Minimizes the negative profit of the strategic player across all scenarios
        (equivalent to maximizing total profit).
        """
        s = self.scenario_index

        self.model.objective = Objective(expr =
                                            sum(
                                                self.model.lambda_var[t] * self.demand_scenarios[s][t]
                                                    + sum(
                                                          -self.bid_scenarios[s][t][i] * self.model.P[i, t]
                                                          for i in self.model.non_strategic_index) 
                                                    + sum(-self.model.mu_upper_bound[i, t] * self.pmax_scenarios[s][t][i]
                                                          +self.model.mu_upper_bound[i, t] * self.pmin_scenarios[s][t][i]
                                                          -self.model.mu_ramp_up[i, t]     * self.ramp_vector_up[i]
                                                          -self.model.mu_ramp_down[i, t]   * self.ramp_vector_down[i]
                                                          for i in self.model.n_gen)
                                                for t in self.model.time_steps)
                                                # Initial conditions for ramp constraints (t=0)
                                                + sum(
                                                    -self.model.mu_ramp_up[i, 0]   * self.P_init[s][i]
                                                    +self.model.mu_ramp_down[i, 0] * self.P_init[s][i]
                                                    for i in self.model.n_gen)
                                          + sum(
                                                sum(
                                                    self.model.mu_upper_bound[i, t] * self.pmax_scenarios[s][t][i]
                                                   -self.model.mu_lower_bound[i, t] * self.pmin_scenarios[s][t][i]
                                                    for i in self.model.strategic_index)
                                                for t in self.model.time_steps)
                                          + sum(self.model.mu_ramp_up[i, 0]   * (self.P_init[s][i] + self.ramp_vector_up[i])
                                               -self.model.mu_ramp_down[i, 0] * (self.P_init[s][i] - self.ramp_vector_down[i])
                                                +sum(self.model.mu_ramp_up[i, t]   * self.ramp_vector_up[i]
                                                    +self.model.mu_ramp_down[i, t] * self.ramp_vector_down[i]
                                                     for t in range(1, self.num_time_steps))
                                                for i in self.model.strategic_index)   
                                         + sum(
                                               sum(self.cost_vector[i] * self.model.P[i, t]
                                                   for i in self.model.strategic_index) 
                                               for t in self.model.time_steps)
 
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
        # self._build_bid_seperation_constraints()

    def _build_upper_level_constraints(self) -> None:
        """
        Function to build the upper-level constraints for the MPEC model. 
        """
        def min_bid_rule(model, i, t): 
            return model.alpha[i, t] >= self.alpha_min
        
        def max_bid_rule(model, i, t):
            return model.alpha[i, t] <= self.alpha_max
        
        def tmp_rule(model, i, t):
            return model.alpha[i, t] <= 2 * self.cost_vector[i]

        self.model.min_bid_constraint = Constraint(self.model.strategic_index, self.model.time_steps, rule=min_bid_rule)
        self.model.max_bid_constraint = Constraint(self.model.strategic_index, self.model.time_steps, rule=max_bid_rule)

        self.model.tmp_rule = Constraint(self.model.strategic_index, self.model.time_steps, rule=tmp_rule)

    def _build_lower_level_constraints(self) -> None:
        """
        Function to build the lower-level constraints for the MPEC model. 
        """
        s = self.scenario_index

        def power_balance_rule(m, t):
            return sum(m.P[i, t] for i in m.n_gen) - self.demand_scenarios[s][t] == 0
        
        def generation_upper_rule(m, i, t):
            return m.P[i, t] - self.pmax_scenarios[s][t][i] <= 0

        def generation_lower_rule(m, i, t):
            return -m.P[i, t] + self.pmin_scenarios[s][t][i] <= 0
        
        def ramp_up_rule(m, i, t):
            return m.P[i, t] - m.P[i, t-1] - self.ramp_vector_up[i] <= 0
        
        def ramp_up_initial_rule(m, i):
            return m.P[i, 0] - self.P_init[s][i] - self.ramp_vector_up[i] <= 0
        
        def ramp_down_rule(m, i, t):
            return -m.P[i, t] + m.P[i, t-1] - self.ramp_vector_down[i] <= 0

        def ramp_down_initial_rule(m, i):
            return - m.P[i, 0] + self.P_init[s][i] - self.ramp_vector_down[i] <= 0

        self.model.power_balance_constraint = Constraint(self.model.time_steps, rule=power_balance_rule)
        self.model.generation_upper_bound_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=generation_upper_rule)
        self.model.generation_lower_bound_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=generation_lower_rule)
        self.model.ramp_up_constraints = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_rule)
        self.model.ramp_down_constraints = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_rule)
        self.model.ramp_up_initial_feasibility_constraints = Constraint(self.model.n_gen, rule=ramp_up_initial_rule)
        self.model.ramp_down_initial_feasibility_constraints = Constraint(self.model.n_gen, rule=ramp_down_initial_rule)

    def _build_KKT_stationarity_constraints(self) -> None:
        """
        Function to build the KKT stationarity constraints for the MPEC model. 
        """
        s = self.scenario_index

        def stationarity_rule_strategic_agent(m, i, t):
            return m.alpha[i, t] - m.lambda_var[t] + m.mu_upper_bound[i, t] - m.mu_lower_bound[i, t] + m.mu_ramp_up[i, t] - m.mu_ramp_up[i, t+1] - m.mu_ramp_down[i, t] + m.mu_ramp_down[i, t+1] == 0

        def stationarity_rule_non_strategic_agents(m, i, t):
            return self.bid_scenarios[s][t][i] - m.lambda_var[t] + m.mu_upper_bound[i, t] - m.mu_lower_bound[i, t] + m.mu_ramp_up[i, t] - m.mu_ramp_up[i, t+1] - m.mu_ramp_down[i, t] + m.mu_ramp_down[i, t+1] == 0

        def final_ramp_up_dual_rule(m, i):
            return m.mu_ramp_up[i, self.num_time_steps] == 0

        def final_ramp_down_dual_rule(m, i):
            return m.mu_ramp_down[i, self.num_time_steps] == 0

        self.model.stationarity_constraint_strategic = Constraint(self.model.strategic_index, self.model.time_steps, rule=stationarity_rule_strategic_agent)
        self.model.stationarity_constraint_non_strategic = Constraint(self.model.non_strategic_index, self.model.time_steps, rule=stationarity_rule_non_strategic_agents)
        self.model.final_ramp_up_dual_constraint = Constraint(self.model.n_gen, rule=final_ramp_up_dual_rule)
        self.model.final_ramp_down_dual_constraint = Constraint(self.model.n_gen, rule=final_ramp_down_dual_rule)

    def _upper_slack_expr(self, m, i: int, t: int):
        s = self.scenario_index
        return self.pmax_scenarios[s][t][i] - m.P[i, t]

    def _lower_slack_expr(self, m, i: int, t: int):
        s = self.scenario_index
        return m.P[i, t] - self.pmin_scenarios[s][t][i]

    def _ramp_up_slack_expr(self, m, i: int, t: int):
        if t == 0:
            return self.P_init[self.scenario_index][i] + self.ramp_vector_up[i] - m.P[i, 0]
        return self.ramp_vector_up[i] - (m.P[i, t] - m.P[i, t - 1])

    def _ramp_down_slack_expr(self, m, i: int, t: int):
        if t == 0:
            return m.P[i, 0] - self.P_init[self.scenario_index][i] + self.ramp_vector_down[i]
        return self.ramp_vector_down[i] - (m.P[i, t - 1] - m.P[i, t])

    def _manual_big_m_lookup(self, name: str, index: Tuple[int, int], fallback: float) -> float:
        values = self.manual_big_m_values.get(name, {})
        if not values:
            warnings.warn(
                f"No manual Big-M dictionary supplied for {name}; using fallback {fallback}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float(fallback)

        key_options = [
            index,
            str(index),
            f"{index[0]},{index[1]}",
            f"{index[0]}_{index[1]}",
        ]
        for key in key_options:
            if key in values:
                return max(self.big_m_floor, float(values[key]))

        warnings.warn(
            f"No manual Big-M value supplied for {name}{index}; using fallback {fallback}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(fallback)

    def _initialize_big_m_storage(self) -> None:
        self.model._M_upper_slack = {}
        self.model._M_lower_slack = {}
        self.model._M_ramp_up_slack = {}
        self.model._M_ramp_down_slack = {}
        self.model._M_upper_dual = {}
        self.model._M_lower_dual = {}
        self.model._M_ramp_up_dual = {}
        self.model._M_ramp_down_dual = {}
        self.model._big_m_stats = {}

    def _store_global_big_m_values(self) -> None:
        fallback = float(self.big_m_complementarity)
        for i in self.model.n_gen:
            for t in self.model.time_steps:
                idx = (i, t)
                self.model._M_upper_slack[idx] = fallback
                self.model._M_lower_slack[idx] = fallback
                self.model._M_ramp_up_slack[idx] = fallback
                self.model._M_ramp_down_slack[idx] = fallback
                self.model._M_upper_dual[idx] = fallback
                self.model._M_lower_dual[idx] = fallback
                self.model._M_ramp_up_dual[idx] = fallback
                self.model._M_ramp_down_dual[idx] = fallback

    def _store_manual_big_m_values(self) -> None:
        fallback = float(self.big_m_complementarity)
        for i in self.model.n_gen:
            for t in self.model.time_steps:
                idx = (i, t)
                self.model._M_upper_slack[idx] = self._manual_big_m_lookup("upper_slack", idx, fallback)
                self.model._M_lower_slack[idx] = self._manual_big_m_lookup("lower_slack", idx, fallback)
                self.model._M_ramp_up_slack[idx] = self._manual_big_m_lookup("ramp_up_slack", idx, fallback)
                self.model._M_ramp_down_slack[idx] = self._manual_big_m_lookup("ramp_down_slack", idx, fallback)
                self.model._M_upper_dual[idx] = self._manual_big_m_lookup("upper_dual", idx, fallback)
                self.model._M_lower_dual[idx] = self._manual_big_m_lookup("lower_dual", idx, fallback)
                self.model._M_ramp_up_dual[idx] = self._manual_big_m_lookup("ramp_up_dual", idx, fallback)
                self.model._M_ramp_down_dual[idx] = self._manual_big_m_lookup("ramp_down_dual", idx, fallback)

    def build_fbbt_slack_big_m_values(self) -> None:
        """
        Compute expression-specific Big-M values for primal slack expressions.

        FBBT tightens variable bounds before ``compute_bounds_on_expr`` is used
        on each complementarity slack. Dual Big-M values remain configured/global
        because reliable dual bounds should come from economic information such
        as bid ranges, price bounds, and stationarity structure.
        """
        run_fbbt_for_big_m(self.model)
        fallback = float(self.big_m_complementarity)

        bound_specs = [
            ("upper_generation_slack", self.model._M_upper_slack, self._upper_slack_expr),
            ("lower_generation_slack", self.model._M_lower_slack, self._lower_slack_expr),
            ("ramp_up_slack", self.model._M_ramp_up_slack, self._ramp_up_slack_expr),
            ("ramp_down_slack", self.model._M_ramp_down_slack, self._ramp_down_slack_expr),
        ]

        for label, storage, expr_builder in bound_specs:
            stats = {"fallback_count": 0, "missing_bounds": 0}
            for i in self.model.n_gen:
                for t in self.model.time_steps:
                    idx = (i, t)
                    storage[idx] = safe_expr_upper_bound(
                        expr_builder(self.model, i, t),
                        f"{label}{idx}",
                        min_value=self.big_m_floor,
                        fallback=fallback,
                        stats=stats,
                    )
            self.model._big_m_stats[label] = stats

        for i in self.model.n_gen:
            for t in self.model.time_steps:
                idx = (i, t)
                self.model._M_upper_dual[idx] = fallback
                self.model._M_lower_dual[idx] = fallback
                self.model._M_ramp_up_dual[idx] = fallback
                self.model._M_ramp_down_dual[idx] = fallback

    def _prepare_big_m_values(self) -> None:
        self._initialize_big_m_storage()
        if self.big_m_method == "global":
            self._store_global_big_m_values()
        elif self.big_m_method == "manual":
            self._store_manual_big_m_values()
        elif self.big_m_method == "fbbt":
            self.build_fbbt_slack_big_m_values()
        else:
            raise ValueError(f"Unknown big_m_method '{self.big_m_method}'")

        if self.print_big_m_summary:
            self._print_big_m_summary()

    def _print_big_m_summary(self) -> None:
        def summarize(label: str, values: Dict[Tuple[int, int], float]) -> None:
            data = [float(v) for v in values.values()]
            stats = self.model._big_m_stats.get(label, {})
            fallback_count = stats.get("fallback_count", 0)
            missing_bounds = stats.get("missing_bounds", 0)
            if not data:
                print(f"{label} Big-M: no values")
                return
            print(
                f"{label} Big-M: "
                f"min={min(data):.6g}, max={max(data):.6g}, mean={float(np.mean(data)):.6g}, "
                f"fallback_count={fallback_count}, missing_bounds={missing_bounds}"
            )

        print(f"Big-M method: {self.big_m_method}")
        summarize("upper_generation_slack", self.model._M_upper_slack)
        summarize("lower_generation_slack", self.model._M_lower_slack)
        summarize("ramp_up_slack", self.model._M_ramp_up_slack)
        summarize("ramp_down_slack", self.model._M_ramp_down_slack)

    def _build_KKT_complementarity_constraints(self) -> None:
        """
        Function to build the KKT complementarity constraints for the MPEC model. 
        """
        self._prepare_big_m_values()

        def upper_bound_complementarity_rule(m, i, t):
            return self._upper_slack_expr(m, i, t) <= m._M_upper_slack[(i, t)] * m.z_upper_bound[i, t]

        def upper_bound_complementarity_rule_dual(m, i, t):
            return m.mu_upper_bound[i, t] <= m._M_upper_dual[(i, t)] * (1 - m.z_upper_bound[i, t])
        
        def lower_bound_complementarity_rule(m, i, t):
            return self._lower_slack_expr(m, i, t) <= m._M_lower_slack[(i, t)] * m.z_lower_bound[i, t]

        def lower_bound_complementarity_rule_dual(m, i, t):
            return m.mu_lower_bound[i, t] <= m._M_lower_dual[(i, t)] * (1 - m.z_lower_bound[i, t])

        def ramp_up_complementarity_rule(m, i, t):
            return self._ramp_up_slack_expr(m, i, t) <= m._M_ramp_up_slack[(i, t)] * m.z_ramp_up[i, t]
        
        def ramp_up_complementarity_initial_rule(m, i):
            return self._ramp_up_slack_expr(m, i, 0) <= m._M_ramp_up_slack[(i, 0)] * m.z_ramp_up[i, 0]

        def ramp_up_complementarity_rule_dual(m, i, t):
            return m.mu_ramp_up[i, t] <= m._M_ramp_up_dual[(i, t)] * (1 - m.z_ramp_up[i, t])

        def ramp_down_complementarity_rule(m, i, t):
            return self._ramp_down_slack_expr(m, i, t) <= m._M_ramp_down_slack[(i, t)] * m.z_ramp_down[i, t]

        def ramp_down_complementarity_initial_rule(m, i):
            return self._ramp_down_slack_expr(m, i, 0) <= m._M_ramp_down_slack[(i, 0)] * m.z_ramp_down[i, 0]

        def ramp_down_complementarity_rule_dual(m, i, t):
            return m.mu_ramp_down[i, t] <= m._M_ramp_down_dual[(i, t)] * (1 - m.z_ramp_down[i, t])

        self.model.upper_bound_complementarity_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_rule)
        self.model.upper_bound_complementarity_constraints_dual = Constraint(self.model.n_gen, self.model.time_steps, rule=upper_bound_complementarity_rule_dual)
        self.model.lower_bound_complementarity_constraints = Constraint(self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_rule)
        self.model.lower_bound_complementarity_constraints_dual = Constraint(self.model.n_gen, self.model.time_steps, rule=lower_bound_complementarity_rule_dual)

        self.model.ramp_up_complementarity_constraints = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_up_complementarity_rule)
        self.model.ramp_up_complementarity_initial_constraints = Constraint(self.model.n_gen, rule=ramp_up_complementarity_initial_rule)
        self.model.ramp_up_complementarity_constraints_dual = Constraint(self.model.n_gen, self.model.time_steps, rule=ramp_up_complementarity_rule_dual)

        self.model.ramp_down_complementarity_constraints = Constraint(self.model.n_gen, self.model.time_steps_minus_1, rule=ramp_down_complementarity_rule)
        self.model.ramp_down_complementarity_initial_constraints = Constraint(self.model.n_gen, rule=ramp_down_complementarity_initial_rule)
        self.model.ramp_down_complementarity_constraints_dual = Constraint(self.model.n_gen, self.model.time_steps, rule=ramp_down_complementarity_rule_dual)

    def _build_bid_seperation_constraints(self) -> None:
        """
        Function to build the bid seperation constraints for the MPEC model. 
        """
        BigM = self.big_m_bid_separation
        epsilon = self.bid_separation_epsilon
        s = self.scenario_index

        def alpha_upper_seperation(m, i, k, t):
            return m.alpha[i, t] >= self.bid_scenarios[s][t][k] + epsilon - BigM * (1 - m.tau[i, k, t])
        
        def alpha_lower_seperation(m, i, k, t):
            return m.alpha[i, t] <= self.bid_scenarios[s][t][k] - epsilon + BigM * m.tau[i, k, t]

        self.model.alpha_upper_seperation_constraints = Constraint(self.model.strategic_index, self.model.non_strategic_index, self.model.time_steps, rule=alpha_upper_seperation)
        self.model.alpha_lower_seperation_constraints = Constraint(self.model.strategic_index, self.model.non_strategic_index, self.model.time_steps, rule=alpha_lower_seperation)

    def solve(
        self,
        tee: bool = False,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        solver_threads: Optional[int] = None,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Solve the optimization model.
        """
        if self.model is None:
            if not parallel:
                raise ValueError("No single-scenario model is built. Call build_model(..., scenario_index=s) or solve with parallel=True.")
            self._solve_scenarios_parallel(
                max_workers=max_workers,
                tee=tee,
                solver_threads=solver_threads,
                solver_options=solver_options,
            )
            return

        # Create solver
        solver = SolverFactory("gurobi")
        if solver_threads is not None:
            solver.options["Threads"] = int(solver_threads)
        if solver_options:
            for option, value in solver_options.items():
                solver.options[option] = value

        # Solve
        results = solver.solve(self.model, tee=tee)

        # Check solver status
        if not (str(results.solver.status).lower() == 'ok' and str(results.solver.termination_condition).lower() == 'optimal'):
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)
            raise ValueError("Solver did not find an optimal solution")

    def _solve_scenarios_parallel(
        self,
        max_workers: Optional[int] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self, "strategic_player_id"):
            raise ValueError("Must call build_model(strategic_player_id) before solving model")

        optimal_bids = [None] * self.num_scenarios
        profits = [None] * self.num_scenarios
        if max_workers is None:
            cpu_count = os.cpu_count() or 1
            workers = min(self.num_scenarios, max(1, min(12, cpu_count)))
        else:
            workers = max(1, min(int(max_workers), self.num_scenarios))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._solve_one_scenario, s, tee, solver_threads, solver_options): s
                for s in range(self.num_scenarios)
            }
            for future in as_completed(futures):
                s = futures[future]
                bid_profile, profit = future.result()
                optimal_bids[s] = bid_profile
                profits[s] = profit

        self._parallel_optimal_bid_scenarios = optimal_bids
        self._parallel_scenario_profits = profits

    def _solve_one_scenario(
        self,
        scenario_index: int,
        tee: bool,
        solver_threads: Optional[int],
        solver_options: Optional[Dict[str, Any]],
    ) -> Tuple[List[List[float]], float]:
        scenario_model = self.__class__(
            self.scenarios_df.iloc[[scenario_index]].reset_index(drop=True),
            self.costs_df,
            self.ramps_df,
            self.players_config,
            p_init=[self.P_init[scenario_index]],
            pmin_default=self.pmin_default,
            config_overrides=self.config,
        )
        scenario_model.build_model(self.strategic_player_id, scenario_index=0)
        scenario_model.solve(
            tee=tee,
            parallel=False,
            solver_threads=solver_threads,
            solver_options=solver_options,
        )
        return scenario_model.get_optimal_bids()[0], scenario_model.get_scenario_profits()[0]
    
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
            if self._parallel_optimal_bid_scenarios is not None:
                return self._parallel_optimal_bid_scenarios
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
        
        # Extract optimal alpha values for strategic generators in the built scenario.
        s = self.scenario_index
        for t in range(self.num_time_steps):
            for i in strategic_indices:
                try:
                    alpha_value = self.model.alpha[i, t].value
                    if alpha_value is not None:
                        optimal_bid_scenarios[s][t][i] = float(alpha_value)
                    else:
                        print(f"Warning: Alpha value for scenario {s}, time {t}, generator {i} is None")
                except KeyError:
                    print(f"Warning: Alpha variable for scenario {s}, time {t}, generator {i} not found")
                    
        return optimal_bid_scenarios
    
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

            for s in range(self.num_scenarios):
                profile = [float(optimal_bid_scenarios[s][t][gen_idx]) for t in range(self.num_time_steps)]
                updated_df.at[s, bid_profile_col] = profile

        self.scenarios_df = updated_df
        self._extract_scenario_data(updated_df, self.costs_df, self.ramps_df, self.pmin_default)

        return updated_df

    def get_scenario_profits(self) -> List[float]:
        """
        Calculate the profit for the strategic player in each scenario based on the optimal bids and dispatch.
        
        Returns
        -------
        List[float]
            List of profits for the strategic player in each scenario
        """
        if self.model is None:
            if self._parallel_scenario_profits is not None:
                return self._parallel_scenario_profits
            raise ValueError("Model has not been built yet. Call build_model(strategic_player_id) first.")

        if not hasattr(self.model, 'lambda_var'):
            raise ValueError("Market clearing price variable (lambda_var) not found. Model may not be properly built.")
        
        profits = [0.0 for _ in range(self.num_scenarios)]
        s = self.scenario_index
        profit_scenario = 0.0
        for t in range(self.num_time_steps):
            lambda_value = self.model.lambda_var[t].value
            for i in self.strategic_generators:
                dispatch = self.model.P[i, t].value
                cost = self.cost_vector[i]
                profit_scenario += (lambda_value * dispatch - cost * dispatch)
        profits[s] = float(profit_scenario)
        
        return profits
