from pyomo.environ import *
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from .utilities.MPEC_utils import get_mpec_parameters
from .feature_setup import FeatureBuilder, DEFAULT_FEATURES

class MPECModel:
    def __init__(self, 
                 scenarios_df: pd.DataFrame,
                 costs_df: pd.DataFrame,
                 players_config: List[Dict[str, Any]],
                 strategic_player_id: int = None,
                 config_overrides: Optional[Dict[str, Any]] = None,
                 features: Optional[List[str]] = None):
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
        config_overrides : Dict[str, Any], optional
            Configuration overrides for MPEC parameters
        features : List[str], optional
            Ordered list of feature names for the bidding policy.
            Defaults to DEFAULT_FEATURES from features.yaml.
        """
        
        # Load default configuration
        self.config = get_mpec_parameters()
        
        # Apply any overrides
        if config_overrides:
            self.config.update(config_overrides)
              
        # Initialize model parameters from config using utility functions
        self.alpha_min = self.config.get("alpha_min") 
        self.alpha_max = self.config.get("alpha_max")
        self.big_m_complementarity = self.config.get("big_m_complementarity")
        self.big_m_bid_separation = self.config.get("big_m_bid_separation")
        self.bid_separation_epsilon = self.config.get("bid_separation_epsilon")
        
        # Extract scenario and generator information from DataFrames
        self._extract_scenario_data(scenarios_df, costs_df)
        
        # Store player configurations and strategic player
        self.players_config = players_config
        self.strategic_player_id = strategic_player_id
        self.strategic_generators = []
        
        # Extract strategic generator indices if strategic player is specified
        if self.strategic_player_id is not None and self.players_config:
            strategic_player = next((p for p in self.players_config if p['id'] == self.strategic_player_id), None)
            if strategic_player:
                self.strategic_generators = strategic_player['controlled_generators']
            else:
                raise ValueError(f"Strategic player {self.strategic_player_id} not found in players_config")
        
        # Store DataFrames
        self.scenarios_df = scenarios_df
        self.costs_df = costs_df

        # Feature builder for policy-based bidding
        self.feature_builder = FeatureBuilder(features or DEFAULT_FEATURES)
        self.feature_matrix: Dict = {}   # populated by _build_feature_matrix()
        self.num_policy_features: int = 0

        self.model = None

    def _extract_scenario_data(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame):
        """Extract scenario and generator data from DataFrames"""
        
        # Auto-detect demand column
        demand_col = None
        for col in scenarios_df.columns:
            if any(keyword in col.lower() for keyword in ['demand', 'load']):
                demand_col = col
                break
        
        if demand_col is None:
            raise ValueError("No demand column found. Expected column name containing 'demand' or 'load'")
        
        # Auto-detect generator capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'")
        
        # Extract generator information
        self.generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(scenarios_df)
        
        # Extract data
        self.demand_scenarios = scenarios_df[demand_col].tolist()
        
        # Build capacity and cost data structures
        self.pmax_scenarios = []
        self.pmin_scenarios = []
        
        for _, row in scenarios_df.iterrows():
            pmax_scenario = [row[f"{gen}_cap"] for gen in self.generator_names]
            pmin_scenario = [0.0] * self.num_generators  # Default Pmin = 0
            self.pmax_scenarios.append(pmax_scenario)
            self.pmin_scenarios.append(pmin_scenario)
        
        # Extract costs
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in self.generator_names]
        
        # Extract bid data for each scenario
        self.bid_scenarios = []
        for _, row in scenarios_df.iterrows():
            scenario_bids = [row[f"{gen}_bid"] for gen in self.generator_names]
            self.bid_scenarios.append(scenario_bids)

    def _build_feature_matrix(self) -> None:
        """
        Build the feature matrix for the current strategic player.

        Produces one feature vector per (scenario, generator) pair and stores
        them in ``self.feature_matrix`` keyed by ``(s, i)`` where *s* is the
        scenario index and *i* is the **global** generator index.

        Also sets ``self.num_policy_features`` to the length of each vector.
        """
        self.feature_matrix = {}

        for gen_idx in self.strategic_generators:
            # Build observations treating this single generator as its own "player"
            observations = FeatureBuilder._extract_observations(
                self.scenarios_df,
                self.costs_df,
                player_generators=[gen_idx],
                generator_names=self.generator_names,
            )
            for s, obs in enumerate(observations):
                self.feature_matrix[s, gen_idx] = self.feature_builder.build(obs)

        # All vectors have the same length – grab it from any entry
        if self.feature_matrix:
            sample = next(iter(self.feature_matrix.values()))
            self.num_policy_features = len(sample)
        else:
            self.num_policy_features = self.feature_builder.num_features_expanded(1)

    def update_strategic_player(self, strategic_player_id: int) -> None:
        """
        Update the model for a new strategic player without rebuilding everything.
        Only updates the constraints that depend on the strategic player.
        
        Parameters
        ----------
        strategic_player_id : int
            ID of the player to optimize (must match a player id in players_config)
        """            
        # Find the player and get their controlled generators
        strategic_player = next((p for p in self.players_config if p['id'] == strategic_player_id), None)
        if not strategic_player:
            raise ValueError(f"Strategic player {strategic_player_id} not found in players_config")
            
        self.strategic_player_id = strategic_player_id
        self.strategic_generators = strategic_player['controlled_generators']
        
        # If model doesn't exist, build it completely
        if self.model is None:
            self._build_model()
        else:
            # Update only the strategic player dependent constraints
            self._update_strategic_constraints()
    
    def _build_model(self) -> None:
        """
        Build the complete MPEC model structure.
        This is called once, then update_strategic_player() updates the changing parts between each strategic player.
        """
        if self.strategic_player_id is None:
            raise ValueError("Must call update_strategic_player() before building model")

        self.model = ConcreteModel()
        
        # Define sets
        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.n_scenarios = Set(initialize=range(self.num_scenarios))
        self.model.strategic_index = Set(initialize=self.strategic_generators)

        # Create set of non-strategic generators  
        non_strategic_gens = [i for i in range(self.num_generators) if i not in self.strategic_generators]
        self.model.non_strategic_index = Set(initialize=non_strategic_gens)

        # Build feature matrix for the current strategic player
        self._build_feature_matrix()

        self._build_variables()
        self._build_objective()
        self._build_constraints()
        
    def _update_strategic_constraints(self) -> None:
        """
        Update only the constraints that depend on the strategic player.
        This is much more efficient than rebuilding the entire model.
        """
        # Update strategic set
        strategic_set = self.strategic_generators
            
        # Update the strategic index set
        self.model.strategic_index.clear()
        self.model.strategic_index.update(strategic_set)
        
        # Update non-strategic set
        non_strategic_gens = [i for i in range(self.num_generators) if i not in strategic_set]
        self.model.non_strategic_index.clear()
        self.model.non_strategic_index.update(non_strategic_gens)
        
        # Update alpha variables to match new strategic set (BEFORE building constraints)
        self.model.del_component(self.model.alpha)
        self.model.alpha = Var(self.model.n_scenarios, self.model.strategic_index, domain=Reals)
        
        # Update tau variables to match new strategic/non-strategic split (BEFORE building constraints)
        self.model.del_component(self.model.tau)
        self.model.tau = Var(self.model.n_scenarios, self.model.strategic_index, self.model.non_strategic_index, domain=Binary)
        
        # Update objective (depends on strategic player)
        self.model.del_component(self.model.objective)
        self._build_objective()
        
        # Update upper level constraints (bid bounds for strategic players)
        self.model.del_component(self.model.min_bid_constraint)
        self.model.del_component(self.model.max_bid_constraint)
        self.model.del_component(self.model.min_bid_constraint_2)
        self._build_upper_level_constraints()
        
        # Update KKT stationarity constraints (different rules for strategic vs non-strategic)
        self.model.del_component(self.model.stationarity_constraint_strategic)
        self.model.del_component(self.model.stationarity_constraint_non_strategic)
        self._build_KKT_stationarity_constraints()
        
        # Update bid separation constraints (only between strategic and non-strategic)
        self.model.del_component(self.model.alpha_upper_seperation_constraints)
        self.model.del_component(self.model.alpha_lower_seperation_constraints)
        self._build_bid_seperation_constraints()

        # Rebuild feature matrix for the new strategic player
        self._build_feature_matrix()

        # Update policy variables and constraints (if they exist)
        if hasattr(self.model, 'policy_constraint'):
            self.model.del_component(self.model.policy_constraint)
        if hasattr(self.model, 'theta'):
            self.model.del_component(self.model.theta)
        if hasattr(self.model, 'n_features'):
            self.model.del_component(self.model.n_features)
        self._build_policy_variables()
        self._build_policy_constraints()

    def _build_variables(self) -> None:
        """
        Function to build the Pyomo variables for the MPEC model. 
        """
        self._build_upper_level_primal_variables()
        self._build_upper_level_dual_variables()
        self._build_lower_level_primal_variables()
        self._build_complementarity_variables()
        self._build_bid_seperation_variables()
        self._build_policy_variables()

    def _build_upper_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the upper-level problem. 
        """
        #Bid variable for the strategic player(s)
        self.model.alpha = Var(self.model.n_scenarios, self.model.strategic_index, domain=Reals)

    def _build_upper_level_dual_variables(self) -> None:
        """
        Function to build the dual variables for the upper-level problem. 
        """
        #Dual variables for the market clearing problem (one per scenario)
        self.model.lambda_var = Var(self.model.n_scenarios, domain=Reals)
        self.model.mu_upper_bound = Var(self.model.n_scenarios, self.model.n_gen, domain=NonNegativeReals)  # Upper bound duals
        self.model.mu_lower_bound = Var(self.model.n_scenarios, self.model.n_gen, domain=NonNegativeReals)  # Lower bound duals

    def _build_lower_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the lower-level problem. 
        """
        #Production variable for each generator in each scenario
        self.model.P = Var(self.model.n_scenarios, self.model.n_gen, domain=NonNegativeReals)

    def _build_complementarity_variables(self) -> None:
        """
        Function to build the complementarity variables for the MPEC model. 
        """
        #Complementarity variables for the upper and lower bounds (one per generator per scenario)
        self.model.z_upper_bound = Var(self.model.n_scenarios, self.model.n_gen, domain=Binary)
        self.model.z_lower_bound = Var(self.model.n_scenarios, self.model.n_gen, domain=Binary)

    def _build_policy_variables(self) -> None:
        """
        Function to build the policy variables for the MPEC model.
        theta is the weight vector shared across all strategic generators;
        its dimension equals the expanded feature size (num_policy_features).
        """
        self.model.n_features = Set(initialize=range(self.num_policy_features))
        self.model.theta = Var(self.model.n_features, domain=Reals)

    def _build_bid_seperation_variables(self) -> None:
        """
        Function to build the bid separation variables for the MPEC model.
        """
        
        #Binary bid separation variables for strategic players vs competitors
        self.model.tau = Var(self.model.n_scenarios, self.model.strategic_index, self.model.non_strategic_index, domain=Binary)

    def _build_objective(self) -> None:
        """
        Function to build the objective function for the MPEC model.
        Minimizes the negative *average* profit of the strategic player
        across all scenarios and accumulated iterations (equivalent to
        maximizing the mean profit per observation).
        """
        self.model.objective = Objective(expr=
            1 / self.num_scenarios * (
            -(
            sum(self.model.lambda_var[s] * self.demand_scenarios[s] for s in self.model.n_scenarios) -
            sum(self.model.mu_upper_bound[s, i] * self.pmax_scenarios[s][i] for s in self.model.n_scenarios for i in self.model.n_gen) +
            sum(self.model.mu_lower_bound[s, i] * self.pmin_scenarios[s][i] for s in self.model.n_scenarios for i in self.model.n_gen) -
            sum(self.bid_scenarios[s][i] * self.model.P[s, i] for s in self.model.n_scenarios for i in self.model.non_strategic_index) +
            sum(self.model.mu_upper_bound[s, i] * self.pmax_scenarios[s][i] for s in self.model.n_scenarios for i in self.model.strategic_index) -
            sum(self.model.mu_lower_bound[s, i] * self.pmin_scenarios[s][i] for s in self.model.n_scenarios for i in self.model.strategic_index)
            )
            + sum(self.cost_vector[i] * self.model.P[s, i] for s in self.model.n_scenarios for i in self.model.strategic_index)
            ),
            sense=minimize)

    def _build_constraints(self) -> None:
        """
        Function to build the constraints for the MPEC model. 
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
        def min_bid_rule(model, s, i): 
            return model.alpha[s, i] >= self.alpha_min
        
        def max_bid_rule(model, s, i):
            return model.alpha[s, i] <= self.alpha_max
        
        def min_bid_rule_2(model, s, i):
            return model.alpha[s, i] >= self.cost_vector[i]
        
        self.model.min_bid_constraint = Constraint(self.model.n_scenarios, self.model.strategic_index, rule=min_bid_rule)
        self.model.max_bid_constraint = Constraint(self.model.n_scenarios, self.model.strategic_index, rule=max_bid_rule)
        self.model.min_bid_constraint_2 = Constraint(self.model.n_scenarios, self.model.strategic_index, rule=min_bid_rule_2)

    def _build_lower_level_constraints(self) -> None:
        """
        Function to build the lower-level constraints for the MPEC model. 
        """

        def power_balance_rule(m, s):
            return sum(m.P[s, i] for i in m.n_gen) == self.demand_scenarios[s]
        
        def generation_upper_rule(m, s, i):
            return 0 <= self.pmax_scenarios[s][i] - m.P[s, i]

        def generation_lower_rule(m, s, i):
            return 0 <= m.P[s, i] - self.pmin_scenarios[s][i]
        
        self.model.power_balance_constraint = Constraint(self.model.n_scenarios, rule=power_balance_rule)
        self.model.generation_upper_bound_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=generation_upper_rule)
        self.model.generation_lower_bound_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=generation_lower_rule)
    
    def _build_KKT_stationarity_constraints(self) -> None:
        """
        Function to build the KKT stationarity constraints for the MPEC model. 
        """
        def stationarity_rule_strategic_agent(m, s, i):
            return m.alpha[s, i] - m.lambda_var[s] + m.mu_upper_bound[s, i] - m.mu_lower_bound[s, i] == 0

        def stationarity_rule_non_strategic_agents(m, s, i):
            return self.bid_scenarios[s][i] - m.lambda_var[s] + m.mu_upper_bound[s, i] - m.mu_lower_bound[s, i] == 0

        self.model.stationarity_constraint_strategic = Constraint(self.model.n_scenarios, self.model.strategic_index, rule=stationarity_rule_strategic_agent)
        self.model.stationarity_constraint_non_strategic = Constraint(self.model.n_scenarios, self.model.non_strategic_index, rule=stationarity_rule_non_strategic_agents)

    def _build_KKT_complementarity_constraints(self) -> None:
        """
        Function to build the KKT complementarity constraints for the MPEC model. 
        """
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_rule(m, s, i):
            return self.pmax_scenarios[s][i] - m.P[s, i] <= BigM * (1 - m.z_upper_bound[s, i]) 

        def upper_bound_complementarity_rule_dual(m, s, i):
            return m.mu_upper_bound[s, i] <= BigM * m.z_upper_bound[s, i] 
        
        def lower_bound_complementarity_rule(m, s, i):
            return m.P[s, i] - self.pmin_scenarios[s][i] <= BigM * (1 - m.z_lower_bound[s, i]) 

        def lower_bound_complementarity_rule_dual(m, s, i):
            return m.mu_lower_bound[s, i] <= BigM * m.z_lower_bound[s, i] 

        self.model.upper_bound_complementarity_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=upper_bound_complementarity_rule)
        self.model.upper_bound_complementarity_constraints_dual = Constraint(self.model.n_scenarios, self.model.n_gen, rule=upper_bound_complementarity_rule_dual)
        self.model.lower_bound_complementarity_constraints = Constraint(self.model.n_scenarios, self.model.n_gen, rule=lower_bound_complementarity_rule)
        self.model.lower_bound_complementarity_constraints_dual = Constraint(self.model.n_scenarios, self.model.n_gen, rule=lower_bound_complementarity_rule_dual)

    def _build_bid_seperation_constraints(self) -> None:
        """
        Function to build the bid seperation constraints for the MPEC model. 
        """
        BigM = self.big_m_bid_separation
        epsilon = self.bid_separation_epsilon

        def alpha_upper_seperation(m, s, i, k):
            return m.alpha[s, i] >= self.bid_scenarios[s][k] + epsilon - BigM * (1 - m.tau[s, i, k])
        
        def alpha_lower_seperation(m, s, i, k):
            return m.alpha[s, i] <= self.bid_scenarios[s][k] - epsilon + BigM * m.tau[s, i, k]

        self.model.alpha_upper_seperation_constraints = Constraint(self.model.n_scenarios, self.model.strategic_index, self.model.non_strategic_index, rule=alpha_upper_seperation)
        self.model.alpha_lower_seperation_constraints = Constraint(self.model.n_scenarios, self.model.strategic_index, self.model.non_strategic_index, rule=alpha_lower_seperation)

    def _build_policy_constraints(self) -> None:
        """
        Function to build the policy constraints for the MPEC model.
        Links each strategic bid to the shared policy weights:
            alpha[s, i] = sum_k  theta[k] * feature_matrix[s, i, k]
        """
        def policy_rule(m, s, i):
            phi = self.feature_matrix[s, i]          # 1-D numpy array
            return m.alpha[s, i] == sum(
                m.theta[k] * float(phi[k]) for k in m.n_features
            )
        self.model.policy_constraint = Constraint(
            self.model.n_scenarios, self.model.strategic_index, rule=policy_rule
        )

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
    
    def get_optimal_bids(self) -> List[List[float]]:
        """
        Extract optimal strategic bids from the solved model.
        
        Returns
        -------
        List[List[float]]
            Complete bid matrix where optimal_bids[s][i] is the bid for generator i 
            in scenario s. Strategic generators get optimal alpha values, non-strategic 
            generators keep their original bids.
        """        
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been built yet. Call update_strategic_player() first.")
            
        if not hasattr(self.model, 'alpha'):
            raise ValueError("Strategic bid variables (alpha) not found. Model may not be properly built.")
            
        # Get strategic generator indices
        strategic_indices = self.strategic_generators
            
        # Initialize optimal bids matrix (copy current bids first)
        optimal_bid_scenarios = [scenario_bids[:] for scenario_bids in self.bid_scenarios]
        
        # Extract optimal alpha values for strategic generators
        optimal_alphas = {}
        for s in range(self.num_scenarios):
            optimal_alphas[s] = {}
            for i in strategic_indices:
                try:
                    alpha_value = self.model.alpha[s, i].value
                    if alpha_value is not None:
                        optimal_alphas[s][i] = alpha_value
                        # Update the bid in the optimal bids matrix
                        optimal_bid_scenarios[s][i] = alpha_value
                    else:
                        print(f"Warning: Alpha value for scenario {s}, generator {i} is None")
                        optimal_alphas[s][i] = self.bid_scenarios[s][i]  # Keep original bid
                except KeyError:
                    print(f"Warning: Alpha variable for scenario {s}, generator {i} not found")
                    optimal_alphas[s][i] = self.bid_scenarios[s][i]  # Keep original bid
                    
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
        # Get optimal bids
        optimal_bid_scenarios = self.get_optimal_bids()
        
        # Get strategic indices
        strategic_indices = self.strategic_generators
        
        # Create update scenarios DataFrame
        updated_df = scenarios_df.copy()        
            
        # Update bid columns in the DataFrame
        for gen_idx in strategic_indices:
            gen_name = self.generator_names[gen_idx]
            bid_col = f"{gen_name}_bid"
            if bid_col in updated_df.columns:
                for s in range(self.num_scenarios):
                    updated_df.at[s, bid_col] = optimal_bid_scenarios[s][gen_idx]
                    
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
            lambda_value = self.model.lambda_var[s].value
            profit_scenario = 0.0
            for i in self.strategic_generators:
                bid = self.model.alpha[s, i].value if self.model.alpha[s, i].value is not None else self.bid_scenarios[s][i]
                dispatch = self.model.P[s, i].value if self.model.P[s, i].value is not None else 0.0
                cost = self.cost_vector[i]
                profit_scenario += (lambda_value * dispatch - cost * dispatch)
            profits.append(profit_scenario)
        
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

    # ── scenario accumulation for regret minimization ─────────────────

    def update_scenarios(self, new_scenarios_df: pd.DataFrame) -> None:
        """
        Replace the scenario data with an updated (growing) DataFrame.

        This is the main entry point for the best-response loop.  After each
        iteration the caller appends the latest bid state for **every** market
        scenario and passes the full accumulated DataFrame here so that the
        next ``solve()`` optimises theta over all scenarios **and** all
        previous iterations.

        Example accumulation with *S* base scenarios and *T* iterations::

            accumulated = base_df.copy()          # S rows (iteration 0)
            for t in range(1, T):
                iter_df = ...                     # S rows with updated bids
                accumulated = pd.concat([accumulated, iter_df], ignore_index=True)
                mpec.update_scenarios(accumulated) # S*t rows

        The Pyomo model is invalidated and will be rebuilt on the next
        ``update_strategic_player`` / ``solve`` call.

        Parameters
        ----------
        new_scenarios_df : pd.DataFrame
            The accumulated scenarios DataFrame (S_base * num_iterations rows).
        """
        # Re-extract all data from the new DataFrame
        self._extract_scenario_data(new_scenarios_df, self.costs_df)
        self.scenarios_df = new_scenarios_df.copy().reset_index(drop=True)

        # Force a full model rebuild next time
        self.model = None

    def get_optimal_theta(self) -> np.ndarray:
        """
        Extract the optimal policy weights from the solved model.

        Returns
        -------
        np.ndarray of shape (num_policy_features,)
            The solved theta vector.
        """
        if self.model is None:
            raise ValueError("Model has not been built / solved yet.")
        if not hasattr(self.model, 'theta'):
            raise ValueError("Policy variables (theta) not found on the model.")

        return np.array([
            self.model.theta[k].value for k in self.model.n_features
        ], dtype=np.float64)

    def get_policy_bids(self, theta: np.ndarray, scenarios_df: pd.DataFrame) -> List[List[float]]:
        """
        Compute the bids a given theta would produce on an arbitrary set of scenarios.

        This is useful for *evaluating* a policy on data it was not trained on
        (e.g. testing the current theta against a single new iteration).

        Parameters
        ----------
        theta : np.ndarray
            Policy weight vector of length ``num_policy_features``.
        scenarios_df : pd.DataFrame
            Scenarios to evaluate (same column format as the training data).

        Returns
        -------
        List[List[float]]
            ``policy_bids[s][i]`` is the bid for generator *i* in scenario *s*.
            Non-strategic generators keep their DataFrame bids.
        """
        # Build feature vectors for the current strategic player
        bid_matrix: List[List[float]] = []
        for s_idx, (_, row) in enumerate(scenarios_df.iterrows()):
            row_bids = [row[f"{g}_bid"] for g in self.generator_names]
            for gen_idx in self.strategic_generators:
                obs_list = FeatureBuilder._extract_observations(
                    scenarios_df.iloc[[s_idx]].reset_index(drop=True),
                    self.costs_df,
                    player_generators=[gen_idx],
                    generator_names=self.generator_names,
                )
                phi = self.feature_builder.build(obs_list[0])
                row_bids[gen_idx] = float(theta @ phi)
            bid_matrix.append(row_bids)
        return bid_matrix

