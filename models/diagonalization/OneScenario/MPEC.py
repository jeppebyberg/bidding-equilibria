from pyomo.environ import *
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

# Add workspace root to path (go up 3 levels: diagonalization -> models -> workspace_root)
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.utils.MPEC_utils import get_mpec_parameters
from models.diagonalization.economic_dispatch import EconomicDispatchModel

class MPECModel:
    def __init__(self, 
                 demand: List[float],
                 pmax_list: List[float], 
                 pmin_list: List[float],
                 num_generators: int,
                 strategic_player: int | tuple[int] = None,
                 bid_vector: List[float] = None,
                 cost_vector: List[float] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize MPEC model with case data and configuration
        
        Parameters
        ----------
        demand : List[float]
            Demand vector for the market
        pmax_list : List[float]
            Maximum power output for each generator
        pmin_list : List[float] 
            Minimum power output for each generator
        num_generators : int
            Number of generators in the system
        config_overrides : Dict[str, Any], optional
            Configuration overrides for MPEC parameters
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
        
        # Case data (fixed for all strategic players)
        self.demand = demand
        self.Pmax = pmax_list
        self.Pmin = pmin_list 
        self.num_generators = num_generators
        
        self.strategic_player = strategic_player
        self.bid_vector = bid_vector
        self.cost_vector = cost_vector

        self.model = None
        # self._build_model()  # Build initial model structure

    def update_strategic_player(self, 
                                strategic_player: int | tuple[int], 
                                bid_vector: List[float], 
                                cost_vector: List[float]) -> None:
        """
        Update the model for a new strategic player without rebuilding everything.
        Only updates the constraints that depend on the strategic player.
        
        Parameters
        ----------
        strategic_player : int or tuple[int]
            Index or indexes of generator(s) the strategic player controls
        bid_vector : List[float]
            Current bid vector for all generators
        cost_vector : List[float]
            Cost vector for all generators
        """
        self.strategic_player = strategic_player
        self.bid_vector = bid_vector
        self.cost_vector = cost_vector
        
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
        if self.strategic_player is None:
            raise ValueError("Must call update_strategic_player() before building model")

        self.model = ConcreteModel()
        
        # Determine strategic set
        if isinstance(self.strategic_player, int):
            strategic_set = [self.strategic_player]
        else:
            strategic_set = list(self.strategic_player)

        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.strategic_index = Set(initialize=strategic_set)

        #Create set of non-strategic generators  
        non_strategic_gens = [i for i in range(self.num_generators) if i not in self.model.strategic_index]
        self.model.non_strategic_index = Set(initialize=non_strategic_gens)

        self._build_variables()
        self._build_objective()
        self._build_constraints()
        
    def _update_strategic_constraints(self) -> None:
        """
        Update only the constraints that depend on the strategic player.
        This is much more efficient than rebuilding the entire model.
        """
        # Update strategic set
        if isinstance(self.strategic_player, int):
            strategic_set = [self.strategic_player]
        else:
            strategic_set = list(self.strategic_player)
            
        # Update the strategic index set
        self.model.strategic_index.clear()
        self.model.strategic_index.update(strategic_set)
        
        # Update non-strategic set
        non_strategic_gens = [i for i in range(self.num_generators) if i not in strategic_set]
        self.model.non_strategic_index.clear()
        self.model.non_strategic_index.update(non_strategic_gens)
        
        # Update alpha variables to match new strategic set (BEFORE building constraints)
        self.model.del_component(self.model.alpha)
        self.model.alpha = Var(self.model.strategic_index, domain=Reals)
        
        # Update tau variables to match new strategic/non-strategic split (BEFORE building constraints)
        self.model.del_component(self.model.tau)
        self.model.tau = Var(self.model.strategic_index, self.model.non_strategic_index, domain=Binary)
        
        # Update objective (depends on strategic player)
        self.model.del_component(self.model.objective)
        self._build_objective()
        
        # Update upper level constraints (bid bounds for strategic players)
        self.model.del_component(self.model.min_bid_constraint)
        self.model.del_component(self.model.max_bid_constraint)
        self._build_upper_level_constraints()
        
        # Update KKT stationarity constraints (different rules for strategic vs non-strategic)
        self.model.del_component(self.model.stationarity_constraint_strategic)
        self.model.del_component(self.model.stationarity_constraint_non_strategic)
        self._build_KKT_stationarity_constraints()
        
        # Update bid separation constraints (only between strategic and non-strategic)
        self.model.del_component(self.model.alpha_upper_seperation_constraints)
        self.model.del_component(self.model.alpha_lower_seperation_constraints)
        self._build_bid_seperation_constraints()

    def _build_variables(self) -> None:
        """
        Function to build the Pyomo variables for the MPEC model. 
        """
        self._build_upper_level_primal_variables()
        self._build_upper_level_dual_variables()
        self._build_lower_level_primal_variables()
        self._build_complementarity_variables()
        self._build_bid_seperation_variables()
        # self._build_policy_variables()

    def _build_upper_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the upper-level problem. 
        """
        #Bid variable for the strategic player(s)
        self.model.alpha = Var(self.model.strategic_index, domain=Reals)

    def _build_upper_level_dual_variables(self) -> None:
        """
        Function to build the dual variables for the upper-level problem. 
        """
        #Dual variables for the market clearing problem
        self.model.lambda_var = Var(domain=Reals)
        self.model.mu_upper_bound = Var(self.model.n_gen, domain=NonNegativeReals)  # Upper bound duals
        self.model.mu_lower_bound = Var(self.model.n_gen, domain=NonNegativeReals)  # Lower bound duals

    def _build_lower_level_primal_variables(self) -> None:
        """
        Function to build the primal variables for the lower-level problem. 
        """
        #Production variable for each generator
        self.model.P = Var(self.model.n_gen, domain=NonNegativeReals)

    def _build_complementarity_variables(self) -> None:
        """
        Function to build the complementarity variables for the MPEC model. 
        """
        #Complementarity variables for the upper and lower bounds (one per generator)
        self.model.z_upper_bound = Var(self.model.n_gen, domain=Binary)
        self.model.z_lower_bound = Var(self.model.n_gen, domain=Binary)

    def _build_policy_variables(self) -> None:
        """
        Function to build the policy variables for the MPEC model. 
        """
        #Policy variable for the strategic player
        self.model.theta = Var(domain=Reals)

    def _build_bid_seperation_variables(self) -> None:
        """
        Function to build the bid separation variables for the MPEC model.
        """
        
        #Binary bid separation variables for strategic players vs competitors
        self.model.tau = Var(self.model.strategic_index, self.model.non_strategic_index, domain=Binary)

    def _build_objective(self) -> None:
        """
        Function to build the objective function for the MPEC model. 
        """
        self.model.objective = Objective(expr= -(
                                        self.model.lambda_var * self.demand -
                                        sum(self.model.mu_upper_bound[i] * self.Pmax[i] for i in self.model.n_gen) +
                                        sum(self.model.mu_lower_bound[i] * self.Pmin[i] for i in self.model.n_gen) -
                                        sum(self.bid_vector[i] * self.model.P[i] for i in self.model.non_strategic_index) +
                                        sum(self.model.mu_upper_bound[i] * self.Pmax[i] for i in self.model.strategic_index) -
                                        sum(self.model.mu_lower_bound[i] * self.Pmin[i] for i in self.model.strategic_index)
                                        )
                                        + sum(self.cost_vector[i] * self.model.P[i] for i in self.model.strategic_index),
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
        # self._build_policy_constraints()

    def _build_upper_level_constraints(self) -> None:
        """
        Function to build the upper-level constraints for the MPEC model. 
        """
        def min_bid_rule(model, i): 
            return model.alpha[i] >= self.alpha_min
        
        def max_bid_rule(model, i):
            return model.alpha[i] <= self.alpha_max
        
        self.model.min_bid_constraint = Constraint(self.model.strategic_index, rule=min_bid_rule)
        self.model.max_bid_constraint = Constraint(self.model.strategic_index, rule=max_bid_rule)

    def _build_lower_level_constraints(self) -> None:
        """
        Function to build the lower-level constraints for the MPEC model. 
        """

        def power_balance_rule(m):
            return sum(m.P[i] for i in m.n_gen) == self.demand
        
        def generation_upper_rule(m, i):
            return 0 <= self.Pmax[i] - m.P[i]

        def generation_lower_rule(m, i):
            return 0 <= m.P[i] - self.Pmin[i]
        
        self.model.power_balance_constraint = Constraint(rule=power_balance_rule)
        self.model.generation_upper_bound_constraints = Constraint(self.model.n_gen, rule=generation_upper_rule)
        self.model.generation_lower_bound_constraints = Constraint(self.model.n_gen, rule=generation_lower_rule)
    
    def _build_KKT_stationarity_constraints(self) -> None:
        """
        Function to build the KKT stationarity constraints for the MPEC model. 
        """
        def stationarity_rule_strategic_agent(m, i):
            return m.alpha[i] - m.lambda_var + m.mu_upper_bound[i] - m.mu_lower_bound[i] == 0

        def stationarity_rule_non_strategic_agents(m, i):
            return self.bid_vector[i] - m.lambda_var + m.mu_upper_bound[i] - m.mu_lower_bound[i] == 0

        self.model.stationarity_constraint_strategic = Constraint(self.model.strategic_index, rule=stationarity_rule_strategic_agent)
        self.model.stationarity_constraint_non_strategic = Constraint(self.model.non_strategic_index, rule=stationarity_rule_non_strategic_agents)

    def _build_KKT_complementarity_constraints(self) -> None:
        """
        Function to build the KKT complementarity constraints for the MPEC model. 
        """
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_rule(m, i):
            return self.Pmax[i] - m.P[i] <= BigM * (1 - m.z_upper_bound[i]) 

        def upper_bound_complementarity_rule_dual(m, i):
            return m.mu_upper_bound[i] <= BigM * m.z_upper_bound[i] 
        
        def lower_bound_complementarity_rule(m, i):
            return m.P[i] - self.Pmin[i] <= BigM * (1 - m.z_lower_bound[i]) 

        def lower_bound_complementarity_rule_dual(m, i):
            return m.mu_lower_bound[i] <= BigM * m.z_lower_bound[i] 

        self.model.upper_bound_complementarity_constraints = Constraint(self.model.n_gen, rule=upper_bound_complementarity_rule)
        self.model.upper_bound_complementarity_constraints_dual = Constraint(self.model.n_gen, rule=upper_bound_complementarity_rule_dual)
        self.model.lower_bound_complementarity_constraints = Constraint(self.model.n_gen, rule=lower_bound_complementarity_rule)
        self.model.lower_bound_complementarity_constraints_dual = Constraint(self.model.n_gen, rule=lower_bound_complementarity_rule_dual)

    def _build_bid_seperation_constraints(self) -> None:
        """
        Function to build the bid seperation constraints for the MPEC model. 
        """
        BigM = self.big_m_bid_separation
        epsilon = self.bid_separation_epsilon

        def alpha_upper_seperation(m, i, k):
            return m.alpha[i] >= self.bid_vector[k] + epsilon - BigM * (1 - m.tau[i, k])
        
        def alpha_lower_seperation(m, i, k):
            return m.alpha[i] <= self.bid_vector[k] - epsilon + BigM * m.tau[i, k]

        self.model.alpha_upper_seperation_constraints = Constraint(self.model.strategic_index, self.model.non_strategic_index, rule=alpha_upper_seperation)
        self.model.alpha_lower_seperation_constraints = Constraint(self.model.strategic_index, self.model.non_strategic_index, rule=alpha_lower_seperation)

    def _build_policy_constraints(self) -> None:
        """
        Function to build the policy constraints for the MPEC model. 
        """
        def policy_rule(m):
            return m.alpha == m.theta * self.features
        self.model.policy_constraint = Constraint(rule=policy_rule)

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

if __name__ == "__main__":
    # Import test data loader
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.utils.cases_utils import load_setup_data
    
    print("=== Testing MPEC Model ===")
    
    # Load test case data
    num_generators, pmax_list, pmin_list, cost_vector, demand = load_setup_data()
    
    initial_bids = cost_vector.copy()
    
    strategic_player = 1

    # Create MPEC model
    mpec_model = MPECModel(
        demand=demand,
        pmax_list=pmax_list,
        pmin_list=pmin_list,
        num_generators=num_generators,
        strategic_player=strategic_player,
        bid_vector=initial_bids,
        cost_vector=cost_vector
    )
    
    print(f"\n=== Testing Strategic Player {strategic_player} ===")
    
    mpec_model._build_model()
    mpec_model.solve()

    print("\n=== MPEC Solution ===")
    print(f"Objective value: {-mpec_model.model.objective.expr()}")

    model = EconomicDispatchModel()
    dispatch, clearing_price = model.economic_dispatch(num_generators=num_generators, demand=demand, Pmax=pmax_list, Pmin=pmin_list, bid_list=initial_bids)

    print(f"Strategic player: {strategic_player}")
    print(f"Initial bids: {initial_bids}")

    alpha = []  
    for i in mpec_model.model.strategic_index:
        alpha_val = mpec_model.model.alpha[i].value
        if alpha_val is None:
            print(f"Warning: No solution for strategic player {i}, using initial bid")
            alpha_val = initial_bids[i]
        alpha.append(alpha_val)
    for i in mpec_model.model.non_strategic_index:
        alpha.append(mpec_model.bid_vector[i])  # Non-strategic players have fixed bids
    print(f"Alpha:        {np.array(alpha).round(2)}")

    strategic_player = strategic_player + 1
    print(f"\n=== Testing Strategic Player {strategic_player} ===")

    mpec_model.update_strategic_player(strategic_player=strategic_player, bid_vector=initial_bids, cost_vector=cost_vector)
    mpec_model.solve()

    print(f"Strategic player: {strategic_player}")
    print("\n=== MPEC Solution ===")
    print(f"Objective value: {-mpec_model.model.objective.expr()}")

    alpha = []  
    for i in mpec_model.model.strategic_index:
        alpha_val = mpec_model.model.alpha[i].value
        if alpha_val is None:
            print(f"Warning: No solution for strategic player {i}, using initial bid")
            alpha_val = initial_bids[i]
        alpha.append(alpha_val)
    for i in mpec_model.model.non_strategic_index:
        alpha.append(mpec_model.bid_vector[i])  # Non-strategic players have fixed bids
    print(f"Alpha:        {np.array(alpha).round(2)}")


    print(f"\nProfit of each agent")
    for i in mpec_model.model.n_gen:
        profit = mpec_model.model.lambda_var.value * mpec_model.model.P[i].value - mpec_model.cost_vector[i] * mpec_model.model.P[i].value
        profit_ED = clearing_price * dispatch[i] - mpec_model.cost_vector[i] * dispatch[i]
        print(f"Profit agent {i}: {profit} (ED: {profit_ED})")
