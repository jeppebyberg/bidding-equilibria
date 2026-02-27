from pyomo.environ import *
from typing import List
import pandas as pd

class EconomicDispatchModel:
    def __init__(self):
        pass

    def economic_dispatch(self, num_generators: int, demand_scenarios: List[float], Pmax, Pmin, bid_list, num_scenarios: int = 1) -> tuple[List[List[float]], List[float]]:
        """
        Solve the economic dispatch problem for multiple scenarios with the bids placed.
        Now optimized to solve all scenarios in a single optimization problem.

        Parameters
        ----------
        num_generators : int
            Number of generators
        demand_scenarios : List[float]
            List of demand values for each scenario  
        Pmax : List[List[float]] or List[float]
            Maximum power output - either [scenario][generator] or [generator] for uniform
        Pmin : List[List[float]] or List[float]  
            Minimum power output - either [scenario][generator] or [generator] for uniform
        bid_list : List[List[float]] or List[float]
            Cost/bid values - either [scenario][generator] or [generator] for uniform
        num_scenarios : int
            Number of scenarios to solve
            
        Returns
        -------
        all_dispatches : List[List[float]]
            Optimal dispatch for each generator in each scenario
        clearing_prices : List[float]
            Market clearing price for each scenario
        """
        
        # Handle backward compatibility - convert single arrays to scenario arrays
        if not isinstance(Pmax[0], list):
            # Old API: single arrays for all scenarios
            Pmax = [Pmax] * num_scenarios
            Pmin = [Pmin] * num_scenarios  
            bid_list = [bid_list] * num_scenarios

        model = ConcreteModel()
        
        # Define sets
        model.scenarios = Set(initialize=range(num_scenarios))
        model.generators = Set(initialize=range(num_generators))
        
        # Define variables with scenario and generator indices
        model.P_G = Var(model.scenarios, model.generators, domain=Reals)
        
        # Objective: minimize total cost across all scenarios
        model.objective = Objective(
            expr=sum(bid_list[s][g] * model.P_G[s, g] 
                    for s in model.scenarios 
                    for g in model.generators),
            sense=minimize
        )
        
        # Power balance constraint for each scenario
        def power_balance_rule(m, s):
            return demand_scenarios[s] - sum(m.P_G[s, g] for g in m.generators) == 0
        model.power_balance = Constraint(model.scenarios, rule=power_balance_rule)
        
        # Generator minimum limits for each scenario
        def gen_min_rule(m, s, g):
            return m.P_G[s, g] >= Pmin[s][g]
        model.gen_min = Constraint(model.scenarios, model.generators, rule=gen_min_rule)
        
        # Generator maximum limits for each scenario
        def gen_max_rule(m, s, g):
            return m.P_G[s, g] <= Pmax[s][g]
        model.gen_max = Constraint(model.scenarios, model.generators, rule=gen_max_rule)

        # Attach suffix to capture duals
        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=False)
        
        all_dispatches = []
        clearing_prices = []
        
        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            for s in range(num_scenarios):
                dispatch = [model.P_G[s, g].value for g in range(num_generators)]
                clearing_price = -model.dual[model.power_balance[s]]
                all_dispatches.append(dispatch)
                clearing_prices.append(clearing_price)
        else:
            print(f"Solver status:", results.solver.status)
            print(f"Termination condition:", results.solver.termination_condition)
            # Return empty results for failed scenarios
            for s in range(num_scenarios):
                all_dispatches.append([0.0] * num_generators)
                clearing_prices.append(0.0)
                
        return all_dispatches, clearing_prices

    def economic_dispatch_from_dataframe(self, scenarios_df: pd.DataFrame, bid_list: List[float] = None, 
                                       demand_col: str = 'demand', 
                                       generator_cols: List[str] = None,
                                       bid_cols: List[str] = None,
                                       pmin_default: float = 0.0) -> tuple[List[List[float]], List[float]]:
        """
        Solve economic dispatch for multiple scenarios using a DataFrame.
        
        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data with demand, generator capacity, and optionally bid columns
        bid_list : List[float], optional
            Array of cost/bid values for each generator (used if bid_cols not provided)
        demand_col : str, optional
            Name of the demand column (default: 'demand')
        generator_cols : List[str], optional
            List of generator column names. If None, auto-detects non-demand/bid columns
        bid_cols : List[str], optional
            List of bid column names for scenario-specific bids. If None, uses single bid_list
        pmin_default : float, optional
            Default Pmin value for all generators (default: 0.0)
            
        Returns
        -------
        all_dispatches : List[List[float]]
            Optimal dispatch for each generator in each scenario
        clearing_prices : List[float]
            Market clearing price for each scenario
        """
        
        # Auto-detect generator columns if not provided
        if generator_cols is None:
            # Get capacity/generation columns (exclude demand, scenario_id, and bid columns)
            exclude_cols = [demand_col, 'scenario_id']
            if bid_cols:
                exclude_cols.extend(bid_cols)
            generator_cols = [col for col in scenarios_df.columns 
                            if col not in exclude_cols]
        
        num_generators = len(generator_cols)
        
        # Handle bid data - either scenario-specific or single bid_list
        use_scenario_bids = bid_cols is not None
        
        if use_scenario_bids:
            # Validate bid columns match generator columns
            if len(bid_cols) != num_generators:
                raise ValueError(f"bid_cols length ({len(bid_cols)}) must match number of generators ({num_generators})")
        else:
            # Validate single bid_list
            if bid_list is None:
                raise ValueError("Either bid_list or bid_cols must be provided")
            if len(bid_list) != num_generators:
                raise ValueError(f"bid_list length ({len(bid_list)}) must match number of generators ({num_generators})")
        
        # Prepare data for single optimization call
        demand_scenarios = scenarios_df[demand_col].tolist()
        pmax_scenarios = []
        pmin_scenarios = []
        bid_scenarios = []
        
        for _, row in scenarios_df.iterrows():
            # Extract Pmax for this scenario
            pmax_scenario = [row[col] for col in generator_cols]
            pmin_scenario = [pmin_default] * num_generators
            
            # Extract bids for this scenario
            if use_scenario_bids:
                scenario_bids = [row[col] for col in bid_cols]
            else:
                scenario_bids = bid_list
            
            pmax_scenarios.append(pmax_scenario)
            pmin_scenarios.append(pmin_scenario)
            bid_scenarios.append(scenario_bids)
        
        # Call economic_dispatch once for all scenarios
        all_dispatches, clearing_prices = self.economic_dispatch(
            num_generators=num_generators,
            demand_scenarios=demand_scenarios,
            Pmax=pmax_scenarios,
            Pmin=pmin_scenarios,
            bid_list=bid_scenarios,
            num_scenarios=len(scenarios_df)
        )
            
        return all_dispatches, clearing_prices