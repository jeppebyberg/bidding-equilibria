from pyomo.environ import *
from typing import List
import pandas as pd

class EconomicDispatchModel:
    def __init__(self):
        pass

    def economic_dispatch(self, num_generators: int, demand_scenarios: List[float], Pmax, Pmin, bid_list) -> tuple[List[List[float]], List[float]]:
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
            
        Returns
        -------
        all_dispatches : List[List[float]]
            Optimal dispatch for each generator in each scenario
        clearing_prices : List[float]
            Market clearing price for each scenario
        """
        
        num_scenarios = len(demand_scenarios)

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
                
        return all_dispatches, clearing_prices

    def economic_dispatch_from_dataframe(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame,
                                       pmin_default: float = 0.0) -> tuple[List[List[float]], List[float], List[dict]]:
        """
        Solve economic dispatch for multiple scenarios using separate DataFrames.
        
        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data with demand, generator capacity, and bid columns
            Expected columns:
            - Demand column: should contain 'demand' or 'load' in name
            - Generator capacity columns: should end with '_cap' (e.g., 'G1_cap', 'G2_cap')
            - Generator bid columns: should end with '_bid' (e.g., 'G1_bid', 'G2_bid')
        costs_df : pd.DataFrame
            DataFrame containing static generator costs for profit calculation
            Expected columns: should end with '_cost' (e.g., 'G1_cost', 'G2_cost')
        pmin_default : float, optional
            Default Pmin value for all generators (default: 0.0)
            
        Returns
        -------
        all_dispatches : List[List[float]]
            Optimal dispatch for each generator in each scenario
        clearing_prices : List[float]
            Market clearing price for each scenario
        all_profits : List[dict]
            Profit for each generator in each scenario as dictionaries with generator names as keys
            Format: [{'G1': profit, 'G2': profit, ...}, {'G1': profit, 'G2': profit, ...}, ...]
            Profit = (clearing_price - cost) * dispatch
        """
        
        # Auto-detect demand column
        demand_col = None
        for col in scenarios_df.columns:
            if any(keyword in col.lower() for keyword in ['demand', 'load']):
                demand_col = col
                break
        
        if demand_col is None:
            raise ValueError("No demand column found. Expected column name containing 'demand' or 'load'")
        
        # Auto-detect generator capacity columns (those ending with "_cap")
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'")
        
        # Extract generator names from capacity columns
        generator_names = [col.replace('_cap', '') for col in capacity_cols]
        num_generators = len(generator_names)
        
        # Get bid columns (must exist in scenarios_df)
        bid_cols = [f"{gen_name}_bid" for gen_name in generator_names]
        missing_bid_cols = [col for col in bid_cols if col not in scenarios_df.columns]
        if missing_bid_cols:
            raise ValueError(f"Missing bid columns in scenarios_df: {missing_bid_cols}")
        
        # Get cost columns (must exist in costs_df)
        cost_cols = [f"{gen_name}_cost" for gen_name in generator_names]
        missing_cost_cols = [col for col in cost_cols if col not in costs_df.columns]
        if missing_cost_cols:
            raise ValueError(f"Missing cost columns in costs_df: {missing_cost_cols}")
        
        # Prepare data for optimization
        demand_scenarios = scenarios_df[demand_col].tolist()
        pmax_scenarios = []
        pmin_scenarios = []
        bid_scenarios = []
        
        for _, row in scenarios_df.iterrows():
            # Extract capacity and bids for this scenario
            pmax_scenario = [row[col] for col in capacity_cols]
            pmin_scenario = [pmin_default] * num_generators
            scenario_bids = [row[col] for col in bid_cols]
            
            pmax_scenarios.append(pmax_scenario)
            pmin_scenarios.append(pmin_scenario)
            bid_scenarios.append(scenario_bids)
        
        # Solve economic dispatch for all scenarios
        all_dispatches, clearing_prices = self.economic_dispatch(
            num_generators=num_generators,
            demand_scenarios=demand_scenarios,
            Pmax=pmax_scenarios,
            Pmin=pmin_scenarios,
            bid_list=bid_scenarios
        )

        # Calculate profits for each generator in each scenario
        generator_costs = [costs_df[col].iloc[0] for col in cost_cols]
        all_profits = []
        
        for scenario_idx, (dispatch, clearing_price) in enumerate(zip(all_dispatches, clearing_prices)):
            scenario_profits = {}
            for gen_idx, (gen_name, dispatch_amount, cost) in enumerate(zip(generator_names, dispatch, generator_costs)):
                # Profit = (clearing_price - cost) * dispatch_amount
                profit = (clearing_price - cost) * dispatch_amount
                scenario_profits[gen_name] = profit
            all_profits.append(scenario_profits)
                     
        return all_dispatches, clearing_prices, all_profits