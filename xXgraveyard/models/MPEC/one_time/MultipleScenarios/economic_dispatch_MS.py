from pyomo.environ import *
from typing import List
import pandas as pd


class EconomicDispatchModel:
    """
    Economic Dispatch model for multiple scenarios.
    Same DataFrame-based interface as the MPEC MultipleScenarios model.
    """

    def __init__(self,
                 scenarios_df: pd.DataFrame,
                 costs_df: pd.DataFrame,
                 pmin_default: float = 0.0):
        """
        Initialize Economic Dispatch model with scenario data.

        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data (one row per scenario).
            Expected columns:
            - Demand column: should contain 'demand' or 'load' in name
            - Generator capacity columns: should end with '_cap' (e.g., 'G1_cap')
            - Generator bid columns: should end with '_bid' (e.g., 'G1_bid')
        costs_df : pd.DataFrame
            DataFrame containing static generator costs.
            Expected columns: should end with '_cost' (e.g., 'G1_cost')
        pmin_default : float
            Default minimum generation for all generators (default: 0.0)
        """
        self._extract_data(scenarios_df, costs_df, pmin_default)

        # Results (populated after solve)
        self.dispatches = None
        self.clearing_prices = None

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_data(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame,
                      pmin_default: float) -> None:
        """Extract scenario and generator data from DataFrames."""
        # Auto-detect demand column
        demand_col = None
        for col in scenarios_df.columns:
            if any(kw in col.lower() for kw in ['demand', 'load']):
                demand_col = col
                break
        if demand_col is None:
            raise ValueError("No demand column found. Expected column name containing 'demand' or 'load'")

        # Auto-detect generator capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'")

        self.generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(scenarios_df)

        # Extract per-scenario data
        self.demand_scenarios = scenarios_df[demand_col].tolist()
        self.pmax_scenarios = []
        self.pmin_scenarios = []
        self.bid_scenarios = []

        for _, row in scenarios_df.iterrows():
            self.pmax_scenarios.append([row[f"{gen}_cap"] for gen in self.generator_names])
            self.pmin_scenarios.append([pmin_default] * self.num_generators)
            self.bid_scenarios.append([row[f"{gen}_bid"] for gen in self.generator_names])

        # Extract costs (static across scenarios)
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in self.generator_names]

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Build and solve the economic dispatch LP for all scenarios jointly."""
        model = ConcreteModel()

        model.scenarios = Set(initialize=range(self.num_scenarios))
        model.generators = Set(initialize=range(self.num_generators))
        model.P_G = Var(model.scenarios, model.generators, domain=Reals)

        model.objective = Objective(
            expr=sum(
                self.bid_scenarios[s][g] * model.P_G[s, g]
                for s in model.scenarios
                for g in model.generators
            ),
            sense=minimize
        )

        def power_balance_rule(m, s):
            return self.demand_scenarios[s] - sum(m.P_G[s, g] for g in m.generators) == 0
        model.power_balance = Constraint(model.scenarios, rule=power_balance_rule)

        def gen_min_rule(m, s, g):
            return m.P_G[s, g] >= self.pmin_scenarios[s][g]
        model.gen_min = Constraint(model.scenarios, model.generators, rule=gen_min_rule)

        def gen_max_rule(m, s, g):
            return m.P_G[s, g] <= self.pmax_scenarios[s][g]
        model.gen_max = Constraint(model.scenarios, model.generators, rule=gen_max_rule)

        # Attach suffix to capture duals
        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=False)

        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            self.dispatches = []
            self.clearing_prices = []
            for s in range(self.num_scenarios):
                self.dispatches.append(
                    [model.P_G[s, g].value for g in range(self.num_generators)]
                )
                self.clearing_prices.append(-model.dual[model.power_balance[s]])
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_dispatches(self) -> List[List[float]]:
        """Return optimal dispatch for each generator in each scenario."""
        return self.dispatches

    def get_clearing_prices(self) -> List[float]:
        """Return the market clearing price for each scenario."""
        return self.clearing_prices

    def get_generator_profits(self) -> List[List[float]]:
        """
        Return profit for each generator in each scenario:
        profits[s][g] = (clearing_price[s] - cost[g]) * dispatch[s][g]
        """
        all_profits = []
        for s in range(self.num_scenarios):
            profits = [
                (self.clearing_prices[s] - self.cost_vector[g]) * self.dispatches[s][g]
                for g in range(self.num_generators)
            ]
            all_profits.append(profits)
        return all_profits