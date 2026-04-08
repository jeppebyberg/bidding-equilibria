from pyomo.environ import *
from typing import List
import pandas as pd


class EconomicDispatchModel:
    """
    Economic Dispatch model for a single scenario.
    Same DataFrame-based interface as the MPEC OneScenario model.
    """

    def __init__(self,
                 scenarios_df: pd.DataFrame,
                 costs_df: pd.DataFrame,
                 pmin_default: float = 0.0,
                 scenario_id: int = 0):
        """
        Initialize Economic Dispatch model with scenario data.

        Parameters
        ----------
        scenarios_df : pd.DataFrame
            Single-row DataFrame containing scenario data.
            Expected columns:
            - Demand column: should contain 'demand' or 'load' in name
            - Generator capacity columns: should end with '_cap' (e.g., 'G1_cap')
            - Generator bid columns: should end with '_bid' (e.g., 'G1_bid')
        costs_df : pd.DataFrame
            DataFrame containing static generator costs.
            Expected columns: should end with '_cost' (e.g., 'G1_cost')
        pmin_default : float
            Default minimum generation for all generators (default: 0.0)
        scenario_id : int
            ID for the scenario (default: 0)
        """
        self.scenario_id = scenario_id
        self._extract_data(scenarios_df, costs_df, pmin_default)

        # Results (populated after solve)
        self.dispatch = None
        self.clearing_price = None

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

        # Extract data from first row (single scenario)
        row = scenarios_df.iloc[self.scenario_id]
        self.demand = row[demand_col]
        self.Pmax = [row[f"{gen}_cap"] for gen in self.generator_names]
        self.Pmin = [pmin_default] * self.num_generators
        self.bid_vector = [row[f"{gen}_bid"] for gen in self.generator_names]
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in self.generator_names]

    def build_model(self) -> None:
        self.model = ConcreteModel()
        
        # Define sets
        self.model.n_gen = Set(initialize=range(self.num_generators))

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        """Build Pyomo variables."""
        self.model.P_G = Var(self.model.n_gen, domain=Reals)
    
    def _build_objective(self) -> None:
        """Build Pyomo objective."""
        self.model.objective = Objective(
            expr=sum(self.bid_vector[i] * self.model.P_G[i] for i in self.model.n_gen),
            sense=minimize
        )

    def _build_constraints(self) -> None:
        """Build Pyomo constraints."""
        self.model.power_balance = Constraint(
            expr=self.demand - sum(self.model.P_G[i] for i in self.model.n_gen) == 0
        )
        self.model.gen_min = Constraint(self.model.n_gen, rule=lambda m, i: m.P_G[i] >= self.Pmin[i])
        self.model.gen_max = Constraint(self.model.n_gen, rule=lambda m, i: m.P_G[i] <= self.Pmax[i])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Build and solve the economic dispatch LP."""
        self.build_model()

        # Attach suffix to capture duals
        self.model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        results = solver.solve(self.model, tee=False)

        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            self.dispatch = [self.model.P_G[i].value for i in self.model.n_gen]
            self.clearing_price = -self.model.dual[self.model.power_balance]
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_dispatch(self) -> List[float]:
        """Return optimal dispatch for each generator."""
        return self.dispatch

    def get_clearing_price(self) -> float:
        """Return the market clearing price."""
        return self.clearing_price

    def get_generator_profits(self) -> List[float]:
        """
        Return profit for each generator: (clearing_price - cost) * dispatch.
        """
        return [
            (self.clearing_price - self.cost_vector[i]) * self.dispatch[i]
            for i in range(self.num_generators)
        ]