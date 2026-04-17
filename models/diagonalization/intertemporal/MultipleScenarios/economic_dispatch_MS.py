from pyomo.environ import *
from typing import List, Any, Optional
import pandas as pd
import ast


class EconomicDispatchModel:
    """
    Intertemporal Economic Dispatch model for multiple scenarios.

    This model optimizes dispatch across scenarios and time steps jointly,
    enforcing per-time power balance and ramp constraints between consecutive
    time steps.
    """

    def __init__(self,
                 scenarios_df: pd.DataFrame,
                 costs_df: pd.DataFrame,
                 ramps_df: pd.DataFrame,
                 p_init: Optional[List[List[float]]] = None,
                 pmin_default: float = 0.0):
        """
        Initialize Economic Dispatch model with scenario data.

        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data (one row per scenario).
            Expected columns:
            - Demand profile column: should contain 'demand_profile'
            - Generator capacity columns: should end with '_cap' (e.g., 'G1_cap')
            - Generator bid columns: should end with '_bid' (e.g., 'G1_bid')
            - Optional wind profile columns: 'W*_profile' (list values)
            - Optional 'time_steps' column
        costs_df : pd.DataFrame
            DataFrame containing static generator costs.
            Expected columns: should end with '_cost' (e.g., 'G1_cost')
        p_init : Optional[List[List[float]]]
            Initial dispatch levels indexed as [scenario][generator]. If
            provided, the first time-step ramp constraints are enforced
            against these values to mirror the MPEC formulation.
        pmin_default : float
            Default minimum generation for all generators (default: 0.0)
        """
        self.P_init = p_init
        self._extract_data(scenarios_df, costs_df, ramps_df, pmin_default)

        if self.P_init is not None:
            if len(self.P_init) != self.num_scenarios:
                raise ValueError(
                    f"p_init must have one row per scenario: expected {self.num_scenarios}, got {len(self.P_init)}"
                )
            for s, row in enumerate(self.P_init):
                if len(row) != self.num_generators:
                    raise ValueError(
                        f"p_init row {s} must have one value per generator: expected {self.num_generators}, got {len(row)}"
                    )

        # Results (populated after solve)
        self.dispatches = None
        self.clearing_prices = None

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_profile(value: Any, expected_len: int, column_name: str) -> List[float]:
        """Convert profile-like input into a numeric list with expected length. This gives per row profile"""
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except Exception as exc:
                raise ValueError(f"Could not parse profile column '{column_name}': {exc}") from exc
            value = parsed

        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Column '{column_name}' must contain a list/tuple of length {expected_len}")

        if len(value) != expected_len:
            raise ValueError(
                f"Profile length mismatch in column '{column_name}': expected {expected_len}, got {len(value)}"
            )

        try:
            profile = [float(v) for v in value]
        except Exception as exc:
            raise ValueError(f"Profile column '{column_name}' contains non-numeric values") from exc

        return profile

    def _extract_data(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame, ramps_df: pd.DataFrame,
                      pmin_default: float) -> None:
        """Extract scenario and generator data from DataFrames."""
        # Auto-detect demand profile column
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

                    pmax_t.append(cap)
                    pmin_t.append(float(pmin_default))
                    bid_t.append(float(row[f"{gen}_bid_profile"][t]))

                pmax_scenario_by_time.append(pmax_t)
                pmin_scenario_by_time.append(pmin_t)
                bid_scenario_by_time.append(bid_t)

            self.pmax_scenarios.append(pmax_scenario_by_time)
            self.pmin_scenarios.append(pmin_scenario_by_time)
            self.bid_scenarios.append(bid_scenario_by_time)

        # Extract costs (static across scenarios)
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in self.generator_names]

        # Extract ramps (static across scenarios)
        self.ramp_vector_up = [ramps_df[f"{gen}_ramp_up"].iloc[0] for gen in self.generator_names]
        self.ramp_vector_down = [ramps_df[f"{gen}_ramp_down"].iloc[0] for gen in self.generator_names]

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Build and solve the intertemporal economic dispatch LP."""
        model = ConcreteModel()

        model.scenarios = Set(initialize=range(self.num_scenarios))
        model.time_steps = Set(initialize=range(self.num_time_steps))
        model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        model.generators = Set(initialize=range(self.num_generators))
        model.P = Var(model.scenarios, model.time_steps, model.generators, domain=Reals)

        model.objective = Objective(
            expr= 1 / self.num_scenarios * sum(
                self.bid_scenarios[s][t][g] * model.P[s, t, g]
                for s in model.scenarios
                for t in model.time_steps
                for g in model.generators
            ),
            sense=minimize
        )

        def power_balance_rule(m, s, t):
            return self.demand_scenarios[s][t] - sum(m.P[s, t, g] for g in m.generators) == 0
        model.power_balance = Constraint(model.scenarios, model.time_steps, rule=power_balance_rule)

        def gen_max_rule(m, s, t, g):
            return m.P[s, t, g] - self.pmax_scenarios[s][t][g] <= 0
        model.gen_max = Constraint(model.scenarios, model.time_steps, model.generators, rule=gen_max_rule)

        def gen_min_rule(m, s, t, g):
            return - m.P[s, t, g] + self.pmin_scenarios[s][t][g] <= 0
        model.gen_min = Constraint(model.scenarios, model.time_steps, model.generators, rule=gen_min_rule)

        def ramp_up_rule(m, s, t, g):
            return m.P[s, t, g] - m.P[s, t - 1, g] - self.ramp_vector_up[g] <= 0

        def ramp_down_rule(m, s, t, g):
            return - m.P[s, t, g] + m.P[s, t - 1, g] - self.ramp_vector_down[g] <= 0

        def ramp_up_initial_rule(m, s, g):
            return m.P[s, 0, g] - self.P_init[s][g] - self.ramp_vector_up[g] <= 0

        def ramp_down_initial_rule(m, s, g):
            return - m.P[s, 0, g] + self.P_init[s][g] - self.ramp_vector_down[g] <= 0

        model.ramp_up = Constraint(model.scenarios, model.time_steps_minus_1, model.generators, rule=ramp_up_rule)
        model.ramp_down = Constraint(model.scenarios, model.time_steps_minus_1, model.generators, rule=ramp_down_rule)
        model.ramp_up_initial = Constraint(model.scenarios, model.generators, rule=ramp_up_initial_rule)
        model.ramp_down_initial = Constraint(model.scenarios, model.generators, rule=ramp_down_initial_rule)

        # Attach suffix to capture duals
        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=False)

        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            self.dispatches = []
            self.clearing_prices = []
            for s in range(self.num_scenarios):
                scenario_dispatch = []
                scenario_prices = []
                for t in range(self.num_time_steps):
                    scenario_dispatch.append(
                        [model.P[s, t, g].value for g in range(self.num_generators)]
                    )
                    scenario_prices.append(-model.dual[model.power_balance[s, t]] * self.num_scenarios)

                self.dispatches.append(
                    scenario_dispatch
                )
                self.clearing_prices.append(scenario_prices)
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_dispatches(self) -> List[List[List[float]]]:
        """Return optimal dispatch as [scenario][time][generator]."""
        return self.dispatches

    def get_clearing_prices(self) -> List[List[float]]:
        """Return clearing prices as [scenario][time]."""
        return self.clearing_prices

    def get_generator_profits(self) -> List[List[float]]:
        """
        Return total profit per generator in each scenario (summed across time):
        profits[s][g] = sum_t (clearing_price[s][t] - cost[g]) * dispatch[s][t][g]
        """
        all_profits = []
        for s in range(self.num_scenarios):
            profits = []
            for g in range(self.num_generators):
                profit_g = sum(
                    (self.clearing_prices[s][t] - self.cost_vector[g]) * self.dispatches[s][t][g]
                    for t in range(self.num_time_steps)
                )
                profits.append(float(profit_g))
            all_profits.append(profits)
        return all_profits