from pyomo.environ import *
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from models.helper import (
    BlockStructure,
    available_block_capacity,
    block_cost_vector,
    block_structure_from_dataframes,
    ensure_profile,
    find_demand_profile_column,
    infer_num_time_steps,
    parse_profile_exact_length,
    ramp_vectors,
)


class EconomicDispatchModel:
    """
    Intertemporal Economic Dispatch model for multiple scenarios.

    This model optimizes dispatch across scenarios and time steps jointly,
    enforcing per-time power balance and ramp constraints between consecutive
    time steps. Dispatch variables use generator-local bidding blocks:
    P[i, b, t, s] is block b local to physical generator i. Physical-generator
    ramping sums over those local blocks directly. Generators may own
    heterogeneous numbers of bidding blocks.
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
            - Bidding-block capacity columns: should end with '_cap' (e.g., 'G1_B1_cap')
            - Bidding-block bid columns: '<block>_bid_profile' or '<block>_bid'
            - Optional wind block profile columns: '<wind_block>_profile' (list values)
            - Optional 'time_steps' column
        costs_df : pd.DataFrame
            DataFrame containing static bidding-block costs.
            Expected columns: should end with '_cost' (e.g., 'G1_B1_cost')
        ramps_df : pd.DataFrame
            DataFrame containing physical-generator ramp limits.
            Expected columns: '<physical_generator>_ramp_up' and
            '<physical_generator>_ramp_down'. Physical generator ownership of
            blocks is inferred from names such as G1_B1 -> G1.
        p_init : Optional[List[List[float]]]
            Initial dispatch levels indexed as either [scenario][block] or
            [scenario][physical_generator]. The shape is inferred and converted
            to physical-generator totals internally, because initial ramp
            constraints are defined at the physical-generator level. If
            provided, the first time-step ramp constraints are enforced against
            these values to mirror the MPEC formulation.
        pmin_default : float
            Default minimum generation for all generators (default: 0.0)
        """
        self.P_init = p_init
        self._extract_data(scenarios_df, costs_df, ramps_df, pmin_default)

        if self.P_init is not None:
            self.P_init = self._normalize_p_init(self.P_init)

        # Results (populated after solve)
        self.dispatches = None
        self.block_dispatches = None
        self.block_dispatches_by_generator = None
        self.physical_dispatches = None
        self.clearing_prices = None
        self.dual_variables = None

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _set_block_structure(self, block_structure: BlockStructure) -> None:
        """Copy shared block/physical-generator mappings onto this model."""
        self.block_names = list(block_structure.block_names)
        self.num_blocks = len(self.block_names)
        self.physical_generator_names = list(block_structure.physical_generator_names)
        self.num_physical_generators = len(self.physical_generator_names)
        self.generator_names = list(self.physical_generator_names)
        self.block_to_physical = dict(block_structure.block_to_physical)
        self.block_to_physical_idx = list(block_structure.block_to_physical_idx)
        self.physical_to_block_indices = [
            list(blocks) for blocks in block_structure.physical_to_block_indices
        ]
        self.blocks_by_generator = {
            int(generator_idx): list(block_indices)
            for generator_idx, block_indices in block_structure.blocks_by_generator.items()
        }
        self.local_blocks_by_generator = {
            int(generator_idx): list(block_indices)
            for generator_idx, block_indices in block_structure.local_blocks_by_generator.items()
        }
        self.local_to_global_block = dict(block_structure.local_to_global_block)
        self.global_to_local_block = dict(block_structure.global_to_local_block)
        self.generator_block_pairs = list(block_structure.generator_block_pairs)

    def _extract_data(self, scenarios_df: pd.DataFrame, costs_df: pd.DataFrame, ramps_df: pd.DataFrame,
                      pmin_default: float) -> None:
        """Extract scenario and generator data from DataFrames."""
        demand_profile_col = find_demand_profile_column(scenarios_df)
        self._set_block_structure(block_structure_from_dataframes(scenarios_df, ramps_df))
        self.num_scenarios = len(scenarios_df)
        self.num_time_steps = infer_num_time_steps(scenarios_df)

        # Extract per-scenario data
        self.demand_scenarios: List[List[float]] = []
        self.pmax_scenarios = []
        self.pmin_scenarios = []
        self.bid_scenarios = []

        for scenario_id, (_, row) in enumerate(scenarios_df.iterrows()):
            demand_profile = parse_profile_exact_length(
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

                for block in self.block_names:
                    pmax_t.append(available_block_capacity(scenarios_df, block, scenario_id, t))
                    pmin_t.append(float(pmin_default))
                    bid_profile_col = f"{block}_bid_profile"
                    bid_col = (
                        bid_profile_col
                        if bid_profile_col in scenarios_df.columns
                        else f"{block}_bid"
                    )
                    bid_profile = ensure_profile(row[bid_col], self.num_time_steps, bid_col)
                    bid_t.append(float(bid_profile[t]))

                pmax_scenario_by_time.append(pmax_t)
                pmin_scenario_by_time.append(pmin_t)
                bid_scenario_by_time.append(bid_t)

            self.pmax_scenarios.append(pmax_scenario_by_time)
            self.pmin_scenarios.append(pmin_scenario_by_time)
            self.bid_scenarios.append(bid_scenario_by_time)

        # Extract static block costs and physical ramp limits.
        self.cost_vector = block_cost_vector(costs_df, self.block_names)
        self.block_cost_vector = list(self.cost_vector)
        self.ramp_vector_up, self.ramp_vector_down = ramp_vectors(
            ramps_df,
            self.physical_generator_names,
        )

    def _normalize_p_init(self, p_init: List[List[float]]) -> List[List[float]]:
        """Return physical initial output as [scenario][physical_generator]."""
        if len(p_init) != self.num_scenarios:
            raise ValueError(
                f"p_init must have one row per scenario: expected {self.num_scenarios}, got {len(p_init)}"
            )

        normalized: List[List[float]] = []
        for s, row in enumerate(p_init):
            values = [float(v) for v in row]
            if len(values) == self.num_physical_generators:
                normalized.append(values)
            elif len(values) == self.num_blocks:
                normalized.append([
                    sum(values[b] for b in block_indices)
                    for block_indices in self.physical_to_block_indices
                ])
            else:
                raise ValueError(
                    f"p_init row {s} has {len(values)} values; expected either "
                    f"{self.num_physical_generators} physical generators or {self.num_blocks} blocks"
                )

        return normalized

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """
        Build and solve the intertemporal economic dispatch LP.

        The optimization variable P[i, b, t, s] uses b as a local bidding-block
        index for physical generator i. Flattened block results are reconstructed
        after the solve for compatibility with the sensitivity code and scripts.
        """
        self._reset_results()

        model = ConcreteModel()

        model.scenarios = Set(initialize=range(self.num_scenarios))
        model.time_steps = Set(initialize=range(self.num_time_steps))
        model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        model.physical_generators = Set(initialize=range(self.num_physical_generators))
        model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)
        model.P = Var(model.generator_blocks, model.time_steps, model.scenarios, domain=Reals)

        objective_expr = 1 / self.num_scenarios * sum(
            self.bid_scenarios[s][t][self.local_to_global_block[(i, b)]] * model.P[i, b, t, s]
            for s in model.scenarios
            for t in model.time_steps
            for (i, b) in model.generator_blocks
        )

        model.objective = Objective(expr=objective_expr, sense=minimize)

        def power_balance_rule(m, t, s):
            return self.demand_scenarios[s][t] - sum(m.P[i, b, t, s] for (i, b) in m.generator_blocks) == 0
        
        def gen_max_rule(m, i, b, t, s):
            global_block = self.local_to_global_block[(i, b)]
            return m.P[i, b, t, s] - self.pmax_scenarios[s][t][global_block] <= 0
        
        def gen_min_rule(m, i, b, t, s):
            global_block = self.local_to_global_block[(i, b)]
            return - m.P[i, b, t, s] + self.pmin_scenarios[s][t][global_block] <= 0

        def ramp_up_rule(m, g, t, s):
            return (
                  sum(m.P[g, b, t, s] for b in self.local_blocks_by_generator[g])
                - sum(m.P[g, b, t - 1, s] for b in self.local_blocks_by_generator[g])
                - self.ramp_vector_up[g]
                <= 0
            )

        def ramp_down_rule(m, g, t, s):
            return (
                - sum(m.P[g, b, t, s] for b in self.local_blocks_by_generator[g])
                + sum(m.P[g, b, t - 1, s] for b in self.local_blocks_by_generator[g])
                - self.ramp_vector_down[g]
                <= 0
            )

        def ramp_up_initial_rule(m, g, s):
            return sum(m.P[g, b, 0, s] for b in self.local_blocks_by_generator[g]) - self.P_init[s][g] - self.ramp_vector_up[g] <= 0

        def ramp_down_initial_rule(m, g, s):
            return -sum(m.P[g, b, 0, s] for b in self.local_blocks_by_generator[g]) + self.P_init[s][g] - self.ramp_vector_down[g] <= 0

        model.power_balance = Constraint(model.time_steps, model.scenarios, rule=power_balance_rule)
        model.gen_max = Constraint(model.generator_blocks, model.time_steps, model.scenarios, rule=gen_max_rule)
        model.gen_min = Constraint(model.generator_blocks, model.time_steps, model.scenarios, rule=gen_min_rule)
        
        model.ramp_up = Constraint(model.physical_generators, model.time_steps_minus_1, model.scenarios, rule=ramp_up_rule)
        model.ramp_down = Constraint(model.physical_generators, model.time_steps_minus_1, model.scenarios, rule=ramp_down_rule)
        if self.P_init is not None:
            model.ramp_up_initial = Constraint(model.physical_generators, model.scenarios, rule=ramp_up_initial_rule)
            model.ramp_down_initial = Constraint(model.physical_generators, model.scenarios, rule=ramp_down_initial_rule)

        # Attach suffix to capture duals
        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        solver.options["OutputFlag"] = 0
        results = solver.solve(model, tee=False)

        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            self._store_solution_results(model)
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    def _reset_results(self) -> None:
        self.dispatches = None
        self.block_dispatches = None
        self.block_dispatches_by_generator = None
        self.physical_dispatches = None
        self.clearing_prices = None
        self.dual_variables = None

    def _block_dispatch_at(
        self,
        model: ConcreteModel,
        scenario_id: int,
        time_id: int,
    ) -> Tuple[List[float], List[List[float]]]:
        """Read flattened and generator-local block dispatch for one time step."""
        block_dispatch = [0.0] * self.num_blocks
        block_dispatch_by_generator = []
        for generator_idx in range(self.num_physical_generators):
            generator_blocks = []
            for local_block in self.local_blocks_by_generator[generator_idx]:
                global_block = self.local_to_global_block[(generator_idx, local_block)]
                value = model.P[generator_idx, local_block, time_id, scenario_id].value
                block_dispatch[global_block] = value
                generator_blocks.append(value)
            block_dispatch_by_generator.append(generator_blocks)
        return block_dispatch, block_dispatch_by_generator

    def _physical_dispatch_from_blocks(self, block_dispatch: List[float]) -> List[float]:
        return [
            sum(block_dispatch[block_idx] for block_idx in block_indices)
            for block_indices in self.physical_to_block_indices
        ]

    def _capacity_duals_at(
        self,
        model: ConcreteModel,
        scenario_id: int,
        time_id: int,
    ) -> Tuple[List[float], List[float]]:
        mu_max = []
        mu_min = []
        for global_block in range(self.num_blocks):
            generator_idx, local_block = self.global_to_local_block[global_block]
            mu_max.append(
                self._nonnegative_scaled_dual(
                    model,
                    model.gen_max[generator_idx, local_block, time_id, scenario_id],
                )
            )
            mu_min.append(
                self._nonnegative_scaled_dual(
                    model,
                    model.gen_min[generator_idx, local_block, time_id, scenario_id],
                )
            )
        return mu_max, mu_min

    def _ramp_duals_at(
        self,
        model: ConcreteModel,
        scenario_id: int,
        time_id: int,
    ) -> Tuple[List[float], List[float]]:
        if time_id == 0 and self.P_init is None:
            zeros = [0.0] * self.num_physical_generators
            return zeros, list(zeros)

        mu_up = []
        mu_down = []
        for generator_idx in range(self.num_physical_generators):
            ramp_up_constraint = (
                model.ramp_up_initial[generator_idx, scenario_id]
                if time_id == 0
                else model.ramp_up[generator_idx, time_id, scenario_id]
            )
            ramp_down_constraint = (
                model.ramp_down_initial[generator_idx, scenario_id]
                if time_id == 0
                else model.ramp_down[generator_idx, time_id, scenario_id]
            )
            mu_up.append(self._nonnegative_scaled_dual(model, ramp_up_constraint))
            mu_down.append(self._nonnegative_scaled_dual(model, ramp_down_constraint))
        return mu_up, mu_down

    def _store_solution_results(self, model: ConcreteModel) -> None:
        """Populate dispatch, price, and dual result containers after an optimal solve."""
        self.dispatches = []
        self.block_dispatches = []
        self.block_dispatches_by_generator = []
        self.physical_dispatches = []
        self.clearing_prices = []
        self.dual_variables = {
            "lambda": [],
            "mu_max": [],
            "mu_min": [],
            "mu_up": [],
            "mu_down": [],
        }

        for scenario_id in range(self.num_scenarios):
            scenario_dispatch = []
            scenario_prices = []
            scenario_lambda = []
            scenario_mu_max = []
            scenario_mu_min = []
            scenario_mu_up = []
            scenario_mu_down = []
            scenario_block_dispatch = []
            scenario_block_dispatch_by_generator = []

            for time_id in range(self.num_time_steps):
                block_dispatch, block_dispatch_by_generator = self._block_dispatch_at(
                    model,
                    scenario_id,
                    time_id,
                )
                scenario_block_dispatch.append(block_dispatch)
                scenario_block_dispatch_by_generator.append(block_dispatch_by_generator)
                scenario_dispatch.append(self._physical_dispatch_from_blocks(block_dispatch))

                kkt_lambda = -self._scaled_dual(
                    model,
                    model.power_balance[time_id, scenario_id],
                )
                scenario_lambda.append(kkt_lambda)
                scenario_prices.append(kkt_lambda)

                mu_max, mu_min = self._capacity_duals_at(model, scenario_id, time_id)
                mu_up, mu_down = self._ramp_duals_at(model, scenario_id, time_id)
                scenario_mu_max.append(mu_max)
                scenario_mu_min.append(mu_min)
                scenario_mu_up.append(mu_up)
                scenario_mu_down.append(mu_down)

            self.block_dispatches.append(scenario_block_dispatch)
            self.block_dispatches_by_generator.append(scenario_block_dispatch_by_generator)
            self.physical_dispatches.append(scenario_dispatch)
            self.dispatches.append(scenario_dispatch)
            self.clearing_prices.append(scenario_prices)
            self.dual_variables["lambda"].append(scenario_lambda)
            self.dual_variables["mu_max"].append(scenario_mu_max)
            self.dual_variables["mu_min"].append(scenario_mu_min)
            self.dual_variables["mu_up"].append(scenario_mu_up)
            self.dual_variables["mu_down"].append(scenario_mu_down)

    def _scaled_dual(self, model: ConcreteModel, constraint: Constraint) -> float:
        """
        Return the Pyomo dual scaled back from the averaged multi-scenario objective.

        The model objective is divided by num_scenarios, so KKT multipliers for
        each scenario are scaled by the same factor. Multiplying by
        num_scenarios gives the per-scenario ED dual.
        """
        return float(model.dual.get(constraint, 0.0)) * self.num_scenarios

    def _nonnegative_scaled_dual(self, model: ConcreteModel, constraint: Constraint) -> float:
        """
        Return a nonnegative KKT multiplier for a <= constraint.

        Gurobi/Pyomo reports <= minimization duals with the opposite sign from
        the nonnegative multipliers used in the KKT equations, so we negate and
        scale them here.
        """
        return max(0.0, -self._scaled_dual(model, constraint))

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_dispatches(self) -> List[List[List[float]]]:
        """Return physical dispatch as [scenario][time][physical_generator]."""
        return self.physical_dispatches

    def get_block_dispatches(self) -> List[List[List[float]]]:
        """Return flattened block dispatch as [scenario][time][global_block]."""
        return self.block_dispatches

    def get_block_dispatches_by_generator(self) -> List[List[List[List[float]]]]:
        """
        Return local-block dispatch as [scenario][time][physical_generator][local_block].

        The local_block index is generator-specific, so generators can have
        different numbers of blocks.
        """
        return self.block_dispatches_by_generator

    def get_physical_dispatches(self) -> List[List[List[float]]]:
        """Return physical dispatch as [scenario][time][physical_generator]."""
        return self.physical_dispatches

    def get_block_names(self) -> List[str]:
        return list(self.block_names)

    def get_physical_generator_names(self) -> List[str]:
        return list(self.physical_generator_names)

    def get_block_to_physical_mapping(self) -> Dict[str, str]:
        return dict(self.block_to_physical)

    def get_blocks_by_generator(self) -> Dict[int, List[int]]:
        """Return physical generator index -> flattened global block indices."""
        return {generator_idx: list(blocks) for generator_idx, blocks in self.blocks_by_generator.items()}

    def get_local_to_global_block_mapping(self) -> Dict[Tuple[int, int], int]:
        """Return (physical_generator, local_block) -> flattened global block index."""
        return dict(self.local_to_global_block)

    def get_clearing_prices(self) -> List[List[float]]:
        """Return clearing prices as [scenario][time]."""
        return self.clearing_prices

    def get_dual_variables(self) -> Dict[str, List[Any]]:
        """
        Return ED dual variables.

        Shapes are:
        - lambda: [scenario][time]
        - mu_max, mu_min: [scenario][time][global_block]
        - mu_up, mu_down: [scenario][time][physical_generator]

        The inequality multipliers are returned as nonnegative KKT multipliers
        for constraints written in the model as g(P) <= 0. The lambda values use
        the KKT sign convention for D[t] - sum_i P[i,t] = 0, matching the
        clearing prices returned by get_clearing_prices().
        """
        return self.dual_variables

if __name__ == "__main__":
    from config.scenarios.scenario_generator import ScenarioManager

    case = "test_case_bidding_blocks"
    regime_set = "debugging"
    seed = 1

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )
    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]
    players_config = scenario_manager.get_players_config()

    ed_model = EconomicDispatchModel(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        p_init=None
    )

    ed_model.solve()
