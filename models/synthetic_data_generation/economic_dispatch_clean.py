from __future__ import annotations

from typing import Optional

import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Objective,
    Reals,
    Set,
    SolverFactory,
    Suffix,
    Var,
    minimize,
)

from models.helper import (
    available_block_capacity,
    block_structure_from_dataframes,
    ensure_profile,
    find_demand_profile_column,
    infer_num_time_steps,
    parse_profile_exact_length,
    ramp_vectors,
)


class EconomicDispatchModel:
    """
    Minimal economic dispatch model used by merit_order_best_response.

    Public API intentionally kept to the methods used by the heuristic:
    solve(), get_dispatches(), get_block_dispatches(), and get_clearing_prices().
    """

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        p_init: Optional[list[list[float]]] = None,
        pmin_default: float = 0.0,
    ) -> None:
        # Kept in the signature so MeritOrderHeuristic can swap ED implementations.
        _ = costs_df
        self._load_input_data(scenarios_df, ramps_df, pmin_default)
        self.P_init = None if p_init is None else self._normalize_p_init(p_init)

        self.dispatches: Optional[list[list[list[float]]]] = None
        self.block_dispatches: Optional[list[list[list[float]]]] = None
        self.clearing_prices: Optional[list[list[float]]] = None

    def _load_input_data(self, scenarios_df: pd.DataFrame, ramps_df: pd.DataFrame, pmin_default: float) -> None:
        block_structure = block_structure_from_dataframes(scenarios_df, ramps_df)
        self.block_names = list(block_structure.block_names)
        self.physical_generator_names = list(block_structure.physical_generator_names)
        self.physical_to_block_indices = [
            list(blocks) for blocks in block_structure.physical_to_block_indices
        ]
        self.local_blocks_by_generator = dict(block_structure.local_blocks_by_generator)
        self.local_to_global_block = dict(block_structure.local_to_global_block)
        self.generator_block_pairs = list(block_structure.generator_block_pairs)

        self.num_blocks = len(self.block_names)
        self.num_physical_generators = len(self.physical_generator_names)
        self.num_scenarios = len(scenarios_df)
        self.num_time_steps = infer_num_time_steps(scenarios_df)

        demand_column = find_demand_profile_column(scenarios_df)
        self.demand_scenarios: list[list[float]] = []
        self.pmax_scenarios: list[list[list[float]]] = []
        self.pmin_scenarios: list[list[list[float]]] = []
        self.bid_scenarios: list[list[list[float]]] = []

        for scenario_id, (_, row) in enumerate(scenarios_df.iterrows()):
            self.demand_scenarios.append(
                parse_profile_exact_length(
                    row[demand_column],
                    self.num_time_steps,
                    demand_column,
                )
            )

            pmax_by_time = []
            pmin_by_time = []
            bid_by_time = []
            for time_id in range(self.num_time_steps):
                pmax_t = []
                pmin_t = []
                bid_t = []
                for block_name in self.block_names:
                    pmax_t.append(
                        available_block_capacity(
                            scenarios_df,
                            block_name,
                            scenario_id,
                            time_id,
                        )
                    )
                    pmin_t.append(float(pmin_default))

                    bid_profile_col = f"{block_name}_bid_profile"
                    bid_col = (
                        bid_profile_col
                        if bid_profile_col in scenarios_df.columns
                        else f"{block_name}_bid"
                    )
                    bid_profile = ensure_profile(
                        row[bid_col],
                        self.num_time_steps,
                        bid_col,
                    )
                    bid_t.append(float(bid_profile[time_id]))

                pmax_by_time.append(pmax_t)
                pmin_by_time.append(pmin_t)
                bid_by_time.append(bid_t)

            self.pmax_scenarios.append(pmax_by_time)
            self.pmin_scenarios.append(pmin_by_time)
            self.bid_scenarios.append(bid_by_time)

        self.ramp_vector_up, self.ramp_vector_down = ramp_vectors(
            ramps_df,
            self.physical_generator_names,
        )

    def _normalize_p_init(self, p_init: list[list[float]]) -> list[list[float]]:
        if len(p_init) != self.num_scenarios:
            raise ValueError(
                f"p_init must have one row per scenario: expected {self.num_scenarios}, got {len(p_init)}"
            )

        normalized = []
        for scenario_id, row in enumerate(p_init):
            values = [float(value) for value in row]
            if len(values) == self.num_physical_generators:
                normalized.append(values)
                continue
            if len(values) == self.num_blocks:
                normalized.append(
                    [
                        sum(values[block_idx] for block_idx in block_indices)
                        for block_indices in self.physical_to_block_indices
                    ]
                )
                continue
            raise ValueError(
                f"p_init row {scenario_id} has {len(values)} values; expected either "
                f"{self.num_physical_generators} physical generators or {self.num_blocks} blocks"
            )

        return normalized

    def solve(self) -> None:
        self.dispatches = None
        self.block_dispatches = None
        self.clearing_prices = None

        model = ConcreteModel()
        model.scenarios = Set(initialize=range(self.num_scenarios))
        model.time_steps = Set(initialize=range(self.num_time_steps))
        model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        model.physical_generators = Set(initialize=range(self.num_physical_generators))
        model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)
        model.P = Var(
            model.generator_blocks,
            model.time_steps,
            model.scenarios,
            domain=Reals,
        )

        model.objective = Objective(
            expr=(1 / self.num_scenarios)
            * sum(
                self.bid_scenarios[s][t][self.local_to_global_block[(i, b)]]
                * model.P[i, b, t, s]
                for s in model.scenarios
                for t in model.time_steps
                for (i, b) in model.generator_blocks
            ),
            sense=minimize,
        )

        def power_balance_rule(m, t, s):
            return (
                self.demand_scenarios[s][t]
                - sum(m.P[i, b, t, s] for (i, b) in m.generator_blocks)
                == 0
            )

        def gen_max_rule(m, i, b, t, s):
            global_block = self.local_to_global_block[(i, b)]
            return m.P[i, b, t, s] <= self.pmax_scenarios[s][t][global_block]

        def gen_min_rule(m, i, b, t, s):
            global_block = self.local_to_global_block[(i, b)]
            return m.P[i, b, t, s] >= self.pmin_scenarios[s][t][global_block]

        def ramp_up_rule(m, g, t, s):
            return (
                sum(m.P[g, b, t, s] for b in self.local_blocks_by_generator[g])
                - sum(m.P[g, b, t - 1, s] for b in self.local_blocks_by_generator[g])
                <= self.ramp_vector_up[g]
            )

        def ramp_down_rule(m, g, t, s):
            return (
                sum(m.P[g, b, t - 1, s] for b in self.local_blocks_by_generator[g])
                - sum(m.P[g, b, t, s] for b in self.local_blocks_by_generator[g])
                <= self.ramp_vector_down[g]
            )

        model.power_balance = Constraint(
            model.time_steps,
            model.scenarios,
            rule=power_balance_rule,
        )
        model.gen_max = Constraint(
            model.generator_blocks,
            model.time_steps,
            model.scenarios,
            rule=gen_max_rule,
        )
        model.gen_min = Constraint(
            model.generator_blocks,
            model.time_steps,
            model.scenarios,
            rule=gen_min_rule,
        )
        model.ramp_up = Constraint(
            model.physical_generators,
            model.time_steps_minus_1,
            model.scenarios,
            rule=ramp_up_rule,
        )
        model.ramp_down = Constraint(
            model.physical_generators,
            model.time_steps_minus_1,
            model.scenarios,
            rule=ramp_down_rule,
        )

        if self.P_init is not None:

            def ramp_up_initial_rule(m, g, s):
                return (
                    sum(m.P[g, b, 0, s] for b in self.local_blocks_by_generator[g])
                    - self.P_init[s][g]
                    <= self.ramp_vector_up[g]
                )

            def ramp_down_initial_rule(m, g, s):
                return (
                    self.P_init[s][g]
                    - sum(m.P[g, b, 0, s] for b in self.local_blocks_by_generator[g])
                    <= self.ramp_vector_down[g]
                )

            model.ramp_up_initial = Constraint(
                model.physical_generators,
                model.scenarios,
                rule=ramp_up_initial_rule,
            )
            model.ramp_down_initial = Constraint(
                model.physical_generators,
                model.scenarios,
                rule=ramp_down_initial_rule,
            )

        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        solver.options["OutputFlag"] = 0
        results = solver.solve(model, tee=False)

        if str(results.solver.status).lower() != "ok" or str(results.solver.termination_condition).lower() != "optimal":
            raise RuntimeError(
                "Economic dispatch solve failed: "
                f"status={results.solver.status}, "
                f"termination={results.solver.termination_condition}"
            )

        self._store_results(model)

    def _store_results(self, model: ConcreteModel) -> None:
        self.dispatches = []
        self.block_dispatches = []
        self.clearing_prices = []

        for scenario_id in range(self.num_scenarios):
            scenario_dispatches = []
            scenario_block_dispatches = []
            scenario_prices = []

            for time_id in range(self.num_time_steps):
                block_dispatch = [0.0] * self.num_blocks
                for generator_idx, local_block in self.generator_block_pairs:
                    global_block = self.local_to_global_block[(generator_idx, local_block)]
                    block_dispatch[global_block] = float(
                        model.P[generator_idx, local_block, time_id, scenario_id].value
                    )

                scenario_block_dispatches.append(block_dispatch)
                scenario_dispatches.append(
                    [
                        sum(block_dispatch[block_idx] for block_idx in block_indices)
                        for block_indices in self.physical_to_block_indices
                    ]
                )
                scenario_prices.append(
                    -float(model.dual.get(model.power_balance[time_id, scenario_id], 0.0))
                    * self.num_scenarios
                )

            self.dispatches.append(scenario_dispatches)
            self.block_dispatches.append(scenario_block_dispatches)
            self.clearing_prices.append(scenario_prices)

    def get_dispatches(self) -> Optional[list[list[list[float]]]]:
        return self.dispatches

    def get_block_dispatches(self) -> Optional[list[list[list[float]]]]:
        return self.block_dispatches

    def get_clearing_prices(self) -> Optional[list[list[float]]]:
        return self.clearing_prices
