from pyomo.environ import *
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import ast
import numpy as np


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

    @staticmethod
    def _ensure_profile(value: Any, expected_len: int, column_name: str) -> List[float]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                return [float(value)] * expected_len
        if isinstance(value, (list, tuple)):
            if len(value) == expected_len:
                return [float(v) for v in value]
        return [float(value)] * expected_len

    @staticmethod
    def _infer_physical_from_block_name(block_name: str) -> str:
        if "_B" in block_name:
            return block_name.rsplit("_B", 1)[0]
        return block_name

    @staticmethod
    def _ramp_physical_names(ramps_df: pd.DataFrame) -> List[str]:
        return [
            str(col).removesuffix("_ramp_up")
            for col in ramps_df.columns
            if str(col).endswith("_ramp_up")
        ]

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

        # Auto-detect block capacity columns. With block-aware scenario data,
        # these are bidding blocks; old one-block cases continue to work.
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        if not capacity_cols:
            raise ValueError("No generator capacity columns found. Expected columns ending with '_cap'")

        self.block_names = [col.replace('_cap', '') for col in capacity_cols]
        self.num_blocks = len(self.block_names)
        ramp_physical_names = self._ramp_physical_names(ramps_df)
        if ramp_physical_names:
            self.physical_generator_names = ramp_physical_names
            self.block_to_physical = {
                block: self._infer_physical_from_block_name(block)
                for block in self.block_names
            }
        else:
            # Backwards-compatible fallback for old one-column-per-generator frames.
            self.physical_generator_names = list(self.block_names)
            self.block_to_physical = {name: name for name in self.block_names}

        missing_mapping = [name for name in self.block_names if name not in self.block_to_physical]
        if missing_mapping:
            raise ValueError(f"Missing block_to_physical mapping for blocks: {missing_mapping}")

        unknown_physical = sorted({
            physical
            for physical in self.block_to_physical.values()
            if physical not in self.physical_generator_names
        })
        if unknown_physical:
            available = ", ".join(self.physical_generator_names)
            raise ValueError(
                "Could not infer physical generator ownership for blocks. "
                f"Unknown physical generators from block names: {unknown_physical}. "
                f"Available ramp generators: {available}"
            )

        self.block_to_physical_idx = [
            self.physical_generator_names.index(self.block_to_physical[block])
            for block in self.block_names
        ]
        self.physical_to_block_indices: List[List[int]] = [
            [
                block_idx
                for block_idx, physical_idx in enumerate(self.block_to_physical_idx)
                if physical_idx == idx
            ]
            for idx in range(len(self.physical_generator_names))
        ]
        self.blocks_by_generator: Dict[int, List[int]] = {
            generator_idx: list(block_indices)
            for generator_idx, block_indices in enumerate(self.physical_to_block_indices)
        }
        self.local_blocks_by_generator: Dict[int, List[int]] = {
            generator_idx: list(range(len(block_indices)))
            for generator_idx, block_indices in self.blocks_by_generator.items()
        }
        self.local_to_global_block: Dict[Tuple[int, int], int] = {
            (generator_idx, local_block_idx): global_block_idx
            for generator_idx, block_indices in self.blocks_by_generator.items()
            for local_block_idx, global_block_idx in enumerate(block_indices)
        }
        self.global_to_local_block: Dict[int, Tuple[int, int]] = {
            global_block_idx: (generator_idx, local_block_idx)
            for (generator_idx, local_block_idx), global_block_idx in self.local_to_global_block.items()
        }
        self.generator_block_pairs: List[Tuple[int, int]] = list(self.local_to_global_block.keys())
        self.num_physical_generators = len(self.physical_generator_names)
        self.generator_names = list(self.physical_generator_names)
        self.num_generators = self.num_physical_generators
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

                for block in self.block_names:
                    cap = float(row[f"{block}_cap"])
                    if f"{block}_profile" in scenarios_df.columns:
                        wind_profile = self._convert_profile(
                            row[f"{block}_profile"],
                            self.num_time_steps,
                            f"{block}_profile",
                        )
                        cap = wind_profile[t]

                    pmax_t.append(cap)
                    pmin_t.append(float(pmin_default))
                    bid_profile_col = f"{block}_bid_profile"
                    if bid_profile_col in scenarios_df.columns:
                        bid_profile = self._ensure_profile(row[bid_profile_col], self.num_time_steps, bid_profile_col)
                        bid_t.append(float(bid_profile[t]))
                    elif f"{block}_bid" in scenarios_df.columns:
                        bid_t.append(float(row[f"{block}_bid"]))
                    else:
                        bid_t.append(float(costs_df[f"{block}_cost"].iloc[0]))

                pmax_scenario_by_time.append(pmax_t)
                pmin_scenario_by_time.append(pmin_t)
                bid_scenario_by_time.append(bid_t)

            self.pmax_scenarios.append(pmax_scenario_by_time)
            self.pmin_scenarios.append(pmin_scenario_by_time)
            self.bid_scenarios.append(bid_scenario_by_time)

        # Extract static block costs and physical ramp limits.
        self.cost_vector = [float(costs_df[f"{block}_cost"].iloc[0]) for block in self.block_names]
        self.block_cost_vector = list(self.cost_vector)

        self.ramp_vector_up = []
        self.ramp_vector_down = []
        for physical in self.physical_generator_names:
            if f"{physical}_ramp_up" in ramps_df.columns:
                if f"{physical}_ramp_down" not in ramps_df.columns:
                    raise ValueError(f"Missing ramp-down column for physical generator '{physical}'")
                self.ramp_vector_up.append(float(ramps_df[f"{physical}_ramp_up"].iloc[0]))
                self.ramp_vector_down.append(float(ramps_df[f"{physical}_ramp_down"].iloc[0]))
            else:
                # Backwards-compatible fallback for old frames.
                first_block = self.block_names[self.physical_to_block_indices[len(self.ramp_vector_up)][0]]
                self.ramp_vector_up.append(float(ramps_df[f"{first_block}_ramp_up"].iloc[0]))
                self.ramp_vector_down.append(float(ramps_df[f"{first_block}_ramp_down"].iloc[0]))

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
        self.dispatches = None
        self.block_dispatches = None
        self.block_dispatches_by_generator = None
        self.physical_dispatches = None
        self.clearing_prices = None
        self.dual_variables = None

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
            return self.demand_scenarios[s][t] - sum(
                m.P[i, b, t, s]
                for (i, b) in m.generator_blocks
            ) == 0
        
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
                -sum(m.P[g, b, t, s] for b in self.local_blocks_by_generator[g])
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
            for s in range(self.num_scenarios):
                scenario_dispatch = []
                scenario_prices = []
                scenario_lambda = []
                scenario_mu_max = []
                scenario_mu_min = []
                scenario_mu_up = []
                scenario_mu_down = []
                scenario_block_dispatch = []
                scenario_block_dispatch_by_generator = []
                for t in range(self.num_time_steps):
                    block_dispatch_t = [0.0] * self.num_blocks
                    block_dispatch_by_generator_t = []
                    for i in range(self.num_physical_generators):
                        generator_blocks_t = []
                        for local_block in self.local_blocks_by_generator[i]:
                            global_block = self.local_to_global_block[(i, local_block)]
                            value = model.P[i, local_block, t, s].value
                            block_dispatch_t[global_block] = value
                            generator_blocks_t.append(value)
                        block_dispatch_by_generator_t.append(generator_blocks_t)
                    physical_dispatch_t = [
                        sum(block_dispatch_t[b] for b in block_indices)
                        for block_indices in self.physical_to_block_indices
                    ]
                    scenario_block_dispatch.append(block_dispatch_t)
                    scenario_block_dispatch_by_generator.append(block_dispatch_by_generator_t)
                    scenario_dispatch.append(physical_dispatch_t)
                    balance_dual = self._scaled_dual(model, model.power_balance[t, s])
                    kkt_lambda = -balance_dual
                    scenario_lambda.append(kkt_lambda)
                    scenario_prices.append(kkt_lambda)

                    mu_max_t = []
                    mu_min_t = []
                    for global_block in range(self.num_blocks):
                        generator_idx, local_block = self.global_to_local_block[global_block]
                        mu_max_t.append(
                            self._nonnegative_scaled_dual(
                                model,
                                model.gen_max[generator_idx, local_block, t, s],
                            )
                        )
                        mu_min_t.append(
                            self._nonnegative_scaled_dual(
                                model,
                                model.gen_min[generator_idx, local_block, t, s],
                            )
                        )
                    scenario_mu_max.append(mu_max_t)
                    scenario_mu_min.append(mu_min_t)
                    scenario_mu_up.append([
                        self._nonnegative_scaled_dual(
                            model,
                            model.ramp_up_initial[i, s] if (t == 0 and self.P_init is not None) else model.ramp_up[i, t, s],
                        )
                        if not (t == 0 and self.P_init is None) else 0.0
                        for i in range(self.num_physical_generators)
                    ])
                    scenario_mu_down.append([
                        self._nonnegative_scaled_dual(
                            model,
                            model.ramp_down_initial[i, s] if (t == 0 and self.P_init is not None) else model.ramp_down[i, t, s],
                        )
                        if not (t == 0 and self.P_init is None) else 0.0
                        for i in range(self.num_physical_generators)
                    ])

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
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

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
        Return KKT-style ED dual variables.

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

    def get_scenario_kkt_data(self, scenario_idx: int) -> Dict[str, np.ndarray]:
        """
        Return one scenario's primal and dual solution arrays for sensitivity code.

        P_block, mu_max, and mu_min are flattened global-block arrays. P_phys,
        mu_up, and mu_down are physical-generator-level. P is kept as a
        compatibility alias for block dispatch because KKT stationarity is now
        block-level.
        """
        if self.block_dispatches is None or self.dual_variables is None:
            raise RuntimeError("Solve the economic dispatch model before requesting KKT data.")

        s = int(scenario_idx)
        if s < 0 or s >= self.num_scenarios:
            raise IndexError(f"scenario_idx must be in [0, {self.num_scenarios - 1}], got {s}")

        return {
            "P": np.asarray(self.block_dispatches[s], dtype=np.float64).T,
            "P_block": np.asarray(self.block_dispatches[s], dtype=np.float64).T,
            "P_phys": np.asarray(self.physical_dispatches[s], dtype=np.float64).T,
            "lambda_": np.asarray(self.dual_variables["lambda"][s], dtype=np.float64),
            "mu_max": np.asarray(self.dual_variables["mu_max"][s], dtype=np.float64).T,
            "mu_min": np.asarray(self.dual_variables["mu_min"][s], dtype=np.float64).T,
            "mu_up": np.asarray(self.dual_variables["mu_up"][s], dtype=np.float64).T,
            "mu_down": np.asarray(self.dual_variables["mu_down"][s], dtype=np.float64).T,
        }

    def get_scenario_kkt_parameters(self, scenario_idx: int) -> Dict[str, np.ndarray]:
        """
        Return one scenario's ED parameters in the shape expected by kkt_sensitivity.py.

        The quadratic ED objective in this model uses beta_coeff * P^2. The
        sensitivity module writes the objective as 0.5 * beta * P^2, so this
        method returns beta = 2 * beta_coeff.
        """
        s = int(scenario_idx)
        if s < 0 or s >= self.num_scenarios:
            raise IndexError(f"scenario_idx must be in [0, {self.num_scenarios - 1}], got {s}")
        if self.P_init is None:
            raise RuntimeError("p_init is required to build KKT parameter data with initial ramp constraints.")

        return {
            "alpha": np.asarray(self.bid_scenarios[s], dtype=np.float64).T,
            "beta": np.full(
                (self.num_blocks, self.num_time_steps),
                2.0 * float(self.beta_coeff),
                dtype=np.float64,
            ),
            "demand": np.asarray(self.demand_scenarios[s], dtype=np.float64),
            "pmax": np.asarray(self.pmax_scenarios[s], dtype=np.float64).T,
            "pmin": np.asarray(self.pmin_scenarios[s], dtype=np.float64).T,
            "ramp_up": np.asarray(self.ramp_vector_up, dtype=np.float64),
            "ramp_down": np.asarray(self.ramp_vector_down, dtype=np.float64),
            "p_initial": np.asarray(self.P_init[s], dtype=np.float64),
            "physical_to_block_indices": [list(blocks) for blocks in self.physical_to_block_indices],
            "block_to_physical_idx": list(self.block_to_physical_idx),
            "num_physical_generators": self.num_physical_generators,
            "num_blocks": self.num_blocks,
        }

    def get_generator_profits(self) -> List[List[float]]:
        """
        Return total profit per physical generator in each scenario.

        Revenue is based on physical dispatch, while production cost is summed
        over the generator-local bidding blocks owned by that physical generator.
        """
        all_profits = []
        for s in range(self.num_scenarios):
            profits = []
            for g in range(self.num_physical_generators):
                revenue_g = sum(
                    self.clearing_prices[s][t] * self.physical_dispatches[s][t][g]
                    for t in range(self.num_time_steps)
                )
                cost_g = sum(
                    self.block_cost_vector[self.local_to_global_block[(g, local_block)]]
                    * self.block_dispatches_by_generator[s][t][g][local_block]
                    for t in range(self.num_time_steps)
                    for local_block in self.local_blocks_by_generator[g]
                )
                profits.append(float(revenue_g - cost_g))
            all_profits.append(profits)
        return all_profits

    def get_block_profits(self) -> List[List[float]]:
        """
        Return total profit per block in each scenario:
        profits[s][global_block] =
        sum_t (lambda[s][t] - cost_block[global_block]) * P_block[s][t][global_block].
        """
        all_profits = []
        for s in range(self.num_scenarios):
            profits = []
            for b in range(self.num_blocks):
                profit_g = sum(
                    (self.clearing_prices[s][t] - self.block_cost_vector[b]) * self.block_dispatches[s][t][b]
                    for t in range(self.num_time_steps)
                )
                profits.append(float(profit_g))
            all_profits.append(profits)
        return all_profits
