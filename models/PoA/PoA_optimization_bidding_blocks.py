from pyomo.environ import *
import numpy as np
import pandas as pd
import json
import yaml
import ast
from pathlib import Path
from typing import Any, Optional

import time

from config.scenarios.scenario_generator import ScenarioManager
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel


class PoAOptimizationBiddingBlocks:
    """
    Block-aware Price of Anarchy optimization.

    Dispatch and bids are indexed by physical generator and local bidding block:
    P_eq[i, b, t], P_opt[i, b, t], alpha[i, b, t].
    Physical ramp constraints sum over each generator's local bidding blocks.
    """

    normalization_epsilon = 1e-12

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        p_init: Optional[list[float] | list[list[float]]] = None,
        num_time_steps: Optional[int] = None,
        support_set_config: Optional[dict[str, Any]] = None,
        nn_model_dir: Optional[str | Path] = None,
        nn_normalization_stats_path: Optional[str | Path] = None,
        big_m_complementarity: float = 1e3,
        lambda_bound: float = 75.0,
        capacity_dual_bound: float = 100.0,
        ramp_dual_bound: float = 5.0,
        big_m_relu: Optional[float] = None,
        reference_case: str = "test_case_bidding_blocks",
    ):
        self.scenarios_df = scenarios_df.reset_index(drop=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.requested_p_init = p_init
        self.support_set_config = support_set_config or {}
        self.nn_model_dir = Path(nn_model_dir) if nn_model_dir is not None else None
        self.nn_normalization_stats_path = (
            Path(nn_normalization_stats_path)
            if nn_normalization_stats_path is not None
            else None
        )
        self.big_m_complementarity = float(big_m_complementarity)
        self.lambda_bound = float(lambda_bound)
        self.capacity_dual_bound = float(capacity_dual_bound)
        self.ramp_dual_bound = float(ramp_dual_bound)
        self.big_m_relu = float(big_m_relu or big_m_complementarity)
        self.reference_case = reference_case

        self._initialize_block_structure_from_ed()
        self.num_time_steps = int(num_time_steps or self._infer_num_time_steps())
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive")

        self.static_block_capacity = [
            float(self.scenarios_df[f"{block}_cap"].iloc[0])
            for block in self.block_names
        ]
        self.static_physical_capacity = [
            sum(self.static_block_capacity[g] for g in self.physical_to_block_indices[i])
            for i in range(self.num_physical_generators)
        ]
        self.p_init = self._normalize_p_init(self.requested_p_init)
        self._configure_support_set_parameters()

        self.nn_policies: dict[str, dict[str, Any]] = {}
        self.nn_stats: dict[str, Any] = {}
        if self.nn_model_dir is not None:
            self._load_nn_policies()
            self._load_nn_normalization_stats()

    # ------------------------------------------------------------------
    # Data and configuration
    # ------------------------------------------------------------------

    def _initialize_block_structure_from_ed(self) -> None:
        mapping_model = EconomicDispatchModel(
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
            p_init=None,
        )

        self.block_names = list(mapping_model.block_names)
        self.num_blocks = int(mapping_model.num_blocks)
        self.physical_generator_names = list(mapping_model.physical_generator_names)
        self.num_physical_generators = int(mapping_model.num_physical_generators)
        self.block_to_physical = dict(mapping_model.block_to_physical)
        self.block_to_physical_idx = list(mapping_model.block_to_physical_idx)
        self.physical_to_block_indices = [
            list(blocks) for blocks in mapping_model.physical_to_block_indices
        ]
        self.blocks_by_generator = {
            int(i): list(blocks) for i, blocks in mapping_model.blocks_by_generator.items()
        }
        self.local_blocks_by_generator = {
            int(i): list(blocks)
            for i, blocks in mapping_model.local_blocks_by_generator.items()
        }
        self.local_to_global_block = dict(mapping_model.local_to_global_block)
        self.global_to_local_block = dict(mapping_model.global_to_local_block)
        self.generator_block_pairs = list(mapping_model.generator_block_pairs)
        self.block_cost_vector = [float(v) for v in mapping_model.block_cost_vector]
        self.ramp_vector_up = [float(v) for v in mapping_model.ramp_vector_up]
        self.ramp_vector_down = [float(v) for v in mapping_model.ramp_vector_down]

        self.wind_physical_generator_ids = [
            i
            for i, name in enumerate(self.physical_generator_names)
            if self._is_wind_name(name)
        ]
        self.conventional_physical_generator_ids = [
            i
            for i in range(self.num_physical_generators)
            if i not in self.wind_physical_generator_ids
        ]
        self.wind_block_pairs = [
            (i, b)
            for (i, b) in self.generator_block_pairs
            if i in self.wind_physical_generator_ids
        ]
        self.conventional_block_pairs = [
            (i, b)
            for (i, b) in self.generator_block_pairs
            if i in self.conventional_physical_generator_ids
        ]

    @staticmethod
    def _is_wind_name(name: str) -> bool:
        stripped = str(name).strip()
        return stripped.upper().startswith("W") or "wind" in stripped.lower()

    def _infer_num_time_steps(self) -> int:
        if "time_steps" in self.scenarios_df.columns:
            return int(self.scenarios_df["time_steps"].iloc[0])
        demand_col = self._demand_profile_column()
        value = self.scenarios_df[demand_col].iloc[0]
        return len(self._parse_profile(value, demand_col))

    def _demand_profile_column(self) -> str:
        for col in self.scenarios_df.columns:
            if "demand_profile" in str(col).lower():
                return str(col)
        raise ValueError("No demand profile column found in scenarios_df")

    @staticmethod
    def _parse_profile(value: Any, column_name: str) -> list[float]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception as exc:
                raise ValueError(f"Could not parse profile column '{column_name}': {exc}") from exc
        if not isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            raise ValueError(f"Column '{column_name}' must contain a profile")
        return [float(v) for v in value]

    @staticmethod
    def _as_profile(value: Any, horizon: int, name: str) -> list[float]:
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    value = parsed
            except Exception:
                pass
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            profile = [float(v) for v in value]
        else:
            profile = [float(value)] * horizon
        if len(profile) < horizon:
            raise ValueError(f"{name} must have at least {horizon} entries, got {len(profile)}")
        return profile[:horizon]

    def _normalize_p_init(
        self, p_init: Optional[list[float] | list[list[float]]]
    ) -> list[float]:
        if p_init is None:
            return [0.5 * cap for cap in self.static_physical_capacity]

        values: Any = p_init
        if values and isinstance(values[0], (list, tuple, np.ndarray, pd.Series)):
            values = values[0]
        values = [float(v) for v in values]
        if len(values) == self.num_physical_generators:
            return values
        if len(values) == self.num_blocks:
            return [
                sum(values[g] for g in self.physical_to_block_indices[i])
                for i in range(self.num_physical_generators)
            ]
        raise ValueError(
            f"p_init has {len(values)} values; expected {self.num_physical_generators} "
            f"physical-generator values or {self.num_blocks} block values"
        )

    def _available_block_capacity_from_scenarios_df(
        self, scenario_id: int, global_block_idx: int, time_id: int
    ) -> float:
        row = self.scenarios_df.iloc[int(scenario_id)]
        block_name = self.block_names[int(global_block_idx)]
        for suffix in ("_cap_profile", "_profile"):
            column = f"{block_name}{suffix}"
            if column in self.scenarios_df.columns:
                profile = self._parse_profile(row[column], column)
                return float(profile[int(time_id)])
        return float(row[f"{block_name}_cap"])

    def _demand_from_scenarios_df(self, scenario_id: int, time_id: int) -> float:
        col = self._demand_profile_column()
        profile = self._parse_profile(self.scenarios_df[col].iloc[int(scenario_id)], col)
        return float(profile[int(time_id)])

    def _per_generator_config_value(
        self, raw: Any, generator_idx: int, default: Any
    ) -> Any:
        if raw is None:
            return default
        if isinstance(raw, dict):
            name = self.physical_generator_names[int(generator_idx)]
            for key in (generator_idx, str(generator_idx), name, name.upper(), name.lower()):
                if key in raw:
                    return raw[key]
            return default
        if isinstance(raw, (list, tuple, np.ndarray, pd.Series)):
            return raw[int(generator_idx)]
        return raw

    def _wind_generator_config_value(
        self, cfg: dict[str, Any], field_name: str, generator_idx: int, default: Any
    ) -> Any:
        grouped = cfg.get("wind_generators")
        name = self.physical_generator_names[int(generator_idx)]
        if isinstance(grouped, dict):
            for key in (generator_idx, str(generator_idx), name, name.upper(), name.lower()):
                if key in grouped and isinstance(grouped[key], dict) and field_name in grouped[key]:
                    return grouped[key][field_name]
        legacy_key = {"reference": "wind_reference", "min": "wind_min", "max": "wind_max"}[
            field_name
        ]
        return self._per_generator_config_value(cfg.get(legacy_key), generator_idx, default)

    def _configure_support_set_parameters(self) -> None:
        cfg = self.support_set_config
        reference_demand = [
            self._demand_from_scenarios_df(0, t) for t in range(self.num_time_steps)
        ]
        self.support_demand_reference = self._as_profile(
            cfg.get("demand_reference", reference_demand),
            self.num_time_steps,
            "demand_reference",
        )
        demand_min_default = min(reference_demand)
        demand_max_default = max(reference_demand)
        self.support_demand_min = float(cfg.get("demand_min", 0.8 * demand_min_default))
        self.support_demand_max = float(cfg.get("demand_max", 1.2 * demand_max_default))
        if self.support_demand_min > self.support_demand_max:
            raise ValueError("support_set_config demand_min cannot exceed demand_max")
        demand_range = self.support_demand_max - self.support_demand_min
        self.support_demand_ramp = float(cfg.get("demand_ramp", demand_range))
        self.support_demand_budget = float(
            cfg.get("demand_budget", self.num_time_steps * demand_range)
        )

        self.support_wind_reference: dict[int, list[float]] = {}
        self.support_wind_min: dict[int, float] = {}
        self.support_wind_max: dict[int, float] = {}
        for i in self.wind_physical_generator_ids:
            default_reference = [
                sum(
                    self._available_block_capacity_from_scenarios_df(0, g, t)
                    for g in self.physical_to_block_indices[i]
                )
                for t in range(self.num_time_steps)
            ]
            self.support_wind_reference[i] = self._as_profile(
                self._wind_generator_config_value(cfg, "reference", i, default_reference),
                self.num_time_steps,
                f"wind_generators[{self.physical_generator_names[i]}].reference",
            )
            static_total = self.static_physical_capacity[i]
            self.support_wind_min[i] = float(
                self._wind_generator_config_value(cfg, "min", i, 0.0)
            )
            self.support_wind_max[i] = float(
                self._wind_generator_config_value(cfg, "max", i, static_total)
            )
            if self.support_wind_min[i] > self.support_wind_max[i]:
                raise ValueError(
                    f"support_set_config wind_min cannot exceed wind_max for generator {i}"
                )

        total_wind_range = sum(
            self.support_wind_max[i] - self.support_wind_min[i]
            for i in self.wind_physical_generator_ids
        )
        self.support_wind_ramp = float(cfg.get("wind_ramp", max(total_wind_range, 0.0)))
        self.support_wind_budget = float(
            cfg.get("wind_budget", self.num_time_steps * max(total_wind_range, 0.0))
        )

    @staticmethod
    def load_support_set_config(
        config_path: str = "models/PoA/support_set_config.yaml",
        config_name: Optional[str] = None,
    ) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Support-set config not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle) or {}
        if "support_sets" not in raw_config:
            return raw_config
        support_sets = raw_config.get("support_sets")
        if not isinstance(support_sets, dict) or not support_sets:
            raise ValueError("'support_sets' must be a non-empty mapping")
        selected_name = config_name or raw_config.get("default_support_set") or next(iter(support_sets))
        if selected_name not in support_sets:
            raise ValueError(
                f"Unknown support-set config '{selected_name}'. "
                f"Available: {', '.join(support_sets.keys())}"
            )
        return support_sets[selected_name] or {}

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        self.model = ConcreteModel()
        
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))

        self.model.physical_generators = Set(initialize=range(self.num_physical_generators))
        self.model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)

        self.model.wind_physical_generators = Set(initialize=self.wind_physical_generator_ids)
        self.model.conventional_physical_generators = Set(initialize=self.conventional_physical_generator_ids)
        self.model.wind_blocks = Set(dimen=2, initialize=self.wind_block_pairs)
        self.model.conventional_blocks = Set(dimen=2, initialize=self.conventional_block_pairs)

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        self._build_PoA_variables()
        self._build_equilibrium_variables()
        self._build_complementarity_equilibrium_variables()
        self._build_optimal_variables()
        self._build_complementarity_optimal_variables()  

    def _build_PoA_variables(self) -> None:
        self.model.D = Var(self.model.time_steps, domain=NonNegativeReals)
        self.model.P_max_block = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.C_eq = Var(domain=Reals)
        self.model.C_opt = Var(domain=Reals)
        self.model.PoA = Var(domain=Reals)

        # Auxiliary variables for support set deviations and budgets
        self.model.D_abs_deviation = Var(self.model.time_steps, domain=NonNegativeReals)
        self.model.P_max_phys_abs_deviation = Var(self.model.wind_physical_generators, self.model.time_steps, domain=NonNegativeReals)

    def _build_equilibrium_variables(self) -> None:
        self.model.P_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.alpha = Var(self.model.generator_blocks, self.model.time_steps, domain=Reals)
        self.model.lambda_eq = Var(self.model.time_steps, domain=Reals, bounds=(-self.lambda_bound, self.lambda_bound))
        self.model.mu_upper_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals, bounds=(0.0, self.capacity_dual_bound))
        self.model.mu_lower_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals, bounds=(0.0, self.capacity_dual_bound))
        self.model.mu_ramp_up_eq = Var(self.model.physical_generators, self.model.time_steps_plus_1, domain=NonNegativeReals, bounds=(0.0, self.ramp_dual_bound))
        self.model.mu_ramp_down_eq = Var(self.model.physical_generators, self.model.time_steps_plus_1, domain=NonNegativeReals, bounds=(0.0, self.ramp_dual_bound))

    def _build_complementarity_equilibrium_variables(self) -> None:
        self.model.z_upper_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_lower_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_eq = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_eq = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        self.model.P_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.lambda_opt = Var(self.model.time_steps, domain=Reals, bounds=(-self.lambda_bound, self.lambda_bound))
        self.model.mu_upper_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals, bounds=(0.0, self.capacity_dual_bound))
        self.model.mu_lower_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals, bounds=(0.0, self.capacity_dual_bound))
        self.model.mu_ramp_up_opt = Var(self.model.physical_generators, self.model.time_steps_plus_1, domain=NonNegativeReals, bounds=(0.0, self.ramp_dual_bound))
        self.model.mu_ramp_down_opt = Var(self.model.physical_generators, self.model.time_steps_plus_1, domain=NonNegativeReals, bounds=(0.0, self.ramp_dual_bound))

    def _build_complementarity_optimal_variables(self) -> None:
        self.model.z_upper_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_lower_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_opt = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_opt = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)

    def _build_objective(self) -> None:
        self.model.objective = Objective(expr=self.model.PoA, sense=maximize)

    def _build_constraints(self) -> None:
        self._build_support_set()
        self._build_policy_constraints()
        self._build_lower_level_equilibrium_constraints()
        self._build_lower_level_optimal_constraints()
        self._build_KKT_stationarity_equilibrium_constraints()
        self._build_KKT_stationarity_optimal_constraints()
        self._build_KKT_complementarity_equilibrium_constraints()
        self._build_KKT_complementarity_optimal_constraints()
        self._build_PoA_constraints()

    # ------------------------------------------------------------------
    # Support set
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_support_set_demand(self) -> None:
        if self.support_demand_ramp < 0 or self.support_demand_budget < 0:
            raise ValueError("Demand ramp and budget must be non-negative")
        def demand_lower_rule(m, t):
            return m.D[t] >= self.support_demand_min

        def demand_upper_rule(m, t):
            return m.D[t] <= self.support_demand_max

        def demand_ramp_up_rule(m, t):
            return m.D[t] - m.D[t - 1] <= self.support_demand_ramp

        def demand_ramp_down_rule(m, t):
            return m.D[t - 1] - m.D[t] <= self.support_demand_ramp
        
        # Budget constraints 
        def demand_abs_deviation_pos_rule(m, t):
            return m.D_abs_deviation[t] >= m.D[t] - self.support_demand_reference[int(t)]

        def demand_abs_deviation_neg_rule(m, t):
            return m.D_abs_deviation[t] >= self.support_demand_reference[int(t)] - m.D[t]

        def demand_budget_rule(m):
            return sum(m.D_abs_deviation[t] for t in m.time_steps) <= self.support_demand_budget

        self.model.demand_lower_bound_constraints = Constraint(self.model.time_steps, rule=demand_lower_rule)
        self.model.demand_upper_bound_constraints = Constraint(self.model.time_steps, rule=demand_upper_rule)
        self.model.demand_ramp_up_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_up_rule)
        self.model.demand_ramp_down_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_down_rule)
        self.model.demand_abs_deviation_pos_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_pos_rule)
        self.model.demand_abs_deviation_neg_constraints = Constraint(self.model.time_steps, rule=demand_abs_deviation_neg_rule)
        self.model.demand_budget_constraint = Constraint(rule=demand_budget_rule)

    def _build_support_set_wind(self) -> None:
        if self.support_wind_ramp < 0 or self.support_wind_budget < 0:
            raise ValueError("Wind ramp and budget must be non-negative")

        def conventional_capacity_rule(m, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.P_max_block[i, b, t] == self.static_block_capacity[global_block]

        def wind_total_lower_rule(m, i, t):
            return (sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)]) >= self.support_wind_min[int(i)])

        def wind_total_upper_rule(m, i, t):
            return (sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)]) <= self.support_wind_max[int(i)])

        def wind_ramp_up_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)]) 
              - sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                <= self.support_wind_ramp)

        def wind_ramp_down_rule(m, i, t):
            return (
                sum(m.P_max_block[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= self.support_wind_ramp
            )

        def wind_abs_deviation_pos_rule(m, i, t):
            return (
                m.P_max_phys_abs_deviation[i, t]
                >= sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - self.support_wind_reference[int(i)][int(t)]
            )

        def wind_abs_deviation_neg_rule(m, i, t):
            return (
                m.P_max_phys_abs_deviation[i, t]
                >= self.support_wind_reference[int(i)][int(t)]
                - sum(m.P_max_block[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            )

        def wind_budget_rule(m):
            return sum(
                m.P_max_phys_abs_deviation[i, t]
                for i in m.wind_physical_generators
                for t in m.time_steps
            ) <= self.support_wind_budget

        self.model.conventional_capacity = Constraint(self.model.conventional_blocks, self.model.time_steps, rule=conventional_capacity_rule)
        self.model.wind_total_lower_bound = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_total_lower_rule)
        self.model.wind_total_upper_bound = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_total_upper_rule)
        self.model.wind_ramp_up = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ramp_up_rule)
        self.model.wind_ramp_down = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ramp_down_rule)
        self.model.wind_abs_deviation_pos = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_pos_rule)
        self.model.wind_abs_deviation_neg = Constraint(self.model.wind_physical_generators, self.model.time_steps, rule=wind_abs_deviation_neg_rule)
        self.model.wind_budget = Constraint(rule=wind_budget_rule)

    # ------------------------------------------------------------------
    # Lower level equilibrium and optimality constraints
    # ------------------------------------------------------------------

    def _build_lower_level_equilibrium_constraints(self) -> None:
        def power_balance_eq_rule(m, t):
            return m.D[t] - sum(m.P_eq[i, b, t] for (i, b) in m.generator_blocks) == 0

        def generation_upper_eq_rule(m, i, b, t):
            return m.P_eq[i, b, t] - m.P_max_block[i, b, t] <= 0

        def generation_lower_eq_rule(m, i, b, t):
            return m.P_eq[i, b, t] >= 0
        
        def ramp_up_eq_rule(m, i, t):
            return sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)]) - sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_up[int(i)] <= 0

        def ramp_up_initial_eq_rule(m, i):
            return sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) - self.p_init[int(i)] <= self.ramp_vector_up[int(i)]

        def ramp_down_eq_rule(m, i, t):
            return - sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)]) + sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_down[int(i)] <= 0
        
        def ramp_down_initial_eq_rule(m, i):
            return - sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) + self.p_init[int(i)] - self.ramp_vector_down[int(i)] <= 0

        self.model.power_balance_eq     = Constraint(self.model.time_steps, rule=power_balance_eq_rule)
        self.model.generation_upper_eq  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=generation_upper_eq_rule)
        self.model.generation_lower_eq  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=generation_lower_eq_rule)
        self.model.ramp_up_eq           = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_up_eq_rule)
        self.model.ramp_up_initial_eq   = Constraint(self.model.physical_generators, rule=ramp_up_initial_eq_rule)
        self.model.ramp_down_eq         = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_down_eq_rule)
        self.model.ramp_down_initial_eq = Constraint(self.model.physical_generators, rule=ramp_down_initial_eq_rule)

    def _build_lower_level_optimal_constraints(self) -> None:
        def power_balance_opt_rule(m, t):
            return m.D[t] - sum(m.P_opt[i, b, t] for (i, b) in m.generator_blocks) == 0

        def generation_upper_opt_rule(m, i, b, t):
            return m.P_opt[i, b, t] - m.P_max_block[i, b, t] <= 0

        def generation_lower_opt_rule(m, i, b, t):
            return m.P_opt[i, b, t] >= 0

        def ramp_up_opt_rule(m, i, t):
            return sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)]) - sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_up[int(i)] <= 0

        def ramp_up_initial_opt_rule(m, i):
            return sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) - self.p_init[int(i)] <= self.ramp_vector_up[int(i)]

        def ramp_down_opt_rule(m, i, t):
            return - sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)]) + sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_down[int(i)] <= 0
        
        def ramp_down_initial_opt_rule(m, i):
            return - sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) + self.p_init[int(i)] - self.ramp_vector_down[int(i)] <= 0

        self.model.power_balance_opt     = Constraint(self.model.time_steps, rule=power_balance_opt_rule)
        self.model.generation_upper_opt  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=generation_upper_opt_rule)
        self.model.generation_lower_opt  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=generation_lower_opt_rule)
        self.model.ramp_up_opt           = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_up_opt_rule)
        self.model.ramp_up_initial_opt   = Constraint(self.model.physical_generators, rule=ramp_up_initial_opt_rule)
        self.model.ramp_down_opt         = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_down_opt_rule)
        self.model.ramp_down_initial_opt = Constraint(self.model.physical_generators, rule=ramp_down_initial_opt_rule)

    # ------------------------------------------------------------------
    # KKT stationarity conditions
    # ------------------------------------------------------------------

    def _build_KKT_stationarity_equilibrium_constraints(self) -> None:
        def stationarity_eq_rule(m, i, b, t):
            return (
                m.alpha[i, b, t]
                - m.lambda_eq[t]
                + m.mu_upper_eq[i, b, t]
                - m.mu_lower_eq[i, b, t]
                + m.mu_ramp_up_eq[i, t]
                - m.mu_ramp_up_eq[i, t + 1]
                - m.mu_ramp_down_eq[i, t]
                + m.mu_ramp_down_eq[i, t + 1]
                == 0
            )

        def final_ramp_up_dual_eq_rule(m, i):
            return m.mu_ramp_up_eq[i, self.num_time_steps] == 0

        def final_ramp_down_dual_eq_rule(m, i):
            return m.mu_ramp_down_eq[i, self.num_time_steps] == 0

        self.model.stationarity_eq = Constraint(self.model.generator_blocks, self.model.time_steps, rule=stationarity_eq_rule)
        self.model.final_ramp_up_dual_eq = Constraint(self.model.physical_generators, rule=final_ramp_up_dual_eq_rule)
        self.model.final_ramp_down_dual_eq = Constraint(self.model.physical_generators, rule=final_ramp_down_dual_eq_rule)

    def _build_KKT_stationarity_optimal_constraints(self) -> None:
        def stationarity_opt_rule(m, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return (
                self.block_cost_vector[global_block]
                - m.lambda_opt[t]
                + m.mu_upper_opt[i, b, t]
                - m.mu_lower_opt[i, b, t]
                + m.mu_ramp_up_opt[i, t]
                - m.mu_ramp_up_opt[i, t + 1]
                - m.mu_ramp_down_opt[i, t]
                + m.mu_ramp_down_opt[i, t + 1]
                == 0
            )


        def final_ramp_up_dual_opt_rule(m, i):
            return m.mu_ramp_up_opt[i, self.num_time_steps] == 0

        def final_ramp_down_dual_opt_rule(m, i):
            return m.mu_ramp_down_opt[i, self.num_time_steps] == 0

        self.model.stationarity_opt = Constraint(self.model.generator_blocks, self.model.time_steps, rule=stationarity_opt_rule)
        self.model.final_ramp_up_dual_opt = Constraint(self.model.physical_generators, rule=final_ramp_up_dual_opt_rule)
        self.model.final_ramp_down_dual_opt = Constraint(self.model.physical_generators, rule=final_ramp_down_dual_opt_rule)

    # ------------------------------------------------------------------
    # KKT complementarity conditions
    # ------------------------------------------------------------------

    def _build_KKT_complementarity_equilibrium_constraints(self) -> None:
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_eq_rule(m, i, b, t):
            return -BigM * (1 - m.z_upper_eq[i, b, t]) <= m.P_eq[i, b, t] - m.P_max_block[i, b, t] 

        def upper_bound_complementarity_dual_eq_rule(m, i, b, t):
            return m.mu_upper_eq[i, b, t] <= self.capacity_dual_bound * m.z_upper_eq[i, b, t]

        def lower_bound_complementarity_eq_rule(m, i, b, t):
            return -BigM * (1 - m.z_lower_eq[i, b, t]) <= -m.P_eq[i, b, t]

        def lower_bound_complementarity_dual_eq_rule(m, i, b, t):
            return m.mu_lower_eq[i, b, t] <= self.capacity_dual_bound * m.z_lower_eq[i, b, t]
        
        def ramp_up_complementarity_eq_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_up_eq[i, t]) <= sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)]) - sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_up[int(i)]

        def ramp_up_initial_complementarity_eq_rule(m, i):
            return -BigM * (1 - m.z_ramp_up_eq[i, 0]) <= sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) - self.p_init[int(i)] - self.ramp_vector_up[int(i)]
        
        def ramp_up_complementarity_dual_eq_rule(m, i, t):
            return m.mu_ramp_up_eq[i, t] <= self.ramp_dual_bound * m.z_ramp_up_eq[i, t]
        
        def ramp_down_complementarity_eq_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_down_eq[i, t]) <= - sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)]) + sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_down[int(i)]

        def ramp_down_initial_complementarity_eq_rule(m, i):
            return -BigM * (1 - m.z_ramp_down_eq[i, 0]) <= - sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) + self.p_init[int(i)] - self.ramp_vector_down[int(i)]

        def ramp_down_complementarity_dual_eq_rule(m, i, t):
            return m.mu_ramp_down_eq[i, t] <= self.ramp_dual_bound * m.z_ramp_down_eq[i, t]

        self.model.upper_bound_complementarity_eq       = Constraint(self.model.generator_blocks, self.model.time_steps, rule=upper_bound_complementarity_eq_rule)
        self.model.upper_bound_complementarity_dual_eq  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=upper_bound_complementarity_dual_eq_rule)

        self.model.lower_bound_complementarity_eq       = Constraint(self.model.generator_blocks, self.model.time_steps, rule=lower_bound_complementarity_eq_rule)
        self.model.lower_bound_complementarity_dual_eq  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=lower_bound_complementarity_dual_eq_rule)

        self.model.ramp_up_complementarity_eq           = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_up_complementarity_eq_rule)
        self.model.ramp_up_complementarity_dual_eq      = Constraint(self.model.physical_generators, self.model.time_steps, rule=ramp_up_complementarity_dual_eq_rule)
        self.model.ramp_up_initial_complementarity_eq   = Constraint(self.model.physical_generators, rule=ramp_up_initial_complementarity_eq_rule)

        self.model.ramp_down_complementarity_eq         = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_down_complementarity_eq_rule)
        self.model.ramp_down_complementarity_dual_eq    = Constraint(self.model.physical_generators, self.model.time_steps, rule=ramp_down_complementarity_dual_eq_rule)
        self.model.ramp_down_initial_complementarity_eq = Constraint(self.model.physical_generators, rule=ramp_down_initial_complementarity_eq_rule)

    def _build_KKT_complementarity_optimal_constraints(self) -> None:
        BigM = self.big_m_complementarity

        def upper_bound_complementarity_opt_rule(m, i, b, t):
            return -BigM * (1 - m.z_upper_opt[i, b, t]) <= m.P_opt[i, b, t] - m.P_max_block[i, b, t] 

        def upper_bound_complementarity_dual_opt_rule(m, i, b, t):
            return m.mu_upper_opt[i, b, t] <= self.capacity_dual_bound * m.z_upper_opt[i, b, t]

        def lower_bound_complementarity_opt_rule(m, i, b, t):
            return -BigM * (1 - m.z_lower_opt[i, b, t]) <= -m.P_opt[i, b, t]

        def lower_bound_complementarity_dual_opt_rule(m, i, b, t):
            return m.mu_lower_opt[i, b, t] <= self.capacity_dual_bound * m.z_lower_opt[i, b, t]
        
        def ramp_up_complementarity_opt_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_up_opt[i, t]) <= sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)]) - sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_up[int(i)]

        def ramp_up_initial_complementarity_opt_rule(m, i):
            return -BigM * (1 - m.z_ramp_up_opt[i, 0]) <= sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) - self.p_init[int(i)] - self.ramp_vector_up[int(i)]
        
        def ramp_up_complementarity_dual_opt_rule(m, i, t):
            return m.mu_ramp_up_opt[i, t] <= self.ramp_dual_bound * m.z_ramp_up_opt[i, t]
        
        def ramp_down_complementarity_opt_rule(m, i, t):
            return -BigM * (1 - m.z_ramp_down_opt[i, t]) <= - sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)]) + sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)]) - self.ramp_vector_down[int(i)]

        def ramp_down_initial_complementarity_opt_rule(m, i):
            return -BigM * (1 - m.z_ramp_down_opt[i, 0]) <= - sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)]) + self.p_init[int(i)] - self.ramp_vector_down[int(i)]

        def ramp_down_complementarity_dual_opt_rule(m, i, t):
            return m.mu_ramp_down_opt[i, t] <= self.ramp_dual_bound * m.z_ramp_down_opt[i, t]

        self.model.upper_bound_complementarity_opt       = Constraint(self.model.generator_blocks, self.model.time_steps, rule=upper_bound_complementarity_opt_rule)
        self.model.upper_bound_complementarity_dual_opt  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=upper_bound_complementarity_dual_opt_rule)

        self.model.lower_bound_complementarity_opt       = Constraint(self.model.generator_blocks, self.model.time_steps, rule=lower_bound_complementarity_opt_rule)
        self.model.lower_bound_complementarity_dual_opt  = Constraint(self.model.generator_blocks, self.model.time_steps, rule=lower_bound_complementarity_dual_opt_rule)

        self.model.ramp_up_complementarity_opt           = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_up_complementarity_opt_rule)
        self.model.ramp_up_complementarity_dual_opt      = Constraint(self.model.physical_generators, self.model.time_steps, rule=ramp_up_complementarity_dual_opt_rule)
        self.model.ramp_up_initial_complementarity_opt   = Constraint(self.model.physical_generators, rule=ramp_up_initial_complementarity_opt_rule)

        self.model.ramp_down_complementarity_opt         = Constraint(self.model.physical_generators, self.model.time_steps_minus_1, rule=ramp_down_complementarity_opt_rule)
        self.model.ramp_down_complementarity_dual_opt    = Constraint(self.model.physical_generators, self.model.time_steps, rule=ramp_down_complementarity_dual_opt_rule)
        self.model.ramp_down_initial_complementarity_opt = Constraint(self.model.physical_generators, rule=ramp_down_initial_complementarity_opt_rule)

    # ------------------------------------------------------------------
    # PoA constraints
    # ------------------------------------------------------------------

    def _build_PoA_constraints(self) -> None:
        def cost_eq_rule(m):
            return m.C_eq == sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * m.P_eq[i, b, t]
                for (i, b) in m.generator_blocks
                for t in m.time_steps
            )

        def cost_opt_rule(m):
            return m.C_opt == sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * m.P_opt[i, b, t]
                for (i, b) in m.generator_blocks
                for t in m.time_steps
            )
        
        def PoA_rule(m):
            return m.C_eq - m.C_opt == m.PoA 

        self.model.cost_definition_eq = Constraint(rule=cost_eq_rule)
        self.model.cost_definition_opt = Constraint(rule=cost_opt_rule)
        self.model.poa_definition = Constraint(rule=PoA_rule)

    # ------------------------------------------------------------------
    # Neural-network policy embedding
    # ------------------------------------------------------------------

    def _build_policy_constraints(self) -> None:
        if self.nn_model_dir is not None:
            self._build_nn_policy_constraints()
            return

        def true_cost_alpha_rule(m, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.alpha[i, b, t] == self.block_cost_vector[global_block]

        self.model.true_cost_alpha = Constraint(
            self.model.generator_blocks,
            self.model.time_steps,
            rule=true_cost_alpha_rule,
        )

    def _load_nn_policies(self) -> None:
        if self.nn_model_dir is None or not self.nn_model_dir.exists():
            raise FileNotFoundError(f"NN model directory not found: {self.nn_model_dir}")
        self.nn_policies = {}
        for generator_name in self.physical_generator_names:
            weights_path = self.nn_model_dir / f"{generator_name}_policy_weights.json"
            metadata_path = self.nn_model_dir / f"{generator_name}_policy_metadata.json"
            if not weights_path.exists():
                raise FileNotFoundError(f"Missing NN weights file: {weights_path}")
            with weights_path.open("r", encoding="utf-8") as file_handle:
                weights = json.load(file_handle)
            metadata = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as file_handle:
                    metadata = json.load(file_handle)

            feature_columns = list(weights.get("feature_columns") or metadata.get("feature_columns") or [])
            target_columns = list(weights.get("target_columns") or metadata.get("target_columns") or [])
            layers = list(weights.get("layers", []))
            if not feature_columns or not target_columns or not layers:
                raise ValueError(f"Invalid NN policy payload for {generator_name}")
            self._validate_nn_policy(generator_name, feature_columns, target_columns, layers)
            self.nn_policies[generator_name] = {
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "layers": layers,
                "metadata": metadata,
                "target_map": self._target_columns_to_local_blocks(generator_name, target_columns),
            }

    def _validate_nn_policy(
        self, generator_name: str, feature_columns: list[str], target_columns: list[str], layers: list[dict[str, Any]]
    ) -> None:
        expected_input = len(feature_columns)
        current_dim = expected_input
        previous_was_hidden_linear = False
        linear_count = 0
        for idx, layer in enumerate(layers):
            layer_type = str(layer.get("type", "")).lower()
            if layer_type == "linear":
                weight = np.asarray(layer.get("weight"), dtype=float)
                bias = np.asarray(layer.get("bias"), dtype=float)
                if weight.ndim != 2 or bias.ndim != 1:
                    raise ValueError(f"{generator_name}: linear layer {idx} has invalid dimensions")
                if weight.shape[1] != current_dim or weight.shape[0] != bias.shape[0]:
                    raise ValueError(f"{generator_name}: inconsistent dimensions in linear layer {idx}")
                current_dim = int(weight.shape[0])
                previous_was_hidden_linear = idx < len(layers) - 1
                linear_count += 1
            elif layer_type == "relu":
                if not previous_was_hidden_linear:
                    raise ValueError(f"{generator_name}: ReLU layer {idx} must follow a hidden linear layer")
                previous_was_hidden_linear = False
            else:
                raise ValueError(f"{generator_name}: unsupported layer type '{layer_type}'")
        if str(layers[-1].get("type", "")).lower() != "linear":
            raise ValueError(f"{generator_name}: final NN layer must be linear")
        if current_dim != len(target_columns):
            raise ValueError(
                f"{generator_name}: output dimension {current_dim} does not match "
                f"{len(target_columns)} target columns"
            )
        if linear_count < 1:
            raise ValueError(f"{generator_name}: NN must contain at least one linear layer")

    def _target_columns_to_local_blocks(
        self, generator_name: str, target_columns: list[str]
    ) -> dict[int, int]:
        generator_idx = self.physical_generator_names.index(generator_name)
        output_to_local_block: dict[int, int] = {}
        seen_local_blocks: set[int] = set()
        for output_idx, column in enumerate(target_columns):
            prefix = "target_bid_"
            if not column.startswith(prefix):
                raise ValueError(f"{generator_name}: target column must start with '{prefix}': {column}")
            block_name = column.removeprefix(prefix)
            if block_name not in self.block_names:
                raise ValueError(f"{generator_name}: unknown target block '{block_name}'")
            global_block = self.block_names.index(block_name)
            block_generator_idx, local_block = self.global_to_local_block[global_block]
            if block_generator_idx != generator_idx:
                raise ValueError(
                    f"{generator_name}: target block '{block_name}' belongs to "
                    f"{self.physical_generator_names[block_generator_idx]}"
                )
            output_to_local_block[output_idx] = local_block
            seen_local_blocks.add(local_block)
        expected = set(self.local_blocks_by_generator[generator_idx])
        if seen_local_blocks != expected:
            raise ValueError(
                f"{generator_name}: target columns must cover local blocks {sorted(expected)}, "
                f"got {sorted(seen_local_blocks)}"
            )
        return output_to_local_block

    def _load_nn_normalization_stats(self) -> None:
        if self.nn_normalization_stats_path is None:
            self.nn_stats = {}
            return
        if not self.nn_normalization_stats_path.exists():
            raise FileNotFoundError(
                f"NN normalization stats not found: {self.nn_normalization_stats_path}"
            )
        with self.nn_normalization_stats_path.open("r", encoding="utf-8") as file_handle:
            self.nn_stats = json.load(file_handle)

    def _raw_nn_feature_expression(self, feature_name: str, t: int, physical_generator_idx: int):
        m = self.model
        previous_t = self.num_time_steps - 1 if int(t) == 0 else int(t) - 1
        next_t = 0 if int(t) == self.num_time_steps - 1 else int(t) + 1

        total_wind_capacity = lambda time_idx: sum(
            m.P_max_block[i, b, time_idx] for (i, b) in self.wind_block_pairs
        )
        total_capacity = lambda time_idx: sum(
            m.P_max_block[i, b, time_idx] for (i, b) in self.generator_block_pairs
        )
        own_capacity = lambda time_idx: sum(
            m.P_max_block[physical_generator_idx, b, time_idx]
            for b in self.local_blocks_by_generator[physical_generator_idx]
        )

        if feature_name == "demand":
            return m.D[t]
        if feature_name == "total_wind_generation_capacity":
            return total_wind_capacity(t)
        if feature_name == "total_generation_capacity":
            return total_capacity(t)
        if feature_name == "residual_demand":
            return m.D[t] - total_wind_capacity(t)
        if feature_name == "previous_generation_capacity":
            return total_capacity(previous_t)
        if feature_name == "previous_demand":
            return m.D[previous_t]
        if feature_name == "next_generation_capacity":
            return total_capacity(next_t)
        if feature_name == "next_demand":
            return m.D[next_t]
        if feature_name == "own_generation_capacity":
            return own_capacity(t)
        if feature_name == "previous_own_generation_capacity":
            return own_capacity(previous_t)
        if feature_name == "next_own_generation_capacity":
            return own_capacity(next_t)
        if feature_name == "average_true_cost":
            costs = [
                self.block_cost_vector[self.local_to_global_block[(physical_generator_idx, b)]]
                for b in self.local_blocks_by_generator[physical_generator_idx]
            ]
            return float(np.mean(costs))
        if feature_name == "minimum_true_cost":
            return float(
                min(
                    self.block_cost_vector[self.local_to_global_block[(physical_generator_idx, b)]]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        if feature_name == "maximum_true_cost":
            return float(
                max(
                    self.block_cost_vector[self.local_to_global_block[(physical_generator_idx, b)]]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        raise ValueError(f"Unsupported NN feature name: {feature_name}")

    def _normalized_nn_feature_expression(
        self, generator_name: str, feature_name: str, t: int, physical_generator_idx: int
    ):
        raw = self._raw_nn_feature_expression(feature_name, t, physical_generator_idx)
        feature_min, feature_max = self._nn_feature_bounds(generator_name, feature_name)
        denominator = feature_max - feature_min
        if abs(denominator) <= self.normalization_epsilon:
            return 0.0
        return (raw - feature_min) / denominator

    def _nn_feature_bounds(self, generator_name: str, feature_name: str) -> tuple[float, float]:
        stats = self.nn_stats or {}
        if bool(stats.get("per_generator")):
            generator_stats = stats.get("stats", {}).get(generator_name, {})
            mins = generator_stats.get("feature_min", {})
            maxs = generator_stats.get("feature_max", {})
            if feature_name in mins and feature_name in maxs:
                return float(mins[feature_name]), float(maxs[feature_name])
        if "feature_min" in stats and "feature_max" in stats:
            mins = stats["feature_min"]
            maxs = stats["feature_max"]
            if isinstance(mins, dict) and feature_name in mins:
                return float(mins[feature_name]), float(maxs[feature_name])
        return 0.0, 1.0

    def _build_nn_policy_constraints(self) -> None:
        m = self.model
        nn_input_indices: list[tuple[int, int, int]] = []
        nn_z_indices: list[tuple[int, int, int, int]] = []
        nn_h_indices: list[tuple[int, int, int, int]] = []
        nn_output_indices: list[tuple[int, int, int]] = []

        linear_layer_dims: dict[tuple[int, int], int] = {}
        relu_after_linear_layers: set[tuple[int, int]] = set()
        output_dims: dict[int, int] = {}
        for i, generator_name in enumerate(self.physical_generator_names):
            policy = self.nn_policies[generator_name]
            for f_idx, _ in enumerate(policy["feature_columns"]):
                for t in range(self.num_time_steps):
                    nn_input_indices.append((i, t, f_idx))
            linear_idx = 0
            layers = policy["layers"]
            for layer_pos, layer in enumerate(layers):
                if str(layer.get("type", "")).lower() != "linear":
                    continue
                output_dim = len(layer["bias"])
                linear_layer_dims[(i, linear_idx)] = output_dim
                is_final_linear = layer_pos == len(layers) - 1
                if is_final_linear:
                    output_dims[i] = output_dim
                    for k in range(output_dim):
                        for t in range(self.num_time_steps):
                            nn_output_indices.append((i, t, k))
                else:
                    for node in range(output_dim):
                        for t in range(self.num_time_steps):
                            nn_z_indices.append((i, t, linear_idx, node))
                            nn_h_indices.append((i, t, linear_idx, node))
                    if (
                        layer_pos + 1 < len(layers)
                        and str(layers[layer_pos + 1].get("type", "")).lower() == "relu"
                    ):
                        relu_after_linear_layers.add((i, linear_idx))
                linear_idx += 1

        m.nn_input_index = Set(dimen=3, initialize=nn_input_indices)
        m.nn_z_index = Set(dimen=4, initialize=nn_z_indices)
        m.nn_h_index = Set(dimen=4, initialize=nn_h_indices)
        m.nn_output_index = Set(dimen=3, initialize=nn_output_indices)
        m.nn_delta_index = Set(dimen=4, initialize=nn_h_indices)
        m.nn_input = Var(m.nn_input_index, domain=Reals)
        m.nn_z = Var(m.nn_z_index, domain=Reals)
        m.nn_h = Var(m.nn_h_index, domain=NonNegativeReals)
        m.nn_delta = Var(m.nn_delta_index, domain=Binary)
        m.nn_output = Var(m.nn_output_index, domain=Reals)
        m.nn_constraints = ConstraintList()

        for i, generator_name in enumerate(self.physical_generator_names):
            policy = self.nn_policies[generator_name]
            feature_columns = policy["feature_columns"]
            for t in range(self.num_time_steps):
                for f_idx, feature_name in enumerate(feature_columns):
                    m.nn_constraints.add(
                        m.nn_input[i, t, f_idx]
                        == self._normalized_nn_feature_expression(generator_name, feature_name, t, i)
                    )

                previous_values = [m.nn_input[i, t, f] for f in range(len(feature_columns))]
                linear_idx = 0
                for layer_pos, layer in enumerate(policy["layers"]):
                    if str(layer.get("type", "")).lower() == "relu":
                        continue
                    weights = np.asarray(layer["weight"], dtype=float)
                    bias = np.asarray(layer["bias"], dtype=float)
                    is_final_linear = layer_pos == len(policy["layers"]) - 1
                    current_values = []
                    for node in range(weights.shape[0]):
                        expr = float(bias[node]) + sum(
                            float(weights[node, prev_idx]) * previous_values[prev_idx]
                            for prev_idx in range(weights.shape[1])
                        )
                        if is_final_linear:
                            m.nn_constraints.add(m.nn_output[i, t, node] == expr)
                            current_values.append(m.nn_output[i, t, node])
                        else:
                            m.nn_constraints.add(m.nn_z[i, t, linear_idx, node] == expr)
                            h = m.nn_h[i, t, linear_idx, node]
                            z = m.nn_z[i, t, linear_idx, node]
                            delta = m.nn_delta[i, t, linear_idx, node]
                            m.nn_constraints.add(h >= z)
                            m.nn_constraints.add(h >= 0)
                            m.nn_constraints.add(h <= z + self.big_m_relu * (1 - delta))
                            m.nn_constraints.add(h <= self.big_m_relu * delta)
                            current_values.append(h)
                    previous_values = current_values
                    linear_idx += 1

            for output_idx, local_block in policy["target_map"].items():
                for t in range(self.num_time_steps):
                    m.nn_constraints.add(
                        m.alpha[i, local_block, t] == m.nn_output[i, t, output_idx]
                    )

    # ------------------------------------------------------------------
    # Solve and results
    # ------------------------------------------------------------------

    def solve(self, time_limit: Optional[float] = None) -> Any:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        solver = SolverFactory("gurobi")
        solver.options["OutputFlag"] = 1
        solver.options["NumericFocus"] = 1
        solver.options["IntFeasTol"] = 1e-8
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        self.solver_results = solver.solve(self.model, tee=True)
        return self.solver_results

    def _safe_value(self, expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        if raw_value is None:
            return None
        return float(raw_value)

    def _profile_values(self, var: Any, *leading_indices: int) -> list[Optional[float]]:
        return [
            self._safe_value(var[(*leading_indices, t)] if leading_indices else var[t])
            for t in range(self.num_time_steps)
        ]

    def check_dual_bound_activity(self, tol: float = 1e-5) -> list[dict]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        active_bounds: list[dict] = []

        def add_if_active(component_name: str, var: Any, indices: tuple, bound: float) -> None:
            var_value = self._safe_value(var[indices])
            if var_value is None or var_value < bound - tol:
                return
            active_bounds.append(
                {
                    "component": component_name,
                    "indices": [int(idx) for idx in indices],
                    "value": var_value,
                    "bound": float(bound),
                    "relative_to_bound": var_value / bound if bound != 0 else None,
                }
            )

        capacity_components = (
            ("mu_upper_eq", m.mu_upper_eq),
            ("mu_lower_eq", m.mu_lower_eq),
            ("mu_upper_opt", m.mu_upper_opt),
            ("mu_lower_opt", m.mu_lower_opt),
        )
        for component_name, var in capacity_components:
            for i, b in m.generator_blocks:
                for t in m.time_steps:
                    add_if_active(
                        component_name,
                        var,
                        (i, b, t),
                        self.capacity_dual_bound,
                    )

        ramp_components = (
            ("mu_ramp_up_eq", m.mu_ramp_up_eq),
            ("mu_ramp_down_eq", m.mu_ramp_down_eq),
            ("mu_ramp_up_opt", m.mu_ramp_up_opt),
            ("mu_ramp_down_opt", m.mu_ramp_down_opt),
        )
        for component_name, var in ramp_components:
            for i in m.physical_generators:
                for t in m.time_steps:
                    add_if_active(
                        component_name,
                        var,
                        (i, t),
                        self.ramp_dual_bound,
                    )

        return active_bounds

    def extract_results(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        objective = {
            "PoA_difference": self._safe_value(m.PoA),
            "C_eq": self._safe_value(m.C_eq),
            "C_opt": self._safe_value(m.C_opt),
            "PoA_ratio": None,
        }
        if objective["C_eq"] is not None and objective["C_opt"] not in (None, 0.0):
            objective["PoA_ratio"] = objective["C_eq"] / objective["C_opt"]

        generators: dict[str, Any] = {}
        for i, generator_name in enumerate(self.physical_generator_names):
            block_results = []
            for b in self.local_blocks_by_generator[i]:
                global_block = self.local_to_global_block[(i, b)]
                block_results.append(
                    {
                        "local_block_index": int(b),
                        "global_block_index": int(global_block),
                        "block_name": self.block_names[global_block],
                        "capacity_profile": self._profile_values(m.P_max_block, i, b),
                        "alpha_profile": self._profile_values(m.alpha, i, b),
                        "equilibrium_dispatch": self._profile_values(m.P_eq, i, b),
                        "optimal_dispatch": self._profile_values(m.P_opt, i, b),
                        "true_cost": float(self.block_cost_vector[global_block]),
                    }
                )
            generators[generator_name] = {
                "physical_generator_index": int(i),
                "is_wind": i in self.wind_physical_generator_ids,
                "physical_capacity_profile": [
                    sum(
                        self._safe_value(m.P_max_block[i, b, t]) or 0.0
                        for b in self.local_blocks_by_generator[i]
                    )
                    for t in range(self.num_time_steps)
                ],
                "equilibrium_physical_dispatch": [
                    sum(
                        self._safe_value(m.P_eq[i, b, t]) or 0.0
                        for b in self.local_blocks_by_generator[i]
                    )
                    for t in range(self.num_time_steps)
                ],
                "optimal_physical_dispatch": [
                    sum(
                        self._safe_value(m.P_opt[i, b, t]) or 0.0
                        for b in self.local_blocks_by_generator[i]
                    )
                    for t in range(self.num_time_steps)
                ],
                "blocks": block_results,
            }

        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(self.solver_results.solver.termination_condition),
            }

        try:
            dual_bound_activity = self.check_dual_bound_activity()
        except Exception:
            dual_bound_activity = []

        return {
            "reference_case": self.reference_case,
            "num_time_steps": self.num_time_steps,
            "objective": objective,
            "demand_profile": self._profile_values(m.D),
            "block_names": list(self.block_names),
            "physical_generator_names": list(self.physical_generator_names),
            "block_to_physical": dict(self.block_to_physical),
            "physical_to_block_indices": {
                str(i): list(blocks)
                for i, blocks in enumerate(self.physical_to_block_indices)
            },
            "generators": generators,
            "equilibrium_price_profile": self._profile_values(m.lambda_eq),
            "optimal_price_profile": self._profile_values(m.lambda_opt),
            "support_set": {
                "demand": {
                    "reference": list(self.support_demand_reference),
                    "min": float(self.support_demand_min),
                    "max": float(self.support_demand_max),
                    "ramp": float(self.support_demand_ramp),
                    "budget": float(self.support_demand_budget),
                },
                "wind": {
                    self.physical_generator_names[i]: {
                        "reference": list(self.support_wind_reference[i]),
                        "min": float(self.support_wind_min[i]),
                        "max": float(self.support_wind_max[i]),
                    }
                    for i in self.wind_physical_generator_ids
                },
                "wind_ramp": float(self.support_wind_ramp),
                "wind_budget": float(self.support_wind_budget),
            },
            "solver": solver_summary,
            "policy_type": "neural_network" if self.nn_model_dir is not None else "true_cost_baseline",
            "dual_bound_activity": dual_bound_activity,
        }

    def save_results(self, output_path: str | Path) -> Path:
        results = self.extract_results()
        path = Path(output_path)
        if not path.suffix:
            path = path.with_suffix(".json")
        if path.suffix.lower() != ".json":
            raise ValueError("output_path must end with .json or have no suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2)
        return path


if __name__ == "__main__":
    case = "test_case_bidding_blocks"
    regime_set = "PoA_analysis"
    seed = 1

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    support_set_config = PoAOptimizationBiddingBlocks.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case_bidding_blocks_base",
    )

    optimizer = PoAOptimizationBiddingBlocks(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        p_init=None,
        num_time_steps=4,
        support_set_config=support_set_config,
        nn_model_dir="models/neural_network/training/trained_models",
        nn_normalization_stats_path="models/neural_network/features/generated/normalized/min_max_stats.json",
        big_m_complementarity=100.0,
        lambda_bound=75.0,
        capacity_dual_bound=200.0,
        ramp_dual_bound=5.0,
        big_m_relu=50,
        reference_case=case,
    )

    start = time.perf_counter()
    optimizer.build_model()
    optimizer.solve()
    dual_bound_activity = optimizer.check_dual_bound_activity()
    if dual_bound_activity:
        capacity_hits = sum(
            1
            for item in dual_bound_activity
            if item["component"] in {"mu_upper_eq", "mu_lower_eq", "mu_upper_opt", "mu_lower_opt"}
        )
        ramp_hits = len(dual_bound_activity) - capacity_hits
        print(f"Dual bound activity detected: {len(dual_bound_activity)} active upper-bound duals")
        print(f"  capacity dual hits: {capacity_hits}")
        print(f"  ramp dual hits: {ramp_hits}")
        if capacity_hits:
            print("  If many capacity duals hit 100, increase capacity_dual_bound to 150.")
        if ramp_hits:
            print("  If many ramp duals hit 5, increase ramp_dual_bound to 20.")
    optimizer.save_results("results/poa_optimization_bidding_blocks_results.json")
    end = time.perf_counter()
    print(f"Total time: {end - start}")
