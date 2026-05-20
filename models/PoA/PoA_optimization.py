from pyomo.environ import *
import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
import sys
from typing import Any, Optional

import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.helper import (
    available_block_capacity,
    ensure_profile,
    infer_num_time_steps,
    scenario_demand,
    target_columns_to_local_blocks,
    wind_generator_config_value,
)
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel

class PoAOptimization:
    """
    Block-aware Price of Anarchy optimization.

    Dispatch and bids are indexed by physical generator and local bidding block:
    P_eq[i, b, t], P_opt[i, b, t], alpha[i, b, t].
    Physical ramp constraints sum over each generator's local bidding blocks.
    """

    normalization_epsilon = 1e-12
    default_lambda_bound = 40.0
    default_capacity_dual_bound = 40.02
    default_ramp_dual_bound = 20.0

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
        nn_policy_generators: Optional[list[int | str]] = None,
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
        self.requested_nn_policy_generators = nn_policy_generators
        self.lambda_bound = float(self.default_lambda_bound)
        self.capacity_dual_bound = float(self.default_capacity_dual_bound)
        self.ramp_dual_bound = float(self.default_ramp_dual_bound)

        self.nn_relu_bounds_report: dict[str, Any] = {}
        self.nn_relu_bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
        self.nn_feature_bounds: dict[str, Any] = {}
        self.nn_bound_warnings: list[str] = []
        self.tight_big_m: dict[str, dict[str, Any]] = {}
        self.aggregate_dual_bounds: dict[str, Any] = {}
        self.lambda_bounds: dict[str, Any] = {}
        self.reference_case = reference_case

        self._initialize_block_structure_from_ed()
        self.num_time_steps = int(num_time_steps or infer_num_time_steps(self.scenarios_df))
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
        self._configure_nn_policy_generators()

        self.nn_policies: dict[str, dict[str, Any]] = {}
        self.nn_stats: dict[str, Any] = {}
        if self.nn_model_dir is not None and self.nn_policy_generator_ids:
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

    def _configure_support_set_parameters(self) -> None:
        cfg = self.support_set_config
        reference_demand = [
            scenario_demand(self.scenarios_df, 0, t) for t in range(self.num_time_steps)
        ]
        self.support_demand_reference = ensure_profile(
            cfg.get("demand_reference", reference_demand),
            self.num_time_steps,
            "demand_reference",
            allow_truncate=True,
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
                    available_block_capacity(
                        self.scenarios_df,
                        self.block_names[int(g)],
                        0,
                        t,
                    )
                    for g in self.physical_to_block_indices[i]
                )
                for t in range(self.num_time_steps)
            ]
            self.support_wind_reference[i] = ensure_profile(
                wind_generator_config_value(
                    cfg,
                    "reference",
                    i,
                    self.physical_generator_names,
                    default_reference,
                ),
                self.num_time_steps,
                f"wind_generators[{self.physical_generator_names[i]}].reference",
                allow_truncate=True,
            )
            static_total = self.static_physical_capacity[i]
            self.support_wind_min[i] = float(
                wind_generator_config_value(
                    cfg,
                    "min",
                    i,
                    self.physical_generator_names,
                    0.0,
                )
            )
            self.support_wind_max[i] = float(
                wind_generator_config_value(
                    cfg,
                    "max",
                    i,
                    self.physical_generator_names,
                    static_total,
                )
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

    def _configure_nn_policy_generators(self) -> None:
        if self.nn_model_dir is None:
            self.nn_policy_generator_ids = []
            self.nn_policy_generator_names = []
            return

        requested = self.requested_nn_policy_generators
        if requested is None:
            ids = list(range(self.num_physical_generators))
        else:
            ids = []
            for raw_generator in requested:
                if isinstance(raw_generator, str):
                    generator_name = raw_generator.strip()
                    if generator_name not in self.physical_generator_names:
                        raise ValueError(
                            f"Unknown NN policy generator '{raw_generator}'. "
                            f"Available: {self.physical_generator_names}"
                        )
                    generator_idx = self.physical_generator_names.index(generator_name)
                else:
                    generator_idx = int(raw_generator)
                    if not 0 <= generator_idx < self.num_physical_generators:
                        raise ValueError(
                            f"NN policy generator index {generator_idx} is outside "
                            f"0..{self.num_physical_generators - 1}"
                        )
                if generator_idx not in ids:
                    ids.append(generator_idx)

        self.nn_policy_generator_ids = ids
        self.nn_policy_generator_names = [
            self.physical_generator_names[i] for i in self.nn_policy_generator_ids
        ]

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

    @staticmethod
    def _optional_float_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("bound", "value", "tight_bound", "tight_big_m"):
                if value_key in payload:
                    return PoAOptimization._optional_float_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    def _tight_lambda_bound(
        self,
        lambda_name: Optional[str],
        time_idx: int,
        bound_name: str,
        default_bound: float,
    ) -> float:
        if lambda_name is None:
            return float(default_bound)

        report = getattr(self, "tightening_report", {}) or {}
        source_payloads: list[Any] = [getattr(self, "lambda_bounds", {}) or {}]
        if isinstance(report, dict):
            source_payloads.append(report.get("lambda_bounds", {}) or {})

        for payload in source_payloads:
            if not isinstance(payload, dict):
                continue
            entries = payload.get(lambda_name, {}) or {}
            if not isinstance(entries, dict):
                continue
            details = entries.get(str(int(time_idx)), entries.get(int(time_idx)))
            if not isinstance(details, dict):
                continue
            tight_value = self._optional_float_bound(details.get(bound_name))
            if tight_value is not None:
                if bound_name == "lower":
                    return max(float(default_bound), tight_value)
                return min(float(default_bound), tight_value)

        return float(default_bound)

    def _lambda_lower_bound(self, time_idx: int, lambda_name: Optional[str] = None) -> float:
        return self._tight_lambda_bound(
            lambda_name,
            int(time_idx),
            "lower",
            -self.lambda_bound,
        )

    def _lambda_upper_bound(self, time_idx: int, lambda_name: Optional[str] = None) -> float:
        return self._tight_lambda_bound(
            lambda_name,
            int(time_idx),
            "upper",
            self.lambda_bound,
        )

    def _tight_dual_upper_bound(
        self,
        dual_name: Optional[str],
        index: tuple[int, ...],
        default_bound: float,
    ) -> float:
        """
        Return a certified dual bound from the tightening report when present.

        The constants below remain necessary construction fallbacks: lambda is
        not tightened by the current preprocessing pipeline, and any missing
        dual Big-M entry must fall back to a valid global bound. When a
        `tight_big_m` entry exists, use it both as the variable upper bound and
        as the coefficient in `mu <= M z`; using it only as a variable bound
        would leave a weaker default Big-M coefficient in the complementarity
        relaxation.
        """
        if dual_name is None:
            return float(default_bound)

        tight_big_m = getattr(self, "tight_big_m", {}) or {}
        entries = tight_big_m.get(dual_name, {}) or {}
        details = entries.get(self._json_key(index))
        if not details:
            return float(default_bound)

        tight_value = details.get("tight_big_m")
        if tight_value is None:
            return float(default_bound)
        return max(0.0, min(float(default_bound), float(tight_value)))

    @staticmethod
    def _optional_numeric_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("tight_big_m", "upper_bound", "ub", "bound", "value"):
                if value_key in payload:
                    return PoAOptimization._optional_numeric_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return max(0.0, numeric_value)

    def _lookup_optional_time_bound(
        self,
        payload: Any,
        side: str,
        time_idx: int,
    ) -> Optional[float]:
        numeric_value = self._optional_numeric_bound(payload)
        if numeric_value is not None:
            return numeric_value

        if isinstance(payload, (list, tuple)):
            if 0 <= int(time_idx) < len(payload):
                return self._optional_numeric_bound(payload[int(time_idx)])
            return None

        if not isinstance(payload, dict):
            return None

        side_candidates = tuple(dict.fromkeys((side, str(side), side.lower(), side.upper())))
        time_candidates = tuple(dict.fromkeys((int(time_idx), str(int(time_idx)))))
        composite_candidates = tuple(
            dict.fromkeys(
                (
                    f"{side},{int(time_idx)}",
                    f"{side}:{int(time_idx)}",
                    f"{side}_{int(time_idx)}",
                    f"{side}-{int(time_idx)}",
                    f"{side.upper()},{int(time_idx)}",
                    f"{side.upper()}:{int(time_idx)}",
                )
            )
        )

        for key in side_candidates:
            if key in payload:
                value = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if value is not None:
                    return value

        for key in time_candidates:
            if key in payload:
                value = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if value is not None:
                    return value

        for key in composite_candidates:
            if key in payload:
                value = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if value is not None:
                    return value

        return None

    @staticmethod
    def _aggregate_dual_bound_key_candidates(
        generic_key: str,
        side: str,
        dual_name: str,
    ) -> tuple[str, ...]:
        root = (
            generic_key[: -len("_sum_ub")]
            if generic_key.endswith("_sum_ub")
            else generic_key
        )
        dual_root = dual_name
        for suffix in ("_eq", "_opt"):
            if dual_root.endswith(suffix):
                dual_root = dual_root[: -len(suffix)]

        aliases = {
            "mu_max_sum_ub": ("mu_upper_sum_ub", "mu_upper_bound_sum_ub"),
            "mu_min_sum_ub": ("mu_lower_sum_ub", "mu_lower_bound_sum_ub"),
            "mu_ramp_up_sum_ub": ("rho_up_sum_ub",),
            "mu_ramp_down_sum_ub": ("rho_down_sum_ub",),
        }
        candidates = (
            f"{dual_name}_sum_ub",
            f"{dual_root}_{side}_sum_ub",
            f"{root}_{side}_sum_ub",
            f"{dual_root}_sum_ub",
            generic_key,
            *aliases.get(generic_key, ()),
        )
        return tuple(dict.fromkeys(candidates))

    def _aggregate_dual_sum_upper_bound(
        self,
        generic_key: str,
        side: str,
        time_idx: int,
        dual_name: str,
    ) -> Optional[float]:
        """
        Return an optional aggregate dual sum bound for one KKT side and time.

        The tightening report is intentionally permissive about where these
        values live so older reports remain valid and newer reports can store
        them either in a dedicated `aggregate_dual_bounds` block or alongside
        other dual Big-M data.
        """
        report = getattr(self, "tightening_report", {}) or {}
        source_payloads: list[Any] = [
            getattr(self, "aggregate_dual_bounds", {}) or {},
        ]
        if isinstance(report, dict):
            source_payloads.extend(
                [
                    report.get("aggregate_dual_bounds", {}) or {},
                    report,
                ]
            )
        source_payloads.append(getattr(self, "tight_big_m", {}) or {})

        support_dual_bounds = self.support_set_config.get("dual_bounds", {})
        source_payloads.extend(
            [
                self.support_set_config.get("aggregate_dual_bounds", {}),
                support_dual_bounds.get("aggregate_dual_bounds", {})
                if isinstance(support_dual_bounds, dict)
                else {},
                support_dual_bounds,
                self.support_set_config,
            ]
        )

        for payload in source_payloads:
            if not isinstance(payload, dict):
                continue
            for key in self._aggregate_dual_bound_key_candidates(
                generic_key,
                side,
                dual_name,
            ):
                if key not in payload:
                    continue
                bound = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if bound is not None:
                    return bound
        return None

    def _capacity_dual_upper_bound(
        self,
        bound_key: str,
        physical_generator_idx: int,
        local_block_idx: int,
        time_idx: int,
        dual_name: Optional[str] = None,
    ) -> float:
        return self._tight_dual_upper_bound(
            dual_name=dual_name,
            index=(int(physical_generator_idx), int(local_block_idx), int(time_idx)),
            default_bound=self.capacity_dual_bound,
        )

    def _ramp_dual_upper_bound(
        self,
        bound_key: str,
        physical_generator_idx: int,
        time_idx: int,
        dual_name: Optional[str] = None,
    ) -> float:
        return self._tight_dual_upper_bound(
            dual_name=dual_name,
            index=(int(physical_generator_idx), int(time_idx)),
            default_bound=self.ramp_dual_bound,
        )

    def _block_capacity_big_m(self, physical_generator_idx: int, local_block_idx: int) -> float:
        global_block = self.local_to_global_block[(physical_generator_idx, local_block_idx)]

        if physical_generator_idx in self.conventional_physical_generator_ids:
            return float(self.static_block_capacity[global_block])

        # For wind, if there is only one local block, use the support-set max directly.
        local_blocks = self.local_blocks_by_generator[physical_generator_idx]
        if len(local_blocks) == 1:
            return float(self.support_wind_max[physical_generator_idx])

        # If wind has multiple blocks, distribute support max according to static block shares.
        static_total = sum(
            self.static_block_capacity[self.local_to_global_block[(physical_generator_idx, b)]]
            for b in local_blocks
        )
        if static_total <= 0:
            return 0.0

        block_share = self.static_block_capacity[global_block] / static_total
        return float(block_share * self.support_wind_max[physical_generator_idx])

    def _physical_capacity_big_m(self, physical_generator_idx: int) -> float:
        if physical_generator_idx in self.wind_physical_generator_ids:
            return float(self.support_wind_max[physical_generator_idx])
        return float(self.static_physical_capacity[physical_generator_idx])

    def _ramp_up_big_m(self, physical_generator_idx: int) -> float:
        return float(
            self.ramp_vector_up[physical_generator_idx]
            + self._physical_capacity_big_m(physical_generator_idx)
        )

    def _ramp_down_big_m(self, physical_generator_idx: int) -> float:
        return float(
            self.ramp_vector_down[physical_generator_idx]
            + self._physical_capacity_big_m(physical_generator_idx)
        )

    def _ramp_up_initial_big_m(self, physical_generator_idx: int) -> float:
        return float(
            self.p_init[physical_generator_idx]
            + self.ramp_vector_up[physical_generator_idx]
        )

    def _ramp_down_initial_big_m(self, physical_generator_idx: int) -> float:
        return float(
            max(
                0.0,
                self._physical_capacity_big_m(physical_generator_idx)
                - self.p_init[physical_generator_idx]
                + self.ramp_vector_down[physical_generator_idx],
            )
        )

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
        self.model.lambda_eq = Var(
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, t: (
                self._lambda_lower_bound(int(t), "lambda_eq"),
                self._lambda_upper_bound(int(t), "lambda_eq"),
            ),
        )
        self.model.mu_upper_eq = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self._capacity_dual_upper_bound(
                    "mu_max_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_upper_eq",
                ),
            ),
        )
        self.model.mu_lower_eq = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self._capacity_dual_upper_bound(
                    "mu_min_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_lower_eq",
                ),
            ),
        )
        self.model.mu_ramp_up_eq = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self._ramp_dual_upper_bound(
                    "rho_up_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_up_eq",
                )
                if int(t) < self.num_time_steps
                else self.ramp_dual_bound,
            ),
        )
        self.model.mu_ramp_down_eq = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self._ramp_dual_upper_bound(
                    "rho_down_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_down_eq",
                )
                if int(t) < self.num_time_steps
                else self.ramp_dual_bound,
            ),
        )

    def _build_complementarity_equilibrium_variables(self) -> None:
        self.model.z_upper_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_lower_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_eq = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_eq = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        self.model.P_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.lambda_opt = Var(
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, t: (
                self._lambda_lower_bound(int(t), "lambda_opt"),
                self._lambda_upper_bound(int(t), "lambda_opt"),
            ),
        )
        self.model.mu_upper_opt = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self._capacity_dual_upper_bound(
                    "mu_max_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_upper_opt",
                ),
            ),
        )
        self.model.mu_lower_opt = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self._capacity_dual_upper_bound(
                    "mu_min_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_lower_opt",
                ),
            ),
        )
        self.model.mu_ramp_up_opt = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self._ramp_dual_upper_bound(
                    "rho_up_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_up_opt",
                )
                if int(t) < self.num_time_steps
                else self.ramp_dual_bound,
            ),
        )
        self.model.mu_ramp_down_opt = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self._ramp_dual_upper_bound(
                    "rho_down_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_down_opt",
                )
                if int(t) < self.num_time_steps
                else self.ramp_dual_bound,
            ),
        )

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
        self._build_aggregate_dual_bound_constraints()

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
        # self.model.demand_ramp_up_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_up_rule)
        # self.model.demand_ramp_down_constraints = Constraint(self.model.time_steps_minus_1, rule=demand_ramp_down_rule)
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

        def wind_even_block_split_rule(m, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return (
                len(local_blocks) * m.P_max_block[i, b, t]
                == sum(m.P_max_block[i, other_b, t] for other_b in local_blocks)
            )

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
        self.model.wind_even_block_split = Constraint(self.model.wind_blocks, self.model.time_steps, rule=wind_even_block_split_rule)
        # self.model.wind_ramp_up = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ramp_up_rule)
        # self.model.wind_ramp_down = Constraint(self.model.wind_physical_generators, self.model.time_steps_minus_1, rule=wind_ramp_down_rule)
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
        def upper_bound_complementarity_eq_rule(m, i, b, t):
            M_cap = self._block_capacity_big_m(int(i), int(b))
            return -M_cap * (1 - m.z_upper_eq[i, b, t]) <= m.P_eq[i, b, t] - m.P_max_block[i, b, t]

        def upper_bound_complementarity_dual_eq_rule(m, i, b, t):
            return m.mu_upper_eq[i, b, t] <= (
                self._capacity_dual_upper_bound(
                    "mu_max_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_upper_eq",
                )
                * m.z_upper_eq[i, b, t]
            )

        def lower_bound_complementarity_eq_rule(m, i, b, t):
            M_cap = self._block_capacity_big_m(int(i), int(b))
            return -M_cap * (1 - m.z_lower_eq[i, b, t]) <= -m.P_eq[i, b, t]

        def lower_bound_complementarity_dual_eq_rule(m, i, b, t):
            return m.mu_lower_eq[i, b, t] <= (
                self._capacity_dual_upper_bound(
                    "mu_min_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_lower_eq",
                )
                * m.z_lower_eq[i, b, t]
            )

        def ramp_up_complementarity_eq_rule(m, i, t):
            M_ramp_up = self._ramp_up_big_m(int(i))
            return -M_ramp_up * (1 - m.z_ramp_up_eq[i, t]) <= (
                sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_eq_rule(m, i):
            M_ramp_up_initial = self._ramp_up_initial_big_m(int(i))
            return -M_ramp_up_initial * (1 - m.z_ramp_up_eq[i, 0]) <= (
                sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_eq_rule(m, i, t):
            return m.mu_ramp_up_eq[i, t] <= (
                self._ramp_dual_upper_bound(
                    "rho_up_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_up_eq",
                )
                * m.z_ramp_up_eq[i, t]
            )
        
        def ramp_down_complementarity_eq_rule(m, i, t):
            M_ramp_down = self._ramp_down_big_m(int(i))
            return -M_ramp_down * (1 - m.z_ramp_down_eq[i, t]) <= (
                -sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_eq_rule(m, i):
            M_ramp_down_initial = self._ramp_down_initial_big_m(int(i))
            return -M_ramp_down_initial * (1 - m.z_ramp_down_eq[i, 0]) <= (
                -sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_eq_rule(m, i, t):
            return m.mu_ramp_down_eq[i, t] <= (
                self._ramp_dual_upper_bound(
                    "rho_down_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_down_eq",
                )
                * m.z_ramp_down_eq[i, t]
            )

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
        def upper_bound_complementarity_opt_rule(m, i, b, t):
            M_cap = self._block_capacity_big_m(int(i), int(b))
            return -M_cap * (1 - m.z_upper_opt[i, b, t]) <= m.P_opt[i, b, t] - m.P_max_block[i, b, t]
    
        def upper_bound_complementarity_dual_opt_rule(m, i, b, t):
            return m.mu_upper_opt[i, b, t] <= (
                self._capacity_dual_upper_bound(
                    "mu_max_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_upper_opt",
                )
                * m.z_upper_opt[i, b, t]
            )

        def lower_bound_complementarity_opt_rule(m, i, b, t):
            M_cap = self._block_capacity_big_m(int(i), int(b))
            return -M_cap * (1 - m.z_lower_opt[i, b, t]) <= -m.P_opt[i, b, t]

        def lower_bound_complementarity_dual_opt_rule(m, i, b, t):
            return m.mu_lower_opt[i, b, t] <= (
                self._capacity_dual_upper_bound(
                    "mu_min_ub",
                    int(i),
                    int(b),
                    int(t),
                    dual_name="mu_lower_opt",
                )
                * m.z_lower_opt[i, b, t]
            )
        
        def ramp_up_complementarity_opt_rule(m, i, t):
            M_ramp_up = self._ramp_up_big_m(int(i))
            return -M_ramp_up * (1 - m.z_ramp_up_opt[i, t]) <= (
                sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_opt_rule(m, i):
            M_ramp_up_initial = self._ramp_up_initial_big_m(int(i))
            return -M_ramp_up_initial * (1 - m.z_ramp_up_opt[i, 0]) <= (
                sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_opt_rule(m, i, t):
            return m.mu_ramp_up_opt[i, t] <= (
                self._ramp_dual_upper_bound(
                    "rho_up_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_up_opt",
                )
                * m.z_ramp_up_opt[i, t]
            )

        def ramp_down_complementarity_opt_rule(m, i, t):
            M_ramp_down = self._ramp_down_big_m(int(i))
            return -M_ramp_down * (1 - m.z_ramp_down_opt[i, t]) <= (
                -sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_opt_rule(m, i):
            M_ramp_down_initial = self._ramp_down_initial_big_m(int(i))
            return -M_ramp_down_initial * (1 - m.z_ramp_down_opt[i, 0]) <= (
                -sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_opt_rule(m, i, t):
            return m.mu_ramp_down_opt[i, t] <= (
                self._ramp_dual_upper_bound(
                    "rho_down_ub",
                    int(i),
                    int(t),
                    dual_name="mu_ramp_down_opt",
                )
                * m.z_ramp_down_opt[i, t]
            )

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
    # Aggregate dual-bound valid inequalities
    # ------------------------------------------------------------------

    aggregate_dual_bound_component_names = (
        "aggregate_mu_max_bound",
        "aggregate_mu_min_bound",
        "aggregate_mu_ramp_up_bound",
        "aggregate_mu_ramp_down_bound",
    )

    def _build_aggregate_dual_bound_constraints(self) -> int:
        """
        Add optional aggregate dual-bound inequalities.

        Componentwise dual Big-M values are safe but rectangular: each
        individual maximum may come from a different support-set trajectory. The
        aggregate inequalities tighten that artificial feasible space by
        limiting jointly attainable sums of dual variables without changing the
        KKT stationarity or complementarity equations themselves.
        """
        m = self.model
        if not hasattr(m, "kkt_sides"):
            m.kkt_sides = Set(initialize=("eq", "opt"))

        def dual_component(side: str, constraint_type: str) -> Any:
            return getattr(
                m,
                {
                    ("eq", "upper"): "mu_upper_eq",
                    ("eq", "lower"): "mu_lower_eq",
                    ("eq", "ramp_up"): "mu_ramp_up_eq",
                    ("eq", "ramp_down"): "mu_ramp_down_eq",
                    ("opt", "upper"): "mu_upper_opt",
                    ("opt", "lower"): "mu_lower_opt",
                    ("opt", "ramp_up"): "mu_ramp_up_opt",
                    ("opt", "ramp_down"): "mu_ramp_down_opt",
                }[(side, constraint_type)],
            )

        def dual_name(side: str, constraint_type: str) -> str:
            return {
                ("eq", "upper"): "mu_upper_eq",
                ("eq", "lower"): "mu_lower_eq",
                ("eq", "ramp_up"): "mu_ramp_up_eq",
                ("eq", "ramp_down"): "mu_ramp_down_eq",
                ("opt", "upper"): "mu_upper_opt",
                ("opt", "lower"): "mu_lower_opt",
                ("opt", "ramp_up"): "mu_ramp_up_opt",
                ("opt", "ramp_down"): "mu_ramp_down_opt",
            }[(side, constraint_type)]

        def aggregate_bound(side: str, constraint_type: str, t: int) -> Optional[float]:
            key = {
                "upper": "mu_max_sum_ub",
                "lower": "mu_min_sum_ub",
                "ramp_up": "mu_ramp_up_sum_ub",
                "ramp_down": "mu_ramp_down_sum_ub",
            }[constraint_type]
            return self._aggregate_dual_sum_upper_bound(
                key,
                side,
                int(t),
                dual_name(side, constraint_type),
            )

        def capacity_sum_rule(constraint_type: str):
            def rule(model, side, t):
                side = str(side)
                bound = aggregate_bound(side, constraint_type, int(t))
                if bound is None:
                    return Constraint.Skip
                mu = dual_component(side, constraint_type)
                return sum(mu[i, b, t] for (i, b) in model.generator_blocks) <= bound

            return rule

        def ramp_sum_rule(constraint_type: str):
            def rule(model, side, t):
                side = str(side)
                bound = aggregate_bound(side, constraint_type, int(t))
                if bound is None:
                    return Constraint.Skip
                mu = dual_component(side, constraint_type)
                return sum(mu[i, t] for i in model.physical_generators) <= bound

            return rule

        m.aggregate_mu_max_bound = Constraint(
            m.kkt_sides,
            m.time_steps,
            rule=capacity_sum_rule("upper"),
        )
        m.aggregate_mu_min_bound = Constraint(
            m.kkt_sides,
            m.time_steps,
            rule=capacity_sum_rule("lower"),
        )
        m.aggregate_mu_ramp_up_bound = Constraint(
            m.kkt_sides,
            m.time_steps,
            rule=ramp_sum_rule("ramp_up"),
        )
        m.aggregate_mu_ramp_down_bound = Constraint(
            m.kkt_sides,
            m.time_steps,
            rule=ramp_sum_rule("ramp_down"),
        )

        return sum(
            len(getattr(m, component_name))
            for component_name in self.aggregate_dual_bound_component_names
        )

    def _refresh_aggregate_dual_bound_constraints(self) -> int:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        for component_name in self.aggregate_dual_bound_component_names:
            if hasattr(self.model, component_name):
                self.model.del_component(component_name)
        return self._build_aggregate_dual_bound_constraints()

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------



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
        def true_cost_alpha_rule(m, i, b, t):
            if int(i) in self.nn_policy_generator_ids:
                return Constraint.Skip
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.alpha[i, b, t] == self.block_cost_vector[global_block]

        self.model.true_cost_alpha = Constraint(
            self.model.generator_blocks,
            self.model.time_steps,
            rule=true_cost_alpha_rule,
        )

        if self.nn_policy_generator_ids:
            self._build_nn_policy_constraints()

    def _load_nn_policies(self) -> None:
        if self.nn_model_dir is None or not self.nn_model_dir.exists():
            raise FileNotFoundError(f"NN model directory not found: {self.nn_model_dir}")
        self.nn_policies = {}
        for generator_name in self.nn_policy_generator_names:
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
                "target_map": target_columns_to_local_blocks(
                    generator_name=generator_name,
                    target_columns=target_columns,
                    block_names=self.block_names,
                    physical_generator_names=self.physical_generator_names,
                    global_to_local_block=self.global_to_local_block,
                    local_blocks_by_generator=self.local_blocks_by_generator,
                ),
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
        if not self.nn_relu_bounds:
            raise ValueError(
                "NN ReLU bounds are required before building NN policy constraints. "
                "Run compute_nn_relu_bounds.py or call load_nn_relu_bounds_report(...)."
            )
        nn_input_indices: list[tuple[int, int, int]] = []
        nn_z_indices: list[tuple[int, int, int, int]] = []
        nn_h_indices: list[tuple[int, int, int, int]] = []
        nn_output_indices: list[tuple[int, int, int]] = []

        linear_layer_dims: dict[tuple[int, int], int] = {}
        relu_after_linear_layers: set[tuple[int, int]] = set()
        output_dims: dict[int, int] = {}
        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            if generator_name not in self.nn_relu_bounds:
                raise ValueError(
                    f"NN ReLU bounds report is missing bounds for generator "
                    f"{generator_name}."
                )
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

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[i]
            relu_bounds = self.nn_relu_bounds[generator_name]
            for (t, linear_idx, node), bounds in relu_bounds.items():
                index = (i, int(t), int(linear_idx), int(node))
                if index not in m.nn_z:
                    continue
                m.nn_z[index].setlb(float(bounds["L"]))
                m.nn_z[index].setub(float(bounds["U"]))
                m.nn_h[index].setlb(float(bounds["h_lower"]))
                m.nn_h[index].setub(float(bounds["h_upper"]))

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            relu_bounds = self.nn_relu_bounds[generator_name]
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
                            bound_key = (int(t), int(linear_idx), int(node))
                            if bound_key not in relu_bounds:
                                raise ValueError(
                                    f"NN ReLU bounds report is missing bounds for "
                                    f"{generator_name} at time {t}, linear layer "
                                    f"{linear_idx}, node {node}."
                                )
                            bounds = relu_bounds[bound_key]
                            status = bounds["status"]
                            if status == "inactive":
                                m.nn_constraints.add(h == 0)
                                delta.fix(0)
                            elif status == "active":
                                m.nn_constraints.add(h == z)
                                delta.fix(1)
                            elif status == "ambiguous":
                                L = float(bounds["L"])
                                U = float(bounds["U"])
                                m.nn_constraints.add(h >= z)
                                m.nn_constraints.add(h >= 0)
                                m.nn_constraints.add(h <= z - L * (1 - delta))
                                m.nn_constraints.add(h <= U * delta)
                            else:
                                raise ValueError(
                                    f"{generator_name}: unknown ReLU bound status '{status}'"
                                )
                            current_values.append(h)
                    previous_values = current_values
                    linear_idx += 1

            for output_idx, local_block in policy["target_map"].items():
                for t in range(self.num_time_steps):
                    m.nn_constraints.add(
                        m.alpha[i, local_block, t] == m.nn_output[i, t, output_idx]
                    )

    # ------------------------------------------------------------------
    # Apply precomputed tightening reports
    # ------------------------------------------------------------------

    # The alpha-bound computation, slack-based binary fixing, and dual Big-M
    # maximization live in models/PoA/PoA_tightening/bidding_blocks_tightening.py.
    # This class only knows how to import their JSON output into the final PoA
    # model before solve().
    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in str(key).split(",") if part != "")

    @staticmethod
    def _parse_nn_relu_bound_key(key: str) -> tuple[int, int, int]:
        parts = tuple(int(part) for part in str(key).split(",") if part != "")
        if len(parts) != 3:
            raise ValueError(
                f"NN ReLU bound key '{key}' must have the form 'time_idx,linear_idx,node'"
            )
        return int(parts[0]), int(parts[1]), int(parts[2])

    def _set_nn_relu_bounds_from_report(self, report: dict[str, Any]) -> None:
        self.nn_relu_bounds_report = report
        self.nn_feature_bounds = report.get("nn_feature_bounds", {}) or {}
        self.nn_bound_warnings = list(
            report.get("warnings", report.get("nn_bound_warnings", [])) or []
        )

        parsed_bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
        for generator_name, entries in (report.get("nn_relu_bounds", {}) or {}).items():
            generator_bounds: dict[tuple[int, int, int], dict[str, Any]] = {}
            for key, details in (entries or {}).items():
                if not isinstance(details, dict):
                    raise ValueError(
                        f"Invalid NN ReLU bound entry for {generator_name}, key {key}"
                    )
                time_idx, linear_idx, node = self._parse_nn_relu_bound_key(key)
                parsed_details = dict(details)
                for numeric_key in ("L", "U", "h_lower", "h_upper"):
                    if numeric_key not in parsed_details:
                        raise ValueError(
                            f"NN ReLU bound entry {generator_name}[{key}] is missing "
                            f"'{numeric_key}'"
                        )
                    parsed_details[numeric_key] = float(parsed_details[numeric_key])
                parsed_details["status"] = str(parsed_details.get("status", "")).lower()
                if parsed_details["status"] not in {"active", "inactive", "ambiguous"}:
                    raise ValueError(
                        f"NN ReLU bound entry {generator_name}[{key}] has invalid "
                        f"status '{parsed_details['status']}'"
                    )
                parsed_details.setdefault("time_idx", time_idx)
                parsed_details.setdefault("linear_idx", linear_idx)
                parsed_details.setdefault("node", node)
                generator_bounds[(time_idx, linear_idx, node)] = parsed_details
            parsed_bounds[str(generator_name)] = generator_bounds

        self.nn_relu_bounds = parsed_bounds

    def load_nn_relu_bounds_report(
        self,
        report_path: str | Path = "results/poa_nn_relu_bounds_report.json",
    ) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"NN ReLU bounds report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)

        self.nn_relu_bounds_report_path = path
        self._set_nn_relu_bounds_from_report(report)
        return report

    def apply_nn_relu_bounds_to_model(
        self,
        report: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        if report is not None:
            self._set_nn_relu_bounds_from_report(report)
        if not self.nn_relu_bounds:
            raise ValueError(
                "No NN ReLU bounds loaded. Call load_nn_relu_bounds_report() first."
            )

        m = self.model
        stats = {
            "z_bounds_applied": 0,
            "h_bounds_applied": 0,
            "delta_fixed_active": 0,
            "delta_fixed_inactive": 0,
            "delta_left_ambiguous": 0,
        }

        for physical_generator_idx in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(physical_generator_idx)]
            for (time_idx, linear_idx, node), bounds in (
                self.nn_relu_bounds.get(generator_name, {}) or {}
            ).items():
                index = (
                    int(physical_generator_idx),
                    int(time_idx),
                    int(linear_idx),
                    int(node),
                )
                if hasattr(m, "nn_z") and index in m.nn_z:
                    m.nn_z[index].setlb(float(bounds["L"]))
                    m.nn_z[index].setub(float(bounds["U"]))
                    stats["z_bounds_applied"] += 1
                if hasattr(m, "nn_h") and index in m.nn_h:
                    m.nn_h[index].setlb(float(bounds["h_lower"]))
                    m.nn_h[index].setub(float(bounds["h_upper"]))
                    stats["h_bounds_applied"] += 1
                if not hasattr(m, "nn_delta") or index not in m.nn_delta:
                    continue

                status = str(bounds.get("status", "")).lower()
                if status == "inactive":
                    m.nn_delta[index].fix(0)
                    stats["delta_fixed_inactive"] += 1
                elif status == "active":
                    m.nn_delta[index].fix(1)
                    stats["delta_fixed_active"] += 1
                elif status == "ambiguous":
                    if m.nn_delta[index].fixed:
                        m.nn_delta[index].unfix()
                    stats["delta_left_ambiguous"] += 1
                else:
                    raise ValueError(
                        f"{generator_name}: unknown ReLU bound status '{status}'"
                    )

        self.applied_nn_relu_stats = stats
        return stats

    def summarize_nn_feature_bounds(self) -> dict[str, Any]:
        return dict(self.nn_feature_bounds or {})

    def summarize_nn_relu_bounds(self) -> dict[str, Any]:
        report_summary = (self.nn_relu_bounds_report or {}).get("summary")
        if isinstance(report_summary, dict):
            return report_summary
        if not self.nn_relu_bounds:
            return {}

        summary: dict[str, Any] = {}
        for generator_name, bounds in self.nn_relu_bounds.items():
            values = list(bounds.values())
            if not values:
                continue
            L_values = [float(item["L"]) for item in values]
            U_values = [float(item["U"]) for item in values]
            summary[generator_name] = {
                "num_hidden_neurons_time_indexed": len(values),
                "num_active": sum(1 for item in values if item["status"] == "active"),
                "num_inactive": sum(1 for item in values if item["status"] == "inactive"),
                "num_ambiguous": sum(
                    1 for item in values if item["status"] == "ambiguous"
                ),
                "min_L": float(min(L_values)),
                "max_L": float(max(L_values)),
                "min_U": float(min(U_values)),
                "max_U": float(max(U_values)),
                "max_M_minus": float(max(max(0.0, -L) for L in L_values)),
                "max_M_plus": float(max(max(0.0, U) for U in U_values)),
            }
        return summary

    def load_tightening_report(
        self,
        report_path: str | Path = "results/poa_bidding_blocks_tightening_report.json",
    ) -> dict[str, Any]:
        """
        Load a previously saved tightening report.

        The loaded data is kept in both JSON-friendly form and tuple-indexed
        form. The tuple-indexed alpha bounds can be reused by the OBBT helpers,
        while the JSON-friendly fixed-binary and Big-M dictionaries are applied
        directly to a built Pyomo model.
        """
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"Tightening report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)

        self.tightening_report_path = path
        self.tightening_report = report
        self.fixed_binaries = report.get("fixed_binaries", {}) or {}
        self.tight_big_m = report.get("tight_big_m", {}) or {}
        self.aggregate_dual_bounds = report.get("aggregate_dual_bounds", {}) or {}
        self.lambda_bounds = report.get("lambda_bounds", {}) or {}
        self.alpha_bound_optimization_results = (
            report.get("alpha_optimization_results", {}) or {}
        )
        self.alpha_bounds = {
            self._parse_json_index(key): {
                "lower": float(value["lower"]),
                "upper": float(value["upper"]),
            }
            for key, value in (report.get("alpha_bounds", {}) or {}).items()
        }
        return report

    def apply_tightened_bounds_to_model(
        self,
        report: Optional[dict[str, Any]] = None,
        apply_alpha_bounds: bool = True,
        apply_fixed_binaries: bool = True,
        apply_dual_bounds: bool = True,
    ) -> dict[str, int]:
        """
        Apply a tightening report to the already-built PoA model.

        Fixed binaries remove complementarity cases that slack OBBT proved can
        never bind. Tight dual upper bounds strengthen the `mu <= M z` side of
        complementarity because the dual variable itself now has a smaller upper
        bound. Alpha bounds are redundant when the exact NN is embedded, but they
        still help the solver by tightening variable domains.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        report = report or getattr(self, "tightening_report", None)
        if report is None:
            raise ValueError("No tightening report loaded. Call load_tightening_report() first.")
        self.tightening_report = report
        self.tight_big_m = report.get("tight_big_m", {}) or {}
        self.aggregate_dual_bounds = report.get("aggregate_dual_bounds", {}) or {}
        self.lambda_bounds = report.get("lambda_bounds", {}) or {}

        m = self.model
        stats = {
            "fixed_binaries": 0,
            "lambda_bounds": 0,
            "dual_upper_bounds": 0,
            "aggregate_dual_bounds": 0,
            "alpha_bounds": 0,
        }

        if apply_fixed_binaries:
            for var_name, entries in (report.get("fixed_binaries", {}) or {}).items():
                binary_var = getattr(m, var_name, None)
                if binary_var is None:
                    continue
                for key, details in entries.items():
                    index = self._parse_json_index(key)
                    if index not in binary_var:
                        continue
                    binary_var[index].fix(int(details.get("fixed_value", 0)))
                    stats["fixed_binaries"] += 1

        if apply_dual_bounds:
            stats["lambda_bounds"] = self._apply_lambda_bounds_to_model()

            for var_name, entries in (report.get("tight_big_m", {}) or {}).items():
                dual_var = getattr(m, var_name, None)
                if dual_var is None:
                    continue
                for key, details in entries.items():
                    tight_value = details.get("tight_big_m")
                    if tight_value is None:
                        continue
                    index = self._parse_json_index(key)
                    if index not in dual_var:
                        continue
                    current_ub = dual_var[index].ub
                    new_ub = max(0.0, float(tight_value))
                    if current_ub is not None:
                        new_ub = min(float(current_ub), new_ub)
                    dual_var[index].setub(new_ub)
                    stats["dual_upper_bounds"] += 1

            self.aggregate_dual_bounds = report.get("aggregate_dual_bounds", {}) or {}
            stats["aggregate_dual_bounds"] = (
                self._refresh_aggregate_dual_bound_constraints()
            )

        if apply_alpha_bounds:
            alpha_entries = report.get("alpha_bounds", {}) or {}
            for key, bounds in alpha_entries.items():
                index = self._parse_json_index(key)
                if index not in m.alpha:
                    continue
                lower = float(bounds["lower"])
                upper = float(bounds["upper"])
                current_lb = m.alpha[index].lb
                current_ub = m.alpha[index].ub
                if current_lb is not None:
                    lower = max(float(current_lb), lower)
                if current_ub is not None:
                    upper = min(float(current_ub), upper)
                if lower <= upper:
                    m.alpha[index].setlb(lower)
                    m.alpha[index].setub(upper)
                    stats["alpha_bounds"] += 1

        self.applied_tightening_stats = stats
        return stats

    def _apply_lambda_bounds_to_model(self) -> int:
        m = self.model
        applied = 0
        for lambda_name in ("lambda_eq", "lambda_opt"):
            lambda_var = getattr(m, lambda_name, None)
            if lambda_var is None:
                continue
            lambda_entries = (self.lambda_bounds or {}).get(lambda_name, {}) or {}
            for t in m.time_steps:
                details = (
                    lambda_entries.get(str(int(t)), lambda_entries.get(int(t)))
                    if isinstance(lambda_entries, dict)
                    else None
                )
                has_report_bound = isinstance(details, dict) and (
                    self._optional_float_bound(details.get("lower")) is not None
                    or self._optional_float_bound(details.get("upper")) is not None
                )
                lower = self._lambda_lower_bound(int(t), lambda_name)
                upper = self._lambda_upper_bound(int(t), lambda_name)
                current_lb = lambda_var[t].lb
                current_ub = lambda_var[t].ub
                if current_lb is not None:
                    lower = max(float(current_lb), lower)
                if current_ub is not None:
                    upper = min(float(current_ub), upper)
                if lower <= upper:
                    changed = (
                        current_lb is None
                        or current_ub is None
                        or not np.isclose(float(current_lb), lower)
                        or not np.isclose(float(current_ub), upper)
                    )
                    lambda_var[t].setlb(lower)
                    lambda_var[t].setub(upper)
                    if has_report_bound or changed:
                        applied += 1
        return applied

    # ------------------------------------------------------------------
    # Solve and results
    # ------------------------------------------------------------------

    def solve(self, time_limit: Optional[float] = None) -> Any:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        solver = SolverFactory("gurobi")
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
            ("mu_upper_eq", m.mu_upper_eq, "mu_max_ub"),
            ("mu_lower_eq", m.mu_lower_eq, "mu_min_ub"),
            ("mu_upper_opt", m.mu_upper_opt, "mu_max_ub"),
            ("mu_lower_opt", m.mu_lower_opt, "mu_min_ub"),
        )
        for component_name, var, bound_key in capacity_components:
            for i, b in m.generator_blocks:
                for t in m.time_steps:
                    add_if_active(
                        component_name,
                        var,
                        (i, b, t),
                        self._capacity_dual_upper_bound(
                            bound_key,
                            int(i),
                            int(b),
                            int(t),
                            dual_name=component_name,
                        ),
                    )

        ramp_components = (
            ("mu_ramp_up_eq", m.mu_ramp_up_eq, "rho_up_ub"),
            ("mu_ramp_down_eq", m.mu_ramp_down_eq, "rho_down_ub"),
            ("mu_ramp_up_opt", m.mu_ramp_up_opt, "rho_up_ub"),
            ("mu_ramp_down_opt", m.mu_ramp_down_opt, "rho_down_ub"),
        )
        for component_name, var, bound_key in ramp_components:
            for i in m.physical_generators:
                for t in m.time_steps:
                    add_if_active(
                        component_name,
                        var,
                        (i, t),
                        self._ramp_dual_upper_bound(
                            bound_key,
                            int(i),
                            int(t),
                            dual_name=component_name,
                        ),
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
            "policy_type": (
                "true_cost_baseline"
                if not self.nn_policy_generator_ids
                else (
                    "neural_network"
                    if len(self.nn_policy_generator_ids) == self.num_physical_generators
                    else "mixed_neural_network_true_cost"
                )
            ),
            "nn_policy_generators": list(self.nn_policy_generator_names),
            "true_cost_policy_generators": [
                generator_name
                for i, generator_name in enumerate(self.physical_generator_names)
                if i not in self.nn_policy_generator_ids
            ],
            "dual_bounds": {
                "lambda_bound": float(self.lambda_bound),
                "lambda_bounds": self.lambda_bounds,
                "capacity_dual_bound": float(self.capacity_dual_bound),
                "ramp_dual_bound": float(self.ramp_dual_bound),
            },
            "dual_bound_activity": dual_bound_activity,
            "nn_feature_bounds": self.summarize_nn_feature_bounds(),
            "nn_relu_bounds": self.summarize_nn_relu_bounds(),
            "nn_bound_warnings": list(self.nn_bound_warnings),
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
    horizon = 4
    nn_relu_report_path = "results/poa_nn_relu_bounds_report.json"
    tightening_report_path = "results/poa_bidding_blocks_tightening_report.json"

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    support_set_config = PoAOptimization.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case_bidding_blocks_base",
    )

    optimizer = PoAOptimization(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        p_init=None,
        num_time_steps=horizon,
        support_set_config=support_set_config,
        nn_model_dir="models/neural_network/training/trained_models",
        nn_normalization_stats_path=(
            "models/neural_network/features/generated/normalized/min_max_stats.json"
        ),
        # None means all generators with available NN files. Use [] for true
        # costs only, or e.g. ["G2", "W1"] / [1, 2] for a selected subset.
        nn_policy_generators=[1, 2],
        reference_case=case,
    )

    start = time.perf_counter()
    optimizer.load_nn_relu_bounds_report(nn_relu_report_path)
    optimizer.load_tightening_report(tightening_report_path)
    optimizer.build_model()
    print("\nNN ReLU bound summary after model build")
    print(json.dumps(optimizer.summarize_nn_relu_bounds(), indent=2))
    applied_nn_relu_stats = optimizer.apply_nn_relu_bounds_to_model()
    applied_stats = optimizer.apply_tightened_bounds_to_model()
    optimizer.solve(time_limit=400)
    result_path = optimizer.save_results(
        "results/poa_optimization_bidding_blocks_results_tightened.json"
    )
    elapsed = time.perf_counter() - start

    print("\nPoA solve with precomputed tightening complete")
    print(f"  NN ReLU bounds report: {nn_relu_report_path}")
    print(f"  Tightening report: {tightening_report_path}")
    print(f"  Active NN ReLU binaries fixed: {applied_nn_relu_stats['delta_fixed_active']}")
    print(f"  Inactive NN ReLU binaries fixed: {applied_nn_relu_stats['delta_fixed_inactive']}")
    print(f"  Ambiguous NN ReLU binaries left binary: {applied_nn_relu_stats['delta_left_ambiguous']}")
    print(f"  Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"  Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"  Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"  Results: {result_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
