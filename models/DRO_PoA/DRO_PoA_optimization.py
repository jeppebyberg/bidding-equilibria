from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from pyomo.environ import *

from config.scenarios.scenario_generator import ScenarioManager
from config.utils.cases_utils import load_setup_data, normalize_generators


class DRO_PoAOptimization:
    """
    Distributionally robust, block-aware Price of Anarchy optimization.

    The lower-level equilibrium and social optimum use the same indexing as
    models/PoA/PoA_optimization.py, with an additional empirical-scenario index:

        P_eq[k, i, b, t], P_opt[k, i, b, t], alpha[k, i, b, t]

    where k is the empirical scenario, i is the physical generator, and b is a
    generator-local bidding block. Physical ramping sums over local blocks.
    """

    normalization_epsilon = 1e-12

    def __init__(
        self,
        P_init: Optional[Sequence[float] | Sequence[Sequence[float]]] = None,
        num_time_steps: Optional[int] = None,
        reference_case: str = "test_case_bidding_blocks",
        support_set_config: Optional[dict[str, Any]] = None,
        eta: float = 0.0,
        empirical_scenario: Optional[Any] = None,
        big_m_complementarity: float = 1e8,
        feature_normalizer_stats_path: str | Path = "results/feature_normalizer_stats.json",
        policy_results_path: Optional[str | Path] = None,
        policy_data: Optional[dict[str, Any]] = None,
    ):
        self.requested_p_init = P_init
        self.requested_num_time_steps = num_time_steps
        self.reference_case = reference_case
        self.support_set_config = support_set_config or {}
        self.eta = float(eta)
        self.big_m_complementarity = float(big_m_complementarity)
        self.feature_normalizer_stats_path = Path(feature_normalizer_stats_path)
        self.policy_results_path = Path(policy_results_path) if policy_results_path is not None else None
        self.policy_data = policy_data

        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if self.big_m_complementarity <= 0:
            raise ValueError("big_m_complementarity must be positive")

        self.feature_names: list[str] = []
        self.feature_min = np.asarray([], dtype=float)
        self.feature_max = np.asarray([], dtype=float)
        self.private_feature_names: list[str] = []
        self.player_private_min_max: dict[int, dict[str, np.ndarray]] = {}
        self.policy_type: Optional[str] = None
        self.policy_by_generator: dict[int, Any] = {}
        self.alpha_bounds: dict[tuple[int, ...], dict[str, float]] = {}
        self.fixed_binaries: dict[str, dict[str, Any]] = {}
        self.tight_big_m: dict[str, dict[str, Any]] = {}

        self._initialize_block_structure_from_reference_case()
        self.num_time_steps = int(self.requested_num_time_steps or self.reference_time_steps)
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive")
        if self.num_time_steps > self.reference_time_steps:
            raise ValueError(
                f"num_time_steps={self.num_time_steps} exceeds reference-case horizon "
                f"{self.reference_time_steps}"
            )

        self.p_init = self._normalize_p_init(self.requested_p_init)
        self._load_feature_normalization_stats_if_available()
        self._configure_support_set_parameters()
        self._configure_empirical_scenarios(empirical_scenario)
        self._load_policy_if_requested()

    # ------------------------------------------------------------------
    # Data and configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _is_wind_name(name: str) -> bool:
        stripped = str(name).strip()
        return stripped.upper().startswith("W") or "wind" in stripped.lower()

    @staticmethod
    def _as_profile(
        value: Any,
        horizon: int,
        name: str,
        *,
        allow_truncate: bool = True,
    ) -> list[float]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                return [float(value)] * horizon

        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            profile = [float(v) for v in value]
        else:
            profile = [float(value)] * horizon

        if allow_truncate and len(profile) >= horizon:
            return profile[:horizon]
        if len(profile) != horizon:
            raise ValueError(f"{name} must have length {horizon}, got {len(profile)}")
        return profile

    @staticmethod
    def load_support_set_config(
        config_path: str | Path = "models/PoA/support_set_config.yaml",
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

    def _initialize_block_structure_from_reference_case(self) -> None:
        try:
            (
                num_generators,
                _pmax_list,
                _pmin_list,
                _cost_vector,
                ramp_up,
                ramp_down,
                demand,
                generators,
                players,
                time_steps,
            ) = load_setup_data(self.reference_case)
        except Exception as exc:
            raise ValueError(f"Failed to load reference case '{self.reference_case}': {exc}") from exc

        normalized = normalize_generators(generators)
        physical_generators = list(normalized["physical_generators"])
        blocks = list(normalized["blocks"])

        self.reference_time_steps = int(time_steps)
        self.num_physical_generators = int(num_generators)
        self.physical_generator_names = [str(gen["physical_name"]) for gen in physical_generators]
        self.generators = generators
        self.players_config = players
        self.reference_demand = float(demand)

        self.block_names = [str(block["block_name"]) for block in blocks]
        self.num_blocks = len(self.block_names)
        self.static_block_capacity = [float(block["pmax"]) for block in blocks]
        self.block_cost_vector = [float(block["cost"]) for block in blocks]
        self.static_physical_capacity = [float(gen["pmax"]) for gen in physical_generators]
        self.ramp_vector_up = [float(v) for v in ramp_up]
        self.ramp_vector_down = [float(v) for v in ramp_down]

        physical_idx_by_name = {
            name: idx for idx, name in enumerate(self.physical_generator_names)
        }
        self.block_to_physical = {
            str(block["block_name"]): str(block["physical_name"])
            for block in blocks
        }
        self.block_to_physical_idx = [
            physical_idx_by_name[str(block["physical_name"])]
            for block in blocks
        ]
        self.physical_to_block_indices: list[list[int]] = [
            [] for _ in range(self.num_physical_generators)
        ]
        for global_block, physical_idx in enumerate(self.block_to_physical_idx):
            self.physical_to_block_indices[physical_idx].append(global_block)

        self.blocks_by_generator = {
            i: list(blocks_for_generator)
            for i, blocks_for_generator in enumerate(self.physical_to_block_indices)
        }
        self.local_blocks_by_generator = {
            i: list(range(len(blocks_for_generator)))
            for i, blocks_for_generator in self.blocks_by_generator.items()
        }
        self.local_to_global_block = {
            (i, local_block): global_block
            for i, blocks_for_generator in self.blocks_by_generator.items()
            for local_block, global_block in enumerate(blocks_for_generator)
        }
        self.global_to_local_block = {
            global_block: local_pair
            for local_pair, global_block in self.local_to_global_block.items()
        }
        self.generator_block_pairs = list(self.local_to_global_block.keys())

        self.wind_physical_generator_ids = [
            i
            for i, generator in enumerate(physical_generators)
            if bool(generator.get("is_wind", False)) or self._is_wind_name(self.physical_generator_names[i])
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

    def _normalize_p_init(
        self,
        p_init: Optional[Sequence[float] | Sequence[Sequence[float]]],
    ) -> list[float]:
        if p_init is None:
            return [0.5 * cap for cap in self.static_physical_capacity]

        values: Any = p_init
        if isinstance(values, np.ndarray):
            values = values.tolist()
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
            f"P_init has {len(values)} values; expected {self.num_physical_generators} "
            f"physical-generator values or {self.num_blocks} block values"
        )

    def _generator_name(self, generator_idx: int) -> str:
        return self.physical_generator_names[int(generator_idx)]

    def _feature_bound(self, feature_name: str, bound: str, default: float) -> float:
        if feature_name not in self.feature_names:
            return float(default)
        feature_idx = self.feature_names.index(feature_name)
        values = self.feature_min if bound == "min" else self.feature_max
        if feature_idx >= len(values):
            return float(default)
        return float(values[feature_idx])

    def _load_feature_normalization_stats_if_available(self) -> None:
        if not self.feature_normalizer_stats_path.exists():
            return
        with self.feature_normalizer_stats_path.open("r", encoding="utf-8") as file_handle:
            stats = json.load(file_handle)

        self.feature_names = list(stats.get("feature_names", []))
        self.feature_min = np.asarray(stats.get("min", []), dtype=float)
        self.feature_max = np.asarray(stats.get("max", []), dtype=float)
        self.private_feature_names = list(stats.get("private_feature_names", []))

        for pid_str, player_stats in stats.get("player_private_min_max", {}).items():
            self.player_private_min_max[int(pid_str)] = {
                "min": np.asarray(player_stats.get("min", []), dtype=float),
                "max": np.asarray(player_stats.get("max", []), dtype=float),
            }

    def _per_generator_config_value(self, raw: Any, generator_idx: int, default: Any) -> Any:
        if raw is None:
            return default
        if isinstance(raw, dict):
            name = self._generator_name(generator_idx)
            for key in (generator_idx, str(generator_idx), name, name.upper(), name.lower()):
                if key in raw:
                    return raw[key]
            return default
        if isinstance(raw, (list, tuple, np.ndarray, pd.Series)):
            return raw[int(generator_idx)]
        return raw

    def _wind_generator_config_value(self, cfg: dict[str, Any], field_name: str, generator_idx: int, default: Any) -> Any:
        grouped = cfg.get("wind_generators")
        name = self._generator_name(generator_idx)
        if isinstance(grouped, dict):
            for key in (generator_idx, str(generator_idx), name, name.upper(), name.lower()):
                if key in grouped and isinstance(grouped[key], dict) and field_name in grouped[key]:
                    return grouped[key][field_name]

        legacy_key = {"reference": "wind_reference", "min": "wind_min", "max": "wind_max"}[field_name]
        return self._per_generator_config_value(cfg.get(legacy_key), generator_idx, default)

    def _configure_support_set_parameters(self) -> None:
        cfg = self.support_set_config
        self.support_reference_mode = "empirical_scenario"

        self.support_demand_reference = self._as_profile(
            cfg.get("demand_reference", self.reference_demand),
            self.num_time_steps,
            "demand_reference",
        )
        demand_min_default = self._feature_bound("demand", "min", 0.8 * self.reference_demand)
        demand_max_default = self._feature_bound("demand", "max", 1.2 * self.reference_demand)
        self.support_demand_min = float(cfg.get("demand_min", demand_min_default))
        self.support_demand_max = float(cfg.get("demand_max", demand_max_default))
        if self.support_demand_min > self.support_demand_max:
            raise ValueError("support_set_config demand_min cannot exceed demand_max")

        demand_range = self.support_demand_max - self.support_demand_min
        self.support_demand_ramp = float(cfg.get("demand_ramp", demand_range))
        self.support_demand_budget = float(
            cfg.get("demand_budget", self.num_time_steps * demand_range)
        )

        total_wind_capacity = sum(
            self.static_physical_capacity[i] for i in self.wind_physical_generator_ids
        )
        wind_count = max(len(self.wind_physical_generator_ids), 1)
        wind_min_default = self._feature_bound("wind_forecast", "min", 0.5 * total_wind_capacity) / wind_count
        wind_max_default = self._feature_bound("wind_forecast", "max", total_wind_capacity) / wind_count

        self.support_wind_reference: dict[int, list[float]] = {}
        self.support_wind_min: dict[int, float] = {}
        self.support_wind_max: dict[int, float] = {}
        for i in self.wind_physical_generator_ids:
            self.support_wind_reference[i] = self._as_profile(
                self._wind_generator_config_value(
                    cfg,
                    "reference",
                    i,
                    self.static_physical_capacity[i],
                ),
                self.num_time_steps,
                f"wind_generators[{self._generator_name(i)}].reference",
            )
            self.support_wind_min[i] = float(
                self._wind_generator_config_value(cfg, "min", i, wind_min_default)
            )
            self.support_wind_max[i] = float(
                self._wind_generator_config_value(
                    cfg,
                    "max",
                    i,
                    min(self.static_physical_capacity[i], wind_max_default),
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

    def _row_block_capacity_profile(self, row: dict[str, Any], block_name: str, block_idx: int) -> list[float]:
        for suffix in ("_cap_profile", "_profile"):
            column = f"{block_name}{suffix}"
            if column not in row:
                continue
            raw_value = row.get(column)
            if raw_value is None:
                continue
            if isinstance(raw_value, float) and np.isnan(raw_value):
                continue
            return self._as_profile(raw_value, self.num_time_steps, column)

        cap_value = row.get(f"{block_name}_cap", self.static_block_capacity[int(block_idx)])
        return [float(cap_value)] * self.num_time_steps

    def _configure_empirical_scenarios(self, empirical_scenario: Optional[Any]) -> None:
        if empirical_scenario is None:
            empirical_rows = [
                {
                    "regime": "reference",
                    "scenario_id": None,
                    "demand_profile": list(self.support_demand_reference),
                }
            ]
        elif isinstance(empirical_scenario, pd.DataFrame):
            empirical_rows = [row.to_dict() for _, row in empirical_scenario.iterrows()]
        elif isinstance(empirical_scenario, dict):
            empirical_rows = [empirical_scenario]
        else:
            empirical_rows = list(empirical_scenario)

        if not empirical_rows:
            raise ValueError("At least one empirical scenario is required")

        self.num_empirical_scenarios = len(empirical_rows)
        self.empirical_regime = str(empirical_rows[0].get("regime", "unknown"))
        self.empirical_scenario_ids = [
            None if row.get("scenario_id") is None else int(row.get("scenario_id"))
            for row in empirical_rows
        ]

        self.empirical_demand_profiles: dict[int, list[float]] = {}
        self.empirical_capacity_profiles: dict[int, dict[int, list[float]]] = {}

        for k, row in enumerate(empirical_rows):
            self.empirical_demand_profiles[k] = self._as_profile(
                row["demand_profile"],
                self.num_time_steps,
                f"empirical_scenario[{k}]['demand_profile']",
            )

            block_profiles = {
                block_idx: self._row_block_capacity_profile(row, block_name, block_idx)
                for block_idx, block_name in enumerate(self.block_names)
            }
            capacity_profiles: dict[int, list[float]] = {}
            for i in range(self.num_physical_generators):
                if i in self.wind_physical_generator_ids:
                    capacity_profiles[i] = [
                        sum(block_profiles[g][t] for g in self.physical_to_block_indices[i])
                        for t in range(self.num_time_steps)
                    ]
                else:
                    capacity_profiles[i] = [
                        sum(
                            float(row.get(f"{self.block_names[g]}_cap", self.static_block_capacity[g]))
                            for g in self.physical_to_block_indices[i]
                        )
                    ] * self.num_time_steps
            self.empirical_capacity_profiles[k] = capacity_profiles

    def _load_policy_if_requested(self) -> None:
        if self.policy_data is None and self.policy_results_path is None:
            return
        raise NotImplementedError(
            "The streamlined DRO model currently uses true block costs as bids. "
            "Policy embedding should be added against the block-indexed alpha[k,i,b,t] surface."
        )

    # ------------------------------------------------------------------
    # Big-M helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in str(key).split(",") if part != "")

    def _tight_dual_upper_bound(
        self,
        dual_name: str,
        index: tuple[int, ...],
        default_bound: float,
    ) -> float:
        entries = (getattr(self, "tight_big_m", {}) or {}).get(dual_name, {}) or {}
        details = entries.get(self._json_key(index))
        if not details:
            return float(default_bound)

        tight_value = details.get("tight_big_m")
        if tight_value is None:
            return float(default_bound)
        return max(0.0, min(float(default_bound), float(tight_value)))

    def _block_capacity_big_m(self, physical_generator_idx: int, local_block_idx: int) -> float:
        global_block = self.local_to_global_block[(int(physical_generator_idx), int(local_block_idx))]
        if int(physical_generator_idx) in self.conventional_physical_generator_ids:
            return float(self.static_block_capacity[global_block])

        local_blocks = self.local_blocks_by_generator[int(physical_generator_idx)]
        if len(local_blocks) == 1:
            return float(self.support_wind_max[int(physical_generator_idx)])

        static_total = sum(
            self.static_block_capacity[self.local_to_global_block[(int(physical_generator_idx), b)]]
            for b in local_blocks
        )
        if static_total <= 0:
            return 0.0
        block_share = self.static_block_capacity[global_block] / static_total
        return float(block_share * self.support_wind_max[int(physical_generator_idx)])

    def _physical_capacity_big_m(self, physical_generator_idx: int) -> float:
        if int(physical_generator_idx) in self.wind_physical_generator_ids:
            return float(self.support_wind_max[int(physical_generator_idx)])
        return float(self.static_physical_capacity[int(physical_generator_idx)])

    def _ramp_up_big_m(self, physical_generator_idx: int) -> float:
        return float(
            self.ramp_vector_up[int(physical_generator_idx)]
            + self._physical_capacity_big_m(int(physical_generator_idx))
        )

    def _ramp_down_big_m(self, physical_generator_idx: int) -> float:
        return float(
            self.ramp_vector_down[int(physical_generator_idx)]
            + self._physical_capacity_big_m(int(physical_generator_idx))
        )

    def _ramp_up_initial_big_m(self, physical_generator_idx: int) -> float:
        return float(
            self.p_init[int(physical_generator_idx)]
            + self.ramp_vector_up[int(physical_generator_idx)]
        )

    def _ramp_down_initial_big_m(self, physical_generator_idx: int) -> float:
        return float(
            max(
                0.0,
                self._physical_capacity_big_m(int(physical_generator_idx))
                - self.p_init[int(physical_generator_idx)]
                + self.ramp_vector_down[int(physical_generator_idx)],
            )
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        self.model = ConcreteModel()

        self.model.scenarios = Set(initialize=range(self.num_empirical_scenarios))
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))

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
        m = self.model
        m.D = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_block = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.C_eq = Var(m.scenarios, domain=Reals)
        m.C_opt = Var(m.scenarios, domain=Reals)
        m.PoA = Var(m.scenarios, domain=Reals)
        m.wasserstein_distance = Var(m.scenarios, domain=NonNegativeReals)

        m.D_abs_deviation = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_phys_abs_deviation = Var(
            m.scenarios,
            m.wind_physical_generators,
            m.time_steps,
            domain=NonNegativeReals,
        )
        m.D_transport_abs_deviation = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_phys_transport_abs_deviation = Var(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            domain=NonNegativeReals,
        )

    def _build_equilibrium_variables(self) -> None:
        m = self.model
        m.P_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.alpha = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Reals)
        m.lambda_eq = Var(m.scenarios, m.time_steps, domain=Reals)
        m.mu_upper_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.mu_lower_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.mu_ramp_up_eq = Var(m.scenarios, m.physical_generators, m.time_steps_plus_1, domain=NonNegativeReals)
        m.mu_ramp_down_eq = Var(m.scenarios, m.physical_generators, m.time_steps_plus_1, domain=NonNegativeReals)

    def _build_complementarity_equilibrium_variables(self) -> None:
        m = self.model
        m.z_upper_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_lower_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_ramp_up_eq = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)
        m.z_ramp_down_eq = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        m = self.model
        m.P_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.lambda_opt = Var(m.scenarios, m.time_steps, domain=Reals)
        m.mu_upper_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.mu_lower_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.mu_ramp_up_opt = Var(m.scenarios, m.physical_generators, m.time_steps_plus_1, domain=NonNegativeReals)
        m.mu_ramp_down_opt = Var(m.scenarios, m.physical_generators, m.time_steps_plus_1, domain=NonNegativeReals)

    def _build_complementarity_optimal_variables(self) -> None:
        m = self.model
        m.z_upper_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_lower_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_ramp_up_opt = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)
        m.z_ramp_down_opt = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)

    def _build_objective(self) -> None:
        m = self.model
        m.objective = Objective(
            expr=sum(
                m.PoA[k] - self.eta * m.wasserstein_distance[k]
                for k in m.scenarios
            ) / self.num_empirical_scenarios,
            sense=maximize,
        )

    def _build_constraints(self) -> None:
        self._build_support_set()
        self._build_policy_constraints()
        self._build_lower_level_equilibrium_constraints()
        self._build_lower_level_optimal_constraints()
        self._build_KKT_stationarity_equilibrium_constraints()
        self._build_KKT_stationarity_optimal_constraints()
        self._build_KKT_complementarity_equilibrium_constraints()
        self._build_KKT_complementarity_optimal_constraints()
        self._build_wasserstein_distance_constraints()
        self._build_PoA_constraints()

    # ------------------------------------------------------------------
    # Support set and DRO distance
    # ------------------------------------------------------------------

    def _build_support_set(self) -> None:
        if self.support_demand_ramp < 0 or self.support_demand_budget < 0:
            raise ValueError("Demand ramp and budget must be non-negative")
        if self.support_wind_ramp < 0 or self.support_wind_budget < 0:
            raise ValueError("Wind ramp and budget must be non-negative")

        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_support_set_demand(self) -> None:
        m = self.model

        def demand_lower_rule(model, k, t):
            return model.D[k, t] >= self.support_demand_min

        def demand_upper_rule(model, k, t):
            return model.D[k, t] <= self.support_demand_max

        def demand_ramp_up_rule(model, k, t):
            return model.D[k, t] - model.D[k, t - 1] <= self.support_demand_ramp

        def demand_ramp_down_rule(model, k, t):
            return model.D[k, t - 1] - model.D[k, t] <= self.support_demand_ramp

        def demand_abs_deviation_pos_rule(model, k, t):
            return (
                model.D_abs_deviation[k, t]
                >= model.D[k, t] - self.empirical_demand_profiles[int(k)][int(t)]
            )

        def demand_abs_deviation_neg_rule(model, k, t):
            return (
                model.D_abs_deviation[k, t]
                >= self.empirical_demand_profiles[int(k)][int(t)] - model.D[k, t]
            )

        def demand_budget_rule(model, k):
            return sum(model.D_abs_deviation[k, t] for t in model.time_steps) <= self.support_demand_budget

        m.demand_lower_bound_constraints = Constraint(m.scenarios, m.time_steps, rule=demand_lower_rule)
        m.demand_upper_bound_constraints = Constraint(m.scenarios, m.time_steps, rule=demand_upper_rule)
        m.demand_ramp_up_constraints = Constraint(m.scenarios, m.time_steps_minus_1, rule=demand_ramp_up_rule)
        m.demand_ramp_down_constraints = Constraint(m.scenarios, m.time_steps_minus_1, rule=demand_ramp_down_rule)
        m.demand_abs_deviation_pos_constraints = Constraint(m.scenarios, m.time_steps, rule=demand_abs_deviation_pos_rule)
        m.demand_abs_deviation_neg_constraints = Constraint(m.scenarios, m.time_steps, rule=demand_abs_deviation_neg_rule)
        m.demand_budget_constraint = Constraint(m.scenarios, rule=demand_budget_rule)

    def _build_support_set_wind(self) -> None:
        m = self.model

        def conventional_capacity_rule(model, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return model.P_max_block[k, i, b, t] == self.static_block_capacity[global_block]

        def wind_total_lower_rule(model, k, i, t):
            return (
                sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                >= self.support_wind_min[int(i)]
            )

        def wind_total_upper_rule(model, k, i, t):
            return (
                sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= self.support_wind_max[int(i)]
            )

        def wind_even_block_split_rule(model, k, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return (
                len(local_blocks) * model.P_max_block[k, i, b, t]
                == sum(model.P_max_block[k, i, other_b, t] for other_b in local_blocks)
            )

        def wind_ramp_up_rule(model, k, i, t):
            return (
                sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(model.P_max_block[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                <= self.support_wind_ramp
            )

        def wind_ramp_down_rule(model, k, i, t):
            return (
                sum(model.P_max_block[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= self.support_wind_ramp
            )

        def wind_abs_deviation_pos_rule(model, k, i, t):
            return (
                model.P_max_phys_abs_deviation[k, i, t]
                >= sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - self.empirical_capacity_profiles[int(k)][int(i)][int(t)]
            )

        def wind_abs_deviation_neg_rule(model, k, i, t):
            return (
                model.P_max_phys_abs_deviation[k, i, t]
                >= self.empirical_capacity_profiles[int(k)][int(i)][int(t)]
                - sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            )

        def wind_budget_rule(model, k):
            return (
                sum(
                    model.P_max_phys_abs_deviation[k, i, t]
                    for i in model.wind_physical_generators
                    for t in model.time_steps
                )
                <= self.support_wind_budget
            )

        m.conventional_capacity = Constraint(m.scenarios, m.conventional_blocks, m.time_steps, rule=conventional_capacity_rule)
        m.wind_total_lower_bound = Constraint(m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_total_lower_rule)
        m.wind_total_upper_bound = Constraint(m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_total_upper_rule)
        m.wind_even_block_split = Constraint(m.scenarios, m.wind_blocks, m.time_steps, rule=wind_even_block_split_rule)
        m.wind_ramp_up = Constraint(m.scenarios, m.wind_physical_generators, m.time_steps_minus_1, rule=wind_ramp_up_rule)
        m.wind_ramp_down = Constraint(m.scenarios, m.wind_physical_generators, m.time_steps_minus_1, rule=wind_ramp_down_rule)
        m.wind_abs_deviation_pos = Constraint(m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_abs_deviation_pos_rule)
        m.wind_abs_deviation_neg = Constraint(m.scenarios, m.wind_physical_generators, m.time_steps, rule=wind_abs_deviation_neg_rule)
        m.wind_budget = Constraint(m.scenarios, rule=wind_budget_rule)

    def _physical_capacity_expr(self, k: int, i: int, t: int):
        return sum(
            self.model.P_max_block[k, i, b, t]
            for b in self.local_blocks_by_generator[int(i)]
        )

    def _build_wasserstein_distance_constraints(self) -> None:
        m = self.model

        def demand_transport_pos_rule(model, k, t):
            return model.D_transport_abs_deviation[k, t] >= model.D[k, t] - self.empirical_demand_profiles[int(k)][int(t)]

        def demand_transport_neg_rule(model, k, t):
            return model.D_transport_abs_deviation[k, t] >= self.empirical_demand_profiles[int(k)][int(t)] - model.D[k, t]

        def capacity_transport_pos_rule(model, k, i, t):
            physical_capacity = sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            return (
                model.P_max_phys_transport_abs_deviation[k, i, t]
                >= physical_capacity - self.empirical_capacity_profiles[int(k)][int(i)][int(t)]
            )

        def capacity_transport_neg_rule(model, k, i, t):
            physical_capacity = sum(model.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            return (
                model.P_max_phys_transport_abs_deviation[k, i, t]
                >= self.empirical_capacity_profiles[int(k)][int(i)][int(t)] - physical_capacity
            )

        def wasserstein_distance_rule(model, k):
            return model.wasserstein_distance[k] == (
                sum(model.D_transport_abs_deviation[k, t] for t in model.time_steps)
                + sum(
                    model.P_max_phys_transport_abs_deviation[k, i, t]
                    for i in model.physical_generators
                    for t in model.time_steps
                )
            )

        m.demand_transport_pos = Constraint(m.scenarios, m.time_steps, rule=demand_transport_pos_rule)
        m.demand_transport_neg = Constraint(m.scenarios, m.time_steps, rule=demand_transport_neg_rule)
        m.capacity_transport_pos = Constraint(m.scenarios, m.physical_generators, m.time_steps, rule=capacity_transport_pos_rule)
        m.capacity_transport_neg = Constraint(m.scenarios, m.physical_generators, m.time_steps, rule=capacity_transport_neg_rule)
        m.wasserstein_distance_definition = Constraint(m.scenarios, rule=wasserstein_distance_rule)

    # ------------------------------------------------------------------
    # Lower level equilibrium and optimality constraints
    # ------------------------------------------------------------------

    def _build_policy_constraints(self) -> None:
        def true_cost_alpha_rule(model, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return model.alpha[k, i, b, t] == self.block_cost_vector[global_block]

        self.model.true_cost_alpha = Constraint(
            self.model.scenarios,
            self.model.generator_blocks,
            self.model.time_steps,
            rule=true_cost_alpha_rule,
        )

    def _build_lower_level_equilibrium_constraints(self) -> None:
        m = self.model

        def power_balance_eq_rule(model, k, t):
            return model.D[k, t] - sum(model.P_eq[k, i, b, t] for (i, b) in model.generator_blocks) == 0

        def generation_upper_eq_rule(model, k, i, b, t):
            return model.P_eq[k, i, b, t] - model.P_max_block[k, i, b, t] <= 0

        def generation_lower_eq_rule(model, k, i, b, t):
            return model.P_eq[k, i, b, t] >= 0

        def ramp_up_eq_rule(model, k, i, t):
            return (
                sum(model.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(model.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
                <= 0
            )

        def ramp_up_initial_eq_rule(model, k, i):
            return (
                sum(model.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
                <= 0
            )

        def ramp_down_eq_rule(model, k, i, t):
            return (
                -sum(model.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(model.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        def ramp_down_initial_eq_rule(model, k, i):
            return (
                -sum(model.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        m.power_balance_eq = Constraint(m.scenarios, m.time_steps, rule=power_balance_eq_rule)
        m.generation_upper_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=generation_upper_eq_rule)
        m.generation_lower_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=generation_lower_eq_rule)
        m.ramp_up_eq = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_up_eq_rule)
        m.ramp_up_initial_eq = Constraint(m.scenarios, m.physical_generators, rule=ramp_up_initial_eq_rule)
        m.ramp_down_eq = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_down_eq_rule)
        m.ramp_down_initial_eq = Constraint(m.scenarios, m.physical_generators, rule=ramp_down_initial_eq_rule)

    def _build_lower_level_optimal_constraints(self) -> None:
        m = self.model

        def power_balance_opt_rule(model, k, t):
            return model.D[k, t] - sum(model.P_opt[k, i, b, t] for (i, b) in model.generator_blocks) == 0

        def generation_upper_opt_rule(model, k, i, b, t):
            return model.P_opt[k, i, b, t] - model.P_max_block[k, i, b, t] <= 0

        def generation_lower_opt_rule(model, k, i, b, t):
            return model.P_opt[k, i, b, t] >= 0

        def ramp_up_opt_rule(model, k, i, t):
            return (
                sum(model.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(model.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
                <= 0
            )

        def ramp_up_initial_opt_rule(model, k, i):
            return (
                sum(model.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
                <= 0
            )

        def ramp_down_opt_rule(model, k, i, t):
            return (
                -sum(model.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(model.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        def ramp_down_initial_opt_rule(model, k, i):
            return (
                -sum(model.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        m.power_balance_opt = Constraint(m.scenarios, m.time_steps, rule=power_balance_opt_rule)
        m.generation_upper_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=generation_upper_opt_rule)
        m.generation_lower_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=generation_lower_opt_rule)
        m.ramp_up_opt = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_up_opt_rule)
        m.ramp_up_initial_opt = Constraint(m.scenarios, m.physical_generators, rule=ramp_up_initial_opt_rule)
        m.ramp_down_opt = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_down_opt_rule)
        m.ramp_down_initial_opt = Constraint(m.scenarios, m.physical_generators, rule=ramp_down_initial_opt_rule)

    # ------------------------------------------------------------------
    # KKT stationarity and complementarity
    # ------------------------------------------------------------------

    def _build_KKT_stationarity_equilibrium_constraints(self) -> None:
        m = self.model

        def stationarity_eq_rule(model, k, i, b, t):
            return (
                model.alpha[k, i, b, t]
                - model.lambda_eq[k, t]
                + model.mu_upper_eq[k, i, b, t]
                - model.mu_lower_eq[k, i, b, t]
                + model.mu_ramp_up_eq[k, i, t]
                - model.mu_ramp_up_eq[k, i, t + 1]
                - model.mu_ramp_down_eq[k, i, t]
                + model.mu_ramp_down_eq[k, i, t + 1]
                == 0
            )

        def final_ramp_up_dual_eq_rule(model, k, i):
            return model.mu_ramp_up_eq[k, i, self.num_time_steps] == 0

        def final_ramp_down_dual_eq_rule(model, k, i):
            return model.mu_ramp_down_eq[k, i, self.num_time_steps] == 0

        m.stationarity_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=stationarity_eq_rule)
        m.final_ramp_up_dual_eq = Constraint(m.scenarios, m.physical_generators, rule=final_ramp_up_dual_eq_rule)
        m.final_ramp_down_dual_eq = Constraint(m.scenarios, m.physical_generators, rule=final_ramp_down_dual_eq_rule)

    def _build_KKT_stationarity_optimal_constraints(self) -> None:
        m = self.model

        def stationarity_opt_rule(model, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return (
                self.block_cost_vector[global_block]
                - model.lambda_opt[k, t]
                + model.mu_upper_opt[k, i, b, t]
                - model.mu_lower_opt[k, i, b, t]
                + model.mu_ramp_up_opt[k, i, t]
                - model.mu_ramp_up_opt[k, i, t + 1]
                - model.mu_ramp_down_opt[k, i, t]
                + model.mu_ramp_down_opt[k, i, t + 1]
                == 0
            )

        def final_ramp_up_dual_opt_rule(model, k, i):
            return model.mu_ramp_up_opt[k, i, self.num_time_steps] == 0

        def final_ramp_down_dual_opt_rule(model, k, i):
            return model.mu_ramp_down_opt[k, i, self.num_time_steps] == 0

        m.stationarity_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=stationarity_opt_rule)
        m.final_ramp_up_dual_opt = Constraint(m.scenarios, m.physical_generators, rule=final_ramp_up_dual_opt_rule)
        m.final_ramp_down_dual_opt = Constraint(m.scenarios, m.physical_generators, rule=final_ramp_down_dual_opt_rule)

    def _build_KKT_complementarity_equilibrium_constraints(self) -> None:
        m = self.model

        def upper_bound_complementarity_eq_rule(model, k, i, b, t):
            return -self._block_capacity_big_m(int(i), int(b)) * (1 - model.z_upper_eq[k, i, b, t]) <= (
                model.P_eq[k, i, b, t] - model.P_max_block[k, i, b, t]
            )

        def upper_bound_complementarity_dual_eq_rule(model, k, i, b, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_upper_eq",
                (int(k), int(i), int(b), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_upper_eq[k, i, b, t] <= dual_big_m * model.z_upper_eq[k, i, b, t]

        def lower_bound_complementarity_eq_rule(model, k, i, b, t):
            return -self._block_capacity_big_m(int(i), int(b)) * (1 - model.z_lower_eq[k, i, b, t]) <= -model.P_eq[k, i, b, t]

        def lower_bound_complementarity_dual_eq_rule(model, k, i, b, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_lower_eq",
                (int(k), int(i), int(b), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_lower_eq[k, i, b, t] <= dual_big_m * model.z_lower_eq[k, i, b, t]

        def ramp_up_complementarity_eq_rule(model, k, i, t):
            return -self._ramp_up_big_m(int(i)) * (1 - model.z_ramp_up_eq[k, i, t]) <= (
                sum(model.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(model.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_eq_rule(model, k, i):
            return -self._ramp_up_initial_big_m(int(i)) * (1 - model.z_ramp_up_eq[k, i, 0]) <= (
                sum(model.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_eq_rule(model, k, i, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_ramp_up_eq",
                (int(k), int(i), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_ramp_up_eq[k, i, t] <= dual_big_m * model.z_ramp_up_eq[k, i, t]

        def ramp_down_complementarity_eq_rule(model, k, i, t):
            return -self._ramp_down_big_m(int(i)) * (1 - model.z_ramp_down_eq[k, i, t]) <= (
                -sum(model.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(model.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_eq_rule(model, k, i):
            return -self._ramp_down_initial_big_m(int(i)) * (1 - model.z_ramp_down_eq[k, i, 0]) <= (
                -sum(model.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_eq_rule(model, k, i, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_ramp_down_eq",
                (int(k), int(i), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_ramp_down_eq[k, i, t] <= dual_big_m * model.z_ramp_down_eq[k, i, t]

        m.upper_bound_complementarity_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=upper_bound_complementarity_eq_rule)
        m.upper_bound_complementarity_dual_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=upper_bound_complementarity_dual_eq_rule)
        m.lower_bound_complementarity_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=lower_bound_complementarity_eq_rule)
        m.lower_bound_complementarity_dual_eq = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=lower_bound_complementarity_dual_eq_rule)
        m.ramp_up_complementarity_eq = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_up_complementarity_eq_rule)
        m.ramp_up_initial_complementarity_eq = Constraint(m.scenarios, m.physical_generators, rule=ramp_up_initial_complementarity_eq_rule)
        m.ramp_up_complementarity_dual_eq = Constraint(m.scenarios, m.physical_generators, m.time_steps, rule=ramp_up_complementarity_dual_eq_rule)
        m.ramp_down_complementarity_eq = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_down_complementarity_eq_rule)
        m.ramp_down_initial_complementarity_eq = Constraint(m.scenarios, m.physical_generators, rule=ramp_down_initial_complementarity_eq_rule)
        m.ramp_down_complementarity_dual_eq = Constraint(m.scenarios, m.physical_generators, m.time_steps, rule=ramp_down_complementarity_dual_eq_rule)

    def _build_KKT_complementarity_optimal_constraints(self) -> None:
        m = self.model

        def upper_bound_complementarity_opt_rule(model, k, i, b, t):
            return -self._block_capacity_big_m(int(i), int(b)) * (1 - model.z_upper_opt[k, i, b, t]) <= (
                model.P_opt[k, i, b, t] - model.P_max_block[k, i, b, t]
            )

        def upper_bound_complementarity_dual_opt_rule(model, k, i, b, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_upper_opt",
                (int(k), int(i), int(b), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_upper_opt[k, i, b, t] <= dual_big_m * model.z_upper_opt[k, i, b, t]

        def lower_bound_complementarity_opt_rule(model, k, i, b, t):
            return -self._block_capacity_big_m(int(i), int(b)) * (1 - model.z_lower_opt[k, i, b, t]) <= -model.P_opt[k, i, b, t]

        def lower_bound_complementarity_dual_opt_rule(model, k, i, b, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_lower_opt",
                (int(k), int(i), int(b), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_lower_opt[k, i, b, t] <= dual_big_m * model.z_lower_opt[k, i, b, t]

        def ramp_up_complementarity_opt_rule(model, k, i, t):
            return -self._ramp_up_big_m(int(i)) * (1 - model.z_ramp_up_opt[k, i, t]) <= (
                sum(model.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(model.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_opt_rule(model, k, i):
            return -self._ramp_up_initial_big_m(int(i)) * (1 - model.z_ramp_up_opt[k, i, 0]) <= (
                sum(model.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_opt_rule(model, k, i, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_ramp_up_opt",
                (int(k), int(i), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_ramp_up_opt[k, i, t] <= dual_big_m * model.z_ramp_up_opt[k, i, t]

        def ramp_down_complementarity_opt_rule(model, k, i, t):
            return -self._ramp_down_big_m(int(i)) * (1 - model.z_ramp_down_opt[k, i, t]) <= (
                -sum(model.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(model.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_opt_rule(model, k, i):
            return -self._ramp_down_initial_big_m(int(i)) * (1 - model.z_ramp_down_opt[k, i, 0]) <= (
                -sum(model.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_opt_rule(model, k, i, t):
            dual_big_m = self._tight_dual_upper_bound(
                "mu_ramp_down_opt",
                (int(k), int(i), int(t)),
                self.big_m_complementarity,
            )
            return model.mu_ramp_down_opt[k, i, t] <= dual_big_m * model.z_ramp_down_opt[k, i, t]

        m.upper_bound_complementarity_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=upper_bound_complementarity_opt_rule)
        m.upper_bound_complementarity_dual_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=upper_bound_complementarity_dual_opt_rule)
        m.lower_bound_complementarity_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=lower_bound_complementarity_opt_rule)
        m.lower_bound_complementarity_dual_opt = Constraint(m.scenarios, m.generator_blocks, m.time_steps, rule=lower_bound_complementarity_dual_opt_rule)
        m.ramp_up_complementarity_opt = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_up_complementarity_opt_rule)
        m.ramp_up_initial_complementarity_opt = Constraint(m.scenarios, m.physical_generators, rule=ramp_up_initial_complementarity_opt_rule)
        m.ramp_up_complementarity_dual_opt = Constraint(m.scenarios, m.physical_generators, m.time_steps, rule=ramp_up_complementarity_dual_opt_rule)
        m.ramp_down_complementarity_opt = Constraint(m.scenarios, m.physical_generators, m.time_steps_minus_1, rule=ramp_down_complementarity_opt_rule)
        m.ramp_down_initial_complementarity_opt = Constraint(m.scenarios, m.physical_generators, rule=ramp_down_initial_complementarity_opt_rule)
        m.ramp_down_complementarity_dual_opt = Constraint(m.scenarios, m.physical_generators, m.time_steps, rule=ramp_down_complementarity_dual_opt_rule)

    # ------------------------------------------------------------------
    # PoA objective terms
    # ------------------------------------------------------------------

    def _build_PoA_constraints(self) -> None:
        m = self.model

        def cost_eq_rule(model, k):
            return model.C_eq[k] == sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * model.P_eq[k, i, b, t]
                for (i, b) in model.generator_blocks
                for t in model.time_steps
            )

        def cost_opt_rule(model, k):
            return model.C_opt[k] == sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * model.P_opt[k, i, b, t]
                for (i, b) in model.generator_blocks
                for t in model.time_steps
            )

        def poa_rule(model, k):
            return model.C_eq[k] - model.C_opt[k] == model.PoA[k]

        m.cost_definition_eq = Constraint(m.scenarios, rule=cost_eq_rule)
        m.cost_definition_opt = Constraint(m.scenarios, rule=cost_opt_rule)
        m.poa_definition = Constraint(m.scenarios, rule=poa_rule)

    # ------------------------------------------------------------------
    # Apply precomputed tightening reports
    # ------------------------------------------------------------------

    def load_tightening_report(
        self,
        report_path: str | Path = "results/dro_poa_bidding_blocks_tightening_report.json",
    ) -> dict[str, Any]:
        """
        Load a DRO tightening report.

        Alpha and dual/binary entries are expected to carry the empirical
        scenario index first, e.g. alpha/upper/lower keys are `k,i,b,t` and ramp
        keys are `k,i,t`.
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
        Apply a scenario-indexed tightening report to an already-built DRO model.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        report = report or getattr(self, "tightening_report", None)
        if report is None:
            raise ValueError("No tightening report loaded. Call load_tightening_report() first.")

        m = self.model
        stats = {"fixed_binaries": 0, "dual_upper_bounds": 0, "alpha_bounds": 0}

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
                    new_ub = max(0.0, float(tight_value))
                    current_ub = dual_var[index].ub
                    if current_ub is not None:
                        new_ub = min(float(current_ub), new_ub)
                    dual_var[index].setub(new_ub)
                    stats["dual_upper_bounds"] += 1

        if apply_alpha_bounds:
            for key, bounds in (report.get("alpha_bounds", {}) or {}).items():
                index = self._parse_json_index(key)
                if not hasattr(m, "alpha") or index not in m.alpha:
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

    # ------------------------------------------------------------------
    # Solve and results
    # ------------------------------------------------------------------

    def solve(
        self,
        solver_name: str = "gurobi",
        tee: bool = True,
        time_limit: Optional[float] = None,
    ) -> Any:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        solver = SolverFactory(solver_name)
        solver.options["IntFeasTol"] = 1e-8
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        self.solver_results = solver.solve(self.model, tee=tee)
        return self.solver_results

    def _safe_value(self, expr: Any) -> Optional[float]:
        raw_value = value(expr, exception=False)
        if raw_value is None:
            return None
        return float(raw_value)

    def _profile_values(self, var: Any, scenario_idx: int, *leading_indices: int) -> list[Optional[float]]:
        return [
            self._safe_value(var[(scenario_idx, *leading_indices, t)] if leading_indices else var[scenario_idx, t])
            for t in range(self.num_time_steps)
        ]

    def extract_results(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        m = self.model
        scenario_results: dict[str, Any] = {}
        for k in m.scenarios:
            scenario_key = str(int(k))
            c_eq = self._safe_value(m.C_eq[k])
            c_opt = self._safe_value(m.C_opt[k])
            scenario_generators: dict[str, Any] = {}
            for i, generator_name in enumerate(self.physical_generator_names):
                blocks = []
                for b in self.local_blocks_by_generator[i]:
                    global_block = self.local_to_global_block[(i, b)]
                    blocks.append(
                        {
                            "local_block_index": int(b),
                            "global_block_index": int(global_block),
                            "block_name": self.block_names[global_block],
                            "capacity_profile": self._profile_values(m.P_max_block, int(k), i, b),
                            "alpha_profile": self._profile_values(m.alpha, int(k), i, b),
                            "equilibrium_dispatch": self._profile_values(m.P_eq, int(k), i, b),
                            "optimal_dispatch": self._profile_values(m.P_opt, int(k), i, b),
                            "true_cost": float(self.block_cost_vector[global_block]),
                        }
                    )

                scenario_generators[generator_name] = {
                    "physical_generator_index": int(i),
                    "is_wind": i in self.wind_physical_generator_ids,
                    "physical_capacity_profile": [
                        sum(
                            self._safe_value(m.P_max_block[int(k), i, b, t]) or 0.0
                            for b in self.local_blocks_by_generator[i]
                        )
                        for t in range(self.num_time_steps)
                    ],
                    "equilibrium_physical_dispatch": [
                        sum(
                            self._safe_value(m.P_eq[int(k), i, b, t]) or 0.0
                            for b in self.local_blocks_by_generator[i]
                        )
                        for t in range(self.num_time_steps)
                    ],
                    "optimal_physical_dispatch": [
                        sum(
                            self._safe_value(m.P_opt[int(k), i, b, t]) or 0.0
                            for b in self.local_blocks_by_generator[i]
                        )
                        for t in range(self.num_time_steps)
                    ],
                    "blocks": blocks,
                }

            scenario_results[scenario_key] = {
                "scenario_id": self.empirical_scenario_ids[int(k)],
                "C_eq": c_eq,
                "C_opt": c_opt,
                "PoA_difference": self._safe_value(m.PoA[k]),
                "PoA_ratio": c_eq / c_opt if c_eq is not None and c_opt not in (None, 0.0) else None,
                "wasserstein_distance": self._safe_value(m.wasserstein_distance[k]),
                "demand_profile": self._profile_values(m.D, int(k)),
                "equilibrium_price_profile": self._profile_values(m.lambda_eq, int(k)),
                "optimal_price_profile": self._profile_values(m.lambda_opt, int(k)),
                "generators": scenario_generators,
            }

        poa_values = [
            self._safe_value(m.PoA[k])
            for k in m.scenarios
            if self._safe_value(m.PoA[k]) is not None
        ]
        distance_values = [
            self._safe_value(m.wasserstein_distance[k])
            for k in m.scenarios
            if self._safe_value(m.wasserstein_distance[k]) is not None
        ]
        ratio_values = [
            scenario["PoA_ratio"]
            for scenario in scenario_results.values()
            if scenario["PoA_ratio"] is not None
        ]

        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(self.solver_results.solver.termination_condition),
            }

        return {
            "reference_case": self.reference_case,
            "regime": self.empirical_regime,
            "num_time_steps": self.num_time_steps,
            "num_empirical_scenarios": self.num_empirical_scenarios,
            "eta": float(self.eta),
            "objective": self._safe_value(m.objective),
            "average_poa_proxy": float(np.mean(poa_values)) if poa_values else None,
            "average_poa_ratio": float(np.mean(ratio_values)) if ratio_values else None,
            "min_poa_ratio": float(np.min(ratio_values)) if ratio_values else None,
            "max_poa_ratio": float(np.max(ratio_values)) if ratio_values else None,
            "average_wasserstein_distance": float(np.mean(distance_values)) if distance_values else None,
            "block_names": list(self.block_names),
            "physical_generator_names": list(self.physical_generator_names),
            "block_to_physical": dict(self.block_to_physical),
            "physical_to_block_indices": {
                str(i): list(blocks)
                for i, blocks in enumerate(self.physical_to_block_indices)
            },
            "support_set": {
                "reference_mode": self.support_reference_mode,
                "demand": {
                    "configured_reference_fallback": list(self.support_demand_reference),
                    "min": float(self.support_demand_min),
                    "max": float(self.support_demand_max),
                    "ramp": float(self.support_demand_ramp),
                    "budget": float(self.support_demand_budget),
                },
                "wind": {
                    self.physical_generator_names[i]: {
                        "configured_reference_fallback": list(self.support_wind_reference[i]),
                        "min": float(self.support_wind_min[i]),
                        "max": float(self.support_wind_max[i]),
                    }
                    for i in self.wind_physical_generator_ids
                },
                "wind_ramp": float(self.support_wind_ramp),
                "wind_budget": float(self.support_wind_budget),
            },
            "scenarios": scenario_results,
            "solver": solver_summary,
        }

    def solution_summary(self) -> dict[str, Any]:
        results = self.extract_results()
        return {
            "regime": results["regime"],
            "scenario_ids": list(self.empirical_scenario_ids),
            "eta": results["eta"],
            "inner_value": results["objective"],
            "average_poa_proxy": results["average_poa_proxy"],
            "average_wasserstein_distance": results["average_wasserstein_distance"],
            "average_poa": results["average_poa_ratio"],
            "min_poa": results["min_poa_ratio"],
            "max_poa": results["max_poa_ratio"],
            "termination_condition": results.get("solver", {}).get("termination_condition"),
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

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    @classmethod
    def load_regime_scenarios(
        cls,
        reference_case: str = "test_case_bidding_blocks",
        regime_config_path: str = "config/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        manager = ScenarioManager(base_case_reference=reference_case)
        scenario_set = manager.create_scenario_set_from_regimes(
            regime_config_path=regime_config_path,
            regime_set=regime_set,
            seed=seed,
        )
        return scenario_set["scenarios_df"]

    @classmethod
    def build_support_set_config_from_scenarios(
        cls,
        scenarios_df: pd.DataFrame,
        reference_case: str = "test_case_bidding_blocks",
        regime: Optional[str] = None,
    ) -> dict[str, Any]:
        if regime is not None:
            local_df = scenarios_df[scenarios_df["regime"].astype(str) == str(regime)].copy()
        else:
            local_df = scenarios_df.copy()
        if local_df.empty:
            raise ValueError(f"No scenarios available for regime '{regime}'")

        (
            _num_generators,
            _pmax_list,
            _pmin_list,
            _cost_vector,
            _ramp_up,
            _ramp_down,
            _demand,
            generators,
            _players,
            time_steps,
        ) = load_setup_data(reference_case)
        normalized = normalize_generators(generators)
        physical_generators = list(normalized["physical_generators"])
        blocks = list(normalized["blocks"])
        block_names = [str(block["block_name"]) for block in blocks]
        static_block_capacity = [float(block["pmax"]) for block in blocks]

        horizon = int(time_steps)
        demand_profiles = np.vstack(
            [
                np.asarray(cls._as_profile(row["demand_profile"], horizon, "demand_profile"), dtype=float)
                for _, row in local_df.iterrows()
            ]
        )
        demand_reference = np.mean(demand_profiles, axis=0)
        demand_ramp = float(np.max(np.abs(np.diff(demand_profiles, axis=1)))) if horizon > 1 else 0.0

        def max_pairwise_l1(profile_matrix: np.ndarray) -> float:
            if profile_matrix.shape[0] <= 1:
                return 0.0
            pairwise_distances = np.abs(
                profile_matrix[:, None, :] - profile_matrix[None, :, :]
            ).sum(axis=2)
            return float(np.max(pairwise_distances))

        demand_budget = max_pairwise_l1(demand_profiles)

        physical_idx_by_name = {
            str(gen["physical_name"]): idx
            for idx, gen in enumerate(physical_generators)
        }
        physical_to_block_indices: dict[int, list[int]] = {
            i: [] for i in range(len(physical_generators))
        }
        for block_idx, block in enumerate(blocks):
            physical_to_block_indices[physical_idx_by_name[str(block["physical_name"])]].append(block_idx)

        def block_capacity_profile(row: pd.Series, block_idx: int) -> list[float]:
            block_name = block_names[block_idx]
            for suffix in ("_cap_profile", "_profile"):
                column = f"{block_name}{suffix}"
                if column in row and row[column] is not None:
                    return cls._as_profile(row[column], horizon, column)
            return [float(row.get(f"{block_name}_cap", static_block_capacity[block_idx]))] * horizon

        wind_generators: dict[str, dict[str, Any]] = {}
        wind_ramp = 0.0
        wind_profiles_for_budget: list[np.ndarray] = []

        for gen_idx, generator in enumerate(physical_generators):
            gen_name = str(generator["physical_name"])
            if not (bool(generator.get("is_wind", False)) or cls._is_wind_name(gen_name)):
                continue

            physical_profiles = []
            for _, row in local_df.iterrows():
                block_profiles = [
                    block_capacity_profile(row, block_idx)
                    for block_idx in physical_to_block_indices[int(gen_idx)]
                ]
                physical_profiles.append(
                    [
                        sum(profile[t] for profile in block_profiles)
                        for t in range(horizon)
                    ]
                )
            wind_profiles = np.asarray(physical_profiles, dtype=float)
            reference = np.mean(wind_profiles, axis=0)
            wind_profiles_for_budget.append(wind_profiles)

            wind_generators[gen_name] = {
                "reference": reference.tolist(),
                "min": float(np.min(wind_profiles)),
                "max": float(min(float(generator["pmax"]), np.max(wind_profiles))),
            }
            if horizon > 1:
                wind_ramp = max(wind_ramp, float(np.max(np.abs(np.diff(wind_profiles, axis=1)))))

        if wind_profiles_for_budget:
            wind_budget_profiles = np.concatenate(wind_profiles_for_budget, axis=1)
            wind_budget = max_pairwise_l1(wind_budget_profiles)
        else:
            wind_budget = 0.0

        return {
            "reference_mode": "empirical_scenario",
            "demand_reference": demand_reference.tolist(),
            "demand_min": float(np.min(demand_profiles)),
            "demand_max": float(np.max(demand_profiles)),
            "demand_ramp": demand_ramp,
            "demand_budget": demand_budget,
            "wind_generators": wind_generators,
            "wind_ramp": wind_ramp,
            "wind_budget": wind_budget,
        }

    @classmethod
    def run_eta_sweep_by_regime(
        cls,
        eta_values: Sequence[float],
        P_init: Optional[Sequence[float]] = None,
        scenarios_df: Optional[pd.DataFrame] = None,
        regimes: Optional[Sequence[str]] = None,
        reference_case: str = "test_case_bidding_blocks",
        regime_config_path: str = "config/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        support_set_name: Optional[str] = None,
        solver_name: str = "gurobi",
        tee: bool = False,
        time_limit: Optional[float] = None,
        max_scenarios_per_regime: Optional[int] = None,
        big_m_complementarity: float = 1e8,
    ) -> pd.DataFrame:
        if scenarios_df is None:
            scenarios_df = cls.load_regime_scenarios(
                reference_case=reference_case,
                regime_config_path=regime_config_path,
                regime_set=regime_set,
            )

        if regimes is None:
            regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())

        records: list[dict[str, Any]] = []
        for regime in regimes:
            regime_df = scenarios_df[scenarios_df["regime"].astype(str) == str(regime)].copy()
            if regime_df.empty:
                raise ValueError(f"No scenarios available for regime '{regime}'")
            if max_scenarios_per_regime is not None:
                regime_df = regime_df.head(int(max_scenarios_per_regime)).copy()

            if support_set_name is None:
                support_set_config = cls.build_support_set_config_from_scenarios(
                    regime_df,
                    reference_case=reference_case,
                    regime=str(regime),
                )
            else:
                support_set_config = cls.load_support_set_config(config_name=support_set_name)

            for eta in eta_values:
                optimizer = cls(
                    P_init=P_init,
                    num_time_steps=int(regime_df.iloc[0]["time_steps"]),
                    reference_case=reference_case,
                    support_set_config=support_set_config,
                    eta=float(eta),
                    empirical_scenario=regime_df,
                    big_m_complementarity=big_m_complementarity,
                )
                optimizer.build_model()
                optimizer.solve(solver_name=solver_name, tee=tee, time_limit=time_limit)
                summary = optimizer.solution_summary()
                records.append(
                    {
                        "regime": summary["regime"],
                        "n_scenarios": len(summary["scenario_ids"]),
                        "eta": summary["eta"],
                        "inner_value": summary["inner_value"],
                        "average_PoA_proxy": summary["average_poa_proxy"],
                        "average_PoA": summary["average_poa"],
                        "min_PoA": summary["min_poa"],
                        "max_PoA": summary["max_poa"],
                        "average_distance": summary["average_wasserstein_distance"],
                        "termination_condition": summary["termination_condition"],
                    }
                )

        return pd.DataFrame(records)

    def plot_demand_capacity_trajectory(
        self,
        save_path: str | Path = "results/dro_poa_demand_capacity_trajectory.png",
        show: bool = True,
        scenario_index: int = 0,
    ) -> None:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        if scenario_index not in self.model.scenarios:
            raise ValueError(f"Invalid scenario_index {scenario_index}")

        import matplotlib.pyplot as plt

        time_points = list(self.model.time_steps)
        demand_vals = [
            self._safe_value(self.model.D[scenario_index, t])
            for t in time_points
        ]

        capacity_by_generator: dict[int, list[float]] = {}
        for i in self.model.physical_generators:
            capacity_by_generator[int(i)] = [
                sum(
                    self._safe_value(self.model.P_max_block[scenario_index, int(i), b, t]) or 0.0
                    for b in self.local_blocks_by_generator[int(i)]
                )
                for t in time_points
            ]

        total_capacity = [
            sum(capacity_by_generator[int(i)][t_idx] for i in self.model.physical_generators)
            for t_idx in range(len(time_points))
        ]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(time_points, demand_vals, color="black", linewidth=2.5, label="Demand")
        ax.plot(time_points, total_capacity, color="#1f77b4", linestyle="--", linewidth=2.2, label="Total Capacity")
        for i in self.model.physical_generators:
            ax.plot(
                time_points,
                capacity_by_generator[int(i)],
                linewidth=1.4,
                label=f"{self.physical_generator_names[int(i)]} Capacity",
            )
        ax.set_title(f"DRO Demand and Capacity Trajectories (Scenario {scenario_index})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2, fontsize=9)
        fig.tight_layout()

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    case = "test_case_bidding_blocks"
    regime_set = "PoA_analysis"
    seed = 1
    eta = 0.5
    max_scenarios = 10
    tightening_report_path = Path("results/dro_poa_bidding_blocks_tightening_report.json")

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )
    scenarios_df = scenarios["scenarios_df"].head(max_scenarios).copy()

    support_set_config = DRO_PoAOptimization.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case_bidding_blocks_base",
    )

    optimizer = DRO_PoAOptimization(
        num_time_steps=int(scenarios_df.iloc[0]["time_steps"]),
        P_init=None,
        reference_case=case,
        support_set_config=support_set_config,
        eta=eta,
        empirical_scenario=scenarios_df,
    )

    start = time.perf_counter()
    if tightening_report_path.exists():
        optimizer.load_tightening_report(tightening_report_path)
    optimizer.build_model()
    applied_stats = (
        optimizer.apply_tightened_bounds_to_model()
        if tightening_report_path.exists()
        else {"fixed_binaries": 0, "dual_upper_bounds": 0, "alpha_bounds": 0}
    )
    optimizer.solve(time_limit=400)
    result_path = optimizer.save_results("results/dro_poa_optimization_bidding_blocks_results.json")
    elapsed = time.perf_counter() - start

    print("\nDRO-PoA solve complete")
    print(f"  Regime set: {regime_set}")
    print(f"  Empirical scenarios: {len(scenarios_df)}")
    print(f"  Eta: {eta}")
    print(f"  Tightening report: {tightening_report_path if tightening_report_path.exists() else 'not applied'}")
    print(f"  Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"  Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"  Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"  Results: {result_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
