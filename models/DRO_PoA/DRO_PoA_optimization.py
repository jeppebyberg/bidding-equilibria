from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Expression,
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    SolverFactory,
    Var,
    maximize,
    value,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.helper import (
    ensure_profile,
    find_demand_profile_column,
    infer_num_time_steps,
    target_columns_to_local_blocks,
)
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel


class DRO_PoAOptimization:
    """
    Scenario-indexed distributionally robust Price of Anarchy optimization.

    The model keeps the block-aware physical-generator/local-block indexing of
    PoAOptimization and adds an empirical scenario index k as the first index on
    state, dispatch, KKT, policy, and PoA variables.
    """

    default_lambda_bound = 40.0
    default_capacity_dual_bound = 40.02
    default_ramp_dual_bound = 20.0
    default_primal_big_m_placeholder = 1.0e5
    normalization_epsilon = 1e-12
    aggregate_dual_bound_component_names = (
        "aggregate_mu_max_bound",
        "aggregate_mu_min_bound",
        "aggregate_mu_ramp_up_bound",
        "aggregate_mu_ramp_down_bound",
    )
    allowed_objective_modes = {
        "difference",
        "ratio_mccormick",
        "ratio_piecewise_mccormick",
    }
    ratio_bounds_tolerance = 1e-7

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        p_init: Optional[list[float] | list[list[float]]] = None,
        num_time_steps: Optional[int] = None,
        regime_config_path: str | Path = "config/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        regime_name: Optional[str] = None,
        eta: float = 0.0,
        epsilon: float = 0.0,
        nn_model_dir: Optional[str | Path] = None,
        nn_normalization_stats_path: Optional[str | Path] = None,
        nn_policy_generators: Optional[list[int | str]] = None,
        reference_case: str = "test_case_bidding_blocks",
        objective_mode: str = "difference",
        ratio_bounds: Optional[dict[str, Any]] = None,
        defer_ratio_bound_validation: bool = False,
    ):
        if float(eta) < 0.0:
            raise ValueError("eta must be nonnegative")
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be nonnegative")

        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.requested_p_init = p_init
        self.nn_model_dir = Path(nn_model_dir) if nn_model_dir is not None else None
        self.nn_normalization_stats_path = (
            Path(nn_normalization_stats_path)
            if nn_normalization_stats_path is not None
            else None
        )
        self.requested_nn_policy_generators = nn_policy_generators
        self.regime_config_path = Path(regime_config_path)
        self.regime_set = str(regime_set)
        self.eta = float(eta)
        self.epsilon = float(epsilon)
        self.reference_case = reference_case
        self.objective_mode = self._validate_objective_mode(objective_mode)
        self.defer_ratio_bound_validation = bool(defer_ratio_bound_validation)
        self._raw_ratio_bounds = ratio_bounds
        if (
            self.defer_ratio_bound_validation
            and self.objective_mode in {
                "ratio_mccormick",
                "ratio_piecewise_mccormick",
            }
            and (ratio_bounds is None or "C_opt" not in ratio_bounds)
        ):
            self.ratio_bounds = self._validate_deferred_ratio_bounds(ratio_bounds)
        else:
            self.ratio_bounds = self._validate_ratio_bounds(ratio_bounds)
        self.lambda_bound = float(self.default_lambda_bound)
        self.capacity_dual_bound = float(self.default_capacity_dual_bound)
        self.ramp_dual_bound = float(self.default_ramp_dual_bound)
        self.primal_big_m_placeholder = float(self.default_primal_big_m_placeholder)
        self.nn_policy_generator_ids: list[int] = []
        self.nn_policy_generator_names: list[str] = []
        self.nn_relu_bounds_report: dict[str, Any] = {}
        self.nn_relu_bounds: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {}
        self.nn_feature_bounds: dict[str, Any] = {}
        self.nn_bound_warnings: list[str] = []
        self.nn_policies: dict[str, Any] = {}
        self.nn_stats: dict[str, Any] = {}

        self.selected_regime = self.load_regime_config(
            self.regime_config_path,
            self.regime_set,
            regime_name,
        )
        self.regime_name = str(self.selected_regime["name"])
        self.selected_regime_parameters = dict(self.selected_regime)

        self.scenarios_df = self._filter_scenarios_to_regime(
            scenarios_df,
            self.regime_name,
            regime_name_was_explicit=regime_name is not None,
        )
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
        self._configure_nn_policy_generators()

        self.num_empirical_scenarios = len(self.scenarios_df)
        self.empirical_scenario_ids = [
            self._scenario_id_from_row(row, fallback=index)
            for index, (_, row) in enumerate(self.scenarios_df.iterrows())
        ]
        self.empirical_D = self._parse_empirical_demand_profiles()
        self.empirical_Pmax_block = self._parse_empirical_block_capacity_profiles()
        self.empirical_Pmax_phys = self._build_empirical_physical_capacity_profiles()
        self.p_init = self._normalize_p_init(self.requested_p_init)

        self._configure_fixed_regime_parameters()
        self._configure_regime_shape_profiles()
        if self.nn_model_dir is not None and self.nn_policy_generator_ids:
            self._load_nn_policies()
            self._load_nn_normalization_stats()
        self._initialize_big_m_placeholders()
        self.tightening_report: dict[str, Any] = {}
        self.fixed_binaries: dict[str, dict[str, Any]] = {}
        self.primal_big_m: dict[str, dict[str, Any]] = {}
        self.tight_big_m: dict[str, dict[str, Any]] = {}
        self.aggregate_dual_bounds: dict[str, Any] = {}
        self.lambda_bounds: dict[str, Any] = {}
        self.alpha_bounds: dict[tuple[int, ...], dict[str, float]] = {}
        self.alpha_bound_optimization_results: dict[str, Any] = {}
        self.optimal_cost_bounds: dict[str, Any] = {}
        self.scenario_optimal_cost_bounds: dict[str, Any] = {}
        self.optimal_cost_bound_optimization_results: dict[str, Any] = {}
        self._loaded_bounds_prepared = False

    def _validate_objective_mode(self, objective_mode: str) -> str:
        normalized_mode = str(objective_mode).strip().lower()
        if normalized_mode not in self.allowed_objective_modes:
            allowed = ", ".join(sorted(self.allowed_objective_modes))
            raise ValueError(
                f"objective_mode must be one of {{{allowed}}}; got {objective_mode!r}"
            )
        return normalized_mode

    def _validate_deferred_ratio_bounds(
        self,
        ratio_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if ratio_bounds is None:
            raise ValueError(
                f"ratio_bounds with at least 'phi' is required when "
                f"objective_mode='{self.objective_mode}' and "
                "defer_ratio_bound_validation=True"
            )
        if not isinstance(ratio_bounds, dict):
            raise ValueError("ratio_bounds must be a dictionary")
        if "phi" not in ratio_bounds:
            raise ValueError("ratio_bounds must contain 'phi' for deferred ratio validation")
        raw_phi = ratio_bounds["phi"]
        if not isinstance(raw_phi, (list, tuple)) or len(raw_phi) != 2:
            raise ValueError("ratio_bounds['phi'] must be a pair (lower, upper)")
        phi_L = float(raw_phi[0])
        phi_U = float(raw_phi[1])
        if not np.isfinite(phi_L) or not np.isfinite(phi_U):
            raise ValueError("ratio_bounds['phi'] entries must be finite")
        if phi_L < 0.0:
            raise ValueError("ratio_bounds['phi'][0] must be nonnegative")
        if phi_U <= phi_L:
            raise ValueError(
                "ratio_bounds['phi'][1] must be greater than ratio_bounds['phi'][0]"
            )
        if self.objective_mode == "ratio_piecewise_mccormick":
            if "num_pieces" not in ratio_bounds and "C_opt_breakpoints" not in ratio_bounds:
                raise ValueError(
                    "deferred ratio_piecewise_mccormick bounds must include "
                    "'num_pieces' or 'C_opt_breakpoints'"
                )
            if "num_pieces" in ratio_bounds:
                try:
                    num_pieces = int(ratio_bounds["num_pieces"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("ratio_bounds['num_pieces'] must be an integer") from exc
                if num_pieces < 2:
                    raise ValueError("ratio_bounds['num_pieces'] must be at least 2")
        return dict(ratio_bounds)

    def _validate_ratio_bounds(
        self,
        ratio_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return ratio_bounds

        if ratio_bounds is None:
            raise ValueError(
                f"ratio_bounds is required when objective_mode='{self.objective_mode}'"
            )
        if not isinstance(ratio_bounds, dict):
            raise ValueError("ratio_bounds must be a dictionary")
        missing = [key for key in ("phi", "C_opt") if key not in ratio_bounds]
        if missing:
            raise ValueError(
                "ratio_bounds must contain bounds for: " + ", ".join(missing)
            )

        def parse_bounds(key: str) -> tuple[float, float]:
            raw_bounds = ratio_bounds[key]
            if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
                raise ValueError(f"ratio_bounds['{key}'] must be a pair (lower, upper)")
            lower = float(raw_bounds[0])
            upper = float(raw_bounds[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"ratio_bounds['{key}'] entries must be finite")
            return lower, upper

        phi_L, phi_U = parse_bounds("phi")
        C_opt_L, C_opt_U = parse_bounds("C_opt")

        if C_opt_L <= 0.0:
            raise ValueError("ratio_bounds['C_opt'][0] must be strictly positive")
        if C_opt_U < C_opt_L:
            raise ValueError(
                "ratio_bounds['C_opt'][1] must be greater than or equal to "
                "ratio_bounds['C_opt'][0]"
            )
        if phi_L < 0.0:
            raise ValueError("ratio_bounds['phi'][0] must be nonnegative")
        if phi_U <= phi_L:
            raise ValueError(
                "ratio_bounds['phi'][1] must be greater than ratio_bounds['phi'][0]"
            )

        validated_bounds: dict[str, Any] = {
            "phi": (phi_L, phi_U),
            "C_opt": (C_opt_L, C_opt_U),
        }
        if self.objective_mode == "ratio_piecewise_mccormick":
            validated_bounds["C_opt_breakpoints"] = self._validate_ratio_breakpoints(
                ratio_bounds,
                C_opt_L,
                C_opt_U,
            )
            validated_bounds["num_pieces"] = (
                len(validated_bounds["C_opt_breakpoints"]) - 1
            )
        return validated_bounds

    def _validate_ratio_breakpoints(
        self,
        ratio_bounds: dict[str, Any],
        C_opt_L: float,
        C_opt_U: float,
    ) -> list[float]:
        tolerance = self.ratio_bounds_tolerance
        if "C_opt_breakpoints" in ratio_bounds:
            raw_breakpoints = ratio_bounds["C_opt_breakpoints"]
            if not isinstance(raw_breakpoints, (list, tuple)):
                raise ValueError("ratio_bounds['C_opt_breakpoints'] must be a list")
            breakpoints = [float(value) for value in raw_breakpoints]
            if len(breakpoints) < 3:
                raise ValueError(
                    "ratio_bounds['C_opt_breakpoints'] must contain at least 3 values"
                )
            if not all(np.isfinite(value) for value in breakpoints):
                raise ValueError("ratio_bounds['C_opt_breakpoints'] entries must be finite")
            if abs(breakpoints[0] - C_opt_L) > tolerance:
                raise ValueError(
                    "ratio_bounds['C_opt_breakpoints'][0] must match "
                    "ratio_bounds['C_opt'][0]"
                )
            if abs(breakpoints[-1] - C_opt_U) > tolerance:
                raise ValueError(
                    "ratio_bounds['C_opt_breakpoints'][-1] must match "
                    "ratio_bounds['C_opt'][1]"
                )
            if any(
                breakpoints[idx + 1] <= breakpoints[idx]
                for idx in range(len(breakpoints) - 1)
            ):
                raise ValueError(
                    "ratio_bounds['C_opt_breakpoints'] must be strictly increasing"
                )
            return breakpoints

        if "num_pieces" not in ratio_bounds:
            raise ValueError(
                "ratio_bounds must include either 'num_pieces' or "
                "'C_opt_breakpoints' for objective_mode='ratio_piecewise_mccormick'"
            )
        try:
            num_pieces = int(ratio_bounds["num_pieces"])
        except (TypeError, ValueError) as exc:
            raise ValueError("ratio_bounds['num_pieces'] must be an integer") from exc
        if num_pieces < 2:
            raise ValueError("ratio_bounds['num_pieces'] must be at least 2")
        return [
            float(value)
            for value in np.linspace(C_opt_L, C_opt_U, num_pieces + 1)
        ]

    def _ratio_bounds_with_loaded_C_opt_bounds(
        self,
        ratio_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return ratio_bounds
        if ratio_bounds is None:
            raise ValueError(
                f"ratio_bounds is required when objective_mode='{self.objective_mode}'"
            )
        if "C_opt" in ratio_bounds:
            return ratio_bounds

        C_opt_bounds = (self.optimal_cost_bounds or {}).get("C_opt", {})
        lower = C_opt_bounds.get("lower")
        upper = C_opt_bounds.get("upper")
        if lower is None or upper is None:
            raise ValueError(
                "Ratio objective modes require denominator bounds. Pass "
                "ratio_bounds['C_opt'] explicitly or run/load the DRO "
                "optimal-cost-bound tightening stage first."
            )
        completed = dict(ratio_bounds)
        completed["C_opt"] = (float(lower), float(upper))
        return completed

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

    @staticmethod
    def load_regime_config(
        regime_config_path: str | Path,
        regime_set: str,
        regime_name: Optional[str],
    ) -> dict[str, Any]:
        path = Path(regime_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Regime config not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle) or {}

        regime_sets = raw_config.get("regime_sets")
        if not isinstance(regime_sets, dict) or not regime_sets:
            raise ValueError("Regime config must contain a non-empty 'regime_sets' mapping")
        if regime_set not in regime_sets:
            available = ", ".join(str(key) for key in regime_sets)
            raise ValueError(f"Unknown regime_set '{regime_set}'. Available: {available}")

        selected_set = regime_sets[regime_set] or {}
        regimes = selected_set.get("regimes")
        if not isinstance(regimes, list) or not regimes:
            raise ValueError(f"regime_set '{regime_set}' has no non-empty 'regimes' list")

        if regime_name is None:
            selected = regimes[0]
        else:
            selected = next(
                (regime for regime in regimes if str(regime.get("name")) == str(regime_name)),
                None,
            )
            if selected is None:
                available = ", ".join(str(regime.get("name")) for regime in regimes)
                raise ValueError(
                    f"Unknown regime '{regime_name}' in regime_set '{regime_set}'. "
                    f"Available: {available}"
                )

        required_fields = ("name", "mu_D", "rho_D", "sigma_D", "mu_W", "rho_W", "sigma_W")
        missing = [field for field in required_fields if field not in selected]
        if missing:
            raise ValueError(
                f"Regime '{selected.get('name', regime_name)}' is missing fields: {missing}"
            )
        peak_key = "peak_W" if "peak_W" in selected else "tau_W"
        if peak_key not in selected:
            raise ValueError(
                f"Regime '{selected.get('name', regime_name)}' must include peak_W or tau_W"
            )

        selected = dict(selected)
        selected["peak_W"] = float(selected[peak_key])
        for field in ("mu_D", "rho_D", "sigma_D", "mu_W", "rho_W", "sigma_W"):
            selected[field] = float(selected[field])
        selected["name"] = str(selected["name"])
        return selected

    @staticmethod
    def load_regime_scenarios(
        reference_case: str = "test_case_bidding_blocks",
        regime_config_path: str | Path = "config/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        seed: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scenario_manager = ScenarioManager(reference_case)
        scenario_set = scenario_manager.create_scenario_set_from_regimes(
            regime_config_path=str(regime_config_path),
            regime_set=regime_set,
            seed=seed,
        )
        return (
            scenario_set["scenarios_df"],
            scenario_set["costs_df"],
            scenario_set["ramps_df"],
        )

    def _filter_scenarios_to_regime(
        self,
        scenarios_df: pd.DataFrame,
        regime_name: str,
        regime_name_was_explicit: bool,
    ) -> pd.DataFrame:
        if scenarios_df.empty:
            raise ValueError("scenarios_df must contain at least one empirical scenario")
        if "regime" not in scenarios_df.columns:
            if regime_name_was_explicit:
                raise ValueError(
                    "scenarios_df does not contain a 'regime' column, so it cannot "
                    f"be filtered to selected regime '{regime_name}'"
                )
            return scenarios_df.reset_index(drop=True).copy()

        filtered = scenarios_df[
            scenarios_df["regime"].astype(str) == str(regime_name)
        ].reset_index(drop=True)
        if filtered.empty:
            available = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())
            raise ValueError(
                f"scenarios_df contains no rows for selected regime '{regime_name}'. "
                f"Available regimes: {available}"
            )
        return filtered.copy()

    @staticmethod
    def _scenario_id_from_row(row: pd.Series, fallback: int) -> Any:
        return row["scenario_id"] if "scenario_id" in row.index else int(fallback)

    def _configure_fixed_regime_parameters(self) -> None:
        self.mu_D_fixed = float(self.selected_regime["mu_D"])
        self.sigma_D_fixed = float(self.selected_regime["sigma_D"])
        self.demand_rho_fixed = float(self.selected_regime["rho_D"])
        self.mu_W_fixed = float(self.selected_regime["mu_W"])
        self.sigma_W_fixed = float(self.selected_regime["sigma_W"])
        self.wind_rho_fixed = float(self.selected_regime["rho_W"])
        self.peak_W_fixed = float(self.selected_regime["peak_W"])

        if self.sigma_D_fixed < 0.0 or self.sigma_W_fixed < 0.0:
            raise ValueError("Selected regime sigma_D and sigma_W must be nonnegative")
        if not -0.999 <= self.demand_rho_fixed <= 0.999:
            raise ValueError("Selected regime rho_D must be in [-0.999, 0.999]")
        if not -0.999 <= self.wind_rho_fixed <= 0.999:
            raise ValueError("Selected regime rho_W must be in [-0.999, 0.999]")
        if not 0.0 <= self.peak_W_fixed <= 24.0:
            raise ValueError("Selected regime peak_W must be in [0, 24]")

    def _configure_regime_shape_profiles(self) -> None:
        scenario_manager = ScenarioManager(self.reference_case)
        self.demand_D_ref = float(scenario_manager.base_case["demand"])
        if self.demand_D_ref <= 0.0 or not np.isfinite(self.demand_D_ref):
            raise ValueError("Reference-case scalar demand must be positive and finite")

        reference_horizon = int(scenario_manager.base_case["time_steps"])
        raw_demand_shape = scenario_manager._build_demand_shape(reference_horizon)
        raw_wind_shape = scenario_manager._build_wind_shape(
            reference_horizon,
            self.peak_W_fixed,
        )
        self.demand_shape = ensure_profile(
            raw_demand_shape,
            self.num_time_steps,
            "demand_shape",
            allow_truncate=True,
        )
        self.wind_shape = ensure_profile(
            raw_wind_shape,
            self.num_time_steps,
            "wind_shape",
            allow_truncate=True,
        )
        self.demand_delta_shape = {
            t: abs(self.demand_shape[t] - self.demand_shape[t - 1])
            for t in range(1, self.num_time_steps)
        }
        self.wind_delta_shape = {
            t: abs(self.wind_shape[t] - self.wind_shape[t - 1])
            for t in range(1, self.num_time_steps)
        }

    def _normalize_p_init(
        self,
        p_init: Optional[list[float] | list[list[float]]],
    ) -> list[list[float]]:
        default_row = [0.5 * cap for cap in self.static_physical_capacity]
        if p_init is None:
            return [list(default_row) for _ in range(self.num_empirical_scenarios)]

        values: Any = p_init
        if values and isinstance(values[0], (list, tuple, np.ndarray, pd.Series)):
            if len(values) == self.num_empirical_scenarios:
                return [self._normalize_p_init_row(row) for row in values]
            if len(values) == 1:
                row = self._normalize_p_init_row(values[0])
                return [list(row) for _ in range(self.num_empirical_scenarios)]
            raise ValueError(
                f"p_init has {len(values)} rows; expected 1 or "
                f"{self.num_empirical_scenarios} empirical-scenario rows"
            )

        row = self._normalize_p_init_row(values)
        return [list(row) for _ in range(self.num_empirical_scenarios)]

    def _normalize_p_init_row(self, values: Any) -> list[float]:
        row = [float(v) for v in values]
        if len(row) == self.num_physical_generators:
            return row
        if len(row) == self.num_blocks:
            return [
                sum(row[g] for g in self.physical_to_block_indices[i])
                for i in range(self.num_physical_generators)
            ]
        raise ValueError(
            f"p_init row has {len(row)} values; expected "
            f"{self.num_physical_generators} physical-generator values or "
            f"{self.num_blocks} block values"
        )

    def _initialize_big_m_placeholders(self) -> None:
        cap_indices = [
            (int(i), int(b), int(t))
            for i, b in self.generator_block_pairs
            for t in range(self.num_time_steps)
        ]
        ramp_indices = [
            (int(i), int(t))
            for i in range(self.num_physical_generators)
            for t in range(self.num_time_steps)
        ]
        self.M_cap = {index: self.primal_big_m_placeholder for index in cap_indices}
        self.M_lower = {index: self.primal_big_m_placeholder for index in cap_indices}
        self.M_ramp_up = {index: self.primal_big_m_placeholder for index in ramp_indices}
        self.M_ramp_down = {index: self.primal_big_m_placeholder for index in ramp_indices}
        self.M_ramp_up_initial = {
            int(i): self.primal_big_m_placeholder
            for i in range(self.num_physical_generators)
        }
        self.M_ramp_down_initial = {
            int(i): self.primal_big_m_placeholder
            for i in range(self.num_physical_generators)
        }
        self.M_mu_upper_eq = {index: self.capacity_dual_bound for index in cap_indices}
        self.M_mu_lower_eq = {index: self.capacity_dual_bound for index in cap_indices}
        self.M_mu_upper_opt = {index: self.capacity_dual_bound for index in cap_indices}
        self.M_mu_lower_opt = {index: self.capacity_dual_bound for index in cap_indices}
        self.M_mu_ramp_up_eq = {index: self.ramp_dual_bound for index in ramp_indices}
        self.M_mu_ramp_down_eq = {index: self.ramp_dual_bound for index in ramp_indices}
        self.M_mu_ramp_up_opt = {index: self.ramp_dual_bound for index in ramp_indices}
        self.M_mu_ramp_down_opt = {index: self.ramp_dual_bound for index in ramp_indices}
        self.lambda_eq_bounds = {
            int(t): (-self.lambda_bound, self.lambda_bound)
            for t in range(self.num_time_steps)
        }
        self.lambda_opt_bounds = dict(self.lambda_eq_bounds)

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

            feature_columns = list(
                weights.get("feature_columns") or metadata.get("feature_columns") or []
            )
            target_columns = list(
                weights.get("target_columns") or metadata.get("target_columns") or []
            )
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
        self,
        generator_name: str,
        feature_columns: list[str],
        target_columns: list[str],
        layers: list[dict[str, Any]],
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
                    raise ValueError(
                        f"{generator_name}: ReLU layer {idx} must follow a hidden linear layer"
                    )
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

    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in str(key).split(",") if part != "")
        except ValueError as exc:
            raise ValueError(f"Malformed tightening-report key '{key}'") from exc

    @staticmethod
    def _optional_numeric_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("tight_big_m", "big_m", "upper_bound", "ub", "bound", "value"):
                if value_key in payload:
                    return DRO_PoAOptimization._optional_numeric_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return max(0.0, numeric_value)

    @staticmethod
    def _optional_float_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("bound", "value", "tight_bound", "tight_big_m"):
                if value_key in payload:
                    return DRO_PoAOptimization._optional_float_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    def _validate_regime_wide_tightening_metadata(self, report: dict[str, Any]) -> None:
        metadata = report.get("metadata", {}) or {}
        if metadata.get("tightening_type") != "regime_wide":
            raise ValueError("DRO tightening report metadata must set tightening_type='regime_wide'")
        if metadata.get("tightening_scope") not in {None, "regime_wide", "scenario_wise"}:
            raise ValueError(
                "DRO tightening report metadata tightening_scope must be "
                "'scenario_wise' or 'regime_wide'"
            )
        if metadata.get("model_type") not in {None, "DRO_PoA"}:
            raise ValueError("DRO tightening report metadata model_type must be 'DRO_PoA'")
        if str(metadata.get("regime_name")) != str(self.regime_name):
            raise ValueError(
                "DRO tightening report regime_name mismatch: "
                f"report has {metadata.get('regime_name')!r}, optimizer has {self.regime_name!r}"
            )
        if int(metadata.get("num_time_steps", self.num_time_steps)) != int(self.num_time_steps):
            raise ValueError(
                "DRO tightening report num_time_steps mismatch: "
                f"report has {metadata.get('num_time_steps')}, optimizer has {self.num_time_steps}"
            )

    @staticmethod
    def _parse_nn_relu_bound_key(key: str) -> tuple[int, ...]:
        parts = tuple(int(part) for part in str(key).split(",") if part != "")
        if len(parts) not in {3, 4}:
            raise ValueError(
                f"NN ReLU bound key '{key}' must have the form "
                "'time_idx,linear_idx,node' or 'k,time_idx,linear_idx,node'"
            )
        return tuple(int(part) for part in parts)

    def _set_nn_relu_bounds_from_report(self, report: dict[str, Any]) -> None:
        self.nn_relu_bounds_report = report
        self.nn_feature_bounds = report.get("nn_feature_bounds", {}) or {}
        self.nn_bound_warnings = list(
            report.get("warnings", report.get("nn_bound_warnings", [])) or []
        )
        relu_entries = report.get("scenario_nn_relu_bounds", {}) or report.get(
            "nn_relu_bounds",
            {},
        )
        parsed_bounds: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {}
        for generator_name, entries in (relu_entries or {}).items():
            generator_bounds: dict[tuple[int, ...], dict[str, Any]] = {}
            for key, details in (entries or {}).items():
                parsed_key = self._parse_nn_relu_bound_key(key)
                time_idx, linear_idx, node = parsed_key[-3:]
                parsed_details = dict(details or {})
                for numeric_key in ("L", "U", "h_lower", "h_upper"):
                    if numeric_key not in parsed_details:
                        raise ValueError(
                            f"NN ReLU bound entry {generator_name}[{key}] is missing "
                            f"'{numeric_key}'"
                        )
                    parsed_details[numeric_key] = float(parsed_details[numeric_key])
                status = str(parsed_details.get("status", "ambiguous")).lower()
                if status not in {"active", "inactive", "ambiguous"}:
                    raise ValueError(
                        f"NN ReLU bound entry {generator_name}[{key}] has invalid "
                        f"status '{status}'"
                    )
                parsed_details["status"] = status
                if len(parsed_key) == 4:
                    parsed_details.setdefault("scenario_idx", parsed_key[0])
                parsed_details.setdefault("time_idx", time_idx)
                parsed_details.setdefault("linear_idx", linear_idx)
                parsed_details.setdefault("node", node)
                generator_bounds[parsed_key] = parsed_details
            parsed_bounds[str(generator_name)] = generator_bounds
        self.nn_relu_bounds = parsed_bounds

    def load_nn_relu_bounds_report(
        self,
        report_path: str | Path = "results/dro_poa_tightening/final_tightening_report.json",
    ) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"NN ReLU bounds report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        relu_report = report.get("nn_relu_bounds_report", {}) or report
        self.nn_relu_bounds_report_path = path
        self._set_nn_relu_bounds_from_report(relu_report)
        return relu_report

    def _set_regime_wide_tightening_report_data(
        self,
        report: dict[str, Any],
        report_path: Optional[Path] = None,
    ) -> None:
        self._validate_regime_wide_tightening_metadata(report)
        self.tightening_report = report
        if report_path is not None:
            self.tightening_report_path = Path(report_path)
        relu_report = report.get("nn_relu_bounds_report", {}) or {}
        if not relu_report and (
            "nn_relu_bounds" in report or "scenario_nn_relu_bounds" in report
        ):
            relu_report = report
        if relu_report:
            self._set_nn_relu_bounds_from_report(relu_report)
        self.fixed_binaries = report.get("scenario_fixed_binaries", {}) or report.get(
            "fixed_binaries",
            {},
        ) or {}
        self.primal_big_m = report.get("primal_big_m", {}) or {}
        self.tight_big_m = report.get("scenario_tight_big_m", {}) or report.get(
            "tight_big_m",
            {},
        ) or {}
        self.aggregate_dual_bounds = report.get("aggregate_dual_bounds", {}) or {}
        self.lambda_bounds = report.get("scenario_lambda_bounds", {}) or report.get(
            "lambda_bounds",
            {},
        ) or {}
        self.alpha_bound_optimization_results = (
            report.get("alpha_optimization_results", {}) or {}
        )
        self.alpha_bounds = {
            self._parse_json_index(key): {
                "lower": float(value["lower"]),
                "upper": float(value["upper"]),
            }
            for key, value in (
                report.get("scenario_alpha_bounds", {}) or report.get("alpha_bounds", {}) or {}
            ).items()
        }
        if (
            "optimal_cost_bounds" in report
            or "scenario_optimal_cost_bounds" in report
            or "optimal_cost_bound_optimization_results" in report
        ):
            self._set_optimal_cost_bounds_from_report(report)
        self._loaded_bounds_prepared = False

    def load_regime_wide_tightening_report(self, report_path: str | Path) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"DRO regime-wide tightening report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        self._set_regime_wide_tightening_report_data(report, path)
        self._prepare_loaded_bounds()
        return report

    def _set_optimal_cost_bounds_from_report(self, report: dict[str, Any]) -> None:
        raw_bounds = report.get("optimal_cost_bounds", {}) or {}
        if "C_opt" in raw_bounds and isinstance(raw_bounds.get("C_opt"), dict):
            C_opt_payload = raw_bounds.get("C_opt", {}) or {}
        else:
            C_opt_payload = raw_bounds

        if C_opt_payload:
            lower = C_opt_payload.get("lower")
            upper = C_opt_payload.get("upper")
            if lower is None or upper is None:
                raise ValueError(
                    "optimal_cost_bounds must contain finite 'lower' and 'upper' entries"
                )
            self.optimal_cost_bounds = {
                "C_opt": {
                    "lower": float(lower),
                    "upper": float(upper),
                    "raw_lower": (
                        float(C_opt_payload["raw_lower"])
                        if C_opt_payload.get("raw_lower") is not None
                        else None
                    ),
                    "raw_upper": (
                        float(C_opt_payload["raw_upper"])
                        if C_opt_payload.get("raw_upper") is not None
                        else None
                    ),
                }
            }

        self.scenario_optimal_cost_bounds = (
            report.get("scenario_optimal_cost_bounds", {}) or {}
        )
        self.optimal_cost_bound_optimization_results = (
            report.get("optimal_cost_bound_optimization_results", {}) or {}
        )

    def load_optimal_cost_bounds_report(self, report_path: str | Path) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"DRO optimal-cost bounds report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        self._set_optimal_cost_bounds_from_report(report)
        return report

    def _indexed_numeric_entries(
        self,
        entries: Any,
        bound_type: str,
        allowed_dimensions: tuple[int, ...],
        expected_key_format: str,
    ) -> dict[tuple[int, ...], float]:
        if not isinstance(entries, dict):
            return {}
        parsed: dict[tuple[int, ...], float] = {}
        for raw_key, details in entries.items():
            index = self._parse_json_index(str(raw_key))
            if len(index) not in allowed_dimensions:
                scenario_note = (
                    "including scenario index k"
                    if "k," in expected_key_format
                    else "without scenario index k"
                )
                raise ValueError(
                    f"Invalid {bound_type} key '{raw_key}'. Expected key format "
                    f"{expected_key_format} {scenario_note}."
                )
            numeric_value = self._optional_numeric_bound(details)
            if numeric_value is None:
                raise ValueError(
                    f"Invalid {bound_type} entry at key '{raw_key}'. Expected a "
                    "finite numeric bound in tight_big_m, big_m, upper_bound, ub, "
                    "bound, or value."
                )
            parsed[index] = float(numeric_value)
        return parsed

    def _missing_loaded_bound_error(
        self,
        bound_type: str,
        index: tuple[int, ...] | int,
        expected_key_format: str,
    ) -> ValueError:
        scenario_note = (
            "including scenario index k"
            if "k," in expected_key_format
            else "without scenario index k"
        )
        return ValueError(
            f"Missing {bound_type} for index {index}. Expected key format "
            f"{expected_key_format} {scenario_note}. If this is a reused DRO "
            "tightening report, rerun the corresponding stage for the current "
            "num_empirical_scenarios."
        )

    @staticmethod
    def _scenario_or_regime_value(
        bound_map: dict[tuple[int, ...], float],
        scenario_idx: int,
        regime_index: tuple[int, ...],
    ) -> float:
        scenario_key = (int(scenario_idx), *tuple(int(part) for part in regime_index))
        regime_key = tuple(int(part) for part in regime_index)
        if scenario_key in bound_map:
            return float(bound_map[scenario_key])
        return float(bound_map[regime_key])

    @staticmethod
    def _scenario_or_regime_lambda_bounds(
        bound_map: dict[Any, tuple[float, float]],
        scenario_idx: int,
        time_idx: int,
    ) -> tuple[float, float]:
        scenario_key = (int(scenario_idx), int(time_idx))
        if scenario_key in bound_map:
            return bound_map[scenario_key]
        return bound_map[int(time_idx)]

    def _prepare_block_time_primal_big_m(
        self,
        component_name: str,
        entries: Any,
        expected_key_format: str,
    ) -> dict[tuple[int, int, int], float]:
        parsed = self._indexed_numeric_entries(
            entries,
            f"primal Big-M '{component_name}'",
            (2, 3),
            expected_key_format,
        )
        prepared: dict[tuple[int, int, int], float] = {}
        for i, b in self.generator_block_pairs:
            for t in range(self.num_time_steps):
                time_index = (int(i), int(b), int(t))
                block_index = (int(i), int(b))
                if time_index in parsed:
                    prepared[time_index] = parsed[time_index]
                elif block_index in parsed:
                    prepared[time_index] = parsed[block_index]
                else:
                    raise self._missing_loaded_bound_error(
                        f"primal Big-M '{component_name}'",
                        time_index,
                        expected_key_format,
                    )
        return prepared

    def _prepare_generator_time_primal_big_m(
        self,
        component_name: str,
        entries: Any,
        expected_key_format: str,
    ) -> dict[tuple[int, int], float]:
        parsed = self._indexed_numeric_entries(
            entries,
            f"primal Big-M '{component_name}'",
            (1, 2),
            expected_key_format,
        )
        prepared: dict[tuple[int, int], float] = {}
        for i in range(self.num_physical_generators):
            for t in range(self.num_time_steps):
                time_index = (int(i), int(t))
                generator_index = (int(i),)
                if time_index in parsed:
                    prepared[time_index] = parsed[time_index]
                elif generator_index in parsed:
                    prepared[time_index] = parsed[generator_index]
                else:
                    raise self._missing_loaded_bound_error(
                        f"primal Big-M '{component_name}'",
                        time_index,
                        expected_key_format,
                    )
        return prepared

    def _prepare_generator_primal_big_m(
        self,
        component_name: str,
        entries: Any,
        expected_key_format: str,
    ) -> dict[int, float]:
        parsed = self._indexed_numeric_entries(
            entries,
            f"primal Big-M '{component_name}'",
            (1, 2),
            expected_key_format,
        )
        prepared: dict[int, float] = {}
        for i in range(self.num_physical_generators):
            generator_index = (int(i),)
            initial_time_index = (int(i), 0)
            if generator_index in parsed:
                prepared[int(i)] = parsed[generator_index]
            elif initial_time_index in parsed:
                prepared[int(i)] = parsed[initial_time_index]
            else:
                raise self._missing_loaded_bound_error(
                    f"primal Big-M '{component_name}'",
                    int(i),
                    expected_key_format,
                )
        return prepared

    def _prepare_dual_big_m(
        self,
        dual_name: str,
        expected_indices: list[tuple[int, ...]],
        default_bound: float,
        expected_key_format: str,
    ) -> dict[tuple[int, ...], float]:
        entries = (getattr(self, "tight_big_m", {}) or {}).get(dual_name, {}) or {}
        if not entries:
            return {index: float(default_bound) for index in expected_indices}
        parsed = self._indexed_numeric_entries(
            entries,
            f"dual Big-M '{dual_name}'",
            (len(expected_indices[0]), len(expected_indices[0]) + 1),
            f"{expected_key_format} or k,{expected_key_format}",
        )
        prepared: dict[tuple[int, ...], float] = {}
        has_scenario_keys = any(len(index) == len(expected_indices[0]) + 1 for index in parsed)
        for regime_index in expected_indices:
            if has_scenario_keys:
                for k in range(self.num_empirical_scenarios):
                    scenario_index = (int(k), *tuple(regime_index))
                    if scenario_index not in parsed:
                        raise self._missing_loaded_bound_error(
                            f"scenario-wise dual Big-M '{dual_name}'",
                            scenario_index,
                            f"k,{expected_key_format}",
                        )
                    prepared[scenario_index] = max(
                        0.0,
                        min(float(default_bound), float(parsed[scenario_index])),
                    )
                continue
            if regime_index not in parsed:
                raise self._missing_loaded_bound_error(
                    f"dual Big-M '{dual_name}'",
                    regime_index,
                    expected_key_format,
                )
            prepared[regime_index] = max(
                0.0,
                min(float(default_bound), float(parsed[regime_index])),
            )
        return prepared

    def _prepare_lambda_bounds(self, lambda_name: str) -> dict[Any, tuple[float, float]]:
        entries = (getattr(self, "lambda_bounds", {}) or {}).get(lambda_name, {}) or {}
        if not entries:
            return {
                int(t): (-float(self.lambda_bound), float(self.lambda_bound))
                for t in range(self.num_time_steps)
            }
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid lambda bound block '{lambda_name}'. Expected keys 't' or 'k,t'.")
        parsed_entries: dict[tuple[int, ...], Any] = {
            self._parse_json_index(str(key)): details
            for key, details in entries.items()
        }
        has_scenario_keys = any(len(index) == 2 for index in parsed_entries)
        prepared: dict[Any, tuple[float, float]] = {}
        for t in range(self.num_time_steps):
            keys = (
                [(int(k), int(t)) for k in range(self.num_empirical_scenarios)]
                if has_scenario_keys
                else [(int(t),)]
            )
            for key in keys:
                details = parsed_entries.get(key)
                if not isinstance(details, dict):
                    raise self._missing_loaded_bound_error(
                        f"lambda bounds '{lambda_name}'",
                        key,
                        "t or k,t",
                    )
                lower = self._optional_float_bound(details.get("lower"))
                upper = self._optional_float_bound(details.get("upper"))
                if lower is None or upper is None:
                    raise ValueError(
                        f"Invalid lambda bounds '{lambda_name}' for index {key}. "
                        "Expected finite lower and upper entries."
                    )
                lower = max(-float(self.lambda_bound), float(lower))
                upper = min(float(self.lambda_bound), float(upper))
                if lower > upper:
                    raise ValueError(
                        f"Invalid lambda bounds '{lambda_name}' for index {key}: "
                        f"lower {lower} exceeds upper {upper}."
                    )
                prepared[key if len(key) == 2 else int(t)] = (lower, upper)
        return prepared

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

    def _prepare_loaded_bounds(self) -> None:
        if not getattr(self, "tightening_report", None) and not getattr(
            self,
            "primal_big_m",
            None,
        ):
            return
        if not getattr(self, "primal_big_m", None):
            raise ValueError(
                "Missing primal_big_m in DRO regime-wide tightening report."
            )

        primal_big_m = self.primal_big_m
        self.M_cap = self._prepare_block_time_primal_big_m(
            "block_capacity",
            primal_big_m.get("block_capacity", {}),
            "i,b or i,b,t",
        )
        lower_entries = (
            primal_big_m.get("lower_generation", {})
            or primal_big_m.get("generation_lower", {})
            or primal_big_m.get("lower_bound", {})
        )
        self.M_lower = (
            self._prepare_block_time_primal_big_m(
                "lower_generation",
                lower_entries,
                "i,b or i,b,t",
            )
            if lower_entries
            else dict(self.M_cap)
        )
        self.M_physical_capacity = self._prepare_generator_primal_big_m(
            "physical_capacity",
            primal_big_m.get("physical_capacity", {}),
            "i or i,t",
        )
        self.M_ramp_up = self._prepare_generator_time_primal_big_m(
            "ramp_up",
            primal_big_m.get("ramp_up", {}),
            "i or i,t",
        )
        self.M_ramp_down = self._prepare_generator_time_primal_big_m(
            "ramp_down",
            primal_big_m.get("ramp_down", {}),
            "i or i,t",
        )
        self.M_ramp_up_initial = self._prepare_generator_primal_big_m(
            "ramp_up_initial",
            primal_big_m.get("ramp_up_initial", {}),
            "i",
        )
        self.M_ramp_down_initial = self._prepare_generator_primal_big_m(
            "ramp_down_initial",
            primal_big_m.get("ramp_down_initial", {}),
            "i",
        )

        capacity_indices = [
            (int(i), int(b), int(t))
            for i, b in self.generator_block_pairs
            for t in range(self.num_time_steps)
        ]
        ramp_indices = [
            (int(i), int(t))
            for i in range(self.num_physical_generators)
            for t in range(self.num_time_steps)
        ]
        self.M_mu_upper_eq = self._prepare_dual_big_m(
            "mu_upper_eq",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_lower_eq = self._prepare_dual_big_m(
            "mu_lower_eq",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_ramp_up_eq = self._prepare_dual_big_m(
            "mu_ramp_up_eq",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.M_mu_ramp_down_eq = self._prepare_dual_big_m(
            "mu_ramp_down_eq",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.M_mu_upper_opt = self._prepare_dual_big_m(
            "mu_upper_opt",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_lower_opt = self._prepare_dual_big_m(
            "mu_lower_opt",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_ramp_up_opt = self._prepare_dual_big_m(
            "mu_ramp_up_opt",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.M_mu_ramp_down_opt = self._prepare_dual_big_m(
            "mu_ramp_down_opt",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.lambda_eq_bounds = self._prepare_lambda_bounds("lambda_eq")
        self.lambda_opt_bounds = self._prepare_lambda_bounds("lambda_opt")
        self._loaded_bounds_prepared = True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _parse_empirical_demand_profiles(self) -> list[list[float]]:
        demand_column = find_demand_profile_column(self.scenarios_df)
        profiles: list[list[float]] = []
        for row_idx, row in self.scenarios_df.iterrows():
            try:
                profiles.append(
                    ensure_profile(
                        row[demand_column],
                        self.num_time_steps,
                        demand_column,
                        allow_truncate=True,
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not parse empirical demand profile for row {row_idx}"
                ) from exc
        return profiles

    def _block_capacity_profile_from_row(self, row: pd.Series, block_name: str) -> list[float]:
        for column in (f"{block_name}_cap_profile", f"{block_name}_profile"):
            if column in row.index:
                return ensure_profile(
                    row[column],
                    self.num_time_steps,
                    column,
                    allow_truncate=True,
                )
        cap_column = f"{block_name}_cap"
        if cap_column in row.index:
            return ensure_profile(row[cap_column], self.num_time_steps, cap_column)

        global_block = self.block_names.index(block_name)
        return [float(self.static_block_capacity[global_block])] * self.num_time_steps

    def _parse_empirical_block_capacity_profiles(self) -> list[list[list[float]]]:
        profiles: list[list[list[float]]] = []
        for row_idx, row in self.scenarios_df.iterrows():
            scenario_profiles = []
            for block_name in self.block_names:
                try:
                    scenario_profiles.append(
                        self._block_capacity_profile_from_row(row, block_name)
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Could not parse empirical capacity profile for block "
                        f"'{block_name}' in row {row_idx}"
                    ) from exc
            profiles.append(scenario_profiles)
        return profiles

    def _build_empirical_physical_capacity_profiles(self) -> list[list[list[float]]]:
        physical_profiles: list[list[list[float]]] = []
        for k in range(self.num_empirical_scenarios):
            by_generator = []
            for i in range(self.num_physical_generators):
                by_generator.append(
                    [
                        sum(
                            self.empirical_Pmax_block[k][global_block][t]
                            for global_block in self.physical_to_block_indices[i]
                        )
                        for t in range(self.num_time_steps)
                    ]
                )
            physical_profiles.append(by_generator)
        return physical_profiles

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        if getattr(self, "tightening_report", None):
            self._prepare_loaded_bounds()
        self.model = ConcreteModel()

        self.model.scenarios = Set(initialize=range(self.num_empirical_scenarios))
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))
        self.model.physical_generators = Set(initialize=range(self.num_physical_generators))
        self.model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)
        self.model.wind_physical_generators = Set(initialize=self.wind_physical_generator_ids)
        self.model.conventional_physical_generators = Set(
            initialize=self.conventional_physical_generator_ids
        )
        self.model.wind_blocks = Set(dimen=2, initialize=self.wind_block_pairs)
        self.model.conventional_blocks = Set(dimen=2, initialize=self.conventional_block_pairs)

        if self.objective_mode in {"ratio_mccormick", "ratio_piecewise_mccormick"}:
            ratio_bounds = self.ratio_bounds or self._raw_ratio_bounds
            completed_bounds = self._ratio_bounds_with_loaded_C_opt_bounds(ratio_bounds)
            self.ratio_bounds = self._validate_ratio_bounds(completed_bounds)

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
        self._build_regime_variables()
        m = self.model
        m.D = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_block = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.D_transport_abs_deviation = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_phys_transport_abs_deviation = Var(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            domain=NonNegativeReals,
        )
        m.wasserstein_distance = Var(m.scenarios, domain=NonNegativeReals)
        m.C_eq = Var(m.scenarios, domain=Reals)
        c_opt_bounds = (
            self.ratio_bounds["C_opt"]
            if self.objective_mode in {
                "ratio_mccormick",
                "ratio_piecewise_mccormick",
            } and self.ratio_bounds is not None
            else (None, None)
        )
        m.C_opt = Var(m.scenarios, domain=Reals, bounds=c_opt_bounds)
        m.PoA = Var(m.scenarios, domain=Reals)
        if self.objective_mode in {"ratio_mccormick", "ratio_piecewise_mccormick"}:
            self._build_ratio_mccormick_variables()
        if self.objective_mode == "ratio_piecewise_mccormick":
            self._build_ratio_piecewise_mccormick_variables()

    def _build_ratio_mccormick_variables(self) -> None:
        if self.ratio_bounds is None:
            raise ValueError(
                f"ratio_bounds is required when objective_mode='{self.objective_mode}'"
            )
        m = self.model
        phi_L, phi_U = self.ratio_bounds["phi"]
        C_opt_L, C_opt_U = self.ratio_bounds["C_opt"]
        m.phi = Var(m.scenarios, bounds=(phi_L, phi_U))
        m.z_ratio_product = Var(
            m.scenarios,
            domain=Reals,
            bounds=(phi_L * C_opt_L, phi_U * C_opt_U),
        )

    def _build_ratio_piecewise_mccormick_variables(self) -> None:
        if self.ratio_bounds is None:
            raise ValueError(
                "ratio_bounds is required when "
                "objective_mode='ratio_piecewise_mccormick'"
        )
        m = self.model
        breakpoints = list(self.ratio_bounds["C_opt_breakpoints"])
        _phi_L, phi_U = self.ratio_bounds["phi"]
        m.ratio_piece_index = Set(initialize=range(len(breakpoints) - 1))
        m.ratio_piece_active = Var(m.scenarios, m.ratio_piece_index, domain=Binary)
        m.C_opt_piece = Var(
            m.scenarios,
            m.ratio_piece_index,
            domain=NonNegativeReals,
            bounds=lambda m, k, p: (0.0, breakpoints[int(p) + 1]),
        )
        m.phi_piece = Var(
            m.scenarios,
            m.ratio_piece_index,
            domain=NonNegativeReals,
            bounds=(0.0, phi_U),
        )
        m.z_ratio_piece = Var(
            m.scenarios,
            m.ratio_piece_index,
            domain=NonNegativeReals,
            bounds=lambda m, k, p: (0.0, phi_U * breakpoints[int(p) + 1]),
        )

    def _build_regime_variables(self) -> None:
        m = self.model
        m.mu_D = Var(bounds=(self.mu_D_fixed, self.mu_D_fixed))
        m.sigma_D = Var(bounds=(self.sigma_D_fixed, self.sigma_D_fixed))
        m.mu_W = Var(bounds=(self.mu_W_fixed, self.mu_W_fixed))
        m.sigma_W = Var(bounds=(self.sigma_W_fixed, self.sigma_W_fixed))
        m.rho_D = Var(bounds=(self.demand_rho_fixed, self.demand_rho_fixed))
        m.rho_W = Var(bounds=(self.wind_rho_fixed, self.wind_rho_fixed))
        m.peak_W = Var(bounds=(self.peak_W_fixed, self.peak_W_fixed))

    def _build_equilibrium_variables(self) -> None:
        m = self.model
        m.P_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.alpha = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Reals)
        m.lambda_eq = Var(
            m.scenarios,
            m.time_steps,
            domain=Reals,
            bounds=lambda m, k, t: self._scenario_or_regime_lambda_bounds(
                self.lambda_eq_bounds,
                int(k),
                int(t),
            ),
        )
        m.mu_upper_eq = Var(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            domain=Reals,
            bounds=lambda m, k, i, b, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_upper_eq,
                    int(k),
                    (int(i), int(b), int(t)),
                ),
            ),
        )
        m.mu_lower_eq = Var(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            domain=Reals,
            bounds=lambda m, k, i, b, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_lower_eq,
                    int(k),
                    (int(i), int(b), int(t)),
                ),
            ),
        )
        m.mu_ramp_up_eq = Var(
            m.scenarios,
            m.physical_generators,
            m.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, k, i, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_ramp_up_eq,
                    int(k),
                    (int(i), int(t)),
                )
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )
        m.mu_ramp_down_eq = Var(
            m.scenarios,
            m.physical_generators,
            m.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, k, i, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_ramp_down_eq,
                    int(k),
                    (int(i), int(t)),
                )
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )

    def _build_complementarity_equilibrium_variables(self) -> None:
        m = self.model
        m.z_upper_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_lower_eq = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_ramp_up_eq = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)
        m.z_ramp_down_eq = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        m = self.model
        m.P_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=NonNegativeReals)
        m.lambda_opt = Var(
            m.scenarios,
            m.time_steps,
            domain=Reals,
            bounds=lambda m, k, t: self._scenario_or_regime_lambda_bounds(
                self.lambda_opt_bounds,
                int(k),
                int(t),
            ),
        )
        m.mu_upper_opt = Var(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            domain=Reals,
            bounds=lambda m, k, i, b, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_upper_opt,
                    int(k),
                    (int(i), int(b), int(t)),
                ),
            ),
        )
        m.mu_lower_opt = Var(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            domain=Reals,
            bounds=lambda m, k, i, b, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_lower_opt,
                    int(k),
                    (int(i), int(b), int(t)),
                ),
            ),
        )
        m.mu_ramp_up_opt = Var(
            m.scenarios,
            m.physical_generators,
            m.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, k, i, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_ramp_up_opt,
                    int(k),
                    (int(i), int(t)),
                )
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )
        m.mu_ramp_down_opt = Var(
            m.scenarios,
            m.physical_generators,
            m.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, k, i, t: (
                0.0,
                self._scenario_or_regime_value(
                    self.M_mu_ramp_down_opt,
                    int(k),
                    (int(i), int(t)),
                )
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )

    def _build_complementarity_optimal_variables(self) -> None:
        m = self.model
        m.z_upper_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_lower_opt = Var(m.scenarios, m.generator_blocks, m.time_steps, domain=Binary)
        m.z_ramp_up_opt = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)
        m.z_ramp_down_opt = Var(m.scenarios, m.physical_generators, m.time_steps, domain=Binary)

    def _build_objective(self) -> None:
        if self.objective_mode == "difference":
            self._build_difference_objective()
        elif self.objective_mode == "ratio_mccormick":
            self._build_ratio_mccormick_objective()
        elif self.objective_mode == "ratio_piecewise_mccormick":
            self._build_ratio_piecewise_mccormick_objective()
        else:
            raise ValueError(f"Unsupported objective_mode: {self.objective_mode}")

    def _build_difference_objective(self) -> None:
        m = self.model
        m.objective = Objective(
            expr=sum(
                m.PoA[k] - self.eta * m.wasserstein_distance[k]
                for k in m.scenarios
            )
            / self.num_empirical_scenarios,
            sense=maximize,
        )

    def _build_ratio_mccormick_objective(self) -> None:
        m = self.model
        m.objective = Objective(
            expr=sum(
                m.phi[k] - self.eta * m.wasserstein_distance[k]
                for k in m.scenarios
            )
            / self.num_empirical_scenarios,
            sense=maximize,
        )

    def _build_ratio_piecewise_mccormick_objective(self) -> None:
        m = self.model
        m.objective = Objective(
            expr=sum(
                m.phi[k] - self.eta * m.wasserstein_distance[k]
                for k in m.scenarios
            )
            / self.num_empirical_scenarios,
            sense=maximize,
        )

    def _build_constraints(self) -> None:
        self._build_support_set()
        self._build_transport_constraints()
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
        self._build_regime_fixing_constraints()
        self._build_support_set_demand()
        self._build_support_set_wind()

    def _build_regime_fixing_constraints(self) -> None:
        m = self.model
        m.regime_mu_D_fixed = Constraint(expr=m.mu_D == self.mu_D_fixed)
        m.regime_sigma_D_fixed = Constraint(expr=m.sigma_D == self.sigma_D_fixed)
        m.regime_mu_W_fixed = Constraint(expr=m.mu_W == self.mu_W_fixed)
        m.regime_sigma_W_fixed = Constraint(expr=m.sigma_W == self.sigma_W_fixed)
        m.regime_rho_D_fixed = Constraint(expr=m.rho_D == self.demand_rho_fixed)
        m.regime_rho_W_fixed = Constraint(expr=m.rho_W == self.wind_rho_fixed)
        m.regime_peak_W_fixed = Constraint(expr=m.peak_W == self.peak_W_fixed)

    def _build_support_set_demand(self) -> None:
        m = self.model
        m.demand_reference = Expression(
            m.time_steps,
            rule=lambda m, t: self.demand_D_ref * m.mu_D * self.demand_shape[int(t)],
        )
        m.demand_lower = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t] - self.demand_D_ref * m.sigma_D,
        )
        m.demand_upper = Expression(
            m.time_steps,
            rule=lambda m, t: m.demand_reference[t] + self.demand_D_ref * m.sigma_D,
        )
        m.demand_ramp = Expression(
            m.time_steps_minus_1,
            rule=lambda m, t: self.demand_D_ref
            * (
                m.mu_D * self.demand_delta_shape[int(t)]
                + (1.0 - self.demand_rho_fixed) * m.sigma_D
            ),
        )

        def demand_lower_rule(m, k, t):
            return m.D[k, t] >= m.demand_lower[t]

        def demand_upper_rule(m, k, t):
            return m.D[k, t] <= m.demand_upper[t]

        def demand_ramp_up_rule(m, k, t):
            return m.D[k, t] - m.D[k, t - 1] <= m.demand_ramp[t]

        def demand_ramp_down_rule(m, k, t):
            return m.D[k, t - 1] - m.D[k, t] <= m.demand_ramp[t]

        def demand_feasibility_rule(m, t):
            return m.demand_reference[t] - self.demand_D_ref * m.sigma_D >= 0

        m.demand_lower_bound_constraints = Constraint(
            m.scenarios,
            m.time_steps,
            rule=demand_lower_rule,
        )
        m.demand_upper_bound_constraints = Constraint(
            m.scenarios,
            m.time_steps,
            rule=demand_upper_rule,
        )
        m.demand_ramp_up_constraints = Constraint(
            m.scenarios,
            m.time_steps_minus_1,
            rule=demand_ramp_up_rule,
        )
        m.demand_ramp_down_constraints = Constraint(
            m.scenarios,
            m.time_steps_minus_1,
            rule=demand_ramp_down_rule,
        )
        m.demand_lower_feasibility = Constraint(m.time_steps, rule=demand_feasibility_rule)

    def _build_support_set_wind(self) -> None:
        m = self.model
        m.wind_reference = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: self.static_physical_capacity[int(i)]
            * m.mu_W
            * self.wind_shape[int(t)],
        )
        m.wind_lower = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t]
            - self.static_physical_capacity[int(i)] * m.sigma_W,
        )
        m.wind_upper = Expression(
            m.wind_physical_generators,
            m.time_steps,
            rule=lambda m, i, t: m.wind_reference[i, t]
            + self.static_physical_capacity[int(i)] * m.sigma_W,
        )
        m.wind_ramp = Expression(
            m.wind_physical_generators,
            m.time_steps_minus_1,
            rule=lambda m, i, t: self.static_physical_capacity[int(i)]
            * (
                m.mu_W * self.wind_delta_shape[int(t)]
                + (1.0 - self.wind_rho_fixed) * m.sigma_W
            ),
        )

        def conventional_capacity_rule(m, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.P_max_block[k, i, b, t] == self.static_block_capacity[global_block]

        def wind_total_lower_rule(m, k, i, t):
            return (
                sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                >= m.wind_lower[i, t]
            )

        def wind_total_upper_rule(m, k, i, t):
            return (
                sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= m.wind_upper[i, t]
            )

        def wind_even_block_split_rule(m, k, i, b, t):
            local_blocks = self.local_blocks_by_generator[int(i)]
            return (
                len(local_blocks) * m.P_max_block[k, i, b, t]
                == sum(m.P_max_block[k, i, other_b, t] for other_b in local_blocks)
            )

        def wind_ramp_up_rule(m, k, i, t):
            return (
                sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(
                    m.P_max_block[k, i, b, t - 1]
                    for b in self.local_blocks_by_generator[int(i)]
                )
                <= m.wind_ramp[i, t]
            )

        def wind_ramp_down_rule(m, k, i, t):
            return (
                sum(
                    m.P_max_block[k, i, b, t - 1]
                    for b in self.local_blocks_by_generator[int(i)]
                )
                - sum(m.P_max_block[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                <= m.wind_ramp[i, t]
            )

        def wind_capacity_factor_lower_feasibility_rule(m, t):
            return m.mu_W * self.wind_shape[int(t)] - m.sigma_W >= 0

        def wind_capacity_factor_upper_feasibility_rule(m, t):
            return m.mu_W * self.wind_shape[int(t)] + m.sigma_W <= 1

        def dispatch_capacity_feasibility_rule(m, k, t):
            return m.D[k, t] <= sum(
                m.P_max_block[k, i, b, t]
                for i, b in m.generator_blocks
            )

        m.conventional_capacity = Constraint(
            m.scenarios,
            m.conventional_blocks,
            m.time_steps,
            rule=conventional_capacity_rule,
        )
        m.wind_total_lower_bound = Constraint(
            m.scenarios,
            m.wind_physical_generators,
            m.time_steps,
            rule=wind_total_lower_rule,
        )
        m.wind_total_upper_bound = Constraint(
            m.scenarios,
            m.wind_physical_generators,
            m.time_steps,
            rule=wind_total_upper_rule,
        )
        m.wind_even_block_split = Constraint(
            m.scenarios,
            m.wind_blocks,
            m.time_steps,
            rule=wind_even_block_split_rule,
        )
        m.wind_ramp_up = Constraint(
            m.scenarios,
            m.wind_physical_generators,
            m.time_steps_minus_1,
            rule=wind_ramp_up_rule,
        )
        m.wind_ramp_down = Constraint(
            m.scenarios,
            m.wind_physical_generators,
            m.time_steps_minus_1,
            rule=wind_ramp_down_rule,
        )
        m.wind_capacity_factor_lower_feasibility = Constraint(
            m.time_steps,
            rule=wind_capacity_factor_lower_feasibility_rule,
        )
        m.wind_capacity_factor_upper_feasibility = Constraint(
            m.time_steps,
            rule=wind_capacity_factor_upper_feasibility_rule,
        )
        m.dispatch_capacity_feasibility = Constraint(
            m.scenarios,
            m.time_steps,
            rule=dispatch_capacity_feasibility_rule,
        )

    def _build_transport_constraints(self) -> None:
        m = self.model

        def physical_capacity_expr(k: int, i: int, t: int):
            return sum(
                m.P_max_block[k, i, b, t]
                for b in self.local_blocks_by_generator[int(i)]
            )

        def demand_transport_pos_rule(m, k, t):
            return m.D_transport_abs_deviation[k, t] >= m.D[k, t] - self.empirical_D[int(k)][int(t)]

        def demand_transport_neg_rule(m, k, t):
            return m.D_transport_abs_deviation[k, t] >= self.empirical_D[int(k)][int(t)] - m.D[k, t]

        def pmax_transport_pos_rule(m, k, i, t):
            return (
                m.P_max_phys_transport_abs_deviation[k, i, t]
                >= physical_capacity_expr(int(k), int(i), int(t))
                - self.empirical_Pmax_phys[int(k)][int(i)][int(t)]
            )

        def pmax_transport_neg_rule(m, k, i, t):
            return (
                m.P_max_phys_transport_abs_deviation[k, i, t]
                >= self.empirical_Pmax_phys[int(k)][int(i)][int(t)]
                - physical_capacity_expr(int(k), int(i), int(t))
            )

        def wasserstein_distance_rule(m, k):
            return m.wasserstein_distance[k] == (
                sum(m.D_transport_abs_deviation[k, t] for t in m.time_steps)
                + sum(
                    m.P_max_phys_transport_abs_deviation[k, i, t]
                    for i in m.physical_generators
                    for t in m.time_steps
                )
            )

        m.demand_transport_abs_pos = Constraint(
            m.scenarios,
            m.time_steps,
            rule=demand_transport_pos_rule,
        )
        m.demand_transport_abs_neg = Constraint(
            m.scenarios,
            m.time_steps,
            rule=demand_transport_neg_rule,
        )
        m.pmax_phys_transport_abs_pos = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            rule=pmax_transport_pos_rule,
        )
        m.pmax_phys_transport_abs_neg = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            rule=pmax_transport_neg_rule,
        )
        m.wasserstein_distance_definition = Constraint(
            m.scenarios,
            rule=wasserstein_distance_rule,
        )

    # ------------------------------------------------------------------
    # Lower level equilibrium and optimality constraints
    # ------------------------------------------------------------------

    def _build_lower_level_equilibrium_constraints(self) -> None:
        m = self.model

        def power_balance_eq_rule(m, k, t):
            return m.D[k, t] - sum(m.P_eq[k, i, b, t] for (i, b) in m.generator_blocks) == 0

        def generation_upper_eq_rule(m, k, i, b, t):
            return m.P_eq[k, i, b, t] - m.P_max_block[k, i, b, t] <= 0

        def generation_lower_eq_rule(m, k, i, b, t):
            return m.P_eq[k, i, b, t] >= 0

        def ramp_up_eq_rule(m, k, i, t):
            return (
                sum(m.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
                <= 0
            )

        def ramp_up_initial_eq_rule(m, k, i):
            return (
                sum(m.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(k)][int(i)]
                <= self.ramp_vector_up[int(i)]
            )

        def ramp_down_eq_rule(m, k, i, t):
            return (
                -sum(m.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        def ramp_down_initial_eq_rule(m, k, i):
            return (
                -sum(m.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(k)][int(i)]
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        m.power_balance_eq = Constraint(m.scenarios, m.time_steps, rule=power_balance_eq_rule)
        m.generation_upper_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=generation_upper_eq_rule,
        )
        m.generation_lower_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=generation_lower_eq_rule,
        )
        m.ramp_up_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_up_eq_rule,
        )
        m.ramp_up_initial_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_up_initial_eq_rule,
        )
        m.ramp_down_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_down_eq_rule,
        )
        m.ramp_down_initial_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_down_initial_eq_rule,
        )

    def _build_lower_level_optimal_constraints(self) -> None:
        m = self.model

        def power_balance_opt_rule(m, k, t):
            return m.D[k, t] - sum(m.P_opt[k, i, b, t] for (i, b) in m.generator_blocks) == 0

        def generation_upper_opt_rule(m, k, i, b, t):
            return m.P_opt[k, i, b, t] - m.P_max_block[k, i, b, t] <= 0

        def generation_lower_opt_rule(m, k, i, b, t):
            return m.P_opt[k, i, b, t] >= 0

        def ramp_up_opt_rule(m, k, i, t):
            return (
                sum(m.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
                <= 0
            )

        def ramp_up_initial_opt_rule(m, k, i):
            return (
                sum(m.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(k)][int(i)]
                <= self.ramp_vector_up[int(i)]
            )

        def ramp_down_opt_rule(m, k, i, t):
            return (
                -sum(m.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        def ramp_down_initial_opt_rule(m, k, i):
            return (
                -sum(m.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(k)][int(i)]
                - self.ramp_vector_down[int(i)]
                <= 0
            )

        m.power_balance_opt = Constraint(m.scenarios, m.time_steps, rule=power_balance_opt_rule)
        m.generation_upper_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=generation_upper_opt_rule,
        )
        m.generation_lower_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=generation_lower_opt_rule,
        )
        m.ramp_up_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_up_opt_rule,
        )
        m.ramp_up_initial_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_up_initial_opt_rule,
        )
        m.ramp_down_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_down_opt_rule,
        )
        m.ramp_down_initial_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_down_initial_opt_rule,
        )

    # ------------------------------------------------------------------
    # KKT stationarity conditions
    # ------------------------------------------------------------------

    def _build_KKT_stationarity_equilibrium_constraints(self) -> None:
        m = self.model

        def stationarity_eq_rule(m, k, i, b, t):
            return (
                m.alpha[k, i, b, t]
                - m.lambda_eq[k, t]
                + m.mu_upper_eq[k, i, b, t]
                - m.mu_lower_eq[k, i, b, t]
                + m.mu_ramp_up_eq[k, i, t]
                - m.mu_ramp_up_eq[k, i, t + 1]
                - m.mu_ramp_down_eq[k, i, t]
                + m.mu_ramp_down_eq[k, i, t + 1]
                == 0
            )

        def final_ramp_up_dual_eq_rule(m, k, i):
            return m.mu_ramp_up_eq[k, i, self.num_time_steps] == 0

        def final_ramp_down_dual_eq_rule(m, k, i):
            return m.mu_ramp_down_eq[k, i, self.num_time_steps] == 0

        m.stationarity_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=stationarity_eq_rule,
        )
        m.final_ramp_up_dual_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=final_ramp_up_dual_eq_rule,
        )
        m.final_ramp_down_dual_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=final_ramp_down_dual_eq_rule,
        )

    def _build_KKT_stationarity_optimal_constraints(self) -> None:
        m = self.model

        def stationarity_opt_rule(m, k, i, b, t):
            global_block = self.local_to_global_block[(int(i), int(b))]
            return (
                self.block_cost_vector[global_block]
                - m.lambda_opt[k, t]
                + m.mu_upper_opt[k, i, b, t]
                - m.mu_lower_opt[k, i, b, t]
                + m.mu_ramp_up_opt[k, i, t]
                - m.mu_ramp_up_opt[k, i, t + 1]
                - m.mu_ramp_down_opt[k, i, t]
                + m.mu_ramp_down_opt[k, i, t + 1]
                == 0
            )

        def final_ramp_up_dual_opt_rule(m, k, i):
            return m.mu_ramp_up_opt[k, i, self.num_time_steps] == 0

        def final_ramp_down_dual_opt_rule(m, k, i):
            return m.mu_ramp_down_opt[k, i, self.num_time_steps] == 0

        m.stationarity_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=stationarity_opt_rule,
        )
        m.final_ramp_up_dual_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=final_ramp_up_dual_opt_rule,
        )
        m.final_ramp_down_dual_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=final_ramp_down_dual_opt_rule,
        )

    # ------------------------------------------------------------------
    # KKT complementarity conditions
    # ------------------------------------------------------------------

    def _build_KKT_complementarity_equilibrium_constraints(self) -> None:
        m = self.model

        def upper_bound_complementarity_eq_rule(m, k, i, b, t):
            return (
                -self.M_cap[int(i), int(b), int(t)] * (1 - m.z_upper_eq[k, i, b, t])
                <= m.P_eq[k, i, b, t] - m.P_max_block[k, i, b, t]
            )

        def upper_bound_complementarity_dual_eq_rule(m, k, i, b, t):
            return (
                m.mu_upper_eq[k, i, b, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_upper_eq,
                    int(k),
                    (int(i), int(b), int(t)),
                ) * m.z_upper_eq[k, i, b, t]
            )

        def lower_bound_complementarity_eq_rule(m, k, i, b, t):
            return (
                -self.M_lower[int(i), int(b), int(t)] * (1 - m.z_lower_eq[k, i, b, t])
                <= -m.P_eq[k, i, b, t]
            )

        def lower_bound_complementarity_dual_eq_rule(m, k, i, b, t):
            return (
                m.mu_lower_eq[k, i, b, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_lower_eq,
                    int(k),
                    (int(i), int(b), int(t)),
                ) * m.z_lower_eq[k, i, b, t]
            )

        def ramp_up_complementarity_eq_rule(m, k, i, t):
            return -self.M_ramp_up[int(i), int(t)] * (1 - m.z_ramp_up_eq[k, i, t]) <= (
                sum(m.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_eq_rule(m, k, i):
            return -self.M_ramp_up_initial[int(i)] * (1 - m.z_ramp_up_eq[k, i, 0]) <= (
                sum(m.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(k)][int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_eq_rule(m, k, i, t):
            return (
                m.mu_ramp_up_eq[k, i, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_ramp_up_eq,
                    int(k),
                    (int(i), int(t)),
                ) * m.z_ramp_up_eq[k, i, t]
            )

        def ramp_down_complementarity_eq_rule(m, k, i, t):
            return -self.M_ramp_down[int(i), int(t)] * (1 - m.z_ramp_down_eq[k, i, t]) <= (
                -sum(m.P_eq[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_eq[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_eq_rule(m, k, i):
            return -self.M_ramp_down_initial[int(i)] * (1 - m.z_ramp_down_eq[k, i, 0]) <= (
                -sum(m.P_eq[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(k)][int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_eq_rule(m, k, i, t):
            return (
                m.mu_ramp_down_eq[k, i, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_ramp_down_eq,
                    int(k),
                    (int(i), int(t)),
                ) * m.z_ramp_down_eq[k, i, t]
            )

        m.upper_bound_complementarity_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=upper_bound_complementarity_eq_rule,
        )
        m.upper_bound_complementarity_dual_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=upper_bound_complementarity_dual_eq_rule,
        )
        m.lower_bound_complementarity_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=lower_bound_complementarity_eq_rule,
        )
        m.lower_bound_complementarity_dual_eq = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=lower_bound_complementarity_dual_eq_rule,
        )
        m.ramp_up_complementarity_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_up_complementarity_eq_rule,
        )
        m.ramp_up_complementarity_dual_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            rule=ramp_up_complementarity_dual_eq_rule,
        )
        m.ramp_up_initial_complementarity_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_up_initial_complementarity_eq_rule,
        )
        m.ramp_down_complementarity_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_down_complementarity_eq_rule,
        )
        m.ramp_down_complementarity_dual_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            rule=ramp_down_complementarity_dual_eq_rule,
        )
        m.ramp_down_initial_complementarity_eq = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_down_initial_complementarity_eq_rule,
        )

    def _build_KKT_complementarity_optimal_constraints(self) -> None:
        m = self.model

        def upper_bound_complementarity_opt_rule(m, k, i, b, t):
            return (
                -self.M_cap[int(i), int(b), int(t)] * (1 - m.z_upper_opt[k, i, b, t])
                <= m.P_opt[k, i, b, t] - m.P_max_block[k, i, b, t]
            )

        def upper_bound_complementarity_dual_opt_rule(m, k, i, b, t):
            return (
                m.mu_upper_opt[k, i, b, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_upper_opt,
                    int(k),
                    (int(i), int(b), int(t)),
                ) * m.z_upper_opt[k, i, b, t]
            )

        def lower_bound_complementarity_opt_rule(m, k, i, b, t):
            return (
                -self.M_lower[int(i), int(b), int(t)] * (1 - m.z_lower_opt[k, i, b, t])
                <= -m.P_opt[k, i, b, t]
            )

        def lower_bound_complementarity_dual_opt_rule(m, k, i, b, t):
            return (
                m.mu_lower_opt[k, i, b, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_lower_opt,
                    int(k),
                    (int(i), int(b), int(t)),
                ) * m.z_lower_opt[k, i, b, t]
            )

        def ramp_up_complementarity_opt_rule(m, k, i, t):
            return -self.M_ramp_up[int(i), int(t)] * (1 - m.z_ramp_up_opt[k, i, t]) <= (
                sum(m.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_opt_rule(m, k, i):
            return -self.M_ramp_up_initial[int(i)] * (1 - m.z_ramp_up_opt[k, i, 0]) <= (
                sum(m.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(k)][int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_opt_rule(m, k, i, t):
            return (
                m.mu_ramp_up_opt[k, i, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_ramp_up_opt,
                    int(k),
                    (int(i), int(t)),
                ) * m.z_ramp_up_opt[k, i, t]
            )

        def ramp_down_complementarity_opt_rule(m, k, i, t):
            return -self.M_ramp_down[int(i), int(t)] * (1 - m.z_ramp_down_opt[k, i, t]) <= (
                -sum(m.P_opt[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_opt[k, i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_opt_rule(m, k, i):
            return -self.M_ramp_down_initial[int(i)] * (1 - m.z_ramp_down_opt[k, i, 0]) <= (
                -sum(m.P_opt[k, i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(k)][int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_opt_rule(m, k, i, t):
            return (
                m.mu_ramp_down_opt[k, i, t]
                <= self._scenario_or_regime_value(
                    self.M_mu_ramp_down_opt,
                    int(k),
                    (int(i), int(t)),
                ) * m.z_ramp_down_opt[k, i, t]
            )

        m.upper_bound_complementarity_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=upper_bound_complementarity_opt_rule,
        )
        m.upper_bound_complementarity_dual_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=upper_bound_complementarity_dual_opt_rule,
        )
        m.lower_bound_complementarity_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=lower_bound_complementarity_opt_rule,
        )
        m.lower_bound_complementarity_dual_opt = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=lower_bound_complementarity_dual_opt_rule,
        )
        m.ramp_up_complementarity_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_up_complementarity_opt_rule,
        )
        m.ramp_up_complementarity_dual_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            rule=ramp_up_complementarity_dual_opt_rule,
        )
        m.ramp_up_initial_complementarity_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_up_initial_complementarity_opt_rule,
        )
        m.ramp_down_complementarity_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps_minus_1,
            rule=ramp_down_complementarity_opt_rule,
        )
        m.ramp_down_complementarity_dual_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            m.time_steps,
            rule=ramp_down_complementarity_dual_opt_rule,
        )
        m.ramp_down_initial_complementarity_opt = Constraint(
            m.scenarios,
            m.physical_generators,
            rule=ramp_down_initial_complementarity_opt_rule,
        )

    # ------------------------------------------------------------------
    # Aggregate dual-bound valid inequalities
    # ------------------------------------------------------------------

    def _build_aggregate_dual_bound_constraints(self) -> int:
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
            def rule(model, k, side, t):
                side = str(side)
                bound = aggregate_bound(side, constraint_type, int(t))
                if bound is None:
                    return Constraint.Skip
                mu = dual_component(side, constraint_type)
                return (
                    sum(mu[k, i, b, t] for (i, b) in model.generator_blocks)
                    <= float(bound)
                )

            return rule

        def ramp_sum_rule(constraint_type: str):
            def rule(model, k, side, t):
                side = str(side)
                bound = aggregate_bound(side, constraint_type, int(t))
                if bound is None:
                    return Constraint.Skip
                mu = dual_component(side, constraint_type)
                return (
                    sum(mu[k, i, t] for i in model.physical_generators)
                    <= float(bound)
                )

            return rule

        m.aggregate_mu_max_bound = Constraint(
            m.scenarios,
            m.kkt_sides,
            m.time_steps,
            rule=capacity_sum_rule("upper"),
        )
        m.aggregate_mu_min_bound = Constraint(
            m.scenarios,
            m.kkt_sides,
            m.time_steps,
            rule=capacity_sum_rule("lower"),
        )
        m.aggregate_mu_ramp_up_bound = Constraint(
            m.scenarios,
            m.kkt_sides,
            m.time_steps,
            rule=ramp_sum_rule("ramp_up"),
        )
        m.aggregate_mu_ramp_down_bound = Constraint(
            m.scenarios,
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
    # PoA constraints
    # ------------------------------------------------------------------

    def _build_PoA_constraints(self) -> None:
        m = self.model

        def cost_eq_rule(m, k):
            return m.C_eq[k] == sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * m.P_eq[k, i, b, t]
                for (i, b) in m.generator_blocks
                for t in m.time_steps
            )

        def cost_opt_rule(m, k):
            return m.C_opt[k] == sum(
                self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
                * m.P_opt[k, i, b, t]
                for (i, b) in m.generator_blocks
                for t in m.time_steps
            )

        def poa_rule(m, k):
            return m.C_eq[k] - m.C_opt[k] == m.PoA[k]

        m.cost_definition_eq = Constraint(m.scenarios, rule=cost_eq_rule)
        m.cost_definition_opt = Constraint(m.scenarios, rule=cost_opt_rule)
        m.poa_definition = Constraint(m.scenarios, rule=poa_rule)
        if self.objective_mode == "ratio_mccormick":
            self._build_ratio_mccormick_constraints()
        if self.objective_mode == "ratio_piecewise_mccormick":
            self._build_ratio_piecewise_mccormick_constraints()

    def _build_ratio_mccormick_constraints(self) -> None:
        if self.ratio_bounds is None:
            raise ValueError(
                "ratio_bounds is required when objective_mode='ratio_mccormick'"
            )
        m = self.model
        phi_L, phi_U = self.ratio_bounds["phi"]
        C_opt_L, C_opt_U = self.ratio_bounds["C_opt"]

        def ratio_link_eq_cost_rule(m, k):
            return m.z_ratio_product[k] == m.C_eq[k]

        def lower_1_rule(m, k):
            return (
                m.z_ratio_product[k]
                >= phi_L * m.C_opt[k] + C_opt_L * m.phi[k] - phi_L * C_opt_L
            )

        def lower_2_rule(m, k):
            return (
                m.z_ratio_product[k]
                >= phi_U * m.C_opt[k] + C_opt_U * m.phi[k] - phi_U * C_opt_U
            )

        def upper_1_rule(m, k):
            return (
                m.z_ratio_product[k]
                <= phi_U * m.C_opt[k] + C_opt_L * m.phi[k] - phi_U * C_opt_L
            )

        def upper_2_rule(m, k):
            return (
                m.z_ratio_product[k]
                <= phi_L * m.C_opt[k] + C_opt_U * m.phi[k] - phi_L * C_opt_U
            )

        m.ratio_link_eq_cost = Constraint(m.scenarios, rule=ratio_link_eq_cost_rule)
        m.ratio_mccormick_lower_1 = Constraint(m.scenarios, rule=lower_1_rule)
        m.ratio_mccormick_lower_2 = Constraint(m.scenarios, rule=lower_2_rule)
        m.ratio_mccormick_upper_1 = Constraint(m.scenarios, rule=upper_1_rule)
        m.ratio_mccormick_upper_2 = Constraint(m.scenarios, rule=upper_2_rule)

    def _build_ratio_piecewise_mccormick_constraints(self) -> None:
        if self.ratio_bounds is None:
            raise ValueError(
                "ratio_bounds is required when "
                "objective_mode='ratio_piecewise_mccormick'"
            )
        m = self.model
        phi_L, phi_U = self.ratio_bounds["phi"]
        breakpoints = list(self.ratio_bounds["C_opt_breakpoints"])

        def select_one_rule(m, k):
            return sum(
                m.ratio_piece_active[k, p] for p in m.ratio_piece_index
            ) == 1

        def C_opt_link_rule(m, k):
            return m.C_opt[k] == sum(
                m.C_opt_piece[k, p] for p in m.ratio_piece_index
            )

        def phi_link_rule(m, k):
            return m.phi[k] == sum(
                m.phi_piece[k, p] for p in m.ratio_piece_index
            )

        def z_link_rule(m, k):
            return m.z_ratio_product[k] == sum(
                m.z_ratio_piece[k, p] for p in m.ratio_piece_index
            )

        def C_eq_link_rule(m, k):
            return m.C_eq[k] == m.z_ratio_product[k]

        def C_opt_piece_lower_rule(m, k, p):
            return (
                breakpoints[int(p)] * m.ratio_piece_active[k, p]
                <= m.C_opt_piece[k, p]
            )

        def C_opt_piece_upper_rule(m, k, p):
            return (
                m.C_opt_piece[k, p]
                <= breakpoints[int(p) + 1] * m.ratio_piece_active[k, p]
            )

        def phi_piece_lower_rule(m, k, p):
            return phi_L * m.ratio_piece_active[k, p] <= m.phi_piece[k, p]

        def phi_piece_upper_rule(m, k, p):
            return m.phi_piece[k, p] <= phi_U * m.ratio_piece_active[k, p]

        def lower_1_rule(m, k, p):
            p_int = int(p)
            y_L = breakpoints[p_int]
            return (
                m.z_ratio_piece[k, p]
                >= phi_L * m.C_opt_piece[k, p]
                + y_L * m.phi_piece[k, p]
                - phi_L * y_L * m.ratio_piece_active[k, p]
            )

        def lower_2_rule(m, k, p):
            p_int = int(p)
            y_U = breakpoints[p_int + 1]
            return (
                m.z_ratio_piece[k, p]
                >= phi_U * m.C_opt_piece[k, p]
                + y_U * m.phi_piece[k, p]
                - phi_U * y_U * m.ratio_piece_active[k, p]
            )

        def upper_1_rule(m, k, p):
            p_int = int(p)
            y_L = breakpoints[p_int]
            return (
                m.z_ratio_piece[k, p]
                <= phi_U * m.C_opt_piece[k, p]
                + y_L * m.phi_piece[k, p]
                - phi_U * y_L * m.ratio_piece_active[k, p]
            )

        def upper_2_rule(m, k, p):
            p_int = int(p)
            y_U = breakpoints[p_int + 1]
            return (
                m.z_ratio_piece[k, p]
                <= phi_L * m.C_opt_piece[k, p]
                + y_U * m.phi_piece[k, p]
                - phi_L * y_U * m.ratio_piece_active[k, p]
            )

        m.ratio_piece_select_one = Constraint(m.scenarios, rule=select_one_rule)
        m.ratio_piece_C_opt_link = Constraint(m.scenarios, rule=C_opt_link_rule)
        m.ratio_piece_phi_link = Constraint(m.scenarios, rule=phi_link_rule)
        m.ratio_piece_z_link = Constraint(m.scenarios, rule=z_link_rule)
        m.ratio_piece_C_eq_link = Constraint(m.scenarios, rule=C_eq_link_rule)
        m.ratio_piece_C_opt_lower = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=C_opt_piece_lower_rule,
        )
        m.ratio_piece_C_opt_upper = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=C_opt_piece_upper_rule,
        )
        m.ratio_piece_phi_lower = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=phi_piece_lower_rule,
        )
        m.ratio_piece_phi_upper = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=phi_piece_upper_rule,
        )
        m.ratio_piece_mccormick_lower_1 = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=lower_1_rule,
        )
        m.ratio_piece_mccormick_lower_2 = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=lower_2_rule,
        )
        m.ratio_piece_mccormick_upper_1 = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=upper_1_rule,
        )
        m.ratio_piece_mccormick_upper_2 = Constraint(
            m.scenarios,
            m.ratio_piece_index,
            rule=upper_2_rule,
        )

    # ------------------------------------------------------------------
    # Policy constraints
    # ------------------------------------------------------------------

    def _build_policy_constraints(self) -> None:
        m = self.model

        def true_cost_alpha_rule(m, k, i, b, t):
            if int(i) in self.nn_policy_generator_ids:
                return Constraint.Skip
            global_block = self.local_to_global_block[(int(i), int(b))]
            return m.alpha[k, i, b, t] == self.block_cost_vector[global_block]

        m.true_cost_alpha = Constraint(
            m.scenarios,
            m.generator_blocks,
            m.time_steps,
            rule=true_cost_alpha_rule,
        )

        if self.nn_policy_generator_ids:
            self._build_nn_policy_constraints()
        else:
            m.nn_constraints = ConstraintList()

    def _raw_nn_feature_expression(
        self,
        feature_name: str,
        k: int,
        t: int,
        physical_generator_idx: int,
    ):
        m = self.model
        k = int(k)
        t = int(t)
        physical_generator_idx = int(physical_generator_idx)
        previous_t = self.num_time_steps - 1 if t == 0 else t - 1
        next_t = 0 if t == self.num_time_steps - 1 else t + 1

        total_wind_capacity = lambda time_idx: sum(
            m.P_max_block[k, i, b, time_idx] for (i, b) in self.wind_block_pairs
        )
        total_capacity = lambda time_idx: sum(
            m.P_max_block[k, i, b, time_idx] for (i, b) in self.generator_block_pairs
        )
        own_capacity = lambda time_idx: sum(
            m.P_max_block[k, physical_generator_idx, b, time_idx]
            for b in self.local_blocks_by_generator[physical_generator_idx]
        )

        if feature_name == "demand":
            return m.D[k, t]
        if feature_name == "total_wind_generation_capacity":
            return total_wind_capacity(t)
        if feature_name == "total_generation_capacity":
            return total_capacity(t)
        if feature_name == "residual_demand":
            return m.D[k, t] - total_wind_capacity(t)
        if feature_name == "previous_generation_capacity":
            return total_capacity(previous_t)
        if feature_name == "previous_demand":
            return m.D[k, previous_t]
        if feature_name == "next_generation_capacity":
            return total_capacity(next_t)
        if feature_name == "next_demand":
            return m.D[k, next_t]
        if feature_name == "own_generation_capacity":
            return own_capacity(t)
        if feature_name == "previous_own_generation_capacity":
            return own_capacity(previous_t)
        if feature_name == "next_own_generation_capacity":
            return own_capacity(next_t)
        if feature_name == "average_true_cost":
            costs = [
                self.block_cost_vector[
                    self.local_to_global_block[(physical_generator_idx, b)]
                ]
                for b in self.local_blocks_by_generator[physical_generator_idx]
            ]
            return float(np.mean(costs))
        if feature_name == "minimum_true_cost":
            return float(
                min(
                    self.block_cost_vector[
                        self.local_to_global_block[(physical_generator_idx, b)]
                    ]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        if feature_name == "maximum_true_cost":
            return float(
                max(
                    self.block_cost_vector[
                        self.local_to_global_block[(physical_generator_idx, b)]
                    ]
                    for b in self.local_blocks_by_generator[physical_generator_idx]
                )
            )
        raise ValueError(f"Unsupported NN feature name: {feature_name}")

    def _normalized_nn_feature_expression(
        self,
        generator_name: str,
        feature_name: str,
        k: int,
        t: int,
        physical_generator_idx: int,
    ):
        raw = self._raw_nn_feature_expression(feature_name, k, t, physical_generator_idx)
        feature_min, feature_max = self._nn_feature_bounds(generator_name, feature_name)
        denominator = feature_max - feature_min
        if abs(denominator) <= self.normalization_epsilon:
            return 0.0
        return (raw - feature_min) / denominator

    def _build_nn_policy_constraints(self) -> None:
        m = self.model
        if not self.nn_relu_bounds:
            raise ValueError(
                "NN ReLU bounds are required before building NN policy constraints. "
                "Run the DRO ReLU-bound tightening stage or call "
                "load_nn_relu_bounds_report(...)."
            )
        nn_input_indices: list[tuple[int, int, int, int]] = []
        nn_z_indices: list[tuple[int, int, int, int, int]] = []
        nn_h_indices: list[tuple[int, int, int, int, int]] = []
        nn_output_indices: list[tuple[int, int, int, int]] = []

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(i)]
            if generator_name not in self.nn_policies:
                raise ValueError(f"NN policy for generator {generator_name} was not loaded.")
            if generator_name not in self.nn_relu_bounds:
                raise ValueError(
                    "NN ReLU bounds report is missing bounds for generator "
                    f"{generator_name}."
                )
            policy = self.nn_policies[generator_name]
            for k in range(self.num_empirical_scenarios):
                for t in range(self.num_time_steps):
                    for f_idx, _ in enumerate(policy["feature_columns"]):
                        nn_input_indices.append((int(k), int(i), int(t), int(f_idx)))

            linear_idx = 0
            layers = policy["layers"]
            for layer_pos, layer in enumerate(layers):
                if str(layer.get("type", "")).lower() != "linear":
                    continue
                output_dim = len(layer["bias"])
                is_final_linear = layer_pos == len(layers) - 1
                for k in range(self.num_empirical_scenarios):
                    for t in range(self.num_time_steps):
                        for node in range(output_dim):
                            if is_final_linear:
                                nn_output_indices.append((int(k), int(i), int(t), int(node)))
                            else:
                                nn_z_indices.append(
                                    (int(k), int(i), int(t), int(linear_idx), int(node))
                                )
                                nn_h_indices.append(
                                    (int(k), int(i), int(t), int(linear_idx), int(node))
                                )
                linear_idx += 1

        m.nn_input_index = Set(dimen=4, initialize=nn_input_indices)
        m.nn_z_index = Set(dimen=5, initialize=nn_z_indices)
        m.nn_h_index = Set(dimen=5, initialize=nn_h_indices)
        m.nn_delta_index = Set(dimen=5, initialize=nn_h_indices)
        m.nn_output_index = Set(dimen=4, initialize=nn_output_indices)
        m.nn_input = Var(m.nn_input_index, domain=Reals)
        m.nn_z = Var(m.nn_z_index, domain=Reals)
        m.nn_h = Var(m.nn_h_index, domain=NonNegativeReals)
        m.nn_delta = Var(m.nn_delta_index, domain=Binary)
        m.nn_output = Var(m.nn_output_index, domain=Reals)
        m.nn_constraints = ConstraintList()

        def relu_bound_for_node(
            relu_bounds: dict[tuple[int, ...], dict[str, Any]],
            k: int,
            t: int,
            linear_idx: int,
            node: int,
            generator_name: str,
        ) -> dict[str, Any]:
            scenario_key = (int(k), int(t), int(linear_idx), int(node))
            regime_key = (int(t), int(linear_idx), int(node))
            if scenario_key in relu_bounds:
                return relu_bounds[scenario_key]
            if regime_key in relu_bounds:
                return relu_bounds[regime_key]
            raise ValueError(
                f"NN ReLU bounds report is missing bounds for {generator_name} "
                f"at scenario {k}, time {t}, linear layer {linear_idx}, node {node}."
            )

        for i in self.nn_policy_generator_ids:
            generator_name = self.physical_generator_names[int(i)]
            relu_bounds = self.nn_relu_bounds[generator_name]
            for raw_key, bounds in relu_bounds.items():
                key = tuple(int(part) for part in raw_key)
                scenario_keys = (
                    [key]
                    if len(key) == 4
                    else [(int(k), *key) for k in m.scenarios]
                )
                for scenario_key in scenario_keys:
                    k, time_idx, linear_idx, node = scenario_key
                    index = (int(k), int(i), int(time_idx), int(linear_idx), int(node))
                    if index not in m.nn_z:
                        continue
                    m.nn_z[index].setlb(float(bounds["L"]))
                    m.nn_z[index].setub(float(bounds["U"]))
                    m.nn_h[index].setlb(float(bounds["h_lower"]))
                    m.nn_h[index].setub(float(bounds["h_upper"]))

        for i in self.nn_policy_generator_ids:
            i = int(i)
            generator_name = self.physical_generator_names[i]
            policy = self.nn_policies[generator_name]
            relu_bounds = self.nn_relu_bounds[generator_name]
            feature_columns = policy["feature_columns"]
            for k in range(self.num_empirical_scenarios):
                for t in range(self.num_time_steps):
                    for f_idx, feature_name in enumerate(feature_columns):
                        m.nn_constraints.add(
                            m.nn_input[k, i, t, f_idx]
                            == self._normalized_nn_feature_expression(
                                generator_name,
                                feature_name,
                                k,
                                t,
                                i,
                            )
                        )

                    previous_values = [
                        m.nn_input[k, i, t, f_idx]
                        for f_idx in range(len(feature_columns))
                    ]
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
                                m.nn_constraints.add(m.nn_output[k, i, t, node] == expr)
                                current_values.append(m.nn_output[k, i, t, node])
                                continue

                            m.nn_constraints.add(
                                m.nn_z[k, i, t, linear_idx, node] == expr
                            )
                            h = m.nn_h[k, i, t, linear_idx, node]
                            z = m.nn_z[k, i, t, linear_idx, node]
                            delta = m.nn_delta[k, i, t, linear_idx, node]
                            bounds = relu_bound_for_node(
                                relu_bounds,
                                int(k),
                                int(t),
                                int(linear_idx),
                                int(node),
                                generator_name,
                            )
                            status = str(bounds["status"]).lower()
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
                for k in m.scenarios:
                    for t in m.time_steps:
                        m.nn_constraints.add(
                            m.alpha[int(k), i, int(local_block), int(t)]
                            == m.nn_output[int(k), i, int(t), int(output_idx)]
                        )

    # ------------------------------------------------------------------
    # Applying regime-wide tightening reports
    # ------------------------------------------------------------------

    def apply_regime_wide_tightening_to_model(
        self,
        report: Optional[dict[str, Any]] = None,
        apply_alpha_bounds: bool = True,
        apply_fixed_binaries: bool = True,
        apply_dual_bounds: bool = True,
        apply_relu_bounds: bool = True,
    ) -> dict[str, Any]:
        """
        Apply DRO tightening to the Pyomo model.

        Scenario-indexed keys are applied to their exact scenario copy k.
        Regime-wide keys are still accepted and broadcast across scenarios.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        current_report = getattr(self, "tightening_report", None)
        report = report or current_report
        if not report:
            raise ValueError(
                "No DRO regime-wide tightening report loaded. "
                "Call load_regime_wide_tightening_report() first."
            )
        if report is not current_report:
            self._set_regime_wide_tightening_report_data(
                report,
                getattr(self, "tightening_report_path", None),
            )
        self._prepare_loaded_bounds()

        stats = {
            "alpha_bounds": 0,
            "fixed_binaries": 0,
            "lambda_bounds": 0,
            "dual_upper_bounds": 0,
            "aggregate_dual_bounds": 0,
            "relu_bounds": {},
        }
        if apply_alpha_bounds:
            stats["alpha_bounds"] = self._apply_alpha_bounds_to_model(report)
        if apply_fixed_binaries:
            stats["fixed_binaries"] = self._apply_fixed_binaries_to_model(report)
        if apply_dual_bounds:
            stats["lambda_bounds"] = self._apply_lambda_bounds_to_model()
            stats["dual_upper_bounds"] = self._apply_dual_bounds_to_model()
            self.aggregate_dual_bounds = report.get("aggregate_dual_bounds", {}) or {}
            stats["aggregate_dual_bounds"] = (
                self._refresh_aggregate_dual_bound_constraints()
            )
        if apply_relu_bounds:
            stats["relu_bounds"] = self._apply_relu_bounds_to_model(report)
        self.applied_tightening_stats = stats
        return stats

    def _apply_alpha_bounds_to_model(self, report: dict[str, Any]) -> int:
        m = self.model
        alpha_var = getattr(m, "alpha", None)
        if alpha_var is None:
            return 0
        applied = 0
        alpha_bounds = report.get("scenario_alpha_bounds", {}) or report.get(
            "alpha_bounds",
            {},
        )
        for key, bounds in (alpha_bounds or {}).items():
            index = self._parse_json_index(key)
            if len(index) not in {3, 4}:
                raise ValueError(f"Alpha-bound key '{key}' must have format i,b,t or k,i,b,t")
            lower = float(bounds["lower"])
            upper = float(bounds["upper"])
            scenario_indices = [index] if len(index) == 4 else [
                (int(k), *index) for k in m.scenarios
            ]
            for scenario_index in scenario_indices:
                if scenario_index not in alpha_var:
                    continue
                current_lb = alpha_var[scenario_index].lb
                current_ub = alpha_var[scenario_index].ub
                new_lower = max(float(current_lb), lower) if current_lb is not None else lower
                new_upper = min(float(current_ub), upper) if current_ub is not None else upper
                if new_lower <= new_upper:
                    alpha_var[scenario_index].setlb(new_lower)
                    alpha_var[scenario_index].setub(new_upper)
                    applied += 1
        return applied

    def _apply_fixed_binaries_to_model(self, report: dict[str, Any]) -> int:
        m = self.model
        applied = 0
        fixed_binaries = report.get("scenario_fixed_binaries", {}) or report.get(
            "fixed_binaries",
            {},
        )
        for var_name, entries in (fixed_binaries or {}).items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, details in (entries or {}).items():
                index = self._parse_json_index(key)
                fixed_value = int((details or {}).get("fixed_value", 0))
                expected_without_k = binary_var.dim() - 1
                if len(index) == binary_var.dim():
                    if index in binary_var:
                        binary_var[index].fix(fixed_value)
                        applied += 1
                    continue
                if len(index) != expected_without_k:
                    raise ValueError(
                        f"Fixed-binary key '{key}' for {var_name} must have "
                        f"{binary_var.dim()} scenario-indexed indices or omit k and "
                        f"have {expected_without_k} indices."
                    )
                for k in m.scenarios:
                    scenario_index = (int(k), *index)
                    if scenario_index in binary_var:
                        binary_var[scenario_index].fix(fixed_value)
                        applied += 1
        return applied

    def _apply_lambda_bounds_to_model(self) -> int:
        m = self.model
        applied = 0
        lambda_bound_maps = {
            "lambda_eq": self.lambda_eq_bounds,
            "lambda_opt": self.lambda_opt_bounds,
        }
        for lambda_name, bound_map in lambda_bound_maps.items():
            lambda_var = getattr(m, lambda_name, None)
            if lambda_var is None:
                continue
            for k in m.scenarios:
                for t in m.time_steps:
                    lower, upper = self._scenario_or_regime_lambda_bounds(
                        bound_map,
                        int(k),
                        int(t),
                    )
                    index = (int(k), int(t))
                    current_lb = lambda_var[index].lb
                    current_ub = lambda_var[index].ub
                    new_lower = max(float(current_lb), lower) if current_lb is not None else lower
                    new_upper = min(float(current_ub), upper) if current_ub is not None else upper
                    if new_lower <= new_upper:
                        lambda_var[index].setlb(new_lower)
                        lambda_var[index].setub(new_upper)
                        applied += 1
        return applied

    def _apply_dual_bounds_to_model(self) -> int:
        m = self.model
        applied = 0
        dual_bound_maps: tuple[tuple[str, dict[Any, float]], ...] = (
            ("mu_upper_eq", self.M_mu_upper_eq),
            ("mu_lower_eq", self.M_mu_lower_eq),
            ("mu_ramp_up_eq", self.M_mu_ramp_up_eq),
            ("mu_ramp_down_eq", self.M_mu_ramp_down_eq),
            ("mu_upper_opt", self.M_mu_upper_opt),
            ("mu_lower_opt", self.M_mu_lower_opt),
            ("mu_ramp_up_opt", self.M_mu_ramp_up_opt),
            ("mu_ramp_down_opt", self.M_mu_ramp_down_opt),
        )
        for var_name, bound_map in dual_bound_maps:
            dual_var = getattr(m, var_name, None)
            if dual_var is None:
                continue
            for raw_index, upper in bound_map.items():
                index = tuple(int(part) for part in raw_index)
                scenario_indices = (
                    [index]
                    if len(index) == dual_var.dim()
                    else [(int(k), *index) for k in m.scenarios]
                )
                for scenario_index in scenario_indices:
                    if scenario_index not in dual_var:
                        continue
                    current_ub = dual_var[scenario_index].ub
                    new_ub = max(0.0, float(upper))
                    if current_ub is not None:
                        new_ub = min(float(current_ub), new_ub)
                    dual_var[scenario_index].setub(new_ub)
                    applied += 1
        for var_name in (
            "mu_ramp_up_eq",
            "mu_ramp_down_eq",
            "mu_ramp_up_opt",
            "mu_ramp_down_opt",
        ):
            dual_var = getattr(m, var_name, None)
            if dual_var is None:
                continue
            for k in m.scenarios:
                for i in m.physical_generators:
                    index = (int(k), int(i), self.num_time_steps)
                    if index in dual_var:
                        dual_var[index].setub(0.0)
        return applied

    def _apply_relu_bounds_to_model(self, report: dict[str, Any]) -> dict[str, int]:
        relu_report = report.get("nn_relu_bounds_report", {}) or {}
        if not relu_report and (
            "nn_relu_bounds" in report or "scenario_nn_relu_bounds" in report
        ):
            relu_report = report
        stats = {
            "z_bounds_applied": 0,
            "h_bounds_applied": 0,
            "delta_fixed_active": 0,
            "delta_fixed_inactive": 0,
            "delta_left_ambiguous": 0,
        }
        if not relu_report:
            return stats
        self._set_nn_relu_bounds_from_report(relu_report)

        m = self.model
        if not hasattr(m, "nn_z") or not hasattr(m, "nn_h") or not hasattr(m, "nn_delta"):
            return stats

        for physical_generator_idx in self.nn_policy_generator_ids:
            i = int(physical_generator_idx)
            generator_name = self.physical_generator_names[i]
            for raw_key, bounds in (self.nn_relu_bounds.get(generator_name, {}) or {}).items():
                key = tuple(int(part) for part in raw_key)
                scenario_keys = (
                    [key]
                    if len(key) == 4
                    else [(int(k), *key) for k in m.scenarios]
                )
                for scenario_key in scenario_keys:
                    k, time_idx, linear_idx, node = scenario_key
                    index = (int(k), i, int(time_idx), int(linear_idx), int(node))
                    if index in m.nn_z:
                        m.nn_z[index].setlb(float(bounds["L"]))
                        m.nn_z[index].setub(float(bounds["U"]))
                        stats["z_bounds_applied"] += 1
                    if index in m.nn_h:
                        m.nn_h[index].setlb(float(bounds["h_lower"]))
                        m.nn_h[index].setub(float(bounds["h_upper"]))
                        stats["h_bounds_applied"] += 1
                    if index not in m.nn_delta:
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

    def compute_optimal_cost_bounds(
        self,
        output_path: str | Path = "results/dro_poa_tightening/optimal_cost_bounds_report.json",
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        from models.DRO_PoA.DRO_PoA_tightening.compute_optimal_cost_bounds import (
            DROOptimalCostBoundsComputer,
        )

        stage = DROOptimalCostBoundsComputer.__new__(DROOptimalCostBoundsComputer)
        stage.poa = self
        stage.dro = self
        stage.dro_poa = self
        stage.tightening_data = {
            "primal_big_m": getattr(self, "primal_big_m", {}) or {},
            "optimal_cost_bounds": getattr(self, "optimal_cost_bounds", {}) or {},
            "scenario_optimal_cost_bounds": getattr(
                self,
                "scenario_optimal_cost_bounds",
                {},
            ) or {},
            "optimal_cost_bound_optimization_results": getattr(
                self,
                "optimal_cost_bound_optimization_results",
                {},
            ) or {},
        }
        stage.stage_reports = {}
        report = stage.run_optimal_cost_bounds(
            output_path=output_path,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        self._set_optimal_cost_bounds_from_report(report)
        return report

    def make_ratio_bounds_from_optimal_cost_bounds(
        self,
        phi_bounds: tuple[float, float],
        num_pieces: Optional[int] = None,
        C_opt_breakpoints: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        if num_pieces is not None and C_opt_breakpoints is not None:
            raise ValueError("Provide either num_pieces or C_opt_breakpoints, not both")
        C_opt_bounds = (self.optimal_cost_bounds or {}).get("C_opt", {})
        lower = C_opt_bounds.get("lower")
        upper = C_opt_bounds.get("upper")
        if lower is None or upper is None:
            raise ValueError(
                "Optimal cost bounds are not loaded. Run compute_optimal_cost_bounds() "
                "or load_optimal_cost_bounds_report() first."
            )
        ratio_bounds: dict[str, Any] = {
            "phi": phi_bounds,
            "C_opt": (float(lower), float(upper)),
        }
        if num_pieces is not None:
            ratio_bounds["num_pieces"] = int(num_pieces)
        if C_opt_breakpoints is not None:
            ratio_bounds["C_opt_breakpoints"] = [
                float(value) for value in C_opt_breakpoints
            ]
        return ratio_bounds

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
            self._safe_value(var[(*leading_indices, t)])
            for t in range(self.num_time_steps)
        ]

    def _physical_capacity_profile_values(self, k: int, i: int) -> list[Optional[float]]:
        m = self.model
        return [
            self._safe_value(
                sum(
                    m.P_max_block[k, i, b, t]
                    for b in self.local_blocks_by_generator[int(i)]
                )
            )
            for t in range(self.num_time_steps)
        ]

    def _physical_dispatch_profile_values(self, var: Any, k: int, i: int) -> list[Optional[float]]:
        return [
            self._safe_value(
                sum(var[k, i, b, t] for b in self.local_blocks_by_generator[int(i)])
            )
            for t in range(self.num_time_steps)
        ]

    def _json_serializable_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {str(key): self._json_serializable_payload(value) for key, value in payload.items()}
        if isinstance(payload, (list, tuple)):
            return [self._json_serializable_payload(value) for value in payload]
        if isinstance(payload, np.generic):
            return payload.item()
        if isinstance(payload, (str, int, float, bool)) or payload is None:
            return payload
        return str(payload)

    def _json_serializable_ratio_bounds(self) -> Optional[dict[str, Any]]:
        if self.ratio_bounds is None:
            return None
        return self._json_serializable_payload(self.ratio_bounds)

    def extract_results(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        inner_objective = self._safe_value(m.objective)
        ratio_mode = self.objective_mode in {
            "ratio_mccormick",
            "ratio_piecewise_mccormick",
        }
        poa_values = [self._safe_value(m.PoA[k]) for k in m.scenarios]
        wasserstein_values = [self._safe_value(m.wasserstein_distance[k]) for k in m.scenarios]
        average_poa = (
            float(np.mean([v for v in poa_values if v is not None]))
            if any(v is not None for v in poa_values)
            else None
        )
        average_wasserstein = (
            float(np.mean([v for v in wasserstein_values if v is not None]))
            if any(v is not None for v in wasserstein_values)
            else None
        )
        scenario_ratio_metrics: dict[int, dict[str, Optional[float]]] = {}
        poa_ratio_values: list[float] = []
        relaxed_phi_values: list[float] = []
        for k in range(self.num_empirical_scenarios):
            C_eq = self._safe_value(m.C_eq[k])
            C_opt = self._safe_value(m.C_opt[k])
            poa_ratio = (
                C_eq / C_opt
                if C_eq is not None and C_opt not in (None, 0.0)
                else None
            )
            relaxed_phi = self._safe_value(m.phi[k]) if ratio_mode else None
            ratio_relaxation_gap = (
                relaxed_phi - poa_ratio
                if relaxed_phi is not None and poa_ratio is not None
                else None
            )
            if poa_ratio is not None:
                poa_ratio_values.append(float(poa_ratio))
            if relaxed_phi is not None:
                relaxed_phi_values.append(float(relaxed_phi))
            scenario_ratio_metrics[k] = {
                "C_eq": C_eq,
                "C_opt": C_opt,
                "PoA_ratio": poa_ratio,
                "relaxed_phi": relaxed_phi,
                "ratio_relaxation_gap": ratio_relaxation_gap,
            }
        average_poa_ratio = (
            float(np.mean(poa_ratio_values)) if poa_ratio_values else None
        )
        average_relaxed_phi = (
            float(np.mean(relaxed_phi_values))
            if ratio_mode and relaxed_phi_values
            else None
        )

        scenario_results = []
        for k in range(self.num_empirical_scenarios):
            ratio_metrics = scenario_ratio_metrics[k]
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
                            "capacity_profile": self._profile_values(m.P_max_block, k, i, b),
                            "alpha_profile": self._profile_values(m.alpha, k, i, b),
                            "equilibrium_dispatch": self._profile_values(m.P_eq, k, i, b),
                            "optimal_dispatch": self._profile_values(m.P_opt, k, i, b),
                            "true_cost": float(self.block_cost_vector[global_block]),
                        }
                    )
                generators[generator_name] = {
                    "physical_generator_index": int(i),
                    "is_wind": i in self.wind_physical_generator_ids,
                    "empirical_physical_capacity_profile": list(
                        self.empirical_Pmax_phys[k][i]
                    ),
                    "optimized_physical_capacity_profile": self._physical_capacity_profile_values(k, i),
                    "equilibrium_physical_dispatch": self._physical_dispatch_profile_values(
                        m.P_eq,
                        k,
                        i,
                    ),
                    "optimal_physical_dispatch": self._physical_dispatch_profile_values(
                        m.P_opt,
                        k,
                        i,
                    ),
                    "blocks": block_results,
                }

            scenario_results.append(
                {
                    "k": int(k),
                    "scenario_id": self.empirical_scenario_ids[k],
                    "empirical_demand_profile": list(self.empirical_D[k]),
                    "optimized_demand_profile": self._profile_values(m.D, k),
                    "empirical_physical_capacity_profiles": {
                        self.physical_generator_names[i]: list(self.empirical_Pmax_phys[k][i])
                        for i in range(self.num_physical_generators)
                    },
                    "optimized_physical_capacity_profiles": {
                        self.physical_generator_names[i]: self._physical_capacity_profile_values(k, i)
                        for i in range(self.num_physical_generators)
                    },
                    "wasserstein_distance": self._safe_value(m.wasserstein_distance[k]),
                    "C_eq": ratio_metrics["C_eq"],
                    "C_opt": ratio_metrics["C_opt"],
                    "PoA_difference": self._safe_value(m.PoA[k]),
                    "PoA_ratio": ratio_metrics["PoA_ratio"],
                    "relaxed_phi": ratio_metrics["relaxed_phi"],
                    "ratio_relaxation_gap": ratio_metrics["ratio_relaxation_gap"],
                    "equilibrium_price_profile": self._profile_values(m.lambda_eq, k),
                    "optimal_price_profile": self._profile_values(m.lambda_opt, k),
                    "generators": generators,
                }
            )

        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(
                    self.solver_results.solver.termination_condition
                ),
            }

        dro_objective_with_epsilon = (
            inner_objective + self.eta * self.epsilon
            if inner_objective is not None
            else None
        )
        return {
            "reference_case": self.reference_case,
            "regime_set": self.regime_set,
            "regime_name": self.regime_name,
            "num_time_steps": self.num_time_steps,
            "num_empirical_scenarios": self.num_empirical_scenarios,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "objective_mode": self.objective_mode,
            "ratio_bounds": self._json_serializable_ratio_bounds(),
            "optimal_cost_bounds": self._json_serializable_payload(
                self.optimal_cost_bounds
            ),
            "scenario_optimal_cost_bounds": self._json_serializable_payload(
                self.scenario_optimal_cost_bounds
            ),
            "optimal_cost_bound_optimization_results": self._json_serializable_payload(
                self.optimal_cost_bound_optimization_results
            ),
            "inner_objective": inner_objective,
            "dro_objective_with_epsilon": dro_objective_with_epsilon,
            "average_poa_difference": average_poa,
            "average_poa_ratio": average_poa_ratio,
            "average_relaxed_phi": average_relaxed_phi,
            "average_wasserstein_distance": average_wasserstein,
            "solver": solver_summary,
            "selected_regime_parameters": dict(self.selected_regime_parameters),
            "demand_shape": list(self.demand_shape),
            "wind_shape": list(self.wind_shape),
            "block_names": list(self.block_names),
            "physical_generator_names": list(self.physical_generator_names),
            "block_to_physical": dict(self.block_to_physical),
            "physical_to_block_indices": {
                str(i): list(blocks)
                for i, blocks in enumerate(self.physical_to_block_indices)
            },
            "policy_type": (
                "nn_policy"
                if self.nn_policy_generator_ids
                else "true_cost_baseline"
            ),
            "scenarios": scenario_results,
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

    def solution_summary(self) -> dict[str, Any]:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        m = self.model
        inner_objective = self._safe_value(m.objective)
        average_poa = self._safe_value(
            sum(m.PoA[k] for k in m.scenarios) / self.num_empirical_scenarios
        )
        ratio_mode = self.objective_mode in {
            "ratio_mccormick",
            "ratio_piecewise_mccormick",
        }
        poa_ratio_values = []
        relaxed_phi_values = []
        for k in m.scenarios:
            C_eq = self._safe_value(m.C_eq[k])
            C_opt = self._safe_value(m.C_opt[k])
            if C_eq is not None and C_opt not in (None, 0.0):
                poa_ratio_values.append(C_eq / C_opt)
            if ratio_mode:
                relaxed_phi = self._safe_value(m.phi[k])
                if relaxed_phi is not None:
                    relaxed_phi_values.append(relaxed_phi)
        average_poa_ratio = (
            float(np.mean(poa_ratio_values)) if poa_ratio_values else None
        )
        average_relaxed_phi = (
            float(np.mean(relaxed_phi_values))
            if ratio_mode and relaxed_phi_values
            else None
        )
        average_wasserstein = self._safe_value(
            sum(m.wasserstein_distance[k] for k in m.scenarios)
            / self.num_empirical_scenarios
        )
        solver_summary: dict[str, Any] = {}
        if hasattr(self, "solver_results"):
            solver_summary = {
                "status": str(self.solver_results.solver.status),
                "termination_condition": str(
                    self.solver_results.solver.termination_condition
                ),
            }
        return {
            "reference_case": self.reference_case,
            "regime_set": self.regime_set,
            "regime_name": self.regime_name,
            "num_time_steps": self.num_time_steps,
            "num_empirical_scenarios": self.num_empirical_scenarios,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "objective_mode": self.objective_mode,
            "ratio_bounds": self._json_serializable_ratio_bounds(),
            "optimal_cost_bounds": self._json_serializable_payload(
                self.optimal_cost_bounds
            ),
            "scenario_optimal_cost_bounds": self._json_serializable_payload(
                self.scenario_optimal_cost_bounds
            ),
            "optimal_cost_bound_optimization_results": self._json_serializable_payload(
                self.optimal_cost_bound_optimization_results
            ),
            "inner_objective": inner_objective,
            "dro_objective_with_epsilon": (
                inner_objective + self.eta * self.epsilon
                if inner_objective is not None
                else None
            ),
            "average_poa_difference": average_poa,
            "average_poa_ratio": average_poa_ratio,
            "average_relaxed_phi": average_relaxed_phi,
            "average_wasserstein_distance": average_wasserstein,
            "solver": solver_summary,
        }


def load_regime_scenarios(
    reference_case: str = "test_case_bidding_blocks",
    regime_config_path: str | Path = "config/regime_definitions.yaml",
    regime_set: str = "PoA_analysis",
    seed: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return DRO_PoAOptimization.load_regime_scenarios(
        reference_case=reference_case,
        regime_config_path=regime_config_path,
        regime_set=regime_set,
        seed=seed,
    )


def run_eta_sweep_by_regime(
    etas: list[float],
    regimes: Optional[list[str]] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
    costs_df: Optional[pd.DataFrame] = None,
    ramps_df: Optional[pd.DataFrame] = None,
    reference_case: str = "test_case_bidding_blocks",
    regime_config_path: str | Path = "config/regime_definitions.yaml",
    regime_set: str = "PoA_analysis",
    epsilon: float = 0.0,
    num_time_steps: Optional[int] = None,
    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    objective_mode: str = "difference",
    ratio_bounds: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    if scenarios_df is None or costs_df is None or ramps_df is None:
        scenarios_df, costs_df, ramps_df = load_regime_scenarios(
            reference_case=reference_case,
            regime_config_path=regime_config_path,
            regime_set=regime_set,
            seed=seed,
        )

    if regimes is None:
        if "regime" not in scenarios_df.columns:
            raise ValueError("regimes must be provided when scenarios_df has no 'regime' column")
        regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())

    summaries: list[dict[str, Any]] = []
    for regime_name in regimes:
        for eta in etas:
            optimizer = DRO_PoAOptimization(
                scenarios_df=scenarios_df,
                costs_df=costs_df,
                ramps_df=ramps_df,
                p_init=None,
                num_time_steps=num_time_steps,
                regime_config_path=regime_config_path,
                regime_set=regime_set,
                regime_name=regime_name,
                eta=float(eta),
                epsilon=float(epsilon),
                nn_model_dir=None,
                reference_case=reference_case,
                objective_mode=objective_mode,
                ratio_bounds=ratio_bounds,
            )
            optimizer.build_model()
            optimizer.solve(time_limit=time_limit)
            summaries.append(optimizer.solution_summary())

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    case = "test_case_bidding_blocks"
    regime_set = "PoA_analysis"
    regime_name = "normal"
    seed = 1
    eta = 0.5
    epsilon = 0.0
    horizon = 4

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )
    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    optimizer = DRO_PoAOptimization(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        p_init=None,
        num_time_steps=horizon,
        regime_config_path="config/regime_definitions.yaml",
        regime_set=regime_set,
        regime_name=regime_name,
        eta=eta,
        epsilon=epsilon,
        nn_model_dir=None,
        reference_case=case,
        # objective_mode="ratio_mccormick",
        # ratio_bounds={
        #     "phi": (1.0, 5.0),
        #     "C_opt": (1000.0, 20000.0),
        # },
        # objective_mode="ratio_piecewise_mccormick",
        # ratio_bounds={
        #     "phi": (1.0, 5.0),
        #     "C_opt": (1000.0, 20000.0),
        #     "num_pieces": 4,
        # },
    )

    start = time.perf_counter()
    optimizer.build_model()
    optimizer.solve(time_limit=400)
    result_path = optimizer.save_results(
        "results/dro_poa/dro_poa_optimization_results.json"
    )
    elapsed = time.perf_counter() - start

    print("\nDRO PoA solve complete")
    print(f"  Regime: {regime_set}/{regime_name}")
    print(f"  Eta: {eta}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Results: {result_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
