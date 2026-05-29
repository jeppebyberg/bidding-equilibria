from __future__ import annotations

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
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.helper import (
    ensure_profile,
    find_demand_profile_column,
    infer_num_time_steps,
)
from models.DRO_PoA.dro_poa_model.nn_policy_embedding import DROPoAPolicyEmbedding
from models.DRO_PoA.dro_poa_model.results import DROPoAResults
from models.DRO_PoA.dro_poa_model.support_set import DROPoASupportSet
from models.DRO_PoA.dro_poa_model.tightening_reports import DROPoATighteningReports
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel


class DRO_PoAOptimization(
    DROPoAPolicyEmbedding,
    DROPoATighteningReports,
    DROPoASupportSet,
    DROPoAResults,
):
    """
    Scenario-indexed distributionally robust Price of Anarchy optimization.

    The model keeps the block-aware physical-generator/local-block indexing of
    PoAOptimization and adds an empirical scenario index k as the first index on
    state, dispatch, KKT, policy, and PoA variables.
    """

    normalization_epsilon = 1e-12
    DEFAULT_LOOSE_RELU_BOUND = 1e4
    DEFAULT_LOOSE_ALPHA_LOWER = 0.0
    DEFAULT_LOOSE_ALPHA_UPPER = 1e4
    DEFAULT_LOOSE_DUAL_BIG_M = 1e6
    DEFAULT_LOOSE_LAMBDA_LOWER = 0
    DEFAULT_LOOSE_LAMBDA_UPPER = 50
    DEFAULT_LOOSE_AGGREGATE_DUAL_UPPER = 1e8
    DEFAULT_LOOSE_C_OPT_LOWER = 1e-3
    DEFAULT_LOOSE_C_OPT_UPPER = 1e8
    DEFAULT_PoA_LOWER = 1.0
    DEFAULT_PoA_UPPER = 10.0
    DEFAULT_PHI_LOWER = DEFAULT_PoA_LOWER
    DEFAULT_PHI_UPPER = DEFAULT_PoA_UPPER

    normalization_epsilon = 1e-12
    allowed_objective_modes = {
        "difference",
        "mccormick",
        "piecewise_mccormick",
    }

    mccormick_bounds_tolerance = 1e-7

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
        reference_case: str = "base_test_case",
        objective_mode: str = "difference",
        mccormick_bounds: Optional[dict[str, Any]] = None,
        ratio_bounds: Optional[dict[str, Any]] = None,
        ambiguity_kappa: float = 0.3,
        defer_mccormick_bound_validation: bool = False,
        use_default_bounds: bool = False,
        default_relu_bound: float = DEFAULT_LOOSE_RELU_BOUND,
        default_alpha_lower: float = DEFAULT_LOOSE_ALPHA_LOWER,
        default_alpha_upper: float = DEFAULT_LOOSE_ALPHA_UPPER,
        default_dual_big_m: float = DEFAULT_LOOSE_DUAL_BIG_M,
        default_lambda_lower: float = DEFAULT_LOOSE_LAMBDA_LOWER,
        default_lambda_upper: float = DEFAULT_LOOSE_LAMBDA_UPPER,
        default_aggregate_dual_upper: float = DEFAULT_LOOSE_AGGREGATE_DUAL_UPPER,
        default_c_opt_lower: float = DEFAULT_LOOSE_C_OPT_LOWER,
        default_c_opt_upper: float = DEFAULT_LOOSE_C_OPT_UPPER,
        default_PoA_lower: float = DEFAULT_PoA_LOWER,
        default_PoA_upper: float = DEFAULT_PoA_UPPER,
        default_phi_lower: Optional[float] = None,
        default_phi_upper: Optional[float] = None,
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
        self.ambiguity_kappa = float(ambiguity_kappa)
        self.reference_case = reference_case
        self.use_default_bounds = bool(use_default_bounds)
        self.default_relu_bound = float(default_relu_bound)
        self.default_alpha_lower = float(default_alpha_lower)
        self.default_alpha_upper = float(default_alpha_upper)
        self.default_dual_big_m = float(default_dual_big_m)
        self.default_lambda_lower = float(default_lambda_lower)
        self.default_lambda_upper = float(default_lambda_upper)
        self.default_aggregate_dual_upper = float(default_aggregate_dual_upper)
        self.default_c_opt_lower = float(default_c_opt_lower)
        self.default_c_opt_upper = float(default_c_opt_upper)
        self.default_PoA_lower = float(
            default_PoA_lower if default_phi_lower is None else default_phi_lower
        )
        self.default_PoA_upper = float(
            default_PoA_upper if default_phi_upper is None else default_phi_upper
        )
        if self.default_c_opt_lower <= 0.0:
            raise ValueError("default_c_opt_lower must be strictly positive")
        if self.default_c_opt_upper < self.default_c_opt_lower:
            raise ValueError("default_c_opt_upper must be >= default_c_opt_lower")
        if self.default_lambda_lower >= self.default_lambda_upper:
            raise ValueError("default_lambda_lower must be < default_lambda_upper")
        self.lambda_bound = max(
            abs(float(self.default_lambda_lower)),
            abs(float(self.default_lambda_upper)),
        )
        self.capacity_dual_bound = float(self.default_dual_big_m)
        self.ramp_dual_bound = float(self.default_dual_big_m)
        self.primal_big_m_placeholder = float(self.default_dual_big_m)
        self.default_bounds_used: dict[str, Any] = self._empty_default_bounds_used()
        self.objective_mode = self._validate_objective_mode(objective_mode)
        self.defer_mccormick_bound_validation = bool(defer_mccormick_bound_validation)
        mccormick_bounds = self._normalize_mccormick_bounds_alias(
            mccormick_bounds,
            ratio_bounds,
        )
        mccormick_bounds = self._mccormick_bounds_with_defaults(mccormick_bounds)
        self._raw_mccormick_bounds = mccormick_bounds
        if (
            self.defer_mccormick_bound_validation
            and self.objective_mode in {
                "mccormick",
                "piecewise_mccormick",
            }
            and (mccormick_bounds is None or "C_opt" not in mccormick_bounds)
        ):
            self.mccormick_bounds = self._validate_deferred_mccormick_bounds(mccormick_bounds)
        else:
            self.mccormick_bounds = self._validate_mccormick_bounds(mccormick_bounds)
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

    def _normalize_mccormick_bounds_alias(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
        ratio_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        raw_bounds = mccormick_bounds if mccormick_bounds is not None else ratio_bounds
        if raw_bounds is None or not isinstance(raw_bounds, dict):
            return raw_bounds
        normalized = dict(raw_bounds)
        if "PoA" not in normalized and "phi" in normalized:
            normalized["PoA"] = normalized["phi"]
        return normalized

    def _default_mccormick_bounds_payload(
        self,
        mccormick_bounds: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        completed = dict(mccormick_bounds or {})
        completed.setdefault(
            "PoA",
            (float(self.default_PoA_lower), float(self.default_PoA_upper)),
        )
        completed.setdefault(
            "C_opt",
            (float(self.default_c_opt_lower), float(self.default_c_opt_upper)),
        )
        if self.objective_mode == "piecewise_mccormick":
            completed.setdefault("num_pieces", 10)
        return completed

    def _mccormick_bounds_with_defaults(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return mccormick_bounds
        if mccormick_bounds is None:
            if not self.use_default_bounds:
                return None
            self._mark_default_bound_used(
                "optimal_cost_bounds",
                "Default loose mccormick/C_opt bounds were used because mccormick_bounds was missing.",
            )
            return self._default_mccormick_bounds_payload()
        if self.use_default_bounds and (
            "C_opt" not in mccormick_bounds or "PoA" not in mccormick_bounds
        ):
            self._mark_default_bound_used(
                "optimal_cost_bounds",
                "Default loose mccormick/C_opt bounds filled missing mccormick_bounds entries.",
            )
            return self._default_mccormick_bounds_payload(mccormick_bounds)
        return mccormick_bounds

    def _validate_deferred_mccormick_bounds(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds with at least 'PoA' is required when "
                f"objective_mode='{self.objective_mode}' and "
                "defer_mccormick_bound_validation=True"
            )
        if not isinstance(mccormick_bounds, dict):
            raise ValueError("mccormick_bounds must be a dictionary")
        mccormick_bounds = self._normalize_mccormick_bounds_alias(mccormick_bounds, None)
        if "PoA" not in mccormick_bounds:
            raise ValueError("mccormick_bounds must contain 'PoA' for deferred mccormick validation")
        raw_PoA = mccormick_bounds["PoA"]
        if not isinstance(raw_PoA, (list, tuple)) or len(raw_PoA) != 2:
            raise ValueError("mccormick_bounds['PoA'] must be a pair (lower, upper)")
        PoA_L = float(raw_PoA[0])
        PoA_U = float(raw_PoA[1])
        if not np.isfinite(PoA_L) or not np.isfinite(PoA_U):
            raise ValueError("mccormick_bounds['PoA'] entries must be finite")
        if PoA_L < 0.0:
            raise ValueError("mccormick_bounds['PoA'][0] must be nonnegative")
        if PoA_U <= PoA_L:
            raise ValueError(
                "mccormick_bounds['PoA'][1] must be greater than mccormick_bounds['PoA'][0]"
            )
        if self.objective_mode == "piecewise_mccormick":
            if "num_pieces" not in mccormick_bounds and "C_opt_breakpoints" not in mccormick_bounds:
                raise ValueError(
                    "deferred piecewise_mccormick bounds must include "
                    "'num_pieces' or 'C_opt_breakpoints'"
                )
            if "num_pieces" in mccormick_bounds:
                try:
                    num_pieces = int(mccormick_bounds["num_pieces"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("mccormick_bounds['num_pieces'] must be an integer") from exc
                if num_pieces < 2:
                    raise ValueError("mccormick_bounds['num_pieces'] must be at least 2")
        return dict(mccormick_bounds)

    def _validate_mccormick_bounds(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return mccormick_bounds

        if mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds is required when objective_mode='{self.objective_mode}'"
            )
        if not isinstance(mccormick_bounds, dict):
            raise ValueError("mccormick_bounds must be a dictionary")
        mccormick_bounds = self._normalize_mccormick_bounds_alias(mccormick_bounds, None)
        missing = [key for key in ("PoA", "C_opt") if key not in mccormick_bounds]
        if missing:
            raise ValueError(
                "mccormick_bounds must contain bounds for: " + ", ".join(missing)
            )

        def parse_bounds(key: str) -> tuple[float, float]:
            raw_bounds = mccormick_bounds[key]
            if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
                raise ValueError(f"mccormick_bounds['{key}'] must be a pair (lower, upper)")
            lower = float(raw_bounds[0])
            upper = float(raw_bounds[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"mccormick_bounds['{key}'] entries must be finite")
            return lower, upper

        PoA_L, PoA_U = parse_bounds("PoA")
        C_opt_L, C_opt_U = parse_bounds("C_opt")

        if C_opt_L <= 0.0:
            raise ValueError("mccormick_bounds['C_opt'][0] must be strictly positive")
        if C_opt_U < C_opt_L:
            raise ValueError(
                "mccormick_bounds['C_opt'][1] must be greater than or equal to "
                "mccormick_bounds['C_opt'][0]"
            )
        if PoA_L < 0.0:
            raise ValueError("mccormick_bounds['PoA'][0] must be nonnegative")
        if PoA_U <= PoA_L:
            raise ValueError(
                "mccormick_bounds['PoA'][1] must be greater than mccormick_bounds['PoA'][0]"
            )

        validated_bounds: dict[str, Any] = {
            "PoA": (PoA_L, PoA_U),
            "C_opt": (C_opt_L, C_opt_U),
        }
        if self.objective_mode == "piecewise_mccormick":
            validated_bounds["C_opt_breakpoints"] = self._validate_mccormick_breakpoints(
                mccormick_bounds,
                C_opt_L,
                C_opt_U,
            )
            validated_bounds["num_pieces"] = (
                len(validated_bounds["C_opt_breakpoints"]) - 1
            )
        return validated_bounds

    def _validate_mccormick_breakpoints(
        self,
        mccormick_bounds: dict[str, Any],
        C_opt_L: float,
        C_opt_U: float,
    ) -> list[float]:
        tolerance = self.mccormick_bounds_tolerance
        if "C_opt_breakpoints" in mccormick_bounds:
            raw_breakpoints = mccormick_bounds["C_opt_breakpoints"]
            if not isinstance(raw_breakpoints, (list, tuple)):
                raise ValueError("mccormick_bounds['C_opt_breakpoints'] must be a list")
            breakpoints = [float(value) for value in raw_breakpoints]
            if len(breakpoints) < 3:
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'] must contain at least 3 values"
                )
            if not all(np.isfinite(value) for value in breakpoints):
                raise ValueError("mccormick_bounds['C_opt_breakpoints'] entries must be finite")
            if abs(breakpoints[0] - C_opt_L) > tolerance:
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'][0] must match "
                    "mccormick_bounds['C_opt'][0]"
                )
            if abs(breakpoints[-1] - C_opt_U) > tolerance:
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'][-1] must match "
                    "mccormick_bounds['C_opt'][1]"
                )
            if any(
                breakpoints[idx + 1] <= breakpoints[idx]
                for idx in range(len(breakpoints) - 1)
            ):
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'] must be strictly increasing"
                )
            return breakpoints

        if "num_pieces" not in mccormick_bounds:
            raise ValueError(
                "mccormick_bounds must include either 'num_pieces' or "
                "'C_opt_breakpoints' for objective_mode='piecewise_mccormick'"
            )
        try:
            num_pieces = int(mccormick_bounds["num_pieces"])
        except (TypeError, ValueError) as exc:
            raise ValueError("mccormick_bounds['num_pieces'] must be an integer") from exc
        if num_pieces < 2:
            raise ValueError("mccormick_bounds['num_pieces'] must be at least 2")
        return [
            float(value)
            for value in np.linspace(C_opt_L, C_opt_U, num_pieces + 1)
        ]

    def _mccormick_bounds_with_loaded_C_opt_bounds(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return mccormick_bounds
        if mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds is required when objective_mode='{self.objective_mode}'"
            )
        mccormick_bounds = self._normalize_mccormick_bounds_alias(mccormick_bounds, None)
        if "C_opt" in mccormick_bounds:
            return mccormick_bounds

        C_opt_bounds = self.optimal_cost_bounds or {}
        if "C_opt" in C_opt_bounds and isinstance(C_opt_bounds.get("C_opt"), dict):
            C_opt_bounds = C_opt_bounds.get("C_opt", {}) or {}
        lower = C_opt_bounds.get("lower")
        upper = C_opt_bounds.get("upper")
        if lower is None or upper is None:
            raise ValueError(
                "Mccormick objective modes require denominator bounds. Pass "
                "mccormick_bounds['C_opt'] explicitly or run/load the DRO "
                "optimal-cost-bound tightening stage first."
            )
        completed = dict(mccormick_bounds)
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
        reference_case: str = "base_test_case",
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

        self.demand_shape = [
            float(value)
            for value in scenario_manager._build_demand_shape(self.num_time_steps)
        ]
        self.wind_shape = [
            float(value)
            for value in scenario_manager._build_wind_shape(
                self.num_time_steps,
                self.peak_W_fixed,
            )
        ]
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

        if self.objective_mode in {"mccormick", "piecewise_mccormick"}:
            mccormick_bounds = self.mccormick_bounds or self._raw_mccormick_bounds
            completed_bounds = self._mccormick_bounds_with_loaded_C_opt_bounds(mccormick_bounds)
            self.mccormick_bounds = self._validate_mccormick_bounds(completed_bounds)

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
            self.mccormick_bounds["C_opt"]
            if self.objective_mode in {
                "mccormick",
                "piecewise_mccormick",
            } and self.mccormick_bounds is not None
            else (None, None)
        )
        m.C_opt = Var(m.scenarios, domain=Reals, bounds=c_opt_bounds)
        if self.objective_mode in {"mccormick", "piecewise_mccormick"}:
            self._build_mccormick_variables()
        else:
            m.PoA = Var(m.scenarios, domain=Reals)
        if self.objective_mode == "piecewise_mccormick":
            self._build_piecewise_mccormick_variables()

    def _build_mccormick_variables(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds is required when objective_mode='{self.objective_mode}'"
        )
        m = self.model
        PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        C_opt_L, C_opt_U = self.mccormick_bounds["C_opt"]
        m.PoA = Var(m.scenarios, bounds=(PoA_L, PoA_U))
        m.z_mccormick_product = Var(
            m.scenarios,
            domain=Reals,
            bounds=(PoA_L * C_opt_L, PoA_U * C_opt_U),
        )

    def _build_piecewise_mccormick_variables(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds is required when "
                "objective_mode='piecewise_mccormick'"
        )
        m = self.model
        breakpoints = list(self.mccormick_bounds["C_opt_breakpoints"])
        _PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        m.mccormick_piece_index = Set(initialize=range(len(breakpoints) - 1))
        m.mccormick_piece_active = Var(m.scenarios, m.mccormick_piece_index, domain=Binary)
        m.C_opt_piece = Var(
            m.scenarios,
            m.mccormick_piece_index,
            domain=NonNegativeReals,
            bounds=lambda m, k, p: (0.0, breakpoints[int(p) + 1]),
        )
        m.PoA_piece = Var(
            m.scenarios,
            m.mccormick_piece_index,
            domain=NonNegativeReals,
            bounds=(0.0, PoA_U),
        )
        m.z_mccormick_piece = Var(
            m.scenarios,
            m.mccormick_piece_index,
            domain=NonNegativeReals,
            bounds=lambda m, k, p: (0.0, PoA_U * breakpoints[int(p) + 1]),
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
        m.D_abs_deviation = Var(m.scenarios, m.time_steps, domain=NonNegativeReals)
        m.P_max_phys_abs_deviation = Var(
            m.scenarios, m.wind_physical_generators, m.time_steps, domain=NonNegativeReals
        )

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
        elif self.objective_mode == "mccormick":
            self._build_mccormick_objective()
        elif self.objective_mode == "piecewise_mccormick":
            self._build_piecewise_mccormick_objective()
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

    def _build_mccormick_objective(self) -> None:
        m = self.model
        m.objective = Objective(
            expr=sum(
                m.PoA[k] - self.eta * m.wasserstein_distance[k]
                for k in m.scenarios
            )
            / self.num_empirical_scenarios,
            sense=maximize,
        )

    def _build_piecewise_mccormick_objective(self) -> None:
        m = self.model
        m.objective = Objective(
            expr=sum(
                m.PoA[k] - self.eta * m.wasserstein_distance[k]
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
        self._build_PoA_constraints()

    # ------------------------------------------------------------------
    # Support set
    # ------------------------------------------------------------------

    def diagnose_empirical_support_set_violations(self) -> list[dict]:
        """For each empirical scenario, compute the minimum achievable Wasserstein
        distance and flag which support-set constraint families are violated.

        This reveals the W-floor that high-eta runs cannot cross: if empirical_D[k]
        or empirical_Pmax_phys[k] lies outside the support set, the optimizer cannot
        set D[k,t] = empirical_D[k][t] and W[k] > 0 is unavoidable.

        Returns a list of dicts, one per scenario.
        """
        import numpy as np
        from scipy.stats import norm as _norm

        T = self.num_time_steps
        D_ref = float(self.demand_D_ref)
        kappa = 1.96
        kappa_ar1 = float(_norm.ppf((1.0 + 0.95 ** (1.0 / T)) / 2.0))

        from config.scenarios.scenario_generator import ScenarioManager
        demand_shape = ScenarioManager._build_demand_shape(T)
        wind_shape = ScenarioManager._build_wind_shape(T, self.peak_W_fixed)

        results = []
        for k in range(self.num_empirical_scenarios):
            D_emp = np.asarray(self.empirical_D[k], dtype=float)
            D_ref_vec = D_ref * self.mu_D_fixed * demand_shape
            stationary_std_D = self.sigma_D_fixed / np.sqrt(1.0 - self.demand_rho_fixed ** 2)
            margin_D = kappa * D_ref * stationary_std_D

            lb_D = np.maximum(D_ref_vec - margin_D, 0.0)
            ub_D = D_ref_vec + margin_D
            D_proj = np.clip(D_emp, lb_D, ub_D)
            min_W_demand = float(np.sum(np.abs(D_emp - D_proj)))

            innov_margin_D = kappa_ar1 * D_ref * self.sigma_D_fixed
            ar1_ref_D = D_ref * self.mu_D_fixed * (
                demand_shape[1:] - self.demand_rho_fixed * demand_shape[:-1]
            )
            innov_D = D_emp[1:] - self.demand_rho_fixed * D_emp[:-1]
            ar1_violations_D = int(np.sum(np.abs(innov_D - ar1_ref_D) > innov_margin_D + 1e-9))

            min_W_wind = 0.0
            wind_ar1_violations = 0
            stationary_std_W = self.sigma_W_fixed / np.sqrt(1.0 - self.wind_rho_fixed ** 2)
            innov_margin_W = kappa_ar1 * self.sigma_W_fixed

            for i in self.wind_physical_generator_ids:
                cap = float(self.static_physical_capacity[int(i)])
                P_emp = np.asarray(self.empirical_Pmax_phys[k][int(i)], dtype=float)
                P_ref_vec = cap * self.mu_W_fixed * wind_shape
                margin_W = kappa * cap * stationary_std_W
                lb_W = np.maximum(P_ref_vec - margin_W, 0.0)
                ub_W = np.minimum(P_ref_vec + margin_W, cap)
                P_proj = np.clip(P_emp, lb_W, ub_W)
                min_W_wind += float(np.sum(np.abs(P_emp - P_proj)))

                ar1_ref_W = cap * self.mu_W_fixed * (
                    wind_shape[1:] - self.wind_rho_fixed * wind_shape[:-1]
                )
                innov_W = P_emp[1:] - self.wind_rho_fixed * P_emp[:-1]
                wind_ar1_violations += int(
                    np.sum(np.abs(innov_W - ar1_ref_W) > innov_margin_W * cap + 1e-9)
                )

            results.append({
                "scenario_k": k,
                "min_W_demand": min_W_demand,
                "min_W_wind": min_W_wind,
                "min_W_total": min_W_demand + min_W_wind,
                "demand_pointwise_violations": int(
                    np.sum(D_emp < lb_D - 1e-9) + np.sum(D_emp > ub_D + 1e-9)
                ),
                "demand_ar1_violations": ar1_violations_D,
                "wind_ar1_violations": wind_ar1_violations,
            })
        return results

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

        # Hard Wasserstein ball constraint: W[k] <= epsilon for every scenario.
        # Without this, only the Lagrangian penalty (eta * W) is active, which
        # cannot drive W to zero when the PoA landscape has discontinuities
        # (bid-stack switches create arbitrarily steep gradients at finite eta).
        # At epsilon = 0 this forces W = 0, recovering the nominal PoA.
        def wasserstein_ball_rule(m, k):
            return m.wasserstein_distance[k] <= self.epsilon

        m.wasserstein_ball = Constraint(m.scenarios, rule=wasserstein_ball_rule)

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
        if self.objective_mode == "difference":
            m.poa_definition = Constraint(m.scenarios, rule=poa_rule)
        if self.objective_mode == "mccormick":
            self._build_mccormick_constraints()
        if self.objective_mode == "piecewise_mccormick":
            self._build_piecewise_mccormick_constraints()

    def _build_mccormick_constraints(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds is required when objective_mode='mccormick'"
        )
        m = self.model
        PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        C_opt_L, C_opt_U = self.mccormick_bounds["C_opt"]

        def mccormick_link_eq_cost_rule(m, k):
            return m.z_mccormick_product[k] == m.C_eq[k]

        def lower_1_rule(m, k):
            return (
                m.z_mccormick_product[k]
                >= PoA_L * m.C_opt[k] + C_opt_L * m.PoA[k] - PoA_L * C_opt_L
            )

        def lower_2_rule(m, k):
            return (
                m.z_mccormick_product[k]
                >= PoA_U * m.C_opt[k] + C_opt_U * m.PoA[k] - PoA_U * C_opt_U
            )

        def upper_1_rule(m, k):
            return (
                m.z_mccormick_product[k]
                <= PoA_U * m.C_opt[k] + C_opt_L * m.PoA[k] - PoA_U * C_opt_L
            )

        def upper_2_rule(m, k):
            return (
                m.z_mccormick_product[k]
                <= PoA_L * m.C_opt[k] + C_opt_U * m.PoA[k] - PoA_L * C_opt_U
            )

        m.mccormick_link_eq_cost = Constraint(m.scenarios, rule=mccormick_link_eq_cost_rule)
        m.mccormick_lower_1 = Constraint(m.scenarios, rule=lower_1_rule)
        m.mccormick_lower_2 = Constraint(m.scenarios, rule=lower_2_rule)
        m.mccormick_upper_1 = Constraint(m.scenarios, rule=upper_1_rule)
        m.mccormick_upper_2 = Constraint(m.scenarios, rule=upper_2_rule)

    def _build_piecewise_mccormick_constraints(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds is required when "
                "objective_mode='piecewise_mccormick'"
        )
        m = self.model
        PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        breakpoints = list(self.mccormick_bounds["C_opt_breakpoints"])

        def select_one_rule(m, k):
            return sum(
                m.mccormick_piece_active[k, p] for p in m.mccormick_piece_index
            ) == 1

        def C_opt_link_rule(m, k):
            return m.C_opt[k] == sum(
                m.C_opt_piece[k, p] for p in m.mccormick_piece_index
            )

        def PoA_link_rule(m, k):
            return m.PoA[k] == sum(
                m.PoA_piece[k, p] for p in m.mccormick_piece_index
            )

        def z_link_rule(m, k):
            return m.z_mccormick_product[k] == sum(
                m.z_mccormick_piece[k, p] for p in m.mccormick_piece_index
            )

        def C_eq_link_rule(m, k):
            return m.C_eq[k] == m.z_mccormick_product[k]

        def C_opt_piece_lower_rule(m, k, p):
            return (
                breakpoints[int(p)] * m.mccormick_piece_active[k, p]
                <= m.C_opt_piece[k, p]
            )

        def C_opt_piece_upper_rule(m, k, p):
            return (
                m.C_opt_piece[k, p]
                <= breakpoints[int(p) + 1] * m.mccormick_piece_active[k, p]
            )

        def PoA_piece_lower_rule(m, k, p):
            return PoA_L * m.mccormick_piece_active[k, p] <= m.PoA_piece[k, p]

        def PoA_piece_upper_rule(m, k, p):
            return m.PoA_piece[k, p] <= PoA_U * m.mccormick_piece_active[k, p]

        def lower_1_rule(m, k, p):
            p_int = int(p)
            y_L = breakpoints[p_int]
            return (
                m.z_mccormick_piece[k, p]
                >= PoA_L * m.C_opt_piece[k, p]
                + y_L * m.PoA_piece[k, p]
                - PoA_L * y_L * m.mccormick_piece_active[k, p]
            )

        def lower_2_rule(m, k, p):
            p_int = int(p)
            y_U = breakpoints[p_int + 1]
            return (
                m.z_mccormick_piece[k, p]
                >= PoA_U * m.C_opt_piece[k, p]
                + y_U * m.PoA_piece[k, p]
                - PoA_U * y_U * m.mccormick_piece_active[k, p]
            )

        def upper_1_rule(m, k, p):
            p_int = int(p)
            y_L = breakpoints[p_int]
            return (
                m.z_mccormick_piece[k, p]
                <= PoA_U * m.C_opt_piece[k, p]
                + y_L * m.PoA_piece[k, p]
                - PoA_U * y_L * m.mccormick_piece_active[k, p]
            )

        def upper_2_rule(m, k, p):
            p_int = int(p)
            y_U = breakpoints[p_int + 1]
            return (
                m.z_mccormick_piece[k, p]
                <= PoA_L * m.C_opt_piece[k, p]
                + y_U * m.PoA_piece[k, p]
                - PoA_L * y_U * m.mccormick_piece_active[k, p]
            )

        m.mccormick_piece_select_one = Constraint(m.scenarios, rule=select_one_rule)
        m.mccormick_piece_C_opt_link = Constraint(m.scenarios, rule=C_opt_link_rule)
        m.mccormick_piece_PoA_link = Constraint(m.scenarios, rule=PoA_link_rule)
        m.mccormick_piece_z_link = Constraint(m.scenarios, rule=z_link_rule)
        m.mccormick_piece_C_eq_link = Constraint(m.scenarios, rule=C_eq_link_rule)
        m.mccormick_piece_C_opt_lower = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=C_opt_piece_lower_rule,
        )
        m.mccormick_piece_C_opt_upper = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=C_opt_piece_upper_rule,
        )
        m.mccormick_piece_PoA_lower = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=PoA_piece_lower_rule,
        )
        m.mccormick_piece_PoA_upper = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=PoA_piece_upper_rule,
        )
        m.mccormick_piece_mccormick_lower_1 = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=lower_1_rule,
        )
        m.mccormick_piece_mccormick_lower_2 = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=lower_2_rule,
        )
        m.mccormick_piece_mccormick_upper_1 = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=upper_1_rule,
        )
        m.mccormick_piece_mccormick_upper_2 = Constraint(
            m.scenarios,
            m.mccormick_piece_index,
            rule=upper_2_rule,
        )

    # ------------------------------------------------------------------
    # Policy constraints
    # ------------------------------------------------------------------

    def solve(self, time_limit: Optional[float] = None) -> Any:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        solver = SolverFactory("gurobi_persistent")
        solver.set_instance(self.model)
        solver.options["IntFeasTol"] = 1e-8
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        self.solver_results = solver.solve(tee=True, load_solutions=False)
        termination = self.solver_results.solver.termination_condition
        if termination == TerminationCondition.infeasible:
            from pyomo.contrib.iis import write_iis
            iis_path = "infeasible.ilp"
            write_iis(self.model, iis_path, solver="gurobi")
            print(f"IIS written to {iis_path}")
        elif len(self.solver_results.solution) > 0:
            self.model.solutions.load_from(self.solver_results)
        return self.solver_results

def load_regime_scenarios(
    reference_case: str = "base_test_case",
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
    reference_case: str = "base_test_case",
    regime_config_path: str | Path = "config/regime_definitions.yaml",
    regime_set: str = "PoA_analysis",
    epsilon: float = 0.0,
    num_time_steps: Optional[int] = None,
    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    objective_mode: str = "difference",
    mccormick_bounds: Optional[dict[str, Any]] = None,
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
                mccormick_bounds=mccormick_bounds,
            )
            optimizer.build_model()
            optimizer.solve(time_limit=time_limit)
            summaries.append(optimizer.solution_summary())

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    case = "base_test_case"
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
        # objective_mode="mccormick",
        # mccormick_bounds={
        #     "PoA": (1.0, 5.0),
        #     "C_opt": (1000.0, 20000.0),
        # },
        # objective_mode="piecewise_mccormick",
        # mccormick_bounds={
        #     "PoA": (1.0, 5.0),
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
