from __future__ import annotations

import logging
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
    block_cost_vector,
    block_structure_from_dataframes,
    ensure_profile,
    find_demand_profile_column,
    gurobi_log_filter,
    infer_num_time_steps,
    is_wind_generator_name,
    parse_gurobi_node_log,
    ramp_vectors,
)
from models.DRO_PoA.dro_poa_model.mccormick import DROPoAMcCormick
from models.DRO_PoA.dro_poa_model.nn_policy_embedding import DROPoAPolicyEmbedding
from models.DRO_PoA.dro_poa_model.results import DROPoAResults
from models.DRO_PoA.dro_poa_model.auxiliary_LPs import (
    DROPoAMIPStart,
    DROPoASupportDiagnostics,
)
from models.DRO_PoA.dro_poa_model.support_set import (
    DROWassersteinSupportSet,
    _ar1_kappa,
)
from models.DRO_PoA.dro_poa_model.tightening_reports import DROPoATighteningReports



class DRO_PoAOptimization(
    DROPoAPolicyEmbedding,
    DROPoATighteningReports,
    DROPoAMcCormick,
    DROWassersteinSupportSet,
    DROPoASupportDiagnostics,
    DROPoAMIPStart,
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
    # alpha (bids) is domain=Reals: strategic/NN bids may be negative, so the loose
    # lower bound must be a valid outer bound, not 0. A 0 lower bound invalidly
    # clamps bids to non-negative and suppresses the true worst-case PoA.
    DEFAULT_LOOSE_ALPHA_LOWER = -1e4
    DEFAULT_LOOSE_ALPHA_UPPER = 1e4
    DEFAULT_LOOSE_DUAL_BIG_M = 1e6
    # lambda_eq is the Reals dual of the power-balance equality; with negative bids
    # the marginal block can set a negative clearing price, so the loose lower bound
    # must allow negatives. There is no lambda tightening stage, so this default is
    # used in every run. Kept symmetric with the upper price scale.
    DEFAULT_LOOSE_LAMBDA_LOWER = -100
    DEFAULT_LOOSE_LAMBDA_UPPER = 100
    # Outward margin applied to the bid-implied clearing-price (lambda) bounds to
    # absorb ramp-dual contributions; see _bid_implied_lambda_bounds.
    LAMBDA_RAMP_SAFETY_FACTOR = 0.10
    DEFAULT_LOOSE_AGGREGATE_DUAL_UPPER = 1e8
    DEFAULT_LOOSE_C_OPT_LOWER = 1e-3
    DEFAULT_LOOSE_C_OPT_UPPER = 1e8
    DEFAULT_PoA_LOWER = 1.0
    DEFAULT_PoA_UPPER = 10.0
    DEFAULT_PHI_LOWER = DEFAULT_PoA_LOWER
    DEFAULT_PHI_UPPER = DEFAULT_PoA_UPPER
    DEFAULT_ALPHA_ORDERING_EPSILON = 1e-6

    normalization_epsilon = 1e-12
    # Scenarios whose minimum achievable Wasserstein distance to the support set
    # is at or below this tolerance are treated as inside the support set, so the
    # nominal market state (W[k]=0) is feasible and the per-scenario objective
    # term PoA[k] - eta * W[k] can be validly floored at 1.
    SUPPORT_FLOOR_TOLERANCE = 1e-6
    allowed_objective_modes = {
        "difference",
        "mccormick",
        "piecewise_mccormick",
    }

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
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
        case_label: str = "",
        objective_mode: str = "piecewise_mccormick",
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
        ar1_coverage: Optional[float] = None,
        alpha_ordering_epsilon: float = DEFAULT_ALPHA_ORDERING_EPSILON,
    ):
        if float(eta) < 0.0:
            raise ValueError("eta must be nonnegative")
        if float(epsilon) < 0.0:
            raise ValueError("epsilon must be nonnegative")

        self.costs_df = costs_df
        self.ramps_df = ramps_df
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
        self.case_label = case_label
        self.ar1_coverage = float(
            ar1_coverage
            if ar1_coverage is not None
            else DROWassersteinSupportSet.AR1_JOINT_COVERAGE
        )
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
        self.alpha_ordering_epsilon = float(alpha_ordering_epsilon)
        if self.default_c_opt_lower <= 0.0:
            raise ValueError("default_c_opt_lower must be strictly positive")
        if self.default_c_opt_upper < self.default_c_opt_lower:
            raise ValueError("default_c_opt_upper must be >= default_c_opt_lower")
        if self.default_lambda_lower >= self.default_lambda_upper:
            raise ValueError("default_lambda_lower must be < default_lambda_upper")
        if self.alpha_ordering_epsilon < 0.0:
            raise ValueError("alpha_ordering_epsilon must be nonnegative")
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
        self._initialize_block_structure()
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
        self._configure_fixed_regime_parameters()
        self._configure_regime_shape_profiles()
        self.p_init = self.compute_p_init_from_ed()
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

    # ------------------------------------------------------------------
    # Data and configuration
    # ------------------------------------------------------------------

    def _initialize_block_structure(self) -> None:
        block_structure = block_structure_from_dataframes(
            self.scenarios_df,
            self.ramps_df,
        )

        self.block_names = list(block_structure.block_names)
        self.num_blocks = len(self.block_names)
        self.physical_generator_names = list(block_structure.physical_generator_names)
        self.num_physical_generators = len(self.physical_generator_names)
        self.block_to_physical = dict(block_structure.block_to_physical)
        self.block_to_physical_idx = list(block_structure.block_to_physical_idx)
        self.physical_to_block_indices = [
            list(blocks) for blocks in block_structure.physical_to_block_indices
        ]
        self.blocks_by_generator = {
            int(i): list(blocks) for i, blocks in block_structure.blocks_by_generator.items()
        }
        self.local_blocks_by_generator = {
            int(i): list(blocks)
            for i, blocks in block_structure.local_blocks_by_generator.items()
        }
        self.local_to_global_block = dict(block_structure.local_to_global_block)
        self.global_to_local_block = dict(block_structure.global_to_local_block)
        self.generator_block_pairs = list(block_structure.generator_block_pairs)
        self.block_cost_vector = block_cost_vector(self.costs_df, self.block_names)
        self.ramp_vector_up, self.ramp_vector_down = ramp_vectors(
            self.ramps_df,
            self.physical_generator_names,
        )

        self.wind_physical_generator_ids = [
            i
            for i, name in enumerate(self.physical_generator_names)
            if is_wind_generator_name(name)
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

    def compute_p_init_from_ed(self) -> list[list[float]]:
        """Compute p_init by solving a 1-step economic dispatch at the regime's
        deterministic (mu_D_fixed, mu_W_fixed) operating point, using true
        marginal costs as bids.

        Replaces the merit-order heuristic in compute_deterministic_p_init() with
        an exact LP.  Returns the same p_init row for every empirical scenario,
        shaped [num_empirical_scenarios][num_physical_generators].  A single shared
        p_init keeps the regime-wide (single-representative-scenario) tightening
        valid for every empirical scenario.
        """
        from models.synthetic_data_generation.economic_dispatch_clean import EconomicDispatchModel

        demand_shape = ScenarioManager._build_demand_shape(self.num_time_steps)
        wind_shape = ScenarioManager._build_wind_shape(self.num_time_steps, self.peak_W_fixed)
        demand_t0 = float(self.demand_D_ref * self.mu_D_fixed * demand_shape[0])

        wind_ids = set(int(i) for i in self.wind_physical_generator_ids)
        row: dict[str, Any] = {"demand_profile": [demand_t0], "time_steps": 1}
        for global_b, block_name in enumerate(self.block_names):
            phys_idx = int(self.block_to_physical_idx[global_b])
            if phys_idx in wind_ids:
                n_local = len(self.local_blocks_by_generator[phys_idx])
                block_cap = float(
                    self.static_physical_capacity[phys_idx] * self.mu_W_fixed * wind_shape[0] / n_local
                )
            else:
                block_cap = float(self.static_block_capacity[global_b])
            row[f"{block_name}_cap"] = block_cap
            row[f"{block_name}_bid"] = float(self.block_cost_vector[global_b])

        scenarios_df = pd.DataFrame([row])
        ed = EconomicDispatchModel(
            scenarios_df=scenarios_df,
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
        )
        ed.solve()
        dispatches = ed.get_dispatches()
        p_init_row = dispatches[0][0]  # type: ignore[index]
        return [list(p_init_row) for _ in range(self.num_empirical_scenarios)]

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
            int(t): (float(self.default_lambda_lower), float(self.default_lambda_upper))
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
        if not getattr(self, "tightening_report", None) and getattr(
            self,
            "use_default_bounds",
            False,
        ):
            # No tightening report loaded: self-provide model-construction bounds,
            # always computing primal_big_m (analytic) and optimal_cost_bounds (real
            # solve). Mirrors the PoA model fallback.
            self.ensure_default_bounds_available()
        elif getattr(self, "tightening_report", None):
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
        PoA_U = self.mccormick_bounds["PoA"][1]
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
            # Inactive pieces must be zero; PoA_L is enforced by
            # mccormick_piece_PoA_lower after the active binary is known.
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
        self._build_support_floor_constraints()

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

        m.cost_definition_eq = Constraint(m.scenarios, rule=cost_eq_rule)
        m.cost_definition_opt = Constraint(m.scenarios, rule=cost_opt_rule)
        if self.objective_mode == "difference":
            def poa_rule(m, k):
                return m.C_eq[k] - m.C_opt[k] == m.PoA[k]
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

    def support_set_diagnostics(self, solver_name: str = "gurobi") -> list[dict]:
        """Per-scenario support-set diagnostics, computed once and cached.

        The diagnostic (minimum Wasserstein distance onto the support set, plus
        per-family violation counts) does not depend on eta, so it is computed a
        single time and reused -- both by the support-floor constraints and across
        an eta sweep -- rather than re-solving the projection LPs for every eta.
        """
        if getattr(self, "_support_diagnostics_cache", None) is None:
            self._support_diagnostics_cache = (
                self.diagnose_empirical_support_set_violations(solver_name=solver_name)
            )
        return self._support_diagnostics_cache

    def _build_support_floor_constraints(self) -> None:
        """Floor the per-scenario objective term at 1 where W[k]=0 is feasible.

        For empirical scenarios inside the Wasserstein support set (minimum
        achievable transport distance <= SUPPORT_FLOOR_TOLERANCE, as reported by
        DROPoASupportDiagnostics) the nominal market state W[k]=0 is feasible and
        yields PoA[k] >= 1, so the per-scenario maximand PoA[k] - eta * W[k] is at
        least 1.  Imposing this as a valid lower bound tightens the relaxation
        without removing the optimum.  Scenarios outside the support set
        (min_W_total > tolerance) are skipped because the floor need not hold
        there (W[k] is forced strictly positive and the penalty -eta * W[k] can
        pull the term below 1).

        This constraint depends on eta, so update_eta() rebuilds it during an eta
        sweep.  The set of floored scenarios is eta-independent, so it is created
        once and kept; only the constraint coefficients change between etas.
        """
        m = self.model
        diagnostics = self.support_set_diagnostics()
        inside_support = [
            int(row["scenario_k"])
            for row in diagnostics
            if float(row["min_W_total"]) <= self.SUPPORT_FLOOR_TOLERANCE
        ]
        self.support_floor_scenarios = inside_support
        if not inside_support:
            return

        if not hasattr(m, "support_floor_scenario_set"):
            m.support_floor_scenario_set = Set(initialize=inside_support)

        def support_objective_floor_rule(m, k):
            return m.PoA[k] - self.eta * m.wasserstein_distance[k] >= 1.0

        m.support_objective_floor = Constraint(
            m.support_floor_scenario_set, rule=support_objective_floor_rule
        )

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def attach_persistent_solver(self) -> Any:
        """Create a persistent Gurobi solver and load the model into it once.

        This is the setup step for an eta sweep: the model is loaded a single
        time, after which update_eta() pushes only the eta-dependent objective and
        support-floor constraints to the live solver instead of rebuilding and
        re-loading the whole model for every eta.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        solver = SolverFactory("gurobi_persistent")
        solver.set_instance(self.model)
        solver.options["IntFeasTol"] = 1e-8
        self._persistent_solver = solver

        # Provide Gurobi with a feasible starting point: the empirical trajectories
        # (W[k]=0) dispatched under the NN bids.  This gives an immediate incumbent
        # at the root node so Gurobi starts pruning rather than searching.
        self.compute_empirical_mip_start()

        return solver

    def update_eta(self, eta: float) -> None:
        """Re-point the eta-dependent parts of the model at a new eta, in place.

        Only the objective term -eta * W[k] and the support-floor constraints
        PoA[k] - eta * W[k] >= 1 depend on eta; everything else (support set, KKT,
        ReLU embedding, McCormick) is unchanged.  When a persistent solver is
        attached, the rebuilt objective and constraints are pushed to it so an eta
        sweep reuses the already-loaded model.
        """
        self.eta = float(eta)
        m = self.model
        solver = getattr(self, "_persistent_solver", None)

        # Rebuild the eta-dependent objective.
        m.del_component(m.objective)
        self._build_objective()
        if solver is not None:
            solver.set_objective(m.objective)

        # Rebuild the eta-dependent support-floor constraints.  The floored-
        # scenario set is unchanged, so only the Constraint component is dropped
        # and recreated; support_floor_scenario_set is left in place.
        previous_floor = (
            list(m.support_objective_floor.values())
            if hasattr(m, "support_objective_floor")
            else []
        )
        if hasattr(m, "support_objective_floor"):
            m.del_component(m.support_objective_floor)
        self._build_support_floor_constraints()
        if solver is not None:
            for constraint_data in previous_floor:
                solver.remove_constraint(constraint_data)
            if hasattr(m, "support_objective_floor"):
                for constraint_data in m.support_objective_floor.values():
                    solver.add_constraint(constraint_data)

    def solve(self, time_limit: Optional[float] = None, warm_start: bool = False) -> Any:
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")
        solver = getattr(self, "_persistent_solver", None)
        if solver is None:
            solver = self.attach_persistent_solver()
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)

        # warm_start=True hands Gurobi the variable values as a MIP-start incumbent.
        # The persistent model appends to one log file across the eta sweep;
        # pos_holder tracks the read offset so the tail continues from where it
        # left off rather than replaying the full file on every eta.
        if not hasattr(self, "_gurobi_log_pos"):
            self._gurobi_log_path = None
            self._gurobi_log_pos = {"pos": 0}

        start = time.perf_counter()
        with gurobi_log_filter(solver, self._gurobi_log_path, self._gurobi_log_pos) as log_path:
            self._gurobi_log_path = log_path
            log_file = Path(log_path)
            log_offset = log_file.stat().st_size if log_file.exists() else 0
            self.solver_results = solver.solve(
                tee=False, load_solutions=False, warmstart=warm_start
            )
        self.solve_wall_time_seconds = time.perf_counter() - start

        # Slice this solve's segment out of the shared (appended) Gurobi log and
        # parse the node log into an incumbent/BestBd/gap time series, so each
        # eta's bound movement can be saved alongside its result.
        self.last_solve_gurobi_log = None
        self.solve_bound_progression = []
        try:
            with log_file.open("r", encoding="utf-8", errors="replace") as file_handle:
                file_handle.seek(log_offset)
                self.last_solve_gurobi_log = file_handle.read()
        except OSError:
            pass
        if self.last_solve_gurobi_log:
            self.solve_bound_progression = parse_gurobi_node_log(self.last_solve_gurobi_log)

        # Certified bracket from Gurobi (maximization): the incumbent is a lower
        # bound and ObjBound an upper bound on the optimum, so a time-limited
        # solve still yields a reportable interval. Reset first: the persistent
        # solver is reused across the eta sweep and stale values must not leak.
        self.best_objective_bound = None
        self.mip_gap = None
        gurobi_model = getattr(solver, "_solver_model", None)
        if gurobi_model is not None:
            try:
                bound = float(gurobi_model.ObjBound)
                if np.isfinite(bound):
                    self.best_objective_bound = bound
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                gap = float(gurobi_model.MIPGap)
                if np.isfinite(gap):
                    self.mip_gap = gap
            except (AttributeError, TypeError, ValueError):
                pass

        termination = self.solver_results.solver.termination_condition
        if termination == TerminationCondition.infeasible:
            # Computing an IIS on this MILP can take many minutes and, in an eta
            # sweep, would fire on every infeasible eta. Default to a cheap log;
            # set write_iis_on_infeasible=True to diagnose a single solve.
            if getattr(self, "write_iis_on_infeasible", False):
                from pyomo.contrib.iis import write_iis
                iis_path = "infeasible.ilp"
                write_iis(self.model, iis_path, solver="gurobi")
                print(f"IIS written to {iis_path}")
            else:
                print(
                    "DRO solve infeasible (IIS skipped; set "
                    "optimizer.write_iis_on_infeasible=True to diagnose)."
                )
        elif len(self.solver_results.solution) > 0:
            # Gurobi can return sign-bounded KKT duals (e.g. mu_lower_eq) a hair
            # past their 0 lower bound, within FeasibilityTol (~1e-6). The value
            # is genuinely zero; the only effect is a cosmetic W1002 on load.
            # Silence pyomo.core just for this load so real out-of-bounds loads
            # elsewhere still surface.
            pyomo_core_logger = logging.getLogger("pyomo.core")
            previous_level = pyomo_core_logger.level
            pyomo_core_logger.setLevel(logging.ERROR)
            try:
                self.model.solutions.load_from(self.solver_results)
            finally:
                pyomo_core_logger.setLevel(previous_level)
        return self.solver_results

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
