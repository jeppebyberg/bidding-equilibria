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
    block_cost_vector,
    block_structure_from_dataframes,
    infer_num_time_steps,
    is_wind_generator_name,
    ramp_vectors,
)
from models.PoA.poa_model.nn_policy_embedding import PoAPolicyEmbedding
from models.PoA.poa_model.mccormick import PoAMcCormick
from models.PoA.poa_model.results import PoAResults
from models.PoA.poa_model.support_set import PoASupportSet
from models.PoA.poa_model.tightening_reports import PoATighteningReports

class PoAOptimization(
        PoAPolicyEmbedding, 
        PoATighteningReports, 
        PoAMcCormick,
        PoASupportSet, 
        PoAResults
    ):
    """
    Block-aware Price of Anarchy optimization.

    Dispatch and bids are indexed by physical generator and local bidding block:
    P_eq[i, b, t], P_opt[i, b, t], alpha[i, b, t].
    Physical ramp constraints sum over each generator's local bidding blocks.
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
        p_init: Optional[list[float] | list[list[float]]] = None,
        num_time_steps: Optional[int] = None,
        ambiguity_set_config: Optional[dict[str, Any]] = None,
        nn_model_dir: Optional[str | Path] = None,
        nn_normalization_stats_path: Optional[str | Path] = None,
        nn_policy_generators: Optional[list[int | str]] = None,
        reference_case: str = "base_test_case",
        objective_mode: str = "difference",
        mccormick_bounds: Optional[dict[str, Any]] = None,
        ratio_bounds: Optional[dict[str, Any]] = None,
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
    ):
        self.scenarios_df = scenarios_df.reset_index(drop=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.requested_p_init = p_init
        self.ambiguity_set_config = ambiguity_set_config or {}
        self.nn_model_dir = Path(nn_model_dir) if nn_model_dir is not None else None
        self.nn_normalization_stats_path = (
            Path(nn_normalization_stats_path)
            if nn_normalization_stats_path is not None
            else None
        )
        self.requested_nn_policy_generators = nn_policy_generators
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
        self.default_PoA_lower = float(default_PoA_lower)
        self.default_PoA_upper = float(default_PoA_upper)
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
        self.default_bounds_used: dict[str, Any] = self._empty_default_bounds_used()

        self.nn_relu_bounds_report: dict[str, Any] = {}
        self.nn_relu_bounds: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
        self.nn_feature_bounds: dict[str, Any] = {}
        self.nn_bound_warnings: list[str] = []
        self.primal_big_m: dict[str, dict[str, Any]] = {}
        self.tight_big_m: dict[str, dict[str, Any]] = {}
        self.aggregate_dual_bounds: dict[str, Any] = {}
        self.lambda_bounds: dict[str, Any] = {}
        self.optimal_cost_bounds: dict[str, Any] = {}
        self.optimal_cost_bound_optimization_results: dict[str, Any] = {}
        self.reference_case = reference_case
        self.objective_mode = self._validate_objective_mode(objective_mode)
        mccormick_bounds = self._normalize_mccormick_bounds_alias(
            mccormick_bounds,
            ratio_bounds,
        )
        mccormick_bounds = self._mccormick_bounds_with_defaults(mccormick_bounds)
        self.mccormick_bounds = self._validate_mccormick_bounds(mccormick_bounds)
        self._initialize_loaded_bound_dictionaries()

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
        self.p_init = self._normalize_p_init(self.requested_p_init)
        self._configure_ambiguity_set_parameters()
        self._configure_nn_policy_generators()

        self.nn_policies: dict[str, dict[str, Any]] = {}
        self.nn_stats: dict[str, Any] = {}
        if self.nn_model_dir is not None and self.nn_policy_generator_ids:
            self._load_nn_policies()
            self._load_nn_normalization_stats()

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
        block_structure = block_structure_from_dataframes(self.scenarios_df, self.ramps_df)

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
            int(i): list(blocks)
            for i, blocks in block_structure.blocks_by_generator.items()
        }
        self.local_blocks_by_generator = {
            int(i): list(blocks)
            for i, blocks in block_structure.local_blocks_by_generator.items()
        }
        self.local_to_global_block = dict(block_structure.local_to_global_block)
        self.global_to_local_block = dict(block_structure.global_to_local_block)
        self.generator_block_pairs = list(block_structure.generator_block_pairs)
        self.block_cost_vector = [
            float(v) for v in block_cost_vector(self.costs_df, self.block_names)
        ]
        ramp_up, ramp_down = ramp_vectors(self.ramps_df, self.physical_generator_names)
        self.ramp_vector_up = [float(v) for v in ramp_up]
        self.ramp_vector_down = [float(v) for v in ramp_down]

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
        return is_wind_generator_name(name)

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

    @staticmethod
    def _required_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
        cursor: Any = config
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                rendered_path = "".join(f"['{part}']" for part in path)
                raise ValueError(f"Missing ambiguity_set_config{rendered_path}")
            cursor = cursor[key]
        return cursor

    @staticmethod
    def _required_float(config: dict[str, Any], path: tuple[str, ...]) -> float:
        value = PoAOptimization._required_nested(config, path)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            rendered_path = "".join(f"['{part}']" for part in path)
            raise ValueError(
                f"ambiguity_set_config{rendered_path} must be numeric"
            ) from exc
        if not np.isfinite(numeric_value):
            rendered_path = "".join(f"['{part}']" for part in path)
            raise ValueError(
                f"ambiguity_set_config{rendered_path} must be finite"
            )
        return numeric_value

    @staticmethod
    def _required_bounds(
        config: dict[str, Any],
        section: str,
        parameter: str,
    ) -> tuple[float, float]:
        lower = PoAOptimization._required_float(config, (section, parameter, "min"))
        upper = PoAOptimization._required_float(config, (section, parameter, "max"))
        if lower > upper:
            raise ValueError(
                f"ambiguity_set_config['{section}']['{parameter}']['min'] "
                "cannot exceed max"
            )
        return (lower, upper)

    def _configure_ambiguity_set_parameters(self) -> None:
        cfg = self.ambiguity_set_config
        if not isinstance(cfg, dict) or not cfg:
            raise ValueError(
                "ambiguity_set_config is required; use load_ambiguity_set_config()"
            )

        self.ambiguity_kappa = self._required_float(cfg, ("kappa",))
        self.mu_D_bounds = self._required_bounds(cfg, "demand", "mu")
        self.sigma_D_bounds = self._required_bounds(cfg, "demand", "sigma")
        self.demand_rho_fixed = self._required_float(cfg, ("demand", "rho_fixed"))
        self.mu_W_bounds = self._required_bounds(cfg, "wind", "mu")
        self.sigma_W_bounds = self._required_bounds(cfg, "wind", "sigma")
        self.wind_rho_fixed = self._required_float(cfg, ("wind", "rho_fixed"))
        self.wind_tau_fixed = self._required_float(cfg, ("wind", "tau_fixed"))

        if self.sigma_D_bounds[0] < 0:
            raise ValueError("ambiguity_set_config['demand']['sigma']['min'] must be non-negative")
        if self.sigma_W_bounds[0] < 0:
            raise ValueError("ambiguity_set_config['wind']['sigma']['min'] must be non-negative")
        if self.ambiguity_kappa < 0:
            raise ValueError("ambiguity_set_config['kappa'] must be non-negative")
        if not -0.999 <= self.demand_rho_fixed <= 0.999:
            raise ValueError("ambiguity_set_config['demand']['rho_fixed'] must be in [-0.999, 0.999]")
        if not -0.999 <= self.wind_rho_fixed <= 0.999:
            raise ValueError("ambiguity_set_config['wind']['rho_fixed'] must be in [-0.999, 0.999]")
        if not 0 <= self.wind_tau_fixed <= 24:
            raise ValueError("ambiguity_set_config['wind']['tau_fixed'] must be in [0, 24]")

        scenario_manager = ScenarioManager(self.reference_case)
        self.demand_D_ref = float(scenario_manager.base_case["demand"])
        if self.demand_D_ref <= 0 or not np.isfinite(self.demand_D_ref):
            raise ValueError("Reference-case scalar demand must be positive and finite")

        self.demand_shape = [
            float(value)
            for value in scenario_manager._build_demand_shape(self.num_time_steps)
        ]
        self.wind_shape = [
            float(value)
            for value in scenario_manager._build_wind_shape(
                self.num_time_steps,
                self.wind_tau_fixed,
            )
        ]

        for name, shape in (
            ("demand_shape", self.demand_shape),
            ("wind_shape", self.wind_shape),
        ):
            if any((not np.isfinite(value)) or value <= 0 for value in shape):
                raise ValueError(f"{name} entries must be finite and positive")

        self.demand_delta_shape = {
            t: abs(self.demand_shape[t] - self.demand_shape[t - 1])
            for t in range(1, self.num_time_steps)
        }
        self.wind_delta_shape = {
            t: abs(self.wind_shape[t] - self.wind_shape[t - 1])
            for t in range(1, self.num_time_steps)
        }

        for i in self.wind_physical_generator_ids:
            installed_capacity = self.static_physical_capacity[i]
            if installed_capacity <= 0 or not np.isfinite(installed_capacity):
                generator_name = self.physical_generator_names[i]
                raise ValueError(
                    f"Wind generator {generator_name} must have positive installed capacity"
                )

    def compute_deterministic_p_init(self) -> list[float]:
        """Compute p_init from the merit-order dispatch at the context scenario's
        deterministic (mu_D, mu_W) operating point at t=0.

        Mirrors DRO_PoAOptimization.compute_deterministic_p_init so both
        pipelines seed the ramp constraints consistently.  Returns one value
        per physical generator.
        """
        row = self.scenarios_df.iloc[0]
        mu_D = float(row["mu_D"]) if "mu_D" in self.scenarios_df.columns else 1.0
        mu_W = float(row["mu_W"]) if "mu_W" in self.scenarios_df.columns else 1.0
        D0 = float(self.demand_D_ref * mu_D * self.demand_shape[0])

        wind_ids = set(int(i) for i in self.wind_physical_generator_ids)
        blocks: list[tuple[float, float, int]] = []
        for i, b in self.generator_block_pairs:
            global_b = self.local_to_global_block[(int(i), int(b))]
            cost = self.block_cost_vector[global_b]
            if int(i) in wind_ids:
                phys_cap_0 = (
                    self.static_physical_capacity[int(i)] * mu_W * self.wind_shape[0]
                )
                n_blocks = len(self.local_blocks_by_generator[int(i)])
                block_cap = phys_cap_0 / n_blocks
            else:
                block_cap = float(self.static_block_capacity[global_b])
            blocks.append((cost, block_cap, int(i)))

        blocks.sort(key=lambda x: x[0])
        p_dispatch: dict[int, float] = {i: 0.0 for i in range(self.num_physical_generators)}
        remaining = D0
        for cost, cap, i in blocks:
            dispatched = min(cap, remaining)
            p_dispatch[i] += dispatched
            remaining -= dispatched
            if remaining <= 1e-9:
                break

        return [p_dispatch[i] for i in range(self.num_physical_generators)]

    @staticmethod
    def load_ambiguity_set(
        config_path: str = "models/PoA/ambiguity_set_config.yaml",
        config_name: Optional[str] = None,
    ) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Ambiguity-set config not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle) or {}
        if "ambiguity_sets" not in raw_config:
            return raw_config
        ambiguity_sets = raw_config.get("ambiguity_sets")
        if not isinstance(ambiguity_sets, dict) or not ambiguity_sets:
            raise ValueError("'ambiguity_sets' must be a non-empty mapping")
        selected_name = (
            config_name
            or raw_config.get("default_ambiguity_set")
            or next(iter(ambiguity_sets))
        )
        if selected_name not in ambiguity_sets:
            raise ValueError(
                f"Unknown ambiguity-set config '{selected_name}'. "
                f"Available: {', '.join(ambiguity_sets.keys())}"
            )
        return ambiguity_sets[selected_name] or {}

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        if self.use_default_bounds:
            self.ensure_default_bounds_available(
                include_nn_relu_bounds=bool(self.nn_policy_generator_ids),
                include_tight_big_m=True,
                include_optimal_cost_bounds=(
                    self.objective_mode
                    in {"mccormick", "piecewise_mccormick"}
                ),
                overwrite_existing=False,
            )
        self._ensure_loaded_bounds_prepared()
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
        self._build_ambiguity_set_variables()
        self.model.D = Var(self.model.time_steps, domain=NonNegativeReals)
        self.model.P_max_block = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.C_eq = Var(domain=Reals)
        c_opt_bounds = (
            self.mccormick_bounds["C_opt"]
            if self.objective_mode in {
                "mccormick",
                "piecewise_mccormick",
            } and self.mccormick_bounds is not None
            else (None, None)
        )
        self.model.C_opt = Var(domain=Reals, bounds=c_opt_bounds)
        if self.objective_mode in {"mccormick", "piecewise_mccormick"}:
            self._build_mccormick_variables()
        else:
            self.model.PoA = Var(domain=Reals)
        if self.objective_mode == "piecewise_mccormick":
            self._build_piecewise_mccormick_variables()

    def _build_ambiguity_set_variables(self) -> None:
        self.model.mu_D = Var(bounds=self.mu_D_bounds)
        self.model.sigma_D = Var(bounds=self.sigma_D_bounds)
        self.model.mu_W = Var(bounds=self.mu_W_bounds)
        self.model.sigma_W = Var(bounds=self.sigma_W_bounds)

        # Auxiliary variables for support set deviations and budgets
        self.model.D_abs_deviation = Var(self.model.time_steps, domain=NonNegativeReals)
        self.model.P_max_phys_abs_deviation = Var(self.model.wind_physical_generators, self.model.time_steps, domain=NonNegativeReals)

    def _build_equilibrium_variables(self) -> None:
        self._ensure_loaded_bounds_prepared()
        self.model.P_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.alpha = Var(self.model.generator_blocks, self.model.time_steps, domain=Reals)
        self.model.lambda_eq = Var(
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, t: self.lambda_eq_bounds[int(t)],
        )
        self.model.mu_upper_eq = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self.M_mu_upper_eq[int(i), int(b), int(t)],
            ),
        )
        self.model.mu_lower_eq = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self.M_mu_lower_eq[int(i), int(b), int(t)],
            ),
        )
        self.model.mu_ramp_up_eq = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self.M_mu_ramp_up_eq[int(i), int(t)]
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )
        self.model.mu_ramp_down_eq = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self.M_mu_ramp_down_eq[int(i), int(t)]
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )

    def _build_complementarity_equilibrium_variables(self) -> None:
        self.model.z_upper_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_lower_eq = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_eq = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_eq = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)

    def _build_optimal_variables(self) -> None:
        self._ensure_loaded_bounds_prepared()
        self.model.P_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=NonNegativeReals)
        self.model.lambda_opt = Var(
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, t: self.lambda_opt_bounds[int(t)],
        )
        self.model.mu_upper_opt = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self.M_mu_upper_opt[int(i), int(b), int(t)],
            ),
        )
        self.model.mu_lower_opt = Var(
            self.model.generator_blocks,
            self.model.time_steps,
            domain=Reals,
            bounds=lambda m, i, b, t: (
                0.0,
                self.M_mu_lower_opt[int(i), int(b), int(t)],
            ),
        )
        self.model.mu_ramp_up_opt = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self.M_mu_ramp_up_opt[int(i), int(t)]
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )
        self.model.mu_ramp_down_opt = Var(
            self.model.physical_generators,
            self.model.time_steps_plus_1,
            domain=Reals,
            bounds=lambda m, i, t: (
                0.0,
                self.M_mu_ramp_down_opt[int(i), int(t)]
                if int(t) < self.num_time_steps
                else 0.0,
            ),
        )

    def _build_complementarity_optimal_variables(self) -> None:
        self.model.z_upper_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_lower_opt = Var(self.model.generator_blocks, self.model.time_steps, domain=Binary)
        self.model.z_ramp_up_opt = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)
        self.model.z_ramp_down_opt = Var(self.model.physical_generators, self.model.time_steps, domain=Binary)

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
        # Difference proxy objective: maximize C_eq - C_opt.
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
            return (
                -self.M_cap[int(i), int(b), int(t)] * (1 - m.z_upper_eq[i, b, t])
                <= m.P_eq[i, b, t] - m.P_max_block[i, b, t]
            )

        def upper_bound_complementarity_dual_eq_rule(m, i, b, t):
            return (
                m.mu_upper_eq[i, b, t]
                <= self.M_mu_upper_eq[int(i), int(b), int(t)]
                * m.z_upper_eq[i, b, t]
            )

        def lower_bound_complementarity_eq_rule(m, i, b, t):
            return (
                -self.M_lower[int(i), int(b), int(t)] * (1 - m.z_lower_eq[i, b, t])
                <= -m.P_eq[i, b, t]
            )

        def lower_bound_complementarity_dual_eq_rule(m, i, b, t):
            return (
                m.mu_lower_eq[i, b, t]
                <= self.M_mu_lower_eq[int(i), int(b), int(t)]
                * m.z_lower_eq[i, b, t]
            )

        def ramp_up_complementarity_eq_rule(m, i, t):
            return -self.M_ramp_up[int(i), int(t)] * (1 - m.z_ramp_up_eq[i, t]) <= (
                sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_eq_rule(m, i):
            return -self.M_ramp_up_initial[int(i)] * (1 - m.z_ramp_up_eq[i, 0]) <= (
                sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_eq_rule(m, i, t):
            return (
                m.mu_ramp_up_eq[i, t]
                <= self.M_mu_ramp_up_eq[int(i), int(t)] * m.z_ramp_up_eq[i, t]
            )
        
        def ramp_down_complementarity_eq_rule(m, i, t):
            return -self.M_ramp_down[int(i), int(t)] * (1 - m.z_ramp_down_eq[i, t]) <= (
                -sum(m.P_eq[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_eq[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_eq_rule(m, i):
            return -self.M_ramp_down_initial[int(i)] * (1 - m.z_ramp_down_eq[i, 0]) <= (
                -sum(m.P_eq[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_eq_rule(m, i, t):
            return (
                m.mu_ramp_down_eq[i, t]
                <= self.M_mu_ramp_down_eq[int(i), int(t)] * m.z_ramp_down_eq[i, t]
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
            return (
                -self.M_cap[int(i), int(b), int(t)] * (1 - m.z_upper_opt[i, b, t])
                <= m.P_opt[i, b, t] - m.P_max_block[i, b, t]
            )
    
        def upper_bound_complementarity_dual_opt_rule(m, i, b, t):
            return (
                m.mu_upper_opt[i, b, t]
                <= self.M_mu_upper_opt[int(i), int(b), int(t)]
                * m.z_upper_opt[i, b, t]
            )

        def lower_bound_complementarity_opt_rule(m, i, b, t):
            return (
                -self.M_lower[int(i), int(b), int(t)] * (1 - m.z_lower_opt[i, b, t])
                <= -m.P_opt[i, b, t]
            )

        def lower_bound_complementarity_dual_opt_rule(m, i, b, t):
            return (
                m.mu_lower_opt[i, b, t]
                <= self.M_mu_lower_opt[int(i), int(b), int(t)]
                * m.z_lower_opt[i, b, t]
            )
        
        def ramp_up_complementarity_opt_rule(m, i, t):
            return -self.M_ramp_up[int(i), int(t)] * (1 - m.z_ramp_up_opt[i, t]) <= (
                sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                - sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_initial_complementarity_opt_rule(m, i):
            return -self.M_ramp_up_initial[int(i)] * (1 - m.z_ramp_up_opt[i, 0]) <= (
                sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                - self.p_init[int(i)]
                - self.ramp_vector_up[int(i)]
            )

        def ramp_up_complementarity_dual_opt_rule(m, i, t):
            return (
                m.mu_ramp_up_opt[i, t]
                <= self.M_mu_ramp_up_opt[int(i), int(t)] * m.z_ramp_up_opt[i, t]
            )

        def ramp_down_complementarity_opt_rule(m, i, t):
            return -self.M_ramp_down[int(i), int(t)] * (1 - m.z_ramp_down_opt[i, t]) <= (
                -sum(m.P_opt[i, b, t] for b in self.local_blocks_by_generator[int(i)])
                + sum(m.P_opt[i, b, t - 1] for b in self.local_blocks_by_generator[int(i)])
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_initial_complementarity_opt_rule(m, i):
            return -self.M_ramp_down_initial[int(i)] * (1 - m.z_ramp_down_opt[i, 0]) <= (
                -sum(m.P_opt[i, b, 0] for b in self.local_blocks_by_generator[int(i)])
                + self.p_init[int(i)]
                - self.ramp_vector_down[int(i)]
            )

        def ramp_down_complementarity_dual_opt_rule(m, i, t):
            return (
                m.mu_ramp_down_opt[i, t]
                <= self.M_mu_ramp_down_opt[int(i), int(t)] * m.z_ramp_down_opt[i, t]
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
    # PoA constraints
    # ------------------------------------------------------------------

    def _get_equilibrium_cost_expression(self, m: ConcreteModel) -> Any:
        return sum(
            self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
            * m.P_eq[i, b, t]
            for (i, b) in m.generator_blocks
            for t in m.time_steps
        )

    def _get_optimal_cost_expression(self, m: ConcreteModel) -> Any:
        return sum(
            self.block_cost_vector[self.local_to_global_block[(int(i), int(b))]]
            * m.P_opt[i, b, t]
            for (i, b) in m.generator_blocks
            for t in m.time_steps
        )

    def _build_PoA_constraints(self) -> None:
        def cost_eq_rule(m):
            return m.C_eq == self._get_equilibrium_cost_expression(m)

        def cost_opt_rule(m):
            return m.C_opt == self._get_optimal_cost_expression(m)
        
        self.model.cost_definition_eq = Constraint(rule=cost_eq_rule)
        self.model.cost_definition_opt = Constraint(rule=cost_opt_rule)
        if self.objective_mode == "difference":
            def PoA_rule(m):
                return m.C_eq - m.C_opt == m.PoA

            self.model.poa_definition = Constraint(rule=PoA_rule)
        if self.objective_mode == "mccormick":
            self._build_mccormick_constraints()
        if self.objective_mode == "piecewise_mccormick":
            self._build_piecewise_mccormick_constraints()

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

if __name__ == "__main__":
    case = "base_test_case"
    regime_set = "PoA_analysis"
    seed = 1
    horizon = 4
    nn_relu_report_path = "results/poa_tightening/relu_bounds_report.json"
    tightening_report_path = "results/poa_tightening/final_tightening_report.json"
    diagnostic_mode = False

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]

    ambiguity_set_config = PoAOptimization.load_ambiguity_set(
        config_path="config/ambiguity_set_config.yaml",
        config_name="base_test_case",
    )

    optimizer_kwargs = {
        "scenarios_df": scenarios_df,
        "costs_df": costs_df,
        "ramps_df": ramps_df,
        "p_init": None,
        "num_time_steps": horizon,
        "ambiguity_set_config": ambiguity_set_config,
        "nn_model_dir": "models/neural_network/training/trained_models",
        "nn_normalization_stats_path": (
            "models/neural_network/features/generated/normalized/min_max_stats.json"
        ),
        # None means all generators with available NN files. Use [] for true
        # costs only, or e.g. ["G2", "W1"] / [1, 2] for a selected subset.
        "nn_policy_generators": [1, 2],
        "reference_case": case,
    }

    optimizer = PoAOptimization(**optimizer_kwargs)

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
