# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-14
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Regime-based stochastic scenario generation for intertemporal studies."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm as _norm


if __package__ in {None, ""}:
	sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.utils.cases_utils import load_setup_data, normalize_generators


@dataclass(frozen=True)
class RegimeParameters:
	"""Container for one stochastic regime."""

	name: str
	n_scenarios: int
	mu_D: float
	rho_D: float
	sigma_D: float
	mu_W: float
	peak_W: float
	rho_W: float
	sigma_W: float


class ScenarioManager:
	"""Generate stochastic demand and wind scenarios from YAML-defined regimes."""

	DEFAULT_REGIME_CONFIG_PATH = Path(__file__).resolve().parents[1] / "regime_definitions.yaml"
	AMBIGUITY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "ambiguity_set_config.yaml"

	def __init__(self, base_case_reference: str = "base_test_case") -> None:
		self.base_case_reference = base_case_reference

		(
			num_generators,
			pmax_list,
			pmin_list,
			cost_vector,
			r_rates_up_list,
			r_rates_down_list,
			demand,
			generators,
			players,
		) = load_setup_data(self.base_case_reference)

		self.base_case = {
			"case_name": self.base_case_reference,
			"num_generators": num_generators,
			"generators": generators,
			"players": players,
			"demand": float(demand),
			"pmax_list": [float(v) for v in pmax_list],
			"pmin_list": [float(v) for v in pmin_list],
			"cost_vector": [float(v) for v in cost_vector],
			"r_rates_up_list": [float(v) for v in r_rates_up_list],
			"r_rates_down_list": [float(v) for v in r_rates_down_list],
		}
		self.normalized_generators = normalize_generators(self.base_case["generators"])
		self.physical_generators = self.normalized_generators["physical_generators"]
		self.blocks = self.normalized_generators["blocks"]
		self.block_names = self.normalized_generators["block_names"]
		self.physical_generator_names = self.normalized_generators["physical_generator_names"]
		self.block_to_physical = self.normalized_generators["block_to_physical"]
		self.physical_to_blocks = self.normalized_generators["physical_to_blocks"]

		self.players_config = self.base_case["players"]

		self.wind_generator_indices = [
			i
			for i, gen in enumerate(self.physical_generators)
			if bool(gen["is_wind"])
		]
		self.conventional_generator_indices = [
			i for i in range(len(self.physical_generators)) if i not in self.wind_generator_indices
		]

		self.total_conventional_capacity = float(
			sum(self.physical_generators[i]["pmax"] for i in self.conventional_generator_indices)
		)

	@staticmethod
	def _generator_name(generator: Any) -> str:
		return generator["name"] if isinstance(generator, dict) else str(generator)

	@staticmethod
	def _is_wind_generator(generator_name: str) -> bool:
		return generator_name.startswith("W")

	@staticmethod
	def _get_value(
		source: Dict[str, Any],
		keys: List[str],
		default: Any = None,
		required: bool = False,
		context: str = "",
	) -> Any:
		for key in keys:
			if key in source:
				return source[key]

		if required:
			keys_text = ", ".join(keys)
			raise ValueError(f"Missing required field ({keys_text}) in {context}")

		return default

	@staticmethod
	def _to_float(value: Any, name: str, regime_name: str) -> float:
		try:
			return float(value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"Regime '{regime_name}': field '{name}' must be numeric") from exc

	@staticmethod
	def _to_int(value: Any, name: str, regime_name: str) -> int:
		try:
			converted = int(value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"Regime '{regime_name}': field '{name}' must be integer") from exc

		if float(value) != float(converted):
			raise ValueError(f"Regime '{regime_name}': field '{name}' must be an integer value")

		return converted

	def _parse_regime(self, regime_raw: Dict[str, Any], index: int) -> RegimeParameters:
		if not isinstance(regime_raw, dict):
			raise ValueError(f"Regime entry at index {index} must be a mapping")

		if "name" not in regime_raw:
			raise ValueError(f"Regime entry at index {index} is missing required field 'name'")
		regime_name = str(regime_raw["name"]).strip()
		if not regime_name:
			raise ValueError(f"Regime entry at index {index} has an empty 'name'")

		n_scenarios = self._to_int(
			self._get_value(
				regime_raw,
				["n_scenarios", "n", "num_scenarios"],
				required=True,
				context=f"regime '{regime_name}'",
			),
			"n_scenarios",
			regime_name,
		)

		mu_D = self._to_float(
			self._get_value(regime_raw, ["mu_D"], required=True, context=regime_name),
			"mu_D",
			regime_name,
		)
		rho_D = self._to_float(
			self._get_value(regime_raw, ["rho_D"], required=True, context=regime_name),
			"rho_D",
			regime_name,
		)
		sigma_D = self._to_float(
			self._get_value(regime_raw, ["sigma_D"], required=True, context=regime_name),
			"sigma_D",
			regime_name,
		)
		mu_W = self._to_float(
			self._get_value(regime_raw, ["mu_W"], required=True, context=regime_name),
			"mu_W",
			regime_name,
		)
		peak_W = self._to_float(
			self._get_value(
				regime_raw,
				["peak_W", "tau_W"],
				required=True,
				context=f"regime '{regime_name}'",
			),
			"peak_W",
			regime_name,
		)
		rho_W = self._to_float(
			self._get_value(regime_raw, ["rho_W"], required=True, context=regime_name),
			"rho_W",
			regime_name,
		)
		sigma_W = self._to_float(
			self._get_value(regime_raw, ["sigma_W"], required=True, context=regime_name),
			"sigma_W",
			regime_name,
		)

		if n_scenarios <= 0:
			raise ValueError(f"Regime '{regime_name}': n_scenarios must be positive")
		if mu_D <= 0:
			raise ValueError(f"Regime '{regime_name}': mu_D must be positive")
		if not -0.999 <= rho_D <= 0.999:
			raise ValueError(f"Regime '{regime_name}': rho_D must be in [-0.999, 0.999]")
		if sigma_D < 0:
			raise ValueError(f"Regime '{regime_name}': sigma_D must be non-negative")
		if not 0 <= mu_W <= 1:
			raise ValueError(f"Regime '{regime_name}': mu_W must be in [0, 1]")
		if not 0 <= peak_W <= 24:
			raise ValueError(f"Regime '{regime_name}': peak_W/tau_W must be in [0, 24]")
		if not -0.999 <= rho_W <= 0.999:
			raise ValueError(f"Regime '{regime_name}': rho_W must be in [-0.999, 0.999]")
		if sigma_W < 0:
			raise ValueError(f"Regime '{regime_name}': sigma_W must be non-negative")

		return RegimeParameters(
			name=regime_name,
			n_scenarios=n_scenarios,
			mu_D=mu_D,
			rho_D=rho_D,
			sigma_D=sigma_D,
			mu_W=mu_W,
			peak_W=peak_W,
			rho_W=rho_W,
			sigma_W=sigma_W,
		)

	def load_regime_configuration(
		self,
		regime_config_path: Optional[str] = None,
		regime_set: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Load and validate regime configuration from YAML."""
		config_path = Path(regime_config_path) if regime_config_path is not None else self.DEFAULT_REGIME_CONFIG_PATH
		if not config_path.is_absolute() and not config_path.exists():
			repo_relative = Path(__file__).resolve().parents[2] / config_path
			config_path = repo_relative if repo_relative.exists() else Path(__file__).resolve().parents[1] / config_path.name
		if not config_path.exists():
			raise FileNotFoundError(f"Regime YAML was not found: {config_path}")

		with open(config_path, "r", encoding="utf-8") as file:
			raw_config = yaml.safe_load(file) or {}

		if "regime_sets" in raw_config:
			regime_sets = raw_config.get("regime_sets")
			if not isinstance(regime_sets, dict) or not regime_sets:
				raise ValueError("'regime_sets' must be a non-empty mapping")

			selected_name = regime_set or raw_config.get("default_regime_set") or next(iter(regime_sets))
			if selected_name not in regime_sets:
				available = ", ".join(regime_sets.keys())
				raise ValueError(f"Unknown regime_set '{selected_name}'. Available: {available}")

			selected_config = regime_sets[selected_name] or {}
		else:
			selected_name = regime_set or str(raw_config.get("name", "default"))
			selected_config = raw_config

		regimes_raw = selected_config.get("regimes")
		if not isinstance(regimes_raw, list) or not regimes_raw:
			raise ValueError("Regime configuration must include a non-empty 'regimes' list")

		max_draw_attempts = int(selected_config.get("max_draw_attempts", raw_config.get("max_draw_attempts", 500)))
		if max_draw_attempts <= 0:
			raise ValueError("max_draw_attempts must be a positive integer")

		enforce_feasibility = bool(
			selected_config.get(
				"enforce_dispatch_feasibility",
				raw_config.get("enforce_dispatch_feasibility", True),
			)
		)
		contingency_margin = float(
			selected_config.get(
				"n_minus_one_margin",
				selected_config.get(
					"contingency_margin",
					raw_config.get("n_minus_one_margin", raw_config.get("contingency_margin", 0.0)),
				),
			)
		)
		if contingency_margin < 0:
			raise ValueError("n_minus_one_margin/contingency_margin must be non-negative")

		seed = selected_config.get("seed", raw_config.get("seed"))
		if seed is not None:
			seed = int(seed)

		parsed_regimes = [self._parse_regime(regime_raw, idx) for idx, regime_raw in enumerate(regimes_raw)]

		return {
			"name": selected_name,
			"seed": seed,
			"max_draw_attempts": max_draw_attempts,
			"enforce_dispatch_feasibility": enforce_feasibility,
			"n_minus_one_margin": contingency_margin,
			"regimes": parsed_regimes,
			"raw": selected_config,
		}

	def load_ambiguity_set_configuration(
		self,
		ambiguity_config_path: Optional[str] = None,
		ambiguity_set: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Load and validate ambiguity-set configuration from YAML."""
		config_path = (
			Path(ambiguity_config_path)
			if ambiguity_config_path is not None
			else self.AMBIGUITY_CONFIG_PATH
		)
		if not config_path.is_absolute() and not config_path.exists():
			repo_relative = Path(__file__).resolve().parents[2] / config_path
			config_path = (
				repo_relative
				if repo_relative.exists()
				else Path(__file__).resolve().parents[1] / config_path.name
			)
		if not config_path.exists():
			raise FileNotFoundError(f"Ambiguity-set YAML was not found: {config_path}")

		with open(config_path, "r", encoding="utf-8") as file:
			raw_config = yaml.safe_load(file) or {}
		if not isinstance(raw_config, dict):
			raise ValueError("Ambiguity configuration YAML must contain a mapping at the top level")

		ambiguity_sets = raw_config.get("ambiguity_sets")
		if not isinstance(ambiguity_sets, dict) or not ambiguity_sets:
			raise ValueError("Ambiguity configuration must include a non-empty 'ambiguity_sets' mapping")

		selected_name = ambiguity_set or raw_config.get("default_ambiguity_set") or next(iter(ambiguity_sets))
		if selected_name not in ambiguity_sets:
			available = ", ".join(ambiguity_sets.keys())
			raise ValueError(f"Unknown ambiguity_set '{selected_name}'. Available: {available}")

		selected_config = ambiguity_sets[selected_name] or {}
		if not isinstance(selected_config, dict):
			raise ValueError(f"Ambiguity set '{selected_name}' must be a mapping")

		def require_mapping(source: Dict[str, Any], key: str, context: str) -> Dict[str, Any]:
			value = source.get(key)
			field_name = f"{context}.{key}" if context else key
			if not isinstance(value, dict):
				raise ValueError(f"Ambiguity set '{selected_name}': '{field_name}' must be a mapping")
			return value

		def require_float(source: Dict[str, Any], key: str, context: str) -> float:
			field_name = f"{context}.{key}" if context else key
			if key not in source:
				raise ValueError(f"Ambiguity set '{selected_name}': missing required field '{field_name}'")
			try:
				return float(source[key])
			except (TypeError, ValueError) as exc:
				raise ValueError(
					f"Ambiguity set '{selected_name}': field '{field_name}' must be numeric"
				) from exc

		demand = require_mapping(selected_config, "demand", "")
		demand_mu = require_mapping(demand, "mu", "demand")
		demand_sigma = require_mapping(demand, "sigma", "demand")
		wind = require_mapping(selected_config, "wind", "")
		wind_mu = require_mapping(wind, "mu", "wind")
		wind_sigma = require_mapping(wind, "sigma", "wind")

		demand_mu_min = require_float(demand_mu, "min", "demand.mu")
		demand_mu_max = require_float(demand_mu, "max", "demand.mu")
		demand_sigma_min = require_float(demand_sigma, "min", "demand.sigma")
		demand_sigma_max = require_float(demand_sigma, "max", "demand.sigma")
		rho_D = require_float(demand, "rho_fixed", "demand")
		wind_mu_min = require_float(wind_mu, "min", "wind.mu")
		wind_mu_max = require_float(wind_mu, "max", "wind.mu")
		wind_sigma_min = require_float(wind_sigma, "min", "wind.sigma")
		wind_sigma_max = require_float(wind_sigma, "max", "wind.sigma")
		rho_W = require_float(wind, "rho_fixed", "wind")
		peak_W = require_float(wind, "tau_fixed", "wind")

		if demand_mu_min > demand_mu_max:
			raise ValueError(f"Ambiguity set '{selected_name}': demand.mu.min must be <= demand.mu.max")
		if demand_sigma_min > demand_sigma_max:
			raise ValueError(
				f"Ambiguity set '{selected_name}': demand.sigma.min must be <= demand.sigma.max"
			)
		if wind_mu_min > wind_mu_max:
			raise ValueError(f"Ambiguity set '{selected_name}': wind.mu.min must be <= wind.mu.max")
		if wind_sigma_min > wind_sigma_max:
			raise ValueError(f"Ambiguity set '{selected_name}': wind.sigma.min must be <= wind.sigma.max")
		if demand_mu_min <= 0:
			raise ValueError(f"Ambiguity set '{selected_name}': demand mu bounds must be positive")
		if demand_sigma_min < 0:
			raise ValueError(f"Ambiguity set '{selected_name}': demand sigma bounds must be non-negative")
		if not 0 <= wind_mu_min <= 1 or not 0 <= wind_mu_max <= 1:
			raise ValueError(f"Ambiguity set '{selected_name}': wind mu bounds must be in [0, 1]")
		if wind_sigma_min < 0:
			raise ValueError(f"Ambiguity set '{selected_name}': wind sigma bounds must be non-negative")
		if not -0.999 <= rho_D <= 0.999:
			raise ValueError(f"Ambiguity set '{selected_name}': demand.rho_fixed must be in [-0.999, 0.999]")
		if not -0.999 <= rho_W <= 0.999:
			raise ValueError(f"Ambiguity set '{selected_name}': wind.rho_fixed must be in [-0.999, 0.999]")
		if not 0 <= peak_W <= 24:
			raise ValueError(f"Ambiguity set '{selected_name}': wind.tau_fixed must be in [0, 24]")

		max_draw_attempts = int(
			selected_config.get("max_draw_attempts", raw_config.get("max_draw_attempts", 500))
		)
		if max_draw_attempts <= 0:
			raise ValueError("max_draw_attempts must be a positive integer")

		enforce_feasibility = bool(
			selected_config.get(
				"enforce_dispatch_feasibility",
				raw_config.get("enforce_dispatch_feasibility", True),
			)
		)
		contingency_margin = float(
			selected_config.get(
				"n_minus_one_margin",
				selected_config.get(
					"contingency_margin",
					raw_config.get("n_minus_one_margin", raw_config.get("contingency_margin", 0.0)),
				),
			)
		)
		if contingency_margin < 0:
			raise ValueError("n_minus_one_margin/contingency_margin must be non-negative")

		seed = selected_config.get("seed", raw_config.get("seed"))
		if seed is not None:
			seed = int(seed)

		kappa = selected_config.get("kappa", raw_config.get("kappa"))
		if kappa is not None:
			kappa = float(kappa)

		return {
			"name": selected_name,
			"description": str(selected_config.get("description", "")),
			"demand_mu_min": demand_mu_min,
			"demand_mu_max": demand_mu_max,
			"demand_sigma_min": demand_sigma_min,
			"demand_sigma_max": demand_sigma_max,
			"rho_D": rho_D,
			"wind_mu_min": wind_mu_min,
			"wind_mu_max": wind_mu_max,
			"wind_sigma_min": wind_sigma_min,
			"wind_sigma_max": wind_sigma_max,
			"rho_W": rho_W,
			"peak_W": peak_W,
			"kappa": kappa,
			"seed": seed,
			"max_draw_attempts": max_draw_attempts,
			"enforce_dispatch_feasibility": enforce_feasibility,
			"n_minus_one_margin": contingency_margin,
			"raw": selected_config,
		}

	def _draw_regime_parameters_from_ambiguity_set(
		self,
		ambiguity_config: Dict[str, Any],
		rng: np.random.Generator,
		scenario_index: int,
	) -> RegimeParameters:
		"""Draw one stochastic-process parameter vector from an ambiguity set."""
		return RegimeParameters(
			name=f"{ambiguity_config['name']}_draw_{scenario_index}",
			n_scenarios=1,
			mu_D=float(rng.uniform(ambiguity_config["demand_mu_min"], ambiguity_config["demand_mu_max"])),
			rho_D=float(ambiguity_config["rho_D"]),
			sigma_D=float(
				rng.uniform(ambiguity_config["demand_sigma_min"], ambiguity_config["demand_sigma_max"])
			),
			mu_W=float(rng.uniform(ambiguity_config["wind_mu_min"], ambiguity_config["wind_mu_max"])),
			peak_W=float(ambiguity_config["peak_W"]),
			rho_W=float(ambiguity_config["rho_W"]),
			sigma_W=float(
				rng.uniform(ambiguity_config["wind_sigma_min"], ambiguity_config["wind_sigma_max"])
			),
		)

	@staticmethod
	def _build_demand_shape(horizon: int) -> np.ndarray:
		"""Create a positive mean-one daily demand multiplier profile."""
		if horizon <= 0:
			raise ValueError(f"horizon must be positive, got {horizon}")

		if horizon == 1:
			return np.array([1.0], dtype=float)

		hours = np.linspace(0.0, 24.0, horizon)
		morning_peak = np.exp(-0.5 * ((hours - 7.5) / 2.5) ** 2)
		evening_peak = 1.35 * np.exp(-0.5 * ((hours - 15.0) / 3.0) ** 2)
		midday_dip = -0.35 * np.exp(-0.5 * ((hours - 11.0) / 2.0) ** 2)

		raw_shape = morning_peak + evening_peak + midday_dip
		centered = raw_shape - np.mean(raw_shape)
		scale = np.max(np.abs(centered))

		if scale <= 1e-12:
			return np.ones(horizon, dtype=float)

		multiplier = 1.0 + 0.125 * (centered / scale)
		multiplier = np.maximum(multiplier, 1e-9)
		return multiplier / np.mean(multiplier)

	@staticmethod
	def _build_wind_shape(horizon: int, peak_hour: float, width: float = 4.0) -> np.ndarray:
		"""Create a positive mean-one circular wind multiplier profile."""
		if horizon <= 0:
			raise ValueError(f"horizon must be positive, got {horizon}")
		if not 0 <= peak_hour <= 24:
			raise ValueError(f"peak_hour must be in [0, 24], got {peak_hour}")
		if width <= 0:
			raise ValueError(f"width must be positive, got {width}")

		if horizon == 1:
			return np.array([1.0], dtype=float)

		hours = np.linspace(0.0, 24.0, horizon, endpoint=False)
		peak_hour = peak_hour % 24.0
		circular_distance = np.abs(((hours - peak_hour + 12.0) % 24.0) - 12.0)
		raw_shape = np.exp(-0.5 * (circular_distance / width) ** 2)
		centered = raw_shape - np.mean(raw_shape)
		scale = np.max(np.abs(centered))

		if scale <= 1e-12:
			return np.ones(horizon, dtype=float)

		multiplier = 1.0 + 0.15 * (centered / scale)
		multiplier = np.maximum(multiplier, 1e-9)
		return multiplier / np.mean(multiplier)

	def _simulate_single_scenario(
		self,
		regime: RegimeParameters,
		rng: np.random.Generator,
	) -> Tuple[List[float], Dict[str, List[float]], float]:
		"""Draw demand and physical-generator wind availability profiles."""
		horizon = self.base_case["time_steps"]
		demand_shape = self._build_demand_shape(horizon)
		reference_demand = self.base_case["demand"]
		# Demand is simulated in reference-demand units, so sigma_D is dimensionless like sigma_W,
		# and the final demand profile is scaled back to MW.
		demand_mean = reference_demand * regime.mu_D
		wind_shape = self._build_wind_shape(horizon, regime.peak_W)
		# The deterministic shape functions are mean-one multiplier profiles.
		# Intraday amplitude is embedded in the shapes, not in regime parameters.
		demand_trend = regime.mu_D * demand_shape
		wind_trend = regime.mu_W * wind_shape

		z_d = rng.standard_normal(horizon)

		demand_residual = np.zeros(horizon, dtype=float)
		demand_residual[0] = regime.sigma_D * z_d[0]

		for t in range(1, horizon):
			demand_residual[t] = regime.rho_D * demand_residual[t - 1] + regime.sigma_D * z_d[t]

		demand_profile = reference_demand * np.maximum(demand_trend + demand_residual, 0.0)

		generator_wind_profiles: Dict[str, List[float]] = {}

		# Wind availability is generated separately for each physical wind generator:
		# they share the regime-level deterministic shape and persistence parameters,
		# but draw independent generator-specific innovations. This keeps the process
		# interpretable without an extra wind_idiosyncratic_share parameter.
		for gen_idx in self.wind_generator_indices:
			generator = self.physical_generators[gen_idx]
			gen_name = generator["physical_name"]
			nominal_capacity = float(generator["pmax"])

			z_w = rng.standard_normal(horizon)

			wind_residual = np.zeros(horizon, dtype=float)
			wind_residual[0] = regime.sigma_W * z_w[0]

			for t in range(1, horizon):
				wind_residual[t] = regime.rho_W * wind_residual[t - 1] + regime.sigma_W * z_w[t]

			wind_factor = np.clip(wind_trend + wind_residual, 0.0, 1.0)
			generator_wind_profiles[gen_name] = [float(value * nominal_capacity) for value in wind_factor]

		return [float(v) for v in demand_profile], generator_wind_profiles, demand_mean

	def _available_capacity_by_generator(
		self,
		generator_wind_profiles: Dict[str, List[float]],
		t: int,
	) -> Dict[int, float]:
		"""Return available capacity at time t keyed by physical generator id."""
		available_capacity: Dict[int, float] = {}
		for generator in self.physical_generators:
			physical_id = int(generator["physical_id"])
			physical_name = generator["physical_name"]
			if bool(generator["is_wind"]):
				available_capacity[physical_id] = float(generator_wind_profiles[physical_name][t])
			else:
				available_capacity[physical_id] = float(generator["pmax"])
		return available_capacity

	def _is_dispatch_feasible(
		self,
		demand_profile: List[float],
		generator_wind_profiles: Dict[str, List[float]],
		contingency_margin: float = 0.0,
		enforce_n_minus_one: bool = True,
	) -> bool:
		"""Check base feasibility and optionally N-1 contingency across all time steps.

		A scenario is feasible only if:
		1) Total available capacity can serve demand at each time step.
		2) (When enforce_n_minus_one=True) Demand can still be served when any
		   single generator is unavailable (N-1).
		"""
		horizon = self.base_case["time_steps"]

		for t in range(horizon):
			available_capacity_t = self._available_capacity_by_generator(generator_wind_profiles, t)
			total_capacity_t = float(sum(available_capacity_t.values()))
			demand_t = float(demand_profile[t])

			# Base feasibility (no outage).
			if demand_t + contingency_margin > total_capacity_t + 1e-9:
				return False

			if not enforce_n_minus_one:
				continue

			# N-1 contingency: demand must still be met without any single physical generator.
			for generator_capacity_t in available_capacity_t.values():
				remaining_capacity_t = total_capacity_t - generator_capacity_t
				if demand_t + contingency_margin > remaining_capacity_t + 1e-9:
					return False

		return True

	def _is_within_support_set(
		self,
		regime: RegimeParameters,
		demand_profile: List[float],
		generator_wind_profiles: Dict[str, List[float]],
	) -> bool:
		"""Return True iff the scenario satisfies all DRO support set constraints.

		Mirrors support_set.py: point-wise level bounds (kappa=1.96) and AR(1)
		innovation bounds (joint 95% coverage across T steps).
		"""
		kappa = 1.96
		T = len(demand_profile)
		kappa_ar1 = float(_norm.ppf((1.0 + 0.95 ** (1.0 / T)) / 2.0))
		tol = 1e-9

		D_ref = float(self.base_case["demand"])
		demand_shape = self._build_demand_shape(T)
		wind_shape = self._build_wind_shape(T, regime.peak_W)

		D = np.asarray(demand_profile, dtype=float)
		D_reference = D_ref * regime.mu_D * demand_shape
		stationary_std_D = regime.sigma_D / np.sqrt(1.0 - regime.rho_D ** 2)
		margin_D = kappa * D_ref * stationary_std_D

		if np.any(D < np.maximum(D_reference - margin_D, 0.0) - tol):
			return False
		if np.any(D > D_reference + margin_D + tol):
			return False
		if T > 1:
			ar1_ref_D = D_ref * regime.mu_D * (demand_shape[1:] - regime.rho_D * demand_shape[:-1])
			if np.any(np.abs(D[1:] - regime.rho_D * D[:-1] - ar1_ref_D) > kappa_ar1 * D_ref * regime.sigma_D + tol):
				return False

		for gen_idx in self.wind_generator_indices:
			gen = self.physical_generators[gen_idx]
			capacity = float(gen["pmax"])
			P = np.asarray(generator_wind_profiles[gen["physical_name"]], dtype=float)
			P_reference = capacity * regime.mu_W * wind_shape
			stationary_std_W = regime.sigma_W / np.sqrt(1.0 - regime.rho_W ** 2)
			margin_W = kappa * capacity * stationary_std_W

			if np.any(P < np.maximum(P_reference - margin_W, 0.0) - tol):
				return False
			if np.any(P > np.minimum(P_reference + margin_W, capacity) + tol):
				return False
			if T > 1:
				ar1_ref_W = capacity * regime.mu_W * (wind_shape[1:] - regime.rho_W * wind_shape[:-1])
				if np.any(np.abs(P[1:] - regime.rho_W * P[:-1] - ar1_ref_W) > kappa_ar1 * capacity * regime.sigma_W + tol):
					return False

		return True

	def _build_costs_and_ramps(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		cost_data: Dict[str, float] = {}
		ramp_data: Dict[str, float] = {}

		for block in self.blocks:
			cost_data[f"{block['block_name']}_cost"] = float(block["cost"])
		for generator in self.physical_generators:
			gen_name = generator["physical_name"]
			ramp_data[f"{gen_name}_ramp_up"] = float(generator["ramp_up"])
			ramp_data[f"{gen_name}_ramp_down"] = float(generator["ramp_down"])

		return pd.DataFrame([cost_data]), pd.DataFrame([ramp_data])

	def _build_scenario_row(
		self,
		scenario_id: int,
		regime_name: str,
		demand_mean: float,
		demand_profile: List[float],
		generator_wind_profiles: Dict[str, List[float]],
		regime_parameters: Optional[RegimeParameters] = None,
		scenario_source: str = "regime",
		ambiguity_set: Optional[str] = None,
	) -> Dict[str, Any]:
		horizon = self.base_case["time_steps"]

		scenario_row: Dict[str, Any] = {
			"scenario_id": scenario_id,
			"regime": regime_name,
			"scenario_source": scenario_source,
			"ambiguity_set": ambiguity_set,
			"demand": demand_mean,
			"time_steps": horizon,
			"demand_profile": demand_profile,
		}

		if regime_parameters is not None:
			scenario_row.update(
				{
					"mu_D": float(regime_parameters.mu_D),
					"rho_D": float(regime_parameters.rho_D),
					"sigma_D": float(regime_parameters.sigma_D),
					"mu_W": float(regime_parameters.mu_W),
					"rho_W": float(regime_parameters.rho_W),
					"sigma_W": float(regime_parameters.sigma_W),
					"peak_W": float(regime_parameters.peak_W),
				}
			)

		for block in self.blocks:
			block_name = block["block_name"]
			physical_name = block["physical_name"]
			block_pmax = float(block["pmax"])
			cost = float(block["cost"])

			scenario_row[f"{block_name}_bid_profile"] = [cost] * horizon
			scenario_row[f"{block_name}_cap"] = block_pmax

			if bool(block["is_wind"]):
				physical = next(
					gen for gen in self.physical_generators
					if gen["physical_name"] == physical_name
				)
				physical_pmax = max(float(physical["pmax"]), 1e-12)
				scenario_row[f"{block_name}_profile"] = [
					float(value) * block_pmax / physical_pmax
					for value in generator_wind_profiles[physical_name]
				]

		return scenario_row

	def get_players_config(self) -> List[Dict[str, Any]]:
		return self.players_config

	@staticmethod
	def _profile_to_float_list(profile_raw: Any, expected_horizon: int) -> List[float]:
		"""Convert profile payload to a numeric list and validate horizon length."""
		if isinstance(profile_raw, str):
			profile_raw = ast.literal_eval(profile_raw)

		if not isinstance(profile_raw, (list, tuple, np.ndarray)):
			raise ValueError("Each profile must be a list-like sequence")

		profile = [float(value) for value in profile_raw]
		if len(profile) != expected_horizon:
			raise ValueError(
				f"Demand profile length mismatch: expected {expected_horizon}, got {len(profile)}"
			)

		return profile

	def create_scenario_set_from_regimes(
		self,
		regime_config_path: Optional[str] = None,
		regime_set: Optional[str] = None,
		seed: Optional[int] = None,
		enforce_support_set: bool = False,
		enforce_dispatch_feasibility: Optional[bool] = None,
		n_minus_one_margin: Optional[float] = None,
		enforce_n_minus_one: Optional[bool] = None,
	) -> Dict[str, Any]:
		"""Generate stochastic scenarios from a YAML-defined list of regimes.

		Set enforce_support_set=True (DRO poa scenarios) to reject draws that
		violate the support set point-wise level bounds or AR(1) innovation bounds.
		enforce_dispatch_feasibility, n_minus_one_margin, and enforce_n_minus_one
		override the values from the YAML config when provided.  Pass
		enforce_n_minus_one=False to require only base feasibility (total capacity
		>= demand) without the stricter N-1 contingency check.
		"""
		config = self.load_regime_configuration(regime_config_path=regime_config_path, regime_set=regime_set)

		effective_seed = seed if seed is not None else config["seed"]
		effective_enforce_feasibility = (
			enforce_dispatch_feasibility
			if enforce_dispatch_feasibility is not None
			else config["enforce_dispatch_feasibility"]
		)
		effective_n_minus_one_margin = (
			float(n_minus_one_margin)
			if n_minus_one_margin is not None
			else float(config["n_minus_one_margin"])
		)
		effective_enforce_n_minus_one = (
			enforce_n_minus_one if enforce_n_minus_one is not None else True
		)
		rng = np.random.default_rng(effective_seed)

		scenarios_table: List[Dict[str, Any]] = []
		horizon = self.base_case["time_steps"]
		accepted_scenario_count = 0

		for regime in config["regimes"]:
			for draw_idx in range(regime.n_scenarios):
				demand_profile: List[float] = []
				generator_wind_profiles: Dict[str, List[float]] = {}

				for _ in range(config["max_draw_attempts"]):
					demand_profile, generator_wind_profiles, demand_mean = self._simulate_single_scenario(
						regime=regime,
						rng=rng,
					)

					if not effective_enforce_feasibility:
						break
					if self._is_dispatch_feasible(
						demand_profile,
						generator_wind_profiles,
						contingency_margin=effective_n_minus_one_margin,
						enforce_n_minus_one=effective_enforce_n_minus_one,
					) and (
						not enforce_support_set
						or self._is_within_support_set(
							regime,
							demand_profile,
							generator_wind_profiles,
						)
					):
						break
				else:
					raise ValueError(
						f"Failed to draw a feasible scenario for regime '{regime.name}' "
						f"after {config['max_draw_attempts']} attempts. "
						"Try reducing mu_D/sigma_D, increasing mu_W, or relaxing N-1 strictness."
					)

				accepted_scenario_count += 1
				scenario_id = len(scenarios_table) + 1
				scenarios_table.append(
					self._build_scenario_row(
						scenario_id=scenario_id,
						regime_name=regime.name,
						demand_mean=demand_mean,
						demand_profile=demand_profile,
						generator_wind_profiles=generator_wind_profiles,
						regime_parameters=regime,
						scenario_source="regime",
						ambiguity_set=None,
					)
				)

		scenarios_df = pd.DataFrame(scenarios_table)
		costs_df, ramps_df = self._build_costs_and_ramps()

		regime_counts = scenarios_df["regime"].value_counts().sort_index().to_dict()
		regime_summary = ", ".join(f"{name}: {count}" for name, count in regime_counts.items())
		description_text = (
			"=== Scenario Set Summary (Regime-based Stochastic Process) ===\n"
			f"Reference Case: {self.base_case_reference}\n"
			f"Regime Set: {config['name']}\n"
			f"Accepted Stochastic Scenarios: {accepted_scenario_count}\n"
			f"Generated Scenario Rows: {len(scenarios_df)}\n"
			f"Time Steps: {horizon}\n"
			f"Physical Generators: {len(self.physical_generators)}\n"
			f"Bidding Blocks: {len(self.blocks)}\n"
			"N-1 Feasibility: generator-level feasibility filter only\n"
			f"Regime Breakdown: {regime_summary}"
		)

		return {
			"description_text": description_text,
			"scenarios_df": scenarios_df,
			"costs_df": costs_df,
			"ramps_df": ramps_df,
			"regime_config": config,
		}

	def create_scenario_set_from_ambiguity_set(
		self,
		ambiguity_config_path: Optional[str] = None,
		ambiguity_set: Optional[str] = None,
		n_scenarios: int = 100,
		seed: Optional[int] = None,
		enforce_dispatch_feasibility: Optional[bool] = None,
		max_draw_attempts: Optional[int] = None,
		n_minus_one_margin: Optional[float] = None,
		enforce_support_set: bool = False,
	) -> Dict[str, Any]:
		"""Generate stochastic scenarios from uniformly drawn ambiguity-set parameters.

		Set enforce_support_set=True (DRO pipeline only) to reject samples that violate
		the support set constraints: point-wise level bounds and AR(1) innovation bounds.
		"""
		if n_scenarios <= 0:
			raise ValueError(f"n_scenarios must be positive, got {n_scenarios}")

		config = self.load_ambiguity_set_configuration(
			ambiguity_config_path=ambiguity_config_path,
			ambiguity_set=ambiguity_set,
		)

		effective_seed = seed if seed is not None else config["seed"]
		effective_enforce_feasibility = (
			enforce_dispatch_feasibility
			if enforce_dispatch_feasibility is not None
			else config["enforce_dispatch_feasibility"]
		)
		effective_max_draw_attempts = (
			int(max_draw_attempts)
			if max_draw_attempts is not None
			else int(config["max_draw_attempts"])
		)
		effective_n_minus_one_margin = (
			float(n_minus_one_margin)
			if n_minus_one_margin is not None
			else float(config["n_minus_one_margin"])
		)

		if effective_max_draw_attempts <= 0:
			raise ValueError("max_draw_attempts must be a positive integer")
		if effective_n_minus_one_margin < 0:
			raise ValueError("n_minus_one_margin must be non-negative")

		rng = np.random.default_rng(effective_seed)
		scenarios_table: List[Dict[str, Any]] = []
		horizon = self.base_case["time_steps"]
		accepted_scenario_count = 0

		for scenario_index in range(1, n_scenarios + 1):
			demand_profile: List[float] = []
			generator_wind_profiles: Dict[str, List[float]] = {}
			regime_parameters: Optional[RegimeParameters] = None
			demand_mean = 0.0

			for _ in range(effective_max_draw_attempts):
				regime_parameters = self._draw_regime_parameters_from_ambiguity_set(
					ambiguity_config=config,
					rng=rng,
					scenario_index=scenario_index,
				)
				demand_profile, generator_wind_profiles, demand_mean = self._simulate_single_scenario(
					regime=regime_parameters,
					rng=rng,
				)

				if not effective_enforce_feasibility:
					break
				if self._is_dispatch_feasible(
					demand_profile,
					generator_wind_profiles,
					contingency_margin=effective_n_minus_one_margin,
				) and (
					not enforce_support_set
					or self._is_within_support_set(
						regime_parameters,
						demand_profile,
						generator_wind_profiles,
					)
				):
					break
			else:
				raise ValueError(
					f"Failed to draw a feasible scenario from ambiguity set '{config['name']}' "
					f"after {effective_max_draw_attempts} attempts. "
					"The ambiguity-set bounds may be too wide or the feasibility filter may be too strict."
				)

			if regime_parameters is None:
				raise ValueError(f"Failed to draw parameters from ambiguity set '{config['name']}'")

			accepted_scenario_count += 1
			scenario_id = len(scenarios_table) + 1
			scenarios_table.append(
				self._build_scenario_row(
					scenario_id=scenario_id,
					regime_name=config["name"],
					demand_mean=demand_mean,
					demand_profile=demand_profile,
					generator_wind_profiles=generator_wind_profiles,
					regime_parameters=regime_parameters,
					scenario_source="ambiguity_set",
					ambiguity_set=config["name"],
				)
			)

		scenarios_df = pd.DataFrame(scenarios_table)
		costs_df, ramps_df = self._build_costs_and_ramps()

		description_text = (
			"=== Scenario Set Summary (Ambiguity-set Stochastic Process) ===\n"
			f"Reference Case: {self.base_case_reference}\n"
			f"Ambiguity Set: {config['name']}\n"
			f"Accepted Stochastic Scenarios: {accepted_scenario_count}\n"
			f"Generated Scenario Rows: {len(scenarios_df)}\n"
			f"Time Steps: {horizon}\n"
			f"Physical Generators: {len(self.physical_generators)}\n"
			f"Bidding Blocks: {len(self.blocks)}\n"
			"N-1 Feasibility: generator-level feasibility filter only\n"
			"Parameter draws: uniform over ambiguity-set bounds"
		)

		return {
			"description_text": description_text,
			"scenarios_df": scenarios_df,
			"costs_df": costs_df,
			"ramps_df": ramps_df,
			"ambiguity_config": config,
		}

if __name__ == "__main__":
	manager = ScenarioManager(base_case_reference="base_test_case")
	regime_yaml = ScenarioManager.DEFAULT_REGIME_CONFIG_PATH
	ambiguity_yaml = ScenarioManager.AMBIGUITY_CONFIG_PATH
	regime_scenario_set = manager.create_scenario_set_from_regimes(
		regime_config_path=str(regime_yaml),
		regime_set="policy_training",
		seed=42,
	)
	print(regime_scenario_set["description_text"])

	ambiguity_scenario_set = manager.create_scenario_set_from_ambiguity_set(
		ambiguity_config_path=str(ambiguity_yaml),
		ambiguity_set="base_test_case",
		n_scenarios=500,
		seed=42,
	)
	print(ambiguity_scenario_set["description_text"])