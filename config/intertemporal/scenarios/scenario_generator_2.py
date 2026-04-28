"""Regime-based stochastic scenario generation for intertemporal studies."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from config.intertemporal.utils.cases_utils import load_setup_data


@dataclass(frozen=True)
class RegimeParameters:
	"""Container for one stochastic regime."""

	name: str
	n_scenarios: int
	mu_D: float
	A_D: float
	rho_D: float
	sigma_D: float
	mu_W: float
	A_W: float
	peak_W: float
	rho_W: float
	sigma_W: float
	Corr: float


class ScenarioManagerV2:
	"""Generate stochastic demand and wind scenarios from YAML-defined regimes."""

	DEFAULT_REGIME_CONFIG_PATH = Path(__file__).resolve().with_name("regime_definitions.yaml")

	def __init__(self, base_case_reference: str = "test_case1") -> None:
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
			time_steps,
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
			"time_steps": int(time_steps),
		}

		self.players_config = self.base_case["players"]

		if self.base_case["time_steps"] <= 0:
			raise ValueError(f"time_steps must be positive, got {self.base_case['time_steps']}")

		self.wind_generator_indices = [
			i
			for i, gen in enumerate(self.base_case["generators"])
			if self._is_wind_generator(self._generator_name(gen))
		]
		self.conventional_generator_indices = [
			i for i in range(self.base_case["num_generators"]) if i not in self.wind_generator_indices
		]

		self.total_conventional_capacity = float(
			sum(self.base_case["pmax_list"][i] for i in self.conventional_generator_indices)
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

		regime_name = str(regime_raw.get("name", f"regime_{index + 1}"))

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
			self._get_value(regime_raw, ["mu_D", "mu_D(r)"], required=True, context=regime_name),
			"mu_D",
			regime_name,
		)
		A_D = self._to_float(
			self._get_value(regime_raw, ["A_D", "A_D(r)"], required=True, context=regime_name),
			"A_D",
			regime_name,
		)
		rho_D = self._to_float(
			self._get_value(regime_raw, ["rho_D", "rho_D(r)"], required=True, context=regime_name),
			"rho_D",
			regime_name,
		)
		sigma_D = self._to_float(
			self._get_value(regime_raw, ["sigma_D", "sigma_D(r)"], required=True, context=regime_name),
			"sigma_D",
			regime_name,
		)
		mu_W = self._to_float(
			self._get_value(regime_raw, ["mu_W", "mu_W(r)"], required=True, context=regime_name),
			"mu_W",
			regime_name,
		)
		A_W = self._to_float(
			self._get_value(regime_raw, ["A_W", "A_W(r)", "wind_amplitude"], default=0.0, context=regime_name),
			"A_W",
			regime_name,
		)
		peak_W = self._to_float(
			self._get_value(
				regime_raw,
				["peak_W", "peak_W(r)", "wind_peak_hour", "peak_hour_W"],
				default=12.0,
				context=regime_name,
			),
			"peak_W",
			regime_name,
		)
		rho_W = self._to_float(
			self._get_value(regime_raw, ["rho_W", "rho_W(r)"], required=True, context=regime_name),
			"rho_W",
			regime_name,
		)
		sigma_W = self._to_float(
			self._get_value(regime_raw, ["sigma_W", "sigma_W(r)"], required=True, context=regime_name),
			"sigma_W",
			regime_name,
		)
		corr = self._to_float(
			self._get_value(regime_raw, ["Corr", "Corr(r)", "corr", "correlation"], required=True, context=regime_name),
			"Corr",
			regime_name,
		)

		if n_scenarios <= 0:
			raise ValueError(f"Regime '{regime_name}': n_scenarios must be positive")
		if mu_D <= 0:
			raise ValueError(f"Regime '{regime_name}': mu_D must be positive")
		if A_D < 0:
			raise ValueError(f"Regime '{regime_name}': A_D must be non-negative")
		if not -0.999 <= rho_D <= 0.999:
			raise ValueError(f"Regime '{regime_name}': rho_D must be in [-0.999, 0.999]")
		if sigma_D < 0:
			raise ValueError(f"Regime '{regime_name}': sigma_D must be non-negative")
		if not 0 <= mu_W <= 1:
			raise ValueError(f"Regime '{regime_name}': mu_W must be in [0, 1]")
		if A_W < 0:
			raise ValueError(f"Regime '{regime_name}': A_W must be non-negative")
		if not 0 <= peak_W <= 24:
			raise ValueError(f"Regime '{regime_name}': peak_W must be in [0, 24]")
		if not -0.999 <= rho_W <= 0.999:
			raise ValueError(f"Regime '{regime_name}': rho_W must be in [-0.999, 0.999]")
		if sigma_W < 0:
			raise ValueError(f"Regime '{regime_name}': sigma_W must be non-negative")
		if not -1 <= corr <= 1:
			raise ValueError(f"Regime '{regime_name}': Corr must be in [-1, 1]")

		return RegimeParameters(
			name=regime_name,
			n_scenarios=n_scenarios,
			mu_D=mu_D,
			A_D=A_D,
			rho_D=rho_D,
			sigma_D=sigma_D,
			mu_W=mu_W,
			A_W=A_W,
			peak_W=peak_W,
			rho_W=rho_W,
			sigma_W=sigma_W,
			Corr=corr,
		)

	def load_regime_configuration(
		self,
		regime_config_path: Optional[str] = None,
		regime_set: Optional[str] = None,
	) -> Dict[str, Any]:
		"""Load and validate regime configuration from YAML."""
		config_path = Path(regime_config_path) if regime_config_path is not None else self.DEFAULT_REGIME_CONFIG_PATH
		if not config_path.is_absolute() and not config_path.exists():
			config_path = Path(__file__).resolve().parent / config_path
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

		wind_idio_share = float(
			selected_config.get("wind_idiosyncratic_share", raw_config.get("wind_idiosyncratic_share", 0.25))
		)
		if not 0 <= wind_idio_share <= 1:
			raise ValueError("wind_idiosyncratic_share must be in [0, 1]")

		max_draw_attempts = int(selected_config.get("max_draw_attempts", raw_config.get("max_draw_attempts", 500)))
		if max_draw_attempts <= 0:
			raise ValueError("max_draw_attempts must be a positive integer")

		enforce_feasibility = bool(
			selected_config.get(
				"enforce_dispatch_feasibility",
				raw_config.get("enforce_dispatch_feasibility", True),
			)
		)

		seed = selected_config.get("seed", raw_config.get("seed"))
		if seed is not None:
			seed = int(seed)

		parsed_regimes = [self._parse_regime(regime_raw, idx) for idx, regime_raw in enumerate(regimes_raw)]

		return {
			"name": selected_name,
			"seed": seed,
			"wind_idiosyncratic_share": wind_idio_share,
			"max_draw_attempts": max_draw_attempts,
			"enforce_dispatch_feasibility": enforce_feasibility,
			"regimes": parsed_regimes,
			"raw": selected_config,
		}

	@staticmethod
	def _build_demand_peak_shape(horizon: int) -> np.ndarray:
		"""Create a normalized daily shape with morning and afternoon/evening peaks."""
		if horizon <= 0:
			raise ValueError(f"horizon must be positive, got {horizon}")

		if horizon == 1:
			return np.array([0.0], dtype=float)

		hours = np.linspace(0.0, 24.0, horizon)
		morning_peak = np.exp(-0.5 * ((hours - 7.5) / 2.5) ** 2)
		evening_peak = 1.35 * np.exp(-0.5 * ((hours - 15.0) / 3.0) ** 2)
		midday_dip = -0.35 * np.exp(-0.5 * ((hours - 11.0) / 2.0) ** 2)

		raw_shape = morning_peak + evening_peak + midday_dip
		centered = raw_shape - np.mean(raw_shape)
		scale = np.max(np.abs(centered))

		if scale <= 1e-12:
			return np.zeros(horizon, dtype=float)

		return centered / scale

	@staticmethod
	def _build_wind_peak_shape(horizon: int, peak_hour: float, width: float = 4.0) -> np.ndarray:
		"""Create a normalized circular daily shape with a configurable peak hour."""
		if horizon <= 0:
			raise ValueError(f"horizon must be positive, got {horizon}")
		if not 0 <= peak_hour <= 24:
			raise ValueError(f"peak_hour must be in [0, 24], got {peak_hour}")
		if width <= 0:
			raise ValueError(f"width must be positive, got {width}")

		if horizon == 1:
			return np.array([0.0], dtype=float)

		hours = np.linspace(0.0, 24.0, horizon, endpoint=False)
		peak_hour = peak_hour % 24.0
		circular_distance = np.abs(((hours - peak_hour + 12.0) % 24.0) - 12.0)
		raw_shape = np.exp(-0.5 * (circular_distance / width) ** 2)
		centered = raw_shape - np.mean(raw_shape)
		scale = np.max(np.abs(centered))

		if scale <= 1e-12:
			return np.zeros(horizon, dtype=float)

		return centered / scale

	@staticmethod
	def _draw_correlated_normals(corr: float, horizon: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
		covariance = np.array([[1.0, corr], [corr, 1.0]], dtype=float)
		samples = rng.multivariate_normal(mean=[0.0, 0.0], cov=covariance, size=horizon)
		return samples[:, 0], samples[:, 1]

	def _simulate_single_scenario(
		self,
		regime: RegimeParameters,
		rng: np.random.Generator,
		wind_idiosyncratic_share: float,
	) -> Tuple[List[float], Dict[str, List[float]], float]:
		horizon = self.base_case["time_steps"]
		demand_shape = self._build_demand_peak_shape(horizon)
		demand_trend = regime.mu_D * (1.0 + regime.A_D * demand_shape)
		wind_shape = self._build_wind_peak_shape(horizon, regime.peak_W)
		wind_mean = np.clip(regime.mu_W * (1.0 + regime.A_W * wind_shape), 0.0, 1.0)

		z_d, z_w_common = self._draw_correlated_normals(regime.Corr, horizon, rng)

		demand_residual = np.zeros(horizon, dtype=float)
		demand_stationary_scale = regime.sigma_D / np.sqrt(max(1.0 - regime.rho_D**2, 1e-8))
		demand_residual[0] = demand_stationary_scale * z_d[0]

		for t in range(1, horizon):
			demand_residual[t] = regime.rho_D * demand_residual[t - 1] + regime.sigma_D * z_d[t]

		demand_profile = np.maximum(demand_trend + demand_residual, 1e-3)

		wind_profiles: Dict[str, List[float]] = {}
		common_weight = np.sqrt(max(1.0 - wind_idiosyncratic_share**2, 0.0))

		for gen_idx in self.wind_generator_indices:
			generator = self.base_case["generators"][gen_idx]
			gen_name = self._generator_name(generator)
			nominal_capacity = self.base_case["pmax_list"][gen_idx]

			z_w_idio = rng.standard_normal(horizon)
			z_w = common_weight * z_w_common + wind_idiosyncratic_share * z_w_idio

			wind_factor = np.zeros(horizon, dtype=float)
			wind_stationary_scale = regime.sigma_W / np.sqrt(max(1.0 - regime.rho_W**2, 1e-8))
			wind_factor[0] = wind_mean[0] + wind_stationary_scale * z_w[0]

			for t in range(1, horizon):
				wind_factor[t] = (
					wind_mean[t]
					+ regime.rho_W * (wind_factor[t - 1] - wind_mean[t - 1])
					+ regime.sigma_W * z_w[t]
				)

			wind_factor = np.clip(wind_factor, 0.0, 1.0)
			wind_profiles[gen_name] = [float(value * nominal_capacity) for value in wind_factor]

		return [float(v) for v in demand_profile], wind_profiles, regime.mu_D

	def _is_dispatch_feasible(
		self,
		demand_profile: List[float],
		wind_profiles: Dict[str, List[float]],
	) -> bool:
		"""Check base feasibility and N-1 contingency across all time steps.

		A scenario is feasible only if:
		1) Total available capacity can serve demand at each time step.
		2) Demand can still be served when any single generator is unavailable (N-1).
		"""
		horizon = self.base_case["time_steps"]

		for t in range(horizon):
			capacities_t: List[float] = []
			for gen_idx, generator in enumerate(self.base_case["generators"]):
				gen_name = self._generator_name(generator)
				if self._is_wind_generator(gen_name):
					cap_t = float(wind_profiles[gen_name][t])
				else:
					cap_t = float(self.base_case["pmax_list"][gen_idx])
				capacities_t.append(cap_t)

			total_capacity_t = float(sum(capacities_t))
			demand_t = float(demand_profile[t])

			# Base feasibility (no outage).
			if demand_t > total_capacity_t + 1e-6:
				return False

			# N-1 contingency: demand must still be met without any single generator.
			for generator_capacity_t in capacities_t:
				remaining_capacity_t = total_capacity_t - generator_capacity_t
				if demand_t > remaining_capacity_t + 1e-6:
					return False

		return True

	def _build_costs_and_ramps(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		cost_data: Dict[str, float] = {}
		ramp_data: Dict[str, float] = {}

		for i, generator in enumerate(self.base_case["generators"]):
			gen_name = self._generator_name(generator)
			cost_data[f"{gen_name}_cost"] = float(self.base_case["cost_vector"][i])
			ramp_data[f"{gen_name}_ramp_up"] = float(self.base_case["r_rates_up_list"][i])
			ramp_data[f"{gen_name}_ramp_down"] = float(self.base_case["r_rates_down_list"][i])

		return pd.DataFrame([cost_data]), pd.DataFrame([ramp_data])

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

	def plot_demand_profiles_by_regime(
		self,
		scenarios_df: pd.DataFrame,
		title: str = "Demand Profiles by Regime",
		max_profiles_per_regime: Optional[int] = None,
		show_regime_mean: bool = True,
		alpha: float = 0.25,
		line_width: float = 1.0,
		mean_line_width: float = 2.5,
		figsize: Tuple[float, float] = (11.0, 6.0),
		random_state: Optional[int] = None,
		save_path: Optional[str] = None,
		show: bool = True,
	):
		"""Plot demand profiles and color them by regime.

		Args:
			scenarios_df: DataFrame from create_scenario_set_from_regimes.
			title: Plot title.
			max_profiles_per_regime: Optional cap on plotted trajectories per regime.
			show_regime_mean: If True, draw one bold mean line per regime.
			alpha: Transparency for individual profiles.
			line_width: Width for individual profiles.
			mean_line_width: Width for regime mean lines.
			figsize: Matplotlib figure size.
			random_state: Optional seed when down-sampling profiles.
			save_path: Optional figure output path.
			show: If True, call plt.show().

		Returns:
			(fig, ax, regime_colors)
		"""
		if "regime" not in scenarios_df.columns:
			raise ValueError("scenarios_df must include a 'regime' column")
		if "demand_profile" not in scenarios_df.columns:
			raise ValueError("scenarios_df must include a 'demand_profile' column")

		import matplotlib.pyplot as plt
		from matplotlib.lines import Line2D

		horizon = self.base_case["time_steps"]
		regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())
		if not regimes:
			raise ValueError("No regimes found in scenarios_df['regime']")

		rng = np.random.default_rng(random_state)
		cmap = plt.get_cmap("tab10")
		regime_colors = {regime: cmap(i % cmap.N) for i, regime in enumerate(regimes)}

		fig, ax = plt.subplots(figsize=figsize)
		legend_handles: List[Line2D] = []

		for regime in regimes:
			regime_df = scenarios_df[scenarios_df["regime"].astype(str) == regime]
			if regime_df.empty:
				continue

			indices = regime_df.index.to_list()
			if max_profiles_per_regime is not None and max_profiles_per_regime > 0 and len(indices) > max_profiles_per_regime:
				indices = rng.choice(indices, size=max_profiles_per_regime, replace=False).tolist()

			profiles: List[np.ndarray] = []
			for idx in indices:
				profile = self._profile_to_float_list(scenarios_df.at[idx, "demand_profile"], expected_horizon=horizon)
				profile_np = np.array(profile, dtype=float)
				profiles.append(profile_np)
				ax.plot(
					range(horizon),
					profile_np,
					color=regime_colors[regime],
					alpha=alpha,
					linewidth=line_width,
				)

			if show_regime_mean and profiles:
				mean_profile = np.mean(np.vstack(profiles), axis=0)
				ax.plot(
					range(horizon),
					mean_profile,
					color=regime_colors[regime],
					linewidth=mean_line_width,
				)

			legend_handles.append(
				Line2D(
					[0],
					[0],
					color=regime_colors[regime],
					lw=mean_line_width,
					label=f"{regime} (n={len(regime_df)})",
				)
			)

		ax.set_title(title)
		ax.set_xlabel("Time")
		ax.set_ylabel("Demand")
		ax.grid(True, alpha=0.25)
		ax.legend(handles=legend_handles, title="Regime", loc="best")

		if save_path:
			save_target = Path(save_path)
			save_target.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(save_target, dpi=200, bbox_inches="tight")

		if show:
			plt.show()

		return fig, ax, regime_colors

	def plot_wind_profiles_by_regime(
		self,
		scenarios_df: pd.DataFrame,
		title: str = "Wind Profiles by Regime",
		wind_generators: Optional[List[str]] = None,
		max_profiles_per_regime: Optional[int] = None,
		show_regime_mean: bool = True,
		alpha: float = 0.22,
		line_width: float = 0.9,
		mean_line_width: float = 2.2,
		figsize_per_generator: Tuple[float, float] = (11.0, 3.2),
		random_state: Optional[int] = None,
		save_path: Optional[str] = None,
		show: bool = True,
	):
		"""Plot wind profiles for each wind generator and color scenarios by regime.

		Args:
			scenarios_df: DataFrame from create_scenario_set_from_regimes.
			title: Figure title.
			wind_generators: Optional subset of wind generator names (e.g. ["W3", "W4"]).
			max_profiles_per_regime: Optional cap on trajectories per regime and generator.
			show_regime_mean: If True, draw one bold mean line per regime on each subplot.
			alpha: Transparency for individual profiles.
			line_width: Width for individual profiles.
			mean_line_width: Width for regime mean lines.
			figsize_per_generator: (width, height) used per subplot row.
			random_state: Optional seed when down-sampling profiles.
			save_path: Optional output path for the figure.
			show: If True, call plt.show().

		Returns:
			(fig, axes, regime_colors)
		"""
		if "regime" not in scenarios_df.columns:
			raise ValueError("scenarios_df must include a 'regime' column")

		import matplotlib.pyplot as plt
		from matplotlib.lines import Line2D

		all_wind_generators = [
			self._generator_name(self.base_case["generators"][i])
			for i in self.wind_generator_indices
		]

		if wind_generators is None:
			wind_generators = all_wind_generators

		if not wind_generators:
			raise ValueError("No wind generators available for plotting")

		for wind_name in wind_generators:
			if wind_name not in all_wind_generators:
				raise ValueError(f"Unknown wind generator '{wind_name}'. Available: {all_wind_generators}")
			profile_col = f"{wind_name}_profile"
			if profile_col not in scenarios_df.columns:
				raise ValueError(f"Missing column '{profile_col}' in scenarios_df")

		horizon = self.base_case["time_steps"]
		regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())
		if not regimes:
			raise ValueError("No regimes found in scenarios_df['regime']")

		rng = np.random.default_rng(random_state)
		cmap = plt.get_cmap("tab10")
		regime_colors = {regime: cmap(i % cmap.N) for i, regime in enumerate(regimes)}

		fig_width = float(figsize_per_generator[0])
		fig_height = float(figsize_per_generator[1]) * len(wind_generators)
		fig, axes = plt.subplots(
			nrows=len(wind_generators),
			ncols=1,
			figsize=(fig_width, fig_height),
			sharex=True,
		)
		if len(wind_generators) == 1:
			axes = [axes]

		for ax, wind_name in zip(axes, wind_generators):
			profile_col = f"{wind_name}_profile"

			for regime in regimes:
				regime_df = scenarios_df[scenarios_df["regime"].astype(str) == regime]
				if regime_df.empty:
					continue

				indices = regime_df.index.to_list()
				if max_profiles_per_regime is not None and max_profiles_per_regime > 0 and len(indices) > max_profiles_per_regime:
					indices = rng.choice(indices, size=max_profiles_per_regime, replace=False).tolist()

				profiles: List[np.ndarray] = []
				for idx in indices:
					profile = self._profile_to_float_list(scenarios_df.at[idx, profile_col], expected_horizon=horizon)
					profile_np = np.array(profile, dtype=float)
					profiles.append(profile_np)
					ax.plot(
						range(horizon),
						profile_np,
						color=regime_colors[regime],
						alpha=alpha,
						linewidth=line_width,
					)

				if show_regime_mean and profiles:
					mean_profile = np.mean(np.vstack(profiles), axis=0)
					ax.plot(
						range(horizon),
						mean_profile,
						color=regime_colors[regime],
						linewidth=mean_line_width,
					)

			ax.set_title(f"{wind_name} Wind Profile")
			ax.set_ylabel("MW")
			ax.grid(True, alpha=0.25)

		axes[-1].set_xlabel("Time")
		fig.suptitle(title)

		counts_by_regime = scenarios_df["regime"].astype(str).value_counts().to_dict()
		legend_handles = [
			Line2D(
				[0],
				[0],
				color=regime_colors[regime],
				lw=mean_line_width,
				label=f"{regime} (n={counts_by_regime.get(regime, 0)})",
			)
			for regime in regimes
		]
		fig.legend(handles=legend_handles, title="Regime", loc="upper right")
		fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])

		if save_path:
			save_target = Path(save_path)
			save_target.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(save_target, dpi=200, bbox_inches="tight")

		if show:
			plt.show()

		return fig, axes, regime_colors

	def create_scenario_set_from_regimes(
		self,
		regime_config_path: Optional[str] = None,
		regime_set: Optional[str] = None,
		seed: Optional[int] = None,
	) -> Dict[str, Any]:
		"""Generate stochastic scenarios from a YAML-defined list of regimes."""
		config = self.load_regime_configuration(regime_config_path=regime_config_path, regime_set=regime_set)

		effective_seed = seed if seed is not None else config["seed"]
		rng = np.random.default_rng(effective_seed)

		scenarios_table: List[Dict[str, Any]] = []
		horizon = self.base_case["time_steps"]

		for regime in config["regimes"]:
			for draw_idx in range(regime.n_scenarios):
				demand_profile: List[float] = []
				wind_profiles: Dict[str, List[float]] = {}

				for _ in range(config["max_draw_attempts"]):
					demand_profile, wind_profiles, demand_mean = self._simulate_single_scenario(
						regime=regime,
						rng=rng,
						wind_idiosyncratic_share=config["wind_idiosyncratic_share"],
					)

					if not config["enforce_dispatch_feasibility"]:
						break
					if self._is_dispatch_feasible(demand_profile, wind_profiles):
						break
				else:
					raise ValueError(
						f"Failed to draw a feasible scenario for regime '{regime.name}' "
						f"after {config['max_draw_attempts']} attempts. "
						"Try reducing mu_D/A_D/sigma_D, increasing mu_W, or relaxing N-1 strictness."
					)

				scenario_id = len(scenarios_table) + 1
				scenario_row: Dict[str, Any] = {
					"scenario_id": scenario_id,
					"regime": regime.name,
					"demand": demand_mean,
					"time_steps": horizon,
					"demand_profile": demand_profile,
				}

				for i, generator in enumerate(self.base_case["generators"]):
					gen_name = self._generator_name(generator)
					pmax = float(self.base_case["pmax_list"][i])
					cost = float(self.base_case["cost_vector"][i])

					scenario_row[f"{gen_name}_bid_profile"] = [cost] * horizon

					if self._is_wind_generator(gen_name):
						profile = wind_profiles[gen_name]
						scenario_row[f"{gen_name}_profile"] = profile
						scenario_row[f"{gen_name}_cap"] = pmax
					else:
						scenario_row[f"{gen_name}_cap"] = pmax

				scenarios_table.append(scenario_row)

		scenarios_df = pd.DataFrame(scenarios_table)
		costs_df, ramps_df = self._build_costs_and_ramps()

		regime_counts = scenarios_df["regime"].value_counts().sort_index().to_dict()
		regime_summary = ", ".join(f"{name}: {count}" for name, count in regime_counts.items())
		description_text = (
			"=== Scenario Set Summary (Regime-based Stochastic Process) ===\n"
			f"Reference Case: {self.base_case_reference}\n"
			f"Regime Set: {config['name']}\n"
			f"Total Scenarios: {len(scenarios_df)}\n"
			f"Time Steps: {horizon}\n"
			f"Wind Generators: {len(self.wind_generator_indices)}\n"
			f"Regime Breakdown: {regime_summary}"
		)

		return {
			"description_text": description_text,
			"scenarios_df": scenarios_df,
			"costs_df": costs_df,
			"ramps_df": ramps_df,
			"regime_config": config,
		}

if __name__ == "__main__":
	manager = ScenarioManagerV2(base_case_reference="test_case1")
	default_yaml = Path(__file__).with_name("regime_definitions.yaml")
	
	scenario_set = manager.create_scenario_set_from_regimes(str(default_yaml), regime_set="policy_training")
	print(scenario_set["description_text"])
	print("\nScenarios preview:")
	print(scenario_set["scenarios_df"].head())

	manager.plot_demand_profiles_by_regime(
		scenario_set["scenarios_df"],
		title="Demand Profiles by Regime",
		max_profiles_per_regime=20,
		show_regime_mean=True,
		alpha=0.22,
		save_path="demand_profiles_by_regime.png",
	)

	manager.plot_wind_profiles_by_regime(
		scenario_set["scenarios_df"],
		title="Wind Profiles by Regime",
		max_profiles_per_regime=20,
		show_regime_mean=True,
		alpha=0.22,
		save_path="wind_profiles_by_regime.png",
	)


	scenario_set = manager.create_scenario_set_from_regimes(str(default_yaml), regime_set="PoA_analysis")
	print(scenario_set["description_text"])
	print("\nScenarios preview:")
	print(scenario_set["scenarios_df"].head())

	stop = True

	manager.plot_demand_profiles_by_regime(
		scenario_set["scenarios_df"],
		title="Demand Profiles by Regime",
		max_profiles_per_regime=20,
		show_regime_mean=True,
		alpha=0.22,
		save_path="demand_profiles_by_regime.png",
	)

	manager.plot_wind_profiles_by_regime(
		scenario_set["scenarios_df"],
		title="Wind Profiles by Regime",
		max_profiles_per_regime=20,
		show_regime_mean=True,
		alpha=0.22,
		save_path="wind_profiles_by_regime.png",
	)

	stop = True
