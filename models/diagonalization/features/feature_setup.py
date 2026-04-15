"""
Feature Setup Module for Policy-Based Bidding

Provides a configurable FeatureBuilder that converts scenario data and
player-specific information into feature vectors used by bidding policies.

Feature taxonomy
----------------
**Market-level (public) features** - observable by every player:
  bias              : constant 1 (intercept / bias term)
  demand            : total system demand  [MW]
  demand_sq         : demand²              [MW²]
  wind_forecast     : wind forecast  [MW]
  total_capacity    : sum of all generator capacities  [MW]

**Player-private features** - specific to the strategic player:
  marginal_cost     : (capacity-weighted) avg marginal cost of the player  [$/MWh]
  player_capacity   : player's total available capacity  [MW]

**Historical supply-curve features** - pre-computed from historical market data:
  supply_intercept  : intercept of price ~ demand regression  [$/MWh]
  supply_slope      : slope of price ~ demand regression      [$/MW]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from config.intertemporal.scenarios.scenario_generator import ScenarioManager
from config.default_loader import load_test_case_config

# YAML-based feature configuration loader

def load_feature_config(config_path: str = None) -> Dict[str, List[str]]:
    """Load feature configuration from YAML file.

    Returns a dict with keys ``available_features`` and ``default_features``.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "features.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    features_cfg = data.get("features", {})
    hist_cfg     = data.get("historical_scenarios", {})
    return {
        "available_features":   features_cfg.get("available_features", []),
        "default_features":     features_cfg.get("default_features", []),
        "historical_scenarios": hist_cfg,
    }

# Historical supply-curve estimation
# def compute_historical_supply_curve(
#     reference_case: str = "test_case",
#     config_path: str = None,
#     plot = False,
# ) -> Dict[str, float]:
#     """
#     Estimate the supply-curve from *independent* historical scenarios.

#     Generates a separate set of demand / capacity scenarios (controlled by
#     ``features.yaml → historical_scenarios``), runs competitive (cost-based)
#     economic dispatch on them, and fits:

#         clearing_price ≈ supply_intercept + supply_slope x demand

#     The resulting coefficients are **exogenous prior knowledge** — they
#     do not depend on any BR-iteration data.

#     Parameters
#     ----------
#     scenario_manager : ScenarioManager
#         Used to generate the historical scenario set.
#     costs_df : pd.DataFrame
#         Static generator cost data.
#     config_path : str, optional
#         Path to ``features.yaml``.  ``None`` → default location.

#     Returns
#     -------
#     dict with keys ``supply_intercept`` and ``supply_slope``.
#     """

#     cfg = load_feature_config(config_path)
#     hist_cfg = cfg.get("historical_scenarios", {})

#     num_demand   = int(hist_cfg.get("num_demand", None ))
#     demand_min   = float(hist_cfg.get("demand_min_factor", None))
#     demand_max   = float(hist_cfg.get("demand_max_factor", None))
#     num_capacity = int(hist_cfg.get("num_capacity", None ))
#     cap_min      = float(hist_cfg.get("capacity_min_factor", None))
#     cap_max      = float(hist_cfg.get("capacity_max_factor", None))

#     scenario_manager = ScenarioManager(reference_case)  

#     # Generate independent historical scenarios
#     demand_scenarios = scenario_manager.generate_demand_scenarios(
#         "linear", num_scenarios=num_demand,
#         min_factor=demand_min, max_factor=demand_max,
#     )
#     capacity_scenarios = scenario_manager.generate_capacity_scenarios(
#         "linear", num_scenarios=num_capacity,
#         min_factor=cap_min, max_factor=cap_max,
#     )
#     hist_set = scenario_manager.create_scenario_set(
#         demand_scenarios=demand_scenarios,
#         capacity_scenarios=capacity_scenarios,
#     )

#     hist_df = hist_set["scenarios_df"]
#     hist_cost_df = hist_set["costs_df"]
#     hist_ramps_df = hist_set["ramps_df"]

#     # Cost-based (competitive) bids
#     gen_names = [c.replace("_cap", "") for c in hist_df.columns if c.endswith("_cap")]
#     for g in gen_names:
#         hist_df[f"{g}_bid"] = hist_cost_df[f"{g}_cost"].iloc[0]

#     ed = EconomicDispatchModel(hist_df, hist_cost_df, hist_ramps_df)
#     ed.solve()
#     prices = np.array(ed.get_clearing_prices())
#     demands = hist_df["demand"].values.astype(float)

#     # Fit price = a + b * demand
#     if len(demands) >= 2 and np.std(demands) > 1e-12:
#         coeffs = np.polyfit(demands, prices, deg=1)  # [slope, intercept]
#         slope, intercept = float(coeffs[0]), float(coeffs[1])
#     else:
#         intercept = float(prices.mean())
#         slope = 0.0

#     if plot:
#         plt.scatter(demands, prices, label="Historical Scenarios")
#         plt.plot(demands, intercept + slope * demands, color="red", label=f"Fitted Supply Curve\n ($y={intercept:.2f} + {slope:.2f}x$)")
#         plt.xlabel("Demand [MW]")
#         plt.ylabel("Clearing Price [$/MWh]")
#         plt.title("Historical Supply Curve Estimation")
#         plt.legend()
#         plt.grid()
#         plt.show()

#     return {"supply_intercept": intercept, "supply_slope": slope}

def get_tm1_data(reference_case: str, feature_config_path: str = None) -> Dict[str, float]:
    
    feat_cfg = load_feature_config(feature_config_path)
    hist_cfg = feat_cfg.get("historical_scenarios", {})

    test_cfg = load_test_case_config(reference_case)

    demand_cfg = test_cfg["demand"]
    capacity_cfg = test_cfg["capacity"]

    demand_persistence   = float(hist_cfg.get("demand_persistence", None))
    wind_persistence     = float(hist_cfg.get("wind_persistence", None))
    noise_scale          = float(hist_cfg.get("noise_scale", None))
    
    scenario_manager = ScenarioManager(reference_case)

    # Demand
    demand_scenarios = scenario_manager.generate_demand_scenarios(
        demand_cfg["type"],
        num_scenarios=demand_cfg["num_scenarios"],
        min_factor=demand_cfg["min_factor"],
        max_factor=demand_cfg["max_factor"],
    )

    # Capacity
    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        capacity_cfg["type"],
        num_scenarios=capacity_cfg["num_scenarios"],
        min_factor=capacity_cfg["min_factor"],
        max_factor=capacity_cfg["max_factor"],
    )

    hist_set = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )

    scenario_df = hist_set["scenarios_df"]
    
    historic_df = scenario_manager.generate_historic_df(scenario_df, demand_persistence, wind_persistence, noise_scale)

    return historic_df

# Dataclasses that capture a single observation for one player
@dataclass
class MarketState:
    """Public / market-level information for a single scenario."""
    demand: float
    wind_forecast: float
    total_capacity: float

@dataclass
class PrivateInfo:
    """Player-specific (private) information for a single scenario."""
    player_cost: List[float]
    player_capacity: List[float]

@dataclass
class HistoricData:
    """Historical supply-curve coefficients."""
    supply_intercept: float
    supply_slope: float
    supply_curve: float

@dataclass
class HistoricTM1Data:
    """Historical data for t-1 features."""
    demand_tm1: float
    wind_tm1: float
    total_capacity_tm1: float
    
@dataclass
class Observation:
    """Full observation = market state + private info."""
    market: MarketState
    private: PrivateInfo
    historic: Optional[HistoricData] = None
    historic_tm1: Optional[HistoricTM1Data] = None

# Catalogue of all available features (loaded from features.yaml)
_FEATURE_CFG = load_feature_config()
AVAILABLE_FEATURES: List[str] = _FEATURE_CFG["available_features"]

# FeatureBuilder
class FeatureBuilder:
    """
    Converts an ``Observation`` into a numeric feature vector.

    Parameters
    ----------
    features : list[str]
        Ordered list of feature names to include (see ``AVAILABLE_FEATURES``).
    """

    def __init__(self, reference_case: str, features: List[str]):
        unknown = set(features) - set(AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {unknown}. "
                f"Available: {AVAILABLE_FEATURES}"
            )

        self.features = list(features)
        self.reference_case = reference_case

        # Pre-compute supply-curve coefficients if needed
        supply_features = {"supply_intercept", "supply_slope", "supply_curve"}
        if supply_features & set(features):
            # self.supply_coeffs = compute_historical_supply_curve(reference_case)
            self.supply_coeffs = {"supply_intercept": 0.0, "supply_slope": 0.0}

        # Pre-compute historical t-1 data if needed
        tm1_features = {"wind_tm1", "demand_tm1", "total_capacity_tm1"}
        if tm1_features & set(features):
            self.historic_tm1_df = get_tm1_data(reference_case)
        else: 
            self.historic_tm1_df = None

        # Map each feature name → its handler method.
        # To add a new feature: write a _feat_<name> method and add the
        # name to AVAILABLE_FEATURES – everything else is automatic.
        self._handlers: Dict[str, callable] = {
            "bias":               self._feat_bias,
            "demand":             self._feat_demand,
            "demand_sq":          self._feat_demand_sq,
            "wind_forecast":      self._feat_wind_forecast,
            "total_capacity":     self._feat_total_capacity,
            "player_cost":        self._feat_player_cost,
            "player_capacity":    self._feat_player_capacity,
            "scarcity_ratio":     self._feat_scarcity_ratio,
            "residual_demand":    self._feat_residual_demand,
            "supply_intercept":   self._feat_supply_intercept,
            "supply_slope":       self._feat_supply_slope,
            "supply_curve":       self._feat_supply_curve,
            "demand_tm1":         self._feat_demand_tm1,
            "wind_tm1":           self._feat_wind_tm1,
            "total_capacity_tm1": self._feat_total_capacity_tm1,
        }

    # Individual feature handlers 
    # Each handler receives an Observation and returns a scalar or 1-D array.
    # To extend: add a new _feat_<name> method and register it in __init__.

    @staticmethod
    def _feat_bias(obs: Observation):
        """Constant intercept term."""
        return 1.0

    @staticmethod
    def _feat_demand(obs: Observation):
        """Total system demand [MW]."""
        return obs.market.demand

    @staticmethod
    def _feat_demand_sq(obs: Observation):
        """Demand squared [MW²]."""
        return obs.market.demand ** 2

    @staticmethod
    def _feat_wind_forecast(obs: Observation):
        """Wind forecast [MW]."""
        return obs.market.wind_forecast

    @staticmethod
    def _feat_total_capacity(obs: Observation):
        """Sum of all generator capacities [MW]."""
        return obs.market.total_capacity

    @staticmethod
    def _feat_player_cost(obs: Observation):
        """Per-generator marginal costs for the strategic player."""
        return np.atleast_1d(obs.private.player_cost)

    @staticmethod
    def _feat_player_capacity(obs: Observation):
        """Per-generator available capacities for the strategic player."""
        return np.atleast_1d(obs.private.player_capacity)

    @staticmethod
    def _feat_scarcity_ratio(obs: Observation):
        """Ratio of demand to total capacity (unitless)."""
        if obs.market.total_capacity > 0:
            return obs.market.demand / obs.market.total_capacity
        else:
            return 0.0  # Avoid division by zero; interpret as no scarcity
    
    @staticmethod
    def _feat_residual_demand(obs: Observation):
        """Residual demand after accounting for player's capacity [MW]."""
        residual = obs.market.demand - obs.market.wind_forecast
        return max(residual, 0.0)  # Residual demand can't be negative

    def _feat_supply_intercept(self, obs: Observation):
        """Intercept of historical price~demand regression [$/MWh]."""
        return self.supply_coeffs.get("supply_intercept", 0.0)

    def _feat_supply_slope(self, obs: Observation):
        """Slope of historical price~demand regression [$/MW]."""
        return self.supply_coeffs.get("supply_slope", 0.0)

    def _feat_supply_curve(self, obs: Observation):
        """Full historical supply curve value at current demand [$/MWh]."""
        intercept = self._feat_supply_intercept(obs)
        slope = self._feat_supply_slope(obs)
        return intercept + slope * obs.market.demand

    @staticmethod
    def _feat_demand_tm1(obs: Observation):
        return obs.historic_tm1.demand_tm1 if obs.historic_tm1 is not None else 0.0

    @staticmethod
    def _feat_wind_tm1(obs: Observation):
        return obs.historic_tm1.wind_tm1 if obs.historic_tm1 is not None else 0.0

    @staticmethod
    def _feat_total_capacity_tm1(obs: Observation):
        return obs.historic_tm1.total_capacity_tm1 if obs.historic_tm1 is not None else 0.0

    @staticmethod
    def make_observation(
        demand: float,
        wind_forecast: float,
        total_capacity: float,
        player_cost: List[float],
        player_capacity: List[float],
        demand_tm1: float = 0.0,
        wind_tm1: float = 0.0,
        total_capacity_tm1: float = 0.0,
        supply_intercept: float = 0.0,
        supply_slope: float = 0.0,
        supply_curve: float = 0.0,
    ) -> Observation:
        """Create a full observation from raw values.

        This is the preferred entry point for intertemporal policies because it
        keeps current and lagged values in one reusable structure.
        """
        return Observation(
            market=MarketState(
                demand=demand,
                wind_forecast=wind_forecast,
                total_capacity=total_capacity,
            ),
            private=PrivateInfo(
                player_cost=player_cost,
                player_capacity=player_capacity,
            ),
            historic=HistoricData(
                supply_intercept=supply_intercept,
                supply_slope=supply_slope,
                supply_curve=supply_curve,
            ),
            historic_tm1=HistoricTM1Data(
                demand_tm1=demand_tm1,
                wind_tm1=wind_tm1,
                total_capacity_tm1=total_capacity_tm1,
            ),
        )

    def build_intertemporal_features(
        self,
        demand: float,
        wind_forecast: float,
        total_capacity: float,
        player_cost: List[float],
        player_capacity: List[float],
        demand_tm1: float = 0.0,
        wind_tm1: float = 0.0,
        total_capacity_tm1: float = 0.0,
        supply_intercept: float = 0.0,
        supply_slope: float = 0.0,
        supply_curve: float = 0.0,
    ) -> np.ndarray:
        """Build a feature vector directly from raw current and lagged values."""
        obs = self.make_observation(
            demand=demand,
            wind_forecast=wind_forecast,
            total_capacity=total_capacity,
            player_cost=player_cost,
            player_capacity=player_capacity,
            demand_tm1=demand_tm1,
            wind_tm1=wind_tm1,
            total_capacity_tm1=total_capacity_tm1,
            supply_intercept=supply_intercept,
            supply_slope=supply_slope,
            supply_curve=supply_curve,
        )
        parts: List[np.ndarray] = []
        for feature_name in self.features:
            val = self._handlers[feature_name](obs)
            parts.append(np.atleast_1d(val))
        return np.concatenate(parts).astype(np.float64)

    @staticmethod
    def compute_intertemporal_context(
        demand_scenarios: List[List[float]],
        pmax_scenarios: List[List[List[float]]],
        generator_names: List[str],
        s: int,
        t: int,
    ) -> Dict[str, float]:
        """Compute current and t-1 market context used by intertemporal features."""
        num_generators = len(generator_names)

        demand_t = float(demand_scenarios[s][t])
        total_capacity_t = float(sum(float(pmax_scenarios[s][t][g]) for g in range(num_generators)))
        wind_t = float(
            sum(
                float(pmax_scenarios[s][t][g])
                for g, name in enumerate(generator_names)
                if name.startswith("W")
            )
        )

        if t > 0:
            demand_tm1 = float(demand_scenarios[s][t - 1])
            total_capacity_tm1 = float(
                sum(float(pmax_scenarios[s][t - 1][g]) for g in range(num_generators))
            )
            wind_tm1 = float(
                sum(
                    float(pmax_scenarios[s][t - 1][g])
                    for g, name in enumerate(generator_names)
                    if name.startswith("W")
                )
            )
        else:
            demand_tm1 = 0.0
            total_capacity_tm1 = 0.0
            wind_tm1 = 0.0

        return {
            "demand_t": demand_t,
            "total_capacity_t": total_capacity_t,
            "wind_t": wind_t,
            "demand_tm1": demand_tm1,
            "total_capacity_tm1": total_capacity_tm1,
            "wind_tm1": wind_tm1,
        }

    def build_intertemporal_feature_tensor(
        self,
        demand_scenarios: List[List[float]],
        pmax_scenarios: List[List[List[float]]],
        cost_vector: List[float],
        generator_names: List[str],
    ) -> Dict[Tuple[int, int, int], List[float]]:
        """Build full intertemporal feature tensor keyed by (scenario, time, generator)."""
        num_scenarios = len(demand_scenarios)
        num_time_steps = len(demand_scenarios[0]) if num_scenarios > 0 else 0
        num_generators = len(generator_names)

        full_feature_matrix: Dict[Tuple[int, int, int], List[float]] = {}
        expected_dim: Optional[int] = None

        for s in range(num_scenarios):
            for t in range(num_time_steps):
                ctx = self.compute_intertemporal_context(
                    demand_scenarios=demand_scenarios,
                    pmax_scenarios=pmax_scenarios,
                    generator_names=generator_names,
                    s=s,
                    t=t,
                )
                for i in range(num_generators):
                    phi = self.build_intertemporal_features(
                        demand=ctx["demand_t"],
                        wind_forecast=ctx["wind_t"],
                        total_capacity=ctx["total_capacity_t"],
                        player_cost=[float(cost_vector[i])],
                        player_capacity=[float(pmax_scenarios[s][t][i])],
                        demand_tm1=ctx["demand_tm1"],
                        wind_tm1=ctx["wind_tm1"],
                        total_capacity_tm1=ctx["total_capacity_tm1"],
                    )
                    phi_list = np.atleast_1d(phi).astype(np.float64).tolist()

                    if expected_dim is None:
                        expected_dim = len(phi_list)
                    elif len(phi_list) != expected_dim:
                        raise ValueError(
                            f"Inconsistent feature dimension at (s={s}, t={t}, i={i}). Expected {expected_dim}, got {len(phi_list)}."
                        )

                    full_feature_matrix[s, t, i] = phi_list

        return full_feature_matrix

    @staticmethod
    def split_feature_tensor_by_player(
        full_feature_matrix: Dict[Tuple[int, int, int], List[float]],
        players_config: List[Dict[str, Any]],
    ) -> Dict[int, Dict[Tuple[int, int, int], List[float]]]:
        """Split full feature tensor into one dictionary per player id."""
        feature_matrix_by_player: Dict[int, Dict[Tuple[int, int, int], List[float]]] = {}
        for player in players_config:
            pid = player["id"]
            controlled = set(player["controlled_generators"])
            feature_matrix_by_player[pid] = {
                key: value
                for key, value in full_feature_matrix.items()
                if key[2] in controlled
            }
        return feature_matrix_by_player

    def build_intertemporal_feature_matrix_by_player(
        self,
        demand_scenarios: List[List[float]],
        pmax_scenarios: List[List[List[float]]],
        cost_vector: List[float],
        generator_names: List[str],
        players_config: List[Dict[str, Any]],
    ) -> Dict[int, Dict[Tuple[int, int, int], List[float]]]:
        """Build per-player intertemporal feature dictionaries from this FeatureBuilder."""
        full_feature_matrix = self.build_intertemporal_feature_tensor(
            demand_scenarios=demand_scenarios,
            pmax_scenarios=pmax_scenarios,
            cost_vector=cost_vector,
            generator_names=generator_names,
        )
        return self.split_feature_tensor_by_player(full_feature_matrix, players_config)

    @property
    def num_features(self) -> int:
        """Number of *named* feature slots (before per-generator expansion)."""
        return len(self.features)

    def num_features_expanded(self, num_player_generators: int) -> int:
        """Actual dimensionality after per-generator features are expanded.

        Parameters
        ----------
        num_player_generators : int
            How many generators the strategic player controls.
        """
        PER_GEN_FEATURES = {"player_cost", "player_capacity"}
        dim = 0
        for f in self.features:
            dim += num_player_generators if f in PER_GEN_FEATURES else 1
        return dim

    def __repr__(self) -> str:
        return f"FeatureBuilder(features={self.features})"

# Convenience: default feature set (loaded from features.yaml)
DEFAULT_FEATURES: List[str] = _FEATURE_CFG["default_features"]

def build_intertemporal_feature_matrix_by_player(
    feature_builder: FeatureBuilder,
    demand_scenarios: List[List[float]],
    pmax_scenarios: List[List[List[float]]],
    cost_vector: List[float],
    generator_names: List[str],
    players_config: List[Dict[str, Any]],
) -> Dict[int, Dict[Tuple[int, int, int], List[float]]]:
    """Build per-player intertemporal feature dictionaries.

    Returns
    -------
    dict[int, dict[(s, t, i), feature_vector]]
        Player id -> sparse feature map for generators controlled by that player.
    """
    return feature_builder.build_intertemporal_feature_matrix_by_player(
        demand_scenarios=demand_scenarios,
        pmax_scenarios=pmax_scenarios,
        cost_vector=cost_vector,
        generator_names=generator_names,
        players_config=players_config,
    )

if __name__ == "__main__":
    from config.intertemporal.scenarios.scenario_generator import ScenarioManager

    # Configuration
    TEST_CASE  = "test_case1"

    test_config = load_test_case_config(TEST_CASE)

    demand_cfg = test_config["demand"]
    capacity_cfg = test_config["capacity"]

    # Scenario generation
    scenario_manager = ScenarioManager(TEST_CASE)
    players_config   = scenario_manager.get_players_config()

    # Demand
    demand_scenarios = scenario_manager.generate_demand_scenarios(
        demand_cfg["type"],
        num_scenarios=demand_cfg["num_scenarios"],
        min_factor=demand_cfg["min_factor"],
        max_factor=demand_cfg["max_factor"],
    )

    # Capacity
    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        capacity_cfg["type"],
        num_scenarios=capacity_cfg["num_scenarios"],
        min_factor=capacity_cfg["min_factor"],
        max_factor=capacity_cfg["max_factor"],
    )

    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )
    scenarios_df = scenarios["scenarios_df"]
    costs_df     = scenarios["costs_df"]

    # Generator names from the DataFrame columns
    generator_names = [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    print("Scenarios DataFrame:")
    print(scenarios_df)
    print("\nCosts DataFrame:")
    print(costs_df)

    # Build feature vectors per player
    fb = FeatureBuilder(TEST_CASE, DEFAULT_FEATURES)

    # compute_historical_supply_curve(reference_case=TEST_CASE, plot=True)

    demand_scenarios_intertemporal = [
        [float(d)]
        for d in scenarios_df["demand"].tolist()
    ]
    pmax_scenarios_intertemporal = [
        [[float(row[f"{g}_cap"]) for g in generator_names]]
        for _, row in scenarios_df.iterrows()
    ]
    cost_vector = [float(costs_df[f"{g}_cost"].iloc[0]) for g in generator_names]

    feature_matrix_by_player = fb.build_intertemporal_feature_matrix_by_player(
        demand_scenarios=demand_scenarios_intertemporal,
        pmax_scenarios=pmax_scenarios_intertemporal,
        cost_vector=cost_vector,
        generator_names=generator_names,
        players_config=players_config,
    )

    print(f"Built feature matrices for players: {sorted(feature_matrix_by_player.keys())}")
    for pid, feature_map in feature_matrix_by_player.items():
        print(f"Player {pid}: {len(feature_map)} feature rows")

    stop = True