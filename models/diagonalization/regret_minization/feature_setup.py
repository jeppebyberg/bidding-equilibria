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
from typing import Any, Dict, List

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from models.diagonalization.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from config.base_case.scenarios.scenario_generator import ScenarioManager

# YAML-based feature configuration loader

def load_feature_config(config_path: str = None) -> Dict[str, List[str]]:
    """Load feature configuration from YAML file.

    Returns a dict with keys ``available_features`` and ``default_features``.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "utilities" / "features.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    features_cfg = data.get("features", {})
    hist_cfg     = data.get("historical_scenarios", {})
    return {
        "available_features":   features_cfg.get("available_features", []),
        "default_features":     features_cfg.get("default_features", []),
        "historical_scenarios": hist_cfg,
    }

# ──────────────────────────────────────────────────────────────────────
# Historical supply-curve estimation
# ──────────────────────────────────────────────────────────────────────

def compute_historical_supply_curve(
    reference_case: str = "test_case",
    config_path: str = None,
) -> Dict[str, float]:
    """
    Estimate the supply-curve from *independent* historical scenarios.

    Generates a separate set of demand / capacity scenarios (controlled by
    ``features.yaml → historical_scenarios``), runs competitive (cost-based)
    economic dispatch on them, and fits:

        clearing_price ≈ supply_intercept + supply_slope × demand

    The resulting coefficients are **exogenous prior knowledge** — they
    do not depend on any BR-iteration data.

    Parameters
    ----------
    scenario_manager : ScenarioManager
        Used to generate the historical scenario set.
    costs_df : pd.DataFrame
        Static generator cost data.
    config_path : str, optional
        Path to ``features.yaml``.  ``None`` → default location.

    Returns
    -------
    dict with keys ``supply_intercept`` and ``supply_slope``.
    """

    cfg = load_feature_config(config_path)
    hist_cfg = cfg.get("historical_scenarios", {})

    num_demand   = int(hist_cfg.get("num_demand", 7))
    demand_min   = float(hist_cfg.get("demand_min_factor", 0.4))
    demand_max   = float(hist_cfg.get("demand_max_factor", 1.0))
    num_capacity = int(hist_cfg.get("num_capacity", 3))
    cap_min      = float(hist_cfg.get("capacity_min_factor", 0.4))
    cap_max      = float(hist_cfg.get("capacity_max_factor", 1.0))

    scenario_manager = ScenarioManager(reference_case)  

    # Generate independent historical scenarios
    demand_scenarios = scenario_manager.generate_demand_scenarios(
        "linear", num_scenarios=num_demand,
        min_factor=demand_min, max_factor=demand_max,
    )
    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        "linear", num_scenarios=num_capacity,
        min_factor=cap_min, max_factor=cap_max,
    )
    hist_set = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )

    hist_df = hist_set["scenarios_df"]
    hist_cost_df = hist_set["costs_df"]

    # Cost-based (competitive) bids
    gen_names = [c.replace("_cap", "") for c in hist_df.columns if c.endswith("_cap")]
    for g in gen_names:
        hist_df[f"{g}_bid"] = hist_cost_df[f"{g}_cost"].iloc[0]

    ed = EconomicDispatchModel(hist_df, hist_cost_df)
    ed.solve()
    prices = np.array(ed.get_clearing_prices())
    demands = hist_df["demand"].values.astype(float)

    # Fit price = a + b * demand
    if len(demands) >= 2 and np.std(demands) > 1e-12:
        coeffs = np.polyfit(demands, prices, deg=1)  # [slope, intercept]
        slope, intercept = float(coeffs[0]), float(coeffs[1])
    else:
        intercept = float(prices.mean())
        slope = 0.0

    return {"supply_intercept": intercept, "supply_slope": slope}

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
class Observation:
    """Full observation = market state + private info."""
    market: MarketState
    private: PrivateInfo

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

    def __init__(self, features: List[str], supply_coeffs: Dict[str, float] = None):
        unknown = set(features) - set(AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {unknown}. "
                f"Available: {AVAILABLE_FEATURES}"
            )
        self.features = list(features)
        self.supply_coeffs = supply_coeffs or {}

        # Map each feature name → its handler method.
        # To add a new feature: write a _feat_<name> method and add the
        # name to AVAILABLE_FEATURES – everything else is automatic.
        self._handlers: Dict[str, callable] = {
            "bias":             self._feat_bias,
            "demand":           self._feat_demand,
            "demand_sq":        self._feat_demand_sq,
            "wind_forecast":    self._feat_wind_forecast,
            "total_capacity":   self._feat_total_capacity,
            "player_cost":      self._feat_player_cost,
            "player_capacity":  self._feat_player_capacity,
            "scarcity_ratio":   self._feat_scarcity_ratio,
            "residual_demand":  self._feat_residual_demand,
            "supply_intercept": self._feat_supply_intercept,
            "supply_slope":     self._feat_supply_slope,
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

    # Core feature builder
    def build(self, obs: Observation) -> np.ndarray:
        """Return a 1-D feature vector for a single observation."""
        parts: List[np.ndarray] = []
        for f in self.features:
            val = self._handlers[f](obs)
            parts.append(np.atleast_1d(val))
        return np.concatenate(parts).astype(np.float64)

    # Builder for all scenarios at once
    def build_scenario_matrix(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        player_generators: List[int],
        generator_names: List[str],
    ) -> np.ndarray:
        """
        Build the (num_scenarios x num_features) matrix directly from DataFrames.

        Parameters
        ----------
        scenarios_df : pd.DataFrame
            Scenario table (columns: demand, {gen}_cap, {gen}_bid, ...).
        costs_df : pd.DataFrame
            Static costs table (columns: {gen}_cost, ...).
        player_generators : list[int]
            Indices of generators controlled by the strategic player.
        generator_names : list[str]
            Ordered list of all generator names (e.g. ['G1', 'G2', ...]).

        Returns
        -------
        np.ndarray of shape (num_scenarios, num_features)
        """
        observations = self._extract_observations(
            scenarios_df, costs_df, player_generators, generator_names,
        )
        return np.vstack([self.build(obs) for obs in observations])

    # Helpers
    @staticmethod
    def _extract_observations(
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        player_generators: List[int],
        generator_names: List[str],
    ) -> List[Observation]:
        """Convert DataFrame rows into a list of ``Observation`` objects."""

        # Static cost info (same across scenarios)
        costs = np.array([costs_df[f"{g}_cost"].iloc[0] for g in generator_names])
        player_gen_names = [generator_names[i] for i in player_generators]

        observations: List[Observation] = []
        for _, row in scenarios_df.iterrows():
            # --- capacities this scenario ---
            caps = np.array([row[f"{g}_cap"] for g in generator_names])
            total_cap = float(caps.sum())

            # Wind capacity = sum of capacities for generators whose name starts with 'W'
            wind_cap = float(sum(
                row[f"{g}_cap"] for g in generator_names if g.startswith("W")
            ))

            demand = float(row["demand"])

            # --- player-private ---
            player_caps = np.array([row[f"{g}_cap"] for g in player_gen_names])
            player_costs = np.array([costs_df[f"{g}_cost"].iloc[0] for g in player_gen_names])

            market = MarketState(
                demand=demand,
                wind_forecast=wind_cap,
                total_capacity=total_cap,
            )
            private = PrivateInfo(
                player_cost=player_costs,
                player_capacity=player_caps,
            )
            observations.append(Observation(market=market, private=private))

        return observations

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


# ──────────────────────────────────────────────────────────────────────
# Convenience: default feature set (loaded from features.yaml)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_FEATURES: List[str] = _FEATURE_CFG["default_features"]

def create_feature_builder(
    features: List[str] = None,
) -> FeatureBuilder:
    """
    Factory that creates a :class:`FeatureBuilder`, automatically computing
    historical supply-curve coefficients when ``supply_intercept`` or
    ``supply_slope`` are among the requested features.

    All configuration (reference case, scenario grid) is read from
    ``features.yaml`` — no external dependencies need to be passed in.

    Parameters
    ----------
    features : list[str], optional
        Feature names.  ``None`` → ``DEFAULT_FEATURES``.

    Returns
    -------
    FeatureBuilder
    """
    if features is None:
        features = DEFAULT_FEATURES

    supply_features = {"supply_intercept", "supply_slope"}
    needs_supply = bool(supply_features & set(features))

    supply_coeffs = None
    if needs_supply:
        supply_coeffs = compute_historical_supply_curve()

    return FeatureBuilder(features, supply_coeffs=supply_coeffs)

# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test when run directly
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config.base_case.scenarios.scenario_generator import ScenarioManager

    # ── 1. Load scenarios from ScenarioManager ──────────────────────
    manager = ScenarioManager("test_case_multiple_owners")

    demand_scenarios = manager.generate_demand_scenarios(
        "linear", num_scenarios=5, min_factor=0.8, max_factor=1.0
    )
    capacity_scenarios = manager.generate_capacity_scenarios(
        "linear", num_scenarios=3
    )
    scenario_set = manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )

    scenarios_df = scenario_set["scenarios_df"]
    costs_df     = scenario_set["costs_df"]
    players_cfg  = manager.players_config

    # Generator names from the DataFrame columns
    generator_names = [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    print("Scenarios DataFrame:")
    print(scenarios_df)
    print("\nCosts DataFrame:")
    print(costs_df)

    # ── 2. Build feature vectors per player ─────────────────────────
    fb = create_feature_builder(DEFAULT_FEATURES)

    for player in players_cfg:
        pid  = player["id"]
        gens = player["controlled_generators"]

        # Extract observations for this player across all scenarios
        observations = FeatureBuilder._extract_observations(
            scenarios_df, costs_df, gens, generator_names
        )

        # Build the feature matrix (num_scenarios x num_features)
        feature_matrix = fb.build_scenario_matrix(
            scenarios_df, costs_df, gens, generator_names
        )

        print(f"\n── Player {pid} (generators {gens}) ──")
        print(f"Feature names : {fb.features}")
        print(f"Feature matrix ({feature_matrix.shape}):")
        print(feature_matrix)

        # Show first observation details
        obs0 = observations[0]
        print(f"  First obs → market : demand={obs0.market.demand}, "
              f"wind={obs0.market.wind_forecast}, total_cap={obs0.market.total_capacity}")
        print(f"              private: cost={obs0.private.player_cost}, "
              f"cap={obs0.private.player_capacity}")

    stop = True