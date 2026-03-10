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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

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
    return {
        "available_features": features_cfg.get("available_features", []),
        "default_features":   features_cfg.get("default_features", []),
    }


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

    def __init__(self, features: List[str]):
        unknown = set(features) - set(AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {unknown}. "
                f"Available: {AVAILABLE_FEATURES}"
            )
        self.features = list(features)

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
            scenarios_df, costs_df, player_generators, generator_names
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
    fb = FeatureBuilder(DEFAULT_FEATURES)

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