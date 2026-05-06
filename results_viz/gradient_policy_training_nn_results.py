"""
Visualize final NN policy bids on mean regime trajectories.

Default inputs:
    results/gradient_policy_training_nn_results.json
    results/feature_normalizer_stats_gradient.json

The script regenerates the scenario dataframe with ScenarioManagerV2, averages
the profiles within each regime, evaluates the saved final_policy_params on
those mean trajectories, solves economic dispatch, and saves one four-panel
figure per (regime, generator).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.intertemporal.scenarios.scenario_generator_2 import ScenarioManagerV2
from models.diagonalization.features.feature_setup import FeatureBuilder
from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel
from models.gradient_based.sensitivities.policy_sensitivity import (
    PolicyParameters,
    compute_policy_bids,
)


def as_profile(value: Any, expected_len: Optional[int] = None, column_name: str = "profile") -> List[float]:
    """Convert a stored profile to a numeric list."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Column '{column_name}' must contain a list-like profile")

    profile = [float(v) for v in value]
    if expected_len is not None and len(profile) != expected_len:
        raise ValueError(
            f"Profile length mismatch for '{column_name}': expected {expected_len}, got {len(profile)}"
        )
    return profile


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


class GradientNNPolicyMeanRegimeVisualizer:
    """Create mean-regime trajectory plots for saved gradient NN policy results."""

    def __init__(
        self,
        results: Dict[str, Any],
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        reference_case: str,
        output_dir: Path,
        normalizer_stats_path: Optional[Path] = None,
        beta_smooth: Optional[float] = None,
    ) -> None:
        self.results = results
        self.scenarios_df = scenarios_df.copy(deep=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.reference_case = reference_case
        self.output_dir = output_dir
        self.normalizer_stats_path = normalizer_stats_path
        self.beta_smooth = float(beta_smooth if beta_smooth is not None else results.get("beta_smooth", 0.001))

        self.generator_names = list(results.get("generator_names", []))
        if not self.generator_names:
            self.generator_names = [col.replace("_cap", "") for col in self.scenarios_df.columns if col.endswith("_cap")]
        if not self.generator_names:
            raise ValueError("Could not infer generator names")

        self.features = list(results.get("features", []))
        if not self.features:
            raise ValueError("Results file does not include the feature order used by the policy")

        self.num_time_steps = int(results.get("num_time_steps", self.scenarios_df["time_steps"].iloc[0]))
        self.final_policy_params = results.get("final_policy_params", {})
        if not self.final_policy_params:
            raise ValueError("Results file does not include final_policy_params")

        self.mean_scenarios_df = self._build_mean_scenarios_df()
        self._apply_final_policy_to_mean_scenarios()
        self.dispatches, self.clearing_prices = self._solve_mean_dispatch()

    @classmethod
    def from_json(
        cls,
        results_path: Path,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        reference_case: str,
        output_dir: Path,
        normalizer_stats_path: Optional[Path] = None,
        beta_smooth: Optional[float] = None,
    ) -> "GradientNNPolicyMeanRegimeVisualizer":
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        return cls(
            results=results,
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            players_config=players_config,
            reference_case=reference_case,
            output_dir=output_dir,
            normalizer_stats_path=normalizer_stats_path,
            beta_smooth=beta_smooth,
        )

    def _build_mean_scenarios_df(self) -> pd.DataFrame:
        if "regime" not in self.scenarios_df.columns:
            raise ValueError("scenarios_df must include a 'regime' column")
        if "demand_profile" not in self.scenarios_df.columns:
            raise ValueError("scenarios_df must include a 'demand_profile' column")

        rows: List[Dict[str, Any]] = []
        regimes = sorted(self.scenarios_df["regime"].dropna().astype(str).unique())
        for scenario_idx, regime in enumerate(regimes):
            regime_df = self.scenarios_df[self.scenarios_df["regime"].astype(str) == regime]
            row: Dict[str, Any] = {
                "scenario_id": scenario_idx + 1,
                "regime": regime,
                "time_steps": self.num_time_steps,
            }

            demand_profiles = np.vstack(
                [
                    as_profile(value, self.num_time_steps, "demand_profile")
                    for value in regime_df["demand_profile"]
                ]
            )
            demand_profile = np.mean(demand_profiles, axis=0).astype(float).tolist()
            row["demand_profile"] = demand_profile
            row["demand"] = float(np.mean(demand_profile))

            for gen_name in self.generator_names:
                cap_col = f"{gen_name}_cap"
                if cap_col not in regime_df.columns:
                    raise ValueError(f"Missing capacity column '{cap_col}'")
                row[cap_col] = float(regime_df[cap_col].astype(float).mean())

                for profile_col in (f"{gen_name}_profile", f"{gen_name}_cap_profile"):
                    if profile_col in regime_df.columns:
                        profiles = np.vstack(
                            [
                                as_profile(value, self.num_time_steps, profile_col)
                                for value in regime_df[profile_col]
                            ]
                        )
                        row[profile_col] = np.mean(profiles, axis=0).astype(float).tolist()

                if f"{gen_name}_profile" not in row and f"{gen_name}_cap_profile" in row:
                    row[f"{gen_name}_profile"] = list(row[f"{gen_name}_cap_profile"])

                cost = float(self.costs_df[f"{gen_name}_cost"].iloc[0])
                row[f"{gen_name}_bid"] = cost
                row[f"{gen_name}_bid_profile"] = [cost] * self.num_time_steps

            rows.append(row)

        return pd.DataFrame(rows)

    def _policy_parameters_for_player(self, player_id: int) -> PolicyParameters:
        raw = self.final_policy_params.get(str(player_id), self.final_policy_params.get(player_id))
        if raw is None:
            raise KeyError(f"No final policy parameters found for player {player_id}")
        return PolicyParameters(
            Gamma=np.asarray(raw["Gamma"], dtype=np.float64),
            gamma=np.asarray(raw["gamma"], dtype=np.float64),
            Theta=np.asarray(raw["Theta"], dtype=np.float64),
            rho=np.asarray(raw["rho"], dtype=np.float64),
        )

    def _build_feature_matrix_by_player(self) -> Dict[int, Dict[tuple[int, int, int], List[float]]]:
        feature_builder = FeatureBuilder(self.reference_case, self.features)
        if self.normalizer_stats_path and self.normalizer_stats_path.exists():
            feature_builder.load_feature_normalizer_stats(str(self.normalizer_stats_path))

        return feature_builder.build_intertemporal_feature_matrix_by_player_from_frames(
            scenarios_df=self.mean_scenarios_df,
            costs_df=self.costs_df,
            generator_names=self.generator_names,
            players_config=self.players_config,
            fit_normalizer=False,
        )

    def _apply_final_policy_to_mean_scenarios(self) -> None:
        feature_matrix_by_player = self._build_feature_matrix_by_player()

        for player in self.players_config:
            player_id = int(player["id"])
            controlled = [int(g) for g in player["controlled_generators"]]
            params = self._policy_parameters_for_player(player_id)
            player_features = feature_matrix_by_player[player_id]

            for scenario_idx in range(len(self.mean_scenarios_df)):
                feature_rows = []
                for t in range(self.num_time_steps):
                    per_generator_features = [
                        np.asarray(player_features[(scenario_idx, t, gen_idx)], dtype=np.float64)
                        for gen_idx in controlled
                    ]
                    first = per_generator_features[0]
                    if any(not np.allclose(first, phi, rtol=0.0, atol=1e-12) for phi in per_generator_features[1:]):
                        raise ValueError(
                            "Feature vectors differ across generators controlled by player "
                            f"{player_id} for mean scenario {scenario_idx}, time {t}"
                        )
                    feature_rows.append(first)

                features_s = np.vstack(feature_rows)
                bids = compute_policy_bids(features_s, params)
                lower = self.results.get("alpha_bounds", {}).get("min")
                upper = self.results.get("alpha_bounds", {}).get("max")
                if lower is not None or upper is not None:
                    bids = np.clip(
                        bids,
                        -np.inf if lower is None else float(lower),
                        np.inf if upper is None else float(upper),
                    )

                for local_idx, gen_idx in enumerate(controlled):
                    gen_name = self.generator_names[gen_idx]
                    bid_profile = bids[local_idx, :].astype(float).tolist()
                    self.mean_scenarios_df.at[scenario_idx, f"{gen_name}_bid_profile"] = bid_profile
                    self.mean_scenarios_df.at[scenario_idx, f"{gen_name}_bid"] = float(bid_profile[0])

    def _compute_p_init(self) -> List[List[float]]:
        initial_dispatch = []
        for _, row in self.mean_scenarios_df.iterrows():
            initial_dispatch.append([0.5 * float(row[f"{gen}_cap"]) for gen in self.generator_names])

        cost_bid_df = self.mean_scenarios_df.copy(deep=True)
        for gen_name in self.generator_names:
            cost = float(self.costs_df[f"{gen_name}_cost"].iloc[0])
            cost_bid_df[f"{gen_name}_bid"] = cost
            cost_bid_df[f"{gen_name}_bid_profile"] = [[cost] * self.num_time_steps for _ in range(len(cost_bid_df))]

        ed = EconomicDispatchQuadraticModel(
            cost_bid_df,
            self.costs_df,
            self.ramps_df,
            p_init=initial_dispatch,
            beta_coeff=self.beta_smooth,
        )
        ed.solve()
        dispatches = ed.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Could not compute initial dispatch for mean scenarios")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def _solve_mean_dispatch(self) -> tuple[List[List[List[float]]], List[List[float]]]:
        """Solve each mean regime independently to get scenario-local dual prices."""
        p_init_all = self._compute_p_init()
        dispatches_all: List[List[List[float]]] = []
        prices_all: List[List[float]] = []

        for scenario_idx in range(len(self.mean_scenarios_df)):
            scenario_df = self.mean_scenarios_df.iloc[[scenario_idx]].copy(deep=True).reset_index(drop=True)
            ed = EconomicDispatchQuadraticModel(
                scenario_df,
                self.costs_df,
                self.ramps_df,
                p_init=[p_init_all[scenario_idx]],
                beta_coeff=self.beta_smooth,
            )
            ed.solve()
            dispatches = ed.get_dispatches()
            prices = ed.get_clearing_prices()
            if dispatches is None or prices is None:
                regime = self.mean_scenarios_df.at[scenario_idx, "regime"]
                raise RuntimeError(f"Economic dispatch failed for mean regime '{regime}'")

            dispatches_all.append(dispatches[0])
            prices_all.append(prices[0])

        return dispatches_all, prices_all

    def _available_capacity_profile(self, scenario_idx: int, gen_name: str) -> np.ndarray:
        row = self.mean_scenarios_df.iloc[scenario_idx]
        for profile_col in (f"{gen_name}_cap_profile", f"{gen_name}_profile"):
            if profile_col in self.mean_scenarios_df.columns and isinstance(row.get(profile_col), list):
                return np.asarray(as_profile(row[profile_col], self.num_time_steps, profile_col), dtype=float)
        return np.full(self.num_time_steps, float(row[f"{gen_name}_cap"]), dtype=float)

    def plot_regime_generator(self, scenario_idx: int, gen_idx: int, show: bool = False) -> None:
        row = self.mean_scenarios_df.iloc[scenario_idx]
        regime = str(row["regime"])
        gen_name = self.generator_names[gen_idx]
        time_axis = np.arange(self.num_time_steps)

        demand = np.asarray(row["demand_profile"], dtype=float)
        capacity = self._available_capacity_profile(scenario_idx, gen_name)
        bid = np.asarray(row[f"{gen_name}_bid_profile"], dtype=float)
        price = np.asarray(self.clearing_prices[scenario_idx], dtype=float)
        dispatch = np.asarray(self.dispatches[scenario_idx], dtype=float)[:, gen_idx]
        marginal_offer = bid + 2.0 * self.beta_smooth * dispatch

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        fig.suptitle(f"Mean Regime Trajectory - {regime} - {gen_name}", fontsize=14)

        axes[0].plot(time_axis, demand, color="tab:blue", marker="o", linewidth=2.0)
        axes[0].set_ylabel("Demand")
        axes[0].set_title("Demand Trajectory")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_axis, capacity, color="tab:green", marker="o", linewidth=2.0)
        axes[1].set_ylabel("MW")
        axes[1].set_title(f"{gen_name} Available Capacity")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time_axis, bid, color="tab:orange", marker="o", linewidth=2.0, label=f"{gen_name} bid")

        axes[2].plot(
            time_axis,
            price,
            color="tab:red",
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="Clearing price",
        )
        
        if abs(self.beta_smooth) > 0.0:
            axes[2].plot(
                time_axis,
                marginal_offer,
                color="tab:brown",
                marker="^",
                linewidth=1.6,
                linestyle=":",
                label=f"{gen_name} bid + 2 beta dispatch",
            )

        axes[2].set_ylabel("Price / Bid")
        axes[2].set_title("Policy Bid and Market Clearing Price")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best", fontsize=9)

        axes[3].plot(time_axis, dispatch, color="tab:purple", marker="o", linewidth=2.0)
        axes[3].set_ylabel("MW")
        axes[3].set_xlabel("Time step")
        axes[3].set_title(f"{gen_name} Dispatch")
        axes[3].grid(True, alpha=0.3)

        for ax in axes:
            ax.set_xticks(time_axis)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        regime_dir = self.output_dir / slugify(regime)
        regime_dir.mkdir(parents=True, exist_ok=True)
        out = regime_dir / f"gradient_nn_mean_{gen_name}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"[saved] {out}")
        if show:
            plt.show()
        plt.close(fig)

    def run(self, show: bool = False) -> None:
        print("\n=== Gradient NN Mean-Regime Visualization ===")
        print(f"Regimes: {', '.join(self.mean_scenarios_df['regime'].astype(str).tolist())}")
        print(f"Generators: {', '.join(self.generator_names)}")
        print(f"Features: {', '.join(self.features)}")
        if self.normalizer_stats_path and self.normalizer_stats_path.exists():
            print(f"Feature normalizer: {self.normalizer_stats_path}")
        else:
            print("Feature normalizer: fitted on mean scenarios because no stats file was loaded")

        for scenario_idx in range(len(self.mean_scenarios_df)):
            for gen_idx in range(len(self.generator_names)):
                self.plot_regime_generator(scenario_idx=scenario_idx, gen_idx=gen_idx, show=show)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize gradient NN policy results on mean regime trajectories")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/gradient_policy_training_nn_results.json"),
        help="Path to gradient NN training JSON results",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("results/feature_normalizer_stats_gradient.json"),
        help="Path to saved feature normalizer stats",
    )
    parser.add_argument("--case", default="test_case1", help="ScenarioManagerV2 base case reference")
    parser.add_argument("--regime-set", default="policy_training", help="Regime set name from regime_definitions.yaml")
    parser.add_argument("--seed", type=int, default=1, help="Scenario generation seed")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_viz/figures/gradient_policy_training_nn"),
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--beta-smooth",
        type=float,
        default=None,
        help="Override beta_smooth used in economic dispatch; defaults to value in results JSON",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively in addition to saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    scenario_manager = ScenarioManagerV2(args.case)
    scenario_set = scenario_manager.create_scenario_set_from_regimes(
        regime_set=args.regime_set,
        seed=args.seed,
    )

    viz = GradientNNPolicyMeanRegimeVisualizer.from_json(
        results_path=args.results,
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        players_config=scenario_manager.get_players_config(),
        reference_case=args.case,
        output_dir=args.outdir,
        normalizer_stats_path=args.normalizer,
        beta_smooth=args.beta_smooth,
    )
    viz.run(show=args.show)


if __name__ == "__main__":
    main()
