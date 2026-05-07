"""
Visualize final direct-gradient bid labels on the first scenario in each regime.

Default input:
    results/gradient_bid_training_results.json

The direct bid trainer has no policy to evaluate on a synthetic scenario.
Instead, this script regenerates the training scenarios, selects the first
scenario in each regime, takes that scenario's saved final bid profiles from
``bid_history[-1]``, solves economic dispatch, and saves one four-panel figure
per (regime, generator).
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
from models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel


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


class GradientBidFirstScenarioRegimeVisualizer:
    """Create first-scenario-per-regime trajectory plots for direct bid-gradient results."""

    def __init__(
        self,
        results: Dict[str, Any],
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        output_dir: Path,
        beta_smooth: Optional[float] = None,
    ) -> None:
        self.results = results
        self.scenarios_df = scenarios_df.copy(deep=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.output_dir = output_dir
        self.beta_smooth = float(beta_smooth if beta_smooth is not None else results.get("beta_smooth", 0.001))

        self.generator_names = list(results.get("generator_names", []))
        if not self.generator_names:
            self.generator_names = [
                col.replace("_cap", "")
                for col in self.scenarios_df.columns
                if col.endswith("_cap")
            ]
        if not self.generator_names:
            raise ValueError("Could not infer generator names")

        self.num_time_steps = int(results.get("num_time_steps", self.scenarios_df["time_steps"].iloc[0]))
        self.final_bids = self._extract_final_bids()
        self.regime_scenarios_df = self._build_regime_scenarios_df()
        self.dispatches, self.clearing_prices = self._solve_regime_dispatch()

    @classmethod
    def from_json(
        cls,
        results_path: Path,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        output_dir: Path,
        beta_smooth: Optional[float] = None,
    ) -> "GradientBidFirstScenarioRegimeVisualizer":
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        return cls(
            results=results,
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            output_dir=output_dir,
            beta_smooth=beta_smooth,
        )

    def _extract_final_bids(self) -> np.ndarray:
        bid_history = self.results.get("bid_history", [])
        if not bid_history:
            raise ValueError("Results file does not include bid_history")
        final_bids = np.asarray(bid_history[-1], dtype=np.float64)
        expected = (len(self.scenarios_df), len(self.generator_names), self.num_time_steps)
        if final_bids.shape != expected:
            raise ValueError(f"Final bid history must have shape {expected}, got {final_bids.shape}")
        if not np.all(np.isfinite(final_bids)):
            raise ValueError("Final bid history contains non-finite values")
        return final_bids

    def _build_regime_scenarios_df(self) -> pd.DataFrame:
        if "regime" not in self.scenarios_df.columns:
            raise ValueError("scenarios_df must include a 'regime' column")
        if "demand_profile" not in self.scenarios_df.columns:
            raise ValueError("scenarios_df must include a 'demand_profile' column")

        rows: List[Dict[str, Any]] = []
        regimes = sorted(self.scenarios_df["regime"].dropna().astype(str).unique())
        for regime_scenario_idx, regime in enumerate(regimes):
            regime_mask = self.scenarios_df["regime"].astype(str) == regime
            source_positions = np.flatnonzero(regime_mask.to_numpy())
            if source_positions.size == 0:
                continue
            source_idx = int(source_positions[0])
            source_row = self.scenarios_df.iloc[source_idx]

            row: Dict[str, Any] = {
                "scenario_id": regime_scenario_idx + 1,
                "source_scenario_index": source_idx,
                "regime": regime,
                "time_steps": self.num_time_steps,
            }

            demand_profile = as_profile(source_row["demand_profile"], self.num_time_steps, "demand_profile")
            row["demand_profile"] = demand_profile
            row["demand"] = float(np.mean(demand_profile))

            for gen_idx, gen_name in enumerate(self.generator_names):
                cap_col = f"{gen_name}_cap"
                if cap_col not in self.scenarios_df.columns:
                    raise ValueError(f"Missing capacity column '{cap_col}'")
                row[cap_col] = float(source_row[cap_col])

                for profile_col in (f"{gen_name}_profile", f"{gen_name}_cap_profile"):
                    if profile_col in self.scenarios_df.columns:
                        row[profile_col] = as_profile(source_row[profile_col], self.num_time_steps, profile_col)

                if f"{gen_name}_profile" not in row and f"{gen_name}_cap_profile" in row:
                    row[f"{gen_name}_profile"] = list(row[f"{gen_name}_cap_profile"])

                bid_profile = self.final_bids[source_idx, gen_idx, :]
                bid_profile_list = bid_profile.astype(float).tolist()
                row[f"{gen_name}_bid_profile"] = bid_profile_list
                row[f"{gen_name}_bid"] = float(bid_profile_list[0])

            rows.append(row)

        return pd.DataFrame(rows)

    def _compute_p_init(self) -> List[List[float]]:
        initial_dispatch = []
        for _, row in self.regime_scenarios_df.iterrows():
            initial_dispatch.append([0.5 * float(row[f"{gen}_cap"]) for gen in self.generator_names])

        cost_bid_df = self.regime_scenarios_df.copy(deep=True)
        for gen_name in self.generator_names:
            cost = float(self.costs_df[f"{gen_name}_cost"].iloc[0])
            cost_bid_df[f"{gen_name}_bid"] = cost
            cost_bid_df[f"{gen_name}_bid_profile"] = [
                [cost] * self.num_time_steps
                for _ in range(len(cost_bid_df))
            ]

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
            raise RuntimeError("Could not compute initial dispatch for regime scenarios")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def _solve_regime_dispatch(self) -> tuple[List[List[List[float]]], List[List[float]]]:
        """Solve each selected regime scenario independently to get scenario-local dual prices."""
        p_init_all = self._compute_p_init()
        dispatches_all: List[List[List[float]]] = []
        prices_all: List[List[float]] = []

        for scenario_idx in range(len(self.regime_scenarios_df)):
            scenario_df = self.regime_scenarios_df.iloc[[scenario_idx]].copy(deep=True).reset_index(drop=True)
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
                regime = self.regime_scenarios_df.at[scenario_idx, "regime"]
                raise RuntimeError(f"Economic dispatch failed for regime scenario '{regime}'")

            dispatches_all.append(dispatches[0])
            prices_all.append(prices[0])

        return dispatches_all, prices_all

    def _available_capacity_profile(self, scenario_idx: int, gen_name: str) -> np.ndarray:
        row = self.regime_scenarios_df.iloc[scenario_idx]
        for profile_col in (f"{gen_name}_cap_profile", f"{gen_name}_profile"):
            if profile_col in self.regime_scenarios_df.columns and isinstance(row.get(profile_col), list):
                return np.asarray(as_profile(row[profile_col], self.num_time_steps, profile_col), dtype=float)
        return np.full(self.num_time_steps, float(row[f"{gen_name}_cap"]), dtype=float)

    def plot_regime_generator(self, scenario_idx: int, gen_idx: int, show: bool = False) -> None:
        row = self.regime_scenarios_df.iloc[scenario_idx]
        regime = str(row["regime"])
        source_idx = int(row["source_scenario_index"])
        gen_name = self.generator_names[gen_idx]
        time_axis = np.arange(self.num_time_steps)

        demand = np.asarray(row["demand_profile"], dtype=float)
        capacity = self._available_capacity_profile(scenario_idx, gen_name)
        bid = np.asarray(row[f"{gen_name}_bid_profile"], dtype=float)
        price = np.asarray(self.clearing_prices[scenario_idx], dtype=float)
        dispatch = np.asarray(self.dispatches[scenario_idx], dtype=float)[:, gen_idx]
        marginal_offer = bid + 2.0 * self.beta_smooth * dispatch

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        fig.suptitle(f"First Scenario in Regime - {regime} - {gen_name} (scenario {source_idx})", fontsize=14)

        axes[0].plot(time_axis, demand, color="tab:blue", marker="o", linewidth=2.0)
        axes[0].set_ylabel("Demand")
        axes[0].set_title("Demand Trajectory")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_axis, capacity, color="tab:green", marker="o", linewidth=2.0)
        axes[1].set_ylabel("MW")
        axes[1].set_title(f"{gen_name} Available Capacity")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            time_axis,
            bid,
            color="tab:orange",
            marker="o",
            linewidth=2.0,
            label=f"{gen_name} bid",
        )
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
        axes[2].set_title("Direct Bid Label and Market Clearing Price")
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
        out = regime_dir / f"gradient_bid_first_scenario_{gen_name}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"[saved] {out}")
        if show:
            plt.show()
        plt.close(fig)

    def run(self, show: bool = False) -> None:
        print("\n=== Gradient Bid First-Scenario Regime Visualization ===")
        print(f"Regimes: {', '.join(self.regime_scenarios_df['regime'].astype(str).tolist())}")
        print(f"Generators: {', '.join(self.generator_names)}")
        print("Bid profiles: final labels from the first generated scenario in each regime")

        for scenario_idx in range(len(self.regime_scenarios_df)):
            for gen_idx in range(len(self.generator_names)):
                self.plot_regime_generator(scenario_idx=scenario_idx, gen_idx=gen_idx, show=show)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize direct gradient bid training results for the first scenario in each regime"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/gradient_bid_training_results.json"),
        help="Path to direct gradient bid training JSON results",
    )
    parser.add_argument("--case", default="test_case1", help="ScenarioManagerV2 base case reference")
    parser.add_argument("--regime-set", default="policy_training", help="Regime set name from regime_definitions.yaml")
    parser.add_argument("--seed", type=int, default=1, help="Scenario generation seed")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_viz/figures/gradient_bid_training"),
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

    viz = GradientBidFirstScenarioRegimeVisualizer.from_json(
        results_path=args.results,
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        output_dir=args.outdir,
        beta_smooth=args.beta_smooth,
    )
    viz.run(show=args.show)


if __name__ == "__main__":
    main()
