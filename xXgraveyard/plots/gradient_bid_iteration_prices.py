"""
Plot bid trajectories over expanded-training iterations with actual ED price.

Default input:
    results/gradient_bid_training_expanded_results.json

The script reads ``scenario_histories`` from the expanded bid-gradient trainer.
For each update-ready row of a selected base scenario, it solves the actual
quadratic ED and overlays the selected block bids with the ED clearing price at
one time step.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from xXgraveyard.models.gradient_based.economic_dispatch_quad import EconomicDispatchQuadraticModel


def as_profile(value: Any, expected_len: Optional[int] = None, column_name: str = "profile") -> List[float]:
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


class ExpandedBidIterationPricePlotter:
    def __init__(
        self,
        results: Dict[str, Any],
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        self.results = results
        self.scenarios_df = scenarios_df.copy(deep=True).reset_index(drop=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.output_dir = output_dir
        self.block_names = list(results.get("block_names", []))
        if not self.block_names:
            self.block_names = [
                str(col).removesuffix("_cap")
                for col in self.scenarios_df.columns
                if str(col).endswith("_cap")
            ]
        self.block_to_physical = dict(results.get("block_to_physical", {})) or {
            block: block.rsplit("_B", 1)[0] if "_B" in block else block
            for block in self.block_names
        }
        self.physical_generator_names = list(results.get("physical_generator_names", [])) or sorted(
            set(self.block_to_physical.values())
        )
        self.num_time_steps = int(results.get("num_time_steps", self.scenarios_df["time_steps"].iloc[0]))
        params = results.get("parameters", {})
        self.beta_smooth = float(results.get("beta_smooth", params.get("beta_smooth", 0.001)))
        self.scenario_histories = results.get("scenario_histories", {})
        if not self.scenario_histories:
            raise ValueError("Results file does not include scenario_histories")

    @classmethod
    def from_json(
        cls,
        results_path: Path,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        output_dir: Path,
    ) -> "ExpandedBidIterationPricePlotter":
        with results_path.open("r", encoding="utf-8") as file_handle:
            results = json.load(file_handle)
        return cls(
            results=results,
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            output_dir=output_dir,
        )

    def _history_rows(self, scenario_idx: int, include_true_cost: bool) -> List[Dict[str, Any]]:
        key = str(int(scenario_idx))
        if key not in self.scenario_histories and int(scenario_idx) in self.scenario_histories:
            key = int(scenario_idx)
        if key not in self.scenario_histories:
            available = sorted(str(k) for k in self.scenario_histories)
            raise ValueError(f"Unknown scenario {scenario_idx}. Available histories: {available}")

        rows = list(self.scenario_histories[key])
        if not include_true_cost:
            rows = [row for row in rows if str(row.get("history_role")) == "update_ready"]
        rows.sort(key=lambda row: int(row.get("training_iteration", 0)))
        return rows

    def _row_df(self, row: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([row]).copy(deep=True)
        for block_name in self.block_names:
            profile_col = f"{block_name}_bid_profile"
            if profile_col in df.columns:
                profile = as_profile(df.at[0, profile_col], self.num_time_steps, profile_col)
                df.at[0, profile_col] = profile
                df.at[0, f"{block_name}_bid"] = float(profile[0])
        return df

    def _half_capacity_initial_dispatch(self, scenario_df: pd.DataFrame) -> List[List[float]]:
        physical_initial = []
        for physical_name in self.physical_generator_names:
            block_indices = [
                idx for idx, block_name in enumerate(self.block_names)
                if self.block_to_physical.get(block_name, block_name) == physical_name
            ]
            capacity = sum(float(scenario_df.at[0, f"{self.block_names[idx]}_cap"]) for idx in block_indices)
            physical_initial.append(0.5 * capacity)
        return [physical_initial]

    def _cost_bid_df(self, scenario_df: pd.DataFrame) -> pd.DataFrame:
        cost_df = scenario_df.copy(deep=True)
        for block_name in self.block_names:
            cost = float(self.costs_df[f"{block_name}_cost"].iloc[0])
            cost_df.at[0, f"{block_name}_bid_profile"] = [cost] * self.num_time_steps
            cost_df.at[0, f"{block_name}_bid"] = cost
        return cost_df

    def _compute_p_init(self, scenario_df: pd.DataFrame) -> List[List[float]]:
        cost_df = self._cost_bid_df(scenario_df)
        ed_for_p_init = EconomicDispatchQuadraticModel(
            cost_df,
            self.costs_df,
            self.ramps_df,
            p_init=self._half_capacity_initial_dispatch(cost_df),
            beta_coeff=self.beta_smooth,
        )
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Could not compute ED initial dispatch")
        return [list(dispatches[0][0])]

    def _solve_price(self, row: Dict[str, Any], time_step: int) -> float:
        scenario_df = self._row_df(row)
        ed = EconomicDispatchQuadraticModel(
            scenario_df,
            self.costs_df,
            self.ramps_df,
            p_init=self._compute_p_init(scenario_df),
            beta_coeff=self.beta_smooth,
        )
        ed.solve()
        prices = ed.get_clearing_prices()
        if prices is None:
            raise RuntimeError("Actual ED solve did not return clearing prices")
        return float(prices[0][int(time_step)])

    def _select_block_indices(
        self,
        blocks: Optional[str],
        physical: Optional[str],
    ) -> List[int]:
        if blocks:
            selected = []
            for item in blocks.split(","):
                item = item.strip()
                if not item:
                    continue
                if item.lstrip("-").isdigit():
                    idx = int(item)
                    if idx < 0 or idx >= len(self.block_names):
                        raise ValueError(f"Block index {idx} is out of range")
                    selected.append(idx)
                else:
                    if item not in self.block_names:
                        raise ValueError(f"Unknown block '{item}'. Available: {self.block_names}")
                    selected.append(self.block_names.index(item))
            return selected

        if physical:
            selected = [
                idx for idx, block_name in enumerate(self.block_names)
                if self.block_to_physical.get(block_name, block_name) == physical
            ]
            if not selected:
                available = sorted(set(self.block_to_physical.values()))
                raise ValueError(f"Unknown physical generator '{physical}'. Available: {available}")
            return selected

        return list(range(len(self.block_names)))

    def _bid_at(self, row: Dict[str, Any], block_idx: int, time_step: int) -> float:
        block_name = self.block_names[int(block_idx)]
        profile = as_profile(row[f"{block_name}_bid_profile"], self.num_time_steps, f"{block_name}_bid_profile")
        return float(profile[int(time_step)])

    def plot(
        self,
        scenario_idx: int,
        time_step: int,
        block_indices: Iterable[int],
        include_true_cost: bool = False,
        show: bool = False,
    ) -> Path:
        rows = self._history_rows(scenario_idx, include_true_cost=include_true_cost)
        if not rows:
            raise ValueError(f"No history rows selected for scenario {scenario_idx}")

        iterations = np.asarray([int(row["training_iteration"]) for row in rows], dtype=int)
        prices = np.asarray([self._solve_price(row, time_step) for row in rows], dtype=float)
        regime = str(rows[-1].get("regime", f"scenario_{scenario_idx}"))

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            iterations,
            prices,
            color="#111111",
            linewidth=2.5,
            marker="D",
            label=f"Actual ED clearing price t={time_step}",
        )

        for block_idx in block_indices:
            block_name = self.block_names[int(block_idx)]
            bids = np.asarray([self._bid_at(row, block_idx, time_step) for row in rows], dtype=float)
            ax.plot(
                iterations,
                bids,
                linewidth=2.0,
                marker="o",
                label=f"{block_name} bid",
            )

        ax.set_title(
            f"Bid Trajectories and Actual ED Price\n"
            f"Regime {regime}, Scenario {scenario_idx}, Time {time_step}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Bid / Price")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        out_dir = self.output_dir / slugify(regime)
        out_dir.mkdir(parents=True, exist_ok=True)
        block_label = "all_blocks" if len(list(block_indices)) == len(self.block_names) else "selected_blocks"
        out = out_dir / f"bid_price_iterations_s{int(scenario_idx):03d}_t{int(time_step):02d}_{block_label}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"[saved] {out}")
        if show:
            plt.show()
        plt.close(fig)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot expanded-training bids over iterations with actual ED clearing price."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/gradient_bid_training_expanded_results.json"),
        help="Path to expanded gradient bid training JSON results",
    )
    parser.add_argument("--case", default="test_case_bidding_blocks", help="ScenarioManager base case reference")
    parser.add_argument("--regime-set", default="policy_training", help="Regime set name")
    parser.add_argument("--seed", type=int, default=1, help="Scenario generation seed")
    parser.add_argument("--scenario-index", type=int, default=0, help="Base scenario index to plot")
    parser.add_argument("--time-step", type=int, default=0, help="Time step to plot")
    parser.add_argument(
        "--blocks",
        default=None,
        help="Comma-separated block names or indices. Defaults to all blocks unless --physical is set.",
    )
    parser.add_argument("--physical", default=None, help="Plot all blocks belonging to this physical generator")
    parser.add_argument("--include-true-cost", action="store_true", help="Include the true_cost reference row")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_viz/figures/gradient_bid_iteration_prices"),
        help="Directory for generated figures",
    )
    parser.add_argument("--show", action="store_true", help="Display figure interactively in addition to saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results.exists():
        raise FileNotFoundError(f"Results file not found: {args.results}")

    scenario_manager = ScenarioManager(args.case)
    scenario_set = scenario_manager.create_scenario_set_from_regimes(
        regime_set=args.regime_set,
        seed=args.seed,
    )

    plotter = ExpandedBidIterationPricePlotter.from_json(
        results_path=args.results,
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        output_dir=args.outdir,
    )
    block_indices = plotter._select_block_indices(args.blocks, args.physical)
    plotter.plot(
        scenario_idx=args.scenario_index,
        time_step=args.time_step,
        block_indices=block_indices,
        include_true_cost=args.include_true_cost,
        show=args.show,
    )


if __name__ == "__main__":
    main()
