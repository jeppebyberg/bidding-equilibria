"""
Plot merit-order curves for true block costs versus trained direct bid labels.

Default input:
    results/gradient_bid_training_results.json

The plot is built directly from regenerated training scenarios and the final
block-level bid profiles stored in ``bid_history[-1]``. It does not solve ED:
clearing markers are read from the merit stack at the selected demand level.
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


class GradientBidMeritOrderVisualizer:
    """Compare competitive cost stack and trained bid stack for selected scenarios."""

    def __init__(
        self,
        results: Dict[str, Any],
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        output_dir: Path,
        result_view: str = "final",
        result_iteration: Optional[int] = None,
        result_role: str = "update_ready",
    ) -> None:
        self.results = results
        self.scenarios_df = scenarios_df.copy(deep=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.output_dir = output_dir
        self.result_view = result_view
        self.result_iteration = result_iteration
        self.result_role = result_role

        self.num_time_steps = int(results.get("num_time_steps", self.scenarios_df["time_steps"].iloc[0]))
        self.block_names = self._infer_block_names()
        self.block_to_physical = self._infer_block_to_physical()
        self.final_bids = self._extract_final_bids()
        self.true_costs = self._extract_true_costs()

    @classmethod
    def from_json(
        cls,
        results_path: Path,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        output_dir: Path,
        result_view: str = "final",
        result_iteration: Optional[int] = None,
        result_role: str = "update_ready",
    ) -> "GradientBidMeritOrderVisualizer":
        with results_path.open("r", encoding="utf-8") as file_handle:
            results = json.load(file_handle)
        return cls(
            results=results,
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            output_dir=output_dir,
            result_view=result_view,
            result_iteration=result_iteration,
            result_role=result_role,
        )

    def _infer_block_names(self) -> List[str]:
        block_names = list(self.results.get("block_names", []))
        if block_names:
            return block_names
        return [
            str(col).removesuffix("_cap")
            for col in self.scenarios_df.columns
            if str(col).endswith("_cap")
        ]

    def _infer_block_to_physical(self) -> Dict[str, str]:
        mapping = dict(self.results.get("block_to_physical", {}))
        if mapping:
            return mapping
        return {
            block_name: block_name.rsplit("_B", 1)[0] if "_B" in block_name else block_name
            for block_name in self.block_names
        }

    def _extract_final_bids(self) -> np.ndarray:
        if self.result_view == "expanded":
            return self._extract_expanded_bids()

        bid_history = self.results.get("bid_history", [])
        if not bid_history:
            raise ValueError("Results file does not include bid_history")

        final_bids = np.asarray(bid_history[-1], dtype=np.float64)
        if final_bids.ndim != 3:
            raise ValueError("Final bid history must have shape [scenario][block][time]")
        if final_bids.shape[0] < len(self.scenarios_df):
            self.scenarios_df = self.scenarios_df.iloc[:final_bids.shape[0]].copy(deep=True).reset_index(drop=True)
        expected = (len(self.scenarios_df), len(self.block_names), self.num_time_steps)
        if final_bids.shape != expected:
            raise ValueError(f"Final bid history must have shape {expected}, got {final_bids.shape}")
        if not np.all(np.isfinite(final_bids)):
            raise ValueError("Final bid history contains non-finite values")
        return final_bids

    def _extract_expanded_bids(self) -> np.ndarray:
        expanded_rows = self.results.get("expanded_bid_history", [])
        if not expanded_rows:
            raise ValueError(
                "Results file does not include expanded_bid_history. "
                "Use --result-view final or regenerate expanded results."
            )

        selected_by_scenario: Dict[int, Dict[str, Any]] = {}
        for row in expanded_rows:
            if str(row.get("history_role")) != self.result_role:
                continue
            if self.result_iteration is not None and int(row.get("training_iteration")) != int(self.result_iteration):
                continue
            base_idx = int(row["base_scenario_idx"])
            current = selected_by_scenario.get(base_idx)
            if current is None or int(row["training_iteration"]) > int(current["training_iteration"]):
                selected_by_scenario[base_idx] = row

        selected_indices = sorted(selected_by_scenario)
        if not selected_indices:
            raise ValueError(f"No expanded result rows matched role={self.result_role}, iteration={self.result_iteration}")
        max_selected_idx = max(selected_indices)
        if max_selected_idx >= len(self.scenarios_df):
            raise ValueError(
                "Expanded result references scenario index "
                f"{max_selected_idx}, but regenerated scenarios only contain {len(self.scenarios_df)} rows"
            )
        missing_between = [
            idx for idx in range(max_selected_idx + 1)
            if idx not in selected_by_scenario
        ]
        if missing_between:
            selector = f"role={self.result_role}"
            if self.result_iteration is not None:
                selector += f", iteration={self.result_iteration}"
            raise ValueError(f"Expanded results missing selected rows for scenarios {missing_between} ({selector})")

        self.scenarios_df = self.scenarios_df.iloc[:max_selected_idx + 1].copy(deep=True).reset_index(drop=True)

        bids = np.asarray(
            [
                selected_by_scenario[idx]["bids"]
                for idx in range(max_selected_idx + 1)
            ],
            dtype=np.float64,
        )
        expected = (len(self.scenarios_df), len(self.block_names), self.num_time_steps)
        if bids.shape != expected:
            raise ValueError(f"Selected expanded bids must have shape {expected}, got {bids.shape}")
        if not np.all(np.isfinite(bids)):
            raise ValueError("Selected expanded bids contain non-finite values")
        return bids

    def _extract_true_costs(self) -> np.ndarray:
        costs = np.asarray(
            [float(self.costs_df[f"{block_name}_cost"].iloc[0]) for block_name in self.block_names],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(costs)):
            raise ValueError("True cost vector contains non-finite values")
        return costs

    def _demand_at(self, scenario_idx: int, time_step: int) -> float:
        profile = as_profile(
            self.scenarios_df.at[scenario_idx, "demand_profile"],
            self.num_time_steps,
            "demand_profile",
        )
        return float(profile[time_step])

    def _available_capacity_at(self, scenario_idx: int, time_step: int) -> np.ndarray:
        capacities, _ = self._available_capacity_details_at(scenario_idx, time_step)
        return capacities

    def _available_capacity_details_at(
        self,
        scenario_idx: int,
        time_step: int,
    ) -> tuple[np.ndarray, List[str]]:
        capacities = []
        sources = []
        for block_name in self.block_names:
            profile_col = f"{block_name}_profile"
            cap_profile_col = f"{block_name}_cap_profile"
            if profile_col in self.scenarios_df.columns:
                profile = as_profile(
                    self.scenarios_df.at[scenario_idx, profile_col],
                    self.num_time_steps,
                    profile_col,
                )
                capacities.append(float(profile[time_step]))
                sources.append(profile_col)
            elif cap_profile_col in self.scenarios_df.columns:
                profile = as_profile(
                    self.scenarios_df.at[scenario_idx, cap_profile_col],
                    self.num_time_steps,
                    cap_profile_col,
                )
                capacities.append(float(profile[time_step]))
                sources.append(cap_profile_col)
            else:
                capacities.append(float(self.scenarios_df.at[scenario_idx, f"{block_name}_cap"]))
                sources.append(f"{block_name}_cap")

        capacities_arr = np.asarray(capacities, dtype=np.float64)
        if np.any(capacities_arr < -1e-9) or not np.all(np.isfinite(capacities_arr)):
            raise ValueError(f"Invalid capacities for scenario {scenario_idx}, time {time_step}")
        return np.maximum(capacities_arr, 0.0), sources

    def _available_capacity_from_row(
        self,
        row: pd.Series,
        block_name: str,
        time_step: int,
    ) -> float:
        for col in (f"{block_name}_profile", f"{block_name}_cap_profile"):
            if col not in row.index:
                continue
            value = row[col]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            try:
                profile = as_profile(value, self.num_time_steps, col)
                return max(float(profile[int(time_step)]), 0.0)
            except (TypeError, ValueError):
                continue
        return max(float(row[f"{block_name}_cap"]), 0.0)

    def _scenario_df_with_bids(self, scenario_idx: int) -> pd.DataFrame:
        row = self.scenarios_df.iloc[[int(scenario_idx)]].copy(deep=True).reset_index(drop=True)
        for block_idx, block_name in enumerate(self.block_names):
            bid_profile = self.final_bids[int(scenario_idx), block_idx, :].astype(float).tolist()
            row.at[0, f"{block_name}_bid_profile"] = bid_profile
            row.at[0, f"{block_name}_bid"] = float(bid_profile[0])
        return row

    def _cost_bid_scenario_df(self, scenario_idx: int) -> pd.DataFrame:
        row = self.scenarios_df.iloc[[int(scenario_idx)]].copy(deep=True).reset_index(drop=True)
        for block_name in self.block_names:
            cost = float(self.costs_df[f"{block_name}_cost"].iloc[0])
            row.at[0, f"{block_name}_bid_profile"] = [cost] * self.num_time_steps
            row.at[0, f"{block_name}_bid"] = cost
        return row

    def _half_capacity_initial_dispatch(self, scenario_df: pd.DataFrame) -> List[List[float]]:
        physical_names = list(self.results.get("physical_generator_names", []))
        if not physical_names:
            physical_names = sorted(set(self.block_to_physical.values()))
        physical_initial = []
        row = scenario_df.iloc[0]
        for physical_name in physical_names:
            block_indices = [
                idx for idx, block_name in enumerate(self.block_names)
                if self.block_to_physical.get(block_name, block_name) == physical_name
            ]
            capacity = sum(
                self._available_capacity_from_row(row, self.block_names[idx], time_step=0)
                for idx in block_indices
            )
            physical_initial.append(0.5 * capacity)
        return [physical_initial]

    def _compute_ed_p_init(self, scenario_idx: int) -> List[List[float]]:
        cost_df = self._cost_bid_scenario_df(scenario_idx)
        ed_for_p_init = EconomicDispatchQuadraticModel(
            cost_df,
            self.costs_df,
            self.ramps_df,
            p_init=self._half_capacity_initial_dispatch(cost_df),
            beta_coeff=float(self.results.get("beta_smooth", self.results.get("parameters", {}).get("beta_smooth", 0.001))),
        )
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Could not compute ED initial dispatch for merit-order overlay")
        return [list(dispatches[0][0])]

    def _solve_actual_ed(self, scenario_idx: int) -> Dict[str, Any]:
        scenario_df = self._scenario_df_with_bids(scenario_idx)
        ed = EconomicDispatchQuadraticModel(
            scenario_df,
            self.costs_df,
            self.ramps_df,
            p_init=self._compute_ed_p_init(scenario_idx),
            beta_coeff=float(self.results.get("beta_smooth", self.results.get("parameters", {}).get("beta_smooth", 0.001))),
        )
        ed.solve()
        block_dispatches = ed.get_block_dispatches()
        physical_dispatches = ed.get_dispatches()
        prices = ed.get_clearing_prices()
        if block_dispatches is None or physical_dispatches is None or prices is None:
            raise RuntimeError("Actual ED solve did not return dispatches and prices")
        return {
            "block_dispatches": np.asarray(block_dispatches[0], dtype=np.float64),
            "physical_dispatches": np.asarray(physical_dispatches[0], dtype=np.float64),
            "prices": np.asarray(prices[0], dtype=np.float64),
        }

    def _block_quadratic_cost(self, block_name: str) -> float:
        for col in (f"{block_name}_quadratic_cost", f"{block_name}_cost_quadratic", f"{block_name}_quad_cost"):
            if col in self.costs_df.columns:
                return float(self.costs_df[col].iloc[0])
        return 0.0

    def _ed_physical_summary_at_time(
        self,
        ed_result: Dict[str, Any],
        time_step: int,
    ) -> List[str]:
        block_dispatch = np.asarray(ed_result["block_dispatches"][time_step], dtype=np.float64)
        price = float(ed_result["prices"][time_step])
        physical_names = list(self.results.get("physical_generator_names", []))
        if not physical_names:
            physical_names = sorted(set(self.block_to_physical.values()))

        lines = []
        for physical_name in physical_names:
            block_indices = [
                idx for idx, block_name in enumerate(self.block_names)
                if self.block_to_physical.get(block_name, block_name) == physical_name
            ]
            dispatch = float(np.sum(block_dispatch[block_indices]))
            profit = 0.0
            for block_idx in block_indices:
                block_name = self.block_names[block_idx]
                p = float(block_dispatch[block_idx])
                cost = float(self.true_costs[block_idx])
                quad = self._block_quadratic_cost(block_name)
                profit += price * p - cost * p - 0.5 * quad * p**2
            lines.append(f"{physical_name}: P={dispatch:.1f}, pi={profit:.1f}")
        return lines

    def _build_stack(
        self,
        prices: np.ndarray,
        capacities: np.ndarray,
    ) -> Dict[str, Any]:
        valid = capacities > 1e-9
        order = sorted(
            [idx for idx in range(len(self.block_names)) if valid[idx]],
            key=lambda idx: (float(prices[idx]), idx),
        )
        if not order:
            raise ValueError("No positive-capacity blocks are available for this scenario/time")

        ordered_caps = capacities[order]
        ordered_prices = prices[order]
        cumulative = np.cumsum(ordered_caps)
        edges = np.concatenate(([0.0], cumulative))
        step_values = np.concatenate((ordered_prices, [ordered_prices[-1]]))
        return {
            "order": order,
            "capacities": ordered_caps,
            "prices": ordered_prices,
            "cumulative": cumulative,
            "edges": edges,
            "step_values": step_values,
        }

    @staticmethod
    def _clearing_price(stack: Dict[str, Any], demand: float) -> float:
        cumulative = np.asarray(stack["cumulative"], dtype=np.float64)
        prices = np.asarray(stack["prices"], dtype=np.float64)
        idx = int(np.searchsorted(cumulative, float(demand), side="left"))
        idx = min(idx, len(prices) - 1)
        return float(prices[idx])

    def _annotate_stack(
        self,
        ax: plt.Axes,
        stack: Dict[str, Any],
        prices_label: str,
        color: str,
        y_offset: float,
    ) -> None:
        for local_idx, block_idx in enumerate(stack["order"]):
            left = 0.0 if local_idx == 0 else float(stack["cumulative"][local_idx - 1])
            right = float(stack["cumulative"][local_idx])
            midpoint = 0.5 * (left + right)
            price = float(stack["prices"][local_idx])
            block_name = self.block_names[block_idx]
            physical_name = self.block_to_physical.get(block_name, block_name)
            ax.annotate(
                f"{block_name}\n{prices_label}: {price:.2f}\n({physical_name})",
                xy=(midpoint, price),
                xytext=(midpoint, price + y_offset),
                ha="center",
                va="bottom" if y_offset >= 0 else "top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.75),
            )

    def _shade_ed_dispatch(
        self,
        ax: plt.Axes,
        stack: Dict[str, Any],
        block_dispatch: np.ndarray,
        y_min: float,
        y_span: float,
    ) -> None:
        bar_y = y_min + 0.025 * y_span
        bar_h = 0.035 * y_span
        label_added = False
        for local_idx, block_idx in enumerate(stack["order"]):
            dispatched = float(block_dispatch[block_idx])
            if dispatched <= 1e-9:
                continue
            left = 0.0 if local_idx == 0 else float(stack["cumulative"][local_idx - 1])
            width = min(dispatched, float(stack["capacities"][local_idx]))
            ax.broken_barh(
                [(left, width)],
                (bar_y, bar_h),
                facecolors="#111111",
                alpha=0.55,
                label="Actual ED Dispatch" if not label_added else None,
            )
            label_added = True

    def plot_scenario_time(
        self,
        scenario_idx: int,
        time_step: int,
        annotate: bool = True,
        show: bool = False,
    ) -> Path:
        scenario_idx = int(scenario_idx)
        time_step = int(time_step)
        if scenario_idx < 0 or scenario_idx >= len(self.scenarios_df):
            raise ValueError(f"scenario_idx must be in [0, {len(self.scenarios_df) - 1}], got {scenario_idx}")
        if time_step < 0 or time_step >= self.num_time_steps:
            raise ValueError(f"time_step must be in [0, {self.num_time_steps - 1}], got {time_step}")

        demand = self._demand_at(scenario_idx, time_step)
        capacities, capacity_sources = self._available_capacity_details_at(scenario_idx, time_step)
        trained_bids = self.final_bids[scenario_idx, :, time_step]
        print(
            f"[data] scenario={scenario_idx}, time={time_step}, "
            f"demand_profile[{time_step}]={demand:.6f}"
        )
        print(
            "[data] block availability used: "
            + ", ".join(
                f"{block}={cap:.3f} ({source})"
                for block, cap, source in zip(self.block_names, capacities, capacity_sources)
            )
        )

        cost_stack = self._build_stack(self.true_costs, capacities)
        bid_stack = self._build_stack(trained_bids, capacities)
        cost_clearing = self._clearing_price(cost_stack, demand)
        bid_clearing = self._clearing_price(bid_stack, demand)
        ed_result = self._solve_actual_ed(scenario_idx)
        ed_price = float(ed_result["prices"][time_step])
        ed_block_dispatch = np.asarray(ed_result["block_dispatches"][time_step], dtype=np.float64)
        total_capacity = float(np.sum(capacities))

        regime = str(self.scenarios_df.at[scenario_idx, "regime"]) if "regime" in self.scenarios_df.columns else "scenario"
        y_values = np.concatenate((self.true_costs, trained_bids, [cost_clearing, bid_clearing, ed_price]))
        y_min = min(0.0, float(np.min(y_values)))
        y_max = max(float(np.max(y_values)), cost_clearing, bid_clearing)
        y_span = max(y_max - y_min, 1.0)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.step(
            cost_stack["edges"],
            cost_stack["step_values"],
            where="post",
            color="#2636ff",
            linewidth=2.6,
            label="Competitive Merit Order (True Costs)",
        )
        ax.step(
            bid_stack["edges"],
            bid_stack["step_values"],
            where="post",
            color="#ff2b2b",
            linewidth=2.6,
            linestyle="--",
            label="Strategic Merit Order (Trained Bids)",
        )

        ax.axvline(demand, color="#2ca02c", linewidth=2.4, label=f"Demand ({demand:.1f} MW)")
        ax.scatter(
            [demand],
            [cost_clearing],
            s=150,
            color="#2636ff",
            edgecolor="black",
            zorder=5,
            label=f"Competitive Clearing ({cost_clearing:.2f})",
        )
        ax.scatter(
            [demand],
            [bid_clearing],
            s=150,
            marker="s",
            color="#ff2b2b",
            edgecolor="black",
            zorder=5,
            label=f"Strategic Clearing ({bid_clearing:.2f})",
        )
        ax.axhline(
            ed_price,
            color="#111111",
            linewidth=2.0,
            linestyle=":",
            label=f"Actual ED Price ({ed_price:.2f})",
        )
        ax.scatter(
            [demand],
            [ed_price],
            s=130,
            marker="D",
            color="#111111",
            edgecolor="white",
            zorder=6,
            label="Actual ED Price at Demand",
        )

        if annotate:
            self._annotate_stack(ax, cost_stack, "Cost", "#2636ff", y_offset=0.08 * y_span)
            self._annotate_stack(ax, bid_stack, "Bid", "#ff2b2b", y_offset=-0.12 * y_span)
        self._shade_ed_dispatch(ax, bid_stack, ed_block_dispatch, y_min, y_span)

        summary_lines = ["Actual ED at time step", f"lambda={ed_price:.2f}"] + self._ed_physical_summary_at_time(
            ed_result,
            time_step,
        )
        ax.text(
            0.985,
            0.03,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#111111", alpha=0.88),
        )

        ax.set_title(
            f"Merit Order Comparison: True Costs vs Trained Bids\n"
            f"Regime {regime}, Scenario {scenario_idx}, Time {time_step}",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlabel("Cumulative Available Capacity (MW)", fontsize=12)
        ax.set_ylabel("Price / Bid ($/MWh)", fontsize=12)
        ax.set_xlim(0.0, max(total_capacity, demand) * 1.05)
        ax.set_ylim(y_min - 0.18 * y_span, y_max + 0.22 * y_span)
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()

        regime_dir = self.output_dir / slugify(regime)
        regime_dir.mkdir(parents=True, exist_ok=True)
        out = regime_dir / f"merit_order_s{scenario_idx:03d}_t{time_step:02d}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"[saved] {out}")
        if show:
            plt.show()
        plt.close(fig)
        return out

    def first_scenario_indices_by_regime(self) -> List[int]:
        if "regime" not in self.scenarios_df.columns:
            return [0]
        indices = []
        for regime in sorted(self.scenarios_df["regime"].dropna().astype(str).unique()):
            positions = np.flatnonzero((self.scenarios_df["regime"].astype(str) == regime).to_numpy())
            if positions.size:
                indices.append(int(positions[0]))
        return indices

    def scenario_indices_for_regime(self, regime: str) -> List[int]:
        if "regime" not in self.scenarios_df.columns:
            raise ValueError("Cannot select by regime because scenarios_df has no 'regime' column")
        positions = np.flatnonzero((self.scenarios_df["regime"].astype(str) == str(regime)).to_numpy())
        if positions.size == 0:
            available = sorted(self.scenarios_df["regime"].dropna().astype(str).unique())
            raise ValueError(f"Unknown regime '{regime}'. Available: {available}")
        return [int(positions[0])]

    def run(
        self,
        scenario_indices: Iterable[int],
        time_step: int,
        annotate: bool,
        show: bool,
    ) -> List[Path]:
        outputs = []
        for scenario_idx in scenario_indices:
            outputs.append(
                self.plot_scenario_time(
                    scenario_idx=int(scenario_idx),
                    time_step=time_step,
                    annotate=annotate,
                    show=show,
                )
            )
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot merit-order curves comparing true block costs and trained bid labels."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/gradient_bid_training_expanded_results.json"),
        help="Path to direct gradient bid training JSON results",
    )
    parser.add_argument("--case", default="test_case_bidding_blocks", help="ScenarioManager base case reference")
    parser.add_argument("--regime-set", default="policy_training", help="Regime set name from regime_definitions.yaml")
    parser.add_argument("--seed", type=int, default=1, help="Scenario generation seed")
    parser.add_argument("--time-step", type=int, default=0, help="Time step to plot")
    parser.add_argument("--scenario-index", type=int, default=None, help="Specific training scenario index to plot")
    parser.add_argument("--regime", default=None, help="Plot the first scenario from this regime")
    parser.add_argument(
        "--result-view",
        choices=["final", "expanded"],
        default="final",
        help="Use final bid_history-compatible bids or select rows from expanded_bid_history",
    )
    parser.add_argument(
        "--result-iteration",
        type=int,
        default=None,
        help="Expanded result iteration to plot. Omit to use the latest selected row per scenario.",
    )
    parser.add_argument(
        "--result-role",
        default="update_ready",
        help="Expanded result history_role to plot, e.g. update_ready or true_cost",
    )
    parser.add_argument(
        "--all-regimes",
        action="store_true",
        help="Plot the first scenario in every regime. This is the default if no scenario/regime is selected.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_viz/figures/gradient_bid_merit_order"),
        help="Directory for generated figures",
    )
    parser.add_argument("--no-annotate", action="store_true", help="Disable block labels/arrows")
    parser.add_argument("--show", action="store_true", help="Display figures interactively in addition to saving")
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

    viz = GradientBidMeritOrderVisualizer.from_json(
        results_path=args.results,
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        output_dir=args.outdir,
        result_view=args.result_view,
        result_iteration=args.result_iteration,
        result_role=args.result_role,
    )

    if args.scenario_index is not None:
        scenario_indices = [int(args.scenario_index)]
    elif args.regime is not None:
        scenario_indices = viz.scenario_indices_for_regime(args.regime)
    else:
        scenario_indices = viz.first_scenario_indices_by_regime()

    viz.run(
        scenario_indices=scenario_indices,
        time_step=args.time_step,
        annotate=not args.no_annotate,
        show=args.show,
    )


if __name__ == "__main__":
    main()
