"""
Visualize block-aware PoA optimization results.

The expected input is the JSON produced by
models/PoA/PoA_optimization_bidding_blocks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


class PoABiddingBlocksVisualizer:
    def __init__(self, results: dict[str, Any], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time = np.arange(int(results["num_time_steps"]))
        self.generators = results["generators"]

    @classmethod
    def from_json(cls, path: Path, output_dir: Path) -> "PoABiddingBlocksVisualizer":
        with path.open("r", encoding="utf-8") as file_handle:
            results = json.load(file_handle)
        return cls(results=results, output_dir=output_dir)

    @staticmethod
    def _series(values: list[Any]) -> np.ndarray:
        return np.asarray([np.nan if value is None else float(value) for value in values])

    def _save(self, fig: plt.Figure, filename: str, show: bool) -> Path:
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_system_overview(self, show: bool = False) -> Path:
        objective = self.results.get("objective", {})
        demand = self._series(self.results["demand_profile"])
        eq_price = self._series(self.results["equilibrium_price_profile"])
        opt_price = self._series(self.results["optimal_price_profile"])
        total_capacity = np.zeros_like(demand, dtype=float)
        total_eq_dispatch = np.zeros_like(demand, dtype=float)
        total_opt_dispatch = np.zeros_like(demand, dtype=float)

        for generator in self.generators.values():
            total_capacity += self._series(generator["physical_capacity_profile"])
            total_eq_dispatch += self._series(generator["equilibrium_physical_dispatch"])
            total_opt_dispatch += self._series(generator["optimal_physical_dispatch"])

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        title = (
            f"PoA={objective.get('PoA_difference', np.nan):.2f}, "
            f"C_eq={objective.get('C_eq', np.nan):.2f}, "
            f"C_opt={objective.get('C_opt', np.nan):.2f}, "
            f"ratio={objective.get('PoA_ratio', np.nan):.3f}"
        )
        fig.suptitle(title, fontsize=12)

        axes[0].plot(self.time, demand, color="black", marker="o", linewidth=2.0, label="Demand")
        axes[0].plot(
            self.time,
            total_capacity,
            color="tab:green",
            marker="s",
            linewidth=1.8,
            label="Available capacity",
        )
        axes[0].set_ylabel("MW")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(
            self.time,
            total_eq_dispatch,
            color="tab:red",
            marker="o",
            linewidth=2.0,
            label="Equilibrium dispatch",
        )
        axes[1].plot(
            self.time,
            total_opt_dispatch,
            color="tab:blue",
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="Optimal dispatch",
        )
        axes[1].set_ylabel("MW")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(
            self.time,
            eq_price,
            color="tab:red",
            marker="o",
            linewidth=2.0,
            label="Equilibrium price",
        )
        axes[2].plot(
            self.time,
            opt_price,
            color="tab:blue",
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="Optimal price",
        )
        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Price")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.25)

        return self._save(fig, "poa_system_overview.png", show)

    def plot_physical_dispatch(self, show: bool = False) -> Path:
        generator_names = list(self.generators.keys())
        eq_dispatch = np.vstack(
            [
                self._series(self.generators[name]["equilibrium_physical_dispatch"])
                for name in generator_names
            ]
        )
        opt_dispatch = np.vstack(
            [
                self._series(self.generators[name]["optimal_physical_dispatch"])
                for name in generator_names
            ]
        )
        demand = self._series(self.results["demand_profile"])

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].stackplot(self.time, eq_dispatch, labels=generator_names, alpha=0.85)
        axes[0].plot(self.time, demand, color="black", linewidth=2.0, label="Demand")
        axes[0].set_title("Equilibrium Physical Dispatch")
        axes[0].set_ylabel("MW")
        axes[0].grid(True, alpha=0.25)

        axes[1].stackplot(self.time, opt_dispatch, labels=generator_names, alpha=0.85)
        axes[1].plot(self.time, demand, color="black", linewidth=2.0, label="Demand")
        axes[1].set_title("Optimal Physical Dispatch")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("MW")
        axes[1].grid(True, alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6))
        fig.subplots_adjust(bottom=0.14)
        return self._save(fig, "poa_physical_dispatch_stack.png", show)

    def plot_bid_price_comparison(self, show: bool = False) -> Path:
        eq_price = self._series(self.results["equilibrium_price_profile"])
        opt_price = self._series(self.results["optimal_price_profile"])
        n_generators = len(self.generators)
        fig, axes = plt.subplots(
            n_generators,
            1,
            figsize=(11, max(2.6 * n_generators, 5)),
            sharex=True,
            squeeze=False,
        )

        for axis, (generator_name, generator) in zip(axes[:, 0], self.generators.items()):
            axis.plot(self.time, eq_price, color="black", linewidth=1.8, label="Eq price")
            axis.plot(
                self.time,
                opt_price,
                color="0.45",
                linewidth=1.6,
                linestyle="--",
                label="Opt price",
            )
            for block in generator["blocks"]:
                axis.plot(
                    self.time,
                    self._series(block["alpha_profile"]),
                    marker="o",
                    linewidth=1.4,
                    label=f"{block['block_name']} bid",
                )
            axis.set_title(generator_name)
            axis.set_ylabel("Price / bid")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", fontsize=8, ncol=2)

        axes[-1, 0].set_xlabel("Time step")
        return self._save(fig, "poa_bid_price_comparison.png", show)

    def plot_block_dispatch_by_generator(self, show: bool = False) -> list[Path]:
        saved_paths = []
        for generator_name, generator in self.generators.items():
            blocks = generator["blocks"]
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            for block in blocks:
                label = block["block_name"]
                axes[0].plot(
                    self.time,
                    self._series(block["equilibrium_dispatch"]),
                    marker="o",
                    linewidth=1.8,
                    label=label,
                )
                axes[1].plot(
                    self.time,
                    self._series(block["optimal_dispatch"]),
                    marker="s",
                    linewidth=1.8,
                    label=label,
                )
                axes[0].plot(
                    self.time,
                    self._series(block["capacity_profile"]),
                    color="0.6",
                    linewidth=1.0,
                    linestyle=":",
                )
                axes[1].plot(
                    self.time,
                    self._series(block["capacity_profile"]),
                    color="0.6",
                    linewidth=1.0,
                    linestyle=":",
                )

            axes[0].set_title(f"{generator_name}: Equilibrium Block Dispatch")
            axes[1].set_title(f"{generator_name}: Optimal Block Dispatch")
            for axis in axes:
                axis.set_ylabel("MW")
                axis.grid(True, alpha=0.25)
                axis.legend(loc="best")
            axes[1].set_xlabel("Time step")
            filename = f"poa_block_dispatch_{generator_name}.png"
            saved_paths.append(self._save(fig, filename, show))
        return saved_paths

    def plot_generator_capacity_dispatch_and_bids(self, show: bool = False) -> list[Path]:
        eq_price = self._series(self.results["equilibrium_price_profile"])
        saved_paths = []

        for generator_name, generator in self.generators.items():
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(11, 8.5),
                sharex=True,
                gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
            )

            axes[0].plot(
                self.time,
                self._series(generator["physical_capacity_profile"]),
                color="tab:green",
                marker="s",
                linewidth=2.0,
                label="Available physical capacity",
            )
            axes[0].plot(
                self.time,
                self._series(generator["equilibrium_physical_dispatch"]),
                color="tab:red",
                marker="o",
                linewidth=2.0,
                label="Equilibrium dispatch",
            )
            axes[0].plot(
                self.time,
                self._series(generator["optimal_physical_dispatch"]),
                color="tab:blue",
                marker="^",
                linewidth=1.8,
                linestyle="--",
                label="Optimal dispatch",
            )
            axes[0].set_title(f"{generator_name}: Capacity, Dispatch, Bids, and Price")
            axes[0].set_ylabel("MW")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="best")

            for block in generator["blocks"]:
                axes[1].plot(
                    self.time,
                    self._series(block["capacity_profile"]),
                    marker="s",
                    linewidth=1.7,
                    label=f"{block['block_name']} capacity",
                )
            axes[1].set_ylabel("Block capacity (MW)")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(loc="best", ncol=3)

            for block in generator["blocks"]:
                axes[2].plot(
                    self.time,
                    self._series(block["alpha_profile"]),
                    marker="o",
                    linewidth=1.7,
                    label=f"{block['block_name']} bid",
                )
            axes[2].plot(
                self.time,
                eq_price,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Equilibrium clearing price",
            )
            axes[2].set_xlabel("Time step")
            axes[2].set_ylabel("Bid / price")
            axes[2].grid(True, alpha=0.25)
            axes[2].legend(loc="best", ncol=3)

            filename = f"poa_capacity_dispatch_bids_{generator_name}.png"
            saved_paths.append(self._save(fig, filename, show))

        return saved_paths

    def plot_clearing_price_diagnostics(self, show: bool = False) -> Path:
        block_rows = []
        for generator_name, generator in self.generators.items():
            for block in generator["blocks"]:
                block_rows.append((generator_name, block))

        block_names = [block["block_name"] for _, block in block_rows]
        bids = np.vstack([self._series(block["alpha_profile"]) for _, block in block_rows])
        dispatch = np.vstack(
            [self._series(block["equilibrium_dispatch"]) for _, block in block_rows]
        )
        capacity = np.vstack([self._series(block["capacity_profile"]) for _, block in block_rows])
        eq_price = self._series(self.results["equilibrium_price_profile"])
        demand = self._series(self.results["demand_profile"])

        max_bid = np.nanmax(bids, axis=0)
        min_bid = np.nanmin(bids, axis=0)
        outside_bid_range = (eq_price > max_bid + 1e-6) | (eq_price < min_bid - 1e-6)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        axes[0].fill_between(
            self.time,
            min_bid,
            max_bid,
            color="0.88",
            label="Bid range",
        )
        axes[0].plot(self.time, eq_price, color="tab:red", marker="o", linewidth=2.0, label="Eq price")
        axes[0].scatter(
            self.time[outside_bid_range],
            eq_price[outside_bid_range],
            color="black",
            s=55,
            zorder=4,
            label="Outside bid range",
        )
        axes[0].set_title("Clearing Price Compared with Bid Range")
        axes[0].set_ylabel("Price / bid")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        binding_upper_count = np.sum(np.isclose(dispatch, capacity, atol=1e-5), axis=0)
        zero_dispatch_count = np.sum(dispatch <= 1e-5, axis=0)
        available_count = dispatch.shape[0]
        axes[1].bar(self.time - 0.18, binding_upper_count, width=0.36, label="At capacity")
        axes[1].bar(self.time + 0.18, zero_dispatch_count, width=0.36, label="At zero dispatch")
        axes[1].set_title("Block Bound Activity in Equilibrium")
        axes[1].set_ylabel("Number of blocks")
        axes[1].set_ylim(0, available_count + 1)
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend(loc="best")

        t_star = int(np.nanargmax(np.abs(eq_price - np.clip(eq_price, min_bid, max_bid))))
        y_pos = np.arange(len(block_names))
        colors = [
            "tab:green" if dispatch[row, t_star] > 1e-5 else "0.75"
            for row in range(len(block_names))
        ]
        axes[2].barh(y_pos, capacity[:, t_star], color="0.88", label="Capacity")
        axes[2].barh(y_pos, dispatch[:, t_star], color=colors, label="Equilibrium dispatch")
        for row, bid_value in enumerate(bids[:, t_star]):
            axes[2].text(
                capacity[row, t_star] + 0.5,
                row,
                f"bid {bid_value:.2f}",
                va="center",
                fontsize=8,
            )
        axes[2].axvline(demand[t_star], color="black", linestyle="--", linewidth=1.4, label="Demand")
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(block_names)
        axes[2].invert_yaxis()
        axes[2].set_title(
            f"Dispatch and Capacity at t={t_star}, price={eq_price[t_star]:.2f}"
        )
        axes[2].set_xlabel("MW")
        axes[2].grid(True, axis="x", alpha=0.25)
        axes[2].legend(loc="lower right")

        return self._save(fig, "poa_clearing_price_diagnostics.png", show)

    def print_price_diagnostics(self) -> None:
        block_rows = [
            block
            for generator in self.generators.values()
            for block in generator["blocks"]
        ]
        bids = np.vstack([self._series(block["alpha_profile"]) for block in block_rows])
        dispatch = np.vstack([self._series(block["equilibrium_dispatch"]) for block in block_rows])
        capacity = np.vstack([self._series(block["capacity_profile"]) for block in block_rows])
        eq_price = self._series(self.results["equilibrium_price_profile"])

        print("\nPrice diagnostics")
        print("t  price      min_bid    max_bid    at_cap  at_zero")
        for t in range(len(self.time)):
            at_cap = int(np.sum(np.isclose(dispatch[:, t], capacity[:, t], atol=1e-5)))
            at_zero = int(np.sum(dispatch[:, t] <= 1e-5))
            marker = "*" if eq_price[t] > np.max(bids[:, t]) + 1e-6 else " "
            print(
                f"{t:<2}{marker} {eq_price[t]:>8.3f}  "
                f"{np.min(bids[:, t]):>8.3f}  {np.max(bids[:, t]):>8.3f}  "
                f"{at_cap:>6}  {at_zero:>7}"
            )
        print("* price is above the maximum submitted block bid")

    def validate_equilibrium_lower_level_lp(self) -> None:
        block_rows = []
        physical_generator_names = list(self.results["physical_generator_names"])
        for generator_idx, generator_name in enumerate(physical_generator_names):
            for block in self.generators[generator_name]["blocks"]:
                block_rows.append((generator_idx, block))

        num_blocks = len(block_rows)
        num_generators = len(physical_generator_names)
        horizon = len(self.time)
        variable_count = num_blocks * horizon
        flat_index = lambda block_idx, time_idx: block_idx * horizon + time_idx

        alpha = np.asarray(
            [block["alpha_profile"] for _, block in block_rows],
            dtype=float,
        )
        capacity = np.asarray(
            [block["capacity_profile"] for _, block in block_rows],
            dtype=float,
        )
        reported_dispatch = np.asarray(
            [block["equilibrium_dispatch"] for _, block in block_rows],
            dtype=float,
        )
        demand = self._series(self.results["demand_profile"])

        support = self.results.get("support_set", {})
        wind_generators = support.get("wind", {})
        static_capacity = [
            float(np.nanmax(self._series(self.generators[name]["physical_capacity_profile"])))
            for name in physical_generator_names
        ]
        p_init = np.asarray([0.5 * cap for cap in static_capacity], dtype=float)
        ramp_up = np.asarray(
            [
                50.0 if name in wind_generators else 30.0
                for name in physical_generator_names
            ],
            dtype=float,
        )
        ramp_down = ramp_up.copy()

        objective = alpha.reshape(variable_count)
        equality_matrix = []
        equality_rhs = []
        for t in range(horizon):
            row = np.zeros(variable_count)
            for block_idx in range(num_blocks):
                row[flat_index(block_idx, t)] = 1.0
            equality_matrix.append(row)
            equality_rhs.append(demand[t])

        inequality_matrix = []
        inequality_rhs = []
        for generator_idx in range(num_generators):
            owned_blocks = [
                block_idx
                for block_idx, (owner_idx, _) in enumerate(block_rows)
                if owner_idx == generator_idx
            ]
            row = np.zeros(variable_count)
            for block_idx in owned_blocks:
                row[flat_index(block_idx, 0)] = 1.0
            inequality_matrix.append(row)
            inequality_rhs.append(p_init[generator_idx] + ramp_up[generator_idx])

            row = np.zeros(variable_count)
            for block_idx in owned_blocks:
                row[flat_index(block_idx, 0)] = -1.0
            inequality_matrix.append(row)
            inequality_rhs.append(ramp_down[generator_idx] - p_init[generator_idx])

            for t in range(1, horizon):
                row = np.zeros(variable_count)
                for block_idx in owned_blocks:
                    row[flat_index(block_idx, t)] = 1.0
                    row[flat_index(block_idx, t - 1)] -= 1.0
                inequality_matrix.append(row)
                inequality_rhs.append(ramp_up[generator_idx])

                row = np.zeros(variable_count)
                for block_idx in owned_blocks:
                    row[flat_index(block_idx, t)] -= 1.0
                    row[flat_index(block_idx, t - 1)] += 1.0
                inequality_matrix.append(row)
                inequality_rhs.append(ramp_down[generator_idx])

        bounds = [
            (0.0, capacity[block_idx, t])
            for block_idx in range(num_blocks)
            for t in range(horizon)
        ]
        result = linprog(
            objective,
            A_ub=np.asarray(inequality_matrix),
            b_ub=np.asarray(inequality_rhs),
            A_eq=np.asarray(equality_matrix),
            b_eq=np.asarray(equality_rhs),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            print(f"\nLower-level LP validation failed to solve: {result.message}")
            return

        reported_cost = float(np.sum(alpha * reported_dispatch))
        optimal_cost = float(result.fun)
        gap = reported_cost - optimal_cost
        print("\nLower-level equilibrium LP validation")
        print(f"reported bid-cost dispatch: {reported_cost:.3f}")
        print(f"LP minimum bid cost:        {optimal_cost:.3f}")
        print(f"reported minus LP optimum: {gap:.3f}")
        if abs(gap) > 1e-4:
            print("WARNING: reported P_eq is not the lower-level minimum-bid dispatch.")

    def plot_all(self, show: bool = False) -> list[Path]:
        self.print_price_diagnostics()
        self.validate_equilibrium_lower_level_lp()
        saved = [
            self.plot_system_overview(show=show),
            self.plot_physical_dispatch(show=show),
            self.plot_bid_price_comparison(show=show),
            self.plot_clearing_price_diagnostics(show=show),
        ]
        saved.extend(self.plot_generator_capacity_dispatch_and_bids(show=show))
        saved.extend(self.plot_block_dispatch_by_generator(show=show))
        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize block-aware PoA optimization results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/poa_optimization_bidding_blocks_results_tightened_T6.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results_viz/figures/poa_optimization_bidding_blocks"),
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    visualizer = PoABiddingBlocksVisualizer.from_json(args.results, args.outdir)
    saved_paths = visualizer.plot_all(show=args.show)
    for path in saved_paths:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
