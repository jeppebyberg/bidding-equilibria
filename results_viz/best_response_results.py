"""
Intertemporal best-response results visualization.

This script reads results saved by:
	drivers/intertemporal/best_response_algo_regret_min.py

Expected default input:
	results/best_response_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class IntertemporalResultsVisualizer:
	"""Create analysis plots from intertemporal best-response JSON results."""

	def __init__(self, results: Dict[str, Any], output_dir: Path) -> None:
		self.results = results
		self.output_dir = output_dir
		self.output_dir.mkdir(parents=True, exist_ok=True)

		self.bid_history = results.get("bid_history", [])
		self.dispatch_history = results.get("dispatch_history", [])
		self.clearing_price_history = results.get("clearing_price_history", [])
		self.profit_history_mpec = results.get("profit_history_mpec", [])
		self.profit_history_ed = results.get("profit_history_ed", [])
		self.theta_history = results.get("theta_history", [])
		self.nn_policy_weights = results.get("nn_policy_weights", {})
		self.generator_costs = results.get("generator_costs", [])

		self.num_iterations, self.num_scenarios, self.num_generators, self.num_timesteps = self._infer_dims()

	@classmethod
	def from_json(cls, path: Path, output_dir: Path) -> "IntertemporalResultsVisualizer":
		with path.open("r", encoding="utf-8") as f:
			results = json.load(f)
		return cls(results=results, output_dir=output_dir)

	def _infer_dims(self) -> Tuple[int, int, int, int]:
		if not self.bid_history:
			return 0, 0, 0, 0

		num_iterations = len(self.bid_history)
		num_scenarios = len(self.bid_history[0])
		num_generators = len(self.bid_history[0][0])
		num_timesteps = len(self.bid_history[0][0][0])
		return num_iterations, num_scenarios, num_generators, num_timesteps

	def _save(self, fig: plt.Figure, name: str) -> None:
		out = self.output_dir / name
		fig.savefig(out, dpi=150, bbox_inches="tight")
		print(f"[saved] {out}")

	def print_summary(self) -> None:
		print("\n=== Intertemporal Results Summary ===")
		print(f"Iterations: {self.num_iterations}")
		print(f"Scenarios: {self.num_scenarios}")
		print(f"Generators: {self.num_generators}")
		print(f"Time steps: {self.num_timesteps}")
		print(f"Players (from profits): {len(self.profit_history_mpec[0]) if self.profit_history_mpec else 0}")
		if self.nn_policy_weights:
			print(f"NN policy weights available for {len(self.nn_policy_weights)} strategic generators")

	def print_nn_policy_weights(self) -> None:
		if not self.nn_policy_weights:
			print("No neural-network policy weights stored in the results file")
			return

		print("\n=== Neural Network Policy Weights ===")
		for gen_key, weights in self.nn_policy_weights.items():
			gamma = np.array(weights.get("gamma", []), dtype=float)
			theta = np.array(weights.get("theta", []), dtype=float)
			Gamma = np.array(weights.get("Gamma", []), dtype=float)
			output_bias = weights.get("output_bias", None)
			print(f"Generator {gen_key}:")
			print(f"  gamma shape: {gamma.shape}")
			print(f"  theta shape: {theta.shape}")
			print(f"  Gamma shape: {Gamma.shape}")
			print(f"  output_bias: {output_bias}")

	def plot_bid_evolution(self, scenario_id: int = 0, time_step: int = 0, show: bool = False) -> None:
		if self.num_iterations == 0:
			print("No bid history available")
			return
		if scenario_id < 0 or scenario_id >= self.num_scenarios:
			print(f"Invalid scenario_id {scenario_id}. Available: 0-{self.num_scenarios - 1}")
			return
		if time_step < 0 or time_step >= self.num_timesteps:
			print(f"Invalid time_step {time_step}. Available: 0-{self.num_timesteps - 1}")
			return

		iterations = np.arange(self.num_iterations)
		fig, ax = plt.subplots(figsize=(12, 7))

		for g in range(self.num_generators):
			y = [self.bid_history[i][scenario_id][g][time_step] for i in range(self.num_iterations)]
			cost_txt = ""
			if g < len(self.generator_costs):
				cost_txt = f" (cost={self.generator_costs[g]:.1f})"
			ax.plot(iterations, y, marker="o", linewidth=1.8, label=f"Gen {g}{cost_txt}")

		ax.set_xlabel("Iteration")
		ax.set_ylabel("Bid")
		ax.set_title(f"Bid Evolution - Scenario {scenario_id}, Time {time_step}")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=9)
		fig.tight_layout()
		self._save(fig, f"bid_evolution_S{scenario_id}_T{time_step}.png")
		if show:
			plt.show()
		plt.close(fig)

	def plot_final_bid_heatmap(self, scenario_id: int = 0, show: bool = False) -> None:
		if self.num_iterations == 0:
			print("No bid history available")
			return
		if scenario_id < 0 or scenario_id >= self.num_scenarios:
			print(f"Invalid scenario_id {scenario_id}. Available: 0-{self.num_scenarios - 1}")
			return

		final_bids = np.array(self.bid_history[-1][scenario_id], dtype=float)
		fig, ax = plt.subplots(figsize=(10, 5))
		im = ax.imshow(final_bids, aspect="auto", cmap="viridis")
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_label("Bid")

		ax.set_xlabel("Time step")
		ax.set_ylabel("Generator")
		ax.set_title(f"Final Bid Heatmap - Scenario {scenario_id}")
		ax.set_yticks(np.arange(self.num_generators))
		ax.set_yticklabels([f"G{g}" for g in range(self.num_generators)])
		ax.set_xticks(np.arange(self.num_timesteps))
		fig.tight_layout()
		self._save(fig, f"final_bid_heatmap_S{scenario_id}.png")
		if show:
			plt.show()
		plt.close(fig)

	def plot_final_dispatch_profiles(self, scenario_id: int = 0, show: bool = False) -> None:
		if not self.dispatch_history:
			print("No dispatch history available")
			return
		if scenario_id < 0 or scenario_id >= self.num_scenarios:
			print(f"Invalid scenario_id {scenario_id}. Available: 0-{self.num_scenarios - 1}")
			return

		final_dispatch = np.array(self.dispatch_history[-1][scenario_id], dtype=float)
		time_axis = np.arange(final_dispatch.shape[1])

		fig, ax = plt.subplots(figsize=(12, 6))
		for g in range(final_dispatch.shape[0]):
			ax.plot(time_axis, final_dispatch[g], marker="o", linewidth=1.8, label=f"Gen {g}")

		ax.set_xlabel("Time step")
		ax.set_ylabel("Dispatch")
		ax.set_title(f"Final Dispatch Profiles - Scenario {scenario_id}")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=9)
		fig.tight_layout()
		self._save(fig, f"final_dispatch_profiles_S{scenario_id}.png")
		if show:
			plt.show()
		plt.close(fig)

	def plot_final_price_heatmap(self, show: bool = False) -> None:
		if not self.clearing_price_history:
			print("No clearing price history available")
			return

		final_prices = np.array(self.clearing_price_history[-1], dtype=float)
		fig, ax = plt.subplots(figsize=(10, 5))
		im = ax.imshow(final_prices, aspect="auto", cmap="magma")
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_label("Price")

		ax.set_xlabel("Time step")
		ax.set_ylabel("Scenario")
		ax.set_title("Final Clearing Price Heatmap")
		ax.set_yticks(np.arange(final_prices.shape[0]))
		ax.set_yticklabels([f"S{s}" for s in range(final_prices.shape[0])])
		ax.set_xticks(np.arange(final_prices.shape[1]))
		fig.tight_layout()
		self._save(fig, "final_clearing_price_heatmap.png")
		if show:
			plt.show()
		plt.close(fig)

	def plot_merit_order_curve(
		self,
		scenario_id: int = 0,
		time_step: int = 0,
		iteration: Optional[int] = None,
		show: bool = False,
	) -> None:
		"""
		Plot merit-order curve for one scenario and one time step.

		Uses dispatched quantity as segment width and bid as segment height.
		Optionally overlays a cost-based reference curve (same dispatched widths,
		sorted by generator marginal costs).
		"""
		if self.num_iterations == 0:
			print("No bid history available")
			return
		if not self.dispatch_history:
			print("No dispatch history available")
			return
		if scenario_id < 0 or scenario_id >= self.num_scenarios:
			print(f"Invalid scenario_id {scenario_id}. Available: 0-{self.num_scenarios - 1}")
			return
		if time_step < 0 or time_step >= self.num_timesteps:
			print(f"Invalid time_step {time_step}. Available: 0-{self.num_timesteps - 1}")
			return

		it = self.num_iterations - 1 if iteration is None else iteration
		if it < 0 or it >= self.num_iterations:
			print(f"Invalid iteration {it}. Available: 0-{self.num_iterations - 1}")
			return

		bids = np.array(self.bid_history[it][scenario_id], dtype=float)[:, time_step]

		dispatch_matrix = np.array(self.dispatch_history[-1][scenario_id], dtype=float)
		# Intertemporal dispatch is typically [time_step, generator], but handle both orientations.
		if (
			dispatch_matrix.ndim == 2
			and dispatch_matrix.shape[0] == self.num_timesteps
			and dispatch_matrix.shape[1] == self.num_generators
		):
			dispatch = dispatch_matrix[time_step, :]
		elif (
			dispatch_matrix.ndim == 2
			and dispatch_matrix.shape[0] == self.num_generators
			and dispatch_matrix.shape[1] == self.num_timesteps
		):
			dispatch = dispatch_matrix[:, time_step]
		else:
			raise ValueError(
				"Unexpected dispatch shape for merit-order plotting: "
				f"{dispatch_matrix.shape}. Expected (T, G) or (G, T)."
			)

		order_by_bid = np.argsort(bids)
		fig, ax = plt.subplots(figsize=(12, 7))

		x_bid = [0.0]
		y_bid = [0.0]
		cum_bid = 0.0
		for g in order_by_bid:
			q = max(float(dispatch[g]), 0.0)
			if q <= 1e-9:
				continue
			p = float(bids[g])
			x_bid.extend([cum_bid, cum_bid + q])
			y_bid.extend([p, p])
			cum_bid += q

		if cum_bid <= 1e-9:
			print("No positive dispatch found for this scenario/time step")
			plt.close(fig)
			return

		ax.step(x_bid, y_bid, where="post", linewidth=2.4, color="tab:blue", label="Strategic merit order (by bid)")

		if self.generator_costs and len(self.generator_costs) >= self.num_generators:
			costs = np.array(self.generator_costs, dtype=float)
			order_by_cost = np.argsort(costs)
			x_cost = [0.0]
			y_cost = [0.0]
			cum_cost = 0.0
			for g in order_by_cost:
				q = max(float(dispatch[g]), 0.0)
				if q <= 1e-9:
					continue
				p = float(costs[g])
				x_cost.extend([cum_cost, cum_cost + q])
				y_cost.extend([p, p])
				cum_cost += q

			ax.step(
				x_cost,
				y_cost,
				linewidth=2.0,
				color="tab:orange",
				linestyle="--",
				label="Cost reference (by marginal cost)",
			)

		demand = float(np.sum(dispatch))
		ax.axvline(demand, color="tab:green", linewidth=2.0, alpha=0.85, label=f"Demand served ({demand:.1f})")

		if self.clearing_price_history:
			price = float(self.clearing_price_history[-1][scenario_id][time_step])
			ax.axhline(
				price,
				color="tab:red",
				linewidth=1.8,
				alpha=0.8,
				linestyle=":",
				label=f"Clearing price ({price:.2f})",
			)

		ax.set_xlabel("Cumulative dispatched quantity")
		ax.set_ylabel("Price / Bid")
		ax.set_title(f"Merit Order Curve - Scenario {scenario_id}, Time {time_step}, Iteration {it}")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=9)
		fig.tight_layout()
		self._save(fig, f"merit_order_S{scenario_id}_T{time_step}_I{it}.png")
		if show:
			plt.show()
		plt.close(fig)

	def plot_profit_evolution(self, show: bool = False) -> None:
		if not self.profit_history_mpec:
			print("No MPEC profit history available")
			return

		mpec = np.array(self.profit_history_mpec, dtype=float)
		iterations = np.arange(mpec.shape[0])

		fig, ax = plt.subplots(figsize=(12, 6))
		for p in range(mpec.shape[1]):
			ax.plot(iterations, mpec[:, p], marker="o", linewidth=1.8, label=f"Player {p} MPEC")

		if self.profit_history_ed:
			ed = np.array(self.profit_history_ed, dtype=float)
			ed_iters = np.arange(ed.shape[0])
			for p in range(ed.shape[1]):
				ax.plot(
					ed_iters,
					ed[:, p],
					marker="s",
					linestyle="--",
					linewidth=1.3,
					alpha=0.8,
					label=f"Player {p} ED",
				)

		ax.set_xlabel("Iteration")
		ax.set_ylabel("Profit")
		ax.set_title("Player Profit Evolution")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=8)
		fig.tight_layout()
		self._save(fig, "profit_evolution.png")
		if show:
			plt.show()
		plt.close(fig)

	def plot_theta_norm_evolution(self, show: bool = False) -> None:
		if not self.theta_history:
			print("No theta history available")
			return

		player_ids = sorted(list(self.theta_history[0].keys()), key=lambda x: int(x))
		iterations = np.arange(len(self.theta_history))
		theta_norms: Dict[str, List[float]] = {str(pid): [] for pid in player_ids}

		for it in self.theta_history:
			for pid in player_ids:
				pid_key = str(pid)
				player_theta = it.get(pid_key, it.get(pid, {}))
				if not player_theta:
					theta_norms[pid_key].append(0.0)
					continue

				norms = []
				for vec in player_theta.values():
					arr = np.array(vec, dtype=float)
					norms.append(float(np.linalg.norm(arr)))
				theta_norms[pid_key].append(float(np.mean(norms)) if norms else 0.0)

		fig, ax = plt.subplots(figsize=(12, 6))
		for pid_key in sorted(theta_norms.keys(), key=lambda x: int(x)):
			ax.plot(iterations, theta_norms[pid_key], marker="o", linewidth=1.8, label=f"Player {pid_key}")

		ax.set_xlabel("Iteration")
		ax.set_ylabel("Average theta L2 norm")
		ax.set_title("Policy Parameter Magnitude Evolution")
		ax.grid(True, alpha=0.3)
		ax.legend(loc="best", fontsize=9)
		fig.tight_layout()
		self._save(fig, "theta_norm_evolution.png")
		if show:
			plt.show()
		plt.close(fig)

	def run_default_analysis(self, scenario_id: int, time_step: int, show: bool = False) -> None:
		self.print_summary()
		self.print_nn_policy_weights()
		self.plot_bid_evolution(scenario_id=scenario_id, time_step=time_step, show=show)
		self.plot_merit_order_curve(scenario_id=scenario_id, time_step=time_step, show=show)
		self.plot_final_bid_heatmap(scenario_id=scenario_id, show=show)
		self.plot_final_dispatch_profiles(scenario_id=scenario_id, show=show)
		self.plot_final_price_heatmap(show=show)
		self.plot_profit_evolution(show=show)
		self.plot_theta_norm_evolution(show=show)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize intertemporal best-response results")
	parser.add_argument(
		"--results",
		type=Path,
		default=Path("results/best_response_results.json"),
		help="Path to saved JSON results",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=Path("results_viz/figures"),
		help="Directory for generated figures",
	)
	parser.add_argument(
		"--scenario",
		type=int,
		default=0,
		help="Scenario index used for scenario-specific plots",
	)
	parser.add_argument(
		"--time-step",
		type=int,
		default=0,
		help="Time-step index used in bid-evolution plot",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display figures interactively in addition to saving",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.results.exists():
		raise FileNotFoundError(f"Results file not found: {args.results}")

	viz = IntertemporalResultsVisualizer.from_json(path=args.results, output_dir=args.outdir)
	viz.run_default_analysis(scenario_id=args.scenario, time_step=args.time_step, show=args.show)


if __name__ == "__main__":
	main()
