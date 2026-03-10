import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.diagonalization.OneScenario.MPEC import MPECModel
from models.diagonalization.OneScenario.economic_dispatch import EconomicDispatchModel
from models.diagonalization.OneScenario.utilities.diagonalization_loader import load_diagonalization


class BestResponseAlgorithmPerScenario:
    """
    Best response algorithm that solves each scenario independently, then aggregates results.
    
    Unlike BestResponseAlgorithmMS which solves all scenarios jointly in a single MPEC,
    this class runs a separate best response loop per scenario. Each scenario converges
    independently, and the final results are aggregated across all scenarios.
    """

    def __init__(self,
                 scenarios_df: pd.DataFrame,
                 costs_df: pd.DataFrame,
                 players_config: List[Dict[str, Any]],
                 seed: int = 123):
        """
        Initialize the per-scenario best response algorithm.

        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data with demand, generator capacity, and bid columns
        costs_df : pd.DataFrame
            DataFrame containing static generator costs
        players_config : List[Dict[str, Any]]
            List of player configurations
        seed : int
            Random seed for reproducibility
        """
        self.scenarios_df = scenarios_df.copy()
        self.costs_df = costs_df
        self.players_config = players_config
        self.seed = seed

        # Auto-detect generator names from capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        self.generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.num_generators = len(self.generator_names)
        self.num_scenarios = len(scenarios_df)

        # Detect demand column
        self.demand_col = None
        for col in scenarios_df.columns:
            if any(kw in col.lower() for kw in ['demand', 'load']):
                self.demand_col = col
                break

        # Extract costs
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in self.generator_names]

        # Load algorithm parameters
        diag_config = load_diagonalization()
        self.max_iterations = int(diag_config.get("max_iterations"))
        self.conv_tolerance = float(diag_config.get("conv_tolerance"))

        # Per-scenario history: [scenario] -> list of per-iteration data
        self.scenario_bid_history = [[] for _ in range(self.num_scenarios)]         # [scenario][iteration][generator]
        self.scenario_profit_history_mpec = [[] for _ in range(self.num_scenarios)]  # [scenario][iteration][player]
        self.scenario_iterations = [0] * self.num_scenarios                          # iterations used per scenario
        self.scenario_converged = [False] * self.num_scenarios

        # Aggregated history (aligned with global iterations): [iteration][scenario][generator]
        self.bid_history = []
        self.profit_history_agent_perspective = []
        self.profit_history_agent_perspective_scenario = []
        self.profit_history_ED_perspective = []
        self.profit_history_ED_perspective_scenario = []
        self.dispatch_history = []
        self.clearing_price_history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def check_convergence(self, parameter_1: float, parameter_2: float) -> bool:
        """Check if two values are within the convergence tolerance."""
        if abs(parameter_1 - parameter_2) <= self.conv_tolerance * abs(parameter_2) + self.conv_tolerance:
            return True
        return False

    # ------------------------------------------------------------------
    # Single-scenario solve
    # ------------------------------------------------------------------

    def _run_ed_single(self, scenario_idx: int):
        """
        Run economic dispatch for a single scenario using current bids in scenarios_df.

        Parameters
        ----------
        scenario_idx : int
            Index of the scenario in scenarios_df
        """
        single_df = self.scenarios_df.iloc[[scenario_idx]].copy().reset_index(drop=True)
        ed = EconomicDispatchModel(single_df, self.costs_df)
        ed.solve()

        dispatch = ed.get_dispatch()
        clearing_price = ed.get_clearing_price()
        gen_profits = ed.get_generator_profits()

        # Aggregate by player
        ed_player_profits = []
        for pc in self.players_config:
            pp = sum(gen_profits[g] for g in pc['controlled_generators'])
            ed_player_profits.append(pp)

        return dispatch, clearing_price, ed_player_profits

    def _solve_scenario(self, scenario_idx: int) -> Dict[str, Any]:
        """
        Run the full best-response loop for a single scenario.

        Parameters
        ----------
        scenario_idx : int
            Index of the scenario in scenarios_df

        Returns
        -------
        dict
            Per-scenario results including final bids, dispatch, price, and player profits
        """
        # Read initial bids for this scenario
        bid_vector = [self.scenarios_df.iloc[scenario_idx][f"{gen}_bid"] for gen in self.generator_names]

        bid_history_s = []   # [iteration][generator]
        profit_history_s = []  # [iteration][player]
        converged = False
        iteration = 0

        print(f"\n  === Scenario {scenario_idx} ===")

        while iteration < self.max_iterations:
            print(f"    --- Iteration {iteration + 1} ---")

            profit_per_player = [None] * len(self.players_config)
            indices = list(range(len(self.players_config)))

            for player_idx in indices:
                player_config = self.players_config[player_idx]
                player_id = player_config['id']
                controlled_generators = player_config['controlled_generators']

                # Create MPEC pointing at the right scenario row
                mpec = MPECModel(
                    scenarios_df=self.scenarios_df,
                    costs_df=self.costs_df,
                    players_config=self.players_config,
                    scenario_id=scenario_idx
                )
                mpec.update_strategic_player(strategic_player_id=player_id)
                mpec.solve()

                # Extract profit
                profit_per_player[player_idx] = mpec.get_scenario_profits()[0]

                # Update bids directly with optimal values
                self.scenarios_df = mpec.update_bids_with_optimal_values(self.scenarios_df)
                for gen_id in controlled_generators:
                    bid_vector[gen_id] = self.scenarios_df.at[scenario_idx, f"{self.generator_names[gen_id]}_bid"]

            # Record bids after all players have updated
            current_bids = bid_vector.copy()
            bid_history_s.append(current_bids)
            profit_history_s.append(profit_per_player)

            bids_str = ", ".join([f"{b:.1f}" for b in current_bids])
            profits_str = ", ".join([f"P{i}={profit_per_player[i]:.1f}" for i in range(len(self.players_config))])
            print(f"      Bids: [{bids_str}]")
            print(f"      MPEC profits: {profits_str}")

            # Convergence check (MPEC profit stability)
            if iteration > 0:
                all_stable = all(
                    self.check_convergence(profit_per_player[p], profit_history_s[iteration - 1][p])
                    for p in range(len(self.players_config))
                )

                if all_stable:
                    # Run ED validation (reads bids from self.scenarios_df)
                    ed_dispatch, ed_price, ed_player_profits = self._run_ed_single(scenario_idx)

                    # Check MPEC ≈ ED
                    mpec_eq_ed = all(
                        self.check_convergence(profit_per_player[p], ed_player_profits[p])
                        for p in range(len(self.players_config))
                    )

                    if mpec_eq_ed:
                        print(f"    Scenario {scenario_idx} converged at iteration {iteration + 1}")
                        converged = True
                        break
                    else:
                        ed_str = ", ".join([f"P{i}={ed_player_profits[i]:.1f}" for i in range(len(self.players_config))])
                        print(f"      Bids stable but MPEC != ED — ED profits: {ed_str}")

            iteration += 1

        if not converged:
            print(f"    Scenario {scenario_idx}: max iterations reached without convergence")
            ed_dispatch, ed_price, ed_player_profits = self._run_ed_single(scenario_idx)

        # Store per-scenario tracking
        self.scenario_bid_history[scenario_idx] = bid_history_s
        self.scenario_profit_history_mpec[scenario_idx] = profit_history_s
        self.scenario_iterations[scenario_idx] = iteration + (0 if converged else 0)
        self.scenario_converged[scenario_idx] = converged

        return {
            "scenario_idx": scenario_idx,
            "converged": converged,
            "iterations": iteration + 1,
            "final_bids": current_bids,
            "final_dispatch": ed_dispatch,
            "final_price": ed_price,
            "final_player_profits": ed_player_profits,
            "bid_history": bid_history_s,
            "profit_history": profit_history_s,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run the best response algorithm independently for each scenario.

        Returns
        -------
        dict
            Aggregated results across all scenarios
        """
        print("=== Starting Per-Scenario Best Response Algorithm ===")
        print(f"Number of generators: {self.num_generators}")
        print(f"Number of scenarios: {self.num_scenarios}")
        print(f"Generator costs: {[f'{c:.2f}' for c in self.cost_vector]}")

        scenario_results = []
        for s in range(self.num_scenarios):
            result = self._solve_scenario(s)
            scenario_results.append(result)

        # Print summary
        print("\n=== Per-Scenario Summary ===")
        for r in scenario_results:
            s = r["scenario_idx"]
            status = "CONVERGED" if r["converged"] else "NOT CONVERGED"
            bids_str = ", ".join([f"{b:.1f}" for b in r["final_bids"]])
            profits_str = ", ".join([f"P{i}={r['final_player_profits'][i]:.1f}" for i in range(len(self.players_config))])
            print(f"  S{s}: {status} in {r['iterations']} iters | Bids: [{bids_str}] | Price: {r['final_price']:.2f} | ED profits: {profits_str}")

        self.results = self._aggregate_results(scenario_results)
        return self.results

    # ------------------------------------------------------------------
    # Results aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-scenario results into the same format as BestResponseAlgorithmMS."""
        scenario_bids = [r["final_bids"] for r in scenario_results]
        scenario_dispatches = [r["final_dispatch"] for r in scenario_results]
        scenario_prices = [r["final_price"] for r in scenario_results]
        scenario_player_profits = [r["final_player_profits"] for r in scenario_results]
        scenario_welfare = [sum(r["final_player_profits"]) for r in scenario_results]

        num_players = len(self.players_config)

        results = {
            "iterations": max(r["iterations"] for r in scenario_results),
            "scenario_iterations": [r["iterations"] for r in scenario_results],
            "scenario_converged": [r["converged"] for r in scenario_results],
            "num_scenarios": self.num_scenarios,
            "generator_costs": self.cost_vector.copy(),
            "bid_history": None,  # per-scenario histories available via scenario_bid_history
            "profit_history_agent_perspective": None,
            "dispatch_history": None,
            "clearing_price_history": None,
            "profit_history_ED_perspective": None,
            "final_scenarios_data": {
                "scenario_bids": scenario_bids,
                "scenario_dispatches": scenario_dispatches,
                "scenario_prices": scenario_prices,
                "scenario_player_profits": scenario_player_profits,
                "scenario_welfare": scenario_welfare,
            },
            "summary_stats": {
                "avg_dispatch": [
                    sum(scenario_dispatches[s][g] for s in range(self.num_scenarios)) / self.num_scenarios
                    for g in range(self.num_generators)
                ],
                "avg_price": sum(scenario_prices) / self.num_scenarios,
                "avg_player_profits": [
                    sum(scenario_player_profits[s][p] for s in range(self.num_scenarios)) / self.num_scenarios
                    for p in range(num_players)
                ],
                "avg_welfare": sum(scenario_welfare) / self.num_scenarios,
            },
        }
        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_bid_evolution(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize how bids evolve over iterations for each generator.

        Parameters
        ----------
        scenario_id : int, optional
            Scenario to plot. If None, plots one subplot per scenario.
        """
        if scenario_id is not None:
            self._plot_bid_evolution_single(scenario_id)
        else:
            cols = min(3, self.num_scenarios)
            rows = (self.num_scenarios + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

            for s in range(self.num_scenarios):
                ax = axes[s // cols][s % cols]
                history = self.scenario_bid_history[s]
                if not history:
                    continue
                iterations = list(range(len(history)))
                for g in range(self.num_generators):
                    bids_over_time = [history[it][g] for it in iterations]
                    ax.plot(iterations, bids_over_time, marker='o', linewidth=1.5,
                            label=f'Gen {g} (${self.cost_vector[g]:.1f})')
                ax.set_xlabel('Iteration', fontsize=10)
                ax.set_ylabel('Bid ($/MWh)', fontsize=10)
                ax.set_title(f'Scenario {s}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            for idx in range(self.num_scenarios, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)

            fig.suptitle('Bid Evolution Over Iterations (Per-Scenario)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

    def _plot_bid_evolution_single(self, scenario_id: int) -> None:
        """Plot bid evolution for a single scenario."""
        history = self.scenario_bid_history[scenario_id]
        if not history:
            print(f"No bid history for scenario {scenario_id}")
            return

        iterations = list(range(len(history)))
        plt.figure(figsize=(12, 8))
        for g in range(self.num_generators):
            bids_over_time = [history[it][g] for it in iterations]
            plt.plot(iterations, bids_over_time, marker='o', linewidth=2,
                     label=f'Generator {g} (Cost: ${self.cost_vector[g]:.1f})')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Bid ($/MWh)', fontsize=12)
        plt.title(f'Bid Evolution (Scenario {scenario_id})', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Per-Scenario Best Response Algorithm ===")

    from config.base_case.scenarios.scenario_generator import ScenarioManager

    scenario_manager = ScenarioManager("test_case2")
    players_config = scenario_manager.get_players_config()

    demand_scenarios = scenario_manager.generate_demand_scenarios(
        "linear",
        num_scenarios=5,
        min_factor=0.6,
        max_factor=1.0,
    )

    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        "linear",
        num_scenarios=1,
        min_factor=1.0,
        max_factor=1.0,
    )

    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios,
    )

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]

    algo = BestResponseAlgorithmPerScenario(scenarios_df, costs_df, players_config, seed=0)
    algo.run()
    results = algo.results

    # Visualize
    algo.visualize_bid_evolution()
