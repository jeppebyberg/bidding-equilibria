"""
Best Response Algorithm with Regret Minimization (Policy-Based Bidding)

Instead of finding free-form bids per scenario, each player optimises a shared
policy weight vector **theta** such that bids are computed as:

    alpha[s, i] = theta^T * phi[s, i]

where *phi* is the feature vector for generator *i* in scenario *s*.

Theta is optimised over the *accumulated* history of all previous iterations
(and all base-case demand/capacity scenarios), so the policy is robust to
all market states seen so far — this is the **regret minimisation** property.

The Gauss-Seidel iteration structure is the same as ``BestResponseAlgorithmMS``:
each player solves their MPEC sequentially, seeing the latest bids of the
players solved before them in the same iteration.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from models.diagonalization.regret_minization.MPEC_regret_min import MPECModel
from models.diagonalization.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from models.diagonalization.regret_minization.utilities.diagonalization_loader import load_diagonalization

from models.diagonalization.regret_minization.feature_setup import FeatureBuilder, create_feature_builder

class BestResponseAlgorithmRegretMin:
    """
    Best response algorithm where each player uses a feature-based bidding
    policy (theta) optimised via regret minimisation across accumulated
    scenarios and iterations.
    """

    def __init__(
        self,
        reference_case: str,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        feature_list: Optional[List[str]] = None,
        results_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        reference_case : str
            Reference case name (used for feature building to load historic data similiar to test data).
        scenarios_df : pd.DataFrame
            Base scenario set (S rows — one per demand/capacity combination).
            Bids should be initialised to marginal costs.
        costs_df : pd.DataFrame
            Static generator cost data (single row).
        players_config : list[dict]
            Player configuration list (id, controlled_generators).
        feature_list : list[str], optional
            List of features to include in the feature vector.  ``None`` → default features from features.yaml.
        results_dir : str, optional
            Directory to save figures.  ``None`` → ``results/`` in project root.
        initial_theta : dict[int, np.ndarray], optional
            Initial policy weights per player (keyed by player id).
            ``None`` → cost-based default (bid = marginal cost).
        """
        self.scenarios_df = scenarios_df.copy().reset_index(drop=True)
        self.initial_scenarios = scenarios_df.copy().reset_index(drop=True)
        self.costs_df = costs_df
        self.players_config = players_config
        self.feature_builder = create_feature_builder(reference_case, feature_list)

        # Auto-detect columns
        capacity_cols = [c for c in scenarios_df.columns if c.endswith('_cap')]
        self.generator_names = [c.replace('_cap', '') for c in capacity_cols]
        self.num_generators = len(self.generator_names)

        demand_col = None
        for c in scenarios_df.columns:
            if any(kw in c.lower() for kw in ['demand', 'load']):
                demand_col = c
                break
        self.demand_col = demand_col

        self.cost_vector = [costs_df[f"{g}_cost"].iloc[0] for g in self.generator_names]
        self.pmax_list = [scenarios_df[f"{g}_cap"].iloc[0] for g in self.generator_names]
        self.num_base_scenarios = len(scenarios_df)

        # Results directory for saving figures
        if results_dir is None:
            self.results_dir = Path(__file__).resolve().parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load diagonalization config
        diag_config = load_diagonalization()
        self.max_iterations = int(diag_config.get("max_iterations"))
        self.conv_tolerance = float(diag_config.get("conv_tolerance"))
        self.history_window = int(diag_config.get("history_window", 0))

        # Build MPEC model (will be rebuilt via update_scenarios each iteration)
        self.mpec_model = MPECModel(
            reference_case=reference_case,
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            players_config=self.players_config,
            feature_builder=self.feature_builder,
        )

        # Accumulated history: list of scenario snapshots (one S-row df per iteration)
        self.history_snapshots: List[pd.DataFrame] = []

        # History tracking 
        # bid_history[t][s][g]  — bids at end of iteration t
        self.bid_history: List[List[List[float]]] = []
        # theta_history[t][player_idx] — theta after iteration t
        self.theta_history: List[Dict[int, np.ndarray]] = []
        # profit_history_mpec[t][player_idx] — MPEC total profit (summed across current scenarios)
        self.profit_history_mpec: List[List[float]] = []
        # profit_history_mpec_scenario[t][player_idx][s] — MPEC per-scenario profits
        self.profit_history_mpec_scenario: List[List[List[float]]] = []
        # profit_history_ed[t] — ED total player profits (summed across scenarios)
        self.profit_history_ed: List[List[float]] = []
        # profit_history_ed_scenario[t][player_idx][s] — ED per-scenario profits
        self.profit_history_ed_scenario: List[List[List[float]]] = []
        self.dispatch_history: List[List[List[float]]] = []
        self.clearing_price_history: List[List[float]] = []

        self.iteration = 0
        self.results: Optional[Dict[str, Any]] = None

        # Initial policy ────────────────────────────────────────────────
        self.initial_theta = self._compute_cost_theta()

    # Helpers

    def _build_accumulated_df(self) -> pd.DataFrame:
        """
        Concatenate history snapshots with the current ``scenarios_df``
        to form the accumulated DataFrame fed to the MPEC.

        If ``self.history_window > 0``, only the most recent
        *history_window* snapshots are included (sliding window) so the
        MILP size stays bounded.
        """
        snapshots = self.history_snapshots
        if self.history_window > 0 and len(snapshots) > self.history_window:
            snapshots = snapshots[-self.history_window:]
        parts = snapshots + [self.scenarios_df]
        return pd.concat(parts, ignore_index=True)

    def check_convergence(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.conv_tolerance * abs(b) + self.conv_tolerance

    def _compute_cost_theta(self) -> Dict[int, np.ndarray]:
        """
        Build a theta vector per player that reproduces marginal-cost bidding.


        For the default feature set ``[demand, demand_sq, player_cost,
        player_capacity]`` the cost-based theta is ``[0, 0, 1, 0]``.
        Works for any feature set by putting weight 1 on ``player_cost``
        (one per controlled generator) and 0 everywhere else.
        """
        features = self.mpec_model.feature_builder.features
        cost_theta: Dict[int, np.ndarray] = {}
        for pc in self.players_config:
            pid = pc['id']
            n_gens = len(pc['controlled_generators'])
            dim = self.mpec_model.feature_builder.num_features_expanded(n_gens)
            th = np.zeros(dim)
            # Set weight = 1 for each player_cost slot
            idx = 0
            for f in features:
                if f == 'player_cost':
                    th[idx:idx + n_gens] = 1.0
                    idx += n_gens
                elif f == 'player_capacity':
                    idx += n_gens
                else:
                    idx += 1
            cost_theta[pid] = th
        return cost_theta

    def _apply_initial_policy(self) -> None:
        """Overwrite bids in ``scenarios_df`` using ``self.initial_theta``."""

        fb = self.mpec_model.feature_builder
        S = self.num_base_scenarios

        for pc in self.players_config:
            pid = pc['id']
            theta = self.initial_theta[pid]
            for s in range(S):
                for gen_idx in pc['controlled_generators']:
                    obs_list = fb._extract_observations(
                        self.initial_scenarios.iloc[[s]].reset_index(drop=True),
                        self.costs_df,
                        player_generators=[gen_idx],
                        generator_names=self.generator_names,
                    )
                    phi = fb.build(obs_list[0])
                    gen_name = self.generator_names[gen_idx]
                    self.scenarios_df.at[s, f"{gen_name}_bid"] = float(theta @ phi)

        # Record initial thetas and bids so plots show the cost-based start
        self.theta_history.append(
            {pc['id']: self.initial_theta[pc['id']].copy()
             for pc in self.players_config}
        )
        bid_snapshot = []
        for s in range(S):
            bid_snapshot.append(
                [self.scenarios_df.at[s, f"{g}_bid"] for g in self.generator_names]
            )
        self.bid_history.append(bid_snapshot)

        # Include cost-based bids in the accumulated history so the MPEC
        # regret minimisation optimises over them as well.
        self.history_snapshots.append(self.scenarios_df.copy())

        print("\n--- Initial policy applied (cost-based) ---")
        for s in range(S):
            demand = self.scenarios_df.at[s, self.demand_col]
            print(f"  S{s} (D={demand:.0f}): [{', '.join(f'{b:.1f}' for b in bid_snapshot[s])}]")

    def _save_fig(self, fig, name: str) -> None:
        """Save *fig* as PNG to ``self.results_dir / name``."""
        path = self.results_dir / name
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [saved] {path}")

    # Per-player solve

    def solve_strategic_player_problem(self, player_id: int) -> Tuple[float, List[float], np.ndarray]:
        """
        Solve the regret-min MPEC for one strategic player.

        Returns
        -------
        total_profit : float
            Sum of profits across the *current* base scenarios (last S rows
            of the accumulated DataFrame).
        scenario_profits : list[float]
            Profits for the current S base scenarios.
        theta : np.ndarray
            Optimal policy weight vector.
        """
        accumulated = self._build_accumulated_df()

        # Feed full history to MPEC and solve
        self.mpec_model.update_scenarios(accumulated)
        self.mpec_model.update_strategic_player(player_id)
        self.mpec_model.solve()

        theta = self.mpec_model.get_optimal_theta()

        # Profits on ALL accumulated rows
        all_profits = self.mpec_model.get_scenario_profits()

        # Extract profits for the CURRENT iteration (last S rows)
        S = self.num_base_scenarios
        current_profits = all_profits[-S:]

        total_profit = sum(current_profits)

        # Compute policy bids for the CURRENT scenarios and update scenarios_df
        policy_bids = self.mpec_model.get_policy_bids(theta, self.scenarios_df)

        controlled_gens = next(
            p['controlled_generators'] for p in self.players_config if p['id'] == player_id
        )
        for s in range(S):
            for gen_idx in controlled_gens:
                gen_name = self.generator_names[gen_idx]
                self.scenarios_df.at[s, f"{gen_name}_bid"] = policy_bids[s][gen_idx]

        return total_profit, current_profits, theta

    # ED validation for second convergence check

    def calculate_ED(self) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
        """
        Run economic dispatch on the *current* scenarios_df to validate
        MPEC bids.

        Returns (dispatches, prices, player_profits_by_scenario, total_player_profits).
        """
        ed = EconomicDispatchModel(self.scenarios_df, self.costs_df)
        ed.solve()
        dispatches = ed.get_dispatches()
        prices = ed.get_clearing_prices()
        gen_profits = ed.get_generator_profits()  # [scenario][generator]

        player_profits_by_scenario: List[List[float]] = []
        for s_profits in gen_profits:
            player_row = []
            for pc in self.players_config:
                player_row.append(sum(s_profits[g] for g in pc['controlled_generators']))
            player_profits_by_scenario.append(player_row)

        num_players = len(self.players_config)
        total_player_profits = [
            sum(player_profits_by_scenario[s][p] for s in range(len(player_profits_by_scenario)))
            for p in range(num_players)
        ]

        return dispatches, prices, player_profits_by_scenario, total_player_profits

    # Main algorithm loop

    def run(self) -> Dict[str, Any]:
        """Run the best-response algorithm with regret minimisation."""

        S = self.num_base_scenarios
        num_players = len(self.players_config)

        print("=== Starting Regret-Min Best Response Algorithm ===")
        print(f"Generators    : {self.generator_names}")
        print(f"Generator costs: {[f'{c:.2f}' for c in self.cost_vector]}")
        print(f"Base scenarios : {S}")
        print(f"Features      : {self.mpec_model.feature_builder.features}")
        print(f"Players       : {num_players}")
        # sc = self.mpec_model.feature_builder.supply_coeffs
        # if sc:
        #     print(f"Supply curve  : price ≈ {sc['supply_intercept']:.2f} + {sc['supply_slope']:.4f} * demand")

        # Apply initial policy and record iteration 0
        self._apply_initial_policy()

        self.iteration = 0

        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")

            iteration_profits_mpec: List[Optional[float]] = [None] * num_players
            iteration_profits_mpec_scenario: List[Optional[List[float]]] = [None] * num_players
            iteration_thetas: Dict[int, np.ndarray] = {}

            # Gauss-Seidel: solve each player sequentially
            for player_idx, player_config in enumerate(self.players_config):
                player_id = player_config['id']
                controlled = player_config['controlled_generators']
                # print(f"  Player {player_id} (generators {controlled})...")

                total_profit, scenario_profits, theta = self.solve_strategic_player_problem(player_id)

                iteration_profits_mpec[player_idx] = total_profit
                iteration_profits_mpec_scenario[player_idx] = scenario_profits
                iteration_thetas[player_id] = theta

            # ── record history ────────────────────────────────────────
            # Bid snapshot: [scenario][generator]
            iteration_bids = []
            for s in range(S):
                iteration_bids.append(
                    [self.scenarios_df.at[s, f"{g}_bid"] for g in self.generator_names]
                )
            self.bid_history.append(iteration_bids)
            self.theta_history.append(iteration_thetas)
            self.profit_history_mpec.append(iteration_profits_mpec)
            self.profit_history_mpec_scenario.append(iteration_profits_mpec_scenario)

            # Snapshot current bids into history for next iteration's accumulation
            self.history_snapshots.append(self.scenarios_df.copy())

            # ── print summary ─────────────────────────────────────────
            mpec_str = ", ".join(
                f"P{pc['id']}={iteration_profits_mpec[i]:.1f}"
                for i, pc in enumerate(self.players_config)
            )
            # print(f"  MPEC profits (current scenarios): {mpec_str}")
            for s in range(S):
                bids_str = ", ".join(f"{iteration_bids[s][g]:.1f}" for g in range(self.num_generators))
                demand = self.scenarios_df.at[s, self.demand_col]
                # print(f"  S{s} (D={demand:.0f}): [{bids_str}]")

            # Print theta for each player
            for pid, th in iteration_thetas.items():
                print(f"  theta[P{pid}] = {np.round(th, 4)}")

            # ── convergence checks ────────────────────────────────────
            if self.iteration > 0:
                # Check 1: MPEC profit stability
                convergence_1 = []
                cur_profits = self.profit_history_mpec[self.iteration]
                prev_profits = self.profit_history_mpec[self.iteration - 1]
                for p_idx in range(num_players):
                    convergence_1.append(
                        self.check_convergence(cur_profits[p_idx], prev_profits[p_idx])
                    )

                if all(convergence_1):
                    print("  Check 1 PASSED (profits stable) — running ED validation...")

                    dispatches, prices, ed_player_profits, ed_total_profits = self.calculate_ED()
                    self.dispatch_history.append(dispatches)
                    self.clearing_price_history.append(prices)
                    self.profit_history_ed.append(ed_total_profits)
                    self.profit_history_ed_scenario.append(ed_player_profits)

                    ed_str = ", ".join(
                        f"P{pc['id']}={ed_total_profits[i]:.1f}"
                        for i, pc in enumerate(self.players_config)
                    )
                    print(f"  ED profits:   {ed_str}")

                    # Check 2: MPEC ≈ ED
                    convergence_2 = []
                    for p_idx in range(num_players):
                        mpec_p = cur_profits[p_idx]
                        ed_p = ed_total_profits[p_idx]
                        convergence_2.append(self.check_convergence(mpec_p, ed_p))

                    if all(convergence_2):
                        print("  Convergence achieved! (MPEC profits stable AND MPEC ~ ED)")
                        for p_idx in range(num_players):
                            mpec_p = cur_profits[p_idx]
                            ed_p = ed_total_profits[p_idx]
                            print(f"    Player {self.players_config[p_idx]['id']}: "
                                  f"MPEC={mpec_p:.2f}, ED={ed_p:.2f}, gap={mpec_p - ed_p:.2f}")
                        self.iteration += 1
                        self.results = self.get_results()
                        return self.results
                    else:
                        for p_idx in range(num_players):
                            mpec_p = cur_profits[p_idx]
                            ed_p = ed_total_profits[p_idx]
                            print(f"    Player {self.players_config[p_idx]['id']}: "
                                  f"MPEC={mpec_p:.2f}, ED={ed_p:.2f}, gap={mpec_p - ed_p:.2f}")
                        print("  Check 2 FAILED (MPEC and ED profits differ)")
                else:
                    n_fail = sum(1 for c in convergence_1 if not c)
                    print(f"  Check 1 FAILED ({n_fail}/{num_players} players still changing)")

            self.iteration += 1

        # Max iterations reached
        print("\nMaximum iterations reached without convergence.")
        dispatches, prices, ed_player_profits, ed_total_profits = self.calculate_ED()
        self.dispatch_history.append(dispatches)
        self.clearing_price_history.append(prices)
        self.profit_history_ed.append(ed_total_profits)
        self.profit_history_ed_scenario.append(ed_player_profits)
        self.results = self.get_results()
        return self.results

    # Results

    def get_results(self) -> Dict[str, Any]:
        """Compile algorithm results into a dictionary."""
        dispatches, prices, player_profits, total_player_profits = self.calculate_ED()
        scenario_welfare = [sum(player_profits[s]) for s in range(len(player_profits))]

        scenario_bids = []
        for s in range(self.num_base_scenarios):
            scenario_bids.append(
                [self.scenarios_df.at[s, f"{g}_bid"] for g in self.generator_names]
            )

        # Collect final theta per player
        final_thetas = self.theta_history[-1] if self.theta_history else {}

        return {
            "iterations": self.iteration,
            "num_scenarios": self.num_base_scenarios,
            "generator_costs": self.cost_vector.copy(),
            "bid_history": self.bid_history,
            "theta_history": self.theta_history,
            "final_thetas": final_thetas,
            "profit_history_mpec": self.profit_history_mpec,
            "profit_history_mpec_scenario": self.profit_history_mpec_scenario,
            "profit_history_ed": self.profit_history_ed,
            "profit_history_ed_scenario": self.profit_history_ed_scenario,
            "dispatch_history": self.dispatch_history,
            "clearing_price_history": self.clearing_price_history,
            "final_scenarios_data": {
                "scenario_bids": scenario_bids,
                "scenario_dispatches": dispatches,
                "scenario_prices": prices,
                "scenario_player_profits": player_profits,
                "scenario_welfare": scenario_welfare,
            },
            "summary_stats": {
                "avg_dispatch": [
                    sum(dispatches[s][g] for s in range(len(dispatches))) / len(dispatches)
                    for g in range(self.num_generators)
                ],
                "avg_price": sum(prices) / len(prices),
                "avg_player_profits": [
                    sum(player_profits[s][p] for s in range(len(player_profits))) / len(player_profits)
                    for p in range(len(self.players_config))
                ],
                "avg_welfare": sum(scenario_welfare) / len(scenario_welfare),
            },
        }

    # Visualization

    def _run_competitive_ed(self, scenario_id: Optional[int] = None):
        """
        Run ED with cost-based bidding (perfect competition).

        Parameters
        ----------
        scenario_id : int, optional
            If given, run only for that scenario row.  Otherwise use first row.

        Returns
        -------
        tuple
            (dispatch_list, clearing_price, ed_model)
        """
        sid = scenario_id if scenario_id is not None else 0
        comp_df = self.scenarios_df.iloc[[sid]].copy().reset_index(drop=True)
        for gen in self.generator_names:
            comp_df[f"{gen}_bid"] = self.costs_df[f"{gen}_cost"].iloc[0]
        ed = EconomicDispatchModel(comp_df, self.costs_df)
        ed.solve()
        return ed.get_dispatches()[0], ed.get_clearing_prices()[0], ed

    def visualize_bid_evolution(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize how bids evolve over iterations for each generator.

        Parameters
        ----------
        scenario_id : int, optional
            Scenario to plot. If ``None``, one subplot per scenario.
        """
        if not self.bid_history:
            print("No bid history available for visualization")
            return

        num_iterations = len(self.bid_history)
        num_scenarios = len(self.bid_history[0])
        iterations = list(range(num_iterations))

        if scenario_id is not None:
            if scenario_id >= num_scenarios:
                print(f"Invalid scenario_id {scenario_id}. Available: 0-{num_scenarios-1}")
                return

            plt.figure(figsize=(12, 8))
            for g in range(self.num_generators):
                bids = [self.bid_history[t][scenario_id][g] for t in range(num_iterations)]
                plt.plot(iterations, bids, marker='o', linewidth=2,
                         label=f'Gen {g} (Cost: ${self.cost_vector[g]:.1f})')

            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Bid ($/MWh)', fontsize=12)
            plt.title(f'Bid Evolution Over Iterations (Scenario {scenario_id})',
                      fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            self._save_fig(plt.gcf(), f'bid_evolution_S{scenario_id}.png')
            plt.show()

            # Summary table
            print(f"\n=== Bid Evolution Summary (Scenario {scenario_id}) ===")
            header = "Iter " + " ".join([f"{'Gen '+str(g):>8}" for g in range(self.num_generators)])
            print(header)
            print("-" * len(header))
            for t in range(num_iterations):
                bid_str = " ".join(f"{self.bid_history[t][scenario_id][g]:8.2f}"
                                    for g in range(self.num_generators))
                print(f"{t:<4} {bid_str}")
        else:
            cols = min(3, num_scenarios)
            rows = (num_scenarios + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)

            for s in range(num_scenarios):
                ax = axes[s // cols][s % cols]
                for g in range(self.num_generators):
                    bids = [self.bid_history[t][s][g] for t in range(num_iterations)]
                    ax.plot(iterations, bids, marker='o', linewidth=1.5,
                            label=f'Gen {g} (${self.cost_vector[g]:.1f})')
                ax.set_xlabel('Iteration', fontsize=10)
                ax.set_ylabel('Bid ($/MWh)', fontsize=10)
                ax.set_title(f'Scenario {s}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            for idx in range(num_scenarios, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)

            fig.suptitle('Bid Evolution Over Iterations (All Scenarios)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            self._save_fig(fig, 'bid_evolution_all.png')
            plt.show()

    def visualize_theta_evolution(self) -> None:
        """
        Visualize theta (policy weight) evolution over iterations for each player.
        """
        if not self.theta_history:
            print("No theta history available for visualization")
            return

        num_iterations = len(self.theta_history)
        iterations = list(range(num_iterations))
        feature_names = self.mpec_model.feature_builder.features

        num_players = len(self.players_config)
        cols = min(3, num_players)
        rows = (num_players + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)

        for p_idx, pc in enumerate(self.players_config):
            pid = pc['id']
            ax = axes[p_idx // cols][p_idx % cols]

            # Collect theta for this player across iterations
            for k, fname in enumerate(feature_names):
                vals = []
                for t in range(num_iterations):
                    th = self.theta_history[t].get(pid)
                    vals.append(th[k] if th is not None else 0.0)
                ax.plot(iterations, vals, marker='o', linewidth=1.5, label=fname)

            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Theta value', fontsize=10)
            ax.set_title(f'Player {pid} (gens {pc["controlled_generators"]})',
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for idx in range(num_players, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle('Policy Weight (Theta) Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'theta_evolution.png')
        plt.show()

    def visualize_final_theta(self) -> None:
        feature_names = self.mpec_model.feature_builder.features
        final_thetas = self.results['final_thetas']

        num_features = len(feature_names)
        num_players = len(self.players_config)

        cols = min(3, num_players)
        rows = (num_players + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)

        for p_idx, pc in enumerate(self.players_config):
            pid = pc['id']
            theta = final_thetas.get(pid)
            ax = axes[p_idx // cols][p_idx % cols]
            if theta is not None:
                ax.bar(np.arange(num_features), theta[:num_features], color='skyblue', alpha=0.8)
            ax.set_xticks(np.arange(num_features))
            ax.set_xticklabels(feature_names, rotation=30, ha='right')
            ax.set_xlabel('Feature', fontsize=12)
            ax.set_ylabel('Theta Weight', fontsize=12)
            ax.set_title(f'Player {pid} Final Policy Weights', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        for idx in range(num_players, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle('Final Policy Weights (Theta)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'final_theta.png')
        plt.show()

    def visualize_supply_demand_curve(self, scenario_id: int = 0) -> None:
        """
        Visualize the supply-demand curve with market clearing point.

        Parameters
        ----------
        scenario_id : int
            Scenario to visualize (default: 0).
        """
        if not self.results:
            print("No results available for visualization")
            return

        if scenario_id >= self.results['num_scenarios']:
            print(f"Invalid scenario_id {scenario_id}. "
                  f"Available: 0-{self.results['num_scenarios']-1}")
            return

        sd = self.results['final_scenarios_data']
        bids = sd['scenario_bids'][scenario_id]
        dispatch = sd['scenario_dispatches'][scenario_id]
        price = sd['scenario_prices'][scenario_id]
        scenario_pmax = [self.scenarios_df.at[scenario_id, f"{g}_cap"]
                         for g in self.generator_names]
        demand_level = self.scenarios_df.at[scenario_id, self.demand_col]

        gen_data = sorted(
            [(i, bids[i], scenario_pmax[i], dispatch[i]) for i in range(self.num_generators)],
            key=lambda x: x[1],
        )

        # Build supply curve
        cum = 0
        sq, sp = [0], [0]
        for _, bid, pmax, _ in gen_data:
            sq.append(cum); sp.append(bid)
            cum += pmax
            sq.append(cum); sp.append(bid)

        plt.figure(figsize=(12, 8))
        plt.step(sq, sp, where='post', linewidth=2.5, color='blue',
                 label='Supply Curve', alpha=0.8)
        max_price = max(sp) * 1.1
        plt.axvline(x=demand_level, color='red', linewidth=2.5,
                    label=f'Demand ({demand_level:.0f} MW)', alpha=0.8)
        plt.scatter([demand_level], [price], color='green', s=150, zorder=5,
                    label=f'Clearing\n(${price:.2f}/MWh)')

        # Annotations
        cum_d = 0
        for gid, bid, pmax, disp in gen_data:
            if disp > 0.1:
                plt.annotate(f'Gen {gid}\n${bid:.1f}/MWh\n{disp:.0f} MW',
                             xy=(cum_d + disp/2, bid),
                             xytext=(cum_d + disp/2, bid + max_price*0.05),
                             ha='center', va='bottom', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='yellow', alpha=0.7),
                             arrowprops=dict(arrowstyle='->', color='black',
                                             alpha=0.5))
            cum_d += disp

        plt.xlabel('Quantity (MW)', fontsize=12)
        plt.ylabel('Price ($/MWh)', fontsize=12)
        plt.title(f'Supply-Demand Curve (Scenario {scenario_id})',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(cum, demand_level) * 1.1)
        plt.ylim(0, max_price)
        plt.tight_layout()
        self._save_fig(plt.gcf(), f'supply_demand_S{scenario_id}.png')
        plt.show()

        # Print dispatch summary
        print(f"\n=== Market Dispatch Summary (Scenario {scenario_id}) ===")
        print(f"{'Gen ID':<6} {'Bid':<8} {'Capacity':<8} {'Dispatch':<8} {'Status':<10}")
        print("-" * 50)
        total_d = 0
        for gid, bid, pmax, disp in gen_data:
            status = "Dispatched" if disp > 0.1 else "Not Used"
            print(f"{gid:<6} ${bid:<7.2f} {pmax:<8.0f} {disp:<8.0f} {status:<10}")
            total_d += disp
        print("-" * 50)
        print(f"Total Dispatch: {total_d:.0f} MW")
        print(f"Total Demand: {demand_level:.0f} MW")
        print(f"Market Clearing Price: ${price:.2f}/MWh")

    def visualize_agent_profits(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize agent profit evolution and compare with perfect competition.

        Parameters
        ----------
        scenario_id : int, optional
            Scenario for profit comparison.  ``None`` → average across scenarios.
        """
        if not self.results:
            print("No results available for visualization")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: profit evolution over iterations
        if len(self.profit_history_mpec) > 1:
            iterations = list(range(len(self.profit_history_mpec)))
            for p_idx, pc in enumerate(self.players_config):
                profits = [self.profit_history_mpec[t][p_idx]
                           for t in range(len(self.profit_history_mpec))]
                ax1.plot(iterations, profits, marker='o', linewidth=2,
                         label=f'Player {pc["id"]} (gens {pc["controlled_generators"]})')
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Profit ($)', fontsize=12)
            ax1.set_title('MPEC Profit Evolution Over Iterations',
                          fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12)

        # Right: final equilibrium vs perfect competition
        if scenario_id is not None and scenario_id < self.results['num_scenarios']:
            final_profits = self.results['final_scenarios_data']['scenario_player_profits'][scenario_id]
            title_suffix = f" (Scenario {scenario_id})"
        else:
            final_profits = self.results['summary_stats']['avg_player_profits']
            title_suffix = " (Average Across Scenarios)"

        # Competitive benchmark (average across scenarios)
        comp_profits = []
        for s in range(self.num_base_scenarios):
            _, _, comp_ed = self._run_competitive_ed(s)
            comp_gen_profits = comp_ed.get_generator_profits()[0]
            for p_idx, pc in enumerate(self.players_config):
                player_p = sum(comp_gen_profits[g] for g in pc['controlled_generators'])
                if s == 0:
                    comp_profits.append(player_p)
                else:
                    comp_profits[p_idx] += player_p
        comp_profits = [p / self.num_base_scenarios for p in comp_profits]

        num_players = len(self.players_config)
        x_pos = np.arange(num_players)
        width = 0.35

        ax2.bar(x_pos - width/2, final_profits, width,
                label='Equilibrium Profits', color='blue', alpha=0.7)
        ax2.bar(x_pos + width/2, comp_profits, width,
                label='Perfect Competition', color='red', alpha=0.7)

        for i, (eq, cp) in enumerate(zip(final_profits, comp_profits)):
            ax2.annotate(f'${eq:.0f}', xy=(i - width/2, eq),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9)
            ax2.annotate(f'${cp:.0f}', xy=(i + width/2, cp),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=9)

        ax2.set_xlabel('Player', fontsize=12)
        ax2.set_ylabel('Profit ($)', fontsize=12)
        ax2.set_title(f'Equilibrium vs Perfect Competition{title_suffix}',
                      fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'P{pc["id"]}' for pc in self.players_config])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_fig(fig, 'agent_profits.png')
        plt.show()

        # Detailed profit analysis table
        total_eq = sum(final_profits)
        total_comp = sum(comp_profits)
        print(f"\n=== Profit Analysis{title_suffix} ===")
        print(f"{'Player':<8} {'Gens':<15} {'Equilibrium':<12} {'Competitive':<12} {'Diff':<12}")
        print("-" * 60)
        for p_idx, pc in enumerate(self.players_config):
            eq = final_profits[p_idx]
            cp = comp_profits[p_idx]
            print(f"{pc['id']:<8} {str(pc['controlled_generators']):<15} "
                  f"${eq:<11.0f} ${cp:<11.0f} ${eq - cp:<+11.0f}")
        print("-" * 60)
        print(f"Total Welfare — Equilibrium: ${total_eq:.0f}")
        print(f"Total Welfare — Competitive: ${total_comp:.0f}")
        pct = (total_comp - total_eq) / total_comp * 100 if total_comp else 0
        print(f"Welfare Loss: ${total_comp - total_eq:.0f} ({pct:.1f}%)")

    def _compute_merit_order_data(self, scenario_id: int) -> Dict[str, Any]:
        """Compute merit-order data for a single scenario (competitive & strategic)."""
        sd = self.results['final_scenarios_data']
        strategic_bids = sd['scenario_bids'][scenario_id]
        strategic_dispatch = sd['scenario_dispatches'][scenario_id]
        strategic_price = sd['scenario_prices'][scenario_id]

        demand_level = self.scenarios_df.at[scenario_id, self.demand_col]
        scenario_pmax = [self.scenarios_df.at[scenario_id, f"{g}_cap"]
                         for g in self.generator_names]

        # Competitive dispatch for this scenario
        comp_df = self.scenarios_df.iloc[[scenario_id]].copy().reset_index(drop=True)
        for gen in self.generator_names:
            comp_df[f"{gen}_bid"] = self.costs_df[f"{gen}_cost"].iloc[0]
        comp_ed = EconomicDispatchModel(comp_df, self.costs_df)
        comp_ed.solve()
        comp_dispatch = comp_ed.get_dispatches()[0]
        comp_price = comp_ed.get_clearing_prices()[0]

        comp_gen_data = sorted(
            [(i, self.cost_vector[i], scenario_pmax[i], comp_dispatch[i])
             for i in range(self.num_generators)],
            key=lambda x: x[1],
        )
        strategic_gen_data = sorted(
            [(i, strategic_bids[i], scenario_pmax[i], strategic_dispatch[i])
             for i in range(self.num_generators)],
            key=lambda x: x[1],
        )

        def _build_curve(gen_data):
            cum = 0; sq, sp = [0], [0]
            for _, price, pmax, _ in gen_data:
                sq.append(cum); sp.append(price)
                cum += pmax
                sq.append(cum); sp.append(price)
            return sq, sp, cum

        sq_c, sp_c, cum_c = _build_curve(comp_gen_data)
        sq_s, sp_s, cum_s = _build_curve(strategic_gen_data)

        return {
            'demand_level': demand_level,
            'comp_price': comp_price,
            'strategic_price': strategic_price,
            'comp_gen_data': comp_gen_data,
            'strategic_gen_data': strategic_gen_data,
            'supply_quantities_comp': sq_c,
            'supply_prices_comp': sp_c,
            'supply_quantities_strategic': sq_s,
            'supply_prices_strategic': sp_s,
            'cumulative_capacity_comp': cum_c,
            'cumulative_capacity_strategic': cum_s,
        }

    def _plot_merit_order_on_ax(self, ax, data: Dict[str, Any],
                                scenario_id: int, detailed: bool = True) -> None:
        """Plot merit-order comparison on a given axes."""
        lw = 3 if detailed else 2
        fs = 12 if detailed else 8
        ms = 200 if detailed else 100

        ax.step(data['supply_quantities_comp'], data['supply_prices_comp'],
                where='post', linewidth=lw, color='blue',
                label='Competitive (Costs)', alpha=0.8)
        ax.step(data['supply_quantities_strategic'], data['supply_prices_strategic'],
                where='post', linewidth=lw, color='red',
                label='Strategic (Bids)', alpha=0.8, linestyle='--')

        max_price = max(
            max(data['supply_prices_comp']),
            max(data['supply_prices_strategic']),
        ) * 1.1
        ax.axvline(x=data['demand_level'], color='green', linewidth=lw,
                   label=f'Demand ({data["demand_level"]:.0f} MW)', alpha=0.8)
        ax.scatter([data['demand_level']], [data['comp_price']], color='blue',
                   s=ms, marker='o', zorder=5, edgecolor='black',
                   label=f'Comp: ${data["comp_price"]:.2f}/MWh')
        ax.scatter([data['demand_level']], [data['strategic_price']], color='red',
                   s=ms, marker='s', zorder=5, edgecolor='black',
                   label=f'Strat: ${data["strategic_price"]:.2f}/MWh')

        if detailed:
            cum = 0
            for cnt, (gid, cost, pmax, disp) in enumerate(data['comp_gen_data']):
                if disp > 0.1:
                    voff = max_price * (0.12 + 0.04 * (cnt % 3))
                    ax.annotate(
                        f'Gen {gid}\nCost: ${cost:.1f}',
                        xy=(cum + disp/2, cost),
                        xytext=(cum + disp/2, cost + voff),
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightblue', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='blue',
                                        alpha=0.7, lw=1.5),
                    )
                cum += disp

            cum = 0
            for cnt, (gid, bid, pmax, disp) in enumerate(data['strategic_gen_data']):
                if disp > 0.1 and abs(bid - self.cost_vector[gid]) > 1.0:
                    voff = max_price * (0.12 + 0.04 * (cnt % 3))
                    ax.annotate(
                        f'Gen {gid}\nBid: ${bid:.1f}',
                        xy=(cum + disp/2, bid),
                        xytext=(cum + disp/2, bid - voff),
                        ha='center', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightcoral', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        alpha=0.7, lw=1.5),
                    )
                cum += disp

        ax.set_xlabel('Cumulative Capacity (MW)', fontsize=fs)
        ax.set_ylabel('Price ($/MWh)', fontsize=fs)
        ax.set_title(f'Merit Order Comparison (Scenario {scenario_id})',
                     fontsize=fs + 4, fontweight='bold')
        ax.legend(loc='upper left', fontsize=fs - 2)
        ax.grid(True, alpha=0.3)
        max_cap = max(data['cumulative_capacity_comp'],
                      data['cumulative_capacity_strategic'])
        ax.set_xlim(0, max(max_cap, data['demand_level']) * 1.1)
        ax.set_ylim(0, max_price)

    def visualize_merit_order_comparison(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize merit-order curves (competitive vs strategic).

        Parameters
        ----------
        scenario_id : int, optional
            Scenario to visualize.  ``None`` → one subplot per scenario.
        """
        if not self.results:
            print("No results available for merit order comparison")
            return

        num_scenarios = self.results['num_scenarios']

        if scenario_id is not None:
            if scenario_id >= num_scenarios:
                print(f"Invalid scenario_id {scenario_id}. Available: 0-{num_scenarios-1}")
                return
            data = self._compute_merit_order_data(scenario_id)
            fig, ax = plt.subplots(figsize=(14, 10))
            self._plot_merit_order_on_ax(ax, data, scenario_id, detailed=True)
            plt.tight_layout()
            self._save_fig(fig, f'merit_order_S{scenario_id}.png')
            plt.show()
            self._print_merit_order_summary(data, scenario_id)
        else:
            cols = min(3, num_scenarios)
            rows = (num_scenarios + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows), squeeze=False)
            for s in range(num_scenarios):
                ax = axes[s // cols][s % cols]
                data = self._compute_merit_order_data(s)
                self._plot_merit_order_on_ax(ax, data, s, detailed=False)
            for idx in range(num_scenarios, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)
            fig.suptitle('Merit Order: Competitive vs Strategic (All Scenarios)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            self._save_fig(fig, 'merit_order_all.png')
            plt.show()

    def _print_merit_order_summary(self, data: Dict[str, Any], scenario_id: int) -> None:
        """Print merit-order comparison table for one scenario."""
        print(f"\n=== Merit Order Comparison (Scenario {scenario_id}) ===")
        print(f"Demand: {data['demand_level']:.0f} MW")

        print("\nCompetitive Merit Order (by Cost):")
        print(f"{'Rank':<4} {'Gen':<4} {'Cost':<8} {'Cap':<8} {'Dispatch':<8}")
        print("-" * 40)
        for rank, (gid, cost, pmax, disp) in enumerate(data['comp_gen_data'], 1):
            print(f"{rank:<4} {gid:<4} ${cost:<7.2f} {pmax:<8.0f} {disp:<8.0f}")

        print(f"\nStrategic Merit Order (by Bid):")
        print(f"{'Rank':<4} {'Gen':<4} {'Bid':<8} {'Cap':<8} {'Dispatch':<8} {'Cost':<8}")
        print("-" * 50)
        for rank, (gid, bid, pmax, disp) in enumerate(data['strategic_gen_data'], 1):
            print(f"{rank:<4} {gid:<4} ${bid:<7.2f} {pmax:<8.0f} {disp:<8.0f} ${self.cost_vector[gid]:<7.2f}")

        cp = data['comp_price']
        sp = data['strategic_price']
        print(f"\nCompetitive price : ${cp:.2f}/MWh")
        print(f"Strategic price   : ${sp:.2f}/MWh")
        pct = ((sp / cp - 1) * 100) if cp else 0
        print(f"Price increase    : ${sp - cp:.2f}/MWh ({pct:+.1f}%)")

        comp_order = [gid for gid, _, _, d in data['comp_gen_data'] if d > 0.1]
        strat_order = [gid for gid, _, _, d in data['strategic_gen_data'] if d > 0.1]
        if comp_order != strat_order:
            print("Merit order CHANGED due to strategic bidding!")
            print(f"  Competitive: {comp_order}")
            print(f"  Strategic  : {strat_order}")
        else:
            print("Merit order UNCHANGED (only prices affected)")

    # Printing summary

    def print_summary(self) -> None:
        """Print a concise summary of the final results."""
        if self.results is None:
            print("No results — call run() first.")
            return

        r = self.results
        print(f"\n{'='*60}")
        print(f"  Regret-Min Best Response — Summary")
        print(f"{'='*60}")
        print(f"Iterations        : {r['iterations']}")
        print(f"Base scenarios    : {r['num_scenarios']}")
        print(f"Accumulated rows  : {len(self.history_snapshots) * self.num_base_scenarios}")

        print(f"\nFinal bids per scenario:")
        for s in range(r['num_scenarios']):
            bids_str = ", ".join(f"{r['final_scenarios_data']['scenario_bids'][s][g]:.1f}"
                                for g in range(self.num_generators))
            demand = self.scenarios_df.at[s, self.demand_col]
            print(f"  S{s} (D={demand:.0f}): [{bids_str}]")

        print(f"\nFinal theta per player:")
        for pid, th in r['final_thetas'].items():
            print(f"  Player {pid}: {np.round(th, 4)}")

        stats = r['summary_stats']
        print(f"\nAvg clearing price : ${stats['avg_price']:.2f}/MWh")
        print(f"Avg player profits : {[f'{p:.1f}' for p in stats['avg_player_profits']]}")
        print(f"Avg welfare        : ${stats['avg_welfare']:.1f}")

def load_config(path: str = "config/defaults.yaml") -> dict:
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    print("=== Testing Regret-Min Best Response Algorithm ===")

    from config.base_case.scenarios.scenario_generator import ScenarioManager
    from config.default_loader import load_test_case_config

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

    # Run algorithm
    algo = BestResponseAlgorithmRegretMin(
        reference_case=TEST_CASE,
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        players_config=players_config,
    )
    algo.run()
    algo.print_summary()

    # Visualisations
    algo.visualize_bid_evolution()
    algo.visualize_theta_evolution()
    algo.visualize_merit_order_comparison()
    algo.visualize_final_theta()

    stop = True