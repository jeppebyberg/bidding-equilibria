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

from models.diagonalization.intertemporal.regret_minization.MPEC_regret_min import MPECModel
from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from models.diagonalization.intertemporal.regret_minization.utilities.diagonalization_loader import load_diagonalization

from models.diagonalization.features.feature_setup import FeatureBuilder, DEFAULT_FEATURES

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
        ramps_df: pd.DataFrame,
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
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.feature_builder = FeatureBuilder(reference_case, feature_list or DEFAULT_FEATURES)

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

        # Fix P_init from the first ED run and reuse it across all later
        # accumulated scenario snapshots.
        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)
        self.feature_matrix_by_player = self._build_feature_matrix_by_player(self.scenarios_df, fit_normalizer=True)

        # Build MPEC model (will be rebuilt via update_scenarios each iteration)
        self.mpec_model = MPECModel(
            reference_case=reference_case,
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            players_config=self.players_config,
            feature_matrix_by_player=self.feature_matrix_by_player,
            ramps_df=self.ramps_df,
            p_init=self.P_init,
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
    def _compute_p_init_from_ed(self, scenarios_df: pd.DataFrame) -> List[List[float]]:
        """Run ED once and extract t=0 dispatch as [scenario][generator]."""
        ed_for_p_init = EconomicDispatchModel(scenarios_df, self.costs_df, self.ramps_df)
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def _build_feature_matrix_by_player(
        self,
        scenarios_df: pd.DataFrame,
        fit_normalizer: bool = False,
    ) -> Dict[int, Dict[Tuple[int, int, int], List[float]]]:
        return self.feature_builder.build_intertemporal_feature_matrix_by_player_from_frames(
            scenarios_df=scenarios_df,
            costs_df=self.costs_df,
            generator_names=self.generator_names,
            players_config=self.players_config,
            fit_normalizer=fit_normalizer,
        )

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

        Initialization rule:
        - theta for `player_cost` feature(s) is set to the player's cost(s)
        - all other theta entries are set to 0
        """
        features = self.feature_builder.features
        cost_theta: Dict[int, np.ndarray] = {}

        for pc in self.players_config:
            pid = pc['id']
            controlled = list(pc['controlled_generators'])
            player_features = self.feature_matrix_by_player[pid]

            if not player_features:
                cost_theta[pid] = np.zeros(1, dtype=np.float64)
                continue

            # Dimension equals the actual feature vector length used by the model.
            dim = len(next(iter(player_features.values())))
            theta = np.zeros(dim, dtype=np.float64)

            idx = 0
            for feature_name in features:
                if idx >= dim:
                    break

                if feature_name == 'player_cost':
                    # Expanded block when feature vectors carry one slot per controlled generator.
                    if len(controlled) > 1 and idx + len(controlled) <= dim:
                        for off, gen_idx in enumerate(controlled):
                            theta[idx + off] = float(self.cost_vector[gen_idx])
                        idx += len(controlled)
                    else:
                        # Single slot case: use mean cost (equals the generator cost for single-gen players).
                        theta[idx] = float(np.mean([self.cost_vector[g] for g in controlled]))
                        idx += 1

                elif feature_name == 'player_capacity':
                    # Keep zero-initialized, just advance pointer with matching span.
                    if len(controlled) > 1 and idx + len(controlled) <= dim:
                        idx += len(controlled)
                    else:
                        idx += 1

                else:
                    idx += 1

            cost_theta[pid] = theta

        return cost_theta

    def _apply_initial_policy(self) -> None:
        """Overwrite bids in ``scenarios_df`` using ``self.initial_theta``."""
        S = self.num_base_scenarios
        T = self.mpec_model.num_time_steps

        for pc in self.players_config:
            pid = pc['id']
            theta = self.initial_theta[pid]
            phi_by_player = self.mpec_model.feature_matrix_by_player[pid]
            for s in range(S):
                for gen_idx in pc['controlled_generators']:
                    bid_profile = [
                        float(theta @ np.asarray(phi_by_player[(s, t, gen_idx)], dtype=np.float64))
                        for t in range(T)
                    ]
                    gen_name = self.generator_names[gen_idx]
                    self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = bid_profile
                    self.scenarios_df.at[s, f"{gen_name}_bid"] = bid_profile[0]

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
        accumulated_feature_matrix = self._build_feature_matrix_by_player(accumulated)

        # Feed full history to MPEC and solve
        self.mpec_model.update_scenarios(accumulated, accumulated_feature_matrix)
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
        current_feature_matrix = self._build_feature_matrix_by_player(self.scenarios_df)
        policy_bids = self.mpec_model.get_policy_bids(
            theta,
            self.scenarios_df,
            feature_matrix_by_player=current_feature_matrix,
        )

        controlled_gens = next(
            p['controlled_generators'] for p in self.players_config if p['id'] == player_id
        )
        for s in range(S):
            for gen_idx in controlled_gens:
                gen_name = self.generator_names[gen_idx]
                bid_profile = [float(policy_bids[s][t][gen_idx]) for t in range(self.mpec_model.num_time_steps)]
                self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = bid_profile
                self.scenarios_df.at[s, f"{gen_name}_bid"] = bid_profile[0]

        return total_profit, current_profits, theta

    # ED validation for second convergence check

    def calculate_ED(self) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
        """
        Run economic dispatch on the *current* scenarios_df to validate
        MPEC bids.

        Returns (dispatches, prices, player_profits_by_scenario, total_player_profits).
        """
        ed = EconomicDispatchModel(self.scenarios_df, self.costs_df, self.ramps_df)
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
        print(f"Features      : {self.feature_builder.features}")
        print(f"Players       : {num_players}")
       
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
        }


if __name__ == "__main__":
    print("=== Testing Regret-Min Best Response Algorithm ===")

    from config.intertemporal.scenarios.scenario_generator import ScenarioManager
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
    ramps_df     = scenarios["ramps_df"]

    # Run algorithm
    algo = BestResponseAlgorithmRegretMin(
        reference_case=TEST_CASE,
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
    )
    algo.run()

    stop = True