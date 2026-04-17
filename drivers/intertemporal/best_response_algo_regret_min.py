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
import json

from models.diagonalization.intertemporal.regret_minization.MPEC_regret_min import MPECModel
from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from models.diagonalization.intertemporal.regret_minization.utilities.diagonalization_loader import load_diagonalization

class BestResponseAlgorithmRegretMin:
    """
    Best response algorithm where each player uses a feature-based bidding
    policy (theta) optimised via regret minimisation across accumulated
    scenarios and iterations.
    """

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        feature_matrix_by_player: Dict[int, Dict[Tuple[int, int, int], List[float]]],
        features: List[str],
    ):
        """
        Parameters
        ----------
        scenarios_df : pd.DataFrame
            Base scenario set (S rows — one per demand/capacity combination).
            Bids should be initialised to marginal costs.
        costs_df : pd.DataFrame
            Static generator cost data (single row).
        players_config : list[dict]
            Player configuration list (id, controlled_generators).
        feature_matrix_by_player : dict[int, dict[tuple[int, int, int], list[float]]]
            Feature matrix for each player, keyed by player id and scenario.
        features : list[str]
            List of feature names corresponding to the feature vectors in feature_matrix_by_player.
        """
        self.scenarios_df = scenarios_df.copy().reset_index(drop=True)
        self.initial_scenarios = scenarios_df.copy().reset_index(drop=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.feature_matrix_by_player = feature_matrix_by_player
        self.features = features

        # Extract basic data for compatibility with ED model from scenarios_df
        # Auto-detect generator names from capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.generator_names = generator_names
        
        self.num_generators = len(generator_names)
        self.num_base_scenarios = len(self.initial_scenarios)
        
        # Extract generator data from scenarios_df first row and costs_df
        self.pmax_list = [scenarios_df[f"{gen}_cap"].iloc[0] for gen in generator_names]
        self.pmin_list = [0.0] * self.num_generators  # Default Pmin = 0
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in generator_names]

        self.demand = scenarios_df['demand']
        self.demand_profile = scenarios_df['demand_profile']
        self.num_time_steps = len(self.demand_profile[0]) if len(self.demand_profile) > 0 else 0

        # Load diagonalization config
        diag_config = load_diagonalization()
        self.max_iterations = int(diag_config.get("max_iterations"))
        self.conv_tolerance = float(diag_config.get("conv_tolerance"))
        self.history_window = int(diag_config.get("history_window", 0))

        # Fix P_init from the first ED run and reuse it across all later
        # accumulated scenario snapshots.
        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)
        
        # Build MPEC model (will be rebuilt via update_scenarios each iteration)
        self.mpec_model = MPECModel(
            scenarios_df=self.scenarios_df,
            initial_scenarios_df=self.initial_scenarios,
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
            players_config=self.players_config,
            p_init=self.P_init,
            feature_matrix_by_player=self.feature_matrix_by_player,
        )

        # Accumulated history: list of scenario snapshots (one S-row df per iteration)
        self.history_snapshots: List[pd.DataFrame] = []

        #[i] = iteration index, [s] = scenario index, [player_idx] = player index, [g] = generator index, [t] = time step index

        # History tracking 
        # bid_history[i][s][g][t]
        self.bid_history: List[List[List[List[float]]]] = []
        # theta_history[i][player_idx][g] — theta 
        self.theta_history: List[Dict[int, Dict[int, np.ndarray]]] = []
        # profit_history_mpec[i][player_idx] — MPEC total profit (summed across current scenarios in regret min)
        self.profit_history_mpec: List[List[float]] = []
        # profit_history_mpec_scenario[i][s][player_idx] — MPEC per-scenario profits (across current scenarios in regret min)
        self.profit_history_mpec_scenario: List[List[List[float]]] = []
        # profit_history_ed[i][player_idx] — ED total player profits (summed across current scenarios in regret min)
        self.profit_history_ed: List[List[float]] = []
        # profit_history_ed_scenario[i][s][player_idx] — ED per-scenario profits (across current scenarios in regret min)
        self.profit_history_ed_scenario: List[List[List[float]]] = []
        # dispatch_history[i][s][g][t] — dispatch for each scenario and generator
        self.dispatch_history: List[List[List[List[float]]]] = []
        # clearing_price_history[i][s][t] — clearing price for each scenario
        self.clearing_price_history: List[List[List[float]]] = []

        # Initial policy 
        self.initial_theta = self._compute_cost_theta()

    # Helpers

    def _compute_p_init_from_ed(self, scenarios_df) -> List[List[float]]:
        """Solve ED and extract first time-step dispatch as [scenario][generator]."""
        # Use neutral initial conditions: 50% of scenario capacity for every generator, because all generators can ramp more than 50% of their capacity.
        initial_dispatch = []
        for _, row in scenarios_df.iterrows():
            initial_dispatch.append([
                0.5 * float(row[f"{gen}_cap"])
                for gen in self.generator_names
            ])

        ed_for_p_init = EconomicDispatchModel(
            scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=initial_dispatch,
        )
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def _compute_cost_theta(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Build a theta vector per player that reproduces marginal-cost bidding.

        Initialization rule:
        - theta for `player_cost` feature(s) is set to the player's cost(s)
        - all other theta entries are set to 0
        """
        features = self.features
        cost_theta: Dict[int, Dict[int, np.ndarray]] = {}

        for pc in self.players_config:
            pid = pc['id']
            controlled = list(pc['controlled_generators'])
            player_features = self.feature_matrix_by_player[pid]

            if not player_features:
                cost_theta[pid] = {gen_idx: np.zeros(1, dtype=np.float64) for gen_idx in controlled}
                continue

            # Dimension equals the actual feature vector length used by the model.
            dim = len(next(iter(player_features.values())))
            theta_by_generator: Dict[int, np.ndarray] = {}

            for gen_idx in controlled:
                theta = np.zeros(dim, dtype=np.float64)
                idx = 0
                for feature_name in features:
                    if idx >= dim:
                        break

                    if feature_name == 'player_cost':
                        theta[idx] = float(self.cost_vector[gen_idx])
                    idx += 1

                theta_by_generator[gen_idx] = theta

            cost_theta[pid] = theta_by_generator

        return cost_theta

    def _apply_initial_policy(self) -> None:
        """
        Apply cost-based theta as fixed policy and record iteration 0 via ED.
        """
        num_scenarios = len(self.scenarios_df)
        num_time_steps = self.mpec_model.num_time_steps

        for player in self.players_config:
            pid = player['id']
            theta_by_generator = self.initial_theta[pid]
            phi_by_player = self.feature_matrix_by_player[pid]

            for s in range(num_scenarios):
                for gen_idx in player['controlled_generators']:
                    theta = theta_by_generator[gen_idx]
                    bid_profile = [
                        float(theta @ np.asarray(phi_by_player[(s, t, gen_idx)], dtype=np.float64))
                        for t in range(num_time_steps)
                    ]
                    gen_name = self.generator_names[gen_idx]
                    self.scenarios_df.at[s, f"{gen_name}_bid_profile"] = bid_profile
                    self.scenarios_df.at[s, f"{gen_name}_bid"] = bid_profile[0]

        iteration_bids = []
        for s in range(len(self.scenarios_df)):
            scenario_bid_matrix = [
                list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
                for gen_name in self.generator_names
            ]
            iteration_bids.append(scenario_bid_matrix)
        self.bid_history.append(iteration_bids)

        iteration_zero_theta = {
            player['id']: {
                gen_idx: theta.copy()
                for gen_idx, theta in self.initial_theta[player['id']].items()
            }
            for player in self.players_config
        }
        self.theta_history.append(iteration_zero_theta)

        all_dispatches, clearing_prices, all_player_profits, total_player_profits = self.calculate_ED()
        self.dispatch_history.append(all_dispatches)
        self.clearing_price_history.append(clearing_prices)

        self.profit_history_ed.append(total_player_profits)
        self.profit_history_ed_scenario.append(all_player_profits)

        # Iteration 0 has fixed policy bids, so ED is the relevant profit benchmark.
        self.profit_history_mpec.append(total_player_profits.copy())
        self.profit_history_mpec_scenario.append(
            [[all_player_profits[s][p] for s in range(len(all_player_profits))] for p in range(len(self.players_config))]
        )

        # Regret-min uses history snapshots to build accumulated scenarios.
        self.history_snapshots.append(self.scenarios_df.copy())

    def _build_accumulated_df(self) -> pd.DataFrame:
        """
        Concatenate the recorded scenario snapshots to form the accumulated
        DataFrame fed to the MPEC.

        If ``self.history_window > 0``, only the most recent
        *history_window* snapshots are included (sliding window) so the
        MILP size stays bounded.

        ``self.history_snapshots`` already contains the current scenario state,
        so appending ``self.scenarios_df`` here would duplicate the latest row
        block in the accumulated problem.
        """
        snapshots = self.history_snapshots
        if self.history_window > 0 and len(snapshots) > self.history_window:
            snapshots = snapshots[-self.history_window:]

        if not snapshots:
            return self.scenarios_df.copy().reset_index(drop=True)

        return pd.concat(snapshots, ignore_index=True)

    # Per-player solve

    def solve_strategic_player_problem(self, player_id: int) -> Tuple[float, List[float], Dict[int, np.ndarray]]:
        """
        Solve the regret-min MPEC for one strategic player.

        Returns
        -------
        total_profit : float
            Sum of profits across the *current* base scenarios (last S rows
            of the accumulated DataFrame).
        scenario_profits : list[float]
            Profits for the current S base scenarios.
        theta : Dict[int, np.ndarray]
            Optimal policy weights keyed by controlled generator index.
        """
        accumulated = self._build_accumulated_df()

        # Feed full history to MPEC and solve
        self.mpec_model = MPECModel(
            scenarios_df=accumulated,
            initial_scenarios_df=self.initial_scenarios,
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
            players_config=self.players_config,
            feature_matrix_by_player=self.feature_matrix_by_player,
            p_init=self.P_init,
        )
        self.mpec_model.build_model(player_id)
        self.mpec_model.solve()

        theta = self.mpec_model.get_optimal_theta()

        # Profits on ALL accumulated rows
        all_profits = self.mpec_model.get_scenario_profits()

        # Extract profits for the CURRENT iteration (last S rows)
        S = self.num_base_scenarios
        current_profits = all_profits[-S:]

        total_profit = sum(current_profits)

        return total_profit, current_profits, theta

    # ED validation for second convergence check

    def calculate_ED(self) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
        """
        Run economic dispatch on the *current* scenarios_df to validate
        MPEC bids.

        Returns (dispatches, prices, player_profits_by_scenario, total_player_profits).
        """
        S = self.num_base_scenarios
        current_scenarios = self.scenarios_df[-S:].copy().reset_index(drop=True)

        ed = EconomicDispatchModel(current_scenarios, self.costs_df, self.ramps_df, self.P_init)
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

    # Convergence helper function

    def check_convergence(self, a: float, b: float) -> bool:
        """
        Check if the algorithm has converged
        
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        return abs(a - b) <= self.conv_tolerance * abs(b) + self.conv_tolerance

    # Main algorithm loop

    def run(self) -> Dict[str, Any]:
        """Run the best-response algorithm with regret minimisation."""

        S = self.num_base_scenarios
        num_players = len(self.players_config)

        print("=== Starting Regret-Min Best Response Algorithm ===")
        print(f"Generators     : {self.generator_names}")
        print(f"Generator costs: {[f'{c:.2f}' for c in self.cost_vector]}")
        print(f"Number of time steps: {self.num_time_steps}")
        print(f"Base scenarios : {S}")
        print(f"Features       : {self.features}")
        print(f"Players        : {num_players}")
       
        # Apply initial policy and record iteration 0
        self._apply_initial_policy()
        
        init_mpec_str = ", ".join([
            f"P{i}={self.profit_history_mpec[0][i]:.1f}" for i in range(len(self.players_config))
        ])
        print("\n--- Iteration 0 (cost-based policy, fixed alpha) ---")
        print(f"  MPEC profits: {init_mpec_str}")
        
        self.iteration = 0

        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")

            iteration_profits_mpec: List[Optional[float]] = [None] * num_players
            iteration_profits_mpec_scenario: List[Optional[List[float]]] = [None] * num_players
            iteration_thetas: Dict[int, Dict[int, np.ndarray]] = {}

            for player_idx, player_config in enumerate(self.players_config):
                player_id = player_config['id']
                controlled = player_config['controlled_generators']
                print(f"  Solving for player {player_id} (controls generators {controlled})...")

                # Solve MPEC problem for this player 
                total_profit, scenario_profits, theta = self.solve_strategic_player_problem(player_id)

                self.scenarios_df = self.mpec_model.update_current_base_scenario_bids(scenarios_df=self.scenarios_df, num_base_scenarios=S, controlled_generators=controlled)

                iteration_profits_mpec[player_idx] = total_profit
                iteration_profits_mpec_scenario[player_idx] = scenario_profits
                iteration_thetas[player_id] = theta

            # Record history
            iteration_bids = []
            for s in range(S):
                scenario_bid_matrix = [
                    list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
                    for gen_name in self.generator_names
                ]
                iteration_bids.append(scenario_bid_matrix)

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
            print(f"  MPEC profits: {mpec_str}")
            # Print theta for each player
            # for pid, th in iteration_thetas.items():
            #     print(f"  theta[P{pid}] = {np.round(th, 4)}")

            # Convergence checks
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

                    # Only run ED when MPEC has converged
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
            scenario_bid_matrix = [
                list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
                for gen_name in self.generator_names
            ]
            scenario_bids.append(scenario_bid_matrix)

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
        }
    
    @staticmethod
    def _json_default_serializer(obj: Any) -> Any:
        """Convert numpy objects to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save_results(self, output_path: str) -> Path:
        """
        Save algorithm results from get_results() to a JSON file.

        Parameters
        ----------
        output_path : str
            Target file path.

        Returns
        -------
        pathlib.Path
            Path to the written file.
        """
        results = getattr(self, "results", None) or self.get_results()

        path = Path(output_path)
        if path.suffix and path.suffix.lower() != ".json":
            raise ValueError("output_path must end with .json or have no extension")
        if not path.suffix:
            path = path.with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=self._json_default_serializer)
        return path


if __name__ == "__main__":
    print("=== Testing Best Response Algorithm ===")
    
    from config.intertemporal.scenarios.scenario_generator import ScenarioManager
    from config.default_loader import load_test_case_config
    from models.diagonalization.features.feature_setup import FeatureBuilder, DEFAULT_FEATURES

    import time

    # Generate scenarios for the algorithm
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

    # Generator names from the DataFrame columns
    generator_names = [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    # Build feature vectors per player
    fb = FeatureBuilder(TEST_CASE, DEFAULT_FEATURES)

    # compute_historical_supply_curve(reference_case=TEST_CASE, plot=True)

    feature_matrix_by_player = fb.build_intertemporal_feature_matrix_by_player_from_frames(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        generator_names=generator_names,
        players_config=players_config,
        fit_normalizer=True,
    )

    fb.save_feature_normalizer_stats("results/feature_normalizer_stats.json")

    features = fb.features  # List of feature names in the order they appear in the feature vectors for each generator

    # Run algorithm
    algo = BestResponseAlgorithmRegretMin(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        feature_matrix_by_player=feature_matrix_by_player,
        features=features
    )
    
    start = time.perf_counter()
    algo.run()
    end = time.perf_counter()

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")

    stop = True