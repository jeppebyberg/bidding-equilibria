import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt

from models.diagonalization.intertemporal.MultipleScenarios.MPEC_MS import MPECModel
from models.diagonalization.intertemporal.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from models.diagonalization.intertemporal.MultipleScenarios.utilities.diagonalization_loader import load_diagonalization

class BestResponseAlgorithmMS:
    """
    Best response algorithm for finding bidding equilibrium using MPEC and economic dispatch models
    """
    
    def __init__(self,
                 scenarios_df,
                 costs_df,
                 ramps_df,
                 players_config, 
                 feature_matrix_by_player,
                 seed: int = 123
                 ):
        """
        Initialize the best response algorithm
        
        Parameters
        ----------
        scenarios_df : pd.DataFrame
            DataFrame containing scenario data with demand, generator capacity, and bid columns
        costs_df : pd.DataFrame
            DataFrame containing static generator costs
        players_config : List[Dict[str, Any]]
            List of player configurations
        """
        
        self.scenarios_df = scenarios_df
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.feature_matrix_by_player = feature_matrix_by_player
        
        self.seed = seed

        # Extract basic data for compatibility with ED model from scenarios_df
        # Auto-detect generator names from capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        generator_names = [col.replace('_cap', '') for col in capacity_cols]
        self.generator_names = generator_names
        
        self.num_generators = len(generator_names)
        
        # Extract generator data from scenarios_df first row and costs_df
        self.pmax_list = [scenarios_df[f"{gen}_cap"].iloc[0] for gen in generator_names]
        self.pmin_list = [0.0] * self.num_generators  # Default Pmin = 0
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in generator_names]

        self.demand = scenarios_df['demand']
        self.demand_profile = scenarios_df['demand_profile']

        diag_config = load_diagonalization()

        # Algorithm parameters
        self.max_iterations = int(diag_config.get("max_iterations"))
        self.conv_tolerance = float(diag_config.get("conv_tolerance"))
        
        # Create MPEC model (reused for all strategic players)
        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)
        self.mpec_model = MPECModel(
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
            p_init=self.P_init,
            players_config=self.players_config,
            feature_matrix_by_player=self.feature_matrix_by_player,
        )
        
        # # Initialize bid vector with true costs
        # self.bid_vector = self.cost_vector.copy()
               
        # History tracking
        self.bid_history = []
        self.profit_history_agent_perspective = []
        self.profit_history_agent_perspective_scenario = []
        self.profit_history_ED_perspective = []
        self.profit_history_ED_perspective_scenario = []
        self.theta_history = []
        self.dispatch_history = []
        self.clearing_price_history = []

    def _compute_p_init_from_ed(self, scenarios_df) -> List[List[float]]:
        """Solve ED and extract first time-step dispatch as [scenario][generator]."""
        ed_for_p_init = EconomicDispatchModel(scenarios_df, self.costs_df, self.ramps_df)
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def solve_strategic_player_problem(self, player_id: int) -> float:
        """
        Solve MPEC optimization for a strategic player
        
        Parameters
        ----------
        player_id : int
            ID of the strategic player
            
        Returns
        -------
        float
            Total profit for the player (aggregated across all their generators and scenarios)
        """
        
        # Reconstruct MPEC model with current scenarios_df (bids already up-to-date)
        self.mpec_model = MPECModel(
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            ramps_df=self.ramps_df,
            p_init=self.P_init,
            players_config=self.players_config,
            feature_matrix_by_player=self.feature_matrix_by_player,
            strategic_player_id=player_id,
        )
        
        # Solve the MPEC model
        self.mpec_model.solve()
        
        theta = self.mpec_model.get_optimal_theta()

        # Get per-scenario profits (correctly evaluated from solved variable values)
        scenario_profits = self.mpec_model.get_scenario_profits()
        
        # Total profit = sum across all scenarios (not averaged)
        total_profit = sum(scenario_profits)

        return total_profit, scenario_profits, theta

    def calculate_ED(self) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """
        Calculate market dispatch and profits using economic dispatch across all scenarios
        
        Returns
        -------
        tuple
            (all_dispatches, clearing_prices, all_player_profits, total_player_profits)
            - all_dispatches: List[List[float]] - dispatch for each scenario and generator
            - clearing_prices: List[float] - clearing price for each scenario  
            - all_player_profits: List[List[float]] - profits for each scenario and player
            - total_player_profits: List[float] - total profit for each player across scenarios
        """
        ed = EconomicDispatchModel(self.scenarios_df, self.costs_df, self.ramps_df)
        ed.solve()
        all_dispatches = ed.get_dispatches()
        clearing_prices = ed.get_clearing_prices()
        all_gen_profits = ed.get_generator_profits()  # List[List[float]] by [scenario][generator]
        
        # Aggregate generator profits by player for each scenario
        all_player_profits = []
        for scenario_idx in range(len(all_gen_profits)):
            scenario_player_profits = []
            for player_config in self.players_config:
                player_profit = sum(all_gen_profits[scenario_idx][g] for g in player_config['controlled_generators'])
                scenario_player_profits.append(player_profit)
            all_player_profits.append(scenario_player_profits)
        
        # Sum each player's profit across all scenarios
        num_players = len(self.players_config)
        total_player_profits = [
            sum(all_player_profits[s][p] for s in range(len(all_player_profits)))
            for p in range(num_players)
        ]

        return all_dispatches, clearing_prices, all_player_profits, total_player_profits
    
    def check_convergence(self, parameter_1: float, parameter_2: float) -> bool:
        """
        Check if the algorithm has converged
        
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        if abs(parameter_1 - parameter_2) <= self.conv_tolerance * parameter_2 + self.conv_tolerance:
            return True
        else: 
            return False
    
    def run(self) -> Dict[str, Any]:
        """
        Run the best response algorithm
        
        Returns
        -------
        dict
            Dictionary containing results and convergence information
        """
        
        print("=== Starting Best Response Algorithm ===")
        print(f"Number of generators: {self.num_generators}")
        # print(f"Initial bids: {[f'{b:.2f}' for b in self.bid_vector]}")
        print(f"Generator costs: {[f'{c:.2f}' for c in self.cost_vector]}")
        print(f"Demand: {[f'{d:.1f}' for d in self.demand]} MW")

        
        self.iteration = 0
        
        # Cache generator names for bid extraction
        generator_names = [col.replace('_cap', '') for col in self.scenarios_df.columns if col.endswith('_cap')]
        
        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")
            
            # Update each player's bid sequentially (shuffled order each iteration)
            profit_agent_perspective = [None] * len(self.players_config)
            profit_agent_perspective_scenario = [None] * len(self.players_config)
            iteration_thetas: Dict[int, np.ndarray] = {}

            indices = list(range(len(self.players_config)))

            # np.random.seed(self.seed + self.iteration)  # Ensure reproducibility with changing seed each iteration
            # np.random.shuffle(indices)

            for player_idx in indices:
                player_config = self.players_config[player_idx]
                player_id = player_config['id']
                controlled_generators = player_config['controlled_generators']
                print(f"  Solving for player {player_id} (controls generators {controlled_generators})...")
                
                # Solve MPEC problem for this player (using current bid_vector which may have been updated by previous players)
                total_profit, scenario_profits, theta = self.solve_strategic_player_problem(player_id)
                
                # Update scenarios DataFrame with optimal bids
                self.scenarios_df = self.mpec_model.update_bids_with_optimal_values(self.scenarios_df)

                profit_agent_perspective[player_idx] = total_profit  # Store total profit for this player
                profit_agent_perspective_scenario[player_idx] = scenario_profits  # Store per-scenario profits

                iteration_thetas[player_id] = theta  # Store theta for this player and iteration

                # print(f"    Total profit for player {player_id}: {total_profit:.2f}")
                # print(f"    Scenario profits: {[f'{p:.2f}' for p in scenario_profits]}")
            
            # Store scenario-time-dependent bid history: [iteration][scenario][generator][time]
            iteration_bids = []
            for s in range(len(self.scenarios_df)):
                scenario_bid_matrix = [
                    list(self.scenarios_df.at[s, f"{gen_name}_bid_profile"])
                    for gen_name in generator_names
                ]
                iteration_bids.append(scenario_bid_matrix)
            self.bid_history.append(iteration_bids)
            
            # Concise per-iteration summary
            mpec_profits_str = ", ".join([f"P{i}={profit_agent_perspective[i]:.1f}" for i in range(len(self.players_config))])
            print(f"  MPEC profits: {mpec_profits_str}")
            
            # Store the profit history: [iteration][player_idx] -> list of scenario profits
            self.profit_history_agent_perspective.append(profit_agent_perspective.copy())
            self.profit_history_agent_perspective_scenario.append(profit_agent_perspective_scenario.copy())
            self.theta_history.append(iteration_thetas.copy())
            
            if self.iteration > 0:
                self.convergence_check_1 = []
                for player_idx in range(len(self.players_config)):
                    current_profit = self.profit_history_agent_perspective[self.iteration][player_idx]
                    previous_profit = self.profit_history_agent_perspective[self.iteration - 1][player_idx]
                    self.convergence_check_1.append(self.check_convergence(current_profit, previous_profit))

                if all(self.convergence_check_1):
                    print("  Check 1 PASSED (all bids stable) — running ED validation...")
                    
                    # Only run ED when MPEC has converged
                    all_dispatches, clearing_prices, all_player_profits, total_player_profits = self.calculate_ED()
                    self.dispatch_history.append(all_dispatches)
                    self.clearing_price_history.append(clearing_prices)
                    self.profit_history_ED_perspective.append(total_player_profits)
                    self.profit_history_ED_perspective_scenario.append(all_player_profits)
                    
                    ed_profits_str = ", ".join([f"P{i}={total_player_profits[i]:.1f}" for i in range(len(self.players_config))])
                    print(f"  ED profits:   {ed_profits_str}")

                    # --- Convergence Check 2: MPEC profit ≈ ED profit (same iteration) ---
                    self.convergence_check_2 = []
                    for player_idx in range(len(self.players_config)):
                        mpec_p = self.profit_history_agent_perspective[self.iteration][player_idx]
                        ed_p = total_player_profits[player_idx]
                        self.convergence_check_2.append(self.check_convergence(mpec_p, ed_p))

                    if all(self.convergence_check_2):
                        print("Convergence achieved! (MPEC profits stable AND MPEC ≈ ED)")
                        
                        # Print final comparison
                        for player_idx in range(len(self.players_config)):
                            mpec_p = self.profit_history_agent_perspective[self.iteration][player_idx]
                            ed_p = total_player_profits[player_idx]
                            print(f"    Player {player_idx}: MPEC={mpec_p:.2f}, ED={ed_p:.2f}, gap={mpec_p - ed_p:.2f}")
                        
                        self.results = self.get_results()
                        break
                    else:
                        for player_idx in range(len(self.players_config)):
                            mpec_p = self.profit_history_agent_perspective[self.iteration][player_idx]
                            ed_p = total_player_profits[player_idx]
                            print(f"    Player {player_idx}: MPEC={mpec_p:.2f}, ED={ed_p:.2f}, gap={mpec_p - ed_p:.2f}")
                        print("  Check 2 FAILED (MPEC and ED profits differ)")
                else:
                    num_unconverged = sum(1 for c in self.convergence_check_1 if not c)
                    total_checks = len(self.convergence_check_1)
                    print(f"  Check 1 FAILED ({num_unconverged}/{total_checks} bids still changing)")
                
            # Increment iteration counter
            self.iteration += 1
            if self.iteration == self.max_iterations:
                print("Maximum iterations reached without convergence.")
                # Run final ED so results are complete
                all_dispatches, clearing_prices, all_player_profits, total_player_profits = self.calculate_ED()
                self.dispatch_history.append(all_dispatches)
                self.clearing_price_history.append(clearing_prices)
                self.profit_history_ED_perspective.append(total_player_profits)
                self.profit_history_ED_perspective_scenario.append(all_player_profits)
                self.results = self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get algorithm results
        
        Returns
        -------
        dict
            Dictionary containing all results and history with scenario-indexed data
        """
        
        final_dispatches, final_prices, final_player_profits, final_total_player_profits = self.calculate_ED()
        
        # Calculate aggregate welfare for each scenario
        scenario_welfare = [sum(final_player_profits[s]) for s in range(len(final_player_profits))]
        
        # Calculate scenario-specific bid data from scenarios_df
        generator_names = [col.replace('_cap', '') for col in self.scenarios_df.columns if col.endswith('_cap')]
        scenario_bids = []
        for s in range(len(self.scenarios_df)):
            missing_profile_cols = [
                f"{gen_name}_bid_profile" for gen_name in generator_names
                if f"{gen_name}_bid_profile" not in self.scenarios_df.columns
            ]
            if missing_profile_cols:
                raise ValueError(f"Missing required profile columns: {missing_profile_cols}")

            num_time_steps = len(self.scenarios_df.at[s, f"{generator_names[0]}_bid_profile"])
            scenario_bid_matrix = []
            for t in range(num_time_steps):
                scenario_bid_matrix.append([
                    self.scenarios_df.at[s, f"{gen_name}_bid_profile"][t]
                    for gen_name in generator_names
                ])
            scenario_bids.append(scenario_bid_matrix)

        num_scenarios = len(final_dispatches)
        num_time_steps = len(final_dispatches[0]) if num_scenarios > 0 else 0
        num_players = len(self.players_config)

        final_thetas = self.theta_history[-1] if self.theta_history else {}
        
        results = {
            "iterations": self.iteration,
            "num_scenarios": len(final_dispatches),
            "generator_costs": self.cost_vector.copy(),
            "bid_history": self.bid_history.copy(),
            "theta_history": self.theta_history.copy(),
            "final_thetas": final_thetas,
            "profit_history_agent_perspective": self.profit_history_agent_perspective.copy(),
            "dispatch_history": self.dispatch_history.copy(),
            "clearing_price_history": self.clearing_price_history.copy(),
            "profit_history_ED_perspective": self.profit_history_ED_perspective.copy(),
            "final_scenarios_data": {
                "scenario_bids": scenario_bids,  # [scenario][time][generator]
                "scenario_dispatches": final_dispatches,  # [scenario][time][generator]
                "scenario_prices": final_prices,  # [scenario][time]
                "scenario_player_profits": final_player_profits,  # [scenario][player]
                "scenario_welfare": scenario_welfare  # [scenario]
            }
        }
        
        return results

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

    # Create and run algorithm
    algo = BestResponseAlgorithmMS(scenarios_df, costs_df, ramps_df, players_config, feature_matrix_by_player)
    start = time.perf_counter()
    algo.run()
    end = time.perf_counter()

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")
    
    results = algo.results
    saved_path = algo.save_results("results/best_response_results.json")

    print(saved_path)
    
    # # Display final results
    # print(f"\n=== Final Results ====")
    # print(f"Iterations: {results['iterations']}")
    # print(f"Number of scenarios: {results['num_scenarios']}")
    # print(f"Generator costs: {[f'{c:.2f}' for c in results['generator_costs']]}")
    
    # # Show summary statistics across all scenarios
    # summary = results['summary_stats']
    # print(f"\n=== Summary Statistics (Average Across Scenarios) ====")
    # print(f"Average final dispatch: {[f'{d:.1f}' for d in summary['avg_dispatch']]} MW")
    # print(f"Average market price: ${summary['avg_price']:.2f}/MWh")
    # print(f"Average player profits: {[f'{p:.2f}' for p in summary['avg_player_profits']]}")
    # print(f"Average total welfare: ${summary['avg_welfare']:.2f}")
    
    # # Show scenario-specific results
    # print(f"\n=== Scenario-Specific Results ===")
    # for s in range(results['num_scenarios']):
    #     scenario_data = results['final_scenarios_data']
    #     print(f"\nScenario {s}:")
    #     print(f"  Dispatch: {[f'{d:.1f}' for d in scenario_data['scenario_dispatches'][s]]} MW")
    #     print(f"  Price: ${scenario_data['scenario_prices'][s]:.2f}/MWh")
    #     print(f"  Player profits: {[f'{p:.2f}' for p in scenario_data['scenario_player_profits'][s]]}")
    #     print(f"  Total welfare: ${scenario_data['scenario_welfare'][s]:.2f}")
    
    # Visualizations with scenario selection
    # scenario_to_analyze = 0  # Choose which scenario to analyze in detail
    # print(f"\n=== Visualizing Results for Scenario {scenario_to_analyze} ===")
    
    # # Visualize bid evolution
    # print("\n=== Visualizing Bid Evolution ====")
    # algo.visualize_bid_evolution()
    # algo.visualize_merit_order_comparison()