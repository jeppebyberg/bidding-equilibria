import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from models.diagonalization.MultipleScenarios.MPEC_MS import MPECModel
from models.diagonalization.MultipleScenarios.economic_dispatch_MS import EconomicDispatchModel
from models.diagonalization.MultipleScenarios.utilities.diagonalization_loader import load_diagonalization

class BestResponseAlgorithmMS:
    """
    Best response algorithm for finding bidding equilibrium using MPEC and economic dispatch models
    """
    
    def __init__(self,
                 scenarios_df,
                 costs_df,
                 players_config, 
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
        self.players_config = players_config
        
        self.seed = seed

        # Extract basic data for compatibility with ED model from scenarios_df
        # Auto-detect generator names from capacity columns
        capacity_cols = [col for col in scenarios_df.columns if col.endswith('_cap')]
        generator_names = [col.replace('_cap', '') for col in capacity_cols]
        
        self.num_generators = len(generator_names)
        
        # Extract generator data from scenarios_df first row and costs_df
        self.pmax_list = [scenarios_df[f"{gen}_cap"].iloc[0] for gen in generator_names]
        self.pmin_list = [0.0] * self.num_generators  # Default Pmin = 0
        self.cost_vector = [costs_df[f"{gen}_cost"].iloc[0] for gen in generator_names]
        
        # Use first scenario demand for ED compatibility
        demand_col = None
        for col in scenarios_df.columns:
            if any(keyword in col.lower() for keyword in ['demand', 'load']):
                demand_col = col
                break
        self.demand = scenarios_df[demand_col].iloc[0] if demand_col else scenarios_df.iloc[0, 0]

        diag_config = load_diagonalization()

        # Algorithm parameters
        self.max_iterations = int(diag_config.get("max_iterations"))
        self.conv_tolerance = float(diag_config.get("conv_tolerance"))
        self.damping_factor = float(diag_config.get("damping_factor", 0.5))
        
        # Create MPEC model (reused for all strategic players)
        self.mpec_model = MPECModel(
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            players_config=self.players_config
        )
        
        # Create economic dispatch model
        self.ed_model = EconomicDispatchModel()
        
        # Initialize bid vector with true costs
        self.bid_vector = self.cost_vector.copy()
               
        # History tracking
        self.bid_history = []
        self.profit_history_agent_perspective = []
        self.profit_history_agent_perspective_scenario = []
        self.profit_history_ED_perspective = []
        self.profit_history_ED_perspective_scenario = []
        self.dispatch_history = []
        self.clearing_price_history = []
        
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
            players_config=self.players_config
        )
        
        # Update the MPEC model for this strategic player
        self.mpec_model.update_strategic_player(strategic_player_id=player_id)
        
        # Solve the MPEC model
        self.mpec_model.solve()
        
        # Get per-scenario profits (correctly evaluated from solved variable values)
        scenario_profits = self.mpec_model.get_scenario_profits()
        
        # Total profit = sum across all scenarios (not averaged)
        total_profit = sum(scenario_profits)

        return total_profit, scenario_profits
                                
    def calculate_ED(self) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """
        Calculate market dispatch and profits using economic dispatch across all scenarios
        
        Returns
        -------
        tuple
            (all_dispatches, clearing_prices, all_player_profits)
            - all_dispatches: List[List[float]] - dispatch for each scenario and generator
            - clearing_prices: List[float] - clearing price for each scenario  
            - all_player_profits: List[List[float]] - profits for each scenario and player
        """
        
        # Use the DataFrame-based economic dispatch function (scenarios_df already has up-to-date bids)
        generator_names = [col.replace('_cap', '') for col in self.scenarios_df.columns if col.endswith('_cap')]
        all_dispatches, clearing_prices, all_profits = self.ed_model.economic_dispatch_from_dataframe(
            scenarios_df=self.scenarios_df,
            costs_df=self.costs_df,
            pmin_default=0.0
        )
        
        # Extract generator profits and aggregate by player for each scenario
        # all_profits format: [{'G1': profit, 'G2': profit, ...}, ...] (one dict per scenario)
        all_player_profits = []
        for scenario_idx in range(len(all_profits)):
            scenario_player_profits = []
            for player_config in self.players_config:
                player_profit = 0.0
                controlled_generators = player_config['controlled_generators']
                for gen_id in controlled_generators:
                    gen_name = generator_names[gen_id]
                    player_profit += all_profits[scenario_idx][gen_name]
                scenario_player_profits.append(player_profit)
            all_player_profits.append(scenario_player_profits)
        
        # Sum each player's profit across all scenarios: [player] -> total profit
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
        print(f"Initial bids: {[f'{b:.2f}' for b in self.bid_vector]}")
        print(f"Generator costs: {[f'{c:.2f}' for c in self.cost_vector]}")
        print(f"Demand: {self.demand:.1f} MW")
        print(f"Damping factor: {self.damping_factor}")
        
        self.iteration = 0
        
        # Cache generator names for bid extraction
        generator_names = [col.replace('_cap', '') for col in self.scenarios_df.columns if col.endswith('_cap')]
        
        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")
            
            # Update each player's bid sequentially (shuffled order each iteration)
            profit_agent_perspective = [None] * len(self.players_config)
            profit_agent_perspective_scenario = [None] * len(self.players_config)

            indices = list(range(len(self.players_config)))

            # np.random.seed(self.seed + self.iteration)  # Ensure reproducibility with changing seed each iteration
            # np.random.shuffle(indices)

            for player_idx in indices:
                player_config = self.players_config[player_idx]
                player_id = player_config['id']
                controlled_generators = player_config['controlled_generators']
                print(f"  Solving for player {player_id} (controls generators {controlled_generators})...")
                
                # Solve MPEC problem for this player (using current bid_vector which may have been updated by previous players)
                total_profit, scenario_profits = self.solve_strategic_player_problem(player_id)
                
                # # Update scenarios DataFrame with optimal bids using MPEC function
                # self.scenarios_df = self.mpec_model.update_bids_with_optimal_values(self.scenarios_df)
                    
                # Apply damped bid update: new_bid = (1-α)*old_bid + α*optimal_bid
                optimal_bid_scenarios = self.mpec_model.get_optimal_bids()
                strategic_gens = self.mpec_model.strategic_generators
                gen_names_mpec = self.mpec_model.generator_names
                for gen_idx in strategic_gens:
                    bid_col = f"{gen_names_mpec[gen_idx]}_bid"
                    for s in range(len(self.scenarios_df)):
                        old_bid = self.scenarios_df.at[s, bid_col]
                        optimal_bid = optimal_bid_scenarios[s][gen_idx]
                        damped_bid = (1 - self.damping_factor) * old_bid + self.damping_factor * optimal_bid
                        self.scenarios_df.at[s, bid_col] = damped_bid


                profit_agent_perspective[player_idx] = total_profit  # Store total profit for this player
                profit_agent_perspective_scenario[player_idx] = scenario_profits  # Store per-scenario profits

                # print(f"    Total profit for player {player_id}: {total_profit:.2f}")
                # print(f"    Scenario profits: {[f'{p:.2f}' for p in scenario_profits]}")
            
            # Store scenario-dependent bid history: [iteration][scenario][generator]
            iteration_bids = []
            for s in range(len(self.scenarios_df)):
                scenario_bid_vector = [self.scenarios_df.at[s, f"{gen_name}_bid"] for gen_name in generator_names]
                iteration_bids.append(scenario_bid_vector)
            self.bid_history.append(iteration_bids)
            
            # Concise per-iteration summary
            mpec_profits_str = ", ".join([f"P{i}={profit_agent_perspective[i]:.1f}" for i in range(len(self.players_config))])
            print(f"  MPEC profits: {mpec_profits_str}")
            for s in range(len(self.scenarios_df)):
                bids_str = ", ".join([f"{iteration_bids[s][g]:.1f}" for g in range(self.num_generators)])
                print(f"  S{s} bids: [{bids_str}]")
            
            # Store the profit history: [iteration][player_idx] -> list of scenario profits
            self.profit_history_agent_perspective.append(profit_agent_perspective.copy())
            self.profit_history_agent_perspective_scenario.append(profit_agent_perspective_scenario.copy())
            
            # Calculate ED every iteration to track actual market outcomes
            all_dispatches, clearing_prices, all_player_profits, total_player_profits = self.calculate_ED()
            self.dispatch_history.append(all_dispatches)
            self.clearing_price_history.append(clearing_prices)
            self.profit_history_ED_perspective.append(total_player_profits)
            self.profit_history_ED_perspective_scenario.append(all_player_profits)
            
            ed_profits_str = ", ".join([f"P{i}={total_player_profits[i]:.1f}" for i in range(len(self.players_config))])
            print(f"  ED profits:   {ed_profits_str}")
            
            if self.iteration > 0:
                # --- Convergence Check 1: MPEC profit stability between iterations ---
                self.convergence_check_1 = []
                for player_idx in range(len(self.players_config)):
                    current_profit = self.profit_history_agent_perspective[self.iteration][player_idx]
                    previous_profit = self.profit_history_agent_perspective[self.iteration - 1][player_idx]
                    self.convergence_check_1.append(self.check_convergence(current_profit, previous_profit))

                # --- Convergence Check 2: ED profit stability between iterations ---
                self.convergence_check_2 = []
                for player_idx in range(len(self.players_config)):
                    current_ed_profit = self.profit_history_ED_perspective[self.iteration][player_idx]
                    previous_ed_profit = self.profit_history_ED_perspective[self.iteration - 1][player_idx]
                    self.convergence_check_2.append(self.check_convergence(current_ed_profit, previous_ed_profit))

                # Both checks must pass for convergence
                if all(self.convergence_check_1) and all(self.convergence_check_2):
                    print("Convergence achieved! (MPEC profits stable AND ED profits stable)")
                    
                    # Print final comparison
                    for player_idx in range(len(self.players_config)):
                        mpec_p = self.profit_history_agent_perspective[self.iteration][player_idx]
                        ed_p = total_player_profits[player_idx]
                        print(f"    Player {player_idx}: MPEC={mpec_p:.2f}, ED={ed_p:.2f}, gap={mpec_p - ed_p:.2f}")
                    
                    self.results = self.get_results()
                    break
                elif all(self.convergence_check_1) and not all(self.convergence_check_2):
                    print("  Check 1 PASSED (MPEC stable), Check 2 FAILED (ED still changing)")
                elif not all(self.convergence_check_1) and all(self.convergence_check_2):
                    print("  Check 1 FAILED (MPEC changing), Check 2 PASSED (ED stable)")
                
            # Increment iteration counter
            self.iteration += 1
            if self.iteration == self.max_iterations:
                print("Maximum iterations reached without convergence.")
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
            scenario_bid_vector = [self.scenarios_df.at[s, f"{gen_name}_bid"] for gen_name in generator_names]
            scenario_bids.append(scenario_bid_vector)
        
        results = {
            "iterations": self.iteration,
            "num_scenarios": len(final_dispatches),
            "generator_costs": self.cost_vector.copy(),
            "bid_history": self.bid_history.copy(),
            "profit_history_agent_perspective": self.profit_history_agent_perspective.copy(),
            "dispatch_history": self.dispatch_history.copy(),
            "clearing_price_history": self.clearing_price_history.copy(),
            "profit_history_ED_perspective": self.profit_history_ED_perspective.copy(),
            "final_scenarios_data": {
                "scenario_bids": scenario_bids,  # [scenario][generator] 
                "scenario_dispatches": final_dispatches,  # [scenario][generator]
                "scenario_prices": final_prices,  # [scenario]
                "scenario_player_profits": final_player_profits,  # [scenario][player]
                "scenario_welfare": scenario_welfare  # [scenario]
            },
            "summary_stats": {
                "avg_dispatch": [sum(final_dispatches[s][g] for s in range(len(final_dispatches))) / len(final_dispatches) 
                                for g in range(self.num_generators)],
                "avg_price": sum(final_prices) / len(final_prices),
                "avg_player_profits": [sum(final_player_profits[s][p] for s in range(len(final_player_profits))) / len(final_player_profits) 
                                      for p in range(len(self.players_config))],
                "avg_welfare": sum(scenario_welfare) / len(scenario_welfare)
            }
        }
        
        return results
    
    def visualize_bid_evolution(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize how bids evolve over iterations for each generator.
        
        Parameters
        ----------
        scenario_id : int, optional
            Scenario to plot bids for. If None, plots one subplot per scenario.
        """
        if not self.bid_history:
            print("No bid history available for visualization")
            return
        
        # bid_history structure: [iteration][scenario][generator]
        num_iterations = len(self.bid_history)
        num_scenarios = len(self.bid_history[0])
        iterations = list(range(num_iterations))
        
        if scenario_id is not None:
            # Single scenario plot
            if scenario_id >= num_scenarios:
                print(f"Invalid scenario_id {scenario_id}. Available scenarios: 0-{num_scenarios-1}")
                return
            
            plt.figure(figsize=(12, 8))
            for gen_id in range(self.num_generators):
                bids_over_time = [self.bid_history[it][scenario_id][gen_id] for it in range(num_iterations)]
                plt.plot(iterations, bids_over_time, marker='o', linewidth=2,
                        label=f'Generator {gen_id} (Cost: ${self.cost_vector[gen_id]:.1f})')
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Bid ($/MWh)', fontsize=12)
            plt.title(f'Bid Evolution Over Iterations (Scenario {scenario_id})', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Summary table for the selected scenario
            print(f"\n=== Bid Evolution Summary (Scenario {scenario_id}) ===")
            header = "Iter " + " ".join([f"{'Gen '+str(g):>8}" for g in range(self.num_generators)])
            print(header)
            print("-" * len(header))
            for it in range(num_iterations):
                bid_str = " ".join([f"{self.bid_history[it][scenario_id][g]:8.2f}" for g in range(self.num_generators)])
                print(f"{it:<4} {bid_str}")
        else:
            # One subplot per scenario
            cols = min(3, num_scenarios)
            rows = (num_scenarios + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
            
            for s in range(num_scenarios):
                ax = axes[s // cols][s % cols]
                for gen_id in range(self.num_generators):
                    bids_over_time = [self.bid_history[it][s][gen_id] for it in range(num_iterations)]
                    ax.plot(iterations, bids_over_time, marker='o', linewidth=1.5,
                           label=f'Gen {gen_id} (${self.cost_vector[gen_id]:.1f})')
                
                ax.set_xlabel('Iteration', fontsize=10)
                ax.set_ylabel('Bid ($/MWh)', fontsize=10)
                ax.set_title(f'Scenario {s}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            
            # Hide unused subplots
            for idx in range(num_scenarios, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)
            
            fig.suptitle('Bid Evolution Over Iterations (All Scenarios)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    def visualize_supply_demand_curve(self, scenario_id: int = 0) -> None:
        """
        Visualize the supply-demand curve with market clearing point for a specific scenario
        
        Parameters
        ----------
        scenario_id : int, optional
            Scenario ID to visualize (default: 0)
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for visualization")
            return
            
        if scenario_id >= self.results['num_scenarios']:
            print(f"Invalid scenario_id {scenario_id}. Available scenarios: 0-{self.results['num_scenarios']-1}")
            return
            
        scenario_bids = self.results['final_scenarios_data']['scenario_bids'][scenario_id]
        scenario_dispatch = self.results['final_scenarios_data']['scenario_dispatches'][scenario_id]
        scenario_price = self.results['final_scenarios_data']['scenario_prices'][scenario_id]
        
        # Create merit order (sort generators by bid price)
        gen_data = [(i, scenario_bids[i], self.pmax_list[i], scenario_dispatch[i]) for i in range(self.num_generators)]
        gen_data.sort(key=lambda x: x[1])  # Sort by bid price
        
        # Build supply curve
        cumulative_capacity = 0
        supply_quantities = [0]
        supply_prices = [0]
        
        for gen_id, bid, pmax, dispatch in gen_data:
            # Add point before price change
            supply_quantities.append(cumulative_capacity)
            supply_prices.append(bid)
            
            # Add capacity at this price level
            cumulative_capacity += pmax
            supply_quantities.append(cumulative_capacity)
            supply_prices.append(bid)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot supply curve
        plt.step(supply_quantities, supply_prices, where='post', linewidth=2.5, 
                color='blue', label='Supply Curve', alpha=0.8)
        
        # Plot demand curve (vertical line at demand level)
        demand_level = self.scenarios_df.iloc[scenario_id][self.scenarios_df.columns[self.scenarios_df.columns.str.contains('demand|load', case=False)][0]]
        max_price = max(supply_prices) * 1.1
        plt.axvline(x=demand_level, color='red', linewidth=2.5, 
                   label=f'Demand ({demand_level:.0f} MW)', alpha=0.8)
        
        # Mark market clearing point
        plt.scatter([demand_level], [scenario_price], color='green', s=150, 
                   zorder=5, label=f'Market Clearing\n(Price: ${scenario_price:.2f}/MWh)')
        
        # Add annotations for dispatched generators
        cumulative_dispatch = 0
        for gen_id, bid, pmax, dispatch in gen_data:
            if dispatch > 0.1:  # Only show significantly dispatched units
                plt.annotate(f'Gen {gen_id}\n${bid:.1f}/MWh\n{dispatch:.0f} MW', 
                           xy=(cumulative_dispatch + dispatch/2, bid),
                           xytext=(cumulative_dispatch + dispatch/2, bid + max_price*0.05),
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
            cumulative_dispatch += dispatch
        
        # Formatting
        plt.xlabel('Quantity (MW)', fontsize=12)
        plt.ylabel('Price ($/MWh)', fontsize=12)
        plt.title(f'Supply-Demand Curve and Market Clearing (Scenario {scenario_id})', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        plt.xlim(0, max(cumulative_capacity, demand_level) * 1.1)
        plt.ylim(0, max_price)
        
        plt.tight_layout()
        plt.show()
        
        # Print dispatch summary
        print(f"\n=== Market Dispatch Summary (Scenario {scenario_id}) ===")
        print(f"{'Gen ID':<6} {'Bid':<8} {'Capacity':<8} {'Dispatch':<8} {'Status':<10}")
        print("-" * 50)
        total_dispatch = 0
        for gen_id, bid, pmax, dispatch in gen_data:
            status = "Dispatched" if dispatch > 0.1 else "Not Used"
            print(f"{gen_id:<6} ${bid:<7.2f} {pmax:<8.0f} {dispatch:<8.0f} {status:<10}")
            total_dispatch += dispatch
        print("-" * 50)
        print(f"Total Dispatch: {total_dispatch:.0f} MW")
        print(f"Total Demand: {demand_level:.0f} MW")
        print(f"Market Clearing Price: ${scenario_price:.2f}/MWh")
    
    def visualize_agent_profits(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize agent profits over iterations and compare with perfect competition
        
        Parameters
        ----------
        scenario_id : int, optional
            Scenario ID for profit comparison. If None, uses average across scenarios
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for visualization")
            return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Profit evolution over iterations
        if len(self.profit_history_agent_perspective) > 1:
            iterations = list(range(len(self.profit_history_agent_perspective)))
            
            for player_idx, player_config in enumerate(self.players_config):
                player_id = player_config['id']
                controlled_generators = player_config['controlled_generators']
                profits_over_time = [self.profit_history_agent_perspective[iter][player_idx] 
                                   for iter in range(len(self.profit_history_agent_perspective))]
                ax1.plot(iterations, profits_over_time, marker='o', linewidth=2, 
                        label=f'Player {player_id} (gens {controlled_generators})')
            
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Profit ($)', fontsize=12)
            ax1.set_title('Agent Perspective Profit Evolution Over Iterations', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data\nfor profit evolution', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Profit Evolution (No Data)', fontsize=14)
        
        # Right plot: Final profits comparison
        if scenario_id is not None:
            if scenario_id >= self.results['num_scenarios']:
                print(f"Invalid scenario_id {scenario_id}. Using average across scenarios.")
                scenario_id = None
        
        if scenario_id is not None:
            final_profits = self.results['final_scenarios_data']['scenario_player_profits'][scenario_id]
            title_suffix = f" (Scenario {scenario_id})"
        else:
            final_profits = self.results['summary_stats']['avg_player_profits']
            title_suffix = " (Average Across Scenarios)"
        
        # Calculate perfect competition profits (bidding at marginal cost)
        # Use single scenario for perfect competition comparison
        all_comp_dispatches, comp_prices = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand_scenarios=[self.demand],
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector  # Bid at marginal cost
        )
        perfect_comp_dispatch, perfect_comp_price = all_comp_dispatches[0], comp_prices[0]
        
        # Calculate perfect competition profits per generator first
        perfect_comp_gen_profits = []
        for i in range(self.num_generators):
            revenue = perfect_comp_price * perfect_comp_dispatch[i]
            cost = self.cost_vector[i] * perfect_comp_dispatch[i]
            perfect_comp_gen_profits.append(revenue - cost)
            
        # Aggregate perfect competition profits by player
        perfect_comp_profits = []
        for player_config in self.players_config:
            player_profit = 0.0
            controlled_generators = player_config['controlled_generators']
            for gen_id in controlled_generators:
                player_profit += perfect_comp_gen_profits[gen_id]
            perfect_comp_profits.append(player_profit)
        
        # Create bar chart
        num_players = len(self.players_config)
        x_pos = np.arange(num_players)
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, final_profits, width, 
                       label='Equilibrium Profits', color='blue', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, perfect_comp_profits, width, 
                       label='Perfect Competition', color='red', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'${height:.0f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'${height:.0f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Player ID', fontsize=12)
        ax2.set_ylabel('Profit ($)', fontsize=12)
        ax2.set_title(f'Final Profits vs Perfect Competition{title_suffix}', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Player {player["id"]}' for player in self.players_config])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed profit analysis
        print(f"\n=== Profit Analysis{title_suffix} ===")
        print(f"{'Player':<8} {'Generators':<15} {'Total Profit':<12} {'Perfect Comp':<12} {'Difference':<12}")
        print("-" * 80)
        
        total_equil_profit = 0
        total_perfect_profit = 0
        
        for player_idx, player_config in enumerate(self.players_config):
            player_id = player_config['id']
            controlled_generators = player_config['controlled_generators']
            gen_list = ', '.join(map(str, controlled_generators))
            
            equil_profit = final_profits[player_idx]
            comp_profit = perfect_comp_profits[player_idx]
            profit_diff = equil_profit - comp_profit
            
            print(f"{player_id:<8} {gen_list:<15} ${equil_profit:<11.0f} ${comp_profit:<11.0f} ${profit_diff:<11.0f}")
            
            total_equil_profit += equil_profit
            total_perfect_profit += comp_profit
        
        print("-" * 80)
        print(f"Total Welfare - Equilibrium: ${total_equil_profit:.0f}")
        print(f"Total Welfare - Perfect Comp: ${total_perfect_profit:.0f}")
        print(f"Welfare Loss: ${total_perfect_profit - total_equil_profit:.0f} ({((total_perfect_profit - total_equil_profit)/total_perfect_profit*100):.1f}%)")
        
        # Market power analysis 
        if scenario_id is not None:
            clearing_price = self.results['final_scenarios_data']['scenario_prices'][scenario_id]
        else:
            clearing_price = self.results['summary_stats']['avg_price']
            
        print(f"\nMarket Clearing Price - Equilibrium: ${clearing_price:.2f}/MWh")
        print(f"Market Clearing Price - Perfect Comp: ${perfect_comp_price:.2f}/MWh")
        price_markup = ((clearing_price - perfect_comp_price) / perfect_comp_price * 100)
        print(f"Price Markup: {price_markup:.1f}%")
    
    def analyze_competitive_benchmark(self) -> None:
        """
        Analyze what the competitive (perfect competition) outcome should be
        """
        print("\n=== Competitive Benchmark Analysis ===")
        
        # Calculate competitive dispatch (bidding at marginal cost)
        all_comp_dispatches, comp_prices = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand_scenarios=[self.demand],
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector  # Bid at marginal cost
        )
        comp_dispatch, comp_price = all_comp_dispatches[0], comp_prices[0]
        
        # Create merit order based on costs
        gen_data = [(i, self.cost_vector[i], self.pmax_list[i], comp_dispatch[i]) 
                   for i in range(self.num_generators)]
        gen_data.sort(key=lambda x: x[1])  # Sort by cost
        
        print("Merit Order (by marginal cost):")
        print(f"{'Gen ID':<6} {'Cost':<8} {'Capacity':<8} {'Dispatch':<8} {'Status':<12}")
        print("-" * 55)
        
        total_demand = sum(self.demand) if isinstance(self.demand, list) else self.demand
        cumulative_dispatch = 0
        marginal_gen = None
        
        for gen_id, cost, capacity, dispatch in gen_data:
            status = "Dispatched" if dispatch > 0.01 else "Not needed"
            if dispatch > 0.01:
                cumulative_dispatch += dispatch
                if cumulative_dispatch >= total_demand - 0.01:  # Allow small tolerance
                    marginal_gen = gen_id
                    status = "Marginal"
                    
            print(f"{gen_id:<6} ${cost:<7.2f} {capacity:<8.0f} {dispatch:<8.0f} {status:<12}")
        
        print(f"\nCompetitive Market Clearing Price: ${comp_price:.2f}/MWh")
        print(f"Total Demand: {total_demand:.0f} MW")
        print(f"Total Dispatch: {sum(comp_dispatch):.0f} MW")
        
        if marginal_gen is not None:
            print(f"Marginal Generator: {marginal_gen} (Cost: ${self.cost_vector[marginal_gen]:.2f}/MWh)")
        
        # Check if most expensive generator is dispatched
        most_expensive_gen = max(range(self.num_generators), key=lambda i: self.cost_vector[i])
        most_expensive_cost = self.cost_vector[most_expensive_gen]
        
        if comp_dispatch[most_expensive_gen] < 0.01:
            print(f"Most expensive generator {most_expensive_gen} (${most_expensive_cost:.2f}/MWh) is NOT dispatched")
            print(f"Theoretical market price should be just below ${most_expensive_cost:.2f}/MWh")
        else:
            print(f"Most expensive generator {most_expensive_gen} (${most_expensive_cost:.2f}/MWh) IS dispatched")
            print(f"Market price is set by this marginal generator")
        
        # Compare with current equilibrium results if available
        if hasattr(self, 'results') and self.results:
            print(f"\n=== Comparison with Current Equilibrium ===")
            equil_price = self.results['final_market_outcomes']['clearing_price']
            equil_dispatch = self.results['final_market_outcomes']['dispatch']
            
            print(f"Competitive price: ${comp_price:.2f}/MWh")
            print(f"Equilibrium price: ${equil_price:.2f}/MWh")
            print(f"Price difference: ${equil_price - comp_price:.2f}/MWh ({((equil_price/comp_price - 1)*100):.1f}%)")
            
            print(f"\nDispatch comparison:")
            print(f"{'Gen':<4} {'Competitive':<12} {'Equilibrium':<12} {'Difference':<10}")
            print("-" * 45)
            for i in range(self.num_generators):
                diff = equil_dispatch[i] - comp_dispatch[i]
                print(f"{i:<4} {comp_dispatch[i]:<12.1f} {equil_dispatch[i]:<12.1f} {diff:<10.1f}")
        
        return comp_dispatch, comp_price
    
    def compare_dispatch_formulations(self) -> None:
        """
        Compare dispatch and bids between Economic Dispatch (competitive) and MPEC (strategic) formulations
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for comparison")
            return
            
        # Get competitive dispatch (Economic Dispatch with marginal cost bidding)
        all_comp_dispatches, comp_prices = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand_scenarios=[self.demand],
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector  # Bid at marginal cost
        )
        comp_dispatch, comp_price = all_comp_dispatches[0], comp_prices[0]
        
        # Get strategic dispatch (MPEC results)
        strategic_dispatch = self.results['final_market_outcomes']['dispatch']
        strategic_price = self.results['final_market_outcomes']['clearing_price']
        strategic_bids = self.results['final_bids']
        
        # Create comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top Left: Dispatch Comparison
        x_pos = np.arange(self.num_generators)
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, comp_dispatch, width, 
                       label='Economic Dispatch (Competitive)', color='blue', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, strategic_dispatch, width, 
                       label='MPEC (Strategic)', color='red', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0.1:
                ax1.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0.1:
                ax1.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Generator ID')
        ax1.set_ylabel('Dispatch (MW)')
        ax1.set_title('Dispatch Comparison: Economic Dispatch vs MPEC')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Gen {i}' for i in range(self.num_generators)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Top Right: Bid Comparison
        bars3 = ax2.bar(x_pos - width/2, self.cost_vector, width, 
                       label='Marginal Costs', color='green', alpha=0.7)
        bars4 = ax2.bar(x_pos + width/2, strategic_bids, width, 
                       label='Strategic Bids', color='orange', alpha=0.7)
        
        # Add value labels
        for i, (cost, bid) in enumerate(zip(self.cost_vector, strategic_bids)):
            ax2.annotate(f'${cost:.1f}',
                        xy=(i - width/2, cost),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            ax2.annotate(f'${bid:.1f}',
                        xy=(i + width/2, bid),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Generator ID')
        ax2.set_ylabel('Price ($/MWh)')
        ax2.set_title('Bid Comparison: Marginal Costs vs Strategic Bids')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Gen {i}' for i in range(self.num_generators)])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Bottom Left: Dispatch Difference (Strategic - Competitive)
        dispatch_diff = [strategic_dispatch[i] - comp_dispatch[i] for i in range(self.num_generators)]
        colors = ['red' if diff > 0 else 'blue' if diff < 0 else 'gray' for diff in dispatch_diff]
        
        bars5 = ax3.bar(x_pos, dispatch_diff, color=colors, alpha=0.7)
        
        for bar, diff in zip(bars5, dispatch_diff):
            height = bar.get_height()
            if abs(height) > 0.1:
                ax3.annotate(f'{height:+.0f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3 if height > 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Generator ID')
        ax3.set_ylabel('Dispatch Difference (MW)')
        ax3.set_title('Dispatch Change: Strategic - Competitive\n(Red: Increased, Blue: Decreased)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Gen {i}' for i in range(self.num_generators)])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Bottom Right: Bid Markup Percentage
        markup_pct = [((strategic_bids[i] - self.cost_vector[i]) / self.cost_vector[i] * 100) 
                      if self.cost_vector[i] > 0 else 0 for i in range(self.num_generators)]
        
        bars6 = ax4.bar(x_pos, markup_pct, color='purple', alpha=0.7)
        
        for bar, markup in zip(bars6, markup_pct):
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Generator ID')
        ax4.set_ylabel('Markup Percentage (%)')
        ax4.set_title('Strategic Bid Markup Above Marginal Cost')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Gen {i}' for i in range(self.num_generators)])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed comparison table
        print("\n=== Detailed Dispatch & Bid Comparison ===")
        print(f"{'Gen':<4} {'Cost':<8} {'Strategic':<10} {'Markup':<8} {'Comp Disp':<10} {'Strat Disp':<11} {'Diff':<8}")
        print(f"{'ID':<4} {'($/MWh)':<8} {'Bid':<10} {'(%)':<8} {'(MW)':<10} {'(MW)':<11} {'(MW)':<8}")
        print("-" * 70)
        
        total_comp_dispatch = 0
        total_strategic_dispatch = 0
        
        for i in range(self.num_generators):
            markup = markup_pct[i]
            diff = dispatch_diff[i]
            
            print(f"{i:<4} ${self.cost_vector[i]:<7.2f} ${strategic_bids[i]:<9.2f} {markup:<7.1f} "
                  f"{comp_dispatch[i]:<10.1f} {strategic_dispatch[i]:<11.1f} {diff:<+7.1f}")
            
            total_comp_dispatch += comp_dispatch[i]
            total_strategic_dispatch += strategic_dispatch[i]
        
        print("-" * 70)
        print(f"Total Dispatch: {total_comp_dispatch:.1f} MW (Comp) vs {total_strategic_dispatch:.1f} MW (Strategic)")
        print(f"Market Price: ${comp_price:.2f}/MWh (Comp) vs ${strategic_price:.2f}/MWh (Strategic)")
        print(f"Price Difference: ${strategic_price - comp_price:.2f}/MWh ({((strategic_price/comp_price - 1)*100):+.1f}%)")
        
        # Efficiency analysis
        demand_level = sum(self.demand) if isinstance(self.demand, list) else self.demand
        print(f"\nDemand: {demand_level:.1f} MW")
        print(f"Supply adequacy: {'✓' if min(total_comp_dispatch, total_strategic_dispatch) >= demand_level - 0.1 else '✗'}")
    
    def _compute_merit_order_data(self, scenario_id: int) -> Dict[str, Any]:
        """
        Compute merit order data for a single scenario (competitive and strategic).
        
        Parameters
        ----------
        scenario_id : int
            Scenario index
            
        Returns
        -------
        dict
            Dictionary with all merit order data for the scenario
        """
        import pandas as pd
        
        scenario_data = self.results['final_scenarios_data']
        strategic_bids = scenario_data['scenario_bids'][scenario_id]
        strategic_dispatch = scenario_data['scenario_dispatches'][scenario_id]
        strategic_price = scenario_data['scenario_prices'][scenario_id]
        
        generator_names = [col.replace('_cap', '') for col in self.scenarios_df.columns if col.endswith('_cap')]
        demand_col = [col for col in self.scenarios_df.columns if any(kw in col.lower() for kw in ['demand', 'load'])][0]
        demand_level = self.scenarios_df.at[scenario_id, demand_col]
        scenario_pmax = [self.scenarios_df.at[scenario_id, f"{gen}_cap"] for gen in generator_names]
        
        # Compute competitive dispatch for this scenario (bid at marginal cost)
        comp_scenarios_df = self.scenarios_df.iloc[[scenario_id]].copy().reset_index(drop=True)
        for gen_name in generator_names:
            comp_scenarios_df[f"{gen_name}_bid"] = self.costs_df[f"{gen_name}_cost"].iloc[0]
        
        comp_dispatches, comp_prices, _ = self.ed_model.economic_dispatch_from_dataframe(
            scenarios_df=comp_scenarios_df,
            costs_df=self.costs_df,
            pmin_default=0.0
        )
        comp_dispatch = comp_dispatches[0]
        comp_price = comp_prices[0]
        
        # Create merit orders
        comp_gen_data = [(i, self.cost_vector[i], scenario_pmax[i], comp_dispatch[i]) 
                        for i in range(self.num_generators)]
        comp_gen_data.sort(key=lambda x: x[1])
        
        strategic_gen_data = [(i, strategic_bids[i], scenario_pmax[i], strategic_dispatch[i]) 
                             for i in range(self.num_generators)]
        strategic_gen_data.sort(key=lambda x: x[1])
        
        # Build competitive supply curve
        cumulative_capacity_comp = 0
        supply_quantities_comp = [0]
        supply_prices_comp = [0]
        for gen_id, cost, pmax, dispatch in comp_gen_data:
            supply_quantities_comp.append(cumulative_capacity_comp)
            supply_prices_comp.append(cost)
            cumulative_capacity_comp += pmax
            supply_quantities_comp.append(cumulative_capacity_comp)
            supply_prices_comp.append(cost)
        
        # Build strategic supply curve
        cumulative_capacity_strategic = 0
        supply_quantities_strategic = [0]
        supply_prices_strategic = [0]
        for gen_id, bid, pmax, dispatch in strategic_gen_data:
            supply_quantities_strategic.append(cumulative_capacity_strategic)
            supply_prices_strategic.append(bid)
            cumulative_capacity_strategic += pmax
            supply_quantities_strategic.append(cumulative_capacity_strategic)
            supply_prices_strategic.append(bid)
        
        return {
            'demand_level': demand_level,
            'comp_price': comp_price,
            'strategic_price': strategic_price,
            'comp_gen_data': comp_gen_data,
            'strategic_gen_data': strategic_gen_data,
            'supply_quantities_comp': supply_quantities_comp,
            'supply_prices_comp': supply_prices_comp,
            'supply_quantities_strategic': supply_quantities_strategic,
            'supply_prices_strategic': supply_prices_strategic,
            'cumulative_capacity_comp': cumulative_capacity_comp,
            'cumulative_capacity_strategic': cumulative_capacity_strategic,
        }
    
    def _plot_merit_order_on_ax(self, ax, data: Dict[str, Any], scenario_id: int, 
                                 detailed: bool = True) -> None:
        """
        Plot merit order comparison on a given axes object.
        
        Parameters
        ----------
        ax : matplotlib axes
            Axes to plot on
        data : dict
            Merit order data from _compute_merit_order_data
        scenario_id : int
            Scenario index (for title)
        detailed : bool
            If True, add annotations and larger fonts (single-scenario mode)
        """
        lw = 3 if detailed else 2
        fs_label = 12 if detailed else 8
        fs_title = 16 if detailed else 12
        marker_size = 200 if detailed else 100
        
        # Plot both supply curves
        ax.step(data['supply_quantities_comp'], data['supply_prices_comp'], where='post',
                linewidth=lw, color='blue', label='Competitive (Costs)', alpha=0.8)
        ax.step(data['supply_quantities_strategic'], data['supply_prices_strategic'], where='post',
                linewidth=lw, color='red', label='Strategic (Bids)', alpha=0.8, linestyle='--')
        
        # Plot demand line
        max_price = max(max(data['supply_prices_comp']), max(data['supply_prices_strategic'])) * 1.1
        ax.axvline(x=data['demand_level'], color='green', linewidth=lw,
                   label=f'Demand ({data["demand_level"]:.0f} MW)', alpha=0.8)
        
        # Mark market clearing points
        ax.scatter([data['demand_level']], [data['comp_price']], color='blue', s=marker_size,
                   marker='o', zorder=5, edgecolor='black',
                   label=f'Comp: ${data["comp_price"]:.2f}/MWh')
        ax.scatter([data['demand_level']], [data['strategic_price']], color='red', s=marker_size,
                   marker='s', zorder=5, edgecolor='black',
                   label=f'Strat: ${data["strategic_price"]:.2f}/MWh')
        
        # Add annotations only in detailed (single-scenario) mode
        if detailed:
            # Competitive merit order annotations
            cumulative_comp = 0
            comp_annotation_count = 0
            for gen_id, cost, pmax, dispatch in data['comp_gen_data']:
                if dispatch > 0.1:
                    vertical_offset = max_price * (0.12 + 0.04 * (comp_annotation_count % 3))
                    horizontal_offset = 5 * (comp_annotation_count % 2 - 0.5)
                    ax.annotate(f'Gen {gen_id}\nCost: ${cost:.1f}',
                               xy=(cumulative_comp + dispatch/2, cost),
                               xytext=(cumulative_comp + dispatch/2 + horizontal_offset, cost + vertical_offset),
                               ha='center', va='bottom', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7, lw=1.5))
                    comp_annotation_count += 1
                cumulative_comp += dispatch
            
            # Strategic merit order annotations (only show different ones)
            cumulative_strategic = 0
            strat_annotation_count = 0
            for gen_id, bid, pmax, dispatch in data['strategic_gen_data']:
                if dispatch > 0.1:
                    if abs(bid - self.cost_vector[gen_id]) > 1.0:
                        vertical_offset = max_price * (0.12 + 0.04 * (strat_annotation_count % 3))
                        horizontal_offset = 8 * (strat_annotation_count % 2 - 0.5)
                        ax.annotate(f'Gen {gen_id}\nBid: ${bid:.1f}',
                                   xy=(cumulative_strategic + dispatch/2, bid),
                                   xytext=(cumulative_strategic + dispatch/2 + horizontal_offset, bid - vertical_offset),
                                   ha='center', va='top', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5))
                        strat_annotation_count += 1
                cumulative_strategic += dispatch
        
        # Formatting
        ax.set_xlabel('Cumulative Capacity (MW)', fontsize=fs_label)
        ax.set_ylabel('Price ($/MWh)', fontsize=fs_label)
        ax.set_title(f'Merit Order Comparison (Scenario {scenario_id})', fontsize=fs_title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=fs_label - 2)
        ax.grid(True, alpha=0.3)
        
        max_capacity = max(data['cumulative_capacity_comp'], data['cumulative_capacity_strategic'])
        ax.set_xlim(0, max(max_capacity, data['demand_level']) * 1.1)
        ax.set_ylim(0, max_price)
    
    def visualize_merit_order_comparison(self, scenario_id: Optional[int] = None) -> None:
        """
        Visualize merit order curves for both competitive (cost-based) and strategic (bid-based) dispatch.
        
        Parameters
        ----------
        scenario_id : int, optional
            Scenario to visualize. If None, plots one subplot per scenario.
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for merit order comparison")
            return
        
        num_scenarios = self.results['num_scenarios']
        
        if scenario_id is not None:
            # --- Single scenario: detailed plot ---
            if scenario_id >= num_scenarios:
                print(f"Invalid scenario_id {scenario_id}. Available scenarios: 0-{num_scenarios-1}")
                return
            
            data = self._compute_merit_order_data(scenario_id)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            self._plot_merit_order_on_ax(ax, data, scenario_id, detailed=True)
            plt.tight_layout()
            plt.show()
            
            # Print detailed merit order comparison
            self._print_merit_order_summary(data, scenario_id)
        else:
            # --- All scenarios: one subplot per scenario ---
            cols = min(3, num_scenarios)
            rows = (num_scenarios + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
            
            for s in range(num_scenarios):
                ax = axes[s // cols][s % cols]
                data = self._compute_merit_order_data(s)
                self._plot_merit_order_on_ax(ax, data, s, detailed=False)
            
            # Hide unused subplots
            for idx in range(num_scenarios, rows * cols):
                axes[idx // cols][idx % cols].set_visible(False)
            
            fig.suptitle('Merit Order Comparison: Competitive vs Strategic (All Scenarios)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print summary for each scenario
            for s in range(num_scenarios):
                data = self._compute_merit_order_data(s)
                # self._print_merit_order_summary(data, s)
    
    def _print_merit_order_summary(self, data: Dict[str, Any], scenario_id: int) -> None:
        """Print merit order comparison summary for a scenario."""
        print(f"\n=== Merit Order Comparison (Scenario {scenario_id}) ===")
        print(f"Demand: {data['demand_level']:.0f} MW")
        print("\nCompetitive Merit Order (by Cost):")
        print(f"{'Rank':<4} {'Gen':<4} {'Cost':<8} {'Capacity':<8} {'Dispatch':<8}")
        print("-" * 40)
        for rank, (gen_id, cost, pmax, dispatch) in enumerate(data['comp_gen_data'], 1):
            print(f"{rank:<4} {gen_id:<4} ${cost:<7.2f} {pmax:<8.0f} {dispatch:<8.0f}")
        
        print(f"\nStrategic Merit Order (by Bid):")
        print(f"{'Rank':<4} {'Gen':<4} {'Bid':<8} {'Capacity':<8} {'Dispatch':<8} {'Cost':<8}")
        print("-" * 50)
        for rank, (gen_id, bid, pmax, dispatch) in enumerate(data['strategic_gen_data'], 1):
            cost = self.cost_vector[gen_id]
            print(f"{rank:<4} {gen_id:<4} ${bid:<7.2f} {pmax:<8.0f} {dispatch:<8.0f} ${cost:<7.2f}")
        
        print(f"\n=== Merit Order Efficiency Analysis ===")
        comp_price = data['comp_price']
        strategic_price = data['strategic_price']
        print(f"Competitive price: ${comp_price:.2f}/MWh")
        print(f"Strategic price: ${strategic_price:.2f}/MWh")
        print(f"Price increase: ${strategic_price - comp_price:.2f}/MWh ({((strategic_price/comp_price - 1)*100):+.1f}%)")
        
        comp_order = [gen_id for gen_id, _, _, dispatch in data['comp_gen_data'] if dispatch > 0.1]
        strategic_order = [gen_id for gen_id, _, _, dispatch in data['strategic_gen_data'] if dispatch > 0.1]
        
        if comp_order != strategic_order:
            print("Merit order CHANGED due to strategic bidding!")
            print(f"Competitive dispatch order: {comp_order}")
            print(f"Strategic dispatch order: {strategic_order}")
        else:
            print("Merit order UNCHANGED (only prices affected)")

if __name__ == "__main__":
    print("=== Testing Best Response Algorithm ===")
    
    # Generate scenarios for the algorithm
    from config.base_case.scenarios.scenario_generator import ScenarioManager
    
    # Load scenario manager
    scenario_manager = ScenarioManager("test_case")
    players_config = scenario_manager.get_players_config()
    
    # Generate demand scenarios with some variation
    demand_scenarios = scenario_manager.generate_demand_scenarios(
        "linear", 
        num_scenarios=5, 
        min_factor=0.4, 
        max_factor=0.6
    )
    
    # Generate capacity scenarios (use base case capacities)
    capacity_scenarios = scenario_manager.generate_capacity_scenarios(
        "linear",
        num_scenarios=1,  # Use base case capacities
        min_factor=1.0,
        max_factor=1.0
    )
    
    # Create scenario set
    scenarios = scenario_manager.create_scenario_set(
        demand_scenarios=demand_scenarios,
        capacity_scenarios=capacity_scenarios
    )
    
    scenarios_df = scenarios['scenarios_df']
    costs_df = scenarios['costs_df']
    
    seed = 0

    # Create and run algorithm
    algo = BestResponseAlgorithmMS(scenarios_df, costs_df, players_config, seed=seed)
    algo.run()
    results = algo.results
    
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
    scenario_to_analyze = 0  # Choose which scenario to analyze in detail
    print(f"\n=== Visualizing Results for Scenario {scenario_to_analyze} ===")
    
    # Visualize bid evolution
    print("\n=== Visualizing Bid Evolution ====")
    algo.visualize_bid_evolution()
    algo.visualize_merit_order_comparison()