import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

import config.base_case as config
from models.diagonalization.OneScenario.MPEC import MPECModel
from models.diagonalization.OneScenario.economic_dispatch import EconomicDispatchModel

from models.diagonalization.OneScenario.utilities.diagonalization_loader import load_diagonalization

class BestResponseAlgorithm:
    """
    Best response algorithm for finding bidding equilibrium using MPEC and economic dispatch models
    """
    
    def __init__(self,
                 case: str = "test_case"
                 ):
        """
        Initialize the best response algorithm
        
        Parameters
        ----------
        case : str
            Name of the test case to load from config
        """
        
        # Load test case data
        self.num_generators, self.pmax_list, self.pmin_list, self.cost_vector, self.demand, self.generators = config.load_setup_data(case)

        diag_config = load_diagonalization()

        # Algorithm parameters
        self.max_iterations = int(diag_config.get("max_iterations"))
        self.conv_tolerance = float(diag_config.get("conv_tolerance"))
        
        # Create MPEC model (reused for all strategic players)
        self.mpec_model = MPECModel(
            demand=self.demand,
            pmax_list=self.pmax_list,
            pmin_list=self.pmin_list,
            num_generators=self.num_generators,
            generators=self.generators
        )
        
        # Create economic dispatch model
        self.ed_model = EconomicDispatchModel()
        
        # Initialize bid vector with true costs
        self.bid_vector = self.cost_vector.copy()
               
        # History tracking
        self.bid_history = []
        self.profit_history_agent_perspective = []
        self.profit_history_ED_perspective = []
        self.dispatch_history = []
        self.clearing_price_history = []
        
    def solve_strategic_player_problem(self, strategic_player: int) -> float:
        """
        Solve MPEC optimization for a strategic player
        
        Parameters
        ----------
        strategic_player : int
            Index of the strategic generator
            
        Returns
        -------
        float
            Optimal bid for the strategic player
            Objective value (negative profit) for the strategic player
        """
        
        # Update the MPEC model for this strategic player
        self.mpec_model.update_strategic_player(
            strategic_player=strategic_player,
            bid_vector=self.bid_vector,
            cost_vector=self.cost_vector
        )
        
        # Solve the MPEC model
        self.mpec_model.solve()
        
        # Extract optimal bid
        if self.mpec_model.model.alpha[strategic_player].value is not None:
            return self.mpec_model.model.alpha[strategic_player].value, -self.mpec_model.model.objective.expr()
        else:
            print(f"Warning: No solution found for strategic player {strategic_player}")
            ValueError(f"No solution found for strategic player {strategic_player}")
                                
    def calculate_ED(self) -> Tuple[List[float], float, List[float]]:
        """
        Calculate market dispatch and profits using economic dispatch
        
        Returns
        -------
        tuple
            (dispatch, clearing_price, profits)
        """
        
        # Update economic dispatch with current bids
        dispatch, clearing_price = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand=self.demand,
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.bid_vector
        )
                
        # Calculate profits for each generator # -------------- Must be updated if agent owns multiple generators
        profits = []
        for i in range(self.num_generators):
            revenue = clearing_price * dispatch[i]
            cost = self.cost_vector[i] * dispatch[i]
            profit = revenue - cost
            profits.append(profit)
        
        return dispatch, clearing_price, profits
    
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
        
        self.iteration = 0
        
        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")
            
            # Store current bids
            self.bid_history.append(self.bid_vector.copy())
            
            # Update each generator's bid sequentially (Gauss-Seidel style)
            profit_agent_perspective = []

            for strategic_player in range(self.num_generators):
                print(f"  Solving for strategic player {strategic_player}...")
                
                # Solve MPEC problem for this player (using current bid_vector which may have been updated by previous players)
                optimal_bid, optimal_profit = self.solve_strategic_player_problem(strategic_player)
                
                # if optimal_bid < self.cost_vector[strategic_player]:
                #     print(f"    Warning: Optimal bid ${optimal_bid:.2f} is below cost ${self.cost_vector[strategic_player]:.2f}. Adjusting to cost.")
                #     optimal_bid = self.cost_vector[strategic_player]
                #     optimal_profit = self.mpec_model.model.lambda_var.value * self.mpec_model.model.P[strategic_player].value - self.cost_vector[strategic_player] * self.pmax_list[strategic_player]

                # Update this player's bid immediately
                self.bid_vector[strategic_player] = optimal_bid
                profit_agent_perspective.append(optimal_profit)

                print(f"    Optimal bid: {optimal_bid:.2f}, Optimal profit: {optimal_profit:.2f}")
            
            # Store the profit history
            self.profit_history_agent_perspective.append(profit_agent_perspective.copy())
            
            if self.iteration > 0:
                self.convergence_check_1 = []
                # Check convergence based on bid changes
                for i in range(self.num_generators):
                    self.convergence_check_1.append(self.check_convergence(
                        parameter_1 = self.profit_history_agent_perspective[self.iteration][i],
                        parameter_2 = self.profit_history_agent_perspective[self.iteration - 1][i]
                    ))

                # Check if all generators have converged                
                if all(self.convergence_check_1):
                    # Calculate market outcomes
                    dispatch, clearing_price, profits = self.calculate_ED()
                    
                    # Store results
                    self.dispatch_history.append(dispatch)
                    self.clearing_price_history.append(clearing_price)
                    self.profit_history_ED_perspective.append(profits.copy())

                    self.convergence_check_2 = []

                    for i in range(self.num_generators):
                        self.convergence_check_2.append(self.check_convergence(
                            parameter_1 = self.profit_history_ED_perspective[-1][i],
                            parameter_2 = self.profit_history_agent_perspective[self.iteration][i]
                        ))

                    if all(self.convergence_check_2):
                        print("Convergence achieved!")
                        self.results = self.get_results()
                        break
                
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
            Dictionary containing all results and history
        """
        
        final_dispatch, final_price, final_profits = self.calculate_ED()
        
        results = {
            "iterations": self.iteration,
            "final_bids": self.bid_vector.copy(),
            "costs": self.cost_vector.copy(),
            "bid_history": self.bid_history.copy(),
            "profit_history": self.profit_history_agent_perspective.copy(),
            "clearing_price_history": self.clearing_price_history.copy(),
            "final_market_outcomes": {
                "dispatch": final_dispatch,
                "clearing_price": final_price,
                "profits": final_profits,
                "total_welfare": sum(final_profits)
            },
        }
        
        return results
    
    def visualize_bid_evolution(self) -> None:
        """
        Visualize how bids evolve over iterations for each generator
        """
        if not self.bid_history:
            print("No bid history available for visualization")
            return
            
        # Create figure and axis
        plt.figure(figsize=(12, 8))
        
        # Plot bid evolution for each generator
        iterations = list(range(len(self.bid_history)))
        
        for gen_id in range(self.num_generators):
            bids_over_time = [self.bid_history[iter][gen_id] for iter in range(len(self.bid_history))]
            plt.plot(iterations, bids_over_time, marker='o', linewidth=2, 
                    label=f'Generator {gen_id} (Cost: ${self.cost_vector[gen_id]:.1f})')
        
        # Add formatting
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Bid ($/MWh)', fontsize=12)
        plt.title('Bid Evolution Over Iterations', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        # Also create a summary table
        print("\n=== Bid Evolution Summary ===")
        print(f"{'Iter':<4} {'Gen 0':<8} {'Gen 1':<8} {'Gen 2':<8} {'Gen 3':<8} {'Gen 4':<8} {'Gen 5':<8}")
        print("-" * 52)
        for i, bids in enumerate(self.bid_history):
            bid_str = " ".join([f"{bid:7.2f}" for bid in bids])
            print(f"{i:<4} {bid_str}")
    
    def visualize_supply_demand_curve(self) -> None:
        """
        Visualize the supply-demand curve with market clearing point
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for visualization")
            return
            
        final_bids = self.results['final_bids']
        final_dispatch = self.results['final_market_outcomes']['dispatch']
        clearing_price = self.results['final_market_outcomes']['clearing_price']
        
        # Create merit order (sort generators by bid price)
        gen_data = [(i, final_bids[i], self.pmax_list[i], final_dispatch[i]) for i in range(self.num_generators)]
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
        demand_level = sum(self.demand) if isinstance(self.demand, list) else self.demand
        max_price = max(supply_prices) * 1.1
        plt.axvline(x=demand_level, color='red', linewidth=2.5, 
                   label=f'Demand ({demand_level:.0f} MW)', alpha=0.8)
        
        # Mark market clearing point
        plt.scatter([demand_level], [clearing_price], color='green', s=150, 
                   zorder=5, label=f'Market Clearing\n(Price: ${clearing_price:.2f}/MWh)')
        
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
        plt.title('Supply-Demand Curve and Market Clearing', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        plt.xlim(0, max(cumulative_capacity, demand_level) * 1.1)
        plt.ylim(0, max_price)
        
        plt.tight_layout()
        plt.show()
        
        # Print dispatch summary
        print("\n=== Market Dispatch Summary ===")
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
        print(f"Market Clearing Price: ${clearing_price:.2f}/MWh")
    
    def visualize_agent_profits(self) -> None:
        """
        Visualize agent profits over iterations and compare with perfect competition
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for visualization")
            return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Profit evolution over iterations
        if len(self.profit_history_agent_perspective) > 1:
            iterations = list(range(len(self.profit_history_agent_perspective)))
            
            for gen_id in range(self.num_generators):
                profits_over_time = [self.profit_history_agent_perspective[iter][gen_id] 
                                   for iter in range(len(self.profit_history_agent_perspective))]
                ax1.plot(iterations, profits_over_time, marker='o', linewidth=2, 
                        label=f'Generator {gen_id}')
            
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
        final_profits = self.results['final_market_outcomes']['profits']
        
        # Calculate perfect competition profits (bidding at marginal cost)
        perfect_comp_dispatch, perfect_comp_price = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand=self.demand,
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector  # Bid at marginal cost
        )
        
        perfect_comp_profits = []
        for i in range(self.num_generators):
            revenue = perfect_comp_price * perfect_comp_dispatch[i]
            cost = self.cost_vector[i] * perfect_comp_dispatch[i]
            perfect_comp_profits.append(revenue - cost)
        
        # Create bar chart
        x_pos = np.arange(self.num_generators)
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
        
        ax2.set_xlabel('Generator ID', fontsize=12)
        ax2.set_ylabel('Profit ($)', fontsize=12)
        ax2.set_title('Final Profits vs Perfect Competition', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Gen {i}' for i in range(self.num_generators)])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed profit analysis
        print("\n=== Profit Analysis ===")
        print(f"{'Gen':<4} {'Cost':<8} {'Final Bid':<10} {'Dispatch':<10} {'Equil Profit':<12} {'Perfect Comp':<12} {'Markup':<8}")
        print("-" * 80)
        
        total_equil_profit = 0
        total_perfect_profit = 0
        
        for i in range(self.num_generators):
            final_dispatch = self.results['final_market_outcomes']['dispatch'][i]
            markup = ((self.results['final_bids'][i] - self.cost_vector[i]) / self.cost_vector[i] * 100) if self.cost_vector[i] > 0 else 0
            
            print(f"{i:<4} ${self.cost_vector[i]:<7.1f} ${self.results['final_bids'][i]:<9.2f} "
                  f"{final_dispatch:<10.1f} ${final_profits[i]:<11.0f} ${perfect_comp_profits[i]:<11.0f} {markup:<7.1f}%")
            
            total_equil_profit += final_profits[i]
            total_perfect_profit += perfect_comp_profits[i]
        
        print("-" * 80)
        print(f"Total Welfare - Equilibrium: ${total_equil_profit:.0f}")
        print(f"Total Welfare - Perfect Comp: ${total_perfect_profit:.0f}")
        print(f"Welfare Loss: ${total_perfect_profit - total_equil_profit:.0f} ({((total_perfect_profit - total_equil_profit)/total_perfect_profit*100):.1f}%)")
        
        # Market power analysis
        print(f"\nMarket Clearing Price - Equilibrium: ${self.results['final_market_outcomes']['clearing_price']:.2f}/MWh")
        print(f"Market Clearing Price - Perfect Comp: ${perfect_comp_price:.2f}/MWh")
        price_markup = ((self.results['final_market_outcomes']['clearing_price'] - perfect_comp_price) / perfect_comp_price * 100)
        print(f"Price Markup: {price_markup:.1f}%")
    
    def analyze_competitive_benchmark(self) -> None:
        """
        Analyze what the competitive (perfect competition) outcome should be
        """
        print("\n=== Competitive Benchmark Analysis ===")
        
        # Calculate competitive dispatch (bidding at marginal cost)
        comp_dispatch, comp_price = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand=self.demand,
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector  # Bid at marginal cost
        )
        
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
        comp_dispatch, comp_price = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand=self.demand,
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector  # Bid at marginal cost
        )
        
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
    
    def visualize_merit_order_comparison(self) -> None:
        """
        Visualize merit order curves for both competitive (cost-based) and strategic (bid-based) dispatch
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for merit order comparison")
            return
            
        # Get competitive and strategic data
        comp_dispatch, comp_price = self.ed_model.economic_dispatch(
            num_generators=self.num_generators,
            demand=self.demand,
            Pmax=self.pmax_list,
            Pmin=self.pmin_list,
            bid_list=self.cost_vector
        )
        
        strategic_dispatch = self.results['final_market_outcomes']['dispatch']
        strategic_price = self.results['final_market_outcomes']['clearing_price']
        strategic_bids = self.results['final_bids']
        
        # Create merit order for competitive case (sorted by costs)
        comp_gen_data = [(i, self.cost_vector[i], self.pmax_list[i], comp_dispatch[i]) 
                        for i in range(self.num_generators)]
        comp_gen_data.sort(key=lambda x: x[1])  # Sort by cost
        
        # Create merit order for strategic case (sorted by bids)
        strategic_gen_data = [(i, strategic_bids[i], self.pmax_list[i], strategic_dispatch[i]) 
                             for i in range(self.num_generators)]
        strategic_gen_data.sort(key=lambda x: x[1])  # Sort by bid
        
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
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Plot both supply curves
        plt.step(supply_quantities_comp, supply_prices_comp, where='post', linewidth=3, 
                color='blue', label='Competitive Merit Order (Costs)', alpha=0.8)
        plt.step(supply_quantities_strategic, supply_prices_strategic, where='post', linewidth=3, 
                color='red', label='Strategic Merit Order (Bids)', alpha=0.8, linestyle='--')
        
        # Plot demand curve
        demand_level = sum(self.demand) if isinstance(self.demand, list) else self.demand
        max_price = max(max(supply_prices_comp), max(supply_prices_strategic)) * 1.1
        plt.axvline(x=demand_level, color='green', linewidth=3, 
                   label=f'Demand ({demand_level:.0f} MW)', alpha=0.8)
        
        # Mark market clearing points
        plt.scatter([demand_level], [comp_price], color='blue', s=200, marker='o',
                   zorder=5, label=f'Competitive Clearing\n(${comp_price:.2f}/MWh)', edgecolor='black')
        plt.scatter([demand_level], [strategic_price], color='red', s=200, marker='s',
                   zorder=5, label=f'Strategic Clearing\n(${strategic_price:.2f}/MWh)', edgecolor='black')
        
        # Add annotations for key generators with improved spacing
        # Competitive merit order annotations
        cumulative_comp = 0
        comp_annotation_count = 0
        for gen_id, cost, pmax, dispatch in comp_gen_data:
            if dispatch > 0.1:  # Only annotate dispatched units
                # Alternate positioning and add horizontal offset to avoid overlap
                vertical_offset = max_price * (0.12 + 0.04 * (comp_annotation_count % 3))
                horizontal_offset = 5 * (comp_annotation_count % 2 - 0.5)  # Alternate left/right
                
                plt.annotate(f'Gen {gen_id}\nCost: ${cost:.1f}', 
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
        for gen_id, bid, pmax, dispatch in strategic_gen_data:
            if dispatch > 0.1:  # Only annotate dispatched units
                # Only annotate if significantly different from cost
                if abs(bid - self.cost_vector[gen_id]) > 1.0:
                    # Alternate positioning with better spacing below the curve
                    vertical_offset = max_price * (0.12 + 0.04 * (strat_annotation_count % 3))
                    horizontal_offset = 8 * (strat_annotation_count % 2 - 0.5)  # Alternate left/right
                    
                    plt.annotate(f'Gen {gen_id}\nBid: ${bid:.1f}', 
                               xy=(cumulative_strategic + dispatch/2, bid),
                               xytext=(cumulative_strategic + dispatch/2 + horizontal_offset, bid - vertical_offset),
                               ha='center', va='top', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5))
                    strat_annotation_count += 1
            cumulative_strategic += dispatch
        
        # Formatting
        plt.xlabel('Cumulative Capacity (MW)', fontsize=14)
        plt.ylabel('Price ($/MWh)', fontsize=14)
        plt.title('Merit Order Comparison: Competitive vs Strategic Bidding', fontsize=16, fontweight='bold')
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        max_capacity = max(cumulative_capacity_comp, cumulative_capacity_strategic)
        plt.xlim(0, max(max_capacity, demand_level) * 1.1)
        plt.ylim(0, max_price)
        
        plt.tight_layout()
        plt.show()
        
        # Print merit order comparison
        print("\n=== Merit Order Comparison ===")
        print("Competitive Merit Order (by Cost):")
        print(f"{'Rank':<4} {'Gen':<4} {'Cost':<8} {'Capacity':<8} {'Dispatch':<8}")
        print("-" * 40)
        for rank, (gen_id, cost, pmax, dispatch) in enumerate(comp_gen_data, 1):
            print(f"{rank:<4} {gen_id:<4} ${cost:<7.2f} {pmax:<8.0f} {dispatch:<8.0f}")
        
        print(f"\nStrategic Merit Order (by Bid):")
        print(f"{'Rank':<4} {'Gen':<4} {'Bid':<8} {'Capacity':<8} {'Dispatch':<8} {'Cost':<8}")
        print("-" * 50)
        for rank, (gen_id, bid, pmax, dispatch) in enumerate(strategic_gen_data, 1):
            cost = self.cost_vector[gen_id]
            print(f"{rank:<4} {gen_id:<4} ${bid:<7.2f} {pmax:<8.0f} {dispatch:<8.0f} ${cost:<7.2f}")
        
        # Efficiency analysis
        print(f"\n=== Merit Order Efficiency Analysis ===")
        print(f"Competitive price: ${comp_price:.2f}/MWh")
        print(f"Strategic price: ${strategic_price:.2f}/MWh")
        print(f"Price increase: ${strategic_price - comp_price:.2f}/MWh ({((strategic_price/comp_price - 1)*100):+.1f}%)")
        
        # Check if merit order changed
        comp_order = [gen_id for gen_id, _, _, dispatch in comp_gen_data if dispatch > 0.1]
        strategic_order = [gen_id for gen_id, _, _, dispatch in strategic_gen_data if dispatch > 0.1]
        
        if comp_order != strategic_order:
            print("⚠️  Merit order CHANGED due to strategic bidding!")
            print(f"Competitive dispatch order: {comp_order}")
            print(f"Strategic dispatch order: {strategic_order}")
        else:
            print("✓ Merit order UNCHANGED (only prices affected)")


if __name__ == "__main__":
    print("=== Testing Best Response Algorithm ===")
    
    # Create and run algorithm
    algo = BestResponseAlgorithm(case = "test_case")
    algo.run()
    results = algo.results
    
    # Display final results
    print(f"\n=== Final Results ===")
    print(f"Iterations: {results['iterations']}")
    print(f"Final bids: {[f'{b:.2f}' for b in results['final_bids']]}")
    print(f"Generator costs: {[f'{c:.2f}' for c in results['costs']]}")
    print(f"Final market price: ${results['final_market_outcomes']['clearing_price']:.2f}/MWh")
    print(f"Final dispatch: {[f'{d:.1f}' for d in results['final_market_outcomes']['dispatch']]} MW")
    print(f"Final profits: {[f'{p:.2f}' for p in results['final_market_outcomes']['profits']]}")
    print(f"Total welfare: ${results['final_market_outcomes']['total_welfare']:.2f}")
    
    # Visualize bid evolution
    print("\n=== Visualizing Bid Evolution ===")
    algo.visualize_bid_evolution()
    
    # Visualize supply-demand curve
    print("\n=== Visualizing Supply-Demand Curve ===")
    algo.visualize_supply_demand_curve()
    
    # Visualize agent profits
    print("\n=== Visualizing Agent Profits ===")
    algo.visualize_agent_profits()
    
    # Analyze competitive benchmark
    print("\n=== Competitive Benchmark Analysis ===")
    algo.analyze_competitive_benchmark()
    
    # Compare dispatch formulations
    print("\n=== Comparing Economic Dispatch vs MPEC Formulations ===")
    algo.compare_dispatch_formulations()
    
    # Visualize merit order comparison
    print("\n=== Merit Order Curve Comparison ===")
    algo.visualize_merit_order_comparison()