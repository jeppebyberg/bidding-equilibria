import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add parent directory to path for config imports
sys.path.append(str(Path(__file__).parent.parent))
from models.diagonalization.stochastic_MPEC import StochasticMPECModel
from models.diagonalization.economic_dispatch import EconomicDispatchModel
from config.utils.diagonalization_loader import load_diagonalization

class StochasticBestResponseAlgorithm:
    """
    Stochastic best response algorithm where generators optimize expected profit across scenarios
    """
    
    def __init__(self, case_name: str = "stochastic_demand_case"):
        """
        Initialize the stochastic best response algorithm
        
        Parameters
        ----------
        case_name : str
            Name of the stochastic case to load from cases.yaml
        """
        self.case_name = case_name
        self.load_case_config()
        
        config = load_diagonalization()
        
        # Algorithm parameters
        self.max_iterations = int(config.get("max_iterations"))
        self.conv_tolerance = float(config.get("conv_tolerance"))
        
        # Create economic dispatch model for comparisons
        self.ed_model = EconomicDispatchModel()
        
        # Initialize bid vector with true costs
        self.bid_vector = self.cost_vector.copy()
        
        # Algorithm state
        self.iteration = 0
        self.converged = False
        
        # History tracking
        self.bid_history = []
        self.expected_profit_history = []
        self.scenario_outcomes_history = []
        
    def load_case_config(self):
        """Load stochastic case configuration from cases.yaml"""
        config_path = Path(__file__).parent.parent / "config" / "cases.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cases = yaml.safe_load(f)
            
        if self.case_name not in cases:
            raise ValueError(f"Case '{self.case_name}' not found in cases.yaml")
            
        case_config = cases[self.case_name]
        
        # Verify this is a stochastic case
        if case_config.get("optimization_type") != "stochastic":
            raise ValueError(f"Case '{self.case_name}' is not configured for stochastic optimization")
        
        # Extract generator data
        generators = case_config["generators"]
        self.num_generators = len(generators)
        self.pmax_list = [gen["pmax"] for gen in generators]
        self.pmin_list = [gen["pmin"] for gen in generators] 
        self.cost_vector = [gen["cost"] for gen in generators]
        
        # Extract scenarios with probabilities
        self.scenarios = case_config["scenarios"]
        self.num_scenarios = len(self.scenarios)
        
        # Validate probabilities
        total_prob = sum(s['probability'] for s in self.scenarios)
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Scenario probabilities must sum to 1.0, got {total_prob}")
        
        print(f"Loaded stochastic case with {self.num_scenarios} scenarios:")
        for scenario in self.scenarios:
            print(f"  - {scenario['name']}: {scenario['demand'][0]:.0f} MW (prob: {scenario['probability']:.2f})")
    
    def solve_strategic_player_problem(self, strategic_player: int) -> Tuple[float, float]:
        """
        Solve stochastic MPEC optimization for a strategic player
        
        Parameters
        ----------
        strategic_player : int
            Index of the strategic generator
            
        Returns
        -------
        tuple
            (optimal_bid, expected_profit)
        """
        
        # Create stochastic MPEC model
        stochastic_mpec = StochasticMPECModel(
            scenarios=self.scenarios,
            pmax_list=self.pmax_list,
            pmin_list=self.pmin_list,
            num_generators=self.num_generators,
            strategic_player=strategic_player,
            cost_vector=self.cost_vector
        )
        
        # Build and solve model
        stochastic_mpec.build_model(self.bid_vector)
        
        if stochastic_mpec.solve():
            solution = stochastic_mpec.get_solution()
            return solution['optimal_bid'], solution['expected_profit']
        else:
            print(f"Warning: No solution found for strategic player {strategic_player}")
            return self.bid_vector[strategic_player], 0.0
    
    def calculate_scenario_outcomes(self) -> List[Dict[str, Any]]:
        """
        Calculate market outcomes for each scenario given current bids
        
        Returns
        -------
        list
            List of scenario outcomes with dispatch and profits
        """
        scenario_outcomes = []
        
        for scenario in self.scenarios:
            demand = scenario['demand']
            
            # Calculate economic dispatch for this scenario
            dispatch, clearing_price = self.ed_model.economic_dispatch(
                num_generators=self.num_generators,
                demand=demand,
                Pmax=self.pmax_list,
                Pmin=self.pmin_list,
                bid_list=self.bid_vector
            )
            
            # Calculate profits for each generator
            profits = []
            for i in range(self.num_generators):
                revenue = clearing_price * dispatch[i]
                cost = self.cost_vector[i] * dispatch[i]
                profit = revenue - cost
                profits.append(profit)
            
            scenario_outcomes.append({
                'scenario_name': scenario['name'],
                'demand': scenario['demand'][0],
                'probability': scenario['probability'],
                'clearing_price': clearing_price,
                'dispatch': dispatch,
                'profits': profits
            })
        
        return scenario_outcomes
    
    def calculate_expected_profits(self, scenario_outcomes: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate expected profits for each generator across scenarios
        
        Parameters
        ----------
        scenario_outcomes : list
            List of scenario outcomes
            
        Returns
        -------
        list
            Expected profits for each generator
        """
        expected_profits = [0.0] * self.num_generators
        
        for outcome in scenario_outcomes:
            probability = outcome['probability']
            profits = outcome['profits']
            
            for i in range(self.num_generators):
                expected_profits[i] += probability * profits[i]
        
        return expected_profits
    
    def check_convergence(self, parameter_1: float, parameter_2: float) -> bool:
        """
        Check if the algorithm has converged
        
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        if abs(parameter_1 - parameter_2) <= self.conv_tolerance * abs(parameter_2) if parameter_2 != 0 else self.conv_tolerance:
            return True
        else: 
            return False
    
    def run(self) -> Dict[str, Any]:
        """
        Run the stochastic best response algorithm
        
        Returns
        -------
        dict
            Dictionary containing results and convergence information
        """
        
        print("=== Starting Stochastic Best Response Algorithm ===")
        print(f"Number of generators: {self.num_generators}")
        print(f"Number of scenarios: {self.num_scenarios}")
        print(f"Initial bids: {[f'{b:.2f}' for b in self.bid_vector]}")
        print(f"Generator costs: {[f'{c:.2f}' for c in self.cost_vector]}")
        
        self.iteration = 0
        
        while self.iteration < self.max_iterations:
            print(f"\n--- Iteration {self.iteration + 1} ---")
            
            # Store current bids
            self.bid_history.append(self.bid_vector.copy())
            
            # Update each generator's bid using stochastic optimization
            expected_profits = []

            for strategic_player in range(self.num_generators):
                print(f"  Solving stochastic MPEC for strategic player {strategic_player}...")
                
                # Solve stochastic MPEC problem for this player
                optimal_bid, expected_profit = self.solve_strategic_player_problem(strategic_player)
                
                # Update this player's bid immediately (Gauss-Seidel style)
                self.bid_vector[strategic_player] = optimal_bid
                expected_profits.append(expected_profit)

                print(f"    Optimal bid: ${optimal_bid:.2f}/MWh, Expected profit: ${expected_profit:.2f}")
            
            # Store expected profit history
            self.expected_profit_history.append(expected_profits.copy())
            
            # Calculate scenario outcomes with current bids
            scenario_outcomes = self.calculate_scenario_outcomes()
            self.scenario_outcomes_history.append(scenario_outcomes)
            
            # Check convergence based on expected profit changes
            if self.iteration > 0:
                convergence_checks = []
                for i in range(self.num_generators):
                    converged = self.check_convergence(
                        self.expected_profit_history[self.iteration][i],
                        self.expected_profit_history[self.iteration - 1][i]
                    )
                    convergence_checks.append(converged)
                
                all_converged = all(convergence_checks)
                
                print(f"  Convergence check: {'Passed' if all_converged else 'Failed'}")
                print(f"  Updated bids: {[f'{b:.2f}' for b in self.bid_vector]}")
                
                if all_converged:
                    self.converged = True
                    print(f"  *** Converged after {self.iteration + 1} iterations ***")
                    break
                
            # Increment iteration counter
            self.iteration += 1
        
        # Final results
        if not self.converged:
            print(f"\nAlgorithm stopped after {self.max_iterations} iterations without convergence")
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get algorithm results
        
        Returns
        -------
        dict
            Dictionary containing all results and history
        """
        
        # Calculate final scenario outcomes
        final_scenario_outcomes = self.calculate_scenario_outcomes()
        final_expected_profits = self.calculate_expected_profits(final_scenario_outcomes)
        
        results = {
            "converged": self.converged,
            "iterations": self.iteration,
            "final_bids": self.bid_vector.copy(),
            "costs": self.cost_vector.copy(),
            "bid_history": self.bid_history.copy(),
            "expected_profit_history": self.expected_profit_history.copy(),
            "scenario_outcomes_history": self.scenario_outcomes_history.copy(),
            "final_scenario_outcomes": final_scenario_outcomes,
            "final_expected_profits": final_expected_profits,
            "scenarios": self.scenarios,
            "algorithm_parameters": {
                "max_iterations": self.max_iterations,
                "conv_tolerance": self.conv_tolerance,
                "num_generators": self.num_generators,
                "num_scenarios": self.num_scenarios
            }
        }
        
        return results
    
    def visualize_stochastic_results(self) -> None:
        """
        Visualize stochastic optimization results
        """
        if not hasattr(self, 'results') or not self.results:
            print("No results available for visualization")
            return
        
        results = self.results
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Expected Profit Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        if len(results['expected_profit_history']) > 1:
            iterations = list(range(len(results['expected_profit_history'])))
            for gen_id in range(self.num_generators):
                profits = [results['expected_profit_history'][iter][gen_id] for iter in iterations]
                ax1.plot(iterations, profits, marker='o', label=f'Gen {gen_id}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Expected Profit ($)')
        ax1.set_title('Expected Profit Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bid Evolution
        ax2 = fig.add_subplot(gs[0, 1])
        if len(results['bid_history']) > 1:
            iterations = list(range(len(results['bid_history'])))
            for gen_id in range(self.num_generators):
                bids = [results['bid_history'][iter][gen_id] for iter in iterations]
                ax2.plot(iterations, bids, marker='o', label=f'Gen {gen_id} (Cost: ${self.cost_vector[gen_id]:.1f})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Bid ($/MWh)')
        ax2.set_title('Bid Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final Expected Profits vs Costs
        ax3 = fig.add_subplot(gs[0, 2])
        x_pos = np.arange(self.num_generators)
        bars1 = ax3.bar(x_pos - 0.2, results['costs'], 0.4, label='Marginal Costs', alpha=0.7)
        bars2 = ax3.bar(x_pos + 0.2, results['final_expected_profits'], 0.4, label='Expected Profits', alpha=0.7)
        ax3.set_xlabel('Generator ID')
        ax3.set_ylabel('$/MWh or $')
        ax3.set_title('Costs vs Expected Profits')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Gen {i}' for i in range(self.num_generators)])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-9. Scenario-specific outcomes (6 scenarios, 2x3 grid)
        scenario_axes = [fig.add_subplot(gs[r, c]) for r in range(1, 3) for c in range(3)]
        
        for i, (ax, scenario_outcome) in enumerate(zip(scenario_axes, results['final_scenario_outcomes'])):
            if i < len(results['final_scenario_outcomes']):
                # Bar chart of dispatch for this scenario
                dispatch = scenario_outcome['dispatch']
                profits = scenario_outcome['profits']
                
                x_pos = np.arange(self.num_generators)
                bars = ax.bar(x_pos, dispatch, alpha=0.7)
                
                # Color bars by profit
                for bar, profit in zip(bars, profits):
                    if profit > 0:
                        bar.set_color('green')
                    else:
                        bar.set_color('red')
                
                ax.set_title(f"{scenario_outcome['scenario_name']}\\n"
                           f"Demand: {scenario_outcome['demand']:.0f} MW, "
                           f"Price: ${scenario_outcome['clearing_price']:.2f}/MWh\\n"
                           f"Prob: {scenario_outcome['probability']:.2f}")
                ax.set_xlabel('Generator')
                ax.set_ylabel('Dispatch (MW)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'G{i}' for i in range(self.num_generators)])
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')  # Hide unused subplots
        
        plt.suptitle('Stochastic Best Response Algorithm Results', fontsize=16, fontweight='bold')
        plt.show()
        
        # Print summary
        self.print_stochastic_summary()
    
    def print_stochastic_summary(self):
        """Print summary of stochastic results"""
        if not hasattr(self, 'results'):
            return
            
        results = self.results
        
        print("\n=== Stochastic Best Response Summary ===")
        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Final bids: {[f'${b:.2f}' for b in results['final_bids']]}")
        print(f"Generator costs: {[f'${c:.2f}' for c in results['costs']]}")
        print(f"Expected profits: {[f'${p:.2f}' for p in results['final_expected_profits']]}")
        
        print("\\n=== Scenario Outcomes ===")
        for outcome in results['final_scenario_outcomes']:
            print(f"{outcome['scenario_name']}: "
                  f"Price ${outcome['clearing_price']:.2f}/MWh, "
                  f"Total dispatch {sum(outcome['dispatch']):.0f} MW")


if __name__ == "__main__":
    print("=== Testing Stochastic Best Response Algorithm ===")
    
    # Create and run algorithm
    algo = StochasticBestResponseAlgorithm("stochastic_demand_case")
    algo.results = algo.run()
    
    # Visualize results
    print("\\n=== Visualizing Results ===")
    algo.visualize_stochastic_results()