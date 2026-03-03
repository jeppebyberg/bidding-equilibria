"""
Scenario Management Module

This module provides comprehensive utilities for:
1. Loading base case setup and configuration data
2. Referencing the base case setup for analysis
3. Generating demand scenarios based on the base case:
   - Linear demand scenarios (linearly spaced between min/max factors of base demand)
   - Custom demand scenarios (from user-provided demand lists)
4. Future extensibility for scenarios affecting generator capacities and costs
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from config.base_case.utils.cases_utils import load_setup_data

class ScenarioManager:
    """Unified scenario management for loading base cases and generating scenarios."""
    
    def __init__(self, base_case_reference: str = "test_case"):
        """Initialize the scenario manager.
        
        Args:
            base_case_reference: Default reference case name from reference_cases.yaml
        """
        self.base_case_reference = base_case_reference
        
        # Load base case once and reuse
        try:
            num_generators, pmax_list, pmin_list, cost_vector, demand, generators, players = load_setup_data(self.base_case_reference)
            
            self.base_case = {
                'case_name': self.base_case_reference,
                'num_generators': num_generators,
                'generators': generators,
                'players': players,
                'demand': demand,
                'pmax_list': pmax_list,
                'pmin_list': pmin_list,
                'cost_vector': cost_vector
            }
            
            # Validate the loaded base case
            if not self.validate_base_setup():
                raise ValueError(f"Base case '{self.base_case_reference}' failed validation checks")
                
                    
            self.players_config = self.get_players_config()


        except Exception as e:
            raise ValueError(f"Failed to load base case '{self.base_case_reference}': {e}")
    
    def get_base_setup_reference(self) -> Dict[str, Any]:
        """
        Get a reference to the base setup that can be used by scenario analysis.
        
        Returns:
            Dictionary containing reference information to the base setup
        """
        return {
            'type': 'base_reference',
            'base_case_name': self.base_case_reference,
            'reference_file': 'config/base_case/reference_cases.yaml',
            'setup_summary': {
                'num_generators': self.base_case['num_generators'],
                'total_capacity': sum(self.base_case['pmax_list']),
                'demand': self.base_case['demand']
            }
        }
    
    def get_players_config(self) -> List[Dict[str, Any]]:
        """
        Get the players configuration from the loaded base case.
        
        Returns:
            List of player configurations, each containing 'name' and 'controlled_generators'
        """
        return self.base_case['players']
    
    def validate_base_setup(self) -> bool:
        """Validate that the base setup is properly configured."""
        try:
            # Basic validation checks
            if self.base_case['num_generators'] <= 0:
                return False
            if self.base_case['demand'] <= 0:
                return False
            total_capacity = sum(self.base_case['pmax_list'])
            if total_capacity <= self.base_case['demand']:
                print(f"Warning: Total capacity ({total_capacity}) is less than or equal to demand ({self.base_case['demand']})")
            
            return True
        except Exception:
            return False
    
    def generate_demand_scenarios(
        self,
        scenario_type: str = "linear",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate complete scenarios combining base case with demand scenarios.
        
        Args:
            scenario_type: Type of demand scenarios ('linear' or 'custom')
            **kwargs: Additional parameters for scenario generation
                For 'linear': num_scenarios, min_factor, max_factor
                For 'custom': demand_list
        
        Returns:
            Dictionary containing complete scenario configuration
        """
        # Generate demand scenarios based on type
        if scenario_type == "linear":
            demand_scenarios = self._generate_linear_demand_scenarios(**kwargs)
        elif scenario_type == "custom":
            demand_scenarios = self._generate_custom_demand_scenarios(**kwargs)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}. Use 'linear' or 'custom'")
        
        # Create complete scenario configuration
        return demand_scenarios

    def _generate_linear_demand_scenarios(
        self,
        num_scenarios: int = 6,
        min_factor: float = 0.7,
        max_factor: float = 1.3
    ) -> List[float]:
        """Generate linear demand scenarios based on reference case."""
        base_demand = self.base_case['demand']
        
        min_demand = base_demand * min_factor
        max_demand = base_demand * max_factor
        
        demand_scenarios = np.linspace(min_demand, max_demand, num_scenarios)
        return [float(d) for d in demand_scenarios]
    
    def _generate_custom_demand_scenarios(self, demand_list: List[float]) -> List[float]:
        """Generate custom demand scenarios."""
        if not demand_list:
            raise ValueError("Demand list cannot be empty for custom scenarios")
        
        if any(d <= 0 for d in demand_list):
            raise ValueError("All demand values must be positive")
        
        return [float(d) for d in demand_list]

    def generate_capacity_scenarios(
        self,
        scenario_type: str = "linear",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate complete scenarios combining base case with capacity scenarios.
        
        Only wind generators (W prefix) have variable capacity between 0 and pmax.
        Conventional generators (G prefix) maintain their nominal capacity.
        
        Args:
            scenario_type: Type of capacity scenarios ('linear' or 'custom')
            **kwargs: Additional parameters for scenario generation
                For 'linear': num_scenarios, min_factor (default 0.0), max_factor (default 1.0)
                For 'custom': wind_factor_list (list of factors between 0 and 1)
        
        Returns:
            Dictionary containing complete scenario configuration
        """
        # Generate capacity scenarios based on type
        if scenario_type == "linear":
            capacity_scenarios = self._generate_linear_capacity_scenarios(**kwargs)
        elif scenario_type == "custom":
            capacity_scenarios = self._generate_custom_capacity_scenarios(**kwargs)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}. Use 'linear' or 'custom'")
        
        # Create complete scenario configuration
        return capacity_scenarios
    
    def _generate_linear_capacity_scenarios(
        self,
        num_scenarios: int = 6,
        min_factor: float = 0.0,
        max_factor: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Generate linear capacity scenarios based on reference case.
        
        Only wind generators (W prefix) have variable capacity between 0 and pmax.
        Conventional generators (G prefix) maintain their nominal capacity.
        """
        base_capacity_list = self.base_case['pmax_list']
        generators = self.base_case['generators']
        
        # Check if there are any wind generators
        has_wind_generators = any(
            (generator['name'] if isinstance(generator, dict) else generator).startswith('W') 
            for generator in generators
        )
        
        # If no wind generators, return single scenario with base capacities
        if not has_wind_generators:
            return [{
                'wind_capacity_factor': 1.0,
                'capacity_list': [float(cap) for cap in base_capacity_list],
                'total_capacity': float(sum(base_capacity_list)),
                'wind_capacity': 0.0,
                'conventional_capacity': float(sum(base_capacity_list))
            }]
        
        # Generate scaling factors for wind generators
        wind_factors = np.linspace(min_factor, max_factor, num_scenarios)
        
        # Create capacity scenarios
        capacity_scenarios = []
        for factor in wind_factors:
            scenario_capacities = []
            for i, (generator, base_cap) in enumerate(zip(generators, base_capacity_list)):
                gen_name = generator['name'] if isinstance(generator, dict) else generator
                if gen_name.startswith('W'):  # Wind generator - apply scaling
                    new_capacity = float(base_cap * factor)
                else:  # Conventional generator (G) - keep nominal capacity
                    new_capacity = float(base_cap)
                scenario_capacities.append(new_capacity)
            
            capacity_scenarios.append({
                'wind_capacity_factor': float(factor),
                'capacity_list': scenario_capacities,
                'total_capacity': sum(scenario_capacities),
                'wind_capacity': sum([cap for i, cap in enumerate(scenario_capacities) 
                                    if (generators[i]['name'] if isinstance(generators[i], dict) else generators[i]).startswith('W')]),
                'conventional_capacity': sum([cap for i, cap in enumerate(scenario_capacities) 
                                            if (generators[i]['name'] if isinstance(generators[i], dict) else generators[i]).startswith('G')])
            })
        
        return capacity_scenarios
    
    def _generate_custom_capacity_scenarios(
        self, 
        wind_factor_list: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate custom capacity scenarios.
        
        Only wind generators (W prefix) have variable capacity.
        Conventional generators (G prefix) maintain their nominal capacity.
        """
        if not wind_factor_list:
            raise ValueError("Wind factor list cannot be empty for custom scenarios")
        
        if any(f < 0 or f > 1 for f in wind_factor_list):
            raise ValueError("All wind capacity factors must be between 0 and 1")
        
        base_capacity_list = self.base_case['pmax_list']
        generators = self.base_case['generators']
        
        # Check if there are any wind generators
        has_wind_generators = any(
            (generator['name'] if isinstance(generator, dict) else generator).startswith('W') 
            for generator in generators
        )
        
        # If no wind generators, return single scenario with base capacities
        if not has_wind_generators:
            return [{
                'wind_capacity_factor': 1.0,
                'capacity_list': [float(cap) for cap in base_capacity_list],
                'total_capacity': float(sum(base_capacity_list)),
                'wind_capacity': 0.0,
                'conventional_capacity': float(sum(base_capacity_list))
            }]
        
        # Create capacity scenarios
        capacity_scenarios = []
        for factor in wind_factor_list:
            scenario_capacities = []
            for i, (generator, base_cap) in enumerate(zip(generators, base_capacity_list)):
                gen_name = generator['name'] if isinstance(generator, dict) else generator
                if gen_name.startswith('W'):  # Wind generator - apply scaling
                    new_capacity = float(base_cap * factor)
                else:  # Conventional generator (G) - keep nominal capacity
                    new_capacity = float(base_cap)
                scenario_capacities.append(new_capacity)
            
            capacity_scenarios.append({
                'wind_capacity_factor': float(factor),
                'capacity_list': scenario_capacities,
                'total_capacity': sum(scenario_capacities),
                'wind_capacity': sum([cap for i, cap in enumerate(scenario_capacities) 
                                    if (generators[i]['name'] if isinstance(generators[i], dict) else generators[i]).startswith('W')]),
                'conventional_capacity': sum([cap for i, cap in enumerate(scenario_capacities) 
                                            if (generators[i]['name'] if isinstance(generators[i], dict) else generators[i]).startswith('G')])
            })
        
        return capacity_scenarios
    
    def create_scenario_set(
        self,
        demand_scenarios: Optional[List[float]] = None,
        capacity_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a complete scenario set as cartesian product of demand and capacity scenarios.
        
        Args:
            demand_scenarios: List of demand values from generate_demand_scenarios
            capacity_scenarios: List of capacity scenario dictionaries from generate_capacity_scenarios
            
        Returns:
            Dictionary containing cartesian product of all scenario combinations
        """
        
        # Use provided scenarios or default to base case values
        if demand_scenarios is None:
            demand_scenarios = [self.base_case['demand']]
        
        if capacity_scenarios is None:
            # Use base capacity as single scenario
            capacity_scenarios = [{
                'wind_capacity_factor': 1.0,
                'capacity_list': self.base_case['pmax_list'],
                'total_capacity': sum(self.base_case['pmax_list']),
                'wind_capacity': sum([cap for i, cap in enumerate(self.base_case['pmax_list']) 
                                    if (self.base_case['generators'][i]['name'] if isinstance(self.base_case['generators'][i], dict) else self.base_case['generators'][i]).startswith('W')]),
                'conventional_capacity': sum([cap for i, cap in enumerate(self.base_case['pmax_list']) 
                                            if (self.base_case['generators'][i]['name'] if isinstance(self.base_case['generators'][i], dict) else self.base_case['generators'][i]).startswith('G')])
            }]
        
        # Create cartesian product of demand and capacity scenarios
        all_scenarios = []
        scenario_id = 1
        
        for demand in demand_scenarios:
            for capacity_scenario in capacity_scenarios:
                scenario = {
                    'scenario_id': scenario_id,
                    'demand': demand,
                    'capacity_info': capacity_scenario,
                    'total_capacity': capacity_scenario['total_capacity'],
                    'wind_capacity': capacity_scenario['wind_capacity'],
                    'conventional_capacity': capacity_scenario['conventional_capacity'],
                    'capacity_list': capacity_scenario['capacity_list']
                }
                all_scenarios.append(scenario)
                scenario_id += 1
        
        # Validate all scenarios and filter based on results
        validation_results = self.validate_scenario_set(all_scenarios, self.base_case)
        
        # Filter out invalid scenarios (those that failed N-1 or other checks)
        valid_scenarios = []
        invalid_scenarios = []
        
        for i, scenario in enumerate(all_scenarios):
            scenario_validation = validation_results['scenario_details'][i]
            if scenario_validation['is_valid']:
                valid_scenarios.append(scenario)
            else:
                invalid_scenarios.append({
                    **scenario,
                    'validation_errors': scenario_validation['errors']
                })
        
        # Re-number the valid scenarios sequentially and build separate DataFrames directly
        combined_scenarios = []
        scenarios_table = []  # For scenario data (dynamic)
        
        # Create costs DataFrame first (static data) - same for all scenarios
        cost_data = {}
        for j, (generator, _) in enumerate(zip(self.base_case['generators'], self.base_case['pmax_list'])):
            gen_name = generator['name'] if isinstance(generator, dict) else generator
            gen_cost = self.base_case['cost_vector'][j]
            cost_data[f'{gen_name}_cost'] = gen_cost
        costs_df = pd.DataFrame([cost_data])
        
        # Build scenarios DataFrame (dynamic data)
        for i, scenario in enumerate(valid_scenarios, 1):
            scenario['scenario_id'] = i
            combined_scenarios.append(scenario)
            
            # Create scenario row with only dynamic data
            scenario_row = {
                'scenario_id': i,
                'demand': scenario['demand'],
            }
            
            # Add individual generator capacities and bids (no costs)
            for j, (generator, capacity) in enumerate(zip(self.base_case['generators'], scenario['capacity_list'])):
                gen_name = generator['name'] if isinstance(generator, dict) else generator
                gen_cost = self.base_case['cost_vector'][j]
                
                # Add capacity and bid columns for each generator
                scenario_row[f'{gen_name}_cap'] = capacity
                scenario_row[f'{gen_name}_bid'] = gen_cost  # Initialize bid same as cost
            
            scenarios_table.append(scenario_row)
        
        # Convert scenarios table to DataFrame
        scenarios_df = pd.DataFrame(scenarios_table)
        
        # Create formatted description for printing
        wind_generators = [gen['name'] if isinstance(gen, dict) else gen for gen in self.base_case['generators'] if (gen['name'] if isinstance(gen, dict) else gen).startswith('W')]
        conventional_generators = [gen['name'] if isinstance(gen, dict) else gen for gen in self.base_case['generators'] if (gen['name'] if isinstance(gen, dict) else gen).startswith('G')]
        
        total_possible = len(all_scenarios)
        description_text = f"""
            === Scenario Set Summary (Comprehensive N-1 Contingency) ===
            Reference Case: {self.base_case_reference}
            Total Possible Scenarios: {total_possible}
            Valid Scenarios: {len(combined_scenarios)} (passed N-1 contingency and validation checks)
            Invalid Scenarios: {len(invalid_scenarios)}
            Demand Scenarios: {len(demand_scenarios)} (Range: {min(demand_scenarios):.1f} - {max(demand_scenarios):.1f} MW)
            Capacity Scenarios: {len(capacity_scenarios)} (Wind Factor Range: {min([cs['wind_capacity_factor'] for cs in capacity_scenarios]):.1f} - {max([cs['wind_capacity_factor'] for cs in capacity_scenarios]):.1f})
            Generators: {len(self.base_case['generators'])} total ({len(wind_generators)} wind, {len(conventional_generators)} conventional)
            Wind Generators: {', '.join(wind_generators) if wind_generators else 'None'}
            Conventional Generators: {', '.join(conventional_generators) if conventional_generators else 'None'}
        """.strip()
        
        # Update validation results with invalid scenario details
        validation_results['dismissed_details'] = invalid_scenarios
        validation_results['dismissed_scenarios'] = len(invalid_scenarios)

        return {
            'description_text': description_text,
            'scenarios_df': scenarios_df,
            'costs_df': costs_df,
            'validation': validation_results
        }
        
    def validate_scenario_set(self, all_scenarios: List[Dict[str, Any]], base_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate each scenario in the scenario set for feasibility and data integrity.
        
        Args:
            all_scenarios: List of all scenario dictionaries to validate
            base_case: Base case configuration for reference
            
        Returns:
            Dictionary containing validation results and summary
        """
        validation_results = {
            'total_scenarios': len(all_scenarios),
            'dismissed_scenarios': 0,  # Will be updated after filtering
            'valid_scenarios': 0,
            'warnings': [],
            'errors': [],
            'scenario_details': []
        }
        
        for scenario in all_scenarios:
            scenario_validation = self._validate_single_scenario(scenario, base_case)
            validation_results['scenario_details'].append(scenario_validation)
            
            if scenario_validation['is_valid']:
                validation_results['valid_scenarios'] += 1
            
            # Collect errors and warnings
            validation_results['errors'].extend([
                f"Scenario {scenario['scenario_id']}: {error}" 
                for error in scenario_validation['errors']
            ])
            validation_results['warnings'].extend([
                f"Scenario {scenario['scenario_id']}: {warning}" 
                for warning in scenario_validation['warnings']
            ])
        
        # Add summary statistics
        validation_results['validation_success_rate'] = (
            validation_results['valid_scenarios'] / validation_results['total_scenarios'] * 100
            if validation_results['total_scenarios'] > 0 else 0
        )
        
        # Print validation summary
        print(f"\n=== Scenario Validation Summary ===")
        print(f"Total Possible Scenarios: {validation_results['total_scenarios']}")
        print(f"Valid Scenarios: {validation_results['valid_scenarios']}")
        if validation_results['dismissed_scenarios'] > 0:
            print(f"Invalid Scenarios: {validation_results['dismissed_scenarios']} (failed N-1 contingency or validation checks)")
        print(f"Success Rate: {validation_results['validation_success_rate']:.1f}%")
        
        # Show invalid scenario details if any
        if validation_results.get('dismissed_details', []):
            print(f"\nInvalid Scenarios:")
            for invalid in validation_results['dismissed_details'][:5]:  # Show first 5 invalid
                errors = invalid.get('validation_errors', ['Unknown error'])
                print(f"  • Scenario {invalid['scenario_id']}: {errors[0]}")
            if len(validation_results['dismissed_details']) > 5:
                print(f"  ... and {len(validation_results['dismissed_details']) - 5} more invalid scenarios")
        
        if validation_results['errors']:
            print(f"\nErrors ({len(validation_results['errors'])}):")
            for error in validation_results['errors'][:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(validation_results['errors']) > 5:
                print(f"  ... and {len(validation_results['errors']) - 5} more errors")
        
        return validation_results
    
    def _validate_single_scenario(self, scenario: Dict[str, Any], base_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single scenario for N-1 contingency and data integrity.
        
        Args:
            scenario: Single scenario dictionary
            base_case: Base case configuration for reference
            
        Returns:
            Dictionary containing validation results for the scenario
        """
        validation = {
            'scenario_id': scenario['scenario_id'],
            'is_valid': True,
            'errors': [],
            'warnings': [],  # Keep empty as requested
            'checks_performed': []
        }
        
        try:
            # Check 1: N-1 Contingency - demand must be met even without ANY single generator
            validation['checks_performed'].append('n_minus_1_contingency')
            capacity_list = scenario['capacity_list']
            total_capacity = scenario['total_capacity']
            demand = scenario['demand']
            
            for i, generator_capacity in enumerate(capacity_list):
                remaining_capacity = total_capacity - generator_capacity
                if demand > remaining_capacity:
                    # Find generator name for better error message
                    gen_name = f"Generator_{i+1}"  # Default name
                    if i < len(self.base_case['generators']):
                        generator = self.base_case['generators'][i]
                        gen_name = generator['name'] if isinstance(generator, dict) else generator
                    
                    validation['errors'].append(f"N-1 Infeasible: Demand ({demand:.1f} MW) exceeds remaining capacity ({remaining_capacity:.1f} MW) when {gen_name} ({generator_capacity:.1f} MW) is unavailable")
                    validation['is_valid'] = False
                    break  # Found first failing case, no need to check others
            
            # Check 2: Positive demand
            validation['checks_performed'].append('positive_demand')
            if scenario['demand'] <= 0:
                validation['errors'].append(f"Demand must be positive, got {scenario['demand']}")
                validation['is_valid'] = False
            
            # Check 3: Individual generator capacities
            validation['checks_performed'].append('generator_capacities')
            capacity_list = scenario['capacity_list']
            base_pmax_list = self.base_case['pmax_list']
            generators = self.base_case['generators']
            
            for i, (capacity, base_capacity, generator) in enumerate(zip(capacity_list, base_pmax_list, generators)):
                gen_name = generator['name'] if isinstance(generator, dict) else generator
                
                # Check for negative capacity
                if capacity < 0:
                    validation['errors'].append(f"Generator {gen_name} has negative capacity: {capacity}")
                    validation['is_valid'] = False
                
                # Check for NaN or infinite values
                if not np.isfinite(capacity):
                    validation['errors'].append(f"Generator {gen_name} has invalid capacity: {capacity}")
                    validation['is_valid'] = False
                
                # Check wind generator bounds (should be between 0 and nominal)
                if gen_name.startswith('W'):
                    if capacity > base_capacity:
                        validation['errors'].append(f"Wind generator {gen_name} capacity ({capacity:.1f}) exceeds nominal ({base_capacity:.1f})")
                        validation['is_valid'] = False
            
            # Check 4: Data consistency
            validation['checks_performed'].append('data_consistency')
            
            # Verify capacity sums
            calculated_total = sum(capacity_list)
            if abs(calculated_total - scenario['total_capacity']) > 1e-6:
                validation['errors'].append(f"Capacity sum mismatch: calculated {calculated_total:.3f} vs stored {scenario['total_capacity']:.3f}")
                validation['is_valid'] = False
                
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
            validation['is_valid'] = False
        
        return validation

if __name__ == "__main__":
    """Demo of ScenarioManager functionality."""
    
    # Initialize with base case reference
    # manager = ScenarioManager("test_case")
    # manager = ScenarioManager("test_case1")
    manager = ScenarioManager("test_case_multiple_owners")

    players_config = manager.players_config

    demand_linear = manager.generate_demand_scenarios("linear", num_scenarios=10, min_factor=0.8, max_factor=1.2)
    capacity_linear = manager.generate_capacity_scenarios("linear", num_scenarios=3, min_factor=0.7, max_factor=1.0)

    scenarios = manager.create_scenario_set(
        demand_scenarios=demand_linear,
        capacity_scenarios=capacity_linear
    )

    # Print description
    print(scenarios['description_text'])
    
    # Access DataFrame directly
    scenario_df = scenarios['scenarios_df']
    print("\nCombined Scenarios Table:")
    print(scenario_df)
    
    cost_df = scenarios['costs_df']
    print("\nCosts DataFrame:")
    print(cost_df)

    # Players configuration
    print("\nPlayers Configuration:")
    for player in players_config:
        print(f"  • Player {player['id']} controls generators: {player['controlled_generators']}")
