"""
Scenario Generation Module

This module provides utilities for generating different types of scenarios
for electricity market bidding equilibria analysis.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, List, Optional, Any
import yaml
import os

class ScenarioGenerator:
    """Generate scenarios for bidding equilibria analysis."""
    
    def __init__(self):
        """Initialize the scenario generator."""
        pass
    
    def _load_base_case_demand(self, base_case_name: str = "test_case") -> float:
        """Load base demand from cases.yaml file."""
        # Get path to cases.yaml (two directories up from utils)
        cases_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cases.yaml")
        
        try:
            with open(cases_path, 'r') as f:
                cases = yaml.safe_load(f)
            
            if base_case_name not in cases:
                raise ValueError(f"Base case '{base_case_name}' not found in cases.yaml")
            
            base_case = cases[base_case_name]
            demand_list = base_case.get("demand", [])
            
            # Extract demand value (handle both single values and lists)
            if isinstance(demand_list, list):
                return float(demand_list[0]) if demand_list else 225.0
            else:
                return float(demand_list)
                
        except FileNotFoundError:
            print(f"Warning: cases.yaml not found at {cases_path}, using default demand of 225.0")
            return 225.0
        except Exception as e:
            print(f"Warning: Error loading base case demand: {e}, using default demand of 225.0")
            return 225.0
    
    def generate_demand_scenarios(
        self,
        scenario_type: str = "uniform",
        base_demand: Optional[float] = None,
        base_case_name: str = "test_case",
        num_scenarios: int = 6,
        **kwargs
    ) -> List[float]:
        """
        Generate demand scenarios based on specified type.
        
        Args:
            scenario_type: Type of scenario generation ('uniform', 'normal', 'custom')
            base_demand: Base demand level in MW (if None, loads from base_case_name in cases.yaml)
            base_case_name: Name of base case to get demand from (default: "test_case")
            num_scenarios: Number of scenarios to generate
            **kwargs: Additional parameters for specific scenario types
            
        Returns:
            List of demand values in MW
        """
        # If no base_demand provided, load from cases.yaml
        if base_demand is None:
            base_demand = self._load_base_case_demand(base_case_name)
            print(f"Loaded base demand from '{base_case_name}': {base_demand} MW")
        
        if scenario_type == "uniform":
            return self._generate_uniform_demand(base_demand, num_scenarios, **kwargs)
        elif scenario_type == "custom":
            return self._generate_custom_demand(**kwargs)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def _generate_uniform_demand(
        self, 
        base_demand: float, 
        num_scenarios: int,
        min_factor: float = 0.7,
        max_factor: float = 1.3
    ) -> List[float]:
        """Generate uniformly distributed demand scenarios around the base demand with specified factors."""
        min_demand = base_demand * min_factor
        max_demand = base_demand * max_factor
        demands = np.linspace(min_demand, max_demand, num_scenarios)
        return [float(d) for d in demands]
    
    def _generate_custom_demand(self, demands: List[float]) -> List[float]:
        """Use custom demand values."""
        return demands


def load_scenario_config(config_file: str = "scenario_config.yaml") -> Dict[str, Any]:
    """Load scenario configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    
    if not os.path.exists(config_path):
        # Return default configuration
        return {
            'scenario_type': 'uniform',
            'num_scenarios': 6,
            'min_factor': 0.7,
            'max_factor': 1.3
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Predefined scenario generators for common cases
def generate_standard_demand_scenarios(
    base_case_name: str = "test_case",
    num_scenarios: int = 6,
    min_factor: float = 0.7,
    max_factor: float = 1.3
) -> List[float]:
    """Generate standard demand scenarios using uniform distribution."""
    generator = ScenarioGenerator()
    return generator.generate_demand_scenarios(
        scenario_type="uniform",
        base_case_name=base_case_name,
        num_scenarios=num_scenarios,
        min_factor=min_factor,
        max_factor=max_factor
    )

def generate_custom_scenarios(demand_list: List[float]) -> List[float]:
    """Generate specified test scenarios with demand according to specified values."""
    generator = ScenarioGenerator()
    return generator.generate_demand_scenarios(
        scenario_type="custom",
        demands=demand_list
    )

if __name__ == "__main__":
    # Example usage
    generator = ScenarioGenerator()
    
    # Generate standard scenarios
    print("Standard demand scenarios:")
    standard_demands = generate_standard_demand_scenarios(num_scenarios=50)
    print(standard_demands)
    
    # Generate stress test scenarios
    print("\nCustom test scenarios:")
    custom_demands = generate_custom_scenarios(demand_list = [50.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    print(custom_demands)
    