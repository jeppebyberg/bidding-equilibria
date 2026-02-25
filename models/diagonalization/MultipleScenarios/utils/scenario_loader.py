"""
Scenario Loader Module

This module provides utilities for loading and generating scenarios
that integrate with the existing case loading system.
"""

import yaml
import os
from typing import Dict, Any, Optional
from config.utils.scenario_generator import ScenarioGenerator

class ScenarioLoader:
    """Load and generate scenarios for bidding equilibria analysis."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the scenario loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = os.path.dirname(__file__)
        self.config_dir = config_dir
        self.generator = ScenarioGenerator()
    
    def load_base_case(self, case_name: str) -> Dict[str, Any]:
        """
        Load a base case from cases.yaml.
        
        Args:
            case_name: Name of the case to load
            
        Returns:
            Base case configuration
        """
        cases_file = os.path.join(os.path.dirname(self.config_dir), "cases.yaml")
        
        with open(cases_file, 'r') as f:
            cases = yaml.safe_load(f)
        
        if case_name not in cases:
            raise ValueError(f"Case '{case_name}' not found in cases.yaml")
        
        return cases[case_name]
    
    def load_scenario_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load scenario configuration.
        
        Args:
            config_name: Name of the configuration to load
            
        Returns:
            Scenario configuration
        """
        config_file = os.path.join(self.config_dir, "scenario_config.yaml")
        
        with open(config_file, 'r') as f:
            configs = yaml.safe_load(f)
        
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' not found in scenario_config.yaml")
        
        return configs[config_name]
    
    def generate_case(
        self,
        base_case_name: str,
        scenario_config_name: str,
        case_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete case with scenarios.
        
        Args:
            base_case_name: Name of the base case to use
            scenario_config_name: Name of the scenario configuration to use
            case_name: Name for the generated case (defaults to scenario_config_name)
            
        Returns:
            Complete case configuration with scenarios
        """
        # Load base case (use generators from it)
        base_case = self.load_base_case(base_case_name)
        
        # Load scenario configuration
        scenario_config = self.load_scenario_config(scenario_config_name)
        
        # Generate scenarios
        generated_scenarios = generate_scenarios_from_config(base_case, scenario_config)
        
        # Create final case
        final_case = {
            'demand_scenarios': generated_scenarios['demand_scenarios'],
            'generators': generated_scenarios['generators']
        }
        
        return final_case
    
    def update_cases_yaml_with_generated_case(
        self,
        base_case_name: str,
        scenario_config_name: str,
        new_case_name: Optional[str] = None
    ) -> None:
        """
        Generate a case and add it to cases.yaml.
        
        Args:
            base_case_name: Name of the base case to use
            scenario_config_name: Name of the scenario configuration to use
            new_case_name: Name for the new case in cases.yaml
        """
        if new_case_name is None:
            new_case_name = f"{base_case_name}_{scenario_config_name}"
        
        # Generate the case
        generated_case = self.generate_case(base_case_name, scenario_config_name)
        
        # Load existing cases
        cases_file = os.path.join(os.path.dirname(self.config_dir), "cases.yaml")
        
        with open(cases_file, 'r') as f:
            cases = yaml.safe_load(f)
        
        # Add the new case
        cases[new_case_name] = generated_case
        
        # Write back to file
        with open(cases_file, 'w') as f:
            yaml.safe_dump(cases, f, default_flow_style=False, sort_keys=False)
    
    def list_available_base_cases(self) -> list:
        """List available base cases from cases.yaml."""
        cases_file = os.path.join(os.path.dirname(self.config_dir), "cases.yaml")
        
        with open(cases_file, 'r') as f:
            cases = yaml.safe_load(f)
        
        return list(cases.keys())
    
    def list_available_scenario_configs(self) -> list:
        """List available scenario configurations."""
        config_file = os.path.join(self.config_dir, "scenario_config.yaml")
        
        if not os.path.exists(config_file):
            return []
        
        with open(config_file, 'r') as f:
            configs = yaml.safe_load(f)
        
        return list(configs.keys())


def create_scenario_based_case(
    base_case_name: str = "test_case",
    scenario_config_name: str = "multi_scenario_case",
    new_case_name: str = "generated_multi_scenario_case"
) -> Dict[str, Any]:
    """
    Convenience function to create a scenario-based case.
    
    Args:
        base_case_name: Name of the base case to use
        scenario_config_name: Name of the scenario configuration to use
        new_case_name: Name for the generated case
        
    Returns:
        Generated case configuration
    """
    loader = ScenarioLoader()
    return loader.generate_case(base_case_name, scenario_config_name, new_case_name)


def add_generated_case_to_yaml(
    base_case_name: str = "test_case",
    scenario_config_name: str = "multi_scenario_case",
    new_case_name: str = "generated_multi_scenario_case"
) -> None:
    """
    Convenience function to add a generated case to cases.yaml.
    
    Args:
        base_case_name: Name of the base case to use
        scenario_config_name: Name of the scenario configuration to use
        new_case_name: Name for the new case in cases.yaml
    """
    loader = ScenarioLoader()
    loader.update_cases_yaml_with_generated_case(base_case_name, scenario_config_name, new_case_name)


if __name__ == "__main__":
    # Example usage
    loader = ScenarioLoader()
    
    print("Available base cases:")
    print(loader.list_available_base_cases())
    
    print("\nAvailable scenario configurations:")
    print(loader.list_available_scenario_configs())
    
    # Generate a case
    print("\nGenerating multi_scenario_case...")
    case = create_scenario_based_case()
    print("Demand scenarios:", case['demand_scenarios'])
    print("Number of generators:", len(case['generators']))