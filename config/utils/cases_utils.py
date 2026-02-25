"""Load utilities for chosen case"""

import yaml
from typing import List, Dict, Any, Tuple
from pathlib import Path

def load_test_case(case_name: str, case_path: str = "config/reference_case.yaml") -> Dict[str, Any]:
    """Load test case from YAML file"""
    with open(case_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    case_data = data.get(case_name)
    
    # Check if this case needs to be generated
    if case_data and case_data.get("use_generator"):
        case_data = _generate_case_on_demand(case_data)
    
    return case_data

def _generate_case_on_demand(case_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a case using the scenario generator."""
    from .utils.scenario_loader import create_scenario_based_case
    
    base_case = case_config.get("base_case", "test_case")
    scenario_config = case_config.get("scenario_config", "multi_scenario_case")
    
    return create_scenario_based_case(base_case, scenario_config)

def get_demand(case_name: str) -> float:
    """Get demand from test case"""
    test_case = load_test_case(case_name=case_name)
    
    # Handle single demand or demand scenarios
    if "demand" in test_case:
        demand_list = test_case.get("demand")
        return demand_list[0] if isinstance(demand_list, list) else demand_list
    elif "demand_scenarios" in test_case:
        # For multi-scenario cases, return the first scenario as default
        # or the mean of all scenarios
        demand_scenarios = test_case.get("demand_scenarios", [])
        if demand_scenarios:
            return sum(demand_scenarios) / len(demand_scenarios)  # Return mean
        else:
            raise ValueError("No demand scenarios found in case")
    else:
        raise ValueError("No demand information found in case")

def get_demand_scenarios(case_name: str) -> List[float]:
    """Get all demand scenarios from test case"""
    test_case = load_test_case(case_name=case_name)
    
    if "demand_scenarios" in test_case:
        return test_case.get("demand_scenarios", [])
    elif "demand" in test_case:
        # Single demand case - convert to list
        demand_list = test_case.get("demand")
        if isinstance(demand_list, list):
            return demand_list
        else:
            return [demand_list]
    else:
        raise ValueError("No demand information found in case")

def get_generators(case_name: str) -> List[Dict[str, Any]]:
    """Get generator data from test case"""
    test_case = load_test_case(case_name=case_name)
    return test_case.get("generators")

def load_setup_data(case_name: str = "test_case") -> Tuple[List[float], List[float], List[float], float]:
    """
    Load case data for a given case name

    Parameters
    ----------
    case_name : str
        Name of the test case to load (must match a key in the YAML file)
    
    Returns
    -------
    tuple
        (num_generators, pmax_list, pmin_list, cost_vector, demand)
    """
    # Load generators and demand
    generators = get_generators(case_name=case_name)
    demand = get_demand(case_name=case_name)
    
    num_generators = len(generators)

    # Extract generator data into arrays
    pmax_list = [gen["pmax"] for gen in generators]
    pmin_list = [gen["pmin"] for gen in generators]
    cost_vector = [gen["cost"] for gen in generators]
    
    return num_generators, pmax_list, pmin_list, cost_vector, demand

def load_multi_scenario_data(case_name: str) -> Tuple[int, List[float], List[float], List[float], List[float]]:
    """
    Load case data for multi-scenario cases
    
    Parameters
    ----------
    case_name : str
        Name of the test case to load
        
    Returns
    -------
    tuple
        (num_generators, pmax_list, pmin_list, cost_vector, demand_scenarios)
    """
    # Load generators and demand scenarios
    generators = get_generators(case_name=case_name)
    demand_scenarios = get_demand_scenarios(case_name=case_name)
    
    num_generators = len(generators)

    # Extract generator data into arrays
    pmax_list = [gen["pmax"] for gen in generators]
    pmin_list = [gen["pmin"] for gen in generators]
    cost_vector = [gen["cost"] for gen in generators]
    
    return num_generators, pmax_list, pmin_list, cost_vector, demand_scenarios

if __name__ == "__main__":
    # Example usage
    num_generators, pmax, pmin, cost, demand = load_setup_data("test_case")
    print("Number of generators:", num_generators)
    print("Pmax:", pmax)
    print("Pmin:", pmin)
    print("Cost:", cost)
    print("Demand:", demand)