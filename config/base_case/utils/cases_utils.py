"""Load utilities for base case configuration"""

import yaml
from typing import List, Dict, Any, Tuple
from pathlib import Path

def load_test_case(case_name: str, case_path: str = "config/base_case/reference_cases.yaml") -> Dict[str, Any]:
    """Load test case from reference cases YAML file"""
    with open(case_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    case_data = data.get(case_name)
    
    if not case_data:
        raise ValueError(f"Case '{case_name}' not found in {case_path}")
    
    return case_data

def get_demand(case_name: str) -> float:
    """Get demand from base case"""
    test_case = load_test_case(case_name=case_name)
    
    if "demand" in test_case:
        demand_list = test_case.get("demand")
        return demand_list[0] if isinstance(demand_list, list) else demand_list
    else:
        raise ValueError("No demand information found in case")

def get_generators(case_name: str) -> List[Dict[str, Any]]:
    """Get generator data from test case"""
    test_case = load_test_case(case_name=case_name)
    return test_case.get("generators")

def load_setup_data(case_name: str = "test_case") -> Tuple[int, List[float], List[float], List[float], float]:
    """
    Load base case data for a given case name

    Parameters
    ----------
    case_name : str
        Name of the base case to load (must match a key in reference_cases.yaml)
    
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
    
    return num_generators, pmax_list, pmin_list, cost_vector, demand, generators

if __name__ == "__main__":
    # Example usage
    num_generators, pmax, pmin, cost, demand, generators = load_setup_data("test_case")
    print("Number of generators:", num_generators)
    print("Pmax:", pmax)
    print("Pmin:", pmin)
    print("Cost:", cost)
    print("Demand:", demand)
    print("Generators:", generators) 