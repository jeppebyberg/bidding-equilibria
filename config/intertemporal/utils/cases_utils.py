"""Load utilities for base case configuration"""

import yaml
from typing import List, Dict, Any, Tuple
from pathlib import Path

def load_test_case(case_name: str, case_path: str = "config/intertemporal/reference_cases.yaml") -> Dict[str, Any]:
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

def get_time_steps(case_name: str) -> int:
    """Get time steps from base case"""
    test_case = load_test_case(case_name=case_name)
    
    if "time_steps" in test_case:
        time_steps_list = test_case.get("time_steps")
        return time_steps_list[0] if isinstance(time_steps_list, list) else time_steps_list
    else:
        raise ValueError("No time steps information found in case")

def get_generators(case_name: str) -> List[Dict[str, Any]]:
    """Get generator data from test case"""
    test_case = load_test_case(case_name=case_name)
    return test_case.get("generators")

def get_players(case_name: str) -> List[Dict[str, Any]]:
    """Get players data from test case"""
    test_case = load_test_case(case_name=case_name)
    players = test_case.get("players", [])
    if not players:
        raise ValueError(f"No players configuration found in case '{case_name}'")
    
    # Check for overlapping generator ownership
    all_controlled_generators = []
    player_generator_map = {}
    
    for player in players:
        player_name = player.get("id", "Unknown")
        controlled_gens = player.get("controlled_generators", [])
        
        for gen_id in controlled_gens:
            if gen_id in all_controlled_generators:
                # Find which player already controls this generator
                existing_player = player_generator_map.get(gen_id, "Unknown")
                raise ValueError(f"Generator {gen_id} is controlled by both player {existing_player} and player {player_name} in case '{case_name}'")
            
            all_controlled_generators.append(gen_id)
            player_generator_map[gen_id] = player_name
    
    return players

def load_setup_data(case_name: str = "test_case") -> Tuple[int, List[float], List[float], List[float], float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load complete base case data including players for a given case name

    Parameters
    ----------
    case_name : str
        Name of the base case to load (must match a key in reference_cases.yaml)
    
    Returns
    -------
    tuple
        (num_generators, pmax_list, pmin_list, cost_vector, demand, generators, players)
    """
    # Load generators, demand, and players
    generators = get_generators(case_name=case_name)
    demand = get_demand(case_name=case_name)
    players = get_players(case_name=case_name)
    time_steps = get_time_steps(case_name=case_name)
    
    num_generators = len(generators)

    # Extract generator data into arrays
    pmax_list = [gen["pmax"] for gen in generators]
    pmin_list = [gen["pmin"] for gen in generators]
    cost_vector = [gen["cost"] for gen in generators]
    r_rates_up_list = [gen["R_rate_up"] for gen in generators]
    r_rates_down_list = [gen["R_rate_down"] for gen in generators]
    
    return num_generators, pmax_list, pmin_list, cost_vector, r_rates_up_list, r_rates_down_list, demand, generators, players, time_steps

if __name__ == "__main__":
    # Example usage - load complete data including players
    num_generators, pmax, pmin, cost, r_rates_up, r_rates_down, demand, generators, players, time_steps = load_setup_data("test_case")
    print("Number of generators:", num_generators)
    print("Pmax:", pmax)
    print("Pmin:", pmin)
    print("Cost:", cost)
    print("R_rates_up:", r_rates_up)
    print("R_rates_down:", r_rates_down)
    print("Demand:", demand)
    print("Time steps:", time_steps)
    # print("Generators:", generators)
    # print("\nPlayers:", players)
    print(f"\nLoaded {num_generators} generators and {len(players)} players") 