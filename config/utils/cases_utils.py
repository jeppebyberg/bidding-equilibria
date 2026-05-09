"""Load utilities for base case configuration"""

import yaml
import warnings
from typing import List, Dict, Any, Tuple
from pathlib import Path

DEFAULT_REFERENCE_CASES_PATH = Path(__file__).resolve().parents[1] / "reference_cases.yaml"

def load_test_case(case_name: str, case_path: str | Path = DEFAULT_REFERENCE_CASES_PATH) -> Dict[str, Any]:
    """Load test case from reference cases YAML file"""
    case_path = Path(case_path)
    with case_path.open("r", encoding="utf-8") as f:
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

def _is_wind_generator(generator: Dict[str, Any]) -> bool:
    gen_type = str(generator.get("type", "")).lower()
    gen_name = str(generator.get("name", ""))
    return gen_type == "wind" or gen_name.startswith("W")

def normalize_generators(generators: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize physical generators and optional bidding blocks.

    Old cases with one row per generator become one physical generator with
    one block. New cases can provide ``bidding_blocks`` under a physical
    generator; ramping data remains attached to the physical generator.
    """
    physical_generators: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []

    for physical_idx, gen in enumerate(generators):
        physical_id = int(gen.get("id", physical_idx))
        physical_name = str(gen.get("name", f"G{physical_id}"))
        gen_type = str(gen.get("type", "wind" if _is_wind_generator(gen) else "conventional"))
        ramp_up = float(gen.get("R_rate_up", gen.get("ramp_up", 0.0)))
        ramp_down = float(gen.get("R_rate_down", gen.get("ramp_down", 0.0)))
        pmin = float(gen.get("pmin", 0.0))
        is_wind = gen_type.lower() == "wind" or physical_name.startswith("W")

        raw_blocks = gen.get("bidding_blocks")
        if raw_blocks:
            normalized_blocks: List[Dict[str, Any]] = []
            for local_block_idx, block in enumerate(raw_blocks):
                block_id = int(block.get("block_id", local_block_idx))
                block_name = str(block.get("name", f"{physical_name}_B{block_id}"))
                block_record = {
                    "block_id": block_id,
                    "block_name": block_name,
                    "physical_id": physical_id,
                    "physical_name": physical_name,
                    "physical_index": physical_idx,
                    "pmax": float(block["pmax"]),
                    "cost": float(block["cost"]),
                    "is_wind": is_wind,
                }
                normalized_blocks.append(block_record)
                blocks.append(block_record)
        else:
            warnings.warn(
                f"Generator '{physical_name}' uses legacy generator-level pmax/cost; "
                "converting it to a one-block bidding_blocks layout.",
                UserWarning,
                stacklevel=2,
            )
            block_record = {
                "block_id": 0,
                "block_name": f"{physical_name}_B1",
                "physical_id": physical_id,
                "physical_name": physical_name,
                "physical_index": physical_idx,
                "pmax": float(gen["pmax"]),
                "cost": float(gen["cost"]),
                "is_wind": is_wind,
            }
            normalized_blocks = [block_record]
            blocks.append(block_record)

        physical_generators.append(
            {
                "physical_id": physical_id,
                "physical_name": physical_name,
                "type": gen_type,
                "ramp_up": ramp_up,
                "ramp_down": ramp_down,
                "pmin": pmin,
                "is_wind": is_wind,
                "blocks": normalized_blocks,
                "pmax": float(sum(block["pmax"] for block in normalized_blocks)),
            }
        )

    return {
        "physical_generators": physical_generators,
        "blocks": blocks,
        "block_names": [block["block_name"] for block in blocks],
        "physical_generator_names": [gen["physical_name"] for gen in physical_generators],
        "block_to_physical": {block["block_name"]: block["physical_name"] for block in blocks},
        "physical_to_blocks": {
            gen["physical_name"]: [block["block_name"] for block in gen["blocks"]]
            for gen in physical_generators
        },
    }

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
    
    normalized = normalize_generators(generators)
    physical_generators = normalized["physical_generators"]
    num_generators = len(physical_generators)

    # Extract generator data into arrays
    pmax_list = [gen["pmax"] for gen in physical_generators]
    pmin_list = [gen["pmin"] for gen in physical_generators]
    cost_vector = [
        gen["blocks"][0]["cost"] if gen["blocks"] else 0.0
        for gen in physical_generators
    ]
    r_rates_up_list = [gen["ramp_up"] for gen in physical_generators]
    r_rates_down_list = [gen["ramp_down"] for gen in physical_generators]
    
    return num_generators, pmax_list, pmin_list, cost_vector, r_rates_up_list, r_rates_down_list, demand, generators, players, time_steps

if __name__ == "__main__":
    # Example usage - load complete data including players
    num_generators, pmax, pmin, cost, r_rates_up, r_rates_down, demand, generators, players, time_steps = load_setup_data("test_case_bidding_blocks")
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
