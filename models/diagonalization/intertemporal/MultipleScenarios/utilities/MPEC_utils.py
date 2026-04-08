"""Utilities for MPEC model configuration"""

import yaml
from pathlib import Path
from typing import Dict, Any

from config.default_loader import load_defaults

def load_mpec_config(config_path: str = None) -> Dict[str, Any]:
    """Load MPEC configuration from YAML file"""
    if config_path is None:
        # Default to MPEC.yaml in the same directory as this file
        config_path = Path(__file__).parent / "MPEC.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("mpec_config", {})

def get_mpec_parameters() -> Dict[str, Any]:
    """
    Get complete MPEC parameters combining defaults and local config
    
    Returns
    -------
    dict
        Combined MPEC configuration parameters
    """
    # Load defaults from config
    defaults = load_defaults()
    
    # Load local MPEC config
    local_config = load_mpec_config()
    
    # Merge configurations (local overrides defaults)
    combined_config = defaults.copy()
    combined_config.update(local_config)
    
    return combined_config

def get_big_m_complementarity() -> float:
    """Get Big-M value for complementarity constraints"""
    config = get_mpec_parameters()
    return config.get("big_m_complementarity")

def get_big_m_bid_separation() -> float:
    """Get Big-M value for bid separation constraints"""
    config = get_mpec_parameters()
    return config.get("big_m_bid_separation")

def get_bid_separation_epsilon() -> float:
    """Get epsilon value for bid separation constraints"""
    config = get_mpec_parameters()
    return config.get("bid_separation_epsilon")

def get_alpha_bounds() -> tuple[float, float]:
    """Get bid bounds (alpha_min, alpha_max)"""
    config = get_mpec_parameters()
    alpha_min = config.get("alpha_min")
    alpha_max = config.get("alpha_max")
    return alpha_min, alpha_max

def get_solver_config() -> Dict[str, Any]:
    """Get solver configuration"""
    config = get_mpec_parameters()
    return {
        "solver": config.get("solver"),
    }

if __name__ == "__main__":
    # Example usage
    mpec_params = get_mpec_parameters()
    print("MPEC Parameters:", mpec_params)