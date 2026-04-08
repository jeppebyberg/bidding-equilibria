"""Simple loader for diagonalization configuration"""

import yaml
from typing import Dict, Any
from pathlib import Path

def load_diagonalization(config_path: str = None) -> Dict[str, Any]:
    """Load diagonalization configuration from YAML file"""
    if config_path is None:
        # Use the diagonalization.yaml file in the same directory as this loader
        config_path = Path(__file__).parent / "diagonalization.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("diagonalization", {})
