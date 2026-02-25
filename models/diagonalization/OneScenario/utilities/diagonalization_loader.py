"""Simple loader for diagonalization configuration"""

import yaml
from typing import Dict, Any
from pathlib import Path

def load_diagonalization(config_path: str = "config/utils/diagonalization.yaml") -> Dict[str, Any]:
    """Load diagonalization configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("diagonalization", {})
