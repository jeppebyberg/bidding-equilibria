"""Simple loader for default configuration"""

import yaml
from typing import Dict, Any
from pathlib import Path

def load_defaults(config_path: str = "config/defaults.yaml") -> Dict[str, Any]:
    """Load default configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("defaults", {})

