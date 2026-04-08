"""Simple loader for default configuration"""

import yaml
from typing import Dict, Any
from pathlib import Path

def load_defaults(config_path: str = "config/defaults.yaml") -> Dict[str, Any]:
    """Load default configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("defaults", {})

def load_test_case_config(test_case: str, config_path: str = "drivers/intertemporal/utils/test_case.yaml") -> Dict[str, Any]:
    """Load test case specific configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    test_cases = data.get("test_cases", {})
    return test_cases.get(test_case, {})