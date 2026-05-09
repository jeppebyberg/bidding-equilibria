"""Simple loader for default configuration"""

import yaml
from typing import Dict, Any
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

def load_defaults(config_path: str | Path = CONFIG_DIR / "defaults.yaml") -> Dict[str, Any]:
    """Load default configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("defaults", {})

def load_test_case_config(test_case: str, config_path: str | Path = CONFIG_DIR / "reference_cases.yaml") -> Dict[str, Any]:
    """Load test case specific configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    test_cases = data.get("test_cases", data)
    case_config = test_cases.get(test_case, {})
    return case_config if isinstance(case_config, dict) else {}
