"""Scenario generation package"""

def __getattr__(name):
    if name == 'ScenarioManager':
        from .scenario_generator import ScenarioManager
        return ScenarioManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['ScenarioManager']