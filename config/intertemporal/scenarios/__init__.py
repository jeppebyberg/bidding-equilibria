"""Scenario generation package"""

def __getattr__(name):
    if name == 'ScenarioManager':
        from .scenario_generator import ScenarioManager
        return ScenarioManager
    if name in {'ScenarioManagerV2', 'ScenarioManager2'}:
        from .scenario_generator_2 import ScenarioManagerV2
        return ScenarioManagerV2
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['ScenarioManager', 'ScenarioManagerV2', 'ScenarioManager2']