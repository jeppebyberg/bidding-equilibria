"""Base case configuration package"""

def __getattr__(name):
    if name == 'ScenarioManager':
        from .scenarios.scenario_generator import ScenarioManager
        return ScenarioManager
    elif name == 'load_setup_data':
        from .utils.cases_utils import load_setup_data
        return load_setup_data
    elif name == 'get_generators':
        from .utils.cases_utils import get_generators
        return get_generators
    elif name == 'get_demand':
        from .utils.cases_utils import get_demand
        return get_demand
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['load_setup_data', 'get_generators', 'get_demand', 'ScenarioManager']