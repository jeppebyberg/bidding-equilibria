"""
Script to run Economic Dispatch model examples and tests
"""
from .economic_dispatch import EconomicDispatchModel
import config.base_case as config

if __name__ == "__main__":
    num_generators, Pmax, Pmin, bid_list, demand, generators = config.load_setup_data("test_case")

    model = EconomicDispatchModel()
    dispatch, clearing_price = model.economic_dispatch(num_generators=num_generators, demand=demand, Pmax=Pmax, Pmin=Pmin, bid_list=bid_list)
    print("Optimal Dispatch:", dispatch)
    print("Market Clearing Price:", clearing_price)