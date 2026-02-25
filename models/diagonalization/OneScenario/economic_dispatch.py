from pyomo.environ import *
from typing import List

class EconomicDispatchModel:
    def __init__(self):
        pass

    def economic_dispatch(self, num_generators: int, demand: float, Pmax: List[float], Pmin: List[float], bid_list: List[float]) -> tuple[List[float], float]:
        """
        Solve the economic dispatch problem with the bids placed.

        Parameters
        ----------
        bid_list : List[float]
            Array of cost/bids values for each generator
        
        Returns
        -------
        dispatch : List[float]
            Optimal dispatch for each generator
        clearing_price : float
            Market clearing price
        """

        model = ConcreteModel()
        model.n_gen = Set(initialize=range(num_generators))
        model.P_G = Var(model.n_gen, domain=Reals)
        model.objective = Objective(
            expr=sum(bid_list[i] * model.P_G[i] for i in model.n_gen),
            sense=minimize
        )
        model.power_balance = Constraint(expr=demand - sum(model.P_G[i] for i in model.n_gen) == 0)

        model.gen_min = Constraint(model.n_gen, rule=lambda m, i: model.P_G[i] >= Pmin[i])
        model.gen_max = Constraint(model.n_gen, rule=lambda m, i: model.P_G[i] <= Pmax[i])

        # Attach suffix to capture duals
        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=False)
        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            # print("Optimal solution found for economic dispatch.")
            dispatch = [model.P_G[i].value for i in model.n_gen]
            clearing_price = -model.dual[model.power_balance]
            return dispatch, clearing_price
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

if __name__ == "__main__":
    from config.utils.cases_utils import load_setup_data
    num_generators, Pmax, Pmin, bid_list, demand = load_setup_data()

    model = EconomicDispatchModel()
    dispatch, clearing_price = model.economic_dispatch(num_generators=num_generators, demand=demand, Pmax=Pmax, Pmin=Pmin, bid_list=bid_list)
    print("Optimal Dispatch:", dispatch)
    print("Market Clearing Price:", clearing_price)