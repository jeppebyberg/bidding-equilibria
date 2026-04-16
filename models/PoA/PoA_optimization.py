import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import pandas as pd

class PoAOptimization:
    def __init__(
            self,
            P_init
    ):
        self.P_init = P_init

    def _build_model(self) -> None:
        """
        Build the complete MPEC model structure.
        This is called once, then update_strategic_player() updates the changing parts between each strategic player.
        """
        if self.strategic_player_id is None:
            raise ValueError("Must call update_strategic_player() before building model")

        self.model = ConcreteModel()
        
        # Define sets
        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1)) # For ramp constraints
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps)) 

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        # Decision variables
        self._build_PoA_variables()
        self._build_primal_equilibrium_variables()
        self._build_primal_optimal_variables()
        self._build_dual_equilibrium_variables()
        self._build_dual_optimal_variables()
        self._build_complementarity_equilibrium_variables()
        self._build_complementarity_optimal_variables()
        

    def _build_PoA_variables(self) -> None:
        self.model.D = Var(self.model.time_steps, within=NonNegativeReals)  # Demand at each time step
        self.model.P_max = Var(self.model.time_steps,self.model.n_gen, within=NonNegativeReals)  # Max production capacity for each generator and time step
        self.model.C_eq = Var(domain=Reals)
        self.model.C_opt = Var(domain=Reals)

    def _build_objective(self) -> None:
        pass

    def _build_constraints(self) -> None:
        self._build_realistic_constraints()
        self._build_policy_related_constraints()
        self._build_lower_level_equilibrium_constraints()
        self._build_lower_level_optimal_constraints()
        self._build_KKT_stationarity_equilibrium_constraints()
        self._build_KKT_stationarity_optimal_constraints()
        self._build_KKT_complementarity_equilibrium_constraints()
        self._build_KKT_complementarity_optimal_constraints()

    