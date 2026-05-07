"""
SOS1-based variant of the synthetic-data MPEC model.

This module mirrors ``MPEC_MS.MPECModel`` but replaces the Big-M
complementarity formulation with SOS1 pairs between each lower-level dual
variable and its corresponding primal slack.
"""

from pyomo.environ import Constraint, NonNegativeReals, SOSConstraint, Var

from models.synthetic_data_generation.MPEC_MS import MPECModel as BigMMPECModel


class MPECModelSOS1(BigMMPECModel):
    """
    Drop-in SOS1 version of :class:`models.synthetic_data_generation.MPEC_MS.MPECModel`.

    The upper-level problem, lower-level primal constraints, stationarity
    constraints, objective, parallel scenario solve flow, and result extraction
    are inherited from the Big-M implementation. Only complementarity is
    reformulated:

        0 <= dual ⟂ slack >= 0

    becomes an SOS1 set containing ``[dual, slack]``.
    """

    def _build_complementarity_variables(self) -> None:
        """
        Build nonnegative primal slack variables used in the SOS1 pairs.

        The Big-M version creates binary ``z_*`` variables here. SOS1 does not
        need those binaries, but it does need explicit variables for the primal
        slacks because Pyomo SOS constraints operate on variables, not general
        linear expressions.
        """
        self.model.slack_upper_bound = Var(
            self.model.n_gen,
            self.model.time_steps,
            domain=NonNegativeReals,
        )
        self.model.slack_lower_bound = Var(
            self.model.n_gen,
            self.model.time_steps,
            domain=NonNegativeReals,
        )
        self.model.slack_ramp_up = Var(
            self.model.n_gen,
            self.model.time_steps,
            domain=NonNegativeReals,
        )
        self.model.slack_ramp_down = Var(
            self.model.n_gen,
            self.model.time_steps,
            domain=NonNegativeReals,
        )

    def _build_KKT_complementarity_constraints(self) -> None:
        """
        Build slack-defining equalities and SOS1 complementarity pairs.

        The inherited lower-level constraints still define feasibility. These
        equalities mirror those inequalities with nonnegative slacks, and each
        SOS1 set enforces that the slack and corresponding dual cannot both be
        nonzero.
        """
        s = self.scenario_index

        def upper_slack_rule(m, i, t):
            return m.slack_upper_bound[i, t] == self.pmax_scenarios[s][t][i] - m.P[i, t]

        def lower_slack_rule(m, i, t):
            return m.slack_lower_bound[i, t] == m.P[i, t] - self.pmin_scenarios[s][t][i]

        def ramp_up_slack_rule(m, i, t):
            return m.slack_ramp_up[i, t] == self.ramp_vector_up[i] - (m.P[i, t] - m.P[i, t - 1])

        def ramp_up_initial_slack_rule(m, i):
            return m.slack_ramp_up[i, 0] == self.P_init[s][i] + self.ramp_vector_up[i] - m.P[i, 0]

        def ramp_down_slack_rule(m, i, t):
            return m.slack_ramp_down[i, t] == self.ramp_vector_down[i] - (m.P[i, t - 1] - m.P[i, t])

        def ramp_down_initial_slack_rule(m, i):
            return m.slack_ramp_down[i, 0] == m.P[i, 0] - self.P_init[s][i] + self.ramp_vector_down[i]

        self.model.upper_bound_slack_constraints = Constraint(
            self.model.n_gen,
            self.model.time_steps,
            rule=upper_slack_rule,
        )
        self.model.lower_bound_slack_constraints = Constraint(
            self.model.n_gen,
            self.model.time_steps,
            rule=lower_slack_rule,
        )
        self.model.ramp_up_slack_constraints = Constraint(
            self.model.n_gen,
            self.model.time_steps_minus_1,
            rule=ramp_up_slack_rule,
        )
        self.model.ramp_up_initial_slack_constraints = Constraint(
            self.model.n_gen,
            rule=ramp_up_initial_slack_rule,
        )
        self.model.ramp_down_slack_constraints = Constraint(
            self.model.n_gen,
            self.model.time_steps_minus_1,
            rule=ramp_down_slack_rule,
        )
        self.model.ramp_down_initial_slack_constraints = Constraint(
            self.model.n_gen,
            rule=ramp_down_initial_slack_rule,
        )

        def upper_sos_rule(m, i, t):
            return [m.mu_upper_bound[i, t], m.slack_upper_bound[i, t]]

        def lower_sos_rule(m, i, t):
            return [m.mu_lower_bound[i, t], m.slack_lower_bound[i, t]]

        def ramp_up_sos_rule(m, i, t):
            return [m.mu_ramp_up[i, t], m.slack_ramp_up[i, t]]

        def ramp_down_sos_rule(m, i, t):
            return [m.mu_ramp_down[i, t], m.slack_ramp_down[i, t]]

        self.model.upper_bound_complementarity_sos = SOSConstraint(
            self.model.n_gen,
            self.model.time_steps,
            rule=upper_sos_rule,
            sos=1,
        )
        self.model.lower_bound_complementarity_sos = SOSConstraint(
            self.model.n_gen,
            self.model.time_steps,
            rule=lower_sos_rule,
            sos=1,
        )
        self.model.ramp_up_complementarity_sos = SOSConstraint(
            self.model.n_gen,
            self.model.time_steps,
            rule=ramp_up_sos_rule,
            sos=1,
        )
        self.model.ramp_down_complementarity_sos = SOSConstraint(
            self.model.n_gen,
            self.model.time_steps,
            rule=ramp_down_sos_rule,
            sos=1,
        )


# Backwards-friendly alias for scripts that expect the original class name.
MPECModel = MPECModelSOS1
