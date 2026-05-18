from pathlib import Path
import json
from pyomo.environ import *
from typing import Any, Optional

from models.PoA.PoA_optimization import PoAOptimization


class BiddingBlocksTighteningOptimizer(PoAOptimization):
    """Optimizer subclass with the expensive tightening construction routines.

    The base class is kept focused on the final PoA model plus loading and
    applying precomputed tightening reports. This subclass is used only by the
    alpha-bound and Big-M preprocessing scripts.
    """
    # Robust slack-based OBBT and Big-M tightening
    # ------------------------------------------------------------------

    def _build_tightening_sets(self) -> None:
        """
        Build the common index sets used by all explicit preprocessing models.

        These are the same sets created by `PoAOptimizationBiddingBlocks.build_model()`,
        but the preprocessing stages call only the variable and constraint
        builders that are actually needed for the stage.
        """
        self.model.time_steps = Set(initialize=range(self.num_time_steps))
        self.model.time_steps_minus_1 = Set(initialize=range(1, self.num_time_steps))
        self.model.time_steps_plus_1 = Set(initialize=range(self.num_time_steps + 1))
        self.model.physical_generators = Set(initialize=range(self.num_physical_generators))
        self.model.generator_blocks = Set(dimen=2, initialize=self.generator_block_pairs)
        self.model.wind_physical_generators = Set(initialize=self.wind_physical_generator_ids)
        self.model.conventional_physical_generators = Set(
            initialize=self.conventional_physical_generator_ids
        )
        self.model.wind_blocks = Set(dimen=2, initialize=self.wind_block_pairs)
        self.model.conventional_blocks = Set(dimen=2, initialize=self.conventional_block_pairs)

    def _build_alpha_bound_model(self) -> ConcreteModel:
        """
        Explicit Stage 1 model: support set plus policy/ReLU constraints.

        No lower-level dispatch, KKT stationarity, complementarity, or PoA
        objective is present. For NN-controlled generators, alpha is linked to
        the embedded ReLU network. For non-NN generators, alpha is fixed to true
        marginal cost through the same policy builder used in the PoA model.
        """
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()
        self.model.alpha = Var(self.model.generator_blocks, self.model.time_steps, domain=Reals)
        self._build_support_set()
        self._build_policy_constraints()
        return self.model

    def _build_side_kkt_model(
        self,
        side: str,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        include_complementarity: bool = True,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> ConcreteModel:
        """
        Explicit Stage 2/3 model for one lower-level KKT system.

        `side="eq"` builds the policy/equilibrium KKT system and imposes alpha
        through the precomputed bounds, not through ReLU constraints. `side="opt"`
        builds the true-cost/social-optimum KKT system; alpha is not needed there.
        """
        self.model = ConcreteModel()
        self._build_tightening_sets()
        self._build_PoA_variables()

        if side == "eq":
            if alpha_bounds is None:
                raise ValueError("Equilibrium KKT tightening requires alpha_bounds")
            self._build_equilibrium_variables()
            self._build_complementarity_equilibrium_variables()
            self._build_support_set()
            self._apply_alpha_bounds(self.model, alpha_bounds)
            self._build_lower_level_equilibrium_constraints()
            self._build_KKT_stationarity_equilibrium_constraints()
            if include_complementarity:
                self._build_KKT_complementarity_equilibrium_constraints()
                self._apply_fixed_binaries(self.model, fixed_binaries)
            else:
                for var_name in self._binary_components_for_side(side):
                    binary_var = getattr(self.model, var_name, None)
                    if binary_var is not None:
                        for index in binary_var:
                            binary_var[index].fix(0)
        elif side == "opt":
            self._build_optimal_variables()
            self._build_complementarity_optimal_variables()
            self._build_support_set()
            self._build_lower_level_optimal_constraints()
            self._build_KKT_stationarity_optimal_constraints()
            if include_complementarity:
                self._build_KKT_complementarity_optimal_constraints()
                self._apply_fixed_binaries(self.model, fixed_binaries)
            else:
                for var_name in self._binary_components_for_side(side):
                    binary_var = getattr(self.model, var_name, None)
                    if binary_var is not None:
                        for index in binary_var:
                            binary_var[index].fix(0)
        else:
            raise ValueError(f"Unknown KKT tightening side: {side}")

        return self.model

    def compute_nn_certified_bid_bounds(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
    ) -> dict[str, Any]:
        """
        Stage 1: compute exact support-set bid bounds.

        For each bidding block and time step, solve two explicit optimization
        programs:

            min alpha[i,b,t]
            max alpha[i,b,t]

        subject only to the support set and the policy constraints. For
        NN-controlled generators this means the ReLU MILP embedding is active.
        For true-cost generators, the true-cost alpha equality is active.
        """
        if self.nn_policy_generator_ids and not self.nn_policies:
            self._load_nn_policies()
            self._load_nn_normalization_stats()

        alpha_bounds: dict[tuple[int, int, int], dict[str, float]] = {}
        optimization_results: dict[str, dict[str, Any]] = {}
        targets = [
            (int(i), int(b), int(t))
            for i, b in self.generator_block_pairs
            for t in range(self.num_time_steps)
        ]
        total_programs = 2 * len(targets)
        program_number = 0
        print(f"\nAlpha-bound optimization programs: {total_programs}", flush=True)

        for index in targets:
            lower_upper: dict[str, Optional[float]] = {"lower": None, "upper": None}
            for bound_name, sense in (("lower", minimize), ("upper", maximize)):
                program_number += 1
                print(
                    f"[Alpha {program_number}/{total_programs}] "
                    f"{'minimize' if bound_name == 'lower' else 'maximize'} "
                    f"alpha{index}",
                    flush=True,
                )
                m = self._build_alpha_bound_model()
                alpha_expr = m.alpha[index]
                m.tightening_objective = Objective(expr=alpha_expr, sense=sense)
                solved, results = self._solve_tightening_model(
                    m,
                    solver_name=solver_name,
                    time_limit=time_limit,
                    tee=tee,
                )
                bound_value = self._safe_value(alpha_expr) if solved else None
                lower_upper[bound_name] = bound_value
                optimization_results[f"{self._json_key(index)}:{bound_name}"] = {
                    "value": bound_value,
                    "termination_condition": str(results.solver.termination_condition),
                }

            if lower_upper["lower"] is None or lower_upper["upper"] is None:
                raise RuntimeError(f"Could not compute alpha bounds for index {index}")
            alpha_bounds[index] = {
                "lower": float(lower_upper["lower"]),
                "upper": float(lower_upper["upper"]),
            }

        self.alpha_bounds = alpha_bounds
        self.alpha_bound_optimization_results = optimization_results
        return {
            "alpha_bounds": self._jsonify_indexed_dict(alpha_bounds),
            "optimization_results": optimization_results,
            "num_optimization_programs": total_programs,
        }

    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    def _jsonify_indexed_dict(self, payload: dict[tuple[int, ...], Any]) -> dict[str, Any]:
        return {self._json_key(tuple(key)): value for key, value in payload.items()}

    def _apply_alpha_bounds(
        self,
        m: ConcreteModel,
        alpha_bounds: dict[tuple[int, int, int], dict[str, float]],
    ) -> None:
        m.alpha_certified_bounds = ConstraintList()
        for i, b in m.generator_blocks:
            for t in m.time_steps:
                bounds = alpha_bounds[(int(i), int(b), int(t))]
                lower = float(bounds["lower"])
                upper = float(bounds["upper"])
                m.alpha_certified_bounds.add(m.alpha[i, b, t] >= lower)
                m.alpha_certified_bounds.add(m.alpha[i, b, t] <= upper)
                m.alpha[i, b, t].setlb(lower)
                m.alpha[i, b, t].setub(upper)

    def _apply_fixed_binaries(
        self,
        m: ConcreteModel,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        fixed_binaries = fixed_binaries or getattr(self, "fixed_binaries", {})
        for var_name, entries in fixed_binaries.items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, _details in entries.items():
                index = tuple(int(part) for part in str(key).split(",") if part != "")
                binary_var[index].fix(0)

    def _binary_components_for_side(self, side: str) -> tuple[str, ...]:
        if side == "eq":
            return ("z_upper_eq", "z_lower_eq", "z_ramp_up_eq", "z_ramp_down_eq")
        if side == "opt":
            return ("z_upper_opt", "z_lower_opt", "z_ramp_up_opt", "z_ramp_down_opt")
        raise ValueError(f"Unknown tightening side: {side}")

    def _dispatch_var(self, m: ConcreteModel, side: str) -> Any:
        return m.P_eq if side == "eq" else m.P_opt

    def _slack_expression(self, m: ConcreteModel, side: str, constraint_type: str, index: tuple[int, ...]) -> Any:
        """
        Return the nonnegative slack for one lower-level inequality.

        Complementarity is `dual * slack = 0`. Therefore, if the minimum feasible
        slack is at least epsilon over the entire support set and certified bid
        box, the associated dual is always zero and the active-set binary can be
        fixed to zero without changing any feasible KKT point.
        """
        P = self._dispatch_var(m, side)
        if constraint_type == "upper":
            i, b, t = index
            return m.P_max_block[i, b, t] - P[i, b, t]
        if constraint_type == "lower":
            i, b, t = index
            return P[i, b, t]
        if constraint_type == "ramp_up":
            i, t = index
            current = sum(P[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            previous = (
                self.p_init[int(i)]
                if int(t) == 0
                else sum(P[i, b, int(t) - 1] for b in self.local_blocks_by_generator[int(i)])
            )
            return self.ramp_vector_up[int(i)] - (current - previous)
        if constraint_type == "ramp_down":
            i, t = index
            current = sum(P[i, b, t] for b in self.local_blocks_by_generator[int(i)])
            previous = (
                self.p_init[int(i)]
                if int(t) == 0
                else sum(P[i, b, int(t) - 1] for b in self.local_blocks_by_generator[int(i)])
            )
            return self.ramp_vector_down[int(i)] - (previous - current)
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

    def _binary_name(self, side: str, constraint_type: str) -> str:
        return {
            ("eq", "upper"): "z_upper_eq",
            ("eq", "lower"): "z_lower_eq",
            ("eq", "ramp_up"): "z_ramp_up_eq",
            ("eq", "ramp_down"): "z_ramp_down_eq",
            ("opt", "upper"): "z_upper_opt",
            ("opt", "lower"): "z_lower_opt",
            ("opt", "ramp_up"): "z_ramp_up_opt",
            ("opt", "ramp_down"): "z_ramp_down_opt",
        }[(side, constraint_type)]

    def _dual_name(self, side: str, constraint_type: str) -> str:
        return {
            ("eq", "upper"): "mu_upper_eq",
            ("eq", "lower"): "mu_lower_eq",
            ("eq", "ramp_up"): "mu_ramp_up_eq",
            ("eq", "ramp_down"): "mu_ramp_down_eq",
            ("opt", "upper"): "mu_upper_opt",
            ("opt", "lower"): "mu_lower_opt",
            ("opt", "ramp_up"): "mu_ramp_up_opt",
            ("opt", "ramp_down"): "mu_ramp_down_opt",
        }[(side, constraint_type)]

    def _lambda_name(self, side: str) -> str:
        return {
            "eq": "lambda_eq",
            "opt": "lambda_opt",
        }[side]

    def _aggregate_dual_bound_key(self, constraint_type: str) -> str:
        return {
            "upper": "mu_max_sum_ub",
            "lower": "mu_min_sum_ub",
            "ramp_up": "mu_ramp_up_sum_ub",
            "ramp_down": "mu_ramp_down_sum_ub",
        }[constraint_type]

    def _aggregate_dual_expression(
        self,
        m: ConcreteModel,
        side: str,
        constraint_type: str,
        time_idx: int,
    ) -> Any:
        dual_var = getattr(m, self._dual_name(side, constraint_type))
        if constraint_type in {"upper", "lower"}:
            return sum(
                dual_var[i, b, int(time_idx)]
                for i, b in self.generator_block_pairs
            )
        return sum(
            dual_var[i, int(time_idx)]
            for i in range(self.num_physical_generators)
        )

    def _solve_tightening_model(
        self,
        m: ConcreteModel,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
        solver_options: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Any]:
        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        if solver_options:
            for option_name, option_value in solver_options.items():
                solver.options[option_name] = option_value
        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        ok = termination in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
        return bool(ok), results

    def run_lambda_bound_tightening(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
    ) -> dict[str, Any]:
        """
        Maximize and minimize each price dual before the dual Big-M programs.

        These bounds replace the loose global lambda fallback in later
        tightening models and in the final PoA model. The bounds are computed
        over the same KKT-side systems used for dual Big-M tightening, including
        any slack-certified binary fixings.
        """
        alpha_bounds = alpha_bounds or getattr(self, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before lambda tightening")
        fixed_binaries = fixed_binaries or getattr(self, "fixed_binaries", {})

        # Avoid constraining the bound-computation programs with stale lambda
        # bounds from a previous report. Once recomputed, these bounds are used
        # for the remaining dual Big-M programs in this run.
        self.lambda_bounds = {}

        lambda_bounds: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[str, int]] = [
            (side, t)
            for side in ("eq", "opt")
            for t in range(self.num_time_steps)
        ]
        print(
            f"\nLambda-bound optimization programs: {2 * len(tasks)}",
            flush=True,
        )

        for program_idx, (side, time_idx) in enumerate(tasks, start=1):
            lambda_name = self._lambda_name(side)
            entry: dict[str, Any] = {}
            for bound_name, sense in (("lower", minimize), ("upper", maximize)):
                print(
                    f"[Lambda {2 * (program_idx - 1) + (1 if bound_name == 'lower' else 2)}/"
                    f"{2 * len(tasks)}] "
                    f"{'minimize' if bound_name == 'lower' else 'maximize'} "
                    f"{lambda_name}[{time_idx}]",
                    flush=True,
                )
                m = self._build_side_kkt_model(
                    side=side,
                    alpha_bounds=alpha_bounds,
                    include_complementarity=True,
                    fixed_binaries=fixed_binaries,
                )
                lambda_expr = getattr(m, lambda_name)[int(time_idx)]
                m.tightening_objective = Objective(expr=lambda_expr, sense=sense)
                solved, results = self._solve_tightening_model(
                    m,
                    solver_name=solver_name,
                    time_limit=time_limit,
                    tee=tee,
                )
                entry[bound_name] = self._safe_value(lambda_expr) if solved else None
                entry[f"{bound_name}_termination_condition"] = str(
                    results.solver.termination_condition
                )

            lambda_bounds.setdefault(lambda_name, {})[str(int(time_idx))] = entry

        self.lambda_bounds = lambda_bounds
        return {"lambda_bounds": lambda_bounds}

    def run_slack_based_obbt(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        epsilon: float = 1e-6,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        relax_complementarity: bool = False,
        stop_at_zero_slack: bool = True,
        slack_stop_tol: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Stage 2: minimize every constraint slack over the robust support model.

        A positive minimum slack certifies that the lower-level inequality can
        never bind. Its dual must therefore be zero in complementarity, and the
        associated binary is unnecessary and fixed to zero for Stage 3.

        By default the side-specific KKT complementarity constraints remain
        active. This is a stronger, MILP-based certificate: if the target slack
        can be driven to zero only in a relaxation that violates active-set
        logic, it will no longer block binary fixing. Set relax_complementarity
        to True only for a cheap preliminary screen.

        For Gurobi solves, stop_at_zero_slack uses BestObjStop on these slack
        feasibility tests only. The slack objective is nonnegative, so once a
        feasible incumbent has slack <= the tolerance, the tested zero-slack
        KKT-side system is feasible and no tighter minimization certificate is
        useful. This shortcut is intentionally not used for dual Big-M OBBT:
        early incumbent dual values can create invalid overly tight Big-M bounds.
        """
        alpha_bounds = alpha_bounds or getattr(self, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before slack OBBT")

        zero_slack_tol = float(slack_stop_tol if slack_stop_tol is not None else epsilon)
        early_stop_enabled = bool(stop_at_zero_slack and solver_name == "gurobi")

        slack_bounds: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
        fixed_binaries: dict[str, dict[str, Any]] = {}

        tasks: list[tuple[str, str, tuple[int, ...]]] = []
        for side in ("eq", "opt"):
            for i, b in self.generator_block_pairs:
                for t in range(self.num_time_steps):
                    tasks.append((side, "upper", (int(i), int(b), int(t))))
                    tasks.append((side, "lower", (int(i), int(b), int(t))))
            for i in range(self.num_physical_generators):
                for t in range(self.num_time_steps):
                    tasks.append((side, "ramp_up", (int(i), int(t))))
                    tasks.append((side, "ramp_down", (int(i), int(t))))

        total_programs = len(tasks)
        mode = "relaxed LP" if relax_complementarity else "KKT MILP"
        print(f"\nSlack OBBT optimization programs: {total_programs} ({mode})", flush=True)

        for program_number, (side, constraint_type, index) in enumerate(tasks, start=1):
            print(
                f"[Slack OBBT {program_number}/{total_programs}] "
                f"minimize slack side={side}, constraint={constraint_type}, "
                f"index={index}",
                flush=True,
            )
            m = self._build_side_kkt_model(
                side=side,
                alpha_bounds=alpha_bounds,
                include_complementarity=not relax_complementarity,
                fixed_binaries=fixed_binaries,
            )
            slack_expr = self._slack_expression(m, side, constraint_type, index)
            m.target_slack = Var(domain=NonNegativeReals)
            m.target_slack_definition = Constraint(expr=m.target_slack == slack_expr)
            m.tightening_objective = Objective(expr=m.target_slack, sense=minimize)
            solver_options = (
                {"BestObjStop": zero_slack_tol}
                if early_stop_enabled
                else None
            )
            _solved, results = self._solve_tightening_model(
                m,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                solver_options=solver_options,
            )
            termination = results.solver.termination_condition
            incumbent_slack = self._safe_value(m.target_slack)
            is_optimal = termination in {
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
                TerminationCondition.globallyOptimal,
            }

            if incumbent_slack is not None and incumbent_slack <= zero_slack_tol:
                result_classification = "zero_slack_feasible"
                minimum_slack = 0.0 if incumbent_slack < 0.0 else float(incumbent_slack)
                is_inactive = False
            elif is_optimal and incumbent_slack is not None:
                result_classification = "positive_slack_optimal"
                minimum_slack = float(incumbent_slack)
                is_inactive = minimum_slack >= float(epsilon)
            else:
                result_classification = "undetermined"
                minimum_slack = None
                is_inactive = False

            record_key = (side, constraint_type, index)
            slack_bounds[record_key] = {
                "minimum_slack": minimum_slack,
                "incumbent_slack_objective": incumbent_slack,
                "robustly_inactive": bool(is_inactive),
                "early_stop_enabled": early_stop_enabled,
                "slack_stop_tolerance": zero_slack_tol,
                "termination_condition": str(termination),
                "result_classification": result_classification,
            }
            if is_inactive:
                var_name = self._binary_name(side, constraint_type)
                fixed_binaries.setdefault(var_name, {})[self._json_key(index)] = {
                    "fixed_value": 0,
                    "minimum_slack": float(minimum_slack),
                    "side": side,
                    "constraint_type": constraint_type,
                    "result_classification": result_classification,
                }

        self.slack_bounds = slack_bounds
        self.fixed_binaries = fixed_binaries
        return {
            "epsilon": float(epsilon),
            "stop_at_zero_slack": bool(stop_at_zero_slack),
            "early_stop_enabled": early_stop_enabled,
            "slack_stop_tolerance": zero_slack_tol,
            "slack_bounds": {
                f"{side}:{constraint_type}:{self._json_key(index)}": value
                for (side, constraint_type, index), value in slack_bounds.items()
            },
            "fixed_binaries": fixed_binaries,
            "num_fixed_binaries": int(sum(len(entries) for entries in fixed_binaries.values())),
        }

    def run_dual_big_m_tightening(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        fixed_binaries: Optional[dict[str, dict[str, Any]]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
    ) -> dict[str, Any]:
        """
        Stage 3: maximize each dual variable after fixing robustly inactive
        complementarity binaries.

        The optimal dual value is a data-driven Big-M for the corresponding
        `mu <= M z` constraint. If the dual's binary was fixed to zero, the tight
        Big-M is recorded as zero.
        """
        alpha_bounds = alpha_bounds or getattr(self, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before Big-M tightening")
        fixed_binaries = fixed_binaries or getattr(self, "fixed_binaries", {})

        lambda_report = self.run_lambda_bound_tightening(
            alpha_bounds=alpha_bounds,
            fixed_binaries=fixed_binaries,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )

        tight_big_m: dict[str, dict[str, Any]] = {}
        tasks: list[tuple[str, str, tuple[int, ...]]] = []
        for side in ("eq", "opt"):
            for i, b in self.generator_block_pairs:
                for t in range(self.num_time_steps):
                    tasks.append((side, "upper", (int(i), int(b), int(t))))
                    tasks.append((side, "lower", (int(i), int(b), int(t))))
            for i in range(self.num_physical_generators):
                for t in range(self.num_time_steps):
                    tasks.append((side, "ramp_up", (int(i), int(t))))
                    tasks.append((side, "ramp_down", (int(i), int(t))))

        total_candidates = len(tasks)
        skipped_programs = sum(
            1
            for side, constraint_type, index in tasks
            if self._json_key(index)
            in fixed_binaries.get(self._binary_name(side, constraint_type), {})
        )
        total_programs = total_candidates - skipped_programs
        program_number = 0
        print(
            f"\nDual Big-M optimization programs: {total_programs} "
            f"({skipped_programs} skipped because slack fixed the binary)",
            flush=True,
        )

        for side, constraint_type, index in tasks:
            dual_name = self._dual_name(side, constraint_type)
            binary_name = self._binary_name(side, constraint_type)
            key = self._json_key(index)
            if key in fixed_binaries.get(binary_name, {}):
                print(
                    f"[Dual Big-M skip] side={side}, constraint={constraint_type}, "
                    f"index={index}, binary={binary_name} fixed to 0",
                    flush=True,
                )
                tight_big_m.setdefault(dual_name, {})[key] = {
                    "tight_big_m": 0.0,
                    "fixed_by_slack": True,
                    "termination_condition": "fixed_binary_zero",
                }
                continue

            program_number += 1
            print(
                f"[Dual Big-M {program_number}/{total_programs}] "
                f"maximize {dual_name}{index} for side={side}, "
                f"constraint={constraint_type}",
                flush=True,
            )
            m = self._build_side_kkt_model(
                side=side,
                alpha_bounds=alpha_bounds,
                include_complementarity=True,
                fixed_binaries=fixed_binaries,
            )
            dual_var = getattr(m, dual_name)
            dual_expr = dual_var[index]
            m.tightening_objective = Objective(expr=dual_expr, sense=maximize)
            solved, results = self._solve_tightening_model(m, solver_name, time_limit, tee)
            dual_bound = self._safe_value(dual_expr) if solved else None
            tight_big_m.setdefault(dual_name, {})[key] = {
                "tight_big_m": dual_bound,
                "fixed_by_slack": False,
                "termination_condition": str(results.solver.termination_condition),
            }

        # Use the certified componentwise bounds when maximizing aggregate dual
        # sums. Individual dual maxima need not be jointly attainable, so these
        # additional programs compute valid inequalities for the final PoA model
        # that shrink the rectangular Big-M feasible region.
        self.tight_big_m = tight_big_m
        aggregate_dual_bounds: dict[str, dict[str, dict[str, Any]]] = {}
        aggregate_tasks: list[tuple[str, str, int]] = [
            (side, constraint_type, t)
            for side in ("eq", "opt")
            for constraint_type in ("upper", "lower", "ramp_up", "ramp_down")
            for t in range(self.num_time_steps)
        ]
        print(
            f"\nAggregate dual-bound optimization programs: {len(aggregate_tasks)}",
            flush=True,
        )
        for program_number, (side, constraint_type, time_idx) in enumerate(
            aggregate_tasks,
            start=1,
        ):
            bound_key = self._aggregate_dual_bound_key(constraint_type)
            print(
                f"[Aggregate dual bound {program_number}/{len(aggregate_tasks)}] "
                f"maximize {bound_key}[{side},{time_idx}]",
                flush=True,
            )
            m = self._build_side_kkt_model(
                side=side,
                alpha_bounds=alpha_bounds,
                include_complementarity=True,
                fixed_binaries=fixed_binaries,
            )
            aggregate_expr = self._aggregate_dual_expression(
                m,
                side,
                constraint_type,
                time_idx,
            )
            m.tightening_objective = Objective(expr=aggregate_expr, sense=maximize)
            solved, results = self._solve_tightening_model(m, solver_name, time_limit, tee)
            aggregate_bound = self._safe_value(aggregate_expr) if solved else None
            details = {
                "tight_big_m": aggregate_bound,
                "side": side,
                "constraint_type": constraint_type,
                "termination_condition": str(results.solver.termination_condition),
            }
            aggregate_dual_bounds.setdefault(bound_key, {}).setdefault(side, {})[
                str(int(time_idx))
            ] = details
            tight_big_m.setdefault(bound_key, {})[
                f"{side},{int(time_idx)}"
            ] = details

        self.tight_big_m = tight_big_m
        self.aggregate_dual_bounds = aggregate_dual_bounds
        return {
            "lambda_bounds": lambda_report["lambda_bounds"],
            "tight_big_m": tight_big_m,
            "aggregate_dual_bounds": aggregate_dual_bounds,
        }

    def save_tightening_report(
        self,
        output_path: str | Path = "results/poa_bidding_blocks_tightening_report.json",
    ) -> Path:
        """
        Stage 4: save the fixed binaries, slack bounds, and tightened Big-M values.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fixed_binaries": getattr(self, "fixed_binaries", {}),
            "slack_bounds": {
                f"{side}:{constraint_type}:{self._json_key(index)}": value
                for (side, constraint_type, index), value in getattr(self, "slack_bounds", {}).items()
            },
            "lambda_bounds": getattr(self, "lambda_bounds", {}),
            "tight_big_m": getattr(self, "tight_big_m", {}),
            "aggregate_dual_bounds": getattr(self, "aggregate_dual_bounds", {}),
            "alpha_bounds": self._jsonify_indexed_dict(getattr(self, "alpha_bounds", {})),
            "alpha_optimization_results": getattr(
                self,
                "alpha_bound_optimization_results",
                {},
            ),
        }
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
        return path

    def run_robust_bound_tightening(
        self,
        epsilon: float = 1e-6,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        output_path: str | Path = "results/poa_bidding_blocks_tightening_report.json",
        tee: bool = False,
    ) -> dict[str, Any]:
        """
        End-to-end Stage 1-4 driver.
        """
        bid_report = self.compute_nn_certified_bid_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )
        slack_report = self.run_slack_based_obbt(
            epsilon=epsilon,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )
        big_m_report = self.run_dual_big_m_tightening(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
        )
        report_path = self.save_tightening_report(output_path)
        return {
            "bid_bounds": bid_report,
            "slack_obbt": slack_report,
            "big_m_tightening": big_m_report,
            "report_path": str(report_path),
        }
