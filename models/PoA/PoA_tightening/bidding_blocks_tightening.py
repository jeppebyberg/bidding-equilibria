from pathlib import Path
import json
from pyomo.environ import *
from typing import Any, Optional

from models.PoA.PoA_optimization_bidding_blocks_tmp import PoAOptimizationBiddingBlocks


class BiddingBlocksTighteningOptimizer(PoAOptimizationBiddingBlocks):
    """Optimizer subclass with the expensive tightening construction routines.

    The base tmp class is kept focused on the final PoA model plus loading and
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
        self._build_bid_monotonicity_constraints()
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
                self._build_valid_inequalities()
            else:
                for var_name in self._binary_components_for_side(side):
                    binary_var = getattr(self.model, var_name, None)
                    if binary_var is not None:
                        for index in binary_var:
                            binary_var[index].fix(0)
            self._build_bid_monotonicity_constraints()
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

    def _solve_tightening_model(
        self,
        m: ConcreteModel,
        solver_name: str,
        time_limit: Optional[float],
        tee: bool,
    ) -> tuple[bool, Any]:
        solver = SolverFactory(solver_name)
        if time_limit is not None:
            solver.options["TimeLimit"] = float(time_limit)
        results = solver.solve(m, tee=tee)
        termination = results.solver.termination_condition
        ok = termination in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
        return bool(ok), results

    def run_slack_based_obbt(
        self,
        alpha_bounds: Optional[dict[tuple[int, int, int], dict[str, float]]] = None,
        epsilon: float = 1e-6,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        relax_complementarity: bool = False,
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
        """
        alpha_bounds = alpha_bounds or getattr(self, "alpha_bounds", None)
        if alpha_bounds is None:
            raise ValueError("Call compute_nn_certified_bid_bounds() before slack OBBT")

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
            solved, results = self._solve_tightening_model(m, solver_name, time_limit, tee)
            min_slack = self._safe_value(m.target_slack) if solved else None
            is_inactive = min_slack is not None and min_slack >= float(epsilon)

            record_key = (side, constraint_type, index)
            slack_bounds[record_key] = {
                "minimum_slack": min_slack,
                "robustly_inactive": bool(is_inactive),
                "termination_condition": str(results.solver.termination_condition),
            }
            if is_inactive:
                var_name = self._binary_name(side, constraint_type)
                fixed_binaries.setdefault(var_name, {})[self._json_key(index)] = {
                    "fixed_value": 0,
                    "minimum_slack": float(min_slack),
                    "side": side,
                    "constraint_type": constraint_type,
                }

        self.slack_bounds = slack_bounds
        self.fixed_binaries = fixed_binaries
        return {
            "epsilon": float(epsilon),
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

        self.tight_big_m = tight_big_m
        return {"tight_big_m": tight_big_m}

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
            "tight_big_m": getattr(self, "tight_big_m", {}),
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

