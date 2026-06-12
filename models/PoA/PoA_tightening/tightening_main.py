from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from models.PoA.PoA_optimization import PoAOptimization

DEFAULT_TIGHTENING_OUTPUT_PATHS: dict[str, str] = {
    "primal_big_m": "results/poa_tightening/primal_big_m_report.json",
    "relu_bounds": "results/poa_tightening/relu_bounds_report.json",
    "alpha_bounds": "results/poa_tightening/alpha_bounds_report.json",
    "slack_binary_fix": "results/poa_tightening/slack_binary_fix_report.json",
    "dual_big_m": "results/poa_tightening/dual_big_m_report.json",
    "optimal_cost_bounds": "results/poa_tightening/optimal_cost_bounds_report.json",
    "equilibrium_cost_bounds": "results/poa_tightening/equilibrium_cost_bounds_report.json",
    "final": "results/poa_tightening/final_tightening_report.json",
}


class PoATighteningMain:
    """Shared orchestrator for the staged PoA tightening workflow."""

    _MAIN_STATE_ATTRS = {"poa", "tightening_data", "stage_reports", "stage_timings"}

    def __getattr__(self, name: str) -> Any:
        poa = self.__dict__.get("poa")
        if poa is not None:
            return getattr(poa, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._MAIN_STATE_ATTRS or "poa" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        setattr(self.__dict__["poa"], name, value)

    def __init__(
        self,
        scenarios_df,
        costs_df,
        ramps_df,
        num_time_steps: Optional[int] = None,
        ambiguity_set_config: Optional[dict[str, Any]] = None,
        nn_model_dir: Optional[str | Path] = None,
        nn_normalization_stats_path: Optional[str | Path] = None,
        nn_policy_generators: Optional[list[int | str]] = None,
        reference_case: str = "test_case_bidding_blocks",
        objective_mode: str = "difference",
        mccormick_bounds: Optional[dict[str, Any]] = None,
        use_default_bounds: bool = False,
        alpha_ordering_epsilon: float = PoAOptimization.DEFAULT_ALPHA_ORDERING_EPSILON,
    ) -> None:
        self.poa = PoAOptimization(
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            num_time_steps=num_time_steps,
            ambiguity_set_config=ambiguity_set_config,
            nn_model_dir=nn_model_dir,
            nn_normalization_stats_path=nn_normalization_stats_path,
            nn_policy_generators=nn_policy_generators,
            reference_case=reference_case,
            objective_mode=objective_mode,
            mccormick_bounds=mccormick_bounds,
            use_default_bounds=use_default_bounds,
            alpha_ordering_epsilon=alpha_ordering_epsilon,
        )
        self.tightening_data: dict[str, Any] = {}
        self.stage_reports: dict[str, dict[str, Any]] = {}
        self.stage_timings: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _json_key(indices: Any) -> str:
        if isinstance(indices, str):
            return indices
        if isinstance(indices, int):
            return str(indices)
        return ",".join(str(int(index)) for index in tuple(indices))

    def _jsonify_indexed_dict(self, payload: dict[Any, Any]) -> dict[str, Any]:
        return {self._json_key(key): value for key, value in payload.items()}

    @staticmethod
    def _load_json(path: str | Path) -> dict[str, Any]:
        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @staticmethod
    def _save_json(payload: dict[str, Any], path: str | Path) -> Path:
        json_path = Path(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
        return json_path

    def _parse_alpha_bounds(
        self,
        alpha_bounds: dict[str, Any],
    ) -> dict[tuple[int, int, int], dict[str, float]]:
        parsed: dict[tuple[int, int, int], dict[str, float]] = {}
        for key, value in (alpha_bounds or {}).items():
            index = self.poa._parse_json_index(key)
            if len(index) != 3:
                raise ValueError(f"Alpha-bound key '{key}' must have three indices")
            parsed[(int(index[0]), int(index[1]), int(index[2]))] = {
                "lower": float(value["lower"]),
                "upper": float(value["upper"]),
            }
        return parsed

    def _merge_report(self, report: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(report, dict):
            raise ValueError("Tightening report payload must be a dictionary")

        self.tightening_data.setdefault("metadata", {}).update(report.get("metadata", {}) or {})

        if isinstance(report.get("stage_reports"), dict):
            self.stage_reports.update(report["stage_reports"])

        relu_report = report.get("nn_relu_bounds_report")
        if isinstance(relu_report, dict) and relu_report:
            self.tightening_data["nn_relu_bounds_report"] = relu_report
            self.tightening_data["nn_relu_bounds"] = relu_report.get("nn_relu_bounds", {}) or {}
            self.poa._set_nn_relu_bounds_from_report(relu_report)
        elif "nn_relu_bounds" in report:
            self.tightening_data["nn_relu_bounds_report"] = report
            self.tightening_data["nn_relu_bounds"] = report.get("nn_relu_bounds", {}) or {}
            self.poa._set_nn_relu_bounds_from_report(report)

        if "primal_big_m" in report:
            self.tightening_data["primal_big_m"] = report.get("primal_big_m", {}) or {}
            self.poa.primal_big_m = self.tightening_data["primal_big_m"]
            self.poa._loaded_bounds_prepared = False

        if "alpha_bounds" in report:
            alpha_bounds = report.get("alpha_bounds", {}) or {}
            self.tightening_data["alpha_bounds"] = alpha_bounds
            self.poa.alpha_bounds = self._parse_alpha_bounds(alpha_bounds)

        if "alpha_optimization_results" in report:
            self.tightening_data["alpha_optimization_results"] = (
                report.get("alpha_optimization_results", {}) or {}
            )
            self.poa.alpha_bound_optimization_results = self.tightening_data[
                "alpha_optimization_results"
            ]

        if "slack_bounds" in report:
            self.tightening_data["slack_bounds"] = report.get("slack_bounds", {}) or {}
            self.poa.slack_bounds = self.tightening_data["slack_bounds"]

        if "fixed_binaries" in report:
            self.tightening_data["fixed_binaries"] = report.get("fixed_binaries", {}) or {}
            self.poa.fixed_binaries = self.tightening_data["fixed_binaries"]

        if "tight_big_m" in report:
            self.tightening_data["tight_big_m"] = report.get("tight_big_m", {}) or {}
            self.poa.tight_big_m = self.tightening_data["tight_big_m"]
            self.poa._loaded_bounds_prepared = False

        if "optimal_cost_bounds" in report:
            self.tightening_data["optimal_cost_bounds"] = (
                report.get("optimal_cost_bounds", {}) or {}
            )
            self.poa.optimal_cost_bounds = self.tightening_data["optimal_cost_bounds"]
            c_opt_bounds = self._extract_c_opt_bounds(self.poa.optimal_cost_bounds)
            if c_opt_bounds is not None and self.poa.mccormick_bounds is not None:
                self.poa.mccormick_bounds["C_opt"] = c_opt_bounds
                if self.poa.objective_mode == "piecewise_mccormick":
                    num_pieces = int(self.poa.mccormick_bounds.get("num_pieces", 0))
                    if num_pieces >= 2:
                        lower, upper = c_opt_bounds
                        step = (upper - lower) / num_pieces
                        self.poa.mccormick_bounds["C_opt_breakpoints"] = [
                            lower + step * idx for idx in range(num_pieces + 1)
                        ]

        if "optimal_cost_bound_optimization_results" in report:
            self.tightening_data["optimal_cost_bound_optimization_results"] = (
                report.get("optimal_cost_bound_optimization_results", {}) or {}
            )
            self.poa.optimal_cost_bound_optimization_results = self.tightening_data[
                "optimal_cost_bound_optimization_results"
            ]

        if "equilibrium_cost_bounds" in report:
            self.tightening_data["equilibrium_cost_bounds"] = (
                report.get("equilibrium_cost_bounds", {}) or {}
            )
            self.poa.equilibrium_cost_bounds = self.tightening_data["equilibrium_cost_bounds"]

        if "equilibrium_cost_bound_optimization_results" in report:
            self.tightening_data["equilibrium_cost_bound_optimization_results"] = (
                report.get("equilibrium_cost_bound_optimization_results", {}) or {}
            )

        if report.get("derived_poa_bounds"):
            self.tightening_data["derived_poa_bounds"] = report.get("derived_poa_bounds")

        return report

    @staticmethod
    def _extract_c_opt_bounds(
        optimal_cost_bounds: dict[str, Any],
    ) -> tuple[float, float] | None:
        bounds = optimal_cost_bounds or {}
        if "C_opt" in bounds and isinstance(bounds.get("C_opt"), dict):
            bounds = bounds.get("C_opt", {}) or {}
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        if lower is None or upper is None:
            return None
        return (float(lower), float(upper))

    def _load_previous_stage(self, stage_name: str, path: str | Path) -> dict[str, Any]:
        previous_path = Path(path)
        if not previous_path.exists():
            raise FileNotFoundError(
                f"Previous tightening report for stage '{stage_name}' was not found. "
                f"Expected path: {previous_path}"
            )
        report = self._load_json(previous_path)
        self.stage_reports[stage_name] = report
        self._merge_report(report)
        return report

    def _default_stage_report(self, stage_name: str) -> dict[str, Any]:
        """
        Build a loose-but-valid placeholder for a skipped tightening stage.

        These defaults keep downstream tightening diagnostics buildable when a
        component is intentionally disabled. They are not tight preprocessing
        results and should be interpreted as the loose baseline for that stage.
        """
        if stage_name == "primal_big_m":
            from models.PoA.PoA_tightening.compute_primal_big_m import (
                compute_primal_big_m_bounds,
                summarize_primal_big_m,
            )

            primal_big_m = compute_primal_big_m_bounds(self.poa)
            self.poa._mark_default_bound_used(
                "primal_big_m",
                "Analytic primal Big-M defaults were used by tightening_main.",
            )
            return {
                "metadata": {
                    "description": "Analytic primal Big-M defaults.",
                    "source": "analytic_default_bounds",
                    "reference_case": self.poa.reference_case,
                    "num_time_steps": self.poa.num_time_steps,
                    "summary": summarize_primal_big_m(primal_big_m),
                },
                "primal_big_m": primal_big_m,
            }

        if stage_name == "relu_bounds":
            report = self.poa.build_default_nn_relu_bounds_report()
            report.setdefault("metadata", {})["source"] = "default_loose_bounds"
            report["relu_fixed_binaries"] = {}
            self.poa._mark_default_bound_used(
                "nn_relu_bounds_report",
                "Default loose ReLU bounds were used by tightening_main.",
            )
            return report

        if stage_name == "alpha_bounds":
            alpha_bounds = self.poa.build_default_alpha_bounds_report()
            self.poa._mark_default_bound_used(
                "alpha_bounds",
                "Default loose alpha bounds were used by tightening_main.",
            )
            return {
                "metadata": {
                    "description": "Default loose alpha bounds.",
                    "source": "default_loose_bounds",
                    "reference_case": self.poa.reference_case,
                    "num_time_steps": self.poa.num_time_steps,
                    "num_optimization_programs": 0,
                },
                "alpha_bounds": alpha_bounds,
                "alpha_optimization_results": {},
                "num_optimization_programs": 0,
                "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            }

        if stage_name == "slack_binary_fix":
            return {
                "metadata": {
                    "description": "Default empty complementarity-binary fixing.",
                    "source": "default_loose_bounds",
                    "reference_case": self.poa.reference_case,
                    "num_time_steps": self.poa.num_time_steps,
                },
                "slack_bounds": {},
                "fixed_binaries": {},
                "num_fixed_binaries": 0,
                "primal_big_m": self.tightening_data.get("primal_big_m", {}),
                "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
                "alpha_optimization_results": self.tightening_data.get(
                    "alpha_optimization_results",
                    {},
                ),
            }

        if stage_name == "dual_big_m":
            tight_big_m = self.poa.build_default_tight_big_m_report()
            self.poa._mark_default_bound_used(
                "tight_big_m",
                "Default loose dual Big-M values were used by tightening_main.",
            )
            return {
                "metadata": {
                    "description": ("Default loose componentwise dual Big-M bounds."),
                    "source": "default_loose_bounds",
                    "reference_case": self.poa.reference_case,
                    "num_time_steps": self.poa.num_time_steps,
                },
                "tight_big_m": tight_big_m,
                "primal_big_m": self.tightening_data.get("primal_big_m", {}),
                "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
                "alpha_optimization_results": self.tightening_data.get(
                    "alpha_optimization_results",
                    {},
                ),
                "slack_bounds": self.tightening_data.get("slack_bounds", {}),
                "fixed_binaries": self.tightening_data.get("fixed_binaries", {}),
            }

        if stage_name == "optimal_cost_bounds":
            optimal_cost_bounds = self.poa.build_default_optimal_cost_bounds_report()
            self.poa._mark_default_bound_used(
                "optimal_cost_bounds",
                "Default loose optimal-cost bounds were used by tightening_main.",
            )
            return {
                "metadata": {
                    "description": "Default loose optimal-dispatch cost bounds.",
                    "source": "default_loose_bounds",
                    "reference_case": self.poa.reference_case,
                    "num_time_steps": self.poa.num_time_steps,
                    "num_optimization_programs": 0,
                },
                "optimal_cost_bounds": optimal_cost_bounds,
                "optimal_cost_bound_optimization_results": {},
                "num_optimization_programs": 0,
                "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            }

        raise ValueError(f"No default report builder is defined for stage '{stage_name}'")

    def _load_default_stage(
        self,
        stage_name: str,
        output_path: str | Path | None,
    ) -> dict[str, Any]:
        print(
            f"Using default loose inputs for skipped tightening stage: {stage_name}",
            flush=True,
        )
        return self._save_stage_report(
            stage_name,
            self._default_stage_report(stage_name),
            output_path,
        )

    def _save_stage_report(
        self,
        stage_name: str,
        report: dict[str, Any],
        output_path: str | Path | None,
    ) -> dict[str, Any]:
        self.stage_reports[stage_name] = report
        self._merge_report(report)
        if output_path is not None:
            self._save_json(report, output_path)
        return report

    def _record_stage_timing(
        self,
        stage_name: str,
        elapsed_seconds: float,
        mode: str,
        output_path: str | Path | None,
    ) -> None:
        """Record per-stage wall time and, for freshly computed stages, stamp it.

        ``mode`` is "run" (computed now), "loaded" (read from a previous report),
        or "default" (loose placeholder). Only "run" stamps and re-saves the
        stage report, so a loaded report keeps the computation time it was
        originally written with.
        """
        self.stage_timings[stage_name] = {
            "computation_time_seconds": elapsed_seconds,
            "mode": mode,
        }
        if mode != "run":
            return
        report = self.stage_reports.get(stage_name)
        if isinstance(report, dict):
            report.setdefault("metadata", {})["computation_time_seconds"] = elapsed_seconds
            if output_path is not None:
                self._save_json(report, output_path)

    def _load_or_run_stage(
        self,
        stage_name: str,
        run_flag: bool,
        run_callable: Callable[[str | Path | None], dict[str, Any]],
        previous_path: str | Path,
        output_path: str | Path | None,
        use_default_if_missing: bool = True,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if run_flag:
            report = run_callable(output_path)
            mode = "run"
        elif Path(previous_path).exists():
            report = self._load_previous_stage(stage_name, previous_path)
            mode = "loaded"
        elif use_default_if_missing:
            report = self._load_default_stage(stage_name, output_path)
            mode = "default"
        else:
            report = self._load_previous_stage(stage_name, previous_path)
            mode = "loaded"
        self._record_stage_timing(stage_name, time.perf_counter() - start, mode, output_path)
        return report

    def load_existing_tightening_report(self, path: str | Path) -> dict[str, Any]:
        report = self._load_json(path)
        self._merge_report(report)
        return report

    def _ambiguity_metadata(self) -> dict[str, Any]:
        poa = self.poa
        return {
            "regime_bounds": {
                "mu_D": list(poa.mu_D_bounds),
                "sigma_D": list(poa.sigma_D_bounds),
                "mu_W": list(poa.mu_W_bounds),
                "sigma_W": list(poa.sigma_W_bounds),
            },
            "fixed_parameters": {
                "rho_D": float(poa.demand_rho_fixed),
                "rho_W": float(poa.wind_rho_fixed),
                "tau_W": float(poa.wind_tau_fixed),
                "kappa": float(poa.ambiguity_kappa),
                "D_ref": float(poa.demand_D_ref),
            },
            "demand_shape": [float(value) for value in poa.demand_shape],
            "wind_shape": [float(value) for value in poa.wind_shape],
        }

    def _metadata(self) -> dict[str, Any]:
        return {
            "reference_case": self.poa.reference_case,
            "num_time_steps": self.poa.num_time_steps,
            "physical_generator_names": list(self.poa.physical_generator_names),
            "block_names": list(self.poa.block_names),
            "ambiguity_set": self._ambiguity_metadata(),
            "nn_policy_generators": list(getattr(self.poa, "nn_policy_generator_names", [])),
            "nn_model_dir": str(self.poa.nn_model_dir) if self.poa.nn_model_dir else None,
            "nn_normalization_stats_path": (
                str(self.poa.nn_normalization_stats_path)
                if self.poa.nn_normalization_stats_path
                else None
            ),
            "default_bounds_used": getattr(self.poa, "default_bounds_used", {}),
        }

    def save_final_report(self, output_path: str | Path) -> Path:
        payload = {
            "metadata": self._metadata(),
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "nn_relu_bounds_report": self.tightening_data.get("nn_relu_bounds_report", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
            "alpha_optimization_results": self.tightening_data.get(
                "alpha_optimization_results",
                {},
            ),
            "slack_bounds": self.tightening_data.get("slack_bounds", {}),
            "fixed_binaries": self.tightening_data.get("fixed_binaries", {}),
            "tight_big_m": self.tightening_data.get("tight_big_m", {}),
            "optimal_cost_bounds": self.tightening_data.get("optimal_cost_bounds", {}),
            "optimal_cost_bound_optimization_results": self.tightening_data.get(
                "optimal_cost_bound_optimization_results",
                {},
            ),
            "equilibrium_cost_bounds": self.tightening_data.get("equilibrium_cost_bounds", {}),
            "equilibrium_cost_bound_optimization_results": self.tightening_data.get(
                "equilibrium_cost_bound_optimization_results",
                {},
            ),
            "derived_poa_bounds": self.tightening_data.get("derived_poa_bounds"),
            "stage_reports": self.stage_reports,
            "stage_timings": self.stage_timings,
        }
        return self._save_json(payload, output_path)

    def _as_stage(self, stage_cls: type["PoATighteningMain"]) -> "PoATighteningMain":
        stage = stage_cls.__new__(stage_cls)
        stage.__dict__ = self.__dict__
        return stage

    def _require_relu_before_alpha(self) -> None:
        if getattr(self.poa, "nn_policy_generator_ids", []) and not self.poa.nn_relu_bounds:
            raise ValueError(
                "ReLU bounds must be computed or loaded before alpha-bound tightening."
            )

    def _require_alpha_bounds(self) -> None:
        if not getattr(self.poa, "alpha_bounds", None):
            raise ValueError("Alpha bounds must be computed or loaded before this stage.")

    def _require_dual_big_m_inputs(self) -> None:
        if not getattr(self.poa, "primal_big_m", None):
            raise ValueError("Primal Big-M values must be computed or loaded before dual Big-M.")
        self._require_alpha_bounds()
        if "fixed_binaries" not in self.tightening_data:
            raise ValueError(
                "Slack-based complementarity binary fixing must be computed or loaded "
                "before dual Big-M."
            )

    def run_all(
        self,
        run_primal_big_m: bool = True,
        run_relu_bounds: bool = True,
        run_alpha_bounds: bool = True,
        run_slack_binary_fix: bool = True,
        run_dual_big_m: bool = True,
        run_optimal_cost_bounds: bool = True,
        run_equilibrium_cost_bounds: bool = False,
        poa_bounds_lower: float = 1.0,
        poa_bounds_margin: float = 1e-3,
        previous_paths: Optional[dict[str, str | Path]] = None,
        output_paths: Optional[dict[str, str | Path]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        relu_tolerance: float = 1e-9,
        epsilon: float = 1e-6,
        relax_complementarity: bool = False,
        stop_at_zero_slack: bool = True,
        slack_stop_tol: Optional[float] = None,
        parallel_workers: Optional[int] = 1,
        solver_threads: Optional[int] = None,
        use_default_stage_inputs: bool = True,
    ) -> Path:
        from models.PoA.PoA_tightening.compute_alpha_bounds import AlphaBoundsComputer
        from models.PoA.PoA_tightening.compute_dual_big_m import DualBigMComputer
        from models.PoA.PoA_tightening.compute_equilibrium_cost_bounds import (
            EquilibriumCostBoundsComputer,
        )
        from models.PoA.PoA_tightening.compute_optimal_cost_bounds import OptimalCostBoundsComputer
        from models.PoA.PoA_tightening.compute_primal_big_m import PrimalBigMComputer
        from models.PoA.PoA_tightening.compute_relu_bounds import ReLUBoundsComputer
        from models.PoA.PoA_tightening.compute_slack_binary_fix import SlackBinaryFixComputer

        previous_paths = {**DEFAULT_TIGHTENING_OUTPUT_PATHS, **(previous_paths or {})}
        output_paths = {**DEFAULT_TIGHTENING_OUTPUT_PATHS, **(output_paths or {})}

        primal_stage = self._as_stage(PrimalBigMComputer)
        relu_stage = self._as_stage(ReLUBoundsComputer)
        alpha_stage = self._as_stage(AlphaBoundsComputer)
        slack_stage = self._as_stage(SlackBinaryFixComputer)
        dual_stage = self._as_stage(DualBigMComputer)
        optimal_cost_stage = self._as_stage(OptimalCostBoundsComputer)
        equilibrium_cost_stage = self._as_stage(EquilibriumCostBoundsComputer)

        self._load_or_run_stage(
            "primal_big_m",
            run_primal_big_m,
            lambda output_path: primal_stage.run_primal_big_m(output_path=output_path),
            previous_paths["primal_big_m"],
            output_paths["primal_big_m"],
            use_default_if_missing=use_default_stage_inputs,
        )

        self._load_or_run_stage(
            "relu_bounds",
            run_relu_bounds,
            lambda output_path: relu_stage.run_relu_bounds(
                output_path=output_path,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                tolerance=relu_tolerance,
            ),
            previous_paths["relu_bounds"],
            output_paths["relu_bounds"],
            use_default_if_missing=use_default_stage_inputs,
        )

        self._require_relu_before_alpha()
        self._load_or_run_stage(
            "alpha_bounds",
            run_alpha_bounds,
            lambda output_path: alpha_stage.run_alpha_bounds(
                output_path=output_path,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                parallel_workers=parallel_workers,
                solver_threads=solver_threads,
            ),
            previous_paths["alpha_bounds"],
            output_paths["alpha_bounds"],
            use_default_if_missing=use_default_stage_inputs,
        )

        self._require_alpha_bounds()
        self._load_or_run_stage(
            "slack_binary_fix",
            run_slack_binary_fix,
            lambda output_path: slack_stage.run_slack_binary_fix(
                output_path=output_path,
                epsilon=epsilon,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                relax_complementarity=relax_complementarity,
                stop_at_zero_slack=stop_at_zero_slack,
                slack_stop_tol=slack_stop_tol,
                parallel_workers=parallel_workers,
                solver_threads=solver_threads,
            ),
            previous_paths["slack_binary_fix"],
            output_paths["slack_binary_fix"],
            use_default_if_missing=use_default_stage_inputs,
        )

        self._require_dual_big_m_inputs()
        self._load_or_run_stage(
            "dual_big_m",
            run_dual_big_m,
            lambda output_path: dual_stage.run_dual_big_m(
                output_path=output_path,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                parallel_workers=parallel_workers,
                solver_threads=solver_threads,
            ),
            previous_paths["dual_big_m"],
            output_paths["dual_big_m"],
            use_default_if_missing=use_default_stage_inputs,
        )

        self._load_or_run_stage(
            "optimal_cost_bounds",
            run_optimal_cost_bounds,
            lambda output_path: optimal_cost_stage.run_optimal_cost_bounds(
                output_path=output_path,
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                solver_threads=solver_threads,
            ),
            previous_paths["optimal_cost_bounds"],
            output_paths["optimal_cost_bounds"],
            use_default_if_missing=use_default_stage_inputs,
        )

        # Optional equilibrium cost bound (gated by run_equilibrium_cost_bounds).
        # Participates only when computing now or when a previous report exists to
        # reuse; otherwise it is skipped so the final report omits the section and
        # the McCormick PoA box falls back to the hand-set *_mccormick_PoA_bounds.
        eq_previous = previous_paths.get("equilibrium_cost_bounds")
        if run_equilibrium_cost_bounds or (eq_previous and Path(eq_previous).exists()):
            if run_equilibrium_cost_bounds:
                self._require_alpha_bounds()
            self._load_or_run_stage(
                "equilibrium_cost_bounds",
                run_equilibrium_cost_bounds,
                lambda output_path: equilibrium_cost_stage.run_equilibrium_cost_bounds(
                    output_path=output_path,
                    solver_name=solver_name,
                    time_limit=time_limit,
                    tee=tee,
                    solver_threads=solver_threads,
                    poa_bounds_lower=poa_bounds_lower,
                    poa_bounds_margin=poa_bounds_margin,
                ),
                eq_previous,
                output_paths.get("equilibrium_cost_bounds"),
                use_default_if_missing=False,
            )

        return self.save_final_report(output_paths["final"])
