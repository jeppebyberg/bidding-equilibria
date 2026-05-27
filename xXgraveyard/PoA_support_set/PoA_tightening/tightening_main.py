from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from models.PoA.PoA_optimization import PoAOptimization


DEFAULT_TIGHTENING_OUTPUT_PATHS: dict[str, str] = {
    "primal_big_m": "results/poa_tightening/primal_big_m_report.json",
    "relu_bounds": "results/poa_tightening/relu_bounds_report.json",
    "alpha_bounds": "results/poa_tightening/alpha_bounds_report.json",
    "slack_binary_fix": "results/poa_tightening/slack_binary_fix_report.json",
    "dual_big_m": "results/poa_tightening/dual_big_m_report.json",
    "final": "results/poa_tightening/final_tightening_report.json",
}


class PoATighteningMain:
    """Shared orchestrator for the staged PoA tightening workflow."""

    _MAIN_STATE_ATTRS = {"poa", "tightening_data", "stage_reports"}

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
        p_init: Optional[list[float] | list[list[float]]] = None,
        num_time_steps: Optional[int] = None,
        support_set_config: Optional[dict[str, Any]] = None,
        nn_model_dir: Optional[str | Path] = None,
        nn_normalization_stats_path: Optional[str | Path] = None,
        nn_policy_generators: Optional[list[int | str]] = None,
        reference_case: str = "test_case_bidding_blocks",
    ) -> None:
        self.poa = PoAOptimization(
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            p_init=p_init,
            num_time_steps=num_time_steps,
            support_set_config=support_set_config,
            nn_model_dir=nn_model_dir,
            nn_normalization_stats_path=nn_normalization_stats_path,
            nn_policy_generators=nn_policy_generators,
            reference_case=reference_case,
        )
        self.tightening_data: dict[str, Any] = {}
        self.stage_reports: dict[str, dict[str, Any]] = {}

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
            self.poa.alpha_bound_optimization_results = (
                self.tightening_data["alpha_optimization_results"]
            )

        if "slack_bounds" in report:
            self.tightening_data["slack_bounds"] = report.get("slack_bounds", {}) or {}
            self.poa.slack_bounds = self.tightening_data["slack_bounds"]

        if "fixed_binaries" in report:
            self.tightening_data["fixed_binaries"] = report.get("fixed_binaries", {}) or {}
            self.poa.fixed_binaries = self.tightening_data["fixed_binaries"]

        if "lambda_bounds" in report:
            self.tightening_data["lambda_bounds"] = report.get("lambda_bounds", {}) or {}
            self.poa.lambda_bounds = self.tightening_data["lambda_bounds"]
            self.poa._loaded_bounds_prepared = False

        if "tight_big_m" in report:
            self.tightening_data["tight_big_m"] = report.get("tight_big_m", {}) or {}
            self.poa.tight_big_m = self.tightening_data["tight_big_m"]
            self.poa._loaded_bounds_prepared = False

        if "aggregate_dual_bounds" in report:
            self.tightening_data["aggregate_dual_bounds"] = (
                report.get("aggregate_dual_bounds", {}) or {}
            )
            self.poa.aggregate_dual_bounds = self.tightening_data["aggregate_dual_bounds"]
            self.poa._loaded_bounds_prepared = False

        return report

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

    def _load_or_run_stage(
        self,
        stage_name: str,
        run_flag: bool,
        run_callable: Callable[[str | Path | None], dict[str, Any]],
        previous_path: str | Path,
        output_path: str | Path | None,
    ) -> dict[str, Any]:
        if run_flag:
            return run_callable(output_path)
        return self._load_previous_stage(stage_name, previous_path)

    def load_existing_tightening_report(self, path: str | Path) -> dict[str, Any]:
        report = self._load_json(path)
        self._merge_report(report)
        return report

    def _metadata(self) -> dict[str, Any]:
        return {
            "reference_case": self.poa.reference_case,
            "num_time_steps": self.poa.num_time_steps,
            "physical_generator_names": list(self.poa.physical_generator_names),
            "block_names": list(self.poa.block_names),
            "nn_policy_generators": list(getattr(self.poa, "nn_policy_generator_names", [])),
            "nn_model_dir": str(self.poa.nn_model_dir) if self.poa.nn_model_dir else None,
            "nn_normalization_stats_path": (
                str(self.poa.nn_normalization_stats_path)
                if self.poa.nn_normalization_stats_path
                else None
            ),
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
            "lambda_bounds": self.tightening_data.get("lambda_bounds", {}),
            "tight_big_m": self.tightening_data.get("tight_big_m", {}),
            "aggregate_dual_bounds": self.tightening_data.get("aggregate_dual_bounds", {}),
            "stage_reports": self.stage_reports,
        }
        return self._save_json(payload, output_path)

    def _as_stage(self, stage_cls: type["PoATighteningMain"]) -> "PoATighteningMain":
        stage = stage_cls.__new__(stage_cls)
        stage.__dict__ = self.__dict__
        return stage

    def _require_relu_before_alpha(self) -> None:
        if getattr(self.poa, "nn_policy_generator_ids", []) and not self.poa.nn_relu_bounds:
            raise ValueError("ReLU bounds must be computed or loaded before alpha-bound tightening.")

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
    ) -> Path:
        from models.PoA.PoA_tightening.compute_alpha_bounds import AlphaBoundsComputer
        from models.PoA.PoA_tightening.compute_dual_big_m import DualBigMComputer
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

        self._load_or_run_stage(
            "primal_big_m",
            run_primal_big_m,
            lambda output_path: primal_stage.run_primal_big_m(output_path=output_path),
            previous_paths["primal_big_m"],
            output_paths["primal_big_m"],
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
        )

        return self.save_final_report(output_paths["final"])
