from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from models.DRO_PoA.DRO_PoA_optimization_new import DRO_PoAOptimization

DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS: dict[str, str] = {
    "primal_big_m": "results/dro_poa_tightening/{regime_name}/primal_big_m_report.json",
    "relu_bounds": "results/dro_poa_tightening/{regime_name}/relu_bounds_report.json",
    "alpha_bounds": "results/dro_poa_tightening/{regime_name}/alpha_bounds_report.json",
    "slack_binary_fix": "results/dro_poa_tightening/{regime_name}/slack_binary_fix_report.json",
    "dual_big_m": "results/dro_poa_tightening/{regime_name}/dual_big_m_report.json",
    "optimal_cost_bounds": "results/dro_poa_tightening/{regime_name}/optimal_cost_bounds_report.json",
    "equilibrium_cost_bounds": "results/dro_poa_tightening/{regime_name}/equilibrium_cost_bounds_report.json",
    "final": "results/dro_poa_tightening/{regime_name}/final_tightening_report.json",
}


class DROPoATighteningMain:
    """Shared orchestrator for DRO PoA tightening."""

    _MAIN_STATE_ATTRS = {
        "poa",
        "dro",
        "dro_poa",
        "tightening_data",
        "stage_reports",
        "stage_timings",
    }

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
        regime_config_path: str | Path = "config/regime_definitions.yaml",
        regime_set: str = "PoA_analysis",
        regime_name: Optional[str] = None,
        regime_parameters: Optional[dict[str, Any]] = None,
        ambiguity_set_config_path: str | Path = "config/ambiguity_set_config.yaml",
        ambiguity_set_config_name: Optional[str] = None,
        eta: float = 0.0,
        epsilon: float = 0.0,
        nn_model_dir: Optional[str | Path] = None,
        nn_normalization_stats_path: Optional[str | Path] = None,
        nn_policy_generators: Optional[list[int | str]] = None,
        reference_case: str = "base_test_case",
        case_label: str = "",
        objective_mode: str = "piecewise_mccormick",
        mccormick_bounds: Optional[dict[str, Any]] = None,
        defer_mccormick_bound_validation: bool = False,
        ambiguity_kappa: float = 0.3,
        use_default_bounds: bool = False,
        ar1_coverage: Optional[float] = None,
        alpha_ordering_epsilon: float = (
            DRO_PoAOptimization.DEFAULT_ALPHA_ORDERING_EPSILON
        ),
    ) -> None:
        # The optimizer's regime variables r^DRO are always bounded by the ambiguity
        # set R (ambiguity_set_config).  regime_parameters, when given, supplies the
        # regime context directly (e.g. derived from R) and bypasses
        # regime_definitions; when None, the regime is loaded from regime_definitions
        # as before (the empirical regime is then a fixed named regime, still
        # transported to an r^DRO that lives in R).
        self.poa = DRO_PoAOptimization(
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            num_time_steps=num_time_steps,
            regime_config_path=regime_config_path,
            regime_set=regime_set,
            regime_name=regime_name,
            regime_parameters=regime_parameters,
            ambiguity_set_config_path=ambiguity_set_config_path,
            ambiguity_set_config_name=ambiguity_set_config_name,
            eta=eta,
            epsilon=epsilon,
            nn_model_dir=nn_model_dir,
            nn_normalization_stats_path=nn_normalization_stats_path,
            nn_policy_generators=nn_policy_generators,
            reference_case=reference_case,
            case_label=case_label,
            objective_mode=objective_mode,
            mccormick_bounds=mccormick_bounds,
            defer_mccormick_bound_validation=defer_mccormick_bound_validation,
            ambiguity_kappa=ambiguity_kappa,
            use_default_bounds=use_default_bounds,
            ar1_coverage=ar1_coverage,
            alpha_ordering_epsilon=alpha_ordering_epsilon,
        )
        self.dro = self.poa
        self.dro_poa = self.poa
        self.tightening_data: dict[str, Any] = {}
        self.stage_reports: dict[str, dict[str, Any]] = {}
        self.stage_timings: dict[str, dict[str, Any]] = {}

    def _resolve_output_paths(
        self,
        output_paths: Optional[dict[str, str | Path]] = None,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        regime_name = str(self.poa.regime_name)
        for stage_name, template in DEFAULT_DRO_TIGHTENING_OUTPUT_PATHS.items():
            raw_path = (output_paths or {}).get(stage_name, template)
            resolved[stage_name] = str(raw_path).format(regime_name=regime_name)
        return resolved

    @staticmethod
    def _json_key(indices: Any) -> str:
        if isinstance(indices, str):
            return indices
        if isinstance(indices, int):
            return str(indices)
        return ",".join(str(int(index)) for index in tuple(indices))

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in str(key).split(",") if part != "")

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

    def _merge_report(self, report: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(report, dict):
            raise ValueError("Tightening report payload must be a dictionary")

        self._validate_report_metadata_if_present(report)
        self.tightening_data.setdefault("metadata", {}).update(report.get("metadata", {}) or {})
        if isinstance(report.get("stage_reports"), dict):
            self.stage_reports.update(report["stage_reports"])

        relu_report = report.get("nn_relu_bounds_report")
        if isinstance(relu_report, dict) and relu_report:
            self.tightening_data["nn_relu_bounds_report"] = relu_report
            self.tightening_data["scenario_nn_relu_bounds"] = (
                relu_report.get("scenario_nn_relu_bounds", {}) or {}
            )
            self.tightening_data["regime_nn_relu_bounds"] = (
                relu_report.get("regime_nn_relu_bounds", {}) or {}
            )
            self.tightening_data["nn_relu_bounds"] = self._prefer_scenario_report_field(
                relu_report,
                "scenario_nn_relu_bounds",
                "nn_relu_bounds",
            )
            if hasattr(self.poa, "_set_nn_relu_bounds_from_report"):
                self.poa._set_nn_relu_bounds_from_report(
                    {
                        **relu_report,
                        "nn_relu_bounds": self.tightening_data["nn_relu_bounds"],
                    }
                )
        elif "nn_relu_bounds" in report or "scenario_nn_relu_bounds" in report:
            self.tightening_data["nn_relu_bounds_report"] = report
            self.tightening_data["scenario_nn_relu_bounds"] = (
                report.get("scenario_nn_relu_bounds", {}) or {}
            )
            self.tightening_data["regime_nn_relu_bounds"] = (
                report.get("regime_nn_relu_bounds", {}) or {}
            )
            self.tightening_data["nn_relu_bounds"] = self._prefer_scenario_report_field(
                report,
                "scenario_nn_relu_bounds",
                "nn_relu_bounds",
            )
            if hasattr(self.poa, "_set_nn_relu_bounds_from_report"):
                self.poa._set_nn_relu_bounds_from_report(
                    {
                        **report,
                        "nn_relu_bounds": self.tightening_data["nn_relu_bounds"],
                    }
                )

        for key in (
            "primal_big_m",
            "alpha_optimization_results",
            "slack_bounds",
        ):
            if key in report:
                self.tightening_data[key] = report.get(key, {}) or {}
                setattr(self.poa, key, self.tightening_data[key])

        scenario_preferred_fields = {
            "alpha_bounds": "scenario_alpha_bounds",
            "fixed_binaries": "scenario_fixed_binaries",
            "tight_big_m": "scenario_tight_big_m",
        }
        for main_key, scenario_key in scenario_preferred_fields.items():
            regime_key = f"regime_{main_key}"
            if scenario_key in report:
                self.tightening_data[scenario_key] = report.get(scenario_key, {}) or {}
            if regime_key in report:
                self.tightening_data[regime_key] = report.get(regime_key, {}) or {}
            if main_key in report or scenario_key in report:
                selected = self._prefer_scenario_report_field(report, scenario_key, main_key)
                self.tightening_data[main_key] = selected
                setattr(self.poa, main_key, selected)

        if "optimal_cost_bounds" in report:
            self.tightening_data["optimal_cost_bounds"] = (
                report.get("optimal_cost_bounds", {}) or {}
            )
        if "scenario_optimal_cost_bounds" in report:
            self.tightening_data["scenario_optimal_cost_bounds"] = (
                report.get("scenario_optimal_cost_bounds", {}) or {}
            )
        if "optimal_cost_bound_optimization_results" in report:
            self.tightening_data["optimal_cost_bound_optimization_results"] = (
                report.get("optimal_cost_bound_optimization_results", {}) or {}
            )
        if (
            "optimal_cost_bounds" in report
            or "scenario_optimal_cost_bounds" in report
            or "optimal_cost_bound_optimization_results" in report
        ) and hasattr(self.poa, "_set_optimal_cost_bounds_from_report"):
            self.poa._set_optimal_cost_bounds_from_report(report)

        for key in (
            "equilibrium_cost_bounds",
            "scenario_equilibrium_cost_bounds",
            "equilibrium_cost_bound_optimization_results",
        ):
            if key in report:
                self.tightening_data[key] = report.get(key, {}) or {}
        if report.get("derived_poa_bounds"):
            self.tightening_data["derived_poa_bounds"] = report.get("derived_poa_bounds")

        if "alpha_bounds" in self.tightening_data:
            self.poa.alpha_bounds = self._parse_alpha_bounds(
                self.tightening_data.get("alpha_bounds", {}) or {}
            )
        if hasattr(self.poa, "_loaded_bounds_prepared"):
            self.poa._loaded_bounds_prepared = False
        return report

    def _validate_report_metadata_if_present(self, report: dict[str, Any]) -> None:
        metadata = report.get("metadata", {}) or {}
        if not isinstance(metadata, dict) or not metadata:
            return

        report_regime = metadata.get("regime_name")
        if report_regime is not None and str(report_regime) != str(self.poa.regime_name):
            raise ValueError(
                "DRO tightening report regime_name mismatch: "
                f"report has {report_regime!r}, current regime is {self.poa.regime_name!r}"
            )

        report_time_steps = metadata.get("num_time_steps")
        if report_time_steps is not None and int(report_time_steps) != int(self.poa.num_time_steps):
            raise ValueError(
                "DRO tightening report num_time_steps mismatch: "
                f"report has {report_time_steps}, current model has {self.poa.num_time_steps}"
            )

        report_scenarios = metadata.get("num_empirical_scenarios")
        if report_scenarios is not None and int(report_scenarios) != int(
            self.poa.num_empirical_scenarios
        ):
            raise ValueError(
                "DRO tightening report num_empirical_scenarios mismatch: "
                f"report has {report_scenarios}, current model has "
                f"{self.poa.num_empirical_scenarios}. Rerun this DRO tightening "
                "stage for the current scenario set."
            )

    def _load_previous_stage(self, stage_name: str, path: str | Path) -> dict[str, Any]:
        previous_path = Path(path)
        if not previous_path.exists():
            raise FileNotFoundError(
                f"Previous DRO tightening report for stage '{stage_name}' was not found. "
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

    def _record_stage_timing(
        self,
        stage_name: str,
        elapsed_seconds: float,
        mode: str,
        output_path: str | Path | None,
    ) -> None:
        """Record per-stage wall time and, for freshly computed stages, stamp it.

        ``mode`` is "run" (computed now) or "loaded" (read from a previous
        report). Only "run" stamps and re-saves the stage report, so a loaded
        report keeps the computation time it was originally written with.
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
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if run_flag:
            report = run_callable(output_path)
            mode = "run"
        else:
            report = self._load_previous_stage(stage_name, previous_path)
            mode = "loaded"
        self._record_stage_timing(stage_name, time.perf_counter() - start, mode, output_path)
        return report

    def load_existing_tightening_report(self, path: str | Path) -> dict[str, Any]:
        report = self._load_json(path)
        self._merge_report(report)
        return report

    def _as_stage(self, stage_cls: type["DROPoATighteningMain"]) -> "DROPoATighteningMain":
        stage = stage_cls.__new__(stage_cls)
        stage.__dict__ = self.__dict__
        return stage

    def _parallel_stage_state(self) -> dict[str, Any]:
        return {
            "p_init": [list(values) for values in self.poa.p_init],
            "constructor_kwargs": {
                "scenarios_df": self.poa.scenarios_df,
                "costs_df": self.poa.costs_df,
                "ramps_df": self.poa.ramps_df,
                "num_time_steps": self.poa.num_time_steps,
                "regime_config_path": self.poa.regime_config_path,
                "regime_set": self.poa.regime_set,
                "regime_name": self.poa.regime_name,
                # Carry the exact regime context so parallel workers rebuild the
                # optimizer from it (and bound r^DRO by the ambiguity set R) instead
                # of re-reading regime_definitions.  scenarios_df is already the
                # regime's scenarios, so the override path applies no extra filter.
                "regime_parameters": (
                    dict(getattr(self.poa, "selected_regime_parameters", {})) or None
                ),
                "ambiguity_set_config_path": str(self.poa.ambiguity_set_config_path),
                "ambiguity_set_config_name": self.poa.ambiguity_set_config_name,
                "defer_mccormick_bound_validation": getattr(
                    self.poa, "defer_mccormick_bound_validation", False
                ),
                "eta": self.poa.eta,
                "epsilon": self.poa.epsilon,
                "nn_model_dir": self.poa.nn_model_dir,
                "nn_normalization_stats_path": self.poa.nn_normalization_stats_path,
                "nn_policy_generators": self.poa.requested_nn_policy_generators,
                "reference_case": self.poa.reference_case,
                "objective_mode": self.poa.objective_mode,
                "mccormick_bounds": self.poa.mccormick_bounds,
                "use_default_bounds": self.poa.use_default_bounds,
                "ar1_coverage": getattr(self.poa, "ar1_coverage", None),
                "alpha_ordering_epsilon": self.poa.alpha_ordering_epsilon,
            },
            "tightening_data": dict(self.tightening_data),
        }

    @classmethod
    def _from_parallel_stage_state(
        cls,
        state: dict[str, Any],
    ) -> "DROPoATighteningMain":
        stage = cls(**state["constructor_kwargs"])
        if "p_init" in state:
            stage.poa.p_init = state["p_init"]
        tightening_data = dict(state.get("tightening_data", {}) or {})
        stage.tightening_data.update(tightening_data)
        stage._merge_report(tightening_data)
        return stage

    @staticmethod
    def _resolve_parallel_workers(
        parallel_workers: Optional[int],
        total_tasks: int,
    ) -> int:
        if total_tasks <= 1 or parallel_workers is None:
            return 1
        return min(max(1, int(parallel_workers)), int(total_tasks))

    @staticmethod
    def _solver_options_with_threads(
        solver_name: str,
        solver_threads: Optional[int],
        extra_options: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        options = dict(extra_options or {})
        if solver_threads is not None and solver_name == "gurobi":
            options["Threads"] = int(solver_threads)
        return options or None

    def _parse_alpha_bounds(
        self,
        alpha_bounds: dict[str, Any],
    ) -> dict[tuple[int, ...], dict[str, float]]:
        parsed: dict[tuple[int, ...], dict[str, float]] = {}
        for key, value in (alpha_bounds or {}).items():
            index = self._parse_json_index(key)
            if len(index) not in {3, 4}:
                raise ValueError(f"Alpha-bound key '{key}' must have i,b,t or k,i,b,t indices")
            parsed[tuple(int(part) for part in index)] = {
                "lower": float(value["lower"]),
                "upper": float(value["upper"]),
            }
        return parsed

    def _has_scenario_indexed_keys(self, payload: Any, expected_scenario_dim: int) -> bool:
        if not isinstance(payload, dict):
            return False
        entries = (
            payload.values()
            if payload and all(isinstance(value, dict) for value in payload.values())
            else [payload]
        )
        for maybe_entries in entries:
            if not isinstance(maybe_entries, dict):
                continue
            for key in maybe_entries:
                try:
                    if len(self._parse_json_index(key)) == int(expected_scenario_dim):
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    @staticmethod
    def _prefer_scenario_report_field(
        report: dict[str, Any],
        scenario_key: str,
        fallback_key: str,
    ) -> Any:
        scenario_payload = report.get(scenario_key)
        if scenario_payload:
            return scenario_payload
        return report.get(fallback_key, {}) or {}

    def _require_relu_before_alpha(self) -> None:
        if getattr(self.poa, "nn_policy_generator_ids", []) and not getattr(
            self.poa,
            "nn_relu_bounds",
            {},
        ):
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

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_type": "DRO_PoA",
            "tightening_type": "regime_wide",
            "tightening_scope": "scenario_wise",
            "reference_case": self.poa.reference_case,
            "regime_set": self.poa.regime_set,
            "regime_name": self.poa.regime_name,
            "eta": float(self.poa.eta),
            "epsilon": float(self.poa.epsilon),
            "objective_mode": self.poa.objective_mode,
            "mccormick_bounds": getattr(self.poa, "mccormick_bounds", None),
            "num_time_steps": int(self.poa.num_time_steps),
            "num_empirical_scenarios": int(self.poa.num_empirical_scenarios),
            "physical_generator_names": list(self.poa.physical_generator_names),
            "block_names": list(self.poa.block_names),
            "nn_policy_generators": list(getattr(self.poa, "nn_policy_generator_names", [])),
            "nn_model_dir": str(self.poa.nn_model_dir) if self.poa.nn_model_dir else None,
            "nn_normalization_stats_path": (
                str(self.poa.nn_normalization_stats_path)
                if self.poa.nn_normalization_stats_path
                else None
            ),
            "selected_regime_parameters": dict(getattr(self.poa, "selected_regime_parameters", {})),
            "aggregation_rule": (
                "applies only to regime_* diagnostic/fallback fields: minimum lower "
                "bound over k and maximum upper bound over k"
            ),
            "primal_big_m_scope": (
                "regime_wide; primal slack Big-M values remain support-set valid "
                "across empirical scenarios"
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _details_value(details: Any, key: str, default: Any = None) -> Any:
        if isinstance(details, dict):
            return details.get(key, default)
        return default

    def aggregate_lower_upper_bounds_over_k(
        self,
        bounds: dict[Any, Any],
    ) -> dict[str, dict[str, float]]:
        aggregated: dict[tuple[int, ...], dict[str, float]] = {}
        for raw_key, details in (bounds or {}).items():
            index = self._parse_json_index(raw_key) if isinstance(raw_key, str) else tuple(raw_key)
            if len(index) < 2:
                raise ValueError(f"Scenario-indexed bound key '{raw_key}' must include k")
            regime_index = tuple(int(part) for part in index[1:])
            lower = float(self._details_value(details, "lower"))
            upper = float(self._details_value(details, "upper"))
            if regime_index not in aggregated:
                aggregated[regime_index] = {"lower": lower, "upper": upper}
            else:
                aggregated[regime_index]["lower"] = min(aggregated[regime_index]["lower"], lower)
                aggregated[regime_index]["upper"] = max(aggregated[regime_index]["upper"], upper)
        return self._jsonify_indexed_dict(aggregated)

    def aggregate_big_m_over_k(self, bounds: dict[Any, Any]) -> dict[str, Any]:
        aggregated: dict[tuple[int, ...], dict[str, Any]] = {}
        for raw_key, details in (bounds or {}).items():
            index = self._parse_json_index(raw_key) if isinstance(raw_key, str) else tuple(raw_key)
            if len(index) < 2:
                raise ValueError(f"Scenario-indexed Big-M key '{raw_key}' must include k")
            regime_index = tuple(int(part) for part in index[1:])
            value = None
            for value_key in ("big_m", "tight_big_m", "upper_bound", "ub", "bound", "value", "best_bound"):
                value = self._details_value(details, value_key)
                if value is not None:
                    break
            if value is None:
                value = details
            numeric_value = float(value)
            if regime_index not in aggregated or numeric_value > float(
                aggregated[regime_index].get(
                    "big_m", aggregated[regime_index].get("tight_big_m", 0.0)
                )
            ):
                aggregated[regime_index] = dict(details) if isinstance(details, dict) else {}
                if (
                    "tight_big_m" in aggregated[regime_index]
                    and "big_m" not in aggregated[regime_index]
                ):
                    aggregated[regime_index]["tight_big_m"] = numeric_value
                else:
                    aggregated[regime_index]["big_m"] = numeric_value
        return self._jsonify_indexed_dict(aggregated)

    def aggregate_fixed_binaries_over_k(
        self,
        fixed_binaries: dict[str, dict[Any, Any]],
    ) -> dict[str, dict[str, Any]]:
        scenario_count = int(self.poa.num_empirical_scenarios)
        aggregated: dict[str, dict[str, Any]] = {}
        for var_name, entries in (fixed_binaries or {}).items():
            candidates: dict[tuple[int, ...], list[dict[str, Any]]] = {}
            for raw_key, details in (entries or {}).items():
                index = (
                    self._parse_json_index(raw_key) if isinstance(raw_key, str) else tuple(raw_key)
                )
                if len(index) < 2:
                    raise ValueError(
                        f"Scenario-indexed fixed binary key '{raw_key}' must include k"
                    )
                regime_index = tuple(int(part) for part in index[1:])
                candidates.setdefault(regime_index, []).append(dict(details or {}))
            for regime_index, scenario_details in candidates.items():
                fixed_values = {
                    int(details.get("fixed_value"))
                    for details in scenario_details
                    if details.get("fixed_value") is not None
                }
                if len(scenario_details) != scenario_count or len(fixed_values) != 1:
                    continue
                merged = dict(scenario_details[0])
                merged["fixed_value"] = fixed_values.pop()
                merged["regime_wide_agreement_scenarios"] = scenario_count
                aggregated.setdefault(var_name, {})[self._json_key(regime_index)] = merged
        return aggregated

    def aggregate_relu_bounds_over_k(
        self,
        relu_bounds: dict[str, dict[Any, Any]],
        tolerance: float = 1e-9,
    ) -> dict[str, dict[str, Any]]:
        aggregated_by_generator: dict[str, dict[str, Any]] = {}
        for generator_name, entries in (relu_bounds or {}).items():
            accum: dict[tuple[int, int, int], dict[str, float]] = {}
            for raw_key, details in (entries or {}).items():
                index = (
                    self._parse_json_index(raw_key) if isinstance(raw_key, str) else tuple(raw_key)
                )
                if len(index) != 4:
                    raise ValueError(
                        f"Scenario-indexed ReLU key '{raw_key}' must have k,t,linear_idx,node"
                    )
                regime_index = (int(index[1]), int(index[2]), int(index[3]))
                numeric = {
                    "L": float(details["L"]),
                    "U": float(details["U"]),
                    "h_lower": float(details["h_lower"]),
                    "h_upper": float(details["h_upper"]),
                }
                if regime_index not in accum:
                    accum[regime_index] = numeric
                else:
                    accum[regime_index]["L"] = min(accum[regime_index]["L"], numeric["L"])
                    accum[regime_index]["U"] = max(accum[regime_index]["U"], numeric["U"])
                    accum[regime_index]["h_lower"] = min(
                        accum[regime_index]["h_lower"],
                        numeric["h_lower"],
                    )
                    accum[regime_index]["h_upper"] = max(
                        accum[regime_index]["h_upper"],
                        numeric["h_upper"],
                    )
            generator_payload: dict[str, Any] = {}
            for regime_index, details in accum.items():
                status = "ambiguous"
                if details["U"] <= float(tolerance):
                    status = "inactive"
                elif details["L"] >= -float(tolerance):
                    status = "active"
                generator_payload[self._json_key(regime_index)] = {
                    **details,
                    "status": status,
                    "time_idx": regime_index[0],
                    "linear_idx": regime_index[1],
                    "node": regime_index[2],
                }
            aggregated_by_generator[str(generator_name)] = generator_payload
        return aggregated_by_generator

    def save_final_report(self, output_path: str | Path) -> Path:
        payload = {
            "metadata": self._metadata(),
            "primal_big_m": self.tightening_data.get("primal_big_m", {}),
            "nn_relu_bounds_report": self.tightening_data.get("nn_relu_bounds_report", {}),
            "scenario_nn_relu_bounds": self.tightening_data.get("scenario_nn_relu_bounds", {}),
            "regime_nn_relu_bounds": self.tightening_data.get("regime_nn_relu_bounds", {}),
            "nn_relu_bounds": self.tightening_data.get("nn_relu_bounds", {}),
            "scenario_alpha_bounds": self.tightening_data.get("scenario_alpha_bounds", {}),
            "regime_alpha_bounds": self.tightening_data.get("regime_alpha_bounds", {}),
            "alpha_bounds": self.tightening_data.get("alpha_bounds", {}),
            "alpha_optimization_results": self.tightening_data.get(
                "alpha_optimization_results",
                {},
            ),
            "slack_bounds": self.tightening_data.get("slack_bounds", {}),
            "scenario_fixed_binaries": self.tightening_data.get("scenario_fixed_binaries", {}),
            "regime_fixed_binaries": self.tightening_data.get("regime_fixed_binaries", {}),
            "fixed_binaries": self.tightening_data.get("fixed_binaries", {}),
            "scenario_tight_big_m": self.tightening_data.get("scenario_tight_big_m", {}),
            "regime_tight_big_m": self.tightening_data.get("regime_tight_big_m", {}),
            "tight_big_m": self.tightening_data.get("tight_big_m", {}),
            "optimal_cost_bounds": self.tightening_data.get("optimal_cost_bounds", {}),
            "scenario_optimal_cost_bounds": self.tightening_data.get(
                "scenario_optimal_cost_bounds",
                {},
            ),
            "optimal_cost_bound_optimization_results": self.tightening_data.get(
                "optimal_cost_bound_optimization_results",
                {},
            ),
            "equilibrium_cost_bounds": self.tightening_data.get("equilibrium_cost_bounds", {}),
            "scenario_equilibrium_cost_bounds": self.tightening_data.get(
                "scenario_equilibrium_cost_bounds",
                {},
            ),
            "equilibrium_cost_bound_optimization_results": self.tightening_data.get(
                "equilibrium_cost_bound_optimization_results",
                {},
            ),
            "derived_poa_bounds": self.tightening_data.get("derived_poa_bounds"),
            "stage_reports": self.stage_reports,
            "stage_timings": self.stage_timings,
        }
        return self._save_json(payload, output_path)

    def run_essential_bounds(
        self,
        previous_paths: Optional[dict[str, str | Path]] = None,
        output_paths: Optional[dict[str, str | Path]] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> Path:
        """Always (re)compute primal_big_m and optimal_cost_bounds.

        Used when full DRO tightening is disabled but the McCormick C_opt envelope
        still needs valid denominator bounds. Existing reports for the remaining
        stages (relu/alpha/slack/dual) are reused when present; otherwise they are
        left empty, and the final DRO solve falls back to its internal loose
        defaults for them. Unlike ``run_all`` this bypasses the inter-stage
        ``_require_*`` guards, which only make sense for a full tightening pass.
        """
        from models.DRO_PoA.DRO_PoA_tightening.compute_optimal_cost_bounds import (
            DROOptimalCostBoundsComputer,
        )
        from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import DROPrimalBigMComputer

        previous_paths = self._resolve_output_paths(previous_paths)
        output_paths = self._resolve_output_paths(output_paths)

        # Preserve any previously tightened non-essential stages in the final report.
        for stage_name in ("relu_bounds", "alpha_bounds", "slack_binary_fix", "dual_big_m"):
            previous_path = Path(previous_paths[stage_name])
            if previous_path.exists():
                self._load_previous_stage(stage_name, previous_path)

        primal_stage = self._as_stage(DROPrimalBigMComputer)
        primal_stage.run_primal_big_m(output_path=output_paths["primal_big_m"])

        optimal_cost_stage = self._as_stage(DROOptimalCostBoundsComputer)
        optimal_cost_stage.run_optimal_cost_bounds(
            output_path=output_paths["optimal_cost_bounds"],
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )

        return self.save_final_report(output_paths["final"])

    def _compute_cost_ratio_bound(self) -> Optional[dict[str, Any]]:
        """Valid analytical PoA upper bound: max of the overlapping-displacement and
        consecutive-cost ratios.

        Term 1 (overlapping displacement): for each pair of blocks ((i,b), (j,k))
        with c_{j,k} > c_{i,b}, if the certified lower bound on alpha[j,k] is
        strictly below the certified upper bound on alpha[i,b], then (j,k) can
        displace (i,b) in some scenario despite being more expensive, contributing
        the ratio c_{j,k}/c_{i,b}.

        Term 2 (consecutive cost): the largest ratio between cost-adjacent blocks
        (distinct positive costs sorted ascending). This is always applied so the
        bound cannot collapse to 1 when no certified bid ranges overlap.

        kappa = max(max Term 1, max Term 2). Returns a dict with 'kappa', the
        witnessing pair, and which term is binding, or None if unavailable.
        """
        alpha_bounds = getattr(self.poa, "alpha_bounds", None) or {}
        if not alpha_bounds:
            return None
        block_cost: dict[tuple[int, int], float] = {
            (int(i), int(b)): float(
                self.poa.block_cost_vector[self.poa.local_to_global_block[(int(i), int(b))]]
            )
            for i, b in self.poa.generator_block_pairs
        }
        # Aggregate per (i,b): max upper and min lower over all time steps.
        # Keys are (i,b,t) for PoA; offset handles DRO (k,i,b,t) variant too.
        alpha_ub: dict[tuple[int, int], float] = {}
        alpha_lb: dict[tuple[int, int], float] = {}
        for key, bounds in alpha_bounds.items():
            offset = 1 if len(key) == 4 else 0
            ib = (int(key[offset]), int(key[offset + 1]))
            alpha_ub[ib] = max(alpha_ub.get(ib, -1e18), float(bounds["upper"]))
            alpha_lb[ib] = min(alpha_lb.get(ib, 1e18), float(bounds["lower"]))
        # --- Term 1: largest feasible overlapping-displacement ratio ---
        overlap_kappa: Optional[float] = None
        overlap_best: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
        for ib, c_ib in block_cost.items():
            if c_ib <= 0 or ib not in alpha_ub:
                continue
            for jk, c_jk in block_cost.items():
                if c_jk <= c_ib or jk not in alpha_lb:
                    continue
                if alpha_lb[jk] >= alpha_ub[ib]:
                    continue  # j,k always bids above i,b: displacement impossible
                ratio = c_jk / c_ib
                if overlap_kappa is None or ratio > overlap_kappa:
                    overlap_kappa = ratio
                    overlap_best = (ib, jk)

        # --- Term 2: largest ratio between cost-adjacent blocks (always applied) ---
        # Sort the distinct positive block costs ascending and take the largest
        # ratio between consecutive (cost-adjacent) blocks. A more expensive block
        # can be dispatched ahead of the next-cheaper one even when no certified
        # bid ranges overlap, so this keeps kappa from collapsing to 1. Flooring
        # kappa only loosens the bound (the final box is min(C_eq/C_opt, kappa)),
        # so the result stays a valid PoA upper bound.
        adjacent_kappa: Optional[float] = None
        adjacent_best: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
        block_at_cost: dict[float, tuple[int, int]] = {}
        for ib, c_ib in block_cost.items():
            if c_ib > 0.0:
                block_at_cost.setdefault(c_ib, ib)
        positive_costs = sorted(block_at_cost)
        for c_low, c_high in zip(positive_costs, positive_costs[1:]):
            ratio = c_high / c_low
            if adjacent_kappa is None or ratio > adjacent_kappa:
                adjacent_kappa = ratio
                adjacent_best = (block_at_cost[c_low], block_at_cost[c_high])

        # --- kappa = max(Term 1, Term 2) ---
        finite = [
            (k, pair)
            for k, pair in ((overlap_kappa, overlap_best), (adjacent_kappa, adjacent_best))
            if k is not None and pair is not None
        ]
        if not finite:
            return None
        kappa, best = max(finite, key=lambda item: item[0])
        binding_term = "overlap" if best is overlap_best else "consecutive_cost"
        ib, jk = best
        return {
            "kappa": float(kappa),
            "cheap_block": list(ib),
            "expensive_block": list(jk),
            "cheap_cost": float(block_cost[ib]),
            "expensive_cost": float(block_cost[jk]),
            "cheap_alpha_upper": float(alpha_ub[ib]) if ib in alpha_ub else None,
            "expensive_alpha_lower": float(alpha_lb[jk]) if jk in alpha_lb else None,
            "binding_term": binding_term,
            "overlap_kappa": float(overlap_kappa) if overlap_kappa is not None else None,
            "consecutive_cost_kappa": (
                float(adjacent_kappa) if adjacent_kappa is not None else None
            ),
        }

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
        parallel_workers: int = 1,
        solver_threads: Optional[int] = None,
    ) -> Path:
        from models.DRO_PoA.DRO_PoA_tightening.compute_alpha_bounds import DROAlphaBoundsComputer
        from models.DRO_PoA.DRO_PoA_tightening.compute_dual_big_m import DRODualBigMComputer
        from models.DRO_PoA.DRO_PoA_tightening.compute_equilibrium_cost_bounds import (
            DROEquilibriumCostBoundsComputer,
        )
        from models.DRO_PoA.DRO_PoA_tightening.compute_optimal_cost_bounds import (
            DROOptimalCostBoundsComputer,
        )
        from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import DROPrimalBigMComputer
        from models.DRO_PoA.DRO_PoA_tightening.compute_relu_bounds import DROReLUBoundsComputer
        from models.DRO_PoA.DRO_PoA_tightening.compute_slack_binary_fix import (
            DROSlackBinaryFixComputer,
        )

        previous_paths = self._resolve_output_paths(previous_paths)
        output_paths = self._resolve_output_paths(output_paths)

        primal_stage = self._as_stage(DROPrimalBigMComputer)
        relu_stage = self._as_stage(DROReLUBoundsComputer)
        alpha_stage = self._as_stage(DROAlphaBoundsComputer)
        slack_stage = self._as_stage(DROSlackBinaryFixComputer)
        dual_stage = self._as_stage(DRODualBigMComputer)
        optimal_cost_stage = self._as_stage(DROOptimalCostBoundsComputer)
        equilibrium_cost_stage = self._as_stage(DROEquilibriumCostBoundsComputer)

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
                parallel_workers=parallel_workers,
                solver_threads=solver_threads,
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
        )

        # Optional equilibrium cost bound (gated by run_equilibrium_cost_bounds):
        # compute when requested, else reuse a previous report when present, else
        # skip so the final report omits the section and the McCormick PoA box
        # falls back to the hand-set dro_mccormick_PoA_bounds.
        eq_previous = previous_paths.get("equilibrium_cost_bounds")
        if run_equilibrium_cost_bounds:
            self._require_alpha_bounds()
            self._load_or_run_stage(
                "equilibrium_cost_bounds",
                True,
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
            )
        elif eq_previous and Path(eq_previous).exists():
            self._load_previous_stage("equilibrium_cost_bounds", eq_previous)

        # Analytical cost-ratio PoA bound: always applied last so it can tighten
        # whatever derived_poa_bounds was set by equilibrium_cost_bounds (or replace
        # it entirely when that stage is absent). No solver required.
        if getattr(self.poa, "alpha_bounds", None):
            cost_ratio = self._compute_cost_ratio_bound()
            if cost_ratio is not None:
                kappa = float(cost_ratio["kappa"])
                existing = self.tightening_data.get("derived_poa_bounds")
                if existing is not None:
                    derived: list[float] = [float(existing[0]), min(float(existing[1]), kappa)]
                else:
                    derived = [poa_bounds_lower, kappa]
                self.tightening_data["derived_poa_bounds"] = derived
                print(
                    f"[cost_ratio_bound] kappa={kappa:.6g} "
                    f"(block {cost_ratio['cheap_block']} -> {cost_ratio['expensive_block']}) "
                    f"-> derived PoA box ({derived[0]:.6g}, {derived[1]:.6g})",
                    flush=True,
                )

        return self.save_final_report(output_paths["final"])
