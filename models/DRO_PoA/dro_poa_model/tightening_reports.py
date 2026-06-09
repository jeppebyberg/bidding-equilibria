from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class DROPoATighteningReports:
    @staticmethod
    def _empty_default_bounds_used() -> dict[str, Any]:
        return {
            "nn_relu_bounds_report": False,
            "alpha_bounds": False,
            "tight_big_m": False,
            "optimal_cost_bounds": False,
            "primal_big_m": False,
            "notes": [],
        }

    def _mark_default_bound_used(self, component_name: str, note: str) -> None:
        if not getattr(self, "default_bounds_used", None):
            self.default_bounds_used = self._empty_default_bounds_used()
        self.default_bounds_used[component_name] = True
        notes = self.default_bounds_used.setdefault("notes", [])
        if note not in notes:
            notes.append(note)

    def ensure_default_bounds_available(
        self,
        include_optimal_cost_bounds: Optional[bool] = None,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> None:
        """Self-provide model-construction bounds when no tightening report is loaded.

        Mirrors the PoA model: primal_big_m is always recomputed (analytic) and
        optimal_cost_bounds is always recomputed via the real optimal-dispatch KKT
        solve, so the McCormick C_opt envelope has valid, tight denominator bounds.
        The remaining stages (ReLU/alpha/dual Big-M/lambda) fall back to the
        internal loose defaults applied by _prepare_loaded_bounds and the apply step.
        """
        from models.DRO_PoA.DRO_PoA_tightening.compute_primal_big_m import (
            compute_primal_big_m_bounds,
        )

        include_optimal_cost_bounds = (
            self.objective_mode in {"mccormick", "piecewise_mccormick"}
            if include_optimal_cost_bounds is None
            else bool(include_optimal_cost_bounds)
        )

        # primal_big_m always runs (analytic) and is applied immediately so the
        # optimal-cost-bound solve below can build its KKT complementarity model.
        self.primal_big_m = compute_primal_big_m_bounds(self)
        self._loaded_bounds_prepared = False
        self._mark_default_bound_used(
            "primal_big_m",
            "Analytic primal Big-M values were computed for model construction.",
        )

        # optimal_cost_bounds always runs via the real optimal-dispatch KKT solve;
        # _set_optimal_cost_bounds_from_report pushes the range into mccormick_bounds.
        if include_optimal_cost_bounds:
            optimal_cost_report = self._solve_optimal_cost_bounds_report(
                solver_name=solver_name,
                time_limit=time_limit,
                tee=tee,
                solver_threads=solver_threads,
            )
            self._set_optimal_cost_bounds_from_report(
                {"optimal_cost_bounds": optimal_cost_report}
            )
            self._mark_default_bound_used(
                "optimal_cost_bounds",
                "Optimal-cost bounds were computed via the optimal-dispatch KKT solve.",
            )

        self._prepare_loaded_bounds()

    def _solve_optimal_cost_bounds_report(
        self,
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        """Solve the DRO optimal-dispatch KKT problem for true C_opt bounds.

        Requires self.primal_big_m to be populated. The computer builds its model
        on self.tightening_model, leaving self.model untouched.
        """
        from models.DRO_PoA.DRO_PoA_tightening.compute_optimal_cost_bounds import (
            DROOptimalCostBoundsComputer,
        )

        stage = DROOptimalCostBoundsComputer.__new__(DROOptimalCostBoundsComputer)
        stage.poa = self
        stage.tightening_data = {}
        stage.stage_reports = {}
        bound_report = stage.compute_optimal_cost_bounds(
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        c_opt = bound_report["C_opt"]
        return {
            "lower": float(c_opt["lower"]),
            "upper": float(c_opt["upper"]),
            "raw_lower": c_opt.get("raw_lower"),
            "raw_upper": c_opt.get("raw_upper"),
            "source": "computed_optimal_dispatch_kkt_bounds",
        }

    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in str(key).split(",") if part != "")
        except ValueError as exc:
            raise ValueError(f"Malformed tightening-report key '{key}'") from exc

    @staticmethod
    def _optional_numeric_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("tight_big_m", "big_m", "upper_bound", "ub", "bound", "value"):
                if value_key in payload:
                    return DROPoATighteningReports._optional_numeric_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return max(0.0, numeric_value)

    @staticmethod
    def _optional_float_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("bound", "value", "tight_bound", "tight_big_m"):
                if value_key in payload:
                    return DROPoATighteningReports._optional_float_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    def _validate_regime_wide_tightening_metadata(self, report: dict[str, Any]) -> None:
        metadata = report.get("metadata", {}) or {}
        if metadata.get("tightening_type") != "regime_wide":
            raise ValueError("DRO tightening report metadata must set tightening_type='regime_wide'")
        if metadata.get("tightening_scope") not in {None, "regime_wide", "scenario_wise"}:
            raise ValueError(
                "DRO tightening report metadata tightening_scope must be "
                "'scenario_wise' or 'regime_wide'"
            )
        if metadata.get("model_type") not in {None, "DRO_PoA"}:
            raise ValueError("DRO tightening report metadata model_type must be 'DRO_PoA'")
        if str(metadata.get("regime_name")) != str(self.regime_name):
            raise ValueError(
                "DRO tightening report regime_name mismatch: "
                f"report has {metadata.get('regime_name')!r}, optimizer has {self.regime_name!r}"
            )
        if int(metadata.get("num_time_steps", self.num_time_steps)) != int(self.num_time_steps):
            raise ValueError(
                "DRO tightening report num_time_steps mismatch: "
                f"report has {metadata.get('num_time_steps')}, optimizer has {self.num_time_steps}"
            )

    def _set_regime_wide_tightening_report_data(
        self,
        report: dict[str, Any],
        report_path: Optional[Path] = None,
    ) -> None:
        self._validate_regime_wide_tightening_metadata(report)
        report = dict(report)
        for legacy_field in (
            "scenario_lambda_bounds",
            "regime_lambda_bounds",
            "lambda_bounds",
            "aggregate_dual_bounds",
        ):
            report.pop(legacy_field, None)
        legacy_dual_sum_keys = {
            "mu_max_sum_ub",
            "mu_min_sum_ub",
            "mu_ramp_up_sum_ub",
            "mu_ramp_down_sum_ub",
        }
        for field_name in ("scenario_tight_big_m", "regime_tight_big_m", "tight_big_m"):
            payload = report.get(field_name)
            if isinstance(payload, dict):
                report[field_name] = {
                    key: value
                    for key, value in payload.items()
                    if key not in legacy_dual_sum_keys
                }
        self.tightening_report = report
        if report_path is not None:
            self.tightening_report_path = Path(report_path)
        relu_report = report.get("nn_relu_bounds_report", {}) or {}
        if not relu_report and (
            "nn_relu_bounds" in report or "scenario_nn_relu_bounds" in report
        ):
            relu_report = report
        if relu_report:
            self._set_nn_relu_bounds_from_report(relu_report)
        self.fixed_binaries = report.get("scenario_fixed_binaries", {}) or report.get(
            "fixed_binaries",
            {},
        ) or {}
        self.primal_big_m = report.get("primal_big_m", {}) or {}
        self.tight_big_m = report.get("scenario_tight_big_m", {}) or report.get(
            "tight_big_m",
            {},
        ) or {}
        self.alpha_bound_optimization_results = (
            report.get("alpha_optimization_results", {}) or {}
        )
        self.alpha_bounds = {
            self._parse_json_index(key): {
                "lower": float(value["lower"]),
                "upper": float(value["upper"]),
            }
            for key, value in (
                report.get("scenario_alpha_bounds", {}) or report.get("alpha_bounds", {}) or {}
            ).items()
        }
        if (
            "optimal_cost_bounds" in report
            or "scenario_optimal_cost_bounds" in report
            or "optimal_cost_bound_optimization_results" in report
        ):
            self._set_optimal_cost_bounds_from_report(report)
        self._loaded_bounds_prepared = False

    def load_regime_wide_tightening_report(self, report_path: str | Path) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"DRO regime-wide tightening report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        self._set_regime_wide_tightening_report_data(report, path)
        self._prepare_loaded_bounds()
        return self.tightening_report

    def _set_optimal_cost_bounds_from_report(self, report: dict[str, Any]) -> None:
        raw_bounds = report.get("optimal_cost_bounds", {}) or {}
        if "C_opt" in raw_bounds and isinstance(raw_bounds.get("C_opt"), dict):
            C_opt_payload = raw_bounds.get("C_opt", {}) or {}
        else:
            C_opt_payload = raw_bounds

        if C_opt_payload:
            lower = C_opt_payload.get("lower")
            upper = C_opt_payload.get("upper")
            if lower is None or upper is None:
                raise ValueError(
                    "optimal_cost_bounds must contain finite 'lower' and 'upper' entries"
                )
            self.optimal_cost_bounds = {
                "C_opt": {
                    "lower": float(lower),
                    "upper": float(upper),
                    "raw_lower": (
                        float(C_opt_payload["raw_lower"])
                        if C_opt_payload.get("raw_lower") is not None
                        else None
                    ),
                    "raw_upper": (
                        float(C_opt_payload["raw_upper"])
                        if C_opt_payload.get("raw_upper") is not None
                        else None
                    ),
                }
            }

        self.scenario_optimal_cost_bounds = (
            report.get("scenario_optimal_cost_bounds", {}) or {}
        )
        self.optimal_cost_bound_optimization_results = (
            report.get("optimal_cost_bound_optimization_results", {}) or {}
        )

        if self.optimal_cost_bounds and getattr(self, "mccormick_bounds", None) is not None:
            C_opt_bounds = self.optimal_cost_bounds.get("C_opt", {})
            lower = C_opt_bounds.get("lower")
            upper = C_opt_bounds.get("upper")
            if lower is not None and upper is not None:
                c_opt_bounds = (float(lower), float(upper))
                self.mccormick_bounds["C_opt"] = c_opt_bounds
                if getattr(self, "objective_mode", None) == "piecewise_mccormick":
                    num_pieces = int(self.mccormick_bounds.get("num_pieces", 0))
                    if num_pieces >= 2:
                        lower_f, upper_f = c_opt_bounds
                        step = (upper_f - lower_f) / num_pieces
                        self.mccormick_bounds["C_opt_breakpoints"] = [
                            lower_f + step * idx for idx in range(num_pieces + 1)
                        ]

    def load_optimal_cost_bounds_report(self, report_path: str | Path) -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"DRO optimal-cost bounds report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        self._set_optimal_cost_bounds_from_report(report)
        return report

    def _indexed_numeric_entries(
        self,
        entries: Any,
        bound_type: str,
        allowed_dimensions: tuple[int, ...],
        expected_key_format: str,
    ) -> dict[tuple[int, ...], float]:
        if not isinstance(entries, dict):
            return {}
        parsed: dict[tuple[int, ...], float] = {}
        for raw_key, details in entries.items():
            index = self._parse_json_index(str(raw_key))
            if len(index) not in allowed_dimensions:
                scenario_note = (
                    "including scenario index k"
                    if "k," in expected_key_format
                    else "without scenario index k"
                )
                raise ValueError(
                    f"Invalid {bound_type} key '{raw_key}'. Expected key format "
                    f"{expected_key_format} {scenario_note}."
                )
            numeric_value = self._optional_numeric_bound(details)
            if numeric_value is None:
                raise ValueError(
                    f"Invalid {bound_type} entry at key '{raw_key}'. Expected a "
                    "finite numeric bound in tight_big_m, big_m, upper_bound, ub, "
                    "bound, or value."
                )
            parsed[index] = float(numeric_value)
        return parsed

    def _missing_loaded_bound_error(
        self,
        bound_type: str,
        index: tuple[int, ...] | int,
        expected_key_format: str,
    ) -> ValueError:
        scenario_note = (
            "including scenario index k"
            if "k," in expected_key_format
            else "without scenario index k"
        )
        return ValueError(
            f"Missing {bound_type} for index {index}. Expected key format "
            f"{expected_key_format} {scenario_note}. If this is a reused DRO "
            "tightening report, rerun the corresponding stage for the current "
            "num_empirical_scenarios."
        )

    @staticmethod
    def _scenario_or_regime_value(
        bound_map: dict[tuple[int, ...], float],
        scenario_idx: int,
        regime_index: tuple[int, ...],
    ) -> float:
        scenario_key = (int(scenario_idx), *tuple(int(part) for part in regime_index))
        regime_key = tuple(int(part) for part in regime_index)
        if scenario_key in bound_map:
            return float(bound_map[scenario_key])
        return float(bound_map[regime_key])

    @staticmethod
    def _scenario_or_regime_lambda_bounds(
        bound_map: dict[Any, tuple[float, float]],
        scenario_idx: int,
        time_idx: int,
    ) -> tuple[float, float]:
        scenario_key = (int(scenario_idx), int(time_idx))
        if scenario_key in bound_map:
            return bound_map[scenario_key]
        return bound_map[int(time_idx)]

    def _prepare_block_time_primal_big_m(
        self,
        component_name: str,
        entries: Any,
        expected_key_format: str,
    ) -> dict[tuple[int, int, int], float]:
        parsed = self._indexed_numeric_entries(
            entries,
            f"primal Big-M '{component_name}'",
            (2, 3),
            expected_key_format,
        )
        prepared: dict[tuple[int, int, int], float] = {}
        for i, b in self.generator_block_pairs:
            for t in range(self.num_time_steps):
                time_index = (int(i), int(b), int(t))
                block_index = (int(i), int(b))
                if time_index in parsed:
                    prepared[time_index] = parsed[time_index]
                elif block_index in parsed:
                    prepared[time_index] = parsed[block_index]
                else:
                    raise self._missing_loaded_bound_error(
                        f"primal Big-M '{component_name}'",
                        time_index,
                        expected_key_format,
                    )
        return prepared

    def _prepare_generator_time_primal_big_m(
        self,
        component_name: str,
        entries: Any,
        expected_key_format: str,
    ) -> dict[tuple[int, int], float]:
        parsed = self._indexed_numeric_entries(
            entries,
            f"primal Big-M '{component_name}'",
            (1, 2),
            expected_key_format,
        )
        prepared: dict[tuple[int, int], float] = {}
        for i in range(self.num_physical_generators):
            for t in range(self.num_time_steps):
                time_index = (int(i), int(t))
                generator_index = (int(i),)
                if time_index in parsed:
                    prepared[time_index] = parsed[time_index]
                elif generator_index in parsed:
                    prepared[time_index] = parsed[generator_index]
                else:
                    raise self._missing_loaded_bound_error(
                        f"primal Big-M '{component_name}'",
                        time_index,
                        expected_key_format,
                    )
        return prepared

    def _prepare_generator_primal_big_m(
        self,
        component_name: str,
        entries: Any,
        expected_key_format: str,
    ) -> dict[int, float]:
        parsed = self._indexed_numeric_entries(
            entries,
            f"primal Big-M '{component_name}'",
            (1, 2),
            expected_key_format,
        )
        prepared: dict[int, float] = {}
        for i in range(self.num_physical_generators):
            generator_index = (int(i),)
            initial_time_index = (int(i), 0)
            if generator_index in parsed:
                prepared[int(i)] = parsed[generator_index]
            elif initial_time_index in parsed:
                prepared[int(i)] = parsed[initial_time_index]
            else:
                raise self._missing_loaded_bound_error(
                    f"primal Big-M '{component_name}'",
                    int(i),
                    expected_key_format,
                )
        return prepared

    # Minimum fraction of default_bound that an OBBT-derived tight_big_m must
    # reach before it is applied.  Results below this floor (when not confirmed
    # by a primal slack argument) are treated as structurally unreliable: the
    # single-scenario OBBT does not include the adversarial support_objective_floor
    # constraint, so it can claim M=0 for lower-bound duals when the DRO model
    # actually requires M > 0 in adversarially-forced high-PoA scenarios.
    _DUAL_BIG_M_OBBT_MIN_FRACTION: float = 1e-3

    def _prepare_dual_big_m(
        self,
        dual_name: str,
        expected_indices: list[tuple[int, ...]],
        default_bound: float,
        expected_key_format: str,
    ) -> dict[tuple[int, ...], float]:
        entries = (getattr(self, "tight_big_m", {}) or {}).get(dual_name, {}) or {}
        if not entries:
            return {index: float(default_bound) for index in expected_indices}
        parsed = self._indexed_numeric_entries(
            entries,
            f"dual Big-M '{dual_name}'",
            (len(expected_indices[0]), len(expected_indices[0]) + 1),
            f"{expected_key_format} or k,{expected_key_format}",
        )
        # Build a companion map that records whether each entry was confirmed by
        # a primal slack argument (fixed_by_slack=True) or came from the OBBT.
        # The OBBT uses a single-scenario model without adversarial constraints,
        # which can underestimate the bound for lower-bound duals.
        fixed_by_slack_map: dict[tuple[int, ...], bool] = {}
        if isinstance(entries, dict):
            for raw_key, details in entries.items():
                if isinstance(details, dict):
                    idx = self._parse_json_index(str(raw_key))
                    fixed_by_slack_map[idx] = bool(details.get("fixed_by_slack", False))

        min_obbt_bound = float(default_bound) * self._DUAL_BIG_M_OBBT_MIN_FRACTION

        def _safe_bound(index: tuple[int, ...], tight_val: float) -> float:
            fbs = fixed_by_slack_map.get(index, False)
            if not fbs and tight_val < min_obbt_bound:
                # OBBT result is suspiciously small; revert to the default bound
                # to avoid artificial infeasibility in the DRO adversarial model.
                return float(default_bound)
            return max(0.0, min(float(default_bound), tight_val))

        prepared: dict[tuple[int, ...], float] = {}
        has_scenario_keys = any(len(index) == len(expected_indices[0]) + 1 for index in parsed)
        for regime_index in expected_indices:
            if has_scenario_keys:
                for k in range(self.num_empirical_scenarios):
                    scenario_index = (int(k), *tuple(regime_index))
                    if scenario_index not in parsed:
                        raise self._missing_loaded_bound_error(
                            f"scenario-wise dual Big-M '{dual_name}'",
                            scenario_index,
                            f"k,{expected_key_format}",
                        )
                    prepared[scenario_index] = _safe_bound(
                        scenario_index, float(parsed[scenario_index])
                    )
                continue
            if regime_index not in parsed:
                raise self._missing_loaded_bound_error(
                    f"dual Big-M '{dual_name}'",
                    regime_index,
                    expected_key_format,
                )
            prepared[regime_index] = _safe_bound(
                regime_index, float(parsed[regime_index])
            )
        return prepared

    def _bid_implied_lambda_bounds(self, lambda_name: str) -> tuple[float, float]:
        """Clearing-price (lambda) bounds derived from the effective bid range.

        The marginal clearing price equals the marginal block's bid, shifted by
        ramp-dual contributions. We bound lambda by the effective bid range and
        widen it outward by LAMBDA_RAMP_SAFETY_FACTOR to absorb the ramp duals:

            lower = min_bid - sf * |min_bid|
            upper = max_bid + sf * |max_bid|

        For lambda_opt the bids are the true marginal costs.
        For lambda_eq the bids are the alpha bounds: min/max over all loaded
        alpha_bounds entries (tightened OBBT when available, else loose defaults).
        """
        sf = float(getattr(self, "LAMBDA_RAMP_SAFETY_FACTOR", 0.10))
        if lambda_name == "lambda_opt":
            costs = [float(c) for c in self.block_cost_vector]
            min_bid, max_bid = min(costs), max(costs)
        else:
            loaded_alpha = getattr(self, "alpha_bounds", None) or {}
            if loaded_alpha:
                min_bid = min(float(v["lower"]) for v in loaded_alpha.values())
                max_bid = max(float(v["upper"]) for v in loaded_alpha.values())
            else:
                min_bid = float(self.default_alpha_lower)
                max_bid = float(self.default_alpha_upper)
        lower = min_bid - sf * abs(min_bid)
        upper = max_bid + sf * abs(max_bid)
        if lower >= upper:
            lower, upper = min_bid - 1.0, max_bid + 1.0
        return (lower, upper)

    def _prepare_lambda_bounds(self, lambda_name: str) -> dict[Any, tuple[float, float]]:
        # Clearing-price bounds track the effective bid range (alpha + true costs)
        # widened for ramp duals, rather than fixed loose constants.
        dyn_lower, dyn_upper = self._bid_implied_lambda_bounds(lambda_name)
        return {
            int(t): (dyn_lower, dyn_upper)
            for t in range(self.num_time_steps)
        }

    def _prepare_loaded_bounds(self) -> None:
        if not getattr(self, "tightening_report", None) and not getattr(
            self,
            "primal_big_m",
            None,
        ):
            return
        if not getattr(self, "primal_big_m", None):
            raise ValueError(
                "Missing primal_big_m in DRO regime-wide tightening report."
            )

        primal_big_m = self.primal_big_m
        self.M_cap = self._prepare_block_time_primal_big_m(
            "block_capacity",
            primal_big_m.get("block_capacity", {}),
            "i,b or i,b,t",
        )
        lower_entries = (
            primal_big_m.get("lower_generation", {})
            or primal_big_m.get("generation_lower", {})
            or primal_big_m.get("lower_bound", {})
        )
        self.M_lower = (
            self._prepare_block_time_primal_big_m(
                "lower_generation",
                lower_entries,
                "i,b or i,b,t",
            )
            if lower_entries
            else dict(self.M_cap)
        )
        self.M_physical_capacity = self._prepare_generator_primal_big_m(
            "physical_capacity",
            primal_big_m.get("physical_capacity", {}),
            "i or i,t",
        )
        self.M_ramp_up = self._prepare_generator_time_primal_big_m(
            "ramp_up",
            primal_big_m.get("ramp_up", {}),
            "i or i,t",
        )
        self.M_ramp_down = self._prepare_generator_time_primal_big_m(
            "ramp_down",
            primal_big_m.get("ramp_down", {}),
            "i or i,t",
        )
        self.M_ramp_up_initial = self._prepare_generator_primal_big_m(
            "ramp_up_initial",
            primal_big_m.get("ramp_up_initial", {}),
            "i",
        )
        self.M_ramp_down_initial = self._prepare_generator_primal_big_m(
            "ramp_down_initial",
            primal_big_m.get("ramp_down_initial", {}),
            "i",
        )

        capacity_indices = [
            (int(i), int(b), int(t))
            for i, b in self.generator_block_pairs
            for t in range(self.num_time_steps)
        ]
        ramp_indices = [
            (int(i), int(t))
            for i in range(self.num_physical_generators)
            for t in range(self.num_time_steps)
        ]
        self.M_mu_upper_eq = self._prepare_dual_big_m(
            "mu_upper_eq",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_lower_eq = self._prepare_dual_big_m(
            "mu_lower_eq",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_ramp_up_eq = self._prepare_dual_big_m(
            "mu_ramp_up_eq",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.M_mu_ramp_down_eq = self._prepare_dual_big_m(
            "mu_ramp_down_eq",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.M_mu_upper_opt = self._prepare_dual_big_m(
            "mu_upper_opt",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_lower_opt = self._prepare_dual_big_m(
            "mu_lower_opt",
            capacity_indices,
            self.capacity_dual_bound,
            "i,b,t",
        )
        self.M_mu_ramp_up_opt = self._prepare_dual_big_m(
            "mu_ramp_up_opt",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.M_mu_ramp_down_opt = self._prepare_dual_big_m(
            "mu_ramp_down_opt",
            ramp_indices,
            self.ramp_dual_bound,
            "i,t",
        )
        self.lambda_eq_bounds = self._prepare_lambda_bounds("lambda_eq")
        self.lambda_opt_bounds = self._prepare_lambda_bounds("lambda_opt")
        self._loaded_bounds_prepared = True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def apply_regime_wide_tightening_to_model(
        self,
        report: Optional[dict[str, Any]] = None,
        apply_alpha_bounds: bool = True,
        apply_fixed_binaries: bool = True,
        apply_dual_bounds: bool = True,
        apply_relu_bounds: bool = True,
    ) -> dict[str, Any]:
        """
        Apply DRO tightening to the Pyomo model.

        Scenario-indexed keys are applied to their exact scenario copy k.
        Regime-wide keys are still accepted and broadcast across scenarios.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        current_report = getattr(self, "tightening_report", None)
        report = report or current_report
        if not report:
            raise ValueError(
                "No DRO regime-wide tightening report loaded. "
                "Call load_regime_wide_tightening_report() first."
            )
        if report is not current_report:
            self._set_regime_wide_tightening_report_data(
                report,
                getattr(self, "tightening_report_path", None),
            )
        self._prepare_loaded_bounds()

        stats = {
            "alpha_bounds": 0,
            "fixed_binaries": 0,
            "dual_upper_bounds": 0,
            "relu_bounds": {},
        }
        if apply_alpha_bounds:
            stats["alpha_bounds"] = self._apply_alpha_bounds_to_model(report)
        if apply_fixed_binaries:
            stats["fixed_binaries"] = self._apply_fixed_binaries_to_model(report)
        if apply_dual_bounds:
            stats["dual_upper_bounds"] = self._apply_dual_bounds_to_model()
        if apply_relu_bounds:
            stats["relu_bounds"] = self._apply_relu_bounds_to_model(report)
        self.applied_tightening_stats = stats
        return stats

    def _apply_alpha_bounds_to_model(self, report: dict[str, Any]) -> int:
        m = self.model
        alpha_var = getattr(m, "alpha", None)
        if alpha_var is None:
            return 0
        applied = 0
        alpha_bounds = report.get("scenario_alpha_bounds", {}) or report.get(
            "alpha_bounds",
            {},
        )
        for key, bounds in (alpha_bounds or {}).items():
            index = self._parse_json_index(key)
            if len(index) not in {3, 4}:
                raise ValueError(f"Alpha-bound key '{key}' must have format i,b,t or k,i,b,t")
            lower = float(bounds["lower"])
            upper = float(bounds["upper"])
            scenario_indices = [index] if len(index) == 4 else [
                (int(k), *index) for k in m.scenarios
            ]
            for scenario_index in scenario_indices:
                if scenario_index not in alpha_var:
                    continue
                current_lb = alpha_var[scenario_index].lb
                current_ub = alpha_var[scenario_index].ub
                new_lower = max(float(current_lb), lower) if current_lb is not None else lower
                new_upper = min(float(current_ub), upper) if current_ub is not None else upper
                if new_lower <= new_upper:
                    alpha_var[scenario_index].setlb(new_lower)
                    alpha_var[scenario_index].setub(new_upper)
                    applied += 1
        return applied

    def _apply_fixed_binaries_to_model(self, report: dict[str, Any]) -> int:
        m = self.model
        applied = 0
        fixed_binaries = report.get("scenario_fixed_binaries", {}) or report.get(
            "fixed_binaries",
            {},
        )
        for var_name, entries in (fixed_binaries or {}).items():
            binary_var = getattr(m, var_name, None)
            if binary_var is None:
                continue
            for key, details in (entries or {}).items():
                index = self._parse_json_index(key)
                fixed_value = int((details or {}).get("fixed_value", 0))
                expected_without_k = binary_var.dim() - 1
                if len(index) == binary_var.dim():
                    if index in binary_var:
                        binary_var[index].fix(fixed_value)
                        applied += 1
                    continue
                if len(index) != expected_without_k:
                    raise ValueError(
                        f"Fixed-binary key '{key}' for {var_name} must have "
                        f"{binary_var.dim()} scenario-indexed indices or omit k and "
                        f"have {expected_without_k} indices."
                    )
                for k in m.scenarios:
                    scenario_index = (int(k), *index)
                    if scenario_index in binary_var:
                        binary_var[scenario_index].fix(fixed_value)
                        applied += 1
        return applied

    def _apply_dual_bounds_to_model(self) -> int:
        m = self.model
        applied = 0
        dual_bound_maps: tuple[tuple[str, dict[Any, float]], ...] = (
            ("mu_upper_eq", self.M_mu_upper_eq),
            ("mu_lower_eq", self.M_mu_lower_eq),
            ("mu_ramp_up_eq", self.M_mu_ramp_up_eq),
            ("mu_ramp_down_eq", self.M_mu_ramp_down_eq),
            ("mu_upper_opt", self.M_mu_upper_opt),
            ("mu_lower_opt", self.M_mu_lower_opt),
            ("mu_ramp_up_opt", self.M_mu_ramp_up_opt),
            ("mu_ramp_down_opt", self.M_mu_ramp_down_opt),
        )
        for var_name, bound_map in dual_bound_maps:
            dual_var = getattr(m, var_name, None)
            if dual_var is None:
                continue
            for raw_index, upper in bound_map.items():
                index = tuple(int(part) for part in raw_index)
                scenario_indices = (
                    [index]
                    if len(index) == dual_var.dim()
                    else [(int(k), *index) for k in m.scenarios]
                )
                for scenario_index in scenario_indices:
                    if scenario_index not in dual_var:
                        continue
                    current_ub = dual_var[scenario_index].ub
                    new_ub = max(0.0, float(upper))
                    if current_ub is not None:
                        new_ub = min(float(current_ub), new_ub)
                    dual_var[scenario_index].setub(new_ub)
                    applied += 1
        for var_name in (
            "mu_ramp_up_eq",
            "mu_ramp_down_eq",
            "mu_ramp_up_opt",
            "mu_ramp_down_opt",
        ):
            dual_var = getattr(m, var_name, None)
            if dual_var is None:
                continue
            for k in m.scenarios:
                for i in m.physical_generators:
                    index = (int(k), int(i), self.num_time_steps)
                    if index in dual_var:
                        dual_var[index].setub(0.0)
        return applied

    def _apply_relu_bounds_to_model(self, report: dict[str, Any]) -> dict[str, int]:
        relu_report = report.get("nn_relu_bounds_report", {}) or {}
        if not relu_report and (
            "nn_relu_bounds" in report or "scenario_nn_relu_bounds" in report
        ):
            relu_report = report
        stats = {
            "z_bounds_applied": 0,
            "h_bounds_applied": 0,
            "delta_fixed_active": 0,
            "delta_fixed_inactive": 0,
            "delta_left_ambiguous": 0,
        }
        if not relu_report:
            return stats
        self._set_nn_relu_bounds_from_report(relu_report)

        m = self.model
        if not hasattr(m, "nn_z") or not hasattr(m, "nn_h") or not hasattr(m, "nn_delta"):
            return stats

        for physical_generator_idx in self.nn_policy_generator_ids:
            i = int(physical_generator_idx)
            generator_name = self.physical_generator_names[i]
            for raw_key, bounds in (self.nn_relu_bounds.get(generator_name, {}) or {}).items():
                key = tuple(int(part) for part in raw_key)
                scenario_keys = (
                    [key]
                    if len(key) == 4
                    else [(int(k), *key) for k in m.scenarios]
                )
                for scenario_key in scenario_keys:
                    k, time_idx, linear_idx, node = scenario_key
                    index = (int(k), i, int(time_idx), int(linear_idx), int(node))
                    status = str(bounds.get("status", "ambiguous")).lower()
                    z_lb, z_ub, h_lb, h_ub = self._clamp_relu_bounds_to_status(
                        status,
                        float(bounds["L"]),
                        float(bounds["U"]),
                        float(bounds["h_lower"]),
                        float(bounds["h_upper"]),
                    )
                    if index in m.nn_z:
                        m.nn_z[index].setlb(z_lb)
                        m.nn_z[index].setub(z_ub)
                        stats["z_bounds_applied"] += 1
                    if index in m.nn_h:
                        m.nn_h[index].setlb(h_lb)
                        m.nn_h[index].setub(h_ub)
                        stats["h_bounds_applied"] += 1
                    if index not in m.nn_delta:
                        continue

                    status = str(bounds.get("status", "")).lower()
                    if status == "inactive":
                        m.nn_delta[index].fix(0)
                        stats["delta_fixed_inactive"] += 1
                    elif status == "active":
                        m.nn_delta[index].fix(1)
                        stats["delta_fixed_active"] += 1
                    elif status == "ambiguous":
                        if m.nn_delta[index].fixed:
                            m.nn_delta[index].unfix()
                        stats["delta_left_ambiguous"] += 1
                    else:
                        raise ValueError(
                            f"{generator_name}: unknown ReLU bound status '{status}'"
                        )

        self.applied_nn_relu_stats = stats
        return stats

    def compute_optimal_cost_bounds(
        self,
        output_path: str | Path = "results/dro_poa_tightening/optimal_cost_bounds_report.json",
        solver_name: str = "gurobi",
        time_limit: Optional[float] = None,
        tee: bool = False,
        solver_threads: Optional[int] = None,
    ) -> dict[str, Any]:
        from models.DRO_PoA.DRO_PoA_tightening.compute_optimal_cost_bounds import (
            DROOptimalCostBoundsComputer,
        )

        stage = DROOptimalCostBoundsComputer.__new__(DROOptimalCostBoundsComputer)
        stage.poa = self
        stage.dro = self
        stage.dro_poa = self
        stage.tightening_data = {
            "primal_big_m": getattr(self, "primal_big_m", {}) or {},
            "optimal_cost_bounds": getattr(self, "optimal_cost_bounds", {}) or {},
            "scenario_optimal_cost_bounds": getattr(
                self,
                "scenario_optimal_cost_bounds",
                {},
            ) or {},
            "optimal_cost_bound_optimization_results": getattr(
                self,
                "optimal_cost_bound_optimization_results",
                {},
            ) or {},
        }
        stage.stage_reports = {}
        report = stage.run_optimal_cost_bounds(
            output_path=output_path,
            solver_name=solver_name,
            time_limit=time_limit,
            tee=tee,
            solver_threads=solver_threads,
        )
        self._set_optimal_cost_bounds_from_report(report)
        return report

    def make_mccormick_bounds_from_optimal_cost_bounds(
        self,
        PoA_bounds: tuple[float, float],
        num_pieces: Optional[int] = None,
        C_opt_breakpoints: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        if num_pieces is not None and C_opt_breakpoints is not None:
            raise ValueError("Provide either num_pieces or C_opt_breakpoints, not both")
        C_opt_bounds = (self.optimal_cost_bounds or {}).get("C_opt", {})
        lower = C_opt_bounds.get("lower")
        upper = C_opt_bounds.get("upper")
        if lower is None or upper is None:
            raise ValueError(
                "Optimal cost bounds are not loaded. Run compute_optimal_cost_bounds() "
                "or load_optimal_cost_bounds_report() first."
            )
        mccormick_bounds: dict[str, Any] = {
            "PoA": PoA_bounds,
            "C_opt": (float(lower), float(upper)),
        }
        if num_pieces is not None:
            mccormick_bounds["num_pieces"] = int(num_pieces)
        if C_opt_breakpoints is not None:
            mccormick_bounds["C_opt_breakpoints"] = [
                float(value) for value in C_opt_breakpoints
            ]
        return mccormick_bounds

    def make_ratio_bounds_from_optimal_cost_bounds(
        self,
        phi_bounds: tuple[float, float],
        num_pieces: Optional[int] = None,
        C_opt_breakpoints: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        return self.make_mccormick_bounds_from_optimal_cost_bounds(
            PoA_bounds=phi_bounds,
            num_pieces=num_pieces,
            C_opt_breakpoints=C_opt_breakpoints,
        )

