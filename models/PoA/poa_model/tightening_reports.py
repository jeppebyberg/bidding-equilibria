import copy
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class PoATighteningReports:
    tightening_report_components = frozenset(
        {
            "metadata",
            "primal_big_m",
            "nn_relu_bounds_report",
            "alpha_bounds",
            "fixed_binaries",
            "tight_big_m",
            "lambda_bounds",
            "aggregate_dual_bounds",
            "optimal_cost_bounds",
        }
    )

    @staticmethod
    def _empty_default_bounds_used() -> dict[str, Any]:
        return {
            "nn_relu_bounds_report": False,
            "alpha_bounds": False,
            "tight_big_m": False,
            "lambda_bounds": False,
            "aggregate_dual_bounds": False,
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

    @staticmethod
    def _contains_deprecated_aggregate_dual_bounds(
        payload: Any,
        parent_key: Optional[str] = None,
    ) -> bool:
        if isinstance(payload, dict):
            if parent_key == "default_bounds_used":
                return False
            if "aggregate_dual_bounds" in payload:
                return True
            return any(
                PoATighteningReports._contains_deprecated_aggregate_dual_bounds(
                    value,
                    str(key),
                )
                for key, value in payload.items()
            )
        if isinstance(payload, list):
            return any(
                PoATighteningReports._contains_deprecated_aggregate_dual_bounds(
                    value,
                    parent_key,
                )
                for value in payload
            )
        return False

    @staticmethod
    def _optional_float_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("bound", "value", "tight_bound", "tight_big_m"):
                if value_key in payload:
                    return PoATighteningReports._optional_float_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    @staticmethod
    def _optional_numeric_bound(payload: Any) -> Optional[float]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            for value_key in ("tight_big_m", "big_m", "upper_bound", "ub", "bound", "value"):
                if value_key in payload:
                    return PoATighteningReports._optional_numeric_bound(payload[value_key])
            return None
        try:
            numeric_value = float(payload)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return max(0.0, numeric_value)

    def _initialize_loaded_bound_dictionaries(self) -> None:
        # Primal Big-M constants M^p_c for slack-side complementarity:
        #     -M^p_c (1 - z_c) <= g_c(x)
        self.M_cap: dict[tuple[int, int, int], float] = {}
        self.M_lower: dict[tuple[int, int, int], float] = {}
        self.M_physical_capacity: dict[int, float] = {}
        self.M_ramp_up: dict[tuple[int, int], float] = {}
        self.M_ramp_down: dict[tuple[int, int], float] = {}
        self.M_ramp_up_initial: dict[int, float] = {}
        self.M_ramp_down_initial: dict[int, float] = {}

        # Dual Big-M constants M^d_c for dual-side complementarity:
        #     mu_c <= M^d_c z_c
        self.M_mu_upper_eq: dict[tuple[int, int, int], float] = {}
        self.M_mu_lower_eq: dict[tuple[int, int, int], float] = {}
        self.M_mu_ramp_up_eq: dict[tuple[int, int], float] = {}
        self.M_mu_ramp_down_eq: dict[tuple[int, int], float] = {}
        self.M_mu_upper_opt: dict[tuple[int, int, int], float] = {}
        self.M_mu_lower_opt: dict[tuple[int, int, int], float] = {}
        self.M_mu_ramp_up_opt: dict[tuple[int, int], float] = {}
        self.M_mu_ramp_down_opt: dict[tuple[int, int], float] = {}

        self.lambda_eq_bounds: dict[int, tuple[float, float]] = {}
        self.lambda_opt_bounds: dict[int, tuple[float, float]] = {}
        self._loaded_bounds_prepared = False
        self._loaded_bounds_source_signature: tuple[int, int, int, int] | None = None

    def _loaded_bound_source_signature(self) -> tuple[int, int, int, int]:
        return (
            id(getattr(self, "primal_big_m", None)),
            id(getattr(self, "tight_big_m", None)),
            id(getattr(self, "lambda_bounds", None)),
            id(getattr(self, "aggregate_dual_bounds", None)),
        )

    def _ensure_loaded_bounds_prepared(self) -> None:
        current_signature = self._loaded_bound_source_signature()
        if (
            not getattr(self, "_loaded_bounds_prepared", False)
            or getattr(self, "_loaded_bounds_source_signature", None) != current_signature
        ):
            self._prepare_loaded_bounds()

    @staticmethod
    def _json_key(indices: tuple[int, ...]) -> str:
        return ",".join(str(int(index)) for index in indices)

    @staticmethod
    def _parse_json_index(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in str(key).split(",") if part != "")

    @staticmethod
    def _loosen_lower_upper_bounds_recursively(
        payload: Any,
        lower: float,
        upper: float,
    ) -> Any:
        if isinstance(payload, dict):
            loosened: dict[Any, Any] = {}
            for key, value in payload.items():
                key_lower = str(key).lower()
                if key_lower == "lower" and isinstance(value, (int, float)):
                    loosened[key] = float(lower)
                elif key_lower == "upper" and isinstance(value, (int, float)):
                    loosened[key] = float(upper)
                else:
                    loosened[key] = PoATighteningReports._loosen_lower_upper_bounds_recursively(
                        value,
                        lower,
                        upper,
                    )
            return loosened
        if isinstance(payload, list):
            return [
                PoATighteningReports._loosen_lower_upper_bounds_recursively(
                    item,
                    lower,
                    upper,
                )
                for item in payload
            ]
        return copy.deepcopy(payload)

    @staticmethod
    def _replace_numeric_bounds_recursively(
        payload: Any,
        bound_keys: set[str],
        replacement_value: float,
        replace_plain_numbers: bool = False,
    ) -> Any:
        if isinstance(payload, dict):
            replaced: dict[Any, Any] = {}
            for key, value in payload.items():
                key_lower = str(key).lower()
                if key_lower in bound_keys and isinstance(value, (int, float)):
                    replaced[key] = float(replacement_value)
                else:
                    replaced[key] = PoATighteningReports._replace_numeric_bounds_recursively(
                        value,
                        bound_keys,
                        replacement_value,
                        replace_plain_numbers,
                    )
            return replaced
        if isinstance(payload, list):
            return [
                PoATighteningReports._replace_numeric_bounds_recursively(
                    item,
                    bound_keys,
                    replacement_value,
                    replace_plain_numbers,
                )
                for item in payload
            ]
        if replace_plain_numbers and isinstance(payload, (int, float)):
            return float(replacement_value)
        return copy.deepcopy(payload)

    @staticmethod
    def _template_section(
        template_report: Optional[dict[str, Any]],
        section_name: str,
    ) -> dict[str, Any]:
        if not isinstance(template_report, dict) or not template_report:
            return {}
        section = template_report.get(section_name)
        if isinstance(section, dict):
            return copy.deepcopy(section)
        known_report_sections = {
            "metadata",
            "primal_big_m",
            "nn_relu_bounds_report",
            "nn_relu_bounds",
            "alpha_bounds",
            "alpha_optimization_results",
            "fixed_binaries",
            "tight_big_m",
            "lambda_bounds",
            "aggregate_dual_bounds",
            "optimal_cost_bounds",
            "optimal_cost_bound_optimization_results",
        }
        if known_report_sections & set(template_report):
            return {}
        return copy.deepcopy(template_report)

    def build_default_alpha_bounds_report(
        self,
        template_report: Optional[dict[str, Any]] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> dict[str, Any]:
        lower = float(self.default_alpha_lower if lower is None else lower)
        upper = float(self.default_alpha_upper if upper is None else upper)
        template = self._template_section(template_report, "alpha_bounds")
        if not template and getattr(self, "alpha_bounds", None):
            template = {
                self._json_key(index): dict(bounds)
                for index, bounds in self.alpha_bounds.items()
            }
        if not template:
            return {
                self._json_key((int(i), int(b), int(t))): {
                    "lower": lower,
                    "upper": upper,
                    "source": "default_loose_bounds",
                }
                for i, b in self.generator_block_pairs
                for t in range(self.num_time_steps)
            }
        return self._loosen_lower_upper_bounds_recursively(template, lower, upper)

    def build_default_tight_big_m_report(
        self,
        template_report: Optional[dict[str, Any]] = None,
        default_big_m: Optional[float] = None,
    ) -> dict[str, Any]:
        default_big_m = float(
            self.default_dual_big_m if default_big_m is None else default_big_m
        )
        template = self._template_section(template_report, "tight_big_m")
        if not template and getattr(self, "tight_big_m", None):
            template = copy.deepcopy(self.tight_big_m)
        if not template:
            report: dict[str, dict[str, Any]] = {}
            for side in ("eq", "opt"):
                for dual_name in (f"mu_upper_{side}", f"mu_lower_{side}"):
                    for i, b in self.generator_block_pairs:
                        for t in range(self.num_time_steps):
                            report.setdefault(dual_name, {})[
                                self._json_key((int(i), int(b), int(t)))
                            ] = {
                                "tight_big_m": default_big_m,
                                "source": "default_loose_bounds",
                            }
                for dual_name in (f"mu_ramp_up_{side}", f"mu_ramp_down_{side}"):
                    for i in range(self.num_physical_generators):
                        for t in range(self.num_time_steps):
                            report.setdefault(dual_name, {})[
                                self._json_key((int(i), int(t)))
                            ] = {
                                "tight_big_m": default_big_m,
                                "source": "default_loose_bounds",
                            }
            return report
        return self._replace_numeric_bounds_recursively(
            template,
            {"tight_big_m", "big_m", "upper_bound", "ub", "bound", "value"},
            default_big_m,
            replace_plain_numbers=False,
        )

    def build_default_lambda_bounds_report(
        self,
        template_report: Optional[dict[str, Any]] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> dict[str, Any]:
        lower = float(self.default_lambda_lower if lower is None else lower)
        upper = float(self.default_lambda_upper if upper is None else upper)
        template = self._template_section(template_report, "lambda_bounds")
        if not template and getattr(self, "lambda_bounds", None):
            template = copy.deepcopy(self.lambda_bounds)
        if not template:
            return {
                lambda_name: {
                    str(int(t)): {
                        "lower": lower,
                        "upper": upper,
                        "source": "default_loose_bounds",
                    }
                    for t in range(self.num_time_steps)
                }
                for lambda_name in ("lambda_eq", "lambda_opt")
            }
        return self._loosen_lower_upper_bounds_recursively(template, lower, upper)

    def build_default_aggregate_dual_bounds_report(
        self,
        template_report: Optional[dict[str, Any]] = None,
        upper: Optional[float] = None,
    ) -> dict[str, Any]:
        upper = float(self.default_aggregate_dual_upper if upper is None else upper)
        template = self._template_section(template_report, "aggregate_dual_bounds")
        if not template and getattr(self, "aggregate_dual_bounds", None):
            template = copy.deepcopy(self.aggregate_dual_bounds)
        if not template:
            return {
                bound_key: {
                    side: {
                        str(int(t)): {
                            "tight_big_m": upper,
                            "side": side,
                            "constraint_type": constraint_type,
                            "source": "default_loose_bounds",
                        }
                        for t in range(self.num_time_steps)
                    }
                    for side in ("eq", "opt")
                }
                for constraint_type, bound_key in (
                    ("upper", "mu_max_sum_ub"),
                    ("lower", "mu_min_sum_ub"),
                    ("ramp_up", "mu_ramp_up_sum_ub"),
                    ("ramp_down", "mu_ramp_down_sum_ub"),
                )
            }
        return self._replace_numeric_bounds_recursively(
            template,
            {"tight_big_m", "big_m", "upper", "upper_bound", "ub", "bound", "value"},
            upper,
            replace_plain_numbers=False,
        )

    def build_default_optimal_cost_bounds_report(
        self,
        template_report: Optional[dict[str, Any]] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> dict[str, Any]:
        lower = float(self.default_c_opt_lower if lower is None else lower)
        upper = float(self.default_c_opt_upper if upper is None else upper)
        if lower <= 0.0:
            raise ValueError("Default C_opt lower bound must be strictly positive")
        template = self._template_section(template_report, "optimal_cost_bounds")
        if not template and getattr(self, "optimal_cost_bounds", None):
            template = copy.deepcopy(self.optimal_cost_bounds)
        if not template:
            return {
                "lower": lower,
                "upper": upper,
                "source": "default_loose_bounds",
            }
        return self._loosen_lower_upper_bounds_recursively(template, lower, upper)

    @staticmethod
    def _remove_aggregate_dual_bound_payload(
        report: dict[str, Any],
    ) -> dict[str, Any]:
        filtered_report = copy.deepcopy(report)
        filtered_report.pop("aggregate_dual_bounds", None)
        aggregate_keys = {
            "mu_max_sum_ub",
            "mu_min_sum_ub",
            "mu_ramp_up_sum_ub",
            "mu_ramp_down_sum_ub",
        }
        tight_big_m = filtered_report.get("tight_big_m")
        if isinstance(tight_big_m, dict):
            for key in aggregate_keys:
                tight_big_m.pop(key, None)
        return filtered_report

    @classmethod
    def _normalize_tightening_report_components(
        cls,
        components: Optional[set[str] | list[str] | tuple[str, ...] | frozenset[str]],
    ) -> set[str]:
        if components is None:
            return set(cls.tightening_report_components)
        normalized = {str(component) for component in components}
        unknown = normalized - set(cls.tightening_report_components)
        if unknown:
            raise ValueError(
                "Unknown tightening report component(s): "
                f"{', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(cls.tightening_report_components))}"
            )
        return normalized

    @classmethod
    def _filter_tightening_report_components(
        cls,
        report: dict[str, Any],
        components: Optional[set[str] | list[str] | tuple[str, ...] | frozenset[str]] = None,
    ) -> dict[str, Any]:
        enabled_components = cls._normalize_tightening_report_components(components)
        filtered_report = copy.deepcopy(report)
        if "metadata" not in enabled_components:
            filtered_report.pop("metadata", None)
        if "primal_big_m" not in enabled_components:
            filtered_report.pop("primal_big_m", None)
        if "nn_relu_bounds_report" not in enabled_components:
            filtered_report.pop("nn_relu_bounds_report", None)
            filtered_report.pop("nn_relu_bounds", None)
        if "alpha_bounds" not in enabled_components:
            filtered_report.pop("alpha_bounds", None)
            filtered_report.pop("alpha_optimization_results", None)
        if "fixed_binaries" not in enabled_components:
            filtered_report.pop("fixed_binaries", None)
            filtered_report.pop("num_fixed_binaries", None)
        if "tight_big_m" not in enabled_components:
            filtered_report.pop("tight_big_m", None)
        if "lambda_bounds" not in enabled_components:
            filtered_report.pop("lambda_bounds", None)
        if "aggregate_dual_bounds" not in enabled_components:
            filtered_report = cls._remove_aggregate_dual_bound_payload(filtered_report)
        if "optimal_cost_bounds" not in enabled_components:
            filtered_report.pop("optimal_cost_bounds", None)
            filtered_report.pop("optimal_cost_bound_optimization_results", None)
        return filtered_report

    @classmethod
    def _filter_tightening_report_sections(
        cls,
        report: dict[str, Any],
        use_alpha_bounds: bool = True,
        use_fixed_binaries: bool = True,
        use_tight_big_m: bool = True,
        use_lambda_bounds: bool = True,
        use_aggregate_dual_bounds: bool = True,
        use_optimal_cost_bounds: bool = True,
    ) -> dict[str, Any]:
        enabled_components = set(cls.tightening_report_components)
        if not use_alpha_bounds:
            enabled_components.discard("alpha_bounds")
        if not use_fixed_binaries:
            enabled_components.discard("fixed_binaries")
        if not use_tight_big_m:
            enabled_components.discard("tight_big_m")
        if not use_lambda_bounds:
            enabled_components.discard("lambda_bounds")
        if not use_aggregate_dual_bounds:
            enabled_components.discard("aggregate_dual_bounds")
        if not use_optimal_cost_bounds:
            enabled_components.discard("optimal_cost_bounds")
        return cls._filter_tightening_report_components(report, enabled_components)

    def load_tightening_report_data(
        self,
        report: dict[str, Any],
        report_path: Optional[Path] = None,
        prepare_bounds: bool = True,
        merge_existing: bool = False,
    ) -> dict[str, Any]:
        loaded_report = copy.deepcopy(report)
        if merge_existing:
            merged_report = copy.deepcopy(getattr(self, "tightening_report", {}) or {})
            merged_report.update(loaded_report)
            loaded_report = merged_report
        if self._contains_deprecated_aggregate_dual_bounds(loaded_report):
            raise ValueError(
                "Loaded tightening report contains deprecated aggregate_dual_bounds. "
                "Recompute dual Big-M tightening with componentwise bounds only."
            )

        self.tightening_report = loaded_report
        if report_path is not None:
            self.tightening_report_path = Path(report_path)
        nn_relu_report = loaded_report.get("nn_relu_bounds_report", {}) or {}
        if not nn_relu_report and "nn_relu_bounds" in loaded_report:
            nn_relu_report = loaded_report
        if nn_relu_report:
            self._set_nn_relu_bounds_from_report(nn_relu_report)
        self.fixed_binaries = loaded_report.get("fixed_binaries", {}) or {}
        self.primal_big_m = loaded_report.get("primal_big_m", {}) or {}
        self.tight_big_m = loaded_report.get("tight_big_m", {}) or {}
        self.lambda_bounds = loaded_report.get("lambda_bounds", {}) or {}
        self.alpha_bound_optimization_results = (
            loaded_report.get("alpha_optimization_results", {}) or {}
        )
        self.alpha_bounds = {
            self._parse_json_index(key): {
                "lower": float(value["lower"]),
                "upper": float(value["upper"]),
            }
            for key, value in (loaded_report.get("alpha_bounds", {}) or {}).items()
        }
        self.optimal_cost_bounds = loaded_report.get("optimal_cost_bounds", {}) or {}
        self.optimal_cost_bound_optimization_results = (
            loaded_report.get("optimal_cost_bound_optimization_results", {}) or {}
        )
        self._loaded_bounds_prepared = False
        if prepare_bounds:
            self._prepare_loaded_bounds()
        return loaded_report

    def _set_tightening_report_data(
        self,
        report: dict[str, Any],
        report_path: Optional[Path] = None,
    ) -> None:
        self.load_tightening_report_data(
            report,
            report_path=report_path,
            prepare_bounds=False,
            merge_existing=False,
        )

    def load_tightening_report(
        self,
        report_path: str | Path = "results/poa_tightening/final_tightening_report.json",
        components: Optional[set[str] | list[str] | tuple[str, ...] | frozenset[str]] = None,
        use_alpha_bounds: bool = True,
        use_fixed_binaries: bool = True,
        use_tight_big_m: bool = True,
        use_lambda_bounds: bool = True,
        use_aggregate_dual_bounds: bool = True,
        use_optimal_cost_bounds: bool = True,
    ) -> dict[str, Any]:
        """
        Load a tightening report and prepare model constants.

        The raw JSON-friendly report remains available on the object, while all
        Big-M and lambda data used by Pyomo is parsed once into tuple-indexed
        dictionaries before model construction. Individual tightening sections
        can be disabled to solve with default bounds while still loading the
        primal Big-M and NN ReLU data required for model construction.
        """
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"Tightening report not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        if self._contains_deprecated_aggregate_dual_bounds(report):
            raise ValueError(
                "Loaded tightening report contains deprecated aggregate_dual_bounds. "
                "Recompute dual Big-M tightening with componentwise bounds only."
            )

        if components is None:
            report = self._filter_tightening_report_sections(
                report,
                use_alpha_bounds=use_alpha_bounds,
                use_fixed_binaries=use_fixed_binaries,
                use_tight_big_m=use_tight_big_m,
                use_lambda_bounds=use_lambda_bounds,
                use_aggregate_dual_bounds=use_aggregate_dual_bounds,
                use_optimal_cost_bounds=use_optimal_cost_bounds,
            )
        else:
            report = self._filter_tightening_report_components(report, components)
        return self.load_tightening_report_data(
            report,
            report_path=path,
            prepare_bounds=True,
            merge_existing=False,
        )

    def ensure_default_bounds_available(
        self,
        template_report: Optional[dict[str, Any]] = None,
        include_nn_relu_bounds: Optional[bool] = None,
        include_alpha_bounds: bool = False,
        include_tight_big_m: bool = True,
        include_lambda_bounds: bool = False,
        include_aggregate_dual_bounds: bool = False,
        include_optimal_cost_bounds: Optional[bool] = None,
        overwrite_existing: bool = False,
    ) -> dict[str, Any]:
        from models.PoA.PoA_tightening.compute_primal_big_m import (
            compute_primal_big_m_bounds,
        )

        include_nn_relu_bounds = (
            bool(self.nn_policy_generator_ids)
            if include_nn_relu_bounds is None
            else bool(include_nn_relu_bounds)
        )
        include_optimal_cost_bounds = (
            self.objective_mode in {"mccormick", "piecewise_mccormick"}
            if include_optimal_cost_bounds is None
            else bool(include_optimal_cost_bounds)
        )

        default_report = copy.deepcopy(getattr(self, "tightening_report", {}) or {})
        template_report = template_report or default_report

        if overwrite_existing or not default_report.get("primal_big_m"):
            default_report["primal_big_m"] = compute_primal_big_m_bounds(self)
            self._mark_default_bound_used(
                "primal_big_m",
                "Analytic primal Big-M defaults were used for model construction.",
            )

        if include_nn_relu_bounds and (
            overwrite_existing or not (default_report.get("nn_relu_bounds_report") or self.nn_relu_bounds)
        ):
            relu_report = self.build_default_nn_relu_bounds_report(template_report)
            if relu_report:
                default_report["nn_relu_bounds_report"] = relu_report
                default_report["nn_relu_bounds"] = relu_report.get("nn_relu_bounds", {})
                self._mark_default_bound_used(
                    "nn_relu_bounds_report",
                    "Default loose NN ReLU bounds were used.",
                )

        if include_alpha_bounds and (
            overwrite_existing or not default_report.get("alpha_bounds")
        ):
            alpha_report = self.build_default_alpha_bounds_report(template_report)
            if alpha_report:
                default_report["alpha_bounds"] = alpha_report
                self._mark_default_bound_used(
                    "alpha_bounds",
                    "Default loose alpha bounds were used.",
                )

        if include_tight_big_m and (
            overwrite_existing or not default_report.get("tight_big_m")
        ):
            tight_big_m_report = self.build_default_tight_big_m_report(template_report)
            if tight_big_m_report:
                default_report["tight_big_m"] = tight_big_m_report
                self._mark_default_bound_used(
                    "tight_big_m",
                    "Default loose dual Big-M values were used.",
                )

        if include_lambda_bounds and (
            overwrite_existing or not default_report.get("lambda_bounds")
        ):
            lambda_report = self.build_default_lambda_bounds_report(template_report)
            if lambda_report:
                default_report["lambda_bounds"] = lambda_report
                self._mark_default_bound_used(
                    "lambda_bounds",
                    "Default loose lambda bounds were used.",
                )

        if include_aggregate_dual_bounds and (
            overwrite_existing or not default_report.get("aggregate_dual_bounds")
        ):
            aggregate_report = self.build_default_aggregate_dual_bounds_report(
                template_report
            )
            if aggregate_report:
                default_report["aggregate_dual_bounds"] = aggregate_report
                self._mark_default_bound_used(
                    "aggregate_dual_bounds",
                    "Default loose aggregate dual bounds were used.",
                )

        if include_optimal_cost_bounds and (
            overwrite_existing or not default_report.get("optimal_cost_bounds")
        ):
            optimal_cost_report = self.build_default_optimal_cost_bounds_report(
                template_report
            )
            if optimal_cost_report:
                default_report["optimal_cost_bounds"] = optimal_cost_report
                self._mark_default_bound_used(
                    "optimal_cost_bounds",
                    "Default loose optimal-cost bounds were used.",
                )

        self.load_tightening_report_data(
            default_report,
            report_path=getattr(self, "tightening_report_path", None),
            prepare_bounds=True,
            merge_existing=False,
        )
        return default_report

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
                raise ValueError(
                    f"Invalid {bound_type} key '{raw_key}'. Expected key format "
                    f"{expected_key_format}."
                )
            numeric_value = self._optional_numeric_bound(details)
            if numeric_value is None:
                raise ValueError(
                    f"Invalid {bound_type} entry at key '{raw_key}'. Expected a "
                    "finite numeric bound in one of: tight_big_m, big_m, "
                    "upper_bound, ub, bound, or value."
                )
            parsed[index] = float(numeric_value)
        return parsed

    def _missing_loaded_bound_error(
        self,
        bound_type: str,
        index: tuple[int, ...] | int,
        expected_key_format: str,
    ) -> ValueError:
        return ValueError(
            f"Missing {bound_type} for index {index}. Expected key format "
            f"{expected_key_format}. Generate/load a tightening report with the "
            "required Big-M data before build_model()."
        )

    def _primal_component_entries(self, *component_names: str) -> tuple[str, Any]:
        primal_big_m = getattr(self, "primal_big_m", {}) or {}
        for component_name in component_names:
            entries = primal_big_m.get(component_name)
            if entries:
                return component_name, entries
        return component_names[0], {}

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

    def _prepare_dual_big_m(
        self,
        dual_name: str,
        expected_indices: list[tuple[int, ...]],
        default_bound: float,
        expected_key_format: str,
    ) -> dict[tuple[int, ...], float]:
        entries = (getattr(self, "tight_big_m", {}) or {}).get(dual_name, {}) or {}
        if not entries:
            if self.use_default_bounds:
                self._mark_default_bound_used(
                    "tight_big_m",
                    "Default loose dual Big-M values were used where no tightened values were loaded.",
                )
            return {index: float(default_bound) for index in expected_indices}

        parsed = self._indexed_numeric_entries(
            entries,
            f"dual Big-M '{dual_name}'",
            (len(expected_indices[0]),),
            expected_key_format,
        )
        prepared: dict[tuple[int, ...], float] = {}
        for index in expected_indices:
            if index not in parsed:
                raise self._missing_loaded_bound_error(
                    f"dual Big-M '{dual_name}'",
                    index,
                    expected_key_format,
                )
            prepared[index] = max(0.0, min(float(default_bound), float(parsed[index])))
        return prepared

    def _prepare_lambda_bounds(
        self,
        lambda_name: str,
    ) -> dict[int, tuple[float, float]]:
        entries = (getattr(self, "lambda_bounds", {}) or {}).get(lambda_name, {}) or {}
        if not entries:
            if self.use_default_bounds:
                self._mark_default_bound_used(
                    "lambda_bounds",
                    "Default loose lambda bounds were used where no tightened values were loaded.",
                )
            return {
                int(t): (
                    float(self.default_lambda_lower),
                    float(self.default_lambda_upper),
                )
                for t in range(self.num_time_steps)
            }
        if not isinstance(entries, dict):
            raise ValueError(
                f"Invalid lambda bound block '{lambda_name}'. Expected keys 't'."
            )

        prepared: dict[int, tuple[float, float]] = {}
        for t in range(self.num_time_steps):
            details = entries.get(str(int(t)), entries.get(int(t)))
            if not isinstance(details, dict):
                raise self._missing_loaded_bound_error(
                    f"lambda bounds '{lambda_name}'",
                    int(t),
                    "t",
                )
            lower = self._optional_float_bound(details.get("lower"))
            upper = self._optional_float_bound(details.get("upper"))
            if lower is None or upper is None:
                raise ValueError(
                    f"Invalid lambda bounds '{lambda_name}' for time {t}. "
                    "Expected finite 'lower' and 'upper' entries."
                )
            lower = max(float(self.default_lambda_lower), float(lower))
            upper = min(float(self.default_lambda_upper), float(upper))
            if lower > upper:
                raise ValueError(
                    f"Invalid lambda bounds '{lambda_name}' for time {t}: "
                    f"lower {lower} exceeds upper {upper}."
                )
            prepared[int(t)] = (lower, upper)
        return prepared

    def _prepare_loaded_bounds(self) -> None:
        """
        Parse JSON-compatible tightening data into tuple-indexed dictionaries.

        Pyomo variable declarations and complementarity rules read only from
        these prepared dictionaries, mirroring the thesis notation with
        precomputed primal and dual Big-M constants.
        """
        self._initialize_loaded_bound_dictionaries()

        if not getattr(self, "primal_big_m", None):
            raise ValueError(
                "Missing primal Big-M section 'primal_big_m'. Run "
                "compute_primal_big_m.py or load a tightening "
                "report with populated primal Big-M data before build_model()."
            )

        block_component, block_entries = self._primal_component_entries("block_capacity")
        self.M_cap = self._prepare_block_time_primal_big_m(
            block_component,
            block_entries,
            "i,b or i,b,t",
        )

        lower_component, lower_entries = self._primal_component_entries(
            "lower_generation",
            "generation_lower",
            "block_lower",
            "lower_bound",
        )
        self.M_lower = (
            self._prepare_block_time_primal_big_m(
                lower_component,
                lower_entries,
                "i,b or i,b,t",
            )
            if lower_entries
            else dict(self.M_cap)
        )

        physical_component, physical_entries = self._primal_component_entries(
            "physical_capacity"
        )
        self.M_physical_capacity = self._prepare_generator_primal_big_m(
            physical_component,
            physical_entries,
            "i",
        )

        ramp_up_component, ramp_up_entries = self._primal_component_entries("ramp_up")
        self.M_ramp_up = self._prepare_generator_time_primal_big_m(
            ramp_up_component,
            ramp_up_entries,
            "i or i,t",
        )
        ramp_down_component, ramp_down_entries = self._primal_component_entries(
            "ramp_down"
        )
        self.M_ramp_down = self._prepare_generator_time_primal_big_m(
            ramp_down_component,
            ramp_down_entries,
            "i or i,t",
        )

        ramp_up_initial_component, ramp_up_initial_entries = (
            self._primal_component_entries("ramp_up_initial")
        )
        self.M_ramp_up_initial = self._prepare_generator_primal_big_m(
            ramp_up_initial_component,
            ramp_up_initial_entries,
            "i",
        )
        ramp_down_initial_component, ramp_down_initial_entries = (
            self._primal_component_entries("ramp_down_initial")
        )
        self.M_ramp_down_initial = self._prepare_generator_primal_big_m(
            ramp_down_initial_component,
            ramp_down_initial_entries,
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

        self._validate_loaded_big_m_dimensions()
        self._loaded_bounds_prepared = True
        self._loaded_bounds_source_signature = self._loaded_bound_source_signature()

    def _validate_loaded_big_m_dimensions(self) -> None:
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
        generator_indices = list(range(self.num_physical_generators))
        time_indices = list(range(self.num_time_steps))

        def require_keys(
            bound_type: str,
            mapping: dict[Any, Any],
            expected_indices: list[Any],
            expected_key_format: str,
        ) -> None:
            missing = [index for index in expected_indices if index not in mapping]
            if missing:
                raise self._missing_loaded_bound_error(
                    bound_type,
                    missing[0],
                    expected_key_format,
                )

        require_keys("primal Big-M 'block_capacity'", self.M_cap, capacity_indices, "i,b,t")
        require_keys("primal Big-M 'lower_generation'", self.M_lower, capacity_indices, "i,b,t")
        require_keys(
            "primal Big-M 'physical_capacity'",
            self.M_physical_capacity,
            generator_indices,
            "i",
        )
        require_keys("primal Big-M 'ramp_up'", self.M_ramp_up, ramp_indices, "i,t")
        require_keys("primal Big-M 'ramp_down'", self.M_ramp_down, ramp_indices, "i,t")
        require_keys(
            "primal Big-M 'ramp_up_initial'",
            self.M_ramp_up_initial,
            generator_indices,
            "i",
        )
        require_keys(
            "primal Big-M 'ramp_down_initial'",
            self.M_ramp_down_initial,
            generator_indices,
            "i",
        )

        require_keys("dual Big-M 'mu_upper_eq'", self.M_mu_upper_eq, capacity_indices, "i,b,t")
        require_keys("dual Big-M 'mu_lower_eq'", self.M_mu_lower_eq, capacity_indices, "i,b,t")
        require_keys("dual Big-M 'mu_ramp_up_eq'", self.M_mu_ramp_up_eq, ramp_indices, "i,t")
        require_keys("dual Big-M 'mu_ramp_down_eq'", self.M_mu_ramp_down_eq, ramp_indices, "i,t")
        require_keys("dual Big-M 'mu_upper_opt'", self.M_mu_upper_opt, capacity_indices, "i,b,t")
        require_keys("dual Big-M 'mu_lower_opt'", self.M_mu_lower_opt, capacity_indices, "i,b,t")
        require_keys("dual Big-M 'mu_ramp_up_opt'", self.M_mu_ramp_up_opt, ramp_indices, "i,t")
        require_keys("dual Big-M 'mu_ramp_down_opt'", self.M_mu_ramp_down_opt, ramp_indices, "i,t")

        require_keys("lambda bounds 'lambda_eq'", self.lambda_eq_bounds, time_indices, "t")
        require_keys("lambda bounds 'lambda_opt'", self.lambda_opt_bounds, time_indices, "t")

    def _lookup_optional_time_bound(
        self,
        payload: Any,
        side: str,
        time_idx: int,
    ) -> Optional[float]:
        numeric_value = self._optional_numeric_bound(payload)
        if numeric_value is not None:
            return numeric_value

        if isinstance(payload, (list, tuple)):
            if 0 <= int(time_idx) < len(payload):
                return self._optional_numeric_bound(payload[int(time_idx)])
            return None

        if not isinstance(payload, dict):
            return None

        side_candidates = tuple(dict.fromkeys((side, str(side), side.lower(), side.upper())))
        time_candidates = tuple(dict.fromkeys((int(time_idx), str(int(time_idx)))))
        composite_candidates = tuple(
            dict.fromkeys(
                (
                    f"{side},{int(time_idx)}",
                    f"{side}:{int(time_idx)}",
                    f"{side}_{int(time_idx)}",
                    f"{side}-{int(time_idx)}",
                    f"{side.upper()},{int(time_idx)}",
                    f"{side.upper()}:{int(time_idx)}",
                )
            )
        )

        for key in side_candidates:
            if key in payload:
                value = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if value is not None:
                    return value

        for key in time_candidates:
            if key in payload:
                value = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if value is not None:
                    return value

        for key in composite_candidates:
            if key in payload:
                value = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if value is not None:
                    return value

        return None

    @staticmethod
    def _aggregate_dual_bound_key_candidates(
        generic_key: str,
        side: str,
        dual_name: str,
    ) -> tuple[str, ...]:
        root = (
            generic_key[: -len("_sum_ub")]
            if generic_key.endswith("_sum_ub")
            else generic_key
        )
        dual_root = dual_name
        for suffix in ("_eq", "_opt"):
            if dual_root.endswith(suffix):
                dual_root = dual_root[: -len(suffix)]

        aliases = {
            "mu_max_sum_ub": ("mu_upper_sum_ub", "mu_upper_bound_sum_ub"),
            "mu_min_sum_ub": ("mu_lower_sum_ub", "mu_lower_bound_sum_ub"),
            "mu_ramp_up_sum_ub": ("rho_up_sum_ub",),
            "mu_ramp_down_sum_ub": ("rho_down_sum_ub",),
        }
        candidates = (
            f"{dual_name}_sum_ub",
            f"{dual_root}_{side}_sum_ub",
            f"{root}_{side}_sum_ub",
            f"{dual_root}_sum_ub",
            generic_key,
            *aliases.get(generic_key, ()),
        )
        return tuple(dict.fromkeys(candidates))

    def _aggregate_dual_sum_upper_bound(
        self,
        generic_key: str,
        side: str,
        time_idx: int,
        dual_name: str,
    ) -> Optional[float]:
        """
        Return an optional aggregate dual sum bound for one KKT side and time.

        The tightening report is intentionally permissive about where these
        values live so older reports remain valid and newer reports can store
        them either in a dedicated `aggregate_dual_bounds` block or alongside
        other dual Big-M data.
        """
        report = getattr(self, "tightening_report", {}) or {}
        source_payloads: list[Any] = [
            getattr(self, "aggregate_dual_bounds", {}) or {},
        ]
        if isinstance(report, dict):
            source_payloads.extend(
                [
                    report.get("aggregate_dual_bounds", {}) or {},
                    report,
                ]
            )
        source_payloads.append(getattr(self, "tight_big_m", {}) or {})

        support_dual_bounds = self.ambiguity_set_config.get("dual_bounds", {})
        source_payloads.extend(
            [
                self.ambiguity_set_config.get("aggregate_dual_bounds", {}),
                support_dual_bounds.get("aggregate_dual_bounds", {})
                if isinstance(support_dual_bounds, dict)
                else {},
                support_dual_bounds,
                self.ambiguity_set_config,
            ]
        )

        for payload in source_payloads:
            if not isinstance(payload, dict):
                continue
            for key in self._aggregate_dual_bound_key_candidates(
                generic_key,
                side,
                dual_name,
            ):
                if key not in payload:
                    continue
                bound = self._lookup_optional_time_bound(payload[key], side, time_idx)
                if bound is not None:
                    return bound
        return None

    def apply_tightened_bounds_to_model(
        self,
        report: Optional[dict[str, Any]] = None,
        apply_alpha_bounds: bool = True,
        apply_fixed_binaries: bool = True,
        apply_dual_bounds: bool = True,
        apply_lambda_bounds: Optional[bool] = None,
        apply_dual_big_m_bounds: Optional[bool] = None,
        apply_aggregate_dual_bounds: Optional[bool] = None,
    ) -> dict[str, int]:
        """
        Apply a tightening report to the already-built PoA model.

        Fixed binaries remove complementarity cases that slack OBBT proved can
        never bind. Dual and lambda bounds are normally applied during variable
        construction from the prepared dictionaries; this method reapplies those
        variable bounds when a report is refreshed after model construction.
        Alpha bounds are redundant when the exact NN is embedded, but they still
        help the solver by tightening variable domains.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model is not built. Call build_model() first.")

        current_report = getattr(self, "tightening_report", None)
        report = report or current_report
        if report is None:
            raise ValueError("No tightening report loaded. Call load_tightening_report() first.")
        if report is current_report:
            self._ensure_loaded_bounds_prepared()
        else:
            self.load_tightening_report_data(
                report,
                report_path=getattr(self, "tightening_report_path", None),
                prepare_bounds=True,
                merge_existing=False,
            )

        m = self.model
        stats = {
            "fixed_binaries": 0,
            "lambda_bounds": 0,
            "dual_upper_bounds": 0,
            "alpha_bounds": 0,
        }

        if apply_fixed_binaries:
            for var_name, entries in (report.get("fixed_binaries", {}) or {}).items():
                binary_var = getattr(m, var_name, None)
                if binary_var is None:
                    continue
                for key, details in entries.items():
                    index = self._parse_json_index(key)
                    if index not in binary_var:
                        continue
                    binary_var[index].fix(int(details.get("fixed_value", 0)))
                    stats["fixed_binaries"] += 1

        if apply_dual_bounds:
            apply_lambda = (
                True if apply_lambda_bounds is None else bool(apply_lambda_bounds)
            )
            apply_dual_big_m = (
                True
                if apply_dual_big_m_bounds is None
                else bool(apply_dual_big_m_bounds)
            )
            if apply_lambda:
                stats["lambda_bounds"] = self._apply_lambda_bounds_to_model()
            if apply_dual_big_m:
                stats["dual_upper_bounds"] = self._apply_dual_bounds_to_model()

        if apply_alpha_bounds:
            alpha_entries = report.get("alpha_bounds", {}) or {}
            for key, bounds in alpha_entries.items():
                index = self._parse_json_index(key)
                if index not in m.alpha:
                    continue
                lower = float(bounds["lower"])
                upper = float(bounds["upper"])
                current_lb = m.alpha[index].lb
                current_ub = m.alpha[index].ub
                if current_lb is not None:
                    lower = max(float(current_lb), lower)
                if current_ub is not None:
                    upper = min(float(current_ub), upper)
                if lower <= upper:
                    m.alpha[index].setlb(lower)
                    m.alpha[index].setub(upper)
                    stats["alpha_bounds"] += 1

        self.applied_tightening_stats = stats
        return stats

    def _apply_lambda_bounds_to_model(self) -> int:
        m = self.model
        applied = 0
        lambda_bound_maps = {
            "lambda_eq": self.lambda_eq_bounds,
            "lambda_opt": self.lambda_opt_bounds,
        }
        for lambda_name, bound_map in lambda_bound_maps.items():
            lambda_var = getattr(m, lambda_name, None)
            if lambda_var is None:
                continue
            has_report_bound = bool((self.lambda_bounds or {}).get(lambda_name, {}) or {})
            for t in m.time_steps:
                lower, upper = bound_map[int(t)]
                current_lb = lambda_var[t].lb
                current_ub = lambda_var[t].ub
                if current_lb is not None:
                    lower = max(float(current_lb), lower)
                if current_ub is not None:
                    upper = min(float(current_ub), upper)
                if lower <= upper:
                    changed = (
                        current_lb is None
                        or current_ub is None
                        or not np.isclose(float(current_lb), lower)
                        or not np.isclose(float(current_ub), upper)
                    )
                    lambda_var[t].setlb(lower)
                    lambda_var[t].setub(upper)
                    if has_report_bound or changed:
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
            for index, upper in bound_map.items():
                if index not in dual_var:
                    continue
                current_ub = dual_var[index].ub
                new_ub = max(0.0, float(upper))
                if current_ub is not None:
                    new_ub = min(float(current_ub), new_ub)
                dual_var[index].setub(new_ub)
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
            for i in m.physical_generators:
                index = (int(i), self.num_time_steps)
                if index in dual_var:
                    dual_var[index].setub(0.0)
        return applied
