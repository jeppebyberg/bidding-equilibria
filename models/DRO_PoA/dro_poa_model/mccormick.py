from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np


class DROPoAMcCormick:
    """Validation and completion helpers for DRO McCormick objective bounds."""

    mccormick_bounds_tolerance = 1e-7

    def _normalize_mccormick_bounds_alias(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
        ratio_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Normalize legacy ratio/phi aliases to mccormick_bounds/PoA.

        Deprecated:
            Use mccormick_bounds={"PoA": (...), ...}; the legacy "phi" key and
            ratio_bounds argument are retained temporarily for old callers.
        """
        raw_bounds = mccormick_bounds if mccormick_bounds is not None else ratio_bounds
        if raw_bounds is None or not isinstance(raw_bounds, dict):
            return raw_bounds
        normalized = dict(raw_bounds)
        if ratio_bounds is not None and mccormick_bounds is None:
            warnings.warn(
                "ratio_bounds is deprecated; pass mccormick_bounds instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if "PoA" not in normalized and "phi" in normalized:
            warnings.warn(
                "mccormick_bounds['phi'] is deprecated; use 'PoA' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            normalized["PoA"] = normalized["phi"]
        return normalized

    def _default_mccormick_bounds_payload(
        self,
        mccormick_bounds: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        completed = dict(mccormick_bounds or {})
        completed.setdefault(
            "PoA",
            (float(self.default_PoA_lower), float(self.default_PoA_upper)),
        )
        completed.setdefault(
            "C_opt",
            (float(self.default_c_opt_lower), float(self.default_c_opt_upper)),
        )
        if self.objective_mode == "piecewise_mccormick":
            completed.setdefault("num_pieces", 10)
        return completed

    def _mccormick_bounds_with_defaults(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return mccormick_bounds
        if mccormick_bounds is None:
            if not self.use_default_bounds:
                return None
            self._mark_default_bound_used(
                "optimal_cost_bounds",
                "Default loose mccormick/C_opt bounds were used because mccormick_bounds was missing.",
            )
            return self._default_mccormick_bounds_payload()
        if self.use_default_bounds and (
            "C_opt" not in mccormick_bounds or "PoA" not in mccormick_bounds
        ):
            self._mark_default_bound_used(
                "optimal_cost_bounds",
                "Default loose mccormick/C_opt bounds filled missing mccormick_bounds entries.",
            )
            return self._default_mccormick_bounds_payload(mccormick_bounds)
        return mccormick_bounds

    def _validate_deferred_mccormick_bounds(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds with at least 'PoA' is required when "
                f"objective_mode='{self.objective_mode}' and "
                "defer_mccormick_bound_validation=True"
            )
        if not isinstance(mccormick_bounds, dict):
            raise ValueError("mccormick_bounds must be a dictionary")
        mccormick_bounds = self._normalize_mccormick_bounds_alias(mccormick_bounds, None)
        if "PoA" not in mccormick_bounds:
            raise ValueError("mccormick_bounds must contain 'PoA' for deferred mccormick validation")
        raw_PoA = mccormick_bounds["PoA"]
        if not isinstance(raw_PoA, (list, tuple)) or len(raw_PoA) != 2:
            raise ValueError("mccormick_bounds['PoA'] must be a pair (lower, upper)")
        PoA_L = float(raw_PoA[0])
        PoA_U = float(raw_PoA[1])
        if not np.isfinite(PoA_L) or not np.isfinite(PoA_U):
            raise ValueError("mccormick_bounds['PoA'] entries must be finite")
        if PoA_L < 0.0:
            raise ValueError("mccormick_bounds['PoA'][0] must be nonnegative")
        if PoA_U <= PoA_L:
            raise ValueError(
                "mccormick_bounds['PoA'][1] must be greater than mccormick_bounds['PoA'][0]"
            )
        if self.objective_mode == "piecewise_mccormick":
            if "num_pieces" not in mccormick_bounds and "C_opt_breakpoints" not in mccormick_bounds:
                raise ValueError(
                    "deferred piecewise_mccormick bounds must include "
                    "'num_pieces' or 'C_opt_breakpoints'"
                )
            if "num_pieces" in mccormick_bounds:
                try:
                    num_pieces = int(mccormick_bounds["num_pieces"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("mccormick_bounds['num_pieces'] must be an integer") from exc
                if num_pieces < 2:
                    raise ValueError("mccormick_bounds['num_pieces'] must be at least 2")
        return dict(mccormick_bounds)

    def _validate_mccormick_bounds(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return mccormick_bounds

        if mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds is required when objective_mode='{self.objective_mode}'"
            )
        if not isinstance(mccormick_bounds, dict):
            raise ValueError("mccormick_bounds must be a dictionary")
        mccormick_bounds = self._normalize_mccormick_bounds_alias(mccormick_bounds, None)
        missing = [key for key in ("PoA", "C_opt") if key not in mccormick_bounds]
        if missing:
            raise ValueError(
                "mccormick_bounds must contain bounds for: " + ", ".join(missing)
            )

        def parse_bounds(key: str) -> tuple[float, float]:
            raw_bounds = mccormick_bounds[key]
            if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
                raise ValueError(f"mccormick_bounds['{key}'] must be a pair (lower, upper)")
            lower = float(raw_bounds[0])
            upper = float(raw_bounds[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"mccormick_bounds['{key}'] entries must be finite")
            return lower, upper

        PoA_L, PoA_U = parse_bounds("PoA")
        C_opt_L, C_opt_U = parse_bounds("C_opt")

        if C_opt_L <= 0.0:
            raise ValueError("mccormick_bounds['C_opt'][0] must be strictly positive")
        if C_opt_U < C_opt_L:
            raise ValueError(
                "mccormick_bounds['C_opt'][1] must be greater than or equal to "
                "mccormick_bounds['C_opt'][0]"
            )
        if PoA_L < 0.0:
            raise ValueError("mccormick_bounds['PoA'][0] must be nonnegative")
        if PoA_U <= PoA_L:
            raise ValueError(
                "mccormick_bounds['PoA'][1] must be greater than mccormick_bounds['PoA'][0]"
            )

        validated_bounds: dict[str, Any] = {
            "PoA": (PoA_L, PoA_U),
            "C_opt": (C_opt_L, C_opt_U),
        }
        if self.objective_mode == "piecewise_mccormick":
            validated_bounds["C_opt_breakpoints"] = self._validate_mccormick_breakpoints(
                mccormick_bounds,
                C_opt_L,
                C_opt_U,
            )
            validated_bounds["num_pieces"] = (
                len(validated_bounds["C_opt_breakpoints"]) - 1
            )
        return validated_bounds

    def _validate_mccormick_breakpoints(
        self,
        mccormick_bounds: dict[str, Any],
        C_opt_L: float,
        C_opt_U: float,
    ) -> list[float]:
        tolerance = self.mccormick_bounds_tolerance
        if "C_opt_breakpoints" in mccormick_bounds:
            raw_breakpoints = mccormick_bounds["C_opt_breakpoints"]
            if not isinstance(raw_breakpoints, (list, tuple)):
                raise ValueError("mccormick_bounds['C_opt_breakpoints'] must be a list")
            breakpoints = [float(value) for value in raw_breakpoints]
            if len(breakpoints) < 3:
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'] must contain at least 3 values"
                )
            if not all(np.isfinite(value) for value in breakpoints):
                raise ValueError("mccormick_bounds['C_opt_breakpoints'] entries must be finite")
            if abs(breakpoints[0] - C_opt_L) > tolerance:
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'][0] must match "
                    "mccormick_bounds['C_opt'][0]"
                )
            if abs(breakpoints[-1] - C_opt_U) > tolerance:
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'][-1] must match "
                    "mccormick_bounds['C_opt'][1]"
                )
            if any(
                breakpoints[idx + 1] <= breakpoints[idx]
                for idx in range(len(breakpoints) - 1)
            ):
                raise ValueError(
                    "mccormick_bounds['C_opt_breakpoints'] must be strictly increasing"
                )
            return breakpoints

        if "num_pieces" not in mccormick_bounds:
            raise ValueError(
                "mccormick_bounds must include either 'num_pieces' or "
                "'C_opt_breakpoints' for objective_mode='piecewise_mccormick'"
            )
        try:
            num_pieces = int(mccormick_bounds["num_pieces"])
        except (TypeError, ValueError) as exc:
            raise ValueError("mccormick_bounds['num_pieces'] must be an integer") from exc
        if num_pieces < 2:
            raise ValueError("mccormick_bounds['num_pieces'] must be at least 2")
        return [
            float(value)
            for value in np.linspace(C_opt_L, C_opt_U, num_pieces + 1)
        ]

    def _mccormick_bounds_with_loaded_C_opt_bounds(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if self.objective_mode == "difference":
            return mccormick_bounds
        if mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds is required when objective_mode='{self.objective_mode}'"
            )
        mccormick_bounds = self._normalize_mccormick_bounds_alias(mccormick_bounds, None)
        if "C_opt" in mccormick_bounds:
            return mccormick_bounds

        C_opt_bounds = self.optimal_cost_bounds or {}
        if "C_opt" in C_opt_bounds and isinstance(C_opt_bounds.get("C_opt"), dict):
            C_opt_bounds = C_opt_bounds.get("C_opt", {}) or {}
        lower = C_opt_bounds.get("lower")
        upper = C_opt_bounds.get("upper")
        if lower is None or upper is None:
            raise ValueError(
                "Mccormick objective modes require denominator bounds. Pass "
                "mccormick_bounds['C_opt'] explicitly or run/load the DRO "
                "optimal-cost-bound tightening stage first."
            )
        completed = dict(mccormick_bounds)
        completed["C_opt"] = (float(lower), float(upper))
        return completed
