from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pyomo.environ import (
    Binary,
    Constraint,
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    Var,
    maximize,
)


class PoAMcCormick:
    """Validation and Pyomo builders for PoA McCormick objective modes."""

    mccormick_bounds_tolerance = 1e-7

    def _normalize_mccormick_bounds_alias(
        self,
        mccormick_bounds: Optional[dict[str, Any]],
        ratio_bounds: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        raw_bounds = mccormick_bounds if mccormick_bounds is not None else ratio_bounds
        if raw_bounds is None or not isinstance(raw_bounds, dict):
            return raw_bounds
        normalized = dict(raw_bounds)
        if "PoA" not in normalized and "phi" in normalized:
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

    def _validate_mccormick_bounds(
        self,
        mccormick_bounds: Optional[dict[str, tuple[float, float]]],
    ) -> Optional[dict[str, tuple[float, float]]]:
        if self.objective_mode == "difference":
            return mccormick_bounds

        if mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds or ratio_bounds is required when "
                f"objective_mode='{self.objective_mode}'"
            )
        if not isinstance(mccormick_bounds, dict):
            raise ValueError("mccormick_bounds must be a dictionary")
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

    def _build_mccormick_variables(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                f"mccormick_bounds is required when objective_mode='{self.objective_mode}'"
            )
        PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        C_opt_L, C_opt_U = self.mccormick_bounds["C_opt"]
        self.model.PoA = Var(bounds=(PoA_L, PoA_U))
        self.model.z_mccormick_product = Var(
            domain=Reals,
            bounds=(PoA_L * C_opt_L, PoA_U * C_opt_U),
        )

    def _build_piecewise_mccormick_variables(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds is required when "
                "objective_mode='piecewise_mccormick'"
            )
        breakpoints = list(self.mccormick_bounds["C_opt_breakpoints"])
        PoA_U = self.mccormick_bounds["PoA"][1]
        self.model.mccormick_piece_index = Set(initialize=range(len(breakpoints) - 1))
        self.model.mccormick_piece_active = Var(self.model.mccormick_piece_index, domain=Binary)
        self.model.C_opt_piece = Var(
            self.model.mccormick_piece_index,
            domain=NonNegativeReals,
            bounds=lambda m, k: (0.0, breakpoints[int(k) + 1]),
        )
        self.model.PoA_piece = Var(
            self.model.mccormick_piece_index,
            domain=NonNegativeReals,
            bounds=(0.0, PoA_U),
        )
        self.model.z_mccormick_piece = Var(
            self.model.mccormick_piece_index,
            domain=NonNegativeReals,
            bounds=lambda m, k: (0.0, PoA_U * breakpoints[int(k) + 1]),
        )

    def _build_mccormick_objective(self) -> None:
        # McCormick objective: maximize PoA, with C_eq = PoA * C_opt relaxed below.
        self.model.objective = Objective(expr=self.model.PoA, sense=maximize)

    def _build_piecewise_mccormick_objective(self) -> None:
        # Piecewise mccormick objective: maximize PoA, using disaggregated McCormick
        # envelopes over the C_opt denominator intervals.
        self.model.objective = Objective(expr=self.model.PoA, sense=maximize)

    def _build_mccormick_constraints(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds is required when objective_mode='mccormick'"
            )
        m = self.model
        PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        C_opt_L, C_opt_U = self.mccormick_bounds["C_opt"]

        # McCormick relaxation of C_eq = PoA * C_opt. Because this is a
        # relaxation, the realized mccormick C_eq / C_opt should be checked ex post.
        m.mccormick_link_eq_cost = Constraint(expr=m.z_mccormick_product == m.C_eq)
        m.mccormick_lower_1 = Constraint(
            expr=m.z_mccormick_product
            >= PoA_L * m.C_opt + C_opt_L * m.PoA - PoA_L * C_opt_L
        )
        m.mccormick_lower_2 = Constraint(
            expr=m.z_mccormick_product
            >= PoA_U * m.C_opt + C_opt_U * m.PoA - PoA_U * C_opt_U
        )
        m.mccormick_upper_1 = Constraint(
            expr=m.z_mccormick_product
            <= PoA_U * m.C_opt + C_opt_L * m.PoA - PoA_U * C_opt_L
        )
        m.mccormick_upper_2 = Constraint(
            expr=m.z_mccormick_product
            <= PoA_L * m.C_opt + C_opt_U * m.PoA - PoA_L * C_opt_U
        )

    def _build_piecewise_mccormick_constraints(self) -> None:
        if self.mccormick_bounds is None:
            raise ValueError(
                "mccormick_bounds is required when "
                "objective_mode='piecewise_mccormick'"
            )
        m = self.model
        PoA_L, PoA_U = self.mccormick_bounds["PoA"]
        breakpoints = list(self.mccormick_bounds["C_opt_breakpoints"])

        m.mccormick_piece_select_one = Constraint(
            expr=sum(m.mccormick_piece_active[k] for k in m.mccormick_piece_index) == 1
        )
        m.mccormick_piece_C_opt_link = Constraint(
            expr=m.C_opt == sum(m.C_opt_piece[k] for k in m.mccormick_piece_index)
        )
        m.mccormick_piece_PoA_link = Constraint(
            expr=m.PoA == sum(m.PoA_piece[k] for k in m.mccormick_piece_index)
        )
        m.mccormick_piece_z_link = Constraint(
            expr=m.z_mccormick_product
            == sum(m.z_mccormick_piece[k] for k in m.mccormick_piece_index)
        )
        m.mccormick_piece_C_eq_link = Constraint(expr=m.C_eq == m.z_mccormick_product)

        def C_opt_piece_lower_rule(m, k):
            return (
                breakpoints[int(k)] * m.mccormick_piece_active[k]
                <= m.C_opt_piece[k]
            )

        def C_opt_piece_upper_rule(m, k):
            return (
                m.C_opt_piece[k]
                <= breakpoints[int(k) + 1] * m.mccormick_piece_active[k]
            )

        def PoA_piece_lower_rule(m, k):
            return PoA_L * m.mccormick_piece_active[k] <= m.PoA_piece[k]

        def PoA_piece_upper_rule(m, k):
            return m.PoA_piece[k] <= PoA_U * m.mccormick_piece_active[k]

        def lower_1_rule(m, k):
            k_int = int(k)
            y_L = breakpoints[k_int]
            return (
                m.z_mccormick_piece[k]
                >= PoA_L * m.C_opt_piece[k]
                + y_L * m.PoA_piece[k]
                - PoA_L * y_L * m.mccormick_piece_active[k]
            )

        def lower_2_rule(m, k):
            k_int = int(k)
            y_U = breakpoints[k_int + 1]
            return (
                m.z_mccormick_piece[k]
                >= PoA_U * m.C_opt_piece[k]
                + y_U * m.PoA_piece[k]
                - PoA_U * y_U * m.mccormick_piece_active[k]
            )

        def upper_1_rule(m, k):
            k_int = int(k)
            y_L = breakpoints[k_int]
            return (
                m.z_mccormick_piece[k]
                <= PoA_U * m.C_opt_piece[k]
                + y_L * m.PoA_piece[k]
                - PoA_U * y_L * m.mccormick_piece_active[k]
            )

        def upper_2_rule(m, k):
            k_int = int(k)
            y_U = breakpoints[k_int + 1]
            return (
                m.z_mccormick_piece[k]
                <= PoA_L * m.C_opt_piece[k]
                + y_U * m.PoA_piece[k]
                - PoA_L * y_U * m.mccormick_piece_active[k]
            )

        m.mccormick_piece_C_opt_lower = Constraint(
            m.mccormick_piece_index,
            rule=C_opt_piece_lower_rule,
        )
        m.mccormick_piece_C_opt_upper = Constraint(
            m.mccormick_piece_index,
            rule=C_opt_piece_upper_rule,
        )
        m.mccormick_piece_PoA_lower = Constraint(
            m.mccormick_piece_index,
            rule=PoA_piece_lower_rule,
        )
        m.mccormick_piece_PoA_upper = Constraint(
            m.mccormick_piece_index,
            rule=PoA_piece_upper_rule,
        )
        m.mccormick_piece_mccormick_lower_1 = Constraint(
            m.mccormick_piece_index,
            rule=lower_1_rule,
        )
        m.mccormick_piece_mccormick_lower_2 = Constraint(
            m.mccormick_piece_index,
            rule=lower_2_rule,
        )
        m.mccormick_piece_mccormick_upper_1 = Constraint(
            m.mccormick_piece_index,
            rule=upper_1_rule,
        )
        m.mccormick_piece_mccormick_upper_2 = Constraint(
            m.mccormick_piece_index,
            rule=upper_2_rule,
        )
