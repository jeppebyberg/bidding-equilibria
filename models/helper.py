from __future__ import annotations

import ast
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROFILE_TYPES = (list, tuple, np.ndarray, pd.Series)


def parse_profile(value: Any, column_name: str) -> list[float]:
    """Parse a list-like profile from scenario data into floats."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception as exc:
            raise ValueError(f"Could not parse profile column '{column_name}': {exc}") from exc

    if not isinstance(value, PROFILE_TYPES):
        raise ValueError(f"Column '{column_name}' must contain a profile")

    try:
        return [float(v) for v in value]
    except Exception as exc:
        raise ValueError(f"Profile column '{column_name}' contains non-numeric values") from exc


def parse_profile_exact_length(
    value: Any,
    expected_len: int,
    column_name: str,
) -> list[float]:
    """Parse a profile and require exactly expected_len entries."""
    profile = parse_profile(value, column_name)
    if len(profile) != expected_len:
        raise ValueError(
            f"Profile length mismatch in column '{column_name}': "
            f"expected {expected_len}, got {len(profile)}"
        )
    return profile


def ensure_profile(
    value: Any,
    expected_len: int,
    column_name: str,
    *,
    allow_truncate: bool = False,
) -> list[float]:
    """
    Return a numeric profile, expanding scalar values when needed.

    Stringified lists are parsed as profiles. Non-list strings and scalar values
    are repeated to expected_len. If allow_truncate is true, longer profiles are
    accepted and truncated; otherwise list-like inputs must match expected_len.
    """
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return [float(value)] * expected_len
        else:
            value = parsed

    if isinstance(value, PROFILE_TYPES):
        profile = [float(v) for v in value]
        if allow_truncate:
            if len(profile) < expected_len:
                raise ValueError(
                    f"{column_name} must have at least {expected_len} entries, got {len(profile)}"
                )
            return profile[:expected_len]
        if len(profile) != expected_len:
            raise ValueError(
                f"{column_name} must have length {expected_len}, got {len(profile)}"
            )
        return profile

    return [float(value)] * expected_len


def find_demand_profile_column(scenarios_df: pd.DataFrame) -> str:
    """Return the first scenario column whose name contains demand_profile."""
    for column in scenarios_df.columns:
        if "demand_profile" in str(column).lower():
            return str(column)
    raise ValueError("No demand profile column found in scenarios_df")


def infer_num_time_steps(scenarios_df: pd.DataFrame) -> int:
    """Infer the horizon from time_steps, falling back to the demand profile."""
    if "time_steps" in scenarios_df.columns:
        return int(scenarios_df["time_steps"].iloc[0])
    demand_column = find_demand_profile_column(scenarios_df)
    return len(parse_profile(scenarios_df[demand_column].iloc[0], demand_column))


def scenario_demand(
    scenarios_df: pd.DataFrame,
    scenario_id: int,
    time_id: int,
) -> float:
    """Read demand for one scenario and time index."""
    column = find_demand_profile_column(scenarios_df)
    profile = parse_profile(scenarios_df[column].iloc[int(scenario_id)], column)
    return float(profile[int(time_id)])


def available_block_capacity(
    scenarios_df: pd.DataFrame,
    block_name: str,
    scenario_id: int,
    time_id: int,
) -> float:
    """Read time-dependent capacity if present, otherwise static *_cap."""
    row = scenarios_df.iloc[int(scenario_id)]
    for suffix in ("_cap_profile", "_profile"):
        column = f"{block_name}{suffix}"
        if column in scenarios_df.columns:
            profile = parse_profile(row[column], column)
            return float(profile[int(time_id)])
    return float(row[f"{block_name}_cap"])


def per_generator_config_value(
    raw: Any,
    generator_idx: int,
    generator_names: Sequence[str],
    default: Any,
) -> Any:
    """Resolve a scalar/list/dict config value for one generator."""
    if raw is None:
        return default
    if isinstance(raw, dict):
        name = generator_names[int(generator_idx)]
        for key in (generator_idx, str(generator_idx), name, name.upper(), name.lower()):
            if key in raw:
                return raw[key]
        return default
    if isinstance(raw, PROFILE_TYPES):
        return raw[int(generator_idx)]
    return raw


def wind_generator_config_value(
    cfg: dict[str, Any],
    field_name: str,
    generator_idx: int,
    generator_names: Sequence[str],
    default: Any,
) -> Any:
    """Resolve wind support-set config values from grouped or legacy keys."""
    grouped = cfg.get("wind_generators")
    name = generator_names[int(generator_idx)]
    if isinstance(grouped, dict):
        for key in (generator_idx, str(generator_idx), name, name.upper(), name.lower()):
            if key in grouped and isinstance(grouped[key], dict) and field_name in grouped[key]:
                return grouped[key][field_name]

    legacy_key = {"reference": "wind_reference", "min": "wind_min", "max": "wind_max"}[
        field_name
    ]
    return per_generator_config_value(
        cfg.get(legacy_key),
        generator_idx,
        generator_names,
        default,
    )
