from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROFILE_TYPES = (list, tuple, np.ndarray, pd.Series)


@dataclass(frozen=True)
class BlockStructure:
    """Shared block/physical-generator indexing used by ED-like models."""

    block_names: list[str]
    physical_generator_names: list[str]
    block_to_physical: dict[str, str]
    block_to_physical_idx: list[int]
    physical_to_block_indices: list[list[int]]
    blocks_by_generator: dict[int, list[int]]
    local_blocks_by_generator: dict[int, list[int]]
    local_to_global_block: dict[tuple[int, int], int]
    global_to_local_block: dict[int, tuple[int, int]]
    generator_block_pairs: list[tuple[int, int]]


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


def block_names_from_capacity_columns(scenarios_df: pd.DataFrame) -> list[str]:
    """Infer bidding-block names from scenario *_cap columns."""
    return [
        str(column).removesuffix("_cap")
        for column in scenarios_df.columns
        if str(column).endswith("_cap")
    ]


def physical_generator_names_from_ramps(ramps_df: pd.DataFrame) -> list[str]:
    """Infer physical generator names from *_ramp_up columns."""
    return [
        str(column).removesuffix("_ramp_up")
        for column in ramps_df.columns
        if str(column).endswith("_ramp_up")
    ]


def infer_physical_from_block_name(block_name: str) -> str:
    """Infer the physical generator part of a block name such as G1_B2 -> G1."""
    if "_B" in block_name:
        return block_name.rsplit("_B", 1)[0]
    return block_name


def build_block_structure(
    block_names: Sequence[str],
    physical_generator_names: Sequence[str],
) -> BlockStructure:
    """Build consistent global/local block mappings for physical generators."""
    block_names = [str(name) for name in block_names]
    physical_generator_names = [str(name) for name in physical_generator_names]
    physical_idx_by_name = {
        name: idx for idx, name in enumerate(physical_generator_names)
    }

    block_to_physical: dict[str, str] = {}
    block_to_physical_idx: list[int] = []
    physical_to_block_indices: list[list[int]] = [
        [] for _ in physical_generator_names
    ]

    for block_idx, block_name in enumerate(block_names):
        physical_name = infer_physical_from_block_name(block_name)
        if physical_name not in physical_idx_by_name:
            raise ValueError(
                f"Block '{block_name}' maps to physical generator '{physical_name}', "
                "but no matching ramp columns were found."
            )

        physical_idx = physical_idx_by_name[physical_name]
        block_to_physical[block_name] = physical_name
        block_to_physical_idx.append(physical_idx)
        physical_to_block_indices[physical_idx].append(block_idx)

    blocks_by_generator = {
        generator_idx: list(block_indices)
        for generator_idx, block_indices in enumerate(physical_to_block_indices)
    }
    local_blocks_by_generator = {
        generator_idx: list(range(len(block_indices)))
        for generator_idx, block_indices in blocks_by_generator.items()
    }
    local_to_global_block = {
        (generator_idx, local_block_idx): global_block_idx
        for generator_idx, block_indices in blocks_by_generator.items()
        for local_block_idx, global_block_idx in enumerate(block_indices)
    }
    global_to_local_block = {
        global_block_idx: local_block
        for local_block, global_block_idx in local_to_global_block.items()
    }

    return BlockStructure(
        block_names=block_names,
        physical_generator_names=physical_generator_names,
        block_to_physical=block_to_physical,
        block_to_physical_idx=block_to_physical_idx,
        physical_to_block_indices=physical_to_block_indices,
        blocks_by_generator=blocks_by_generator,
        local_blocks_by_generator=local_blocks_by_generator,
        local_to_global_block=local_to_global_block,
        global_to_local_block=global_to_local_block,
        generator_block_pairs=list(local_to_global_block),
    )


def target_columns_to_local_blocks(
    generator_name: str,
    target_columns: Sequence[str],
    block_names: Sequence[str],
    physical_generator_names: Sequence[str],
    global_to_local_block: Mapping[int, tuple[int, int]],
    local_blocks_by_generator: Mapping[int, Sequence[int]],
    target_column_prefix: str = "target_bid_",
) -> dict[int, int]:
    """Map NN target bid columns to local block indices for one generator."""
    block_names = [str(name) for name in block_names]
    physical_generator_names = [str(name) for name in physical_generator_names]
    generator_name = str(generator_name)
    if generator_name not in physical_generator_names:
        raise ValueError(
            f"Unknown generator '{generator_name}'. "
            f"Available: {physical_generator_names}"
        )

    generator_idx = physical_generator_names.index(generator_name)
    output_to_local_block: dict[int, int] = {}
    seen_local_blocks: set[int] = set()
    for output_idx, column in enumerate(target_columns):
        column = str(column)
        if not column.startswith(target_column_prefix):
            raise ValueError(
                f"{generator_name}: target column must start with "
                f"'{target_column_prefix}': {column}"
            )
        block_name = column.removeprefix(target_column_prefix)
        if block_name not in block_names:
            raise ValueError(f"{generator_name}: unknown target block '{block_name}'")

        global_block = block_names.index(block_name)
        block_generator_idx, local_block = global_to_local_block[global_block]
        if block_generator_idx != generator_idx:
            raise ValueError(
                f"{generator_name}: target block '{block_name}' belongs to "
                f"{physical_generator_names[block_generator_idx]}"
            )
        output_to_local_block[output_idx] = int(local_block)
        seen_local_blocks.add(int(local_block))

    expected = set(int(block) for block in local_blocks_by_generator[generator_idx])
    if seen_local_blocks != expected:
        raise ValueError(
            f"{generator_name}: target columns must cover local blocks "
            f"{sorted(expected)}, got {sorted(seen_local_blocks)}"
        )
    return output_to_local_block


def block_structure_from_dataframes(
    scenarios_df: pd.DataFrame,
    ramps_df: pd.DataFrame,
) -> BlockStructure:
    """Infer block names and physical-generator mappings from input data."""
    return build_block_structure(
        block_names_from_capacity_columns(scenarios_df),
        physical_generator_names_from_ramps(ramps_df),
    )


def block_cost_vector(costs_df: pd.DataFrame, block_names: Sequence[str]) -> list[float]:
    """Read static bidding-block costs in block-name order."""
    return [float(costs_df[f"{block}_cost"].iloc[0]) for block in block_names]


def ramp_vectors(
    ramps_df: pd.DataFrame,
    physical_generator_names: Sequence[str],
) -> tuple[list[float], list[float]]:
    """Read physical-generator ramp-up and ramp-down vectors."""
    ramp_up = [
        float(ramps_df[f"{physical}_ramp_up"].iloc[0])
        for physical in physical_generator_names
    ]
    ramp_down = [
        float(ramps_df[f"{physical}_ramp_down"].iloc[0])
        for physical in physical_generator_names
    ]
    return ramp_up, ramp_down


def half_capacity_initial_dispatch(
    scenarios_df: pd.DataFrame,
    block_names: Sequence[str],
    physical_to_block_indices: Sequence[Sequence[int]],
) -> list[list[float]]:
    """Return 50% static physical capacity as [scenario][physical_generator]."""
    initial_dispatch = []
    for _, row in scenarios_df.iterrows():
        physical_initial = []
        for block_indices in physical_to_block_indices:
            physical_capacity = sum(
                float(row[f"{block_names[int(block_idx)]}_cap"])
                for block_idx in block_indices
            )
            physical_initial.append(0.5 * physical_capacity)
        initial_dispatch.append(physical_initial)
    return initial_dispatch


def coerce_index_or_name(value: Any, names: Sequence[str], label: str) -> int:
    """Accept either an integer index or a name from names."""
    names = [str(name) for name in names]
    if isinstance(value, str) and not value.strip().lstrip("-").isdigit():
        if value not in names:
            raise ValueError(f"Unknown {label} name '{value}'. Available: {names}")
        return names.index(value)

    idx = int(value)
    if idx < 0 or idx >= len(names):
        raise ValueError(f"{label} index {idx} is out of range [0, {len(names) - 1}]")
    return idx


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
