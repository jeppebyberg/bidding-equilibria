from __future__ import annotations

import ast
import atexit
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROFILE_TYPES = (list, tuple, np.ndarray, pd.Series)


def derive_poa_upper_bound(
    c_eq_max: float,
    c_opt_min: float,
    lower: float = 1.0,
    margin: float = 1e-3,
) -> tuple[float, float]:
    """Derive the McCormick PoA box (PoA_L, PoA_U) from cost extrema.

    PoA(omega) = C_eq/C_opt <= max_omega C_eq / min_omega C_opt over the support
    set (C_eq >= 0, C_opt > 0). The numerator max C_eq comes from the
    equilibrium_cost_bounds tightening stage and the denominator min C_opt from
    the optimal_cost_bounds stage. ``margin`` is a small relative cushion so the
    box does not clamp a worst case sitting exactly on it; the lower bound is 1.0
    by default (equilibrium cost >= optimal cost).
    """
    c_eq_max = float(c_eq_max)
    c_opt_min = float(c_opt_min)
    if c_opt_min <= 0.0:
        raise ValueError(f"min C_opt must be strictly positive; got {c_opt_min!r}.")
    if c_eq_max < 0.0:
        raise ValueError(f"max C_eq must be non-negative; got {c_eq_max!r}.")
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative; got {margin!r}.")
    poa_lower = float(lower)
    poa_upper = c_eq_max / c_opt_min * (1.0 + float(margin))
    # Guard so the McCormick envelope is non-degenerate (PoA_U strictly > PoA_L).
    poa_upper = max(poa_upper, poa_lower * (1.0 + 1e-6) + 1e-6)
    return (poa_lower, poa_upper)


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
            raise ValueError(f"{column_name} must have length {expected_len}, got {len(profile)}")
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
    physical_idx_by_name = {name: idx for idx, name in enumerate(physical_generator_names)}

    block_to_physical: dict[str, str] = {}
    block_to_physical_idx: list[int] = []
    physical_to_block_indices: list[list[int]] = [[] for _ in physical_generator_names]

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
            f"Unknown generator '{generator_name}'. " f"Available: {physical_generator_names}"
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
        float(ramps_df[f"{physical}_ramp_up"].iloc[0]) for physical in physical_generator_names
    ]
    ramp_down = [
        float(ramps_df[f"{physical}_ramp_down"].iloc[0]) for physical in physical_generator_names
    ]
    return ramp_up, ramp_down


def is_wind_generator_name(name: str) -> bool:
    """Return true for wind generator names used by support-set logic."""
    stripped = str(name).strip()
    return stripped.upper().startswith("W") or "wind" in stripped.lower()


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
                float(row[f"{block_names[int(block_idx)]}_cap"]) for block_idx in block_indices
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

    legacy_key = {"reference": "wind_reference", "min": "wind_min", "max": "wind_max"}[field_name]
    return per_generator_config_value(
        cfg.get(legacy_key),
        generator_idx,
        generator_names,
        default,
    )


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN, Inf) with None.

    Python's json.dump writes NaN and Infinity as bare tokens by default
    (allow_nan=True), which is not valid JSON and causes parse errors in strict
    readers.  Call this before json.dump to produce a valid file.
    """
    import math

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Solve instrumentation (computation time and variable counts)
# ---------------------------------------------------------------------------


def count_integer_variables(model: Any) -> dict[str, int]:
    """Count discrete decision variables in a Pyomo model.

    Binary variables are counted separately from general integers. The ``free``
    counts exclude variables fixed by tightening (e.g. ReLU binaries pinned
    active/inactive); the free binary count is what actually drives MILP
    branch-and-bound difficulty.
    """
    from pyomo.environ import Var

    num_binary = num_binary_fixed = 0
    num_integer = num_integer_fixed = 0
    num_continuous = 0
    for var_data in model.component_data_objects(Var, active=True):
        if var_data.is_binary():
            num_binary += 1
            if var_data.fixed:
                num_binary_fixed += 1
        elif var_data.is_integer():
            num_integer += 1
            if var_data.fixed:
                num_integer_fixed += 1
        else:
            num_continuous += 1

    num_discrete_total = num_binary + num_integer
    num_discrete_fixed = num_binary_fixed + num_integer_fixed
    return {
        "num_binary_variables": num_binary,
        "num_binary_variables_fixed": num_binary_fixed,
        "num_binary_variables_free": num_binary - num_binary_fixed,
        "num_integer_variables": num_integer,
        "num_integer_variables_fixed": num_integer_fixed,
        "num_integer_variables_free": num_integer - num_integer_fixed,
        "num_discrete_variables_total": num_discrete_total,
        "num_discrete_variables_free": num_discrete_total - num_discrete_fixed,
        "num_continuous_variables": num_continuous,
    }


def build_solver_summary(optimizer: Any) -> dict[str, Any]:
    """Build the ``solver`` results block: status, solve time, variable counts.

    Reads ``solver_results`` and ``solve_wall_time_seconds`` (set by the solve
    method) plus the live Pyomo ``model`` off ``optimizer``. Every field is
    optional so the summary degrades gracefully if a solve was skipped.
    """
    summary: dict[str, Any] = {}

    solver_results = getattr(optimizer, "solver_results", None)
    if solver_results is not None:
        summary["status"] = str(solver_results.solver.status)
        summary["termination_condition"] = str(solver_results.solver.termination_condition)
        for attr in ("wallclock_time", "time"):
            reported = getattr(solver_results.solver, attr, None)
            if reported is not None:
                try:
                    summary["solver_reported_time_seconds"] = float(reported)
                except (TypeError, ValueError):
                    pass
                break

    wall_time = getattr(optimizer, "solve_wall_time_seconds", None)
    if wall_time is not None:
        summary["wall_time_seconds"] = float(wall_time)

    # Certified MIP bracket (set by solves that expose it, e.g. the DRO eta
    # sweep): with the incumbent objective this gives a reportable interval
    # even when the solve stopped at a time limit.
    for attr_name in ("best_objective_bound", "mip_gap"):
        attr_value = getattr(optimizer, attr_name, None)
        if attr_value is not None:
            summary[attr_name] = float(attr_value)

    model = getattr(optimizer, "model", None)
    if model is not None:
        summary["variable_counts"] = count_integer_variables(model)

    return summary


# ---------------------------------------------------------------------------
# Gurobi log filtering
#
# Gurobi writes its log to the C-level stdout via a callback, so Python-level
# redirect_stdout cannot intercept it.  Instead, we set solver.options["LogFile"]
# to a temp file and tail that file live with a background thread, forwarding
# only the lines we care about (solve progress) and dropping bookkeeping noise
# (model statistics, presolve, cutting-plane summary, etc.).
#
# Usage -- simple single solve (e.g. PoAOptimization):
#
#     with gurobi_log_filter(solver):
#         solver.solve(model, tee=False)
#
# Usage -- persistent solver across an eta sweep (e.g. DRO_PoAOptimization).
# Pass the same log_path and pos_holder on every call so the tail continues
# from where it left off rather than replaying the whole file:
#
#     if not hasattr(self, "_gurobi_log_pos"):
#         self._gurobi_log_path = None
#         self._gurobi_log_pos = {"pos": 0}
#
#     with gurobi_log_filter(solver, self._gurobi_log_path, self._gurobi_log_pos) as path:
#         self._gurobi_log_path = path
#         solver.solve(tee=False, ...)
# ---------------------------------------------------------------------------

# Lines whose stripped prefix matches any of these are suppressed.
_GUROBI_LOG_DROP_PREFIXES = (
    # Parameter / model setup
    "Set parameter",
    "Non-default parameters:",
    "IntFeasTol",
    "LogFile",
    "Optimize a model with",
    "Model fingerprint:",
    "Model has",
    "Variable types:",
    "Coefficient statistics:",
    "Matrix range",
    "Objective range",
    "Bounds range",
    "RHS range",
    "Warning: Model contains large matrix",
    "Consider reformulating",
    "to avoid numerical issues",
    # Presolve / concurrent LP
    "Presolve removed",
    "Presolve time:",
    "Presolved:",
    "Root relaxation presolved:",
    "Deterministic concurrent LP optimizer",
    "Concurrent LP optimizer",
    "Showing primal log only",
    "Showing dual log only",
    "LogToConsole",
    "Concurrent spin time:",
    "Solved with dual simplex",
    "Solved with barrier",
    "Solved with primal simplex",
    "Extra simplex iterations after uncrush",
    # Post-solve statistics
    "Cutting planes:",
    "Learned:",
    "Gomory:",
    "Cover:",
    "Implied bound:",
    "Clique:",
    "MIR:",
    "Flow cover:",
    "GUB cover:",
    "Zero half:",
    "RLT:",
    "Relax-and-lift:",
    "BQP:",
    "StrongCG:",
    "Mod-K:",
    "Network:",
    "Explored",
    "Thread count was",
    "Solution count",
)

# B&B node-table rows end with e.g. "1649%     -    3s"; this pattern does not
# appear anywhere else in the Gurobi log.
_NODE_DATA_ROW_RE = re.compile(r"\d+%\s+[\d\-]+\s+\d+s\s*$")

# Temp log files registered here are removed at interpreter exit (Windows keeps
# them open while Gurobi is running, so we cannot delete mid-solve).
_TEMP_GUROBI_LOG_PATHS: set[str] = set()


def _cleanup_temp_gurobi_logs() -> None:
    for path in list(_TEMP_GUROBI_LOG_PATHS):
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_gurobi_logs)


def _should_drop_gurobi_line(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in _GUROBI_LOG_DROP_PREFIXES)


def _is_node_data_row(line: str) -> bool:
    return bool(_NODE_DATA_ROW_RE.search(line))


def _is_incumbent_row(line: str) -> bool:
    return bool(line) and not line[0].isspace() and _is_node_data_row(line)


def _tail_gurobi_log(
    path: str,
    line_filter: Callable[[str], bool],
    pos_holder: dict,
    stop_event: threading.Event,
    poll_interval: float = 0.05,
    node_print_every: int = 10,
) -> None:
    """Tail a growing Gurobi log file, forwarding kept lines to stdout live.

    Reads appended bytes on a short poll for near-real-time output.  Only complete
    lines are emitted; a partial trailing line is buffered until its newline arrives
    so the prefix filter never sees a fragment.  ``pos_holder['pos']`` carries the
    read offset across solves that share one log file.

    Blank lines are collapsed to at most one consecutive blank so the output stays
    compact.  B&B node-table rows are thinned to every ``node_print_every`` rows;
    incumbent/heuristic rows (starting with a flag letter like H or *) are always
    printed so every new solution is visible.
    """
    buffer = ""
    consecutive_blank = 0
    node_row_count = 0

    def _drop(line: str) -> bool:
        try:
            return line_filter(line)
        except Exception:
            return False

    def _emit(line: str, out_parts: list) -> None:
        nonlocal consecutive_blank
        if not line.strip():
            consecutive_blank += 1
            if consecutive_blank <= 1:
                out_parts.append(line)
        else:
            consecutive_blank = 0
            out_parts.append(line)

    def process(final: bool = False) -> None:
        nonlocal buffer, node_row_count
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(pos_holder["pos"])
                    buffer += handle.read()
                    pos_holder["pos"] = handle.tell()
        except OSError:
            return

        out_parts: list[str] = []
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            full = line + "\n"
            if _drop(full):
                continue
            if _is_node_data_row(full):
                node_row_count += 1
                if (
                    node_row_count == 1
                    or node_row_count % node_print_every == 0
                    or _is_incumbent_row(full)
                ):
                    _emit(full, out_parts)
            else:
                _emit(full, out_parts)

        if final and buffer:
            if not _drop(buffer):
                if _is_node_data_row(buffer):
                    node_row_count += 1
                    if (
                        node_row_count == 1
                        or node_row_count % node_print_every == 0
                        or _is_incumbent_row(buffer)
                    ):
                        _emit(buffer, out_parts)
                else:
                    _emit(buffer, out_parts)
            buffer = ""

        if out_parts:
            sys.stdout.write("".join(out_parts))
            sys.stdout.flush()

    while not stop_event.is_set():
        process()
        time.sleep(poll_interval)
    process(final=True)


@contextmanager
def gurobi_log_filter(
    solver: Any,
    log_path: Optional[str] = None,
    pos_holder: Optional[dict] = None,
    node_print_every: int = 10,
) -> Iterator[str]:
    """Context manager: route the Gurobi log through a line filter and tail live.

    Sets ``solver.options["LogFile"]`` to ``log_path`` (or a fresh temp file when
    ``log_path`` is None), starts a background tail thread that forwards kept lines
    to stdout, and cleans up on exit.

    Yields the log file path used for this solve so callers can persist it across
    calls (persistent-solver eta sweep).  Pass the same ``log_path`` and
    ``pos_holder`` dict on every subsequent call to continue tailing from where the
    previous solve left off rather than replaying the full file.
    """
    if log_path is None:
        handle_fd, log_path = tempfile.mkstemp(prefix="gurobi_", suffix=".log")
        os.close(handle_fd)
        _TEMP_GUROBI_LOG_PATHS.add(log_path)
    if pos_holder is None:
        pos_holder = {"pos": 0}

    solver.options["LogFile"] = log_path

    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_gurobi_log,
        args=(log_path, _should_drop_gurobi_line, pos_holder, stop_event),
        kwargs={"node_print_every": node_print_every},
        daemon=True,
    )
    tail_thread.start()
    try:
        yield log_path
    finally:
        stop_event.set()
        tail_thread.join()


def _optional_log_number(token: str) -> Optional[float]:
    """Parse a Gurobi log column; '-' (value not yet available) becomes None."""
    if token in ("-", ""):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_gurobi_node_log(log_text: str) -> list:
    """Parse Gurobi MIP node-log lines into a bound-progression time series.

    Returns one record per progress line, in log order, with keys ``time_s``,
    ``nodes_explored``, ``nodes_unexplored``, ``incumbent``, ``best_bound`` and
    ``gap_percent`` (None where Gurobi printed '-'). Node-log columns are
    position-stable from the right -- ``Incumbent BestBd Gap It/Node Time`` --
    which also holds for the shorter ``H``/``*`` incumbent lines, so fields are
    indexed from the line end.
    """
    progression = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if len(line) < 2 or not line.endswith("s"):
            continue
        if line[0] not in "0123456789*H":
            continue
        tokens = line.split()
        if tokens[0] in {"H", "*"}:
            tokens = tokens[1:]
        if len(tokens) < 7 or not tokens[-1][:-1].isdigit():
            continue
        nodes_token = tokens[0].lstrip("*H")
        if not nodes_token.isdigit() or not tokens[1].isdigit():
            continue
        gap_token = tokens[-3]
        if gap_token != "-" and not gap_token.endswith("%"):
            continue
        progression.append(
            {
                "time_s": float(tokens[-1][:-1]),
                "nodes_explored": int(nodes_token),
                "nodes_unexplored": int(tokens[1]),
                "incumbent": _optional_log_number(tokens[-5]),
                "best_bound": _optional_log_number(tokens[-4]),
                "gap_percent": _optional_log_number(gap_token.rstrip("%")),
            }
        )
    return progression
