"""Compute support-set-aware ramp KKT dual bounds for bidding-block PoA models.

The analytical mode is support-set valid to the extent that the configured ramp
dual upper bound is valid. The stress OBBT mode fixes demand, capacities, and
bids for representative support-set trajectories before solving LPs, so its
results are diagnostic unless those trajectories exhaust the support set.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Reals,
    Set,
    SolverFactory,
    Var,
    maximize,
    minimize,
    value,
)
from pyomo.opt import TerminationCondition

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_optimization_bidding_blocks import PoAOptimizationBiddingBlocks


DEFAULT_BID_LB = 0.0
DEFAULT_BID_UB = 60.0
RHO_UB_DEFAULT = 1.0
VALIDITY_NOTE = (
    "Bounds are support-set valid only to the extent that the analytical rho upper "
    "bound is valid. Stress-test OBBT bounds are diagnostic unless stress "
    "trajectories exhaust the support set."
)


logger = logging.getLogger("support_set_ramp_dual_bounds")


def load_support_set_ramp_dual_bounds(path: str | Path) -> dict[str, Any]:
    """Load support-set ramp dual bounds from JSON."""
    bound_path = Path(path)
    if not bound_path.exists():
        raise FileNotFoundError(f"Support-set ramp dual bounds not found: {bound_path}")
    with bound_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid support-set ramp dual bounds format: {bound_path}")
    return payload


def _finite(value_raw: Any, name: str) -> float:
    value_float = float(value_raw)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} is not finite: {value_raw}")
    return value_float


def _as_profile(value_raw: Any, horizon: int, name: str) -> list[float]:
    if isinstance(value_raw, str):
        try:
            parsed = ast.literal_eval(value_raw)
            if isinstance(parsed, (list, tuple)):
                value_raw = parsed
        except Exception:
            pass
    if isinstance(value_raw, (list, tuple, np.ndarray)):
        profile = [_finite(v, name) for v in value_raw]
    else:
        profile = [_finite(value_raw, name)] * horizon
    if len(profile) < horizon:
        raise ValueError(f"{name} must have at least {horizon} entries")
    return profile[:horizon]


def _reference_case_name(reference_case: str) -> str:
    path = Path(reference_case)
    if path.exists() and path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as file_handle:
            raw = yaml.safe_load(file_handle) or {}
        if isinstance(raw, dict) and raw:
            return str(next(iter(raw.keys())))
    return path.stem if path.suffix.lower() in {".yaml", ".yml"} else reference_case


def _load_optimizer_context(args: argparse.Namespace) -> PoAOptimizationBiddingBlocks:
    case_name = _reference_case_name(args.reference_case)
    scenario_manager = ScenarioManager(case_name)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=args.regime_set,
        seed=args.seed,
    )
    support_cfg = PoAOptimizationBiddingBlocks.load_support_set_config(
        config_path=args.support_set_config,
        config_name=args.support_set_name,
    )
    return PoAOptimizationBiddingBlocks(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=args.num_time_steps,
        support_set_config=support_cfg,
        nn_model_dir=None,
        nn_policy_generators=[],
        use_dual_bound_calibration=False,
        reference_case=case_name,
    )


def _empty_indexed(ctx: PoAOptimizationBiddingBlocks) -> dict[str, Any]:
    return {
        "lambda_lb": {str(t): None for t in range(ctx.num_time_steps)},
        "lambda_ub": {str(t): None for t in range(ctx.num_time_steps)},
        "mu_max_ub": {
            ctx.physical_generator_names[i]: {
                ctx.block_names[ctx.local_to_global_block[(i, b)]]: {
                    str(t): None for t in range(ctx.num_time_steps)
                }
                for b in ctx.local_blocks_by_generator[i]
            }
            for i in range(ctx.num_physical_generators)
        },
        "mu_min_ub": {
            ctx.physical_generator_names[i]: {
                ctx.block_names[ctx.local_to_global_block[(i, b)]]: {
                    str(t): None for t in range(ctx.num_time_steps)
                }
                for b in ctx.local_blocks_by_generator[i]
            }
            for i in range(ctx.num_physical_generators)
        },
        "rho_up_ub": {
            ctx.physical_generator_names[i]: {str(t): None for t in range(ctx.num_time_steps)}
            for i in range(ctx.num_physical_generators)
        },
        "rho_down_ub": {
            ctx.physical_generator_names[i]: {str(t): None for t in range(ctx.num_time_steps)}
            for i in range(ctx.num_physical_generators)
        },
    }


def _global_bounds(indexed: dict[str, Any]) -> dict[str, float]:
    def walk_numbers(obj: Any) -> list[float]:
        if isinstance(obj, dict):
            values: list[float] = []
            for child in obj.values():
                values.extend(walk_numbers(child))
            return values
        if obj is None:
            return []
        return [float(obj)]

    lambda_lb_values = walk_numbers(indexed["lambda_lb"])
    lambda_ub_values = walk_numbers(indexed["lambda_ub"])
    return {
        "lambda_lb": min(lambda_lb_values),
        "lambda_ub": max(lambda_ub_values),
        "mu_max_ub": max(walk_numbers(indexed["mu_max_ub"])),
        "mu_min_ub": max(walk_numbers(indexed["mu_min_ub"])),
        "rho_up_ub": max(walk_numbers(indexed["rho_up_ub"])),
        "rho_down_ub": max(walk_numbers(indexed["rho_down_ub"])),
    }


def _bid_bounds(
    ctx: PoAOptimizationBiddingBlocks,
    default_lb: float,
    default_ub: float,
    bid_bound_source: str = "global",
) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int, int], float]]:
    bid_lb: dict[tuple[int, int, int], float] = {}
    bid_ub: dict[tuple[int, int, int], float] = {}
    cfg = ctx.support_set_config or {}
    raw_lb = cfg.get("bid_lb", cfg.get("alpha_lb", default_lb))
    raw_ub = cfg.get("bid_ub", cfg.get("alpha_ub", default_ub))

    def lookup(raw: Any, i: int, b: int, t: int, default: float) -> float:
        if not isinstance(raw, dict):
            return float(raw)
        gen_name = ctx.physical_generator_names[i]
        block_name = ctx.block_names[ctx.local_to_global_block[(i, b)]]
        gen_payload = None
        for gen_key in (gen_name, str(i), i):
            if gen_key in raw:
                gen_payload = raw[gen_key]
                break
        if gen_payload is None:
            return default
        if not isinstance(gen_payload, dict):
            return float(gen_payload)
        block_payload = None
        for block_key in (block_name, str(b), b):
            if block_key in gen_payload:
                block_payload = gen_payload[block_key]
                break
        if block_payload is None:
            return default
        if isinstance(block_payload, (list, tuple)):
            return float(block_payload[t])
        if isinstance(block_payload, dict):
            return float(block_payload.get(str(t), block_payload.get(t, default)))
        return float(block_payload)

    for i, b in ctx.generator_block_pairs:
        global_block = ctx.local_to_global_block[(i, b)]
        for t in range(ctx.num_time_steps):
            if bid_bound_source == "true_cost":
                lb = float(ctx.block_cost_vector[global_block])
                ub = float(ctx.block_cost_vector[global_block])
            else:
                lb = lookup(raw_lb, i, b, t, default_lb)
                ub = lookup(raw_ub, i, b, t, default_ub)
            if lb > ub:
                raise ValueError(f"bid_lb exceeds bid_ub for {(i, b, t)}: {lb} > {ub}")
            bid_lb[(i, b, t)] = lb
            bid_ub[(i, b, t)] = ub
    return bid_lb, bid_ub


def compute_analytical_bounds(
    ctx: PoAOptimizationBiddingBlocks,
    bid_lb: dict[tuple[int, int, int], float],
    bid_ub: dict[tuple[int, int, int], float],
    rho_ub_default: float,
    safety_factor: float,
    epsilon: float,
    nonnegative_prices: bool = True,
) -> dict[str, Any]:
    indexed = _empty_indexed(ctx)
    for t in range(ctx.num_time_steps):
        lb = min(bid_lb[(i, b, t)] for i, b in ctx.generator_block_pairs)
        ub = max(bid_ub[(i, b, t)] for i, b in ctx.generator_block_pairs)
        if nonnegative_prices:
            lb = max(0.0, lb)
        indexed["lambda_lb"][str(t)] = lb
        indexed["lambda_ub"][str(t)] = ub

    for i in range(ctx.num_physical_generators):
        gen_name = ctx.physical_generator_names[i]
        for t in range(ctx.num_time_steps):
            rho_bound = rho_ub_default * safety_factor + epsilon
            indexed["rho_up_ub"][gen_name][str(t)] = rho_bound
            indexed["rho_down_ub"][gen_name][str(t)] = rho_bound

    for i, b in ctx.generator_block_pairs:
        gen_name = ctx.physical_generator_names[i]
        block_name = ctx.block_names[ctx.local_to_global_block[(i, b)]]
        for t in range(ctx.num_time_steps):
            ramp_contribution = 2.0 * rho_ub_default
            if t + 1 < ctx.num_time_steps:
                ramp_contribution += 2.0 * rho_ub_default
            lambda_lb_t = float(indexed["lambda_lb"][str(t)])
            lambda_ub_t = float(indexed["lambda_ub"][str(t)])
            raw_mu_max = max(0.0, lambda_ub_t - bid_lb[(i, b, t)] + ramp_contribution)
            raw_mu_min = max(0.0, bid_ub[(i, b, t)] - lambda_lb_t + ramp_contribution)
            indexed["mu_max_ub"][gen_name][block_name][str(t)] = (
                raw_mu_max * safety_factor + epsilon
            )
            indexed["mu_min_ub"][gen_name][block_name][str(t)] = (
                raw_mu_min * safety_factor + epsilon
            )

    return {
        "indexed_bounds": indexed,
        "global_bounds": _global_bounds(indexed),
    }


def _support_demand_profiles(ctx: PoAOptimizationBiddingBlocks) -> dict[str, list[float]]:
    T = ctx.num_time_steps
    ref = list(ctx.support_demand_reference)
    lo = [ctx.support_demand_min] * T
    hi = [ctx.support_demand_max] * T
    profiles = {
        "high_demand": hi,
        "low_demand": lo,
        "reference_demand": ref,
    }
    for peak_t in range(T):
        p = list(ref)
        p[peak_t] = ctx.support_demand_max
        profiles[f"peak_demand_at_{peak_t}"] = p
    if T > 1:
        profiles["maximum_demand_ramp_up"] = [
            min(ctx.support_demand_max, ctx.support_demand_min + t * ctx.support_demand_ramp)
            for t in range(T)
        ]
        profiles["maximum_demand_ramp_down"] = [
            max(ctx.support_demand_min, ctx.support_demand_max - t * ctx.support_demand_ramp)
            for t in range(T)
        ]
    return profiles


def _wind_total_profile(ctx: PoAOptimizationBiddingBlocks, i: int, kind: str) -> list[float]:
    T = ctx.num_time_steps
    if i not in ctx.wind_physical_generator_ids:
        return [ctx.static_physical_capacity[i]] * T
    if kind == "low":
        return [ctx.support_wind_min[i]] * T
    if kind == "high":
        return [ctx.support_wind_max[i]] * T
    if kind == "ramp_down":
        return [
            max(ctx.support_wind_min[i], ctx.support_wind_max[i] - t * ctx.support_wind_ramp)
            for t in range(T)
        ]
    if kind == "ramp_up":
        return [
            min(ctx.support_wind_max[i], ctx.support_wind_min[i] + t * ctx.support_wind_ramp)
            for t in range(T)
        ]
    return list(ctx.support_wind_reference[i])


def _split_physical_capacity(ctx: PoAOptimizationBiddingBlocks, i: int, total: float) -> dict[int, float]:
    local_blocks = ctx.local_blocks_by_generator[i]
    static_total = sum(ctx.static_block_capacity[ctx.local_to_global_block[(i, b)]] for b in local_blocks)
    if static_total <= 1e-12:
        return {b: 0.0 for b in local_blocks}
    return {
        b: total * ctx.static_block_capacity[ctx.local_to_global_block[(i, b)]] / static_total
        for b in local_blocks
    }


def _capacity_trajectory(
    ctx: PoAOptimizationBiddingBlocks,
    wind_kind: str,
    low_generator: int | None = None,
    low_time: int | None = None,
) -> dict[tuple[int, int, int], float]:
    Pmax: dict[tuple[int, int, int], float] = {}
    for i in range(ctx.num_physical_generators):
        kind = wind_kind
        if low_generator is not None and i != low_generator and i in ctx.wind_physical_generator_ids:
            kind = "high"
        total_profile = _wind_total_profile(ctx, i, kind)
        if low_time is not None and i in ctx.wind_physical_generator_ids:
            total_profile = list(total_profile)
            total_profile[low_time] = ctx.support_wind_min[i]
        for t, total in enumerate(total_profile):
            for b, block_cap in _split_physical_capacity(ctx, i, total).items():
                Pmax[(i, b, t)] = block_cap
    return Pmax


def generate_stress_trajectories(
    ctx: PoAOptimizationBiddingBlocks,
    bid_lb: dict[tuple[int, int, int], float],
    bid_ub: dict[tuple[int, int, int], float],
    max_stress_trajectories: int,
) -> list[dict[str, Any]]:
    demand_profiles = _support_demand_profiles(ctx)
    alpha_cases = {
        "low_bids": bid_lb,
        "high_bids": bid_ub,
    }
    templates: list[tuple[str, list[float], dict[tuple[int, int, int], float]]] = [
        ("high_demand_low_wind", demand_profiles["high_demand"], _capacity_trajectory(ctx, "low")),
        ("low_demand_high_wind", demand_profiles["low_demand"], _capacity_trajectory(ctx, "high")),
        ("maximum_scarcity", demand_profiles["high_demand"], _capacity_trajectory(ctx, "low")),
        ("maximum_demand_ramp_up", demand_profiles.get("maximum_demand_ramp_up", demand_profiles["high_demand"]), _capacity_trajectory(ctx, "low")),
        ("maximum_demand_ramp_down", demand_profiles.get("maximum_demand_ramp_down", demand_profiles["high_demand"]), _capacity_trajectory(ctx, "low")),
        ("minimum_wind_ramp_down", demand_profiles["high_demand"], _capacity_trajectory(ctx, "ramp_down")),
        ("maximum_wind_ramp_up", demand_profiles["high_demand"], _capacity_trajectory(ctx, "ramp_up")),
    ]
    for i in range(ctx.num_physical_generators):
        templates.append(
            (
                f"single_generator_scarcity_{ctx.physical_generator_names[i]}",
                demand_profiles["high_demand"],
                _capacity_trajectory(ctx, "high", low_generator=i),
            )
        )
    for t in range(ctx.num_time_steps):
        templates.append((f"peak_demand_at_{t}", demand_profiles[f"peak_demand_at_{t}"], _capacity_trajectory(ctx, "low")))
        templates.append((f"low_capacity_at_{t}", demand_profiles["high_demand"], _capacity_trajectory(ctx, "high", low_time=t)))

    trajectories: list[dict[str, Any]] = []
    for name, demand, Pmax in templates:
        for alpha_name, alpha in alpha_cases.items():
            trajectories.append(
                {
                    "name": f"{name}__{alpha_name}",
                    "D": list(demand),
                    "Pmax": dict(Pmax),
                    "alpha": dict(alpha),
                }
            )
            if len(trajectories) >= max_stress_trajectories:
                return trajectories
    return trajectories


def _build_fixed_rhs_kkt_model(
    ctx: PoAOptimizationBiddingBlocks,
    trajectory: dict[str, Any],
    lambda_bound: float,
    analytical: dict[str, Any],
) -> ConcreteModel:
    m = ConcreteModel()
    m.T = Set(initialize=range(ctx.num_time_steps))
    m.Tplus = Set(initialize=range(ctx.num_time_steps + 1))
    m.Tminus = Set(initialize=range(1, ctx.num_time_steps))
    m.G = Set(initialize=range(ctx.num_physical_generators))
    m.GB = Set(dimen=2, initialize=ctx.generator_block_pairs)

    m.P = Var(m.GB, m.T, domain=NonNegativeReals)
    m.lambda_var = Var(m.T, domain=Reals, bounds=(-lambda_bound, lambda_bound))
    m.mu_max = Var(m.GB, m.T, domain=NonNegativeReals)
    m.mu_min = Var(m.GB, m.T, domain=NonNegativeReals)
    m.rho_up = Var(m.G, m.Tplus, domain=NonNegativeReals)
    m.rho_down = Var(m.G, m.Tplus, domain=NonNegativeReals)

    for i in m.G:
        m.rho_up[i, ctx.num_time_steps].fix(0.0)
        m.rho_down[i, ctx.num_time_steps].fix(0.0)

    D = trajectory["D"]
    Pmax = trajectory["Pmax"]
    alpha = trajectory["alpha"]

    m.balance = Constraint(m.T, rule=lambda mm, t: D[int(t)] - sum(mm.P[i, b, t] for i, b in mm.GB) == 0)
    m.upper = Constraint(m.GB, m.T, rule=lambda mm, i, b, t: mm.P[i, b, t] - Pmax[(int(i), int(b), int(t))] <= 0)

    def ramp_up_rule(mm, i, t):
        return (
            sum(mm.P[i, b, t] for b in ctx.local_blocks_by_generator[int(i)])
            - sum(mm.P[i, b, t - 1] for b in ctx.local_blocks_by_generator[int(i)])
            - ctx.ramp_vector_up[int(i)]
            <= 0
        )

    def ramp_down_rule(mm, i, t):
        return (
            -sum(mm.P[i, b, t] for b in ctx.local_blocks_by_generator[int(i)])
            + sum(mm.P[i, b, t - 1] for b in ctx.local_blocks_by_generator[int(i)])
            - ctx.ramp_vector_down[int(i)]
            <= 0
        )

    m.ramp_up = Constraint(m.G, m.Tminus, rule=ramp_up_rule)
    m.ramp_down = Constraint(m.G, m.Tminus, rule=ramp_down_rule)
    m.ramp_up_initial = Constraint(
        m.G,
        rule=lambda mm, i: (
            sum(mm.P[i, b, 0] for b in ctx.local_blocks_by_generator[int(i)])
            - ctx.p_init[int(i)]
            - ctx.ramp_vector_up[int(i)]
            <= 0
        ),
    )
    m.ramp_down_initial = Constraint(
        m.G,
        rule=lambda mm, i: (
            -sum(mm.P[i, b, 0] for b in ctx.local_blocks_by_generator[int(i)])
            + ctx.p_init[int(i)]
            - ctx.ramp_vector_down[int(i)]
            <= 0
        ),
    )

    def stationarity_rule(mm, i, b, t):
        return (
            alpha[(int(i), int(b), int(t))]
            - mm.lambda_var[t]
            + mm.mu_max[i, b, t]
            - mm.mu_min[i, b, t]
            + mm.rho_up[i, t]
            - mm.rho_up[i, t + 1]
            - mm.rho_down[i, t]
            + mm.rho_down[i, t + 1]
            == 0
        )

    m.stationarity = Constraint(m.GB, m.T, rule=stationarity_rule)

    primal_obj = sum(alpha[(i, b, t)] * m.P[i, b, t] for i, b in ctx.generator_block_pairs for t in range(ctx.num_time_steps))
    dual_obj = (
        sum(m.lambda_var[t] * D[t] for t in range(ctx.num_time_steps))
        - sum(m.mu_max[i, b, t] * Pmax[(i, b, t)] for i, b in ctx.generator_block_pairs for t in range(ctx.num_time_steps))
        - sum(m.rho_up[i, t] * ctx.ramp_vector_up[i] for i in range(ctx.num_physical_generators) for t in range(1, ctx.num_time_steps))
        - sum(m.rho_down[i, t] * ctx.ramp_vector_down[i] for i in range(ctx.num_physical_generators) for t in range(1, ctx.num_time_steps))
        - sum(m.rho_up[i, 0] * (ctx.p_init[i] + ctx.ramp_vector_up[i]) for i in range(ctx.num_physical_generators))
        + sum(m.rho_down[i, 0] * (ctx.p_init[i] - ctx.ramp_vector_down[i]) for i in range(ctx.num_physical_generators))
    )
    m.strong_duality = Constraint(expr=primal_obj == dual_obj)

    # Analytical bounds keep failed/unbounded stress LPs numerically contained.
    indexed = analytical["indexed_bounds"]
    for t in range(ctx.num_time_steps):
        m.lambda_var[t].setlb(float(indexed["lambda_lb"][str(t)]) * -2.0 - abs(lambda_bound))
        m.lambda_var[t].setub(float(indexed["lambda_ub"][str(t)]) * 2.0 + abs(lambda_bound))
    for i, b in ctx.generator_block_pairs:
        gen_name = ctx.physical_generator_names[i]
        block_name = ctx.block_names[ctx.local_to_global_block[(i, b)]]
        for t in range(ctx.num_time_steps):
            m.mu_max[i, b, t].setub(max(1.0, 2.0 * float(indexed["mu_max_ub"][gen_name][block_name][str(t)])))
            m.mu_min[i, b, t].setub(max(1.0, 2.0 * float(indexed["mu_min_ub"][gen_name][block_name][str(t)])))
    for i in range(ctx.num_physical_generators):
        gen_name = ctx.physical_generator_names[i]
        for t in range(ctx.num_time_steps):
            m.rho_up[i, t].setub(max(1.0, 2.0 * float(indexed["rho_up_ub"][gen_name][str(t)])))
            m.rho_down[i, t].setub(max(1.0, 2.0 * float(indexed["rho_down_ub"][gen_name][str(t)])))

    return m


def _trajectory_capacity_shortfall(
    ctx: PoAOptimizationBiddingBlocks,
    trajectory: dict[str, Any],
    tolerance: float = 1e-8,
) -> list[dict[str, float]]:
    """Return periods where fixed demand exceeds fixed available capacity."""
    shortfalls: list[dict[str, float]] = []
    demand = trajectory["D"]
    pmax = trajectory["Pmax"]
    for t in range(ctx.num_time_steps):
        total_capacity = sum(
            float(pmax[(i, b, t)])
            for i, b in ctx.generator_block_pairs
        )
        demand_t = float(demand[t])
        if demand_t > total_capacity + tolerance:
            shortfalls.append(
                {
                    "time": int(t),
                    "demand": demand_t,
                    "available_capacity": total_capacity,
                    "shortfall": demand_t - total_capacity,
                }
            )
    return shortfalls


def _trajectory_manifest_entry(
    ctx: PoAOptimizationBiddingBlocks,
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    shortfall = _trajectory_capacity_shortfall(ctx, trajectory)
    demand = trajectory["D"]
    pmax = trajectory["Pmax"]
    total_capacity = [
        sum(float(pmax[(i, b, t)]) for i, b in ctx.generator_block_pairs)
        for t in range(ctx.num_time_steps)
    ]
    return {
        "name": str(trajectory["name"]),
        "demand_min": float(min(demand)),
        "demand_max": float(max(demand)),
        "available_capacity_min": float(min(total_capacity)),
        "available_capacity_max": float(max(total_capacity)),
        "capacity_feasible": not bool(shortfall),
        "num_shortfall_periods": len(shortfall),
        "max_shortfall": float(max((item["shortfall"] for item in shortfall), default=0.0)),
    }


def run_stress_obbt(
    ctx: PoAOptimizationBiddingBlocks,
    trajectories: list[dict[str, Any]],
    analytical: dict[str, Any],
    time_limit: float,
    threads: int,
    progress_every: int = 25,
) -> dict[str, Any]:
    summary_indexed = _empty_indexed(ctx)
    for t in range(ctx.num_time_steps):
        summary_indexed["lambda_lb"][str(t)] = math.inf
        summary_indexed["lambda_ub"][str(t)] = -math.inf
    for i, b in ctx.generator_block_pairs:
        gen_name = ctx.physical_generator_names[i]
        block_name = ctx.block_names[ctx.local_to_global_block[(i, b)]]
        for t in range(ctx.num_time_steps):
            summary_indexed["mu_max_ub"][gen_name][block_name][str(t)] = 0.0
            summary_indexed["mu_min_ub"][gen_name][block_name][str(t)] = 0.0
    for i in range(ctx.num_physical_generators):
        gen_name = ctx.physical_generator_names[i]
        for t in range(ctx.num_time_steps):
            summary_indexed["rho_up_ub"][gen_name][str(t)] = 0.0
            summary_indexed["rho_down_ub"][gen_name][str(t)] = 0.0

    failed: list[dict[str, Any]] = []
    trajectory_bounds: dict[str, Any] = {}
    solver = SolverFactory("gurobi")
    if not solver.available(False):
        return {
            "enabled": True,
            "num_trajectories": len(trajectories),
            "num_targets_per_trajectory": 0,
            "planned_optimization_programs": 0,
            "solved_optimization_programs": 0,
            "failed_optimization_programs": 0,
            "skipped_trajectories": 0,
            "skipped_optimization_programs": 0,
            "trajectory_bounds": {},
            "summary": {"available": False},
            "failed_solves": [{"trajectory": "*", "target": "*", "reason": "gurobi unavailable"}],
        }
    solver.options.update(
        {
            "Threads": int(threads),
            "NumericFocus": 1,
            "FeasibilityTol": 1e-7,
            "OptimalityTol": 1e-7,
            "TimeLimit": float(time_limit),
        }
    )

    targets: list[tuple[str, tuple[int, ...], str]] = []
    for i, b in ctx.generator_block_pairs:
        for t in range(ctx.num_time_steps):
            targets.append(("mu_max_ub", (i, b, t), "max"))
            targets.append(("mu_min_ub", (i, b, t), "max"))
    for i in range(ctx.num_physical_generators):
        for t in range(ctx.num_time_steps):
            targets.append(("rho_up_ub", (i, t), "max"))
            targets.append(("rho_down_ub", (i, t), "max"))
    for t in range(ctx.num_time_steps):
        targets.append(("lambda_lb", (t,), "min"))
        targets.append(("lambda_ub", (t,), "max"))

    total_planned_programs = len(trajectories) * len(targets)
    trajectory_manifest = [
        _trajectory_manifest_entry(ctx, trajectory)
        for trajectory in trajectories
    ]
    solved_programs = 0
    failed_programs = 0
    skipped_trajectories = 0
    skipped_programs = 0
    logger.info(
        "Stress OBBT plan: %s trajectories x %s targets = %s LPs",
        len(trajectories),
        len(targets),
        total_planned_programs,
    )
    logger.info(
        "Target breakdown: mu_max=%s, mu_min=%s, rho_up=%s, rho_down=%s, lambda=%s",
        len(ctx.generator_block_pairs) * ctx.num_time_steps,
        len(ctx.generator_block_pairs) * ctx.num_time_steps,
        ctx.num_physical_generators * ctx.num_time_steps,
        ctx.num_physical_generators * ctx.num_time_steps,
        2 * ctx.num_time_steps,
    )
    for entry in trajectory_manifest:
        logger.info(
            "Trajectory '%s': demand=[%.6g, %.6g], capacity=[%.6g, %.6g], feasible=%s, shortfall_periods=%s, max_shortfall=%.6g",
            entry["name"],
            entry["demand_min"],
            entry["demand_max"],
            entry["available_capacity_min"],
            entry["available_capacity_max"],
            entry["capacity_feasible"],
            entry["num_shortfall_periods"],
            entry["max_shortfall"],
        )

    for trajectory in trajectories:
        tname = str(trajectory["name"])
        trajectory_bounds[tname] = {}
        logger.info("Starting stress trajectory '%s'", tname)
        capacity_shortfall = _trajectory_capacity_shortfall(ctx, trajectory)
        if capacity_shortfall:
            skipped_trajectories += 1
            skipped_programs += len(targets)
            logger.info(
                "Skipping trajectory '%s': capacity infeasible; skips %s LPs",
                tname,
                len(targets),
            )
            failed.append(
                {
                    "trajectory": tname,
                    "target": "*",
                    "termination_condition": "skipped_capacity_infeasible",
                    "reason": "Fixed stress trajectory has demand above available capacity.",
                    "capacity_shortfall": capacity_shortfall,
                }
            )
            continue
        for bound_name, indices, sense in targets:
            attempted_program = solved_programs + failed_programs + 1
            model = _build_fixed_rhs_kkt_model(
                ctx,
                trajectory,
                lambda_bound=max(abs(analytical["global_bounds"]["lambda_lb"]), abs(analytical["global_bounds"]["lambda_ub"]), 1.0),
                analytical=analytical,
            )
            if bound_name == "mu_max_ub":
                expr = model.mu_max[indices]
            elif bound_name == "mu_min_ub":
                expr = model.mu_min[indices]
            elif bound_name == "rho_up_ub":
                expr = model.rho_up[indices]
            elif bound_name == "rho_down_ub":
                expr = model.rho_down[indices]
            else:
                expr = model.lambda_var[indices[0]]
            model.obbt_objective = Objective(expr=expr, sense=maximize if sense == "max" else minimize)
            try:
                results = solver.solve(model, tee=False, load_solutions=False)
            except Exception as exc:
                failed_programs += 1
                reason = str(exc)
                failed.append(
                    {
                        "trajectory": tname,
                        "target": bound_name,
                        "indices": list(indices),
                        "termination_condition": "solver_exception",
                        "reason": reason,
                    }
                )
                if "GurobiError" in reason or "license" in reason.lower() or "User name mismatch" in reason:
                    return {
                        "enabled": True,
                        "num_trajectories": len(trajectories),
                        "num_targets_per_trajectory": len(targets),
                        "planned_optimization_programs": total_planned_programs,
                        "trajectory_manifest": trajectory_manifest,
                        "solved_optimization_programs": solved_programs,
                        "failed_optimization_programs": failed_programs,
                        "skipped_trajectories": skipped_trajectories,
                        "skipped_optimization_programs": skipped_programs,
                        "trajectory_bounds": trajectory_bounds,
                        "summary": {
                            "indexed_bounds": copy.deepcopy(analytical["indexed_bounds"]),
                            "global_bounds": dict(analytical["global_bounds"]),
                        },
                        "failed_solves": failed,
                    }
                continue
            tc = results.solver.termination_condition
            if tc not in {TerminationCondition.optimal, TerminationCondition.feasible}:
                failed_programs += 1
                failed.append(
                    {
                        "trajectory": tname,
                        "target": bound_name,
                        "indices": list(indices),
                        "termination_condition": str(tc),
                    }
                )
                if progress_every > 0 and attempted_program % progress_every == 0:
                    logger.info(
                        "Stress OBBT progress: attempted %s/%s LPs; solved=%s failed=%s skipped=%s",
                        attempted_program + skipped_programs,
                        total_planned_programs,
                        solved_programs,
                        failed_programs,
                        skipped_programs,
                    )
                continue
            model.solutions.load_from(results)
            raw_value = value(expr, exception=False)
            if raw_value is None or not math.isfinite(float(raw_value)):
                failed_programs += 1
                failed.append(
                    {
                        "trajectory": tname,
                        "target": bound_name,
                        "indices": list(indices),
                        "termination_condition": "nonfinite_value",
                    }
                )
                continue
            solved_programs += 1
            solved_value = float(raw_value)
            trajectory_bounds[tname][f"{bound_name}:{','.join(map(str, indices))}"] = solved_value
            if bound_name in {"lambda_lb", "lambda_ub"}:
                key = str(indices[0])
                if bound_name == "lambda_lb":
                    summary_indexed["lambda_lb"][key] = min(summary_indexed["lambda_lb"][key], solved_value)
                else:
                    summary_indexed["lambda_ub"][key] = max(summary_indexed["lambda_ub"][key], solved_value)
            elif bound_name in {"mu_max_ub", "mu_min_ub"}:
                i, b, tt = indices
                gen_name = ctx.physical_generator_names[i]
                block_name = ctx.block_names[ctx.local_to_global_block[(i, b)]]
                summary_indexed[bound_name][gen_name][block_name][str(tt)] = max(
                    summary_indexed[bound_name][gen_name][block_name][str(tt)],
                    max(0.0, solved_value),
                )
            else:
                i, tt = indices
                gen_name = ctx.physical_generator_names[i]
                summary_indexed[bound_name][gen_name][str(tt)] = max(
                    summary_indexed[bound_name][gen_name][str(tt)],
                    max(0.0, solved_value),
                )
            if progress_every > 0 and (solved_programs + failed_programs + skipped_programs) % progress_every == 0:
                logger.info(
                    "Stress OBBT progress: completed %s/%s LPs; solved=%s failed=%s skipped=%s",
                    solved_programs + failed_programs + skipped_programs,
                    total_planned_programs,
                    solved_programs,
                    failed_programs,
                    skipped_programs,
                )

    analytical_indexed = analytical["indexed_bounds"]
    for t in range(ctx.num_time_steps):
        if not math.isfinite(float(summary_indexed["lambda_lb"][str(t)])):
            summary_indexed["lambda_lb"][str(t)] = analytical_indexed["lambda_lb"][str(t)]
        if not math.isfinite(float(summary_indexed["lambda_ub"][str(t)])):
            summary_indexed["lambda_ub"][str(t)] = analytical_indexed["lambda_ub"][str(t)]

    return {
        "enabled": True,
        "num_trajectories": len(trajectories),
        "num_targets_per_trajectory": len(targets),
        "planned_optimization_programs": total_planned_programs,
        "trajectory_manifest": trajectory_manifest,
        "solved_optimization_programs": solved_programs,
        "failed_optimization_programs": failed_programs,
        "skipped_trajectories": skipped_trajectories,
        "skipped_optimization_programs": skipped_programs,
        "trajectory_bounds": trajectory_bounds,
        "summary": {
            "indexed_bounds": summary_indexed,
            "global_bounds": _global_bounds(summary_indexed),
        },
        "failed_solves": failed,
    }


def _combine_bounds(
    analytical: dict[str, Any],
    stress: dict[str, Any],
    allow_stress_tightening: bool,
) -> dict[str, Any]:
    final_indexed = copy.deepcopy(analytical["indexed_bounds"])
    stress_indexed = stress.get("summary", {}).get("indexed_bounds") if stress else None
    if not stress_indexed:
        return {"indexed_bounds": final_indexed, "global_bounds": _global_bounds(final_indexed)}

    for t, analytical_lb in analytical["indexed_bounds"]["lambda_lb"].items():
        stress_lb = float(stress_indexed["lambda_lb"][t])
        final_indexed["lambda_lb"][t] = stress_lb if allow_stress_tightening else min(float(analytical_lb), stress_lb)
    for t, analytical_ub in analytical["indexed_bounds"]["lambda_ub"].items():
        stress_ub = float(stress_indexed["lambda_ub"][t])
        final_indexed["lambda_ub"][t] = stress_ub if allow_stress_tightening else max(float(analytical_ub), stress_ub)

    for key in ("mu_max_ub", "mu_min_ub"):
        for gen_name, blocks in analytical["indexed_bounds"][key].items():
            for block_name, times in blocks.items():
                for t, analytical_value in times.items():
                    stress_value = float(stress_indexed[key][gen_name][block_name][t])
                    final_indexed[key][gen_name][block_name][t] = (
                        stress_value
                        if allow_stress_tightening
                        else max(float(analytical_value), stress_value)
                    )
    for key in ("rho_up_ub", "rho_down_ub"):
        for gen_name, times in analytical["indexed_bounds"][key].items():
            for t, analytical_value in times.items():
                stress_value = float(stress_indexed[key][gen_name][t])
                final_indexed[key][gen_name][t] = (
                    stress_value
                    if allow_stress_tightening
                    else max(float(analytical_value), stress_value)
                )
    return {"indexed_bounds": final_indexed, "global_bounds": _global_bounds(final_indexed)}


def _all_values(obj: Any) -> list[float]:
    if isinstance(obj, dict):
        values: list[float] = []
        for child in obj.values():
            values.extend(_all_values(child))
        return values
    if obj is None:
        return []
    return [float(obj)]


def validate_payload(payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    indexed = payload["indexed_bounds"]
    for name, obj in indexed.items():
        for item in _all_values(obj):
            if not math.isfinite(item):
                raise ValueError(f"{name} contains NaN or infinite bound")
            if name in {"mu_max_ub", "mu_min_ub", "rho_up_ub", "rho_down_ub"} and item < -1e-12:
                raise ValueError(f"{name} contains negative upper bound: {item}")
            if abs(item) > 1e4:
                warnings.append(f"WARNING: {name} contains large bound {item:.6g} > 1e4")
    for t, lb in indexed["lambda_lb"].items():
        ub = indexed["lambda_ub"][t]
        if float(lb) > float(ub) + 1e-9:
            raise ValueError(f"lambda_lb[{t}] exceeds lambda_ub[{t}]: {lb} > {ub}")
    return warnings


def print_summary(payload: dict[str, Any]) -> None:
    indexed = payload["indexed_bounds"]
    stress = payload.get("stress_test_bounds", {})
    print("Summary")
    print(f"  max lambda_ub: {max(_all_values(indexed['lambda_ub'])):.6g}")
    print(f"  min lambda_lb: {min(_all_values(indexed['lambda_lb'])):.6g}")
    print(f"  max mu_max_ub: {max(_all_values(indexed['mu_max_ub'])):.6g}")
    print(f"  max mu_min_ub: {max(_all_values(indexed['mu_min_ub'])):.6g}")
    print(f"  max rho_up_ub: {max(_all_values(indexed['rho_up_ub'])):.6g}")
    print(f"  max rho_down_ub: {max(_all_values(indexed['rho_down_ub'])):.6g}")
    for key in ("lambda_lb", "lambda_ub", "mu_max_ub", "mu_min_ub", "rho_up_ub", "rho_down_ub"):
        values = np.asarray(_all_values(indexed[key]), dtype=float)
        quantiles = np.quantile(values, [0.50, 0.90, 0.95, 0.99])
        print(
            f"  {key}: q50={quantiles[0]:.6g}, q90={quantiles[1]:.6g}, "
            f"q95={quantiles[2]:.6g}, q99={quantiles[3]:.6g}, max={values.max():.6g}"
        )
    if stress.get("enabled"):
        print("Stress OBBT")
        print(f"  trajectories: {stress.get('num_trajectories', 0)}")
        print(f"  targets per trajectory: {stress.get('num_targets_per_trajectory', 0)}")
        print(f"  planned LPs: {stress.get('planned_optimization_programs', 0)}")
        print(f"  solved LPs: {stress.get('solved_optimization_programs', 0)}")
        print(f"  failed LPs: {stress.get('failed_optimization_programs', 0)}")
        print(f"  skipped trajectories: {stress.get('skipped_trajectories', 0)}")
        print(f"  skipped LPs: {stress.get('skipped_optimization_programs', 0)}")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    ctx = _load_optimizer_context(args)
    bid_lb, bid_ub = _bid_bounds(
        ctx,
        args.default_bid_lb,
        args.default_bid_ub,
        bid_bound_source=args.bid_bound_source,
    )
    analytical = compute_analytical_bounds(
        ctx,
        bid_lb,
        bid_ub,
        rho_ub_default=args.rho_ub_default,
        safety_factor=args.safety_factor,
        epsilon=args.epsilon,
        nonnegative_prices=not args.allow_negative_prices,
    )
    stress: dict[str, Any] = {
        "enabled": False,
        "num_trajectories": 0,
        "trajectory_bounds": {},
        "summary": {},
        "failed_solves": [],
    }
    if args.mode in {"support_stress_obbt", "hybrid"}:
        trajectories = generate_stress_trajectories(
            ctx,
            bid_lb,
            bid_ub,
            max_stress_trajectories=args.max_stress_trajectories,
        )
        stress = run_stress_obbt(
            ctx,
            trajectories,
            analytical,
            time_limit=args.time_limit,
            threads=args.threads,
            progress_every=args.log_progress_every,
        )

    if args.mode == "analytical_conservative":
        final = analytical
    elif args.mode == "support_stress_obbt":
        final = _combine_bounds(analytical, stress, args.allow_stress_tightening)
    else:
        final = _combine_bounds(analytical, stress, args.allow_stress_tightening)

    if stress.get("summary", {}).get("indexed_bounds"):
        stress_global = stress["summary"]["global_bounds"]
        analytical_global = analytical["global_bounds"]
        for key in ("mu_max_ub", "mu_min_ub", "rho_up_ub", "rho_down_ub"):
            if float(stress_global[key]) > float(analytical_global[key]) + 1e-8:
                print(
                    f"WARNING: stress-test OBBT {key}={stress_global[key]:.6g} exceeds "
                    f"analytical_conservative {key}={analytical_global[key]:.6g}"
                )

    final_equals_analytical = final["global_bounds"] == analytical["global_bounds"]
    final_bound_rule = (
        "stress_tightening_allowed"
        if args.allow_stress_tightening
        else "certified_max_of_analytical_and_stress"
    )
    if final_equals_analytical and args.mode in {"hybrid", "support_stress_obbt"}:
        print(
            "NOTE: Final saved bounds equal the analytical conservative bounds. "
            "This is expected when stress OBBT is unavailable, infeasible, lower than "
            "analytical bounds, or --allow-stress-tightening is not set."
        )

    payload = {
        "metadata": {
            "source": Path(__file__).name,
            "mode": args.mode,
            "support_set_name": args.support_set_name,
            "ramps_in_stationarity": True,
            "bid_bound_source": args.bid_bound_source,
            "default_bid_lb": float(args.default_bid_lb),
            "default_bid_ub": float(args.default_bid_ub),
            "rho_ub_default": float(args.rho_ub_default),
            "safety_factor": float(args.safety_factor),
            "epsilon": float(args.epsilon),
            "allow_stress_tightening": bool(args.allow_stress_tightening),
            "final_bound_rule": final_bound_rule,
            "final_equals_analytical": final_equals_analytical,
            "validity_note": (
                VALIDITY_NOTE
                if not args.allow_stress_tightening
                else VALIDITY_NOTE + " Stress tightening was allowed, so final bounds are not globally certified."
            ),
            "bid_bound_note": (
                "global uses configured/default bid_lb and bid_ub. true_cost fixes each "
                "block bid bound to its marginal cost and is only valid for no-policy or "
                "fixed true-cost experiments, not for learned/strategic bid policies."
            ),
            "rho_activity_warning": (
                "If final PoA solve dual_bound_activity reports rho_up or rho_down at "
                "its bound, increase --rho-ub-default and recompute these bounds."
            ),
        },
        "global_bounds": final["global_bounds"],
        "indexed_bounds": final["indexed_bounds"],
        "analytical_bounds": analytical,
        "stress_test_bounds": stress,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute support-set ramp-aware KKT dual Big-M bounds for bidding-block PoA."
    )
    parser.add_argument("--reference-case", default="test_case_bidding_blocks")
    parser.add_argument("--support-set-name", default="test_case_bidding_blocks_base")
    parser.add_argument("--support-set-config", default="models/PoA/support_set_config.yaml")
    parser.add_argument("--output", default="results/support_set_ramp_dual_bounds_bidding_blocks.json")
    parser.add_argument(
        "--mode",
        choices=["analytical_conservative", "support_stress_obbt", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--rho-ub-default", type=float, default=RHO_UB_DEFAULT)
    parser.add_argument("--safety-factor", type=float, default=1.05)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--max-stress-trajectories", type=int, default=20)
    parser.add_argument("--allow-stress-tightening", action="store_true")
    parser.add_argument("--allow-negative-prices", action="store_true")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level.",
    )
    parser.add_argument(
        "--log-progress-every",
        type=int,
        default=25,
        help="Log stress OBBT progress every N completed/skipped LPs. Use 0 to disable progress logs.",
    )
    parser.add_argument("--default-bid-lb", type=float, default=DEFAULT_BID_LB)
    parser.add_argument("--default-bid-ub", type=float, default=DEFAULT_BID_UB)
    parser.add_argument(
        "--bid-bound-source",
        choices=["global", "true_cost"],
        default="global",
        help=(
            "global uses configured/default bid bounds. true_cost fixes bid_lb=bid_ub "
            "to each block marginal cost and is only valid for fixed true-cost policy runs."
        ),
    )
    parser.add_argument("--regime-set", default="PoA_analysis")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-time-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    payload = build_payload(args)
    warnings = validate_payload(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    print_summary(payload)
    for warning in warnings:
        print(warning)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
