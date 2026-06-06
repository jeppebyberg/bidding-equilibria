"""Out-of-sample PoA evaluation for the DRO bidding policy pipeline.

Draws n_scenarios fresh scenarios from a target regime (different seed from
training/optimization), applies trained NN bidding policies to obtain strategic
bids, then solves two LP dispatch models:
  - equilibrium dispatch: NN bids for strategic generators, true costs for others
  - optimal dispatch:     true marginal costs for all generators (social optimum)

Computes mean and maximum realized PoA ratio across scenarios.
C_eq and C_opt both use true block costs (not bids) as the cost measure.

This is post-processing only -- no MILP is invoked, only LP dispatch (Gurobi).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from driver.core.block1_core import apply_time_steps_override
from models.helper import (
    available_block_capacity,
    block_cost_vector,
    block_structure_from_dataframes,
    find_demand_profile_column,
    infer_num_time_steps,
    parse_profile_exact_length,
)
from models.neural_network.training.model import BiddingPolicyNetwork
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel


# ---------------------------------------------------------------------------
# Regime config helpers
# ---------------------------------------------------------------------------

def _write_oos_regime_config(
    source_config_path: Path,
    source_regime_set: str,
    regime_name: str,
    n_scenarios: int,
    output_path: Path,
) -> str:
    """Copy one regime from source_regime_set, override n_scenarios, write JSON.

    Returns the new regime set name ("oos_{regime_name}") for use with
    ScenarioManager.create_scenario_set_from_regimes().
    """
    with source_config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    regime_sets = raw.get("regime_sets", {}) or {}
    if source_regime_set not in regime_sets:
        raise ValueError(
            f"Regime set '{source_regime_set}' not found in {source_config_path}. "
            f"Available sets: {sorted(regime_sets)}"
        )

    source_set = dict(regime_sets[source_regime_set] or {})
    regimes_in_source = source_set.get("regimes", []) or []
    target_regime = next(
        (dict(r) for r in regimes_in_source if str(r.get("name")) == regime_name),
        None,
    )
    if target_regime is None:
        available = [str(r.get("name")) for r in regimes_in_source]
        raise ValueError(
            f"Regime '{regime_name}' not found in set '{source_regime_set}'. "
            f"Available regimes: {available}"
        )

    target_regime["n_scenarios"] = n_scenarios
    oos_set_name = f"oos_{regime_name}"
    oos_config = {
        "regime_sets": {
            oos_set_name: {
                **{k: v for k, v in source_set.items() if k != "regimes"},
                "regimes": [target_regime],
            }
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(oos_config, f, indent=2)

    return oos_set_name


# ---------------------------------------------------------------------------
# NN policy loading
# ---------------------------------------------------------------------------

def _load_policy_network(
    model_dir: Path,
    generator_name: str,
) -> tuple[BiddingPolicyNetwork, dict[str, Any]]:
    """Load trained policy .pt and metadata JSON for one generator."""
    metadata_path = model_dir / f"{generator_name}_policy_metadata.json"
    model_path = model_dir / f"{generator_name}_policy.pt"

    for path in (metadata_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Policy artifact not found: {path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    model = BiddingPolicyNetwork(
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["output_dim"]),
        hidden_layers=list(metadata["hidden_layers"]),
        final_activation=str(metadata.get("final_activation", "linear")),
    )
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model, metadata


def _load_normalization_stats(
    stats_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load min_max_stats.json.

    Returns {gen_name: {"feature_min": {...}, "feature_max": {...}}}.
    For shared (non-per-generator) stats the key "__shared__" is used.
    """
    with stats_path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    if payload.get("per_generator"):
        return {gen: dict(s) for gen, s in payload["stats"].items()}

    return {"__shared__": dict(payload["stats"])}


def _get_gen_stats(
    norm_stats: dict[str, dict[str, dict[str, float]]],
    generator_name: str,
) -> dict[str, dict[str, float]]:
    if generator_name in norm_stats:
        return norm_stats[generator_name]
    if "__shared__" in norm_stats:
        return norm_stats["__shared__"]
    raise KeyError(
        f"No normalization stats found for generator '{generator_name}'. "
        f"Available keys: {sorted(norm_stats)}"
    )


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _build_market_features(
    scenarios_df: Any,  # pd.DataFrame, typed as Any to avoid import at top level
    block_names: list[str],
    block_to_physical: dict[str, str],
    num_time_steps: int,
) -> dict[str, np.ndarray]:
    """Build market-wide feature arrays of shape (n_scenarios, n_time_steps).

    Mirrors NeuralNetworkFeatureBuilder._compute_market_features() including
    circular (wrap-around) indexing for previous/next time features.
    """
    n = len(scenarios_df)
    T = num_time_steps
    demand_col = find_demand_profile_column(scenarios_df)

    demand = np.zeros((n, T), dtype=float)
    for s in range(n):
        demand[s] = parse_profile_exact_length(
            scenarios_df.at[s, demand_col], T, demand_col
        )

    total_cap = np.zeros((n, T), dtype=float)
    wind_cap = np.zeros((n, T), dtype=float)
    for s in range(n):
        for t in range(T):
            for b_name in block_names:
                cap = float(available_block_capacity(scenarios_df, b_name, s, t))
                total_cap[s, t] += cap
                phys = block_to_physical.get(b_name, b_name)
                if phys.startswith("W") or "wind" in phys.lower():
                    wind_cap[s, t] += cap

    previous_wind = np.roll(wind_cap, 1, axis=1)
    next_wind = np.roll(wind_cap, -1, axis=1)
    previous_demand = np.roll(demand, 1, axis=1)
    next_demand = np.roll(demand, -1, axis=1)
    # Horizon aggregates: per-scenario sums broadcast back across every time step.
    total_demand_horizon = np.broadcast_to(
        demand.sum(axis=1, keepdims=True), demand.shape
    ).copy()
    total_wind_horizon = np.broadcast_to(
        wind_cap.sum(axis=1, keepdims=True), wind_cap.shape
    ).copy()
    return {
        # Current period
        "demand": demand,
        "total_wind_generation_capacity": wind_cap,
        "total_generation_capacity": total_cap,           # legacy
        "residual_demand": demand - wind_cap,
        # Previous period
        "previous_demand": previous_demand,
        "previous_wind_generation_capacity": previous_wind,
        "previous_residual_demand": previous_demand - previous_wind,
        "previous_generation_capacity": np.roll(total_cap, 1, axis=1),  # legacy
        # Next period
        "next_demand": next_demand,
        "next_wind_generation_capacity": next_wind,
        "next_residual_demand": next_demand - next_wind,
        "next_generation_capacity": np.roll(total_cap, -1, axis=1),     # legacy
        # Horizon aggregates
        "total_demand_over_horizon": total_demand_horizon,
        "total_wind_over_horizon": total_wind_horizon,
        "total_residual_over_horizon": total_demand_horizon - total_wind_horizon,
    }


def _build_own_capacity(
    scenarios_df: Any,
    gen_block_names: list[str],
    num_time_steps: int,
) -> np.ndarray:
    """Own-capacity array (n_scenarios, n_time_steps) for one generator."""
    n = len(scenarios_df)
    T = num_time_steps
    own = np.zeros((n, T), dtype=float)
    for s in range(n):
        for t in range(T):
            own[s, t] = sum(
                float(available_block_capacity(scenarios_df, b, s, t))
                for b in gen_block_names
            )
    return own


def _normalize(val: float, f_min: float, f_max: float) -> float:
    denom = f_max - f_min
    return (val - f_min) / denom if denom != 0.0 else 0.0


# ---------------------------------------------------------------------------
# Bid construction
# ---------------------------------------------------------------------------

def _blocks_from_target_columns(target_columns: list[str]) -> list[str]:
    """Strip "target_bid_" prefix from target_columns to get block names."""
    prefix = "target_bid_"
    return [
        tc[len(prefix):] if tc.startswith(prefix) else tc
        for tc in target_columns
    ]


def _build_nn_bid_profiles(
    generator_name: str,
    gen_block_names: list[str],
    feature_columns: list[str],
    model: BiddingPolicyNetwork,
    gen_stats: dict[str, dict[str, float]],
    market_features: dict[str, np.ndarray],
    own_cap: np.ndarray,
) -> dict[str, list[list[float]]]:
    """Forward-pass the NN for every (scenario, time) and collect bid time-series.

    Returns {block_name: [[bid_t0, ..., bid_tT] for each scenario]}.
    """
    n, T = own_cap.shape
    bid_profiles: dict[str, list[list[float]]] = {
        b: [[] for _ in range(n)] for b in gen_block_names
    }
    feat_min = gen_stats.get("feature_min", {})
    feat_max = gen_stats.get("feature_max", {})

    for s in range(n):
        for t in range(T):
            feat_vals: list[float] = []
            for col in feature_columns:
                if col in market_features:
                    raw = float(market_features[col][s, t])
                elif col == "own_generation_capacity":
                    raw = float(own_cap[s, t])
                elif col == "previous_own_generation_capacity":
                    raw = float(own_cap[s, (t - 1) % T])
                elif col == "next_own_generation_capacity":
                    raw = float(own_cap[s, (t + 1) % T])
                else:
                    raise ValueError(
                        f"Unknown feature column '{col}' for generator '{generator_name}'. "
                        f"Supported market features: {sorted(market_features)}"
                    )
                feat_vals.append(
                    _normalize(raw, feat_min.get(col, 0.0), feat_max.get(col, 1.0))
                )

            x = torch.tensor(feat_vals, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                bids_t = model(x).squeeze(0).cpu().numpy()

            for b_idx, b_name in enumerate(gen_block_names):
                bid_profiles[b_name][s].append(float(bids_t[b_idx]))

    return bid_profiles


def _build_dispatch_scenarios(
    scenarios_df: Any,
    block_names: list[str],
    cost_vec: list[float],
    num_time_steps: int,
    nn_policies: dict[str, tuple[BiddingPolicyNetwork, dict[str, Any]]],
    norm_stats: dict[str, dict[str, dict[str, float]]],
    market_features: dict[str, np.ndarray],
) -> tuple[Any, Any]:
    """Return (eq_df, opt_df) with bid columns attached.

    eq_df  — NN bids for NN generators, true costs for all others.
    opt_df — true marginal costs for every generator (social optimum bids).
    """
    import pandas as pd  # local import to keep top-level imports lean

    n = len(scenarios_df)
    T = num_time_steps
    eq_df = scenarios_df.copy()
    opt_df = scenarios_df.copy()

    # Identify which blocks are covered by NN policies
    nn_covered_blocks: set[str] = set()
    for gen_name, (model, metadata) in nn_policies.items():
        gen_block_names = _blocks_from_target_columns(metadata["target_columns"])
        feature_columns: list[str] = metadata["feature_columns"]
        gen_stats = _get_gen_stats(norm_stats, gen_name)
        own_cap = _build_own_capacity(scenarios_df, gen_block_names, T)

        bid_profiles = _build_nn_bid_profiles(
            generator_name=gen_name,
            gen_block_names=gen_block_names,
            feature_columns=feature_columns,
            model=model,
            gen_stats=gen_stats,
            market_features=market_features,
            own_cap=own_cap,
        )
        for b_name, profiles in bid_profiles.items():
            eq_df[f"{b_name}_bid_profile"] = profiles
            nn_covered_blocks.add(b_name)

    # Cost-as-bid for every block (equilibrium non-NN + all optimal)
    for b_idx, b_name in enumerate(block_names):
        cost = cost_vec[b_idx]
        cost_profile = [[cost] * T for _ in range(n)]
        if b_name not in nn_covered_blocks:
            eq_df[f"{b_name}_bid_profile"] = cost_profile
        opt_df[f"{b_name}_bid_profile"] = cost_profile

    return eq_df, opt_df


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

def _true_cost(
    block_dispatches: list[list[list[float]]],
    cost_vec: list[float],
    scenario_id: int,
    num_time_steps: int,
    n_blocks: int,
) -> float:
    """True generation cost (using block_cost_vector, not bid costs) for one scenario."""
    return sum(
        cost_vec[b] * block_dispatches[scenario_id][t][b]
        for t in range(num_time_steps)
        for b in range(n_blocks)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_oos_poa_for_regime(
    case: str,
    regime_name: str,
    source_regime_config: str | Path,
    source_regime_set: str,
    horizon: int,
    model_dir: Path,
    norm_stats_path: Path,
    nn_policy_generators: list[str],
    n_scenarios: int = 100,
    seed: int = 999,
    oos_config_path: Path = Path("results/oos_poa/oos_regime_config.json"),
) -> dict[str, Any]:
    """Draw n_scenarios fresh OOS scenarios and return mean realized PoA ratio.

    The equilibrium dispatch uses NN bids for nn_policy_generators and true
    marginal costs for all other generators. The optimal dispatch uses true
    marginal costs for all generators. C_eq and C_opt are both evaluated at
    true block costs (not bids), consistent with the DRO MILP definition.

    Args:
        case:                 Reference case name (e.g. "base_test_case").
        regime_name:          Target regime name (e.g. "normal").
        source_regime_config: Path to regime_definitions.yaml.
        source_regime_set:    Regime set name in that config (e.g. "PoA_analysis").
        horizon:              Number of time steps T.
        model_dir:            Directory with *_policy.pt and *_policy_metadata.json.
        norm_stats_path:      Path to min_max_stats.json.
        nn_policy_generators: Names of generators with trained NN policies.
        n_scenarios:          Number of fresh scenarios to draw (default 100).
        seed:                 RNG seed for OOS draws (use a seed not used in training).
        oos_config_path:      Temp file path for the OOS regime config YAML/JSON.

    Returns:
        dict with keys:
            regime_name           - regime drawn from.
            n_scenarios_requested - n_scenarios argument.
            n_scenarios_used      - scenarios with C_opt > 0 (used for mean PoA).
            seed                  - seed used.
            oos_mean_poa_ratio    - mean(C_eq / C_opt) across valid scenarios.
            poa_ratios            - per-scenario PoA ratios.
            c_eq                  - per-scenario equilibrium true costs.
            c_opt                 - per-scenario optimal true costs.
    """
    source_regime_config = Path(source_regime_config)

    # 1. Generate fresh scenarios with a separate seed
    oos_set_name = _write_oos_regime_config(
        source_config_path=source_regime_config,
        source_regime_set=source_regime_set,
        regime_name=regime_name,
        n_scenarios=n_scenarios,
        output_path=oos_config_path,
    )
    manager = ScenarioManager(case)
    apply_time_steps_override(manager, horizon)
    raw = manager.create_scenario_set_from_regimes(
        regime_config_path=str(oos_config_path),
        regime_set=oos_set_name,
        seed=seed,
        enforce_support_set=False,
    )

    scenarios_df = raw["scenarios_df"].reset_index(drop=True)
    costs_df = raw["costs_df"]
    ramps_df = raw["ramps_df"]

    if "regime" in scenarios_df.columns:
        scenarios_df = (
            scenarios_df[scenarios_df["regime"] == regime_name]
            .reset_index(drop=True)
        )

    if len(scenarios_df) < n_scenarios:
        warnings.warn(
            f"Only {len(scenarios_df)} scenarios available for regime '{regime_name}'; "
            f"requested {n_scenarios}.",
            stacklevel=2,
        )
    scenarios_df = scenarios_df.iloc[:n_scenarios].reset_index(drop=True)
    actual_n = len(scenarios_df)

    # 2. Block structure
    block_struct = block_structure_from_dataframes(scenarios_df, ramps_df)
    block_names = list(block_struct.block_names)
    block_to_physical = dict(block_struct.block_to_physical)
    cost_vec = list(block_cost_vector(costs_df, block_names))
    T = infer_num_time_steps(scenarios_df)

    # 3. Load NN policies and normalization stats
    nn_policies: dict[str, tuple[BiddingPolicyNetwork, dict[str, Any]]] = {
        gen: _load_policy_network(model_dir, gen)
        for gen in nn_policy_generators
    }
    norm_stats = _load_normalization_stats(norm_stats_path)

    # 4. Build shared market features (used by all generators)
    market_features = _build_market_features(
        scenarios_df, block_names, block_to_physical, T
    )

    # 5. Build equilibrium and optimal scenarios_df copies with bid columns
    eq_df, opt_df = _build_dispatch_scenarios(
        scenarios_df=scenarios_df,
        block_names=block_names,
        cost_vec=cost_vec,
        num_time_steps=T,
        nn_policies=nn_policies,
        norm_stats=norm_stats,
        market_features=market_features,
    )

    # 6. Solve dispatch LPs
    print(f"  Solving equilibrium dispatch ({actual_n} scenarios)...")
    ed_eq = EconomicDispatchModel(eq_df, costs_df, ramps_df)
    ed_eq.solve()

    print(f"  Solving optimal dispatch ({actual_n} scenarios)...")
    ed_opt = EconomicDispatchModel(opt_df, costs_df, ramps_df)
    ed_opt.solve()

    if ed_eq.block_dispatches is None:
        raise RuntimeError("Equilibrium LP dispatch failed to find an optimal solution.")
    if ed_opt.block_dispatches is None:
        raise RuntimeError("Optimal LP dispatch failed to find an optimal solution.")

    # 7. Compute true-cost PoA per scenario
    n_blocks = len(block_names)
    poa_ratios: list[float] = []
    c_eq_list: list[float] = []
    c_opt_list: list[float] = []

    for s in range(actual_n):
        c_eq = _true_cost(ed_eq.block_dispatches, cost_vec, s, T, n_blocks)
        c_opt = _true_cost(ed_opt.block_dispatches, cost_vec, s, T, n_blocks)
        c_eq_list.append(c_eq)
        c_opt_list.append(c_opt)
        if c_opt > 1e-9:
            poa_ratios.append(c_eq / c_opt)
        else:
            warnings.warn(
                f"Scenario {s}: C_opt={c_opt:.3g} near zero; PoA ratio skipped.",
                stacklevel=2,
            )

    oos_mean_poa = float(np.mean(poa_ratios)) if poa_ratios else float("nan")
    oos_max_poa = float(np.max(poa_ratios)) if poa_ratios else float("nan")
    return {
        "regime_name": regime_name,
        "n_scenarios_requested": n_scenarios,
        "n_scenarios_used": len(poa_ratios),
        "seed": seed,
        "oos_mean_poa_ratio": oos_mean_poa,
        "oos_max_poa_ratio": oos_max_poa,
        "poa_ratios": poa_ratios,
        "c_eq": c_eq_list,
        "c_opt": c_opt_list,
    }

def save_oos_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved OOS PoA results: {output_path}")

# ---------------------------------------------------------------------------
# Entry point — edit the config block below, then run this script directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Configuration ────────────────────────────────────────────────────────
    CASE = "base_test_case"
    SOURCE_REGIME_CONFIG = Path("results/sensitivity_pipeline/runtime_regime_definitions.yaml")
    SOURCE_REGIME_SET = "sensitivity_runtime"
    REGIME_NAMES = ["poa_worst_case"]
    HORIZON = 8
    MODEL_DIR = Path("results/sensitivity_pipeline/trained_models")
    NORM_STATS_PATH = Path(
        "results/sensitivity_pipeline/features/normalized/min_max_stats.json"
    )
    NN_POLICY_GENERATORS = ["G1", "G2", "W3"]
    N_SCENARIOS = 200
    SEED = 999                          # different from poa_seed=1/2 used in training/DRO
    OUTPUT_DIR = Path("results/oos_poa/sensitivity_pipeline")
    # ─────────────────────────────────────────────────────────────────────────

    all_results: list[dict[str, Any]] = []
    for regime_name in REGIME_NAMES:
        print(f"\nOOS PoA evaluation: regime='{regime_name}', n={N_SCENARIOS}, seed={SEED}")
        result = evaluate_oos_poa_for_regime(
            case=CASE,
            regime_name=regime_name,
            source_regime_config=SOURCE_REGIME_CONFIG,
            source_regime_set=SOURCE_REGIME_SET,
            horizon=HORIZON,
            model_dir=MODEL_DIR,
            norm_stats_path=NORM_STATS_PATH,
            nn_policy_generators=NN_POLICY_GENERATORS,
            n_scenarios=N_SCENARIOS,
            seed=SEED,
            oos_config_path=OUTPUT_DIR / "oos_regime_config.json",
        )
        oos_mean_poa = result["oos_mean_poa_ratio"]
        oos_max_poa = result["oos_max_poa_ratio"]
        n_used = result["n_scenarios_used"]
        print(
            f"  OOS PoA ratio: mean={oos_mean_poa:.4f}, max={oos_max_poa:.4f} "
            f"(from {n_used} scenarios)"
        )
        all_results.append(result)

    save_oos_results(all_results, OUTPUT_DIR / "oos_poa_results.json")

    # Calibrated epsilon from the sweep frontier
    from results_viz.plot_dro_poa_eta_sweep import (
        calibrated_epsilon,
        load_eta_sweep_records,
        plot_poa_epsilon_frontier,
    )

    dro_results_dir = Path("results/sensitivity_pipeline/dro")
    print()
    for result in all_results:
        reg = result["regime_name"]
        oos_mean_poa = result["oos_mean_poa_ratio"]
        oos_max_poa = result["oos_max_poa_ratio"]
        if np.isnan(oos_mean_poa) and np.isnan(oos_max_poa):
            print(f"Regime '{reg}': OOS PoA is NaN -- skipping calibration.")
            continue
        try:
            records = load_eta_sweep_records(dro_results_dir, reg)
        except FileNotFoundError as exc:
            print(f"Regime '{reg}': no sweep records found ({exc}). Run the DRO sweep first.")
            continue
        if not np.isnan(oos_mean_poa):
            ce_mean = calibrated_epsilon(records, oos_mean_poa)
            print(f"Regime '{reg}' mean: {ce_mean['message']}")
        if not np.isnan(oos_max_poa):
            ce_max = calibrated_epsilon(records, oos_max_poa)
            print(f"Regime '{reg}' max:  {ce_max['message']}")
        frontier_path = plot_poa_epsilon_frontier(
            records=records,
            output_dir=OUTPUT_DIR / "frontiers" / reg,
            regime_name=reg,
            poa_metric="inner_objective",
            oos_mean_poa=oos_mean_poa,
            oos_max_poa=oos_max_poa,
        )
        print(f"Regime '{reg}': saved OOS PoA-epsilon frontier: {frontier_path}")
