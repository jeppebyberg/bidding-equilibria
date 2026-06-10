"""DRO out-of-sample analysis.

Draws N fresh test scenarios from the poa_worst_case regime (the regime
selected by the base PoA optimizer) and evaluates the average PoA under
the fixed NN bidding policies.

For each fresh scenario s:
  C_eq(s)  = sum over (block, t) of true_cost[block] * equilibrium_dispatch[block,t]
             where equilibrium dispatch clears the market under NN-predicted bids
             (for NN generators) and true costs (for non-NN generators).
  C_opt(s) = sum over (block, t) of true_cost[block] * optimal_dispatch[block,t]
             where optimal dispatch uses true costs for all generators.
  PoA(s)   = C_eq(s) / C_opt(s)

The average OOS PoA is plotted as a horizontal band over the in-sample
DRO eta-sweep curve to show the gap between the worst-case DRO bound and
the expected PoA under random scenario draws.

Run:
  .venv/Scripts/python.exe results_viz/dro_out_of_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import RegimeParameters, ScenarioManager
from models.neural_network.training.model import BiddingPolicyNetwork
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POA_RESULT_PATH = Path(
    "results/sensitivity_pipeline/poa/poa_optimization_T8_piecewise_mccormick.json"
)
MODEL_DIR = Path("results/sensitivity_pipeline/trained_models")
NORM_STATS_PATH = Path(
    "results/sensitivity_pipeline/features/normalized/min_max_stats.json"
)
ETA_SUMMARY_PATH = Path("results/sensitivity_pipeline/dro/eta_sweep_summary.json")
OUTPUT_DIR = Path("results_viz/figures/dro_out_of_sample")

REFERENCE_CASE = "base_test_case"
N_OOS = 200        # fresh test scenarios
OOS_SEED = 12345   # independent of training (seed=1) and DRO (seed=1/2)
BATCH_SIZE = 50    # scenarios per EconomicDispatch solve


# ---------------------------------------------------------------------------
# NN policy helpers
# ---------------------------------------------------------------------------

class _NNPolicy:
    def __init__(self, model: BiddingPolicyNetwork, meta: dict[str, Any]) -> None:
        self.model = model
        self.generator_name: str = meta["generator_name"]
        self.feature_columns: list[str] = meta["feature_columns"]
        self.block_names: list[str] = [c.replace("target_bid_", "") for c in meta["target_columns"]]

    def predict(self, feature_row: np.ndarray) -> np.ndarray:
        x = torch.tensor(feature_row.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            return self.model(x).numpy()[0]


def _load_policies(model_dir: Path) -> dict[str, _NNPolicy]:
    policies: dict[str, _NNPolicy] = {}
    for meta_path in sorted(model_dir.glob("*_policy_metadata.json")):
        gen_name = meta_path.stem.replace("_policy_metadata", "")
        model_path = model_dir / f"{gen_name}_policy.pt"
        if not model_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        net = BiddingPolicyNetwork(
            input_dim=meta["input_dim"],
            output_dim=meta["output_dim"],
            hidden_layers=meta["hidden_layers"],
            final_activation=meta.get("final_activation", "linear"),
        )
        net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        net.eval()
        policies[gen_name] = _NNPolicy(net, meta)
    return policies


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _load_norm_stats(stats_path: Path) -> dict[str, Any]:
    """Return raw["stats"] keyed by generator name."""
    return json.loads(stats_path.read_text())["stats"]


def _normalize(
    feature_row: dict[str, float],
    feature_columns: list[str],
    gen_stats: dict[str, Any],
) -> np.ndarray:
    fmin = gen_stats["feature_min"]
    fmax = gen_stats["feature_max"]
    out = []
    for col in feature_columns:
        lo, hi = float(fmin[col]), float(fmax[col])
        val = float(feature_row[col])
        out.append(0.0 if hi <= lo else (val - lo) / (hi - lo))
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature computation (mirrors NeuralNetworkFeatureBuilder)
# ---------------------------------------------------------------------------

def _compute_features(
    demand_profile: list[float],
    wind_profiles: dict[str, list[float]],  # {physical_name: [T values in MW]}
    phys_own_cap: dict[str, list[float]],   # {physical_name: [T values in MW]}
    total_conv_cap: float,
    T: int,
) -> dict[str, list[dict[str, float]]]:
    """Return {gen_name: [T feature dicts]} for all physical generators."""
    total_wind_cap = [
        sum(wind_profiles[g][t] for g in wind_profiles) for t in range(T)
    ]
    total_gen_cap = [total_wind_cap[t] + total_conv_cap for t in range(T)]

    result: dict[str, list[dict[str, float]]] = {}
    for gen_name, own_cap in phys_own_cap.items():
        rows = []
        for t in range(T):
            prev_t = (t - 1) % T
            next_t = (t + 1) % T
            rows.append({
                "demand": demand_profile[t],
                "total_wind_generation_capacity": total_wind_cap[t],
                "total_generation_capacity": total_gen_cap[t],
                "residual_demand": demand_profile[t] - total_wind_cap[t],
                "previous_generation_capacity": total_gen_cap[prev_t],
                "previous_demand": demand_profile[prev_t],
                "next_generation_capacity": total_gen_cap[next_t],
                "next_demand": demand_profile[next_t],
                "own_generation_capacity": own_cap[t],
                "previous_own_generation_capacity": own_cap[prev_t],
                "next_own_generation_capacity": own_cap[next_t],
            })
        result[gen_name] = rows
    return result


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _total_cost_per_scenario(
    ed: EconomicDispatchModel,
) -> list[float]:
    """Compute sum(true_cost * dispatch) per scenario over all blocks and time steps."""
    costs = []
    for s in range(ed.num_scenarios):
        total = sum(
            ed.block_cost_vector[b] * ed.block_dispatches[s][t][b]
            for t in range(ed.num_time_steps)
            for b in range(ed.num_blocks)
        )
        costs.append(total)
    return costs


def _run_batch(
    scenarios_rows: list[dict[str, Any]],
    costs_df: pd.DataFrame,
    ramps_df: pd.DataFrame,
) -> list[float]:
    """Solve dispatch for a batch and return true-cost total per scenario."""
    df = pd.DataFrame(scenarios_rows)
    ed = EconomicDispatchModel(scenarios_df=df, costs_df=costs_df, ramps_df=ramps_df)
    ed.solve()
    if ed.block_dispatches is None:
        raise RuntimeError("EconomicDispatch failed — no solution stored.")
    return _total_cost_per_scenario(ed)


# ---------------------------------------------------------------------------
# Main OOS evaluation
# ---------------------------------------------------------------------------

def run_oos_analysis(
    poa_result_path: Path = POA_RESULT_PATH,
    model_dir: Path = MODEL_DIR,
    norm_stats_path: Path = NORM_STATS_PATH,
    n_oos: int = N_OOS,
    oos_seed: int = OOS_SEED,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Run the out-of-sample PoA evaluation and return per-scenario results."""

    # --- Load regime parameters -----------------------------------------------
    poa_result = json.loads(poa_result_path.read_text())
    amb = poa_result["ambiguity_set"]
    fp = amb["fixed_parameters"]
    sel = amb["selected_regime"]
    T = int(poa_result["num_time_steps"])

    regime = RegimeParameters(
        name="poa_worst_case_oos",
        n_scenarios=n_oos,
        mu_D=float(sel["mu_D"]),
        rho_D=float(fp["rho_D"]),
        sigma_D=float(sel["sigma_D"]),
        mu_W=float(sel["mu_W"]),
        peak_W=float(fp["tau_W"]),
        rho_W=float(fp["rho_W"]),
        sigma_W=float(sel["sigma_W"]),
    )
    print(f"Regime: mu_D={regime.mu_D:.4g}, sigma_D={regime.sigma_D:.4g}, "
          f"mu_W={regime.mu_W:.4g}, sigma_W={regime.sigma_W:.4g}, peak_W={regime.peak_W:.4g}")

    # --- Load NN policies and normalization stats ------------------------------
    policies = _load_policies(model_dir)
    print(f"Loaded NN policies: {sorted(policies.keys())}")
    norm_stats = _load_norm_stats(norm_stats_path)

    # --- Build block structure from ScenarioManager ---------------------------
    manager = ScenarioManager(REFERENCE_CASE)
    manager.base_case["time_steps"] = T
    costs_df, ramps_df = manager._build_costs_and_ramps()

    # Map physical generators to their static block capacity sums.
    block_pmaxes: dict[str, float] = {b["block_name"]: float(b["pmax"]) for b in manager.blocks}
    block_costs: dict[str, float] = {b["block_name"]: float(b["cost"]) for b in manager.blocks}
    wind_physical: list[str] = [g["physical_name"] for g in manager.physical_generators if g["is_wind"]]
    conv_physical: list[str] = [g["physical_name"] for g in manager.physical_generators if not g["is_wind"]]

    # Static own_cap per conventional generator (sum of its block pmaxes).
    conv_own_cap: dict[str, float] = {
        g["physical_name"]: sum(
            block_pmaxes[b["block_name"]]
            for b in manager.blocks
            if b["physical_name"] == g["physical_name"]
        )
        for g in manager.physical_generators if not g["is_wind"]
    }
    total_conv_cap = sum(conv_own_cap.values())

    # --- Generate fresh scenarios ---------------------------------------------
    print(f"\nGenerating {n_oos} fresh OOS scenarios (seed={oos_seed}) ...")
    rng = np.random.default_rng(oos_seed)
    all_demand: list[list[float]] = []
    all_wind: list[dict[str, list[float]]] = []

    for _ in range(n_oos):
        demand_profile, wind_profiles, _ = manager._simulate_single_scenario(regime, rng)
        all_demand.append([float(v) for v in demand_profile])
        all_wind.append({g: [float(v) for v in wind_profiles[g]] for g in wind_physical})

    # --- Predict equilibrium bids for each scenario ---------------------------
    print("Predicting equilibrium bids ...")
    all_eq_bids: list[dict[str, list[float]]] = []

    for s_idx, (demand_profile, wind_profiles) in enumerate(zip(all_demand, all_wind)):
        # Build time-varying own_cap for each generator.
        phys_own_cap: dict[str, list[float]] = {}
        for gen in wind_physical:
            phys_own_cap[gen] = wind_profiles[gen]
        for gen in conv_physical:
            phys_own_cap[gen] = [conv_own_cap[gen]] * T

        features_by_gen = _compute_features(demand_profile, wind_profiles, phys_own_cap, total_conv_cap, T)

        # Start with true costs as bids, then override for NN generators.
        bids: dict[str, list[float]] = {block: [cost] * T for block, cost in block_costs.items()}
        for gen_name, policy in policies.items():
            gen_feats = features_by_gen.get(gen_name)
            gen_stats = norm_stats.get(gen_name)
            if gen_feats is None or gen_stats is None:
                continue
            for t in range(T):
                feat = _normalize(gen_feats[t], policy.feature_columns, gen_stats)
                pred = policy.predict(feat)
                for block_name, val in zip(policy.block_names, pred):
                    bids[block_name][t] = float(val)
        all_eq_bids.append(bids)

    # --- Build scenarios_df rows (equilibrium and optimal) --------------------
    eq_rows: list[dict[str, Any]] = []
    opt_rows: list[dict[str, Any]] = []

    for s_idx, (demand_profile, wind_profiles, eq_bids) in enumerate(
        zip(all_demand, all_wind, all_eq_bids)
    ):
        base_row = manager._build_scenario_row(
            scenario_id=s_idx,
            regime_name="oos",
            demand_mean=float(np.mean(demand_profile)),
            demand_profile=demand_profile,
            generator_wind_profiles=wind_profiles,
        )
        # Optimal: default row uses true costs as bid profiles — already correct.
        opt_rows.append(dict(base_row))

        # Equilibrium: override with NN-predicted bids.
        eq_row = dict(base_row)
        for block_name, bid_profile in eq_bids.items():
            eq_row[f"{block_name}_bid_profile"] = bid_profile
        eq_rows.append(eq_row)

    # --- Run dispatch in batches ----------------------------------------------
    print(f"\nRunning dispatch in batches of {batch_size} ...")
    c_eq_all: list[float] = []
    c_opt_all: list[float] = []

    for start in range(0, n_oos, batch_size):
        end = min(start + batch_size, n_oos)
        batch_n = end - start
        print(f"  Batch {start}–{end-1} ({batch_n} scenarios) ...")
        eq_batch = eq_rows[start:end]
        opt_batch = opt_rows[start:end]
        c_eq_all.extend(_run_batch(eq_batch, costs_df, ramps_df))
        c_opt_all.extend(_run_batch(opt_batch, costs_df, ramps_df))

    # --- Compute PoA ratios ---------------------------------------------------
    c_eq = np.array(c_eq_all)
    c_opt = np.array(c_opt_all)
    poa_ratios = np.where(c_opt > 1e-6, c_eq / c_opt, np.nan)

    valid = ~np.isnan(poa_ratios)
    mean_poa = float(np.mean(poa_ratios[valid]))
    std_poa = float(np.std(poa_ratios[valid]))
    n_valid = int(np.sum(valid))

    print(f"\nOOS results ({n_valid}/{n_oos} valid scenarios):")
    print(f"  Mean PoA ratio:  {mean_poa:.4f}")
    print(f"  Std  PoA ratio:  {std_poa:.4f}")
    print(f"  Min  PoA ratio:  {float(np.nanmin(poa_ratios)):.4f}")
    print(f"  Max  PoA ratio:  {float(np.nanmax(poa_ratios)):.4f}")

    return {
        "n_scenarios": n_oos,
        "n_valid": n_valid,
        "oos_seed": oos_seed,
        "mean_poa_ratio": mean_poa,
        "std_poa_ratio": std_poa,
        "min_poa_ratio": float(np.nanmin(poa_ratios)),
        "max_poa_ratio": float(np.nanmax(poa_ratios)),
        "c_eq": c_eq_all,
        "c_opt": c_opt_all,
        "poa_ratios": poa_ratios.tolist(),
        "regime": {
            "mu_D": regime.mu_D, "sigma_D": regime.sigma_D,
            "mu_W": regime.mu_W, "sigma_W": regime.sigma_W,
            "peak_W": regime.peak_W,
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_oos_vs_eta_sweep(
    oos_result: dict[str, Any],
    eta_summary_path: Path = ETA_SUMMARY_PATH,
    output_dir: Path = OUTPUT_DIR,
    metric: str = "average_poa_ratio",
) -> Path:
    """Plot OOS mean PoA as a horizontal band over the DRO in-sample eta sweep."""
    records = json.loads(eta_summary_path.read_text())
    etas = np.array([float(r["eta"]) for r in records])
    insample = np.array([float(r[metric]) for r in records if r.get(metric) is not None])

    oos_mean = oos_result["mean_poa_ratio"]
    oos_std = oos_result["std_poa_ratio"]
    n = oos_result["n_valid"]
    se = oos_std / np.sqrt(n)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # In-sample DRO curve.
    sorted_idx = np.argsort(etas)
    ax.plot(etas[sorted_idx], insample[sorted_idx],
            "o-", color="tab:blue", linewidth=2.0, markersize=5.5,
            label="DRO in-sample average PoA")

    # OOS mean ± 2 SE band.
    ax.axhline(oos_mean, color="tab:orange", linewidth=2.2, linestyle="--",
               label=f"OOS mean PoA = {oos_mean:.4f}  (N={n})")
    ax.axhspan(oos_mean - 2 * se, oos_mean + 2 * se,
               color="tab:orange", alpha=0.20, label="OOS ±2 SE")
    ax.axhline(1.0, color="0.45", linewidth=1.2, linestyle=":", alpha=0.6,
               label="PoA = 1 (competitive)")

    positive_etas = etas[etas > 0]
    if len(positive_etas) > 0 and max(positive_etas) / min(positive_etas) > 20:
        ax.set_xscale("log")

    ax.set_xlabel("η  (Wasserstein penalty)", fontsize=10)
    ax.set_ylabel("Average PoA ratio", fontsize=10)
    ax.set_title(
        f"DRO In-Sample vs. Out-of-Sample PoA\n"
        f"OOS regime: mu_D={oos_result['regime']['mu_D']:.4g}, "
        f"mu_W={oos_result['regime']['mu_W']:.4g}, "
        f"sigma_D={oos_result['regime']['sigma_D']:.4g}",
        fontsize=11,
    )
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout()

    out_path = output_dir / "poa_oos_vs_eta_sweep.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return out_path


def plot_oos_poa_distribution(
    oos_result: dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Histogram of per-scenario OOS PoA ratios."""
    poa_ratios = np.array(oos_result["poa_ratios"])
    valid = poa_ratios[~np.isnan(poa_ratios)]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(valid, bins=30, color="tab:blue", alpha=0.72, edgecolor="white", linewidth=0.5)
    ax.axvline(oos_result["mean_poa_ratio"], color="tab:orange", linewidth=2.0,
               label=f"Mean = {oos_result['mean_poa_ratio']:.4f}")
    ax.axvline(1.0, color="0.4", linewidth=1.2, linestyle="--", alpha=0.7, label="PoA = 1")
    ax.set_xlabel("PoA ratio  (C_eq / C_opt)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"Out-of-Sample PoA Distribution  (N={oos_result['n_valid']} scenarios)\n"
        f"seed={oos_result['oos_seed']}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = output_dir / "oos_poa_distribution.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    oos_result = run_oos_analysis(
        poa_result_path=POA_RESULT_PATH,
        model_dir=MODEL_DIR,
        norm_stats_path=NORM_STATS_PATH,
        n_oos=N_OOS,
        oos_seed=OOS_SEED,
        batch_size=BATCH_SIZE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "oos_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(oos_result, fh, indent=2)
    print(f"Saved: {summary_path}")

    plot_oos_vs_eta_sweep(oos_result)
    plot_oos_poa_distribution(oos_result)


if __name__ == "__main__":
    main()
