"""Standalone epsilon sweep for the constraint-based DRO PoA formulation.

Reuses the existing base_case tightening report and regime. Sweeps epsilon from
0 to 150 (linearly spaced) and solves the Wasserstein-constrained problem:

    max  (1/K) sum_k PoA(s~^(k))
    s.t. (1/K) sum_k W[k] <= epsilon
         s~^(k) in U_DRO(r)

Results and figures are saved under results/base_case_tmp/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from driver.core.block1_core import apply_time_steps_override
from driver.core.block2_core import discover_trained_policy_generators
from driver.core.block4_core import validate_scenarios_within_wasserstein_support
from models.DRO_PoA.DRO_PoA_optimization_tmp import DRO_PoAOptimization
from models.DRO_PoA.dro_poa_model.support_set import DROWassersteinSupportSet

# ---------------------------------------------------------------------------
# Configuration — edit here to change the sweep
# ---------------------------------------------------------------------------

CASE = "base_test_case"
REGIME_SET = "sensitivity_runtime"
REGIME_NAME = "poa_worst_case"
HORIZON = 8
SEED = 2
AR1_COVERAGE = 0.99
AMBIGUITY_KAPPA = 0.25

RUNTIME_CONFIG = Path("results/base_case/runtime_regime_definitions.yaml")
TIGHTENING_REPORT = Path(
    "results/base_case/dro/tightening/poa_worst_case/final_tightening_report.json"
)
MODEL_DIR = Path("results/base_case/trained_models")
NORM_STATS = Path("results/base_case/features/normalized/min_max_stats.json")

OUTPUT_DIR = Path("results/base_case_tmp")
RESULTS_DIR = OUTPUT_DIR / "epsilon_sweep"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Epsilon grid: 0 to 150, linearly spaced (includes both endpoints)
N_EPSILONS = 15
EPSILON_MAX = 150.0
EPSILON_GRID = np.linspace(0.0, EPSILON_MAX, N_EPSILONS).tolist()

# McCormick objective bounds (from existing tightening report)
C_OPT_BOUNDS = (145.53, 4728.53)
POA_BOUNDS = (1.0, 20.0)
NUM_PIECES = 50

SOLVER_TIME_LIMIT = 400  # seconds per solve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def epsilon_label(eps: float) -> str:
    return f"{eps:.6g}".replace("-", "m").replace(".", "p")


def result_path(eps: float) -> Path:
    return RESULTS_DIR / f"dro_epsilon_{epsilon_label(eps)}_T{HORIZON}.json"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_scenarios() -> dict[str, Any]:
    manager = ScenarioManager(CASE)
    apply_time_steps_override(manager, HORIZON)
    scenarios = manager.create_scenario_set_from_regimes(
        regime_config_path=str(RUNTIME_CONFIG),
        regime_set=REGIME_SET,
        seed=SEED,
        enforce_support_set=False,
    )
    DROWassersteinSupportSet.AR1_JOINT_COVERAGE = AR1_COVERAGE
    scenarios = validate_scenarios_within_wasserstein_support(
        scenarios=scenarios,
        manager=manager,
        horizon=HORIZON,
        ar1_coverage=AR1_COVERAGE,
    )
    print(scenarios["description_text"])
    print(f"  Scenarios after support validation: {len(scenarios['scenarios_df'])}")
    return scenarios


def build_optimizer(scenarios: dict[str, Any]) -> DRO_PoAOptimization:
    nn_generators = discover_trained_policy_generators(MODEL_DIR)
    print(f"\nDiscovered policy generators: {nn_generators}")
    norm_stats = str(NORM_STATS) if NORM_STATS.exists() else None

    return DRO_PoAOptimization(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        num_time_steps=HORIZON,
        regime_config_path=str(RUNTIME_CONFIG),
        regime_set=REGIME_SET,
        regime_name=REGIME_NAME,
        epsilon=EPSILON_GRID[0],
        nn_model_dir=str(MODEL_DIR) if nn_generators else None,
        nn_normalization_stats_path=norm_stats,
        nn_policy_generators=nn_generators,
        reference_case=CASE,
        case_label="base_case_tmp",
        objective_mode="piecewise_mccormick",
        mccormick_bounds={
            "PoA": POA_BOUNDS,
            "C_opt": C_OPT_BOUNDS,
            "num_pieces": NUM_PIECES,
        },
        ambiguity_kappa=AMBIGUITY_KAPPA,
        ar1_coverage=AR1_COVERAGE,
    )


def extract_summary(optimizer: DRO_PoAOptimization, eps: float) -> dict[str, Any]:
    from pyomo.environ import value as pyo_value
    m = optimizer.model
    K = optimizer.num_empirical_scenarios

    poa_ratios = []
    wasserstein_vals = []
    for k in range(K):
        c_eq = optimizer._safe_value(m.C_eq[k])
        c_opt = optimizer._safe_value(m.C_opt[k])
        w = optimizer._safe_value(m.wasserstein_distance[k])
        if c_eq is not None and c_opt not in (None, 0.0):
            poa_ratios.append(c_eq / c_opt)
        if w is not None:
            wasserstein_vals.append(w)

    avg_poa = float(np.mean(poa_ratios)) if poa_ratios else None
    avg_w = float(np.mean(wasserstein_vals)) if wasserstein_vals else None
    obj = optimizer._safe_value(m.objective)

    return {
        "epsilon": eps,
        "objective": obj,
        "average_poa_ratio": avg_poa,
        "per_scenario_poa_ratios": poa_ratios,
        "average_wasserstein": avg_w,
        "per_scenario_wasserstein": wasserstein_vals,
        "solve_wall_time_seconds": getattr(optimizer, "solve_wall_time_seconds", None),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figures(summaries: list[dict[str, Any]]) -> None:
    """Generate and save all figures from the sweep summaries collected so far."""
    if not summaries:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epsilons = [s["epsilon"] for s in summaries]
    avg_poa = [s["average_poa_ratio"] for s in summaries]
    avg_w = [s["average_wasserstein"] for s in summaries]
    solve_times = [s.get("solve_wall_time_seconds") for s in summaries]

    # Strip None values for plotting
    valid_poa = [(e, v) for e, v in zip(epsilons, avg_poa) if v is not None]
    valid_w = [(e, v) for e, v in zip(epsilons, avg_w) if v is not None]

    # Per-scenario series (K x len(summaries))
    n_scenarios = max(
        (len(s.get("per_scenario_poa_ratios", [])) for s in summaries), default=0
    )
    per_scenario_poa: list[list[Optional[float]]] = []
    for k in range(n_scenarios):
        series: list[Optional[float]] = []
        for s in summaries:
            ratios = s.get("per_scenario_poa_ratios", [])
            series.append(ratios[k] if k < len(ratios) else None)
        per_scenario_poa.append(series)

    # --- Figure 1: Average PoA ratio vs epsilon ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if valid_poa:
        eps_v, poa_v = zip(*valid_poa)
        ax.plot(eps_v, poa_v, "o-", color="steelblue", linewidth=2, markersize=5,
                label="Average PoA (ratio)")
    ax.set_xlabel("Wasserstein radius $\\varepsilon$")
    ax.set_ylabel("Average PoA")
    ax.set_title("DRO PoA vs Wasserstein Radius")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dro_avg_poa_vs_epsilon.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "dro_avg_poa_vs_epsilon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Per-scenario PoA + average ---
    if n_scenarios > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, max(n_scenarios, 1)))
        for k, (series, color) in enumerate(zip(per_scenario_poa, colors)):
            pts = [(e, v) for e, v in zip(epsilons, series) if v is not None]
            if pts:
                xe, ye = zip(*pts)
                ax.plot(xe, ye, "o-", color=color, linewidth=1,
                        markersize=3, alpha=0.7, label=f"Scenario {k}" if k < 4 else None)
        if valid_poa:
            eps_v, poa_v = zip(*valid_poa)
            ax.plot(eps_v, poa_v, "o-", color="firebrick", linewidth=2.5, markersize=5,
                    label="Average PoA", zorder=5)
        ax.set_xlabel("Wasserstein radius $\\varepsilon$")
        ax.set_ylabel("PoA (ratio $C^\\mathrm{eq}/C^*$)")
        ax.set_title("Per-Scenario PoA vs Wasserstein Radius")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dro_scenario_poa_vs_epsilon.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "dro_scenario_poa_vs_epsilon.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Figure 3: Average Wasserstein used vs epsilon ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if valid_w:
        eps_v, w_v = zip(*valid_w)
        ax.plot(eps_v, w_v, "s-", color="darkorange", linewidth=2, markersize=5,
                label="Avg transport used")
    ax.plot(epsilons, epsilons, "--", color="gray", linewidth=1, label="Budget $\\varepsilon$")
    ax.set_xlabel("Wasserstein radius $\\varepsilon$")
    ax.set_ylabel("Average Wasserstein distance $W$")
    ax.set_title("Transport Used vs Wasserstein Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dro_wasserstein_vs_epsilon.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "dro_wasserstein_vs_epsilon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: Solve time vs epsilon ---
    valid_times = [(e, t) for e, t in zip(epsilons, solve_times) if t is not None]
    if valid_times:
        eps_t, times_t = zip(*valid_times)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(range(len(eps_t)), times_t, color="steelblue", alpha=0.7)
        ax.set_xticks(range(len(eps_t)))
        ax.set_xticklabels([f"{e:.1f}" for e in eps_t], rotation=45, fontsize=7)
        ax.set_xlabel("Wasserstein radius $\\varepsilon$")
        ax.set_ylabel("Solve time (s)")
        ax.set_title("Solve Time per Epsilon")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "dro_solve_time_vs_epsilon.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "dro_solve_time_vs_epsilon.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures updated in {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_epsilon_sweep() -> list[dict[str, Any]]:
    print("=" * 70)
    print("DRO PoA — Epsilon Sweep (constraint-based Wasserstein formulation)")
    print(f"  Regime: {REGIME_SET}/{REGIME_NAME}")
    print(f"  Horizon: T={HORIZON}")
    print(f"  Epsilon grid: {len(EPSILON_GRID)} points in [0, {EPSILON_MAX:.1f}]")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    scenarios = load_scenarios()
    optimizer = build_optimizer(scenarios)

    print(f"\nLoading tightening report: {TIGHTENING_REPORT}")
    optimizer.load_regime_wide_tightening_report(TIGHTENING_REPORT)

    print("Building model (epsilon=0)...")
    optimizer.build_model()
    applied_stats = optimizer.apply_regime_wide_tightening_to_model()
    print(f"  Applied alpha bounds:     {applied_stats['alpha_bounds']}")
    print(f"  Applied fixed binaries:   {applied_stats['fixed_binaries']}")
    print(f"  Applied dual bounds:      {applied_stats['dual_upper_bounds']}")

    print("\nAttaching persistent solver...")
    optimizer.attach_persistent_solver()

    summaries: list[dict[str, Any]] = []
    summary_path = OUTPUT_DIR / "epsilon_sweep_summary.json"
    n = len(EPSILON_GRID)

    for idx, eps in enumerate(EPSILON_GRID):
        is_first = idx == 0
        print(f"\n{'=' * 60}")
        print(f"  Epsilon {idx+1}/{n}: {eps:.4g}  ({'cold start' if is_first else 'warm start'})")
        print(f"{'=' * 60}")

        if not is_first:
            optimizer.update_epsilon(eps)

        start = time.perf_counter()
        solver_result = optimizer.solve(
            time_limit=SOLVER_TIME_LIMIT,
            warm_start=True,
        )
        elapsed = time.perf_counter() - start

        termination = solver_result.solver.termination_condition
        print(f"  Termination: {termination}  |  Wall time: {elapsed:.1f}s")

        summary = extract_summary(optimizer, eps)
        summary["termination"] = str(termination)
        summaries.append(summary)

        print(f"  Avg PoA ratio:    {summary['average_poa_ratio']}")
        print(f"  Avg Wasserstein:  {summary['average_wasserstein']}")

        # Save full results per epsilon
        out_path = result_path(eps)
        optimizer.save_results(out_path)

        # Save summary and figures incrementally so partial runs are useful
        save_json(summary_path, summaries)
        make_figures(summaries)

    print(f"\nSaved sweep summary: {summary_path}")
    return summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DRO PoA epsilon sweep")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from an existing epsilon_sweep_summary.json without solving",
    )
    args = parser.parse_args()

    if args.figures_only:
        summary_path = OUTPUT_DIR / "epsilon_sweep_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary found at {summary_path}")
        with summary_path.open("r", encoding="utf-8") as fh:
            summaries = json.load(fh)
        print(f"Loaded {len(summaries)} epsilon results from {summary_path}")
        make_figures(summaries)
    else:
        summaries = run_epsilon_sweep()

    print("\nDone.")
