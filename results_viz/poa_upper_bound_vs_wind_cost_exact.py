"""Exact inserted PoA upper bound vs wind cost (one optimization per cost).

Unlike ``poa_upper_bound_vs_wind_cost.py`` (which re-prices a single solved
state), this recomputes the McCormick PoA box upper bound the same way the
tightening pipeline does:

    PoA_U(c) = max C_eq(c) / min C_opt(c)

over the regime-induced support set, with the embedded NN/true-cost bidding
policy and the equilibrium KKT block (see
models/PoA/PoA_tightening/compute_equilibrium_cost_bounds.py and
compute_optimal_cost_bounds.py). For each shared wind cost ``c`` it solves:

  * one ``max C_eq`` MILP (EquilibriumCostBoundsComputer), and
  * one ``min C_opt`` MILP (OptimalCostBoundsComputer, lower bound only),

then combines them with ``derive_poa_upper_bound``. The cost-independent stages
(primal Big-M, ReLU bounds, alpha bounds, fixed binaries, tight Big-M) come from
the wind run's saved final tightening report and are reused for every cost --
only the cost objective changes with ``c``, never the feasible set, so the wind
true cost is applied by overwriting the wind entries of ``poa.block_cost_vector``
between solves.

The trained policies and tightening reports are read from the wind-playing
sensitivity run (results/sensitivity_studies/wind_playing_sweep/wind). Scenarios
and costs are regenerated deterministically from the ScenarioManager, so the
``c == base wind cost`` point reproduces the run's saved ``derived_poa_bounds``.

Solves are persisted to a JSON sidecar so the plot can be regenerated without
re-solving. Edit the settings in ``main()`` and run:
  .\\.venv\\Scripts\\python.exe -m results_viz.poa_upper_bound_vs_wind_cost_exact
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import (
    Constraint,
    Objective,
    SolverFactory,
    TerminationCondition,
    maximize,
    minimize,
    value,
)

from driver.core.block3_core import build_poa_tightening
from driver.sensitivity.sensitivity_config import build_sensitivity_config
from driver.sensitivity.wind_playing_sweep import build_study
from models.helper import derive_poa_upper_bound
from models.PoA.PoA_tightening.compute_equilibrium_cost_bounds import (
    EquilibriumCostBoundsComputer,
)
from models.PoA.PoA_tightening.compute_optimal_cost_bounds import OptimalCostBoundsComputer

_OPTIMAL = {TerminationCondition.optimal, TerminationCondition.locallyOptimal}


def _solver(solver_name: str, time_limit: float | None, solver_threads: int | None):
    solver = SolverFactory(solver_name)
    if time_limit is not None:
        solver.options["TimeLimit"] = float(time_limit)
    if solver_name == "gurobi":
        if solver_threads is not None:
            solver.options["Threads"] = int(solver_threads)
        # DualReductions=0 keeps Gurobi presolve from collapsing a bounded optimum
        # to "infeasibleOrUnbounded" on these KKT models.
        solver.options["DualReductions"] = 0
    return solver


def _safe_lower(raw_lower: float) -> float:
    """Match OptimalCostBoundsComputer._safe_bounds for the lower (min C_opt) side."""
    if raw_lower <= 0.0:
        raise ValueError("Computed C_opt lower bound must be strictly positive")
    return max(1e-5, raw_lower - abs(raw_lower) * 1e-6 - 1e-6)


def build_wind_run_tightening(run_name: str = "wind"):
    """Rebuild the wind-run tightening object with its saved cost-independent stages.

    Returns ``(tightening, cfg, wind_global_block_indices, base_wind_cost)``.
    """
    study = build_study()
    run = next(r for r in study.runs if r.name == run_name)
    cfg = build_sensitivity_config(study, run)
    # Keep the run's actual objective mode (piecewise_mccormick): the KKT cost-bound
    # models are built independently of the objective, but the policy/KKT setup
    # relies on the mode the saved tightening was produced under.

    # Use exactly the generators whose policies were actually trained/saved for
    # this run (training may drop generators with too little label variation, so
    # the run's NN set can be a subset of all physical generators). The remaining
    # generators bid at true cost, matching how the saved tightening was built.
    trained = sorted(
        path.name[: -len("_policy_weights.json")]
        for path in Path(cfg.model_dir).glob("*_policy_weights.json")
    )
    if not trained:
        raise FileNotFoundError(f"No trained policy weights found in {cfg.model_dir}.")
    cfg.nn_policy_generators = trained

    tightening = build_poa_tightening(cfg)
    final_report_path = Path(cfg.poa_result_dir) / "tightening" / "final_tightening_report.json"
    if not final_report_path.exists():
        raise FileNotFoundError(
            f"Final tightening report not found for run '{run_name}': {final_report_path}. "
            "Run the wind_playing_sweep PoA tightening first."
        )
    # Loads primal_big_m, nn_relu_bounds, alpha_bounds, fixed_binaries, tight_big_m
    # into tightening.poa -- all cost-independent and reused across every wind cost.
    tightening._merge_report(tightening._load_json(final_report_path))

    poa = tightening.poa
    wind_global_block_indices = [
        int(poa.local_to_global_block[(int(i), int(b))]) for (i, b) in poa.wind_block_pairs
    ]
    base_wind_cost = float(
        np.mean([poa.block_cost_vector[g] for g in wind_global_block_indices])
    )
    return tightening, cfg, wind_global_block_indices, base_wind_cost


def _wind_cost_coeffs(poa: Any, wind_global_block_indices: list[int], wind_cost: float):
    """True-cost vector with the wind blocks overwritten by ``wind_cost``."""
    wind_set = set(wind_global_block_indices)

    def coef(i: int, b: int) -> float:
        g = int(poa.local_to_global_block[(int(i), int(b))])
        return float(wind_cost) if g in wind_set else float(poa.block_cost_vector[g])

    return coef


def _solve_max_c_eq_repriced(
    tightening: Any,
    wind_global_block_indices: list[int],
    wind_cost: float,
    solver_name: str,
    time_limit: float | None,
    solver_threads: int | None,
) -> dict[str, Any]:
    """max C_eq over the support set, holding the trained equilibrium bids fixed.

    The equilibrium KKT block is built at the ORIGINAL true costs so every certified
    input (alpha bounds, fixed binaries, Big-M) stays valid; only the C_eq cost
    objective is re-priced at ``wind_cost`` for the wind blocks.
    """
    eq_stage = tightening._as_stage(EquilibriumCostBoundsComputer)
    m = eq_stage._build_equilibrium_kkt_bound_model()
    coef = _wind_cost_coeffs(tightening.poa, wind_global_block_indices, wind_cost)

    m.del_component(m.cost_definition_eq)
    m.cost_definition_eq = Constraint(
        expr=m.C_eq
        == sum(coef(i, b) * m.P_eq[i, b, t] for (i, b) in m.generator_blocks for t in m.time_steps)
    )
    m.tightening_objective = Objective(expr=m.C_eq, sense=maximize)

    results = _solver(solver_name, time_limit, solver_threads).solve(m, tee=False)
    termination = results.solver.termination_condition
    if termination not in _OPTIMAL:
        raise RuntimeError(f"max C_eq did not solve to optimality (termination={termination}).")
    raw = float(value(m.C_eq))
    return {"raw": raw, "safe": raw + abs(raw) * 1e-6 + 1e-6, "termination": str(termination)}


def _solve_min_c_opt_repriced(
    tightening: Any,
    wind_global_block_indices: list[int],
    wind_cost: float,
    solver_name: str,
    time_limit: float | None,
    solver_threads: int | None,
) -> dict[str, Any]:
    """min C_opt over the support set with wind re-priced at ``wind_cost``.

    The optimal-dispatch KKT block uses the true cost in both stationarity and the
    objective (and carries no certified binaries), so the wind cost is applied by
    overwriting block_cost_vector for the duration of this solve -- the social
    optimum genuinely re-optimizes dispatch at the new cost.
    """
    poa = tightening.poa
    saved = {g: poa.block_cost_vector[g] for g in wind_global_block_indices}
    for g in wind_global_block_indices:
        poa.block_cost_vector[g] = float(wind_cost)
    try:
        opt_stage = tightening._as_stage(OptimalCostBoundsComputer)
        m = opt_stage._build_optimal_dispatch_kkt_bound_model()
        m.tightening_objective = Objective(expr=m.C_opt, sense=minimize)
        results = _solver(solver_name, time_limit, solver_threads).solve(m, tee=False)
        termination = results.solver.termination_condition
        if termination not in (_OPTIMAL | {TerminationCondition.feasible}):
            raise RuntimeError(
                f"min C_opt did not solve to optimality (termination={termination})."
            )
        raw = float(value(m.C_opt))
    finally:
        for g, v in saved.items():
            poa.block_cost_vector[g] = v
    return {"raw": raw, "safe": _safe_lower(raw), "termination": str(termination)}


def compute_bound_for_cost(
    tightening: Any,
    wind_global_block_indices: list[int],
    wind_cost: float,
    solver_name: str,
    time_limit: float | None,
    solver_threads: int | None,
    poa_bounds_margin: float,
) -> dict[str, Any]:
    """Solve max C_eq and min C_opt at one wind cost and derive PoA_U."""
    eq = _solve_max_c_eq_repriced(
        tightening, wind_global_block_indices, wind_cost, solver_name, time_limit, solver_threads
    )
    opt = _solve_min_c_opt_repriced(
        tightening, wind_global_block_indices, wind_cost, solver_name, time_limit, solver_threads
    )
    _poa_lower, poa_upper = derive_poa_upper_bound(
        eq["safe"], opt["safe"], lower=1.0, margin=poa_bounds_margin
    )
    return {
        "wind_cost": float(wind_cost),
        "max_C_eq": eq["safe"],
        "min_C_opt": opt["safe"],
        "PoA_upper": float(poa_upper),
        "eq_termination": eq["termination"],
        "opt_termination": opt["termination"],
    }


def run_sweep(
    wind_costs: list[float],
    output_json: Path,
    solver_name: str = "gurobi",
    time_limit: float | None = 200.0,
    solver_threads: int | None = 1,
    run_name: str = "wind",
) -> dict[str, Any]:
    tightening, cfg, wind_idx, base_wind_cost = build_wind_run_tightening(run_name)
    margin = float(getattr(cfg, "poa_bounds_derivation_margin", 1e-3))

    print(f"Wind run base wind cost = {base_wind_cost:.4g}; sweeping {len(wind_costs)} cost(s).")
    points: list[dict[str, Any]] = []
    for idx, c in enumerate(wind_costs, start=1):
        start = time.perf_counter()
        point = compute_bound_for_cost(
            tightening, wind_idx, c, solver_name, time_limit, solver_threads, margin
        )
        point["solve_seconds"] = time.perf_counter() - start
        points.append(point)
        print(
            f"  [{idx}/{len(wind_costs)}] c={c:.4g} -> "
            f"max C_eq={point['max_C_eq']:.6g}, min C_opt={point['min_C_opt']:.6g}, "
            f"PoA_U={point['PoA_upper']:.6g}  ({point['solve_seconds']:.1f}s)"
        )

    payload = {
        "run_name": run_name,
        "base_wind_cost": base_wind_cost,
        "poa_bounds_derivation_margin": margin,
        "solver": {"name": solver_name, "time_limit": time_limit, "threads": solver_threads},
        "points": points,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[saved] {output_json}")
    return payload


def plot_sweep(payload: dict[str, Any], output_dir: Path, show: bool = False) -> Path:
    points = sorted(payload["points"], key=lambda p: p["wind_cost"])
    costs = np.array([p["wind_cost"] for p in points])
    poa_upper = np.array([p["PoA_upper"] for p in points])
    c_eq = np.array([p["max_C_eq"] for p in points])
    c_opt = np.array([p["min_C_opt"] for p in points])
    base_wind_cost = float(payload["base_wind_cost"])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(costs, poa_upper, color="tab:blue", marker="o", linewidth=2.2, label=r"$PoA_U(c)=\max C_{eq}/\min C_{opt}$")
    ax.axvline(base_wind_cost, color="gray", linestyle=":", linewidth=1.4,
               label=f"as-trained wind cost = {base_wind_cost:.3g}")
    ax.set_ylabel("inserted PoA upper bound")
    ax.set_title("Exact inserted PoA upper bound vs shared wind cost\n(wind-playing run; one optimization per cost)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(costs, c_eq, color="tab:red", marker="o", linewidth=2.0, label=r"$\max C_{eq}(c)$")
    ax.plot(costs, c_opt, color="tab:green", marker="s", linewidth=2.0, label=r"$\min C_{opt}(c)$")
    ax.axvline(base_wind_cost, color="gray", linestyle=":", linewidth=1.4)
    ax.set_xlabel("shared wind cost  c")
    ax.set_ylabel("cost")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path = output_dir / "poa_upper_bound_vs_wind_cost_exact.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    # Shared wind costs to evaluate (each is two MILP solves). Focused on the
    # cheap-wind regime 0..2.5 where the bound drops sharply.
    wind_costs = [round(0.25 * k, 3) for k in range(11)]  # 0.0, 0.25, ..., 2.5
    run_name = "wind"
    output_json = Path("results_viz/figures/poa_upper_bound_vs_wind_cost_exact/sweep_results.json")
    output_dir = output_json.parent
    solver_name = "gurobi"
    time_limit: float | None = 200.0
    solver_threads: int | None = 1

    payload = run_sweep(
        wind_costs=wind_costs,
        output_json=output_json,
        solver_name=solver_name,
        time_limit=time_limit,
        solver_threads=solver_threads,
        run_name=run_name,
    )
    plot_sweep(payload, output_dir)


if __name__ == "__main__":
    main()
