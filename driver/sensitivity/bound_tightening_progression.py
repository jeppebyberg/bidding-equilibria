"""Sensitivity study: PoA bound-tightening progression (tightening vs. solve time), at T=6.

Runs the base-case learned-policy configuration through five cumulative tightening
settings, recomputing the PoA tightening reports and re-solving the base PoA MILP
for each:

  1. loose         -- no optional stages (loose ReLU/alpha/dual defaults)
  2. relu          -- + ReLU preactivation bounds
  3. alpha         -- + alpha bounds (also triggers analytical cost-ratio PoA bound kappa)
  4. slack         -- + slack-based complementarity binary fixing
  5. dual          -- + dual Big-M tightening                   <- base case

Each step adds exactly one optional stage on top of the previous one, so the
marginal cost (tightening time) and benefit (smaller / faster MILP) of every
stage is isolated. The equilibrium_cost_bounds stage has been removed; the
cost-ratio PoA upper bound (kappa) is now derived analytically from the alpha
bounds and applied automatically in every case that includes alpha_bounds.
Only the PoA formulation is exercised here -- the DRO half is intentionally
left out for now.

Upstream data, features, and trained policies are reused from the base-case run
(results/base_case); only the tightening reports and the final PoA solve are
recomputed, in study-local result folders:
  results/sensitivity_studies/bound_tightening_progression/T6/<case>/poa/

Each run records the two times this study compares:
  - poa/tightening/final_tightening_report.json -> ``stage_timings`` (per-stage
    tightening wall time + which stages ran) = time spent on tightening
  - poa/poa_optimization_T*.json                -> ``solver.wall_time_seconds``
    = PoA solve / computation time (plus variable_counts and objective.PoA)

Companion summary (table + CSV/JSON + plots):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.bound_tightening_progression_summary

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.bound_tightening_progression
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver import block3_poa_pipeline  # noqa: E402
from driver.project_config import load_project_config  # noqa: E402

STUDY_NAME = "bound_tightening_progression"
RESULT_ROOT = PROJECT_ROOT / "results" / "sensitivity_studies" / STUDY_NAME

# Base horizon. Only T=6 is investigated; T=8 is intentionally excluded.
HORIZON = 6

# Upstream artifact root: the base-case run, which trained the NN policies and
# generated the scenarios/features. The progression reuses these directly and
# recomputes only the tightening reports (per flag set) and the final PoA solve.
UPSTREAM_ROOT = PROJECT_ROOT / "results" / "base_case"

# Optional tightening stages added one at a time, in dependency order.
# Note: alpha_bounds also triggers the analytical cost-ratio PoA bound (kappa)
# automatically -- no separate equilibrium_cost_bounds stage is needed.
CUMULATIVE_STAGES = (
    "relu_bounds",
    "alpha_bounds",
    "slack_binary_fix",
    "dual_big_m",
)

# Cumulative cases: (case_name, label, enabled-optional-stages). Each case turns
# on a prefix of CUMULATIVE_STAGES; the always-on primal_big_m and
# optimal_cost_bounds stages run in every case.
TIGHTENING_CASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("S0_loose", "loose (no optional stages)", ()),
    ("S1_relu", "+ ReLU bounds", ("relu_bounds",)),
    ("S2_alpha", "+ alpha bounds (+ kappa PoA bound)", ("relu_bounds", "alpha_bounds")),
    (
        "S3_slack",
        "+ slack binary fix",
        ("relu_bounds", "alpha_bounds", "slack_binary_fix"),
    ),
    (
        "S4_dual",
        "+ dual Big-M (base case)",
        ("relu_bounds", "alpha_bounds", "slack_binary_fix", "dual_big_m"),
    ),
)

# Cases this run actually (re)solves. Subset of TIGHTENING_CASES so a run can
# fill in missing cases without re-solving the ones already on disk.
CASES_TO_RUN = [name for name, _label, _stages in TIGHTENING_CASES]

# Per-run PoA solve time limit (seconds). Runs that hit this cap report the
# incumbent and MIP gap instead of a proven optimum; the summary flags them via
# ``termination_condition``. None = solve every run to global optimality.
POA_TIME_LIMIT: int | None = 3000

# Pin Gurobi Threads/Seed for the final PoA solve so the cases are comparable
# 1:1. Multi-threaded Gurobi has a nondeterministic search path, so solve-time
# differences across cases would otherwise reflect solver variability rather
# than the tightening under study. Single-thread + fixed seed makes each solve
# deterministic; the cost is a slower (but apples-to-apples) solve per case.
POA_SOLVER_THREADS: int | None = 1
POA_SOLVER_SEED: int | None = 0


def horizon_name(horizon: int = HORIZON) -> str:
    return f"T{horizon}"


def run_dir(case_name: str) -> Path:
    return RESULT_ROOT / horizon_name() / case_name


def _build_flags(enabled_stages: tuple[str, ...]) -> dict[str, bool]:
    """Full PoA tightening flag dict for one cumulative case.

    primal_big_m and optimal_cost_bounds are always on; the optional stages are
    enabled iff they appear in this case's prefix of CUMULATIVE_STAGES.
    """
    flags = {"primal_big_m": True, "optimal_cost_bounds": True}
    for stage in CUMULATIVE_STAGES:
        flags[stage] = stage in enabled_stages
    return flags


def _validate_cumulative_flags(flags: dict[str, bool]) -> None:
    """Enforce the stage dependency chain so each case is internally consistent."""
    if flags.get("alpha_bounds") and not flags.get("relu_bounds"):
        raise ValueError("alpha_bounds=True requires relu_bounds=True")
    if flags.get("slack_binary_fix") and not flags.get("alpha_bounds"):
        raise ValueError("slack_binary_fix=True requires alpha_bounds=True")
    if flags.get("dual_big_m") and not flags.get("slack_binary_fix"):
        raise ValueError("dual_big_m=True requires slack_binary_fix=True")


def build_run_config(case_name: str, flags: dict[str, bool]):
    """Base config re-pointed to recompute one tightening case in an isolated dir.

    Upstream data/feature/policy dirs point at the base-case run so every case
    shares identical inputs; only the tightening reports and the final PoA
    outputs are isolated, and only ``poa_tightening_flags`` varies across cases.
    """
    cfg = load_project_config()
    out_dir = run_dir(case_name)

    cfg.case_label = f"{STUDY_NAME}/{horizon_name()}/{case_name}"
    # ``horizon`` set post-construction does not re-derive synthetic_time_steps;
    # set it explicitly so the PoA context scenario matches the horizon.
    cfg.horizon = int(HORIZON)
    cfg.synthetic_time_steps = int(HORIZON)

    # Reuse the base-case upstream artifacts (policies, norm stats, features).
    cfg.model_dir = UPSTREAM_ROOT / "trained_models"
    cfg.training_result_dir = UPSTREAM_ROOT / "training_results"
    cfg.raw_feature_dir = UPSTREAM_ROOT / "features" / "raw"
    cfg.normalized_feature_dir = UPSTREAM_ROOT / "features" / "normalized"
    cfg.synthetic_scenario_dir = UPSTREAM_ROOT / "synthetic_scenarios"
    cfg.heuristic_results_path = UPSTREAM_ROOT / "merit_order_results.json"

    # Isolate just the outputs this case produces.
    cfg.poa_result_dir = out_dir / "poa"
    cfg.figures_dir = out_dir / "figures"
    cfg.poa_scenario_dir = out_dir / "poa_scenarios"
    cfg.runtime_config_path = out_dir / "runtime_regime_definitions.yaml"

    # The field under study: recompute tightening with exactly this flag set.
    cfg.poa_tightening_flags = flags

    # Reuse upstream data; recompute tightening + final PoA solve only.
    cfg.run_scenario_generation = False
    cfg.run_heuristic_labels = False
    cfg.run_feature_building = False
    cfg.run_nn_training = False
    cfg.run_poa_tightening = True
    cfg.run_poa_optimization = True
    # Base PoA only -- keep the DRO half of the pipeline off.
    cfg.run_dro_tightening = False
    cfg.run_dro_optimization = False

    cfg.poa_time_limit = POA_TIME_LIMIT
    # Pin solver threads/seed so every case solves under identical, deterministic
    # conditions -- the prerequisite for reading solve times as 1:1 comparable.
    cfg.poa_solver_threads = POA_SOLVER_THREADS
    cfg.poa_solver_seed = POA_SOLVER_SEED
    # Summary builds its own cross-run plots; skip per-run plotting overhead.
    cfg.plot_results_along_the_way = False
    return cfg


def _clear_tightening_dir(cfg) -> None:
    """Remove any prior tightening reports so disabled stages use loose defaults.

    A stage that is off in this case must fall back to its loose default rather
    than load a stale report left by an earlier run of the same case dir. Wiping
    the dir guarantees enabled stages are recomputed and disabled ones are loose.
    """
    tightening_dir = Path(cfg.poa_result_dir) / "tightening"
    if tightening_dir.exists():
        shutil.rmtree(tightening_dir)


def run() -> dict[str, Any]:
    cases = [case for case in TIGHTENING_CASES if case[0] in CASES_TO_RUN]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  PoA bound-tightening progression ({len(cases)} case(s))")
    print(f"  result_root: {RESULT_ROOT}")
    print(f"  horizon: T={HORIZON}")
    print(f"  upstream: {UPSTREAM_ROOT}")
    print(f"  cases: {[name for name, _l, _s in cases]}")
    print(f"{sep}")

    manifests: dict[str, Any] = {}
    for idx, (name, label, enabled_stages) in enumerate(cases, start=1):
        print(f"\n{sep}")
        print(f"  [{idx}/{len(cases)}] {name}: {label}")
        print(f"{sep}")
        flags = _build_flags(enabled_stages)
        _validate_cumulative_flags(flags)
        cfg = build_run_config(name, flags)
        _clear_tightening_dir(cfg)
        manifest = block3_poa_pipeline.run(cfg)
        manifests[name] = manifest
        print(
            f"\n  [{name}] tightening: "
            f"{manifest.get('tightening_wall_time_seconds', float('nan')):.2f}s"
            f"  |  PoA solve: {manifest.get('solve_wall_time_seconds', float('nan')):.2f}s"
        )

    print(f"\n{sep}")
    print("  Progression complete. Building tightening/compute-time summary...")
    print(f"{sep}")
    # Imported here to avoid a circular import (the summary imports this module).
    from driver.sensitivity.summaries import bound_tightening_progression_summary

    bound_tightening_progression_summary.main()
    return manifests


if __name__ == "__main__":
    run()
