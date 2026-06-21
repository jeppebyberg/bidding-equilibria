# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-16
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Sensitivity study: PoA solve cost vs. number of McCormick pieces, at T=6.

Justifies the choice of ``poa_mccormick_num_pieces``. The number of pieces in
the piecewise-McCormick relaxation of ``C_eq = PoA * C_opt`` is the only thing
that changes across runs: more pieces tighten the relaxation (smaller ex-post
gap) at the cost of more binaries and solve time. Everything upstream -- the
scenarios, heuristic labels, features, trained policies, and the tightening
bounds (ReLU big-M, dual big-M, alpha bounds, and the C_opt range) -- is held
fixed by reusing the base-case run's artifacts, so each run differs only in how
finely the same C_opt interval is subdivided.

The study is run only at the base horizon T=6, reusing the base-case NN policies
and tightening reports directly (results/base_case). T=8 is intentionally not
investigated.

Because the breakpoints are ``linspace(C_opt_L, C_opt_U, num_pieces + 1)``, the
C_opt range comes straight from the base case's ``optimal_cost_bounds`` report
and is identical for every piece count; only the subdivision count varies. The
McCormick PoA box upper bound is likewise derived from that report at solve time
(PoA_U = C_opt_max / C_opt_min), so it too is identical across piece counts.

Each run reuses (with <T> = T6):
  - trained policies      (results/base_case/trained_models)
  - features / norm stats (results/base_case/features)
  - tightening reports    (results/base_case/poa/tightening)
and writes an isolated PoA result to:
  results/sensitivity_studies/mccormick_pieces_sweep/<T>/P<n>/poa/

Each run's ``poa/poa_optimization_T*.json`` records:
  - solver.wall_time_seconds            -> compute time
  - solver.variable_counts.*            -> binary-variable counts
  - objective.PoA                       -> relaxed (upper-bound) PoA
  - objective.ex_post_ratio             -> realized C_eq / C_opt
  - objective.mccormick_gap             -> PoA - ex_post_ratio (relaxation gap)

Companion summary (table + CSV/JSON + plots):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.mccormick_pieces_summary

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.mccormick_pieces_sweep
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

STUDY_NAME = "mccormick_pieces_sweep"
RESULT_ROOT = PROJECT_ROOT / "results" / "sensitivity_studies" / STUDY_NAME

# Base horizon. Only T=6 is investigated; T=8 is intentionally excluded.
HORIZONS = [6]

# Horizons this run actually (re)solves. Subset of HORIZONS so a run can fill
# in missing horizons without re-solving the ones already on disk.
HORIZONS_TO_RUN = list(HORIZONS)

# Per-horizon upstream artifact root: the base-case run, which trained the NN
# policies and computed the tightening reports. The sweep reuses these directly
# so every piece count solves against the same base-case inputs.
UPSTREAM_ROOTS = {
    6: PROJECT_ROOT / "results" / "base_case",
}

# Piece counts under study. Brackets the chosen value (50) on both sides so the
# companion summary can show the compute/binary cost rising while the ex-post
# relaxation gap flattens out.
PIECES = [5, 10, 20, 30, 50, 75, 100, 200, 300, 500]

# Piece counts this run actually (re)solves. Subset of PIECES so a run can fill
# in missing points without re-solving the ones already on disk. This run only
# adds the high-resolution 300 and 500 points; the rest are already on disk.
PIECES_TO_RUN = [300, 500]

# Per-run PoA solve time limit (seconds). Runs that hit this cap report the
# incumbent and MIP gap instead of a proven optimum; the summary flags them via
# ``termination_condition``. None = solve every run to global optimality.
POA_TIME_LIMIT: int | None = 3000

# Pin Gurobi Threads/Seed for the final PoA solve so the cases are comparable
# 1:1. Multi-threaded Gurobi has a nondeterministic search path, so solve-time
# differences across cases would otherwise reflect solver variability rather
# than the piece count under study. Single-thread + fixed seed makes each solve
# deterministic; the cost is a slower (but apples-to-apples) solve per case.
POA_SOLVER_THREADS: int | None = 1
POA_SOLVER_SEED: int | None = 0


def horizon_name(horizon: int) -> str:
    return f"T{horizon}"


def run_name(num_pieces: int) -> str:
    return f"P{num_pieces}"


def run_label(horizon: int, num_pieces: int) -> str:
    return f"T={horizon}, {num_pieces} pieces"


def run_dir(horizon: int, num_pieces: int) -> Path:
    return RESULT_ROOT / horizon_name(horizon) / run_name(num_pieces)


def build_run_config(horizon: int, num_pieces: int):
    """Base config re-pointed to solve one (horizon, piece-count) in an isolated dir.

    Upstream artifact dirs (policies, features, synthetic scenarios, labels,
    tightening reports) are re-pointed to that horizon's
    horizon_poa_compute_sweep run so every piece count at a given horizon reuses
    identical inputs. Only the PoA outputs and the regenerated (deterministic)
    PoA context scenarios are isolated, and only ``poa_mccormick_num_pieces``
    varies within a horizon.
    """
    upstream = UPSTREAM_ROOTS[horizon]
    cfg = load_project_config()
    out_dir = run_dir(horizon, num_pieces)

    cfg.case_label = f"{STUDY_NAME}/{horizon_name(horizon)}/{run_name(num_pieces)}"
    # ``horizon`` set post-construction does not re-derive synthetic_time_steps;
    # set it explicitly so the PoA context scenario matches the horizon.
    cfg.horizon = int(horizon)
    cfg.synthetic_time_steps = int(horizon)

    # Reuse this horizon's upstream artifacts (policies, norm stats, features).
    cfg.model_dir = upstream / "trained_models"
    cfg.training_result_dir = upstream / "training_results"
    cfg.raw_feature_dir = upstream / "features" / "raw"
    cfg.normalized_feature_dir = upstream / "features" / "normalized"
    cfg.synthetic_scenario_dir = upstream / "synthetic_scenarios"
    cfg.heuristic_results_path = upstream / "merit_order_results.json"

    # Isolate just the outputs this run produces.
    cfg.poa_result_dir = out_dir / "poa"
    cfg.figures_dir = out_dir / "figures"
    cfg.poa_scenario_dir = out_dir / "poa_scenarios"
    cfg.runtime_config_path = out_dir / "runtime_regime_definitions.yaml"

    # The single field under study.
    cfg.poa_mccormick_num_pieces = int(num_pieces)

    # Reuse all upstream artifacts; solve only the final PoA MILP.
    cfg.run_scenario_generation = False
    cfg.run_heuristic_labels = False
    cfg.run_feature_building = False
    cfg.run_nn_training = False
    cfg.run_poa_tightening = False
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


def _seed_tightening_reports(cfg, horizon: int) -> None:
    """Copy this horizon's tightening reports into the run's PoA dir.

    block3 always recomputes the cheap ``primal_big_m`` and ``optimal_cost_bounds``
    stages but reuses the optional stages (ReLU/dual/alpha/slack big-Ms) when their
    reports are already present. Seeding the run dir with the per-horizon reports
    makes every piece count use the same tight bounds, so binary counts and solve
    times reflect the chosen number of pieces rather than differences in tightening.
    """
    source = UPSTREAM_ROOTS[horizon] / "poa" / "tightening"
    if not source.exists():
        raise FileNotFoundError(
            f"Tightening reports for T={horizon} not found: {source}. Run the base-case "
            "pipeline (block3 PoA tightening) before running this sweep."
        )
    target = Path(cfg.poa_result_dir) / "tightening"
    target.mkdir(parents=True, exist_ok=True)
    for report in source.glob("*.json"):
        shutil.copy2(report, target / report.name)


def run() -> dict[str, Any]:
    total = len(HORIZONS_TO_RUN) * len(PIECES_TO_RUN)
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  McCormick-pieces sweep ({total} run(s))")
    print(f"  result_root: {RESULT_ROOT}")
    print(f"  horizons: {HORIZONS_TO_RUN}")
    print(f"  pieces: {PIECES_TO_RUN}")
    print(f"{sep}")

    manifests: dict[str, Any] = {}
    idx = 0
    for horizon in HORIZONS_TO_RUN:
        for num_pieces in PIECES_TO_RUN:
            idx += 1
            print(f"\n{sep}")
            print(f"  [{idx}/{total}] {run_label(horizon, num_pieces)}")
            print(f"{sep}")
            cfg = build_run_config(horizon, num_pieces)
            _seed_tightening_reports(cfg, horizon)
            key = f"{horizon_name(horizon)}/{run_name(num_pieces)}"
            manifests[key] = block3_poa_pipeline.run(cfg)

    print(f"\n{sep}")
    print("  Sweep complete. Building cross-pieces comparison...")
    print(f"{sep}")
    # Imported here to avoid a circular import (the summary imports this module).
    from driver.sensitivity.summaries import mccormick_pieces_summary

    mccormick_pieces_summary.main()
    return manifests


if __name__ == "__main__":
    run()
