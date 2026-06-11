"""Sensitivity study: PoA solve cost vs. number of McCormick pieces.

Justifies the choice of ``poa_mccormick_num_pieces``. The number of pieces in
the piecewise-McCormick relaxation of ``C_eq = PoA * C_opt`` is the only thing
that changes across runs: more pieces tighten the relaxation (smaller ex-post
gap) at the cost of more binaries and solve time. Everything upstream -- the
scenarios, heuristic labels, features, trained policies, and the tightening
bounds (ReLU big-M, dual big-M, alpha bounds, and the C_opt range) -- is held
fixed by reusing the locked base-case artifacts, so each run differs only in how
finely the same C_opt interval is subdivided.

Because the breakpoints are ``linspace(C_opt_L, C_opt_U, num_pieces + 1)``, the
C_opt range comes straight from the base-case ``optimal_cost_bounds`` report and
is identical for every run; only the subdivision count varies.

Each run reuses:
  - base-case trained policies      (results/<base>/trained_models)
  - base-case features / norm stats (results/<base>/features)
  - base-case tightening reports    (results/<base>/poa/tightening)  [copied in]
and writes an isolated PoA result to:
  results/sensitivity_studies/mccormick_pieces_sweep/P<n>/poa/

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

# Piece counts under study. Brackets the chosen value (50) on both sides so the
# companion summary can show the compute/binary cost rising while the ex-post
# relaxation gap flattens out.
PIECES = [5, 10, 20, 30, 50, 75, 100]

# Piece counts this run actually (re)solves. Subset of PIECES so a run can fill
# in missing points without re-solving the ones already on disk.
PIECES_TO_RUN = list(PIECES)

# Per-run PoA solve time limit (seconds). Runs that hit this cap report the
# incumbent and MIP gap instead of a proven optimum; the summary flags them via
# ``termination_condition``. None = solve every run to global optimality.
POA_TIME_LIMIT: int | None = 3000


def run_name(num_pieces: int) -> str:
    return f"P{num_pieces}"


def run_label(num_pieces: int) -> str:
    return f"{num_pieces} pieces"


def _base_tightening_dir() -> Path:
    """Tightening reports of the locked base case (the shared bounds source)."""
    return Path(load_project_config().poa_result_dir) / "tightening"


def build_run_config(num_pieces: int):
    """Base config re-pointed to solve one piece-count in an isolated dir.

    Upstream artifact dirs (policies, features, synthetic scenarios, labels) are
    left at their base-case defaults so every run reuses identical inputs. Only
    the PoA outputs and the regenerated (deterministic) PoA context scenarios are
    isolated, and only ``poa_mccormick_num_pieces`` is varied.
    """
    cfg = load_project_config()
    run_dir = RESULT_ROOT / run_name(num_pieces)

    # Isolate just the outputs this run produces; everything else stays shared.
    cfg.case_label = f"{STUDY_NAME}/{run_name(num_pieces)}"
    cfg.poa_result_dir = run_dir / "poa"
    cfg.figures_dir = run_dir / "figures"
    cfg.poa_scenario_dir = run_dir / "poa_scenarios"
    cfg.runtime_config_path = run_dir / "runtime_regime_definitions.yaml"

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
    # Summary builds its own cross-run plots; skip per-run plotting overhead.
    cfg.plot_results_along_the_way = False
    return cfg


def _seed_tightening_reports(cfg) -> None:
    """Copy the base-case tightening reports into this run's PoA dir.

    block3 always recomputes the cheap ``primal_big_m`` and ``optimal_cost_bounds``
    stages but reuses the optional stages (ReLU/dual/alpha/slack big-Ms) when their
    reports are already present. Seeding the run dir with the base-case reports
    makes every run use the same tight bounds, so binary counts and solve times
    reflect the chosen number of pieces rather than differences in tightening.
    """
    source = _base_tightening_dir()
    if not source.exists():
        raise FileNotFoundError(
            f"Base-case tightening reports not found: {source}. Generate the base "
            "case (with run_poa_tightening=True) before running this sweep."
        )
    target = Path(cfg.poa_result_dir) / "tightening"
    target.mkdir(parents=True, exist_ok=True)
    for report in source.glob("*.json"):
        shutil.copy2(report, target / report.name)


def run() -> dict[str, Any]:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  McCormick-pieces sweep ({len(PIECES_TO_RUN)} run(s))")
    print(f"  result_root: {RESULT_ROOT}")
    print(f"  pieces: {PIECES_TO_RUN}")
    print(f"{sep}")

    manifests: dict[str, Any] = {}
    for idx, num_pieces in enumerate(PIECES_TO_RUN, start=1):
        print(f"\n{sep}")
        print(f"  [{idx}/{len(PIECES_TO_RUN)}] {run_label(num_pieces)}")
        print(f"{sep}")
        cfg = build_run_config(num_pieces)
        _seed_tightening_reports(cfg)
        manifests[run_name(num_pieces)] = block3_poa_pipeline.run(cfg)

    print(f"\n{sep}")
    print("  Sweep complete. Building cross-pieces comparison...")
    print(f"{sep}")
    # Imported here to avoid a circular import (the summary imports this module).
    from driver.sensitivity import mccormick_pieces_summary

    mccormick_pieces_summary.main()
    return manifests


if __name__ == "__main__":
    run()
