"""Focused re-run: does the NN alpha-bound floor fix S4's PoA under-report?

Background: at S4 (dual Big-M) the free PoA solve reported 11.351 and a
``max constraint violation 1e-6`` warning, yet fixing alpha to the S0 optimum
showed 11.399 is feasible under S4 (see validate_dual_bigm_no_cut.py). The cause
was a collapsed NN-wind alpha interval (~5e-10 wide) ill-conditioning the
McCormick alpha*P products. compute_alpha_bounds now floors NN-policy alpha
intervals to +/- 1e-5.

This script rebuilds the S4 configuration against the base-case policies,
recomputes the alpha-dependent tightening (relu -> alpha -> slack -> dual) so the
floor takes effect, re-solves the PoA MILP under the same pinned solver settings
as the overnight run (Threads=1, Seed=0), and prints a side-by-side comparison
against the overnight S4 result. Outputs go to an isolated dir so the overnight
S4 results are left intact.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.rerun_s4_alpha_floor
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver import block3_poa_pipeline  # noqa: E402
from driver.sensitivity.bound_tightening_progression import (  # noqa: E402
    RESULT_ROOT,
    TIGHTENING_CASES,
    _build_flags,
    build_run_config,
    horizon_name,
)

SOURCE_CASE = "S4_dual"  # overnight reference
RERUN_CASE = "S4_dual_alpha_floor"  # isolated output dir for this re-run

# Keep tightening parallelism modest so this does not oversubscribe the box if a
# sweep is running. Does not affect the PoA value, only tightening wall time.
RERUN_PARALLEL_WORKERS = 3

RESULT_JSON = "poa_optimization_T6_piecewise_mccormick.json"


def _poa_dir(case_name: str) -> Path:
    return RESULT_ROOT / horizon_name() / case_name / "poa"


def build_rerun_config():
    enabled = next(stages for name, _label, stages in TIGHTENING_CASES if name == SOURCE_CASE)
    flags = _build_flags(enabled)  # relu + alpha + slack + dual (+ always-on)
    cfg = build_run_config(SOURCE_CASE, flags)

    # Recompute the alpha-dependent tightening so the new floor takes effect,
    # then re-solve. (build_run_config already pins Threads=1/Seed=0 and the 3000s
    # PoA time limit, matching the overnight run.)
    cfg.run_poa_tightening = True
    cfg.run_poa_optimization = True
    cfg.poa_parallel_workers = RERUN_PARALLEL_WORKERS

    # Redirect every output to the isolated re-run dir so overnight S4 is intact.
    run_root = RESULT_ROOT / horizon_name() / RERUN_CASE
    cfg.case_label = f"bound_tightening_progression/{horizon_name()}/{RERUN_CASE}"
    cfg.poa_result_dir = run_root / "poa"
    cfg.figures_dir = run_root / "figures"
    cfg.poa_scenario_dir = run_root / "poa_scenarios"
    cfg.runtime_config_path = run_root / "runtime_regime_definitions.yaml"
    return cfg


def _nn_alpha_widths(case_name: str) -> list[tuple[str, float, float, float]]:
    """Return (index, lower, upper, width) for the thinnest alpha intervals."""
    report_path = _poa_dir(case_name) / "tightening" / "dual_big_m_report.json"
    if not report_path.exists():
        return []
    report = json.load(report_path.open(encoding="utf-8"))
    rows = []
    for key, bounds in report.get("alpha_bounds", {}).items():
        lo = bounds.get("lower")
        hi = bounds.get("upper")
        if lo is None or hi is None:
            continue
        rows.append((key, float(lo), float(hi), abs(float(hi) - float(lo))))
    rows.sort(key=lambda r: r[3])
    return rows


def _objective(case_name: str) -> dict:
    path = _poa_dir(case_name) / RESULT_JSON
    if not path.exists():
        return {}
    return json.load(path.open(encoding="utf-8")).get("objective", {})


def _max_constraint_violation(case_name: str) -> float | None:
    """Parse the largest 'max constraint violation' from the Gurobi solve log."""
    log_path = _poa_dir(case_name) / "solve_logs" / f"{Path(RESULT_JSON).stem}_gurobi.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"max constraint violation \(([0-9.eE+-]+)\)", text)
    return max(float(h) for h in hits) if hits else None


def _smallest_nonzero_width(rows) -> tuple[str, float] | None:
    for key, _lo, _hi, width in rows:
        if width > 0.0:
            return key, width
    return None


def report_comparison() -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Comparison: overnight S4 vs S4 with NN alpha-bound floor")
    print(f"{sep}")

    for case in (SOURCE_CASE, RERUN_CASE):
        obj = _objective(case)
        rows = _nn_alpha_widths(case)
        thinnest_nonzero = _smallest_nonzero_width(rows)
        violation = _max_constraint_violation(case)
        label = "overnight" if case == SOURCE_CASE else "alpha-floor"
        print(f"\n  [{label}] {case}")
        if obj:
            print(f"    PoA (relaxed upper bound) : {obj.get('PoA')}")
            print(f"    ex_post_ratio (achieved)  : {obj.get('ex_post_ratio')}")
            print(f"    mccormick_gap             : {obj.get('mccormick_gap')}")
        else:
            print("    (no PoA result found)")
        if thinnest_nonzero is not None:
            key, width = thinnest_nonzero
            print(f"    thinnest nonzero alpha width: {width:.3e}  (index {key})")
        print(f"    max constraint violation  : {violation}")

    new_obj = _objective(RERUN_CASE)
    old_obj = _objective(SOURCE_CASE)
    if new_obj and old_obj:
        new_poa = float(new_obj.get("PoA"))
        old_poa = float(old_obj.get("PoA"))
        print(f"\n{sep}")
        print(f"  PoA: {old_poa:.6f} (overnight)  ->  {new_poa:.6f} (alpha-floor)")
        if new_poa > old_poa + 1e-4:
            print("  => floor RECOVERED the higher PoA the overnight S4 under-reported.")
        elif abs(new_poa - old_poa) <= 1e-4:
            print("  => PoA essentially unchanged; floor alone did not move the solve.")
        else:
            print("  => PoA decreased further; investigate before trusting this.")
        print(f"{sep}")


def main() -> None:
    cfg = build_rerun_config()
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Re-running S4 (dual Big-M) with the NN alpha-bound floor active")
    print(f"  output: {cfg.poa_result_dir}")
    print(f"  tightening: relu -> alpha -> slack -> dual (recomputed)")
    print(f"  solve: Threads={cfg.poa_solver_threads}, Seed={cfg.poa_solver_seed}, "
          f"time_limit={cfg.poa_time_limit}")
    print(f"{sep}")

    block3_poa_pipeline.run(cfg)
    report_comparison()


if __name__ == "__main__":
    main()
