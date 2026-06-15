"""Backfill ``solver.num_constraints`` into PoA result JSONs.

Result JSONs written before constraint instrumentation was added to
``build_solver_summary`` have no constraint count. Re-solving just to record it
is wasteful, so this script reconstructs each run's PoA model build-only (no
solve -- Pyomo model construction is solver-independent), counts the active
constraint rows, and writes ``num_constraints`` into the run's ``solver`` block.

Each run's substantive model parameters are read back from its own result JSON
(horizon, McCormick piece count, NN policy generators, objective mode) and its
artifacts (trained policies, normalization stats, tightening report) are reused
from the run directory. The reference ``case`` is left at the default, which is
correct for every run that keeps the base-case generator/block STRUCTURE
(inflation margin, bound-tightening, McCormick pieces, horizon, and numeric-only
changes such as ramp rate or overlapping costs). Studies that change the
structure (player count, block count, composition, removing wind) need their own
``case``; for those the variable-count self-check below fails and the run is
skipped rather than written with a wrong number.

Self-check: after rebuilding, the variable counts are recompared against those
already stored in the JSON. A mismatch means the rebuilt model is not the one
that was solved, so the script skips that run (never writes a wrong count).

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.summaries.backfill_poa_constraint_counts          # base_case
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.summaries.backfill_poa_constraint_counts --all     # every run on disk
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.summaries.backfill_poa_constraint_counts <run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.core.block0_core import build_config  # noqa: E402
from driver.core.block3_core import build_poa_config, build_poa_optimizer  # noqa: E402
from driver.sensitivity.summaries.poa_solver_latex_table import (  # noqa: E402
    RESULTS_ROOT,
    SENSITIVITY_ROOT,
    find_poa_result,
    iter_run_dirs,
)
from models.PoA.PoA_optimization import PoAOptimization  # noqa: E402
from models.helper import count_constraints, count_integer_variables  # noqa: E402

# Variable-count fields that must match the stored result for a write to proceed.
_SELF_CHECK_KEYS = (
    "num_discrete_variables_total",
    "num_discrete_variables_free",
    "num_continuous_variables",
)


def build_run_pcfg(run_dir: Path, result: dict[str, Any], case: str | None = None) -> Any:
    """Construct a PoAPipelineConfig that rebuilds this run's model.

    Substantive parameters come from the result JSON; artifact directories point
    into the run dir. ``case`` defaults to the base case (structurally correct
    for base-structure runs); pass the study-local case name (auto-discovered
    from config/sensitivity_studies/*.yaml) to rebuild a structure-changing run.
    """
    objective = result.get("objective", {}) or {}
    horizon = int(result["num_time_steps"])

    cfg = build_config()
    if case is not None:
        cfg.case = case
    cfg.horizon = horizon
    cfg.synthetic_time_steps = horizon
    num_pieces = objective.get("num_pieces")
    if isinstance(num_pieces, int):
        cfg.poa_mccormick_num_pieces = num_pieces
    cfg.nn_policy_generators = list(result.get("nn_policy_generators", []))
    obj_mode = objective.get("objective_mode")
    if obj_mode:
        cfg.poa_objective_mode = str(obj_mode)

    # Most runs isolate their own policies/features; some (e.g. the McCormick
    # pieces sweep) reuse the base case's upstream artifacts and only isolate the
    # PoA outputs. Fall back to base_case when the run dir lacks them.
    base_case = RESULTS_ROOT / "base_case"
    model_dir = run_dir / "trained_models"
    if not model_dir.exists():
        model_dir = base_case / "trained_models"
    norm_dir = run_dir / "features" / "normalized"
    if not (norm_dir / "min_max_stats.json").exists():
        norm_dir = base_case / "features" / "normalized"

    cfg.model_dir = model_dir
    cfg.normalized_feature_dir = norm_dir
    cfg.poa_result_dir = run_dir / "poa"

    return build_poa_config(cfg)


def build_model_only(pcfg: Any) -> PoAOptimization:
    """Reproduce run_final_poa's build steps up to (but not including) solve()."""
    optimizer = build_poa_optimizer(pcfg, PoAOptimization)
    if pcfg.tightening_report_path.exists():
        optimizer.load_tightening_report(pcfg.tightening_report_path)
    else:
        optimizer.ensure_default_bounds_available(
            include_nn_relu_bounds=None,
            include_alpha_bounds=True,
            include_tight_big_m=True,
            include_lambda_bounds=True,
            include_optimal_cost_bounds=True,
            overwrite_existing=False,
        )
    optimizer.build_model()
    if optimizer.nn_policy_generator_ids:
        optimizer.apply_nn_relu_bounds_to_model()
    optimizer.apply_tightened_bounds_to_model()
    return optimizer


def _self_check_diff(reconstructed: dict[str, int], stored: dict[str, Any]) -> list[str]:
    diffs = []
    for key in _SELF_CHECK_KEYS:
        if stored.get(key) is not None and reconstructed.get(key) != stored.get(key):
            diffs.append(f"{key}: rebuilt={reconstructed.get(key)} stored={stored.get(key)}")
    return diffs


def _attempt(run_dir: Path, result: dict[str, Any], case: str | None) -> tuple[int, list[str]]:
    """Build this run's model and return (num_constraints, self_check_diffs)."""
    pcfg = build_run_pcfg(run_dir, result, case=case)
    optimizer = build_model_only(pcfg)
    num_constraints = count_constraints(optimizer.model)
    recon_counts = count_integer_variables(optimizer.model)
    stored = (result.get("solver", {}) or {}).get("variable_counts", {}) or {}
    return num_constraints, _self_check_diff(recon_counts, stored)


def backfill_run(run_dir: Path) -> tuple[str, Any]:
    """Return ("ok", n) | ("skip", reason) | ("error", reason) for one run.

    First rebuilds with the base case. If that fails to build or the variable
    counts do not match the solved model, retries with the study-local case named
    after the run directory (e.g. ``players_N4``), which is how structure-changing
    runs are reconstructed.
    """
    result_path = find_poa_result(run_dir)
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    num_constraints: int | None = None
    diffs: list[str] = []
    try:
        num_constraints, diffs = _attempt(run_dir, result, case=None)
    except Exception:  # noqa: BLE001 -- base case could not build this structure
        num_constraints = None

    if num_constraints is None or diffs:
        # Retry with the study-local reference case (auto-discovered by name).
        num_constraints, diffs = _attempt(run_dir, result, case=run_dir.name)

    if diffs:
        return "skip", "; ".join(diffs)

    result.setdefault("solver", {})["num_constraints"] = int(num_constraints)
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return "ok", int(num_constraints)


def _label(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(RESULTS_ROOT).as_posix()
    except ValueError:
        return str(run_dir)


def run_all() -> None:
    run_dirs = [RESULTS_ROOT / "base_case"] + iter_run_dirs(SENSITIVITY_ROOT)
    run_dirs = [d for d in run_dirs if (d / "poa").exists()]

    ok: list[tuple[str, int]] = []
    skipped: list[tuple[str, str]] = []
    errored: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        label = _label(run_dir)
        print(f"  [build] {label} ...", flush=True)
        try:
            status, payload = backfill_run(run_dir)
        except Exception as exc:  # noqa: BLE001 -- report and continue
            errored.append((label, f"{type(exc).__name__}: {exc}"))
            print(f"      ERROR: {type(exc).__name__}: {exc}")
            continue
        if status == "ok":
            ok.append((label, payload))
            print(f"      num_constraints = {payload:,}")
        else:
            skipped.append((label, payload))
            print(f"      SKIP (model mismatch): {payload}")

    print(f"\n{'=' * 72}")
    print(f"Backfill complete: {len(ok)} written, {len(skipped)} skipped, {len(errored)} errored.")
    if skipped:
        print("\nSkipped (structure-changing -- need study-specific case):")
        for label, reason in skipped:
            print(f"  {label}: {reason}")
    if errored:
        print("\nErrored:")
        for label, reason in errored:
            print(f"  {label}: {reason}")


def main(argv: list[str]) -> None:
    if argv and argv[0] in ("--all", "all"):
        run_all()
        return
    run_dir = Path(argv[0]) if argv else RESULTS_ROOT / "base_case"
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    status, payload = backfill_run(run_dir)
    if status == "ok":
        print(f"Backfilled num_constraints = {payload:,} into {find_poa_result(run_dir)}")
    else:
        print(f"Skipped {run_dir}: {payload}")


if __name__ == "__main__":
    main(sys.argv[1:])
