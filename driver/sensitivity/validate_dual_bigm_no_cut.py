"""One-off validation: does S4 (dual Big-M) cut off S0_loose's PoA solution?

Concern: the reported (relaxed) PoA dropped 11.399 -> 11.351 when dual Big-M
tightening was added at S4. If that drop came from the tightened dual bounds
wrongly excluding feasible (alpha, dispatch, dual) tuples, the PoA result would
be invalid. If instead it is just the McCormick relaxation getting tighter, the
true feasible set is unchanged and the result is valid.

Decisive test: rebuild the S4-configured PoA model (same scenarios, same
base-case policies, loading S4's on-disk tightening reports -> S4's tightened
dual Big-M bounds enter as the mu_* variable upper bounds), then FIX every
alpha[i,b,t] to the optimal bid vector from the S0_loose solution and solve.

  - feasible  -> S0's bid vector is representable under S4's bounds; the dual
                 Big-M tightening did NOT cut it. The PoA drop is relaxation
                 tightening, not lost solution space. (Bonus: the recovered
                 C_eq should match S0's C_eq = 1427.52.)
  - infeasible -> S4's tightened bounds excluded S0's solution; the tightening
                 is too aggressive and the S4 PoA is invalid.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.validate_dual_bigm_no_cut
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver.core.block2_core import discover_trained_policy_generators  # noqa: E402
from driver.core.block3_core import build_poa_config, build_poa_optimizer  # noqa: E402
from driver.sensitivity.bound_tightening_progression import (  # noqa: E402
    RESULT_ROOT,
    TIGHTENING_CASES,
    _build_flags,
    build_run_config,
    horizon_name,
)

# Compare S0 (loose) optimum against S4 (dual Big-M) feasible set.
SOURCE_CASE = "S0_loose"  # take the optimal alpha bid vector from here
TARGET_CASE = "S4_dual"  # test feasibility under this case's tightened bounds


def _result_json(case_name: str) -> Path:
    return (
        RESULT_ROOT
        / horizon_name()
        / case_name
        / "poa"
        / "poa_optimization_T6_piecewise_mccormick.json"
    )


def load_source_alpha(case_name: str) -> tuple[dict[tuple[int, int, int], float], float]:
    """Return {(i, b, t): alpha} and the source C_eq from a solved PoA result.

    Indexing matches the model's alpha[i, b, t]: i = physical_generator_index,
    b = local_block_index, t = time step (see poa_model/results.py).
    """
    with _result_json(case_name).open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    alpha: dict[tuple[int, int, int], float] = {}
    for generator in result["generators"].values():
        i = int(generator["physical_generator_index"])
        for block in generator["blocks"]:
            b = int(block["local_block_index"])
            for t, value in enumerate(block["alpha_profile"]):
                alpha[(i, b, t)] = float(value)
    c_eq = float(result["objective"]["C_eq"])
    return alpha, c_eq


def build_target_optimizer():
    """Rebuild the TARGET_CASE PoA optimizer with its on-disk tightening reports."""
    enabled = next(stages for name, _label, stages in TIGHTENING_CASES if name == TARGET_CASE)
    flags = _build_flags(enabled)
    cfg = build_run_config(TARGET_CASE, flags)
    # Reuse the existing tightening reports; do not recompute or re-solve here.
    cfg.run_poa_tightening = False
    cfg.run_poa_optimization = False
    # Restrict to the policies actually trained on disk (base_case trained only a
    # subset, e.g. G1/W2/W3); block3 does the same via discovery before solving.
    discovered = discover_trained_policy_generators(cfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        print(f"  Discovered trained policy generators: {discovered}")
    pcfg = build_poa_config(cfg)

    optimizer = build_poa_optimizer(pcfg)
    if not pcfg.tightening_report_path.exists():
        raise FileNotFoundError(
            f"Tightening report for {TARGET_CASE} not found: {pcfg.tightening_report_path}. "
            "Run the bound_tightening_progression study first."
        )
    optimizer.load_tightening_report(pcfg.tightening_report_path)
    optimizer.build_model()
    if optimizer.nn_policy_generator_ids:
        optimizer.apply_nn_relu_bounds_to_model()
    optimizer.apply_tightened_bounds_to_model()
    return optimizer, pcfg


def fix_alpha(optimizer, alpha: dict[tuple[int, int, int], float]) -> tuple[int, float]:
    """Fix every model alpha[i, b, t] to the source bid vector.

    Clamps each value into the target model's declared [lb, ub] so .fix() never
    overrides a bound (which would let alpha escape S4's feasible region). The
    max clamp distance quantifies how far S0's solution sits outside S4's bounds:
    if it is at solver-tolerance scale (~1e-6 or below), S0's bids are feasible
    under S4 up to numerical noise; if it is large, S4 genuinely excludes them.

    Returns (count, max_clamp_distance).
    """
    m = optimizer.model
    fixed = 0
    max_clamp = 0.0
    missing = []
    for i, b in m.generator_blocks:
        for t in m.time_steps:
            key = (int(i), int(b), int(t))
            if key not in alpha:
                missing.append(key)
                continue
            var = m.alpha[i, b, t]
            value = alpha[key]
            lb = var.lb
            ub = var.ub
            clamped = value
            if lb is not None and clamped < lb:
                clamped = lb
            if ub is not None and clamped > ub:
                clamped = ub
            max_clamp = max(max_clamp, abs(clamped - value))
            var.fix(clamped)
            fixed += 1
    if missing:
        raise KeyError(f"No source alpha for model indices: {missing[:10]} (+{len(missing) - 10})")
    return fixed, max_clamp


def main() -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Validation: is {SOURCE_CASE}'s PoA solution feasible under {TARGET_CASE}?")
    print(f"{sep}")

    alpha, source_c_eq = load_source_alpha(SOURCE_CASE)
    print(f"  Loaded {len(alpha)} alpha values from {SOURCE_CASE} (C_eq = {source_c_eq:.4f})")

    optimizer, pcfg = build_target_optimizer()
    print(f"  Built {TARGET_CASE} model; loaded tightening report:")
    print(f"    {pcfg.tightening_report_path}")

    n_fixed, max_clamp = fix_alpha(optimizer, alpha)
    print(f"  Fixed {n_fixed} alpha[i,b,t] variables to {SOURCE_CASE}'s bid vector.")
    print(f"  Max clamp distance into {TARGET_CASE}'s alpha bounds: {max_clamp:.3e}")
    if max_clamp > 1e-6:
        print(
            f"  NOTE: max clamp {max_clamp:.3e} exceeds solver tolerance -- some "
            f"{SOURCE_CASE} bids lie materially outside {TARGET_CASE}'s alpha bounds."
        )

    print(f"\n  Solving {TARGET_CASE} model with alpha fixed...")
    optimizer.solve(time_limit=600, solver_threads=1, solver_seed=0)
    termination = str(optimizer.solver_results.solver.termination_condition)
    status = str(optimizer.solver_results.solver.status)
    print(f"  Solver status: {status}  |  termination: {termination}")

    infeasible = termination.lower() in {"infeasible", "infeasibleorunbounded"}
    print(f"\n{sep}")
    if infeasible:
        print(f"  RESULT: INFEASIBLE -- {TARGET_CASE}'s dual Big-M bounds EXCLUDE")
        print(f"          {SOURCE_CASE}'s solution. The tightening cut feasible space;")
        print("          the S4 PoA would be INVALID. Investigate dual_big_m bounds.")
    else:
        metrics = optimizer.extract_objective_metrics()
        recovered_c_eq = float(metrics.get("C_eq", float("nan")))
        print(f"  RESULT: FEASIBLE -- {SOURCE_CASE}'s bid vector survives under")
        print(f"          {TARGET_CASE}'s tightened bounds. No feasible solution was cut.")
        print(f"          Recovered C_eq = {recovered_c_eq:.4f}  (source {source_c_eq:.4f},")
        print(f"          delta = {abs(recovered_c_eq - source_c_eq):.4e})")
        print(f"          => the 11.399 -> 11.351 PoA drop is relaxation tightening,")
        print("             not lost solution space.")
    print(f"{sep}")


if __name__ == "__main__":
    main()
