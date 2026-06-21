# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-16
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Write a LaTeX table of PoA solver statistics for one run.

Reads a run's ``poa/poa_optimization_T*.json`` (written by block3) and emits a
booktabs table with four rows:

  - Computation time (s)            -> solver.wall_time_seconds
  - Total number of variables       -> continuous + discrete (binary + integer)
  - Total number of integer variables -> solver.variable_counts.num_discrete_variables_total
        (binary variables are integer variables restricted to {0, 1}; this is
         the count of all discrete decision variables)
  - Completion status               -> solver.termination_condition (title-cased)

The .tex file is self-contained except for ``\\usepackage{booktabs}`` in the
preamble.

Run (default = base_case):
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.summaries.poa_solver_latex_table
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.summaries.poa_solver_latex_table results/base_case
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Curation orders rows by their first element (the run label); the rest of the
# tuple is opaque, so this serves both solver-stats and regime row shapes.
_Row = TypeVar("_Row", bound=tuple)


def find_poa_result(run_dir: Path) -> Path:
    """Return the run's PoA result JSON, or raise if none is present."""
    poa_dir = run_dir / "poa"
    matches = sorted(poa_dir.glob("poa_optimization_T*.json"))
    if not matches:
        raise FileNotFoundError(f"No PoA result JSON under {poa_dir}")
    return matches[0]


# Worst-case regime variables the upper level optimizes over (rho_D / rho_W and
# peak_W are held fixed across runs, so they are omitted from the regime table).
REGIME_KEYS = ("mu_D", "sigma_D", "mu_W", "sigma_W")


def load_regime_params(run_dir: Path) -> dict[str, Any]:
    """Return the run's optimal worst-case regime variables, or {} if absent."""
    params_path = run_dir / "poa" / "poa_worst_case_regime_params.json"
    if not params_path.exists():
        return {}
    with params_path.open("r", encoding="utf-8") as fh:
        params = json.load(fh)
    return {key: params.get(key) for key in REGIME_KEYS}


def extract_stats(result: dict[str, Any]) -> dict[str, Any]:
    solver = result.get("solver", {}) or {}
    counts = solver.get("variable_counts", {}) or {}
    objective = result.get("objective", {}) or {}

    continuous = counts.get("num_continuous_variables")
    discrete = counts.get("num_discrete_variables_total")
    discrete_free = counts.get("num_discrete_variables_free")
    total = None
    if isinstance(continuous, int) and isinstance(discrete, int):
        total = continuous + discrete

    term = solver.get("termination_condition")
    status = str(term).title() if term is not None else "-"

    # Optimal value = the MILP optimal objective (the PoA upper bound); ex-post
    # ratio = the achieved ratio at that solution.
    optimal_value = objective.get("objective_value")
    if optimal_value is None:
        optimal_value = objective.get("PoA")

    return {
        "wall_time_seconds": solver.get("wall_time_seconds"),
        "total_variables": total,
        "continuous_variables": continuous,
        "integer_variables": discrete,
        "integer_variables_free": discrete_free,
        "num_constraints": solver.get("num_constraints"),
        "mip_gap": solver.get("mip_gap"),
        "optimal_value": optimal_value,
        "ex_post_ratio": objective.get("ex_post_ratio"),
        "status": status,
    }


def _fmt_time(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "-"


def _fmt_int(value: Any) -> str:
    return f"{value:,}".replace(",", "\\,") if isinstance(value, int) else "-"


def _fmt_free_total(free: Any, total: Any) -> str:
    if isinstance(free, int) and isinstance(total, int):
        return f"{_fmt_int(free)} / {_fmt_int(total)}"
    return _fmt_int(total)


def _fmt_pct(value: Any) -> str:
    # mip_gap is stored as a fraction (0.0 = proven optimal).
    return f"{value * 100.0:.2f}" if isinstance(value, (int, float)) else "-"


def _fmt_ratio(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def build_latex(stats: dict[str, Any], label: str) -> str:
    rows = [
        ("Computation time (s)", _fmt_time(stats["wall_time_seconds"])),
        ("Total number of variables", _fmt_int(stats["total_variables"])),
        ("Continuous variables", _fmt_int(stats["continuous_variables"])),
        (
            "Integer variables (free / total)",
            _fmt_free_total(stats["integer_variables_free"], stats["integer_variables"]),
        ),
        ("Total number of constraints", _fmt_int(stats["num_constraints"])),
        ("MIP gap (\\%)", _fmt_pct(stats["mip_gap"])),
        ("Optimal value (PoA)", _fmt_ratio(stats["optimal_value"])),
        ("Ex-post ratio", _fmt_ratio(stats["ex_post_ratio"])),
        ("Completion status", stats["status"]),
    ]
    body = "\n".join(f"        {name} & {value} \\\\" for name, value in rows)
    safe_label = label.replace("/", "_").replace(" ", "_")
    return (
        "\\begin{table}[ht]\n"
        "    \\centering\n"
        "    \\begin{tabular}{lr}\n"
        "        \\toprule\n"
        "        Metric & Value \\\\\n"
        "        \\midrule\n"
        f"{body}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        f"    \\caption{{PoA optimization solver statistics ({label}).}}\n"
        f"    \\label{{tab:poa_solver_stats_{safe_label}}}\n"
        "\\end{table}\n"
    )


def load_stats(run_dir: Path) -> dict[str, Any]:
    result_path = find_poa_result(run_dir)
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)
    return extract_stats(result)


def write_table(run_dir: Path, label: str | None = None) -> Path:
    stats = load_stats(run_dir)
    label = label or run_dir.name
    latex = build_latex(stats, label)

    out_dir = run_dir / "LaTeX"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "poa_solver_stats.tex"
    out_path.write_text(latex, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Batch: one table per run across every study, plus a combined overview table.
# ---------------------------------------------------------------------------

RESULTS_ROOT = PROJECT_ROOT / "results"
SENSITIVITY_ROOT = RESULTS_ROOT / "sensitivity_studies"


def iter_run_dirs(root: Path) -> list[Path]:
    """Every directory under ``root`` holding a poa/poa_optimization_T*.json."""
    seen: list[Path] = []
    for result_path in sorted(root.rglob("poa_optimization_T*.json")):
        if result_path.parent.name != "poa":  # skip solve_logs/*_progress.json
            continue
        run_dir = result_path.parent.parent
        if run_dir not in seen:
            seen.append(run_dir)
    return seen


def _run_label(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(RESULTS_ROOT).as_posix()
    except ValueError:
        return run_dir.name


def build_overview_latex(
    rows: list[tuple[str, dict[str, Any]]],
    caption: str,
    label: str,
) -> str:
    header = (
        "        Run & Time (s) & Vars & Int (free/total) & Constr. & Gap (\\%) "
        "& Opt.\\ val. & Ex-post & Status \\\\"
    )
    body_lines = []
    for row_label, st in rows:
        body_lines.append(
            "        "
            + " & ".join(
                [
                    row_label.replace("_", "\\_"),
                    _fmt_time(st["wall_time_seconds"]),
                    _fmt_int(st["total_variables"]),
                    _fmt_free_total(st["integer_variables_free"], st["integer_variables"]),
                    _fmt_int(st["num_constraints"]),
                    _fmt_pct(st["mip_gap"]),
                    _fmt_ratio(st["optimal_value"]),
                    _fmt_ratio(st["ex_post_ratio"]),
                    st["status"],
                ]
            )
            + " \\\\"
        )
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[ht]\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\begin{tabular}{lrrrrrrrr}\n"
        "        \\toprule\n"
        f"{header}\n"
        "        \\midrule\n"
        f"{body}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{table}\n"
    )


def _fmt_regime(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def build_poa_regime_overview_latex(
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
    caption: str,
    label: str,
) -> str:
    """Focused table: PoA (ex-post), time, gap, and the optimal regime variables.

    One row per run. ``rows`` is ``(run_label, stats, regime_params)``; PoA is the
    achieved ex-post ratio (C_eq / C_opt) and the regime columns are the upper
    level's worst-case state (mu_D, sigma_D, mu_W, sigma_W).
    """
    header = (
        "        Run & PoA & Time (s) & Gap (\\%) "
        "& $\\mu_D$ & $\\sigma_D$ & $\\mu_W$ & $\\sigma_W$ \\\\"
    )
    body_lines = []
    for row_label, st, regime in rows:
        body_lines.append(
            "        "
            + " & ".join(
                [
                    row_label.replace("_", "\\_"),
                    _fmt_ratio(st["ex_post_ratio"]),
                    _fmt_time(st["wall_time_seconds"]),
                    _fmt_pct(st["mip_gap"]),
                    _fmt_regime(regime.get("mu_D")),
                    _fmt_regime(regime.get("sigma_D")),
                    _fmt_regime(regime.get("mu_W")),
                    _fmt_regime(regime.get("sigma_W")),
                ]
            )
            + " \\\\"
        )
    body = "\n".join(body_lines)
    return (
        "\\begin{table}[ht]\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\begin{tabular}{lrrrrrrr}\n"
        "        \\toprule\n"
        f"{header}\n"
        "        \\midrule\n"
        f"{body}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{table}\n"
    )


def _study_of(run_dir: Path) -> str | None:
    """First path component under sensitivity_studies, or None for base_case."""
    try:
        return run_dir.relative_to(SENSITIVITY_ROOT).parts[0]
    except (ValueError, IndexError):
        return None


def _study_run_label(run_dir: Path, study: str) -> str:
    """Run path relative to the study root, e.g. 'm5' or 'T6/P10'."""
    return run_dir.relative_to(SENSITIVITY_ROOT / study).as_posix()


def _curate_study_rows(study: str, study_rows: list[_Row]) -> list[_Row]:
    """Restrict/order a study's overview rows to its canonical runs.

    Tuple-shape agnostic: ordering uses only ``row[0]`` (the run label), so it
    serves both the solver-stats rows and the regime rows.

    bound_tightening_progression accumulates experimental side-runs (e.g.
    S4_dual_alpha_floor, S5_equilibrium) in the same study folder. The thesis
    overview table should show only the canonical S0..S4 progression, in stage
    order. Other studies are returned unchanged.
    """
    if study != "bound_tightening_progression":
        return study_rows
    # Lazy import to avoid pulling the pipeline in for unrelated studies.
    from driver.sensitivity.bound_tightening_progression import (
        TIGHTENING_CASES,
        horizon_name,
    )

    order = [f"{horizon_name()}/{name}" for name, _label, _stages in TIGHTENING_CASES]
    rank = {label: idx for idx, label in enumerate(order)}
    kept = [row for row in study_rows if row[0] in rank]
    kept.sort(key=lambda row: rank[row[0]])
    return kept


def run_all() -> None:
    run_dirs = [RESULTS_ROOT / "base_case"] + iter_run_dirs(SENSITIVITY_ROOT)
    run_dirs = [d for d in run_dirs if (d / "poa").exists()]

    rows: list[tuple[str, dict[str, Any]]] = []
    per_study: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    # Parallel regime rows: (label, stats, regime_params) for the focused
    # PoA + optimal-regime table requested across every study.
    regime_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    per_study_regime: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    for run_dir in run_dirs:
        try:
            label = _run_label(run_dir)
            out_path = write_table(run_dir, label=label)
            stats = load_stats(run_dir)
            regime = load_regime_params(run_dir)
            rows.append((label, stats))
            regime_rows.append((label, stats, regime))
            study = _study_of(run_dir)
            if study is not None:
                run_label = _study_run_label(run_dir, study)
                per_study.setdefault(study, []).append((run_label, stats))
                per_study_regime.setdefault(study, []).append((run_label, stats, regime))
            print(f"  wrote {out_path.relative_to(RESULTS_ROOT)}")
        except FileNotFoundError as exc:
            print(f"  [skip] {run_dir}: {exc}")

    # Global overview across every run.
    overview_dir = SENSITIVITY_ROOT / "LaTeX"
    overview_dir.mkdir(parents=True, exist_ok=True)
    overview_path = overview_dir / "poa_solver_stats_overview.tex"
    overview_path.write_text(
        build_overview_latex(
            rows,
            caption="PoA optimization solver statistics across sensitivity runs.",
            label="tab:poa_solver_stats_overview",
        ),
        encoding="utf-8",
    )
    regime_overview_path = overview_dir / "poa_regime_overview.tex"
    regime_overview_path.write_text(
        build_poa_regime_overview_latex(
            regime_rows,
            caption=(
                "Achieved PoA (ex-post ratio), solve time, MIP gap, and the "
                "optimal worst-case regime across sensitivity runs."
            ),
            label="tab:poa_regime_overview",
        ),
        encoding="utf-8",
    )

    # Per-study overviews: only that study's runs, written into the study folder.
    for study, study_rows in per_study.items():
        study_rows = _curate_study_rows(study, study_rows)
        study_dir = SENSITIVITY_ROOT / study / "LaTeX"
        study_dir.mkdir(parents=True, exist_ok=True)
        study_label = study.replace("_", "\\_")
        (study_dir / "poa_solver_stats_overview.tex").write_text(
            build_overview_latex(
                study_rows,
                caption=f"PoA optimization solver statistics ({study_label}).",
                label=f"tab:poa_solver_stats_{study}",
            ),
            encoding="utf-8",
        )

    for study, study_regime_rows in per_study_regime.items():
        study_regime_rows = _curate_study_rows(study, study_regime_rows)
        study_dir = SENSITIVITY_ROOT / study / "LaTeX"
        study_dir.mkdir(parents=True, exist_ok=True)
        study_label = study.replace("_", "\\_")
        (study_dir / "poa_regime_overview.tex").write_text(
            build_poa_regime_overview_latex(
                study_regime_rows,
                caption=(f"Achieved PoA and optimal worst-case regime ({study_label})."),
                label=f"tab:poa_regime_{study}",
            ),
            encoding="utf-8",
        )

    n_missing = sum(1 for _, st in rows if not isinstance(st["num_constraints"], int))
    print(
        f"\nWrote {len(rows)} per-run tables + global solver/regime overviews + "
        f"{len(per_study)} per-study solver + {len(per_study_regime)} per-study "
        f"regime overviews."
    )
    print(f"  global solver: {overview_path}")
    print(f"  global regime: {regime_overview_path}")
    if n_missing:
        print(
            f"  {n_missing}/{len(rows)} runs have no constraint count yet "
            f"(shown as '-'); run the backfill to populate them."
        )


def main(argv: list[str]) -> None:
    if argv and argv[0] in ("--all", "all"):
        run_all()
        return
    run_dir = Path(argv[0]) if argv else PROJECT_ROOT / "results" / "base_case"
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    out_path = write_table(run_dir)
    print(f"Wrote LaTeX table: {out_path}\n")
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
