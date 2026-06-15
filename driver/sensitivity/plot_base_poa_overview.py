"""Cross-study overview of the base PoA (ex-post ratio) across sensitivity setups.

For every sensitivity study under ``results/sensitivity_studies`` this reads each
run's PoA optimization result and plots ``objective.ex_post_ratio`` (the achieved
worst-case C_eq / C_opt at the solution) against that study's swept parameter, as
a grid of small-multiple panels.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.plot_base_poa_overview
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_ROOT = Path("results/sensitivity_studies")
BASE_CASE_POA_DIR = Path("results/base_case/poa")
METRIC_KEY = "ex_post_ratio"
METRIC_LABEL = "Ex-post PoA ratio (C_eq / C_opt)"
OUTPUT_PNG = RESULT_ROOT / "base_poa_ex_post_ratio_overview.png"
OUTPUT_CSV = RESULT_ROOT / "base_poa_ex_post_ratio_overview.csv"
# Per-study figure filename, written into results/sensitivity_studies/<study>/.
STUDY_PLOT_NAME = "base_poa_ex_post_ratio.png"


class StudyMeta(NamedTuple):
    title: str
    xlabel: str
    # run_name -> numeric x (None => categorical, ordered by sort_key)
    x_of: Callable[[str], float | None]


def _trailing_int(run: str, prefix: str) -> float | None:
    m = re.search(rf"{prefix}(\d+)", run)
    return float(m.group(1)) if m else None


# Per-study labelling. Unknown studies fall back to categorical-by-run-name.
STUDY_META: dict[str, StudyMeta] = {
    "bidding_blocks_sweep": StudyMeta(
        "Bidding blocks", "Blocks per conventional gen (B)", lambda r: _trailing_int(r, "blocks_B")
    ),
    "players_sweep": StudyMeta(
        "Number of players", "N (conv + wind, each N)", lambda r: _trailing_int(r, "players_N")
    ),
    "horizon_sweep": StudyMeta(
        "Horizon", "Time steps (T)", lambda r: _trailing_int(r, "T")
    ),
    "ramp_rate_sweep": StudyMeta(
        "Ramp rate", "Conv ramp rate (MW/h)", lambda r: _trailing_int(r, "ramp_R")
    ),
    "overlapping_costs_sweep": StudyMeta(
        "Overlapping costs", "Cost configuration", lambda r: None
    ),
    "wind_playing_sweep": StudyMeta(
        "Wind plays strategically", "Configuration", lambda r: None
    ),
    "composition_sweep": StudyMeta(
        "Generation mix", "Composition", lambda r: None
    ),
}


def _poa_result_file(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("poa/poa_optimization_*.json"))
    return matches[0] if matches else None


def _read_metric(poa_file: Path) -> float | None:
    with poa_file.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    value = (data.get("objective") or {}).get(METRIC_KEY)
    return float(value) if value is not None else None


def _pretty_run(run: str) -> str:
    return run.replace("_", " ")


def collect() -> dict[str, list[dict[str, Any]]]:
    studies: dict[str, list[dict[str, Any]]] = {}
    for study_dir in sorted(p for p in RESULT_ROOT.iterdir() if p.is_dir()):
        meta = STUDY_META.get(study_dir.name)
        rows: list[dict[str, Any]] = []
        for run_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            poa_file = _poa_result_file(run_dir)
            if poa_file is None:
                continue
            metric = _read_metric(poa_file)
            if metric is None:
                continue
            x = meta.x_of(run_dir.name) if meta else None
            rows.append({"run": run_dir.name, "x": x, "metric": metric})
        if rows:
            studies[study_dir.name] = rows
    return studies


def _base_reference() -> float | None:
    poa_file = sorted(BASE_CASE_POA_DIR.glob("poa_optimization_*.json"))
    if not poa_file:
        return None
    return _read_metric(poa_file[0])


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if all(r["x"] is not None for r in rows):
        return sorted(rows, key=lambda r: r["x"])
    return sorted(rows, key=lambda r: r["run"])


def _draw_panel(ax, study: str, rows: list[dict[str, Any]], base_ref: float | None,
                title_size: int = 11, ylabel_size: int = 8) -> None:
    rows = _sort_rows(rows)
    meta = STUDY_META.get(study)
    numeric = all(r["x"] is not None for r in rows)

    if numeric:
        xs = [r["x"] for r in rows]
        tick_labels = [f"{int(x) if float(x).is_integer() else x}" for x in xs]
    else:
        xs = list(range(len(rows)))
        tick_labels = [_pretty_run(r["run"]) for r in rows]
    ys = [r["metric"] for r in rows]

    ax.plot(xs, ys, "-o", color="#1f77b4", zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)

    if base_ref is not None:
        ax.axhline(base_ref, ls="--", lw=1, color="grey", zorder=1)
        ax.text(0.02, base_ref, f" base {base_ref:.2f}", color="grey",
                fontsize=7, va="bottom", ha="left", transform=ax.get_yaxis_transform())

    ax.set_title(meta.title if meta else study, fontsize=title_size)
    ax.set_xlabel(meta.xlabel if meta else "run")
    ax.set_ylabel(METRIC_LABEL, fontsize=ylabel_size)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=0 if numeric else 20, fontsize=8,
                       ha="center" if numeric else "right")
    ax.grid(True, alpha=0.3)


def plot_individual(studies: dict[str, list[dict[str, Any]]], base_ref: float | None) -> None:
    """One standalone figure per study, saved inside that study's folder."""
    for study, rows in sorted(studies.items()):
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        _draw_panel(ax, study, rows, base_ref, title_size=13, ylabel_size=10)
        fig.tight_layout()
        out_png = RESULT_ROOT / study / STUDY_PLOT_NAME
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {out_png}")


def plot_overview(studies: dict[str, list[dict[str, Any]]], base_ref: float | None) -> None:
    n = len(studies)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (study, rows) in enumerate(sorted(studies.items())):
        ax = axes.flat[idx]
        ax.set_visible(True)
        _draw_panel(ax, study, rows, base_ref)

    fig.suptitle("Base PoA (ex-post ratio) across sensitivity setups", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {OUTPUT_PNG}")


def write_csv(studies: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["study,run,x,ex_post_ratio"]
    for study, rows in sorted(studies.items()):
        for r in _sort_rows(rows):
            x = "" if r["x"] is None else r["x"]
            lines.append(f"{study},{r['run']},{x},{r['metric']:.6f}")
    OUTPUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved table:  {OUTPUT_CSV}")


def main() -> None:
    studies = collect()
    base_ref = _base_reference()
    print(f"Base-case ex_post_ratio reference: {base_ref}")
    for study, rows in sorted(studies.items()):
        pretty = ", ".join(f"{r['run']}={r['metric']:.3f}" for r in _sort_rows(rows))
        print(f"  {study}: {pretty}")
    write_csv(studies)
    plot_individual(studies, base_ref)


if __name__ == "__main__":
    main()
