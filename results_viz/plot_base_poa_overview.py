# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-16
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Cross-study overview of the base PoA (ex-post ratio) across sensitivity setups.

For every sensitivity study under ``results/sensitivity_studies`` this reads each
run's PoA optimization result and plots ``objective.ex_post_ratio`` (the achieved
worst-case C_eq / C_opt at the solution) against that study's swept parameter, as
a grid of small-multiple panels.

Run:
  .\\.venv\\Scripts\\python.exe -m results_viz.plot_base_poa_overview
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


# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
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
    # Render PoA as a bar chart instead of a line (for categorical studies).
    bar: bool = False


def _trailing_int(run: str, prefix: str) -> float | None:
    m = re.search(rf"{prefix}(\d+)", run)
    return float(m.group(1)) if m else None


def _p_decimal(run: str, prefix: str) -> float | None:
    """``sigma_0p015`` / ``kappa_0p30`` -> 0.015 / 0.30 ('p' is the decimal point)."""
    m = re.search(rf"{prefix}([0-9p]+)", run)
    return float(m.group(1).replace("p", ".")) if m else None


def _rho_value(run: str) -> float | None:
    """``rho_neg0p25`` -> -0.25, ``rho_pos0p99`` -> 0.99."""
    m = re.search(r"rho_(neg|pos)([0-9p]+)", run)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "neg" else 1.0
    return sign * float(m.group(2).replace("p", "."))


# Per-study labelling. Unknown studies fall back to categorical-by-run-name.
# X labels use mathtext (rendered as Greek/LaTeX) for the thesis appendix figures.
STUDY_META: dict[str, StudyMeta] = {
    "bidding_blocks_sweep": StudyMeta(
        "Bidding blocks",
        r"Bidding blocks per conventional generator $B$",
        lambda r: _trailing_int(r, "blocks_B"),
    ),
    "players_sweep": StudyMeta(
        "Number of players",
        "Number of Wind and Conv. Generators",
        lambda r: _trailing_int(r, "players_N"),
    ),
    "horizon_sweep": StudyMeta("Horizon", "Time steps (T)", lambda r: _trailing_int(r, "T")),
    "ramp_rate_sweep": StudyMeta(
        "Ramp rate",
        r"Conventional ramp rate $R$ (MW/h)",
        lambda r: _trailing_int(r, "ramp_R"),
    ),
    "peak_w_sweep": StudyMeta(
        "Wind peak hour", r"Peak wind hour $\tau$", lambda r: _trailing_int(r, "peak_w_"), bar=True
    ),
    "rho_sweep": StudyMeta("AR(1) autocorrelation", r"AR(1) parameter $\rho$", _rho_value),
    "sigma_max_sweep": StudyMeta(
        "Volatility ceiling", r"Volatility ceiling $\sigma$", lambda r: _p_decimal(r, "sigma_")
    ),
    "wind_bound_sweep": StudyMeta(
        "Wind mean bounds",
        r"Wind mean bounds $(\mu_W^{\min},\ \mu_W^{\max})$",
        lambda r: None,
        bar=True,
    ),
    "demand_ref_sweep": StudyMeta(
        "Reference demand",
        r"Reference demand $D^{\mathrm{ref}}$ (MW)",
        lambda r: _trailing_int(r, "dref_"),
    ),
    "ambiguity_kappa_sweep": StudyMeta(
        "Support-set budget", r"Support-set budget $\gamma$", lambda r: _p_decimal(r, "kappa_")
    ),
    "overlapping_costs_sweep": StudyMeta(
        "Overlapping costs", "Conventional bidding-block cost ranges", lambda r: None, bar=True
    ),
    "wind_playing_sweep": StudyMeta(
        "Wind plays strategically", "Configuration", lambda r: None, bar=True
    ),
    "composition_sweep": StudyMeta(
        "Generation mix", "Generation Mix (Conv., Wind)", lambda r: None, bar=True
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


def _read_optimal(poa_file: Path) -> bool:
    """True only if the run terminated with a proven optimal solution."""
    with poa_file.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    term = str((data.get("solver") or {}).get("termination_condition") or "").lower()
    return term == "optimal"


def _pretty_run(run: str) -> str:
    return run.replace("_", " ")


def _composition_label(run: str) -> str:
    """``comp_2C_4W`` -> ``(2G, 4W)`` (G = conventional generator, W = wind)."""
    m = re.search(r"comp_(\d+)C_(\d+)W", run)
    if not m:
        return _pretty_run(run)
    return f"({m.group(1)}G, {m.group(2)}W)"


def _wind_bound_label(run: str) -> str:
    """``wind_mu_0p1_0p9`` -> ``[0.1, 0.9]`` (the (mu_min, mu_max) pair)."""
    rest = run[len("wind_mu_") :] if run.startswith("wind_mu_") else run
    parts = rest.split("_")
    if len(parts) != 2:
        return _pretty_run(run)
    lo, hi = (p.replace("p", ".") for p in parts)
    return f"[{lo}, {hi}]"


def _overlap_label(run: str) -> str:
    return {"base": "Non-overlapping", "overlapping": "Overlapping"}.get(run, _pretty_run(run))


# Appendix studies that get the thesis figure treatment: no title, no base
# reference line, y-axis "Ex-Post PoA", A4-friendly size, and (for categorical
# x-axes) custom tick labels. Value is a run-name -> tick-label function, or
# None when the x-axis is numeric and labels itself.
THESIS_TICK_LABELS: dict[str, Callable[[str], str] | None] = {
    "players_sweep": None,
    "composition_sweep": _composition_label,
    "peak_w_sweep": None,
    "rho_sweep": None,
    "sigma_max_sweep": None,
    "wind_bound_sweep": _wind_bound_label,
    "demand_ref_sweep": None,
    "ramp_rate_sweep": None,
    "bidding_blocks_sweep": None,
    "ambiguity_kappa_sweep": None,
    "overlapping_costs_sweep": _overlap_label,
}


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
            rows.append(
                {"run": run_dir.name, "x": x, "metric": metric, "optimal": _read_optimal(poa_file)}
            )
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


def _draw_panel(
    ax,
    study: str,
    rows: list[dict[str, Any]],
    base_ref: float | None,
    title_size: int = 11,
    ylabel_size: int = 8,
    *,
    show_title: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_tick_labels: list[str] | None = None,
    annot_fontsize: float = 8,
    xtick_fontsize: float = 8,
    ytick_fontsize: float | None = None,
    xlabel_size: float | None = None,
) -> None:
    rows = _sort_rows(rows)
    meta = STUDY_META.get(study)
    numeric = all(r["x"] is not None for r in rows)

    if numeric:
        xs = [r["x"] for r in rows]
        tick_labels = [f"{int(x) if float(x).is_integer() else x}" for x in xs]
    else:
        xs = list(range(len(rows)))
        tick_labels = (
            list(x_tick_labels)
            if x_tick_labels is not None
            else [_pretty_run(r["run"]) for r in rows]
        )
    ys = [r["metric"] for r in rows]

    opt_flags = [bool(r.get("optimal", True)) for r in rows]

    if meta is not None and meta.bar:
        # Scale the bar width to the data spacing so numeric x-axes (e.g. the
        # unevenly spaced peak-hour tau) get proportionate, not hairline, bars.
        if numeric and len(xs) > 1:
            gaps = [b - a for a, b in zip(sorted(xs), sorted(xs)[1:])]
            bar_width = 0.6 * min(g for g in gaps if g > 0)
        else:
            bar_width = 0.6
        ax.bar(xs, ys, color="#1f77b4", width=bar_width, zorder=3)
    else:
        ax.plot(xs, ys, "-o", color="#1f77b4", zorder=3)
    for x, y, ok in zip(xs, ys, opt_flags):
        ax.annotate(
            f"{y:.2f}" + ("" if ok else "*"),
            (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
            fontsize=annot_fontsize,
            color="black" if ok else "#d62728",
        )

    # Highlight runs that were NOT solved to proven optimality (e.g. time limit).
    non_opt = [(x, y) for x, y, ok in zip(xs, ys, opt_flags) if not ok]
    if non_opt:
        nx, ny = zip(*non_opt)
        ax.scatter(
            nx, ny, s=170, facecolors="none", edgecolors="#d62728", linewidths=2.2,
            zorder=5, label="Not solved to optimality",
        )
        ax.legend(fontsize=8, loc="best")

    if base_ref is not None:
        ax.axhline(base_ref, ls="--", lw=1, color="grey", zorder=1)
        ax.text(
            0.02,
            base_ref,
            f" base {base_ref:.2f}",
            color="grey",
            fontsize=7,
            va="bottom",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )

    if show_title:
        ax.set_title(meta.title if meta else study, fontsize=title_size)
    _xlabel = xlabel if xlabel is not None else (meta.xlabel if meta else "run")
    if xlabel_size is not None:
        ax.set_xlabel(_xlabel, fontsize=xlabel_size)
    else:
        ax.set_xlabel(_xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else METRIC_LABEL, fontsize=ylabel_size)
    horizontal_ticks = numeric or x_tick_labels is not None
    ax.set_xticks(xs)
    ax.set_xticklabels(
        tick_labels,
        rotation=0 if horizontal_ticks else 20,
        fontsize=xtick_fontsize,
        ha="center" if horizontal_ticks else "right",
    )
    if ytick_fontsize is not None:
        ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.grid(True, alpha=0.3)


def plot_individual(studies: dict[str, list[dict[str, Any]]], base_ref: float | None) -> None:
    """One standalone figure per study, saved inside that study's folder."""
    for study, rows in sorted(studies.items()):
        if study in THESIS_TICK_LABELS:
            # Thesis appendix figure: no title, no base reference line, y-axis
            # "Ex-Post PoA", A4-friendly size; non-optimal runs are highlighted
            # by _draw_panel. X-axis label comes from STUDY_META; categorical
            # studies supply custom tick labels.
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            label_fn = THESIS_TICK_LABELS[study]
            tick_labels = [label_fn(r["run"]) for r in _sort_rows(rows)] if label_fn else None
            _draw_panel(
                ax,
                study,
                rows,
                base_ref=None,
                show_title=False,
                ylabel="Ex-Post PoA",
                ylabel_size=12,
                x_tick_labels=tick_labels,
            )
        else:
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
