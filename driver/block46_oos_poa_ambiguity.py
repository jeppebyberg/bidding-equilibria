# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-17
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Block 4.6: realized inefficiency over a wide OOS draw from the ambiguity set.

Unlike block 4.5 (which draws from a single PoA regime), this block draws fresh
out-of-sample scenarios the same way ``create_scenario_set_from_ambiguity_set``
does -- sampling the stochastic-process parameters (mu_D, sigma_D, mu_W, sigma_W)
uniformly over the ambiguity-set bounds. It then applies the trained NN bidding
policy and solves two dispatch LPs per scenario (equilibrium vs. social optimum),
reporting the per-scenario inefficiency C_eq / C_opt across the whole draw.

Use this to see how badly the learned policy degrades market efficiency across
the full support, rather than only at the worst-case regime the MILP reports.
"""

from __future__ import annotations

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from driver.core.block45_core import (  # noqa: E402
    evaluate_oos_poa_for_regime,
    evaluate_oos_poa_from_ambiguity_set,
)
from driver.core.block2_core import discover_trained_policy_generators  # noqa: E402
from driver.block0_system_setup import build_config, write_manifest  # noqa: E402


def _plot_inefficiency_distribution(result: dict[str, Any], output_path: Path) -> None:
    """Histogram + summary lines of the per-scenario inefficiency distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = np.asarray(result["poa_ratios"], dtype=float)
    if ratios.size == 0:
        print("[block4.6] no valid inefficiency ratios to plot.")
        return

    mean = float(np.mean(ratios))
    median = float(np.median(ratios))
    p95 = float(np.percentile(ratios, 95))
    worst = float(np.max(ratios))

    source_label = (
        f"ambiguity set '{result['ambiguity_set']}'"
        if "ambiguity_set" in result
        else f"regime '{result.get('regime_name', '?')}'"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1, label="efficient (1.0)")
    ax.axvline(mean, color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean = {mean:.3f}")
    ax.axvline(median, color="#55A868", linestyle="--", linewidth=1.5, label=f"median = {median:.3f}")
    ax.axvline(p95, color="#8172B3", linestyle="--", linewidth=1.5, label=f"p95 = {p95:.3f}")
    ax.set_xlabel("Realized inefficiency  C_eq / C_opt")
    ax.set_ylabel("Number of OOS scenarios")
    ax.set_title(
        f"OOS inefficiency over {source_label}\n"
        f"n={result['n_scenarios_used']}, worst={worst:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[block4.6] saved inefficiency histogram: {output_path}")


def _scenario_ratios_and_params(
    result: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pair per-scenario inefficiency with its drawn parameters.

    c_eq, c_opt and scenario_params are all full-length and aligned, so the ratio
    is recomputed here (rather than reusing the filtered poa_ratios list) to keep
    it matched 1:1 with the parameter draws. Scenarios with C_opt ~ 0 are dropped.
    """
    c_eq = np.asarray(result["c_eq"], dtype=float)
    c_opt = np.asarray(result["c_opt"], dtype=float)
    params_list = result.get("scenario_params", [])

    valid = c_opt > 1e-9
    ratios = c_eq[valid] / c_opt[valid]

    param_names = sorted({k for p in params_list for k in p})
    params: dict[str, np.ndarray] = {}
    for name in param_names:
        col = np.asarray([p.get(name, np.nan) for p in params_list], dtype=float)
        params[name] = col[valid]
    return ratios, params


def _plot_inefficiency_vs_params(result: dict[str, Any], output_path: Path) -> None:
    """Scatter of realized inefficiency against each varying drawn parameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios, params = _scenario_ratios_and_params(result)
    if ratios.size == 0:
        print("[block4.6] no valid scenarios to scatter.")
        return

    # Only parameters that actually vary across the draw are informative
    # (rho_D, rho_W, peak_W are fixed in the ambiguity set).
    varying = {
        name: col
        for name, col in params.items()
        if np.nanmax(col) - np.nanmin(col) > 1e-9
    }
    if not varying:
        print("[block4.6] no varying parameters to scatter against.")
        return

    names = sorted(varying)
    n_panels = len(names)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False
    )

    for idx, name in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        x = varying[name]
        ax.scatter(x, ratios, s=18, alpha=0.5, color="#4C72B0", edgecolor="none")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        # Pearson correlation as a quick "which state drives it" signal.
        if np.std(x) > 0 and np.std(ratios) > 0:
            r = float(np.corrcoef(x, ratios)[0, 1])
            ax.set_title(f"{name}   (r = {r:+.2f})")
        else:
            ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("C_eq / C_opt")

    # Hide any unused panels.
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(
        f"OOS inefficiency vs. drawn parameters "
        f"(ambiguity set '{result['ambiguity_set']}', n={ratios.size})"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[block4.6] saved inefficiency-vs-parameter scatter: {output_path}")


def _plot_combined_distributions(
    amb_result: dict[str, Any],
    regime_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Overlay the ambiguity-set and worst-case-regime inefficiency histograms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amb = np.asarray(amb_result["poa_ratios"], dtype=float)
    reg = np.asarray(regime_result["poa_ratios"], dtype=float)
    if amb.size == 0 or reg.size == 0:
        print("[block4.6] not enough data for combined distribution plot.")
        return

    # Shared bins across both draws so the bars are directly comparable.
    lo = float(min(amb.min(), reg.min()))
    hi = float(max(amb.max(), reg.max()))
    bins = np.linspace(lo, hi, 40)

    series = [
        (r"Uncertainty set $\mathcal{R}$", amb, "#4C72B0"),
        (r"Regime $r^*$", reg, "#C44E52"),
    ]

    # Sized for the A4 text width (~16 cm) so labels stay readable when included ~1:1.
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    for label, data, color in series:
        ax.hist(
            data,
            bins=bins,
            color=color,
            alpha=0.5,
            edgecolor="white",
            label=f"{label} (mean={data.mean():.2f})",
        )
        ax.axvline(float(data.mean()), color=color, linestyle="--", linewidth=1.5)

    ax.set_xlabel("PoA", fontsize=14)
    ax.set_ylabel("Number of OOS scenarios", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[block4.6] saved combined inefficiency distribution: {output_path}")


def run(
    config=None,
    n_scenarios: int = 500,
    seed: int = 999,
    ambiguity_set: str | None = None,
    enforce_dispatch_feasibility: bool = True,
    enforce_support_set: bool = False,
    plot: bool | None = None,
) -> dict[str, Any]:
    cfg = config or build_config()

    discovered = discover_trained_policy_generators(cfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        print(f"[block4.6] using trained policy generators: {discovered}")

    ambiguity_set = ambiguity_set or cfg.ambiguity_set_config_name

    print(
        f"[block4.6] OOS inefficiency from ambiguity set '{ambiguity_set}': "
        f"n={n_scenarios}, seed={seed}"
    )
    result = evaluate_oos_poa_from_ambiguity_set(
        case=cfg.case,
        horizon=cfg.horizon,
        model_dir=Path(cfg.model_dir),
        norm_stats_path=cfg.nn_normalization_stats_path,
        nn_policy_generators=list(cfg.nn_policy_generators),
        ambiguity_set_config_path=cfg.ambiguity_set_config_path,
        ambiguity_set=ambiguity_set,
        n_scenarios=n_scenarios,
        seed=seed,
        enforce_dispatch_feasibility=enforce_dispatch_feasibility,
        enforce_support_set=enforce_support_set,
    )

    print(
        f"[block4.6] inefficiency C_eq/C_opt: "
        f"mean={result['oos_mean_poa_ratio']:.4f}, "
        f"max={result['oos_max_poa_ratio']:.4f} "
        f"(from {result['n_scenarios_used']} scenarios)"
    )

    out_dir = Path(cfg.figures_dir).parent / "oos_poa"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "oos_poa_ambiguity_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[block4.6] saved results: {results_path}")

    should_plot = cfg.plot_results_along_the_way if plot is None else plot
    if should_plot:
        _plot_inefficiency_distribution(
            result, out_dir / "oos_inefficiency_distribution.png"
        )
        _plot_inefficiency_vs_params(
            result, out_dir / "oos_inefficiency_vs_params.png"
        )

    manifest = {
        "block": "block46_oos_poa_ambiguity",
        "ambiguity_set": ambiguity_set,
        "oos_results_path": str(results_path),
        "n_scenarios_requested": n_scenarios,
        "n_scenarios_used": result["n_scenarios_used"],
        "seed": seed,
        "oos_mean_poa_ratio": result["oos_mean_poa_ratio"],
        "oos_max_poa_ratio": result["oos_max_poa_ratio"],
    }
    write_manifest("block46_oos_poa_ambiguity", manifest, cfg)
    return manifest


def run_worst_case_regime(
    config=None,
    n_scenarios: int = 500,
    seed: int = 999,
    plot: bool | None = None,
) -> dict[str, Any]:
    """OOS inefficiency drawn from the worst-case PoA regime (bar chart only).

    Companion to ``run`` (ambiguity-set draw): same number of scenarios and the
    same C_eq / C_opt measure, but scenarios are drawn from the single
    worst-case regime the PoA pipeline identified rather than uniformly over the
    ambiguity set. Only the inefficiency-distribution histogram is produced.
    """
    cfg = config or build_config()

    discovered = discover_trained_policy_generators(cfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        print(f"[block4.6] using trained policy generators: {discovered}")

    runtime_config_path = Path(cfg.runtime_config_path)
    if not runtime_config_path.exists():
        raise FileNotFoundError(
            "PoA regime runtime config is missing. Run block3_poa_pipeline.py "
            f"first: {runtime_config_path}"
        )

    regime_name = cfg.poa_worst_case_regime_name
    out_dir = Path(cfg.figures_dir).parent / "oos_poa"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[block4.6] OOS inefficiency from worst-case regime '{regime_name}': "
        f"n={n_scenarios}, seed={seed}"
    )
    result = evaluate_oos_poa_for_regime(
        case=cfg.case,
        regime_name=regime_name,
        source_regime_config=runtime_config_path,
        source_regime_set=cfg.poa_regime_set,
        horizon=cfg.horizon,
        model_dir=Path(cfg.model_dir),
        norm_stats_path=cfg.nn_normalization_stats_path,
        nn_policy_generators=list(cfg.nn_policy_generators),
        n_scenarios=n_scenarios,
        seed=seed,
        oos_config_path=out_dir / "oos_worst_case_regime_config.json",
    )

    print(
        f"[block4.6] inefficiency C_eq/C_opt: "
        f"mean={result['oos_mean_poa_ratio']:.4f}, "
        f"max={result['oos_max_poa_ratio']:.4f} "
        f"(from {result['n_scenarios_used']} scenarios)"
    )

    results_path = out_dir / "oos_poa_worst_case_regime_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[block4.6] saved results: {results_path}")

    should_plot = cfg.plot_results_along_the_way if plot is None else plot
    if should_plot:
        _plot_inefficiency_distribution(
            result, out_dir / "oos_inefficiency_distribution_worst_case_regime.png"
        )

    manifest = {
        "block": "block46_oos_poa_worst_case_regime",
        "regime_name": regime_name,
        "oos_results_path": str(results_path),
        "n_scenarios_requested": n_scenarios,
        "n_scenarios_used": result["n_scenarios_used"],
        "seed": seed,
        "oos_mean_poa_ratio": result["oos_mean_poa_ratio"],
        "oos_max_poa_ratio": result["oos_max_poa_ratio"],
    }
    write_manifest("block46_oos_poa_worst_case_regime", manifest, cfg)
    return manifest


def run_both(
    config=None,
    n_scenarios: int = 500,
    seed: int = 999,
    ambiguity_set: str | None = None,
) -> dict[str, Any]:
    """Run both OOS draws and overlay their inefficiency distributions.

    Produces the per-draw outputs (via ``run`` and ``run_worst_case_regime``)
    plus a single combined figure overlaying the two histograms.
    """
    cfg = config or build_config()
    out_dir = Path(cfg.figures_dir).parent / "oos_poa"

    run(
        config=cfg,
        n_scenarios=n_scenarios,
        seed=seed,
        ambiguity_set=ambiguity_set,
    )
    run_worst_case_regime(config=cfg, n_scenarios=n_scenarios, seed=seed)

    with (out_dir / "oos_poa_ambiguity_results.json").open(encoding="utf-8") as f:
        amb_result = json.load(f)
    with (out_dir / "oos_poa_worst_case_regime_results.json").open(encoding="utf-8") as f:
        regime_result = json.load(f)

    combined_path = out_dir / "oos_inefficiency_distribution_combined.png"
    _plot_combined_distributions(amb_result, regime_result, combined_path)

    manifest = {
        "block": "block46_oos_poa_combined",
        "n_scenarios": n_scenarios,
        "seed": seed,
        "combined_figure_path": str(combined_path),
        "ambiguity_mean_poa": amb_result["oos_mean_poa_ratio"],
        "worst_case_regime_mean_poa": regime_result["oos_mean_poa_ratio"],
    }
    write_manifest("block46_oos_poa_combined", manifest, cfg)
    return manifest


if __name__ == "__main__":
    run_both()
