"""Pipeline-triggered visualization hooks.

When FullPipelineConfig.plot_results_along_the_way is True, full_pipeline calls
these helpers at the matching stage boundaries:

  - plot_nn_policy_stage      after NN training
  - plot_base_poa_stage       after the base PoA analysis
  - plot_poa_support_oos_stage after block 3.5 support-band OOS diagnostics
  - run_oos_evaluation_stage  after the base PoA / DRO analysis (auto OOS eval)
  - plot_dro_stage            after the DRO eta sweep (with and without OOS)

All figures land under results/<case>/figures/<plot-group>/ so each run's plots
stay together with its case. Every stage is wrapped so a plotting failure logs a
warning and never aborts the pipeline run.
"""
from __future__ import annotations

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401

import traceback
from pathlib import Path
from typing import Any


def _plot_shape_functions(
    manager: Any,
    regime_config_path: str | None,
    regime_set: str | None,
    title: str,
    figsize: tuple,
    save_path: str | Path | None,
    show: bool,
) -> tuple:
    import matplotlib.pyplot as plt

    config = manager.load_regime_configuration(
        regime_config_path=regime_config_path, regime_set=regime_set
    )
    horizon = manager.base_case["time_steps"]
    time_index = range(horizon)
    demand_shape = manager._build_demand_shape(horizon)
    peak_hours = sorted({float(regime.peak_W) for regime in config["regimes"]})

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

    axes[0].plot(time_index, demand_shape, linewidth=2.2, label="h_D")
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    axes[0].set_title("Demand Shape")
    axes[0].set_ylabel("Multiplier")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    cmap = plt.get_cmap("tab10")
    for idx, peak_hour in enumerate(peak_hours):
        wind_shape = manager._build_wind_shape(horizon, peak_hour)
        axes[1].plot(
            time_index, wind_shape, linewidth=2.2,
            color=cmap(idx % cmap.N), label=f"h_W peak={peak_hour:g}",
        )
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    axes[1].set_title("Wind Shape")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Multiplier")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def _plot_generation_cost_curves(
    manager: Any,
    generator_names: list | None,
    generator_type: str,
    cost_kind: str,
    title: str | None,
    figsize: tuple,
    save_path: str | Path | None,
    show: bool,
) -> tuple:
    import matplotlib.pyplot as plt

    if cost_kind not in {"total", "marginal"}:
        raise ValueError("cost_kind must be either 'total' or 'marginal'")
    if generator_type not in {"all", "conventional", "wind"}:
        raise ValueError("generator_type must be one of: 'all', 'conventional', 'wind'")

    if generator_names is None:
        selected = list(manager.physical_generators)
        if generator_type == "conventional":
            selected = [g for g in selected if not bool(g["is_wind"])]
        elif generator_type == "wind":
            selected = [g for g in selected if bool(g["is_wind"])]
        selected_names = [g["physical_name"] for g in selected]
    else:
        selected_names = [str(n) for n in generator_names]

    unknown = [n for n in selected_names if n not in manager.physical_generator_names]
    if unknown:
        raise ValueError(f"Unknown physical generators: {unknown}")
    if not selected_names:
        raise ValueError(f"No generators match generator_type='{generator_type}'")

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    for gen_idx, gen_name in enumerate(selected_names):
        generator = next(g for g in manager.physical_generators if g["physical_name"] == gen_name)
        blocks = sorted(generator["blocks"], key=lambda b: int(b["block_id"]))
        if not blocks:
            continue
        color = cmap(gen_idx % cmap.N)
        if cost_kind == "total":
            production = [0.0]
            total_cost = [0.0]
            cum_q = cum_c = 0.0
            for b in blocks:
                cum_q += float(b["pmax"])
                cum_c += float(b["pmax"]) * float(b["cost"])
                production.append(cum_q)
                total_cost.append(cum_c)
            ax.plot(production, total_cost, marker="o", linewidth=2.0, color=color, label=gen_name)
        else:
            x_vals: list = []
            y_vals: list = []
            cum_q = 0.0
            for b in blocks:
                nxt = cum_q + float(b["pmax"])
                mc = float(b["cost"])
                x_vals.extend([cum_q, nxt])
                y_vals.extend([mc, mc])
                cum_q = nxt
            ax.step(x_vals, y_vals, where="post", linewidth=2.0, color=color, label=gen_name)

    default_title = (
        "Generation Total Cost Curves" if cost_kind == "total" else "Generation Marginal Cost Curves"
    )
    ax.set_title(title or default_title)
    ax.set_xlabel("Production (MW)")
    ax.set_ylabel("Total variable cost" if cost_kind == "total" else "Marginal cost")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Generator", loc="best")
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def _plot_demand_profiles(
    manager: Any,
    scenarios_df: Any,
    title: str,
    max_profiles_per_regime: int | None,
    show_regime_mean: bool,
    alpha: float,
    line_width: float,
    mean_line_width: float,
    figsize: tuple,
    random_state: int | None,
    save_path: str | Path | None,
    show: bool,
) -> tuple:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if "regime" not in scenarios_df.columns:
        raise ValueError("scenarios_df must include a 'regime' column")
    if "demand_profile" not in scenarios_df.columns:
        raise ValueError("scenarios_df must include a 'demand_profile' column")

    horizon = manager.base_case["time_steps"]
    regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())
    if not regimes:
        raise ValueError("No regimes found in scenarios_df['regime']")

    rng = np.random.default_rng(random_state)
    cmap = plt.get_cmap("tab10")
    regime_colors = {r: cmap(i % cmap.N) for i, r in enumerate(regimes)}

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []

    for regime in regimes:
        regime_df = scenarios_df[scenarios_df["regime"].astype(str) == regime]
        if regime_df.empty:
            continue
        indices = regime_df.index.to_list()
        if max_profiles_per_regime is not None and len(indices) > max_profiles_per_regime:
            indices = rng.choice(indices, size=max_profiles_per_regime, replace=False).tolist()

        profiles = []
        for idx in indices:
            profile = manager._profile_to_float_list(
                scenarios_df.at[idx, "demand_profile"], expected_horizon=horizon
            )
            arr = np.array(profile, dtype=float)
            profiles.append(arr)
            ax.plot(range(horizon), arr, color=regime_colors[regime], alpha=alpha, linewidth=line_width)

        if show_regime_mean and profiles:
            ax.plot(
                range(horizon), np.mean(np.vstack(profiles), axis=0),
                color=regime_colors[regime], linewidth=mean_line_width,
            )
        legend_handles.append(
            Line2D([0], [0], color=regime_colors[regime], lw=mean_line_width,
                   label=f"{regime} (n={len(regime_df)})")
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=legend_handles, title="Regime", loc="best")
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax, regime_colors


def _plot_wind_profiles(
    manager: Any,
    scenarios_df: Any,
    title: str,
    wind_generators: list | None,
    max_profiles_per_regime: int | None,
    show_regime_mean: bool,
    alpha: float,
    line_width: float,
    mean_line_width: float,
    figsize_per_generator: tuple,
    random_state: int | None,
    save_path: str | Path | None,
    show: bool,
) -> tuple:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if "regime" not in scenarios_df.columns:
        raise ValueError("scenarios_df must include a 'regime' column")

    all_wind = [b["block_name"] for b in manager.blocks if bool(b["is_wind"])]
    if wind_generators is None:
        wind_generators = all_wind
    if not wind_generators:
        raise ValueError("No wind generators available for plotting")
    for wn in wind_generators:
        if wn not in all_wind:
            raise ValueError(f"Unknown wind generator '{wn}'. Available: {all_wind}")
        if f"{wn}_profile" not in scenarios_df.columns:
            raise ValueError(f"Missing column '{wn}_profile' in scenarios_df")

    horizon = manager.base_case["time_steps"]
    regimes = sorted(scenarios_df["regime"].dropna().astype(str).unique().tolist())
    if not regimes:
        raise ValueError("No regimes found in scenarios_df['regime']")

    rng = np.random.default_rng(random_state)
    cmap = plt.get_cmap("tab10")
    regime_colors = {r: cmap(i % cmap.N) for i, r in enumerate(regimes)}

    fig_w, fig_h_per = float(figsize_per_generator[0]), float(figsize_per_generator[1])
    fig, axes = plt.subplots(
        nrows=len(wind_generators), ncols=1,
        figsize=(fig_w, fig_h_per * len(wind_generators)), sharex=True,
    )
    if len(wind_generators) == 1:
        axes = [axes]

    for ax, wn in zip(axes, wind_generators):
        profile_col = f"{wn}_profile"
        for regime in regimes:
            regime_df = scenarios_df[scenarios_df["regime"].astype(str) == regime]
            if regime_df.empty:
                continue
            indices = regime_df.index.to_list()
            if max_profiles_per_regime is not None and len(indices) > max_profiles_per_regime:
                indices = rng.choice(indices, size=max_profiles_per_regime, replace=False).tolist()

            profiles = []
            for idx in indices:
                arr = np.array(
                    manager._profile_to_float_list(scenarios_df.at[idx, profile_col], expected_horizon=horizon),
                    dtype=float,
                )
                profiles.append(arr)
                ax.plot(range(horizon), arr, color=regime_colors[regime], alpha=alpha, linewidth=line_width)

            if show_regime_mean and profiles:
                ax.plot(
                    range(horizon), np.mean(np.vstack(profiles), axis=0),
                    color=regime_colors[regime], linewidth=mean_line_width,
                )
        ax.set_title(f"{wn} Wind Profile")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)

    counts = scenarios_df["regime"].astype(str).value_counts().to_dict()
    legend_handles = [
        Line2D([0], [0], color=regime_colors[r], lw=mean_line_width,
               label=f"{r} (n={counts.get(r, 0)})")
        for r in regimes
    ]
    fig.legend(handles=legend_handles, title="Regime", loc="upper right")
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])
    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes, regime_colors


# ---------------------------------------------------------------------------
# Stage: system setup — shape functions, cost curves, scenario profiles
# ---------------------------------------------------------------------------

def plot_setup_stage(config: Any, manager: Any, scenarios: dict) -> None:
    out_dir = figures_root(config) / "setup"
    scenarios_df = scenarios["scenarios_df"]

    _guard(
        "setup shape functions",
        _plot_shape_functions,
        manager,
        regime_config_path=None,
        regime_set=None,
        title="Deterministic Shape Functions",
        figsize=(11.0, 7.0),
        save_path=out_dir / "shape_functions.png",
        show=False,
    )
    _guard(
        "setup cost curves",
        _plot_generation_cost_curves,
        manager,
        generator_names=None,
        generator_type="all",
        cost_kind="total",
        title=None,
        figsize=(10.0, 6.0),
        save_path=out_dir / "cost_curves.png",
        show=False,
    )
    _guard(
        "setup demand profiles",
        _plot_demand_profiles,
        manager,
        scenarios_df,
        title="Sampled Demand Profiles by Regime",
        max_profiles_per_regime=100,
        show_regime_mean=True,
        alpha=0.05,
        line_width=1.0,
        mean_line_width=2.5,
        figsize=(11.0, 6.0),
        random_state=None,
        save_path=out_dir / "demand_profiles.png",
        show=False,
    )
    _guard(
        "setup wind profiles",
        _plot_wind_profiles,
        manager,
        scenarios_df,
        title="Sampled Wind Profiles by Regime",
        wind_generators=None,
        max_profiles_per_regime=100,
        show_regime_mean=True,
        alpha=0.05,
        line_width=1.0,
        mean_line_width=2.5,
        figsize_per_generator=(11.0, 3.2),
        random_state=None,
        save_path=out_dir / "wind_profiles.png",
        show=False,
    )
    print(f"[plot] Saved setup figures to {out_dir}")


def figures_root(config: Any) -> Path:
    """Root folder for all pipeline-triggered figures of a run.

    Uses config.figures_dir when set (sensitivity sweeps point it at each
    composition's folder); otherwise falls back to results/<case>/figures.
    """
    figures_dir = getattr(config, "figures_dir", None)
    return Path(figures_dir) if figures_dir else Path("results") / config.case / "figures"


def _guard(stage_name: str, fn, *args, **kwargs) -> Any:
    """Run a plotting stage, logging and swallowing any error."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - plots must never abort the pipeline
        print(f"\n[plot] Skipping {stage_name}: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Stage: NN policy merit-order + prediction scatter (after training)
# ---------------------------------------------------------------------------

def plot_nn_policy_stage(config: Any) -> None:
    from results_viz.visualize_nn_policy_merit_order import (
        generate_merit_order_figures,
    )

    out_dir = figures_root(config) / "nn_policy_merit_order"
    scenarios_csv = Path(config.synthetic_scenario_dir) / "scenarios.csv"

    # Show one time step at a time: first / middle / last of the horizon, each as
    # its own figure (3 separate merit-order plots).
    horizon = int(config.horizon)
    merit_time_steps = sorted({0, horizon // 2, horizon - 1})

    def _run() -> None:
        generate_merit_order_figures(
            result_json_path=config.heuristic_results_path,
            scenarios_csv_path=scenarios_csv,
            normalized_csv_dir=config.normalized_feature_dir,
            model_dir=config.model_dir,
            output_dir=out_dir,
            nn_generator_names=None,  # auto-detect trained *.pt files
            n_merit_scenarios=1,
            merit_time_steps=merit_time_steps,
            single_time_step_figures=True,
            show=False,
        )
        print(f"[plot] Saved NN policy figures to {out_dir}")

    _guard("NN policy merit-order plots", _run)


# ---------------------------------------------------------------------------
# Stage: base PoA ambiguity trajectories
# ---------------------------------------------------------------------------

def plot_base_poa_stage(config: Any) -> None:
    from results_viz.plot_ambiguity_regime_trajectories import (
        plot_ambiguity_regime_trajectories,
    )
    from results_viz.visualize_poa_trajectory import generate_poa_trajectory_figures

    out_dir = figures_root(config) / "base_poa"
    result_path = Path(config.poa_results_path)
    if not result_path.exists():
        print(f"[plot] Skipping base PoA plots: result not found at {result_path}")
        return

    # Without OOS: optimized trajectory against the support set.
    _guard(
        "ambiguity regime trajectories",
        plot_ambiguity_regime_trajectories,
        result_path=result_path,
        output_dir=out_dir,
        show=False,
    )
    _guard(
        "PoA trajectory merit order",
        generate_poa_trajectory_figures,
        result_json_path=result_path,
        output_dir=out_dir / "merit_order_curves",
        show=False,
    )
    print(f"[plot] Saved base PoA figures to {out_dir}")


def plot_poa_support_oos_stage(config: Any) -> None:
    from results_viz.plot_ambiguity_regime_trajectories import (
        plot_fresh_scenario_coverage,
        plot_fresh_scenario_innovations,
    )

    out_dir = figures_root(config) / "support_oos"
    result_path = Path(config.poa_results_path)
    if not result_path.exists():
        print(f"[plot] Skipping PoA support OOS plots: result not found at {result_path}")
        return

    n_fresh = int(getattr(config, "support_verify_num_draws", 300))
    fresh_seed = int(getattr(config, "support_verify_seed", 42))
    _guard(
        "fresh scenario coverage",
        plot_fresh_scenario_coverage,
        result_path=result_path,
        N=n_fresh,
        seed=fresh_seed,
        output_dir=out_dir,
        show=False,
    )
    _guard(
        "fresh scenario innovations",
        plot_fresh_scenario_innovations,
        result_path=result_path,
        N=n_fresh,
        seed=fresh_seed,
        output_dir=out_dir,
        show=False,
    )
    print(f"[plot] Saved PoA support OOS figures to {out_dir}")


# ---------------------------------------------------------------------------
# Stage: out-of-sample PoA evaluation (auto-run so DRO plots have OOS overlays)
# ---------------------------------------------------------------------------

def run_oos_evaluation_stage(
    config: Any,
    dro_config: Any,
    regime_names: list[str],
    n_scenarios: int = 200,
    seed: int = 999,
) -> Path | None:
    """Run OOS PoA evaluation for each DRO regime; return the results JSON path."""
    if not regime_names:
        print("[plot] Skipping OOS evaluation: no DRO regimes resolved.")
        return None

    def _run() -> Path:
        from driver.core.block45_core import (
            evaluate_oos_poa_for_regime,
            save_oos_results,
        )

        oos_dir = figures_root(config).parent / "oos_poa"
        oos_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        for regime_name in regime_names:
            print(f"[plot] OOS PoA evaluation: regime='{regime_name}', n={n_scenarios}, seed={seed}")
            results.append(
                evaluate_oos_poa_for_regime(
                    case=config.case,
                    regime_name=regime_name,
                    source_regime_config=dro_config.runtime_config_path,
                    source_regime_set=dro_config.poa_regime_set,
                    horizon=config.horizon,
                    model_dir=config.model_dir,
                    norm_stats_path=Path(config.normalized_feature_dir) / "min_max_stats.json",
                    nn_policy_generators=list(config.nn_policy_generators),
                    n_scenarios=n_scenarios,
                    seed=seed,
                    oos_config_path=oos_dir / "oos_regime_config.json",
                )
            )
        out_path = oos_dir / "oos_poa_results.json"
        save_oos_results(results, out_path)
        return out_path

    return _guard("OOS PoA evaluation", _run)


# ---------------------------------------------------------------------------
# Stage: DRO out-of-sample PoA distribution (artifact-only, no Gurobi)
# ---------------------------------------------------------------------------

def plot_dro_oos_stage(config: Any) -> None:
    """Plot the OOS PoA distribution from an existing oos_poa_results.json.

    Reads only the block-4.5 artifact (per-scenario ``poa_ratios`` plus the
    mean/max summaries); it never triggers a (Gurobi) OOS evaluation. The DRO
    eta-frontier overlay is handled separately by plot_dro_stage.
    """
    import json

    import matplotlib.pyplot as plt
    import numpy as np

    oos_path = figures_root(config).parent / "oos_poa" / "oos_poa_results.json"
    if not oos_path.exists():
        print(f"[plot] Skipping DRO OOS distribution: result not found at {oos_path}")
        return

    results = json.loads(oos_path.read_text(encoding="utf-8"))
    if isinstance(results, dict):
        results = [results]
    if not results:
        print(f"[plot] Skipping DRO OOS distribution: no records in {oos_path}")
        return

    out_dir = figures_root(config) / "dro_oos"

    for record in results:
        regime_name = str(record.get("regime_name", "regime"))

        def _run(regime_name: str = regime_name, record: dict = record) -> None:
            ratios = np.array(record.get("poa_ratios", []), dtype=float)
            ratios = ratios[~np.isnan(ratios)]
            if ratios.size == 0:
                print(f"[plot] Skipping DRO OOS distribution for '{regime_name}': no PoA ratios")
                return

            mean_poa = float(record.get("oos_mean_poa_ratio", np.mean(ratios)))
            max_poa = float(record.get("oos_max_poa_ratio", np.max(ratios)))
            n_used = int(record.get("n_scenarios_used", ratios.size))
            seed = record.get("seed", "?")

            out_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.hist(ratios, bins=30, color="tab:blue", alpha=0.72,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(mean_poa, color="tab:orange", linewidth=2.0,
                       label=f"Mean = {mean_poa:.4f}")
            ax.axvline(max_poa, color="tab:red", linewidth=1.6, linestyle="--",
                       label=f"Max = {max_poa:.4f}")
            ax.axvline(1.0, color="0.4", linewidth=1.2, linestyle=":", alpha=0.7,
                       label="PoA = 1")
            ax.set_xlabel("PoA ratio  (C_eq / C_opt)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(
                f"DRO Out-of-Sample PoA Distribution -- regime '{regime_name}'\n"
                f"N={n_used} scenarios, seed={seed}",
                fontsize=11,
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            out_path = out_dir / f"{regime_name}_oos_poa_distribution.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] Saved DRO OOS distribution for '{regime_name}' to {out_path}")

        _guard(f"DRO OOS distribution for '{regime_name}'", _run)


# ---------------------------------------------------------------------------
# Stage: DRO eta sweep + PoA-epsilon frontier (with and without OOS)
# ---------------------------------------------------------------------------

def plot_dro_stage(
    config: Any,
    dro_config: Any,
    regime_names: list[str],
    oos_results_path: Path | None,
) -> None:
    from results_viz.plot_dro_poa_eta_sweep import (
        _load_oos_poa_by_regime,
        discover_regime_names,
        load_eta_sweep_records,
        plot_poa_epsilon_frontier,
        plot_poa_eta_sweep,
        write_summary_csv,
    )

    results_dir = Path(dro_config.dro_result_dir)
    out_root = figures_root(config) / "dro_poa_eta_sweep"

    names = list(regime_names) or discover_regime_names(results_dir)
    if not names:
        print(f"[plot] Skipping DRO plots: no eta-sweep results under {results_dir}")
        return

    oos_by_regime: dict[str, dict[str, float]] = {}
    if oos_results_path is not None and Path(oos_results_path).exists():
        oos_by_regime = _load_oos_poa_by_regime(Path(oos_results_path))

    for regime_name in names:
        def _run(regime_name: str = regime_name) -> None:
            records = load_eta_sweep_records(results_dir, regime_name, include_archives=True)
            # Write directly under dro_poa_eta_sweep/ (no per-regime subfolder).
            # Plot/CSV filenames are regime-prefixed, so multiple regimes do not
            # collide.
            write_summary_csv(records, out_root / f"{regime_name}_eta_sweep_summary.csv")
            plot_poa_eta_sweep(
                records=records, output_dir=out_root, regime_name=regime_name, show=False,
            )
            # Without OOS overlay.
            plot_poa_epsilon_frontier(
                records=records, output_dir=out_root, regime_name=regime_name,
                poa_metric="worst_case_expected_poa", show=False,
            )
            # With OOS overlay (suffixed filename so it does not collide with the
            # no-OOS frontier).
            oos = oos_by_regime.get(regime_name, {})
            if oos:
                plot_poa_epsilon_frontier(
                    records=records, output_dir=out_root,
                    regime_name=regime_name, poa_metric="worst_case_expected_poa",
                    oos_mean_poa=oos.get("mean"), oos_max_poa=oos.get("max"), show=False,
                    filename_suffix="_oos",
                )
            print(f"[plot] Saved DRO figures for '{regime_name}' to {out_root}")

        _guard(f"DRO plots for regime '{regime_name}'", _run)

    # Perturbed-distribution visualizations (fan, summary, transport scatter)
    _guard(
        "DRO perturbed distribution plots",
        plot_dro_perturbed_distribution_stage,
        config=config,
        dro_config=dro_config,
        regime_names=names,
        results_dir=results_dir,
    )


# ---------------------------------------------------------------------------
# Stage: DRO perturbed distribution (fan, summary, transport scatter)
# ---------------------------------------------------------------------------

def plot_dro_perturbed_distribution_stage(
    config: Any,
    regime_names: list[str],
    results_dir: Path,
    **_: Any,
) -> None:
    from results_viz.plot_dro_perturbed_distribution import plot_dro_perturbed_distribution

    out_root = figures_root(config) / "dro_perturbed_distribution"

    for regime_name in regime_names:
        regime_dir = results_dir / regime_name
        if not regime_dir.exists():
            print(f"[plot] Skipping perturbed distribution for '{regime_name}': "
                  f"directory not found at {regime_dir}")
            continue

        def _run(regime_name: str = regime_name, regime_dir: Path = regime_dir) -> None:
            plot_dro_perturbed_distribution(
                regime_dir=regime_dir,
                output_dir=out_root,
                show=False,
            )
            print(f"[plot] Saved perturbed distribution figures for '{regime_name}' "
                  f"to {out_root / regime_name}")

        _guard(f"DRO perturbed distribution for '{regime_name}'", _run)


# ---------------------------------------------------------------------------
# Stage: cross-run sensitivity comparison (after a whole study completes)
# ---------------------------------------------------------------------------

def plot_sensitivity_comparison_stage(
    study_name: str,
    result_root: Path,
    regime_names: list[str] | None = None,
) -> None:
    """Overlay every run's PoA-vs-eta curve and PoA-epsilon frontier.

    Produces one comparison figure per DRO regime, one curve per run, written
    to <result_root>/<study_name>/. Regimes default to the union of those that
    produced DRO eta-sweep results across the study's runs. Each figure is
    guarded so a plotting failure never aborts the study.
    """
    from results_viz.plot_dro_poa_eta_sweep import discover_regime_names
    from results_viz.plot_sensitivity_comparison import (
        plot_epsilon_frontier_comparison,
        plot_eta_sweep_comparison,
    )

    study_dir = Path(result_root) / study_name
    if not study_dir.exists():
        print(f"[plot] Skipping sensitivity comparison: study dir not found at {study_dir}")
        return

    if regime_names:
        names = list(regime_names)
    else:
        names = []
        for case_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            for name in discover_regime_names(case_dir / "dro"):
                if name not in names:
                    names.append(name)

    if not names:
        print(f"[plot] Skipping sensitivity comparison: no DRO regimes under {study_dir}")
        return

    for regime_name in names:
        _guard(
            f"sensitivity eta-sweep comparison ('{regime_name}')",
            plot_eta_sweep_comparison,
            study_name=study_name,
            regime_name=regime_name,
            results_root=Path(result_root),
            output_root=Path(result_root),
        )
        _guard(
            f"sensitivity frontier comparison ('{regime_name}')",
            plot_epsilon_frontier_comparison,
            study_name=study_name,
            regime_name=regime_name,
            results_root=Path(result_root),
            output_root=Path(result_root),
        )
    print(f"[plot] Saved sensitivity comparison figures under {study_dir}")
