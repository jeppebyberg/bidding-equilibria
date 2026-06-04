"""Pipeline-triggered visualization hooks.

When FullPipelineConfig.plot_results_along_the_way is True, full_pipeline calls
these helpers at the matching stage boundaries:

  - plot_nn_policy_stage      after NN training
  - plot_base_poa_stage       after the base PoA analysis
  - run_oos_evaluation_stage  after the base PoA / DRO analysis (auto OOS eval)
  - plot_dro_stage            after the DRO eta sweep (with and without OOS)

All figures land under results/<case>/figures/<plot-group>/ so each run's plots
stay together with its case. Every stage is wrapped so a plotting failure logs a
warning and never aborts the pipeline run.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any


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
    from models.neural_network.tests.visualize_nn_policy_merit_order import (
        generate_merit_order_figures,
    )

    out_dir = figures_root(config) / "nn_policy_merit_order"
    scenarios_csv = Path(config.synthetic_scenario_dir) / "scenarios.csv"

    def _run() -> None:
        generate_merit_order_figures(
            result_json_path=config.heuristic_results_path,
            scenarios_csv_path=scenarios_csv,
            normalized_csv_dir=config.normalized_feature_dir,
            model_dir=config.model_dir,
            output_dir=out_dir,
            nn_generator_names=None,  # auto-detect trained *.pt files
            show=False,
        )
        print(f"[plot] Saved NN policy figures to {out_dir}")

    _guard("NN policy merit-order plots", _run)


# ---------------------------------------------------------------------------
# Stage: base PoA ambiguity trajectories + fresh-scenario coverage (OOS view)
# ---------------------------------------------------------------------------

def plot_base_poa_stage(config: Any, n_fresh: int = 300, fresh_seed: int = 42) -> None:
    from results_viz.plot_ambiguity_regime_trajectories import (
        plot_ambiguity_regime_trajectories,
        plot_fresh_scenario_coverage,
        plot_fresh_scenario_innovations,
    )

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
    # With OOS: fresh scenario draws checked against the AR(1) tube.
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
    print(f"[plot] Saved base PoA figures to {out_dir}")


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
        from driver.run_oos_poa_evaluation import (
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
