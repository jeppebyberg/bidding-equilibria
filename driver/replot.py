"""Re-generate every figure group from existing results -- no Gurobi, no training.

Each plotting stage in ``visualization_core`` only reads result artifacts
(scenario CSVs, heuristic-label JSON, trained ``.pt`` policies, PoA/DRO result
JSONs). This driver wires those stages together against the current
``project_config`` so you can refresh figures after a run without repeating any
computation. It deliberately does NOT call any block's solve/tightening code.

Run all groups:

    .\.venv\Scripts\python.exe driver\replot.py

Run a subset (any of: setup nn base_poa support_oos dro dro_oos):

    .\.venv\Scripts\python.exe driver\replot.py base_poa dro

Figures land in the same place a normal run would write them
(``results/<case_label>/figures/...``), driven by ``config.figures_dir``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.scenarios.scenario_generator import ScenarioManager  # noqa: E402
from driver.core.block1_core import load_or_generate_scenarios  # noqa: E402
from driver.core.block4_core import build_dro_config  # noqa: E402
from driver.core.visualization_core import (  # noqa: E402
    plot_base_poa_stage,
    plot_dro_oos_stage,
    plot_dro_stage,
    plot_nn_policy_stage,
    plot_poa_support_oos_stage,
    plot_setup_stage,
)
from driver.block0_system_setup import build_config  # noqa: E402

ALL_GROUPS = ["setup", "nn", "base_poa", "support_oos", "dro", "dro_oos"]


def _run_group(name: str, fn) -> None:
    """Run one figure group, logging and swallowing any error so others proceed."""
    print(f"\n[replot] === {name} ===")
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - one group must not abort the rest
        print(f"[replot] Skipping '{name}': {exc}")
        traceback.print_exc()


def _replot_setup(cfg) -> None:
    manager = ScenarioManager(cfg.case)
    scenarios = load_or_generate_scenarios(
        config=cfg,
        manager=manager,
        n_scenarios=cfg.synthetic_num_scenarios,
        seed=cfg.synthetic_seed,
        output_dir=cfg.synthetic_scenario_dir,
        should_generate=False,  # reuse existing scenarios.csv
        time_steps=cfg.synthetic_time_steps,
    )
    plot_setup_stage(cfg, manager, scenarios)


def _replot_dro(cfg) -> None:
    dcfg = build_dro_config(cfg)
    regime_names = list(dcfg.dro_regime_names or [cfg.poa_worst_case_regime_name])
    # Overlay OOS PoA on the epsilon frontier only if an existing OOS result is
    # present; never trigger a (Gurobi) OOS evaluation here.
    oos_path = Path(cfg.figures_dir).parent / "oos_poa" / "oos_poa_results.json"
    plot_dro_stage(
        cfg,
        dcfg,
        regime_names,
        oos_results_path=oos_path if oos_path.exists() else None,
    )


def run(groups: list[str] | None = None) -> None:
    cfg = build_config()
    selected = groups or ALL_GROUPS
    unknown = [g for g in selected if g not in ALL_GROUPS]
    if unknown:
        raise SystemExit(
            f"Unknown group(s): {unknown}. Choose from: {', '.join(ALL_GROUPS)}"
        )

    print(f"[replot] case_label={cfg.case_label}, horizon={cfg.horizon}")
    print(f"[replot] figures root: {cfg.figures_dir}")
    print(f"[replot] groups: {', '.join(selected)}")

    if "setup" in selected:
        _run_group("setup", lambda: _replot_setup(cfg))
    if "nn" in selected:
        _run_group("nn policy merit-order", lambda: plot_nn_policy_stage(cfg))
    if "base_poa" in selected:
        _run_group("base PoA", lambda: plot_base_poa_stage(cfg))
    if "support_oos" in selected:
        _run_group("support OOS", lambda: plot_poa_support_oos_stage(cfg))
    if "dro" in selected:
        _run_group("DRO eta sweep", lambda: _replot_dro(cfg))
    if "dro_oos" in selected:
        _run_group("DRO OOS distribution", lambda: plot_dro_oos_stage(cfg))

    print("\n[replot] Done.")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
