"""Block 4.5: evaluate realized PoA on fresh out-of-sample draws."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from driver.core.block45_core import (  # noqa: E402
    evaluate_oos_poa_for_regime,
    save_oos_results,
)
from driver.core.block2_core import discover_trained_policy_generators  # noqa: E402
from driver.core.block4_core import build_dro_config  # noqa: E402
from driver.core.visualization_core import plot_dro_stage  # noqa: E402
from driver.block0_system_setup import (  # noqa: E402
    build_config,
    write_manifest,
)


def run(
    config=None,
    n_scenarios: int = 200,
    seed: int = 999,
    plot_frontier: bool | None = None,
) -> dict[str, Any]:
    cfg = config or build_config()
    discovered = discover_trained_policy_generators(cfg.model_dir)
    if discovered:
        cfg.nn_policy_generators = discovered
        print(f"[block4.5] using trained policy generators: {discovered}")
    dcfg = build_dro_config(cfg)

    if not Path(dcfg.runtime_config_path).exists():
        raise FileNotFoundError(
            "PoA regime runtime config is missing. Run block3_poa_pipeline.py "
            f"first, or point runtime_config_path at an existing file: {dcfg.runtime_config_path}"
        )

    regime_names = list(dcfg.dro_regime_names)
    out_dir = Path(cfg.figures_dir).parent / "oos_poa"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for regime_name in regime_names:
        print(
            f"[block4.5] OOS PoA evaluation: regime='{regime_name}', "
            f"n={n_scenarios}, seed={seed}"
        )
        results.append(
            evaluate_oos_poa_for_regime(
                case=cfg.case,
                regime_name=regime_name,
                source_regime_config=dcfg.runtime_config_path,
                source_regime_set=dcfg.poa_regime_set,
                horizon=cfg.horizon,
                model_dir=cfg.model_dir,
                norm_stats_path=Path(cfg.normalized_feature_dir) / "min_max_stats.json",
                nn_policy_generators=list(cfg.nn_policy_generators),
                n_scenarios=n_scenarios,
                seed=seed,
                oos_config_path=out_dir / "oos_regime_config.json",
            )
        )

    output_path = out_dir / "oos_poa_results.json"
    save_oos_results(results, output_path)

    should_plot = cfg.plot_results_along_the_way if plot_frontier is None else plot_frontier
    if should_plot:
        plot_dro_stage(cfg, dcfg, regime_names, output_path)

    manifest = {
        "block": "block45_oos_poa",
        "runtime_config_path": dcfg.runtime_config_path,
        "oos_results_path": output_path,
        "regime_names": regime_names,
        "n_scenarios": n_scenarios,
        "seed": seed,
        "plotted_dro_frontier": bool(should_plot),
    }
    write_manifest("block45_oos_poa", manifest, cfg)
    return manifest


if __name__ == "__main__":
    run()
