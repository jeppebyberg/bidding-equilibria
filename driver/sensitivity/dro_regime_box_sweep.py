"""Sensitivity study: solve the base-case DRO eta-sweep on regimes placed at the
mean and outer values of the ``base_test_case`` ambiguity-set box.

The base pipeline centers the DRO ambiguity ball on the single PoA-optimized
worst-case state (``poa_worst_case``). This study instead probes how the DRO
frontier moves as the *center regime* is relocated within the ambiguity box. The
four regime parameters that the box bounds -- ``mu_D``, ``sigma_D``, ``mu_W``,
``sigma_W`` -- are swept one-at-a-time: a baseline "mean" regime sits at the box
midpoint, then each parameter is pushed to its lower and upper box bound in turn
while the other three stay at their mean. That isolates each parameter's effect
on the realized PoA. ``rho`` (rho_D / rho_W) and ``peak_W`` are held fixed at the
ambiguity-set values (rho_fixed / tau_fixed) -- they are not swept.

Box bounds and the held-fixed values are read live from the ambiguity-set config
so the sweep stays in sync with ``config/ambiguity_set_config.yaml``. With the
default base_test_case box this yields:

    mu_D    : [0.75, 1.0, 1.25]        sigma_D : [0.001, 0.013, 0.025]
    mu_W    : [0.25, 0.5,  0.75]       sigma_W : [0.001, 0.013, 0.025]
    rho_D = rho_W = 0.75 (fixed)       peak_W  = 14.0 (fixed)

Trained policies, features and normalization stats are reused from an existing
run (default ``results/base_case``); only the DRO leg (tightening + eta sweep) is
recomputed per regime. Each regime is solved into its own isolated run directory
under ``results/sensitivity_studies/dro_regime_box/<regime>/``.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.dro_regime_box_sweep
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driver import block4_dro_poa_pipeline  # noqa: E402
from driver.core.block0_core import ProjectConfig  # noqa: E402
from driver.core.block1_core import write_json  # noqa: E402
from driver.sensitivity.sensitivity_config import (  # noqa: E402
    RESULT_ROOT,
    isolate_run_outputs,
    load_base_config,
)

STUDY_NAME = "dro_regime_box"

# Source of the trained policies / features / normalization stats to reuse. The
# DRO only reads these, so the base case is left untouched.
SOURCE_RUN_DIR = Path("results/base_case")

# Ambiguity-set config that defines the box the regimes are placed in. The base
# case centers its DRO on this same set, so its bounds are the natural span.
AMBIGUITY_CONFIG_PATH = PROJECT_ROOT / "config" / "ambiguity_set_config.yaml"
AMBIGUITY_SET_NAME = "base_test_case"

# Empirical scenarios drawn per regime for the DRO MILP. Matches the base case's
# poa_worst_case_n_scenarios (the DRO is solved per-scenario, so keep it small).
N_EMPIRICAL_SCENARIOS = 10

# Regime-set name written into each run's runtime config and read back by block4.
RUNTIME_REGIME_SET = "dro_box_runtime"


def _load_box() -> dict[str, Any]:
    """Read the swept-parameter box and the held-fixed values from the set."""
    with AMBIGUITY_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    entry = (raw.get("ambiguity_sets", {}) or {}).get(AMBIGUITY_SET_NAME)
    if entry is None:
        raise ValueError(
            f"Ambiguity set '{AMBIGUITY_SET_NAME}' not in {AMBIGUITY_CONFIG_PATH}."
        )
    demand = entry["demand"]
    wind = entry["wind"]
    return {
        "mu_D": (float(demand["mu"]["min"]), float(demand["mu"]["max"])),
        "sigma_D": (float(demand["sigma"]["min"]), float(demand["sigma"]["max"])),
        "mu_W": (float(wind["mu"]["min"]), float(wind["mu"]["max"])),
        "sigma_W": (float(wind["sigma"]["min"]), float(wind["sigma"]["max"])),
        # Held fixed (not swept): rho and peak_W.
        "rho_D": float(demand["rho_fixed"]),
        "rho_W": float(wind["rho_fixed"]),
        "peak_W": float(wind["tau_fixed"]),
    }


# Parameters swept one-at-a-time, paired with a short filesystem-clean tag.
_SWEPT = (
    ("mu_D", "muD"),
    ("sigma_D", "sigD"),
    ("mu_W", "muW"),
    ("sigma_W", "sigW"),
)


@dataclass(frozen=True)
class RegimeSpec:
    """One DRO center-regime: a name plus the full 7-field parameter dict."""

    name: str
    params: dict[str, float]


def build_specs() -> list[RegimeSpec]:
    """Mean regime at the box midpoint, then each parameter to its box bounds."""
    box = _load_box()
    mean = {p: 0.5 * (box[p][0] + box[p][1]) for p, _ in _SWEPT}
    fixed = {"rho_D": box["rho_D"], "rho_W": box["rho_W"], "peak_W": box["peak_W"]}

    def regime(overrides: dict[str, float]) -> dict[str, float]:
        return {**mean, **overrides, **fixed}

    specs = [RegimeSpec("mean", regime({}))]
    for param, tag in _SWEPT:
        lo, hi = box[param]
        specs.append(RegimeSpec(f"{tag}_lo", regime({param: lo})))
        specs.append(RegimeSpec(f"{tag}_hi", regime({param: hi})))
    return specs


def write_box_regime_runtime_config(config: ProjectConfig, spec: RegimeSpec) -> Path:
    """Write a single-regime runtime config centering the DRO on ``spec``.

    Mirrors block3's ``write_poa_regime_runtime_config`` but takes the regime
    parameters from the ambiguity-box spec instead of the PoA-optimized state.
    The regime is written under ``config.poa_regime_set`` and named
    ``config.poa_worst_case_regime_name`` so block4's ``build_dro_config`` picks
    it up unchanged.
    """
    runtime_config = {
        "regime_sets": {
            config.poa_regime_set: {
                "description": (
                    f"DRO box regime '{spec.name}' from ambiguity set "
                    f"'{AMBIGUITY_SET_NAME}' (mean/outer one-at-a-time sweep)."
                ),
                "seed": config.poa_seed,
                "enforce_dispatch_feasibility": True,
                "n_minus_one_margin": 0.0,
                "max_draw_attempts": 500,
                "regimes": [
                    {
                        "name": config.poa_worst_case_regime_name,
                        "n_scenarios": N_EMPIRICAL_SCENARIOS,
                        **spec.params,
                    }
                ],
            }
        }
    }
    output_path = write_json(Path(config.runtime_config_path), runtime_config)
    print(f"[{STUDY_NAME}] wrote DRO box regime '{spec.name}' -> {output_path}")
    return output_path


def build_run_config(spec: RegimeSpec) -> ProjectConfig:
    """Isolated config for one DRO-on-<regime> run, reusing base-case policies."""
    cfg = load_base_config(
        {
            # DRO leg only: reuse the existing policies/features/tightening inputs.
            "run_scenario_generation": False,
            "run_heuristic_labels": False,
            "run_feature_building": False,
            "run_nn_training": False,
            "run_dro_tightening": True,
            "run_dro_optimization": True,
            "plot_results_along_the_way": True,
        }
    )
    isolate_run_outputs(cfg, STUDY_NAME, spec.name, RESULT_ROOT)

    # Reuse the source run's trained policies, features and normalization stats
    # (read-only); the DRO writes only into this run's isolated directories.
    cfg.model_dir = SOURCE_RUN_DIR / "trained_models"
    cfg.raw_feature_dir = SOURCE_RUN_DIR / "features" / "raw"
    cfg.normalized_feature_dir = SOURCE_RUN_DIR / "features" / "normalized"

    # Center the DRO on this regime: build_dro_config sets
    # dro_regime_names=[poa_worst_case_regime_name] and reads poa_regime_set.
    cfg.poa_regime_set = RUNTIME_REGIME_SET
    cfg.poa_worst_case_n_scenarios = N_EMPIRICAL_SCENARIOS
    return cfg


def run() -> dict[str, Any]:
    sep = "=" * 72
    specs = build_specs()
    print(f"\n{sep}\n  Sensitivity study: {STUDY_NAME}")
    print(f"  regimes ({len(specs)}): {', '.join(s.name for s in specs)}")
    print(f"  reusing policies from: {SOURCE_RUN_DIR}\n{sep}")

    if not (SOURCE_RUN_DIR / "trained_models").exists():
        raise FileNotFoundError(
            f"No trained policies under {SOURCE_RUN_DIR / 'trained_models'}. "
            "Generate the base case first (driver/run_full_pipeline.py)."
        )

    manifests: dict[str, Any] = {}
    for spec in specs:
        print(f"\n{sep}\n  DRO centered on regime '{spec.name}': {spec.params}\n{sep}")
        cfg = build_run_config(spec)
        write_box_regime_runtime_config(cfg, spec)
        manifests[spec.name] = block4_dro_poa_pipeline.run(cfg)

    print(f"\n{sep}\n  {STUDY_NAME} complete\n{sep}")
    return manifests


if __name__ == "__main__":
    run()
