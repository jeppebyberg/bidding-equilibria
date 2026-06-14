from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from config.scenarios.scenario_generator import ScenarioManager
from driver.core.block0_core import ProjectConfig
from driver.core.block1_core import apply_time_steps_override, write_json
from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.poa_model.support_set import AR1_JOINT_COVERAGE, _ar1_kappa, _level_scale


_REGIME_PARAM_COLUMNS = ("mu_D", "rho_D", "sigma_D", "mu_W", "rho_W", "sigma_W", "peak_W")


def _parse_profile(value: Any, horizon: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, list):
        values = value
    else:
        values = ast.literal_eval(str(value))
    return np.asarray([float(item) for item in values], dtype=float)[:horizon]


def _load_runtime_regime_set(config: ProjectConfig) -> dict[str, Any]:
    runtime_config_path = Path(config.runtime_config_path)
    if not runtime_config_path.exists():
        raise FileNotFoundError(
            "PoA regime runtime config is missing. Run block3_poa_pipeline.py "
            f"first, or point runtime_config_path at an existing file: {runtime_config_path}"
        )
    with runtime_config_path.open("r", encoding="utf-8") as file_handle:
        raw_config = yaml.safe_load(file_handle) or {}
    regime_sets = raw_config.get("regime_sets", {}) or {}
    regime_set = regime_sets.get(config.poa_regime_set, {}) or {}
    regimes = regime_set.get("regimes", []) or []
    if not regimes:
        raise ValueError(
            f"No regimes found for '{config.poa_regime_set}' in {runtime_config_path}"
        )
    return regime_set


def resolve_poa_runtime_regime_names(config: ProjectConfig) -> list[str]:
    regime_set = _load_runtime_regime_set(config)
    return [
        str(regime["name"])
        for regime in regime_set.get("regimes", [])
        if regime.get("name") is not None
    ]


def _runtime_regime_config_for_diagnostics(config: ProjectConfig) -> dict[str, Any]:
    regime_set = _load_runtime_regime_set(config)
    regimes: list[dict[str, Any]] = []
    for regime in regime_set.get("regimes", []):
        regime_copy = dict(regime)
        regime_copy["n_scenarios"] = int(config.support_verify_num_draws)
        regimes.append(regime_copy)
    return {
        "regime_sets": {
            config.poa_regime_set: {
                **{k: v for k, v in regime_set.items() if k != "regimes"},
                "regimes": regimes,
                "seed": int(config.support_verify_seed),
            }
        }
    }


def draw_poa_regime_oos_samples(config: ProjectConfig) -> dict[str, Any]:
    """Draw fresh samples from the optimized PoA runtime regime.

    These samples are not filtered by the support set; block 3.5 measures how
    often the raw stochastic regime lands inside the base PoA support bands.
    """
    manager = ScenarioManager(config.case)
    apply_time_steps_override(manager, config.horizon)

    diagnostic_config = _runtime_regime_config_for_diagnostics(config)
    draw_config_path = (
        Path(config.support_calibration_report_path).parent
        / "support_oos_regime_draw_config.yaml"
    )
    draw_config_path.parent.mkdir(parents=True, exist_ok=True)
    with draw_config_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(diagnostic_config, file_handle, sort_keys=False)

    scenarios = manager.create_scenario_set_from_regimes(
        regime_config_path=str(draw_config_path),
        regime_set=config.poa_regime_set,
        seed=config.support_verify_seed,
        enforce_support_set=False,
        enforce_n_minus_one=False,
    )
    return {**scenarios, "draw_config_path": draw_config_path}


def _load_poa_support_parameters(config: ProjectConfig) -> dict[str, Any]:
    support_config = PoAOptimization.load_ambiguity_set(
        config.ambiguity_set_config_path,
        config.ambiguity_set_config_name,
    )
    return {
        "ambiguity_kappa": float(support_config["kappa"]),
        "rho_D": float(support_config["demand"]["rho_fixed"]),
        "rho_W": float(support_config["wind"]["rho_fixed"]),
        "tau_W": float(support_config["wind"]["tau_fixed"]),
        "joint_coverage": float(AR1_JOINT_COVERAGE),
    }


def _add_violation(
    violations: list[dict[str, Any]],
    scenario_id: Any,
    regime_name: str,
    family: str,
    max_violation: float,
) -> None:
    if max_violation > 1e-8:
        violations.append(
            {
                "scenario_id": int(scenario_id),
                "regime": str(regime_name),
                "family": family,
                "max_violation": float(max_violation),
            }
        )


def analyze_poa_support_band_coverage(
    config: ProjectConfig,
    scenarios: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manager = ScenarioManager(config.case)
    apply_time_steps_override(manager, config.horizon)
    scenarios = scenarios or draw_poa_regime_oos_samples(config)
    scenarios_df = scenarios["scenarios_df"].copy().reset_index(drop=True)

    support = _load_poa_support_parameters(config)
    horizon = int(config.horizon)
    kappa = _ar1_kappa(horizon, support["joint_coverage"])
    demand_shape = ScenarioManager._build_demand_shape(horizon)
    wind_shape = ScenarioManager._build_wind_shape(horizon, support["tau_W"])
    demand_scales = np.asarray(
        [_level_scale(support["rho_D"], t) for t in range(horizon)],
        dtype=float,
    )
    wind_scales = np.asarray(
        [_level_scale(support["rho_W"], t) for t in range(horizon)],
        dtype=float,
    )

    demand_ref_scalar = float(manager.base_case["demand"])
    wind_generators = [g for g in manager.physical_generators if bool(g["is_wind"])]
    wind_blocks_by_generator = {
        str(g["physical_name"]): [
            block
            for block in manager.blocks
            if block["physical_name"] == g["physical_name"] and bool(block["is_wind"])
        ]
        for g in wind_generators
    }

    missing_columns = [col for col in _REGIME_PARAM_COLUMNS if col not in scenarios_df.columns]
    if missing_columns:
        raise ValueError(
            "PoA support OOS scenarios are missing regime parameter columns: "
            f"{missing_columns}"
        )

    violations: list[dict[str, Any]] = []
    parameter_warnings: list[str] = []
    for _, row in scenarios_df.iterrows():
        scenario_id = row.get("scenario_id", int(row.name) + 1)
        regime_name = str(row.get("regime", ""))
        mu_D = float(row["mu_D"])
        sigma_D = float(row["sigma_D"])
        mu_W = float(row["mu_W"])
        sigma_W = float(row["sigma_W"])

        row_rho_D = float(row["rho_D"])
        row_rho_W = float(row["rho_W"])
        row_peak_W = float(row["peak_W"])
        if abs(row_rho_D - support["rho_D"]) > 1e-9:
            parameter_warnings.append(
                f"{regime_name}: sampled rho_D={row_rho_D} differs from PoA support rho_D={support['rho_D']}"
            )
        if abs(row_rho_W - support["rho_W"]) > 1e-9:
            parameter_warnings.append(
                f"{regime_name}: sampled rho_W={row_rho_W} differs from PoA support rho_W={support['rho_W']}"
            )
        if abs(row_peak_W - support["tau_W"]) > 1e-9:
            parameter_warnings.append(
                f"{regime_name}: sampled peak_W={row_peak_W} differs from PoA support tau_W={support['tau_W']}"
            )

        demand = _parse_profile(row["demand_profile"], horizon)
        demand_ref = demand_ref_scalar * mu_D * demand_shape
        demand_level_half_width = kappa * demand_ref_scalar * sigma_D * demand_scales
        demand_innov_half_width = kappa * demand_ref_scalar * sigma_D
        demand_budget = support["ambiguity_kappa"] * float(np.sum(demand_ref))

        _add_violation(
            violations,
            scenario_id,
            regime_name,
            "demand_level_band",
            float(np.max(np.abs(demand - demand_ref) - demand_level_half_width)),
        )
        demand_t0 = abs(demand[0] - demand_ref[0]) - demand_innov_half_width
        _add_violation(
            violations,
            scenario_id,
            regime_name,
            "demand_ar1_t0_band",
            float(demand_t0),
        )
        if horizon > 1:
            ar1_ref = demand_ref_scalar * mu_D * (
                demand_shape[1:] - support["rho_D"] * demand_shape[:-1]
            )
            demand_innov = demand[1:] - support["rho_D"] * demand[:-1] - ar1_ref
            _add_violation(
                violations,
                scenario_id,
                regime_name,
                "demand_ar1_innovation_band",
                float(np.max(np.abs(demand_innov) - demand_innov_half_width)),
            )
        _add_violation(
            violations,
            scenario_id,
            regime_name,
            "demand_deviation_budget",
            float(np.sum(np.abs(demand - demand_ref)) - demand_budget),
        )

        for generator in wind_generators:
            generator_name = str(generator["physical_name"])
            capacity = float(generator["pmax"])
            wind = np.zeros(horizon, dtype=float)
            for block in wind_blocks_by_generator[generator_name]:
                column = f"{block['block_name']}_profile"
                if column in scenarios_df.columns:
                    wind += _parse_profile(row[column], horizon)

            wind_ref = capacity * mu_W * wind_shape
            wind_level_half_width = kappa * capacity * sigma_W * wind_scales
            wind_innov_half_width = kappa * capacity * sigma_W
            wind_budget = support["ambiguity_kappa"] * float(np.sum(wind_ref))

            _add_violation(
                violations,
                scenario_id,
                regime_name,
                f"wind_{generator_name}_level_band",
                float(np.max(np.abs(wind - wind_ref) - wind_level_half_width)),
            )
            _add_violation(
                violations,
                scenario_id,
                regime_name,
                f"wind_{generator_name}_physical_cap",
                float(np.max(wind - capacity)),
            )
            _add_violation(
                violations,
                scenario_id,
                regime_name,
                f"wind_{generator_name}_ar1_t0_band",
                float(abs(wind[0] - wind_ref[0]) - wind_innov_half_width),
            )
            if horizon > 1:
                ar1_ref_w = capacity * mu_W * (
                    wind_shape[1:] - support["rho_W"] * wind_shape[:-1]
                )
                wind_innov = wind[1:] - support["rho_W"] * wind[:-1] - ar1_ref_w
                _add_violation(
                    violations,
                    scenario_id,
                    regime_name,
                    f"wind_{generator_name}_ar1_innovation_band",
                    float(np.max(np.abs(wind_innov) - wind_innov_half_width)),
                )
            _add_violation(
                violations,
                scenario_id,
                regime_name,
                f"wind_{generator_name}_deviation_budget",
                float(np.sum(np.abs(wind - wind_ref)) - wind_budget),
            )
            _add_violation(
                violations,
                scenario_id,
                regime_name,
                f"wind_{generator_name}_reference_feasibility",
                float(max(np.max(wind_ref - capacity), np.max(-wind_ref))),
            )

    violation_df = pd.DataFrame(violations)
    if violation_df.empty:
        scenario_violation_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        max_violation_by_family: dict[str, float] = {}
        violating_scenario_ids: list[int] = []
    else:
        scenario_violation_counts = {
            str(int(key)): int(value)
            for key, value in violation_df.groupby("scenario_id").size().to_dict().items()
        }
        family_counts = {
            str(key): int(value)
            for key, value in violation_df.groupby("family").size().to_dict().items()
        }
        max_violation_by_family = {
            str(key): float(value)
            for key, value in violation_df.groupby("family")["max_violation"].max().to_dict().items()
        }
        violating_scenario_ids = sorted(int(v) for v in violation_df["scenario_id"].unique())

    report = {
        "case": config.case,
        "regime_set": config.poa_regime_set,
        "runtime_config_path": str(config.runtime_config_path),
        "draw_config_path": str(scenarios.get("draw_config_path", "")),
        "support_model": "base_poa",
        "support_joint_coverage": float(support["joint_coverage"]),
        "support_kappa": float(kappa),
        "ambiguity_kappa": float(support["ambiguity_kappa"]),
        "horizon": horizon,
        "seed": int(config.support_verify_seed),
        "num_draws": int(len(scenarios_df)),
        "num_violating_scenarios": int(len(violating_scenario_ids)),
        "share_violating_scenarios": (
            float(len(violating_scenario_ids) / len(scenarios_df))
            if len(scenarios_df)
            else 0.0
        ),
        "violating_scenario_ids": violating_scenario_ids[:100],
        "scenario_violation_counts": scenario_violation_counts,
        "family_counts": family_counts,
        "max_violation_by_family": max_violation_by_family,
        "parameter_warnings": sorted(set(parameter_warnings)),
        "example_violations": violations[:50],
    }
    output_path = write_json(Path(config.support_calibration_report_path), report)
    print("\nPoA support-band OOS diagnostic")
    print(f"  runtime regime config: {config.runtime_config_path}")
    print(f"  draws: {len(scenarios_df)}  seed: {config.support_verify_seed}")
    print(f"  support coverage: {support['joint_coverage']}  kappa: {kappa:.6g}")
    print(
        "  violating scenarios: "
        f"{report['num_violating_scenarios']}/{report['num_draws']} "
        f"({100.0 * report['share_violating_scenarios']:.2f}%)"
    )
    if not family_counts:
        print("  all sampled scenarios were inside the base PoA support bands.")
    print(f"Saved PoA support-band OOS report: {output_path}")
    return report
