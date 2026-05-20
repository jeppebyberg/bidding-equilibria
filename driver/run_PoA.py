from dataclasses import dataclass
from pathlib import Path
import json
import time

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.PoA_tightening.bidding_blocks_tightening import (
    BiddingBlocksTighteningOptimizer,
)


@dataclass(frozen=True)
class PoARunConfig:
    case: str = "test_case_bidding_blocks"
    regime_set: str = "PoA_analysis"
    seed: int = 1
    horizon: int = 4
    support_set_config_path: str = "models/PoA/support_set_config.yaml"
    support_set_config_name: str = "test_case_bidding_blocks_base"
    nn_model_dir: str = "models/neural_network/training/trained_models"
    nn_normalization_stats_path: str = (
        "models/neural_network/features/generated/normalized/min_max_stats.json"
    )
    nn_policy_generators: tuple[int, ...] = (1, 2)
    solver_name: str = "gurobi"
    preprocessing_time_limit: int = 200
    poa_time_limit: int = 400
    epsilon: float = 1e-6
    # Parallelizes independent tightening submodels; use low Gurobi threads per worker.
    poa_parallel_workers: int = 1
    poa_solver_threads_per_worker: int | None = None

    @property
    def alpha_bounds_path(self) -> Path:
        return Path(f"results/poa_bidding_blocks_alpha_bounds_T{self.horizon}.json")

    @property
    def slack_report_path(self) -> Path:
        return Path(f"results/poa_bidding_blocks_slack_binary_fix_T{self.horizon}.json")

    @property
    def tightening_report_path(self) -> Path:
        return Path(f"results/poa_bidding_blocks_tightening_report_T{self.horizon}.json")

    @property
    def poa_results_path(self) -> Path:
        return Path(
            f"results/poa_optimization_bidding_blocks_results_tightened_T{self.horizon}.json"
        )


CONFIG = PoARunConfig(
    case="test_case_bidding_blocks",
    regime_set="PoA_analysis",
    seed=1,
    horizon=8,
)


RUN_ALPHA_BOUNDS = True
RUN_SLACK_BINARY_FIX = True
RUN_DUAL_BIG_M = True
RUN_FINAL_POA = True


def load_scenario_data(config: PoARunConfig) -> dict:
    scenario_manager = ScenarioManager(config.case)
    return scenario_manager.create_scenario_set_from_regimes(
        regime_set=config.regime_set,
        seed=config.seed,
    )


def load_support_set_config(config: PoARunConfig) -> dict:
    return PoAOptimization.load_support_set_config(
        config_path=config.support_set_config_path,
        config_name=config.support_set_config_name,
    )


def build_optimizer(
    config: PoARunConfig,
    optimizer_cls: type[PoAOptimization] = PoAOptimization,
) -> PoAOptimization:
    scenarios = load_scenario_data(config)
    support_set_config = load_support_set_config(config)
    return optimizer_cls(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=config.horizon,
        support_set_config=support_set_config,
        nn_model_dir=config.nn_model_dir,
        nn_normalization_stats_path=config.nn_normalization_stats_path,
        nn_policy_generators=list(config.nn_policy_generators),
        reference_case=config.case,
    )


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return path


def run_alpha_bounds(config: PoARunConfig) -> Path:
    optimizer = build_optimizer(config, BiddingBlocksTighteningOptimizer)
    start = time.perf_counter()
    alpha_report = optimizer.compute_nn_certified_bid_bounds(
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    elapsed = time.perf_counter() - start

    payload = {
        "metadata": {
            "description": (
                "Exact alpha bounds from support-set optimization with embedded "
                "ReLU policy constraints."
            ),
            "reference_case": optimizer.reference_case,
            "num_time_steps": optimizer.num_time_steps,
            "nn_policy_generators": list(optimizer.nn_policy_generator_names),
            "num_optimization_programs": alpha_report["num_optimization_programs"],
        },
        "alpha_bounds": alpha_report["alpha_bounds"],
        "alpha_optimization_results": alpha_report["optimization_results"],
        "fixed_binaries": {},
        "slack_bounds": {},
        "lambda_bounds": {},
        "tight_big_m": {},
        "aggregate_dual_bounds": {},
    }
    output_path = write_json(config.alpha_bounds_path, payload)

    alpha_values = list(alpha_report["alpha_bounds"].values())
    lower_values = [float(item["lower"]) for item in alpha_values]
    upper_values = [float(item["upper"]) for item in alpha_values]
    print("\nAlpha-bound computation complete")
    print(f"  Report: {output_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Alpha entries: {len(alpha_values)}")
    print(f"  Min alpha lower bound: {min(lower_values):.6g}")
    print(f"  Max alpha upper bound: {max(upper_values):.6g}")
    return output_path


def run_slack_binary_fix(config: PoARunConfig) -> Path:
    optimizer = build_optimizer(config, BiddingBlocksTighteningOptimizer)
    optimizer.load_tightening_report(config.alpha_bounds_path)

    start = time.perf_counter()
    slack_report = optimizer.run_slack_based_obbt(
        alpha_bounds=optimizer.alpha_bounds,
        epsilon=config.epsilon,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    output_path = optimizer.save_tightening_report(config.slack_report_path)
    elapsed = time.perf_counter() - start

    print("\nSlack minimization and binary fixing complete")
    print(f"  Alpha bounds: {config.alpha_bounds_path}")
    print(f"  Report: {output_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Fixed complementarity binaries: {slack_report['num_fixed_binaries']}")
    return output_path


def run_dual_big_m(config: PoARunConfig) -> Path:
    optimizer = build_optimizer(config, BiddingBlocksTighteningOptimizer)
    optimizer.load_tightening_report(config.slack_report_path)

    start = time.perf_counter()
    big_m_report = optimizer.run_dual_big_m_tightening(
        alpha_bounds=optimizer.alpha_bounds,
        fixed_binaries=optimizer.fixed_binaries,
        solver_name=config.solver_name,
        time_limit=config.preprocessing_time_limit,
        tee=False,
        parallel_workers=config.poa_parallel_workers,
        solver_threads=config.poa_solver_threads_per_worker,
    )
    output_path = optimizer.save_tightening_report(config.tightening_report_path)
    elapsed = time.perf_counter() - start

    tightened_values = [
        value["tight_big_m"]
        for component in big_m_report["tight_big_m"].values()
        for value in component.values()
        if value["tight_big_m"] is not None
    ]
    print("\nDual Big-M tightening complete")
    print(f"  Slack/binary report: {config.slack_report_path}")
    print(f"  Final report: {output_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Tightened dual Big-M values: {len(tightened_values)}")
    if tightened_values:
        print(f"  Smallest tight Big-M: {min(tightened_values):.6g}")
        print(f"  Largest tight Big-M: {max(tightened_values):.6g}")
    return output_path


def run_final_poa(config: PoARunConfig) -> Path:
    optimizer = build_optimizer(config, PoAOptimization)
    start = time.perf_counter()
    optimizer.load_tightening_report(config.tightening_report_path)
    optimizer.build_model()
    applied_stats = optimizer.apply_tightened_bounds_to_model()
    optimizer.solve(time_limit=config.poa_time_limit)
    output_path = optimizer.save_results(config.poa_results_path)
    elapsed = time.perf_counter() - start

    print("\nPoA solve with precomputed tightening complete")
    print(f"  Tightening report: {config.tightening_report_path}")
    print(f"  Applied fixed binaries: {applied_stats['fixed_binaries']}")
    print(f"  Applied lambda bounds: {applied_stats['lambda_bounds']}")
    print(f"  Applied dual upper bounds: {applied_stats['dual_upper_bounds']}")
    print(f"  Applied aggregate dual bounds: {applied_stats['aggregate_dual_bounds']}")
    print(f"  Applied alpha bounds: {applied_stats['alpha_bounds']}")
    print(f"  Results: {output_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    return output_path


def main() -> None:
    config = CONFIG
    print(
        "\nPoA run configuration\n"
        f"  case={config.case}\n"
        f"  regime_set={config.regime_set}\n"
        f"  seed={config.seed}\n"
        f"  horizon={config.horizon}\n"
        f"  poa_parallel_workers={config.poa_parallel_workers}"
    )

    if RUN_ALPHA_BOUNDS:
        run_alpha_bounds(config)
    if RUN_SLACK_BINARY_FIX:
        run_slack_binary_fix(config)
    if RUN_DUAL_BIG_M:
        run_dual_big_m(config)
    if RUN_FINAL_POA:
        run_final_poa(config)


if __name__ == "__main__":
    main()
