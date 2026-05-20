from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.PoA_tightening.nn_relu_bounds import NNReLUBoundsOptimizer


def build_optimizer() -> NNReLUBoundsOptimizer:
    case = "test_case_bidding_blocks"
    regime_set = "PoA_analysis"
    seed = 1
    horizon = 4

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    support_set_config = PoAOptimization.load_support_set_config(
        config_path="models/PoA/support_set_config.yaml",
        config_name="test_case_bidding_blocks_base",
    )

    return NNReLUBoundsOptimizer(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=horizon,
        support_set_config=support_set_config,
        nn_model_dir="models/neural_network/training/trained_models",
        nn_normalization_stats_path=(
            "models/neural_network/features/generated/normalized/min_max_stats.json"
        ),
        nn_policy_generators=[1, 2],
        reference_case=case,
    )


def main() -> None:
    output_path = Path("results/poa_nn_relu_bounds_report.json")
    optimizer = build_optimizer()

    start = time.perf_counter()
    report_path = optimizer.save_nn_relu_bounds_report(
        output_path=output_path,
        solver_name="gurobi",
        time_limit=None,
        tee=False,
    )
    elapsed = time.perf_counter() - start

    summary = optimizer.summarize_nn_relu_bounds()
    total_fixed = sum(
        int(details.get("num_active", 0)) + int(details.get("num_inactive", 0))
        for details in summary.values()
    )

    print("\nNN ReLU preactivation-bound report complete")
    print(f"  Report: {report_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  NN policy generators: {len(optimizer.nn_policy_generator_names)}")
    for generator_name in optimizer.nn_policy_generator_names:
        details = summary.get(generator_name, {})
        print(
            f"  {generator_name}: "
            f"active={int(details.get('num_active', 0))}, "
            f"inactive={int(details.get('num_inactive', 0))}, "
            f"ambiguous={int(details.get('num_ambiguous', 0))}"
        )
    print(f"  Total fixed ReLU binaries: {total_fixed}")


if __name__ == "__main__":
    main()
