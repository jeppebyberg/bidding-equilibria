from pathlib import Path
import time

from models.PoA.PoA_tightening.compute_binary_fix_and_big_m_bidding_blocks import (
    build_optimizer,
)


def main() -> None:
    alpha_bounds_path = Path("results/poa_bidding_blocks_alpha_bounds.json")
    output_path = Path("results/poa_bidding_blocks_slack_binary_fix.json")
    epsilon = 1e-6
    solver_name = "gurobi"
    time_limit = 200

    optimizer = build_optimizer()
    optimizer.load_tightening_report(alpha_bounds_path)

    start = time.perf_counter()
    slack_report = optimizer.run_slack_based_obbt(
        alpha_bounds=optimizer.alpha_bounds,
        epsilon=epsilon,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=False,
    )
    report_path = optimizer.save_tightening_report(output_path)
    elapsed = time.perf_counter() - start

    print("\nSlack minimization and binary fixing complete")
    print(f"  Alpha bounds: {alpha_bounds_path}")
    print(f"  Report: {report_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Fixed complementarity binaries: {slack_report['num_fixed_binaries']}")

    print("\nFixed binaries by component")
    for component_name, entries in slack_report["fixed_binaries"].items():
        print(f"  {component_name}: {len(entries)}")


if __name__ == "__main__":
    main()
