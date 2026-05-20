from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.PoA.PoA_tightening.compute_binary_fix_and_big_m_bidding_blocks import (
    build_optimizer,
)


def main() -> None:
    slack_report_path = Path("results/poa_bidding_blocks_slack_binary_fix.json")
    output_path = Path("results/poa_bidding_blocks_tightening_report.json")
    solver_name = "gurobi"
    time_limit = 200

    optimizer = build_optimizer()
    optimizer.load_tightening_report(slack_report_path)

    start = time.perf_counter()
    big_m_report = optimizer.run_dual_big_m_tightening(
        alpha_bounds=optimizer.alpha_bounds,
        fixed_binaries=optimizer.fixed_binaries,
        solver_name=solver_name,
        time_limit=time_limit,
        tee=False,
    )
    report_path = optimizer.save_tightening_report(output_path)
    elapsed = time.perf_counter() - start

    tight_big_m = big_m_report["tight_big_m"]
    tightened_values = [
        value["tight_big_m"]
        for component in tight_big_m.values()
        for value in component.values()
        if value["tight_big_m"] is not None
    ]

    print("\nDual Big-M tightening complete")
    print(f"  Slack/binary report: {slack_report_path}")
    print(f"  Final report: {report_path}")
    print(f"  Runtime: {elapsed:.2f} seconds")
    print(f"  Tightened dual Big-M values: {len(tightened_values)}")
    if tightened_values:
        print(f"  Smallest tight Big-M: {min(tightened_values):.6g}")
        print(f"  Largest tight Big-M: {max(tightened_values):.6g}")


if __name__ == "__main__":
    main()
