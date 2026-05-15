import json
from pathlib import Path
import numpy as np

from config.scenarios.scenario_generator import ScenarioManager
from models.synthetic_data_generation.merit_order_best_response import MeritOrderHeuristic


def flatten(x):
    return np.asarray(x, dtype=float).reshape(-1)


def summarize_duals(duals, safety_factor=2.0):
    lam = flatten(duals["lambda"])
    mu_max = flatten(duals["mu_max"])
    mu_min = flatten(duals["mu_min"])
    mu_up = flatten(duals["mu_up"])
    mu_down = flatten(duals["mu_down"])

    capacity_duals = np.concatenate([mu_max, mu_min])
    ramp_duals = np.concatenate([mu_up, mu_down])

    return {
        "lambda": {
            "max_abs": float(np.max(np.abs(lam))),
            "q99_abs": float(np.quantile(np.abs(lam), 0.99)),
        },
        "capacity_duals": {
            "max_abs": float(np.max(np.abs(capacity_duals))),
            "q99_abs": float(np.quantile(np.abs(capacity_duals), 0.99)),
        },
        "ramp_duals": {
            "max_abs": float(np.max(np.abs(ramp_duals))),
            "q99_abs": float(np.quantile(np.abs(ramp_duals), 0.99)),
        },
        "recommended_bounds": {
            "lambda_bound_max": float(safety_factor * np.max(np.abs(lam))),
            "capacity_dual_bound_max": float(safety_factor * np.max(np.abs(capacity_duals))),
            "ramp_dual_bound_max": float(safety_factor * np.max(np.abs(ramp_duals))),
            "lambda_bound_q99": float(safety_factor * np.quantile(np.abs(lam), 0.99)),
            "capacity_dual_bound_q99": float(safety_factor * np.quantile(np.abs(capacity_duals), 0.99)),
            "ramp_dual_bound_q99": float(safety_factor * np.quantile(np.abs(ramp_duals), 0.99)),
        },
    }


def retrieve_duals_from_merit_order_results(
    results_path="results/merit_order_best_response_results.json",
    output_path="results/dual_bound_calibration_from_merit_order.json",
    case="test_case_bidding_blocks",
    regime_set="policy_training",
    seed=1,
    bid_tolerance=1e-2,
    safety_factor=2.0,
):
    results_path = Path(results_path)
    with results_path.open("r", encoding="utf-8") as f:
        merit_results = json.load(f)

    final_bids = merit_results["final_bids"]

    scenario_manager = ScenarioManager(case)
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )

    scenarios_df = scenarios["scenarios_df"]
    costs_df = scenarios["costs_df"]
    ramps_df = scenarios["ramps_df"]
    players_config = scenario_manager.get_players_config()

    # Instantiate heuristic only to reuse its mapping and P_init construction.
    heuristic = MeritOrderHeuristic(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        bid_tolerance=bid_tolerance,
    )

    # Re-clear ED with the final bids from the JSON.
    dispatches, block_dispatches, prices, ed = heuristic.solve_ed_for_bids(final_bids)

    duals = ed.get_dual_variables()
    if duals is None:
        raise RuntimeError("ED did not return dual variables.")

    dual_summary = summarize_duals(duals, safety_factor=safety_factor)

    payload = {
        "source_results_path": str(results_path),
        "config": {
            "case": case,
            "regime_set": regime_set,
            "seed": seed,
            "num_scenarios": len(final_bids),
            "num_time_steps": heuristic.num_time_steps,
            "num_blocks": heuristic.num_blocks,
            "num_physical_generators": len(heuristic.physical_generator_names),
            "safety_factor": safety_factor,
        },
        "block_names": heuristic.block_names,
        "physical_generator_names": heuristic.physical_generator_names,
        "dual_summary": dual_summary,
        "raw_duals": duals,
        "clearing_prices": prices,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved dual calibration to {output_path}")
    print(json.dumps(dual_summary["recommended_bounds"], indent=2))

    return payload


if __name__ == "__main__":
    retrieve_duals_from_merit_order_results()