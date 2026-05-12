from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.scenarios.scenario_generator import ScenarioManager
from xXgraveyard.models.gradient_based.gradient_bid_training import GradientBidTrainingKKTMS


def as_profile(value: Any, expected_len: Optional[int] = None, column_name: str = "profile") -> List[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Column '{column_name}' must contain a list-like profile")
    profile = [float(v) for v in value]
    if expected_len is not None and len(profile) != expected_len:
        raise ValueError(
            f"Profile length mismatch for '{column_name}': expected {expected_len}, got {len(profile)}"
        )
    return profile


def load_results(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def select_result_row(
    results: Dict[str, Any],
    scenario_idx: int,
    role: str,
    iteration: Optional[int],
) -> Dict[str, Any]:
    histories = results.get("scenario_histories", {})
    key = str(int(scenario_idx))
    if key not in histories and int(scenario_idx) in histories:
        key = int(scenario_idx)
    if key not in histories:
        available = sorted(str(k) for k in histories)
        raise ValueError(f"Results have no scenario history {scenario_idx}. Available: {available}")

    rows = [
        row for row in histories[key]
        if str(row.get("history_role")) == str(role)
    ]
    if iteration is not None:
        rows = [
            row for row in rows
            if int(row.get("training_iteration")) == int(iteration)
        ]
    if not rows:
        raise ValueError(
            f"No result rows matched scenario={scenario_idx}, role={role}, iteration={iteration}"
        )
    return sorted(rows, key=lambda row: int(row.get("training_iteration", 0)))[-1]


def build_single_scenario_df(
    scenario_manager: ScenarioManager,
    regime_set: str,
    seed: int,
    scenario_idx: int,
    result_row: Optional[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_set = scenario_manager.create_scenario_set_from_regimes(
        regime_set=regime_set,
        seed=seed,
    )
    scenarios_df = scenario_set["scenarios_df"].copy(deep=True).reset_index(drop=True)
    if scenario_idx < 0 or scenario_idx >= len(scenarios_df):
        raise ValueError(f"scenario_idx must be in [0, {len(scenarios_df) - 1}], got {scenario_idx}")

    row = scenarios_df.iloc[[scenario_idx]].copy(deep=True).reset_index(drop=True)
    if result_row is not None:
        for column, value in result_row.items():
            if column in row.columns or column.endswith("_bid") or column.endswith("_bid_profile"):
                row.at[0, column] = value

    return row, scenario_set["costs_df"], scenario_set["ramps_df"]


def make_trainer(
    scenarios_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    ramps_df: pd.DataFrame,
    players_config: List[Dict[str, Any]],
    beta_smooth: float,
    kkt_regularization: float,
    alpha_min: Optional[float],
    alpha_max: Optional[float],
) -> GradientBidTrainingKKTMS:
    return GradientBidTrainingKKTMS(
        scenarios_df=scenarios_df.copy(deep=True).reset_index(drop=True),
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        beta_smooth=beta_smooth,
        learning_rate=1.0,
        learning_rate_decay=0.0,
        min_learning_rate=0.0,
        max_iterations=0,
        conv_tolerance=0.0,
        gradient_clip_norm=None,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        kkt_regularization=kkt_regularization,
    )


def player_profit_for_df(
    trainer: GradientBidTrainingKKTMS,
    scenarios_df: pd.DataFrame,
    player_id: int,
) -> float:
    original_df = trainer.scenarios_df
    trainer.scenarios_df = scenarios_df.copy(deep=True).reset_index(drop=True)
    try:
        ed = trainer._solve_training_ed_model()
        dispatches = ed.get_dispatches()
        block_dispatches = ed.get_block_dispatches()
        prices = ed.get_clearing_prices()
        if dispatches is None or block_dispatches is None or prices is None:
            raise RuntimeError("ED solve did not return complete results")
        profit, _ = trainer.compute_player_profit(
            player_id,
            dispatches,
            prices,
            block_dispatches=block_dispatches,
        )
        return float(profit)
    finally:
        trainer.scenarios_df = original_df


def perturb_bid(
    scenarios_df: pd.DataFrame,
    block_name: str,
    time_step: int,
    delta: float,
) -> pd.DataFrame:
    perturbed = scenarios_df.copy(deep=True).reset_index(drop=True)
    profile_col = f"{block_name}_bid_profile"
    profile = as_profile(perturbed.at[0, profile_col], column_name=profile_col)
    profile[int(time_step)] = float(profile[int(time_step)] + delta)
    perturbed.at[0, profile_col] = profile
    perturbed.at[0, f"{block_name}_bid"] = float(profile[0])
    return perturbed


def parse_alpha_bound(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate analytical bid gradient against finite-difference ED profit changes."
    )
    parser.add_argument("--case", default="test_case_bidding_blocks")
    parser.add_argument("--regime-set", default="policy_training")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--results", type=Path, default=Path("results/gradient_bid_training_expanded_results.json"))
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--result-role", default="update_ready")
    parser.add_argument(
        "--result-iteration",
        type=int,
        default=None,
        help="Expanded result iteration. Omit for latest matching row.",
    )
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--block", required=True, help="Controlled block name or global block index")
    parser.add_argument("--time-step", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--beta-smooth", type=float, default=None)
    parser.add_argument("--kkt-regularization", type=float, default=None)
    parser.add_argument("--alpha-min", default="None")
    parser.add_argument("--alpha-max", default="None")
    parser.add_argument(
        "--ignore-results",
        action="store_true",
        help="Use regenerated scenario initial bids instead of a result scenario_history row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")

    scenario_manager = ScenarioManager(args.case)
    players_config = scenario_manager.get_players_config()
    results = None if args.ignore_results else load_results(args.results)
    result_row = None
    if results is not None:
        result_row = select_result_row(
            results,
            scenario_idx=args.scenario_index,
            role=args.result_role,
            iteration=args.result_iteration,
        )

    scenarios_df, costs_df, ramps_df = build_single_scenario_df(
        scenario_manager=scenario_manager,
        regime_set=args.regime_set,
        seed=args.seed,
        scenario_idx=args.scenario_index,
        result_row=result_row,
    )

    result_params = (results or {}).get("parameters", {})
    beta_smooth = float(
        args.beta_smooth
        if args.beta_smooth is not None
        else (results or {}).get("beta_smooth", result_params.get("beta_smooth", 0.001))
    )
    kkt_regularization = float(
        args.kkt_regularization
        if args.kkt_regularization is not None
        else result_params.get("kkt_regularization", 1e-8)
    )
    alpha_min = parse_alpha_bound(args.alpha_min)
    alpha_max = parse_alpha_bound(args.alpha_max)

    trainer = make_trainer(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        beta_smooth=beta_smooth,
        kkt_regularization=kkt_regularization,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    if str(args.block).lstrip("-").isdigit():
        block_idx = int(args.block)
        if block_idx < 0 or block_idx >= len(trainer.block_names):
            raise ValueError(f"block index must be in [0, {len(trainer.block_names) - 1}], got {block_idx}")
        block_name = trainer.block_names[block_idx]
    else:
        block_name = str(args.block)
        if block_name not in trainer.block_names:
            raise ValueError(f"Unknown block '{block_name}'. Available: {trainer.block_names}")
        block_idx = trainer.block_names.index(block_name)

    controlled = trainer._controlled_blocks(args.player_id)
    if block_idx not in controlled:
        controlled_names = [trainer.block_names[idx] for idx in controlled]
        raise ValueError(
            f"Player {args.player_id} does not control block {block_name}. "
            f"Controlled blocks: {controlled_names}"
        )
    if args.time_step < 0 or args.time_step >= trainer.num_time_steps:
        raise ValueError(f"time_step must be in [0, {trainer.num_time_steps - 1}]")

    gradients, baseline_profit, scenario_profits, diagnostics = trainer.compute_player_bid_gradients(args.player_id)
    gradient = np.asarray(gradients[0], dtype=np.float64)
    local_block_idx = controlled.index(block_idx)
    analytic_index = int(args.time_step) * len(controlled) + local_block_idx
    analytic = float(gradient[analytic_index])

    plus_df = perturb_bid(scenarios_df, block_name, args.time_step, args.epsilon)
    minus_df = perturb_bid(scenarios_df, block_name, args.time_step, -args.epsilon)
    plus_profit = player_profit_for_df(trainer, plus_df, args.player_id)
    minus_profit = player_profit_for_df(trainer, minus_df, args.player_id)
    base_profit = player_profit_for_df(trainer, scenarios_df, args.player_id)
    forward_fd = (plus_profit - base_profit) / args.epsilon
    backward_fd = (base_profit - minus_profit) / args.epsilon
    central_fd = (plus_profit - minus_profit) / (2.0 * args.epsilon)

    profile = as_profile(scenarios_df.at[0, f"{block_name}_bid_profile"], trainer.num_time_steps, f"{block_name}_bid_profile")
    print("=== Bid Gradient Finite-Difference Validation ===")
    print(f"scenario_index       : {args.scenario_index}")
    print(f"result_role          : {args.result_role if result_row is not None else 'generated_initial'}")
    print(f"result_iteration     : {result_row.get('training_iteration') if result_row is not None else 'n/a'}")
    print(f"player_id            : {args.player_id}")
    print(f"block                : {block_name} (global index {block_idx})")
    print(f"time_step            : {args.time_step}")
    print(f"bid value            : {profile[args.time_step]:.12g}")
    print(f"epsilon              : {args.epsilon:.12g}")
    print(f"beta_smooth          : {beta_smooth:.12g}")
    print(f"kkt_regularization   : {kkt_regularization:.12g}")
    print(f"baseline_profit      : {base_profit:.12g}")
    print(f"analytic_gradient    : {analytic:.12g}")
    print(f"finite_diff_forward  : {forward_fd:.12g}")
    print(f"finite_diff_backward : {backward_fd:.12g}")
    print(f"finite_diff_central  : {central_fd:.12g}")
    print(f"plus_profit          : {plus_profit:.12g}")
    print(f"minus_profit         : {minus_profit:.12g}")
    print(f"gradient_norm        : {float(np.linalg.norm(gradient)):.12g}")
    print(f"condition_max        : {diagnostics['max_condition_number']:.12g}")

    if np.sign(analytic) != np.sign(central_fd) and abs(analytic) > 1e-9 and abs(central_fd) > 1e-9:
        print("WARNING: analytic and central finite-difference gradients have opposite signs.")


if __name__ == "__main__":
    main()
