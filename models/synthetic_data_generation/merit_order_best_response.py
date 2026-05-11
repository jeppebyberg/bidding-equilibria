from __future__ import annotations

import ast
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.scenarios.scenario_generator import ScenarioManager
from models.synthetic_data_generation.economic_dispatch import EconomicDispatchModel


@dataclass(frozen=True)
class MeritOrderEntry:
    block_id: int
    block_name: str
    generator_id: int
    generator_name: str
    player_id: Optional[int]
    bid: float
    true_cost: float
    available_capacity: float
    dispatch: Optional[float] = None
    cumulative_capacity: float = 0.0


class MeritOrderHeuristic:
    """
    Discrete merit-order best-response search for block bid profiles.

    The heuristic tests one local bid move of the form threshold_bid -
    bid_tolerance, where the threshold is the nearest valid bid in the current
    submitted merit order. A move is accepted only after re-solving ED and
    observing a player-profit increase.
    """

    MAX_ITERATIONS = 20
    PROFIT_TOLERANCE = 1e-4

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        bid_tolerance: float = 1e-6,
    ) -> None:
        if bid_tolerance <= 0:
            raise ValueError("bid_tolerance must be positive")

        self.scenarios_df = scenarios_df.copy(deep=True).reset_index(drop=True)
        self.costs_df = costs_df
        self.ramps_df = ramps_df
        self.players_config = players_config
        self.bid_tolerance = float(bid_tolerance)
        self.debug = False

        self._initialize_block_mapping_from_ed()
        self.num_scenarios = len(self.scenarios_df)
        self.num_time_steps = self._infer_num_time_steps()
        self.cost_vector = np.asarray(
            [float(self.costs_df[f"{block}_cost"].iloc[0]) for block in self.block_names],
            dtype=np.float64,
        )
        self._initialize_missing_bid_profiles()

        self.player_blocks = {
            int(player["id"]): self._controlled_blocks(int(player["id"]))
            for player in self.players_config
        }
        self.block_to_player = {
            int(block_idx): int(player_id)
            for player_id, blocks in self.player_blocks.items()
            for block_idx in blocks
        }
        self.P_init = self._compute_p_init_from_ed(self.scenarios_df)
        self.history: List[Dict[str, Any]] = []
        self.results: Optional[Dict[str, Any]] = None

        self._log_availability_interpretation()

    def _initialize_block_mapping_from_ed(self) -> None:
        mapping_model = EconomicDispatchModel(
            self.scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=None,
        )
        self.block_names = mapping_model.get_block_names()
        self.num_blocks = len(self.block_names)
        self.physical_generator_names = mapping_model.get_physical_generator_names()
        self.block_to_physical = mapping_model.get_block_to_physical_mapping()
        self.block_to_physical_idx = list(mapping_model.block_to_physical_idx)
        self.physical_to_block_indices = [
            list(blocks) for blocks in mapping_model.physical_to_block_indices
        ]

    def _infer_num_time_steps(self) -> int:
        if "time_steps" in self.scenarios_df.columns:
            return int(self.scenarios_df["time_steps"].iloc[0])
        return len(self._as_profile(self.scenarios_df["demand_profile"].iloc[0], "demand_profile"))

    @staticmethod
    def _as_profile(value: Any, column_name: str) -> List[float]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"Column '{column_name}' must contain a list/tuple profile")
        return [float(v) for v in value]

    def _initialize_missing_bid_profiles(self) -> None:
        for block_idx, block_name in enumerate(self.block_names):
            profile_col = f"{block_name}_bid_profile"
            bid_col = f"{block_name}_bid"
            if profile_col not in self.scenarios_df.columns:
                self.scenarios_df[profile_col] = [
                    [float(self.cost_vector[block_idx])] * self.num_time_steps
                    for _ in range(self.num_scenarios)
                ]
            for s in range(self.num_scenarios):
                profile = self._as_profile(self.scenarios_df.at[s, profile_col], profile_col)
                if len(profile) != self.num_time_steps:
                    raise ValueError(
                        f"{profile_col} must have length {self.num_time_steps}, got {len(profile)}"
                    )
                self.scenarios_df.at[s, profile_col] = [float(v) for v in profile]
                self.scenarios_df.at[s, bid_col] = float(profile[0])

    def _controlled_blocks(self, player_id: int) -> List[int]:
        player = self._get_player_config(player_id)
        if "controlled_blocks" in player:
            return [
                self._coerce_index_or_name(value, self.block_names, "bidding block")
                for value in player["controlled_blocks"]
            ]
        controlled_physical = [
            self._coerce_index_or_name(value, self.physical_generator_names, "physical generator")
            for value in player.get("controlled_generators", [])
        ]
        if not controlled_physical:
            raise ValueError(
                f"Player {player_id} must define either controlled_blocks or controlled_generators"
            )
        controlled_blocks: List[int] = []
        for physical_idx in controlled_physical:
            controlled_blocks.extend(self.physical_to_block_indices[int(physical_idx)])
        return controlled_blocks

    def _coerce_index_or_name(self, value: Any, names: List[str], label: str) -> int:
        if isinstance(value, str) and not value.strip().lstrip("-").isdigit():
            if value not in names:
                raise ValueError(f"Unknown {label} name '{value}'. Available: {names}")
            return names.index(value)
        idx = int(value)
        if idx < 0 or idx >= len(names):
            raise ValueError(f"{label} index {idx} is out of range [0, {len(names) - 1}]")
        return idx

    def _get_player_config(self, player_id: int) -> Dict[str, Any]:
        for player in self.players_config:
            if int(player["id"]) == int(player_id):
                return player
        raise KeyError(f"Unknown player_id: {player_id}")

    def _compute_p_init_from_ed(self, scenarios_df: pd.DataFrame) -> List[List[float]]:
        initial_dispatch = []
        for _, row in scenarios_df.iterrows():
            physical_initial = []
            for block_indices in self.physical_to_block_indices:
                physical_capacity = sum(
                    float(row[f"{self.block_names[b]}_cap"]) for b in block_indices
                )
                physical_initial.append(0.5 * physical_capacity)
            initial_dispatch.append(physical_initial)

        ed_for_p_init = EconomicDispatchModel(
            scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=initial_dispatch,
        )
        ed_for_p_init.solve()
        dispatches = ed_for_p_init.get_dispatches()
        if dispatches is None:
            raise RuntimeError("Economic dispatch did not return dispatches. Cannot compute p_init.")
        return [list(dispatches[s][0]) for s in range(len(dispatches))]

    def _log_availability_interpretation(self) -> None:
        for physical_name, block_indices in self._blocks_by_physical_name().items():
            profile_cols = [
                f"{self.block_names[block_idx]}_profile"
                for block_idx in block_indices
                if f"{self.block_names[block_idx]}_profile" in self.scenarios_df.columns
            ]
            if profile_cols:
                print(
                    f"Availability for {physical_name}: block-level profiles in columns "
                    f"{profile_cols}. Wind profiles are distributed across blocks by "
                    "block pmax / physical pmax."
                )
            else:
                print(
                    f"Availability for {physical_name}: static block-level *_cap columns."
                )

    def _blocks_by_physical_name(self) -> Dict[str, List[int]]:
        blocks: Dict[str, List[int]] = {}
        for block_idx, block_name in enumerate(self.block_names):
            physical_name = self.block_to_physical.get(block_name, block_name)
            blocks.setdefault(physical_name, []).append(block_idx)
        return blocks

    def solve_ed_for_bids(
        self, bid_profile: List[List[List[float]]]
    ) -> Tuple[
        List[List[List[float]]],
        List[List[List[float]]],
        List[List[float]],
        EconomicDispatchModel,
    ]:
        scenarios_df = self._scenarios_df_with_bids(bid_profile)
        ed = EconomicDispatchModel(
            scenarios_df,
            self.costs_df,
            self.ramps_df,
            p_init=self.P_init
        )
        ed.solve()
        dispatches = ed.get_dispatches()
        block_dispatches = ed.get_block_dispatches()
        prices = ed.get_clearing_prices()
        if dispatches is None or block_dispatches is None or prices is None:
            raise RuntimeError("ED solve did not return complete results.")
        return dispatches, block_dispatches, prices, ed

    def _scenarios_df_with_bids(self, bid_profile: List[List[List[float]]]) -> pd.DataFrame:
        scenarios_df = self.scenarios_df.copy(deep=True)
        for s in range(self.num_scenarios):
            for block_idx, block_name in enumerate(self.block_names):
                profile = [float(v) for v in bid_profile[s][block_idx]]
                scenarios_df.at[s, f"{block_name}_bid_profile"] = profile
                scenarios_df.at[s, f"{block_name}_bid"] = float(profile[0])
        return scenarios_df

    def _snapshot_all_bids(self) -> List[List[List[float]]]:
        return [
            [
                self._as_profile(
                    self.scenarios_df.at[s, f"{block_name}_bid_profile"],
                    f"{block_name}_bid_profile",
                )
                for block_name in self.block_names
            ]
            for s in range(self.num_scenarios)
        ]

    def compute_player_profit(
        self,
        player_id: int,
        block_dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
    ) -> Tuple[float, List[float]]:
        controlled = self.player_blocks[int(player_id)]
        scenario_profits = []
        for s in range(self.num_scenarios):
            profit_s = 0.0
            for t in range(self.num_time_steps):
                for block_idx in controlled:
                    p = float(block_dispatches[s][t][block_idx])
                    profit_s += (float(clearing_prices[s][t]) - float(self.cost_vector[block_idx])) * p
            scenario_profits.append(float(profit_s))
        return float(np.mean(scenario_profits)), scenario_profits

    def compute_all_player_profits(
        self,
        block_dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
    ) -> Dict[int, float]:
        return {
            int(player["id"]): self.compute_player_profit(
                int(player["id"]), block_dispatches, clearing_prices
            )[0]
            for player in self.players_config
        }

    def build_merit_order(
        self,
        scenario_id: int,
        time_id: int,
        bid_profile: List[List[List[float]]],
        block_dispatches: Optional[List[List[List[float]]]] = None,
    ) -> List[MeritOrderEntry]:
        entries = []
        for block_idx, block_name in enumerate(self.block_names):
            generator_name = self.block_to_physical.get(block_name, block_name)
            generator_id = self.physical_generator_names.index(generator_name)
            entries.append(
                MeritOrderEntry(
                    block_id=block_idx,
                    block_name=block_name,
                    generator_id=generator_id,
                    generator_name=generator_name,
                    player_id=self.block_to_player.get(block_idx),
                    bid=float(bid_profile[scenario_id][block_idx][time_id]),
                    true_cost=float(self.cost_vector[block_idx]),
                    available_capacity=self.available_capacity(scenario_id, block_idx, time_id),
                    dispatch=(
                        None
                        if block_dispatches is None
                        else float(block_dispatches[scenario_id][time_id][block_idx])
                    ),
                )
            )
        sorted_entries = sorted(
            entries,
            key=lambda entry: (entry.bid, entry.generator_id, entry.block_id),
        )
        cumulative = 0.0
        with_cumulative = []
        for entry in sorted_entries:
            cumulative += float(entry.available_capacity)
            with_cumulative.append(
                MeritOrderEntry(
                    **{**asdict(entry), "cumulative_capacity": float(cumulative)}
                )
            )
        return with_cumulative

    def identify_marginal_block(
        self, merit_order: Sequence[MeritOrderEntry], demand: float
    ) -> Optional[MeritOrderEntry]:
        for entry in merit_order:
            if entry.cumulative_capacity >= float(demand) - self.bid_tolerance:
                return entry
        print(f"Warning: no marginal block found; demand {demand:.6f} exceeds merit-order capacity.")
        return None

    def get_candidate_threshold(
        self,
        block_id: int,
        merit_order: Sequence[MeritOrderEntry],
        marginal_block: Optional[MeritOrderEntry],
    ) -> Optional[float]:
        if marginal_block is None:
            return None

        marginal_position = next(
            idx for idx, entry in enumerate(merit_order) if entry.block_id == marginal_block.block_id
        )
        true_cost = float(self.cost_vector[block_id])
        candidates: List[Tuple[int, int, float]] = []
        for idx, entry in enumerate(merit_order):
            if entry.block_id == block_id:
                continue
            if entry.available_capacity <= self.bid_tolerance:
                continue
            if entry.bid <= true_cost + self.bid_tolerance:
                continue
            distance = abs(idx - marginal_position)
            candidates.append((distance, idx, float(entry.bid)))

        seen = set()
        for _, _, bid in sorted(candidates):
            rounded = round(bid, 12)
            if rounded in seen:
                continue
            seen.add(rounded)
            return float(bid)
        return None

    def available_capacity(self, scenario_id: int, block_id: int, time_id: int) -> float:
        block_name = self.block_names[int(block_id)]
        profile_col = f"{block_name}_cap_profile"
        availability_profile_col = f"{block_name}_profile"
        if profile_col in self.scenarios_df.columns:
            profile = self._as_profile(self.scenarios_df.at[scenario_id, profile_col], profile_col)
            return float(profile[time_id])
        if availability_profile_col in self.scenarios_df.columns:
            profile = self._as_profile(
                self.scenarios_df.at[scenario_id, availability_profile_col],
                availability_profile_col,
            )
            return float(profile[time_id])
        return float(self.scenarios_df.at[scenario_id, f"{block_name}_cap"])

    def demand(self, scenario_id: int, time_id: int) -> float:
        profile = self._as_profile(self.scenarios_df.at[scenario_id, "demand_profile"], "demand_profile")
        return float(profile[time_id])

    def run(self) -> Dict[str, Any]:
        print(
            "Starting merit-order best-response heuristic: "
            f"{self.num_scenarios} scenarios, {self.num_time_steps} time steps, "
            f"{self.num_blocks} blocks"
        )
        current_bids = self._snapshot_all_bids()

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"Iteration {iteration}/{self.MAX_ITERATIONS}")
            any_update = False

            for player in self.players_config:
                player_id = int(player["id"])
                dispatches, block_dispatches, prices, _ = self.solve_ed_for_bids(current_bids)
                baseline_profits = self.compute_all_player_profits(block_dispatches, prices)
                current_profit = float(baseline_profits[player_id])
                best_profit = current_profit
                best_update: Optional[Dict[str, Any]] = None
                print(f"  Player {player_id} baseline profit: {current_profit:.4f}")

                for scenario_id in range(self.num_scenarios):
                    for time_id in range(self.num_time_steps):
                        merit_order = self.build_merit_order(
                            scenario_id,
                            time_id,
                            current_bids,
                            block_dispatches=block_dispatches,
                        )
                        marginal_block = self.identify_marginal_block(
                            merit_order,
                            self.demand(scenario_id, time_id),
                        )
                        if self.debug:
                            marginal = None if marginal_block is None else asdict(marginal_block)
                            print(
                                f"    Debug: s={scenario_id} t={time_id} "
                                f"clearing_price={float(prices[scenario_id][time_id]):.6f} "
                                f"marginal={marginal}"
                            )

                        for block_id in self.player_blocks[player_id]:
                            if (
                                self.available_capacity(scenario_id, block_id, time_id)
                                <= self.bid_tolerance
                            ):
                                self._record_rejected_candidate(
                                    iteration,
                                    player_id,
                                    scenario_id,
                                    time_id,
                                    block_id,
                                    current_profit,
                                    "no_capacity",
                                )
                                continue

                            threshold_bid = self.get_candidate_threshold(
                                block_id=block_id,
                                merit_order=merit_order,
                                marginal_block=marginal_block,
                            )
                            if threshold_bid is None:
                                self._record_rejected_candidate(
                                    iteration,
                                    player_id,
                                    scenario_id,
                                    time_id,
                                    block_id,
                                    current_profit,
                                    "no_threshold",
                                )
                                continue
                            else:
                                old_bid = float(current_bids[scenario_id][block_id][time_id])
                                candidate_bid = float(threshold_bid) - self.bid_tolerance
                                reason = self._candidate_rejection_reason(
                                    block_id,
                                    old_bid,
                                    candidate_bid,
                                )
                                if reason is not None:
                                    self._record_rejected_candidate(
                                        iteration,
                                        player_id,
                                        scenario_id,
                                        time_id,
                                        block_id,
                                        current_profit,
                                        reason,
                                        old_bid=old_bid,
                                        new_bid=candidate_bid,
                                        threshold_bid=float(threshold_bid),
                                    )
                                    continue

                                trial_bids = deepcopy(current_bids)
                                trial_bids[scenario_id][block_id][time_id] = candidate_bid
                                _, trial_block_dispatches, trial_prices, _ = self.solve_ed_for_bids(
                                    trial_bids
                                )
                                trial_profit, _ = self.compute_player_profit(
                                    player_id,
                                    trial_block_dispatches,
                                    trial_prices,
                                )

                                if self.debug:
                                    print(
                                        f"    Debug candidate: player={player_id} "
                                        f"s={scenario_id} t={time_id} block={self.block_names[block_id]} "
                                        f"old={old_bid:.6f} new={candidate_bid:.6f} "
                                        f"threshold={float(threshold_bid):.6f} "
                                        f"baseline={current_profit:.6f} candidate={trial_profit:.6f}"
                                    )
                                if trial_profit > best_profit + self.PROFIT_TOLERANCE:
                                    best_profit = float(trial_profit)
                                    best_update = {
                                        "iteration": int(iteration),
                                        "player_id": int(player_id),
                                        "scenario_id": int(scenario_id),
                                        "time_id": int(time_id),
                                        "block_id": int(block_id),
                                        "block_name": self.block_names[block_id],
                                        "accepted_block": self.block_names[block_id],
                                        "old_bid": old_bid,
                                        "new_bid": candidate_bid,
                                        "threshold_bid": float(threshold_bid),
                                        "baseline_profit": current_profit,
                                        "accepted_profit": float(trial_profit),
                                        "profit_improvement": float(trial_profit - current_profit),
                                        "accepted": True,
                                    }
                                else:
                                    self._record_rejected_candidate(
                                        iteration,
                                        player_id,
                                        scenario_id,
                                        time_id,
                                        block_id,
                                        current_profit,
                                        "no_profit_improvement",
                                        old_bid=old_bid,
                                        new_bid=candidate_bid,
                                        threshold_bid=float(threshold_bid),
                                        candidate_profit=float(trial_profit),
                                    )

                if best_update is not None:
                    self._apply_update(current_bids, best_update)
                    self.history.append(best_update)
                    any_update = True
                    print(
                        f"  Accepted: player {player_id}, block {best_update['block_name']}, "
                        f"scenario {best_update['scenario_id']}, time {best_update['time_id']}, "
                        f"bid {best_update['old_bid']:.4f} -> {best_update['new_bid']:.4f}, "
                        f"profit {best_update['baseline_profit']:.4f} -> "
                        f"{best_update['accepted_profit']:.4f}"
                    )
                else:
                    print(f"  No profitable update for player {player_id}; profit remains {current_profit:.4f}")
                    self.history.append(
                        {
                            "iteration": int(iteration),
                            "player_id": int(player_id),
                            "baseline_profit": current_profit,
                            "accepted_profit": current_profit,
                            "accepted_block": None,
                            "old_bid": None,
                            "new_bid": None,
                            "profit_improvement": 0.0,
                            "accepted": False,
                        }
                    )

            if not any_update:
                print("No profitable deviations found. Stopping.")
                break

        self.results = self.get_results(current_bids)
        return self.results

    def _candidate_rejection_reason(
        self, block_id: int, old_bid: float, candidate_bid: float
    ) -> Optional[str]:
        if candidate_bid < float(self.cost_vector[block_id]) - self.bid_tolerance:
            return "below_true_cost"
        if abs(candidate_bid - old_bid) <= self.bid_tolerance:
            return "unchanged_bid"
        if not np.isfinite(candidate_bid):
            return "non_finite_bid"
        return None

    def _record_rejected_candidate(
        self,
        iteration: int,
        player_id: int,
        scenario_id: int,
        time_id: int,
        block_id: int,
        baseline_profit: float,
        reason: str,
        old_bid: Optional[float] = None,
        new_bid: Optional[float] = None,
        threshold_bid: Optional[float] = None,
        candidate_profit: Optional[float] = None,
    ) -> None:
        if self.debug:
            print(
                f"    Debug rejected: player={player_id} s={scenario_id} t={time_id} "
                f"block={self.block_names[block_id]} reason={reason} old={old_bid} "
                f"new={new_bid} threshold={threshold_bid} baseline={baseline_profit} "
                f"candidate={candidate_profit}"
            )

    def _apply_update(self, bids: List[List[List[float]]], update: Dict[str, Any]) -> None:
        s = int(update["scenario_id"])
        block_id = int(update["block_id"])
        t = int(update["time_id"])
        bids[s][block_id][t] = float(update["new_bid"])
        block_name = self.block_names[block_id]
        profile = self._as_profile(self.scenarios_df.at[s, f"{block_name}_bid_profile"], f"{block_name}_bid_profile")
        profile[t] = float(update["new_bid"])
        self.scenarios_df.at[s, f"{block_name}_bid_profile"] = profile
        self.scenarios_df.at[s, f"{block_name}_bid"] = float(profile[0])

    def get_results(self, bid_profile: Optional[List[List[List[float]]]] = None) -> Dict[str, Any]:
        final_bids = self._snapshot_all_bids() if bid_profile is None else bid_profile
        dispatches, block_dispatches, prices, _ = self.solve_ed_for_bids(final_bids)
        final_profits = self.compute_all_player_profits(block_dispatches, prices)
        return {
            "config": {
                "bid_tolerance": self.bid_tolerance,
            },
            "num_scenarios": self.num_scenarios,
            "num_time_steps": self.num_time_steps,
            "physical_generator_names": self.physical_generator_names,
            "block_names": self.block_names,
            "block_to_physical": self.block_to_physical,
            "block_costs": self.cost_vector.tolist(),
            "player_blocks": {
                int(player_id): [int(block) for block in blocks]
                for player_id, blocks in self.player_blocks.items()
            },
            "final_bids": final_bids,
            "final_dispatches": dispatches,
            "final_block_dispatches": block_dispatches,
            "final_clearing_prices": prices,
            "final_player_profits": {int(k): float(v) for k, v in final_profits.items()},
            "history": self.history,
        }

    def save_results(self, output_path: str = "results/merit_order_best_response_results.json") -> Path:
        results = self.results or self.get_results()
        path = Path(output_path)
        if path.suffix and path.suffix.lower() != ".json":
            raise ValueError("output_path must end with .json or have no extension")
        if not path.suffix:
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2, default=self._json_default_serializer)
        return path

    @staticmethod
    def _json_default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(key): value for key, value in obj.items()}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def apply_known_failure_case_bids(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    updated = scenarios_df.copy(deep=True).reset_index(drop=True)
    example_bids = {
        "W2_B1": 3.73,
        "W3_B1": 4.15,
        "W1_B1": 4.51,
        "W1_B2": 4.51,
        "G2_B1": 30.00,
        "W2_B2": 30.51,
        "W3_B2": 32.05,
        "G1_B2": 33.50,
        "G1_B1": 38.80,
        "G2_B2": 40.00,
    }
    horizon = int(updated.at[0, "time_steps"]) if "time_steps" in updated.columns else 1
    for block_name, bid in example_bids.items():
        if f"{block_name}_bid_profile" in updated.columns:
            updated.at[0, f"{block_name}_bid_profile"] = [float(bid)] * horizon
            updated.at[0, f"{block_name}_bid"] = float(bid)
    return updated.iloc[[0]].reset_index(drop=True)

def build_known_failure_case() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    scenario_manager = ScenarioManager("test_case_bidding_blocks")
    scenarios = scenario_manager.create_scenario_set_from_regimes(
        regime_set="policy_training",
        seed=1,
    )
    scenarios_df = apply_known_failure_case_bids(scenarios["scenarios_df"])
    scenarios_df.at[0, "demand_profile"] = [101.4] * int(scenarios_df.at[0, "time_steps"])

    wind_availability = {
        "W3": 20.391,
        "W2": 20.537,
        "W1": 20.744,
    }
    for physical_name, value in wind_availability.items():
        first = True
        availability_cols = [
            col
            for col in scenarios_df.columns
            if col.startswith(f"{physical_name}_B")
            and col.endswith("_profile")
            and not col.endswith("_bid_profile")
            and not col.endswith("_true_cost_profile")
        ]
        for col in availability_cols:
            scenarios_df.at[0, col] = [float(value if first else 0.0)] * int(scenarios_df.at[0, "time_steps"])
            first = False

    return (
        scenarios_df,
        scenarios["costs_df"],
        scenarios["ramps_df"],
        scenario_manager.get_players_config(),
    )

if __name__ == "__main__":
    case = "test_case_bidding_blocks"
    regime_set = "policy_training"
    seed = 1
    bid_tolerance = 1e-6
    output_path = "results/merit_order_best_response_results.json"
    known_failure_case = False
    debug = False

    if known_failure_case:
        scenarios_df, costs_df, ramps_df, players_config = build_known_failure_case()
    else:
        scenario_manager = ScenarioManager(case)
        scenarios = scenario_manager.create_scenario_set_from_regimes(
            regime_set=regime_set,
            seed=seed,
        )
        scenarios_df = scenarios["scenarios_df"]
        costs_df = scenarios["costs_df"]
        ramps_df = scenarios["ramps_df"]
        players_config = scenario_manager.get_players_config()

    heuristic = MeritOrderHeuristic(
        scenarios_df=scenarios_df,
        costs_df=costs_df,
        ramps_df=ramps_df,
        players_config=players_config,
        bid_tolerance=bid_tolerance,
    )
    heuristic.debug = debug
    heuristic.run()
    saved_path = heuristic.save_results(output_path)
    print(f"Saved merit-order best-response results to {saved_path}")
