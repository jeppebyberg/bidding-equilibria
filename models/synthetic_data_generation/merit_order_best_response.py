from __future__ import annotations

import ast
import json
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
    One-pass merit-order bid inflation heuristic for block bid profiles.

    The heuristic solves ED once at the current bids. For each scenario and
    time step, it identifies the marginal price-setting block from that ED
    dispatch and raises only that block's bid to just below the nearest higher
    opponent bid.
    """

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

    def identify_marginal_price_setter(
        self,
        scenario_id: int,
        time_id: int,
        bid_profile: List[List[List[float]]],
        block_dispatches: List[List[List[float]]],
        clearing_prices: List[List[float]],
    ) -> Optional[MeritOrderEntry]:
        merit_order = self.build_merit_order(
            scenario_id,
            time_id,
            bid_profile,
            block_dispatches=block_dispatches,
        )
        dispatched = [
            entry
            for entry in merit_order
            if entry.dispatch is not None and float(entry.dispatch) > self.bid_tolerance
        ]
        if not dispatched:
            return None

        partially_dispatched = [
            entry
            for entry in dispatched
            if float(entry.dispatch) < float(entry.available_capacity) - self.bid_tolerance
        ]
        if partially_dispatched:
            return max(partially_dispatched, key=lambda entry: (entry.bid, entry.generator_id, entry.block_id))

        price = float(clearing_prices[scenario_id][time_id])
        price_consistent = [
            entry
            for entry in dispatched
            if entry.bid <= price + self.bid_tolerance
        ]
        if price_consistent:
            return max(price_consistent, key=lambda entry: (entry.bid, entry.generator_id, entry.block_id))

        return max(dispatched, key=lambda entry: (entry.bid, entry.generator_id, entry.block_id))

    def get_opponent_threshold(
        self,
        marginal_block: MeritOrderEntry,
        merit_order: Sequence[MeritOrderEntry],
    ) -> Optional[float]:
        if marginal_block.player_id is None:
            return None

        opponent_entries = [
            entry
            for entry in merit_order
            if entry.player_id != marginal_block.player_id
            and entry.available_capacity > self.bid_tolerance
            and entry.bid > marginal_block.bid + self.bid_tolerance
        ]
        if not opponent_entries:
            return None
        return float(min(entry.bid for entry in opponent_entries))

    def get_same_player_blocks_to_inflate(
        self,
        marginal_block: MeritOrderEntry,
        merit_order: Sequence[MeritOrderEntry],
        candidate_bid: float,
    ) -> List[MeritOrderEntry]:
        return [
            entry
            for entry in merit_order
            if entry.player_id == marginal_block.player_id
            and entry.available_capacity > self.bid_tolerance
            and entry.bid < float(candidate_bid) - self.bid_tolerance
        ]

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

    def run(self) -> Dict[str, Any]:
        print(
            "Starting one-pass merit-order heuristic: "
            f"{self.num_scenarios} scenarios, {self.num_time_steps} time steps, "
            f"{self.num_blocks} blocks"
        )
        current_bids = self._snapshot_all_bids()
        dispatches, block_dispatches, prices, _ = self.solve_ed_for_bids(current_bids)
        baseline_profits = self.compute_all_player_profits(block_dispatches, prices)

        for scenario_id in range(self.num_scenarios):
            for time_id in range(self.num_time_steps):
                merit_order = self.build_merit_order(
                    scenario_id,
                    time_id,
                    current_bids,
                    block_dispatches=block_dispatches,
                )
                marginal_block = self.identify_marginal_price_setter(
                    scenario_id,
                    time_id,
                    current_bids,
                    block_dispatches,
                    prices,
                )
                if marginal_block is None:
                    self.history.append(
                        {
                            "scenario_id": int(scenario_id),
                            "time_id": int(time_id),
                            "accepted": False,
                            "reason": "no_dispatched_block",
                        }
                    )
                    continue

                threshold_bid = self.get_opponent_threshold(marginal_block, merit_order)
                if threshold_bid is None:
                    self.history.append(
                        {
                            "scenario_id": int(scenario_id),
                            "time_id": int(time_id),
                            "block_id": int(marginal_block.block_id),
                            "block_name": marginal_block.block_name,
                            "player_id": marginal_block.player_id,
                            "old_bid": float(marginal_block.bid),
                            "accepted": False,
                            "reason": "no_higher_opponent_bid",
                        }
                    )
                    continue

                candidate_bid = float(threshold_bid) - self.bid_tolerance
                inflation_blocks = self.get_same_player_blocks_to_inflate(
                    marginal_block,
                    merit_order,
                    candidate_bid,
                )
                valid_updates = []
                rejected_updates = []
                for block in inflation_blocks:
                    reason = self._candidate_rejection_reason(
                        block.block_id,
                        float(block.bid),
                        candidate_bid,
                    )
                    if reason is None:
                        valid_updates.append(block)
                    else:
                        rejected_updates.append((block, reason))

                if not valid_updates:
                    self.history.append(
                        {
                            "scenario_id": int(scenario_id),
                            "time_id": int(time_id),
                            "block_id": int(marginal_block.block_id),
                            "block_name": marginal_block.block_name,
                            "player_id": marginal_block.player_id,
                            "old_bid": float(marginal_block.bid),
                            "new_bid": candidate_bid,
                            "threshold_bid": float(threshold_bid),
                            "accepted": False,
                            "reason": "no_same_player_blocks_to_inflate",
                        }
                    )
                    continue

                for block, reason in rejected_updates:
                    self.history.append(
                        {
                            "scenario_id": int(scenario_id),
                            "time_id": int(time_id),
                            "block_id": int(block.block_id),
                            "block_name": block.block_name,
                            "player_id": block.player_id,
                            "old_bid": float(block.bid),
                            "new_bid": candidate_bid,
                            "threshold_bid": float(threshold_bid),
                            "price_setter_block_id": int(marginal_block.block_id),
                            "price_setter_block_name": marginal_block.block_name,
                            "accepted": False,
                            "reason": reason,
                        }
                    )

                updated_names = []
                for block in valid_updates:
                    update = {
                        "scenario_id": int(scenario_id),
                        "time_id": int(time_id),
                        "block_id": int(block.block_id),
                        "block_name": block.block_name,
                        "accepted_block": block.block_name,
                        "player_id": block.player_id,
                        "old_bid": float(block.bid),
                        "new_bid": candidate_bid,
                        "threshold_bid": float(threshold_bid),
                        "clearing_price": float(prices[scenario_id][time_id]),
                        "dispatch": float(block.dispatch or 0.0),
                        "price_setter_block_id": int(marginal_block.block_id),
                        "price_setter_block_name": marginal_block.block_name,
                        "accepted": True,
                    }
                    self._apply_update(current_bids, update)
                    self.history.append(update)
                    updated_names.append(block.block_name)
                print(
                    f"  Updated s={scenario_id} t={time_id}: "
                    f"{', '.join(updated_names)} bids -> {candidate_bid:.4f}"
                )

        print(f"One-pass heuristic applied {sum(1 for row in self.history if row.get('accepted'))} bid updates.")
        self.results = self.get_results(
            current_bids,
            dispatches=dispatches,
            block_dispatches=block_dispatches,
            prices=prices,
            player_profits=baseline_profits,
            dispatches_are_recleared=False,
        )
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

    def get_results(
        self,
        bid_profile: Optional[List[List[List[float]]]] = None,
        dispatches: Optional[List[List[List[float]]]] = None,
        block_dispatches: Optional[List[List[List[float]]]] = None,
        prices: Optional[List[List[float]]] = None,
        player_profits: Optional[Dict[int, float]] = None,
        dispatches_are_recleared: bool = True,
    ) -> Dict[str, Any]:
        final_bids = self._snapshot_all_bids() if bid_profile is None else bid_profile
        if dispatches is None or block_dispatches is None or prices is None:
            dispatches, block_dispatches, prices, _ = self.solve_ed_for_bids(final_bids)
        final_profits = (
            self.compute_all_player_profits(block_dispatches, prices)
            if player_profits is None
            else player_profits
        )
        return {
            "config": {
                "bid_tolerance": self.bid_tolerance,
                "heuristic": "single_ed_marginal_price_setter_bid_inflation",
                "ed_solves_in_run": 1,
                "dispatches_are_recleared_after_bid_updates": bool(dispatches_are_recleared),
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

if __name__ == "__main__":
    case = "test_case_bidding_blocks"
    regime_set = "policy_training"
    seed = 1
    bid_tolerance = 1e-2
    output_path = "results/merit_order_best_response_results.json"

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
    heuristic.run()
    saved_path = heuristic.save_results(output_path)
    print(f"Saved merit-order best-response results to {saved_path}")

