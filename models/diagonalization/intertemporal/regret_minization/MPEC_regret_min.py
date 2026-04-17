from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from models.diagonalization.intertemporal.MultipleScenarios.MPEC_MS import MPECModel as MSMPECModel


class MPECModel(MSMPECModel):
    """
    Regret-minimization wrapper around the shared intertemporal MS MPEC core.

    The optimization model is identical to the MS MPEC model. The only
    behavioral change is how `p_init` is handled: a base `p_init` matrix
    provided for the initial scenario set is repeated to match the number of
    rows in the current scenario dataframe.
    """

    def __init__(
        self,
        scenarios_df: pd.DataFrame,
        initial_scenarios_df: Optional[pd.DataFrame],
        costs_df: pd.DataFrame,
        ramps_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        p_init: Any,
        feature_matrix_by_player: Dict[int, Dict[Any, List[float]]],
        pmin_default: float = 0.0,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):

        scenarios_df = scenarios_df.copy().reset_index(drop=True)
        if initial_scenarios_df is None:
            initial_scenarios_df = scenarios_df
        else:
            initial_scenarios_df = initial_scenarios_df.copy().reset_index(drop=True)

        if ramps_df is None:
            raise ValueError("ramps_df must be provided for intertemporal regret-min MPEC.")

        p_init_matrix = self._repeat_given_p_init(p_init, initial_scenarios_df, len(scenarios_df))
        repeated_feature_matrix_by_player = self._repeat_feature_matrix_by_player(feature_matrix_by_player, initial_scenarios_df, len(scenarios_df))

        super().__init__(
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            players_config=players_config,
            p_init=p_init_matrix,
            feature_matrix_by_player=repeated_feature_matrix_by_player,
            pmin_default=pmin_default,
            config_overrides=config_overrides,
        )

    @staticmethod
    def _repeat_given_p_init(
        p_init: Optional[Any],
        initial_scenarios_df: pd.DataFrame,
        target_num_scenarios: int,
    ) -> List[List[float]]:
        if p_init is None:
            raise ValueError(
                "p_init must be provided from the initial ED solve in best response. "
                "Expected shape [num_base_scenarios][num_generators]."
            )

        if isinstance(p_init, np.ndarray):
            p_init = p_init.tolist()

        if not isinstance(p_init, (list, tuple)) or len(p_init) == 0:
            raise ValueError("Invalid p_init format. Expected non-empty [scenarios][generators] matrix.")

        num_base_scenarios = len(initial_scenarios_df)
        generator_names = [c.replace("_cap", "") for c in initial_scenarios_df.columns if c.endswith("_cap")]
        num_generators = len(generator_names)

        if len(p_init) != num_base_scenarios:
            raise ValueError(
                f"p_init has {len(p_init)} rows, but initial_scenarios_df has {num_base_scenarios} rows."
            )

        base_rows: List[List[float]] = []
        for row_idx, row in enumerate(p_init):
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"p_init row {row_idx} is not list-like.")
            if len(row) != num_generators:
                raise ValueError(
                    f"p_init row {row_idx} has {len(row)} values, expected {num_generators}."
                )
            base_rows.append([float(v) for v in row])

        repeated: List[List[float]] = []
        while len(repeated) < target_num_scenarios:
            for row in base_rows:
                repeated.append(list(row))
                if len(repeated) == target_num_scenarios:
                    break
        return repeated

    @staticmethod
    def _repeat_feature_matrix_by_player(
        feature_matrix_by_player: Dict[int, Dict[Any, List[float]]],
        initial_scenarios_df: pd.DataFrame,
        target_num_scenarios: int,
    ) -> Dict[int, Dict[Any, List[float]]]:
        """Repeat a base feature tensor so it matches the current scenario count."""
        num_base_scenarios = len(initial_scenarios_df)
        repeated: Dict[int, Dict[Any, List[float]]] = {}

        for pid, player_features in feature_matrix_by_player.items():
            repeated_player_features: Dict[Any, List[float]] = {}
            for s in range(target_num_scenarios):
                base_s = s % num_base_scenarios
                for (scenario_idx, time_idx, gen_idx), feature_vector in player_features.items():
                    if scenario_idx != base_s:
                        continue
                    repeated_player_features[(s, time_idx, gen_idx)] = list(feature_vector)
            repeated[pid] = repeated_player_features

        return repeated

    def update_current_base_scenario_bids(
        self,
        scenarios_df: pd.DataFrame,
        num_base_scenarios: int,
        controlled_generators: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Update bids for the current base scenarios from solved accumulated bids.

        The regret-min MPEC is solved on an accumulated scenario set. This
        helper applies the last ``num_base_scenarios`` rows of the solved bid
        tensor to ``scenarios_df`` (which contains only the current base
        scenarios).
        """
        optimal_bid_scenarios = self.get_optimal_bids()
        num_accumulated = len(optimal_bid_scenarios)
        base_offset = num_accumulated - num_base_scenarios
        if base_offset < 0:
            raise ValueError(
                f"Accumulated bid tensor has fewer scenarios ({num_accumulated}) than base scenarios ({num_base_scenarios})."
            )

        if controlled_generators is None:
            controlled_generators = list(self.strategic_generators)

        updated_df = scenarios_df.copy()
        for s in range(num_base_scenarios):
            s_acc = base_offset + s
            for gen_idx in controlled_generators:
                gen_name = self.generator_names[gen_idx]
                bid_profile = [
                    float(optimal_bid_scenarios[s_acc][t][gen_idx])
                    for t in range(self.num_time_steps)
                ]
                updated_df.at[s, f"{gen_name}_bid_profile"] = bid_profile
                updated_df.at[s, f"{gen_name}_bid"] = bid_profile[0]

        return updated_df
