from typing import List, Optional, Dict, Any, Tuple
import ast

import numpy as np
import pandas as pd

from models.diagonalization.intertemporal.MultipleScenarios.MPEC_MS import MPECModel as MSMPECModel


class MPECModel(MSMPECModel):
    """
    Regret-minimization compatibility wrapper around the shared intertemporal
    MS MPEC core.

    The optimization model itself is reused from `MultipleScenarios/MPEC_MS.py`.
    This wrapper only adapts feature construction and lifecycle methods used by
    the regret-min best-response driver (`update_scenarios`,
    `update_strategic_player`, `get_policy_bids`).
    """

    def __init__(
        self,
        reference_case: str,
        scenarios_df: pd.DataFrame,
        costs_df: pd.DataFrame,
        players_config: List[Dict[str, Any]],
        feature_matrix_by_player: Optional[Dict[int, Dict[Tuple[int, int, int], List[float]]]] = None,
        strategic_player_id: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        ramps_df: Optional[pd.DataFrame] = None,
        pmin_default: float = 0.0,
        p_init: Optional[Any] = None,
    ):
        self.reference_case = reference_case
        self._pmin_default = float(pmin_default)

        scenarios_df = scenarios_df.copy().reset_index(drop=True)
        ramps_df = ramps_df if ramps_df is not None else self._build_default_ramps_df(scenarios_df)
        p_init_matrix = self._validate_given_p_init(p_init, scenarios_df)

        self._base_p_init = [list(row) for row in p_init_matrix]

        self.feature_matrix_by_player = feature_matrix_by_player

        self.features: Dict[Tuple[int, int, int], List[float]] = {}
        self.num_policy_features = 0
        if self.feature_matrix_by_player:
            first_player = next(iter(self.feature_matrix_by_player.values()))
            if first_player:
                self.num_policy_features = len(next(iter(first_player.values())))
        if self.num_policy_features == 0:
            raise ValueError("feature_matrix_by_player is empty; cannot infer policy feature dimension")

        super().__init__(
            scenarios_df=scenarios_df,
            costs_df=costs_df,
            ramps_df=ramps_df,
            players_config=players_config,
            p_init=p_init_matrix,
            feature_matrix_by_player=feature_matrix_by_player,
            strategic_player_id=strategic_player_id,
            pmin_default=pmin_default,
            config_overrides=config_overrides,
        )

    @staticmethod
    def _parse_profile(value: Any) -> List[float]:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected profile-like list/tuple values in scenario columns.")
        return [float(v) for v in value]

    @staticmethod
    def _infer_num_time_steps(scenarios_df: pd.DataFrame) -> int:
        if "time_steps" in scenarios_df.columns:
            return int(scenarios_df["time_steps"].iloc[0])

        demand_profile_col = next((c for c in scenarios_df.columns if "demand_profile" in c.lower()), None)
        if demand_profile_col is None:
            raise ValueError("No demand_profile column found in scenarios_df.")

        return len(MPECModel._parse_profile(scenarios_df[demand_profile_col].iloc[0]))

    @staticmethod
    def _infer_generator_names(scenarios_df: pd.DataFrame) -> List[str]:
        return [c.replace("_cap", "") for c in scenarios_df.columns if c.endswith("_cap")]

    def _build_default_ramps_df(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        generator_names = self._infer_generator_names(scenarios_df)
        ramp_data: Dict[str, float] = {}
        for gen in generator_names:
            cap_col = f"{gen}_cap"
            max_cap = float(pd.to_numeric(scenarios_df[cap_col], errors="coerce").fillna(0.0).max())
            fallback = max(1.0, max_cap)
            ramp_data[f"{gen}_ramp_up"] = fallback
            ramp_data[f"{gen}_ramp_down"] = fallback
        return pd.DataFrame([ramp_data])

    def _validate_given_p_init(self, p_init: Optional[Any], scenarios_df: pd.DataFrame) -> List[List[float]]:
        num_scenarios = len(scenarios_df)
        num_generators = len(self._infer_generator_names(scenarios_df))

        if p_init is None:
            raise ValueError(
                "p_init must be provided from the initial ED solve in best response. "
                "Expected shape [num_scenarios][num_generators]."
            )

        if isinstance(p_init, np.ndarray):
            p_init = p_init.tolist()

        if isinstance(p_init, (list, tuple)) and len(p_init) == num_scenarios:
            matrix: List[List[float]] = []
            for row in p_init:
                if isinstance(row, np.ndarray):
                    row = row.tolist()
                if not isinstance(row, (list, tuple)) or len(row) != num_generators:
                    raise ValueError("Invalid p_init shape. Expected [num_scenarios][num_generators].")
                matrix.append([float(v) for v in row])
            return matrix

        raise ValueError(
            "Invalid p_init format. Expected [num_scenarios][num_generators] from initial ED dispatch at t=0."
        )

    def _repeat_base_p_init(self, target_rows: int) -> List[List[float]]:
        target_cols = self.num_generators

        base_rows = [list(map(float, row)) for row in self._base_p_init]
        if not base_rows:
            return [[0.0 for _ in range(target_cols)] for _ in range(target_rows)]

        for idx, row in enumerate(base_rows):
            if len(row) != target_cols:
                raise ValueError(f"p_init row {idx} has {len(row)} values, expected {target_cols}.")

        repeated: List[List[float]] = []
        while len(repeated) < target_rows:
            for row in base_rows:
                repeated.append(list(row))
                if len(repeated) == target_rows:
                    break
        return repeated

    def update_scenarios(
        self,
        new_scenarios_df: pd.DataFrame,
        feature_matrix_by_player: Optional[Dict[int, Dict[Tuple[int, int, int], List[float]]]] = None,
    ) -> None:
        """
        Replace scenario data and invalidate the built model.
        Feature matrices are regenerated from the supplied dataframe.
        """
        self.scenarios_df = new_scenarios_df.copy().reset_index(drop=True)

        if feature_matrix_by_player is not None:
            self.feature_matrix_by_player = feature_matrix_by_player

        if not self.feature_matrix_by_player:
            raise ValueError(
                "feature_matrix_by_player must be provided on init or update_scenarios, "
                "matching the current scenarios DataFrame."
            )

        self._extract_scenario_data(self.scenarios_df, self.costs_df, self.ramps_df, self._pmin_default)
        self.P_init = self._repeat_base_p_init(self.num_scenarios)

        first_player = next(iter(self.feature_matrix_by_player.values()), {})
        if not first_player:
            raise ValueError("Feature matrix is empty after update_scenarios.")
        self.num_policy_features = len(next(iter(first_player.values())))

        self.model = None

    def update_strategic_player(self, strategic_player_id: int) -> None:
        """Compatibility alias for older regret-min driver workflow."""
        self.build_model(strategic_player_id)

    def get_policy_bids(
        self,
        theta: np.ndarray,
        scenarios_df: pd.DataFrame,
        feature_matrix_by_player: Optional[Dict[int, Dict[Tuple[int, int, int], List[float]]]] = None,
    ) -> List[List[List[float]]]:
        """
        Evaluate a provided theta on scenarios and return full bid tensor [s][t][i].
        """
        if self.strategic_player_id is None:
            raise ValueError("Strategic player must be selected before evaluating policy bids.")

        eval_matrix_by_player = feature_matrix_by_player or self.feature_matrix_by_player
        if not eval_matrix_by_player:
            raise ValueError("feature_matrix_by_player is required to evaluate policy bids.")
        if self.strategic_player_id not in eval_matrix_by_player:
            raise ValueError(f"Missing features for strategic player {self.strategic_player_id}.")

        eval_features = eval_matrix_by_player[self.strategic_player_id]

        eval_df = scenarios_df.copy().reset_index(drop=True)
        num_scenarios = len(eval_df)
        num_time_steps = self._infer_num_time_steps(eval_df)

        bids: List[List[List[float]]] = []
        for s, row in eval_df.iterrows():
            scenario_bids_by_t: List[List[float]] = []
            for t in range(num_time_steps):
                bids_t: List[float] = []
                for gen_name in self.generator_names:
                    bid_profile_col = f"{gen_name}_bid_profile"
                    bid_col = f"{gen_name}_bid"

                    if bid_profile_col in eval_df.columns:
                        profile = row[bid_profile_col]
                        if isinstance(profile, str):
                            profile = ast.literal_eval(profile)
                        if isinstance(profile, (list, tuple)) and len(profile) == num_time_steps:
                            bids_t.append(float(profile[t]))
                            continue

                    if bid_col in eval_df.columns:
                        bids_t.append(float(row[bid_col]))
                    else:
                        bids_t.append(float(self.costs_df[f"{gen_name}_cost"].iloc[0]))

                scenario_bids_by_t.append(bids_t)
            bids.append(scenario_bids_by_t)

        controlled = self.strategic_generators
        for s in range(num_scenarios):
            for t in range(num_time_steps):
                for i in controlled:
                    key = (s, t, i)
                    if key not in eval_features:
                        raise ValueError(
                            f"Missing feature vector for key {key} in feature_matrix_by_player for "
                            f"player {self.strategic_player_id}."
                        )
                    phi = np.array(eval_features[key], dtype=np.float64)
                    bids[s][t][i] = float(theta @ phi)

        return bids

    def update_bids_with_optimal_values(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update strategic bid profiles via shared MS implementation and mirror
        t=0 values to scalar `{gen}_bid` columns for backward compatibility.
        """
        updated_df = super().update_bids_with_optimal_values(scenarios_df)

        for gen_idx in self.strategic_generators:
            gen_name = self.generator_names[gen_idx]
            bid_profile_col = f"{gen_name}_bid_profile"
            bid_col = f"{gen_name}_bid"
            for s in range(self.num_scenarios):
                profile = updated_df.at[s, bid_profile_col]
                if isinstance(profile, str):
                    profile = ast.literal_eval(profile)
                if isinstance(profile, (list, tuple)) and len(profile) > 0:
                    updated_df.at[s, bid_col] = float(profile[0])

        return updated_df
