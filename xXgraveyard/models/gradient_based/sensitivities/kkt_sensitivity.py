from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np


@dataclass(frozen=True)
class EDSolution:
    """Solved primal and dual variables for the intertemporal ED KKT system."""

    P: np.ndarray
    lambda_: np.ndarray
    mu_max: np.ndarray
    mu_min: np.ndarray
    mu_up: np.ndarray
    mu_down: np.ndarray
    P_phys: Optional[np.ndarray] = None


@dataclass(frozen=True)
class EDParameters:
    """Parameters entering the quadratic intertemporal economic dispatch problem."""

    alpha: np.ndarray
    beta: np.ndarray
    demand: np.ndarray
    pmax: np.ndarray
    pmin: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    p_initial: np.ndarray
    physical_to_block_indices: Optional[List[List[int]]] = None
    block_to_physical_idx: Optional[List[int]] = None
    num_physical_generators: Optional[int] = None
    num_blocks: Optional[int] = None

def flatten_time_major(x: np.ndarray) -> np.ndarray:
    """
    Flatten an array with shape (n_gen, n_time) into time-major order.

    The resulting vector is [x[0,0], x[1,0], ..., x[n_gen-1,0], x[0,1], ...].
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array with shape (n_gen, n_time), got {arr.shape}")
    return arr.T.reshape(-1)

def unflatten_time_major(x: np.ndarray, n_gen: int, n_time: int) -> np.ndarray:
    """Unflatten a time-major vector into shape (n_gen, n_time)."""
    vec = np.asarray(x, dtype=np.float64).reshape(-1)
    expected = int(n_gen) * int(n_time)
    if vec.size != expected:
        raise ValueError(f"Expected vector of length {expected}, got {vec.size}")
    return vec.reshape(int(n_time), int(n_gen)).T

def build_balance_matrix(n_gen: int, n_time: int) -> np.ndarray:
    """
    Build M such that M @ P_vec gives total generation at each time step.

    With the convention F_balance = D - M @ P, each row of M selects all
    generators in one time block.
    """
    n_gen, n_time = _validate_dimensions(n_gen, n_time)
    M = np.zeros((n_time, n_gen * n_time), dtype=np.float64)
    for t in range(n_time):
        start = t * n_gen
        M[t, start:start + n_gen] = 1.0
    return M

def build_ramp_up_matrix(
    n_gen: int,
    n_time: int,
    physical_to_block_indices: Optional[List[List[int]]] = None,
    n_phys: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build G_up and a generator index vector for ramp-up rows.

    Row (i,t) of G_up @ P_vec aggregates all blocks for physical generator i.
    For t=0 it is sum_b P[i,b,0]; for t>0 it is
    sum_b P[i,b,t] - sum_b P[i,b,t-1]. The returned generator index vector maps each
    ramp row to the generator whose ramp_up and p_initial parameters it uses.
    """
    n_gen, n_time = _validate_dimensions(n_gen, n_time)
    block_groups = _normalize_block_groups(n_gen, physical_to_block_indices, n_phys)
    n_phys = len(block_groups)
    G = np.zeros((n_phys * n_time, n_gen * n_time), dtype=np.float64)
    row_generators = np.zeros(n_phys * n_time, dtype=np.int64)
    for t in range(n_time):
        for i, block_indices in enumerate(block_groups):
            row = _time_major_index(i, t, n_phys)
            for block_idx in block_indices:
                G[row, _time_major_index(block_idx, t, n_gen)] = 1.0
                if t > 0:
                    G[row, _time_major_index(block_idx, t - 1, n_gen)] = -1.0
            row_generators[row] = i
    return G, row_generators

def build_ramp_down_matrix(
    n_gen: int,
    n_time: int,
    physical_to_block_indices: Optional[List[List[int]]] = None,
    n_phys: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build G_down and a generator index vector for ramp-down rows.

    Row (i,t) of G_down @ P_vec aggregates all blocks for physical generator i.
    For t=0 it is -sum_b P[i,b,0]; for t>0 it is
    -sum_b P[i,b,t] + sum_b P[i,b,t-1]. The returned generator index vector maps each
    ramp row to the generator whose ramp_down and p_initial parameters it uses.
    """
    n_gen, n_time = _validate_dimensions(n_gen, n_time)
    block_groups = _normalize_block_groups(n_gen, physical_to_block_indices, n_phys)
    n_phys = len(block_groups)
    G = np.zeros((n_phys * n_time, n_gen * n_time), dtype=np.float64)
    row_generators = np.zeros(n_phys * n_time, dtype=np.int64)
    for t in range(n_time):
        for i, block_indices in enumerate(block_groups):
            row = _time_major_index(i, t, n_phys)
            for block_idx in block_indices:
                G[row, _time_major_index(block_idx, t, n_gen)] = -1.0
                if t > 0:
                    G[row, _time_major_index(block_idx, t - 1, n_gen)] = 1.0
            row_generators[row] = i
    return G, row_generators

def build_kkt_jacobian(solution: EDSolution, params: EDParameters) -> np.ndarray:
    """
    Build dF/dz for z = [P, lambda, mu_max, mu_min, mu_up, mu_down].

    The residual ordering is [balance, stationarity, max complementarity,
    min complementarity, ramp-up complementarity, ramp-down complementarity].
    """
    n_gen, n_time, n_phys = _validate_solution_and_params(solution, params)
    n_p = n_gen * n_time
    n_ramp = n_phys * n_time
    M = build_balance_matrix(n_gen, n_time)
    G_up, up_generators = build_ramp_up_matrix(
        n_gen, n_time, params.physical_to_block_indices, n_phys
    )
    G_down, down_generators = build_ramp_down_matrix(
        n_gen, n_time, params.physical_to_block_indices, n_phys
    )

    P_vec = flatten_time_major(solution.P)
    beta_vec = flatten_time_major(params.beta)
    pmax_vec = flatten_time_major(params.pmax)
    pmin_vec = flatten_time_major(params.pmin)
    mu_max_vec = flatten_time_major(solution.mu_max)
    mu_min_vec = flatten_time_major(solution.mu_min)
    mu_up_vec = flatten_time_major(solution.mu_up)
    mu_down_vec = flatten_time_major(solution.mu_down)
    R_up_vec = _ramp_rhs_vector(params.ramp_up, params.p_initial, up_generators, sign=1.0)
    R_down_vec = _ramp_rhs_vector(params.ramp_down, params.p_initial, down_generators, sign=-1.0)

    g_max = P_vec - pmax_vec
    g_min = -P_vec + pmin_vec
    g_up = G_up @ P_vec - R_up_vec
    g_down = G_down @ P_vec - R_down_vec

    zero_t_p = np.zeros((n_time, n_p), dtype=np.float64)
    zero_t_r = np.zeros((n_time, n_ramp), dtype=np.float64)
    zero_p_t = np.zeros((n_p, n_time), dtype=np.float64)
    zero_p_p = np.zeros((n_p, n_p), dtype=np.float64)
    zero_p_r = np.zeros((n_p, n_ramp), dtype=np.float64)
    zero_r_t = np.zeros((n_ramp, n_time), dtype=np.float64)
    zero_r_p = np.zeros((n_ramp, n_p), dtype=np.float64)
    zero_r_r = np.zeros((n_ramp, n_ramp), dtype=np.float64)
    I_p = np.eye(n_p, dtype=np.float64)

    return np.block([
        [-M, zero_t_p[:, :n_time], zero_t_p, zero_t_p, zero_t_r, zero_t_r],
        [np.diag(beta_vec), -M.T, I_p, -I_p, G_up.T, G_down.T],
        [np.diag(mu_max_vec), zero_p_t, np.diag(g_max), zero_p_p, zero_p_r, zero_p_r],
        [-np.diag(mu_min_vec), zero_p_t, zero_p_p, np.diag(g_min), zero_p_r, zero_p_r],
        [np.diag(mu_up_vec) @ G_up, zero_r_t, zero_r_p, zero_r_p, np.diag(g_up), zero_r_r],
        [np.diag(mu_down_vec) @ G_down, zero_r_t, zero_r_p, zero_r_p, zero_r_r, np.diag(g_down)],
    ])

def build_bid_derivative_alpha(
    player_generators: List[int],
    n_gen: int,
    n_time: int,
    n_phys: Optional[int] = None,
) -> np.ndarray:
    """
    Build dF/dalpha_j for a player's owned generator-time bid coefficients.

    Alpha enters only stationarity, so this matrix is zero outside the
    stationarity residual block.
    """
    n_gen, n_time = _validate_dimensions(n_gen, n_time)
    owned_indices = _owned_time_major_indices(player_generators, n_gen, n_time)
    n_p = n_gen * n_time

    dim_f = _kkt_residual_dimension(n_gen, n_time, n_gen if n_phys is None else int(n_phys))
    derivative = np.zeros((dim_f, len(owned_indices)), dtype=np.float64)
    for col, p_idx in enumerate(owned_indices):
        derivative[n_time + p_idx, col] = 1.0
    return derivative

def build_bid_derivative_beta(
    solution: EDSolution,
    player_generators: List[int],
    n_gen: int,
    n_time: int,
    n_phys: Optional[int] = None,
) -> np.ndarray:
    """
    Build dF/dbeta_j for a player's owned generator-time quadratic coefficients.

    Beta enters stationarity as beta * P, giving diag(P_vec) @ E_j in the
    stationarity block.
    """
    _validate_dimensions(n_gen, n_time)
    _require_shape("solution.P", solution.P, (n_gen, n_time))
    owned_indices = _owned_time_major_indices(player_generators, n_gen, n_time)
    n_p = n_gen * n_time
    dim_f = _kkt_residual_dimension(n_gen, n_time, n_gen if n_phys is None else int(n_phys))
    P_vec = flatten_time_major(solution.P)
    derivative = np.zeros((dim_f, len(owned_indices)), dtype=np.float64)
    for col, p_idx in enumerate(owned_indices):
        derivative[n_time + p_idx, col] = P_vec[p_idx]
    return derivative

def compute_market_sensitivities(
    solution: EDSolution,
    params: EDParameters,
    player_generators: List[int],
    include_beta: bool = False,
    regularization: float = 0.0,
    condition_warning_threshold: float = 1e10,
) -> Dict[str, np.ndarray | float]:
    """
    Compute KKT implicit-function sensitivities of market outcomes to bids.

    Returns dP/dalpha and dlambda/dalpha in time-major columns for the selected
    player's generator-time bid entries. If include_beta is true, the same
    sensitivities are also returned for beta.
    """
    n_gen, n_time, n_phys = _validate_solution_and_params(solution, params)
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if condition_warning_threshold <= 0:
        raise ValueError("condition_warning_threshold must be positive")

    n_p = n_gen * n_time
    J = build_kkt_jacobian(solution, params)
    if regularization > 0.0:
        J = J + float(regularization) * np.eye(J.shape[0], dtype=np.float64)

    condition_number = float(np.linalg.cond(J))
    if not np.isfinite(condition_number) or condition_number > condition_warning_threshold:
        warnings.warn(
            f"KKT Jacobian is ill-conditioned; condition number is {condition_number:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    A_alpha = build_bid_derivative_alpha(player_generators, n_gen, n_time, n_phys)
    dz_dalpha = -_solve_linear_system(J, A_alpha, "alpha")

    result: Dict[str, np.ndarray | float] = {
        "dP_dalpha": dz_dalpha[:n_p, :],
        "dlambda_dalpha": dz_dalpha[n_p:n_p + n_time, :],
        "condition_number": condition_number,
    }

    if include_beta:
        A_beta = build_bid_derivative_beta(solution, player_generators, n_gen, n_time, n_phys)
        dz_dbeta = -_solve_linear_system(J, A_beta, "beta")
        result["dP_dbeta"] = dz_dbeta[:n_p, :]
        result["dlambda_dbeta"] = dz_dbeta[n_p:n_p + n_time, :]

    return result

def _validate_solution_and_params(solution: EDSolution, params: EDParameters) -> Tuple[int, int, int]:
    alpha = np.asarray(params.alpha, dtype=np.float64)
    if alpha.ndim != 2:
        raise ValueError(f"params.alpha must have shape (n_blocks, n_time), got {alpha.shape}")
    n_gen, n_time = alpha.shape
    _validate_dimensions(n_gen, n_time)
    if params.num_blocks is not None and int(params.num_blocks) != n_gen:
        raise ValueError(f"params.num_blocks={params.num_blocks} does not match alpha rows {n_gen}")
    block_groups = _normalize_block_groups(
        n_gen,
        params.physical_to_block_indices,
        params.num_physical_generators,
    )
    n_phys = len(block_groups)

    for name, value in (
        ("params.beta", params.beta),
        ("params.pmax", params.pmax),
        ("params.pmin", params.pmin),
        ("solution.P", solution.P),
        ("solution.mu_max", solution.mu_max),
        ("solution.mu_min", solution.mu_min),
    ):
        _require_shape(name, value, (n_gen, n_time))

    for name, value in (
        ("solution.mu_up", solution.mu_up),
        ("solution.mu_down", solution.mu_down),
    ):
        _require_shape(name, value, (n_phys, n_time))

    for name, value in (
        ("params.demand", params.demand),
        ("solution.lambda_", solution.lambda_),
    ):
        _require_shape(name, value, (n_time,))

    for name, value in (
        ("params.ramp_up", params.ramp_up),
        ("params.ramp_down", params.ramp_down),
        ("params.p_initial", params.p_initial),
    ):
        _require_shape(name, value, (n_phys,))

    if solution.P_phys is not None:
        _require_shape("solution.P_phys", solution.P_phys, (n_phys, n_time))

    beta = np.asarray(params.beta, dtype=np.float64)
    if not np.all(beta > 0.0):
        raise ValueError("params.beta must be strictly positive for the convex quadratic ED objective")

    arrays = [
        params.alpha, params.beta, params.demand, params.pmax, params.pmin,
        params.ramp_up, params.ramp_down, params.p_initial, solution.P,
        solution.lambda_, solution.mu_max, solution.mu_min, solution.mu_up,
        solution.mu_down,
    ]
    if not all(np.all(np.isfinite(np.asarray(arr, dtype=np.float64))) for arr in arrays):
        raise ValueError("All solution and parameter arrays must contain only finite values")

    return int(n_gen), int(n_time), int(n_phys)

def _validate_dimensions(n_gen: int, n_time: int) -> Tuple[int, int]:
    n_gen = int(n_gen)
    n_time = int(n_time)
    if n_gen <= 0 or n_time <= 0:
        raise ValueError(f"n_gen and n_time must be positive, got {n_gen}, {n_time}")
    return n_gen, n_time

def _require_shape(name: str, value: np.ndarray, expected: Tuple[int, ...]) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")

def _time_major_index(gen_idx: int, time_idx: int, n_gen: int) -> int:
    return int(time_idx) * int(n_gen) + int(gen_idx)

def _owned_time_major_indices(
    player_generators: List[int],
    n_gen: int,
    n_time: int,
) -> List[int]:
    if not player_generators:
        raise ValueError("player_generators must not be empty")
    owned = [int(i) for i in player_generators]
    invalid = [i for i in owned if i < 0 or i >= n_gen]
    if invalid:
        raise ValueError(f"player_generators contains invalid generator indices: {invalid}")

    indices = []
    for t in range(n_time):
        for i in owned:
            indices.append(_time_major_index(i, t, n_gen))
    return indices

def _normalize_block_groups(
    n_blocks: int,
    physical_to_block_indices: Optional[List[List[int]]],
    n_phys: Optional[int] = None,
) -> List[List[int]]:
    """Return physical-generator block groups; default is one block per generator."""
    n_blocks = int(n_blocks)
    if physical_to_block_indices is None:
        if n_phys is not None and int(n_phys) != n_blocks:
            raise ValueError(
                "physical_to_block_indices is required when physical-generator "
                "and bidding-block counts differ"
            )
        return [[i] for i in range(n_blocks)]

    groups = [[int(block_idx) for block_idx in group] for group in physical_to_block_indices]
    if n_phys is not None and len(groups) != int(n_phys):
        raise ValueError(
            f"physical_to_block_indices has {len(groups)} groups, expected {int(n_phys)}"
        )
    if any(len(group) == 0 for group in groups):
        raise ValueError("Every physical generator must own at least one bidding block")

    flattened = [block_idx for group in groups for block_idx in group]
    invalid = [block_idx for block_idx in flattened if block_idx < 0 or block_idx >= n_blocks]
    if invalid:
        raise ValueError(f"physical_to_block_indices contains invalid block indices: {invalid}")
    if sorted(flattened) != list(range(n_blocks)):
        raise ValueError("physical_to_block_indices must cover every block exactly once")
    return groups

def _kkt_residual_dimension(n_blocks: int, n_time: int, n_phys: int) -> int:
    n_blocks, n_time = _validate_dimensions(n_blocks, n_time)
    n_phys = int(n_phys)
    if n_phys <= 0:
        raise ValueError(f"n_phys must be positive, got {n_phys}")
    return n_time + 3 * n_blocks * n_time + 2 * n_phys * n_time

def _ramp_rhs_vector(
    ramp: np.ndarray,
    p_initial: np.ndarray,
    row_generators: np.ndarray,
    sign: float,
) -> np.ndarray:
    ramp = np.asarray(ramp, dtype=np.float64)
    p_initial = np.asarray(p_initial, dtype=np.float64)
    rhs = np.zeros(len(row_generators), dtype=np.float64)
    n_gen = ramp.size
    for row, gen_idx in enumerate(row_generators):
        t = row // n_gen
        rhs[row] = ramp[int(gen_idx)]
        if t == 0:
            rhs[row] += sign * p_initial[int(gen_idx)]
    return rhs

def _solve_linear_system(J: np.ndarray, rhs: np.ndarray, label: str) -> np.ndarray:
    try:
        return np.linalg.solve(J, rhs)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"KKT Jacobian is singular while solving {label} sensitivities; falling back to np.linalg.lstsq.",
            RuntimeWarning,
            stacklevel=2,
        )
        solution, *_ = np.linalg.lstsq(J, rhs, rcond=None)
        return solution

if __name__ == "__main__":
    n_gen = 3
    n_time = 3

    ed_solution = EDSolution(
        P=np.array([[40.0, 45.0, 50.0], [60.0, 55.0, 60.0], [70.0, 65.0, 70.0]]),
        lambda_=np.array([50.0, 50.0, 50.0]),
        mu_max=np.zeros((n_gen, n_time)),
        mu_min=np.zeros((n_gen, n_time)),
        mu_up=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        mu_down=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    ed_params = EDParameters(
        alpha=np.array([[10.0, 10.0, 10.0], [12.0, 12.0, 12.0], [14.0, 14.0, 14.0]]),
        beta=np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
        demand=np.array([100.0, 100.0, 100.0]),
        pmax=np.array([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]),
        pmin=np.zeros((n_gen, n_time)),
        ramp_up=np.array([20.0, 20.0, 20.0]),
        ramp_down=np.array([20.0, 20.0, 20.0]),
        p_initial=np.array([35.0, 65.0, 95.0]),
    )

    sensitivities = compute_market_sensitivities(
        ed_solution,
        ed_params,
        player_generators=[0],
        include_beta=True,
        regularization=1e-8,
    )

    print("dP_dalpha shape:", sensitivities["dP_dalpha"].shape)
    print("dlambda_dalpha shape:", sensitivities["dlambda_dalpha"].shape)
    print("condition number:", sensitivities["condition_number"])
    print("finite dP_dalpha:", np.all(np.isfinite(sensitivities["dP_dalpha"])))
