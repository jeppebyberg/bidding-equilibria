from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProfitParameters:
    """True production-cost parameters used in player profit calculations."""
    c_linear: np.ndarray
    c_quadratic: np.ndarray

def expand_costs_to_time(
    cost: np.ndarray,
    n_gen: int,
    n_time: int,
    name: str,
) -> np.ndarray:
    """
    Expand generator costs to shape ``(n_gen, n_time)``.

    A one-dimensional input with shape ``(n_gen,)`` is treated as
    time-independent. A two-dimensional input with shape ``(n_gen, n_time)``
    is returned as-is apart from conversion to ``float64``.
    """
    n_gen, n_time = _validate_dimensions(n_gen, n_time)
    arr = np.asarray(cost, dtype=np.float64)

    if arr.shape == (n_gen,):
        return np.broadcast_to(arr[:, None], (n_gen, n_time)).copy()
    if arr.shape == (n_gen, n_time):
        return arr.copy()

    raise ValueError(
        f"{name} must have shape {(n_gen,)} or {(n_gen, n_time)}, got {arr.shape}"
    )

def flatten_time_major(x: np.ndarray) -> np.ndarray:
    """
    Flatten an array with shape ``(n_gen, n_time)`` into time-major order.

    The resulting vector is ``[x[0,0], x[1,0], ..., x[n_gen-1,0],
    x[0,1], ...]``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array with shape (n_gen, n_time), got {arr.shape}")
    return arr.T.reshape(-1)

def compute_player_profit(
    P: np.ndarray,
    lambda_: np.ndarray,
    profit_params: ProfitParameters,
    player_generators: list[int],
) -> float:
    """
    Compute one player's profit for one scenario.

    Profit is summed over the player's owned generators and all time periods:
    ``lambda[t] * P[i,t] - c_linear[i,t] * P[i,t]
    - 0.5 * c_quadratic[i,t] * P[i,t]**2``.
    """
    P_arr, lambda_arr, c_linear, c_quadratic, owned = _validate_profit_inputs(
        P,
        lambda_,
        profit_params,
        player_generators,
    )

    owned_P = P_arr[owned, :]
    owned_linear = c_linear[owned, :]
    owned_quadratic = c_quadratic[owned, :]
    revenue = lambda_arr[None, :] * owned_P
    cost = owned_linear * owned_P + 0.5 * owned_quadratic * owned_P**2
    return float(np.sum(revenue - cost))

def compute_profit_sensitivity_dispatch(
    P: np.ndarray,
    lambda_: np.ndarray,
    profit_params: ProfitParameters,
    player_generators: list[int],
    flatten: bool = True,
) -> np.ndarray:
    """
    Compute ``d pi_j / d P`` for one player's profit.

    Owned generators have derivative ``lambda[t] - c_linear[i,t]
    - c_quadratic[i,t] * P[i,t]``. Non-owned generators have derivative zero.
    If ``flatten`` is true, the result is returned in KKT-compatible
    time-major order with shape ``(n_gen * n_time,)``.
    """
    P_arr, lambda_arr, c_linear, c_quadratic, owned = _validate_profit_inputs(
        P,
        lambda_,
        profit_params,
        player_generators,
    )

    dpi_dP = np.zeros_like(P_arr, dtype=np.float64)
    dpi_dP[owned, :] = (
        lambda_arr[None, :]
        - c_linear[owned, :]
        - c_quadratic[owned, :] * P_arr[owned, :]
    )

    if flatten:
        return flatten_time_major(dpi_dP)
    return dpi_dP

def compute_profit_sensitivity_price(
    P: np.ndarray,
    player_generators: list[int],
) -> np.ndarray:
    """
    Compute ``d pi_j / d lambda`` with shape ``(n_time,)``.

    Each entry is total dispatch from the player's owned generators at that
    time: ``sum_{i in Omega_j} P[i,t]``.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2:
        raise ValueError(f"P must have shape (n_gen, n_time), got {P_arr.shape}")
    n_gen, n_time = P_arr.shape
    _validate_dimensions(n_gen, n_time)
    _validate_finite(P_arr, "P")
    owned = _validate_player_generators(player_generators, n_gen)
    return np.sum(P_arr[owned, :], axis=0)

def compute_profit_sensitivities(
    P: np.ndarray,
    lambda_: np.ndarray,
    profit_params: ProfitParameters,
    player_generators: list[int],
    flatten_dispatch: bool = True,
) -> dict[str, float | np.ndarray]:
    """
    Compute player profit, dispatch sensitivity, and price sensitivity.

    Returns a dictionary with ``profit``, ``dpi_dP``, and ``dpi_dlambda``.
    The dispatch sensitivity is flattened in time-major order when
    ``flatten_dispatch`` is true.
    """
    return {
        "profit": compute_player_profit(P, lambda_, profit_params, player_generators),
        "dpi_dP": compute_profit_sensitivity_dispatch(
            P,
            lambda_,
            profit_params,
            player_generators,
            flatten=flatten_dispatch,
        ),
        "dpi_dlambda": compute_profit_sensitivity_price(P, player_generators),
    }

def finite_difference_check(
    P: np.ndarray,
    lambda_: np.ndarray,
    profit_params: ProfitParameters,
    player_generators: list[int],
    eps: float = 1e-6,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """
    Compare analytical profit sensitivities with central finite differences.

    Every dispatch entry ``P[i,t]`` and every price entry ``lambda[t]`` is
    perturbed one at a time. The returned dictionary contains
    ``max_abs_error_P``, ``max_abs_error_lambda``, and ``passed``.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be nonnegative, got {tolerance}")

    P_arr, lambda_arr, _, _, owned = _validate_profit_inputs(
        P,
        lambda_,
        profit_params,
        player_generators,
    )
    n_gen, n_time = P_arr.shape
    analytical_P = compute_profit_sensitivity_dispatch(
        P_arr,
        lambda_arr,
        profit_params,
        owned,
        flatten=False,
    )
    analytical_lambda = compute_profit_sensitivity_price(P_arr, owned)

    numerical_P = np.empty_like(P_arr, dtype=np.float64)
    for i in range(n_gen):
        for t in range(n_time):
            P_plus = P_arr.copy()
            P_minus = P_arr.copy()
            P_plus[i, t] += eps
            P_minus[i, t] -= eps
            profit_plus = compute_player_profit(P_plus, lambda_arr, profit_params, owned)
            profit_minus = compute_player_profit(P_minus, lambda_arr, profit_params, owned)
            numerical_P[i, t] = (profit_plus - profit_minus) / (2.0 * eps)

    numerical_lambda = np.empty_like(lambda_arr, dtype=np.float64)
    for t in range(n_time):
        lambda_plus = lambda_arr.copy()
        lambda_minus = lambda_arr.copy()
        lambda_plus[t] += eps
        lambda_minus[t] -= eps
        profit_plus = compute_player_profit(P_arr, lambda_plus, profit_params, owned)
        profit_minus = compute_player_profit(P_arr, lambda_minus, profit_params, owned)
        numerical_lambda[t] = (profit_plus - profit_minus) / (2.0 * eps)

    abs_error_P = np.abs(analytical_P - numerical_P)
    abs_error_lambda = np.abs(analytical_lambda - numerical_lambda)
    max_abs_error_P = float(np.max(abs_error_P)) if abs_error_P.size else 0.0
    max_abs_error_lambda = (
        float(np.max(abs_error_lambda)) if abs_error_lambda.size else 0.0
    )

    return {
        "max_abs_error_P": max_abs_error_P,
        "max_abs_error_lambda": max_abs_error_lambda,
        "passed": bool(
            max_abs_error_P <= tolerance and max_abs_error_lambda <= tolerance
        ),
    }

def _validate_profit_inputs(
    P: np.ndarray,
    lambda_: np.ndarray,
    profit_params: ProfitParameters,
    player_generators: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2:
        raise ValueError(f"P must have shape (n_gen, n_time), got {P_arr.shape}")

    n_gen, n_time = P_arr.shape
    _validate_dimensions(n_gen, n_time)

    lambda_arr = np.asarray(lambda_, dtype=np.float64)
    if lambda_arr.shape != (n_time,):
        raise ValueError(f"lambda_ must have shape {(n_time,)}, got {lambda_arr.shape}")

    c_linear = expand_costs_to_time(
        profit_params.c_linear,
        n_gen,
        n_time,
        "profit_params.c_linear",
    )
    c_quadratic = expand_costs_to_time(
        profit_params.c_quadratic,
        n_gen,
        n_time,
        "profit_params.c_quadratic",
    )

    _validate_finite(P_arr, "P")
    _validate_finite(lambda_arr, "lambda_")
    _validate_finite(c_linear, "profit_params.c_linear")
    _validate_finite(c_quadratic, "profit_params.c_quadratic")
    if np.any(c_quadratic < 0.0):
        raise ValueError("profit_params.c_quadratic must be nonnegative")

    owned = _validate_player_generators(player_generators, n_gen)
    return P_arr, lambda_arr, c_linear, c_quadratic, owned

def _validate_player_generators(player_generators: list[int], n_gen: int) -> list[int]:
    if not player_generators:
        raise ValueError("player_generators must not be empty")

    owned = [int(i) for i in player_generators]
    invalid = [i for i in owned if i < 0 or i >= n_gen]
    if invalid:
        raise ValueError(f"player_generators contains invalid generator indices: {invalid}")
    return owned

def _validate_dimensions(n_gen: int, n_time: int) -> tuple[int, int]:
    n_gen = int(n_gen)
    n_time = int(n_time)
    if n_gen <= 0 or n_time <= 0:
        raise ValueError(f"n_gen and n_time must be positive, got {n_gen}, {n_time}")
    return n_gen, n_time

def _validate_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

if __name__ == "__main__":
    n_gen = 3
    n_time = 4

    P = np.array(
        [
            [40.0, 42.0, 45.0, 43.0],
            [50.0, 48.0, 47.0, 49.0],
            [20.0, 22.0, 25.0, 24.0],
        ]
    )
    lambda_ = np.array([60.0, 62.0, 65.0, 63.0])
    profit_params = ProfitParameters(
        c_linear=np.array([30.0, 35.0, 10.0]),
        c_quadratic=np.array([0.01, 0.015, 0.0]),
    )
    player_generators = [0, 2]

    sensitivities = compute_profit_sensitivities(
        P,
        lambda_,
        profit_params,
        player_generators,
    )
    check = finite_difference_check(P, lambda_, profit_params, player_generators)

    print(f"profit: {sensitivities['profit']:.6f}")
    print(f"dpi_dP shape: {sensitivities['dpi_dP'].shape}")
    print(f"dpi_dlambda shape: {sensitivities['dpi_dlambda'].shape}")
    print(f"finite-difference max error P: {check['max_abs_error_P']:.6e}")
    print(f"finite-difference max error lambda: {check['max_abs_error_lambda']:.6e}")
    print(f"finite-difference passed: {check['passed']}")
