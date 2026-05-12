from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyParameters:
    """Parameters for one player's one-hidden-layer ReLU bidding policy."""
    Gamma: np.ndarray
    gamma: np.ndarray
    Theta: np.ndarray
    rho: np.ndarray

def relu(x: np.ndarray) -> np.ndarray:
    """Apply the ReLU activation elementwise."""
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)

def relu_derivative(x: np.ndarray, at_zero: float = 0.0) -> np.ndarray:
    """
    Evaluate the elementwise ReLU derivative.

    The derivative is 1 for positive inputs, 0 for negative inputs, and
    ``at_zero`` for inputs exactly equal to zero.
    """
    arr = np.asarray(x, dtype=np.float64)
    deriv = np.zeros_like(arr, dtype=np.float64)
    deriv[arr > 0.0] = 1.0
    deriv[arr == 0.0] = float(at_zero)
    return deriv

def parameter_block_size(n_neurons: int, n_features: int) -> int:
    """Return the number of parameters for one owned generator."""
    n_neurons = _validate_positive_int(n_neurons, "n_neurons")
    n_features = _validate_positive_int(n_features, "n_features")
    return n_neurons * n_features + n_neurons + n_neurons + 1

def total_parameter_size(n_owned: int, n_neurons: int, n_features: int) -> int:
    """Return the total number of policy parameters for one player."""
    n_owned = _validate_positive_int(n_owned, "n_owned")
    return n_owned * parameter_block_size(n_neurons, n_features)

def flatten_parameters(params: PolicyParameters) -> np.ndarray:
    """
    Flatten policy parameters into one vector.

    For each owned generator, the order is row-major ``Gamma_i``, ``gamma_i``,
    ``Theta_i``, and ``rho_i``. Generator blocks follow the local owned
    generator order represented by axis 0 of ``params``.
    """
    n_owned, n_neurons, n_features = _validate_params(params)
    block_size = parameter_block_size(n_neurons, n_features)
    theta = np.empty(n_owned * block_size, dtype=np.float64)

    offset = 0
    for g in range(n_owned):
        theta[offset:offset + n_neurons * n_features] = params.Gamma[g].reshape(-1)
        offset += n_neurons * n_features
        theta[offset:offset + n_neurons] = params.gamma[g]
        offset += n_neurons
        theta[offset:offset + n_neurons] = params.Theta[g]
        offset += n_neurons
        theta[offset] = params.rho[g]
        offset += 1

    return theta

def unflatten_parameters(
    theta: np.ndarray,
    n_owned: int,
    n_neurons: int,
    n_features: int,
) -> PolicyParameters:
    """Invert :func:`flatten_parameters` for the supplied dimensions."""
    n_owned = _validate_positive_int(n_owned, "n_owned")
    n_neurons = _validate_positive_int(n_neurons, "n_neurons")
    n_features = _validate_positive_int(n_features, "n_features")

    vec = np.asarray(theta, dtype=np.float64).reshape(-1)
    expected = total_parameter_size(n_owned, n_neurons, n_features)
    if vec.size != expected:
        raise ValueError(f"Expected theta with length {expected}, got {vec.size}")
    _validate_finite(vec, "theta")

    Gamma = np.empty((n_owned, n_neurons, n_features), dtype=np.float64)
    gamma = np.empty((n_owned, n_neurons), dtype=np.float64)
    Theta = np.empty((n_owned, n_neurons), dtype=np.float64)
    rho = np.empty(n_owned, dtype=np.float64)

    offset = 0
    for g in range(n_owned):
        size = n_neurons * n_features
        Gamma[g] = vec[offset:offset + size].reshape(n_neurons, n_features)
        offset += size
        gamma[g] = vec[offset:offset + n_neurons]
        offset += n_neurons
        Theta[g] = vec[offset:offset + n_neurons]
        offset += n_neurons
        rho[g] = vec[offset]
        offset += 1

    return PolicyParameters(Gamma=Gamma, gamma=gamma, Theta=Theta, rho=rho)

def compute_policy_bids(
    features: np.ndarray,
    params: PolicyParameters,
) -> np.ndarray:
    """
    Compute bid trajectories for one player's owned generators.

    Parameters
    ----------
    features:
        Feature matrix with shape ``(n_time, n_features)``.
    params:
        Policy parameters with local owned-generator axis first.

    Returns
    -------
    np.ndarray
        Bid matrix with shape ``(n_owned, n_time)``.
    """
    features_arr, n_time, _ = _validate_features_and_params(features, params)
    z = np.einsum("gnf,tf->gtn", params.Gamma, features_arr) + params.gamma[:, None, :]
    hidden = relu(z)
    return np.einsum("gn,gtn->gt", params.Theta, hidden) + params.rho[:, None]

def compute_policy_jacobian(
    features: np.ndarray,
    params: PolicyParameters,
    at_zero: float = 0.0,
) -> np.ndarray:
    """
    Compute the Jacobian of time-major flattened bids with respect to parameters.

    Rows are ordered as ``row = t * n_owned + g``. Columns follow
    :func:`flatten_parameters`. A row for generator ``g`` has nonzero entries
    only in generator ``g``'s own parameter block.
    """
    features_arr, n_time, n_features = _validate_features_and_params(features, params)
    n_owned, n_neurons, _ = params.Gamma.shape
    block_size = parameter_block_size(n_neurons, n_features)
    n_theta = total_parameter_size(n_owned, n_neurons, n_features)
    db_dtheta = np.zeros((n_owned * n_time, n_theta), dtype=np.float64)

    z = np.einsum("gnf,tf->gtn", params.Gamma, features_arr) + params.gamma[:, None, :]
    hidden = relu(z)
    active = relu_derivative(z, at_zero=at_zero)

    for g in range(n_owned):
        col_start = g * block_size
        gamma_start = col_start + n_neurons * n_features
        theta_start = gamma_start + n_neurons
        rho_col = theta_start + n_neurons

        for t in range(n_time):
            row = t * n_owned + g
            gamma_deriv = params.Theta[g] * active[g, t]
            gamma_gamma = gamma_deriv[:, None] * features_arr[t, :][None, :]

            db_dtheta[row, col_start:gamma_start] = gamma_gamma.reshape(-1)
            db_dtheta[row, gamma_start:theta_start] = gamma_deriv
            db_dtheta[row, theta_start:rho_col] = hidden[g, t]
            db_dtheta[row, rho_col] = 1.0

    return db_dtheta

def compute_policy_bids_and_jacobian(
    features: np.ndarray,
    params: PolicyParameters,
    at_zero: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both policy bids and their parameter Jacobian."""
    return (
        compute_policy_bids(features, params),
        compute_policy_jacobian(features, params, at_zero=at_zero),
    )

def finite_difference_check(
    features: np.ndarray,
    params: PolicyParameters,
    eps: float = 1e-6,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """
    Compare the analytical policy Jacobian with a central finite difference.

    Returns a dictionary containing ``max_abs_error`` and ``passed``.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be nonnegative, got {tolerance}")

    features_arr, n_time, n_features = _validate_features_and_params(features, params)
    n_owned, n_neurons, _ = params.Gamma.shape
    theta = flatten_parameters(params)
    analytical = compute_policy_jacobian(features_arr, params)
    numerical = np.empty_like(analytical)

    for col in range(theta.size):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[col] += eps
        theta_minus[col] -= eps

        params_plus = unflatten_parameters(theta_plus, n_owned, n_neurons, n_features)
        params_minus = unflatten_parameters(theta_minus, n_owned, n_neurons, n_features)
        bids_plus = _flatten_time_major(compute_policy_bids(features_arr, params_plus))
        bids_minus = _flatten_time_major(compute_policy_bids(features_arr, params_minus))
        numerical[:, col] = (bids_plus - bids_minus) / (2.0 * eps)

    abs_error = np.abs(analytical - numerical)
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    return {
        "max_abs_error": max_abs_error,
        "passed": bool(max_abs_error <= tolerance),
    }

def _validate_features_and_params(
    features: np.ndarray,
    params: PolicyParameters,
) -> tuple[np.ndarray, int, int]:
    n_owned, _, n_features = _validate_params(params)
    if n_owned <= 0:
        raise ValueError("PolicyParameters must contain at least one owned generator")

    features_arr = np.asarray(features, dtype=np.float64)
    if features_arr.ndim != 2:
        raise ValueError(
            f"features must have shape (n_time, n_features), got {features_arr.shape}"
        )
    if features_arr.shape[1] != n_features:
        raise ValueError(
            "features and Gamma disagree on n_features: "
            f"{features_arr.shape[1]} != {n_features}"
        )
    if features_arr.shape[0] <= 0:
        raise ValueError("features must contain at least one time step")
    _validate_finite(features_arr, "features")
    return features_arr, features_arr.shape[0], features_arr.shape[1]

def _validate_params(params: PolicyParameters) -> tuple[int, int, int]:
    Gamma = np.asarray(params.Gamma, dtype=np.float64)
    gamma = np.asarray(params.gamma, dtype=np.float64)
    Theta = np.asarray(params.Theta, dtype=np.float64)
    rho = np.asarray(params.rho, dtype=np.float64)

    if Gamma.ndim != 3:
        raise ValueError(
            f"Gamma must have shape (n_owned, n_neurons, n_features), got {Gamma.shape}"
        )
    n_owned, n_neurons, n_features = Gamma.shape
    if gamma.shape != (n_owned, n_neurons):
        raise ValueError(f"gamma must have shape {(n_owned, n_neurons)}, got {gamma.shape}")
    if Theta.shape != (n_owned, n_neurons):
        raise ValueError(f"Theta must have shape {(n_owned, n_neurons)}, got {Theta.shape}")
    if rho.shape != (n_owned,):
        raise ValueError(f"rho must have shape {(n_owned,)}, got {rho.shape}")

    _validate_finite(Gamma, "Gamma")
    _validate_finite(gamma, "gamma")
    _validate_finite(Theta, "Theta")
    _validate_finite(rho, "rho")
    return n_owned, n_neurons, n_features

def _validate_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def _flatten_time_major(x: np.ndarray) -> np.ndarray:
    """Flatten shape (n_owned, n_time) as [x[0,0], x[1,0], ..., x[0,1], ...]."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {arr.shape}")
    return arr.T.reshape(-1)

if __name__ == "__main__":
    n_owned = 2
    n_time = 3
    n_features = 4
    n_neurons = 3

    rng = np.random.default_rng(1)
    example_features = rng.normal(size=(n_time, n_features))
    example_params = PolicyParameters(
        Gamma=rng.normal(size=(n_owned, n_neurons, n_features)),
        gamma=rng.normal(size=(n_owned, n_neurons)),
        Theta=rng.normal(size=(n_owned, n_neurons)),
        rho=rng.normal(size=n_owned),
    )

    example_bids, example_jacobian = compute_policy_bids_and_jacobian(
        example_features,
        example_params,
    )
    check = finite_difference_check(example_features, example_params)

    print(f"bids shape: {example_bids.shape}")
    print(f"Jacobian shape: {example_jacobian.shape}")
    print(f"total parameter size: {total_parameter_size(n_owned, n_neurons, n_features)}")
    print(f"finite-difference max error: {check['max_abs_error']:.6e}")
    print(f"finite-difference passed: {check['passed']}")
