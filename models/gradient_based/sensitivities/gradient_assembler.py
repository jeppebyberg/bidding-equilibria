from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kkt_sensitivity import (
    EDSolution,
    EDParameters,
    compute_market_sensitivities,
)
from policy_sensitivity import (
    PolicyParameters,
    compute_policy_bids_and_jacobian,
    flatten_parameters,
    unflatten_parameters,
)
from profit_sensitivity import (
    ProfitParameters,
    compute_profit_sensitivities,
)

@dataclass(frozen=True)
class ScenarioData:
    """Data needed to assemble one scenario's policy-parameter gradient."""

    solution: EDSolution
    ed_params: EDParameters
    features: np.ndarray

@dataclass(frozen=True)
class ScenarioGradientResult:
    """Profit-gradient chain-rule result and intermediate sensitivities."""

    profit: float
    dpi_dP: np.ndarray
    dpi_dlambda: np.ndarray
    dP_dalpha: np.ndarray
    dlambda_dalpha: np.ndarray
    dalpha_dtheta: np.ndarray
    dpi_dalpha: np.ndarray
    dpi_dtheta: np.ndarray
    condition_number: float

@dataclass(frozen=True)
class MultiScenarioGradientResult:
    """Average profit-gradient result across several scenarios."""

    average_profit: float
    gradient: np.ndarray
    gradient_norm: float
    scenario_results: list[ScenarioGradientResult]
    max_condition_number: float
    mean_condition_number: float

def validate_chain_rule_shapes(
    dpi_dP: np.ndarray,
    dP_dalpha: np.ndarray,
    dpi_dlambda: np.ndarray,
    dlambda_dalpha: np.ndarray,
    dalpha_dtheta: np.ndarray,
) -> None:
    """Validate that the alpha and theta chain-rule matrix products conform."""
    dpi_dP_arr = np.asarray(dpi_dP, dtype=np.float64)
    dP_dalpha_arr = np.asarray(dP_dalpha, dtype=np.float64)
    dpi_dlambda_arr = np.asarray(dpi_dlambda, dtype=np.float64)
    dlambda_dalpha_arr = np.asarray(dlambda_dalpha, dtype=np.float64)
    dalpha_dtheta_arr = np.asarray(dalpha_dtheta, dtype=np.float64)

    if dpi_dP_arr.ndim != 1:
        raise ValueError(f"dpi_dP must be one-dimensional, got shape {dpi_dP_arr.shape}")
    if dP_dalpha_arr.ndim != 2:
        raise ValueError(f"dP_dalpha must be two-dimensional, got shape {dP_dalpha_arr.shape}")
    if dpi_dP_arr.shape[0] != dP_dalpha_arr.shape[0]:
        raise ValueError(
            "dpi_dP length must match dP_dalpha rows: "
            f"{dpi_dP_arr.shape[0]} != {dP_dalpha_arr.shape[0]}"
        )

    if dpi_dlambda_arr.ndim != 1:
        raise ValueError(
            f"dpi_dlambda must be one-dimensional, got shape {dpi_dlambda_arr.shape}"
        )
    if dlambda_dalpha_arr.ndim != 2:
        raise ValueError(
            f"dlambda_dalpha must be two-dimensional, got shape {dlambda_dalpha_arr.shape}"
        )
    if dpi_dlambda_arr.shape[0] != dlambda_dalpha_arr.shape[0]:
        raise ValueError(
            "dpi_dlambda length must match dlambda_dalpha rows: "
            f"{dpi_dlambda_arr.shape[0]} != {dlambda_dalpha_arr.shape[0]}"
        )

    if dP_dalpha_arr.shape[1] != dlambda_dalpha_arr.shape[1]:
        raise ValueError(
            "dP_dalpha and dlambda_dalpha must have the same number of alpha columns: "
            f"{dP_dalpha_arr.shape[1]} != {dlambda_dalpha_arr.shape[1]}"
        )
    if dalpha_dtheta_arr.ndim != 2:
        raise ValueError(
            f"dalpha_dtheta must be two-dimensional, got shape {dalpha_dtheta_arr.shape}"
        )
    if dP_dalpha_arr.shape[1] != dalpha_dtheta_arr.shape[0]:
        raise ValueError(
            "dpi_dalpha length must match dalpha_dtheta rows: "
            f"{dP_dalpha_arr.shape[1]} != {dalpha_dtheta_arr.shape[0]}"
        )

    for name, arr in (
        ("dpi_dP", dpi_dP_arr),
        ("dP_dalpha", dP_dalpha_arr),
        ("dpi_dlambda", dpi_dlambda_arr),
        ("dlambda_dalpha", dlambda_dalpha_arr),
        ("dalpha_dtheta", dalpha_dtheta_arr),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")

def compute_scenario_profit_gradient(
    scenario: ScenarioData,
    policy_params: PolicyParameters,
    profit_params: ProfitParameters,
    player_generators: list[int],
    include_beta: bool = False,
    regularization: float = 0.0,
) -> ScenarioGradientResult:
    """
    Assemble one scenario's profit gradient with respect to policy parameters.

    The function delegates policy, KKT, and profit sensitivities to the
    standalone modules and only applies the chain rule. The returned result is
    alpha-only; ``include_beta`` is accepted for API compatibility.
    """
    _, dalpha_dtheta = compute_policy_bids_and_jacobian(
        features=scenario.features,
        params=policy_params,
    )

    market_sens = compute_market_sensitivities(
        solution=scenario.solution,
        params=scenario.ed_params,
        player_generators=player_generators,
        include_beta=include_beta,
        regularization=regularization,
    )

    profit_sens = compute_profit_sensitivities(
        P=scenario.solution.P,
        lambda_=scenario.solution.lambda_,
        profit_params=profit_params,
        player_generators=player_generators,
        flatten_dispatch=True,
    )

    profit = float(profit_sens["profit"])
    dpi_dP = np.asarray(profit_sens["dpi_dP"], dtype=np.float64)
    dpi_dlambda = np.asarray(profit_sens["dpi_dlambda"], dtype=np.float64)
    dP_dalpha = np.asarray(market_sens["dP_dalpha"], dtype=np.float64)
    dlambda_dalpha = np.asarray(market_sens["dlambda_dalpha"], dtype=np.float64)
    dalpha_dtheta = np.asarray(dalpha_dtheta, dtype=np.float64)

    validate_chain_rule_shapes(
        dpi_dP=dpi_dP,
        dP_dalpha=dP_dalpha,
        dpi_dlambda=dpi_dlambda,
        dlambda_dalpha=dlambda_dalpha,
        dalpha_dtheta=dalpha_dtheta,
    )

    dpi_dalpha = dpi_dP @ dP_dalpha + dpi_dlambda @ dlambda_dalpha
    if dpi_dalpha.shape != (dalpha_dtheta.shape[0],):
        raise ValueError(
            "dpi_dalpha must match dalpha_dtheta rows after chain rule: "
            f"{dpi_dalpha.shape} != {(dalpha_dtheta.shape[0],)}"
        )

    dpi_dtheta = dpi_dalpha @ dalpha_dtheta
    if dpi_dtheta.ndim != 1:
        raise ValueError(f"dpi_dtheta must be one-dimensional, got shape {dpi_dtheta.shape}")
    if not np.all(np.isfinite(dpi_dtheta)):
        raise ValueError("dpi_dtheta must contain only finite values")

    return ScenarioGradientResult(
        profit=profit,
        dpi_dP=dpi_dP,
        dpi_dlambda=dpi_dlambda,
        dP_dalpha=dP_dalpha,
        dlambda_dalpha=dlambda_dalpha,
        dalpha_dtheta=dalpha_dtheta,
        dpi_dalpha=dpi_dalpha,
        dpi_dtheta=dpi_dtheta,
        condition_number=float(market_sens["condition_number"]),
    )

def compute_multiscenario_profit_gradient(
    scenarios: list[ScenarioData],
    policy_params: PolicyParameters,
    profit_params: ProfitParameters,
    player_generators: list[int],
    regularization: float = 0.0,
) -> MultiScenarioGradientResult:
    """Compute and average profit gradients over multiple scenarios."""
    if not scenarios:
        raise ValueError("scenarios must not be empty")

    scenario_results = [
        compute_scenario_profit_gradient(
            scenario=scenario,
            policy_params=policy_params,
            profit_params=profit_params,
            player_generators=player_generators,
            regularization=regularization,
        )
        for scenario in scenarios
    ]

    gradients = np.stack([result.dpi_dtheta for result in scenario_results], axis=0)
    profits = np.asarray([result.profit for result in scenario_results], dtype=np.float64)
    condition_numbers = np.asarray(
        [result.condition_number for result in scenario_results],
        dtype=np.float64,
    )

    gradient = np.mean(gradients, axis=0)
    gradient_norm = float(np.linalg.norm(gradient, ord=2))

    return MultiScenarioGradientResult(
        average_profit=float(np.mean(profits)),
        gradient=gradient,
        gradient_norm=gradient_norm,
        scenario_results=scenario_results,
        max_condition_number=float(np.max(condition_numbers)),
        mean_condition_number=float(np.mean(condition_numbers)),
    )

def gradient_as_parameter_dict(
    gradient: np.ndarray,
    policy_params: PolicyParameters,
) -> PolicyParameters:
    """Reshape a flat gradient vector into the same structure as policy parameters."""
    n_owned, n_neurons, n_features = _policy_dimensions(policy_params)
    return unflatten_parameters(
        theta=gradient,
        n_owned=n_owned,
        n_neurons=n_neurons,
        n_features=n_features,
    )

def apply_gradient_ascent_step(
    policy_params: PolicyParameters,
    gradient: np.ndarray,
    learning_rate: float,
) -> PolicyParameters:
    """Apply one gradient-ascent update to flattened policy parameters."""
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    theta = flatten_parameters(policy_params)
    gradient_arr = np.asarray(gradient, dtype=np.float64)
    if gradient_arr.shape != theta.shape:
        raise ValueError(
            f"gradient must have the same shape as theta: {gradient_arr.shape} != {theta.shape}"
        )
    if not np.all(np.isfinite(gradient_arr)):
        raise ValueError("gradient must contain only finite values")

    theta_new = theta + float(learning_rate) * gradient_arr
    n_owned, n_neurons, n_features = _policy_dimensions(policy_params)
    return unflatten_parameters(theta_new, n_owned, n_neurons, n_features)

def clip_gradient(
    gradient: np.ndarray,
    max_norm: float | None = None,
) -> np.ndarray:
    """Return a copy of gradient clipped to the requested Euclidean norm."""
    gradient_arr = np.asarray(gradient, dtype=np.float64)
    if not np.all(np.isfinite(gradient_arr)):
        raise ValueError("gradient must contain only finite values")
    if max_norm is None:
        return gradient_arr.copy()
    if max_norm < 0.0:
        raise ValueError(f"max_norm must be nonnegative, got {max_norm}")

    norm = float(np.linalg.norm(gradient_arr, ord=2))
    if norm > max_norm and norm > 0.0:
        return gradient_arr * (float(max_norm) / norm)
    return gradient_arr.copy()

def _policy_dimensions(policy_params: PolicyParameters) -> tuple[int, int, int]:
    Gamma = np.asarray(policy_params.Gamma, dtype=np.float64)
    if Gamma.ndim != 3:
        raise ValueError(
            f"policy_params.Gamma must have shape (n_owned, n_neurons, n_features), got {Gamma.shape}"
        )
    return int(Gamma.shape[0]), int(Gamma.shape[1]), int(Gamma.shape[2])

if __name__ == "__main__":
    n_gen = 3
    n_time = 3
    n_owned = 1
    n_features = 4
    n_neurons = 2
    n_scenarios = 2
    player_generators = [0]

    policy_params = PolicyParameters(
        Gamma=np.array([[[0.10, -0.05, 0.02, 0.03], [0.04, 0.06, -0.01, 0.08]]]),
        gamma=np.array([[0.01, -0.02]]),
        Theta=np.array([[1.2, -0.7]]),
        rho=np.array([20.0]),
    )
    profit_params = ProfitParameters(
        c_linear=np.array([15.0, 18.0, 22.0]),
        c_quadratic=np.array([0.02, 0.025, 0.03]),
    )

    scenarios: list[ScenarioData] = []
    for s in range(n_scenarios):
        P = np.array(
            [
                [30.0 + s, 32.0 + s, 34.0 + s],
                [40.0, 38.0, 36.0],
                [20.0, 22.0, 24.0],
            ],
            dtype=np.float64,
        )
        solution = EDSolution(
            P=P,
            lambda_=np.array([45.0 + s, 47.0 + s, 49.0 + s], dtype=np.float64),
            mu_max=np.zeros((n_gen, n_time), dtype=np.float64),
            mu_min=np.zeros((n_gen, n_time), dtype=np.float64),
            mu_up=np.zeros((n_gen, n_time), dtype=np.float64),
            mu_down=np.zeros((n_gen, n_time), dtype=np.float64),
        )
        ed_params = EDParameters(
            alpha=np.array(
                [
                    [20.0 + s, 20.0 + s, 20.0 + s],
                    [24.0, 24.0, 24.0],
                    [28.0, 28.0, 28.0],
                ],
                dtype=np.float64,
            ),
            beta=np.full((n_gen, n_time), 0.1, dtype=np.float64),
            demand=np.sum(P, axis=0),
            pmax=np.full((n_gen, n_time), 100.0, dtype=np.float64),
            pmin=np.zeros((n_gen, n_time), dtype=np.float64),
            ramp_up=np.full(n_gen, 100.0, dtype=np.float64),
            ramp_down=np.full(n_gen, 100.0, dtype=np.float64),
            p_initial=np.array([30.0, 40.0, 20.0], dtype=np.float64),
        )
        features = np.array(
            [
                [1.0, -0.5 + 0.1 * s, 0.2, 0.0],
                [1.0, 0.0 + 0.1 * s, 0.4, 0.1],
                [1.0, 0.5 + 0.1 * s, 0.6, 0.2],
            ],
            dtype=np.float64,
        )
        scenarios.append(
            ScenarioData(solution=solution, ed_params=ed_params, features=features)
        )

    result = compute_multiscenario_profit_gradient(
        scenarios=scenarios,
        policy_params=policy_params,
        profit_params=profit_params,
        player_generators=player_generators,
        regularization=1e-8,
    )

    print(f"average_profit: {result.average_profit:.6f}")
    print(f"gradient shape: {result.gradient.shape}")
    print(f"gradient norm: {result.gradient_norm:.6e}")
    print(f"number of scenario results: {len(result.scenario_results)}")
    print(f"max condition number: {result.max_condition_number:.6e}")
    print(f"mean condition number: {result.mean_condition_number:.6e}")

    updated_policy_params = apply_gradient_ascent_step(
        policy_params,
        result.gradient,
        learning_rate=1e-3,
    )
    print(f"updated theta shape: {flatten_parameters(updated_policy_params).shape}")
