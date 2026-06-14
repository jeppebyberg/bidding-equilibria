"""Decompose the achieved DRO transport radius into demand vs wind components.

Along the eta-sweep frontier each solved DRO state moves the empirical scenarios
to an adversarial distribution. The realized transport cost (the achieved
Wasserstein radius eps) is, per the model's
``wasserstein_distance_definition`` constraint, an L1 sum of two physically
distinct displacements:

    eps(k) = sum_t |D_opt[k,t]      - D_emp[k,t]|              <- demand transport
           + sum_{i,t} |Pmax_opt[k,i,t] - Pmax_emp[k,i,t]|     <- capacity transport

(see ``D_transport_abs_deviation`` and ``P_max_phys_transport_abs_deviation`` in
models/DRO_PoA/DRO_PoA_optimization.py). Conventional capacities are fixed across
scenarios, so their transport term is ~0 and the capacity component collapses to
the WIND-displacement component. We therefore split the achieved radius into:

    eps_demand = mean_k sum_t |D_opt - D_emp|
    eps_wind   = mean_k sum_{wind i, t} |Pmax_opt - Pmax_emp|
    eps_conv   = mean_k sum_{conv i, t} |Pmax_opt - Pmax_emp|   (sanity ~ 0)

and plot eps_demand and eps_wind against the worst-case expected PoA achieved at
each eta. This turns "wind gives structural deviation" into a defensible
two-lever mechanism: the demand lever is expected to saturate early (matching the
no-wind shelf, where only demand can be displaced) while the wind lever is what
carries PoA the rest of the way up.

This script re-reads already-solved DRO result JSONs only -- no Gurobi, no
re-solve. It reuses ``load_eta_sweep_records`` for the same per-epsilon dedup /
sorting the frontier plot uses, then reopens each record's ``result_path`` to
recover the per-scenario profiles needed for the split.

Edit the paths in ``main()`` and run:
  .\\.venv\\Scripts\\python.exe -m results_viz.dro_epsilon_decomposition_vs_poa
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_viz.plot_dro_poa_eta_sweep import (  # noqa: E402
    discover_regime_names,
    load_eta_sweep_records,
)


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _l1(a: list[float], b: list[float]) -> float:
    """Sum_t |a[t] - b[t]| over the overlapping prefix (defensive on length)."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = min(arr_a.size, arr_b.size)
    return float(np.nansum(np.abs(arr_a[:n] - arr_b[:n])))


@dataclass
class EpsilonSplit:
    """Achieved transport radius split into demand / wind / conventional parts."""

    eps_total: float       # mean achieved Wasserstein radius (sanity: ~ wass field)
    eps_demand: float      # demand-displacement component
    eps_wind: float        # wind-capacity-displacement component
    eps_conv: float        # conventional-capacity component (should be ~ 0)
    reported_total: float  # average_wasserstein_distance straight from the JSON


def decompose_epsilon(result: dict[str, Any]) -> EpsilonSplit:
    """Average the per-scenario demand/wind/conv transport over all scenarios."""
    scenarios = result.get("scenarios", []) or []
    if not scenarios:
        raise ValueError("DRO result has no scenarios; cannot decompose epsilon.")

    # is_wind is recorded per generator inside each scenario; take it from the
    # first scenario (it is a static system property).
    first_gens = scenarios[0]["generators"]
    wind_names = {name for name, g in first_gens.items() if bool(g["is_wind"])}

    demand_terms: list[float] = []
    wind_terms: list[float] = []
    conv_terms: list[float] = []
    total_terms: list[float] = []

    for scenario in scenarios:
        d_dev = _l1(
            scenario["optimized_demand_profile"],
            scenario["empirical_demand_profile"],
        )
        emp_caps = scenario["empirical_physical_capacity_profiles"]
        opt_caps = scenario["optimized_physical_capacity_profiles"]

        wind_dev = 0.0
        conv_dev = 0.0
        for name in emp_caps:
            dev = _l1(opt_caps[name], emp_caps[name])
            if name in wind_names:
                wind_dev += dev
            else:
                conv_dev += dev

        demand_terms.append(d_dev)
        wind_terms.append(wind_dev)
        conv_terms.append(conv_dev)
        total_terms.append(d_dev + wind_dev + conv_dev)

    reported = result.get("average_wasserstein_distance")
    return EpsilonSplit(
        eps_total=float(np.mean(total_terms)),
        eps_demand=float(np.mean(demand_terms)),
        eps_wind=float(np.mean(wind_terms)),
        eps_conv=float(np.mean(conv_terms)),
        reported_total=float(reported) if reported is not None else float("nan"),
    )


@dataclass
class DecompositionPoint:
    eta: float
    poa: float
    split: EpsilonSplit


def build_decomposition_points(
    results_dir: Path,
    regime_name: str,
    poa_metric: str = "worst_case_expected_poa",
    include_archives: bool = True,
) -> list[DecompositionPoint]:
    """One decomposition point per eta, carrying the PoA metric and the eps split."""
    records = load_eta_sweep_records(results_dir, regime_name, include_archives)
    points: list[DecompositionPoint] = []
    for record in records:
        poa = record.get(poa_metric)
        if poa is None:
            continue
        result = _load_json(Path(record["result_path"]))
        split = decompose_epsilon(result)
        points.append(
            DecompositionPoint(eta=float(record["eta"]), poa=float(poa), split=split)
        )
    # Order along the curve by ascending PoA (the frontier sweeps PoA upward as
    # the achieved radius grows / eta shrinks).
    points.sort(key=lambda p: p.poa)
    return points


def plot_epsilon_decomposition_vs_poa(
    results_dir: Path,
    regime_name: str,
    output_dir: Path,
    poa_metric: str = "worst_case_expected_poa",
    no_wind_shelf_poa: float | None = None,
    show: bool = False,
) -> Path:
    points = build_decomposition_points(results_dir, regime_name, poa_metric)
    if not points:
        raise ValueError(
            f"No decomposition points for regime '{regime_name}' under {results_dir}."
        )

    poa = np.array([p.poa for p in points])
    eps_demand = np.array([p.split.eps_demand for p in points])
    eps_wind = np.array([p.split.eps_wind for p in points])
    eps_conv = np.array([p.split.eps_conv for p in points])
    eps_total = np.array([p.split.eps_total for p in points])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)

    # Top: the two levers as lines against PoA.
    ax = axes[0]
    ax.plot(poa, eps_demand, "o-", color="tab:blue", linewidth=2.2,
            markersize=5.5, label="demand displacement")
    ax.plot(poa, eps_wind, "s-", color="tab:green", linewidth=2.2,
            markersize=5.5, label="wind displacement")
    if np.nanmax(np.abs(eps_conv)) > 1e-6:
        ax.plot(poa, eps_conv, "^:", color="tab:gray", linewidth=1.4,
                markersize=4.5, label="conventional displacement")
    ax.plot(poa, eps_total, "--", color="black", linewidth=1.2, alpha=0.6,
            label="total achieved radius")
    if no_wind_shelf_poa is not None:
        ax.axvline(no_wind_shelf_poa, color="tab:red", linestyle=":", linewidth=1.6,
                   label=f"no-wind shelf PoA = {no_wind_shelf_poa:.3g}")
    ax.set_ylabel("achieved transport radius  (L1)")
    ax.set_title(
        f"Achieved Wasserstein radius decomposition vs PoA -- '{regime_name}'\n"
        "demand lever vs wind lever along the eta-sweep frontier"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Bottom: stacked share so saturation of the demand lever is unmistakable.
    ax = axes[1]
    ax.stackplot(
        poa, eps_demand, eps_wind, eps_conv,
        labels=["demand", "wind", "conventional"],
        colors=["tab:blue", "tab:green", "tab:gray"], alpha=0.75,
    )
    if no_wind_shelf_poa is not None:
        ax.axvline(no_wind_shelf_poa, color="tab:red", linestyle=":", linewidth=1.6)
    ax.set_xlabel(poa_metric.replace("_", " "))
    ax.set_ylabel("achieved transport radius  (stacked)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_path = output_dir / f"{regime_name}_epsilon_decomposition_vs_poa.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # Console summary: confirm the split reconstructs the reported radius and
    # report where the demand lever saturates.
    print(f"[decomp] regime '{regime_name}': {len(points)} eta points")
    for p in points:
        resid = p.split.eps_total - p.split.reported_total
        print(
            f"  PoA={p.poa:.4f}  eta={p.eta:<10.5g}  "
            f"demand={p.split.eps_demand:8.3f}  wind={p.split.eps_wind:8.3f}  "
            f"conv={p.split.eps_conv:7.3f}  total={p.split.eps_total:8.3f}  "
            f"(reported={p.split.reported_total:8.3f}, resid={resid:+.2e})"
        )
    print(f"[saved] {out_path}")
    return out_path


def main() -> None:
    # Point this at a WIND-PLAYING DRO sweep to see the two-lever story. Defaults
    # to the base case so the script runs out of the box; swap in the wind run
    # (e.g. results/sensitivity_studies/wind_playing_sweep/wind/dro) once solved.
    results_dir = Path("results/base_case/dro")
    regime_name: str | None = None  # None -> first discovered regime
    poa_metric = "worst_case_expected_poa"
    # PoA where the demand-only (no-wind) frontier tops out; draw as a reference
    # line if known, otherwise leave None.
    no_wind_shelf_poa: float | None = None
    show = False

    if regime_name is None:
        names = discover_regime_names(results_dir)
        if not names:
            raise SystemExit(f"No DRO regimes with eta-sweep results under {results_dir}")
        regime_name = names[0]
        print(f"[decomp] regime not set; using discovered regime '{regime_name}'")

    output_dir = Path("results_viz/figures/dro_epsilon_decomposition") / regime_name
    plot_epsilon_decomposition_vs_poa(
        results_dir=results_dir,
        regime_name=regime_name,
        output_dir=output_dir,
        poa_metric=poa_metric,
        no_wind_shelf_poa=no_wind_shelf_poa,
        show=show,
    )


if __name__ == "__main__":
    main()
