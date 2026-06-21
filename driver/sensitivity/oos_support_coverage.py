# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-16
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

r"""Out-of-sample support-set coverage for the base case, reported per state variable.

For the base-case PoA result we draw N fresh (out-of-sample) trajectories from the
selected regime and count, for each state process D, W_1, W_2, W_3, the fraction that
lie inside the support set U(r).  Membership is checked against the *stored* base-case
support-set bounds, so the numbers match the model exactly:

    level box :  lower[t] <= x[t] <= upper[t]                         for all t
    AR(1) tube:  |x[0]   - ref[0]|                       <= h         (cold start)
                 |(x[t]-ref[t]) - rho*(x[t-1]-ref[t-1])| <= h         for t >= 1

The ell-1 budget constraint is intentionally excluded here.

where ref = reference[t], rho = rho_D / rho_W (fixed_parameters), and the tube
half-width h = upper[0] - reference[0] (scale(0) = 1, so this equals kappa * scale * sigma).

The fresh draws are clipped to [0, cap] for wind in the simulator, matching the
physical bounds in the model, so physical feasibility is automatic.

Run:
    .\.venv\Scripts\python.exe driver\sensitivity\oos_support_coverage.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thesis figure output: vector PDF + high-DPI PNG (results_viz/_thesis_style.py)
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(next((p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()), _pl.Path(__file__).resolve().parents[0])))  # noqa: E402
import results_viz._thesis_style  # noqa: E402,F401
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.scenarios.scenario_generator import RegimeParameters, ScenarioManager

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RESULT_PATH = PROJECT_ROOT / "results" / "base_case" / "poa" / "poa_optimization_T6_piecewise_mccormick.json"
LATEX_PATH = PROJECT_ROOT / "results" / "base_case" / "LaTeX" / "support_oos_table.txt"
PLOT_DIR = PROJECT_ROOT / "results" / "base_case" / "figures" / "support_oos"
LEVEL_PLOT_PATH = PLOT_DIR / "support_oos_level_box.png"
TUBE_PLOT_PATH = PLOT_DIR / "support_oos_ar1_tube.png"
INNOV_PLOT_PATH = PLOT_DIR / "support_oos_innovations.png"
N_DRAWS = 2000  # trajectories used for the coverage counts
N_PLOT = 300  # number of trajectories actually drawn in the visualizations
SEED = 12345
TOL = 1e-9


@dataclass
class Process:
    """One state process (D or a wind generator) with its stored support-set bounds."""

    name: str
    reference: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    rho: float
    sigma_innov: float  # innovation sigma: D_ref*sigma_D (demand) or cap_i*sigma_W (wind)
    trajs: list[np.ndarray]

    @property
    def tex(self) -> str:
        return "$D$" if self.name == "D" else f"$W_{self.name[2:]}$"

    @property
    def half_width(self) -> float:
        """AR(1) tube half-width = kappa * sigma_innov * scale(0), with scale(0)=1."""
        return float(self.upper[0] - self.reference[0])

    def whitened_innovations(self, traj: np.ndarray) -> np.ndarray:
        """xi_t for one trajectory: cold-start dev at t=0, AR(1) residual for t>=1."""
        dev = traj - self.reference
        xi = dev.copy()
        if len(traj) > 1:
            xi[1:] = dev[1:] - self.rho * dev[:-1]
        return xi

    def inside(self, traj: np.ndarray) -> tuple[bool, bool]:
        """Return (level_ok, tube_ok) for one trajectory vs stored bounds (no budget)."""
        level_ok = bool(np.all(traj >= self.lower - TOL) and np.all(traj <= self.upper + TOL))
        xi = self.whitened_innovations(traj)
        tube_ok = bool(np.all(np.abs(xi) <= self.half_width + TOL))
        return level_ok, tube_ok


def main() -> None:
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    amb = results["ambiguity_set"]
    fixed = amb["fixed_parameters"]
    regime = amb["selected_regime"]
    T = int(results["num_time_steps"])
    reference_case = str(results.get("reference_case", "base_test_case"))

    rho_D = float(fixed["rho_D"])
    rho_W = float(fixed["rho_W"])

    # Stored bounds per process.
    demand_block = amb["demand"]
    wind_block = amb["wind"]
    # Order wind generators W_1, W_2, W_3 (ascending), independent of JSON order.
    wind_names = sorted(wind_block.keys(), key=lambda n: int(n[1:]))

    def _arr(block: dict, key: str) -> np.ndarray:
        return np.asarray(block[key], dtype=float)

    # --- Draw N fresh out-of-sample trajectories from the selected regime. ---
    manager = ScenarioManager(base_case_reference=reference_case)
    manager.base_case["time_steps"] = T
    regime_params = RegimeParameters(
        name="base_case_selected",
        n_scenarios=N_DRAWS,
        mu_D=float(regime["mu_D"]),
        rho_D=rho_D,
        sigma_D=float(regime["sigma_D"]),
        mu_W=float(regime["mu_W"]),
        peak_W=float(fixed["tau_W"]),
        rho_W=rho_W,
        sigma_W=float(regime["sigma_W"]),
    )
    rng = np.random.default_rng(SEED)

    demand_trajs: list[np.ndarray] = []
    wind_trajs: dict[str, list[np.ndarray]] = {g: [] for g in wind_names}
    for _ in range(N_DRAWS):
        d, w, _ = manager._simulate_single_scenario(regime=regime_params, rng=rng)
        demand_trajs.append(np.asarray(d, dtype=float))
        for g in wind_names:
            wind_trajs[g].append(np.asarray(w[g], dtype=float))

    # --- Membership per process. ---
    # Innovation sigma per process: demand D_ref*sigma_D, wind cap_i*sigma_W.
    D_ref = float(fixed["D_ref"])
    sigma_innov_D = D_ref * float(regime["sigma_D"])
    sigma_W = float(regime["sigma_W"])

    processes: list[Process] = []
    processes.append(Process(
        name="D",
        reference=_arr(demand_block, "reference"),
        lower=_arr(demand_block, "lower"),
        upper=_arr(demand_block, "upper"),
        rho=rho_D,
        sigma_innov=sigma_innov_D,
        trajs=demand_trajs,
    ))
    for k, g in enumerate(wind_names, start=1):
        blk = wind_block[g]
        processes.append(Process(
            name=f"W_{k}",
            reference=_arr(blk, "reference"),
            lower=_arr(blk, "lower"),
            upper=_arr(blk, "upper"),
            rho=rho_W,
            sigma_innov=float(blk["installed_capacity"]) * sigma_W,
            trajs=wind_trajs[g],
        ))

    print(f"Base case OOS support-set coverage  (N={N_DRAWS}, seed={SEED}, T={T})")
    print(f"  regime: mu_D={regime['mu_D']:.4f}, sigma_D={regime['sigma_D']:.4f}, "
          f"mu_W={regime['mu_W']:.4f}, sigma_W={regime['sigma_W']:.4f}, "
          f"rho_D={rho_D}, rho_W={rho_W}")
    print()
    header = f"{'process':>8} | {'level box':>10} | {'AR(1) tube':>11} | {'support set':>12}"
    print(header)
    print("-" * len(header))

    def _pct(x: float) -> str:
        return f"{100.0 * x:7.2f}%"

    rows: list[tuple[str, float, float, float]] = []
    # Full-sample membership arrays (over all N_DRAWS), keyed by process name.
    membership: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for proc in processes:
        level = np.empty(len(proc.trajs), dtype=bool)
        tube = np.empty(len(proc.trajs), dtype=bool)
        for i, traj in enumerate(proc.trajs):
            level[i], tube[i] = proc.inside(traj)
        membership[proc.name] = (level, tube)
        joint = level & tube
        rows.append((proc.name, float(level.mean()), float(tube.mean()), float(joint.mean())))
        print(f"{proc.name:>8} | {_pct(rows[-1][1]):>10} | {_pct(rows[-1][2]):>11} | "
              f"{_pct(rows[-1][3]):>12}")

    _write_latex_table(rows)
    print(f"\nWrote LaTeX table: {LATEX_PATH}")

    _plot_oos(processes, membership, constraint="level", output_path=LEVEL_PLOT_PATH,
              show_suptitle=False)
    print(f"Wrote level-box plot: {LEVEL_PLOT_PATH}")
    _plot_oos(processes, membership, constraint="tube", output_path=TUBE_PLOT_PATH)
    print(f"Wrote AR(1) tube plot: {TUBE_PLOT_PATH}")
    _plot_innovations(processes, membership)
    print(f"Wrote innovations plot: {INNOV_PLOT_PATH}")


def _plot_oos(
    processes: list[Process],
    membership: dict[str, tuple[np.ndarray, np.ndarray]],
    constraint: str,
    output_path: Path,
    show_suptitle: bool = True,
) -> None:
    """Plot the first N_PLOT fresh trajectories per state, colored inside/outside one constraint.

    constraint = "level" shows only the level box; "tube" shows only the AR(1) tube.
    Trajectory coloring uses the first N_PLOT draws; the title count is over all N_DRAWS.
    ``show_suptitle=False`` drops the overall figure title (per-panel titles are kept).
    """
    band_color = "tab:blue" if constraint == "level" else "tab:orange"
    band_label = "Level box" if constraint == "level" else "AR(1) tube"
    idx = 0 if constraint == "level" else 1  # which membership array
    fig, axes = plt.subplots(len(processes), 1, figsize=(10, 2.9 * len(processes)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, proc in zip(axes, processes):
        t = np.arange(len(proc.reference))
        inside_all = membership[proc.name][idx]

        if constraint == "level":
            ax.fill_between(t, proc.lower, proc.upper, color=band_color, alpha=0.18, label=band_label)
        else:
            ax.fill_between(t, proc.reference - proc.half_width, proc.reference + proc.half_width,
                            color=band_color, alpha=0.28, label=band_label)
        ax.plot(t, proc.reference, color="black", lw=1.8, ls="--", label="Reference")

        for traj, inside in zip(proc.trajs[:N_PLOT], inside_all[:N_PLOT]):
            ax.plot(t, traj, color=("tab:green" if inside else "tab:red"),
                    alpha=(0.18 if inside else 0.55), lw=0.7)

        n_total = len(inside_all)
        n_in = int(inside_all.sum())
        ax.set_title(
            f"{proc.tex}   inside {band_label}: {n_in}/{n_total} "
            f"({100.0 * n_in / n_total:.1f}%)   [showing {min(N_PLOT, n_total)} draws]",
            fontsize=10,
        )
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=7, ncol=2)

    axes[-1].set_xlabel("Time step t")
    if show_suptitle:
        fig.suptitle(
            f"Out-of-sample trajectories vs {band_label} (base_case)\n"
            f"count over {N_DRAWS} draws, {N_PLOT} shown; green = inside, red = outside",
            fontsize=11,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_innovations(
    processes: list[Process],
    membership: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Whitened AR(1) innovations per state: series (left) + pooled histogram (right).

    Mirrors the existing fresh_scenario_innovations figure: xi_t should look like
    N(0, sigma_innov^2) and stay within +/- half_width (the AR(1) tube).  The series
    panel shows the first N_PLOT draws; counts and the histogram use all N_DRAWS.
    """
    from scipy.stats import norm as _norm_dist

    n = len(processes)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.9 * n),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    axes = np.atleast_2d(axes)

    for row, proc in enumerate(processes):
        ax_series, ax_hist = axes[row, 0], axes[row, 1]
        T = len(proc.reference)
        t = np.arange(T)
        bound = proc.half_width
        tube_in = membership[proc.name][1]  # AR(1) tube membership over all draws

        xi_all = np.stack([proc.whitened_innovations(traj) for traj in proc.trajs])  # (N, T)

        # Series: first N_PLOT trajectories, colored by tube containment.
        for xi, inside in zip(xi_all[:N_PLOT], tube_in[:N_PLOT]):
            ax_series.plot(t, xi, color=("tab:green" if inside else "tab:red"),
                           alpha=(0.12 if inside else 0.5), lw=0.7)
        ax_series.axhline(bound, color="tab:orange", lw=1.6, ls="--",
                          label=f"+/- half-width = {bound:.2f}")
        ax_series.axhline(-bound, color="tab:orange", lw=1.6, ls="--")
        ax_series.axhline(0.0, color="gray", lw=0.8, ls=":")
        n_in = int(tube_in.sum())
        ax_series.set_title(
            f"{proc.tex}   inside tube: {n_in}/{len(tube_in)} "
            f"({100.0 * n_in / len(tube_in):.1f}%)   [showing {min(N_PLOT, len(tube_in))}]",
            fontsize=10,
        )
        ax_series.set_ylabel(r"$\xi_t$ (MW)")
        ax_series.set_xlabel("t")
        ax_series.legend(fontsize=7, loc="upper right")
        ax_series.grid(True, alpha=0.2)

        # Histogram: all innovations pooled, with N(0, sigma_innov^2) overlay.
        flat = xi_all.flatten()
        ax_hist.hist(flat, bins=50, density=True, alpha=0.65, color="tab:blue",
                     label=r"$\xi_t$ (all draws)")
        grid = np.linspace(flat.min(), flat.max(), 300)
        ax_hist.plot(grid, _norm_dist.pdf(grid, 0.0, proc.sigma_innov), "r-", lw=2.0,
                     label=f"N(0, {proc.sigma_innov:.2f}$^2$)")
        ax_hist.axvline(bound, color="tab:orange", ls="--", lw=1.4,
                        label=f"+/- half-width = {bound:.2f}")
        ax_hist.axvline(-bound, color="tab:orange", ls="--", lw=1.4)
        ax_hist.set_xlabel(r"$\xi_t$ (MW)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=7, loc="best")
        ax_hist.grid(True, alpha=0.2)

    fig.tight_layout()
    INNOV_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(INNOV_PLOT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_latex_table(rows: list[tuple[str, float, float, float]]) -> None:
    """Write a booktabs coverage table matching the base_case LaTeX style."""
    def _tex_name(name: str) -> str:
        return f"$D$" if name == "D" else f"$W_{name[2:]}$"

    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \begin{tabular}{lrr}",
        r"        \toprule",
        r"        State & Level box (\%) & AR(1) tube (\%) \\",
        r"        \midrule",
    ]
    for name, level, tube, _joint in rows:
        lines.append(
            f"        {_tex_name(name)} & {100.0 * level:.2f} & {100.0 * tube:.2f} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \caption{Out-of-sample support-set coverage by state variable (base\_case): "
        r"fraction of fresh trajectories inside the level box and inside the AR(1) tube.}",
        r"    \label{tab:support_oos_base_case}",
        r"\end{table}",
        "",
    ]
    LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEX_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
