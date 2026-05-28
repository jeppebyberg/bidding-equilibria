"""Visualize DRO PoA results per scenario, mirroring poa_bidding_blocks_results.py.

For each result file (one eta value) the script produces a subfolder
    output_dir/{regime_name}/eta_{label}/scenario_k{k}/
containing one figure per physical generator with the same 3-panel layout as
poa_capacity_dispatch_bids in the base PoA visualizer.

Output structure:
    output_dir/
        {regime_name}/
            eta_{label}/
                scenario_k{k}/
                    poa_capacity_dispatch_bids_{generator_name}.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = Path("results_viz/figures/dro_bidding_results")
DEFAULT_DRO_RESULT_DIR = Path("results/dro_poa")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _series(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values])


def _eta_label(eta: float) -> str:
    return f"{float(eta):.8g}".replace("-", "m").replace(".", "p")


def _load_sorted_result_files(regime_dir: Path) -> list[tuple[float, Path]]:
    pairs: list[tuple[float, Path]] = []
    for path in regime_dir.glob("dro_poa_eta_*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            pairs.append((float(payload["eta"]), path))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _find_regime_dirs(dro_result_dir: Path) -> list[Path]:
    return sorted(
        d for d in dro_result_dir.iterdir()
        if d.is_dir() and any(d.glob("dro_poa_eta_*.json"))
    )


# ---------------------------------------------------------------------------
# Per-result visualizer
# ---------------------------------------------------------------------------

class DROBiddingBlocksVisualizer:
    def __init__(self, result: dict[str, Any], output_dir: Path) -> None:
        self.result = result
        self.output_dir = output_dir
        self.eta = float(result["eta"])
        self.regime_name = str(result.get("regime_name", "unknown"))
        self.time = np.arange(int(result["num_time_steps"]))

    @classmethod
    def from_json(cls, path: Path, output_dir: Path) -> "DROBiddingBlocksVisualizer":
        return cls(result=_load_json(path), output_dir=output_dir)

    def _save(self, fig: plt.Figure, subdir: Path, filename: str, show: bool) -> Path:
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_generator_capacity_dispatch_and_bids(
        self,
        scenario: dict[str, Any],
        show: bool = False,
    ) -> list[Path]:
        k = int(scenario["k"])
        eq_price = _series(scenario["equilibrium_price_profile"])
        opt_price = _series(scenario["optimal_price_profile"])
        c_eq = float(scenario.get("C_eq", float("nan")))
        c_opt = float(scenario.get("C_opt", float("nan")))
        poa_diff = float(scenario.get("PoA_difference", float("nan")))
        poa_ratio = float(scenario.get("PoA_ratio", float("nan")))
        w_dist = float(scenario.get("wasserstein_distance", float("nan")))

        scenario_subdir = (
            self.output_dir
            / f"eta_{_eta_label(self.eta)}"
            / f"scenario_k{k}"
        )

        saved: list[Path] = []
        generators: dict[str, Any] = scenario.get("generators", {})

        for gen_name, gen in generators.items():
            fig, axes = plt.subplots(
                3, 1,
                figsize=(11, 8.5),
                sharex=True,
                gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
            )

            fig.suptitle(
                f"{self.regime_name} | eta={self.eta:.4g} | k={k} | {gen_name}\n"
                f"C_eq={c_eq:.2f}  C_opt={c_opt:.2f}  "
                f"PoA diff={poa_diff:.2f}  ratio={poa_ratio:.3f}  "
                f"W dist={w_dist:.2f}",
                fontsize=10,
            )

            # Panel 1: capacity, dispatch
            axes[0].plot(
                self.time,
                _series(gen["optimized_physical_capacity_profile"]),
                color="tab:green",
                marker="s",
                linewidth=2.0,
                label="Optimized capacity",
            )
            axes[0].plot(
                self.time,
                _series(gen["empirical_physical_capacity_profile"]),
                color="tab:green",
                marker="s",
                linewidth=1.4,
                linestyle="--",
                alpha=0.55,
                label="Empirical capacity",
            )
            axes[0].plot(
                self.time,
                _series(gen["equilibrium_physical_dispatch"]),
                color="tab:red",
                marker="o",
                linewidth=2.0,
                label="Eq dispatch",
            )
            axes[0].plot(
                self.time,
                _series(gen["optimal_physical_dispatch"]),
                color="tab:blue",
                marker="^",
                linewidth=1.8,
                linestyle="--",
                label="Opt dispatch",
            )
            axes[0].set_ylabel("MW")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="best", fontsize=8)

            # Panel 2: block capacities
            for block in gen["blocks"]:
                axes[1].plot(
                    self.time,
                    _series(block["capacity_profile"]),
                    marker="s",
                    linewidth=1.7,
                    label=f"{block['block_name']} cap",
                )
            axes[1].set_ylabel("Block capacity (MW)")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(loc="best", fontsize=8, ncol=3)

            # Panel 3: bids + clearing price
            for block in gen["blocks"]:
                axes[2].plot(
                    self.time,
                    _series(block["alpha_profile"]),
                    marker="o",
                    linewidth=1.7,
                    label=f"{block['block_name']} bid",
                )
            axes[2].plot(
                self.time,
                eq_price,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Eq clearing price",
            )
            axes[2].plot(
                self.time,
                opt_price,
                color="0.5",
                linestyle=":",
                linewidth=1.6,
                label="Opt clearing price",
            )
            axes[2].set_xlabel("Time step")
            axes[2].set_ylabel("Bid / price")
            axes[2].grid(True, alpha=0.25)
            axes[2].legend(loc="best", fontsize=8, ncol=3)

            filename = f"poa_capacity_dispatch_bids_{gen_name}.png"
            saved.append(self._save(fig, scenario_subdir, filename, show))

        return saved

    def plot_all_scenarios(self, show: bool = False) -> list[Path]:
        saved: list[Path] = []
        for scenario in self.result.get("scenarios", []):
            saved.extend(
                self.plot_generator_capacity_dispatch_and_bids(scenario, show=show)
            )
        return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plot_dro_bidding_results(
    dro_result_dir: Path = DEFAULT_DRO_RESULT_DIR,
    eta_values: list[float] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> dict[str, list[Path]]:
    """Plot bidding results for every regime and selected eta values."""
    regime_dirs = _find_regime_dirs(dro_result_dir)
    if not regime_dirs:
        raise FileNotFoundError(
            f"No regime subdirectories with DRO results found in {dro_result_dir}"
        )

    all_paths: dict[str, list[Path]] = {}
    for regime_dir in regime_dirs:
        sorted_files = _load_sorted_result_files(regime_dir)
        if not sorted_files:
            continue

        if eta_values is not None:
            # Select the files whose eta is closest to each requested value.
            eta_arr = np.array([e for e, _ in sorted_files])
            selected: list[tuple[float, Path]] = []
            for target in eta_values:
                idx = int(np.argmin(np.abs(eta_arr - target)))
                if sorted_files[idx] not in selected:
                    selected.append(sorted_files[idx])
        else:
            selected = sorted_files

        regime_name = regime_dir.name
        regime_paths: list[Path] = []
        for eta, result_path in selected:
            result_output_dir = output_dir / regime_name
            viz = DROBiddingBlocksVisualizer.from_json(result_path, result_output_dir)
            paths = viz.plot_all_scenarios(show=show)
            regime_paths.extend(paths)
            print(f"  {regime_name} eta={eta:.4g}: {len(paths)} figures")

        all_paths[regime_name] = regime_paths

    return all_paths


def main() -> None:
    # Edit these settings directly when running this script.
    dro_result_dir = DEFAULT_DRO_RESULT_DIR
    # Set to a list of eta values to plot only those (nearest match is selected).
    # Set to None to plot all available eta values.
    eta_values: list[float] | None = [0.0, 0.5, 10000.0]
    output_dir = DEFAULT_OUTPUT_DIR
    show = False

    print(f"Plotting DRO bidding results from {dro_result_dir}")
    results = plot_dro_bidding_results(
        dro_result_dir=dro_result_dir,
        eta_values=eta_values,
        output_dir=output_dir,
        show=show,
    )
    total = sum(len(v) for v in results.values())
    print(f"Done. {total} figures saved across {len(results)} regime(s).")


if __name__ == "__main__":
    main()
