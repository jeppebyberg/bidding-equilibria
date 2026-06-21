# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-14
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Re-generate sensitivity-study figures from existing results -- no Gurobi, no training.

For each study under ``results/sensitivity_studies/`` this runs, when applicable:
  1. the study's per-study summary (timing / pieces / tightening / NN loss), and
  2. the cross-run DRO comparison overlay (eta-sweep + epsilon frontier).

Every stage only *reads* result artifacts, so it is safe to run after a partial
overnight batch -- missing or unfinished runs are simply skipped, and one failing
study never aborts the rest.

Run all studies found on disk:

    .\\.venv\\Scripts\\python.exe -m driver.sensitivity.replot

Run a subset (by study name):

    .\\.venv\\Scripts\\python.exe -m driver.sensitivity.replot horizon_sweep mccormick_pieces_sweep
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from driver.core.visualization_core import plot_sensitivity_comparison_stage  # noqa: E402
from driver.sensitivity.sensitivity_config import RESULT_ROOT  # noqa: E402

# study name -> dotted path of the summary module exposing a ``main()`` entry point.
# Studies absent here (e.g. wind_playing_sweep, dro_*) only get the DRO comparison
# overlay, which self-skips when a study has no DRO eta-sweep results.
SUMMARY_MODULES = {
    "horizon_sweep": "driver.sensitivity.summaries.pipeline_timing_summary",
    "mccormick_pieces_sweep": "driver.sensitivity.summaries.mccormick_pieces_summary",
    "bound_tightening_progression": "driver.sensitivity.summaries.bound_tightening_progression_summary",
    "hidden_layers_sweep": "driver.sensitivity.summaries.hidden_layers_loss_summary",
}


def _guard(label: str, fn, *args, **kwargs) -> None:
    """Run one replot stage, logging and swallowing any error so others proceed."""
    print(f"\n[replot] === {label} ===")
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - one stage must not abort the rest
        print(f"[replot] Skipping '{label}': {exc}")
        traceback.print_exc()


def _run_summary(module_path: str) -> None:
    """Import a per-study summary module and invoke its ``main()``."""
    module = importlib.import_module(module_path)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{module_path} has no main() entry point")
    main()


def available_studies(root: Path) -> list[str]:
    """Study subdirectories that currently have a result tree on disk."""
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def run(studies: list[str] | None = None) -> None:
    root = Path(RESULT_ROOT)
    available = available_studies(root)
    if not available:
        raise SystemExit(f"No sensitivity studies found under {root}")

    selected = studies or available
    unknown = [s for s in selected if s not in available]
    if unknown:
        raise SystemExit(
            f"Unknown study(ies): {unknown}. Available: {', '.join(available)}"
        )

    print(f"[replot] result root: {root}")
    print(f"[replot] studies: {', '.join(selected)}")

    for study in selected:
        summary_module = SUMMARY_MODULES.get(study)
        if summary_module is not None:
            _guard(f"{study} summary", _run_summary, summary_module)
        # DRO cross-run comparison overlay (self-skips if the study has no DRO runs).
        _guard(f"{study} DRO comparison", plot_sensitivity_comparison_stage, study, root)

    print("\n[replot] Done.")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
