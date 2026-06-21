# -----------------------------------------------------------------------------
# Conducted by Jeppe Urup Byberg.
# Last modified: 2026-06-15
#
# Part of the MSc thesis on strategic bidding equilibria and worst-case market
# inefficiency (Price-of-Anarchy) in electricity markets.
# -----------------------------------------------------------------------------

"""Swap the freshly regenerated base case into every study's base-equivalent run.

Each sensitivity study contains one run that is physically identical to the base
case (same reference case / horizon / wind behaviour). Those runs are supposed to
mirror ``results/base_case`` but can drift when the base case is regenerated. This
script replaces each base-equivalent run directory with an exact copy of the
current ``results/base_case`` tree.

Two safeguards:
  * stray ``sensitivity_*`` manifests living in base_case/pipeline_manifests are
    NOT propagated (they are leftovers from earlier reuse cycles);
  * each run's own ``sensitivity_<study>_<run>.json`` manifest is preserved (with
    a marker recording the swap) so its study/override provenance survives.

Run:
  .\\.venv\\Scripts\\python.exe -m driver.sensitivity.swap_base_case_into_studies
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

BASE_CASE_ROOT = Path("results/base_case")
STUDY_ROOT = Path("results/sensitivity_studies")

# study dir name -> base-equivalent run dir name.
# blocks_B2 / players_N3 / ramp_R020 / overlapping base carry case=base_test_case;
# horizon T6 is base horizon (=6); wind is the wind-plays-strategically (base) arm.
BASE_EQUIVALENT_RUNS: dict[str, str] = {
    "bidding_blocks_sweep": "blocks_B2",
    "players_sweep": "players_N3",
    "ramp_rate_sweep": "ramp_R020",
    "overlapping_costs_sweep": "base",
    "horizon_sweep": "T6",
    "wind_playing_sweep": "wind",
}


def _ignore_stray_sensitivity_manifests(directory: str, names: list[str]) -> set[str]:
    """copytree ignore: drop sensitivity_* manifests inside pipeline_manifests."""
    if Path(directory).name == "pipeline_manifests":
        return {n for n in names if n.startswith("sensitivity_") and n.endswith(".json")}
    return set()


def _swap_one(study: str, run: str, stamp: str) -> bool:
    run_dir = STUDY_ROOT / study / run
    if not run_dir.exists():
        print(f"  SKIP {study}/{run}: run dir not found")
        return False

    manifest_name = f"sensitivity_{study}_{run}.json"
    own_manifest_path = run_dir / "pipeline_manifests" / manifest_name
    own_manifest: dict | None = None
    if own_manifest_path.exists():
        with own_manifest_path.open("r", encoding="utf-8") as fh:
            own_manifest = json.load(fh)
    else:
        print(f"  WARN {study}/{run}: own manifest {manifest_name} not found; will not restore")

    # Replace the run dir with an exact copy of base_case (minus stray manifests).
    shutil.rmtree(run_dir)
    shutil.copytree(BASE_CASE_ROOT, run_dir, ignore=_ignore_stray_sensitivity_manifests)

    # Restore the run's own manifest, marking the swap.
    if own_manifest is not None:
        own_manifest["reused_base_case"] = True
        own_manifest["base_case_results_swapped_in"] = stamp
        (run_dir / "pipeline_manifests").mkdir(parents=True, exist_ok=True)
        with own_manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(own_manifest, fh, indent=2)

    print(f"  OK   {study}/{run} <= base_case")
    return True


def main() -> None:
    if not (BASE_CASE_ROOT / "poa").exists():
        raise SystemExit(f"Base case not found / incomplete: {BASE_CASE_ROOT}")

    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    print(f"Swapping {BASE_CASE_ROOT} into base-equivalent runs ({stamp})")
    n = 0
    for study, run in BASE_EQUIVALENT_RUNS.items():
        if _swap_one(study, run, stamp):
            n += 1
    print(f"Done: {n}/{len(BASE_EQUIVALENT_RUNS)} runs updated.")


if __name__ == "__main__":
    main()
