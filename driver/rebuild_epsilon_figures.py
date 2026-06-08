"""Rebuild epsilon sweep summary and figures from all saved per-epsilon JSONs.

Run any time during or after the sweep to refresh figures from completed solves:
    .\.venv\Scripts\python.exe driver\rebuild_epsilon_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from driver.run_dro_epsilon_sweep import FIGURES_DIR, OUTPUT_DIR, make_figures

SWEEP_DIR = OUTPUT_DIR / "epsilon_sweep"

def _parse_epsilon_from_stem(stem: str) -> float:
    raw = stem.replace("dro_epsilon_", "").replace("_T8", "").replace("p", ".").replace("m", "-")
    return float(raw)

files = sorted(
    SWEEP_DIR.glob("dro_epsilon_*_T8.json"),
    key=lambda p: _parse_epsilon_from_stem(p.stem),
)

if not files:
    print(f"No result files found in {SWEEP_DIR}")
    sys.exit(0)

summaries = []
for f in files:
    r = json.load(f.open())
    scenarios = r.get("scenarios", [])
    poa_ratios = [
        s["C_eq"] / s["C_opt"]
        for s in scenarios
        if s.get("C_eq") and s.get("C_opt") and s["C_opt"] != 0
    ]
    wasserstein = [s.get("wasserstein_distance") or 0.0 for s in scenarios]
    summaries.append({
        "epsilon": r.get("epsilon", 0.0),
        "objective": r.get("inner_objective"),
        "average_poa_ratio": float(np.mean(poa_ratios)) if poa_ratios else None,
        "per_scenario_poa_ratios": poa_ratios,
        "average_wasserstein": float(np.mean(wasserstein)) if wasserstein else None,
        "per_scenario_wasserstein": wasserstein,
        "solve_wall_time_seconds": r.get("solver", {}).get("wall_time"),
        "termination": r.get("solver", {}).get("termination_condition", ""),
    })
    eps = summaries[-1]["epsilon"]
    poa = summaries[-1]["average_poa_ratio"]
    w   = summaries[-1]["average_wasserstein"]
    print(f"  eps={eps:<8.4g}  avg_PoA={poa:.4f}  avg_W={w:.4f}")

summary_path = OUTPUT_DIR / "epsilon_sweep_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(summaries, summary_path.open("w"), indent=2)
print(f"\nSaved summary: {summary_path}  ({len(summaries)} points)")

make_figures(summaries)
