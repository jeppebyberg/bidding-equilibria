"""Scan DRO final tightening reports to quantify how fix #2 (the dual Big-M floor)
would alter each run, and to inspect the distribution of tightened dual values."""
import json
from pathlib import Path

DEFAULT = 1e6
TAU = 1e-3
FLOOR = TAU * DEFAULT

ROOT = Path("results/sensitivity_studies/hidden_layers_sweep")
runs = ["hidden_4_4", "hidden_8_4", "hidden_4_8", "hidden_8_8", "hidden_4_4_4"]

DUALS = [
    "mu_upper_eq", "mu_lower_eq", "mu_ramp_up_eq", "mu_ramp_down_eq",
    "mu_upper_opt", "mu_lower_opt", "mu_ramp_up_opt", "mu_ramp_down_opt",
]

for run in runs:
    p = ROOT / run / "dro/tightening/poa_worst_case/final_tightening_report.json"
    if not p.exists():
        print(f"{run}: (no report)")
        continue
    rep = json.load(p.open())
    tbm = rep.get("tight_big_m", {}) or {}
    print(f"\n===== {run} =====")
    for dual in DUALS:
        entries = tbm.get(dual, {}) or {}
        if not entries:
            continue
        n = len(entries)
        # classify
        reverted = []      # not slack-certified AND value < FLOOR  -> fix #2 changes these
        slack0 = 0         # slack-certified (kept at 0)
        kept = []          # value >= FLOOR (kept as-is)
        vals = []
        for k, v in entries.items():
            if not isinstance(v, dict):
                continue
            val = float(v.get("tight_big_m", DEFAULT))
            fbs = bool(v.get("fixed_by_slack", False))
            vals.append(val)
            if fbs:
                slack0 += 1
            elif val < FLOOR:
                reverted.append((k, val))
            else:
                kept.append((k, val))
        if vals:
            nonslack = [x for x in vals]
            mx = max(vals)
            # distribution of non-slack values
            print(f"  {dual:16s} n={n:3d}  slack0={slack0:3d}  "
                  f"reverted(<{FLOOR:.0f})={len(reverted):3d}  kept(>= floor)={len(kept):3d}  "
                  f"max_val={mx:.3g}")
            if kept:
                kv = sorted(x[1] for x in kept)
                print(f"      kept value range: [{kv[0]:.3g}, {kv[-1]:.3g}]")
