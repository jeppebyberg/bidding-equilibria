# CLAUDE.md

## Overview

Research codebase for a thesis on strategic bidding equilibria and worst-case
market inefficiency in electricity markets. The project quantifies the
Price-of-Anarchy (PoA) via a tri-level model: an upper level maximizes market
inefficiency over market states, a middle level approximates Nash equilibrium
bids using learned per-generator neural-network policies, and a lower level
clears an intertemporal economic dispatch. Learned ReLU policies are embedded
as MILP constraints in a Pyomo/Gurobi model that is solved to global optimality.
A distributionally robust (DRO) variant also exists but is not the default pipeline.

---

## Repository structure

```
bidding-equilibria/
├── driver/                     # Entry-point scripts
├── config/                     # YAML configs and scenario generation
│   ├── scenarios/              # ScenarioManager (regime-based stochastic draws)
│   ├── reference_cases.yaml    # Physical system: generators, costs, capacities
│   ├── regime_definitions.yaml # Demand/wind regime parameters
│   └── ambiguity_set_config.yaml  # DRO Wasserstein ambiguity set bounds
├── models/
│   ├── helper.py               # Shared block-structure, profile, and ramp helpers
│   ├── PoA/                    # Main PoA optimization + 6-stage tightening
│   │   ├── PoA_optimization.py
│   │   ├── poa_model/          # Mixin submodules (nn_policy_embedding, results, support_set, tightening_reports)
│   │   └── PoA_tightening/     # compute_primal_big_m, relu_bounds, alpha_bounds,
│   │                           #   slack_binary_fix, dual_big_m, optimal_cost_bounds,
│   │                           #   tightening_main
│   ├── DRO_PoA/                # Experimental scenario-indexed DRO variant
│   ├── neural_network/
│   │   ├── features/           # NeuralNetworkFeatureBuilder, build_features.py
│   │   └── training/           # BiddingPolicyNetwork, trainer, dataset, trained_models/, training_results/
│   └── synthetic_data_generation/  # Economic dispatch, merit-order heuristic labels
├── results/                    # Generated tightening reports and optimization results (not committed)
├── tests/                      # Smoke tests (pytest)
├── test_outputs/               # Artifact outputs for test cases
├── xXgraveyard/                # Deprecated experiments — reference only
├── pyproject.toml
└── AGENTS.md                   # Guidance for coding agents (including mathematical naming conventions)
```

---

## Scripts

| Script | Purpose | How to run |
|--------|---------|------------|
| `driver/run_full_pipeline.py` | **Main workflow**: scenario gen → heuristic labels → feature building → NN training → PoA tightening → PoA solve. All stage toggles live in `FullPipelineConfig` at the bottom of the file. | `.\.venv\Scripts\python.exe driver\run_full_pipeline.py` |
| `driver/run_full_pipeline_DRO.py` | Same pipeline for the DRO (regime-indexed Wasserstein) variant. Not the default workflow. | `.\.venv\Scripts\python.exe driver\run_full_pipeline_DRO.py` |
| `driver/run_PoA.py` | Older, leaner PoA runner. References legacy import path `bidding_blocks_tightening` — verify before using. | `.\.venv\Scripts\python.exe driver\run_PoA.py` |
| `models/neural_network/training/cross_validate_policy_design.py` | Hyperparameter cross-validation for NN bidding policies. | Direct invocation |
| `models/neural_network/features/build_features.py` | Stand-alone feature construction step. | Direct invocation |

---

## Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project in editable mode with dev extras
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**Gurobi**: The active pipeline requires a Gurobi license. Install separately and
verify the license is reachable. AGENTS.md references a `gurobi_check.py` helper
but that file does not currently exist in the repository.
TODO: confirm Gurobi installation/verification procedure.

---

## Running & testing

### Main pipeline

Edit the `FullPipelineConfig(...)` block at the bottom of
`driver/run_full_pipeline.py`, then:

```powershell
.\.venv\Scripts\python.exe driver\run_full_pipeline.py
```

**Fast-iteration pattern** (reuse existing generated artifacts):

```python
# In FullPipelineConfig(...)
run_scenario_generation=False,
run_heuristic_labels=False,
run_feature_building=False,
run_nn_training=False,
run_tightening=True,
tightening_flags={          # enable only the stages you changed
    "primal_big_m": False,
    "relu_bounds": False,
    ...
},
run_poa_optimization=True,
```

Use a small `horizon` (e.g., 4–8) and the `debugging` regime when testing
formulation changes. Keep `poa_solver_threads_per_worker` low (1) when using
multiple `poa_parallel_workers` to avoid oversubscribing Gurobi.

### Current active configuration (as of last commit)

- Case: `base_test_case`, horizon: 8, 5 generators (G2, G1, W3, W2, W1)
- NN policy generators: `["G1", "W2", "W3"]`
- NN architecture: hidden_layers `[4, 8]`, final activation `linear`
- Objective mode: `piecewise_mccormick` (25 pieces, PoA bounds 1.0–10.0)
- Parallel workers: 6, solver threads: 1, preprocessing time limit: 200 s

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

Tests are lightweight smoke checks. For expensive pipeline stages, reduce
scenario counts, horizon, or time limits in `FullPipelineConfig` before running.

### Linting and formatting

```powershell
# Format
.\.venv\Scripts\python.exe -m black .
.\.venv\Scripts\python.exe -m isort .

# Type check
.\.venv\Scripts\python.exe -m mypy .
```

Configuration: black and isort use line-length 100; mypy targets Python 3.8 with
strict settings (`disallow_untyped_defs`, `warn_return_any`).

---

## Conventions

- **Mathematical naming**: Follow the thesis notation (see AGENTS.md §Mathematical
  Correspondence). Key names: `alpha[i,b,t]` for bids, `P_eq`/`P_opt` for
  dispatch, `lambda` for market-clearing dual, `mu_*` for capacity/ramp duals,
  `z_*` for complementarity binaries.
- **Block indexing**: Always use physical-generator / local-block pairs. Use
  `BlockStructure` and helpers from `models/helper.py` rather than inventing
  alternate index mappings.
- **Profile parsing**: Use `parse_profile()` and `ensure_profile()` from
  `models/helper.py`.
- **Feature names**: Any feature name used in `FullPipelineConfig.nn_feature_columns`
  must be supported both by `NeuralNetworkFeatureBuilder` and by the raw feature
  expression logic inside `PoAOptimization`.
- **JSON output schemas**: Preserve existing keys in tightening report JSON files.
  Downstream stages load these reports by key; schema changes break the pipeline.
- **Solver**: Use Gurobi for MILP/MPEC work. Do not silently switch solvers.
- **Comments**: Keep concise; use mathematical names consistent with the thesis.
  ASCII only in code and docs unless existing files require symbols.
- **Style**: black, line-length 100.

---

## Gotchas & safety

### Correctness-critical: Big-M tightening

Big-M constants are not just a speed optimization — incorrect constants change
the feasible set and invalidate the PoA result. Validate any new tightening
logic on small, fully solvable cases before running the full pipeline.

### Correctness-critical: ReLU MILP embedding

ReLU policies are embedded with exact MILP constraints. Valid preactivation
bounds (from `relu_bounds` tightening stage) affect both tractability and model
correctness. If bounds are too loose the MILP is harder; if they are wrong the
embedding is invalid.

### Generated artifacts

The following directories contain auto-generated files that should not be manually
edited or committed unless explicitly refreshed:

- `results/` — tightening reports and PoA optimization results
- `models/neural_network/features/generated/` — built features and normalization stats
- `models/neural_network/training/trained_models/` — `.pt` model artifacts
- `models/neural_network/training/training_results/` — training history JSON
- `bidding_equilibria.egg-info/` — setuptools build artifact

### Long-running steps

- NN training (500 epochs × 5 generators) — use `run_nn_training=False` to skip.
- PoA tightening (6 stages, each solves multiple optimization problems) — toggle
  individual stages via `tightening_flags` to skip stages whose inputs haven't changed.
- PoA optimization — can be very slow for large horizons; start with horizon ≤ 8.

### xXgraveyard/

Treat as read-only reference material. Do not build new functionality here.

### DRO pipeline

`driver/run_full_pipeline_DRO.py` and `models/DRO_PoA/` are experimental and not
the default workflow. Start from there only when explicitly asked to work on the
DRO formulation.

### No production touchpoints

This is a self-contained research codebase. There are no external APIs, databases,
or cloud services beyond Gurobi licensing and local file I/O.
