# AGENTS.md

Guidance for coding agents working in this repository.

## Project Aim

This repository supports a thesis project on strategic bidding equilibria and
worst-case market inefficiency in power markets. The thesis draft frames the
research problem as a tri-level inefficiency analysis:

1. An upper-level Price-of-Anarchy (PoA) maximization chooses exogenous market
   states, mainly demand and wind availability trajectories, inside a physically
   meaningful support set.
2. A middle-level strategic producer equilibrium would normally choose bids for
   competing agents.
3. A lower-level intertemporal economic dispatch clears the market subject to
   generation capacities and physical ramping.

The current codebase makes this tractable by replacing the explicit middle-level
equilibrium with learned bidding policies. Synthetic scenarios are generated,
heuristic best-response-like bid labels are produced, neural networks are trained
per physical generator, and the trained ReLU policies are embedded in a
single-level Pyomo/Gurobi PoA model using mixed-integer linear constraints.

The thesis draft also discusses distributionally robust optimization (DRO) and
regime-centered Wasserstein ambiguity sets. The current repository contains a
`models/DRO_PoA` implementation, but the main actively wired workflow is the
standard PoA pipeline in `driver/run_full_pipeline.py`.

## Current Code Map

- `driver/run_full_pipeline.py`: Main end-to-end workflow. It can generate
  scenarios, run heuristic labels, build neural-network features, train policies,
  run staged PoA tightening, and solve the final PoA model.
- `driver/run_PoA.py`: Older/leaner PoA runner. It still references the legacy
  `bidding_blocks_tightening` import path, so check it before relying on it.
- `config/reference_cases.yaml`: Reference system definitions. The active case
  is usually `test_case_bidding_blocks`, with 5 physical generators: `G2`, `G1`,
  `W3`, `W2`, `W1`. Conventional generators can have multiple bidding blocks;
  wind generators currently have one active block each.
- `config/regime_definitions.yaml`: Stochastic scenario regimes for policy
  training, PoA analysis, and debugging.
- `config/scenarios/scenario_generator.py`: Regime-based demand and wind scenario
  generation.
- `models/synthetic_data_generation`: Economic dispatch and merit-order heuristic
  label generation.
- `models/neural_network/features`: Feature construction for neural policies.
- `models/neural_network/training`: Per-generator neural-network training,
  exported model artifacts, and training summaries.
- `models/PoA/PoA_optimization.py`: Main block-aware PoA Pyomo model with support
  set constraints, embedded NN policies, KKT blocks for equilibrium and social
  optimum dispatch, and tightening-report loading.
- `models/PoA/PoA_tightening`: Staged preprocessing for primal Big-M values,
  ReLU preactivation bounds, certified alpha/bid bounds, slack-based binary
  fixing, and dual Big-M tightening.
- `models/PoA/support_set_config.yaml`: Deterministic support-set definitions for
  demand and wind trajectories used by PoA and tightening models.
- `models/DRO_PoA/DRO_PoA_optimization.py`: Experimental DRO PoA implementation
  with empirical-scenario indexing and Wasserstein-style penalty support.
- `models/helper.py`: Shared parsing, block-structure, ramp, and generator/block
  mapping helpers. Prefer these helpers over duplicating indexing logic.
- `results/`: Generated scenario, tightening, and optimization outputs.
- `xXgraveyard/`: Historical experiments and deprecated wrappers. Do not build new
  functionality here unless explicitly asked.

## Mathematical Correspondence

Use the thesis terminology when naming new concepts:

- State `s`: demand trajectory plus wind available-capacity trajectory.
- Support set `U`: admissible demand and wind trajectories with bounds, ramp
  limits, and optionally deviation budgets around reference trajectories.
- `alpha[i,b,t]`: bid for physical generator `i`, local block `b`, time `t`.
- `P_eq`: dispatch under learned strategic/equilibrium bids.
- `P_opt`: dispatch under truthful marginal-cost bids.
- `lambda`: market-clearing price dual for demand balance.
- `mu_*`: nonnegative duals for capacity and ramp inequalities.
- `z_*`: complementarity binaries used in Big-M KKT linearizations.
- `C_eq - C_opt`: the optimized PoA proxy; the ratio can be computed after solve.

The current implementation indexes dispatch and bids by physical generator and
local bidding block, not by a single global generator-block index. Use the helper
structures in `models/helper.py` and the initialized mappings in
`PoAOptimization` instead of inventing alternate mappings.

## Main Workflow

The usual full pipeline is:

```powershell
.\.venv\Scripts\python.exe driver\run_full_pipeline.py
```

Important toggles live in the `FullPipelineConfig` instance at the bottom of
`driver/run_full_pipeline.py`.

Common fast iteration pattern:

- Keep `run_scenario_generation=False`, `run_heuristic_labels=False`,
  `run_feature_building=False`, and `run_nn_training=False` when reusing existing
  generated artifacts.
- Toggle individual tightening stages through `tightening_flags`.
- Use `poa_parallel_workers` with low `poa_solver_threads_per_worker` to avoid
  oversubscribing Gurobi.
- Start with small horizons and the `debugging` regime when changing model
  formulation logic.

The main generated tightening reports are:

- `results/poa_tightening/primal_big_m_report.json`
- `results/poa_tightening/relu_bounds_report.json`
- `results/poa_tightening/alpha_bounds_report.json`
- `results/poa_tightening/slack_binary_fix_report.json`
- `results/poa_tightening/dual_big_m_report.json`
- `results/poa_tightening/final_tightening_report.json`

The final PoA result path defaults to `results/poa_optimization_T{horizon}.json`.

## Environment And Dependencies

The project is Python-based and uses Pyomo, NumPy, pandas, SciPy, scikit-learn,
PyTorch, YAML, and Gurobi. `pyproject.toml` declares the main dependencies.

Use the local virtual environment when available:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe gurobi_check.py
```

Gurobi is the expected solver for the active PoA and tightening workflows. Do not
silently switch solvers for MILP/MPEC work unless the user asks or the model is
being deliberately simplified for debugging.

## Testing And Verification

There is no dedicated test suite in the current repo. For model changes, prefer
small deterministic smoke checks:

```powershell
.\.venv\Scripts\python.exe gurobi_check.py
.\.venv\Scripts\python.exe driver\run_full_pipeline.py
```

For expensive pipeline stages, reduce scenario counts, horizon, time limits, or
parallel workers in `FullPipelineConfig` before running a smoke test. If you
cannot run the full pipeline because of time or license constraints, say so
explicitly and report the lighter checks you did run.

## Editing Guidelines

- Keep changes scoped. This codebase is research code with generated artifacts;
  avoid broad refactors unless they directly reduce a modeling error.
- Preserve existing output schemas for JSON reports unless a schema change is the
  task. Downstream stages load these reports by key.
- Prefer shared helpers from `models/helper.py` for profile parsing and
  block/generator mappings.
- Be careful with generated files under `results/`, `models/neural_network/*/generated`,
  and `bidding_equilibria.egg-info/`. Do not commit or overwrite generated
  artifacts unless the user specifically wants refreshed outputs.
- Treat `xXgraveyard/` as reference material only.
- Keep comments concise and mathematical names consistent with the thesis.
- Use ASCII in code and docs unless existing files require mathematical symbols.

## Known Implementation Notes

- The active support-set configuration currently uses deterministic bounds,
  ramping limits, and demand/wind deviation budgets from
  `models/PoA/support_set_config.yaml`.
- The thesis draft's regime-centered DRO formulation is not the default pipeline.
  If asked to implement it, start from `models/DRO_PoA/DRO_PoA_optimization.py`
  and align regime-specific supports with `config/regime_definitions.yaml`.
- Neural policies are trained separately per physical generator. Feature names
  must be supported both by `NeuralNetworkFeatureBuilder` and by the raw feature
  expression logic inside `PoAOptimization`.
- ReLU policies are embedded exactly with MILP constraints, so valid
  preactivation bounds are important for tractability and correctness.
- Big-M tightening is not just an optimization speed-up; invalid Big-M constants
  can change the feasible set. Validate new tightening logic on small cases.
