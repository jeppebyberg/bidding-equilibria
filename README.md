# bidding-equilibria

Research codebase for an MSc thesis on **strategic bidding equilibria and
worst-case market inefficiency in electricity markets**, conducted by
Jeppe Urup Byberg.

The project quantifies the **Price-of-Anarchy (PoA)** of an electricity market
using a tri-level optimization model. Learned per-generator bidding policies
(neural networks) are embedded as exact mixed-integer linear constraints inside a
Pyomo/Gurobi model and solved to global optimality, so that the worst-case loss
of efficiency from strategic bidding can be certified rather than merely
simulated.

---

## The model

The thesis frames market inefficiency as a **tri-level** problem:

1. **Upper level - PoA maximization.** Chooses exogenous market states (demand
   and wind-availability trajectories) inside a physically meaningful *support
   set* so as to maximise market inefficiency.
2. **Middle level - strategic equilibrium.** Competing producers choose bids.
   This level is made tractable by replacing the explicit equilibrium with
   **learned bidding policies**: one ReLU neural network per physical generator,
   trained on heuristic best-response labels.
3. **Lower level - economic dispatch.** An intertemporal economic dispatch clears
   the market subject to generation capacities and physical ramp limits.

The market inefficiency proxy is `C_eq - C_opt`, the cost gap between dispatch
under strategic/equilibrium bids (`P_eq`) and dispatch under truthful
marginal-cost bids (`P_opt`); the PoA ratio is recovered after the solve.

A **distributionally robust (DRO)** variant, using regime-centered Wasserstein
ambiguity sets, is also implemented and is run as the later stages of the full
pipeline.

For the end-to-end stage diagram see [pipeline.md](pipeline.md).

---

## Repository structure

```
bidding-equilibria/
|-- driver/                     # Entry-point scripts and pipeline orchestration
|   |-- full_pipeline.py        # Main end-to-end runner (all blocks)
|   |-- project_config.py       # *** Central run configuration (edit this) ***
|   |-- block0_system_setup.py  # Config build + manifests
|   |-- block1_data_labels_pipeline.py    # Scenarios + heuristic bid labels
|   |-- block2_policy_training_pipeline.py# Features + per-generator NN training
|   |-- block3_poa_pipeline.py            # PoA tightening + PoA MILP solve
|   |-- block35_support_oos_pipeline.py   # Support-set out-of-sample diagnostics
|   |-- block4_dro_poa_pipeline.py        # DRO PoA (Wasserstein eta sweep)
|   |-- block45_oos_poa_pipeline.py       # Out-of-sample PoA evaluation
|   |-- core/                   # Per-block implementation (block*_core.py)
|   `-- sensitivity/            # Sensitivity sweeps and summaries
|-- config/                     # YAML configs + scenario generation
|   |-- reference_cases.yaml    # Physical system: generators, costs, capacities
|   |-- regime_definitions.yaml # Demand/wind regime parameters
|   `-- scenarios/              # ScenarioManager (regime-based stochastic draws)
|-- models/
|   |-- helper.py               # Shared block-structure / profile / ramp helpers
|   |-- synthetic_data_generation/  # Economic dispatch + merit-order labels
|   |-- neural_network/         # Feature building + per-generator NN training
|   |-- PoA/                    # PoA MILP model + 6-stage Big-M/ReLU tightening
|   `-- DRO_PoA/                # Distributionally robust PoA variant
|-- results_viz/                # All standalone plotting / figure scripts
|-- results/                    # Generated outputs (per case; not committed)
|-- tests/                      # Lightweight smoke tests (pytest)
|-- pyproject.toml
|-- AGENTS.md                   # Notation + conventions for contributors
`-- pipeline.md                 # Pipeline stage diagram
```

---

## Setup

Requires **Python >= 3.8** and a working **Gurobi** license (the MILP/MPEC
workflows use Gurobi as the solver).

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the project (with dev extras) in editable mode
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify that Gurobi is licensed and reachable before running the PoA stages, e.g.:

```powershell
.\.venv\Scripts\python.exe -c "import gurobipy; gurobipy.Model()"
```

Core dependencies (declared in [pyproject.toml](pyproject.toml)): NumPy, pandas,
SciPy, scikit-learn, PyTorch, Pyomo, PyYAML, and gurobipy. PyTorch is used as the
CPU build in the reference environment.

---

## How to run

All runtime behaviour is controlled from a single object, `PROJECT_CONFIG`, in
[driver/project_config.py](driver/project_config.py). Every block loads its
configuration from there via `block0_core.build_config()`. **Edit that file to
change the experiment; you do not pass command-line arguments.**

### Full pipeline

```powershell
.\.venv\Scripts\python.exe driver\full_pipeline.py
```

This runs, in order: data + labels (block 1) -> policy training (block 2) ->
PoA tightening + solve (block 3) -> support-set OOS diagnostics (block 3.5) ->
DRO PoA (block 4) -> out-of-sample PoA evaluation (block 4.5).

### Running a single block

Each block is independently runnable as a module (it builds the same config from
`project_config.py`):

```powershell
.\.venv\Scripts\python.exe -m driver.block1_data_labels_pipeline
.\.venv\Scripts\python.exe -m driver.block2_policy_training_pipeline
.\.venv\Scripts\python.exe -m driver.block3_poa_pipeline
.\.venv\Scripts\python.exe -m driver.block35_support_oos_pipeline
.\.venv\Scripts\python.exe -m driver.block4_dro_poa_pipeline
.\.venv\Scripts\python.exe -m driver.block45_oos_poa_pipeline
```

### Fast-iteration pattern (reuse existing artifacts)

The expensive stages (label generation, NN training, tightening, PoA solve) are
individually toggled. To re-run only the part you changed, flip the relevant
`run_*` flags off in `project_config.py` so previously generated artifacts are
reused:

```python
PROJECT_CONFIG = ProjectConfig(
    case_label="base_case",
    horizon=6,
    run_scenario_generation=False,   # reuse existing scenarios
    run_heuristic_labels=False,      # reuse existing labels
    run_feature_building=False,      # reuse existing features
    run_nn_training=False,           # reuse existing trained policies
    run_poa_tightening=True,
    poa_tightening_flags={           # enable only the stages you changed
        "relu_bounds": True,
        "alpha_bounds": True,
        "slack_binary_fix": True,
        "dual_big_m": True,
    },
    run_poa_optimization=True,
)
```

When iterating on the model formulation, use a small `horizon` (e.g. 4-8) and a
low solver thread count to keep solves fast.

### Key configuration knobs

| Field | Purpose |
|-------|---------|
| `case_label` | Reference system / output folder name (e.g. `base_case`). |
| `horizon` | Number of time steps in the dispatch problem. |
| `allow_wind_to_play` | Whether wind generators bid strategically. |
| `synthetic_labels_target` | Target number of heuristic training labels. |
| `run_scenario_generation` / `run_heuristic_labels` / `run_feature_building` / `run_nn_training` | Stage toggles for block 1-2 data preparation. |
| `run_poa_tightening` + `poa_tightening_flags` | Enable the 6-stage Big-M / ReLU tightening and select individual stages. |
| `run_poa_optimization` | Solve the final PoA MILP. |
| `run_dro_tightening` / `run_dro_optimization` + `dro_tightening_flags` | DRO (block 4) controls. |
| `poa_mccormick_num_pieces` / `dro_mccormick_num_pieces` | McCormick piece count for the PoA-ratio bounding. |
| `poa_parallel_workers` / `poa_solver_threads_per_worker` | Parallelism; keep `threads_per_worker=1` when running many workers to avoid oversubscribing Gurobi. |
| `ambiguity_kappa` | DRO Wasserstein ambiguity-set radius scaling. |
| `plot_results_along_the_way` | Emit figures during the run. |

---

## Outputs

Results are written under `results/<case_label>/`, including:

- `figures/` - plots emitted during the run.
- PoA tightening reports (`relu_bounds`, `alpha_bounds`, `slack_binary_fix`,
  `dual_big_m`, and the final combined report) consumed by later stages.
- PoA optimization results, e.g. `poa_optimization_T{horizon}.json`.
- `pipeline_manifests/` - per-block run manifests.

Generated artifacts under `results/`, `models/neural_network/**/generated/`, and
`models/neural_network/training/trained_models/` are not committed.

---

## Visualization

All standalone plotting scripts live in [results_viz/](results_viz/). They read
the JSON/CSV artifacts under `results/` and write figures. Run them from the
repository root, for example:

```powershell
.\.venv\Scripts\python.exe -m results_viz.visualize_poa_trajectory
.\.venv\Scripts\python.exe -m results_viz.plot_base_poa_overview
```

### Figure output format

All plotting code routes through a single thesis figure-output policy in
[results_viz/_thesis_style.py](results_viz/_thesis_style.py). Every figure is
saved as **both** a vector **PDF** (for LaTeX inclusion) and a high-DPI **PNG**
(default 300 DPI), regardless of the extension used in the script's own
`savefig` call. The policy only writes figure files; it never deletes or
modifies `.json`/`.csv` result data.

Override the defaults with environment variables:

```powershell
$env:THESIS_FIG_FORMATS = "pdf"   # vector only (default: "pdf,png")
$env:THESIS_FIG_DPI     = "600"   # raster DPI   (default: 300)
```

---

## Tests

Lightweight smoke tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

For expensive stages, reduce scenario counts, horizon, or time limits in
`project_config.py` before running.

## Formatting and type checking

```powershell
.\.venv\Scripts\python.exe -m black .
.\.venv\Scripts\python.exe -m isort .
.\.venv\Scripts\python.exe -m mypy .
```

`black` and `isort` use line length 100.

---

## Notes for contributors

See [AGENTS.md](AGENTS.md) for mathematical notation, naming conventions, and
correctness-critical details (Big-M tightening and the ReLU MILP embedding both
affect the *feasible set*, not just solver speed - validate changes on small,
fully solvable cases first).

---

## Author

Jeppe Urup Byberg - MSc thesis, Technical University of Denmark (DTU).
</content>
