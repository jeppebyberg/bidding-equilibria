All of these scripts must be run over night. It should be 

I need to run to validate the setup:
- python -m driver.sensitivity.bound_tightening_progression
- python -m driver.sensitivity.mccormick_pieces_sweep

To compare with and without wind playing:
- python -m driver.sensitivity.wind_playing_sweep

To compute the different computation time when having different horizons
- python -m driver.sensitivity.horizon_sweep

To sweep the DRO ambiguity-set parameters (each runs the full pipeline per variant):
- python -m driver.sensitivity.peak_w_sweep
- python -m driver.sensitivity.rho_sweep
- python -m driver.sensitivity.sigma_max_sweep  # max sigma [0.025 base, 0.02, 0.015, 0.01]

To sweep the ambiguity-budget multiplier ambiguity_kappa (reuses base_case
policies/features; recomputes only the PoA + support-OOS + DRO legs per value;
kappa in [0.0, 0.25 base, 0.5, 0.75, 1.0]):
- python -m driver.sensitivity.ambiguity_kappa_sweep

To test the base-case DRO on regimes spanning the ambiguity box (block4 only;
reuses base_case policies; mu_D/sigma_D/mu_W/sigma_W swept one-at-a-time across
their box mean/outer values, rho and peak_W held fixed -- 9 regimes):
- python -m driver.sensitivity.dro_regime_box_sweep
Then build the cross-regime comparison plots + CSV:
- python -m results_viz.plot_dro_regime_box

DRO eta-sweep analysis (block4 only; reuses each case's existing PoA artifacts,
fills in the missing <case>/dro/ folders. Skips T6 / N3 which already have DRO):
- python -m driver.sensitivity.horizon_sweep_dro
- python -m driver.sensitivity.players_sweep_dro
Then build the cross-case comparison plots + summary CSVs:
- python -m results_viz.plot_sensitivity_eta_sweep horizon_sweep
- python -m results_viz.plot_sensitivity_eta_sweep players_sweep

  'driver.sensitivity.peak_w_sweep',
  'driver.sensitivity.rho_sweep',
  'driver.sensitivity.sigma_max_sweep',

## Run all (PowerShell)

```powershell
$logDir = 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$indexLog = 'overnight.log'
# Append to existing log so last night's records are preserved.

$mods = @(
  # Re-run to pick up the NN alpha-bound floor (1e-5) in every tightening case.
  'driver.sensitivity.bound_tightening_progression',
  # Finishes the remaining sigma.max cases (0.015, 0.01); 0.025/0.02 already done.
  'driver.sensitivity.sigma_max_sweep',
  # Adds the high-resolution McCormick points (300, 500 pieces); rest already done.
  'driver.sensitivity.mccormick_pieces_sweep',
  'driver.sensitivity.composition_sweep',
  # Ambiguity-budget sweep: kappa [0.0, 0.25 base, 0.5, 0.75, 1.0]; reuses base_case policies.
  'driver.sensitivity.ambiguity_kappa_sweep',
  'driver.sensitivity.dro_regime_box_sweep',
  'driver.sensitivity.horizon_sweep_dro'
)

foreach ($m in $mods) {
  $logFile = "$logDir\$($m -replace '\.', '_').log"
  $ts = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
  "=== $ts  START $m ===" | Tee-Object $indexLog -Append
  "=== $ts  START $m ===" | Out-File $logFile
  .\.venv\Scripts\python.exe -m $m 2>&1 | Tee-Object $logFile -Append
  $exit = $LASTEXITCODE
  $ts = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
  "=== $ts  END   $m (exit $exit) ===" | Tee-Object $logFile -Append
  "=== $ts  END   $m (exit $exit) ===" | Tee-Object $indexLog -Append
}

# Post-processing plots (read-only; run after the solves complete).
$plots = @(
  'results_viz.plot_dro_regime_box'
)

foreach ($p in $plots) {
  $logFile = "$logDir\$($p -replace '\.', '_').log"
  $ts = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
  "=== $ts  START $p ===" | Tee-Object $indexLog -Append
  "=== $ts  START $p ===" | Out-File $logFile
  .\.venv\Scripts\python.exe -m $p 2>&1 | Tee-Object $logFile -Append
  $exit = $LASTEXITCODE
  $ts = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
  "=== $ts  END   $p (exit $exit) ===" | Tee-Object $logFile -Append
  "=== $ts  END   $p (exit $exit) ===" | Tee-Object $indexLog -Append
}
```

