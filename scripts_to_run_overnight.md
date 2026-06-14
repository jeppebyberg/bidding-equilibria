All of these scripts must be run over night. It should be 

I need to run to validate the setup:
- python -m driver.sensitivity.bound_tightening_progression
- python -m driver.sensitivity.mccormick_pieces_sweep

To compare with and without wind playing:
- python -m driver.sensitivity.wind_playing_sweep

To compute the different computation time when having different horizons
- python -m driver.sensitivity.horizon_sweep

## Run all (PowerShell)

```powershell
$logDir = 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$indexLog = 'overnight.log'
'' | Out-File $indexLog   # clear index from previous run

$mods = @(
  'driver.sensitivity.bound_tightening_progression',
  'driver.sensitivity.horizon_sweep',
  'driver.sensitivity.ramp_rate_sweep',
  'driver.sensitivity.players_sweep',
  'driver.sensitivity.bidding_blocks_sweep',
  'driver.sensitivity.overlapping_costs_sweep'
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
```

