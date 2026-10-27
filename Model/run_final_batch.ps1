# Same as run_final_batch.sh for PowerShell (run after `conda activate coram`).
# Usage (from the Model folder): .\run_final_batch.ps1 [-Out Results/intertemporal] [-Calib Results/calibration_2023]
param([string]$Out = "Results/intertemporal", [string]$Calib = "Results/calibration_2023")
$root = Split-Path -Parent $PSScriptRoot
$o = Join-Path $root $Out; $c = Join-Path $root $Calib; New-Item -ItemType Directory -Force "$o\logs" | Out-Null
$env:FORCE_RESOLVE = "1"; $env:RESULTS_DIR_IT = $o      # code defaults: conveyance 1.38, real-cost logit, existing-desal eff 1.0
$env:DIAG_ONLY = "0"
$p0 = Start-Process python -ArgumentList "-u","run_intertemporal_model.py" -RedirectStandardOutput "$o\logs\s0.txt" -RedirectStandardError "$o\logs\s0_err.txt" -PassThru -NoNewWindow
# 2023 calibration: single-year solve, NO investment (2023 is the observed base year)
$env:CALIB_MODE="1"; $env:AREA_CHANGE_RATE="0"; $env:AREA_CHANGE_RATE_DN="0"; $env:RESULTS_DIR_IT = $c
$pc = Start-Process python -ArgumentList "-u","run_intertemporal_model.py" -RedirectStandardOutput "$o\logs\calib2023.txt" -RedirectStandardError "$o\logs\calib2023_err.txt" -PassThru -NoNewWindow
Remove-Item Env:CALIB_MODE, Env:AREA_CHANGE_RATE, Env:AREA_CHANGE_RATE_DN
$p0 | Wait-Process
# 10 BCM warm-started from the No-limit solution; 5 BCM from Results/warmstart_5BCM (with
# technology-average desalination costs it does not converge from No-limit or 10 BCM; the
# starting point does not change the optimum)
$env:RESULTS_DIR_IT = $o
$ws = @{ 1 = @($o, "No_GW_Limit"); 2 = @((Join-Path $root "Results/warmstart_5BCM"), "5_BCM_GW_Limit") }
$procs = @($pc)
foreach ($i in 1,2) { $env:DIAG_ONLY = "$i"; $env:WS_SOURCE_DIR = $ws[$i][0]; $env:WS_SOURCE_GW = $ws[$i][1]
  $procs += Start-Process python -ArgumentList "-u","run_intertemporal_model.py" -RedirectStandardOutput "$o\logs\s$i.txt" -RedirectStandardError "$o\logs\s${i}_err.txt" -PassThru -NoNewWindow }
$procs | Wait-Process
Get-ChildItem "$o\logs\*.txt" | ForEach-Object { $_.Name; Select-String -Path $_.FullName -Pattern "Termination:|SCENARIO COMPLETE|NPV objective|FEASIBILITY CHECK" | Select-Object -Last 4 }
python results_numbers.py $o | Tee-Object "$o\results_numbers.txt"
python calib_fit.py "$c\2023_STANDARD_No_GW_Limit.xlsx"
