#!/usr/bin/env bash
# Final batch for the national paper: No-limit + 2023 calibration, then the two capped GW
# scenarios (10 BCM warm-started from the No-limit solution, as the sequential warm start
# stalls it; 5 BCM from Results/warmstart_5BCM), then numbers.
# Usage (from the Model folder):  bash run_final_batch.sh [results_dir_name] [calib_dir_name]
#   defaults: Results/intertemporal, Results/calibration_2023
# Needs the coram env activated (`conda activate coram`) so python, IPOPT and the numpy DLLs
# are on PATH (see National_Model_Handoff.md).
set -e
OUT="../${1:-Results/intertemporal}"; CAL="../${2:-Results/calibration_2023}"; mkdir -p "$OUT/logs"
export FORCE_RESOLVE=1 RESULTS_DIR_IT="$OUT"          # code defaults: conveyance 1.38, real-cost logit, existing-desal eff 1.0
DIAG_ONLY=0 python -u run_intertemporal_model.py > "$OUT/logs/s0.txt" 2>&1 &
p0=$!
# 2023 calibration: single-year solve, same structure, NO investment (2023 is the observed base year)
CALIB_MODE=1 AREA_CHANGE_RATE=0 AREA_CHANGE_RATE_DN=0 DIAG_ONLY=0 \
  RESULTS_DIR_IT="$CAL" python -u run_intertemporal_model.py > "$OUT/logs/calib2023.txt" 2>&1 &
pc=$!
wait "$p0" || echo "No-limit run exited non-zero"
WS_SOURCE_DIR="$OUT" WS_SOURCE_GW=No_GW_Limit \
  DIAG_ONLY=1 python -u run_intertemporal_model.py > "$OUT/logs/s1.txt" 2>&1 & p1=$!
# 5 BCM: with technology-average desalination costs IPOPT does not converge from the
# No-limit or 10 BCM solution, so it starts from the paper's 5 BCM solution
# (Results/warmstart_5BCM). The starting point does not change the optimum.
WS_SOURCE_DIR=../Results/warmstart_5BCM WS_SOURCE_GW=5_BCM_GW_Limit \
  DIAG_ONLY=2 python -u run_intertemporal_model.py > "$OUT/logs/s2.txt" 2>&1 & p2=$!
for p in "$pc" "$p1" "$p2"; do wait "$p" || echo "process $p exited non-zero"; done
echo "=== exit statuses (want Optimal / Solved To Acceptable Level, feasibility PASS) ==="
for f in "$OUT"/logs/*.txt; do echo "$f"; grep -n "Termination:\|SCENARIO COMPLETE\|NPV objective\|FEASIBILITY CHECK" "$f" | tail -n 4; done
python results_numbers.py "$OUT" | tee "$OUT/results_numbers.txt"
python calib_fit.py "$CAL/2023_STANDARD_No_GW_Limit.xlsx"
