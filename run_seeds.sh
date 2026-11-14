#!/usr/bin/env bash
# NOTE NEW: launch 5 CACTO runs in sequence with different seeds (N_try = 1..5, seed = N_try).
# The method (ALM or NO ALM) and all the other parameters are taken from conf_single_integrator.py (CACTO_ALM flag):
# do NOT modify the config file while the runs are going, every run reads it at its start.
#
# Usage (from the cacto_ALM folder, with the conda environment active):
#   bash run_seeds.sh
# To run in background and keep it running after closing the terminal:
#   nohup bash run_seeds.sh > run_seeds.log 2>&1 &

set -u

# --- Runs to launch ---
N_TRY_LIST=(11 12 13 14 15)        # N_try of the runs (results in Results .../N_try_k)

# --- Command line arguments of main.py (same for all the runs) ---
PYTHON=${PYTHON:-python3}     # python of the conda environment (override with: PYTHON=/path/to/python bash run_seeds.sh)
SYSTEM_ID="single_integrator"
GPU_DEVICE="None"
W_S="1e-2"
PLOT_FLAG=1
BICSF=0.7                       # biased initial-state sampling factor (0 = no BIC)
UNIF_F=0.3                      # fraction of uniform initial states mixed with the biased ones

cd "$(dirname "$0")"          # run from the cacto_ALM folder (main.py and the config are here)

FAILED=()
START_ALL=$(date +%s)

for N_TRY in "${N_TRY_LIST[@]}"; do
    SEED=$N_TRY               # one different seed per run
    echo "=================================================================="
    echo " Run N_try=${N_TRY}  seed=${SEED}   started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    START=$(date +%s)

    # The terminal output of every run is also saved by main.py in Log/N_try_k/terminal_log.txt
    "$PYTHON" -u main.py \
        --system-id="$SYSTEM_ID" \
        --seed="$SEED" \
        --GPU-device="$GPU_DEVICE" \
        --w-S="$W_S" \
        --test-n="$N_TRY" \
        --plot-flag="$PLOT_FLAG" \
        --BICSf="$BICSF" \
        --unif-f="$UNIF_F"
    STATUS=$?

    ELAPSED=$(( $(date +%s) - START ))
    if [ $STATUS -ne 0 ]; then
        echo ">>> Run N_try=${N_TRY} FAILED (exit code ${STATUS}) after ${ELAPSED} s -> going on with the next one"
        FAILED+=("$N_TRY")
    else
        echo ">>> Run N_try=${N_TRY} finished in $(( ELAPSED / 60 )) min"
    fi
done

echo "=================================================================="
echo " All runs done in $(( ($(date +%s) - START_ALL) / 60 )) min"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Failed runs (N_try): ${FAILED[*]}"
else
    echo " All runs finished correctly"
fi
echo "=================================================================="
