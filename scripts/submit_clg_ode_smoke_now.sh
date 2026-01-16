#!/bin/bash
set -euo pipefail

# Quick submit wrapper for the smoke job on q_ai8.
export RUN_TAG="${RUN_TAG:-smoke}"
export FOLD_ID="${FOLD_ID:-0}"
export MAX_EPOCHS="${MAX_EPOCHS:-8}"
export PATIENCE="${PATIENCE:-3}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export TOPO_SCALE_Q="${TOPO_SCALE_Q:-0.9}"
export TOPO_WARMUP_FRAC="${TOPO_WARMUP_FRAC:-0.2}"

sbatch scripts/submit_clg_ode_smoke.sh
