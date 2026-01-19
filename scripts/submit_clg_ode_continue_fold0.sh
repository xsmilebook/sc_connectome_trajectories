#!/bin/bash
#SBATCH -J clg_ode_continue
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

RESUME_FROM="${RESUME_FROM:-}"
if [[ -z "$RESUME_FROM" ]]; then
  echo "Set RESUME_FROM to the checkpoint path (e.g., outputs/results/clg_ode/runs/<run>/fold0/clg_ode_fold0_best.pt)." >&2
  exit 1
fi

export CLG_CV_FOLD=0
export RUN_BASE="${RUN_BASE:-clg_ode_continue_fold0}"
export RESIDUAL_SKIP="${RESIDUAL_SKIP:-1}"
export RESIDUAL_TAU="${RESIDUAL_TAU:-1.0}"
export RESIDUAL_CAP="${RESIDUAL_CAP:-0.2}"
export LAMBDA_FULL_LOG_MSE="${LAMBDA_FULL_LOG_MSE:-0.05}"
export LAMBDA_ZERO_LOG="${LAMBDA_ZERO_LOG:-0.02}"
export LAMBDA_DELTA_LOG="${LAMBDA_DELTA_LOG:-0.05}"
export ADJACENT_PAIR_PROB="${ADJACENT_PAIR_PROB:-1.0}"
export LATENT_DIM="${LATENT_DIM:-32}"
export HIDDEN_DIM="${HIDDEN_DIM:-64}"
export SOLVER_STEPS="${SOLVER_STEPS:-6}"
export MAX_EPOCHS="${MAX_EPOCHS:-40}"
export PATIENCE="${PATIENCE:-6}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export LAMBDA_KL="${LAMBDA_KL:-0.0}"
export LAMBDA_TOPO="${LAMBDA_TOPO:-0.0}"
export LAMBDA_VEL="${LAMBDA_VEL:-0.0}"
export LAMBDA_ACC="${LAMBDA_ACC:-0.0}"
export GRADNORM_SCOPE="${GRADNORM_SCOPE:-none}"
export SC_POS_EDGE_DROP_PROB="${SC_POS_EDGE_DROP_PROB:-0.0}"
export MORPH_NOISE_SIGMA="${MORPH_NOISE_SIGMA:-0.0}"

export RESUME_FROM

bash scripts/submit_clg_ode.sh
