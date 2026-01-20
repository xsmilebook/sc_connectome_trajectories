#!/bin/bash
#SBATCH -J clg_maskdiag
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

# Diagnostic-only run to inspect:
# - sum(p_hat)/k (mask_ratio)
# - p_hat quantiles (p10/p50/p90)
# - precision@k / recall@k
# - optional AUPRC
export CLG_CV_FOLD=0
export RUN_BASE="clg_maskdiag_fold0"

export RESIDUAL_SKIP=1
export RESIDUAL_TAU=1.0
export RESIDUAL_CAP=0.2
export LAMBDA_FULL_LOG_MSE=0.05

export EDGE_LOSS="bce"
export EDGE_POS_WEIGHT=5

export LAMBDA_DENSITY=0.0
export LAMBDA_ZERO_LOG=0.02

export EARLY_STOP_METRIC="val_sc_log_pearson_sparse"
export VAL_SC_EVAL_EVERY=1
export COMPUTE_MASK_AUPRC=1

export ADJACENT_PAIR_PROB=1.0
export LATENT_DIM=32
export HIDDEN_DIM=64
export SOLVER_STEPS=6
export MAX_EPOCHS=40
export PATIENCE=15
export BATCH_SIZE=2
export LEARNING_RATE=1e-4

export LAMBDA_KL=0.0
export LAMBDA_TOPO=0.0
export LAMBDA_VEL=0.0
export LAMBDA_ACC=0.0
export GRADNORM_SCOPE="none"
export SC_POS_EDGE_DROP_PROB=0.0
export MORPH_NOISE_SIGMA=0.0

bash scripts/submit_clg_ode.sh

