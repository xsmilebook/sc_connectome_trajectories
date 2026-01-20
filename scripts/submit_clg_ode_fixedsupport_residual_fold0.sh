#!/bin/bash
#SBATCH -J clg_fs
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

# Ablation tree root: fixed-support + residual (no innovation).
export CLG_CV_FOLD=0
export RUN_BASE="clg_fs_residual"

export RESIDUAL_SKIP=1
export RESIDUAL_TAU=1.0
export RESIDUAL_CAP=0.5

export FIXED_SUPPORT=1
export INNOVATION_ENABLED=

# L_small (residual shrinkage) - adjust if needed.
export LAMBDA_DELTA_LOG=0.01

export EARLY_STOP_METRIC="val_sc_log_pearson_sparse"
export VAL_SC_EVAL_EVERY=1

export ADJACENT_PAIR_PROB=1.0
export LATENT_DIM=32
export HIDDEN_DIM=64
export SOLVER_STEPS=6
export MAX_EPOCHS=120
export PATIENCE=25
export BATCH_SIZE=2
export LEARNING_RATE=1e-4

export EDGE_LOSS="bce"
export EDGE_POS_WEIGHT=5.0
export LAMBDA_KL=0.0
export LAMBDA_TOPO=0.0
export LAMBDA_VEL=0.0
export LAMBDA_ACC=0.0
export GRADNORM_SCOPE="none"
export SC_POS_EDGE_DROP_PROB=0.0
export MORPH_NOISE_SIGMA=0.0

bash scripts/submit_clg_ode.sh

