#!/bin/bash
#SBATCH -J clg_mask_c
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

# + zero-edge amplitude penalty on A_pred (suppresses false positives).
export CLG_CV_FOLD=0
export RUN_BASE="clg_ode_mask_fold0_c_zero"
export RESIDUAL_SKIP=1
export RESIDUAL_TAU=1.0
export RESIDUAL_CAP=0.2
export LAMBDA_FULL_LOG_MSE=0.05
export LAMBDA_ZERO_LOG=0.05
export ZERO_LOG_WARMUP_EPOCHS=10
export ZERO_LOG_RAMP_EPOCHS=20
export ADJACENT_PAIR_PROB=1.0
export LATENT_DIM=32
export HIDDEN_DIM=64
export SOLVER_STEPS=6
export MAX_EPOCHS=40
export PATIENCE=6
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

