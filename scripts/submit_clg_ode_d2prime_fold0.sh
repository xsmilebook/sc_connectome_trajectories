#!/bin/bash
#SBATCH -J clg_d2p
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

# D2′: More conservative innovation + freeze backbone + long-interval-only gate.
# Base = C2: fixed-support + residual + dt gate, lambda_delta_log=0.
export CLG_CV_FOLD="${CLG_CV_FOLD:-0}"
export RUN_BASE="${RUN_BASE:-clg_d2prime_long_freeze}"

export RESIDUAL_SKIP=1
export RESIDUAL_TAU=1.0
export RESIDUAL_CAP=0.5

export FIXED_SUPPORT=1
export LAMBDA_DELTA_LOG=0.0

export INNOVATION_ENABLED=1
export INNOVATION_TOPM=200
export INNOVATION_K_NEW=40
export INNOVATION_TAU=0.07
export INNOVATION_DELTA_QUANTILE=0.975
export INNOVATION_DT_OFFSET_MONTHS=9
export INNOVATION_DT_RAMP_MONTHS=9
export LAMBDA_NEW_SPARSE=0.20
export NEW_SPARSE_WARMUP_EPOCHS=10
export NEW_SPARSE_RAMP_EPOCHS=0
export INNOVATION_FREEZE_BACKBONE_AFTER=10

export INNOVATION_FOCAL_GAMMA=2.0
export INNOVATION_FOCAL_ALPHA=0.25
export LAMBDA_NEW_REG=0.0

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

export LAMBDA_KL=0.0
export LAMBDA_TOPO=0.0
export LAMBDA_VEL=0.0
export LAMBDA_ACC=0.0
export GRADNORM_SCOPE="none"
export SC_POS_EDGE_DROP_PROB=0.0
export MORPH_NOISE_SIGMA=0.0

bash scripts/submit_clg_ode.sh
