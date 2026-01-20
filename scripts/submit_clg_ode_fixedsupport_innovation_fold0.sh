#!/bin/bash
#SBATCH -J clg_fs_innov
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

# Fixed-support residual + conservative innovation (N=400 defaults).
export CLG_CV_FOLD=0
export RUN_BASE="clg_fs_innov_default"

export RESIDUAL_SKIP=1
export RESIDUAL_TAU=1.0
export RESIDUAL_CAP=0.5

export FIXED_SUPPORT=1
export INNOVATION_ENABLED=1
export INNOVATION_TOPM=400
export INNOVATION_K_NEW=80
export INNOVATION_TAU=0.10
export INNOVATION_DELTA_QUANTILE=0.95
export INNOVATION_DT_SCALE_YEARS=1.0
export INNOVATION_FOCAL_GAMMA=2.0
export INNOVATION_FOCAL_ALPHA=0.25
export LAMBDA_NEW_SPARSE=0.10
export NEW_SPARSE_WARMUP_EPOCHS=10
export NEW_SPARSE_RAMP_EPOCHS=10
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

