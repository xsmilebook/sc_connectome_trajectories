#!/bin/bash
#SBATCH -J clg_ode
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%A_%a.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%A_%a.err

set -euo pipefail

module load singularity

PYTHON_HOST_BIN="/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/python"
if [[ ! -x "$PYTHON_HOST_BIN" ]]; then
  echo "Conda python not found at $PYTHON_HOST_BIN; check your conda installation." >&2
  exit 1
fi

render_paths_output="$("$PYTHON_HOST_BIN" -m scripts.render_paths \
  --set CONTAINER=local.containers.torch_gnn \
        SC_DIR=local.data.sc_connectome_schaefer400 \
        MORPH_ROOT=local.data.morphology \
        SUBJECT_INFO=local.data.subject_info_sc \
        RESULTS_DIR=local.outputs.clg_ode \
  --resolve)" || {
  echo "Failed to resolve paths via scripts.render_paths; check Python environment." >&2
  exit 1
}

eval "$render_paths_output"

if [[ -z "${CONTAINER:-}" || -z "${SC_DIR:-}" || -z "${MORPH_ROOT:-}" || -z "${SUBJECT_INFO:-}" || -z "${RESULTS_DIR:-}" ]]; then
  echo "Missing required path(s); check configs/paths.yaml and scripts.render_paths output." >&2
  exit 1
fi

PYTHON_BIN="/opt/conda/bin/python"
TORCHRUN_BIN="/opt/conda/bin/torchrun"
FOLD_ID="${CLG_CV_FOLD:-${SLURM_ARRAY_TASK_ID:-}}"
RUN_SUFFIX=""
FOLD_ARGS=()
if [[ -n "${FOLD_ID}" ]]; then
  RUN_SUFFIX="_fold${FOLD_ID}"
  FOLD_ARGS=(--cv_fold "${FOLD_ID}")
fi
JOB_ID="${SLURM_JOB_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$JOB_ID}"
TS_DATE="${RUN_DATE:-}"
if [[ -z "${TS_DATE}" ]]; then
  TS_DATE="$(date +%Y%m%d)"
fi
TS_TIME="${RUN_TIME:-}"
if [[ -z "${TS_TIME}" ]]; then
  TS_TIME="$(date +%H%M%S)"
fi
TS="${TS_DATE}_${TS_TIME}"
RUN_BASE="${RUN_BASE:-clg_ode_${TS}_job${ARRAY_JOB_ID}}"
if [[ -n "${FOLD_ID}" ]]; then
  RUN_NAME="${RUN_BASE}/fold${FOLD_ID}"
else
  RUN_NAME="${RUN_BASE}"
fi

MASTER_PORT="$("$PYTHON_HOST_BIN" - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
port = s.getsockname()[1]
s.close()
print(port)
PY
)"

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  "$TORCHRUN_BIN" --nproc_per_node 1 --master_port "$MASTER_PORT" -m scripts.train_clg_ode \
    --sc_dir "$SC_DIR" \
    --morph_root "$MORPH_ROOT" \
    --subject_info_csv "$SUBJECT_INFO" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    ${LATENT_DIM:+--latent_dim "$LATENT_DIM"} \
    ${HIDDEN_DIM:+--hidden_dim "$HIDDEN_DIM"} \
    ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
    ${MAX_EPOCHS:+--max_epochs "$MAX_EPOCHS"} \
    ${PATIENCE:+--patience "$PATIENCE"} \
    ${LEARNING_RATE:+--learning_rate "$LEARNING_RATE"} \
    ${SOLVER_STEPS:+--solver_steps "$SOLVER_STEPS"} \
    ${LAMBDA_KL:+--lambda_kl "$LAMBDA_KL"} \
    ${EDGE_LOSS:+--edge_loss "$EDGE_LOSS"} \
    ${EDGE_POS_WEIGHT:+--edge_pos_weight "$EDGE_POS_WEIGHT"} \
    ${FOCAL_GAMMA:+--focal_gamma "$FOCAL_GAMMA"} \
    ${FOCAL_ALPHA:+--focal_alpha "$FOCAL_ALPHA"} \
    ${LAMBDA_TOPO:+--lambda_topo "$LAMBDA_TOPO"} \
    ${LAMBDA_VEL:+--lambda_vel "$LAMBDA_VEL"} \
    ${LAMBDA_ACC:+--lambda_acc "$LAMBDA_ACC"} \
    ${LAMBDA_FULL_LOG_MSE:+--lambda_full_log_mse "$LAMBDA_FULL_LOG_MSE"} \
    ${LAMBDA_ZERO_LOG:+--lambda_zero_log "$LAMBDA_ZERO_LOG"} \
    ${ZERO_LOG_WARMUP_EPOCHS:+--zero_log_warmup_epochs "$ZERO_LOG_WARMUP_EPOCHS"} \
    ${ZERO_LOG_RAMP_EPOCHS:+--zero_log_ramp_epochs "$ZERO_LOG_RAMP_EPOCHS"} \
    ${LAMBDA_DELTA_LOG:+--lambda_delta_log "$LAMBDA_DELTA_LOG"} \
    ${LAMBDA_DENSITY:+--lambda_density "$LAMBDA_DENSITY"} \
    ${DENSITY_WARMUP_EPOCHS:+--density_warmup_epochs "$DENSITY_WARMUP_EPOCHS"} \
    ${DENSITY_RAMP_EPOCHS:+--density_ramp_epochs "$DENSITY_RAMP_EPOCHS"} \
    ${ADJACENT_PAIR_PROB:+--adjacent_pair_prob "$ADJACENT_PAIR_PROB"} \
    ${RESIDUAL_TAU:+--residual_tau "$RESIDUAL_TAU"} \
    ${RESIDUAL_NO_DT_GATE:+--residual_no_dt_gate} \
    ${RESIDUAL_CAP:+--residual_cap "$RESIDUAL_CAP"} \
    ${FIXED_SUPPORT:+--fixed_support} \
    ${INNOVATION_ENABLED:+--innovation_enabled} \
    ${INNOVATION_TOPM:+--innovation_topm "$INNOVATION_TOPM"} \
    ${INNOVATION_K_NEW:+--innovation_k_new "$INNOVATION_K_NEW"} \
    ${INNOVATION_TAU:+--innovation_tau "$INNOVATION_TAU"} \
    ${INNOVATION_DELTA_QUANTILE:+--innovation_delta_quantile "$INNOVATION_DELTA_QUANTILE"} \
    ${INNOVATION_DT_SCALE_YEARS:+--innovation_dt_scale_years "$INNOVATION_DT_SCALE_YEARS"} \
    ${INNOVATION_DT_OFFSET_MONTHS:+--innovation_dt_offset_months "$INNOVATION_DT_OFFSET_MONTHS"} \
    ${INNOVATION_DT_RAMP_MONTHS:+--innovation_dt_ramp_months "$INNOVATION_DT_RAMP_MONTHS"} \
    ${INNOVATION_FOCAL_GAMMA:+--innovation_focal_gamma "$INNOVATION_FOCAL_GAMMA"} \
    ${INNOVATION_FOCAL_ALPHA:+--innovation_focal_alpha "$INNOVATION_FOCAL_ALPHA"} \
    ${LAMBDA_NEW_SPARSE:+--lambda_new_sparse "$LAMBDA_NEW_SPARSE"} \
    ${NEW_SPARSE_WARMUP_EPOCHS:+--new_sparse_warmup_epochs "$NEW_SPARSE_WARMUP_EPOCHS"} \
    ${NEW_SPARSE_RAMP_EPOCHS:+--new_sparse_ramp_epochs "$NEW_SPARSE_RAMP_EPOCHS"} \
    ${LAMBDA_NEW_REG:+--lambda_new_reg "$LAMBDA_NEW_REG"} \
    ${INNOVATION_FREEZE_BACKBONE_AFTER:+--innovation_freeze_backbone_after "$INNOVATION_FREEZE_BACKBONE_AFTER"} \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
    ${SC_POS_EDGE_DROP_PROB:+--sc_pos_edge_drop_prob "$SC_POS_EDGE_DROP_PROB"} \
    ${MORPH_NOISE_SIGMA:+--morph_noise_sigma "$MORPH_NOISE_SIGMA"} \
    ${GRADNORM_SCOPE:+--gradnorm_scope "$GRADNORM_SCOPE"} \
    ${EARLY_STOP_METRIC:+--early_stop_metric "$EARLY_STOP_METRIC"} \
    ${EARLY_STOP_DENSITY_WEIGHT:+--early_stop_density_weight "$EARLY_STOP_DENSITY_WEIGHT"} \
    ${VAL_SC_EVAL_EVERY:+--val_sc_eval_every "$VAL_SC_EVAL_EVERY"} \
    ${COMPUTE_MASK_AUPRC:+--compute_mask_auprc} \
    ${DISABLE_S_MEAN:+--disable_s_mean} \
    ${DISABLE_TOPO_LOG_COMPRESS:+--disable_topo_log_compress} \
    ${RESIDUAL_SKIP:+--residual_skip} \
    "${FOLD_ARGS[@]}"
