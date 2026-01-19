#!/bin/bash
#SBATCH -J vector_lstm_baseline
#SBATCH -p q_ai8
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/vector_lstm_baseline/%A_%a.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/vector_lstm_baseline/%A_%a.err

set -euo pipefail

module load singularity

mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/vector_lstm_baseline

PYTHON_HOST_BIN="/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/python"
if [[ ! -x "$PYTHON_HOST_BIN" ]]; then
  echo "Conda python not found at $PYTHON_HOST_BIN; check your conda installation." >&2
  exit 1
fi

render_paths_output="$("$PYTHON_HOST_BIN" -m scripts.render_paths \
  --set CONTAINER=local.containers.torch_gnn \
        SC_DIR=local.data.sc_connectome_schaefer400 \
        MORPH_ROOT=local.data.morphology \
        RESULTS_DIR=local.outputs.vector_lstm_baseline \
  --resolve)" || {
  echo "Failed to resolve paths via scripts.render_paths; check Python environment." >&2
  exit 1
}

eval "$render_paths_output"

if [[ -z "${CONTAINER:-}" || -z "${SC_DIR:-}" || -z "${MORPH_ROOT:-}" || -z "${RESULTS_DIR:-}" ]]; then
  echo "Missing required path(s); check configs/paths.yaml and scripts.render_paths output." >&2
  exit 1
fi

PYTHON_BIN="/opt/conda/bin/python"
FOLD_ID="${SLURM_ARRAY_TASK_ID:-0}"
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
RUN_BASE="${RUN_BASE:-vector_lstm_${TS}_job${ARRAY_JOB_ID}}"
RUN_NAME="${RUN_BASE}/fold${FOLD_ID}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LATENT_DIM="${LATENT_DIM:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
RANDOM_STATE="${RANDOM_STATE:-42}"
TOPO_BINS="${TOPO_BINS:-32}"

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  "$PYTHON_BIN" -m scripts.train_vector_lstm_baseline \
    --config configs/paths.yaml \
    --sc_dir "$SC_DIR" \
    --morph_root "$MORPH_ROOT" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --cv_folds 5 \
    --cv_fold "$FOLD_ID" \
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --learning_rate "$LEARNING_RATE" \
    --random_state "$RANDOM_STATE" \
    --topo_bins "$TOPO_BINS"
