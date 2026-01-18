#!/bin/bash
#SBATCH -J vector_lstm_baseline
#SBATCH -p q_ai8
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/vector_lstm_baseline/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/vector_lstm_baseline/%j.err

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
    --latent_dim "$LATENT_DIM" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --learning_rate "$LEARNING_RATE" \
    --random_state "$RANDOM_STATE" \
    --topo_bins "$TOPO_BINS"
