#!/bin/bash
#SBATCH -J clg_ode_smoke
#SBATCH -p q_ai8
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

module load singularity

mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found; load a Python 3 module before submitting." >&2
  exit 1
fi

render_paths_output="$(python3 -m scripts.render_paths \
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
JOB_ID="${SLURM_JOB_ID:-0}"
RUN_TAG="${RUN_TAG:-smoke}"
FOLD_ID="${FOLD_ID:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOPO_SCALE_Q="${TOPO_SCALE_Q:-0.9}"
TOPO_WARMUP_FRAC="${TOPO_WARMUP_FRAC:-0.2}"
RUN_NAME="clg_ode_${RUN_TAG}_job${JOB_ID}"

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
  "$PYTHON_BIN" -m scripts.train_clg_ode \
    --sc_dir "$SC_DIR" \
    --morph_root "$MORPH_ROOT" \
    --subject_info_csv "$SUBJECT_INFO" \
    --results_dir "$RESULTS_DIR" \
    --run_name "$RUN_NAME" \
    --cv_fold "$FOLD_ID" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --topo_scale_quantile "$TOPO_SCALE_Q" \
    --topo_warmup_frac "$TOPO_WARMUP_FRAC"
