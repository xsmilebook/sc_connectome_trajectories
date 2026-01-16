#!/bin/bash
#SBATCH -J clg_ode
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%A_%a.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%A_%a.err

set -euo pipefail

module load singularity

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
TORCHRUN_BIN="/opt/conda/bin/torchrun"
FOLD_ID="${SLURM_ARRAY_TASK_ID:-}"
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

MASTER_PORT="$(python3 - <<'PY'
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
    "${FOLD_ARGS[@]}"
