#!/bin/bash
set -euo pipefail
#SBATCH -J clg_ode
#SBATCH -p q_ai4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o outputs/logs/%j.out
#SBATCH -e outputs/logs/%j.err

module load singularity

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found; load a Python 3 module before submitting." >&2
  exit 1
fi

render_paths_output="$(python3 -m scripts.render_paths \
  --set CONTAINER=local.containers.torch_gnn \
  --set SC_DIR=local.data.sc_connectome_schaefer400 \
  --set MORPH_ROOT=local.data.morphology \
  --set SUBJECT_INFO=local.data.subject_info_sc \
  --set RESULTS_DIR=local.outputs.clg_ode \
  --resolve)" || {
  echo "Failed to resolve paths via scripts.render_paths; check Python environment." >&2
  exit 1
}

eval "$render_paths_output"

if [[ -z "${CONTAINER:-}" || -z "${SC_DIR:-}" || -z "${MORPH_ROOT:-}" || -z "${SUBJECT_INFO:-}" || -z "${RESULTS_DIR:-}" ]]; then
  echo "Missing required path(s); check configs/paths.yaml and scripts.render_paths output." >&2
  exit 1
fi

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  python -m scripts.train_clg_ode \
    --sc_dir "$SC_DIR" \
    --morph_root "$MORPH_ROOT" \
    --subject_info_csv "$SUBJECT_INFO" \
    --results_dir "$RESULTS_DIR"
