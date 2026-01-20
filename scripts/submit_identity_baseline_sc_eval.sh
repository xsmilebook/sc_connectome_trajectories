#!/bin/bash
#SBATCH -J sc_id_base
#SBATCH -p q_ai8,q_ai4
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j.err

set -euo pipefail

module load singularity

PYTHON_HOST_BIN="/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/python"
if [[ ! -x "$PYTHON_HOST_BIN" ]]; then
  echo "Conda python not found at $PYTHON_HOST_BIN; check your conda installation." >&2
  exit 1
fi

render_paths_output="$("$PYTHON_HOST_BIN" -m scripts.render_paths \
  --set CONTAINER=local.containers.torch_gnn \
  --resolve)" || {
  echo "Failed to resolve paths via scripts.render_paths; check Python environment." >&2
  exit 1
}
eval "$render_paths_output"

if [[ -z "${CONTAINER:-}" ]]; then
  echo "Missing CONTAINER path; check configs/paths.yaml and scripts.render_paths output." >&2
  exit 1
fi

PYTHON_BIN="/opt/conda/bin/python"

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  "$PYTHON_BIN" -m scripts.identity_baseline_sc_eval \
    --random_state "${RANDOM_STATE:-42}" \
    --topo_bins "${TOPO_BINS:-32}" \
    --max_nodes "${MAX_NODES:-400}"

