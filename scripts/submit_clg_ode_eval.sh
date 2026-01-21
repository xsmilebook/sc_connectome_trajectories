#!/bin/bash
#SBATCH -J clg_eval
#SBATCH -p q_ai8,q_ai4
#SBATCH --gres=gpu:1
#SBATCH -D /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories
#SBATCH -o /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j_eval.out
#SBATCH -e /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/clg_ode/%j_eval.err

set -euo pipefail

module load singularity

if [[ -z "${EVAL_RUN_DIR:-}" ]]; then
  echo "EVAL_RUN_DIR is required (e.g., outputs/results/clg_ode/runs/<run>/fold0)" >&2
  exit 1
fi

PYTHON_HOST_BIN="/GPFS/cuizaixu_lab_permanent/xuhaoshu/miniconda3/bin/python"
if [[ ! -x "$PYTHON_HOST_BIN" ]]; then
  echo "Conda python not found at $PYTHON_HOST_BIN; check your conda installation." >&2
  exit 1
fi

render_paths_output="$("$PYTHON_HOST_BIN" -m scripts.render_paths \
  --set CONTAINER=local.containers.torch_gnn \
  --resolve)" || {
  echo "Failed to resolve container path via scripts.render_paths; check Python environment." >&2
  exit 1
}

eval "$render_paths_output"

if [[ -z "${CONTAINER:-}" ]]; then
  echo "Missing CONTAINER; check configs/paths.yaml and scripts.render_paths output." >&2
  exit 1
fi

OUT_FILE="${EVAL_OUT_FILE:-test_sc_metrics_ext.json}"

singularity exec --nv \
  --bind /ibmgpfs:/ibmgpfs \
  --bind /GPFS:/GPFS \
  "$CONTAINER" \
  /opt/conda/bin/python -m scripts.eval_clg_ode_run \
    --run_dir "$EVAL_RUN_DIR" \
    --out_file "$OUT_FILE"

