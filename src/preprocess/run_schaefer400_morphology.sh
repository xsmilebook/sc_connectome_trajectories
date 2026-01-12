#!/bin/bash
#SBATCH --job-name=schaefer400_morph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=q_fat_c
#SBATCH --output=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/morph/schaefer400_morph_%A_%a.log
#SBATCH --error=/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/outputs/logs/morph/schaefer400_morph_%A_%a.err

set -euo pipefail

SUBJECTS_DIR_IN="${SUBJECTS_DIR:-}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.yaml}"

module load freesurfer/7.1.1

SUBLIST="${1:-sublist.txt}"
ATLAS_DIR="${ATLAS_DIR:-}"
SUBJECTS_DIR="${SUBJECTS_DIR_IN:-${FREESURFER_DATA:-${SUBJECTS_DIR:-}}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
export SUBJECTS_DIR

if [[ -z "${ATLAS_DIR}" ]]; then
  eval "$(python -m scripts.render_paths --config "${CONFIG_PATH}" --format bash --resolve --set \
    ATLAS_DIR=local.atlas.schaefer400_annot_dir \
  )"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  eval "$(python -m scripts.render_paths --config "${CONFIG_PATH}" --format bash --resolve --set \
    DATA_PROCESSED=local.data.processed \
  )"
  OUTPUT_DIR="${DATA_PROCESSED}/morphology"
fi

if [[ -z "${SUBJECTS_DIR}" ]]; then
  echo "ERROR: SUBJECTS_DIR is empty. Set SUBJECTS_DIR or FREESURFER_DATA." >&2
  exit 1
fi
if [[ -z "${ATLAS_DIR}" ]]; then
  echo "ERROR: ATLAS_DIR is empty. Set ATLAS_DIR to the folder containing lh/rh Schaefer .annot files." >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: OUTPUT_DIR is empty. Set OUTPUT_DIR to a writable folder (recommended under data/processed/ or outputs/results/)." >&2
  exit 1
fi
if [[ ! -f "${SUBLIST}" ]]; then
  echo "ERROR: sublist not found: ${SUBLIST}" >&2
  exit 1
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit as an array job." >&2
  echo "Example:" >&2
  echo "  N=\$(wc -l < sublist.txt)" >&2
  echo "  sbatch --array=1-\${N} run_schaefer400_morphology.sbatch sublist.txt" >&2
  exit 1
fi

SUBID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUBLIST}" | tr -d '\r' | xargs)"
if [[ -z "${SUBID}" ]]; then
  echo "ERROR: empty subject id at line ${SLURM_ARRAY_TASK_ID} in ${SUBLIST}" >&2
  exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
SITE_NAME="${SITE_NAME:-}"
OUTPUT_SITE="${OUTPUT_DIR}"
if [[ -n "${SITE_NAME}" ]]; then
  OUTPUT_SITE="${OUTPUT_DIR}/${SITE_NAME}"
fi
mkdir -p "${OUTPUT_SITE}"
OUT_CSV="${OUTPUT_SITE}/Schaefer400_Morphology_${SUBID}.csv"

python "${SCRIPT_DIR}/extract_schaefer400_morphology.py" \
  --subjects_dir "${SUBJECTS_DIR}" \
  --atlas_dir "${ATLAS_DIR}" \
  --subject_id "${SUBID}" \
  --output_dir "${OUTPUT_SITE}" \
  --out_csv "${OUT_CSV}"

