#!/bin/bash
#SBATCH -J freesurfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20gb
#SBATCH -p q_fat_c
#SBATCH -q high_c
#SBATCH --output=/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/logs/freesurfer/freesurfer_%A_%a.log
#SBATCH --error=/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/logs/freesurfer/freesurfer_%A_%a.err

set -euo pipefail

module load freesurfer

# User inputs
BIDS_ROOT="${BIDS_ROOT:-/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/raw_data/smri}"
FREESURFER_ROOT="${FREESURFER_ROOT:-/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/processed/freesurfer}"
NTHREADS="${NTHREADS:-8}"

SUBLIST="${1:-${SUBLIST:-}}"
if [[ -z "${SUBLIST}" ]]; then
  SUBLIST="${SLURM_SUBMIT_DIR:-$(pwd)}/sublist_freesurfer_all.txt"
  if [[ ! -f "${SUBLIST}" ]]; then
    tmp_list="$(mktemp "${SUBLIST}.XXXXXX")"
    find "${BIDS_ROOT}" -type f -name "*_T1w.nii" | sort > "${tmp_list}"
    mv "${tmp_list}" "${SUBLIST}"
  fi
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit as an array job." >&2
  echo "Example:" >&2
  echo "  N=\$(wc -l < \"${SUBLIST}\")" >&2
  echo "  sbatch --array=1-\${N} freesurfer.sh \"${SUBLIST}\"" >&2
  exit 1
fi

T1W_PATH="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUBLIST}" | tr -d '\r' | xargs)"
if [[ -z "${T1W_PATH}" ]]; then
  echo "ERROR: empty path at line ${SLURM_ARRAY_TASK_ID} in ${SUBLIST}" >&2
  exit 1
fi

rel_path="${T1W_PATH#${BIDS_ROOT}/}"
IFS='/' read -r sesid machine siteid subid _rest <<< "${rel_path}"
if [[ -z "${sesid}" || -z "${machine}" || -z "${siteid}" || -z "${subid}" ]]; then
  echo "ERROR: cannot parse path: ${T1W_PATH}" >&2
  exit 1
fi

SUBJECTS_DIR="${FREESURFER_ROOT}/${sesid}/${machine}/${siteid}"
export SUBJECTS_DIR
mkdir -p "${SUBJECTS_DIR}"

if [[ -f "${SUBJECTS_DIR}/${subid}/scripts/recon-all.done" ]]; then
  echo "Skip finished subject: ${subid} (${sesid}/${machine}/${siteid})"
  exit 0
fi

echo ""
echo "Running freesurfer on participant: ${subid} (${sesid}/${machine}/${siteid})"
echo ""

mkdir -p "${SUBJECTS_DIR}/${subid}/mri/orig"
mri_convert "${T1W_PATH}" "${SUBJECTS_DIR}/${subid}/mri/orig/001.mgz"
mri_convert "${T1W_PATH}" "${SUBJECTS_DIR}/${subid}/mri/orig.mgz"

recon-all -s "${subid}" -all -qcache -no-isrunning -openmp "${NTHREADS}"
