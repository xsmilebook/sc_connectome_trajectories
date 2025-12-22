#!/bin/bash
#SBATCH -J freesurfer
#SBATCH --cpus-per-task=4
#SBATCH -p q_fat
#SBATCH --output=/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/logs/freesurfer/freesurfer_%A_%a.log
#SBATCH --error=/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/logs/freesurfer/freesurfer_%A_%a.err

set -euo pipefail

module load freesurfer

# User inputs
BIDS_ROOT="${BIDS_ROOT:-/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/raw_data/smri}"
FREESURFER_ROOT="${FREESURFER_ROOT:-/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/processed/freesurfer}"
NTHREADS="${NTHREADS:-8}"
SUBJECT_CSV="${SUBJECT_CSV:-/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/subject_info_sc_without_morphology.csv}"

SUBLIST="${1:-${SUBLIST:-}}"
if [[ -z "${SUBLIST}" ]]; then
  SUBLIST="${SLURM_SUBMIT_DIR:-$(pwd)}/sublist_freesurfer_from_csv.txt"
fi
if [[ ! -f "${SUBJECT_CSV}" ]]; then
  echo "ERROR: subject CSV not found: ${SUBJECT_CSV}" >&2
  exit 1
fi
if [[ ! -f "${SUBLIST}" ]]; then
  tmp_list="$(mktemp "${SUBLIST}.XXXXXX")"
  python - "${SUBJECT_CSV}" << 'PY' | while IFS=$'\t' read -r subid sesid; do
import csv
import sys

csv_path = sys.argv[1]
with open(csv_path, newline="") as handle:
    reader = csv.DictReader(handle)
    if "subid" not in reader.fieldnames or "sesid" not in reader.fieldnames:
        raise SystemExit("ERROR: CSV must contain subid and sesid columns.")
    for row in reader:
        subid = (row.get("subid") or "").strip()
        sesid = (row.get("sesid") or "").strip()
        if subid and sesid:
            print(f"{subid}\t{sesid}")
PY
    ses="${sesid#ses-}"
    match="$(find "${BIDS_ROOT}" -type f -path "*/${ses}/*/*/${subid}/anat/${subid}_T1w.nii" | sort | head -n 1)"
    if [[ -n "${match}" ]]; then
      echo "${match}" >> "${tmp_list}"
    else
      echo "Skip missing T1w for ${subid} ${sesid}" >&2
    fi
  done
  mv "${tmp_list}" "${SUBLIST}"
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  N="$(wc -l < "${SUBLIST}")"
  if [[ "${N}" -eq 0 ]]; then
    echo "ERROR: no T1w files found from CSV ${SUBJECT_CSV}" >&2
    exit 1
  fi
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Submit as an array job." >&2
  echo "Example:" >&2
  echo "  sbatch --array=1-${N} ${0} \"${SUBLIST}\"" >&2
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
