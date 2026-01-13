#!/bin/bash
#SBATCH -J freesurfer
#SBATCH --cpus-per-task=8
#SBATCH -p q_fat_c
#SBATCH --output=/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/logs/freesurfer/freesurfer_%A_%a.log
#SBATCH --error=/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/logs/freesurfer/freesurfer_%A_%a.err

set -euo pipefail

module load freesurfer

CONFIG_PATH="${CONFIG_PATH:-configs/paths.yaml}"

# Local (repo-managed) defaults
eval "$(python -m scripts.render_paths --config "${CONFIG_PATH}" --format bash --resolve --set \
  OUTPUTS_LOGS=local.outputs.logs \
  DATA_PROCESSED=local.data.processed \
)"

# HPC-only FreeSurfer roots (do not migrate)
eval "$(python -m scripts.render_paths --config "${CONFIG_PATH}" --format bash --set \
  BIDS_ROOT=hpc.freesurfer.bids_root \
  FREESURFER_ROOT=hpc.freesurfer.freesurfer_root \
)"

NTHREADS="${NTHREADS:-8}"
SUBJECT_CSV="${SUBJECT_CSV:-${DATA_PROCESSED}/table/subject_info_sc_without_morphology.csv}"
mkdir -p "${OUTPUTS_LOGS}/freesurfer"

FREESURFER_FORCE="${FREESURFER_FORCE:-0}"

SUBLIST="${1:-${SUBLIST:-}}"
if [[ -z "${SUBLIST}" ]]; then
  SUBLIST="${OUTPUTS_LOGS}/freesurfer/sublist_freesurfer_from_csv.txt"
fi
if [[ ! -f "${SUBJECT_CSV}" ]]; then
  echo "ERROR: subject CSV not found: ${SUBJECT_CSV}" >&2
  exit 1
fi
if [[ ! -f "${SUBLIST}" ]]; then
  tmp_list="$(mktemp "${SUBLIST}.XXXXXX")"
  python - "${SUBJECT_CSV}" "${BIDS_ROOT}" > "${tmp_list}" << 'PY'
import csv
import sys
from pathlib import Path

csv_path = sys.argv[1]
bids_root = Path(sys.argv[2])

with open(csv_path, newline="") as handle:
    reader = csv.DictReader(handle)
    if "subid" not in reader.fieldnames or "sesid" not in reader.fieldnames:
        raise SystemExit("ERROR: CSV must contain subid and sesid columns.")
    wanted = set()
    for row in reader:
        subid = (row.get("subid") or "").strip()
        sesid = (row.get("sesid") or "").strip()
        if subid and sesid:
            wanted.add((subid, sesid.replace("ses-", "", 1)))

found = set()
for t1w in bids_root.rglob("*_T1w.nii"):
    try:
        rel = t1w.relative_to(bids_root)
    except ValueError:
        continue
    parts = rel.parts
    if len(parts) < 5:
        continue
    sesid = parts[0]
    subid = parts[3]
    if (subid, sesid) in wanted:
        print(str(t1w))
        found.add((subid, sesid))

missing = wanted - found
for subid, sesid in sorted(missing):
    print(f"Skip missing T1w for {subid} ses-{sesid}", file=sys.stderr)
PY
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

subj_root="${SUBJECTS_DIR}/${subid}"
done_file="${subj_root}/scripts/recon-all.done"

is_complete=0
if [[ -f "${done_file}" ]]; then
  if [[ -f "${subj_root}/stats/aseg.stats" && -f "${subj_root}/surf/lh.pial" && -f "${subj_root}/surf/rh.pial" ]]; then
    is_complete=1
  fi
fi

if [[ "${FREESURFER_FORCE}" -eq 1 ]]; then
  echo "Force re-run enabled (FREESURFER_FORCE=1): ${subid} (${sesid}/${machine}/${siteid})"
  rm -f "${done_file}" "${subj_root}/scripts/IsRunning"* 2>/dev/null || true
  is_complete=0
fi

if [[ "${is_complete}" -eq 1 ]]; then
  echo "Skip finished subject: ${subid} (${sesid}/${machine}/${siteid})"
  exit 0
elif [[ -f "${done_file}" ]]; then
  echo "Found recon-all.done but outputs look incomplete; re-running: ${subid} (${sesid}/${machine}/${siteid})"
  rm -f "${done_file}" "${subj_root}/scripts/IsRunning"* 2>/dev/null || true
fi

echo ""
echo "Running freesurfer on participant: ${subid} (${sesid}/${machine}/${siteid})"
echo ""

mkdir -p "${SUBJECTS_DIR}/${subid}/mri/orig"
mri_convert "${T1W_PATH}" "${SUBJECTS_DIR}/${subid}/mri/orig/001.mgz"
mri_convert "${T1W_PATH}" "${SUBJECTS_DIR}/${subid}/mri/orig.mgz"

recon-all -s "${subid}" -all -qcache -no-isrunning -openmp "${NTHREADS}"
