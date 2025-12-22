#!/bin/bash
set -euo pipefail

BIDS_ROOT="${BIDS_ROOT:-/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/raw_data/smri}"
SUBLIST="${1:-sublist_freesurfer_all.txt}"

find "${BIDS_ROOT}" -type f -name "*_T1w.nii" | sort > "${SUBLIST}"
N="$(wc -l < "${SUBLIST}")"

if [[ "${N}" -eq 0 ]]; then
  echo "ERROR: no T1w files found under ${BIDS_ROOT}" >&2
  exit 1
fi

sbatch --array=1-"${N}" "$(dirname "$0")/freesurfer.sh" "${SUBLIST}"


