#!/bin/bash

set -euo pipefail

BASE_ROOT="${1:-/GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/baselineYear1Arm1/SIEMENS}"
SUBLIST_DIR="${2:-${PWD}/sublist_by_site}"
RUN_SCRIPT="${3:-${PWD}/run_schaefer400_morphology.sh}"

ATLAS_DIR="${ATLAS_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "${ATLAS_DIR}" ]]; then
  echo "ERROR: ATLAS_DIR is empty. Set ATLAS_DIR to the folder containing lh/rh Schaefer .annot files." >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "ERROR: OUTPUT_DIR is empty. Set OUTPUT_DIR to a writable results folder under ABCD." >&2
  exit 1
fi
if [[ ! -d "${BASE_ROOT}" ]]; then
  echo "ERROR: BASE_ROOT not found: ${BASE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "ERROR: run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${SUBLIST_DIR}"

for site_dir in "${BASE_ROOT}"/site*; do
  [[ -d "${site_dir}" ]] || continue
  site_name="$(basename "${site_dir}")"
  sublist="${SUBLIST_DIR}/${site_name}.txt"

  find "${site_dir}" -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n" | sort > "${sublist}"
  n="$(wc -l < "${sublist}" | tr -d ' ')"
  if [[ "${n}" -eq 0 ]]; then
    continue
  fi

  sbatch \
    --job-name="morph_${site_name}" \
    --export=ALL,SUBJECTS_DIR="${site_dir}",ATLAS_DIR="${ATLAS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",SITE_NAME="${site_name}" \
    --array=1-"${n}" \
    "${RUN_SCRIPT}" "${sublist}"
done

