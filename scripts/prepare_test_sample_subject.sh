#!/bin/bash
set -euo pipefail

# Copy one subject's processed data (SC CSVs + morphology + subject-info rows) into a destination folder.
# This is intended for *test sample packaging* only. It does not modify original `data/`.
#
# Default destination is under `outputs/results/` (runtime artifacts; not tracked by Git).

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_test_sample_subject.sh [--subid SUBID] [--dest DIR] [--force]

Defaults:
  --subid: auto-pick the first subject found under data/processed/sc_connectome/schaefer400
  --dest : outputs/results/test_sample_subject/<SUBID>

Examples:
  bash scripts/prepare_test_sample_subject.sh
  bash scripts/prepare_test_sample_subject.sh --subid NDARINVXXXX --dest outputs/results/test_sample_subject/NDARINVXXXX
EOF
}

subid=""
dest=""
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subid)
      subid="${2:-}"
      shift 2
      ;;
    --dest)
      dest="${2:-}"
      shift 2
      ;;
    --force)
      force=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

sc_dir="data/processed/sc_connectome/schaefer400"
morph_dir="data/processed/morphology"
subject_info="data/processed/table/subject_info_sc.csv"

if [[ ! -d "$sc_dir" ]]; then
  echo "Missing SC dir: $sc_dir" >&2
  exit 1
fi

if [[ -z "$subid" ]]; then
  first_file="$(ls -1 "$sc_dir"/*.csv 2>/dev/null | head -n 1 || true)"
  if [[ -z "$first_file" ]]; then
    echo "No SC CSV files found under $sc_dir" >&2
    exit 1
  fi
  base="$(basename "$first_file")"
  if [[ "$base" == *_ses-* ]]; then
    subid="$(echo "$base" | sed -E 's/_ses-.*$//')"
  else
    subid="$(echo "$base" | sed -E 's/_.*$//')"
  fi
fi

if [[ -z "$subid" ]]; then
  echo "Failed to resolve subid" >&2
  exit 1
fi

if [[ -z "$dest" ]]; then
  dest="outputs/results/test_sample_subject/${subid}"
fi

if [[ -e "$dest" && "$force" -ne 1 ]]; then
  echo "Destination exists: $dest (use --force to overwrite)" >&2
  exit 1
fi

rm -rf "$dest"
mkdir -p "$dest/sc" "$dest/morphology" "$dest/table"

echo "Copy SC: ${subid}"
shopt -s nullglob
sc_files=( "$sc_dir/${subid}"_ses-*.csv )
if [[ ${#sc_files[@]} -eq 0 ]]; then
  # Fallback: any file starting with subid_
  sc_files=( "$sc_dir/${subid}"_*.csv )
fi
if [[ ${#sc_files[@]} -eq 0 ]]; then
  echo "No SC files found for subid=$subid under $sc_dir" >&2
  exit 1
fi
cp -f "${sc_files[@]}" "$dest/sc/"

echo "Copy morphology (if exists): ${subid}"
morph_file="$morph_dir/Schaefer400_Morphology_${subid}.csv"
if [[ -f "$morph_file" ]]; then
  cp -f "$morph_file" "$dest/morphology/"
else
  echo "WARNING: morphology file not found: $morph_file" >&2
fi

echo "Extract subject rows from subject_info_sc.csv (if exists)"
if [[ -f "$subject_info" ]]; then
  head -n 1 "$subject_info" > "$dest/table/subject_info_sc_sample.csv"
  awk -F',' -v sid="$subid" 'NR>1 && $1 ~ ("^" sid "_") {print $0}' "$subject_info" >> "$dest/table/subject_info_sc_sample.csv" || true
else
  echo "WARNING: subject info not found: $subject_info" >&2
fi

cat > "$dest/README.txt" <<EOF
Test sample subject package

- subid: ${subid}
- copied_at: $(date +%F' '%T)

Contents:
- sc/: SC CSV(s) for this subject
- morphology/: Schaefer400 morphology table (if available)
- table/subject_info_sc_sample.csv: subject_info rows matching this subid (scanid prefix)

Notes:
- This is a convenience package for testing/reporting only.
- Original data remain in ${sc_dir}, ${morph_dir}, ${subject_info}.
EOF

echo "Done: $dest"

