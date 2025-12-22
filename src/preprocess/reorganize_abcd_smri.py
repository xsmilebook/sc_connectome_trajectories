#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


ALLOWED_SESSIONS = {
    "baselineYear1Arm1",
    "2YearFollowUpYArm1",
    "4YearFollowUpYArm1",
}
ALLOWED_MACHINES = {"SIEMENS", "Philips", "GE"}
SITE_PATTERN = re.compile(r"^site\d{1,2}$")
TGZ_PATTERN = re.compile(
    r"^(NDARINV[0-9A-Z]+)_(baselineYear1Arm1|2YearFollowUpYArm1|4YearFollowUpYArm1)_ABCD-T1-NORM_(\d+)\.tgz$"
)


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath(
        [abs_directory, abs_target]
    )


def safe_extract(tar, path):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal in tar: {member.name}")
    tar.extractall(path)


def iter_tgz_files(input_root):
    input_root = Path(input_root)
    for machine_dir in input_root.iterdir():
        if not machine_dir.is_dir():
            continue
        machine = machine_dir.name
        if machine not in ALLOWED_MACHINES:
            continue
        for site_dir in machine_dir.iterdir():
            if not site_dir.is_dir():
                continue
            siteid = site_dir.name
            if not SITE_PATTERN.match(siteid):
                continue
            for tgz_path in site_dir.glob("*.tgz"):
                yield machine, siteid, tgz_path


def find_anat_paths(extract_root, sesid):
    anat_dir = None
    for candidate in Path(extract_root).rglob("anat"):
        if candidate.is_dir() and f"ses-{sesid}" in str(candidate):
            anat_dir = candidate
            break
    if anat_dir is None:
        return None, None

    dcm_dir = None
    json_path = None
    for entry in anat_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("ABCD-T1-NORM_run-"):
            dcm_dir = entry
            json_candidate = anat_dir / f"{entry.name}.json"
            if json_candidate.exists():
                json_path = json_candidate
            break
    if json_path is None:
        for entry in anat_dir.glob("*.json"):
            json_path = entry
            break

    return dcm_dir, json_path


def parse_tgz_name(tgz_path):
    match = TGZ_PATTERN.match(tgz_path.name)
    if not match:
        return None, None
    subject = match.group(1)
    sesid = match.group(2)
    return f"sub-{subject}", sesid


def run_dcm2niix(dcm2niix, dcm_dir, out_dir, out_base, dry_run):
    cmd = [
        dcm2niix,
        "-b",
        "n",
        "-z",
        "n",
        "-f",
        out_base,
        "-o",
        str(out_dir),
        str(dcm_dir),
    ]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def process_tgz(
    tgz_path,
    machine,
    siteid,
    output_root,
    dcm2niix,
    work_dir,
    keep_extracted,
    skip_existing,
    dry_run,
):
    subid, sesid = parse_tgz_name(tgz_path)
    if subid is None or sesid is None:
        print(f"Skip unmatched file name: {tgz_path}")
        return

    dest_anat = Path(output_root) / sesid / machine / siteid / subid / "anat"
    dest_nii = dest_anat / f"{subid}_T1w.nii"
    dest_json = dest_anat / f"{subid}_T1w.json"
    if skip_existing and dest_nii.exists() and dest_json.exists():
        print(f"Skip existing: {dest_nii}")
        return

    if keep_extracted:
        extract_root = Path(work_dir or tgz_path.parent) / tgz_path.stem
        extract_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(dir=work_dir)
        extract_root = Path(temp_ctx.name)

    try:
        if dry_run:
            print(f"DRY RUN: extract {tgz_path} -> {extract_root}")
        else:
            with tarfile.open(tgz_path, "r:gz") as tar:
                safe_extract(tar, str(extract_root))

        dcm_dir, json_path = find_anat_paths(extract_root, sesid)
        if dcm_dir is None:
            print(f"Missing DICOM folder: {tgz_path}")
            return

        if dry_run:
            print(f"DRY RUN: mkdir {dest_anat}")
        else:
            dest_anat.mkdir(parents=True, exist_ok=True)

        run_dcm2niix(dcm2niix, dcm_dir, dest_anat, f"{subid}_T1w", dry_run)

        if json_path is None:
            print(f"Missing JSON sidecar: {tgz_path}")
        else:
            if dry_run:
                print(f"DRY RUN: copy {json_path} -> {dest_json}")
            else:
                shutil.copy2(json_path, dest_json)
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and reorganize ABCD sMRI tgz files into BIDS layout."
    )
    parser.add_argument(
        "--input-root",
        default="/ibmgpfs/cuizaixu_lab/Public_Data/ABCD_20221009/rawdata/sMRI",
        help="Input root containing {machine}/{siteid} folders.",
    )
    parser.add_argument(
        "--output-root",
        default="/GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/raw_data/smri",
        help="Output root for reorganized BIDS data.",
    )
    parser.add_argument(
        "--dcm2niix",
        default=None,
        help="Path to dcm2niix binary. Defaults to the one in the current conda env.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory for temporary extraction.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted folders instead of cleaning them up.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if target NIfTI and JSON already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing.",
    )
    args = parser.parse_args()
    if args.dcm2niix is None:
        dcm2niix_path = shutil.which("dcm2niix")
        if dcm2niix_path is None:
            raise FileNotFoundError(
                "dcm2niix not found in PATH; activate your conda env or pass --dcm2niix."
            )
        args.dcm2niix = dcm2niix_path

    for machine, siteid, tgz_path in iter_tgz_files(args.input_root):
        process_tgz(
            tgz_path=tgz_path,
            machine=machine,
            siteid=siteid,
            output_root=args.output_root,
            dcm2niix=args.dcm2niix,
            work_dir=args.work_dir,
            keep_extracted=args.keep_extracted,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
