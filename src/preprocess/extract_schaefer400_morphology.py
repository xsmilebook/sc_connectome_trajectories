#!/usr/bin/env python3
"""
Extract Schaefer-400 (17 networks) morphology metrics from FreeSurfer recon-all outputs.

This script:
1) Maps Schaefer fsaverage .annot files to each subject using mri_surf2surf.
2) Runs mris_anatomical_stats to extract ROI-averaged metrics:
   - Thickness    (from ?h.thickness, via mris_anatomical_stats default output)
   - Surface Area (via mris_anatomical_stats default output)
   - Mean Curv    (via mris_anatomical_stats default output)
   - GWC          (ROI-averaged mean of ?h.w-g.pct.mgh, via mris_anatomical_stats -t)
3) Aggregates into a wide CSV: SubjectID, <ROI>_<Metric> ...

How to obtain the Schaefer fsaverage .annot files if you don't have them:
- The Schaefer2018 parcellations (including fsaverage .annot) are distributed in CBIG:
  https://github.com/ThomasYeoLab/CBIG
  Look under:
  stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/
  and choose the FreeSurfer fsaverage label/annot set matching your version (e.g., fsaverage).
- You should have files like:
  lh.Schaefer2018_400Parcels_17Networks_order.annot
  rh.Schaefer2018_400Parcels_17Networks_order.annot
  placed in a directory you pass as --atlas_dir (or via $ATLAS_DIR).

Example usage (single subject):
  module load freesurfer/7.1.1
  export SUBJECTS_DIR=/path/to/$FREESURFER_DATA
  export ATLAS_DIR=/path/to/schaefer/fsaverage/label
  python extract_schaefer400_morphology.py --subject_id sub-001 --out_csv Schaefer400_Morphology.csv

Example usage (all subjects under SUBJECTS_DIR):
  python extract_schaefer400_morphology.py --out_csv Schaefer400_Morphology.csv
"""

import argparse
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_ANNOT_BASENAME = "Schaefer2018_400Parcels_17Networks_order.annot"
SESSION_SUFFIX_RE = re.compile(r"_ses-[^_]+$")


def run_checked(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def normalize_subject_id(subject_id: str) -> str:
    return SESSION_SUFFIX_RE.sub("", subject_id.strip())


def detect_annot_paths(atlas_dir: str, annot_basename: str) -> Dict[str, str]:
    lh = os.path.join(atlas_dir, f"lh.{annot_basename}")
    rh = os.path.join(atlas_dir, f"rh.{annot_basename}")
    if not os.path.exists(lh):
        raise FileNotFoundError(f"Missing atlas annot: {lh}")
    if not os.path.exists(rh):
        raise FileNotFoundError(f"Missing atlas annot: {rh}")
    return {"lh": lh, "rh": rh}


def list_subjects(subjects_dir: str) -> List[str]:
    if not os.path.isdir(subjects_dir):
        raise FileNotFoundError(f"SUBJECTS_DIR not found: {subjects_dir}")
    ignore = {
        "fsaverage",
        "fsaverage6",
        "fsaverage5",
        "fsaverage4",
        "fsaverage3",
        "fsaverage_sym",
        "bert",
        "cvs_avg35",
        "cvs_avg35_inMNI152",
        "lh.EC_average",
        "rh.EC_average",
    }
    subjects = []
    for name in sorted(os.listdir(subjects_dir)):
        if name in ignore:
            continue
        p = os.path.join(subjects_dir, name)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "surf")):
            subjects.append(name)
    return subjects


def map_annot_to_subject(
    subjects_dir: str,
    output_dir: str,
    subject_id: str,
    hemi: str,
    src_annot: str,
    force: bool,
) -> str:
    subj_out_root = os.path.join(output_dir, subject_id)
    subj_label_dir = os.path.join(subj_out_root, "label")
    os.makedirs(subj_label_dir, exist_ok=True)
    out_annot = os.path.join(subj_label_dir, os.path.basename(src_annot))
    if os.path.exists(out_annot) and not force:
        return out_annot
    cmd = [
    "mri_surf2surf",
    "--sd", subjects_dir,
    "--srcsubject", "fsaverage",
    "--trgsubject", subject_id,
    "--hemi", hemi,
    "--sval-annot", src_annot,
    "--tval", out_annot,
    "--mapmethod", "nnf",
    ]
    run_checked(cmd)
    return out_annot


def find_gwc_file(subjects_dir: str, subject_id: str, hemi: str) -> str:
    cand = [
        os.path.join(subjects_dir, subject_id, "surf", f"{hemi}.w-g.pct.mgh"),
        os.path.join(subjects_dir, subject_id, "surf", f"{hemi}.w-g.pct.mgz"),
        os.path.join(subjects_dir, subject_id, "surf", f"{hemi}.w-g.pct"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Missing GWC file for {subject_id} {hemi}. Expected one of: {', '.join(cand)}"
    )


def parse_mris_anatomical_stats_table(stats_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(stats_path):
        raise FileNotFoundError(stats_path)
    col_headers: Optional[List[str]] = None
    rows: List[Dict[str, str]] = []
    with open(stats_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# ColHeaders"):
                parts = line.split()
                col_headers = parts[2:]
                continue
            if line.startswith("#"):
                continue
            if col_headers is None:
                continue
            parts = line.split()
            if len(parts) < len(col_headers):
                continue
            row = dict(zip(col_headers, parts[: len(col_headers)]))
            rows.append(row)
    if col_headers is None or not rows:
        raise ValueError(f"Could not parse stats table from: {stats_path}")
    return rows


def pick_first_key(d: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


def stats_to_roi_metrics(
    rows: List[Dict[str, str]],
    hemi: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        name = r.get("StructName")
        if not name or name == "unknown":
            continue
        if name.lower().startswith(("lh_", "rh_", "left_", "right_")):
            roi_name = name
        else:
            roi_name = f"{hemi.upper()}_{name}"

        thickness_key = pick_first_key(
            r, ["ThickAvg", "ThickAvg_mm", "MeanThickness", "ThickAvg_mm"]
        )
        area_key = pick_first_key(r, ["SurfArea", "Area_mm2", "SurfArea_mm2"])
        curv_key = pick_first_key(r, ["MeanCurv", "CurvMean", "Curv"])

        roi_metrics: Dict[str, float] = {}
        if thickness_key is not None:
            roi_metrics["Thickness"] = float(r[thickness_key])
        if area_key is not None:
            roi_metrics["Area"] = float(r[area_key])
        if curv_key is not None:
            roi_metrics["Curv"] = float(r[curv_key])
        if roi_metrics:
            out[roi_name] = roi_metrics
    return out


def gwc_stats_to_roi_mean(rows: List[Dict[str, str]], hemi: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in rows:
        name = r.get("StructName")
        if not name or name == "unknown":
            continue
        if name.lower().startswith(("lh_", "rh_", "left_", "right_")):
            roi_name = name
        else:
            roi_name = f"{hemi.upper()}_{name}"
        mean_key = pick_first_key(r, [
            "ThickAvg", "ThickAvg_mm", "MeanThickness", 
            "Mean", "MeanVal", "MeanValue"
        ])
        if mean_key is None:
            numeric_keys = [k for k in r.keys() if k.lower() in {"mean", "avg", "average"}]
            mean_key = numeric_keys[0] if numeric_keys else None
        if mean_key is None:
            continue
        out[roi_name] = float(r[mean_key])
    return out


def extract_subject_metrics(
    subjects_dir: str,
    output_dir: str,
    subject_id: str,
    atlas_annots: Dict[str, str],
    annot_basename: str,
    force_map: bool,
) -> Dict[str, float]:
    subject_id = normalize_subject_id(subject_id)
    out: Dict[str, float] = {}
    for hemi in ["lh", "rh"]:
        subj_annot = map_annot_to_subject(
            subjects_dir=subjects_dir,
            output_dir=output_dir,
            subject_id=subject_id,
            hemi=hemi,
            src_annot=atlas_annots[hemi],
            force=force_map,
        )

        subj_out_root = os.path.join(output_dir, subject_id)
        stats_dir = os.path.join(subj_out_root, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        base_stats = os.path.join(
            stats_dir,
            f"{hemi}.Schaefer400_17N.anatomical_stats.txt",
        )
        gwc_stats = os.path.join(
            stats_dir,
            f"{hemi}.Schaefer400_17N.gwc_stats.txt",
        )
        run_checked(
            [
                "mris_anatomical_stats",
                "-a",
                subj_annot,
                "-b",
                "-f",
                base_stats,
                subject_id,
                hemi,
            ]
        )
        gwc_file = find_gwc_file(subjects_dir, subject_id, hemi)
        run_checked(
            [
                "mris_anatomical_stats",
                "-a",
                subj_annot,
                "-b",
                "-t",
                gwc_file,
                "-f",
                gwc_stats,
                subject_id,
                hemi,
            ]
        )

        base_rows = parse_mris_anatomical_stats_table(base_stats)
        roi_metrics = stats_to_roi_metrics(base_rows, hemi=hemi)
        gwc_rows = parse_mris_anatomical_stats_table(gwc_stats)
        roi_gwc = gwc_stats_to_roi_mean(gwc_rows, hemi=hemi)

        for roi, mets in roi_metrics.items():
            for metric_name, metric_val in mets.items():
                out[f"{roi}_{metric_name}"] = metric_val
        for roi, gwc_mean in roi_gwc.items():
            out[f"{roi}_GWC"] = gwc_mean
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects_dir", type=str, default=os.environ.get("SUBJECTS_DIR", ""))
    p.add_argument("--output_dir", type=str, default=".", help="Directory to save intermediate label/stats files")
    p.add_argument("--atlas_dir", type=str, default=os.environ.get("ATLAS_DIR", ""))
    p.add_argument("--annot_basename", type=str, default=DEFAULT_ANNOT_BASENAME)
    p.add_argument("--subject_id", type=str, default="")
    p.add_argument("--subject_list", type=str, default="")
    p.add_argument("--out_csv", type=str, default="Schaefer400_Morphology.csv")
    p.add_argument("--force_map", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.subjects_dir:
        raise ValueError("SUBJECTS_DIR is required (set $SUBJECTS_DIR or pass --subjects_dir).")
    if not args.atlas_dir:
        raise ValueError("ATLAS_DIR is required (set $ATLAS_DIR or pass --atlas_dir).")
    atlas_annots = detect_annot_paths(args.atlas_dir, args.annot_basename)

    if args.subject_list:
        with open(args.subject_list, "r", encoding="utf-8") as f:
            subjects = [normalize_subject_id(x) for x in f.read().splitlines() if x.strip()]
    elif args.subject_id:
        subjects = [normalize_subject_id(args.subject_id)]
    else:
        subjects = list_subjects(args.subjects_dir)

    rows: List[Dict[str, float]] = []
    for sid in subjects:
        metrics = extract_subject_metrics(
            subjects_dir=args.subjects_dir,
            output_dir=args.output_dir,
            subject_id=sid,
            atlas_annots=atlas_annots,
            annot_basename=args.annot_basename,
            force_map=args.force_map,
        )
        metrics["SubjectID"] = sid
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if "SubjectID" in df.columns:
        cols = ["SubjectID"] + [c for c in df.columns if c != "SubjectID"]
        df = df[cols]
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

