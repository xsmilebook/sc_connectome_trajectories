import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


SITE_RE = re.compile(r"(site\d+)")


SES_PATTERNS: List[Tuple[str, str]] = [
    ("baselineYear1Arm1", "ses-baselineYear1Arm1"),
    ("BaselineYear1Arm1", "ses-baselineYear1Arm1"),
    ("baseline", "ses-baselineYear1Arm1"),
    ("2YearFollowUpYArm1", "ses-2YearFollowUpYArm1"),
    ("2_year_follow_up_y_arm_1", "ses-2YearFollowUpYArm1"),
    ("2year", "ses-2YearFollowUpYArm1"),
    ("4YearFollowUpYArm1", "ses-4YearFollowUpYArm1"),
    ("4_year_follow_up_y_arm_1", "ses-4YearFollowUpYArm1"),
    ("4year", "ses-4YearFollowUpYArm1"),
]


def infer_siteid(path: str) -> Optional[str]:
    m = SITE_RE.search(path)
    return m.group(1) if m else None


def infer_sesid(path: str) -> Optional[str]:
    for token, sesid in SES_PATTERNS:
        if token in path:
            return sesid
    return None


def infer_subid_from_filename(filename: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.startswith("Schaefer400_Morphology_"):
        base = base[len("Schaefer400_Morphology_") :]
    if base.startswith("sub-"):
        return base
    return None


def find_morphology_files(morph_root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(morph_root):
        return out
    for root, _, files in os.walk(morph_root):
        for f in files:
            if f.startswith("Schaefer400_Morphology_") and f.endswith(".csv"):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def build_morph_success_df(
    morph_files: List[str],
    subject_info_sc: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for p in morph_files:
        subid = infer_subid_from_filename(p)
        if not subid:
            continue
        sesid = infer_sesid(p)
        siteid = infer_siteid(p)
        if not sesid:
            continue
        scanid = f"{subid}_{sesid}"
        rows.append(
            {
                "subid": subid,
                "sesid": sesid,
                "scanid": scanid,
                "siteid": siteid if siteid else "",
                "morph_path": p,
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["scanid"], keep="first")

    cols_keep = ["subid", "sesid", "scanid", "siteid"]
    if df.empty:
        return df.reindex(columns=cols_keep)

    sc_cols = [c for c in ["scanid", "subid", "sesid", "siteid"] if c in subject_info_sc.columns]
    sc_lookup = subject_info_sc[sc_cols].drop_duplicates(subset=["scanid"])
    merged = df.merge(sc_lookup, on="scanid", how="left", suffixes=("", "_sc"))

    merged["subid"] = merged["subid"].fillna(merged.get("subid_sc"))
    merged["sesid"] = merged["sesid"].fillna(merged.get("sesid_sc"))
    merged["siteid"] = merged["siteid"].where(merged["siteid"].astype(str).str.len() > 0, merged.get("siteid_sc"))

    return merged[cols_keep].dropna(subset=["scanid"]).sort_values(["siteid", "subid", "sesid"])


def build_sc_without_morph_df(subject_info_sc: pd.DataFrame, morph_success: pd.DataFrame) -> pd.DataFrame:
    if "scanid" not in subject_info_sc.columns:
        raise ValueError("subject_info_sc.csv must contain a 'scanid' column")
    have = set(morph_success["scanid"].astype(str).tolist()) if not morph_success.empty else set()
    missing = subject_info_sc[~subject_info_sc["scanid"].astype(str).isin(have)].copy()
    keep = [c for c in ["subid", "sesid", "scanid", "siteid", "age", "sex"] if c in missing.columns]
    return missing[keep].sort_values(["siteid", "subid", "sesid"])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument(
        "--morph_root",
        type=str,
        default=os.path.join(project_root, "data", "processed", "morphology"),
    )
    p.add_argument(
        "--subject_info_sc",
        type=str,
        default=os.path.join(project_root, "data", "processed", "table", "subject_info_sc.csv"),
    )
    p.add_argument(
        "--out_success",
        type=str,
        default=os.path.join(project_root, "data", "processed", "table", "subject_info_morphology_success.csv"),
    )
    p.add_argument(
        "--out_missing",
        type=str,
        default=os.path.join(project_root, "data", "processed", "table", "subject_info_sc_without_morphology.csv"),
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    subject_info_sc = pd.read_csv(args.subject_info_sc)
    morph_files = find_morphology_files(args.morph_root)
    success = build_morph_success_df(morph_files, subject_info_sc)
    missing = build_sc_without_morph_df(subject_info_sc, success)

    os.makedirs(os.path.dirname(args.out_success), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_missing), exist_ok=True)
    success.to_csv(args.out_success, index=False)
    missing.to_csv(args.out_missing, index=False)


if __name__ == "__main__":
    main()
