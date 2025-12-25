from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


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

DEFAULT_METRICS = ["Thickness", "Area", "Curv", "GWC"]


def find_morphology_files(morph_root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(morph_root):
        return out
    for root, _, files in os.walk(morph_root):
        for fname in files:
            if fname.startswith("Schaefer400_Morphology_") and fname.endswith(".csv"):
                out.append(os.path.join(root, fname))
    out.sort()
    return out


def infer_sesid(path: str) -> Optional[str]:
    for token, sesid in SES_PATTERNS:
        if token in path:
            return sesid
    return None


def infer_subid(filename: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.startswith("Schaefer400_Morphology_"):
        base = base[len("Schaefer400_Morphology_") :]
    if base.startswith("sub-"):
        return base
    return None


def build_morph_index(morph_root: str) -> Dict[str, str]:
    morph_files = find_morphology_files(morph_root)
    index: Dict[str, str] = {}
    for p in morph_files:
        subid = infer_subid(p)
        sesid = infer_sesid(p)
        if not subid or not sesid:
            continue
        scanid = f"{subid}_{sesid}"
        if scanid not in index:
            index[scanid] = p
    return index


def infer_roi_metric_order(
    columns: Iterable[str],
    metrics: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    metrics = metrics or list(DEFAULT_METRICS)
    roi_set = set()
    metric_set = set()
    for col in columns:
        for metric in metrics:
            suffix = "_" + metric
            if col.endswith(suffix):
                roi_set.add(col[: -len(suffix)])
                metric_set.add(metric)
                break
    roi_order = sorted(roi_set)
    metric_order = [m for m in metrics if m in metric_set]
    return roi_order, metric_order


def load_morphology_matrix(
    path: str,
    roi_order: List[str],
    metric_order: List[str],
    max_nodes: Optional[int] = None,
) -> np.ndarray:
    df = pd.read_csv(path)
    if df.empty:
        return np.zeros((len(roi_order), len(metric_order)), dtype=np.float32)
    row = df.iloc[0]
    mat = np.zeros((len(roi_order), len(metric_order)), dtype=np.float32)
    roi_index = {roi: i for i, roi in enumerate(roi_order)}
    metric_index = {m: j for j, m in enumerate(metric_order)}
    for col, val in row.items():
        if col == "SubjectID":
            continue
        for metric in metric_order:
            suffix = "_" + metric
            if col.endswith(suffix):
                roi = col[: -len(suffix)]
                i = roi_index.get(roi)
                j = metric_index.get(metric)
                if i is not None and j is not None:
                    mat[i, j] = float(val)
                break
    if max_nodes is not None and mat.shape[0] > max_nodes:
        mat = mat[:max_nodes]
    return mat.astype(np.float32)
