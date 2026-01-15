import os
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def infer_session_id(text: str) -> str:
    patterns = [
        ("ses-baselineYear1Arm1", "ses-baselineYear1Arm1"),
        ("baselineYear1Arm1", "ses-baselineYear1Arm1"),
        ("BaselineYear1Arm1", "ses-baselineYear1Arm1"),
        ("baseline", "ses-baselineYear1Arm1"),
        ("ses-2YearFollowUpYArm1", "ses-2YearFollowUpYArm1"),
        ("2YearFollowUpYArm1", "ses-2YearFollowUpYArm1"),
        ("2_year_follow_up_y_arm_1", "ses-2YearFollowUpYArm1"),
        ("2year", "ses-2YearFollowUpYArm1"),
        ("ses-4YearFollowUpYArm1", "ses-4YearFollowUpYArm1"),
        ("4YearFollowUpYArm1", "ses-4YearFollowUpYArm1"),
        ("4_year_follow_up_y_arm_1", "ses-4YearFollowUpYArm1"),
        ("4year", "ses-4YearFollowUpYArm1"),
        ("ses-6YearFollowUpYArm1", "ses-6YearFollowUpYArm1"),
        ("6YearFollowUpYArm1", "ses-6YearFollowUpYArm1"),
        ("6_year_follow_up_y_arm_1", "ses-6YearFollowUpYArm1"),
        ("6year", "ses-6YearFollowUpYArm1"),
    ]
    for token, sesid in patterns:
        if token in text:
            return sesid
    return ""


def parse_subject_session(filename: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    if "_ses-" in base:
        parts = base.split("_ses-")
        subject_id = parts[0]
        session_id = "ses-" + parts[1]
    else:
        subject_id = base.split("_", 1)[0]
        if not subject_id:
            subject_id = base
        if subject_id.startswith("sub-") is False:
            m = re.search(r"(sub-[A-Za-z0-9]+)", base)
            if m:
                subject_id = m.group(1)
            elif re.fullmatch(r"NDARINV[A-Za-z0-9]+", subject_id):
                subject_id = "sub-" + subject_id
        session_id = infer_session_id(base)
    return subject_id, session_id


def session_sort_key(session_id: str) -> Tuple[int, str]:
    priority = {
        "ses-baselineYear1Arm1": 0,
        "ses-2YearFollowUpYArm1": 1,
        "ses-4YearFollowUpYArm1": 2,
        "ses-6YearFollowUpYArm1": 3,
    }
    if session_id in priority:
        return priority[session_id], session_id
    return 999, session_id


def list_subject_sequences(sc_dir: str, min_length: int = 2) -> List[Tuple[str, List[str]]]:
    all_files = [f for f in os.listdir(sc_dir) if f.endswith(".csv")]
    subject_to_sessions: Dict[str, List[Tuple[str, str]]] = {}
    for fname in all_files:
        fpath = os.path.join(sc_dir, fname)
        sid, ses = parse_subject_session(fname)
        subject_to_sessions.setdefault(sid, []).append((ses, fpath))
    sequences: List[Tuple[str, List[str]]] = []
    for sid, ses_list in subject_to_sessions.items():
        ses_list_sorted = sorted(ses_list, key=lambda x: session_sort_key(x[0]))
        paths = [p for _, p in ses_list_sorted]
        sequences.append((sid, paths))
    sequences = [s for s in sequences if len(s[1]) >= min_length]
    sequences.sort(key=lambda x: x[0])
    return sequences


def load_matrix(path: str, max_nodes: int = 400) -> np.ndarray:
    arr = pd.read_csv(path, header=None).values
    n0, n1 = arr.shape
    n = min(max_nodes, n0, n1)
    arr = arr[:n, :n].astype(np.float32)
    return arr


def preprocess_sc(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = matrix.astype(np.float32, copy=True)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    a_log = np.log1p(a)
    return a, a_log


def compute_strengths(a: np.ndarray, eps: float = 1e-8) -> Tuple[float, float]:
    triu_idx = np.triu_indices(a.shape[0], k=1)
    vals = a[triu_idx]
    total = float(np.sum(vals))
    s = float(np.log(total + eps))
    pos = vals[vals > 0]
    if pos.size == 0:
        s_mean = float(np.log(eps))
    else:
        s_mean = float(np.log(float(np.mean(pos)) + eps))
    return s, s_mean


def flatten_upper_triangle(mat: np.ndarray, triu_idx: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return mat[triu_idx].astype(np.float32)


def compute_triu_indices(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.triu_indices(n_nodes, k=1)
    return idx[0], idx[1]
