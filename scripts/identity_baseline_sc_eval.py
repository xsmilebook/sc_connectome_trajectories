import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.configs.paths import get_by_dotted_key, load_simple_yaml, resolve_repo_path


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
    ("6YearFollowUpYArm1", "ses-6YearFollowUpYArm1"),
    ("6_year_follow_up_y_arm_1", "ses-6YearFollowUpYArm1"),
    ("6year", "ses-6YearFollowUpYArm1"),
]


def infer_session_id(text: str) -> str:
    for token, sesid in SES_PATTERNS:
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


def compute_triu_indices(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.triu_indices(n_nodes, k=1)
    return idx[0], idx[1]


def _find_morphology_files(morph_root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(morph_root):
        return out
    for root, _, files in os.walk(morph_root):
        for fname in files:
            if fname.startswith("Schaefer400_Morphology_") and fname.endswith(".csv"):
                out.append(os.path.join(root, fname))
    out.sort()
    return out


def _infer_morph_sesid(path: str) -> str | None:
    for token, sesid in SES_PATTERNS:
        if token in path:
            return sesid
    return None


def _infer_morph_subid(path: str) -> str | None:
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("Schaefer400_Morphology_"):
        base = base[len("Schaefer400_Morphology_") :]
    if base.startswith("sub-"):
        return base
    return None


def build_morph_index(morph_root: str) -> Dict[str, str]:
    morph_files = _find_morphology_files(morph_root)
    index: Dict[str, str] = {}
    for path in morph_files:
        subid = _infer_morph_subid(path)
        sesid = _infer_morph_sesid(path)
        if not subid or not sesid:
            continue
        scanid = f"{subid}_{sesid}"
        if scanid not in index:
            index[scanid] = path
    return index


def compute_ecc(
    a_log: np.ndarray,
    k: int = 32,
    q_min: float = 0.05,
    q_max: float = 0.95,
) -> np.ndarray:
    n = a_log.shape[0]
    tri = np.triu_indices(n, k=1)
    vals = a_log[tri]
    pos = vals[vals > 0]
    if pos.size == 0:
        return np.zeros(k, dtype=np.float32)
    quantiles = np.linspace(q_min, q_max, k)
    thresholds = np.quantile(pos, quantiles).astype(np.float32)
    ecc = np.zeros(k, dtype=np.float32)
    for i, tau in enumerate(thresholds):
        b = (a_log >= tau).astype(np.float32)
        np.fill_diagonal(b, 0.0)
        e = float(np.sum(np.triu(b, k=1)))
        tri_count = float(np.trace(b @ b @ b) / 6.0)
        ecc[i] = float(n) - e + tri_count
    return ecc


def _pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def _sparsify_pred(
    pred_weight: np.ndarray,
    true_raw: np.ndarray,
    triu_idx: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray | None]:
    true_vec = true_raw[triu_idx[0], triu_idx[1]]
    pos_count = int((true_vec > 0).sum())
    if pos_count <= 0:
        return np.zeros_like(pred_weight), None
    pred_vec = pred_weight[triu_idx[0], triu_idx[1]]
    if pos_count >= pred_vec.shape[0]:
        mask = np.ones_like(pred_vec, dtype=bool)
    else:
        idx = np.argpartition(-pred_vec, pos_count - 1)[:pos_count]
        mask = np.zeros_like(pred_vec, dtype=bool)
        mask[idx] = True
    mask_float = mask.astype(pred_weight.dtype)
    mask_full = np.zeros_like(pred_weight)
    mask_full[triu_idx[0], triu_idx[1]] = mask_float
    mask_full = mask_full + mask_full.transpose(-1, -2)
    np.fill_diagonal(mask_full, 0.0)
    pred_sparse = pred_weight * mask_full
    return pred_sparse, mask


def compute_sc_metrics(
    pred_weight: np.ndarray,
    true_raw: np.ndarray,
    triu_idx: Tuple[np.ndarray, np.ndarray],
    topo_bins: int,
) -> Dict[str, float]:
    pred_log = np.log1p(pred_weight)
    true_log = np.log1p(true_raw)
    pred_vec = pred_log[triu_idx[0], triu_idx[1]]
    true_vec = true_log[triu_idx[0], triu_idx[1]]
    diff = pred_vec - true_vec
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = _pearsonr_np(pred_vec, true_vec)
    mask_pos = true_vec > 0
    if int(mask_pos.sum()) > 1:
        corr_pos = _pearsonr_np(pred_vec[mask_pos], true_vec[mask_pos])
    else:
        corr_pos = 0.0

    pred_sparse, mask = _sparsify_pred(pred_weight, true_raw, triu_idx)
    true_vec = true_raw[triu_idx[0], triu_idx[1]]
    corr_topk = 0.0
    if mask is not None and int(mask.sum()) > 1:
        corr_topk = _pearsonr_np(pred_vec[mask], true_vec[mask])
    pred_sparse_log = np.log1p(pred_sparse)
    pred_sparse_vec = pred_sparse_log[triu_idx[0], triu_idx[1]]
    corr_sparse = _pearsonr_np(pred_sparse_vec, true_vec)

    ecc_pred = compute_ecc(pred_sparse_log, k=topo_bins)
    ecc_true = compute_ecc(true_log, k=topo_bins)
    ecc_l2 = float(np.linalg.norm(ecc_pred - ecc_true))
    ecc_corr = _pearsonr_np(ecc_pred, ecc_true)
    return {
        "sc_log_mse": mse,
        "sc_log_mae": mae,
        "sc_log_pearson": corr,
        "sc_log_pearson_pos": corr_pos,
        "sc_log_pearson_topk": corr_topk,
        "sc_log_pearson_sparse": corr_sparse,
        "ecc_l2": ecc_l2,
        "ecc_pearson": ecc_corr,
    }


def _resolve_path(cfg: dict, dotted_key: str) -> str:
    value = get_by_dotted_key(cfg, dotted_key)
    return resolve_repo_path(str(value))


def _scanid_from_path(path: str) -> str:
    base = os.path.basename(path)
    sid, ses = parse_subject_session(base)
    return f"{sid}_{ses}" if ses else sid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute identity-mapping SC metrics on the CLG-ODE test split."
    )
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--topo_bins", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=400)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    sc_dir = _resolve_path(cfg, "local.data.sc_connectome_schaefer400")
    morph_root = _resolve_path(cfg, "local.data.morphology")

    sequences = list_subject_sequences(sc_dir, min_length=1)
    morph_index = build_morph_index(morph_root)
    filtered_sequences = []
    for sid, paths in sequences:
        kept = []
        for path in paths:
            scanid = _scanid_from_path(path)
            if scanid in morph_index:
                kept.append(path)
        if len(kept) >= 1:
            filtered_sequences.append((sid, kept))
    sequences = filtered_sequences
    subjects = [sid for sid, _ in sequences]
    n_subjects = len(subjects)
    n_test = int(np.ceil(0.2 * n_subjects))
    rng = np.random.RandomState(args.random_state)
    perm = rng.permutation(n_subjects)
    test_idx = perm[:n_test]
    test_sequences = [sequences[i] for i in test_idx]

    triu_idx = compute_triu_indices(args.max_nodes)
    sums = {
        "sc_log_mse": 0.0,
        "sc_log_mae": 0.0,
        "sc_log_pearson": 0.0,
        "sc_log_pearson_pos": 0.0,
        "sc_log_pearson_topk": 0.0,
        "sc_log_pearson_sparse": 0.0,
        "ecc_l2": 0.0,
        "ecc_pearson": 0.0,
    }
    count = 0

    for _, paths in test_sequences:
        if not paths:
            continue
        i = 0
        j = 1 if len(paths) >= 2 else 0
        a_i = load_matrix(paths[i], max_nodes=args.max_nodes)
        a_j = load_matrix(paths[j], max_nodes=args.max_nodes)
        a_i_raw, _ = preprocess_sc(a_i)
        a_j_raw, _ = preprocess_sc(a_j)
        metrics = compute_sc_metrics(a_i_raw, a_j_raw, triu_idx, args.topo_bins)
        for k, v in metrics.items():
            sums[k] += float(v)
        count += 1

    denom = count if count > 0 else 1.0
    averaged = {k: v / denom for k, v in sums.items()}
    print("identity_baseline_test_metrics")
    for key in sorted(averaged.keys()):
        print(f"{key}: {averaged[key]:.6f}")


if __name__ == "__main__":
    main()
