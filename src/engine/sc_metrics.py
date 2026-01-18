from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.data.topology import compute_ecc


def vector_to_matrix(
    vec: np.ndarray,
    triu_idx: Tuple[np.ndarray, np.ndarray],
    n_nodes: int,
) -> np.ndarray:
    mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    mat[triu_idx[0], triu_idx[1]] = vec.astype(np.float32)
    mat = mat + mat.T
    np.fill_diagonal(mat, 0.0)
    return mat


def sparsify_pred(
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


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def compute_sc_metrics(
    pred_weight: np.ndarray,
    true_raw: np.ndarray,
    triu_idx: Tuple[np.ndarray, np.ndarray],
    topo_bins: int,
) -> Dict[str, float]:
    pred_weight = np.clip(pred_weight, 0.0, None)
    true_raw = np.clip(true_raw, 0.0, None)
    pred_log = np.log1p(pred_weight)
    true_log = np.log1p(true_raw)
    pred_vec = pred_log[triu_idx[0], triu_idx[1]]
    true_vec = true_log[triu_idx[0], triu_idx[1]]
    diff = pred_vec - true_vec
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = pearsonr_np(pred_vec, true_vec)
    mask_pos = true_vec > 0
    if int(mask_pos.sum()) > 1:
        corr_pos = pearsonr_np(pred_vec[mask_pos], true_vec[mask_pos])
    else:
        corr_pos = 0.0

    pred_sparse, mask = sparsify_pred(pred_weight, true_raw, triu_idx)
    true_vec_raw = true_raw[triu_idx[0], triu_idx[1]]
    corr_topk = 0.0
    if mask is not None and int(mask.sum()) > 1:
        corr_topk = pearsonr_np(pred_vec[mask], true_vec_raw[mask])
    pred_sparse_log = np.log1p(pred_sparse)
    pred_sparse_vec = pred_sparse_log[triu_idx[0], triu_idx[1]]
    corr_sparse = pearsonr_np(pred_sparse_vec, true_vec_raw)

    ecc_pred = compute_ecc(pred_sparse_log, k=topo_bins)
    ecc_true = compute_ecc(true_log, k=topo_bins)
    ecc_l2 = float(np.linalg.norm(ecc_pred - ecc_true))
    ecc_corr = pearsonr_np(ecc_pred, ecc_true)
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


def mean_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in metrics_list]
        out[key] = float(np.mean(vals))
    return out
