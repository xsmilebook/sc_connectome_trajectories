import os
import json
from typing import List, Tuple

import numpy as np

from src.data.utils import (
    parse_subject_session,
    load_matrix,
    compute_triu_indices,
)
from src.configs.paths import ensure_outputs_logs


def collect_session_pairs(sc_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    all_files = [f for f in os.listdir(sc_dir) if f.endswith(".csv")]
    ses_baseline = "ses-baselineYear1Arm1"
    ses_y2 = "ses-2YearFollowUpYArm1"
    ses_y4 = "ses-4YearFollowUpYArm1"
    subject_to_sessions = {}
    for fname in all_files:
        sid, ses = parse_subject_session(fname)
        fpath = os.path.join(sc_dir, fname)
        subject_to_sessions.setdefault(sid, {})[ses] = fpath
    pairs_y2: List[Tuple[str, str]] = []
    pairs_y4: List[Tuple[str, str]] = []
    for sid, ses_dict in subject_to_sessions.items():
        if ses_baseline in ses_dict and ses_y2 in ses_dict:
            pairs_y2.append((ses_dict[ses_baseline], ses_dict[ses_y2]))
        if ses_baseline in ses_dict and ses_y4 in ses_dict:
            pairs_y4.append((ses_dict[ses_baseline], ses_dict[ses_y4]))
    return pairs_y2, pairs_y4


def compute_metrics(pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    triu_idx = compute_triu_indices(400)
    all_true = []
    all_pred = []
    for base_path, foll_path in pairs:
        base = load_matrix(base_path, max_nodes=400)
        foll = load_matrix(foll_path, max_nodes=400)
        base_vec = base[triu_idx].astype(np.float32)
        foll_vec = foll[triu_idx].astype(np.float32)
        all_pred.append(base_vec)
        all_true.append(foll_vec)
    true_vec = np.stack(all_true, axis=0)
    pred_vec = np.stack(all_pred, axis=0)
    diff = pred_vec - true_vec
    mse = float(np.mean(diff ** 2))
    true_flat = true_vec.reshape(-1)
    pred_flat = pred_vec.reshape(-1)
    true_flat = true_flat - true_flat.mean()
    pred_flat = pred_flat - pred_flat.mean()
    num = float((true_flat * pred_flat).sum())
    denom = float(
        np.sqrt((true_flat ** 2).sum() + 1e-8)
        * np.sqrt((pred_flat ** 2).sum() + 1e-8)
    )
    pearson = num / denom if denom > 0 else float("nan")
    return mse, pearson


def main() -> None:
    ensure_outputs_logs()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sc_dir = os.path.join(project_root, "data", "processed", "sc_connectome", "schaefer400")
    pairs_y2, pairs_y4 = collect_session_pairs(sc_dir)
    mse_y2, pearson_y2 = compute_metrics(pairs_y2)
    mse_y4, pearson_y4 = compute_metrics(pairs_y4)
    results = {
        "n_pairs_y2": len(pairs_y2),
        "n_pairs_y4": len(pairs_y4),
        "baseline_vs_y2_mse": mse_y2,
        "baseline_vs_y2_pearson": pearson_y2,
        "baseline_vs_y4_mse": mse_y4,
        "baseline_vs_y4_pearson": pearson_y4,
    }
    out_path = os.path.join(project_root, "outputs", "results", "baseline_eval_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
