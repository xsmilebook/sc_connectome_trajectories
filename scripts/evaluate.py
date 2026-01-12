import os
import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.vector_lstm import VectorLSTM
from src.engine.trainer import Trainer
from src.configs.paths import ensure_outputs_logs
from src.data.utils import (
    list_subject_sequences,
    compute_triu_indices,
    parse_subject_session,
    load_matrix,
)
from src.data.dataset import SCDataset, collate_sequences


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--sc_dir",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "processed",
            "sc_connectome",
            "schaefer400",
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            project_root,
            "outputs",
            "results",
            "vector_lstm",
        ),
    )
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--n_subjects", type=int, default=32)
    parser.add_argument(
        "--timepoint_eval",
        action="store_true",
    )
    return parser


def _compute_mse_pearson(true_list: List[np.ndarray], pred_list: List[np.ndarray]) -> Tuple[float, float]:
    if not true_list:
        return float("nan"), float("nan")
    true_vec = np.stack(true_list, axis=0)
    pred_vec = np.stack(pred_list, axis=0)
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


def _timepoint_eval(
    args: argparse.Namespace,
    sequences: List[Tuple[str, List[str]]],
    test_indices: List[int],
) -> None:
    if not test_indices:
        return
    triu_idx = compute_triu_indices(400)
    feature_dim = triu_idx[0].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorLSTM(
        input_dim=feature_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    results_json = os.path.join(args.results_dir, "vector_lstm_results.json")
    best_model_path = args.model_path
    if not best_model_path and os.path.exists(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        best_model_path = data.get("best_model_path", "")
    if best_model_path and os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    ses_baseline = "ses-baselineYear1Arm1"
    ses_y2 = "ses-2YearFollowUpYArm1"
    ses_y4 = "ses-4YearFollowUpYArm1"
    true_y2: List[np.ndarray] = []
    pred_y2: List[np.ndarray] = []
    true_y4: List[np.ndarray] = []
    pred_y4: List[np.ndarray] = []
    with torch.no_grad():
        for idx in test_indices:
            sid, paths = sequences[idx]
            if len(paths) < 2:
                continue
            mats = []
            sessions = []
            for p in paths:
                mat = load_matrix(p, max_nodes=400)
                mats.append(mat)
                _, ses = parse_subject_session(os.path.basename(p))
                sessions.append(ses)
            vecs = [mat[triu_idx].astype(np.float32) for mat in mats]
            seq = np.stack(vecs, axis=0)
            try:
                b_idx = sessions.index(ses_baseline)
            except ValueError:
                b_idx = -1
            try:
                y2_idx = sessions.index(ses_y2)
            except ValueError:
                y2_idx = -1
            try:
                y4_idx = sessions.index(ses_y4)
            except ValueError:
                y4_idx = -1
            if b_idx >= 0 and y2_idx == b_idx + 1:
                x_seq = seq[b_idx : b_idx + 1][None, :, :]
                x_tensor = torch.from_numpy(x_seq).to(device)
                y_pred = model(x_tensor).cpu().numpy()[0, 0]
                y_true = seq[y2_idx]
                pred_y2.append(y_pred)
                true_y2.append(y_true)
            if b_idx >= 0 and y2_idx >= 0 and y4_idx == y2_idx + 1:
                x_seq = seq[b_idx : y2_idx + 1][None, :, :]
                x_tensor = torch.from_numpy(x_seq).to(device)
                y_pred_seq = model(x_tensor).cpu().numpy()[0]
                step_idx = y2_idx - b_idx
                y_pred = y_pred_seq[step_idx]
                y_true = seq[y4_idx]
                pred_y4.append(y_pred)
                true_y4.append(y_true)
    mse_y2, pearson_y2 = _compute_mse_pearson(true_y2, pred_y2)
    mse_y4, pearson_y4 = _compute_mse_pearson(true_y4, pred_y4)
    out = {
        "n_pairs_y2": len(true_y2),
        "n_pairs_y4": len(true_y4),
        "vector_lstm_vs_y2_mse": mse_y2,
        "vector_lstm_vs_y2_pearson": pearson_y2,
        "vector_lstm_vs_y4_mse": mse_y4,
        "vector_lstm_vs_y4_pearson": pearson_y4,
    }
    out_path = os.path.join(args.results_dir, "vector_lstm_timepoint_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main() -> None:
    ensure_outputs_logs()
    parser = build_arg_parser()
    args = parser.parse_args()
    sequences = list_subject_sequences(args.sc_dir)
    if len(sequences) == 0:
        return
    splits_path = os.path.join(args.results_dir, "vector_lstm_subject_splits.csv")
    subjects_all = [sid for sid, _ in sequences]
    idx_map = {sid: i for i, sid in enumerate(subjects_all)}
    if os.path.exists(splits_path):
        splits_df = __import__("pandas").read_csv(splits_path)
        test_indices = [
            idx_map[sid]
            for sid, s in zip(splits_df["subject_id"], splits_df["set"])
            if s == "test" and sid in idx_map
        ]
    else:
        n = min(args.n_subjects, len(sequences))
        test_indices = list(range(n))
    if args.timepoint_eval:
        _timepoint_eval(args, sequences, test_indices)
        return
    if not test_indices:
        return
    triu_idx = compute_triu_indices(400)
    test_seqs = [sequences[i] for i in test_indices]
    ds = SCDataset(test_seqs, triu_idx)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    example_batch = next(iter(loader))
    feature_dim = example_batch["x"].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VectorLSTM(
        input_dim=feature_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    results_json = os.path.join(args.results_dir, "vector_lstm_results.json")
    best_model_path = args.model_path
    if not best_model_path and os.path.exists(results_json):
        with open(results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        best_model_path = data.get("best_model_path", "")
    if best_model_path and os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state)
    trainer = Trainer(
        sc_dir=args.sc_dir,
        results_dir=args.results_dir,
        model_class=VectorLSTM,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        max_epochs=1,
        patience=1,
        learning_rate=1e-4,
        random_state=args.random_state,
    )
    metrics = trainer._test_metrics(model, loader)
    out_path = os.path.join(args.results_dir, "vector_lstm_eval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

