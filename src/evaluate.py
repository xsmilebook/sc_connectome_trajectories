import os
import argparse
import json

import torch
from torch.utils.data import DataLoader

from models.vector_lstm import VectorLSTM
from engine.trainer import Trainer
from data.utils import list_subject_sequences, compute_triu_indices
from data.dataset import SCDataset, collate_sequences


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--sc_dir",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "ABCD",
            "sc_connectome",
            "schaefer400",
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(
            project_root,
            "data",
            "ABCD",
            "results",
        ),
    )
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sequences = list_subject_sequences(args.sc_dir)
    if len(sequences) == 0:
        return
    splits_path = os.path.join(args.results_dir, "vector_lstm_subject_splits.csv")
    if not os.path.exists(splits_path):
        return
    splits_df = __import__("pandas").read_csv(splits_path)
    subjects_all = [sid for sid, _ in sequences]
    idx_map = {sid: i for i, sid in enumerate(subjects_all)}
    test_indices = [
        idx_map[sid]
        for sid, s in zip(splits_df["subject_id"], splits_df["set"])
        if s == "test" and sid in idx_map
    ]
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

