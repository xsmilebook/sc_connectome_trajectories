import os
import math
import json
from typing import List, Tuple, Dict, Any, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import SCDataset, collate_sequences
from data.utils import ensure_dir, compute_triu_indices, list_subject_sequences


class Trainer:
    def __init__(
        self,
        sc_dir: str,
        results_dir: str,
        model_class: Type[nn.Module],
        latent_dim: int = 512,
        batch_size: int = 4,
        max_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.sc_dir = sc_dir
        self.results_dir = results_dir
        ensure_dir(self.results_dir)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_class = model_class
        self.triu_idx = compute_triu_indices(400)
        self.summary: Dict[str, Any] = {}

    def _build_sequences(self) -> List[Tuple[str, List[str]]]:
        sequences = list_subject_sequences(self.sc_dir)
        return sequences

    def _split_outer(self, sequences: List[Tuple[str, List[str]]]) -> Tuple[List[int], List[int], List[str]]:
        subjects = [sid for sid, _ in sequences]
        indices = np.arange(len(sequences))
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=self.random_state,
        )
        for trainval_idx, test_idx in gss.split(indices, groups=subjects):
            trainval_idx_list = trainval_idx.tolist()
            test_idx_list = test_idx.tolist()
            break
        return trainval_idx_list, test_idx_list, subjects

    def _get_loaders_for_indices(
        self,
        sequences: List[Tuple[str, List[str]]],
        train_indices: List[int],
        val_indices: List[int],
    ) -> Tuple[DataLoader, DataLoader]:
        train_seqs = [sequences[i] for i in train_indices]
        val_seqs = [sequences[i] for i in val_indices]
        train_ds = SCDataset(train_seqs, self.triu_idx)
        val_ds = SCDataset(val_seqs, self.triu_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_sequences,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
        )
        return train_loader, val_loader

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            if not batch:
                continue
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["mask"].to(self.device)
            optimizer.zero_grad()
            y_pred = model(x)
            diff = (y_pred - y) ** 2
            diff = diff * mask.unsqueeze(-1)
            denom = mask.sum() * y.shape[-1]
            if denom.item() == 0:
                continue
            loss = diff.sum() / denom
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        if count == 0:
            return math.inf
        return total_loss / count

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)
                y_pred = model(x)
                diff = (y_pred - y) ** 2
                diff = diff * mask.unsqueeze(-1)
                denom = mask.sum() * y.shape[-1]
                if denom.item() == 0:
                    continue
                loss = diff.sum() / denom
                total_loss += loss.item()
                count += 1
        if count == 0:
            return math.inf
        return total_loss / count

    def _train_cv(
        self,
        sequences: List[Tuple[str, List[str]]],
        trainval_indices: List[int],
        subjects: List[str],
    ) -> str:
        groups = np.array([subjects[i] for i in trainval_indices])
        indices = np.array(trainval_indices)
        gkf = GroupKFold(n_splits=5)
        best_val = math.inf
        best_model_path = ""
        fold_results = []
        for fold_idx, (train_idx_rel, val_idx_rel) in enumerate(gkf.split(indices, groups=groups)):
            train_idx = indices[train_idx_rel].tolist()
            val_idx = indices[val_idx_rel].tolist()
            train_loader, val_loader = self._get_loaders_for_indices(sequences, train_idx, val_idx)
            example_item = next(iter(train_loader))
            feature_dim = example_item["x"].shape[-1]
            model = self.model_class(
                input_dim=feature_dim,
                latent_dim=self.latent_dim,
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            best_fold_val = math.inf
            best_state = None
            epochs_no_improve = 0
            print(f"Starting fold {fold_idx + 1}/5 with {len(train_idx)} train subjects and {len(val_idx)} val subjects")
            for epoch in range(self.max_epochs):
                train_loss = self._train_one_epoch(model, train_loader, optimizer)
                val_loss = self._evaluate(model, val_loader)
                print(
                    f"Fold {fold_idx + 1}/5, epoch {epoch + 1}/{self.max_epochs}, "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )
                if val_loss < best_fold_val:
                    best_fold_val = val_loss
                    best_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"Fold {fold_idx + 1}/5 early stopped at epoch {epoch + 1} "
                        f"with best_val_loss={best_fold_val:.4f}"
                    )
                    break
            fold_results.append(
                {"fold": fold_idx, "best_val_loss": float(best_fold_val)}
            )
            fold_model_path = os.path.join(
                self.results_dir,
                f"vector_lstm_fold{fold_idx}_best.pt",
            )
            if best_state is not None:
                torch.save(best_state, fold_model_path)
            if best_fold_val < best_val:
                best_val = best_fold_val
                best_model_path = fold_model_path
        self.summary["cv_folds"] = fold_results
        self.summary["best_val_loss"] = float(best_val)
        self.summary["best_model_path"] = best_model_path
        return best_model_path

    def _build_loader_from_indices(
        self,
        sequences: List[Tuple[str, List[str]]],
        indices: List[int],
        shuffle: bool,
    ) -> DataLoader:
        seqs = [sequences[i] for i in indices]
        ds = SCDataset(seqs, self.triu_idx)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_sequences,
        )
        return loader

    def _test_metrics(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:
        model.eval()
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].tolist()
                y_pred = model(x)
                for i, L in enumerate(lengths):
                    if L <= 0:
                        continue
                    true_last = y[i, L - 1].cpu().numpy()
                    pred_last = y_pred[i, L - 1].cpu().numpy()
                    all_true.append(true_last)
                    all_pred.append(pred_last)
        if not all_true:
            return {"mse": float("nan"), "pearson": float("nan")}
        true_vec = np.stack(all_true, axis=0).reshape(len(all_true), -1)
        pred_vec = np.stack(all_pred, axis=0).reshape(len(all_pred), -1)
        mse = mean_squared_error(true_vec, pred_vec)
        true_flat = true_vec.reshape(-1)
        pred_flat = pred_vec.reshape(-1)
        true_flat = true_flat - true_flat.mean()
        pred_flat = pred_flat - pred_flat.mean()
        num = float((true_flat * pred_flat).sum())
        denom = float(
            math.sqrt((true_flat ** 2).sum() + 1e-8)
            * math.sqrt((pred_flat ** 2).sum() + 1e-8)
        )
        pearson = num / denom if denom > 0 else float("nan")
        return {"mse": float(mse), "pearson": float(pearson)}

    def run(self) -> None:
        sequences = self._build_sequences()
        if len(sequences) == 0:
            return
        trainval_indices, test_indices, subjects = self._split_outer(sequences)
        best_model_path = self._train_cv(sequences, trainval_indices, subjects)
        test_loader = self._build_loader_from_indices(
            sequences, test_indices, shuffle=False
        )
        example_batch = next(iter(test_loader))
        feature_dim = example_batch["x"].shape[-1]
        model = self.model_class(
            input_dim=feature_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)
        if best_model_path and os.path.exists(best_model_path):
            state = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(state)
        test_metrics = self._test_metrics(model, test_loader)
        self.summary["test_metrics"] = test_metrics
        self.summary["n_subjects"] = len(sequences)
        self.summary["n_trainval"] = len(trainval_indices)
        self.summary["n_test"] = len(test_indices)
        subjects_array = np.array([sid for sid, _ in sequences])
        split_df = pd.DataFrame(
            {
                "subject_id": subjects_array,
                "set": [
                    "test" if i in set(test_indices) else "trainval"
                    for i in range(len(sequences))
                ],
            }
        )
        ensure_dir(self.results_dir)
        split_df.to_csv(
            os.path.join(self.results_dir, "vector_lstm_subject_splits.csv"),
            index=False,
        )
        with open(
            os.path.join(self.results_dir, "vector_lstm_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.summary, f, indent=2)

