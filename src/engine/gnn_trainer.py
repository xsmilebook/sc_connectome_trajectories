from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.morphology import build_morph_index
from src.data.utils import (
    compute_triu_indices,
    ensure_dir,
    list_subject_sequences,
    load_matrix,
    parse_subject_session,
    preprocess_sc,
)
from src.engine.sc_metrics import compute_sc_metrics, mean_metrics


@dataclass(frozen=True)
class GraphPair:
    subject_id: str
    src_path: str
    tgt_path: str


class GraphPairDataset(Dataset):
    def __init__(self, pairs: List[GraphPair], max_nodes: int) -> None:
        self.pairs = pairs
        self.max_nodes = max_nodes

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        a0 = load_matrix(pair.src_path, max_nodes=self.max_nodes)
        a1 = load_matrix(pair.tgt_path, max_nodes=self.max_nodes)
        a0_raw, _ = preprocess_sc(a0)
        a1_raw, _ = preprocess_sc(a1)
        return {
            "a_in": torch.from_numpy(a0_raw),
            "a_out": torch.from_numpy(a1_raw),
        }


def collate_graph_pairs(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    a_in = torch.stack([b["a_in"] for b in batch], dim=0)
    a_out = torch.stack([b["a_out"] for b in batch], dim=0)
    return {"a_in": a_in, "a_out": a_out}


class GNNTrainer:
    def __init__(
        self,
        sc_dir: str,
        morph_root: str,
        results_dir: str,
        model_class: type[nn.Module],
        model_kwargs: Dict[str, int] | None = None,
        batch_size: int = 2,
        max_epochs: int = 50,
        patience: int = 8,
        learning_rate: float = 1e-4,
        random_state: int = 42,
        topo_bins: int = 32,
        max_nodes: int = 400,
    ) -> None:
        self.sc_dir = sc_dir
        self.morph_root = morph_root
        self.results_dir = results_dir
        ensure_dir(self.results_dir)
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.topo_bins = topo_bins
        self.max_nodes = max_nodes
        self.triu_idx = compute_triu_indices(max_nodes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary: Dict[str, float | int | Dict[str, float] | List[Dict[str, float]]] = {}
        self.model: nn.Module | None = None

    @staticmethod
    def _scanid_from_path(path: str) -> str:
        base = os.path.basename(path)
        sid, ses = parse_subject_session(base)
        return f"{sid}_{ses}" if ses else sid

    def _build_sequences(self) -> List[Tuple[str, List[str]]]:
        sequences = list_subject_sequences(self.sc_dir, min_length=1)
        morph_index = build_morph_index(self.morph_root)
        filtered = []
        for sid, paths in sequences:
            kept = []
            for path in paths:
                scanid = self._scanid_from_path(path)
                if scanid in morph_index:
                    kept.append(path)
            if kept:
                filtered.append((sid, kept))
        return filtered

    def _split_outer(self, sequences: List[Tuple[str, List[str]]]) -> Tuple[List[int], List[int], List[str]]:
        subjects = [sid for sid, _ in sequences]
        indices = np.arange(len(sequences))
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=self.random_state,
        )
        for trainval_idx, test_idx in gss.split(indices, groups=subjects):
            return trainval_idx.tolist(), test_idx.tolist(), subjects
        return [], [], subjects

    @staticmethod
    def _pairs_for_sequences(
        sequences: List[Tuple[str, List[str]]],
        indices: List[int],
        include_singleton: bool,
    ) -> List[GraphPair]:
        pairs: List[GraphPair] = []
        for idx in indices:
            sid, paths = sequences[idx]
            if len(paths) >= 2:
                for i in range(len(paths) - 1):
                    pairs.append(GraphPair(subject_id=sid, src_path=paths[i], tgt_path=paths[i + 1]))
            elif include_singleton and len(paths) == 1:
                pairs.append(GraphPair(subject_id=sid, src_path=paths[0], tgt_path=paths[0]))
        return pairs

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=0.0)
        target = torch.clamp(target, min=0.0)
        pred_log = torch.log1p(pred)
        target_log = torch.log1p(target)
        pred_vec = pred_log[:, self.triu_idx[0], self.triu_idx[1]]
        target_vec = target_log[:, self.triu_idx[0], self.triu_idx[1]]
        diff = pred_vec - target_vec
        return torch.mean(diff ** 2)

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            if not batch:
                continue
            a_in = batch["a_in"].to(self.device)
            a_out = batch["a_out"].to(self.device)
            optimizer.zero_grad()
            pred = model(a_in)
            loss = self._loss(pred, a_out)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        return total / count if count > 0 else math.inf

    def _evaluate(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                a_in = batch["a_in"].to(self.device)
                a_out = batch["a_out"].to(self.device)
                pred = model(a_in)
                loss = self._loss(pred, a_out)
                total += float(loss.item())
                count += 1
        return total / count if count > 0 else math.inf

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
        best_state = None
        fold_results = []
        for fold_idx, (train_idx_rel, val_idx_rel) in enumerate(gkf.split(indices, groups=groups)):
            train_idx = indices[train_idx_rel].tolist()
            val_idx = indices[val_idx_rel].tolist()
            train_pairs = self._pairs_for_sequences(sequences, train_idx, include_singleton=False)
            val_pairs = self._pairs_for_sequences(sequences, val_idx, include_singleton=False)
            train_loader = DataLoader(
                GraphPairDataset(train_pairs, self.max_nodes),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_graph_pairs,
            )
            val_loader = DataLoader(
                GraphPairDataset(val_pairs, self.max_nodes),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_graph_pairs,
            )
            model = self.model_class(**self.model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            best_fold_val = math.inf
            best_fold_state = None
            epochs_no_improve = 0
            print(f"GNN fold {fold_idx + 1}/5: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs")
            for epoch in range(self.max_epochs):
                train_loss = self._train_one_epoch(model, train_loader, optimizer)
                val_loss = self._evaluate(model, val_loader)
                print(
                    f"GNN fold {fold_idx + 1}/5, epoch {epoch + 1}/{self.max_epochs}, "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )
                if val_loss < best_fold_val:
                    best_fold_val = val_loss
                    best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"GNN fold {fold_idx + 1}/5 early stopped at epoch {epoch + 1} "
                        f"with best_val_loss={best_fold_val:.4f}"
                    )
                    break
            fold_results.append({"fold": fold_idx, "best_val_loss": float(best_fold_val)})
            if best_fold_val < best_val:
                best_val = best_fold_val
                best_state = best_fold_state
        if best_state is None:
            raise RuntimeError("No training folds produced a model.")
        best_path = os.path.join(self.results_dir, "gnn_baseline_best.pt")
        torch.save(best_state, best_path)
        self.summary["cv_folds"] = fold_results
        self.summary["best_val_loss"] = float(best_val)
        self.summary["best_model_path"] = best_path
        return best_path

    def _test_metrics(
        self,
        sequences: List[Tuple[str, List[str]]],
        indices: List[int],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        pairs = self._pairs_for_sequences(sequences, indices, include_singleton=True)
        if not pairs:
            return {"mse": float("nan"), "pearson": float("nan")}, {}
        loader = DataLoader(
            GraphPairDataset(pairs, self.max_nodes),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_graph_pairs,
        )
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        sc_metrics_list: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                a_in = batch["a_in"].to(self.device)
                a_out = batch["a_out"].to(self.device)
                pred = self.model(a_in)
                pred_np = pred.detach().cpu().numpy()
                true_np = a_out.detach().cpu().numpy()
                for i in range(pred_np.shape[0]):
                    pred_mat = pred_np[i]
                    true_mat = true_np[i]
                    pred_vec = pred_mat[self.triu_idx[0], self.triu_idx[1]]
                    true_vec = true_mat[self.triu_idx[0], self.triu_idx[1]]
                    all_pred.append(pred_vec)
                    all_true.append(true_vec)
                    sc_metrics_list.append(
                        compute_sc_metrics(pred_mat, true_mat, self.triu_idx, self.topo_bins)
                    )
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
        return {"mse": float(mse), "pearson": float(pearson)}, mean_metrics(sc_metrics_list)

    def run(self) -> None:
        sequences = self._build_sequences()
        if not sequences:
            return
        trainval_indices, test_indices, subjects = self._split_outer(sequences)
        best_path = self._train_cv(sequences, trainval_indices, subjects)
        if best_path and os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device)
            self.model = self.model_class(**self.model_kwargs).to(self.device)
            self.model.load_state_dict(state)
        test_metrics, sc_metrics = self._test_metrics(sequences, test_indices)
        self.summary["test_metrics"] = test_metrics
        self.summary["test_sc_metrics"] = sc_metrics
        self.summary["n_subjects"] = len(sequences)
        self.summary["n_trainval"] = len(trainval_indices)
        self.summary["n_test"] = len(test_indices)
        ensure_dir(self.results_dir)
        with open(
            os.path.join(self.results_dir, "gnn_baseline_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.summary, f, indent=2)
        with open(
            os.path.join(self.results_dir, "test_sc_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(sc_metrics, f, indent=2)
