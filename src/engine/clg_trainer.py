from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.clg_dataset import CLGDataset, collate_clg_sequences
from src.data.utils import compute_triu_indices, ensure_dir
from src.engine.losses import topology_loss
from src.models.clg_ode import CLGODE


class CLGTrainer:
    def __init__(
        self,
        sc_dir: str,
        morph_root: str,
        subject_info_csv: str,
        results_dir: str,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        batch_size: int = 2,
        max_epochs: int = 80,
        patience: int = 10,
        learning_rate: float = 1e-4,
        random_state: int = 42,
        lambda_kl: float = 1e-3,
        lambda_topo: float = 0.0,
        lambda_smooth: float = 1e-2,
        topk: int = 20,
        solver_steps: int = 8,
    ) -> None:
        self.sc_dir = sc_dir
        self.morph_root = morph_root
        self.subject_info_csv = subject_info_csv
        self.results_dir = results_dir
        ensure_dir(self.results_dir)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.lambda_kl = lambda_kl
        self.lambda_topo = lambda_topo
        self.lambda_smooth = lambda_smooth
        self.topk = topk
        self.solver_steps = solver_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary: Dict[str, Any] = {}

    def _build_dataset(self) -> CLGDataset:
        return CLGDataset(
            sc_dir=self.sc_dir,
            morph_root=self.morph_root,
            subject_info_csv=self.subject_info_csv,
        )

    def _split_outer(self, subjects: List[str]) -> Tuple[List[int], List[int]]:
        indices = np.arange(len(subjects))
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=self.random_state,
        )
        for trainval_idx, test_idx in gss.split(indices, groups=subjects):
            return trainval_idx.tolist(), test_idx.tolist()
        return [], []

    def _get_loader(self, dataset: CLGDataset, indices: List[int], shuffle: bool) -> DataLoader:
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_clg_sequences,
        )

    def _build_model(self, dataset: CLGDataset) -> CLGODE:
        num_nodes = dataset.max_nodes
        morph_dim = len(dataset.metric_order)
        return CLGODE(
            num_nodes=num_nodes,
            morph_dim=morph_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            sex_dim=dataset.sex_vocab_size,
            site_dim=dataset.site_vocab_size,
            cov_embed_dim=8,
            topk=self.topk,
            solver_steps=self.solver_steps,
        )

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            if not batch:
                continue
            loss = self._compute_loss(model, batch, triu_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
        if count == 0:
            return math.inf
        return total_loss / count

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                loss = self._compute_loss(model, batch, triu_idx)
                total_loss += float(loss.item())
                count += 1
        if count == 0:
            return math.inf
        return total_loss / count

    def _compute_loss(
        self,
        model: CLGODE,
        batch: Dict[str, torch.Tensor],
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        a = batch["a"].to(self.device)
        x = batch["x"].to(self.device)
        times = batch["times"].to(self.device)
        mask = batch["mask"].to(self.device)
        sex = batch["sex"].to(self.device)
        site = batch["site"].to(self.device)
        a0 = a[:, 0]
        x0 = x[:, 0]
        outputs = model(a0, x0, times, sex, site)
        a_hat = outputs.a_hat
        x_hat = outputs.x_hat

        a_true = a[:, :, triu_idx[0], triu_idx[1]]
        a_pred = a_hat[:, :, triu_idx[0], triu_idx[1]]
        loss_a = masked_mse(a_pred, a_true, mask)
        loss_x = masked_mse(x_hat, x, mask)
        loss_recon = loss_a + loss_x

        kl_morph = kl_divergence(outputs.mu_morph, outputs.logvar_morph)
        kl_conn = kl_divergence(outputs.mu_conn, outputs.logvar_conn)
        loss_kl = 0.5 * (kl_morph + kl_conn)

        loss_topo = topology_loss(a_hat, a, mask)
        loss_smooth = smoothness_loss(outputs.z_morph, outputs.z_conn, mask)

        return (
            loss_recon
            + self.lambda_kl * loss_kl
            + self.lambda_topo * loss_topo
            + self.lambda_smooth * loss_smooth
        )

    def _train_cv(
        self,
        dataset: CLGDataset,
        trainval_indices: List[int],
        subjects: List[str],
    ) -> str:
        groups = np.array([subjects[i] for i in trainval_indices])
        indices = np.array(trainval_indices)
        gkf = GroupKFold(n_splits=5)
        triu_idx = compute_triu_indices(dataset.max_nodes)
        best_val = math.inf
        best_model_path = ""
        fold_results = []
        for fold_idx, (train_idx_rel, val_idx_rel) in enumerate(gkf.split(indices, groups=groups)):
            train_idx = indices[train_idx_rel].tolist()
            val_idx = indices[val_idx_rel].tolist()
            train_loader = self._get_loader(dataset, train_idx, shuffle=True)
            val_loader = self._get_loader(dataset, val_idx, shuffle=False)
            model = self._build_model(dataset).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            best_fold_val = math.inf
            best_state = None
            epochs_no_improve = 0
            print(
                f"Starting fold {fold_idx + 1}/5 with {len(train_idx)} train subjects and {len(val_idx)} val subjects"
            )
            for epoch in range(self.max_epochs):
                train_loss = self._train_one_epoch(model, train_loader, optimizer, triu_idx)
                val_loss = self._evaluate(model, val_loader, triu_idx)
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
            fold_results.append({"fold": fold_idx, "best_val_loss": float(best_fold_val)})
            fold_model_path = os.path.join(
                self.results_dir,
                f"clg_ode_fold{fold_idx}_best.pt",
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

    def run(self) -> None:
        dataset = self._build_dataset()
        if len(dataset) == 0:
            return
        subjects = [sid for sid, _ in dataset.sequences]
        trainval_indices, test_indices = self._split_outer(subjects)
        best_model_path = self._train_cv(dataset, trainval_indices, subjects)
        test_loader = self._get_loader(dataset, test_indices, shuffle=False)
        model = self._build_model(dataset).to(self.device)
        if best_model_path and os.path.exists(best_model_path):
            state = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(state)
        triu_idx = compute_triu_indices(dataset.max_nodes)
        test_loss = self._evaluate(model, test_loader, triu_idx)
        self.summary["test_loss"] = float(test_loss)
        self.summary["n_subjects"] = len(dataset)
        self.summary["n_trainval"] = len(trainval_indices)
        self.summary["n_test"] = len(test_indices)
        ensure_dir(self.results_dir)
        with open(
            os.path.join(self.results_dir, "clg_ode_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.summary, f, indent=2)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    diff = diff * mask
    denom = mask.sum() * diff.shape[-1]
    if denom.item() == 0:
        return torch.tensor(0.0, device=diff.device)
    return diff.sum() / denom


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)


def smoothness_loss(
    z_morph: torch.Tensor,
    z_conn: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if z_morph.shape[1] < 3:
        return torch.tensor(0.0, device=z_morph.device)
    dz2_m = z_morph[:, 2:] - 2 * z_morph[:, 1:-1] + z_morph[:, :-2]
    dz2_c = z_conn[:, 2:] - 2 * z_conn[:, 1:-1] + z_conn[:, :-2]
    valid = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    loss_m = masked_mse(dz2_m, torch.zeros_like(dz2_m), valid)
    loss_c = masked_mse(dz2_c, torch.zeros_like(dz2_c), valid)
    return 0.5 * (loss_m + loss_c)
