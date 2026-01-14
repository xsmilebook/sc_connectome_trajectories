from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.clg_dataset import CLGDataset, collate_clg_sequences
from src.data.utils import compute_triu_indices, ensure_dir
from src.engine.losses import edge_bce_loss, weight_huber_loss
from src.models.clg_ode import CLGODE


@dataclass
class NormStats:
    morph_mean: torch.Tensor
    morph_std: torch.Tensor
    topo_mean: torch.Tensor
    topo_std: torch.Tensor


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
        lambda_kl: float = 0.0,
        lambda_weight: float = 1.0,
        use_s_mean: bool = True,
        topo_bins: int = 32,
        adjacent_pair_prob: float = 0.7,
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
        self.lambda_weight = lambda_weight
        self.use_s_mean = use_s_mean
        self.topo_bins = topo_bins
        self.adjacent_pair_prob = adjacent_pair_prob
        self.solver_steps = solver_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary: Dict[str, Any] = {}
        self._rng = np.random.default_rng(self.random_state)

    def _build_dataset(self) -> CLGDataset:
        return CLGDataset(
            sc_dir=self.sc_dir,
            morph_root=self.morph_root,
            subject_info_csv=self.subject_info_csv,
            topo_bins=self.topo_bins,
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

    def _numeric_cov_dim(self) -> int:
        base = 1 + 1 + self.topo_bins
        if self.use_s_mean:
            base += 1
        return base

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
            numeric_cov_dim=self._numeric_cov_dim(),
            solver_steps=self.solver_steps,
        )

    def _compute_norm_stats(self, dataset: CLGDataset, indices: List[int]) -> NormStats:
        n_nodes = dataset.max_nodes
        morph_dim = len(dataset.metric_order)
        topo_dim = dataset.topo_bins
        morph_sum = np.zeros((n_nodes, morph_dim), dtype=np.float64)
        morph_sumsq = np.zeros((n_nodes, morph_dim), dtype=np.float64)
        topo_sum = np.zeros((topo_dim,), dtype=np.float64)
        topo_sumsq = np.zeros((topo_dim,), dtype=np.float64)
        morph_count = 0
        topo_count = 0
        for idx in indices:
            item = dataset[idx]
            x = item["x"].numpy()
            icv = item["icv"].numpy()
            x = self._apply_volume_norm_np(x, icv, dataset.volume_metric_indices)
            morph_sum += x.sum(axis=0)
            morph_sumsq += (x ** 2).sum(axis=0)
            morph_count += x.shape[0]
            topo = item["topo"].numpy()
            topo_sum += topo.sum(axis=0)
            topo_sumsq += (topo ** 2).sum(axis=0)
            topo_count += topo.shape[0]
        morph_mean = morph_sum / max(morph_count, 1)
        morph_var = morph_sumsq / max(morph_count, 1) - morph_mean ** 2
        morph_std = np.sqrt(np.maximum(morph_var, 1e-6))
        topo_mean = topo_sum / max(topo_count, 1)
        topo_var = topo_sumsq / max(topo_count, 1) - topo_mean ** 2
        topo_std = np.sqrt(np.maximum(topo_var, 1e-6))
        return NormStats(
            morph_mean=torch.from_numpy(morph_mean.astype(np.float32)).to(self.device),
            morph_std=torch.from_numpy(morph_std.astype(np.float32)).to(self.device),
            topo_mean=torch.from_numpy(topo_mean.astype(np.float32)).to(self.device),
            topo_std=torch.from_numpy(topo_std.astype(np.float32)).to(self.device),
        )

    @staticmethod
    def _apply_volume_norm_np(
        x: np.ndarray,
        icv: np.ndarray,
        volume_indices: List[int],
    ) -> np.ndarray:
        if not volume_indices:
            return x
        x_adj = x.copy()
        for t in range(x.shape[0]):
            scale = icv[t]
            if not np.isfinite(scale) or scale <= 0:
                continue
            x_adj[t, :, volume_indices] = x_adj[t, :, volume_indices] / scale
        return x_adj

    @staticmethod
    def _apply_volume_norm_torch(
        x: torch.Tensor,
        icv: torch.Tensor,
        volume_indices: List[int],
    ) -> torch.Tensor:
        if not volume_indices:
            return x
        x = x.clone()
        if icv.dim() == 0:
            if not torch.isfinite(icv) or icv <= 0:
                return x
            x[:, :, volume_indices] = x[:, :, volume_indices] / icv
            return x
        scale = icv.view(-1, 1, 1)
        valid = torch.isfinite(scale) & (scale > 0)
        if not torch.any(valid):
            return x
        safe_scale = torch.where(valid, scale, torch.ones_like(scale))
        x[:, :, volume_indices] = x[:, :, volume_indices] / safe_scale
        return x

    @staticmethod
    def _zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std

    def _sample_pair(self, length: int) -> Tuple[int, int]:
        if length <= 2:
            return 0, max(1, length - 1)
        if self._rng.random() < self.adjacent_pair_prob:
            i = int(self._rng.integers(0, length - 1))
            return i, i + 1
        i = int(self._rng.integers(0, length - 1))
        j = int(self._rng.integers(i + 1, length))
        return i, j

    def _prepare_pair_batch(
        self,
        batch: Dict[str, torch.Tensor],
        stats: NormStats,
        volume_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        lengths = batch["lengths"].tolist()
        a_log = batch["a_log"]
        a_raw = batch["a_raw"]
        x = batch["x"]
        ages = batch["ages"]
        topo = batch["topo"]
        strength = batch["strength"]
        icv = batch["icv"]
        sex = batch["sex"]
        site = batch["site"]

        a0_list = []
        x0_list = []
        a_t_list = []
        x_t_list = []
        delta_list = []
        age0_list = []
        topo_list = []
        strength_list = []
        icv_start_list = []
        icv_end_list = []
        sex_list = []
        site_list = []

        for i, length in enumerate(lengths):
            start, end = self._sample_pair(length)
            a0_list.append(a_log[i, start])
            x0_list.append(x[i, start])
            a_t_list.append(a_raw[i, end])
            x_t_list.append(x[i, end])
            age0_val = float(ages[i, start].item())
            age_t_val = float(ages[i, end].item())
            delta = age_t_val - age0_val
            if not np.isfinite(delta) or delta <= 0:
                delta = float(end - start)
            delta_list.append(delta)
            age0_list.append(age0_val)
            topo_list.append(topo[i, start])
            strength_list.append(strength[i, start])
            icv_start_list.append(icv[i, start])
            icv_end_list.append(icv[i, end])
            sex_list.append(sex[i])
            site_list.append(site[i])

        a0 = torch.stack(a0_list).to(self.device)
        x0 = torch.stack(x0_list).to(self.device)
        a_t = torch.stack(a_t_list).to(self.device)
        x_t = torch.stack(x_t_list).to(self.device)
        topo0 = torch.stack(topo_list).to(self.device)
        strength0 = torch.stack(strength_list).to(self.device)
        sex0 = torch.stack(sex_list).to(self.device)
        site0 = torch.stack(site_list).to(self.device)
        icv_start = torch.stack(icv_start_list).to(self.device)
        icv_end = torch.stack(icv_end_list).to(self.device)
        delta_t = torch.tensor(delta_list, dtype=torch.float32, device=self.device)
        age0 = torch.tensor(age0_list, dtype=torch.float32, device=self.device)

        x0 = self._apply_volume_norm_torch(x0, icv_start, volume_indices)
        x_t = self._apply_volume_norm_torch(x_t, icv_end, volume_indices)
        x0 = self._zscore(x0, stats.morph_mean, stats.morph_std)
        x_t = self._zscore(x_t, stats.morph_mean, stats.morph_std)
        topo0 = self._zscore(topo0, stats.topo_mean, stats.topo_std)

        times = torch.stack([torch.zeros_like(delta_t), delta_t], dim=1)

        if self.use_s_mean:
            cov = torch.cat([age0.unsqueeze(-1), strength0, topo0], dim=-1)
        else:
            cov = torch.cat([age0.unsqueeze(-1), strength0[:, :1], topo0], dim=-1)

        return {
            "a0": a0,
            "x0": x0,
            "a_t": a_t,
            "x_t": x_t,
            "times": times,
            "sex": sex0,
            "site": site0,
            "cov": cov,
        }

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        triu_idx: Tuple[np.ndarray, np.ndarray],
        stats: NormStats,
        volume_indices: List[int],
    ) -> float:
        model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            if not batch:
                continue
            loss = self._compute_loss(model, batch, triu_idx, stats, volume_indices)
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
        stats: NormStats,
        volume_indices: List[int],
    ) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                loss = self._compute_loss(model, batch, triu_idx, stats, volume_indices)
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
        stats: NormStats,
        volume_indices: List[int],
    ) -> torch.Tensor:
        prepared = self._prepare_pair_batch(batch, stats, volume_indices)
        outputs = model(
            prepared["a0"],
            prepared["x0"],
            prepared["times"],
            prepared["sex"],
            prepared["site"],
            prepared["cov"],
        )
        a_logit = outputs.a_logit[:, -1]
        a_weight = outputs.a_weight[:, -1]
        x_hat = outputs.x_hat[:, -1]
        a_true = prepared["a_t"]
        x_true = prepared["x_t"]

        loss_morph = torch.mean((x_hat - x_true) ** 2)

        logit_vec = a_logit[:, triu_idx[0], triu_idx[1]]
        target_vec = (a_true[:, triu_idx[0], triu_idx[1]] > 0).float()
        loss_edge = edge_bce_loss(logit_vec, target_vec, pos_weight=5.0)

        pred_log = torch.log1p(a_weight[:, triu_idx[0], triu_idx[1]])
        true_log = torch.log1p(a_true[:, triu_idx[0], triu_idx[1]])
        loss_weight = weight_huber_loss(pred_log, true_log, target_vec)

        kl_morph = kl_divergence(outputs.mu_morph, outputs.logvar_morph)
        kl_conn = kl_divergence(outputs.mu_conn, outputs.logvar_conn)
        loss_kl = 0.5 * (kl_morph + kl_conn)

        return loss_morph + loss_edge + self.lambda_weight * loss_weight + self.lambda_kl * loss_kl

    def _train_cv(
        self,
        dataset: CLGDataset,
        trainval_indices: List[int],
        subjects: List[str],
    ) -> Tuple[str, NormStats]:
        groups = np.array([subjects[i] for i in trainval_indices])
        indices = np.array(trainval_indices)
        gkf = GroupKFold(n_splits=5)
        triu_idx = compute_triu_indices(dataset.max_nodes)
        best_val = math.inf
        best_model_path = ""
        best_stats: NormStats | None = None
        fold_results = []
        volume_indices = dataset.volume_metric_indices
        for fold_idx, (train_idx_rel, val_idx_rel) in enumerate(gkf.split(indices, groups=groups)):
            train_idx = indices[train_idx_rel].tolist()
            val_idx = indices[val_idx_rel].tolist()
            stats = self._compute_norm_stats(dataset, train_idx)
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
                train_loss = self._train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    triu_idx,
                    stats,
                    volume_indices,
                )
                val_loss = self._evaluate(
                    model,
                    val_loader,
                    triu_idx,
                    stats,
                    volume_indices,
                )
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
                best_stats = stats
        self.summary["cv_folds"] = fold_results
        self.summary["best_val_loss"] = float(best_val)
        self.summary["best_model_path"] = best_model_path
        if best_stats is None:
            raise RuntimeError("No training folds produced a model.")
        return best_model_path, best_stats

    def run(self) -> None:
        dataset = self._build_dataset()
        if len(dataset) == 0:
            return
        subjects = [sid for sid, _ in dataset.sequences]
        trainval_indices, test_indices = self._split_outer(subjects)
        best_model_path, best_stats = self._train_cv(dataset, trainval_indices, subjects)
        test_loader = self._get_loader(dataset, test_indices, shuffle=False)
        model = self._build_model(dataset).to(self.device)
        if best_model_path and os.path.exists(best_model_path):
            state = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(state)
        triu_idx = compute_triu_indices(dataset.max_nodes)
        test_loss = self._evaluate(
            model,
            test_loader,
            triu_idx,
            best_stats,
            dataset.volume_metric_indices,
        )
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


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
