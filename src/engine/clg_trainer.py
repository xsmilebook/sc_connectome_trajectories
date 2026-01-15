from __future__ import annotations

import json
import math
import os
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

from src.data.clg_dataset import CLGDataset, collate_clg_sequences
from src.data.utils import compute_triu_indices, ensure_dir
from src.data.topology import compute_ecc
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
        lambda_manifold: float = 1.0,
        lambda_vel: float = 0.1,
        lambda_acc: float = 0.05,
        warmup_manifold_epochs: int = 5,
        warmup_vel_epochs: int = 10,
        morph_noise_sigma: float = 0.05,
        sc_pos_edge_drop_prob: float = 0.02,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
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
        self.lambda_manifold = lambda_manifold
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
        self.warmup_manifold_epochs = warmup_manifold_epochs
        self.warmup_vel_epochs = warmup_vel_epochs
        self.morph_noise_sigma = morph_noise_sigma
        self.sc_pos_edge_drop_prob = sc_pos_edge_drop_prob
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = self.world_size > 1
        self.is_main = self.rank == 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}" if self.is_distributed else "cuda")
        else:
            self.device = torch.device("cpu")
        self.summary: Dict[str, Any] = {}
        self._rng = np.random.default_rng(self.random_state + self.rank)
        self.metrics_csv_path = os.path.join(self.results_dir, "metrics.csv")

    def _build_dataset(self) -> CLGDataset:
        return CLGDataset(
            sc_dir=self.sc_dir,
            morph_root=self.morph_root,
            subject_info_csv=self.subject_info_csv,
            topo_bins=self.topo_bins,
            min_length=1,
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
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate_clg_sequences,
        )

    def _numeric_cov_dim(self) -> int:
        base = 1 + 1 + self.topo_bins
        if self.use_s_mean:
            base += 1
        return base

    def _reduce_loss(self, total_loss: float, count: int) -> float:
        if count == 0:
            return math.inf
        if not self.is_distributed or not dist.is_initialized():
            return total_loss / count
        tensor = torch.tensor([total_loss, count], dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total, cnt = tensor.tolist()
        return total / cnt if cnt > 0 else math.inf

    def _reduce_sum_count(self, total: float, count: int) -> Tuple[float, int]:
        if not self.is_distributed or not dist.is_initialized():
            return total, count
        tensor = torch.tensor([total, count], dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_out, count_out = tensor.tolist()
        return float(total_out), int(count_out)

    def _reduce_metric_sums(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.is_distributed or not dist.is_initialized():
            return metrics
        keys = sorted(metrics.keys())
        tensor = torch.tensor([metrics[k] for k in keys], dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {k: float(v) for k, v in zip(keys, tensor.tolist())}

    def _reduce_count(self, count: int) -> int:
        if not self.is_distributed or not dist.is_initialized():
            return count
        tensor = torch.tensor([count], dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def _append_metrics_row(self, row: Dict[str, Any]) -> None:
        if not self.is_main:
            return
        ensure_dir(self.results_dir)
        write_header = not os.path.exists(self.metrics_csv_path)
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

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
        if not self.is_distributed or self.is_main:
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
            stats = NormStats(
                morph_mean=torch.from_numpy(morph_mean.astype(np.float32)).to(self.device),
                morph_std=torch.from_numpy(morph_std.astype(np.float32)).to(self.device),
                topo_mean=torch.from_numpy(topo_mean.astype(np.float32)).to(self.device),
                topo_std=torch.from_numpy(topo_std.astype(np.float32)).to(self.device),
            )
        else:
            stats = NormStats(
                morph_mean=torch.zeros((n_nodes, morph_dim), dtype=torch.float32, device=self.device),
                morph_std=torch.ones((n_nodes, morph_dim), dtype=torch.float32, device=self.device),
                topo_mean=torch.zeros((topo_dim,), dtype=torch.float32, device=self.device),
                topo_std=torch.ones((topo_dim,), dtype=torch.float32, device=self.device),
            )
        if self.is_distributed and dist.is_initialized():
            dist.broadcast(stats.morph_mean, src=0)
            dist.broadcast(stats.morph_std, src=0)
            dist.broadcast(stats.topo_mean, src=0)
            dist.broadcast(stats.topo_std, src=0)
        return stats

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

    def _sample_triplet(self, length: int) -> Tuple[int, int, int]:
        if length < 3:
            return 0, min(1, length - 1), min(2, length - 1)
        if self._rng.random() < self.adjacent_pair_prob:
            i = int(self._rng.integers(0, length - 2))
            return i, i + 1, i + 2
        i = int(self._rng.integers(0, length - 2))
        j = int(self._rng.integers(i + 1, length - 1))
        k = int(self._rng.integers(j + 1, length))
        return i, j, k

    @staticmethod
    def _clamp_dt(dt: float, fallback: float) -> float:
        if not np.isfinite(dt) or dt <= 0:
            dt = fallback
        return float(max(dt, 1e-3))

    @staticmethod
    def _drop_positive_edges(a_log: torch.Tensor, a_raw: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0:
            return a_log
        with torch.no_grad():
            pos = a_raw > 0
            tri = torch.triu(pos, diagonal=1)
            if tri.sum().item() == 0:
                return a_log
            drop = (torch.rand_like(a_log) < p) & tri
            drop = drop | drop.transpose(-1, -2)
        out = a_log.clone()
        out[drop] = 0.0
        return out

    def _recon_losses(
        self,
        a_logit: torch.Tensor,
        a_weight: torch.Tensor,
        x_hat: torch.Tensor,
        a_true_raw: torch.Tensor,
        x_true: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_x = torch.mean((x_hat - x_true) ** 2)
        logit_vec = a_logit[:, triu_idx[0], triu_idx[1]]
        target_vec = (a_true_raw[:, triu_idx[0], triu_idx[1]] > 0).float()
        loss_edge = edge_bce_loss(logit_vec, target_vec, pos_weight=5.0)
        pred_log = torch.log1p(a_weight[:, triu_idx[0], triu_idx[1]])
        true_log = torch.log1p(a_true_raw[:, triu_idx[0], triu_idx[1]])
        loss_weight = weight_huber_loss(pred_log, true_log, target_vec)
        return loss_x, loss_edge, loss_weight

    @staticmethod
    def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2))
        if denom.item() == 0:
            return 0.0
        return float((x * y).sum().item() / denom.item())

    @staticmethod
    def _pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        if denom == 0:
            return 0.0
        return float(np.sum(x * y) / denom)

    def _sc_metrics(
        self,
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, float]:
        pred_log = torch.log1p(pred_weight)
        true_log = torch.log1p(true_raw)
        pred_vec = pred_log[triu_idx[0], triu_idx[1]]
        true_vec = true_log[triu_idx[0], triu_idx[1]]
        diff = pred_vec - true_vec
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        corr = self._pearsonr_torch(pred_vec, true_vec)

        pred_np = pred_log.detach().cpu().numpy()
        true_np = true_log.detach().cpu().numpy()
        ecc_pred = compute_ecc(pred_np, k=self.topo_bins)
        ecc_true = compute_ecc(true_np, k=self.topo_bins)
        ecc_l2 = float(np.linalg.norm(ecc_pred - ecc_true))
        ecc_corr = self._pearsonr_np(ecc_pred, ecc_true)
        return {
            "sc_log_mse": mse,
            "sc_log_mae": mae,
            "sc_log_pearson": corr,
            "ecc_l2": ecc_l2,
            "ecc_pearson": ecc_corr,
        }

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
        epoch: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        count = 0
        m_sum = 0.0
        v_sum = 0.0
        a_sum = 0.0
        m_count = 0
        v_count = 0
        a_count = 0
        for batch in loader:
            if not batch:
                continue
            loss, metrics = self._compute_loss(model, batch, triu_idx, stats, volume_indices, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
            m_sum += metrics["manifold_sum"]
            v_sum += metrics["vel_sum"]
            a_sum += metrics["acc_sum"]
            m_count += metrics["manifold_count"]
            v_count += metrics["vel_count"]
            a_count += metrics["acc_count"]
        m_sum, m_count = self._reduce_sum_count(m_sum, m_count)
        v_sum, v_count = self._reduce_sum_count(v_sum, v_count)
        a_sum, a_count = self._reduce_sum_count(a_sum, a_count)
        if self.is_main:
            self._last_epoch_train_metrics = {
                "manifold": (m_sum / m_count) if m_count > 0 else 0.0,
                "vel": (v_sum / v_count) if v_count > 0 else 0.0,
                "acc": (a_sum / a_count) if a_count > 0 else 0.0,
            }
        return self._reduce_loss(total_loss, count)

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        triu_idx: Tuple[np.ndarray, np.ndarray],
        stats: NormStats,
        volume_indices: List[int],
        epoch: int,
    ) -> float:
        model.eval()
        total_loss = 0.0
        count = 0
        m_sum = 0.0
        v_sum = 0.0
        a_sum = 0.0
        m_count = 0
        v_count = 0
        a_count = 0
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                loss, metrics = self._compute_loss(model, batch, triu_idx, stats, volume_indices, epoch)
                total_loss += float(loss.item())
                count += 1
                m_sum += metrics["manifold_sum"]
                v_sum += metrics["vel_sum"]
                a_sum += metrics["acc_sum"]
                m_count += metrics["manifold_count"]
                v_count += metrics["vel_count"]
                a_count += metrics["acc_count"]
        m_sum, m_count = self._reduce_sum_count(m_sum, m_count)
        v_sum, v_count = self._reduce_sum_count(v_sum, v_count)
        a_sum, a_count = self._reduce_sum_count(a_sum, a_count)
        if self.is_main:
            self._last_epoch_val_metrics = {
                "manifold": (m_sum / m_count) if m_count > 0 else 0.0,
                "vel": (v_sum / v_count) if v_count > 0 else 0.0,
                "acc": (a_sum / a_count) if a_count > 0 else 0.0,
            }
        return self._reduce_loss(total_loss, count)

    def _compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        triu_idx: Tuple[np.ndarray, np.ndarray],
        stats: NormStats,
        volume_indices: List[int],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        lengths = batch["lengths"].tolist()
        a_raw = batch["a_raw"].to(self.device)
        a_log = batch["a_log"].to(self.device)
        x = batch["x"].to(self.device)
        ages = batch["ages"].to(self.device)
        topo = batch["topo"].to(self.device)
        strength = batch["strength"].to(self.device)
        icv = batch["icv"].to(self.device)
        sex = batch["sex"].to(self.device)
        site = batch["site"].to(self.device)

        enable_vel = epoch >= self.warmup_manifold_epochs
        enable_acc = epoch >= self.warmup_vel_epochs

        loss_manifold = []
        loss_vel = []
        loss_acc = []

        for b, length in enumerate(lengths):
            if length >= 3:
                i, j, k = self._sample_triplet(length)
            elif length == 2:
                i, j = self._sample_pair(length)
                k = None
            else:
                i, j, k = 0, None, None

            age_i = float(ages[b, i].item())
            if not np.isfinite(age_i):
                age_i = float(i)
            topo_i = self._zscore(topo[b, i], stats.topo_mean, stats.topo_std)
            strength_i = strength[b, i]
            if self.use_s_mean:
                cov = torch.cat([torch.tensor([age_i], device=self.device), strength_i, topo_i], dim=0)
            else:
                cov = torch.cat([torch.tensor([age_i], device=self.device), strength_i[:1], topo_i], dim=0)
            cov = cov.unsqueeze(0)
            sex_b = sex[b : b + 1]
            site_b = site[b : b + 1]

            x_i = self._apply_volume_norm_torch(x[b, i], icv[b, i], volume_indices)
            x_i = self._zscore(x_i, stats.morph_mean, stats.morph_std)
            x_i_noisy = x_i + torch.randn_like(x_i) * float(self.morph_noise_sigma)

            a_i_log_clean = a_log[b, i]
            a_i_log_noisy = self._drop_positive_edges(a_i_log_clean, a_raw[b, i], self.sc_pos_edge_drop_prob)

            times0 = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            outputs_i = model(
                a_i_log_clean.unsqueeze(0),
                x_i.unsqueeze(0),
                times0,
                sex_b,
                site_b,
                cov,
                use_mu=True,
            )
            outputs_denoise = model(
                a_i_log_noisy.unsqueeze(0),
                x_i_noisy.unsqueeze(0),
                times0,
                sex_b,
                site_b,
                cov,
                use_mu=True,
            )
            lx, le, lw = self._recon_losses(
                outputs_denoise.a_logit[:, 0],
                outputs_denoise.a_weight[:, 0],
                outputs_denoise.x_hat[:, 0],
                a_raw[b, i].unsqueeze(0),
                x_i.unsqueeze(0),
                triu_idx,
            )
            manifold = lx + le + self.lambda_weight * lw

            z_enc_i = torch.cat([outputs_i.z_morph[:, 0], outputs_i.z_conn[:, 0]], dim=-1)

            z_enc_j = None
            z_enc_k = None
            z_pred_j = None
            z_pred_k = None

            if j is not None:
                age_j = float(ages[b, j].item())
                dt_ij = self._clamp_dt(age_j - age_i, float(j - i))
                times = torch.tensor([[0.0, dt_ij]], dtype=torch.float32, device=self.device)
                outputs_fore = model(
                    a_i_log_clean.unsqueeze(0),
                    x_i.unsqueeze(0),
                    times,
                    sex_b,
                    site_b,
                    cov,
                    use_mu=True,
                )
                lxj, lej, lwj = self._recon_losses(
                    outputs_fore.a_logit[:, -1],
                    outputs_fore.a_weight[:, -1],
                    outputs_fore.x_hat[:, -1],
                    a_raw[b, j].unsqueeze(0),
                    self._zscore(
                        self._apply_volume_norm_torch(x[b, j], icv[b, j], volume_indices),
                        stats.morph_mean,
                        stats.morph_std,
                    ).unsqueeze(0),
                    triu_idx,
                )
                manifold = manifold + (lxj + lej + self.lambda_weight * lwj)
                z_pred_j = torch.cat([outputs_fore.z_morph[:, -1], outputs_fore.z_conn[:, -1]], dim=-1)

                outputs_j = model(
                    a_log[b, j].unsqueeze(0),
                    self._zscore(
                        self._apply_volume_norm_torch(x[b, j], icv[b, j], volume_indices),
                        stats.morph_mean,
                        stats.morph_std,
                    ).unsqueeze(0),
                    times0,
                    sex_b,
                    site_b,
                    cov,
                    use_mu=True,
                )
                z_enc_j = torch.cat([outputs_j.z_morph[:, 0], outputs_j.z_conn[:, 0]], dim=-1)

            if k is not None:
                age_k = float(ages[b, k].item())
                dt_ik = self._clamp_dt(age_k - age_i, float(k - i))
                times = torch.tensor([[0.0, dt_ik]], dtype=torch.float32, device=self.device)
                outputs_fore_k = model(
                    a_i_log_clean.unsqueeze(0),
                    x_i.unsqueeze(0),
                    times,
                    sex_b,
                    site_b,
                    cov,
                    use_mu=True,
                )
                lxk, lek, lwk = self._recon_losses(
                    outputs_fore_k.a_logit[:, -1],
                    outputs_fore_k.a_weight[:, -1],
                    outputs_fore_k.x_hat[:, -1],
                    a_raw[b, k].unsqueeze(0),
                    self._zscore(
                        self._apply_volume_norm_torch(x[b, k], icv[b, k], volume_indices),
                        stats.morph_mean,
                        stats.morph_std,
                    ).unsqueeze(0),
                    triu_idx,
                )
                manifold = manifold + 0.5 * (lxk + lek + self.lambda_weight * lwk)
                z_pred_k = torch.cat([outputs_fore_k.z_morph[:, -1], outputs_fore_k.z_conn[:, -1]], dim=-1)

                outputs_k = model(
                    a_log[b, k].unsqueeze(0),
                    self._zscore(
                        self._apply_volume_norm_torch(x[b, k], icv[b, k], volume_indices),
                        stats.morph_mean,
                        stats.morph_std,
                    ).unsqueeze(0),
                    times0,
                    sex_b,
                    site_b,
                    cov,
                    use_mu=True,
                )
                z_enc_k = torch.cat([outputs_k.z_morph[:, 0], outputs_k.z_conn[:, 0]], dim=-1)

            loss_manifold.append(manifold)

            if enable_vel and (j is not None) and (z_enc_j is not None) and (z_pred_j is not None):
                dt_ij_t = torch.tensor(dt_ij, dtype=torch.float32, device=self.device)
                v_obs = (z_enc_j.detach() - z_enc_i.detach()) / dt_ij_t
                v_model = (z_pred_j - z_enc_i) / dt_ij_t
                loss_vel.append(F.smooth_l1_loss(v_model, v_obs))

            if (
                enable_acc
                and (j is not None)
                and (k is not None)
                and (z_enc_j is not None)
                and (z_enc_k is not None)
                and (z_pred_j is not None)
                and (z_pred_k is not None)
            ):
                age_j = float(ages[b, j].item())
                age_k = float(ages[b, k].item())
                dt_ij = self._clamp_dt(age_j - age_i, float(j - i))
                dt_jk = self._clamp_dt(age_k - age_j, float(k - j))
                denom = torch.tensor(dt_ij + dt_jk, dtype=torch.float32, device=self.device)
                dt_ij_t = torch.tensor(dt_ij, dtype=torch.float32, device=self.device)
                dt_jk_t = torch.tensor(dt_jk, dtype=torch.float32, device=self.device)

                v_obs_ij = (z_enc_j.detach() - z_enc_i.detach()) / dt_ij_t
                v_obs_jk = (z_enc_k.detach() - z_enc_j.detach()) / dt_jk_t
                a_obs = 2.0 * (v_obs_jk - v_obs_ij) / denom

                v_model_ij = (z_pred_j - z_enc_i) / dt_ij_t
                v_model_jk = (z_pred_k - z_pred_j) / dt_jk_t
                a_model = 2.0 * (v_model_jk - v_model_ij) / denom
                loss_acc.append(F.smooth_l1_loss(a_model, a_obs))

        mean_manifold = torch.stack(loss_manifold).mean() if loss_manifold else torch.tensor(0.0, device=self.device)
        mean_vel = torch.stack(loss_vel).mean() if loss_vel else torch.tensor(0.0, device=self.device)
        mean_acc = torch.stack(loss_acc).mean() if loss_acc else torch.tensor(0.0, device=self.device)

        total = self.lambda_manifold * mean_manifold
        if enable_vel:
            total = total + self.lambda_vel * mean_vel
        if enable_acc:
            total = total + self.lambda_acc * mean_acc
        metrics = {
            "manifold_sum": float(mean_manifold.detach().item()) * max(len(loss_manifold), 1),
            "vel_sum": float(mean_vel.detach().item()) * max(len(loss_vel), 1),
            "acc_sum": float(mean_acc.detach().item()) * max(len(loss_acc), 1),
            "manifold_count": int(len(loss_manifold)),
            "vel_count": int(len(loss_vel)),
            "acc_count": int(len(loss_acc)),
        }
        return total, metrics

    def _evaluate_sc_metrics(
        self,
        model: nn.Module,
        loader: DataLoader,
        triu_idx: Tuple[np.ndarray, np.ndarray],
        stats: NormStats,
        volume_indices: List[int],
    ) -> Dict[str, float]:
        model.eval()
        sums = {
            "sc_log_mse": 0.0,
            "sc_log_mae": 0.0,
            "sc_log_pearson": 0.0,
            "ecc_l2": 0.0,
            "ecc_pearson": 0.0,
        }
        count = 0
        with torch.no_grad():
            for batch in loader:
                if not batch:
                    continue
                lengths = batch["lengths"].tolist()
                a_raw = batch["a_raw"].to(self.device)
                a_log = batch["a_log"].to(self.device)
                x = batch["x"].to(self.device)
                ages = batch["ages"].to(self.device)
                topo = batch["topo"].to(self.device)
                strength = batch["strength"].to(self.device)
                icv = batch["icv"].to(self.device)
                sex = batch["sex"].to(self.device)
                site = batch["site"].to(self.device)

                for b, length in enumerate(lengths):
                    if length >= 2:
                        i, j = self._sample_pair(length)
                    else:
                        i, j = 0, None

                    age_i = float(ages[b, i].item())
                    if not np.isfinite(age_i):
                        age_i = float(i)
                    topo_i = self._zscore(topo[b, i], stats.topo_mean, stats.topo_std)
                    strength_i = strength[b, i]
                    if self.use_s_mean:
                        cov = torch.cat([torch.tensor([age_i], device=self.device), strength_i, topo_i], dim=0)
                    else:
                        cov = torch.cat([torch.tensor([age_i], device=self.device), strength_i[:1], topo_i], dim=0)
                    cov = cov.unsqueeze(0)
                    sex_b = sex[b : b + 1]
                    site_b = site[b : b + 1]

                    x_i = self._apply_volume_norm_torch(x[b, i], icv[b, i], volume_indices)
                    x_i = self._zscore(x_i, stats.morph_mean, stats.morph_std)
                    a_i_log_clean = a_log[b, i]

                    if j is None:
                        times = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
                        outputs = model(
                            a_i_log_clean.unsqueeze(0),
                            x_i.unsqueeze(0),
                            times,
                            sex_b,
                            site_b,
                            cov,
                            use_mu=True,
                        )
                        pred_weight = outputs.a_weight[0, 0]
                        true_raw = a_raw[b, i]
                    else:
                        age_j = float(ages[b, j].item())
                        dt_ij = self._clamp_dt(age_j - age_i, float(j - i))
                        times = torch.tensor([[0.0, dt_ij]], dtype=torch.float32, device=self.device)
                        outputs = model(
                            a_i_log_clean.unsqueeze(0),
                            x_i.unsqueeze(0),
                            times,
                            sex_b,
                            site_b,
                            cov,
                            use_mu=True,
                        )
                        pred_weight = outputs.a_weight[0, -1]
                        true_raw = a_raw[b, j]

                    metrics = self._sc_metrics(pred_weight, true_raw, triu_idx)
                    for k, v in metrics.items():
                        sums[k] += float(v)
                    count += 1

        sums = self._reduce_metric_sums(sums)
        total_count = self._reduce_count(count)
        denom = total_count if total_count > 0 else 1.0
        return {k: v / denom for k, v in sums.items()}

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
            if self.is_distributed and dist.is_initialized():
                model = DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            best_fold_val = math.inf
            best_state = None
            epochs_no_improve = 0
            if self.is_main:
                print(
                    f"Starting fold {fold_idx + 1}/5 with {len(train_idx)} train subjects and {len(val_idx)} val subjects"
                )
            for epoch in range(self.max_epochs):
                if (
                    self.is_distributed
                    and dist.is_initialized()
                    and isinstance(train_loader.sampler, DistributedSampler)
                ):
                    train_loader.sampler.set_epoch(epoch)
                train_loss = self._train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    triu_idx,
                    stats,
                    volume_indices,
                    epoch,
                )
                val_loss = self._evaluate(
                    model,
                    val_loader,
                    triu_idx,
                    stats,
                    volume_indices,
                    epoch,
                )
                if self.is_main:
                    print(
                        f"Fold {fold_idx + 1}/5, epoch {epoch + 1}/{self.max_epochs}, "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )
                    train_m = getattr(self, "_last_epoch_train_metrics", {})
                    val_m = getattr(self, "_last_epoch_val_metrics", {})
                    self._append_metrics_row(
                        {
                            "fold": fold_idx,
                            "epoch": epoch + 1,
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "train_manifold": float(train_m.get("manifold", 0.0)),
                            "train_vel": float(train_m.get("vel", 0.0)),
                            "train_acc": float(train_m.get("acc", 0.0)),
                            "val_manifold": float(val_m.get("manifold", 0.0)),
                            "val_vel": float(val_m.get("vel", 0.0)),
                            "val_acc": float(val_m.get("acc", 0.0)),
                            "enable_vel": int(epoch >= self.warmup_manifold_epochs),
                            "enable_acc": int(epoch >= self.warmup_vel_epochs),
                            "lambda_manifold": float(self.lambda_manifold),
                            "lambda_vel": float(self.lambda_vel),
                            "lambda_acc": float(self.lambda_acc),
                            "morph_noise_sigma": float(self.morph_noise_sigma),
                            "sc_pos_edge_drop_prob": float(self.sc_pos_edge_drop_prob),
                        }
                    )
                if val_loss < best_fold_val:
                    best_fold_val = val_loss
                    if self.is_distributed and dist.is_initialized():
                        best_state = model.module.state_dict()
                    else:
                        best_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.is_main:
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
            if best_state is not None and self.is_main:
                torch.save(best_state, fold_model_path)
            if best_fold_val < best_val:
                best_val = best_fold_val
                best_model_path = fold_model_path
                best_stats = stats
        if self.is_main:
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
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
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
            epoch=self.max_epochs,
        )
        sc_metrics = self._evaluate_sc_metrics(
            model,
            test_loader,
            triu_idx,
            best_stats,
            dataset.volume_metric_indices,
        )
        if self.is_main:
            self.summary["test_loss"] = float(test_loss)
            self.summary["test_sc_metrics"] = sc_metrics
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
            with open(
                os.path.join(self.results_dir, "test_sc_metrics.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(sc_metrics, f, indent=2)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
