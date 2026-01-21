from __future__ import annotations

import json
import math
import os
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import average_precision_score

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
from src.engine.losses import edge_bce_loss, edge_focal_loss_with_logits, weight_huber_loss
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
        edge_loss: str = "bce",
        edge_pos_weight: float = 5.0,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        lambda_full_log_mse: float = 0.0,
        lambda_zero_log: float = 0.0,
        zero_log_warmup_epochs: int = 0,
        zero_log_ramp_epochs: int = 0,
        lambda_delta_log: float = 0.0,
        lambda_density: float = 0.0,
        density_warmup_epochs: int = 0,
        density_ramp_epochs: int = 0,
        compute_mask_auprc: bool = False,
        use_s_mean: bool = True,
        topo_bins: int = 32,
        adjacent_pair_prob: float = 0.7,
        solver_steps: int = 8,
        residual_skip: bool = False,
        residual_tau: float = 1.0,
        residual_no_dt_gate: bool = False,
        residual_cap: float = 0.5,
        fixed_support: bool = False,
        innovation_enabled: bool = False,
        innovation_topm: int = 400,
        innovation_k_new: int = 80,
        innovation_tau: float = 0.10,
        innovation_delta_quantile: float = 0.95,
        innovation_dt_scale_years: float = 1.0,
        innovation_dt_offset_months: float = 0.0,
        innovation_dt_ramp_months: float = 12.0,
        innovation_focal_gamma: float = 2.0,
        innovation_focal_alpha: float = 0.25,
        lambda_new_sparse: float = 0.10,
        new_sparse_warmup_epochs: int = 10,
        new_sparse_ramp_epochs: int = 10,
        lambda_new_reg: float = 0.0,
        innovation_freeze_backbone_after: int = -1,
        lambda_manifold: float = 1.0,
        lambda_vel: float = 0.1,
        lambda_acc: float = 0.05,
        warmup_manifold_epochs: int = 5,
        warmup_vel_epochs: int = 10,
        morph_noise_sigma: float = 0.05,
        sc_pos_edge_drop_prob: float = 0.02,
        lambda_topo: float = 0.1,
        topo_loss_bins: int = 8,
        betti_sharpness: float = 20.0,
        betti_t: float = 10.0,
        betti_taylor_order: int = 20,
        betti_probes: int = 2,
        topo_log_compress: bool = True,
        topo_scale_quantile: float = 0.9,
        topo_scale_ema: float = 0.9,
        topo_scale_min: float = 1e-6,
        topo_warmup_frac: float = 0.2,
        gradnorm_scope: str = "manifold_topo",
        gradnorm_alpha: float = 0.5,
        gradnorm_lr: float = 0.1,
        gradnorm_weight_min: float = 0.1,
        gradnorm_weight_max: float = 10.0,
        cv_folds: int = 5,
        cv_fold: int | None = None,
        resume_from: str | None = None,
        early_stop_metric: str = "val_loss",
        early_stop_density_weight: float = 0.0,
        val_sc_eval_every: int = 0,
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
        self.edge_loss = str(edge_loss).strip().lower()
        if self.edge_loss not in {"bce", "focal"}:
            raise ValueError(f"Unsupported edge_loss: {edge_loss}")
        self.edge_pos_weight = float(edge_pos_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = None if focal_alpha is None else float(focal_alpha)
        self.lambda_full_log_mse = float(lambda_full_log_mse)
        self.lambda_zero_log = float(lambda_zero_log)
        self.zero_log_warmup_epochs = int(zero_log_warmup_epochs)
        self.zero_log_ramp_epochs = int(zero_log_ramp_epochs)
        self.lambda_delta_log = float(lambda_delta_log)
        self.lambda_density = float(lambda_density)
        self.density_warmup_epochs = int(density_warmup_epochs)
        self.density_ramp_epochs = int(density_ramp_epochs)
        self.compute_mask_auprc = bool(compute_mask_auprc)
        self.use_s_mean = use_s_mean
        self.topo_bins = topo_bins
        self.adjacent_pair_prob = adjacent_pair_prob
        self.solver_steps = solver_steps
        self.residual_skip = bool(residual_skip)
        self.residual_tau = float(residual_tau)
        self.residual_no_dt_gate = bool(residual_no_dt_gate)
        self.residual_cap = float(residual_cap)
        self.fixed_support = bool(fixed_support)
        self.innovation_enabled = bool(innovation_enabled)
        self.innovation_topm = int(innovation_topm)
        self.innovation_k_new = int(innovation_k_new)
        self.innovation_tau = float(innovation_tau)
        self.innovation_delta_quantile = float(innovation_delta_quantile)
        self.innovation_dt_scale_years = float(innovation_dt_scale_years)
        self.innovation_dt_offset_months = float(innovation_dt_offset_months)
        self.innovation_dt_ramp_months = float(innovation_dt_ramp_months)
        self.innovation_focal_gamma = float(innovation_focal_gamma)
        self.innovation_focal_alpha = float(innovation_focal_alpha)
        self.lambda_new_sparse = float(lambda_new_sparse)
        self.new_sparse_warmup_epochs = int(new_sparse_warmup_epochs)
        self.new_sparse_ramp_epochs = int(new_sparse_ramp_epochs)
        self.lambda_new_reg = float(lambda_new_reg)
        self.innovation_freeze_backbone_after = int(innovation_freeze_backbone_after)
        self.lambda_manifold = lambda_manifold
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
        self.warmup_manifold_epochs = warmup_manifold_epochs
        self.warmup_vel_epochs = warmup_vel_epochs
        self.morph_noise_sigma = morph_noise_sigma
        self.sc_pos_edge_drop_prob = sc_pos_edge_drop_prob
        self.lambda_topo = float(lambda_topo)
        self.topo_loss_bins = int(topo_loss_bins)
        self.betti_sharpness = float(betti_sharpness)
        self.betti_t = float(betti_t)
        self.betti_taylor_order = int(betti_taylor_order)
        self.betti_probes = int(betti_probes)
        self.topo_log_compress = bool(topo_log_compress)
        self.topo_scale_quantile = float(topo_scale_quantile)
        self.topo_scale_ema = float(topo_scale_ema)
        self.topo_scale_min = float(topo_scale_min)
        self.topo_warmup_frac = float(topo_warmup_frac)
        scope = gradnorm_scope.strip().lower()
        if scope not in {"manifold_topo", "none"}:
            raise ValueError(f"Unsupported gradnorm_scope: {gradnorm_scope}")

        if self.innovation_enabled and not self.fixed_support:
            raise ValueError("innovation_enabled is only supported when fixed_support is enabled.")
        if self.innovation_topm <= 0:
            raise ValueError("innovation_topm must be > 0")
        if self.innovation_k_new < 0:
            raise ValueError("innovation_k_new must be >= 0")
        if self.innovation_tau <= 0:
            raise ValueError("innovation_tau must be > 0")
        if not (0.0 < self.innovation_delta_quantile <= 1.0):
            raise ValueError("innovation_delta_quantile must be in (0, 1]")
        if self.innovation_dt_scale_years <= 0:
            raise ValueError("innovation_dt_scale_years must be > 0")
        if self.innovation_dt_ramp_months <= 0:
            raise ValueError("innovation_dt_ramp_months must be > 0")
        self.gradnorm_scope = scope
        self.gradnorm_enabled = scope != "none"
        self.gradnorm_alpha = float(gradnorm_alpha)
        self.gradnorm_lr = float(gradnorm_lr)
        self.gradnorm_weight_min = float(gradnorm_weight_min)
        self.gradnorm_weight_max = float(gradnorm_weight_max)
        self.cv_folds = cv_folds
        self.cv_fold = cv_fold
        self.resume_from = resume_from
        self.early_stop_metric = str(early_stop_metric).strip()
        self.early_stop_density_weight = float(early_stop_density_weight)
        self.val_sc_eval_every = int(val_sc_eval_every)
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
        self._betti_probe_cache: Dict[int, torch.Tensor] = {}
        self._topo_scale = 1.0
        self._gradnorm_weights = {"manifold": 1.0, "topo": 1.0}
        self._gradnorm_init_losses: Dict[str, torch.Tensor] = {}
        self._innovation_backbone_frozen = False

    def _maybe_freeze_backbone_for_innovation(self, model: nn.Module, epoch: int) -> None:
        if not self.innovation_enabled:
            return
        if self.innovation_freeze_backbone_after < 0:
            return
        if epoch < self.innovation_freeze_backbone_after:
            return
        if self._innovation_backbone_frozen:
            return

        target = model.module if hasattr(model, "module") else model
        trainable = {
            "conn_decoder.alpha_new",
            "conn_decoder.delta_new",
            "conn_decoder.gamma_new",
            "conn_decoder.beta_new",
        }
        for name, param in target.named_parameters():
            param.requires_grad = name in trainable
        self._innovation_backbone_frozen = True
        if self.is_main:
            print(f"Freeze backbone at epoch {epoch + 1}: training innovation head only.")

    def _monitor_requires_sc(self) -> bool:
        return self.early_stop_metric in {
            "val_sc_log_mse",
            "val_sc_log_pearson_topk",
            "val_sc_log_pearson_sparse",
            "monitor_mse_plus_density",
        }

    def _monitor_higher_is_better(self) -> bool:
        return self.early_stop_metric in {
            "val_sc_log_pearson_topk",
            "val_sc_log_pearson_sparse",
        }

    def _compute_monitor_value(
        self,
        val_loss: float,
        val_sc_metrics: Dict[str, float],
        val_density: float,
    ) -> float:
        metric = self.early_stop_metric
        if metric == "val_loss":
            return float(val_loss)
        if metric == "val_sc_log_mse":
            return float(val_sc_metrics.get("sc_log_mse", math.inf))
        if metric == "val_sc_log_pearson_topk":
            return float(val_sc_metrics.get("sc_log_pearson_topk", 0.0))
        if metric == "val_sc_log_pearson_sparse":
            return float(val_sc_metrics.get("sc_log_pearson_sparse", 0.0))
        if metric == "monitor_mse_plus_density":
            mse = float(val_sc_metrics.get("sc_log_mse", math.inf))
            return mse + float(self.early_stop_density_weight) * float(val_density)
        raise ValueError(f"Unsupported early_stop_metric: {metric}")

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
            residual_skip=self.residual_skip,
            residual_tau=self.residual_tau,
            residual_no_dt_gate=self.residual_no_dt_gate,
            residual_cap=self.residual_cap,
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

    def _topo_scale_value(self) -> float:
        scale = float(self._topo_scale)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(self.topo_scale_min)
        return max(scale, float(self.topo_scale_min))

    def _topo_warmup_factor(self, epoch: int) -> float:
        frac = float(self.topo_warmup_frac)
        if frac <= 0:
            return 1.0
        warmup_epochs = max(int(math.ceil(self.max_epochs * frac)), 1)
        if epoch >= warmup_epochs:
            return 1.0
        progress = float(epoch + 1) / float(warmup_epochs)
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def _normalize_topo_loss(self, topo_raw: torch.Tensor) -> torch.Tensor:
        scale = self._topo_scale_value()
        scale_t = torch.tensor(scale, device=topo_raw.device, dtype=topo_raw.dtype)
        topo_scaled = topo_raw / scale_t
        if self.topo_log_compress:
            return torch.log1p(topo_scaled)
        return topo_scaled

    def _update_topo_scale(self, values: List[float]) -> None:
        if not values:
            return
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            return
        q = float(self.topo_scale_quantile)
        q = min(max(q, 0.0), 1.0)
        scale = float(np.quantile(arr, q))
        if not np.isfinite(scale):
            return
        scale = max(scale, float(self.topo_scale_min))
        self._topo_scale = (
            self.topo_scale_ema * self._topo_scale + (1.0 - self.topo_scale_ema) * scale
        )

    def _sync_topo_scale(self) -> None:
        if not self.is_distributed or not dist.is_initialized():
            return
        tensor = torch.tensor([self._topo_scale], dtype=torch.float32, device=self.device)
        dist.broadcast(tensor, src=0)
        self._topo_scale = float(tensor.item())

    def _gradnorm_lambda(self, name: str) -> float:
        if name == "manifold":
            return float(self.lambda_manifold)
        if name == "topo":
            return float(self.lambda_topo)
        return 1.0

    def _compute_gradnorm_weights(
        self,
        model: nn.Module,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        if not self.gradnorm_enabled:
            return self._gradnorm_weights
        if "manifold" not in losses or "topo" not in losses:
            return self._gradnorm_weights
        if any(not loss.requires_grad for loss in losses.values()):
            return self._gradnorm_weights
        weights = dict(self._gradnorm_weights)
        if self.is_main:
            params = [p for p in model.parameters() if p.requires_grad]
            g_norms: Dict[str, torch.Tensor] = {}
            for name, loss in losses.items():
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                norms = [g.norm() for g in grads if g is not None]
                if norms:
                    g_norm = torch.norm(torch.stack(norms))
                else:
                    g_norm = torch.tensor(0.0, device=self.device)
                g_norms[name] = g_norm

            skip_update = any(not torch.isfinite(g) or g.item() <= 0 for g in g_norms.values())
            if not skip_update:
                if not self._gradnorm_init_losses:
                    self._gradnorm_init_losses = {k: v.detach() for k, v in losses.items()}

                eps = 1e-6
                loss_ratios = []
                ratio_map: Dict[str, torch.Tensor] = {}
                for name, loss in losses.items():
                    init = self._gradnorm_init_losses.get(name)
                    if init is None or not torch.isfinite(init):
                        init = loss.detach()
                        self._gradnorm_init_losses[name] = init
                    ratio = torch.clamp(loss.detach() / (init + eps), min=eps)
                    ratio_map[name] = ratio
                    loss_ratios.append(ratio)
                avg_ratio = torch.stack(loss_ratios).mean()
                if not torch.isfinite(avg_ratio) or avg_ratio.item() == 0:
                    avg_ratio = torch.tensor(1.0, device=self.device)

                g_vals = []
                for name in losses:
                    g_vals.append(g_norms[name] * self._gradnorm_lambda(name) * weights[name])
                g_avg = torch.stack(g_vals).mean()

                new_weights: Dict[str, float] = {}
                for name in losses:
                    r = ratio_map[name] / avg_ratio
                    target = g_avg * (r ** self.gradnorm_alpha)
                    denom = g_norms[name] * self._gradnorm_lambda(name)
                    denom = torch.clamp(denom, min=eps)
                    new_w = target / denom
                    new_weights[name] = float(new_w.detach().item())

                total = sum(new_weights.values())
                if total > 0:
                    scale = len(new_weights) / total
                    for name in new_weights:
                        new_weights[name] *= scale

                for name in new_weights:
                    new_weights[name] = float(
                        np.clip(new_weights[name], self.gradnorm_weight_min, self.gradnorm_weight_max)
                    )
                    new_weights[name] = (1.0 - self.gradnorm_lr) * weights[name] + self.gradnorm_lr * new_weights[name]

                weights = new_weights
                self._gradnorm_weights = weights

        if self.is_distributed and dist.is_initialized():
            tensor = torch.tensor(
                [weights.get("manifold", 1.0), weights.get("topo", 1.0)],
                dtype=torch.float32,
                device=self.device,
            )
            dist.broadcast(tensor, src=0)
            self._gradnorm_weights = {
                "manifold": float(tensor[0].item()),
                "topo": float(tensor[1].item()),
            }
        return self._gradnorm_weights

    def _sample_pair(self, length: int) -> Tuple[int, int]:
        if length <= 2:
            return 0, max(1, length - 1)
        if self._rng.random() < self.adjacent_pair_prob:
            i = int(self._rng.integers(0, length - 1))
            return i, i + 1
        i = int(self._rng.integers(0, length - 1))
        j = int(self._rng.integers(i + 1, length))
        return i, j

    def _innovation_phase(self, epoch: int) -> bool:
        return (
            self.innovation_enabled
            and self.innovation_freeze_backbone_after >= 0
            and epoch >= self.innovation_freeze_backbone_after
        )

    def _sample_pair_epoch(self, length: int, epoch: int) -> Tuple[int, int]:
        if not self._innovation_phase(epoch):
            return self._sample_pair(length)
        if length <= 2:
            return 0, max(1, length - 1)
        # Prefer non-adjacent pairs to expose longer dt for innovation-only training.
        if length == 3:
            return 0, 2
        i = int(self._rng.integers(0, length - 2))
        j_low = min(length - 1, i + 2)
        j = int(self._rng.integers(j_low, length))
        return i, j

    def _sample_triplet_epoch(self, length: int, epoch: int) -> Tuple[int, int, int]:
        if not self._innovation_phase(epoch):
            return self._sample_triplet(length)
        if length < 3:
            return 0, min(1, length - 1), min(2, length - 1)
        if length == 3:
            return 0, 1, 2
        i = int(self._rng.integers(0, length - 2))
        j = min(i + 1, length - 2)
        k = length - 1
        if k <= j:
            k = j + 1
        return i, j, k

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
        if self.edge_loss == "bce":
            loss_edge = edge_bce_loss(logit_vec, target_vec, pos_weight=self.edge_pos_weight)
        else:
            loss_edge = edge_focal_loss_with_logits(
                logit_vec,
                target_vec,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
            )
        pred_log = torch.log1p(a_weight[:, triu_idx[0], triu_idx[1]])
        true_log = torch.log1p(a_true_raw[:, triu_idx[0], triu_idx[1]])
        loss_weight = weight_huber_loss(pred_log, true_log, target_vec)
        return loss_x, loss_edge, loss_weight

    @staticmethod
    def _expected_weight(a_logit: torch.Tensor, a_weight: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(a_logit) * torch.clamp(a_weight, min=0.0)

    @staticmethod
    def _apply_fixed_support(pred_weight: torch.Tensor, a0_raw: torch.Tensor) -> torch.Tensor:
        support = (a0_raw > 0).to(dtype=pred_weight.dtype)
        out = torch.clamp(pred_weight, min=0.0) * support
        out = out - torch.diag_embed(torch.diagonal(out, dim1=-2, dim2=-1))
        return out

    @staticmethod
    def _linear_warmup_factor(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
        if warmup_epochs <= 0 and ramp_epochs <= 0:
            return 1.0
        if epoch < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return 1.0
        return float(min(1.0, (epoch - warmup_epochs + 1) / float(ramp_epochs)))

    @staticmethod
    def _dt_gate_months(dt_years: float, offset_months: float, ramp_months: float) -> float:
        if not np.isfinite(dt_years) or dt_years <= 0:
            return 0.0
        dt_months = float(dt_years) * 12.0
        x = (dt_months - float(offset_months)) / float(max(ramp_months, 1e-6))
        return float(min(1.0, max(0.0, x)))

    @staticmethod
    def _support_weight_loss(
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        a0_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        pred_log = torch.log1p(torch.clamp(pred_weight, min=0.0))
        true_log = torch.log1p(torch.clamp(true_raw, min=0.0))
        pred_vec = pred_log[triu_idx[0], triu_idx[1]]
        true_vec = true_log[triu_idx[0], triu_idx[1]]
        mask = (a0_raw[triu_idx[0], triu_idx[1]] > 0).to(dtype=pred_vec.dtype)
        if mask.sum().item() == 0:
            return torch.tensor(0.0, device=pred_vec.device)
        loss = F.smooth_l1_loss(pred_vec, true_vec, reduction="none")
        return (loss * mask).sum() / mask.sum()

    def _innovation_step(
        self,
        pred_support: torch.Tensor,
        pred_new_dense: torch.Tensor,
        l_new: torch.Tensor,
        a0_raw: torch.Tensor,
        a_true_raw: torch.Tensor,
        dt_years: float,
        triu_idx: Tuple[np.ndarray, np.ndarray],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | float]]:
        """
        Conservative innovation for edges where a0_raw == 0:
        - candidate TopM by raw innovation logit l_new (upper-tri only)
        - per-sample threshold δ = Pq(l_new over candidate TopM)
        - q = g(dt)*sigmoid((l_new - δ)/tau)
        - hard cap: keep TopK(K_new) by q
        Returns:
          pred_weight_final (NxN) and a dict with innovation losses/diagnostics.
        """
        out: Dict[str, torch.Tensor | float] = {
            "loss_new_edge": torch.tensor(0.0, device=pred_support.device),
            "loss_new_sparse": torch.tensor(0.0, device=pred_support.device),
            "loss_new_reg": torch.tensor(0.0, device=pred_support.device),
            "new_q_mean": 0.0,
            "new_kept": 0.0,
        }
        if not self.innovation_enabled or self.innovation_k_new == 0:
            return pred_support, out

        gate = self._dt_gate_months(
            dt_years,
            offset_months=self.innovation_dt_offset_months,
            ramp_months=self.innovation_dt_ramp_months,
        )
        if gate <= 0:
            return pred_support, out

        if epoch < self.new_sparse_warmup_epochs:
            return pred_support, out

        i_idx_np, j_idx_np = triu_idx
        i_idx = torch.as_tensor(i_idx_np, device=pred_support.device, dtype=torch.long)
        j_idx = torch.as_tensor(j_idx_np, device=pred_support.device, dtype=torch.long)

        l_vec = l_new[i_idx, j_idx]
        pool_mask = a0_raw[i_idx, j_idx] <= 0
        pool_pos = torch.nonzero(pool_mask, as_tuple=False).squeeze(1)
        if pool_pos.numel() == 0:
            return pred_support, out

        topm = min(self.innovation_topm, int(pool_pos.numel()))
        pool_vals = l_vec[pool_pos]
        top = torch.topk(pool_vals, k=topm, largest=True)
        cand_pos = pool_pos[top.indices]
        cand_vals = top.values

        delta = torch.quantile(cand_vals.detach(), q=float(self.innovation_delta_quantile))
        logits_new = (l_vec[cand_pos] - delta) / float(self.innovation_tau)
        q = float(gate) * torch.sigmoid(logits_new)

        out["new_q_mean"] = float(q.mean().detach().item())

        keep_k = min(int(self.innovation_k_new), int(q.numel()))
        if keep_k <= 0:
            return pred_support, out

        keep = torch.topk(q.detach(), k=keep_k, largest=True)
        keep_pos = cand_pos[keep.indices]

        mask_new = torch.zeros_like(pred_support)
        mask_new[i_idx[keep_pos], j_idx[keep_pos]] = 1.0
        mask_new = mask_new + mask_new.transpose(-1, -2)

        pred_weight = pred_support + mask_new * torch.clamp(pred_new_dense, min=0.0)
        pred_weight = pred_weight - torch.diag_embed(torch.diagonal(pred_weight, dim1=-2, dim2=-1))

        y_new_vec = ((a_true_raw[i_idx, j_idx] > 0) & (a0_raw[i_idx, j_idx] <= 0)).to(dtype=logits_new.dtype)
        y_cand = y_new_vec[cand_pos]
        loss_edge = edge_focal_loss_with_logits(
            logits_new,
            y_cand,
            gamma=self.innovation_focal_gamma,
            alpha=self.innovation_focal_alpha,
        )
        out["loss_new_edge"] = float(gate) * loss_edge

        sparse_factor = self._linear_warmup_factor(epoch, self.new_sparse_warmup_epochs, self.new_sparse_ramp_epochs)
        out["loss_new_sparse"] = float(self.lambda_new_sparse) * float(sparse_factor) * q.mean()

        # New-edge metrics on the pool (A0=0) edges, using q as the score.
        y_pool = (a_true_raw[i_idx, j_idx] > 0).to(dtype=logits_new.dtype)[pool_pos]
        score_triu = torch.zeros_like(l_vec.detach())
        score_triu[cand_pos] = q.detach()
        score_pool = score_triu[pool_pos]
        topk_k = min(int(self.innovation_k_new), int(score_pool.numel()))
        if topk_k > 0 and int(y_pool.sum().item()) > 0:
            top_idx = torch.topk(score_pool.detach(), k=topk_k, largest=True).indices
            tp = float(y_pool[top_idx].sum().item())
            out["new_edge_precision_at_knew"] = tp / float(topk_k)
            out["new_edge_recall_at_knew"] = tp / float(y_pool.sum().item())
        else:
            out["new_edge_precision_at_knew"] = 0.0
            out["new_edge_recall_at_knew"] = 0.0

        try:
            y_np = y_pool.detach().cpu().numpy()
            s_np = score_pool.detach().cpu().numpy()
            if y_np.sum() > 0:
                out["new_edge_auprc"] = float(average_precision_score(y_np, s_np))
            else:
                out["new_edge_auprc"] = 0.0
        except Exception:
            out["new_edge_auprc"] = 0.0

        if self.lambda_new_reg > 0:
            pred_new_vec = torch.clamp(pred_new_dense[i_idx, j_idx], min=0.0)[cand_pos]
            true_new_vec = torch.clamp(a_true_raw[i_idx, j_idx], min=0.0)[cand_pos]
            pred_new_log = torch.log1p(pred_new_vec)
            true_new_log = torch.log1p(true_new_vec)
            out["loss_new_reg"] = float(self.lambda_new_reg) * weight_huber_loss(pred_new_log, true_new_log, y_cand)

        out["new_kept"] = float(keep_k)
        return pred_weight, out

    @staticmethod
    def _density_loss(
        p_hat: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        p_vec = p_hat[triu_idx[0], triu_idx[1]]
        true_vec = true_raw[triu_idx[0], triu_idx[1]]
        k = (true_vec > 0).sum()
        if int(k.item()) <= 0:
            return torch.tensor(0.0, device=p_hat.device)
        sum_p = p_vec.sum()
        rel = (sum_p - k.to(dtype=sum_p.dtype)) / k.to(dtype=sum_p.dtype)
        return rel ** 2

    def _mask_diagnostics(
        self,
        p_hat: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, float]:
        p_vec = p_hat[triu_idx[0], triu_idx[1]]
        true_vec = true_raw[triu_idx[0], triu_idx[1]]
        y = (true_vec > 0).to(dtype=torch.float32)
        k = int(y.sum().item())
        if k <= 0:
            return {}

        sum_p = float(p_vec.sum().item())
        ratio = sum_p / float(k)
        mean_p = float(p_vec.mean().item())
        q = torch.quantile(p_vec.detach(), torch.tensor([0.1, 0.5, 0.9], device=p_vec.device))
        p10, p50, p90 = (float(q[0].item()), float(q[1].item()), float(q[2].item()))

        topk = torch.topk(p_vec.detach(), k=k, largest=True)
        pred_idx = topk.indices
        tp = float(y[pred_idx].sum().item())
        prec_k = tp / float(k)
        rec_k = tp / float(k)

        auprc = 0.0
        if self.compute_mask_auprc:
            y_np = y.detach().cpu().numpy()
            p_np = p_vec.detach().cpu().numpy()
            try:
                auprc = float(average_precision_score(y_np, p_np))
            except Exception:
                auprc = 0.0

        return {
            "mask_ratio": ratio,
            "mask_mean_p": mean_p,
            "mask_p10": p10,
            "mask_p50": p50,
            "mask_p90": p90,
            "mask_precision_at_k": prec_k,
            "mask_recall_at_k": rec_k,
            "mask_auprc": auprc,
        }

    @staticmethod
    def _full_log_mse(
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        pred_log = torch.log1p(torch.clamp(pred_weight, min=0.0))
        true_log = torch.log1p(torch.clamp(true_raw, min=0.0))
        pred_vec = pred_log[triu_idx[0], triu_idx[1]]
        true_vec = true_log[triu_idx[0], triu_idx[1]]
        diff = pred_vec - true_vec
        return torch.mean(diff ** 2)

    @staticmethod
    def _zero_log_penalty(
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        pred_log = torch.log1p(torch.clamp(pred_weight, min=0.0))
        true_vec = true_raw[triu_idx[0], triu_idx[1]]
        pred_vec = pred_log[triu_idx[0], triu_idx[1]]
        mask = true_vec <= 0
        if int(mask.sum().item()) == 0:
            return torch.tensor(0.0, device=pred_weight.device)
        return torch.mean(pred_vec[mask] ** 2)

    @staticmethod
    def _delta_log_penalty(
        pred_weight: torch.Tensor,
        base_log: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        pred_log = torch.log1p(torch.clamp(pred_weight, min=0.0))
        pred_vec = pred_log[triu_idx[0], triu_idx[1]]
        base_vec = base_log[triu_idx[0], triu_idx[1]]
        diff = pred_vec - base_vec
        return torch.mean(diff ** 2)

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

    def _betti_probes(self, n: int) -> torch.Tensor:
        probes = max(int(self.betti_probes), 1)
        cached = self._betti_probe_cache.get(n)
        if cached is not None and cached.device == self.device and cached.shape[0] == probes:
            return cached
        rng = np.random.default_rng(self.random_state)
        signs = rng.integers(0, 2, size=(probes, n), dtype=np.int64) * 2 - 1
        t = torch.from_numpy(signs.astype(np.float32)).to(self.device)
        self._betti_probe_cache[n] = t
        return t

    @staticmethod
    def _quantile_thresholds_torch(
        a_true_log: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
        k: int,
        q_min: float = 0.05,
        q_max: float = 0.95,
    ) -> torch.Tensor:
        vals = a_true_log[triu_idx[0], triu_idx[1]]
        pos = vals[vals > 0]
        if pos.numel() == 0:
            return torch.zeros((k,), dtype=a_true_log.dtype, device=a_true_log.device)
        q = torch.linspace(q_min, q_max, k, device=a_true_log.device, dtype=a_true_log.dtype)
        return torch.quantile(pos, q)

    @staticmethod
    def _normalized_laplacian_matvec(
        w: torch.Tensor,
        v: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        d = w.sum(dim=-1)
        d_inv_sqrt = torch.where(d > 0, torch.rsqrt(d + eps), torch.zeros_like(d))
        tmp = d_inv_sqrt * v
        tmp = torch.matmul(w, tmp)
        tmp = d_inv_sqrt * tmp
        return v - tmp

    def _trace_expm_neg_t_lsym(
        self,
        w: torch.Tensor,
        t: float,
        probes: torch.Tensor,
    ) -> torch.Tensor:
        # Hutchinson trace estimator: tr(exp(-t L_sym)) ≈ E[v^T exp(-tL) v]
        # L_sym matvec is computed implicitly from w.
        k = max(int(self.betti_taylor_order), 1)
        tval = float(t)
        estimates = []
        for p in probes:
            y = p
            term = p
            for i in range(1, k + 1):
                term = (-tval / float(i)) * self._normalized_laplacian_matvec(w, term)
                y = y + term
            estimates.append(torch.dot(p, y))
        return torch.stack(estimates).mean()

    def _betti_curve_loss(
        self,
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        if self.lambda_topo <= 0:
            return torch.tensor(0.0, device=self.device)
        n = int(pred_weight.shape[0])
        k = max(int(self.topo_loss_bins), 1)
        probes = self._betti_probes(n)
        true_log = torch.log1p(true_raw)
        thresholds = self._quantile_thresholds_torch(true_log, triu_idx, k=k)
        if torch.all(thresholds == 0):
            return torch.tensor(0.0, device=self.device)

        pred_log = torch.log1p(pred_weight)
        sharp = float(self.betti_sharpness)
        t = float(self.betti_t)

        beta0_losses = []
        beta1_losses = []
        for tau in thresholds:
            w_true = (true_log >= tau).float()
            w_true = w_true - torch.diag_embed(torch.diagonal(w_true, dim1=-2, dim2=-1))
            w_pred = torch.sigmoid(sharp * (pred_log - tau))
            w_pred = w_pred - torch.diag_embed(torch.diagonal(w_pred, dim1=-2, dim2=-1))

            beta0_true = self._trace_expm_neg_t_lsym(w_true, t=t, probes=probes).detach()
            beta0_pred = self._trace_expm_neg_t_lsym(w_pred, t=t, probes=probes)

            e_true = torch.sum(torch.triu(w_true, diagonal=1))
            e_pred = torch.sum(torch.triu(w_pred, diagonal=1))

            beta1_true = (e_true - float(n) + beta0_true).detach()
            beta1_pred = e_pred - float(n) + beta0_pred

            beta0_losses.append(F.smooth_l1_loss(beta0_pred, beta0_true))
            beta1_losses.append(F.smooth_l1_loss(beta1_pred, beta1_true))

        loss0 = torch.stack(beta0_losses).mean() if beta0_losses else torch.tensor(0.0, device=self.device)
        loss1 = torch.stack(beta1_losses).mean() if beta1_losses else torch.tensor(0.0, device=self.device)
        return loss0 + loss1

    def _sc_metrics(
        self,
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
        a0_raw: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        pred_log = torch.log1p(pred_weight)
        true_log = torch.log1p(true_raw)
        pred_vec = pred_log[triu_idx[0], triu_idx[1]]
        true_vec = true_log[triu_idx[0], triu_idx[1]]
        diff = pred_vec - true_vec
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        corr = self._pearsonr_torch(pred_vec, true_vec)
        mask_pos = true_vec > 0
        if int(mask_pos.sum().item()) > 1:
            corr_pos = self._pearsonr_torch(pred_vec[mask_pos], true_vec[mask_pos])
        else:
            corr_pos = 0.0

        pred_sparse, mask = self._sparsify_pred(pred_weight, true_raw, triu_idx)
        true_vec = true_raw[triu_idx[0], triu_idx[1]]
        corr_topk = 0.0
        if mask is not None and int(mask.sum().item()) > 1:
            corr_topk = self._pearsonr_torch(pred_vec[mask], true_vec[mask])
        pred_sparse_log = torch.log1p(pred_sparse)
        pred_sparse_vec = pred_sparse_log[triu_idx[0], triu_idx[1]]
        corr_sparse = self._pearsonr_torch(pred_sparse_vec, true_vec)

        pred_np = pred_sparse_log.detach().cpu().numpy()
        true_np = true_log.detach().cpu().numpy()
        ecc_pred = compute_ecc(pred_np, k=self.topo_bins)
        ecc_true = compute_ecc(true_np, k=self.topo_bins)
        ecc_l2 = float(np.linalg.norm(ecc_pred - ecc_true))
        ecc_corr = self._pearsonr_np(ecc_pred, ecc_true)
        metrics = {
            "sc_log_mse": mse,
            "sc_log_mae": mae,
            "sc_log_pearson": corr,
            "sc_log_pearson_pos": corr_pos,
            "sc_log_pearson_topk": corr_topk,
            "sc_log_pearson_sparse": corr_sparse,
            "ecc_l2": ecc_l2,
            "ecc_pearson": ecc_corr,
        }
        # Zero-edge / new-region diagnostics (log-domain).
        true_raw_vec = true_raw[triu_idx[0], triu_idx[1]]
        pred_log_vec = pred_log[triu_idx[0], triu_idx[1]]
        mask_zero = true_raw_vec <= 0
        if int(mask_zero.sum().item()) > 0:
            metrics["mse_zero"] = float(torch.mean(pred_log_vec[mask_zero] ** 2).item())
        else:
            metrics["mse_zero"] = 0.0
        if a0_raw is not None:
            a0_vec = a0_raw[triu_idx[0], triu_idx[1]]
            mask_new_region = a0_vec <= 0
            if int(mask_new_region.sum().item()) > 0:
                diff_new = pred_log_vec[mask_new_region] - true_vec[mask_new_region]
                metrics["mse_new_region"] = float(torch.mean(diff_new ** 2).item())
            else:
                metrics["mse_new_region"] = 0.0
            mask_zero_strict = mask_new_region & mask_zero
            if int(mask_zero_strict.sum().item()) > 0:
                metrics["mse_zero_strict"] = float(torch.mean(pred_log_vec[mask_zero_strict] ** 2).item())
            else:
                metrics["mse_zero_strict"] = 0.0
        return metrics

    @staticmethod
    def _sparsify_pred(
        pred_weight: torch.Tensor,
        true_raw: torch.Tensor,
        triu_idx: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        true_vec = true_raw[triu_idx[0], triu_idx[1]]
        pos_count = int((true_vec > 0).sum().item())
        if pos_count <= 0:
            return torch.zeros_like(pred_weight), None
        pred_vec = pred_weight[triu_idx[0], triu_idx[1]]
        topk = torch.topk(pred_vec.detach(), k=pos_count, largest=True)
        mask = torch.zeros_like(pred_vec, dtype=torch.bool)
        mask[topk.indices] = True
        mask_float = mask.to(dtype=pred_weight.dtype)
        mask_full = torch.zeros_like(pred_weight)
        mask_full[triu_idx[0], triu_idx[1]] = mask_float
        mask_full = mask_full + mask_full.transpose(-1, -2)
        mask_full = mask_full - torch.diag_embed(torch.diagonal(mask_full, dim1=-2, dim2=-1))
        pred_sparse = pred_weight * mask_full
        return pred_sparse, mask

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
        t_sum = 0.0
        t_raw_sum = 0.0
        f_sum = 0.0
        z_sum = 0.0
        d_sum = 0.0
        den_sum = 0.0
        new_edge_sum = 0.0
        new_sparse_sum = 0.0
        new_reg_sum = 0.0
        new_q_sum = 0.0
        new_kept_sum = 0.0
        mask_ratio_sum = 0.0
        mask_mean_p_sum = 0.0
        mask_p10_sum = 0.0
        mask_p50_sum = 0.0
        mask_p90_sum = 0.0
        mask_precision_sum = 0.0
        mask_recall_sum = 0.0
        mask_auprc_sum = 0.0
        m_count = 0
        v_count = 0
        a_count = 0
        t_count = 0
        f_count = 0
        z_count = 0
        d_count = 0
        den_count = 0
        new_edge_count = 0
        new_sparse_count = 0
        new_reg_count = 0
        new_diag_count = 0
        mask_count = 0
        topo_raw_values: List[float] = []
        for batch in loader:
            if not batch:
                continue
            loss, metrics = self._compute_loss(model, batch, triu_idx, stats, volume_indices, epoch)
            if not loss.requires_grad:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
            m_sum += metrics["manifold_sum"]
            v_sum += metrics["vel_sum"]
            a_sum += metrics["acc_sum"]
            t_sum += metrics.get("topo_sum", 0.0)
            t_raw_sum += metrics.get("topo_raw_sum", 0.0)
            f_sum += metrics.get("full_log_sum", 0.0)
            z_sum += metrics.get("zero_log_sum", 0.0)
            d_sum += metrics.get("delta_log_sum", 0.0)
            den_sum += metrics.get("density_sum", 0.0)
            new_edge_sum += metrics.get("new_edge_sum", 0.0)
            new_sparse_sum += metrics.get("new_sparse_sum", 0.0)
            new_reg_sum += metrics.get("new_reg_sum", 0.0)
            new_q_sum += metrics.get("new_q_mean_sum", 0.0)
            new_kept_sum += metrics.get("new_kept_sum", 0.0)
            mask_ratio_sum += metrics.get("mask_ratio", 0.0)
            mask_mean_p_sum += metrics.get("mask_mean_p", 0.0)
            mask_p10_sum += metrics.get("mask_p10", 0.0)
            mask_p50_sum += metrics.get("mask_p50", 0.0)
            mask_p90_sum += metrics.get("mask_p90", 0.0)
            mask_precision_sum += metrics.get("mask_precision_at_k", 0.0)
            mask_recall_sum += metrics.get("mask_recall_at_k", 0.0)
            mask_auprc_sum += metrics.get("mask_auprc", 0.0)
            m_count += metrics["manifold_count"]
            v_count += metrics["vel_count"]
            a_count += metrics["acc_count"]
            t_count += metrics.get("topo_count", 0)
            f_count += metrics.get("full_log_count", 0)
            z_count += metrics.get("zero_log_count", 0)
            d_count += metrics.get("delta_log_count", 0)
            den_count += metrics.get("density_count", 0)
            new_edge_count += metrics.get("new_edge_count", 0)
            new_sparse_count += metrics.get("new_sparse_count", 0)
            new_reg_count += metrics.get("new_reg_count", 0)
            new_diag_count += metrics.get("new_diag_count", 0)
            mask_count += metrics.get("mask_diag_count", 0)
            if self.is_main:
                raw_vals = metrics.get("topo_raw_values")
                if raw_vals:
                    topo_raw_values.extend(raw_vals)
        m_sum, m_count = self._reduce_sum_count(m_sum, m_count)
        v_sum, v_count = self._reduce_sum_count(v_sum, v_count)
        a_sum, a_count = self._reduce_sum_count(a_sum, a_count)
        t_sum, t_count = self._reduce_sum_count(t_sum, t_count)
        t_raw_sum, _ = self._reduce_sum_count(t_raw_sum, t_count)
        f_sum, f_count = self._reduce_sum_count(f_sum, f_count)
        z_sum, z_count = self._reduce_sum_count(z_sum, z_count)
        d_sum, d_count = self._reduce_sum_count(d_sum, d_count)
        den_sum, den_count = self._reduce_sum_count(den_sum, den_count)
        new_edge_sum, new_edge_count = self._reduce_sum_count(new_edge_sum, new_edge_count)
        new_sparse_sum, new_sparse_count = self._reduce_sum_count(new_sparse_sum, new_sparse_count)
        new_reg_sum, new_reg_count = self._reduce_sum_count(new_reg_sum, new_reg_count)
        new_q_sum, new_diag_count = self._reduce_sum_count(new_q_sum, new_diag_count)
        new_kept_sum, _ = self._reduce_sum_count(new_kept_sum, new_diag_count)
        mask_ratio_sum, mask_count = self._reduce_sum_count(mask_ratio_sum, mask_count)
        mask_mean_p_sum, _ = self._reduce_sum_count(mask_mean_p_sum, mask_count)
        mask_p10_sum, _ = self._reduce_sum_count(mask_p10_sum, mask_count)
        mask_p50_sum, _ = self._reduce_sum_count(mask_p50_sum, mask_count)
        mask_p90_sum, _ = self._reduce_sum_count(mask_p90_sum, mask_count)
        mask_precision_sum, _ = self._reduce_sum_count(mask_precision_sum, mask_count)
        mask_recall_sum, _ = self._reduce_sum_count(mask_recall_sum, mask_count)
        mask_auprc_sum, _ = self._reduce_sum_count(mask_auprc_sum, mask_count)
        if self.is_main:
            self._last_epoch_train_metrics = {
                "manifold": (m_sum / m_count) if m_count > 0 else 0.0,
                "vel": (v_sum / v_count) if v_count > 0 else 0.0,
                "acc": (a_sum / a_count) if a_count > 0 else 0.0,
                "topo": (t_sum / t_count) if t_count > 0 else 0.0,
                "topo_raw": (t_raw_sum / t_count) if t_count > 0 else 0.0,
                "full_log": (f_sum / f_count) if f_count > 0 else 0.0,
                "zero_log": (z_sum / z_count) if z_count > 0 else 0.0,
                "delta_log": (d_sum / d_count) if d_count > 0 else 0.0,
                "density": (den_sum / den_count) if den_count > 0 else 0.0,
                "new_edge": (new_edge_sum / new_edge_count) if new_edge_count > 0 else 0.0,
                "new_sparse": (new_sparse_sum / new_sparse_count) if new_sparse_count > 0 else 0.0,
                "new_reg": (new_reg_sum / new_reg_count) if new_reg_count > 0 else 0.0,
                "new_q_mean": (new_q_sum / new_diag_count) if new_diag_count > 0 else 0.0,
                "new_kept_mean": (new_kept_sum / new_diag_count) if new_diag_count > 0 else 0.0,
                "mask_ratio": (mask_ratio_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_mean_p": (mask_mean_p_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_p10": (mask_p10_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_p50": (mask_p50_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_p90": (mask_p90_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_precision_at_k": (mask_precision_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_recall_at_k": (mask_recall_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_auprc": (mask_auprc_sum / mask_count) if mask_count > 0 else 0.0,
            }
            self._last_epoch_train_counts = {
                "manifold": int(m_count),
                "vel": int(v_count),
                "acc": int(a_count),
                "topo": int(t_count),
                "full_log": int(f_count),
                "zero_log": int(z_count),
                "delta_log": int(d_count),
                "density": int(den_count),
                "new_edge": int(new_edge_count),
                "new_sparse": int(new_sparse_count),
                "new_reg": int(new_reg_count),
                "new_diag": int(new_diag_count),
                "mask_diag": int(mask_count),
            }
        if self.is_main:
            self._update_topo_scale(topo_raw_values)
        self._sync_topo_scale()
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
        t_sum = 0.0
        t_raw_sum = 0.0
        f_sum = 0.0
        z_sum = 0.0
        d_sum = 0.0
        den_sum = 0.0
        new_edge_sum = 0.0
        new_sparse_sum = 0.0
        new_reg_sum = 0.0
        new_q_sum = 0.0
        new_kept_sum = 0.0
        mask_ratio_sum = 0.0
        mask_mean_p_sum = 0.0
        mask_p10_sum = 0.0
        mask_p50_sum = 0.0
        mask_p90_sum = 0.0
        mask_precision_sum = 0.0
        mask_recall_sum = 0.0
        mask_auprc_sum = 0.0
        m_count = 0
        v_count = 0
        a_count = 0
        t_count = 0
        f_count = 0
        z_count = 0
        d_count = 0
        den_count = 0
        new_edge_count = 0
        new_sparse_count = 0
        new_reg_count = 0
        new_diag_count = 0
        mask_count = 0
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
                t_sum += metrics.get("topo_sum", 0.0)
                t_raw_sum += metrics.get("topo_raw_sum", 0.0)
                f_sum += metrics.get("full_log_sum", 0.0)
                z_sum += metrics.get("zero_log_sum", 0.0)
                d_sum += metrics.get("delta_log_sum", 0.0)
                den_sum += metrics.get("density_sum", 0.0)
                new_edge_sum += metrics.get("new_edge_sum", 0.0)
                new_sparse_sum += metrics.get("new_sparse_sum", 0.0)
                new_reg_sum += metrics.get("new_reg_sum", 0.0)
                new_q_sum += metrics.get("new_q_mean_sum", 0.0)
                new_kept_sum += metrics.get("new_kept_sum", 0.0)
                mask_ratio_sum += metrics.get("mask_ratio", 0.0)
                mask_mean_p_sum += metrics.get("mask_mean_p", 0.0)
                mask_p10_sum += metrics.get("mask_p10", 0.0)
                mask_p50_sum += metrics.get("mask_p50", 0.0)
                mask_p90_sum += metrics.get("mask_p90", 0.0)
                mask_precision_sum += metrics.get("mask_precision_at_k", 0.0)
                mask_recall_sum += metrics.get("mask_recall_at_k", 0.0)
                mask_auprc_sum += metrics.get("mask_auprc", 0.0)
                m_count += metrics["manifold_count"]
                v_count += metrics["vel_count"]
                a_count += metrics["acc_count"]
                t_count += metrics.get("topo_count", 0)
                f_count += metrics.get("full_log_count", 0)
                z_count += metrics.get("zero_log_count", 0)
                d_count += metrics.get("delta_log_count", 0)
                den_count += metrics.get("density_count", 0)
                new_edge_count += metrics.get("new_edge_count", 0)
                new_sparse_count += metrics.get("new_sparse_count", 0)
                new_reg_count += metrics.get("new_reg_count", 0)
                new_diag_count += metrics.get("new_diag_count", 0)
                mask_count += metrics.get("mask_diag_count", 0)
        m_sum, m_count = self._reduce_sum_count(m_sum, m_count)
        v_sum, v_count = self._reduce_sum_count(v_sum, v_count)
        a_sum, a_count = self._reduce_sum_count(a_sum, a_count)
        t_sum, t_count = self._reduce_sum_count(t_sum, t_count)
        t_raw_sum, _ = self._reduce_sum_count(t_raw_sum, t_count)
        f_sum, f_count = self._reduce_sum_count(f_sum, f_count)
        z_sum, z_count = self._reduce_sum_count(z_sum, z_count)
        d_sum, d_count = self._reduce_sum_count(d_sum, d_count)
        den_sum, den_count = self._reduce_sum_count(den_sum, den_count)
        new_edge_sum, new_edge_count = self._reduce_sum_count(new_edge_sum, new_edge_count)
        new_sparse_sum, new_sparse_count = self._reduce_sum_count(new_sparse_sum, new_sparse_count)
        new_reg_sum, new_reg_count = self._reduce_sum_count(new_reg_sum, new_reg_count)
        new_q_sum, new_diag_count = self._reduce_sum_count(new_q_sum, new_diag_count)
        new_kept_sum, _ = self._reduce_sum_count(new_kept_sum, new_diag_count)
        mask_ratio_sum, mask_count = self._reduce_sum_count(mask_ratio_sum, mask_count)
        mask_mean_p_sum, _ = self._reduce_sum_count(mask_mean_p_sum, mask_count)
        mask_p10_sum, _ = self._reduce_sum_count(mask_p10_sum, mask_count)
        mask_p50_sum, _ = self._reduce_sum_count(mask_p50_sum, mask_count)
        mask_p90_sum, _ = self._reduce_sum_count(mask_p90_sum, mask_count)
        mask_precision_sum, _ = self._reduce_sum_count(mask_precision_sum, mask_count)
        mask_recall_sum, _ = self._reduce_sum_count(mask_recall_sum, mask_count)
        mask_auprc_sum, _ = self._reduce_sum_count(mask_auprc_sum, mask_count)
        if self.is_main:
            self._last_epoch_val_metrics = {
                "manifold": (m_sum / m_count) if m_count > 0 else 0.0,
                "vel": (v_sum / v_count) if v_count > 0 else 0.0,
                "acc": (a_sum / a_count) if a_count > 0 else 0.0,
                "topo": (t_sum / t_count) if t_count > 0 else 0.0,
                "topo_raw": (t_raw_sum / t_count) if t_count > 0 else 0.0,
                "full_log": (f_sum / f_count) if f_count > 0 else 0.0,
                "zero_log": (z_sum / z_count) if z_count > 0 else 0.0,
                "delta_log": (d_sum / d_count) if d_count > 0 else 0.0,
                "density": (den_sum / den_count) if den_count > 0 else 0.0,
                "new_edge": (new_edge_sum / new_edge_count) if new_edge_count > 0 else 0.0,
                "new_sparse": (new_sparse_sum / new_sparse_count) if new_sparse_count > 0 else 0.0,
                "new_reg": (new_reg_sum / new_reg_count) if new_reg_count > 0 else 0.0,
                "new_q_mean": (new_q_sum / new_diag_count) if new_diag_count > 0 else 0.0,
                "new_kept_mean": (new_kept_sum / new_diag_count) if new_diag_count > 0 else 0.0,
                "mask_ratio": (mask_ratio_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_mean_p": (mask_mean_p_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_p10": (mask_p10_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_p50": (mask_p50_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_p90": (mask_p90_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_precision_at_k": (mask_precision_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_recall_at_k": (mask_recall_sum / mask_count) if mask_count > 0 else 0.0,
                "mask_auprc": (mask_auprc_sum / mask_count) if mask_count > 0 else 0.0,
            }
            self._last_epoch_val_counts = {
                "manifold": int(m_count),
                "vel": int(v_count),
                "acc": int(a_count),
                "topo": int(t_count),
                "full_log": int(f_count),
                "zero_log": int(z_count),
                "delta_log": int(d_count),
                "density": int(den_count),
                "new_edge": int(new_edge_count),
                "new_sparse": int(new_sparse_count),
                "new_reg": int(new_reg_count),
                "new_diag": int(new_diag_count),
                "mask_diag": int(mask_count),
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
        density_factor = self._linear_warmup_factor(
            epoch,
            warmup_epochs=self.density_warmup_epochs,
            ramp_epochs=self.density_ramp_epochs,
        )
        zero_log_factor = self._linear_warmup_factor(
            epoch,
            warmup_epochs=self.zero_log_warmup_epochs,
            ramp_epochs=self.zero_log_ramp_epochs,
        )

        loss_manifold = []
        loss_vel = []
        loss_acc = []
        loss_kl = []
        loss_topo = []
        loss_full_log = []
        loss_zero_log = []
        loss_delta_log = []
        loss_density = []
        loss_new_edge = []
        loss_new_sparse = []
        loss_new_reg = []
        new_q_sum = 0.0
        new_kept_sum = 0.0
        new_diag_count = 0
        mask_diag = []

        for b, length in enumerate(lengths):
            if length >= 3:
                i, j, k = self._sample_triplet_epoch(length, epoch)
            elif length == 2:
                i, j = self._sample_pair_epoch(length, epoch)
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
            if self.residual_skip:
                a_i_log_noisy = a_i_log_clean
            else:
                a_i_log_noisy = self._drop_positive_edges(
                    a_i_log_clean,
                    a_raw[b, i],
                    self.sc_pos_edge_drop_prob,
                )

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
            kl = kl_divergence(outputs_i.mu_morph, outputs_i.logvar_morph)
            kl = kl + kl_divergence(outputs_i.mu_conn, outputs_i.logvar_conn)
            loss_kl.append(kl)
            outputs_denoise = model(
                a_i_log_noisy.unsqueeze(0),
                x_i_noisy.unsqueeze(0),
                times0,
                sex_b,
                site_b,
                cov,
                use_mu=True,
            )
            if self.fixed_support:
                a_pred_i = self._apply_fixed_support(outputs_denoise.a_weight[0, 0], a_raw[b, i])
            else:
                a_pred_i = self._expected_weight(outputs_denoise.a_logit[0, 0], outputs_denoise.a_weight[0, 0])
            pred_sparse_i, _ = self._sparsify_pred(a_pred_i, a_raw[b, i], triu_idx)
            lx = torch.mean((outputs_denoise.x_hat[:, 0] - x_i.unsqueeze(0)) ** 2)
            if self.fixed_support:
                le = torch.tensor(0.0, device=self.device)
                lw = self._support_weight_loss(pred_sparse_i, a_raw[b, i], a_raw[b, i], triu_idx)
            else:
                _, le, lw = self._recon_losses(
                    outputs_denoise.a_logit[:, 0],
                    pred_sparse_i.unsqueeze(0),
                    outputs_denoise.x_hat[:, 0],
                    a_raw[b, i].unsqueeze(0),
                    x_i.unsqueeze(0),
                    triu_idx,
                )
            manifold = lx + le + self.lambda_weight * lw
            if self.lambda_full_log_mse > 0:
                loss_full_log.append(
                    self._full_log_mse(a_pred_i, a_raw[b, i], triu_idx)
                )
            if self.lambda_zero_log > 0 and zero_log_factor > 0:
                loss_zero_log.append(
                    self._zero_log_penalty(a_pred_i, a_raw[b, i], triu_idx)
                )
            if self.lambda_delta_log > 0:
                loss_delta_log.append(
                    self._delta_log_penalty(a_pred_i, a_i_log_clean, triu_idx)
                )
            if self.lambda_density > 0 and density_factor > 0:
                loss_density.append(self._density_loss(torch.sigmoid(outputs_denoise.a_logit[0, 0]), a_raw[b, i], triu_idx))
            mask_diag.append(
                self._mask_diagnostics(
                    torch.sigmoid(outputs_denoise.a_logit[0, 0]),
                    a_raw[b, i],
                    triu_idx,
                )
            )
            topo_loss_i = self._betti_curve_loss(
                pred_sparse_i,
                a_raw[b, i],
                triu_idx,
            )
            loss_topo.append(topo_loss_i)

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
                if self.fixed_support:
                    if outputs_fore.a_weight_new is None or outputs_fore.l_new is None:
                        raise RuntimeError("Model outputs missing a_weight_new/l_new required for innovation.")
                    a_pred_j = self._apply_fixed_support(outputs_fore.a_weight[0, -1], a_raw[b, i])
                    a_pred_j, innov = self._innovation_step(
                        pred_support=a_pred_j,
                        pred_new_dense=torch.clamp(outputs_fore.a_weight_new[0, -1], min=0.0),
                        l_new=outputs_fore.l_new[0, -1],
                        a0_raw=a_raw[b, i],
                        a_true_raw=a_raw[b, j],
                        dt_years=float(dt_ij),
                        triu_idx=triu_idx,
                        epoch=epoch,
                    )
                    if float(innov["new_kept"]) > 0:
                        loss_new_edge.append(innov["loss_new_edge"])
                        loss_new_sparse.append(innov["loss_new_sparse"])
                        loss_new_reg.append(innov["loss_new_reg"])
                        new_q_sum += float(innov["new_q_mean"])
                        new_kept_sum += float(innov["new_kept"])
                        new_diag_count += 1
                else:
                    a_pred_j = self._expected_weight(outputs_fore.a_logit[0, -1], outputs_fore.a_weight[0, -1])
                pred_sparse_j, _ = self._sparsify_pred(a_pred_j, a_raw[b, j], triu_idx)
                x_j_true = self._zscore(
                    self._apply_volume_norm_torch(x[b, j], icv[b, j], volume_indices),
                    stats.morph_mean,
                    stats.morph_std,
                )
                lxj = torch.mean((outputs_fore.x_hat[:, -1] - x_j_true.unsqueeze(0)) ** 2)
                if self.fixed_support:
                    lej = torch.tensor(0.0, device=self.device)
                    lwj = self._support_weight_loss(pred_sparse_j, a_raw[b, j], a_raw[b, i], triu_idx)
                else:
                    _, lej, lwj = self._recon_losses(
                        outputs_fore.a_logit[:, -1],
                        pred_sparse_j.unsqueeze(0),
                        outputs_fore.x_hat[:, -1],
                        a_raw[b, j].unsqueeze(0),
                        x_j_true.unsqueeze(0),
                        triu_idx,
                    )
                manifold = manifold + (lxj + lej + self.lambda_weight * lwj)
                if self.lambda_full_log_mse > 0:
                    loss_full_log.append(
                        self._full_log_mse(a_pred_j, a_raw[b, j], triu_idx)
                    )
                if self.lambda_zero_log > 0 and zero_log_factor > 0:
                    loss_zero_log.append(
                        self._zero_log_penalty(a_pred_j, a_raw[b, j], triu_idx)
                    )
                if self.lambda_delta_log > 0:
                    loss_delta_log.append(
                        self._delta_log_penalty(a_pred_j, a_i_log_clean, triu_idx)
                    )
                if self.lambda_density > 0 and density_factor > 0:
                    loss_density.append(self._density_loss(torch.sigmoid(outputs_fore.a_logit[0, -1]), a_raw[b, j], triu_idx))
                mask_diag.append(
                    self._mask_diagnostics(
                        torch.sigmoid(outputs_fore.a_logit[0, -1]),
                        a_raw[b, j],
                        triu_idx,
                    )
                )
                topo_loss_j = self._betti_curve_loss(
                    pred_sparse_j,
                    a_raw[b, j],
                    triu_idx,
                )
                loss_topo.append(topo_loss_j)
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
                if self.fixed_support:
                    if outputs_fore_k.a_weight_new is None or outputs_fore_k.l_new is None:
                        raise RuntimeError("Model outputs missing a_weight_new/l_new required for innovation.")
                    a_pred_k = self._apply_fixed_support(outputs_fore_k.a_weight[0, -1], a_raw[b, i])
                    a_pred_k, innov = self._innovation_step(
                        pred_support=a_pred_k,
                        pred_new_dense=torch.clamp(outputs_fore_k.a_weight_new[0, -1], min=0.0),
                        l_new=outputs_fore_k.l_new[0, -1],
                        a0_raw=a_raw[b, i],
                        a_true_raw=a_raw[b, k],
                        dt_years=float(dt_ik),
                        triu_idx=triu_idx,
                        epoch=epoch,
                    )
                    if float(innov["new_kept"]) > 0:
                        loss_new_edge.append(0.5 * innov["loss_new_edge"])
                        loss_new_sparse.append(0.5 * innov["loss_new_sparse"])
                        loss_new_reg.append(0.5 * innov["loss_new_reg"])
                        new_q_sum += float(innov["new_q_mean"])
                        new_kept_sum += float(innov["new_kept"])
                        new_diag_count += 1
                else:
                    a_pred_k = self._expected_weight(outputs_fore_k.a_logit[0, -1], outputs_fore_k.a_weight[0, -1])
                pred_sparse_k, _ = self._sparsify_pred(a_pred_k, a_raw[b, k], triu_idx)
                x_k_true = self._zscore(
                    self._apply_volume_norm_torch(x[b, k], icv[b, k], volume_indices),
                    stats.morph_mean,
                    stats.morph_std,
                )
                lxk = torch.mean((outputs_fore_k.x_hat[:, -1] - x_k_true.unsqueeze(0)) ** 2)
                if self.fixed_support:
                    lek = torch.tensor(0.0, device=self.device)
                    lwk = self._support_weight_loss(pred_sparse_k, a_raw[b, k], a_raw[b, i], triu_idx)
                else:
                    _, lek, lwk = self._recon_losses(
                        outputs_fore_k.a_logit[:, -1],
                        pred_sparse_k.unsqueeze(0),
                        outputs_fore_k.x_hat[:, -1],
                        a_raw[b, k].unsqueeze(0),
                        x_k_true.unsqueeze(0),
                        triu_idx,
                    )
                manifold = manifold + 0.5 * (lxk + lek + self.lambda_weight * lwk)
                if self.lambda_full_log_mse > 0:
                    loss_full_log.append(
                        0.5 * self._full_log_mse(a_pred_k, a_raw[b, k], triu_idx)
                    )
                if self.lambda_zero_log > 0 and zero_log_factor > 0:
                    loss_zero_log.append(
                        0.5 * self._zero_log_penalty(a_pred_k, a_raw[b, k], triu_idx)
                    )
                if self.lambda_delta_log > 0:
                    loss_delta_log.append(
                        0.5 * self._delta_log_penalty(a_pred_k, a_i_log_clean, triu_idx)
                    )
                if self.lambda_density > 0 and density_factor > 0:
                    loss_density.append(0.5 * self._density_loss(torch.sigmoid(outputs_fore_k.a_logit[0, -1]), a_raw[b, k], triu_idx))
                mask_diag.append(
                    self._mask_diagnostics(
                        torch.sigmoid(outputs_fore_k.a_logit[0, -1]),
                        a_raw[b, k],
                        triu_idx,
                    )
                )
                topo_loss_k = self._betti_curve_loss(
                    pred_sparse_k,
                    a_raw[b, k],
                    triu_idx,
                )
                loss_topo.append(0.5 * topo_loss_k)
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
        mean_kl = torch.stack(loss_kl).mean() if loss_kl else torch.tensor(0.0, device=self.device)
        if loss_topo:
            topo_raw = torch.stack(loss_topo)
            mean_topo_raw = topo_raw.mean()
            topo_norm = self._normalize_topo_loss(topo_raw)
            mean_topo = topo_norm.mean()
            topo_raw_values = [float(v) for v in topo_raw.detach().cpu().numpy().tolist()]
        else:
            mean_topo_raw = torch.tensor(0.0, device=self.device)
            mean_topo = torch.tensor(0.0, device=self.device)
            topo_raw_values = []

        weights = {"manifold": 1.0, "topo": 1.0}
        if self.gradnorm_enabled and self.lambda_topo > 0 and loss_topo:
            weights = self._compute_gradnorm_weights(
                model,
                {"manifold": mean_manifold, "topo": mean_topo},
            )
        w_manifold = float(weights.get("manifold", 1.0))
        w_topo = float(weights.get("topo", 1.0))
        topo_warmup = self._topo_warmup_factor(epoch)

        total = self.lambda_manifold * w_manifold * mean_manifold
        if enable_vel:
            total = total + self.lambda_vel * mean_vel
        if enable_acc:
            total = total + self.lambda_acc * mean_acc
        if self.lambda_kl > 0:
            total = total + self.lambda_kl * mean_kl
        mean_full_log = (
            torch.stack(loss_full_log).mean()
            if loss_full_log
            else torch.tensor(0.0, device=self.device)
        )
        mean_zero_log = (
            torch.stack(loss_zero_log).mean()
            if loss_zero_log
            else torch.tensor(0.0, device=self.device)
        )
        mean_delta_log = (
            torch.stack(loss_delta_log).mean()
            if loss_delta_log
            else torch.tensor(0.0, device=self.device)
        )
        mean_density = (
            torch.stack(loss_density).mean()
            if loss_density
            else torch.tensor(0.0, device=self.device)
        )
        mean_new_edge = (
            torch.stack([t for t in loss_new_edge if isinstance(t, torch.Tensor)]).mean()
            if loss_new_edge
            else torch.tensor(0.0, device=self.device)
        )
        mean_new_sparse = (
            torch.stack([t for t in loss_new_sparse if isinstance(t, torch.Tensor)]).mean()
            if loss_new_sparse
            else torch.tensor(0.0, device=self.device)
        )
        mean_new_reg = (
            torch.stack([t for t in loss_new_reg if isinstance(t, torch.Tensor)]).mean()
            if loss_new_reg
            else torch.tensor(0.0, device=self.device)
        )
        diag_keys = list(mask_diag[0].keys()) if mask_diag and mask_diag[0] else []
        diag_sums: Dict[str, float] = {k: 0.0 for k in diag_keys}
        diag_count = 0
        for dct in mask_diag:
            if not dct:
                continue
            for k, v in dct.items():
                diag_sums[k] = diag_sums.get(k, 0.0) + float(v)
            diag_count += 1
        diag_means = {k: (v / diag_count if diag_count > 0 else 0.0) for k, v in diag_sums.items()}

        if self.lambda_topo > 0:
            total = total + self.lambda_topo * w_topo * topo_warmup * mean_topo
        if self.lambda_full_log_mse > 0:
            total = total + self.lambda_full_log_mse * mean_full_log
        if self.lambda_zero_log > 0 and zero_log_factor > 0:
            total = total + (self.lambda_zero_log * float(zero_log_factor)) * mean_zero_log
        if self.lambda_delta_log > 0:
            total = total + self.lambda_delta_log * mean_delta_log
        if self.lambda_density > 0 and density_factor > 0:
            total = total + (self.lambda_density * float(density_factor)) * mean_density
        if self.innovation_enabled:
            total = total + mean_new_edge + mean_new_sparse + mean_new_reg
        metrics = {
            "manifold_sum": float(mean_manifold.detach().item()) * max(len(loss_manifold), 1),
            "vel_sum": float(mean_vel.detach().item()) * max(len(loss_vel), 1),
            "acc_sum": float(mean_acc.detach().item()) * max(len(loss_acc), 1),
            "topo_sum": float(mean_topo.detach().item()) * max(len(loss_topo), 1),
            "topo_raw_sum": float(mean_topo_raw.detach().item()) * max(len(loss_topo), 1),
            "full_log_sum": float(mean_full_log.detach().item()) * max(len(loss_full_log), 1),
            "zero_log_sum": float(mean_zero_log.detach().item()) * max(len(loss_zero_log), 1),
            "delta_log_sum": float(mean_delta_log.detach().item()) * max(len(loss_delta_log), 1),
            "density_sum": float(mean_density.detach().item()) * max(len(loss_density), 1),
            "new_edge_sum": float(mean_new_edge.detach().item()) * max(len(loss_new_edge), 1),
            "new_sparse_sum": float(mean_new_sparse.detach().item()) * max(len(loss_new_sparse), 1),
            "new_reg_sum": float(mean_new_reg.detach().item()) * max(len(loss_new_reg), 1),
            "new_q_mean_sum": float(new_q_sum),
            "new_kept_sum": float(new_kept_sum),
            "manifold_count": int(len(loss_manifold)),
            "vel_count": int(len(loss_vel)),
            "acc_count": int(len(loss_acc)),
            "topo_count": int(len(loss_topo)),
            "full_log_count": int(len(loss_full_log)),
            "zero_log_count": int(len(loss_zero_log)),
            "delta_log_count": int(len(loss_delta_log)),
            "density_count": int(len(loss_density)),
            "new_edge_count": int(len(loss_new_edge)),
            "new_sparse_count": int(len(loss_new_sparse)),
            "new_reg_count": int(len(loss_new_reg)),
            "new_diag_count": int(new_diag_count),
            "topo_raw_values": topo_raw_values,
            "mask_diag_count": int(diag_count),
            "edge_loss": self.edge_loss,
            "edge_pos_weight": float(self.edge_pos_weight),
            "focal_gamma": float(self.focal_gamma),
            "focal_alpha": float(self.focal_alpha) if self.focal_alpha is not None else float("nan"),
            "compute_mask_auprc": int(self.compute_mask_auprc),
            "fixed_support": int(self.fixed_support),
            "innovation_enabled": int(self.innovation_enabled),
        }
        for k, v in diag_means.items():
            metrics[k] = float(v)
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
            "sc_log_pearson_pos": 0.0,
            "sc_log_pearson_topk": 0.0,
            "sc_log_pearson_sparse": 0.0,
            "ecc_l2": 0.0,
            "ecc_pearson": 0.0,
            "mse_zero": 0.0,
            "mse_zero_strict": 0.0,
            "mse_new_region": 0.0,
            "new_edge_precision_at_knew": 0.0,
            "new_edge_recall_at_knew": 0.0,
            "new_edge_auprc": 0.0,
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
                        i, j = 0, 1
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
                        if self.fixed_support:
                            pred_weight = self._apply_fixed_support(outputs.a_weight[0, 0], a_raw[b, i])
                        else:
                            pred_weight = self._expected_weight(outputs.a_logit[0, 0], outputs.a_weight[0, 0])
                        true_raw = a_raw[b, i]
                        a0_raw = a_raw[b, i]
                        innov_metrics: Dict[str, float] = {}
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
                        true_raw = a_raw[b, j]
                        a0_raw = a_raw[b, i]
                        innov_metrics = {}
                        if self.fixed_support:
                            if outputs.a_weight_new is None or outputs.l_new is None:
                                raise RuntimeError("Model outputs missing a_weight_new/l_new required for innovation.")
                            pred_weight = self._apply_fixed_support(outputs.a_weight[0, -1], a_raw[b, i])
                            pred_weight, innov = self._innovation_step(
                                pred_support=pred_weight,
                                pred_new_dense=torch.clamp(outputs.a_weight_new[0, -1], min=0.0),
                                l_new=outputs.l_new[0, -1],
                                a0_raw=a_raw[b, i],
                                a_true_raw=true_raw,
                                dt_years=float(dt_ij),
                                triu_idx=triu_idx,
                                epoch=self.max_epochs,
                            )
                            innov_metrics = {
                                k: float(innov.get(k, 0.0))
                                for k in ["new_edge_precision_at_knew", "new_edge_recall_at_knew", "new_edge_auprc"]
                            }
                        else:
                            pred_weight = self._expected_weight(outputs.a_logit[0, -1], outputs.a_weight[0, -1])

                    metrics = self._sc_metrics(pred_weight, true_raw, triu_idx, a0_raw=a0_raw)
                    if innov_metrics:
                        metrics.update(innov_metrics)
                    for k, v in metrics.items():
                        if k in sums:
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
        n_splits = self.cv_folds
        if n_splits < 2:
            raise ValueError("cv_folds must be >= 2")
        if self.cv_fold is not None and not (0 <= self.cv_fold < n_splits):
            raise ValueError(f"cv_fold must be in [0, {n_splits - 1}]")
        gkf = GroupKFold(n_splits=n_splits)
        triu_idx = compute_triu_indices(dataset.max_nodes)
        best_score = math.inf
        best_model_path = ""
        best_stats: NormStats | None = None
        fold_results = []
        volume_indices = dataset.volume_metric_indices
        for fold_idx, (train_idx_rel, val_idx_rel) in enumerate(gkf.split(indices, groups=groups)):
            if self.cv_fold is not None and fold_idx != self.cv_fold:
                continue
            train_idx = indices[train_idx_rel].tolist()
            val_idx = indices[val_idx_rel].tolist()
            if self.is_main:
                train_lengths = [len(dataset.sequences[i][1]) for i in train_idx]
                val_lengths = [len(dataset.sequences[i][1]) for i in val_idx]
                def _count(lengths: List[int]) -> Dict[str, int]:
                    out = {"len1": 0, "len2": 0, "len3p": 0}
                    for L in lengths:
                        if L <= 1:
                            out["len1"] += 1
                        elif L == 2:
                            out["len2"] += 1
                        else:
                            out["len3p"] += 1
                    return out
                train_cnt = _count(train_lengths)
                val_cnt = _count(val_lengths)
                print(
                    "Fold length distribution: "
                    f"train len1={train_cnt['len1']} len2={train_cnt['len2']} len3+={train_cnt['len3p']}; "
                    f"val len1={val_cnt['len1']} len2={val_cnt['len2']} len3+={val_cnt['len3p']}"
                )
            stats = self._compute_norm_stats(dataset, train_idx)
            train_loader = self._get_loader(dataset, train_idx, shuffle=True)
            val_loader = self._get_loader(dataset, val_idx, shuffle=False)
            model = self._build_model(dataset).to(self.device)
            if self.is_distributed and dist.is_initialized():
                model = DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=True,
                )
            if self.resume_from:
                if self.cv_fold is None:
                    raise ValueError("resume_from requires --cv_fold to target a single fold.")
                if os.path.exists(self.resume_from):
                    state = torch.load(self.resume_from, map_location=self.device)
                    if self.is_distributed and dist.is_initialized():
                        model.module.load_state_dict(state)
                    else:
                        model.load_state_dict(state)
                else:
                    raise FileNotFoundError(f"resume_from not found: {self.resume_from}")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            best_fold_val = math.inf
            higher_is_better = self._monitor_higher_is_better()
            best_monitor = -math.inf if higher_is_better else math.inf
            best_state = None
            epochs_no_improve = 0
            if self.is_main:
                print(
                    f"Starting fold {fold_idx + 1}/{n_splits} with {len(train_idx)} train subjects and {len(val_idx)} val subjects"
                )
            for epoch in range(self.max_epochs):
                self._maybe_freeze_backbone_for_innovation(model, epoch)
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
                if val_loss < best_fold_val:
                    best_fold_val = val_loss

                val_sc_metrics: Dict[str, float] = {}
                compute_sc = self._monitor_requires_sc() or (
                    self.val_sc_eval_every > 0 and ((epoch + 1) % self.val_sc_eval_every == 0)
                )
                if compute_sc:
                    val_sc_metrics = self._evaluate_sc_metrics(
                        model,
                        val_loader,
                        triu_idx,
                        stats,
                        volume_indices,
                    )
                val_density = float(getattr(self, "_last_epoch_val_metrics", {}).get("density", 0.0))
                monitor_value = self._compute_monitor_value(val_loss, val_sc_metrics, val_density)
                if self.is_main:
                    print(
                        f"Fold {fold_idx + 1}/{n_splits}, epoch {epoch + 1}/{self.max_epochs}, "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                        f"monitor({self.early_stop_metric})={monitor_value:.6f}"
                    )
                    train_m = getattr(self, "_last_epoch_train_metrics", {})
                    val_m = getattr(self, "_last_epoch_val_metrics", {})
                    train_c = getattr(self, "_last_epoch_train_counts", {})
                    val_c = getattr(self, "_last_epoch_val_counts", {})
                    density_factor = self._linear_warmup_factor(
                        epoch,
                        warmup_epochs=self.density_warmup_epochs,
                        ramp_epochs=self.density_ramp_epochs,
                    )
                    zero_log_factor = self._linear_warmup_factor(
                        epoch,
                        warmup_epochs=self.zero_log_warmup_epochs,
                        ramp_epochs=self.zero_log_ramp_epochs,
                    )
                    self._append_metrics_row(
                        {
                            "fold": fold_idx,
                            "epoch": epoch + 1,
                            "train_loss": float(train_loss),
                            "val_loss": float(val_loss),
                            "monitor_metric": self.early_stop_metric,
                            "monitor_value": float(monitor_value),
                            "train_manifold": float(train_m.get("manifold", 0.0)),
                            "train_vel": float(train_m.get("vel", 0.0)),
                            "train_acc": float(train_m.get("acc", 0.0)),
                            "train_topo": float(train_m.get("topo", 0.0)),
                            "train_topo_raw": float(train_m.get("topo_raw", 0.0)),
                            "train_full_log": float(train_m.get("full_log", 0.0)),
                            "train_zero_log": float(train_m.get("zero_log", 0.0)),
                            "train_delta_log": float(train_m.get("delta_log", 0.0)),
                            "train_density": float(train_m.get("density", 0.0)),
                            "train_new_edge": float(train_m.get("new_edge", 0.0)),
                            "train_new_sparse": float(train_m.get("new_sparse", 0.0)),
                            "train_new_reg": float(train_m.get("new_reg", 0.0)),
                            "train_new_q_mean": float(train_m.get("new_q_mean", 0.0)),
                            "train_new_kept_mean": float(train_m.get("new_kept_mean", 0.0)),
                            "train_mask_ratio": float(train_m.get("mask_ratio", 0.0)),
                            "train_mask_mean_p": float(train_m.get("mask_mean_p", 0.0)),
                            "train_mask_p10": float(train_m.get("mask_p10", 0.0)),
                            "train_mask_p50": float(train_m.get("mask_p50", 0.0)),
                            "train_mask_p90": float(train_m.get("mask_p90", 0.0)),
                            "train_mask_precision_at_k": float(train_m.get("mask_precision_at_k", 0.0)),
                            "train_mask_recall_at_k": float(train_m.get("mask_recall_at_k", 0.0)),
                            "train_mask_auprc": float(train_m.get("mask_auprc", 0.0)),
                            "train_vel_count": int(train_c.get("vel", 0)),
                            "train_acc_count": int(train_c.get("acc", 0)),
                            "train_full_log_count": int(train_c.get("full_log", 0)),
                            "train_zero_log_count": int(train_c.get("zero_log", 0)),
                            "train_delta_log_count": int(train_c.get("delta_log", 0)),
                            "train_density_count": int(train_c.get("density", 0)),
                            "train_new_edge_count": int(train_c.get("new_edge", 0)),
                            "train_new_sparse_count": int(train_c.get("new_sparse", 0)),
                            "train_new_reg_count": int(train_c.get("new_reg", 0)),
                            "train_new_diag_count": int(train_c.get("new_diag", 0)),
                            "train_mask_diag_count": int(train_c.get("mask_diag", 0)),
                            "val_manifold": float(val_m.get("manifold", 0.0)),
                            "val_vel": float(val_m.get("vel", 0.0)),
                            "val_acc": float(val_m.get("acc", 0.0)),
                            "val_topo": float(val_m.get("topo", 0.0)),
                            "val_topo_raw": float(val_m.get("topo_raw", 0.0)),
                            "val_full_log": float(val_m.get("full_log", 0.0)),
                            "val_zero_log": float(val_m.get("zero_log", 0.0)),
                            "val_delta_log": float(val_m.get("delta_log", 0.0)),
                            "val_density": float(val_m.get("density", 0.0)),
                            "val_new_edge": float(val_m.get("new_edge", 0.0)),
                            "val_new_sparse": float(val_m.get("new_sparse", 0.0)),
                            "val_new_reg": float(val_m.get("new_reg", 0.0)),
                            "val_new_q_mean": float(val_m.get("new_q_mean", 0.0)),
                            "val_new_kept_mean": float(val_m.get("new_kept_mean", 0.0)),
                            "val_mask_ratio": float(val_m.get("mask_ratio", 0.0)),
                            "val_mask_mean_p": float(val_m.get("mask_mean_p", 0.0)),
                            "val_mask_p10": float(val_m.get("mask_p10", 0.0)),
                            "val_mask_p50": float(val_m.get("mask_p50", 0.0)),
                            "val_mask_p90": float(val_m.get("mask_p90", 0.0)),
                            "val_mask_precision_at_k": float(val_m.get("mask_precision_at_k", 0.0)),
                            "val_mask_recall_at_k": float(val_m.get("mask_recall_at_k", 0.0)),
                            "val_mask_auprc": float(val_m.get("mask_auprc", 0.0)),
                            "val_vel_count": int(val_c.get("vel", 0)),
                            "val_acc_count": int(val_c.get("acc", 0)),
                            "val_full_log_count": int(val_c.get("full_log", 0)),
                            "val_zero_log_count": int(val_c.get("zero_log", 0)),
                            "val_delta_log_count": int(val_c.get("delta_log", 0)),
                            "val_density_count": int(val_c.get("density", 0)),
                            "val_new_edge_count": int(val_c.get("new_edge", 0)),
                            "val_new_sparse_count": int(val_c.get("new_sparse", 0)),
                            "val_new_reg_count": int(val_c.get("new_reg", 0)),
                            "val_new_diag_count": int(val_c.get("new_diag", 0)),
                            "val_mask_diag_count": int(val_c.get("mask_diag", 0)),
                            "enable_vel": int(epoch >= self.warmup_manifold_epochs),
                            "enable_acc": int(epoch >= self.warmup_vel_epochs),
                            "lambda_manifold": float(self.lambda_manifold),
                            "lambda_vel": float(self.lambda_vel),
                            "lambda_acc": float(self.lambda_acc),
                            "lambda_kl": float(self.lambda_kl),
                            "lambda_topo": float(self.lambda_topo),
                            "lambda_full_log_mse": float(self.lambda_full_log_mse),
                            "lambda_zero_log": float(self.lambda_zero_log),
                            "lambda_delta_log": float(self.lambda_delta_log),
                            "lambda_density": float(self.lambda_density),
                            "fixed_support": int(self.fixed_support),
                            "innovation_enabled": int(self.innovation_enabled),
                            "innovation_topm": int(self.innovation_topm),
                            "innovation_k_new": int(self.innovation_k_new),
                            "innovation_tau": float(self.innovation_tau),
                            "innovation_delta_quantile": float(self.innovation_delta_quantile),
                            "innovation_dt_scale_years": float(self.innovation_dt_scale_years),
                            "innovation_focal_gamma": float(self.innovation_focal_gamma),
                            "innovation_focal_alpha": float(self.innovation_focal_alpha),
                            "lambda_new_sparse": float(self.lambda_new_sparse),
                            "new_sparse_warmup_epochs": int(self.new_sparse_warmup_epochs),
                            "new_sparse_ramp_epochs": int(self.new_sparse_ramp_epochs),
                            "lambda_new_reg": float(self.lambda_new_reg),
                            "density_warmup_epochs": int(self.density_warmup_epochs),
                            "density_ramp_epochs": int(self.density_ramp_epochs),
                            "density_factor": float(density_factor),
                            "edge_loss": str(self.edge_loss),
                            "edge_pos_weight": float(self.edge_pos_weight),
                            "focal_gamma": float(self.focal_gamma),
                            "focal_alpha": float(self.focal_alpha) if self.focal_alpha is not None else float("nan"),
                            "compute_mask_auprc": int(self.compute_mask_auprc),
                            "zero_log_warmup_epochs": int(self.zero_log_warmup_epochs),
                            "zero_log_ramp_epochs": int(self.zero_log_ramp_epochs),
                            "zero_log_factor": float(zero_log_factor),
                            "topo_scale": float(self._topo_scale),
                            "topo_scale_quantile": float(self.topo_scale_quantile),
                            "topo_log_compress": int(self.topo_log_compress),
                            "topo_warmup_frac": float(self.topo_warmup_frac),
                            "topo_warmup_factor": float(self._topo_warmup_factor(epoch)),
                            "gradnorm_scope": self.gradnorm_scope,
                            "gradnorm_alpha": float(self.gradnorm_alpha),
                            "gradnorm_weight_manifold": float(self._gradnorm_weights.get("manifold", 1.0)),
                            "gradnorm_weight_topo": float(self._gradnorm_weights.get("topo", 1.0)),
                            "topo_loss_bins": int(self.topo_loss_bins),
                            "betti_sharpness": float(self.betti_sharpness),
                            "betti_t": float(self.betti_t),
                            "betti_taylor_order": int(self.betti_taylor_order),
                            "betti_probes": int(self.betti_probes),
                            "morph_noise_sigma": float(self.morph_noise_sigma),
                            "sc_pos_edge_drop_prob": float(self.sc_pos_edge_drop_prob),
                            "residual_skip": int(self.residual_skip),
                            "residual_tau": float(self.residual_tau),
                            "residual_cap": float(self.residual_cap),
                            "val_sc_eval_every": int(self.val_sc_eval_every),
                            "early_stop_metric": self.early_stop_metric,
                            "early_stop_density_weight": float(self.early_stop_density_weight),
                            "val_sc_log_mse": float(val_sc_metrics.get("sc_log_mse", 0.0)),
                            "val_sc_log_mae": float(val_sc_metrics.get("sc_log_mae", 0.0)),
                            "val_sc_log_pearson": float(val_sc_metrics.get("sc_log_pearson", 0.0)),
                            "val_sc_log_pearson_pos": float(val_sc_metrics.get("sc_log_pearson_pos", 0.0)),
                            "val_sc_log_pearson_topk": float(val_sc_metrics.get("sc_log_pearson_topk", 0.0)),
                            "val_sc_log_pearson_sparse": float(val_sc_metrics.get("sc_log_pearson_sparse", 0.0)),
                            "val_ecc_l2": float(val_sc_metrics.get("ecc_l2", 0.0)),
                            "val_ecc_pearson": float(val_sc_metrics.get("ecc_pearson", 0.0)),
                        }
                    )

                improved = (monitor_value > best_monitor) if higher_is_better else (monitor_value < best_monitor)
                if improved:
                    best_monitor = monitor_value
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
                            f"Fold {fold_idx + 1}/{n_splits} early stopped at epoch {epoch + 1} "
                            f"with best_monitor({self.early_stop_metric})={best_monitor:.6f}"
                        )
                    break
            fold_results.append(
                {
                    "fold": fold_idx,
                    "best_val_loss": float(best_fold_val),
                    "best_monitor_metric": self.early_stop_metric,
                    "best_monitor": float(best_monitor),
                }
            )
            fold_model_path = os.path.join(
                self.results_dir,
                f"clg_ode_fold{fold_idx}_best.pt",
            )
            if best_state is not None and self.is_main:
                torch.save(best_state, fold_model_path)
            fold_score = -best_monitor if higher_is_better else best_monitor
            if fold_score < best_score:
                best_score = fold_score
                best_model_path = fold_model_path
                best_stats = stats
        if self.is_main:
            self.summary["cv_folds"] = fold_results
            self.summary["cv_fold"] = self.cv_fold
            self.summary["cv_folds_total"] = n_splits
            self.summary["best_cv_score"] = float(best_score)
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
