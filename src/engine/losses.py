from __future__ import annotations

import torch
from torch.nn import functional as F


def edge_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    weight = torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

def edge_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = target * p + (1.0 - target) * (1.0 - p)
    modulating = (1.0 - pt) ** float(gamma)
    loss = modulating * bce
    if alpha is not None:
        alpha_t = target * float(alpha) + (1.0 - target) * (1.0 - float(alpha))
        loss = alpha_t * loss
    return loss.mean()


def weight_huber_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=pred_log.device)
    loss = F.smooth_l1_loss(pred_log, target_log, reduction="none")
    return (loss * mask).sum() / mask.sum()
