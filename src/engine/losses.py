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


def weight_huber_loss(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=pred_log.device)
    loss = F.smooth_l1_loss(pred_log, target_log, reduction="none")
    return (loss * mask).sum() / mask.sum()
