from __future__ import annotations

import torch


def topology_loss(
    a_pred: torch.Tensor,
    a_true: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    _ = a_true
    _ = mask
    return torch.tensor(0.0, device=a_pred.device)
