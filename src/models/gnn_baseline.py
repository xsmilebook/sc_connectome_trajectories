from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = adj.shape[-1]
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj_hat = adj + eye
        deg = adj_hat.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)
        h = torch.matmul(norm, x)
        return self.linear(h)


class GNNBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(1, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True)
        x = torch.log1p(deg)
        h = F.relu(self.gcn1(x, adj))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, adj))
        scores = torch.matmul(h, h.transpose(-1, -2))
        pred = F.softplus(scores)
        pred = 0.5 * (pred + pred.transpose(-1, -2))
        diag = torch.diagonal(pred, dim1=-2, dim2=-1)
        pred = pred - torch.diag_embed(diag)
        return pred
