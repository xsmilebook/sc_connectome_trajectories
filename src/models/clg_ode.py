from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CLGOutput:
    x_hat: torch.Tensor
    a_logit: torch.Tensor
    a_weight: torch.Tensor
    z_morph: torch.Tensor
    z_conn: torch.Tensor
    mu_morph: torch.Tensor
    logvar_morph: torch.Tensor
    mu_conn: torch.Tensor
    logvar_conn: torch.Tensor
    a_weight_new: torch.Tensor | None = None
    l_new: torch.Tensor | None = None


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_norm = normalize_adj(a)
        h = torch.matmul(a_norm, x)
        h = self.act(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CovariateEncoder(nn.Module):
    def __init__(self, sex_dim: int, site_dim: int, embed_dim: int, numeric_dim: int) -> None:
        super().__init__()
        self.sex_embed = nn.Embedding(sex_dim, embed_dim)
        self.site_embed = nn.Embedding(site_dim, embed_dim)
        self.numeric_dim = numeric_dim

    def forward(self, sex: torch.Tensor, site: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        sex_emb = self.sex_embed(sex)
        site_emb = self.site_embed(site)
        if self.numeric_dim == 0:
            return torch.cat([sex_emb, site_emb], dim=-1)
        return torch.cat([sex_emb, site_emb, numeric], dim=-1)


class CoupledODEFunc(nn.Module):
    def __init__(self, latent_dim: int, cov_dim: int, hidden_dim: int) -> None:
        super().__init__()
        in_dim = latent_dim * 2 + cov_dim + 1
        self.morph_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.conn_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        z_morph, z_conn = torch.chunk(z, 2, dim=-1)
        b, n, _ = z_morph.shape
        t_exp = t.view(b, 1, 1).expand(b, n, 1)
        cov_exp = cov.view(b, 1, -1).expand(b, n, -1)
        feat = torch.cat([z_morph, z_conn, cov_exp, t_exp], dim=-1)
        dz_morph = self.morph_net(feat)
        dz_conn = self.conn_net(feat)
        return torch.cat([dz_morph, dz_conn], dim=-1)


class MorphDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConnDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_bias = nn.Parameter(torch.tensor(0.0))
        self.alpha_new = nn.Parameter(torch.tensor(10.0))
        self.delta_new = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        z: torch.Tensor,
        a0_log: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
        residual_skip: bool = False,
        residual_tau: float = 1.0,
        residual_cap: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        score = torch.matmul(z, z.transpose(-1, -2))
        logit = self.alpha * (score - self.delta)
        l_new = self.alpha_new * (score - self.delta_new)
        weight_new = F.softplus(self.gamma * score + self.beta)
        if residual_skip and a0_log is not None and times is not None:
            delta_log = self.residual_scale * score + self.residual_bias
            delta_log = torch.tanh(delta_log) * float(residual_cap)
            if times.dim() == 1:
                times = times.unsqueeze(0).expand(z.shape[0], -1)
            scale = times / (times + float(residual_tau))
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            if a0_log.dim() == 3 and z.dim() == 4:
                base = a0_log.unsqueeze(1)
            else:
                base = a0_log
            support = (base > 0).to(dtype=delta_log.dtype)
            pred_log = base + scale * (delta_log * support)
            weight = torch.expm1(pred_log)
        else:
            weight = weight_new
        weight = weight - torch.diag_embed(torch.diagonal(weight, dim1=-2, dim2=-1))
        weight_new = weight_new - torch.diag_embed(torch.diagonal(weight_new, dim1=-2, dim2=-1))
        return logit, weight, weight_new, l_new


class CLGODE(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        morph_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        sex_dim: int = 3,
        site_dim: int = 128,
        cov_embed_dim: int = 8,
        numeric_cov_dim: int = 0,
        solver_steps: int = 8,
        residual_skip: bool = False,
        residual_tau: float = 1.0,
        residual_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.solver_steps = solver_steps
        self.residual_skip = residual_skip
        self.residual_tau = residual_tau
        self.residual_cap = residual_cap
        self.morph_encoder = GraphEncoder(morph_dim, hidden_dim, latent_dim)
        self.conn_encoder = GraphEncoder(morph_dim, hidden_dim, latent_dim)
        self.cov_encoder = CovariateEncoder(
            sex_dim=sex_dim,
            site_dim=site_dim,
            embed_dim=cov_embed_dim,
            numeric_dim=numeric_cov_dim,
        )
        self.ode_func = CoupledODEFunc(
            latent_dim=latent_dim,
            cov_dim=cov_embed_dim * 2 + numeric_cov_dim,
            hidden_dim=hidden_dim,
        )
        self.morph_decoder = MorphDecoder(latent_dim, morph_dim, hidden_dim)
        self.conn_decoder = ConnDecoder()

    def forward(
        self,
        a0: torch.Tensor,
        x0: torch.Tensor,
        times: torch.Tensor,
        sex: torch.Tensor,
        site: torch.Tensor,
        covariates: torch.Tensor,
        use_mu: bool = False,
    ) -> CLGOutput:
        mu_morph, logvar_morph = self.morph_encoder(x0, a0)
        mu_conn, logvar_conn = self.conn_encoder(x0, a0)
        if use_mu:
            z_morph0 = mu_morph
            z_conn0 = mu_conn
        else:
            z_morph0 = reparameterize(mu_morph, logvar_morph, self.training)
            z_conn0 = reparameterize(mu_conn, logvar_conn, self.training)
        z0 = torch.cat([z_morph0, z_conn0], dim=-1)
        cov = self.cov_encoder(sex, site, covariates)
        zt = integrate_latent(self.ode_func, z0, times, cov, self.solver_steps)
        z_morph_t, z_conn_t = torch.chunk(zt, 2, dim=-1)
        x_hat = self.morph_decoder(z_morph_t)
        a_logit, a_weight, a_weight_new, l_new = self.conn_decoder(
            z_conn_t,
            a0_log=a0,
            times=times,
            residual_skip=self.residual_skip,
            residual_tau=self.residual_tau,
            residual_cap=self.residual_cap,
        )
        return CLGOutput(
            x_hat=x_hat,
            a_logit=a_logit,
            a_weight=a_weight,
            z_morph=z_morph_t,
            z_conn=z_conn_t,
            mu_morph=mu_morph,
            logvar_morph=logvar_morph,
            mu_conn=mu_conn,
            logvar_conn=logvar_conn,
            a_weight_new=a_weight_new,
            l_new=l_new,
        )


def normalize_adj(a: torch.Tensor) -> torch.Tensor:
    n = a.shape[-1]
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    a_hat = a + eye
    deg = a_hat.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    return deg_inv_sqrt @ a_hat @ deg_inv_sqrt


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def integrate_latent(
    func: CoupledODEFunc,
    z0: torch.Tensor,
    times: torch.Tensor,
    cov: torch.Tensor,
    steps_per_interval: int,
) -> torch.Tensor:
    if times.dim() == 1:
        times = times.unsqueeze(0).expand(z0.shape[0], -1)
    batch, t_len = times.shape
    zt = []
    for b in range(batch):
        z = z0[b]
        t_seq = times[b]
        out = []
        for idx in range(t_len):
            if idx == 0:
                out.append(z)
                continue
            t0 = t_seq[idx - 1]
            t1 = t_seq[idx]
            z = rk4_integrate(func, z, t0, t1, cov[b], steps_per_interval)
            out.append(z)
        zt.append(torch.stack(out, dim=0))
    return torch.stack(zt, dim=0)


def rk4_integrate(
    func: CoupledODEFunc,
    z0: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    cov: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    if steps < 1:
        steps = 1
    h = (t1 - t0) / steps
    z = z0
    t = t0
    for _ in range(steps):
        k1 = func(t, z.unsqueeze(0), cov.unsqueeze(0)).squeeze(0)
        k2 = func(t + 0.5 * h, (z + 0.5 * h * k1).unsqueeze(0), cov.unsqueeze(0)).squeeze(0)
        k3 = func(t + 0.5 * h, (z + 0.5 * h * k2).unsqueeze(0), cov.unsqueeze(0)).squeeze(0)
        k4 = func(t + h, (z + h * k3).unsqueeze(0), cov.unsqueeze(0)).squeeze(0)
        z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return z
