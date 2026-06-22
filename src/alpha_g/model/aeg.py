"""Adaptive Energy Geometry — context encoder + learned Riemannian metric."""

import torch
from torch import Tensor, nn

from alpha_g.config import ArchConfig


class AdaptiveEnergyGeometry(nn.Module):
    """
    Encodes input context and produces a task-conditioned Riemannian metric.

    The metric tensor g(x) defines the geometry of the latent reasoning space:
    - Different inputs → different energy geometries
    - HMC dynamics naturally follow important directions
    - Energy = Mahalanobis distance with learned metric
    """

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.d_latent = cfg.d_latent
        self.metric_rank = cfg.metric_rank

        # Context encoder
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ffn, dropout=cfg.dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.aeg_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        # Metric network: context → low-rank metric g = I + U Λ Uᵀ
        self.metric_U = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_latent * cfg.metric_rank),
        )
        self.metric_logL = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.metric_rank),
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            tokens: (B, N, d_model)
            mask: (B, N) padding mask
        Returns:
            context: (B, d_model) — CLS representation
            seq_features: (B, N, d_model) — full sequence features
            metric: (B, d_latent, d_latent) — Riemannian metric tensor
        """
        h = self.encoder(tokens, src_key_padding_mask=mask)
        h = self.norm(h)
        context = h[:, 0]  # CLS token

        # Compute metric
        U = self.metric_U(context).reshape(-1, self.d_latent, self.metric_rank)
        log_L = self.metric_logL(context)
        Lambda = torch.diag_embed(torch.exp(log_L.clamp(-5, 5)))
        eye = torch.eye(self.d_latent, device=context.device).unsqueeze(0)
        metric = eye + U @ Lambda @ U.transpose(-1, -2)

        return context, h, metric

    def energy(self, z_pred: Tensor, z_target: Tensor, metric: Tensor) -> Tensor:
        """Riemannian energy: (Δz)ᵀ g (Δz)."""
        diff = (z_pred - z_target).unsqueeze(-1)
        return (diff.transpose(-2, -1) @ metric @ diff).squeeze(-1).squeeze(-1)
