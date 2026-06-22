"""Bidirectional Equilibrium Processor — two counter-flowing streams negotiate."""

import torch
from torch import Tensor, nn

from alpha_g.config import ArchConfig


class BEPLayer(nn.Module):
    """
    Single bidirectional layer with bottom-up, top-down, and equilibrium gate.
    """

    def __init__(self, d_model: int, d_latent: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        # Bottom-up (data-driven)
        self.bu_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.bu_norm1 = nn.LayerNorm(d_model)
        self.bu_ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.bu_norm2 = nn.LayerNorm(d_model)

        # Top-down (hypothesis-driven)
        self.td_z_proj = nn.Linear(d_latent, d_model)
        self.td_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.td_norm1 = nn.LayerNorm(d_model)
        self.td_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.td_norm2 = nn.LayerNorm(d_model)
        self.td_ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
        self.td_norm3 = nn.LayerNorm(d_model)

        # Equilibrium gate
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )

    def forward(self, h_bu: Tensor, h_td: Tensor, z: Tensor,
                context: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        # Bottom-up: self-attention + FFN
        r = h_bu
        h_bu = self.bu_norm1(h_bu)
        h_bu, _ = self.bu_attn(h_bu, h_bu, h_bu, key_padding_mask=mask)
        h_bu = r + h_bu
        h_bu = h_bu + self.bu_ffn(self.bu_norm2(h_bu))

        # Top-down: cross-attend to (z, context), then self-attend + FFN
        r = h_td
        h_td = self.td_norm1(h_td)
        z_tok = self.td_z_proj(z).unsqueeze(1)
        kv = torch.cat([z_tok, context.unsqueeze(1)], dim=1)
        h_td_c, _ = self.td_cross(h_td, kv, kv)
        h_td = r + h_td_c

        r2 = h_td
        h_td = self.td_norm2(h_td)
        h_td_a, _ = self.td_attn(h_td, h_td, h_td, key_padding_mask=mask)
        h_td = r2 + h_td_a
        h_td = h_td + self.td_ffn(self.td_norm3(h_td))

        # Gate: negotiate
        g = self.gate(torch.cat([h_bu, h_td], dim=-1))
        merged = g * h_bu + (1 - g) * h_td

        return h_bu, h_td, merged


class BidirectionalEquilibriumProcessor(nn.Module):
    """Stack of BEP layers with optional equilibrium iterations."""

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            BEPLayer(cfg.d_model, cfg.d_latent, cfg.n_heads, cfg.d_ffn, cfg.dropout)
            for _ in range(cfg.bep_layers)
        ])
        self.n_iters = cfg.bep_equilibrium_iters
        self.z_pred_proj = nn.Linear(cfg.d_model, cfg.d_latent)

    def forward(self, tokens: Tensor, z: Tensor, context: Tensor,
                mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Returns:
            merged: (B, N, d_model) — equilibrium features
            z_pred: (B, d_model) — predicted target representation
        """
        h_bu = tokens
        h_td = tokens

        for _ in range(self.n_iters):
            for layer in self.layers:
                h_bu, h_td, merged = layer(h_bu, h_td, z, context, mask)
            if self.n_iters > 1:
                h_bu = merged
                h_td = merged

        z_pred = self.z_pred_proj(merged[:, 0])
        return merged, z_pred
