"""Universal Decoder — latent z + BEP features → output tokens."""

import torch
from torch import Tensor, nn

from alpha_g.config import ArchConfig
from alpha_g.model.serializer import Learned2DPE


class UniversalDecoder(nn.Module):
    """Decodes latent z + BEP features to per-cell output logits."""

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.token_seed = nn.Linear(cfg.d_model + cfg.d_latent, cfg.d_model)
        self.pos_2d = Learned2DPE(cfg.d_model, cfg.max_grid)

        self.cross_layers = nn.ModuleList()
        self.self_layers = nn.ModuleList()
        for _ in range(cfg.dec_layers):
            self.cross_layers.append(
                nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            )
            self.self_layers.append(
                nn.TransformerEncoderLayer(
                    cfg.d_model, cfg.n_heads, cfg.d_ffn,
                    dropout=cfg.dropout, activation='gelu',
                    batch_first=True, norm_first=True,
                )
            )

        self.cross_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.dec_layers)])
        self.head = nn.Linear(cfg.d_model, cfg.max_vocab)

    def forward(self, z: Tensor, context: Tensor, bep_features: Tensor,
                H: int, W: int, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            z: (B, d_latent)
            context: (B, d_model)
            bep_features: (B, N_in, d_model) from BEP
            H, W: output grid dimensions
        Returns:
            (B, H, W, max_vocab) logits
        """
        B = z.shape[0]
        N_out = H * W

        # Create output seeds from (z, context)
        seed = self.token_seed(torch.cat([z, context], dim=-1))  # (B, d_model)
        h = seed.unsqueeze(1).expand(B, N_out, -1).clone()

        # Add 2D position
        h = h + self.pos_2d(H, W)

        # Cross-attend to BEP features + self-attend
        for cross, self_attn, norm in zip(self.cross_layers, self.self_layers, self.cross_norms):
            r = h
            h_n = norm(h)
            h_c, _ = cross(h_n, bep_features, bep_features, key_padding_mask=mask)
            h = r + h_c
            h = self_attn(h)

        logits = self.head(h).reshape(B, H, W, -1)
        return logits
