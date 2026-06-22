"""Universal Serializer — converts any structured data to token sequences."""

import math
import torch
from torch import Tensor, nn

from alpha_g.config import ArchConfig


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, n: int) -> Tensor:
        return self.pe[:, :n]


class Learned2DPE(nn.Module):
    """Learned 2D positional encoding: row_embed + col_embed."""
    def __init__(self, d_model: int, max_size: int = 32):
        super().__init__()
        self.row_embed = nn.Embedding(max_size, d_model)
        self.col_embed = nn.Embedding(max_size, d_model)

    def forward(self, H: int, W: int) -> Tensor:
        """Returns (1, H*W, d_model)."""
        rows = torch.arange(H, device=self.row_embed.weight.device)
        cols = torch.arange(W, device=self.col_embed.weight.device)
        row_pe = self.row_embed(rows.unsqueeze(1).expand(H, W).reshape(-1))
        col_pe = self.col_embed(cols.unsqueeze(0).expand(H, W).reshape(-1))
        return (row_pe + col_pe).unsqueeze(0)


class UniversalSerializer(nn.Module):
    """
    Converts structured data to (B, N, d_model) token sequences.
    Handles 2D grids of any size, 1D sequences, or flat sets.
    """
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.value_embed = nn.Embedding(cfg.max_vocab, cfg.d_model)
        self.pos_1d = SinusoidalPE(cfg.d_model, cfg.max_seq_len)
        self.pos_2d = Learned2DPE(cfg.d_model, cfg.max_grid)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.sep_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.proj_norm = nn.LayerNorm(cfg.d_model)

    def serialize_grid(self, grid: Tensor, H: int, W: int) -> Tensor:
        """
        grid: (B, H, W) integer values → (B, H*W, d_model).
        """
        B = grid.shape[0]
        flat = grid.reshape(B, H * W)
        tokens = self.value_embed(flat)
        tokens = tokens + self.pos_2d(H, W)
        return tokens

    def forward(
        self,
        grids: list[Tensor],
        shapes: list[tuple[int, int]],
    ) -> tuple[Tensor, Tensor]:
        """
        Serialize one or more grids into a single padded token sequence.
        Grids are separated by [SEP] tokens, prepended with [CLS].

        Args:
            grids: list of (B, H_i, W_i) integer tensors
            shapes: list of (H_i, W_i) tuples
        Returns:
            tokens: (B, N_total, d_model)
            padding_mask: (B, N_total) — True = padded position
        """
        B = grids[0].shape[0]
        device = grids[0].device
        parts = [self.cls_token.expand(B, -1, -1)]

        for grid, (H, W) in zip(grids, shapes):
            parts.append(self.serialize_grid(grid, H, W))
            parts.append(self.sep_token.expand(B, -1, -1))

        tokens = torch.cat(parts, dim=1)
        tokens = self.proj_norm(tokens)
        mask = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=device)
        return tokens, mask
