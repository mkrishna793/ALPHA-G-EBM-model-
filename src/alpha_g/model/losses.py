"""Loss functions for Alpha-G training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from alpha_g.config import TrainConfig


@dataclass
class LossOutput:
    """All loss components."""
    total: Tensor
    energy: Tensor
    vicreg: Tensor
    decode: Tensor
    consistency: Tensor
    geometry: Tensor


def vicreg_loss(z: Tensor, var_w: float = 1.0, cov_w: float = 0.01) -> Tensor:
    """VICReg: variance + covariance regularization to prevent collapse."""
    z = z - z.mean(dim=0)
    # Safe std calculation to prevent NaN gradients when variance is 0
    std = torch.sqrt(z.var(dim=0) + 1e-04)
    var_loss = F.relu(1.0 - std).mean()

    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(cov.diag())
    cov_loss = (off ** 2).sum() / d

    return var_w * var_loss + cov_w * cov_loss


def geometry_loss(metric: Tensor) -> Tensor:
    """
    Regularize metric tensor to prevent degeneration.
    log_det + trace/d encourages well-conditioned metric.
    """
    d = metric.shape[-1]
    # Log-determinant (numerical stability via slogdet)
    sign, logdet = torch.linalg.slogdet(metric)
    log_det_loss = -logdet.mean()  # encourage non-zero volume

    trace = torch.diagonal(metric, dim1=-2, dim2=-1).sum(dim=-1)
    trace_loss = ((trace / d - 1.0) ** 2).mean()  # encourage trace ≈ d

    return log_det_loss + trace_loss


def compute_loss(
    energy: Tensor,
    z_rule: Tensor,
    logits: Tensor,
    query_target: Tensor,
    cfg: TrainConfig,
) -> LossOutput:
    """
    Combined training loss for In-Context Rule Learning.

    Args:
        energy: (1,) average energy from support pairs
        z_rule: (B, d_model) The derived rule vector
        logits: (B, H, W, vocab) decoded query output
        query_target: (B, H, W) integer targets for query
        cfg: training config
    """
    energy_loss = energy.mean()
    
    # VICReg applied directly to the z_rule to prevent rule collapse!
    vreg = vicreg_loss(z_rule, cfg.vicreg_var_weight, cfg.vicreg_cov_weight)

    # Decode loss: cross-entropy exclusively on the query pair
    B, H, W, V = logits.shape
    decode_loss = F.cross_entropy(
        logits.reshape(-1, V), query_target.reshape(-1).long()
    )

    # We removed geometry loss and consistency loss since rule induction 
    # natively handles alignment via the bottleneck.
    consistency_loss = torch.tensor(0.0, device=logits.device)
    geo_loss = torch.tensor(0.0, device=logits.device)

    total = (
        energy_loss
        + vreg
        + cfg.decode_weight * decode_loss
    )

    return LossOutput(
        total=total, energy=energy_loss, vicreg=vreg,
        decode=decode_loss, consistency=consistency_loss, geometry=geo_loss,
    )
