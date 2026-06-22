"""Alpha-G: The Orchestrator."""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from alpha_g.config import ArchConfig, TrainConfig
from alpha_g.model.aeg import AdaptiveEnergyGeometry
from alpha_g.model.bep import BidirectionalEquilibriumProcessor
from alpha_g.model.decoder import UniversalDecoder
from alpha_g.model.hmc import HMCEngine
from alpha_g.model.serializer import UniversalSerializer


@dataclass
class AlphaGOutput:
    """Output from a training forward pass."""
    energy: Tensor
    z_pred: Tensor
    z_target: Tensor
    context: Tensor
    metric: Tensor
    logits: Tensor
    merged_features: Tensor


class AlphaG(nn.Module):
    """
    Alpha-G: General-Purpose Energy-Based Reasoning Model.

    Combines Universal Serializer, Adaptive Energy Geometry,
    Bidirectional Equilibrium Processor, and HMC Engine.
    """

    def __init__(self, arch_cfg: ArchConfig, train_cfg: TrainConfig):
        super().__init__()
        self.arch_cfg = arch_cfg
        self.train_cfg = train_cfg

        self.serializer = UniversalSerializer(arch_cfg)
        self.aeg = AdaptiveEnergyGeometry(arch_cfg)
        self.bep = BidirectionalEquilibriumProcessor(arch_cfg)
        self.decoder = UniversalDecoder(arch_cfg)
        self.hmc = HMCEngine(
            d_latent=arch_cfg.d_latent,
            n_leapfrog=train_cfg.hmc_leapfrog,
            n_steps=train_cfg.hmc_steps,
            n_chains=train_cfg.hmc_chains,
            step_size=train_cfg.hmc_step_size,
        )

        # Target encoder: EMA copy of AEG context encoder
        self.target_encoder = copy.deepcopy(self.aeg.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.z_encoder = nn.Sequential(
            nn.Linear(arch_cfg.d_model, arch_cfg.d_latent),
            nn.LayerNorm(arch_cfg.d_latent)
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        """EMA update for the target encoder."""
        for t_p, s_p in zip(self.target_encoder.parameters(), self.aeg.encoder.parameters()):
            t_p.lerp_(s_p, 1.0 - momentum)

    def forward(self, input_grid: Tensor, target_grid: Tensor, shapes: list[tuple[int, int]]) -> AlphaGOutput:
        """
        Training forward pass.

        Args:
            input_grid: (B, H, W)
            target_grid: (B, H, W)
            shapes: list of (H, W) pairs
        """
        # Serialize input
        tokens, mask = self.serializer([input_grid], shapes)

        # 1. AEG: Geometry & Context
        context, h_seq, metric = self.aeg(tokens, mask)

        # 2. Target Embedding
        with torch.no_grad():
            t_tokens, t_mask = self.serializer([target_grid], shapes)
            t_h = self.target_encoder(t_tokens, src_key_padding_mask=t_mask)
            z_target = self.z_encoder(t_h[:, 0])
            z_target = F.normalize(z_target, dim=-1)

        # Add noise to target z for training
        z_noisy = z_target + self.train_cfg.z_noise * torch.randn_like(z_target)

        # 3. BEP: Bidirectional Processing
        merged, z_pred = self.bep(tokens, z_noisy, context, mask)

        # 4. Energy
        energy = self.aeg.energy(z_pred, z_target, metric)

        # 5. Decode
        H, W = shapes[0]
        logits = self.decoder(z_noisy, context, merged, H, W, mask)

        return AlphaGOutput(
            energy=energy,
            z_pred=z_pred,
            z_target=z_target,
            context=context,
            metric=metric,
            logits=logits,
            merged_features=merged,
        )

    def solve(self, input_grid: Tensor, shapes: list[tuple[int, int]], out_H: int, out_W: int) -> tuple[Tensor, dict]:
        """
        Inference: find best solution via HMC.
        """
        device = input_grid.device
        B = input_grid.shape[0]

        # 1. Encode Context
        tokens, mask = self.serializer([input_grid], shapes)
        context, _, metric = self.aeg(tokens, mask)

        C = self.train_cfg.hmc_chains
        tokens_exp = tokens.repeat_interleave(C, dim=0)
        mask_exp = mask.repeat_interleave(C, dim=0)
        context_exp = context.repeat_interleave(C, dim=0)
        metric_exp = metric.repeat_interleave(C, dim=0)

        # Define energy function for HMC
        def energy_fn(z: Tensor) -> Tensor:
            merged, z_pred = self.bep(tokens_exp, z, context_exp, mask_exp)
            logits = self.decoder(z, context_exp, merged, out_H, out_W, mask_exp)

            # Self-consistency check
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                # Soft-serialization (continuous relaxation of value_embed)
                flat_probs = probs.reshape(B * C, out_H * out_W, -1)
                t_tokens = flat_probs @ self.serializer.value_embed.weight
                t_tokens = t_tokens + self.serializer.pos_2d(out_H, out_W)
                # Prepend CLS
                cls = self.serializer.cls_token.expand(B * C, -1, -1)
                t_tokens = torch.cat([cls, t_tokens], dim=1)

                t_h = self.target_encoder(t_tokens)
                z_re = self.z_encoder(t_h[:, 0])
                z_re = F.normalize(z_re, dim=-1)

            return self.aeg.energy(z_pred, z_re, metric_exp)

        # 2. Run HMC
        z_star, diag = self.hmc(energy_fn, metric, B, device)

        # 3. Final Decode
        merged, _ = self.bep(tokens, z_star, context, mask)
        logits = self.decoder(z_star, context, merged, out_H, out_W, mask)

        return logits.argmax(dim=-1), diag
