"""HMC Reasoning Engine — Hamiltonian Monte Carlo with learned mass matrix."""

import torch
from torch import Tensor, nn


class HMCEngine(nn.Module):
    """
    Hamiltonian Monte Carlo inference with task-conditioned mass matrix.

    Mass matrix M = metric tensor g(x) from AEG.
    Momentum p ~ N(0, M), so dynamics follow important directions.
    Leapfrog integration + Metropolis-Hastings correction.
    """

    def __init__(self, d_latent: int, n_leapfrog: int = 8,
                 n_steps: int = 20, n_chains: int = 4,
                 step_size: float = 0.02):
        super().__init__()
        self.d_latent = d_latent
        self.n_leapfrog = n_leapfrog
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.log_eps = nn.Parameter(torch.tensor(step_size).log())

    def _leapfrog(self, z: Tensor, p: Tensor, energy_fn,
                  M_inv: Tensor, eps: float, n: int) -> tuple[Tensor, Tensor, Tensor]:
        """Leapfrog integrator."""
        z = z.detach().requires_grad_(True)
        E = energy_fn(z)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0]
        p = p - 0.5 * eps * g

        for i in range(n):
            z_new = z + eps * (M_inv @ p.unsqueeze(-1)).squeeze(-1)
            z_new = z_new.detach().requires_grad_(True)
            E = energy_fn(z_new)
            g = torch.autograd.grad(E.sum(), z_new, create_graph=False)[0]
            p = p - (eps if i < n - 1 else 0.5 * eps) * g
            z = z_new

        return z, p, E

    def forward(self, energy_fn, metric: Tensor,
                batch_size: int, device: torch.device) -> tuple[Tensor, dict]:
        """
        Run HMC to find lowest-energy z.

        Args:
            energy_fn: z → (B*C,) energy
            metric: (B, d, d) from AEG
            batch_size: B
            device: torch device
        Returns:
            z_star: (B, d_latent)
            diag: dict with acceptance_rate, best_energy
        """
        eps = self.log_eps.exp().item()
        C = self.n_chains
        BC = batch_size * C

        # Expand metric for chains
        M = metric.repeat_interleave(C, dim=0)

        # Safe Cholesky: add small diagonal for numerical stability
        M_stable = M + 1e-5 * torch.eye(self.d_latent, device=device).unsqueeze(0)
        try:
            L = torch.linalg.cholesky(M_stable)
        except torch.linalg.LinAlgError:
            L = torch.eye(self.d_latent, device=device).unsqueeze(0).expand(BC, -1, -1)

        M_inv = torch.linalg.inv(M_stable)

        z = torch.randn(BC, self.d_latent, device=device)
        best_z = z.clone()
        best_E = torch.full((BC,), float('inf'), device=device)
        accepted = 0

        with torch.enable_grad():
            for _ in range(self.n_steps):
                # Sample momentum p ~ N(0, M)
                noise = torch.randn(BC, self.d_latent, device=device)
                p = (L @ noise.unsqueeze(-1)).squeeze(-1)

                # Current H
                z_req = z.detach().requires_grad_(True)
                E_old = energy_fn(z_req)
                K_old = 0.5 * (p.unsqueeze(-2) @ M_inv @ p.unsqueeze(-1)).squeeze()

                # Leapfrog
                z_new, p_new, E_new = self._leapfrog(
                    z_req, p, energy_fn, M_inv, eps, self.n_leapfrog
                )

                K_new = 0.5 * (p_new.unsqueeze(-2) @ M_inv @ p_new.unsqueeze(-1)).squeeze()

                # Metropolis-Hastings
                dH = (E_new + K_new) - (E_old.detach() + K_old)
                accept_p = torch.exp(-dH.clamp(min=-20, max=20)).clamp(max=1.0)
                accept = torch.rand(BC, device=device) < accept_p

                z = torch.where(accept.unsqueeze(-1), z_new.detach(), z)
                accepted += accept.float().sum().item()

                # Track best
                with torch.no_grad():
                    cur_E = energy_fn(z)
                    better = cur_E < best_E
                    best_z = torch.where(better.unsqueeze(-1), z, best_z)
                    best_E = torch.where(better, cur_E, best_E)

        # Select best chain per input
        best_E_r = best_E.reshape(batch_size, C)
        best_idx = best_E_r.argmin(dim=1)
        best_z_r = best_z.reshape(batch_size, C, self.d_latent)
        z_star = best_z_r[torch.arange(batch_size, device=device), best_idx]

        diag = {
            'acceptance_rate': accepted / (self.n_steps * BC),
            'best_energy': best_E_r.min(dim=1).values.mean().item(),
        }
        return z_star, diag
