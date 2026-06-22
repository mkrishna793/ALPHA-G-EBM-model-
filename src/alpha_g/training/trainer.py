"""Training loop with mixed precision and compiling for speed."""

import math
import time

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from alpha_g.config import config, ArchConfig, TrainConfig
from alpha_g.model.alpha_g import AlphaG
from alpha_g.model.losses import compute_loss

class Trainer:
    def __init__(self, model: AlphaG, train_dl, val_dl, t_cfg: TrainConfig):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = t_cfg
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if self.cfg.use_compile and hasattr(torch, "compile"):
            print("Compiling model for speed...")
            self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
        # bfloat16 does not need scaling, so we explicitly disable it
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.step = 0
        self.total_steps = len(train_dl) * t_cfg.epochs

    def get_lr(self):
        """Linear warmup, cosine decay."""
        if self.step < self.cfg.warmup_steps:
            return self.cfg.lr * (self.step / max(1, self.cfg.warmup_steps))
        progress = (self.step - self.cfg.warmup_steps) / max(1, self.total_steps - self.cfg.warmup_steps)
        return self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}")

        for i, batch in enumerate(pbar):
            # Update LR
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            s_in = batch['s_in'].to(self.device)
            s_out = batch['s_out'].to(self.device)
            q_in = batch['q_in'].to(self.device)
            q_out = batch['q_out'].to(self.device)
            
            H, W = batch['batch_shape']
            shapes = [(H, W)] * s_in.shape[0]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.cfg.use_amp):
                out = self.model(s_in, s_out, q_in, shapes)
                loss_out = compute_loss(
                    out.energy, out.z_pred, out.logits, q_out, self.cfg
                )
                loss = loss_out.total / self.cfg.grad_accumulation

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\\nWARNING: NaN/Inf loss detected at Epoch {epoch}, Batch {i}! Skipping batch to prevent GPU crash.")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()

            if (i + 1) % self.cfg.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # EMA update
                progress = self.step / self.total_steps
                momentum = self.cfg.ema_start + (self.cfg.ema_end - self.cfg.ema_start) * progress
                m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
                m.update_target_encoder(momentum)

            self.step += 1
            total_loss += loss.item() * self.cfg.grad_accumulation
            
            pbar.set_postfix({'loss': f"{loss_out.total.item():.4f}", 'lr': f"{lr:.2e}"})

        return total_loss / len(self.train_dl)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_dl:
                s_in = batch['s_in'].to(self.device)
                s_out = batch['s_out'].to(self.device)
                q_in = batch['q_in'].to(self.device)
                q_out = batch['q_out'].to(self.device)
                
                H, W = batch['batch_shape']
                shapes = [(H, W)] * s_in.shape[0]

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.cfg.use_amp):
                    out = self.model(s_in, s_out, q_in, shapes)
                    loss_out = compute_loss(
                        out.energy, out.z_pred, out.logits, q_out, self.cfg
                    )
                    total_loss += loss_out.total.item()
                    
        return total_loss / len(self.val_dl)

    def run(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            t1 = time.time()
            
            if epoch % self.cfg.val_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {t1-t0:.1f}s")
            else:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Time: {t1-t0:.1f}s")
