"""Configuration for Alpha-G model, training, and data."""

from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application-level config from environment."""
    model_config = SettingsConfigDict(env_file='.env')
    wandb_api_key: str | None = None
    wandb_project: str | None = Field(default='alpha-g')
    wandb_entity: str | None = None
    data_dir: Path = Field(default=Path('data'))


class ArchConfig(BaseModel):
    """Architecture hyperparameters."""
    d_model: int = 256
    d_latent: int = 128
    n_heads: int = 8
    d_ffn: int = 1024
    dropout: float = 0.1
    max_vocab: int = 32         # color/symbol vocabulary
    max_seq_len: int = 1024     # max tokens
    max_grid: int = 32          # max H or W
    # AEG
    aeg_layers: int = 4
    metric_rank: int = 16       # low-rank metric approximation
    # BEP
    bep_layers: int = 4
    bep_equilibrium_iters: int = 1
    # Decoder
    dec_layers: int = 3


class TrainConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    epochs: int = 100
    grad_clip: float = 1.0
    # EMA
    ema_start: float = 0.996
    ema_end: float = 1.0
    # Loss weights
    decode_weight: float = 1.0
    consistency_weight: float = 0.5
    geometry_weight: float = 0.01
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 0.01
    # z noise
    z_noise: float = 0.1
    # HMC
    hmc_steps: int = 20
    hmc_leapfrog: int = 8
    hmc_chains: int = 4
    hmc_step_size: float = 0.02
    # Speed
    use_compile: bool = False    # torch.compile (set True if GPU supports)
    use_amp: bool = True         # mixed precision
    grad_accumulation: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    # Checkpointing
    checkpoint_dir: Path = Field(default=Path('checkpoints'))
    keep_top_k: int = 3
    val_every: int = 1           # validate every N epochs


config = Config()
