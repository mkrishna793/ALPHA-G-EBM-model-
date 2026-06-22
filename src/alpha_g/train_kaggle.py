"""Entry point for training Alpha-G on Kaggle ARC Data."""

import os
from pathlib import Path

from alpha_g.config import ArchConfig, TrainConfig
from alpha_g.data.kaggle_dataset import get_kaggle_dataloaders
from alpha_g.model.alpha_g import AlphaG
from alpha_g.training.trainer import Trainer

def main():
    # Expect data in ./kaggle_data unless overridden by env
    data_dir = os.environ.get('KAGGLE_DATA_DIR', './kaggle_data')
    
    arch_cfg = ArchConfig(
        d_model=256,
        d_latent=128,
        # ARC uses colors 0-9
        max_vocab=10, 
        max_grid=32,
    )
    
    train_cfg = TrainConfig(
        epochs=50,             # 50 Epochs for 100k
        batch_size=64,         # Reduced from 256 to fit compilation graph in 40GB
        use_amp=True,          # BF16 Mixed Precision
        use_compile=False,     # Disabled so training starts INSTANTLY
        lr=3e-4,
        hmc_steps=10           # Reduced from 20 to save memory footprint
    )

    print("Initializing Alpha-G for Kaggle Data...")
    model = AlphaG(arch_cfg, train_cfg)
    
    print(f"Loading data from {data_dir}...")
    train_dl, val_dl = get_kaggle_dataloaders(
        data_dir=data_dir, 
        batch_size=train_cfg.batch_size, 
        num_workers=8  # High workers for cloud GPU
    )
    
    # If no data found, exit cleanly
    if len(train_dl.dataset) == 0:
        print(f"ERROR: No valid data found in {data_dir}. Please download the Kaggle dataset first.")
        return

    print("Starting Cloud GPU Training...")
    trainer = Trainer(model, train_dl, val_dl, train_cfg)
    trainer.run()
    
    print("Training complete! Saving weights...")
    os.makedirs('weights', exist_ok=True)
    import torch
    torch.save(model.state_dict(), 'weights/alpha_g_kaggle.pth')
    print("Weights saved to weights/alpha_g_kaggle.pth")

if __name__ == "__main__":
    main()
