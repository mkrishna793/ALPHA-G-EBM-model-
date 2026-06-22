"""Entry point for training Alpha-G."""

from alpha_g.config import ArchConfig, TrainConfig
from alpha_g.data.dataset import get_dataloaders
from alpha_g.model.alpha_g import AlphaG
from alpha_g.training.trainer import Trainer

def main():
    arch_cfg = ArchConfig()
    train_cfg = TrainConfig(
        epochs=5,
        batch_size=32,
        use_amp=True,
        # Set to True if you have PyTorch 2.0+ and a supported GPU
        use_compile=False 
    )

    print("Initializing Alpha-G...")
    model = AlphaG(arch_cfg, train_cfg)
    
    print("Loading data...")
    train_dl, val_dl = get_dataloaders(train_cfg.batch_size, train_cfg.num_workers)
    
    trainer = Trainer(model, train_dl, val_dl, train_cfg)
    trainer.run()

if __name__ == "__main__":
    main()
