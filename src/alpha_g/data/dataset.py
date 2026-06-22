"""Dummy JSON Dataset for rapid prototyping and testing."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

class DummyARCData(Dataset):
    """
    Generates dummy grid pairs on the fly for testing the architecture.
    In reality, you'd load from Kaggle's JSON format.
    """
    def __init__(self, size: int = 1000, max_vocab: int = 10, max_grid: int = 10):
        self.size = size
        self.max_vocab = max_vocab
        self.max_grid = max_grid

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random H, W between 3 and max_grid
        H = torch.randint(3, self.max_grid + 1, (1,)).item()
        W = torch.randint(3, self.max_grid + 1, (1,)).item()

        # Input grid
        in_grid = torch.randint(0, self.max_vocab, (H, W))

        # Output grid (some random transformation: e.g. flip and recolor)
        out_grid = torch.flip(in_grid, [0, 1])
        out_grid = (out_grid + 1) % self.max_vocab

        return {
            'input': in_grid,
            'target': out_grid,
            'shape': (H, W),
        }

def collate_fn(batch):
    """Pads grids to max size in batch."""
    shapes = [item['shape'] for item in batch]
    max_H = max(s[0] for s in shapes)
    max_W = max(s[1] for s in shapes)

    in_grids = []
    target_grids = []

    for item in batch:
        pad_H = max_H - item['shape'][0]
        pad_W = max_W - item['shape'][1]
        
        ig = torch.nn.functional.pad(item['input'], (0, pad_W, 0, pad_H), value=0)
        tg = torch.nn.functional.pad(item['target'], (0, pad_W, 0, pad_H), value=0)
        
        in_grids.append(ig)
        target_grids.append(tg)

    return {
        'input': torch.stack(in_grids),
        'target': torch.stack(target_grids),
        'shapes': shapes, # Using original shapes for serializer if needed
        'batch_shape': (max_H, max_W)
    }

def get_dataloaders(batch_size: int, num_workers: int = 2):
    """Returns train and val dataloaders."""
    train_ds = DummyARCData(size=10000)
    val_ds = DummyARCData(size=500)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl
