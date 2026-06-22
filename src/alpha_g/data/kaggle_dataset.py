"""Kaggle ARC Dataset Loader."""

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class KaggleARCDataset(Dataset):
    """
    Loads ARC JSON files (like the 100k synthetic dataset).
    Expects a directory containing .json files.
    Each JSON should have 'train' and/or 'test' keys with 'input' and 'output' grids.
    """
    def __init__(self, data_dir: str | Path, max_grid: int = 32):
        self.data_dir = Path(data_dir)
        self.max_grid = max_grid
        self.samples = self._load_all_samples()

    def _load_all_samples(self) -> list[dict[str, Any]]:
        samples = []
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist. (Ignore if running inside Modal for the first time)")
            return samples

        # Find all JSON files recursively
        json_files = list(self.data_dir.rglob("*.json"))
        print(f"Found {len(json_files)} JSON files in {self.data_dir}")

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                
                # Recursively extract any dictionary with 'input' and 'output'
                def extract_pairs(node, samples_list):
                    if isinstance(node, dict):
                        # Check if this node is an ARC pair
                        if 'input' in node and ('output' in node or 'target' in node):
                            inp = node['input']
                            tgt = node.get('output', node.get('target'))
                            samples_list.append({
                                'input': torch.tensor(inp, dtype=torch.long),
                                'target': torch.tensor(tgt, dtype=torch.long)
                            })
                        else:
                            for k, v in node.items():
                                extract_pairs(v, samples_list)
                    elif isinstance(node, list):
                        for item in node:
                            extract_pairs(item, samples_list)

                extract_pairs(task, samples)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(samples)} valid input-output pairs.")
        return samples

    def __len__(self):
        return max(1, len(self.samples)) # Prevent DataLoader crash if empty

    def __getitem__(self, idx):
        if not self.samples:
            # Return dummy if no data (e.g. before data is downloaded)
            return {
                'input': torch.zeros((3, 3), dtype=torch.long),
                'target': torch.zeros((3, 3), dtype=torch.long),
                'shape': (3, 3)
            }
            
        sample = self.samples[idx]
        # Clamp values to safe vocabulary range (0-99) to prevent ANY out-of-bounds CUDA asserts
        inp = torch.clamp(sample['input'], min=0, max=99)
        tgt = torch.clamp(sample['target'], min=0, max=99)
        
        # Dimensions
        H_in, W_in = inp.shape
        H_out, W_out = tgt.shape
        
        # Skip garbage synthetic puzzles that exceed official ARC dimensions (32x32) or are empty
        # This absolutely prevents O(N^2) memory blowouts (OOMs) in the Attention mechanism
        if H_in > 32 or W_in > 32 or H_out > 32 or W_out > 32 or H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
            return self.__getitem__((idx + 1) % len(self.samples))
        
        # For simplicity in this version, we pad to the max of (input, target)
        H = max(H_in, H_out)
        W = max(W_in, W_out)
        
        # Pad grids to match shapes if they differ (ARC tasks can change shape)
        inp_padded = torch.nn.functional.pad(inp, (0, W - W_in, 0, H - H_in), value=0)
        tgt_padded = torch.nn.functional.pad(tgt, (0, W - W_out, 0, H - H_out), value=0)

        return {
            'input': inp_padded,
            'target': tgt_padded,
            'shape': (H, W)
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
        'shapes': shapes,
        'batch_shape': (max_H, max_W)
    }


def get_kaggle_dataloaders(data_dir: str | Path, batch_size: int, num_workers: int = 4):
    """Returns train and val dataloaders for Kaggle data."""
    dataset = KaggleARCDataset(data_dir)
    
    if len(dataset.samples) == 0:
        print("WARNING: Dataset is empty. Returning empty dataloaders.")
        return DataLoader(dataset, batch_size=batch_size), DataLoader(dataset, batch_size=batch_size)

    # 95% Train, 5% Val split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl
