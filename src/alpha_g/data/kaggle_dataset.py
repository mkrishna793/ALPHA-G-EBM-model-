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
            return samples

        json_files = list(self.data_dir.rglob("*.json"))
        print(f"Found {len(json_files)} JSON files in {self.data_dir}")

        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                
                def extract_tasks(node, tasks_list):
                    if isinstance(node, dict):
                        if 'train' in node and 'test' in node:
                            # Official ARC format handling (if present)
                            support = []
                            for p in node['train']:
                                if 'input' in p and ('output' in p or 'target' in p):
                                    support.append((
                                        torch.tensor(p['input'], dtype=torch.long),
                                        torch.tensor(p.get('output', p.get('target')), dtype=torch.long)
                                    ))
                            query = []
                            for p in node['test']:
                                if 'input' in p and ('output' in p or 'target' in p):
                                    query.append((
                                        torch.tensor(p['input'], dtype=torch.long),
                                        torch.tensor(p.get('output', p.get('target')), dtype=torch.long)
                                    ))
                            if support and query:
                                tasks_list.append({'support': support, 'query': query})
                        else:
                            for k, v in node.items():
                                extract_tasks(v, tasks_list)
                    elif isinstance(node, list):
                        # Kaggle format: A list of dicts, where each dict has input/output.
                        # This entire list represents a single Task/Rule.
                        is_kaggle_task = all(isinstance(item, dict) and 'input' in item and ('output' in item or 'target' in item) for item in node)
                        if is_kaggle_task and len(node) >= 2:
                            pairs = []
                            for p in node:
                                pairs.append((
                                    torch.tensor(p['input'], dtype=torch.long),
                                    torch.tensor(p.get('output', p.get('target')), dtype=torch.long)
                                ))
                            # We store all pairs as 'support', and in __getitem__ we will dynamically split them
                            # into support and query.
                            tasks_list.append({'support': pairs, 'query': pairs}) # __getitem__ will pick from these
                        else:
                            for item in node:
                                extract_tasks(item, tasks_list)

                extract_tasks(task, samples)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(samples)} valid tasks.")
        return samples

    def __len__(self):
        # Reduced multiplier so epochs complete much faster while still ensuring good variety.
        return max(1, len(self.samples)) * 50

    def __getitem__(self, idx):
        if not self.samples:
            return {
                's_in': torch.zeros((2, 3, 3), dtype=torch.long),
                's_out': torch.zeros((2, 3, 3), dtype=torch.long),
                'q_in': torch.zeros((3, 3), dtype=torch.long),
                'q_out': torch.zeros((3, 3), dtype=torch.long),
                'shape': (3, 3)
            }
            
        # Modulo to safely wrap around the artificially expanded length
        actual_idx = idx % len(self.samples)
        task = self.samples[actual_idx]
        all_pairs = task['support']
        
        import random
        if len(all_pairs) >= 3:
            chosen = random.sample(all_pairs, 3)
            s_pairs = chosen[:2]
            q_pair = chosen[2]
        elif len(all_pairs) == 2:
            s_pairs = [all_pairs[0], all_pairs[0]]
            q_pair = all_pairs[1]
        else:
            s_pairs = [all_pairs[0], all_pairs[0]]
            q_pair = all_pairs[0]
            
        def clamp(t): return torch.clamp(t, min=0, max=99)
        
        s_in = [clamp(p[0]) for p in s_pairs]
        s_out = [clamp(p[1]) for p in s_pairs]
        q_in = clamp(q_pair[0])
        q_out = clamp(q_pair[1])
        
        # Check size limits
        for t in s_in + s_out + [q_in, q_out]:
            h, w = t.shape
            if h > 32 or w > 32 or h == 0 or w == 0:
                return self.__getitem__((idx + 1) % len(self.samples))
                
        # Find max H, W across all grids in this task
        max_h = max([t.shape[0] for t in s_in + s_out + [q_in, q_out]])
        max_w = max([t.shape[1] for t in s_in + s_out + [q_in, q_out]])
        
        def pad(t):
            h, w = t.shape
            return torch.nn.functional.pad(t, (0, max_w - w, 0, max_h - h), value=0)
            
        return {
            's_in': torch.stack([pad(t) for t in s_in]),
            's_out': torch.stack([pad(t) for t in s_out]),
            'q_in': pad(q_in),
            'q_out': pad(q_out),
            'shape': (max_h, max_w)
        }


def collate_fn(batch):
    """Pads grids to max size in batch."""
    shapes = [item['shape'] for item in batch]
    max_H = max(s[0] for s in shapes)
    max_W = max(s[1] for s in shapes)

    s_in_b, s_out_b, q_in_b, q_out_b = [], [], [], []

    for item in batch:
        pad_H = max_H - item['shape'][0]
        pad_W = max_W - item['shape'][1]
        
        def pad(t): return torch.nn.functional.pad(t, (0, pad_W, 0, pad_H), value=0)
        
        s_in_b.append(pad(item['s_in']))
        s_out_b.append(pad(item['s_out']))
        q_in_b.append(pad(item['q_in']))
        q_out_b.append(pad(item['q_out']))

    return {
        's_in': torch.stack(s_in_b),
        's_out': torch.stack(s_out_b),
        'q_in': torch.stack(q_in_b),
        'q_out': torch.stack(q_out_b),
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
