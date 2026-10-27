"""PyTorch Dataset and DataLoader collation for Medusa trajectory training."""

import glob
import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class TrajectoryDataset(Dataset):
    """Loads trajectory chunks containing precomputed hidden states and labels."""

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        pattern = os.path.join(data_dir, f"{split}_chunk_*.pt")
        self.chunk_files = sorted(glob.glob(pattern))
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found matching {pattern}")

        # Load samples from all chunks into memory (128 GB system RAM allows fast caching)
        self.samples = []
        for cf in self.chunk_files:
            chunk = torch.load(cf, map_location="cpu")
            self.samples.extend(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_trajectories(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences into a uniform batch."""
    batch_size = len(batch)
    max_len = max(sample["seq_len"] for sample in batch)
    hidden_size = batch[0]["hidden_states"].shape[-1]

    padded_hidden = torch.zeros(batch_size, max_len, hidden_size, dtype=batch[0]["hidden_states"].dtype)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        slen = sample["seq_len"]
        padded_hidden[i, :slen, :] = sample["hidden_states"]
        padded_labels[i, :slen] = sample["labels"]
        attention_mask[i, :slen] = True

    return {
        "hidden_states": padded_hidden,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }
