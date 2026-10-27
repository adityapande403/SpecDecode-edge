"""Dataset manifest, checksum calculation, and integrity verification."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class TrajectoryChunkMeta:
    chunk_file: str
    num_samples: int
    num_tokens: int
    file_size_bytes: int
    sha256_checksum: str


@dataclass
class DatasetManifest:
    dataset_name: str
    target_model_id: str
    feature_layer: int
    hidden_size: int
    vocab_size: int
    num_train_samples: int
    num_val_samples: int
    total_samples: int
    total_tokens: int
    chunks: List[TrajectoryChunkMeta]
    created_at: str

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "DatasetManifest":
        with open(filepath, "r") as f:
            data = json.load(f)
        chunks = [TrajectoryChunkMeta(**c) for c in data.pop("chunks")]
        return cls(chunks=chunks, **data)


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()
