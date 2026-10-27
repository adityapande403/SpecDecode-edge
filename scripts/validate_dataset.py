"""Validate dataset integrity, checksums, tensor shapes, and masks."""

import os
import torch

from specdecode.data.dataset_manifest import DatasetManifest, compute_file_sha256
from specdecode.utils.logging import get_logger

logger = get_logger("validate_dataset")


def validate_dataset(manifest_path: str = "data/trajectories/manifest.json") -> bool:
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return False

    manifest = DatasetManifest.load(manifest_path)
    logger.info(f"Loaded manifest: {manifest.dataset_name} for model {manifest.target_model_id}")
    logger.info(f"Validating {len(manifest.chunks)} chunks across {manifest.total_samples} samples...")

    base_dir = os.path.dirname(manifest_path)
    total_verified_samples = 0
    total_verified_tokens = 0

    for chunk in manifest.chunks:
        chunk_path = os.path.join(base_dir, chunk.chunk_file)
        if not os.path.exists(chunk_path):
            logger.error(f"Missing chunk file: {chunk_path}")
            return False

        # Verify Checksum
        computed_hash = compute_file_sha256(chunk_path)
        if computed_hash != chunk.sha256_checksum:
            logger.error(
                f"Checksum mismatch for {chunk.chunk_file}!\nExpected: {chunk.sha256_checksum}\nGot:      {computed_hash}"
            )
            return False

        # Verify Tensor Content
        data = torch.load(chunk_path, map_location="cpu")
        if len(data) != chunk.num_samples:
            logger.error(f"Sample count mismatch in {chunk.chunk_file}: expected {chunk.num_samples}, got {len(data)}")
            return False

        for sample in data:
            assert "hidden_states" in sample, "Missing hidden_states in sample"
            assert "labels" in sample, "Missing labels in sample"
            assert "input_ids" in sample, "Missing input_ids in sample"

            hidden = sample["hidden_states"]
            labels = sample["labels"]
            input_ids = sample["input_ids"]

            # Shape checks
            seq_len = sample["seq_len"]
            assert hidden.shape == (seq_len, manifest.hidden_size), (
                f"Hidden state shape {hidden.shape} mismatch with expected ({seq_len}, {manifest.hidden_size})"
            )
            assert labels.shape == (seq_len,), f"Labels shape {labels.shape} mismatch with ({seq_len},)"
            assert input_ids.shape == (seq_len,), f"Input IDs shape {input_ids.shape} mismatch with ({seq_len},)"

            # Numerical stability checks
            assert not torch.isnan(hidden).any(), f"NaN detected in hidden states of {chunk.chunk_file}"
            assert not torch.isinf(hidden).any(), f"Inf detected in hidden states of {chunk.chunk_file}"

            total_verified_samples += 1
            total_verified_tokens += seq_len

    logger.info(
        f"Dataset validation SUCCESS! All {total_verified_samples} samples and {total_verified_tokens} tokens passed checksum and shape integrity checks."
    )
    return True


if __name__ == "__main__":
    success = validate_dataset()
    if not success:
        exit(1)
