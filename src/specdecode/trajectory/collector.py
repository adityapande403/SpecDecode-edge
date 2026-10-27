"""Trajectory and hidden-state extraction pipeline for training speculative draft components."""

import os
import time
from typing import Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from specdecode.data.dataset_manifest import DatasetManifest, TrajectoryChunkMeta, compute_file_sha256
from specdecode.utils.logging import get_logger

logger = get_logger("trajectory_collector")


class TrajectoryCollector:
    """Extracts target model hidden states and aligned token labels for Medusa training."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        chunk_size: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk_size = chunk_size

    @torch.no_grad()
    def extract_from_texts(
        self,
        text_samples: List[str],
        output_dir: str,
        split: str = "train",
        max_seq_len: int = 256,
        mask_prompt: bool = True,
    ) -> List[TrajectoryChunkMeta]:
        """Extract hidden states and token labels from text sequences in chunks."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        chunk_metas: List[TrajectoryChunkMeta] = []
        current_chunk_data: List[Dict[str, torch.Tensor]] = []
        chunk_idx = 0
        total_tokens_in_chunk = 0

        logger.info(f"Extracting {len(text_samples)} samples for split '{split}'...")

        for i, text in enumerate(tqdm(text_samples, desc=f"Trajectory extraction ({split})")):
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(self.device)

            input_ids = tokens["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 4:
                continue

            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            # Extract final layer hidden states (1, seq_len, hidden_size)
            hidden_states = outputs.hidden_states[-1].squeeze(0).to(torch.bfloat16).cpu()
            labels = input_ids.squeeze(0).cpu().clone()

            # If masking prompt: for demonstration, let first 1/3 be prompt and remaining 2/3 be target
            if mask_prompt and seq_len > 10:
                prompt_cutoff = max(1, seq_len // 3)
                labels[:prompt_cutoff] = -100

            sample = {
                "input_ids": input_ids.squeeze(0).cpu(),
                "labels": labels,
                "hidden_states": hidden_states,
                "seq_len": seq_len,
            }

            current_chunk_data.append(sample)
            total_tokens_in_chunk += seq_len

            if len(current_chunk_data) >= self.chunk_size or i == len(text_samples) - 1:
                chunk_file = f"{split}_chunk_{chunk_idx:04d}.pt"
                chunk_path = os.path.join(output_dir, chunk_file)
                torch.save(current_chunk_data, chunk_path)

                checksum = compute_file_sha256(chunk_path)
                file_size = os.path.getsize(chunk_path)

                chunk_meta = TrajectoryChunkMeta(
                    chunk_file=chunk_file,
                    num_samples=len(current_chunk_data),
                    num_tokens=total_tokens_in_chunk,
                    file_size_bytes=file_size,
                    sha256_checksum=checksum,
                )
                chunk_metas.append(chunk_meta)

                logger.info(
                    f"Saved {chunk_file}: {len(current_chunk_data)} samples, {total_tokens_in_chunk} tokens ({file_size / (1024*1024):.2f} MB)"
                )

                chunk_idx += 1
                current_chunk_data = []
                total_tokens_in_chunk = 0

        return chunk_metas
