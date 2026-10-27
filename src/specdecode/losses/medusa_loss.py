"""Medusa Multi-Head Cross-Entropy Loss implementation."""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class MedusaLoss(nn.Module):
    """Computes masked multi-head cross-entropy loss for Medusa speculative heads.

    Each head k (0-indexed) predicts tokens at offset (k + 2) from the input position,
    which corresponds to (k + 1) tokens ahead of the standard next-token prediction.
    """

    def __init__(
        self,
        num_heads: int = 3,
        decay_factor: float = 0.8,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.decay_factor = decay_factor
        self.ignore_index = ignore_index
        # Lambda weights: [1.0, decay, decay^2, ...]
        self.weights = [decay_factor**i for i in range(num_heads)]
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(
        self,
        medusa_logits: List[torch.Tensor],
        target_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the weighted multi-head cross entropy loss.

        Args:
            medusa_logits: List of K tensors, each of shape (batch_size, seq_len, vocab_size)
            target_labels: Tensor of shape (batch_size, seq_len) with -100 for ignored tokens

        Returns:
            total_loss: Scalar torch.Tensor
            metrics: Dict with per-head losses and accuracy metrics
        """
        batch_size, seq_len = target_labels.shape
        total_loss = torch.tensor(0.0, device=target_labels.device, dtype=torch.float32)
        metrics = {}

        for k, logits in enumerate(medusa_logits):
            shift = k + 2  # Offset from current token
            if seq_len <= shift:
                # Sequence too short for this lookahead
                continue

            # Slice logits and labels
            # logits at position t predict token at position t + shift
            logits_slice = logits[:, :-shift, :].contiguous()
            labels_slice = target_labels[:, shift:].contiguous()

            # Flatten for cross entropy
            vocab_size = logits_slice.shape[-1]
            flat_logits = logits_slice.view(-1, vocab_size)
            flat_labels = labels_slice.view(-1)

            head_loss = self.loss_fct(flat_logits, flat_labels)

            # Accuracy on non-masked tokens
            with torch.no_grad():
                valid_mask = flat_labels != self.ignore_index
                if valid_mask.sum() > 0:
                    preds = flat_logits.argmax(dim=-1)
                    correct = (preds[valid_mask] == flat_labels[valid_mask]).float().sum()
                    acc = (correct / valid_mask.sum()).item()
                else:
                    acc = 0.0

            weight = self.weights[k]
            total_loss = total_loss + weight * head_loss

            metrics[f"loss_head_{k}"] = head_loss.item()
            metrics[f"acc_head_{k}"] = acc
            metrics[f"weight_head_{k}"] = weight

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
