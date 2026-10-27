"""Medusa Multi-Head Architecture implementation."""

import torch
import torch.nn as nn
from typing import List, Optional


class ResBlock(nn.Module):
    """Residual MLP block used inside each Medusa prediction head."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    """A single Medusa speculative head predicting tokens at offset t + k + 1."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        blocks = [ResBlock(hidden_size) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Args:

        hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        features = self.blocks(hidden_states)
        return self.head(features)


class MedusaHeads(nn.Module):
    """Collection of K Medusa speculative heads."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 3,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.heads = nn.ModuleList(
            [MedusaHead(hidden_size, vocab_size, num_layers=num_layers) for _ in range(num_heads)]
        )

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Forward through all K heads.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            list of K logits tensors, each of shape (batch_size, seq_len, vocab_size)
        """
        return [head(hidden_states) for head in self.heads]

    def count_parameters(self) -> int:
        """Return total trainable parameters in heads."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
