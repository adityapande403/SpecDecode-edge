"""Unit tests for MedusaLoss."""

import pytest
import torch
from specdecode.losses.medusa_loss import MedusaLoss


def test_medusa_loss_computation():
    batch_size = 2
    seq_len = 10
    vocab_size = 50
    num_heads = 3

    loss_fn = MedusaLoss(num_heads=num_heads, decay_factor=0.8, ignore_index=-100)

    # Synthetic logits and labels
    logits = [torch.randn(batch_size, seq_len, vocab_size, requires_grad=True) for _ in range(num_heads)]
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    total_loss, metrics = loss_fn(logits, labels)

    assert total_loss.item() > 0.0
    assert not torch.isnan(total_loss)
    assert "loss_head_0" in metrics
    assert "loss_head_1" in metrics
    assert "loss_head_2" in metrics
    assert "acc_head_0" in metrics

    # Test backward pass
    total_loss.backward()
    for k in range(num_heads):
        assert logits[k].grad is not None
        assert not torch.isnan(logits[k].grad).any()


def test_medusa_loss_masking():
    batch_size = 1
    seq_len = 8
    vocab_size = 20
    num_heads = 2

    loss_fn = MedusaLoss(num_heads=num_heads, decay_factor=0.8, ignore_index=-100)

    logits = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(num_heads)]
    # All masked labels
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    total_loss, metrics = loss_fn(logits, labels)
    assert metrics["acc_head_0"] == 0.0
