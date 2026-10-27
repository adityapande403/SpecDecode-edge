"""Unit tests for MedusaHead and MedusaHeads."""

import pytest
import torch
from specdecode.models.medusa_head import MedusaHead, MedusaHeads, ResBlock


def test_resblock_forward():
    hidden_size = 64
    block = ResBlock(hidden_size)
    x = torch.randn(2, 8, hidden_size)
    out = block(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_medusa_head_forward():
    hidden_size = 64
    vocab_size = 256
    head = MedusaHead(hidden_size, vocab_size, num_layers=1)
    x = torch.randn(2, 10, hidden_size)
    logits = head(x)
    assert logits.shape == (2, 10, vocab_size)
    assert not torch.isnan(logits).any()


def test_medusa_heads_forward_and_params():
    hidden_size = 64
    vocab_size = 256
    num_heads = 3
    heads = MedusaHeads(hidden_size, vocab_size, num_heads=num_heads, num_layers=1)
    assert heads.count_parameters() > 0

    x = torch.randn(2, 12, hidden_size)
    out_logits = heads(x)
    assert len(out_logits) == num_heads
    for k in range(num_heads):
        assert out_logits[k].shape == (2, 12, vocab_size)
