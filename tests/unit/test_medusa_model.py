"""Unit tests for MedusaModel with a small synthetic CausalLM."""

import os
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from specdecode.models.medusa_model import MedusaModel


@pytest.fixture
def tiny_causal_lm():
    # Use small Qwen2 configuration for fast CPU testing
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model


def test_medusa_model_initialization_and_forward(tiny_causal_lm):
    medusa_model = MedusaModel(tiny_causal_lm, num_heads=3, num_layers=1, freeze_base=True)

    # Verify base parameters are frozen
    for p in medusa_model.base_model.parameters():
        assert not p.requires_grad

    # Verify medusa heads are trainable
    for p in medusa_model.medusa_heads.parameters():
        assert p.requires_grad

    # Forward pass
    input_ids = torch.randint(0, 128, (2, 8))
    base_logits, medusa_logits, past_kv, hidden_states = medusa_model(input_ids)

    assert base_logits.shape == (2, 8, 128)
    assert len(medusa_logits) == 3
    for k in range(3):
        assert medusa_logits[k].shape == (2, 8, 128)
    assert hidden_states.shape == (2, 8, 64)


def test_medusa_model_save_and_load(tiny_causal_lm, tmp_path):
    medusa_model = MedusaModel(tiny_causal_lm, num_heads=2, num_layers=1)

    # Change weights so it's not identical to initialization
    with torch.no_grad():
        for p in medusa_model.medusa_heads.parameters():
            p.fill_(0.42)

    save_dir = str(tmp_path / "medusa_checkpoint")
    medusa_model.save_medusa_heads(save_dir)

    assert os.path.exists(os.path.join(save_dir, "medusa_heads.safetensors"))
    assert os.path.exists(os.path.join(save_dir, "medusa_config.json"))

    # Create new model and load
    new_model = MedusaModel(tiny_causal_lm, num_heads=2, num_layers=1)
    new_model.load_medusa_heads(os.path.join(save_dir, "medusa_heads.safetensors"))

    for p in new_model.medusa_heads.parameters():
        assert torch.allclose(p, torch.tensor(0.42))
