"""End-to-end token-by-token greedy equivalence tests."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from specdecode.models.medusa_model import MedusaModel
from specdecode.inference.baseline_generator import BaselineGenerator
from specdecode.inference.speculative_generator import SpeculativeGenerator


class DummyTokenizer:
    """Minimal mock tokenizer for synthetic tests."""
    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None):
        # Convert simple characters to token IDs
        tokens = [ord(c) % self.vocab_size for c in text]
        if not tokens:
            tokens = [1]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens], dtype=torch.long)}
        return {"input_ids": tokens}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(t) if 32 <= t < 127 else "?" for t in token_ids)


@pytest.fixture
def synthetic_pipeline():
    torch.manual_seed(42)
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    base_model = AutoModelForCausalLM.from_config(config)
    base_model.eval()

    medusa_model = MedusaModel(base_model, num_heads=3, num_layers=1)
    medusa_model.eval()

    tokenizer = DummyTokenizer(vocab_size=128)
    return base_model, medusa_model, tokenizer


def test_token_by_token_greedy_equivalence(synthetic_pipeline):
    base_model, medusa_model, tokenizer = synthetic_pipeline

    baseline_gen = BaselineGenerator(base_model, tokenizer)
    speculative_gen = SpeculativeGenerator(medusa_model, tokenizer)

    test_prompts = [
        "Hello speculative decoding",
        "Short",
        "A slightly longer prompt with complex structure 1234567890",
    ]

    for prompt in test_prompts:
        base_out = baseline_gen.generate(prompt, max_new_tokens=32, do_sample=False)
        spec_out = speculative_gen.generate(prompt, max_new_tokens=32)

        # EXACT token-by-token comparison
        assert base_out.token_ids == spec_out.token_ids, (
            f"Equivalence failed for prompt: '{prompt}'\n"
            f"Baseline tokens:    {base_out.token_ids}\n"
            f"Speculative tokens: {spec_out.token_ids}"
        )
        assert base_out.num_generated_tokens == spec_out.num_generated_tokens
