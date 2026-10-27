"""Unified Medusa Model wrapping a frozen base LLM and trainable Medusa speculative heads."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from transformers import PreTrainedModel, AutoModelForCausalLM

from specdecode.models.medusa_head import MedusaHeads


class MedusaModel(nn.Module):
    """Integrates a frozen base CausalLM with trainable Medusa speculative heads."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_heads: int = 3,
        num_layers: int = 1,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        config = base_model.config
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
        vocab_size = getattr(config, "vocab_size", None)
        if hidden_size is None or vocab_size is None:
            raise ValueError("Could not determine hidden_size or vocab_size from base model config.")

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # Medusa heads are initialized with the base model's device and dtype
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.medusa_heads = MedusaHeads(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None if "Any" in globals() else None,
        use_cache: Optional[bool] = None,
        return_medusa_logits: bool = True,
    ):
        """Forward pass through base model and speculative heads.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            position_ids: (batch_size, seq_len)
            past_key_values: KV cache
            use_cache: Whether to use and return KV cache
            return_medusa_logits: Whether to compute and return Medusa logits

        Returns:
            Tuple: (base_logits, medusa_logits, past_key_values, hidden_states)
        """
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        base_logits = base_outputs.logits
        hidden_states = base_outputs.hidden_states[-1]

        medusa_logits = None
        if return_medusa_logits:
            medusa_logits = self.medusa_heads(hidden_states)

        return base_logits, medusa_logits, base_outputs.past_key_values, hidden_states

    def save_medusa_heads(self, save_directory: str, filename: str = "medusa_heads.safetensors"):
        """Save only the Medusa head weights to safetensors."""
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, filename)
        state_dict = {f"medusa_heads.{k}": v.contiguous().cpu() for k, v in self.medusa_heads.state_dict().items()}
        save_file(state_dict, path)

        # Save config metadata
        meta = {
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "base_model_name_or_path": getattr(self.base_model.config, "_name_or_path", "unknown"),
        }
        import json
        with open(os.path.join(save_directory, "medusa_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load_medusa_heads(self, checkpoint_path: str):
        """Load Medusa head weights from safetensors or pt checkpoint."""
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Strip prefix if present
        clean_sd = {}
        for k, v in state_dict.items():
            if k.startswith("medusa_heads."):
                clean_sd[k[len("medusa_heads."):]] = v
            else:
                clean_sd[k] = v

        self.medusa_heads.load_state_dict(clean_sd)
