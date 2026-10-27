"""Speculative decoding generation engine with Medusa multi-head proposals and KV cache rollback."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from specdecode.models.medusa_model import MedusaModel
from specdecode.verification.verifier import SpeculativeVerifier, VerificationResult


@dataclass
class SpeculativeGenerationOutput:
    token_ids: List[int]
    generated_text: str
    num_generated_tokens: int
    prompt_tokens: int
    ttft_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float
    decode_tokens_per_second: float
    peak_vram_gb: float
    speculative_steps: int
    accepted_tokens_per_step: List[int]
    mean_accepted_per_step: float
    head_acceptance_rates: Dict[str, float]


def crop_cache(past_key_values, target_length: int):
    """Crop past_key_values back to target_length after candidate rejection."""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, DynamicCache):
        cur_len = past_key_values.get_seq_length()
        if cur_len > target_length:
            delta = target_length - cur_len  # Negative value
            past_key_values.crop(delta)
        return past_key_values
    elif isinstance(past_key_values, tuple):
        new_cache = []
        for layer_k, layer_v in past_key_values:
            new_k = layer_k[:, :, :target_length, :]
            new_v = layer_v[:, :, :target_length, :]
            new_cache.append((new_k, new_v))
        return tuple(new_cache)
    return past_key_values


class SpeculativeGenerator:
    """Speculative decoding generation engine using Medusa multi-head verification."""

    def __init__(self, medusa_model: MedusaModel, tokenizer: PreTrainedTokenizerBase):
        self.medusa_model = medusa_model
        self.tokenizer = tokenizer
        self.device = next(medusa_model.parameters()).device
        self.num_heads = medusa_model.num_heads

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
    ) -> SpeculativeGenerationOutput:
        """Execute speculative generation with Medusa multi-head drafting and authoritative verification."""
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        raw_inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        generated_ids: List[int] = []
        accepted_per_step: List[int] = []
        head_accept_counts = [0] * self.num_heads
        head_eval_counts = [0] * self.num_heads

        start_time = time.perf_counter()

        # Step 0: Prefill
        prefill_start = time.perf_counter()
        base_logits, medusa_logits, past_key_values, hidden_states = self.medusa_model(
            input_ids=input_ids,
            use_cache=True,
            return_medusa_logits=True,
        )
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_time = (time.perf_counter() - prefill_start) * 1000.0

        # Emit first token from base model greedy prediction
        first_token = torch.argmax(base_logits[:, -1, :], dim=-1).item()
        generated_ids.append(first_token)

        # Propose initial candidates from Medusa heads at the prefill boundary
        last_hidden = hidden_states[:, -1:, :]
        curr_candidates = [
            torch.argmax(head(last_hidden)[:, -1, :], dim=-1).item()
            for head in self.medusa_model.medusa_heads.heads
        ]

        current_token = first_token
        total_cache_len = prompt_len

        decode_start = time.perf_counter()

        # Speculative decoding loop
        while len(generated_ids) < max_new_tokens:
            if eos_token_id is not None and current_token == eos_token_id:
                break

            active_k = min(len(curr_candidates), max_new_tokens - len(generated_ids))
            if active_k == 0:
                break

            eval_candidates = curr_candidates[:active_k]
            verif_tokens = [current_token] + eval_candidates
            verif_input = torch.tensor([verif_tokens], device=self.device, dtype=torch.long)

            verif_out = self.medusa_model.base_model(
                input_ids=verif_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            target_logits = verif_out.logits[0]  # shape: (1 + active_k, vocab_size)
            verif_res = SpeculativeVerifier.verify_greedy(
                candidate_tokens=eval_candidates,
                target_logits=target_logits,
                eos_token_id=eos_token_id,
            )

            # Track acceptance statistics per head
            for i in range(len(eval_candidates)):
                if i < len(verif_res.acceptance_mask):
                    head_eval_counts[i] += 1
                    if verif_res.acceptance_mask[i]:
                        head_accept_counts[i] += 1

            new_tokens = verif_res.accepted_tokens
            available_slots = max_new_tokens - len(generated_ids)
            if len(new_tokens) > available_slots:
                new_tokens = new_tokens[:available_slots]

            generated_ids.extend(new_tokens)
            accepted_per_step.append(len(new_tokens))

            # Number of candidate tokens accepted (excluding the bonus/authoritative token)
            num_spec_accepted = verif_res.num_speculative_accepted
            valid_in_cache = 1 + num_spec_accepted

            target_cache_len = total_cache_len + valid_in_cache
            past_key_values = crop_cache(verif_out.past_key_values, target_cache_len)
            total_cache_len = target_cache_len

            current_token = new_tokens[-1]
            if eos_token_id is not None and current_token == eos_token_id:
                break

            # Predict next candidates from the last valid token position in the verification step
            last_valid_idx = valid_in_cache - 1
            committed_hidden = verif_out.hidden_states[-1][:, last_valid_idx : last_valid_idx + 1, :]
            curr_candidates = [
                torch.argmax(head(committed_hidden)[:, -1, :], dim=-1).item()
                for head in self.medusa_model.medusa_heads.heads
            ]

        total_time = (time.perf_counter() - start_time) * 1000.0
        decode_time = (time.perf_counter() - decode_start) * 1000.0
        num_tokens = len(generated_ids)

        tps = (num_tokens / (total_time / 1000.0)) if total_time > 0 else 0.0
        decode_tps = (
            ((num_tokens - 1) / (decode_time / 1000.0)) if (num_tokens > 1 and decode_time > 0) else 0.0
        )

        mean_accepted = sum(accepted_per_step) / len(accepted_per_step) if accepted_per_step else 1.0

        rates = {}
        for i in range(self.num_heads):
            denom = head_eval_counts[i]
            rates[f"head_{i}"] = (head_accept_counts[i] / denom) if denom > 0 else 0.0

        peak_vram = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0.0
        )

        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return SpeculativeGenerationOutput(
            token_ids=generated_ids,
            generated_text=decoded_text,
            num_generated_tokens=num_tokens,
            prompt_tokens=prompt_len,
            ttft_ms=prefill_time,
            decode_latency_ms=decode_time,
            total_latency_ms=total_time,
            tokens_per_second=tps,
            decode_tokens_per_second=decode_tps,
            peak_vram_gb=peak_vram,
            speculative_steps=len(accepted_per_step),
            accepted_tokens_per_step=accepted_per_step,
            mean_accepted_per_step=mean_accepted,
            head_acceptance_rates=rates,
        )
