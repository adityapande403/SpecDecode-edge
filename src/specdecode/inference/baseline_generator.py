"""Autoregressive baseline generation engine with detailed performance profiling."""

import time
from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class BaselineGenerationOutput:
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
    inter_token_latencies_ms: List[float]


class BaselineGenerator:
    """Standard autoregressive generator for establishing the authoritative baseline."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> BaselineGenerationOutput:
        """Execute baseline autoregressive generation with step-by-step profiling."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        raw_inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        eos_token_id = self.tokenizer.eos_token_id

        generated_ids: List[int] = []
        inter_latencies: List[float] = []

        start_time = time.perf_counter()

        # Step 0: Prefill
        prefill_start = time.perf_counter()
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_time = (time.perf_counter() - prefill_start) * 1000.0

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        if do_sample and temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        generated_ids.append(token_id)
        current_input = next_token

        # Decode loop
        decode_start = time.perf_counter()
        step_start = decode_start

        while len(generated_ids) < max_new_tokens:
            if eos_token_id is not None and token_id == eos_token_id:
                break

            outputs = self.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            step_end = time.perf_counter()
            inter_latencies.append((step_end - step_start) * 1000.0)
            step_start = step_end

            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated_ids.append(token_id)
            current_input = next_token

        total_time = (time.perf_counter() - start_time) * 1000.0
        decode_time = sum(inter_latencies) if inter_latencies else 0.0

        num_tokens = len(generated_ids)
        tps = (num_tokens / (total_time / 1000.0)) if total_time > 0 else 0.0
        decode_tps = (
            ((num_tokens - 1) / (decode_time / 1000.0)) if (num_tokens > 1 and decode_time > 0) else 0.0
        )

        peak_vram = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0.0
        )

        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return BaselineGenerationOutput(
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
            inter_token_latencies_ms=inter_latencies,
        )
