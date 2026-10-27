"""Diagnostic script to inspect token mismatch between baseline and speculative generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from specdecode.data.prompt_dataset import BENCHMARK_PROMPTS
from specdecode.inference.baseline_generator import BaselineGenerator
from specdecode.inference.speculative_generator import SpeculativeGenerator
from specdecode.models.medusa_model import MedusaModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
)
medusa_model = MedusaModel(base_model, num_heads=3, num_layers=1, freeze_base=True)
ckpt = torch.load("checkpoints/medusa/best_medusa_heads.pt", map_location=device)
medusa_model.medusa_heads.load_state_dict(ckpt["state_dict"])
medusa_model.medusa_heads.to(device=device, dtype=torch.bfloat16)

baseline_gen = BaselineGenerator(base_model, tokenizer)
spec_gen = SpeculativeGenerator(medusa_model, tokenizer)

# Test instruction_01
prompt_item = [p for p in BENCHMARK_PROMPTS if p.id == "instruction_01"][0]
base_out = baseline_gen.generate(prompt_item.prompt, max_new_tokens=40, do_sample=False)
spec_out = spec_gen.generate(prompt_item.prompt, max_new_tokens=40)

print("Baseline tokens:", base_out.token_ids[:25])
print("Speculative tokens:", spec_out.token_ids[:25])

for i in range(min(len(base_out.token_ids), len(spec_out.token_ids))):
    b_tok = base_out.token_ids[i]
    s_tok = spec_out.token_ids[i]
    if b_tok != s_tok:
        print(f"Mismatch at index {i}:")
        print(f"  Baseline token:    {b_tok} ('{tokenizer.decode([b_tok])}')")
        print(f"  Speculative token: {s_tok} ('{tokenizer.decode([s_tok])}')")
        break
