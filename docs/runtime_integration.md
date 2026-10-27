# Runtime Integration Guide

## 1. Native PyTorch Integration

The native execution engine is in `src/specdecode/inference/speculative_generator.py`.
It provides:
- Single-pass candidate verification
- Dynamic KV cache management (`crop_cache`)
- Accurate TTFT and TPS profiling

```python
from specdecode.models.medusa_model import MedusaModel
from specdecode.inference.speculative_generator import SpeculativeGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")

medusa_model = MedusaModel(base_model, num_heads=3, num_layers=1)
medusa_model.load_medusa_heads("checkpoints/medusa/best_medusa_heads.pt")

generator = SpeculativeGenerator(medusa_model, tokenizer)
out = generator.generate("Explain how speculative decoding works:", max_new_tokens=64)
print(out.generated_text)
```

## 2. vLLM Speculative Decoding Integration

The trained weights in `artifacts/exported_models/medusa_qwen2.5_1.5b` are formatted for vLLM's Medusa worker:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-model artifacts/exported_models/medusa_qwen2.5_1.5b \
    --num-speculative-tokens 3
```
