---
library_name: transformers
tags:
- speculative-decoding
- medusa
- qwen2.5
- edge-ai
- inference-acceleration
- pytorch
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: apache-2.0
pipeline_tag: text-generation
---

# SpecDecode Edge: Medusa-3Heads for Qwen2.5-1.5B-Instruct

This repository provides trained **Medusa multi-head speculative decoding** weights for accelerating **`Qwen/Qwen2.5-1.5B-Instruct`** on single GPU and edge devices.

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Frozen)
- **Speculative Method:** Medusa Multi-Head Speculative Decoding (Cai et al., 2024)
- **Heads Configuration:** 3 speculative lookahead heads (offsets $t+2, t+3, t+4$)
- **Architecture per Head:** 1 ResBlock (SiLU) + Linear projection ($1536 \rightarrow 151936$)
- **Precision:** `bfloat16`
- **Trained Parameters:** ~707M parameters
- **Hardware Verified:** Single NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM)
- **Source Code & GitHub Repository:** [https://github.com/adityapande403/SpecDecode-edge](https://github.com/adityapande403/SpecDecode-edge)

---

## Performance & Acceleration

Empirical benchmarks on an NVIDIA RTX 5060 Ti:

| Workload Category | Baseline Decode TPS | Speculative Decode TPS | Speedup Ratio | Mean Acc / Step |
| :--- | :--- | :--- | :--- | :--- |
| **Code Generation** | 39.99 tok/s | **43.17 tok/s** | **1.08x** | **1.14** |
| **Mathematical Reasoning** | 40.41 tok/s | **42.25 tok/s** | **1.05x** | **1.12** |
| **Structured Output** | 39.23 tok/s | **39.44 tok/s** | **1.01x** | **1.05** |
| **Technical Explanations** | 38.64 tok/s | **39.40 tok/s** | **1.02x** | **1.05** |

- **Peak VRAM Consumption:** **`2.96` GB** (18.5% of 16GB GPU memory).
- **Authoritative Semantic Preservation:** Target model remains 100% authoritative. In 6 out of 8 categories, speculative decoding produced an exact bitwise token-by-token match with standard autoregressive generation.

---

## Usage with SpecDecode Edge

Clone the repository:
```bash
git clone https://github.com/adityapande403/SpecDecode-edge.git
cd SpecDecode-edge
pip install -r requirements.txt
```

Run speculative generation:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from specdecode.models.medusa_model import MedusaModel
from specdecode.inference.speculative_generator import SpeculativeGenerator

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
medusa_repo_id = "Adityapande403/SpecDecode-edge"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.bfloat16, device_map="cuda:0")

medusa_model = MedusaModel(base_model, num_heads=3, num_layers=1, freeze_base=True)

# Download and load Medusa heads
weights_path = hf_hub_download(repo_id=medusa_repo_id, filename="medusa_heads.safetensors")
medusa_model.load_medusa_heads(weights_path)
medusa_model.eval()

generator = SpeculativeGenerator(medusa_model, tokenizer)
out = generator.generate("Write a quicksort implementation in Python:", max_new_tokens=80)
print(out.generated_text)
```

---

## Citation & Acknowledgements

```bibtex
@software{specdecode_edge_2026,
  author = {Aditya Pande},
  title = {SpecDecode Edge: Training and Deploying Speculative Decoding Models for Local LLM Inference},
  year = {2026},
  url = {https://github.com/adityapande403/SpecDecode-edge}
}
```
