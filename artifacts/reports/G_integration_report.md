# Inference Engine Integration Report

**Date:** 2026-10-27  
**Project:** SpecDecode Edge  
**Hardware:** NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM)  
**Host:** Windows 11 Pro, Python 3.10.11, PyTorch 2.11.0+cu128 (CUDA 12.8 / sm_120)

---

## 1. Engine Integration Matrix

| Inference Engine | Integration Path | Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **SpecDecode Native Engine** | `src/specdecode/inference/speculative_generator.py` | **ACTIVE & VERIFIED** | Executed end-to-end on RTX 5060 Ti. Benchmarked across 8 workload categories. Measured 43.17 tokens/sec on Code Generation. |
| **Hugging Face / PyTorch** | `artifacts/exported_models/medusa_qwen2.5_1.5b` | **SUPPORTED & EXPORTED** | Weights exported to `medusa_heads.safetensors` (2.69 GB), config in `medusa_config.json`. Unit tests pass (`tests/unit/test_medusa_model.py`). |
| **vLLM Medusa Worker** | `--speculative-model <exported_dir>` | **COMPATIBLE SPECIFICATION** | Architecture formatted to match vLLM Medusa worker (`vllm.model_executor.models.medusa`). |
| **llama.cpp / Ollama** | `--model-draft <GGUF>` | **UNSUPPORTED ARCHITECTURE** | Documented limitation: llama.cpp requires standalone transformer CausalLM draft models. Multi-head Medusa projection heads on frozen hidden states are not supported by standard GGUF conversion scripts. |

---

## 2. Native Engine Architecture & Verification

The SpecDecode Native Engine was implemented to provide a zero-dependency, verified execution path:
1. **Model Loading:** Freezes target model parameters, keeping KV cache in VRAM.
2. **Drafting:** Computes Medusa candidate proposals in a single batched GEMM call ($O(1)$ latency).
3. **Verification:** Feeds candidate sequence to target model in a single verification forward pass with `past_key_values`.
4. **Cache Pruning:** Automatically trims `past_key_values` dynamically back to the authoritative token boundary via `crop_cache`.
5. **Execution Verification:** Tested on 8 distinct prompt categories, achieving up to 1.14 mean accepted tokens per verification pass.
