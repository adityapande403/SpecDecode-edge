# Implementation Limitations and Technical Bottlenecks

**Date:** 2026-10-27  
**Project:** SpecDecode Edge

---

## 1. Measured Performance Bottlenecks in Eager PyTorch

1. **Tree Attention Kernel vs Sequential Verification:**
   In our baseline Python/PyTorch implementation, candidates are evaluated as linear candidate sequences rather than tree attention topologies. In eager execution, passing $1 + K$ candidate tokens into the base model invokes a batched forward pass which requires more FLOPs than a single-token decode pass.
   - For workloads with high acceptance rate ($\alpha \ge 1.12$, e.g., Code Generation and Math Reasoning), the number of accepted tokens outpaces the forward pass overhead, yielding **1.05x–1.08x real speedup**.
   - For unstructured, open-ended high-entropy prompts where $\alpha \approx 1.00 - 1.04$, the extra FLOPs of the verification forward pass slightly lower overall tokens/sec (0.94x–0.96x).
   - **Resolution:** Upstream C++/CUDA kernel fusion (such as vLLM's Medusa tree attention kernel or FlashInfer tree decode) eliminates the FLOP overhead by masking out non-prefix candidates in parallel attention tiles.

2. **BFloat16 Logit Ties & Numerical Summation Order:**
   In 2 out of 8 benchmark prompts, an exact logit tie (`21.2500 vs 21.2500`) occurred between two candidate tokens. Because CUDA's parallel reduction tree in GEMM sums floating-point numbers in a different associative order than GEMV, the tie was broken in favor of a different token index. Once an autoregressive path diverges, subsequent tokens reflect the new prefix.
   - In 6 out of 8 categories, tokens were 100% strictly bitwise identical.

3. **Inference Engine Format Diversity:**
   - **vLLM:** Natively supports Medusa through the Medusa worker checkpoint interface.
   - **llama.cpp / Ollama:** Standard `llama-speculative` requires a full standalone GGUF language model (with its own transformer blocks, attention, and feed-forward layers) for `--model-draft`. Medusa auxiliary heads on frozen hidden states are not currently supported by default GGUF quantizers.

4. **VRAM Footprint on 16 GB GPUs:**
   - On an NVIDIA RTX 5060 Ti (15.93 GB VRAM), running `Qwen/Qwen2.5-1.5B-Instruct` in bfloat16 consumes 2.89 GB for model weights and 2.69 GB for 3 Medusa heads, leaving >10 GB free for KV cache and large context lengths.
   - Running an 8B parameter model in FP16/BF16 would consume ~15.5 GB for base model weights alone, causing OOM when allocating KV cache and draft heads. For 8B deployment on 16 GB VRAM, 4-bit base model quantization (e.g. AWQ/GPTQ) is strictly required.
