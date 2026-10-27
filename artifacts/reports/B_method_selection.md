# Method Selection Report: Medusa-Style Multi-Head Speculative Decoding

**Date:** 2026-10-27  
**Project:** SpecDecode Edge  
**Selected Speculative Method:** Medusa-Style Multi-Head Speculative Decoding (Cai et al., 2024)  
**Target Model Family:** Qwen2.5 / Qwen3 (`Qwen/Qwen2.5-1.5B-Instruct` as primary verified baseline, with configuration scaling to 7B/8B)

---

## 1. Comparative Analysis of Candidate Methods

| Evaluation Dimension | Medusa Multi-Head | EAGLE / EAGLE-2 | Independent Draft Model |
| :--- | :--- | :--- | :--- |
| **Draft Architecture** | $K$ lightweight residual heads on frozen target hidden states | 1-layer auto-regressive transformer decoder | Standalone SLM (e.g., 0.5B-1B LLM) |
| **Parameter Overhead** | ~15M–45M params (< 100 MB) | ~150M–250M params (~400 MB) | 500M–1.5B params (1.0–3.0 GB) |
| **VRAM Footprint** | Extremely low (< 100 MB weights) | Moderate (~500 MB) | High (requires 2nd model KV cache) |
| **Draft Latency** | Single parallel linear/MLP pass ($O(1)$) | Auto-regressive sequential steps ($O(K)$) | Auto-regressive sequential steps ($O(K)$) |
| **Training Complexity** | Supervised multi-step token cross-entropy on frozen target features | Multi-loss: Feature MSE/Smooth-L1 + LM cross-entropy | Full language model pre-training / fine-tuning |
| **Target Hidden-State Access** | Direct access to final hidden state $h_t$ | Access to intermediate/top hidden states & embeddings | None (only token IDs communicated) |
| **Verification Algorithm** | Single target-model forward pass with tree attention or candidate sequence | Single target-model tree attention forward pass | Sequential or batched target-model verification |
| **Greedy Output Equivalence** | Guaranteed exact match via authoritative target verification | Guaranteed exact match via authoritative target verification | Guaranteed exact match via authoritative target verification |
| **Runtime Support** | Native PyTorch, vLLM Medusa worker, Hugging Face custom generate | vLLM EAGLE worker, native PyTorch | vLLM `--speculative-model`, llama.cpp `--model-draft` |
| **Quantization Compatibility** | Base model can be FP16, BF16, FP8, or INT4 (AWQ/GPTQ/bitsandbytes); heads operate in BF16/FP16 | Base model can be quantized; draft model requires separate precision | Both models can be independently quantized |

---

## 2. Selection Rationale

For edge and local LLM deployment on an NVIDIA RTX 5060 Ti (15.93 GB VRAM):

1. **Memory Budget & Overhead**:
   On a single 16 GB GPU, loading an authoritative model alongside a separate 1B+ draft model severely constrains KV-cache allocation and maximum context length. In contrast, Medusa adds only $K$ residual heads (~25MB total), consuming virtually zero additional VRAM and allowing 99%+ of GPU memory to be dedicated to the base model's KV cache.

2. **Draft Generation Speed**:
   Independent draft models and EAGLE require $K$ sequential autoregressive steps to generate $K$ candidate tokens. Medusa computes all $K$ candidate heads in parallel in a single sub-millisecond tensor operation directly from the target model's final hidden state $h_t$.

3. **Authoritative Semantic Preservation**:
   The target model is 100% authoritative. In greedy decoding ($T=0$), candidate tokens $c_1, \dots, c_K$ are verified simultaneously in the target model. If candidate $c_i$ matches the target model's $\operatorname{argmax}$, it is accepted. At the first mismatch, the target model's true greedy token is accepted, and all remaining candidates are discarded.

4. **Training Stability & Data Efficiency**:
   Medusa heads are trained with cross-entropy against future ground-truth target tokens:
   $$\mathcal{L} = \sum_{k=1}^K \lambda_k \mathcal{L}_{CE}^{(k)}$$
   Because the base model is frozen, feature representations $h_t$ are deterministic, enabling rapid training without catastrophic forgetting or gradient instability.

---

## 3. Architecture Specification

- **Number of Heads ($K$):** 3 speculative heads (predicting offsets $t+1, t+2, t+3$).
- **Head Architecture:**
  - Input: Hidden state $h_t \in \mathbb{R}^{d}$ (for Qwen2.5-1.5B, $d=1536$).
  - Residual connection: $z_k = h_t + \operatorname{SiLU}(W_1^{(k)} h_t)$ where $W_1^{(k)} \in \mathbb{R}^{d \times d}$.
  - Projection to Vocab: $\mathrm{logits}_k = W_2^{(k)} z_k$ where $W_2^{(k)} \in \mathbb{R}^{V \times d}$ (or tied to LM head).
- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (BF16, 28 layers, hidden size 1536, vocab 151936), readily extensible to 7B/8B.
