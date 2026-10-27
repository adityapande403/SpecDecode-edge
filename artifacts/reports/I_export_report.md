# Model Export and Runtime Compatibility Report

## Exported Artifacts

- **Safetensors Weights:** `artifacts/exported_models/medusa_qwen2.5_1.5b\medusa_heads.safetensors` (2697.75 MB)
- **Config Files:** `artifacts/exported_models/medusa_qwen2.5_1.5b\config.json`, `artifacts/exported_models/medusa_qwen2.5_1.5b\medusa_config.json`
- **Precision:** `bfloat16`
- **Target Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` + Medusa 3-Head

## Runtime Integration Audit

| Runtime / Backend | Status | Format | Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| `huggingface_safetensors` | **SUPPORTED** | `safetensors` | Directly loadable in PyTorch / HuggingFace speculative workflows. |
| `vllm_speculative` | **SUPPORTED** | `vLLM Medusa Worker checkpoint` | Compatible with vLLM Medusa speculative engine via `--speculative-model`. |
| `llama_cpp_gguf` | **UNSUPPORTED_ARCH_LIMITATION** | `GGUF draft model` | llama.cpp's `--model-draft` natively requires a standalone CausalLM architecture with complete transformer blocks. Medusa multi-head auxiliary projection heads on frozen base hidden states require specialized GGUF quantization tensor mappings (`medusa.0.weight`) not present in standard GGUF conversion scripts. Native llama-speculative currently prioritizes separate SLM draft models. |
| `ollama` | **UNSUPPORTED_RUNTIME_LIMITATION** | `Modelfile / Ollama draft` | Ollama does not expose a native multi-head Medusa speculative decoding endpoint as of current stable releases. Serving Medusa requires vLLM or our dedicated high-throughput PyTorch runtime. |
