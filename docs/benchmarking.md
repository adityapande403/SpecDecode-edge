# Benchmarking Methodology and Metrics

## 1. Benchmarking Protocol

All benchmarks strictly enforce matched conditions:
- **Model Weights & Checkpoint:** Identical target model revision (`Qwen/Qwen2.5-1.5B-Instruct`) in `bfloat16`.
- **Hardware:** Single NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM, Driver 610.88, CUDA 12.8, sm_120).
- **Workloads:** 8 standardized prompt categories (QA, Code, Math, Structured, Technical, Short, Instruction, Long Context).
- **Warmup Policy:** 2 unmeasured warmup requests prior to benchmark measurement to populate CUDA kernel caches and stabilize GPU power/clock state.
- **Repetitions:** 3 measured runs per prompt category; medians reported to mitigate OS scheduling noise.

## 2. Metrics Definition

1. **Time To First Token (TTFT):**
   $$TTFT = t_{prefill\_end} - t_{prefill\_start} \quad \text{(ms)}$$
2. **Decode Throughput (Tokens/s):**
   $$TPS_{decode} = \frac{N_{generated} - 1}{t_{decode\_total}} \quad \text{(tokens/sec)}$$
3. **Speculative Compression Ratio ($\alpha$):**
   $$\alpha = \frac{\sum_{s=1}^{S} N_{accepted}(s)}{S}$$
   Average number of accepted tokens emitted per target model verification forward pass.
4. **Speedup Ratio:**
   $$\text{Speedup} = \frac{TPS_{speculative}}{TPS_{baseline}}$$
