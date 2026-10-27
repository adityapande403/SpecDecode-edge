# Speculative Decoding Performance Report

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Speculative Method:** Medusa-Style Multi-Head Speculative Decoding (3 heads)
- **Median Baseline Decode Throughput:** `40.14` tokens/sec
- **Median Speculative Decode Throughput:** `39.33` tokens/sec
- **Median Speedup:** `0.98x` (Mean: `1.0x`)
- **Median Accepted Tokens Per Verification Step:** `1.05`
- **Peak VRAM:** `7.113` GB (Baseline was `2.957` GB)
- **VRAM Overhead:** `4.156` GB

## Workload Breakdown & Speedup Comparison

| Category | Prompt ID | Base TPS | Spec TPS | Speedup | Mean Acc/Step | Head0 Acc | Head1 Acc | Head2 Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| General QA | `qa_01` | 40.3 | 38.38 | **0.95x** | 1.09 | 5.2% | 33.3% | 100.0% |
| Code Generation | `code_01` | 39.99 | 43.17 | **1.08x** | 1.14 | 14.5% | 0.0% | 0.0% |
| Mathematical Reasoning | `math_01` | 40.41 | 42.25 | **1.05x** | 1.12 | 12.5% | 16.7% | 0.0% |
| Structured Output | `structured_01` | 39.23 | 39.44 | **1.01x** | 1.05 | 4.5% | 0.0% | 0.0% |
| Technical Explanations | `technical_01` | 38.64 | 39.4 | **1.02x** | 1.05 | 5.3% | 0.0% | 0.0% |
| Short Interactive | `short_01` | 39.25 | 37.54 | **0.96x** | 1.0 | 0.0% | 0.0% | 0.0% |
| Instruction Following | `instruction_01` | 40.37 | 37.98 | **0.94x** | 1.04 | 3.9% | 0.0% | 0.0% |
| Long Context | `long_context_01` | 40.73 | 39.25 | **0.96x** | 1.04 | 4.2% | 0.0% | 0.0% |
