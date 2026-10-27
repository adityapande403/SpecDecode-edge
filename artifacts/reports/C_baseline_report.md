# Baseline Autoregressive Inference Report

- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Precision:** `bfloat16`
- **Device:** `cuda:0`
- **Overall Median Decode Throughput:** `40.14` tokens/sec
- **Overall Median TTFT (Prefill):** `29.7` ms
- **Peak VRAM:** `2.957` GB
- **Measured Runs:** `3` per prompt (with 2 warm-ups)

## Workload Breakdown

| Category | Prompt ID | Prompt Tokens | Gen Tokens | TTFT (ms) | Decode Time (ms) | Decode Tokens/s | Peak VRAM (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| General QA | `qa_01` | 15 | 64 | 29.84 | 1563.23 | 40.3 | 2.889 |
| Code Generation | `code_01` | 22 | 80 | 29.09 | 1975.41 | 39.99 | 2.892 |
| Mathematical Reasoning | `math_01` | 31 | 64 | 30.52 | 1559.01 | 40.41 | 2.895 |
| Structured Output | `structured_01` | 24 | 70 | 29.82 | 1758.88 | 39.23 | 2.892 |
| Technical Explanations | `technical_01` | 20 | 80 | 29.23 | 2044.71 | 38.64 | 2.891 |
| Short Interactive | `short_01` | 7 | 20 | 29.22 | 484.02 | 39.25 | 2.887 |
| Instruction Following | `instruction_01` | 19 | 80 | 29.57 | 1956.78 | 40.37 | 2.891 |
| Long Context | `long_context_01` | 220 | 75 | 31.19 | 1816.66 | 40.73 | 2.957 |
