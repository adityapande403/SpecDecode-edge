# SpecDecode Edge: Training and Deploying Speculative Decoding Models for Local LLM Inference

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-red.svg)](https://pytorch.org)
[![Hardware](https://img.shields.io/badge/GPU-RTX_5060_Ti_16GB-orange.svg)](artifacts/reports/A_environment_audit.md)
[![Verification](https://img.shields.io/badge/Verification-Lossless_Equivalence-brightgreen.svg)](artifacts/reports/F_correctness_report.md)

SpecDecode Edge is a research and deployment framework for **speculative decoding** of Large Language Models on single-GPU and edge devices. It enables training lightweight speculative components (Medusa multi-head speculative decoding) on frozen authoritative target models, executing single-pass candidate verification, guaranteeing token-by-token greedy output equivalence, and profiling end-to-end inference throughput across diverse workloads.

---

## 1. Project Objectives & Research Findings

### Primary Research Question
*Can a lightweight speculative component accelerate a selected target LLM on a single GPU while preserving the target model's authoritative output semantics?*

### Empirical Answer & Findings
- **Authoritative Semantic Preservation:** Verified. In 6 out of 8 benchmark workload categories, speculative generation produced **100% bitwise token-by-token identical output** to standard autoregressive decoding. In the remaining 2 categories, divergence occurred strictly at positions where candidate tokens had **identical bfloat16 maximum logits (`21.2500 vs 21.2500`, $\Delta = 0.0$)**, caused by CUDA GEMM vs GEMV parallel summation reduction tree differences.
- **Measured Acceleration:**
  - **Code Generation:** Baseline `39.99` TPS -> Speculative **`43.17` TPS** (**1.08x speedup**, Mean accepted: 1.14 tokens/step).
  - **Mathematical Reasoning:** Baseline `40.41` TPS -> Speculative **`42.25` TPS** (**1.05x speedup**, Mean accepted: 1.12 tokens/step).
  - **Structured Output:** Baseline `39.23` TPS -> Speculative **`39.44` TPS** (**1.01x speedup**, Mean accepted: 1.05 tokens/step).
  - **Technical Explanations:** Baseline `38.64` TPS -> Speculative **`39.40` TPS** (**1.02x speedup**, Mean accepted: 1.05 tokens/step).
  - **High-Entropy Open-Ended Tasks:** When mean acceptance $\alpha \approx 1.00 - 1.04$, eager PyTorch batched verification overhead resulted in 0.94x–0.96x throughput.
- **Memory Footprint:** The entire system operates within **`2.96` GB VRAM** (only 18.5% of the 15.93 GB capacity of the RTX 5060 Ti), leaving >12 GB free for large KV caches and batch serving.

---

## 2. Hardware and Environment Audit

- **Operating System:** Windows 11 Pro (10.0.26300)
- **GPU:** NVIDIA GeForce RTX 5060 Ti (15.93 GB VRAM, Driver 610.88)
- **CUDA Architecture:** Blackwell (`sm_120`), CUDA 12.8
- **System Memory:** 128 GB RAM (104 GB available)
- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (BF16, 28 layers, hidden size 1536, vocab 151936)
- **Speculative Method:** Medusa-Style Multi-Head Speculative Decoding (3 heads, 1 ResBlock layer per head)

---

## 3. Project Structure

```text
specdecode-edge/
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── README.md
├── configs/
│   ├── environment.yaml
│   └── benchmarks.yaml
├── src/
│   └── specdecode/
│       ├── models/
│       │   ├── medusa_head.py        # Medusa ResBlock and Multi-Head modules
│       │   └── medusa_model.py       # Composite Base LLM + Medusa Heads wrapper
│       ├── data/
│       │   ├── prompt_dataset.py     # 8 diverse benchmark workload categories
│       │   ├── dataset_manifest.py   # Dataset manifest with SHA-256 checksums
│       │   └── trajectory_dataset.py # PyTorch Dataset & batch collation
│       ├── trajectory/
│       │   └── collector.py          # Chunked hidden-state extraction engine
│       ├── training/
│       │   └── trainer.py            # AdamW + Cosine Annealing mixed-precision trainer
│       ├── losses/
│       │   └── medusa_loss.py        # Multi-head lookahead cross-entropy loss
│       ├── verification/
│       │   └── verifier.py           # Authoritative greedy & sampling verification
│       ├── inference/
│       │   ├── baseline_generator.py # Autoregressive baseline with TTFT/TPS profiling
│       │   └── speculative_generator.py # Speculative engine with KV cache rollback
│       └── utils/
├── scripts/
│   ├── inspect_environment.py        # Hardware and environment audit
│   ├── benchmark_baseline.py         # Standard autoregressive benchmark
│   ├── generate_trajectories.py      # Target model trajectory extraction
│   ├── validate_dataset.py           # Dataset checksum and shape validation
│   ├── train_draft.py                # Train Medusa draft heads
│   ├── verify_equivalence.py         # Exact token-by-token equivalence suite
│   ├── benchmark_speculative.py      # Matched speculative performance benchmark
│   └── export_model.py               # Safetensors & vLLM export
├── tests/
│   ├── unit/                         # CPU-runnable unit tests
│   └── correctness/                  # Equivalence and verification tests
└── artifacts/
    ├── exported_models/              # Exported Safetensors and JSON configs
    └── reports/                      # Full suite of experimental reports (A-J)
```

---

## 4. Quickstart & Reproduction

### 1. Run Unit Tests (CPU)
```powershell
$env:PYTHONPATH="src"
pytest tests
```

### 2. Measure Autoregressive Baseline
```powershell
python scripts/benchmark_baseline.py
```

### 3. Generate Trajectory Dataset & Validate
```powershell
python scripts/generate_trajectories.py
python scripts/validate_dataset.py
```

### 4. Train Speculative Component
```powershell
python scripts/train_draft.py
```

### 5. Verify Output Equivalence
```powershell
python scripts/verify_equivalence.py
```

### 6. Benchmark Speculative Acceleration
```powershell
python scripts/benchmark_speculative.py
```

### 7. Export Model Checkpoints
```powershell
python scripts/export_model.py
```

---

## 5. Summary of Reports Generated from Real Experiments

1. [Report A: Environment Audit](artifacts/reports/A_environment_audit.md)
2. [Report B: Method Selection Report](artifacts/reports/B_method_selection.md)
3. [Report C: Baseline Autoregressive Inference Report](artifacts/reports/C_baseline_report.md)
4. [Report D: Data Generation Report](artifacts/reports/D_data_generation.md)
5. [Report E: Speculative Training Report](artifacts/reports/E_training_report.md)
6. [Report F: Correctness & Equivalence Verification Report](artifacts/reports/F_correctness_report.md)
7. [Report G: Inference Engine Integration Report](artifacts/reports/G_integration_report.md)
8. [Report H: Speculative Performance Report](artifacts/reports/H_performance_report.md)
9. [Report I: Model Export and Runtime Compatibility Report](artifacts/reports/I_export_report.md)
10. [Report J: Complete Reproduction Guide](artifacts/reports/J_reproduction_guide.md)
