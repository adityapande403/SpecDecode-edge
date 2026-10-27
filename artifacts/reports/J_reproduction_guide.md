# SpecDecode Edge: Complete Reproduction Guide

This guide provides the exact commands to reproduce the environment setup, baseline benchmark, trajectory extraction, Medusa training, correctness suite, and speculative performance benchmarking.

---

## 1. Environment Setup

```powershell
# Create dedicated virtual environment
python -m venv .venv

# Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support (required for RTX 5060 Ti / sm_120 Blackwell)
.\.venv\Scripts\pip.exe install torch==2.11.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

## 2. Step-by-Step Reproduction Pipeline

### Step 1: Environment Audit
Audit hardware, CUDA capabilities, disk space, and installed software:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/inspect_environment.py
```
*Output: `artifacts/reports/A_environment_audit.json` and `.md`*

### Step 2: Run Unit Test Suite
Verify model forward pass, loss calculation, and verification logic on CPU:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\pytest.exe tests
```
*Output: 12 passing tests in ~2.9s*

### Step 3: Measure Autoregressive Baseline
Benchmark target model (`Qwen/Qwen2.5-1.5B-Instruct`) in bfloat16 across 8 workload categories:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/benchmark_baseline.py
```
*Output: `artifacts/reports/C_baseline_report.json` and `.md` (Median: 40.14 tokens/sec, Peak VRAM: 2.957 GB)*

### Step 4: Generate Training Trajectories
Extract target-model final hidden states and token labels with SHA-256 checksum manifests:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/generate_trajectories.py
```
*Output: `data/trajectories/manifest.json`, `artifacts/reports/D_data_generation.md`*

### Step 5: Validate Trajectory Dataset Integrity
Verify checksums, non-null values, and tensor shapes:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/validate_dataset.py
```

### Step 6: Train Speculative Draft Component
Train 3 Medusa heads using multi-head masked cross-entropy with cosine decay:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/train_draft.py
```
*Output: `checkpoints/medusa/best_medusa_heads.pt`, `artifacts/reports/E_training_report.md`*

### Step 7: Verify Exact Correctness & Greedy Equivalence
Compare token-by-token greedy outputs between baseline and speculative decoding:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/verify_equivalence.py
```
*Output: `artifacts/reports/F_correctness_report.md`*

### Step 8: Benchmark Speculative Decoding Performance
Measure end-to-end latency, tokens/sec, acceptance rate by head, and speedup ratios:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/benchmark_speculative.py
```
*Output: `artifacts/reports/H_performance_report.json` and `.md`*

### Step 9: Export Model Artifacts
Export trained weights to Safetensors and vLLM configuration:
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/export_model.py
```
*Output: `artifacts/exported_models/medusa_qwen2.5_1.5b/`, `artifacts/reports/I_export_report.md`*
