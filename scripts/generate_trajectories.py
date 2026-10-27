"""Generate target-model training trajectories and feature representations.
Saves chunked tensor files with SHA-256 checksums and dataset manifest.
"""

import json
import os
import time
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdecode.data.dataset_manifest import DatasetManifest
from specdecode.trajectory.collector import TrajectoryCollector
from specdecode.utils.logging import get_logger
from specdecode.utils.seed import set_seed

logger = get_logger("generate_trajectories")

# Diverse seed corpus across 8 workload domains for high-quality feature extraction
CORPUS_PROMPTS = [
    # Code generation & systems
    "Write a Python class implementing an LRU Cache with O(1) get and put operations using OrderedDict.",
    "Implement binary search in C++ using templates and robust boundary handling.",
    "Explain how CUDA warps execute SIMT instructions and why branch divergence degrades performance.",
    "Write an async Python function using asyncio to fetch data with exponential backoff and timeout.",
    "Describe the difference between process memory space, stack memory, and heap allocation.",
    "How does virtual memory paging translate virtual addresses to physical frame numbers via page tables?",
    "Explain how TCP flow control uses sliding windows and how it differs from congestion control.",
    "Write a Rust function that safely reverses a linked list in-place using Option and Box.",
    # Mathematics & reasoning
    "Derive the closed-form solution for the sum of the first N squares: 1^2 + 2^2 + ... + N^2.",
    "Explain Bayes' Theorem with a concrete medical diagnosis example having 99% test sensitivity and 1% disease prevalence.",
    "Calculate the eigenvalues and eigenvectors of a 2x2 symmetric matrix [[4, 1], [1, 4]].",
    "Prove by mathematical induction that 2^n > n for all natural numbers n >= 1.",
    "What is the mathematical difference between L1 regularization (Lasso) and L2 regularization (Ridge) regarding sparsity?",
    "Explain gradient descent convergence rate for convex functions under Lipschitz continuous gradients.",
    # Technical explanations & ML systems
    "Explain how FlashAttention uses GPU SRAM tiling to avoid reading and writing the full N x N attention matrix to HBM.",
    "What is the role of Rotary Position Embedding (RoPE) and how does it encode relative token distances?",
    "Compare FP16, BF16, and FP8 floating-point formats in terms of dynamic range, precision bits, and numerical stability.",
    "Describe how speculative decoding accelerates autoregressive generation without altering target model probability distribution.",
    "Explain how multi-query attention (MQA) and grouped-query attention (GQA) reduce KV cache memory bandwidth.",
    "What is the difference between continuous batching and static batching in modern LLM serving engines?",
    # Structured outputs
    "Generate a structured JSON response schema for an edge AI inference gateway logging latency, throughput, and error codes.",
    "Write a YAML configuration for a Docker Compose stack running Redis, PostgreSQL, and a FastAPI worker.",
    "Create a Markdown table comparing SQLite, PostgreSQL, and DuckDB across storage engine, concurrency, and target use case.",
    # General QA & Instruction following
    "What were the major technological innovations that enabled the rapid scaling of deep learning in the 2010s?",
    "List five foundational principles for designing resilient distributed systems under network partitions.",
    "Explain why modern high-performance databases prefer log-structured merge-trees (LSM trees) for write-heavy workloads.",
    "Describe the architectural differences between RISC-V and x86-64 instruction set architectures.",
    "Provide a step-by-step troubleshooting checklist for diagnosing sudden memory leaks in a Python production service.",
]


def generate_trajectories():
    set_seed(42)
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading model for trajectory extraction: {model_id} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_str,
        attn_implementation="sdpa",
    )
    model.eval()

    # Generate completions to create natural target-model trajectories
    logger.info(f"Generating authoritative trajectories from {len(CORPUS_PROMPTS)} prompt seeds...")
    full_trajectories = []

    for prompt in CORPUS_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        full_trajectories.append(text)

    # Train/Val split: 80% train, 20% val
    num_train = int(len(full_trajectories) * 0.8)
    train_texts = full_trajectories[:num_train]
    val_texts = full_trajectories[num_train:]

    output_dir = "data/trajectories"
    collector = TrajectoryCollector(model, tokenizer, device, chunk_size=10)

    t0 = time.time()
    train_chunks = collector.extract_from_texts(train_texts, output_dir=output_dir, split="train", max_seq_len=256)
    val_chunks = collector.extract_from_texts(val_texts, output_dir=output_dir, split="val", max_seq_len=256)
    extraction_time = time.time() - t0

    all_chunks = train_chunks + val_chunks
    total_samples = sum(c.num_samples for c in all_chunks)
    total_tokens = sum(c.num_tokens for c in all_chunks)
    total_storage_bytes = sum(c.file_size_bytes for c in all_chunks)

    manifest = DatasetManifest(
        dataset_name="specdecode_edge_trajectories_v1",
        target_model_id=model_id,
        feature_layer=len(model.model.layers),
        hidden_size=model.config.hidden_size,
        vocab_size=model.config.vocab_size,
        num_train_samples=len(train_texts),
        num_val_samples=len(val_texts),
        total_samples=total_samples,
        total_tokens=total_tokens,
        chunks=all_chunks,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest.save(manifest_path)

    # Generate Data Generation Report
    os.makedirs("artifacts/reports", exist_ok=True)
    report_data = {
        "dataset_name": manifest.dataset_name,
        "target_model": model_id,
        "total_prompts": len(CORPUS_PROMPTS),
        "total_trajectories": total_samples,
        "total_tokens": total_tokens,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "total_storage_mb": round(total_storage_bytes / (1024 * 1024), 2),
        "extraction_duration_seconds": round(extraction_time, 2),
        "chunks": [c.__dict__ for c in all_chunks],
    }

    with open("artifacts/reports/D_data_generation.json", "w") as f:
        json.dump(report_data, f, indent=2)

    with open("artifacts/reports/D_data_generation.md", "w") as f:
        f.write("# Data Generation Report\n\n")
        f.write(f"- **Target Model:** `{model_id}`\n")
        f.write(f"- **Total Extracted Trajectories:** `{total_samples}`\n")
        f.write(f"- **Total Tokens Extracted:** `{total_tokens}`\n")
        f.write(f"- **Train / Val Split:** `{len(train_texts)}` train / `{len(val_texts)}` val\n")
        f.write(f"- **Storage Format:** Chunked PyTorch Tensors (`torch.bfloat16`) with SHA-256 Checksums\n")
        f.write(f"- **Total Storage Size:** `{report_data['total_storage_mb']}` MB\n")
        f.write(f"- **Extraction Time:** `{report_data['extraction_duration_seconds']}` s\n")
        f.write(f"- **Feature Dimension:** `{manifest.hidden_size}` (Layer `{manifest.feature_layer}`)\n")
        f.write(f"- **Vocab Dimension:** `{manifest.vocab_size}`\n\n")

        f.write("## Chunk Manifest\n\n")
        f.write("| Chunk File | Samples | Tokens | Size (MB) | SHA-256 Checksum |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for c in all_chunks:
            f.write(
                f"| `{c.chunk_file}` | {c.num_samples} | {c.num_tokens} | "
                f"{c.file_size_bytes / (1024*1024):.2f} | `{c.sha256_checksum[:16]}...` |\n"
            )

    logger.info(
        f"Trajectory generation complete! Total: {total_samples} samples, {total_tokens} tokens, {report_data['total_storage_mb']} MB."
    )


if __name__ == "__main__":
    generate_trajectories()
