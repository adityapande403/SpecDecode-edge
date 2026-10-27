"""Matched Speculative vs Baseline Performance Benchmark Harness.
Measures latency, decode throughput, speedup ratio, mean acceptance rate, and VRAM across workload categories.
"""

import json
import os
import statistics
import time
from typing import Dict, List
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdecode.data.prompt_dataset import BENCHMARK_PROMPTS
from specdecode.inference.speculative_generator import SpeculativeGenerator
from specdecode.models.medusa_model import MedusaModel
from specdecode.utils.logging import get_logger
from specdecode.utils.seed import set_seed

logger = get_logger("benchmark_speculative")


def run_speculative_benchmark(config_path: str = "configs/benchmarks.yaml"):
    set_seed(42)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["target_model"]["model_id"]
    dtype_str = cfg["target_model"].get("torch_dtype", "bfloat16")
    device_str = cfg["target_model"].get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Check baseline report
    baseline_json_path = "artifacts/reports/C_baseline_report.json"
    if not os.path.exists(baseline_json_path):
        raise FileNotFoundError(f"Baseline report not found at {baseline_json_path}. Run benchmark_baseline.py first.")

    with open(baseline_json_path, "r") as f:
        baseline_report = json.load(f)

    baseline_by_id = {p["id"]: p for p in baseline_report["results_per_prompt"]}

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading base target model: {model_id} on {device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_str,
        attn_implementation="sdpa",
    )
    base_model.eval()

    logger.info("Initializing MedusaModel and loading trained checkpoint...")
    medusa_model = MedusaModel(base_model, num_heads=3, num_layers=1, freeze_base=True)

    ckpt_path = "checkpoints/medusa/best_medusa_heads.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        medusa_model.medusa_heads.load_state_dict(ckpt["state_dict"])
        medusa_model.medusa_heads.to(device=device, dtype=torch.bfloat16)
        logger.info(f"Loaded trained Medusa checkpoint from {ckpt_path}")
    else:
        logger.warning("No checkpoint found; running with initialized weights.")

    medusa_model.eval()
    generator = SpeculativeGenerator(medusa_model, tokenizer)

    warmup_runs = cfg["benchmarking"].get("warmup_runs", 2)
    measured_runs = cfg["benchmarking"].get("measured_runs", 3)

    logger.info(f"Executing {warmup_runs} warm-up speculative runs...")
    for _ in range(warmup_runs):
        generator.generate("Warmup query for GPU speculative execution.", max_new_tokens=16)

    logger.info(f"Starting matched speculative benchmark ({measured_runs} runs per prompt)...")

    results_per_prompt = []

    for item in BENCHMARK_PROMPTS:
        logger.info(f"Benchmarking prompt [{item.id}] ({item.category})...")
        runs_data = []

        for r in range(measured_runs):
            out = generator.generate(
                prompt=item.prompt,
                max_new_tokens=item.target_output_length,
            )
            runs_data.append(
                {
                    "num_generated_tokens": out.num_generated_tokens,
                    "prompt_tokens": out.prompt_tokens,
                    "ttft_ms": out.ttft_ms,
                    "decode_latency_ms": out.decode_latency_ms,
                    "total_latency_ms": out.total_latency_ms,
                    "tokens_per_second": out.tokens_per_second,
                    "decode_tokens_per_second": out.decode_tokens_per_second,
                    "speculative_steps": out.speculative_steps,
                    "mean_accepted_per_step": out.mean_accepted_per_step,
                    "head_acceptance_rates": out.head_acceptance_rates,
                    "peak_vram_gb": out.peak_vram_gb,
                }
            )

        # Medians across runs
        med_ttft = statistics.median(d["ttft_ms"] for d in runs_data)
        med_decode_ms = statistics.median(d["decode_latency_ms"] for d in runs_data)
        med_total_ms = statistics.median(d["total_latency_ms"] for d in runs_data)
        med_tps = statistics.median(d["tokens_per_second"] for d in runs_data)
        med_decode_tps = statistics.median(d["decode_tokens_per_second"] for d in runs_data)
        med_accepted = statistics.median(d["mean_accepted_per_step"] for d in runs_data)
        peak_vram = max(d["peak_vram_gb"] for d in runs_data)

        # Baseline comparison
        base_match = baseline_by_id[item.id]
        base_decode_tps = base_match["median_decode_tokens_per_sec"]
        base_total_tps = base_match["median_tokens_per_sec"]

        decode_speedup = med_decode_tps / base_decode_tps if base_decode_tps > 0 else 1.0
        total_speedup = med_tps / base_total_tps if base_total_tps > 0 else 1.0

        prompt_summary = {
            "id": item.id,
            "category": item.category,
            "prompt_tokens": runs_data[0]["prompt_tokens"],
            "generated_tokens": runs_data[0]["num_generated_tokens"],
            "median_ttft_ms": round(med_ttft, 2),
            "median_decode_ms": round(med_decode_ms, 2),
            "median_total_ms": round(med_total_ms, 2),
            "speculative_decode_tps": round(med_decode_tps, 2),
            "baseline_decode_tps": round(base_decode_tps, 2),
            "decode_speedup": round(decode_speedup, 2),
            "total_speedup": round(total_speedup, 2),
            "mean_accepted_per_step": round(med_accepted, 2),
            "head_0_acc_rate": round(runs_data[0]["head_acceptance_rates"]["head_0"] * 100, 1),
            "head_1_acc_rate": round(runs_data[0]["head_acceptance_rates"]["head_1"] * 100, 1),
            "head_2_acc_rate": round(runs_data[0]["head_acceptance_rates"]["head_2"] * 100, 1),
            "peak_vram_gb": round(peak_vram, 3),
        }
        results_per_prompt.append(prompt_summary)
        logger.info(
            f"Prompt [{item.id}] -> Spec Decode TPS: {med_decode_tps:.2f} (Base: {base_decode_tps:.2f}, Speedup: {decode_speedup:.2f}x, Mean Acc: {med_accepted:.2f})"
        )

    # Summary aggregates
    all_spec_decode_tps = [p["speculative_decode_tps"] for p in results_per_prompt]
    all_base_decode_tps = [p["baseline_decode_tps"] for p in results_per_prompt]
    all_decode_speedups = [p["decode_speedup"] for p in results_per_prompt]
    all_accepted = [p["mean_accepted_per_step"] for p in results_per_prompt]
    overall_peak_vram = max(p["peak_vram_gb"] for p in results_per_prompt)

    overall_summary = {
        "target_model": model_id,
        "speculative_method": "Medusa-3Heads",
        "median_baseline_decode_tps": round(statistics.median(all_base_decode_tps), 2),
        "median_speculative_decode_tps": round(statistics.median(all_spec_decode_tps), 2),
        "median_decode_speedup": round(statistics.median(all_decode_speedups), 2),
        "mean_decode_speedup": round(statistics.mean(all_decode_speedups), 2),
        "median_accepted_per_step": round(statistics.median(all_accepted), 2),
        "peak_vram_gb": round(overall_peak_vram, 3),
    }

    report = {
        "summary": overall_summary,
        "results_per_prompt": results_per_prompt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs("artifacts/reports", exist_ok=True)
    json_path = "artifacts/reports/H_performance_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = "artifacts/reports/H_performance_report.md"
    with open(md_path, "w") as f:
        f.write("# Speculative Decoding Performance Report\n\n")
        f.write(f"- **Target Model:** `{model_id}`\n")
        f.write(f"- **Speculative Method:** Medusa-Style Multi-Head Speculative Decoding (3 heads)\n")
        f.write(f"- **Median Baseline Decode Throughput:** `{overall_summary['median_baseline_decode_tps']}` tokens/sec\n")
        f.write(f"- **Median Speculative Decode Throughput:** `{overall_summary['median_speculative_decode_tps']}` tokens/sec\n")
        f.write(f"- **Median Speedup:** `{overall_summary['median_decode_speedup']}x` (Mean: `{overall_summary['mean_decode_speedup']}x`)\n")
        f.write(f"- **Median Accepted Tokens Per Verification Step:** `{overall_summary['median_accepted_per_step']}`\n")
        f.write(f"- **Peak VRAM:** `{overall_summary['peak_vram_gb']}` GB (Baseline was `{baseline_report['summary']['peak_vram_gb']}` GB)\n")
        f.write(f"- **VRAM Overhead:** `{round(overall_summary['peak_vram_gb'] - baseline_report['summary']['peak_vram_gb'], 3)}` GB\n\n")

        f.write("## Workload Breakdown & Speedup Comparison\n\n")
        f.write("| Category | Prompt ID | Base TPS | Spec TPS | Speedup | Mean Acc/Step | Head0 Acc | Head1 Acc | Head2 Acc |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for p in results_per_prompt:
            f.write(
                f"| {p['category']} | `{p['id']}` | {p['baseline_decode_tps']} | {p['speculative_decode_tps']} | "
                f"**{p['decode_speedup']}x** | {p['mean_accepted_per_step']} | "
                f"{p['head_0_acc_rate']}% | {p['head_1_acc_rate']}% | {p['head_2_acc_rate']}% |\n"
            )

    logger.info(f"Performance benchmark complete. Reports saved to {json_path} and {md_path}")
    print(f"Speculative Decode Throughput: {overall_summary['median_speculative_decode_tps']} tokens/sec")
    print(f"Median Speedup: {overall_summary['median_decode_speedup']}x")
    return report


if __name__ == "__main__":
    run_speculative_benchmark()
