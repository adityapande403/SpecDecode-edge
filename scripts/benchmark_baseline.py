"""Baseline Autoregressive Benchmark Script.
Measures target-model generation latency, TTFT, decode throughput, and VRAM across workload categories.
"""

import json
import os
import statistics
import time
from typing import Any, Dict, List
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdecode.data.prompt_dataset import BENCHMARK_PROMPTS
from specdecode.inference.baseline_generator import BaselineGenerator
from specdecode.utils.logging import get_logger
from specdecode.utils.seed import set_seed

logger = get_logger("baseline_benchmark")


def run_baseline_benchmark(config_path: str = "configs/benchmarks.yaml"):
    set_seed(42)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["target_model"]["model_id"]
    dtype_str = cfg["target_model"].get("torch_dtype", "bfloat16")
    device_str = cfg["target_model"].get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    attn_impl = cfg["target_model"].get("attn_implementation", "sdpa")

    dtype = getattr(torch, dtype_str) if hasattr(torch, dtype_str) else torch.bfloat16

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading target model: {model_id} in {dtype} on {device_str}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_str,
        attn_implementation=attn_impl,
    )
    model.eval()

    generator = BaselineGenerator(model, tokenizer)

    warmup_runs = cfg["benchmarking"].get("warmup_runs", 2)
    measured_runs = cfg["benchmarking"].get("measured_runs", 3)
    logger.info(f"Executing {warmup_runs} warm-up runs...")

    for i in range(warmup_runs):
        generator.generate("Warmup test query for GPU kernel caches.", max_new_tokens=16)

    logger.info("Starting baseline benchmarking across workload categories...")

    results_per_prompt = []
    category_metrics: Dict[str, List[Dict[str, float]]] = {}

    for item in BENCHMARK_PROMPTS:
        logger.info(f"Benchmarking prompt [{item.id}] ({item.category})...")
        runs_data = []

        for r in range(measured_runs):
            out = generator.generate(
                prompt=item.prompt,
                max_new_tokens=item.target_output_length,
                do_sample=False,
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
                    "peak_vram_gb": out.peak_vram_gb,
                }
            )

        # Compute medians and std
        med_ttft = statistics.median(d["ttft_ms"] for d in runs_data)
        med_decode_ms = statistics.median(d["decode_latency_ms"] for d in runs_data)
        med_total_ms = statistics.median(d["total_latency_ms"] for d in runs_data)
        med_tps = statistics.median(d["tokens_per_second"] for d in runs_data)
        med_decode_tps = statistics.median(d["decode_tokens_per_second"] for d in runs_data)
        med_vram = max(d["peak_vram_gb"] for d in runs_data)

        prompt_summary = {
            "id": item.id,
            "category": item.category,
            "prompt": item.prompt,
            "prompt_tokens": runs_data[0]["prompt_tokens"],
            "generated_tokens": runs_data[0]["num_generated_tokens"],
            "median_ttft_ms": round(med_ttft, 2),
            "median_decode_ms": round(med_decode_ms, 2),
            "median_total_ms": round(med_total_ms, 2),
            "median_tokens_per_sec": round(med_tps, 2),
            "median_decode_tokens_per_sec": round(med_decode_tps, 2),
            "peak_vram_gb": round(med_vram, 3),
            "runs": runs_data,
        }
        results_per_prompt.append(prompt_summary)
        category_metrics.setdefault(item.category, []).append(prompt_summary)

    # Compute overall aggregates
    all_decode_tps = [p["median_decode_tokens_per_sec"] for p in results_per_prompt]
    all_total_tps = [p["median_tokens_per_sec"] for p in results_per_prompt]
    all_ttft = [p["median_ttft_ms"] for p in results_per_prompt]
    overall_peak_vram = max(p["peak_vram_gb"] for p in results_per_prompt)

    overall_summary = {
        "target_model": model_id,
        "dtype": dtype_str,
        "device": device_str,
        "total_prompts": len(BENCHMARK_PROMPTS),
        "measured_runs_per_prompt": measured_runs,
        "median_decode_tokens_per_sec": round(statistics.median(all_decode_tps), 2),
        "mean_decode_tokens_per_sec": round(statistics.mean(all_decode_tps), 2),
        "median_ttft_ms": round(statistics.median(all_ttft), 2),
        "peak_vram_gb": round(overall_peak_vram, 3),
    }

    report = {
        "summary": overall_summary,
        "results_per_prompt": results_per_prompt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save JSON report
    os.makedirs("artifacts/reports", exist_ok=True)
    json_path = "artifacts/reports/C_baseline_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_path = "artifacts/reports/C_baseline_report.md"
    with open(md_path, "w") as f:
        f.write("# Baseline Autoregressive Inference Report\n\n")
        f.write(f"- **Target Model:** `{model_id}`\n")
        f.write(f"- **Precision:** `{dtype_str}`\n")
        f.write(f"- **Device:** `{device_str}`\n")
        f.write(f"- **Overall Median Decode Throughput:** `{overall_summary['median_decode_tokens_per_sec']}` tokens/sec\n")
        f.write(f"- **Overall Median TTFT (Prefill):** `{overall_summary['median_ttft_ms']}` ms\n")
        f.write(f"- **Peak VRAM:** `{overall_summary['peak_vram_gb']}` GB\n")
        f.write(f"- **Measured Runs:** `{measured_runs}` per prompt (with 2 warm-ups)\n\n")

        f.write("## Workload Breakdown\n\n")
        f.write("| Category | Prompt ID | Prompt Tokens | Gen Tokens | TTFT (ms) | Decode Time (ms) | Decode Tokens/s | Peak VRAM (GB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for p in results_per_prompt:
            f.write(
                f"| {p['category']} | `{p['id']}` | {p['prompt_tokens']} | {p['generated_tokens']} | "
                f"{p['median_ttft_ms']} | {p['median_decode_ms']} | {p['median_decode_tokens_per_sec']} | {p['peak_vram_gb']} |\n"
            )

    logger.info(f"Baseline benchmark complete. Reports saved to {json_path} and {md_path}")
    print(f"Overall Baseline Decode Throughput: {overall_summary['median_decode_tokens_per_sec']} tokens/sec")
    print(f"Peak VRAM: {overall_summary['peak_vram_gb']} GB")
    return report


if __name__ == "__main__":
    run_baseline_benchmark()
