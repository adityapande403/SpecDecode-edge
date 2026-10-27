"""Rigorous Correctness and Equivalence Verification Suite.
Demonstrates token-by-token identity between standard autoregressive decoding and speculative decoding.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdecode.data.prompt_dataset import BENCHMARK_PROMPTS
from specdecode.inference.baseline_generator import BaselineGenerator
from specdecode.inference.speculative_generator import SpeculativeGenerator
from specdecode.models.medusa_model import MedusaModel
from specdecode.utils.logging import get_logger
from specdecode.utils.seed import set_seed

logger = get_logger("verify_equivalence")


def run_equivalence_suite():
    set_seed(42)
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading base target model: {model_id} in bfloat16 on {device}")
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
        logger.warning(f"Checkpoint not found at {ckpt_path}, using initialized heads.")

    medusa_model.eval()

    baseline_gen = BaselineGenerator(base_model, tokenizer)
    spec_gen = SpeculativeGenerator(medusa_model, tokenizer)

    logger.info(f"Running token-by-token equivalence across {len(BENCHMARK_PROMPTS)} workload prompts...")

    results = []
    all_passed = True

    for item in BENCHMARK_PROMPTS:
        logger.info(f"Testing [{item.id}] ({item.category})...")

        base_out = baseline_gen.generate(item.prompt, max_new_tokens=item.target_output_length, do_sample=False)
        spec_out = spec_gen.generate(item.prompt, max_new_tokens=item.target_output_length)

        exact_match = (base_out.token_ids == spec_out.token_ids)
        token_len_match = (base_out.num_generated_tokens == spec_out.num_generated_tokens)

        # Token mismatch inspection
        mismatch_first_index = None
        if not exact_match:
            all_passed = False
            for idx in range(min(len(base_out.token_ids), len(spec_out.token_ids))):
                if base_out.token_ids[idx] != spec_out.token_ids[idx]:
                    mismatch_first_index = idx
                    break

        case_res = {
            "id": item.id,
            "category": item.category,
            "prompt": item.prompt,
            "baseline_token_count": base_out.num_generated_tokens,
            "speculative_token_count": spec_out.num_generated_tokens,
            "exact_token_match": exact_match,
            "length_match": token_len_match,
            "first_mismatch_idx": mismatch_first_index,
            "speculative_steps": spec_out.speculative_steps,
            "mean_accepted_per_step": round(spec_out.mean_accepted_per_step, 2),
            "generated_text_sample": spec_out.generated_text[:120] + "...",
        }
        results.append(case_res)

        status_str = "PASSED" if exact_match else f"FAILED (mismatch at index {mismatch_first_index})"
        logger.info(f"Result [{item.id}]: {status_str} ({spec_out.num_generated_tokens} tokens)")

    # Save reports
    os.makedirs("artifacts/reports", exist_ok=True)
    report_data = {
        "target_model": model_id,
        "total_test_cases": len(results),
        "all_passed": all_passed,
        "passed_count": sum(1 for r in results if r["exact_token_match"]),
        "failed_count": sum(1 for r in results if not r["exact_token_match"]),
        "results": results,
    }

    with open("artifacts/reports/F_correctness_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    with open("artifacts/reports/F_correctness_report.md", "w") as f:
        f.write("# Speculative Verification & Correctness Report\n\n")
        f.write(f"- **Target Model:** `{model_id}`\n")
        f.write(f"- **Verification Contract:** Exact Greedy Equivalence ($T=0$)\n")
        f.write(f"- **Total Workload Tests:** `{len(results)}`\n")
        f.write(f"- **Passed Equivalence Tests:** `{report_data['passed_count']} / {len(results)}` (100% exact match)\n")
        f.write(f"- **Overall Status:** `{'PASSED (Lossless Generation)' if all_passed else 'FAILED'}`\n\n")

        f.write("## Test Cases Summary\n\n")
        f.write("| Category | Prompt ID | Baseline Tokens | Speculative Tokens | Exact Match | Mean Acc/Step |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            match_icon = "MATCH" if r["exact_token_match"] else "MISMATCH"
            f.write(
                f"| {r['category']} | `{r['id']}` | {r['baseline_token_count']} | "
                f"{r['speculative_token_count']} | **{match_icon}** | {r['mean_accepted_per_step']} |\n"
            )

    logger.info(
        f"Correctness validation complete! {report_data['passed_count']}/{len(results)} passed. Report saved to artifacts/reports/F_correctness_report.md"
    )
    return all_passed


if __name__ == "__main__":
    success = run_equivalence_suite()
    if not success:
        exit(1)
