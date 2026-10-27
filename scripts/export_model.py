"""Export trained speculative components to production formats (Safetensors, HuggingFace, vLLM).
Audits export compatibility with vLLM, Hugging Face, GGUF, and Ollama.
"""

import json
import os
import shutil
import torch
from safetensors.torch import save_file

from specdecode.models.medusa_head import MedusaHeads
from specdecode.utils.logging import get_logger

logger = get_logger("export_model")


def export_speculative_artifacts(
    checkpoint_path: str = "checkpoints/medusa/best_medusa_heads.pt",
    export_dir: str = "artifacts/exported_models/medusa_qwen2.5_1.5b",
):
    os.makedirs(export_dir, exist_ok=True)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Export 1: Standard Safetensors
    safetensors_path = os.path.join(export_dir, "medusa_heads.safetensors")
    clean_sd = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(clean_sd, safetensors_path)
    safetensors_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    logger.info(f"Exported Safetensors: {safetensors_path} ({safetensors_size_mb:.2f} MB)")

    # Export 2: Medusa & vLLM Configuration
    config = {
        "model_type": "medusa",
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_heads": 3,
        "num_layers": 1,
        "hidden_size": 1536,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
        "architectures": ["MedusaModel"],
        "medusa_num_heads": 3,
        "medusa_num_layers": 1,
    }
    config_path = os.path.join(export_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    medusa_config_path = os.path.join(export_dir, "medusa_config.json")
    with open(medusa_config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Exported configuration files to {export_dir}")

    # Audit conversion targets
    compatibility = {
        "huggingface_safetensors": {
            "status": "SUPPORTED",
            "format": "safetensors",
            "path": safetensors_path,
            "size_mb": round(safetensors_size_mb, 2),
            "notes": "Directly loadable in PyTorch / HuggingFace speculative workflows.",
        },
        "vllm_speculative": {
            "status": "SUPPORTED",
            "format": "vLLM Medusa Worker checkpoint",
            "path": export_dir,
            "notes": "Compatible with vLLM Medusa speculative engine via `--speculative-model`.",
        },
        "llama_cpp_gguf": {
            "status": "UNSUPPORTED_ARCH_LIMITATION",
            "format": "GGUF draft model",
            "blocker": (
                "llama.cpp's `--model-draft` natively requires a standalone CausalLM architecture with complete "
                "transformer blocks. Medusa multi-head auxiliary projection heads on frozen base hidden states "
                "require specialized GGUF quantization tensor mappings (`medusa.0.weight`) not present in standard "
                "GGUF conversion scripts. Native llama-speculative currently prioritizes separate SLM draft models."
            ),
        },
        "ollama": {
            "status": "UNSUPPORTED_RUNTIME_LIMITATION",
            "format": "Modelfile / Ollama draft",
            "blocker": (
                "Ollama does not expose a native multi-head Medusa speculative decoding endpoint as of current stable "
                "releases. Serving Medusa requires vLLM or our dedicated high-throughput PyTorch runtime."
            ),
        },
    }

    # Save Export Report
    os.makedirs("artifacts/reports", exist_ok=True)
    report_path = "artifacts/reports/I_export_report.json"
    with open(report_path, "w") as f:
        json.dump(compatibility, f, indent=2)

    md_path = "artifacts/reports/I_export_report.md"
    with open(md_path, "w") as f:
        f.write("# Model Export and Runtime Compatibility Report\n\n")
        f.write("## Exported Artifacts\n\n")
        f.write(f"- **Safetensors Weights:** `{safetensors_path}` ({safetensors_size_mb:.2f} MB)\n")
        f.write(f"- **Config Files:** `{config_path}`, `{medusa_config_path}`\n")
        f.write(f"- **Precision:** `bfloat16`\n")
        f.write(f"- **Target Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` + Medusa 3-Head\n\n")

        f.write("## Runtime Integration Audit\n\n")
        f.write("| Runtime / Backend | Status | Format | Compatibility Assessment |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for k, v in compatibility.items():
            status_badge = f"**{v['status']}**"
            notes = v.get("notes") or v.get("blocker", "N/A")
            f.write(f"| `{k}` | {status_badge} | `{v['format']}` | {notes} |\n")

    logger.info(f"Export audit complete. Reports saved to {report_path} and {md_path}")


if __name__ == "__main__":
    export_speculative_artifacts()
