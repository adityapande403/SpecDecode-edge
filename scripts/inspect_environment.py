"""Environment inspection script for SpecDecode Edge.
Collects and audits hardware, OS, CUDA runtime, Python packages, and model cache status.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch


def inspect_environment():
    report = {}

    # System and OS
    report["os"] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

    # Python
    report["python"] = {
        "version": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
    }

    # RAM
    vmem = psutil.virtual_memory()
    report["memory"] = {
        "total_gb": round(vmem.total / (1024**3), 2),
        "available_gb": round(vmem.available / (1024**3), 2),
        "used_gb": round(vmem.used / (1024**3), 2),
        "percent_used": vmem.percent,
    }

    # Storage
    disk_c = shutil.disk_usage("C:\\") if os.path.exists("C:\\") else None
    disk_d = shutil.disk_usage("D:\\") if os.path.exists("D:\\") else None
    report["storage"] = {
        "drive_c": {
            "total_gb": round(disk_c.total / (1024**3), 2) if disk_c else None,
            "free_gb": round(disk_c.free / (1024**3), 2) if disk_c else None,
        },
        "drive_d": {
            "total_gb": round(disk_d.total / (1024**3), 2) if disk_d else None,
            "free_gb": round(disk_d.free / (1024**3), 2) if disk_d else None,
        },
    }

    # GPU & CUDA
    cuda_available = torch.cuda.is_available()
    gpu_info = {
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }

    if cuda_available:
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_info["total_vram_gb"] = round(gpu_props.total_memory / (1024**3), 2)
        gpu_info["major_capability"] = gpu_props.major
        gpu_info["minor_capability"] = gpu_props.minor
        gpu_info["compute_capability"] = f"sm_{gpu_props.major}{gpu_props.minor}"
        gpu_info["supported_arch_list"] = (
            torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        )
        # nvidia-smi driver check
        try:
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
            gpu_info["nvidia_driver"] = smi_out
        except Exception as e:
            gpu_info["nvidia_driver"] = f"Error querying driver: {e}"
    report["gpu"] = gpu_info

    # Compilers & Tooling
    tooling = {}
    for tool in ["nvcc", "cl", "git"]:
        path = shutil.which(tool)
        tooling[tool] = {"found": path is not None, "path": path}
        if path:
            try:
                res = subprocess.check_output([tool, "--version"], text=True, stderr=subprocess.STDOUT)
                tooling[tool]["version"] = res.splitlines()[0]
            except Exception:
                tooling[tool]["version"] = "Found (version check failed)"
    report["tooling"] = tooling

    # Python Packages
    pkgs = {}
    for pkg in [
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "datasets",
        "numpy",
        "yaml",
        "tqdm",
        "pytest",
        "rich",
    ]:
        try:
            mod = __import__(pkg)
            pkgs[pkg] = getattr(mod, "__version__", "Installed")
        except ImportError:
            pkgs[pkg] = "Not installed"
    report["packages"] = pkgs

    # HF Cache Scan
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    cached_models = []
    if os.path.exists(hf_cache_dir):
        for entry in os.listdir(hf_cache_dir):
            if entry.startswith("models--"):
                cached_models.append(entry.replace("models--", "").replace("--", "/"))
    report["hf_cached_models"] = cached_models

    return report


def main():
    report = inspect_environment()
    os.makedirs("artifacts/reports", exist_ok=True)

    json_path = "artifacts/reports/A_environment_audit.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = "artifacts/reports/A_environment_audit.md"
    with open(md_path, "w") as f:
        f.write("# Environment Audit Report\n\n")
        f.write(f"- **OS**: {report['os']['platform']}\n")
        f.write(f"- **Python**: {report['python']['version'].splitlines()[0]} (`{report['python']['executable']}`)\n")
        f.write(f"- **System RAM**: {report['memory']['total_gb']} GB (Available: {report['memory']['available_gb']} GB)\n")
        f.write(f"- **Storage**: C Drive {report['storage']['drive_c']['free_gb']} GB free / D Drive {report['storage']['drive_d']['free_gb']} GB free\n")
        f.write(f"- **GPU**: {report['gpu']['device_name'] if report['gpu']['cuda_available'] else 'None'}\n")
        f.write(f"- **VRAM**: {report['gpu'].get('total_vram_gb', 'N/A')} GB\n")
        f.write(f"- **Compute Capability**: {report['gpu'].get('compute_capability', 'N/A')}\n")
        f.write(f"- **NVIDIA Driver**: {report['gpu'].get('nvidia_driver', 'N/A')}\n")
        f.write(f"- **PyTorch**: {report['gpu']['torch_version']} (CUDA {report['gpu']['torch_cuda_version']})\n")
        f.write(f"- **Supported CUDA Archs**: {', '.join(report['gpu'].get('supported_arch_list', []))}\n\n")

        f.write("## Installed Key Packages\n")
        for k, v in report["packages"].items():
            f.write(f"- `{k}`: {v}\n")

        f.write("\n## Tooling\n")
        for k, v in report["tooling"].items():
            f.write(f"- `{k}`: {'Available (' + v.get('path', '') + ')' if v['found'] else 'Not found'}\n")

        f.write("\n## Cached HuggingFace Models\n")
        if report["hf_cached_models"]:
            for m in report["hf_cached_models"]:
                f.write(f"- `{m}`\n")
        else:
            f.write("- None detected in default cache\n")

    print(f"Audit report saved to {json_path} and {md_path}")
    print(f"Summary: GPU={report['gpu'].get('device_name')} ({report['gpu'].get('total_vram_gb')} GB VRAM), "
          f"RAM={report['memory']['total_gb']} GB, D Free={report['storage']['drive_d']['free_gb']} GB, "
          f"PyTorch={report['gpu']['torch_version']}")


if __name__ == "__main__":
    main()
