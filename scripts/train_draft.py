
"""Train Medusa speculative draft heads on extracted trajectory dataset."""

import json
import os
import time
import torch
from torch.utils.data import DataLoader

from specdecode.data.dataset_manifest import DatasetManifest
from specdecode.data.trajectory_dataset import TrajectoryDataset, collate_trajectories
from specdecode.models.medusa_head import MedusaHeads
from specdecode.training.trainer import MedusaTrainer
from specdecode.utils.logging import get_logger
from specdecode.utils.seed import set_seed

logger = get_logger("train_draft")


def train_draft():
    set_seed(42)
    manifest_path = "data/trajectories/manifest.json"
    manifest = DatasetManifest.load(manifest_path)

    hidden_size = manifest.hidden_size
    vocab_size = manifest.vocab_size
    num_heads = 3
    num_layers = 1
    batch_size = 4
    num_epochs = 10
    lr = 1e-3

    logger.info(f"Initializing Medusa heads: hidden={hidden_size}, vocab={vocab_size}, heads={num_heads}...")
    medusa_heads = MedusaHeads(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    param_count = medusa_heads.count_parameters()
    logger.info(f"Medusa heads parameter count: {param_count:,} ({param_count * 2 / (1024*1024):.2f} MB in BF16)")

    logger.info("Loading train and val trajectory datasets...")
    train_dataset = TrajectoryDataset("data/trajectories", split="train")
    val_dataset = TrajectoryDataset("data/trajectories", split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = MedusaTrainer(
        medusa_heads=medusa_heads,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=1e-4,
        num_epochs=num_epochs,
        grad_clip=1.0,
        device=device,
        checkpoint_dir="checkpoints/medusa",
    )

    t0 = time.time()
    history = trainer.train()
    training_duration = time.time() - t0

    # Save final report
    os.makedirs("artifacts/reports", exist_ok=True)
    report_data = {
        "target_model": manifest.target_model_id,
        "num_heads": num_heads,
        "parameter_count": param_count,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "initial_lr": lr,
        "training_duration_seconds": round(training_duration, 2),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "final_val_loss": round(history["val_loss"][-1], 4),
        "best_val_loss": round(min(history["val_loss"]), 4),
        "final_val_acc_head_0": round(history["val_acc_head_0"][-1], 4),
        "final_val_acc_head_1": round(history["val_acc_head_1"][-1], 4),
        "final_val_acc_head_2": round(history["val_acc_head_2"][-1], 4),
        "history": history,
    }

    with open("artifacts/reports/E_training_report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    with open("artifacts/reports/E_training_report.md", "w") as f:
        f.write("# Speculative Component Training Report\n\n")
        f.write(f"- **Target Model:** `{manifest.target_model_id}`\n")
        f.write(f"- **Speculative Architecture:** Medusa Multi-Head ({num_heads} heads)\n")
        f.write(f"- **Trainable Parameters:** `{param_count:,}`\n")
        f.write(f"- **Precision:** Mixed Precision (`bfloat16`)\n")
        f.write(f"- **Optimizer:** AdamW (lr={lr}, weight_decay=1e-4, betas=(0.9, 0.95))\n")
        f.write(f"- **Scheduler:** CosineAnnealingLR\n")
        f.write(f"- **Gradient Clipping:** `1.0`\n")
        f.write(f"- **Epochs:** `{num_epochs}`\n")
        f.write(f"- **Wall-Clock Duration:** `{report_data['training_duration_seconds']}` s\n")
        f.write(f"- **Best Validation Loss:** `{report_data['best_val_loss']}`\n\n")

        f.write("## Validation Accuracy by Head Position\n\n")
        f.write(f"- **Head 0 (Lookahead +1):** `{report_data['final_val_acc_head_0'] * 100:.2f}%`\n")
        f.write(f"- **Head 1 (Lookahead +2):** `{report_data['final_val_acc_head_1'] * 100:.2f}%`\n")
        f.write(f"- **Head 2 (Lookahead +3):** `{report_data['final_val_acc_head_2'] * 100:.2f}%`\n\n")

        f.write("## Training History\n\n")
        f.write("| Epoch | Train Loss | Val Loss | Head 0 Val Acc | Head 1 Val Acc | Head 2 Val Acc | LR |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for ep in range(num_epochs):
            f.write(
                f"| {ep+1} | {history['train_loss'][ep]:.4f} | {history['val_loss'][ep]:.4f} | "
                f"{history['val_acc_head_0'][ep]*100:.2f}% | {history['val_acc_head_1'][ep]*100:.2f}% | "
                f"{history['val_acc_head_2'][ep]*100:.2f}% | {history['lr'][ep]:.2e} |\n"
            )

    logger.info(
        f"Training complete in {training_duration:.1f}s! Best Val Loss: {report_data['best_val_loss']}. Report saved to artifacts/reports/E_training_report.md"
    )


if __name__ == "__main__":
    train_draft()
