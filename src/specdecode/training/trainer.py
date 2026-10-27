"""Medusa Speculative Heads Training Engine."""

import os
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from specdecode.losses.medusa_loss import MedusaLoss
from specdecode.models.medusa_head import MedusaHeads
from specdecode.utils.logging import get_logger

logger = get_logger("medusa_trainer")


class MedusaTrainer:
    """Trains Medusa speculative heads on extracted target-model hidden state trajectories."""

    def __init__(
        self,
        medusa_heads: MedusaHeads,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 5,
        grad_clip: float = 1.0,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir: str = "checkpoints/medusa",
    ):
        self.medusa_heads = medusa_heads.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.loss_fn = MedusaLoss(
            num_heads=medusa_heads.num_heads,
            decay_factor=0.8,
            ignore_index=-100,
        )

        self.optimizer = torch.optim.AdamW(
            self.medusa_heads.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = len(train_loader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps),
            eta_min=lr * 0.05,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.medusa_heads.train()
        epoch_loss = 0.0
        head_accs = {f"acc_head_{k}": 0.0 for k in range(self.medusa_heads.num_heads)}
        total_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        for batch in pbar:
            hidden_states = batch["hidden_states"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits_list = self.medusa_heads(hidden_states)
                loss, metrics = self.loss_fn(logits_list, labels)

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.medusa_heads.parameters(), self.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
            for k in range(self.medusa_heads.num_heads):
                head_accs[f"acc_head_{k}"] += metrics.get(f"acc_head_{k}", 0.0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc0": f"{metrics.get('acc_head_0', 0):.2%}"})

        avg_loss = epoch_loss / total_batches
        avg_accs = {k: v / total_batches for k, v in head_accs.items()}
        return {"train_loss": avg_loss, **avg_accs}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.medusa_heads.eval()
        val_loss = 0.0
        head_accs = {f"val_acc_head_{k}": 0.0 for k in range(self.medusa_heads.num_heads)}
        total_batches = len(self.val_loader)

        for batch in self.val_loader:
            hidden_states = batch["hidden_states"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits_list = self.medusa_heads(hidden_states)
                loss, metrics = self.loss_fn(logits_list, labels)

            val_loss += loss.item()
            for k in range(self.medusa_heads.num_heads):
                head_accs[f"val_acc_head_{k}"] += metrics.get(f"acc_head_{k}", 0.0)

        avg_val_loss = val_loss / total_batches
        avg_accs = {k: v / total_batches for k, v in head_accs.items()}
        return {"val_loss": avg_val_loss, **avg_accs}

    def train(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        for k in range(self.medusa_heads.num_heads):
            history[f"acc_head_{k}"] = []
            history[f"val_acc_head_{k}"] = []

        best_val_loss = float("inf")
        logger.info(f"Starting Medusa heads training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            cur_lr = self.scheduler.get_last_lr()[0]
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["lr"].append(cur_lr)

            for k in range(self.medusa_heads.num_heads):
                history[f"acc_head_{k}"].append(train_metrics[f"acc_head_{k}"])
                history[f"val_acc_head_{k}"].append(val_metrics[f"val_acc_head_{k}"])

            logger.info(
                f"Epoch {epoch+1:02d} | Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Head0 Acc: {val_metrics['val_acc_head_0']:.2%} | "
                f"Head1 Acc: {val_metrics['val_acc_head_1']:.2%} | Time: {elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_path = os.path.join(self.checkpoint_dir, "best_medusa_heads.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": self.medusa_heads.state_dict(),
                        "val_loss": best_val_loss,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
                logger.info(f"-> Saved new best checkpoint to {best_path} (val_loss={best_val_loss:.4f})")

        return history
