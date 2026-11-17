"""EfficientNet training (2025 recipe).

Recipe (unchanged from 2025 so only the data differs):
    full fine-tuning of ImageNet EfficientNet-B0, batch 16, AdamW lr 1e-3 wd 1e-4,
    class-weighted cross-entropy, ReduceLROnPlateau(0.5, patience 2) on val loss,
    gradient clipping 1.0, max 10 epochs, early stopping patience 3 on val loss,
    best checkpoint = lowest val loss.
Additions: fixed random seed, mixed precision (AMP) on GPU, auto-numbered
checkpoints so reruns never overwrite earlier results (REPSOL convention).

Usage: python -m src.EfficientNet.train --split main --run EfficientNet_01
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src import config
from src.dataloaders import get_dataloaders
from src.EfficientNet.model import get_model, get_weighted_criterion


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def next_checkpoint_path(run_name: str) -> Path:
    """Models_output/<run>_best_01.pth, _02, ... (never overwrite)."""
    config.MODELS_OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = run_name.lower()
    k = 1
    while (config.MODELS_OUTPUT / f"{stem}_best_{k:02d}.pth").exists():
        k += 1
    return config.MODELS_OUTPUT / f"{stem}_best_{k:02d}.pth"


class Trainer:
    def __init__(self, split_dir: Path, checkpoint_path: Path, batch_size: int = 16,
                 max_epochs: int = 10, patience: int = 3, lr: float = 1e-3,
                 device: str | None = None, seed: int = config.SEED):
        set_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs, self.patience = max_epochs, patience
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = self.checkpoint_path.with_name(f"{self.checkpoint_path.stem}_training_history.csv")

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            split_dir, batch_size=batch_size, pin_memory=self.device.startswith("cuda"))
        self.classes = self.train_loader.dataset.classes
        self.num_classes = len(self.classes)

        self.model = get_model("efficientnet", self.num_classes, freeze_backbone=False).to(self.device)
        train_labels = [lbl for _, lbl in self.train_loader.dataset.samples]
        self.criterion = get_weighted_criterion(train_labels, self.num_classes, device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        self.use_amp = self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            torch.backends.cudnn.benchmark = True

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        with open(self.history_path, "w", encoding="utf-8") as fh:
            fh.write("epoch,train_loss,val_loss,train_acc,val_acc,lr,seconds\n")
        print(f"device={self.device} | classes={self.num_classes} | train={len(self.train_loader.dataset)} "
              f"val={len(self.val_loader.dataset)} test={len(self.test_loader.dataset)}", flush=True)

    def _run_epoch(self, loader, train: bool, desc: str):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, desc=desc, ncols=100, unit="batch", leave=False,
                             disable=not sys.stderr.isatty()):  # no bars inside notebooks
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    out = self.model(x)
                    loss = self.criterion(out, y)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return total_loss / len(loader), 100.0 * correct / total

    def fit(self):
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_loader, True, f"epoch {epoch} train")
            va_loss, va_acc = self._run_epoch(self.val_loader, False, f"epoch {epoch} val")
            secs = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{self.max_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.2f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.2f} | lr {lr:.2e} | {secs:.0f}s", flush=True)
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(f"{epoch},{tr_loss:.6f},{va_loss:.6f},{tr_acc:.4f},{va_acc:.4f},{lr},{secs:.1f}\n")
            self.scheduler.step(va_loss)
            if va_loss < self.best_val_loss:
                self.best_val_loss = va_loss
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  saved best checkpoint -> {self.checkpoint_path.name}", flush=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"  early stopping (no val-loss improvement for {self.patience} epochs)", flush=True)
                    break
        # reload best weights for evaluation
        self.model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True))
        print("Training finished.", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["main", "grouped"], default="main")
    ap.add_argument("--run", default="EfficientNet_01")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    split_dir = config.MAIN_SPLIT_DIR if args.split == "main" else config.GROUPED_SPLIT_DIR
    Trainer(split_dir, next_checkpoint_path(args.run), max_epochs=args.epochs).fit()
