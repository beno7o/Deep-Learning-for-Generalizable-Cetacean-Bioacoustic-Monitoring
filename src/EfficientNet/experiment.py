"""Run one training configuration with several seeds (DL Lab 3: confirm with seeds).

Model selection rule (decided before looking at results): configurations are
compared on mean VALIDATION macro-F1 over seeds. Test metrics are recorded for
the report but never used to choose settings.

Writes per seed: checkpoint + history (Models_output/), test predictions/reports
(outputs/evaluation/<run>_s<seed>), and one row in outputs/evaluation/experiments.csv.
"""
import json
import time

import pandas as pd
import torch

from src import config
from src.augment import SpectrogramAugment
from src.EfficientNet.train import Trainer, next_checkpoint_path
from src.evaluate import compute_metrics, predict, save_evaluation

SEEDS = (42, 43, 44)
EXPERIMENTS_CSV = config.EVAL_DIR / "experiments.csv"


def run_seeds(run: str, split: str = "grouped", seeds=SEEDS, augment: bool = False,
              augment_kwargs: dict | None = None, **trainer_kwargs) -> pd.DataFrame:
    split_dir = config.MAIN_SPLIT_DIR if split == "main" else config.GROUPED_SPLIT_DIR
    rows = []
    for seed in seeds:
        t0 = time.time()
        name = f"{run}_s{seed}"
        tf = SpectrogramAugment(seed=seed, **(augment_kwargs or {})) if augment else None
        trainer = Trainer(split_dir, next_checkpoint_path(name), seed=seed, train_transform=tf,
                          random_offset=augment, **trainer_kwargs)
        trainer.fit()

        yv, ypv, _ = predict(trainer.model, trainer.val_loader, trainer.device)
        val = compute_metrics(yv, ypv)
        yt, ypt, pt = predict(trainer.model, trainer.test_loader, trainer.device)
        test = save_evaluation(name, split, trainer.classes, trainer.test_loader.dataset.filenames, yt, ypt, pt)

        rows.append({
            "run": run, "seed": seed, "split": split,
            "config": json.dumps({"augment": augment, **(augment_kwargs or {}), **trainer_kwargs}),
            "best_epoch": trainer.best_epoch, "epochs_run": trainer.epochs_run,
            "val_loss": round(trainer.best_val_loss, 4),
            "val_acc": round(val["accuracy"], 4), "val_f1_macro": round(val["f1_macro"], 4),
            "test_acc": round(test["accuracy"], 4), "test_f1_macro": round(test["f1_macro"], 4),
            "minutes": round((time.time() - t0) / 60, 1),
        })
        print(f"--> {name}: val F1 {val['f1_macro']:.3f} acc {val['accuracy']:.3f} | "
              f"best epoch {trainer.best_epoch}/{trainer.epochs_run}", flush=True)
        del trainer
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    old = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if len(old):
        old = old[old.run != run]  # rerunning a config replaces its rows
    pd.concat([old, df], ignore_index=True).to_csv(EXPERIMENTS_CSV, index=False)
    return df


def summarise(path=EXPERIMENTS_CSV) -> pd.DataFrame:
    """Mean ± std over seeds per run. Sorted by validation macro-F1 (the selection metric)."""
    df = pd.read_csv(path)
    g = df.groupby("run")
    out = pd.DataFrame({
        "seeds": g.size(),
        "val_f1_macro": g.val_f1_macro.mean(), "val_f1_std": g.val_f1_macro.std(),
        "val_acc": g.val_acc.mean(),
        "test_f1_macro": g.test_f1_macro.mean(), "test_f1_std": g.test_f1_macro.std(),
        "test_acc": g.test_acc.mean(), "test_acc_std": g.test_acc.std(),
        "best_epoch": g.best_epoch.mean(), "minutes_per_seed": g.minutes.mean(),
    })
    return out.sort_values("val_f1_macro", ascending=False).round(3)
