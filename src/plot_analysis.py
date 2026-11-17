"""Plots for training runs and evaluation."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_learning_curves(history_csv: Path, title: str, out_path: Path | None = None):
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train")
    ax[0].plot(h.epoch, h.val_loss, "o-", label="val")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(h.epoch, h.train_acc, "o-", label="train")
    ax[1].plot(h.epoch, h.val_acc, "o-", label="val")
    ax[1].set(title="Accuracy (%)", xlabel="epoch")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes, title: str, out_path: Path | None = None):
    """Row-normalised confusion matrix (each row = true class sums to 1)."""
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.32 * n + 4, 0.32 * n + 3))
    sns.heatmap(cmn, ax=ax, cmap="Blues", vmin=0, vmax=1, square=True,
                xticklabels=classes, yticklabels=classes, cbar_kws={"shrink": 0.6})
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_f1(report_csv: Path, title: str, out_path: Path | None = None):
    rep = pd.read_csv(report_csv, index_col=0)
    rep = rep.drop(index=[i for i in ("accuracy", "macro avg", "weighted avg") if i in rep.index])
    rep = rep.sort_values("f1-score")
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(rep) + 1.5))
    ax.barh(rep.index, rep["f1-score"], color="#4C72B0")
    for y, (f, s) in enumerate(zip(rep["f1-score"], rep["support"])):
        ax.text(f + 0.01, y, f"{f:.2f} (n={int(s)})", va="center", fontsize=7)
    ax.set(xlim=(0, 1.15), xlabel="F1 (test)", title=title)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
