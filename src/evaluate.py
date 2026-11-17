"""Test-set evaluation: predictions, metrics, saved reports."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config


def predict(model, loader, device):
    import torch  # imported here so the Perch environment (no torch) can use this module

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                out = model(x.to(device))
            probs.append(torch.softmax(out.float(), 1).cpu())
            ys.append(y)
    probs = torch.cat(probs).numpy()
    return np.concatenate(ys), probs.argmax(1), probs


def compute_metrics(y_true, y_pred) -> dict:
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        m.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    return m


def save_evaluation(run: str, split: str, classes, filenames, y_true, y_pred, probs,
                    out_dir: Path = config.EVAL_DIR) -> dict:
    """Writes preds_<run>.csv, classification_report_<run>.csv, confusion_matrix_<run>.csv
    and appends a row to summary_metrics.csv. Returns the metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(classes)
    pd.DataFrame({
        "filename": filenames,
        "y_true": [classes[i] for i in y_true],
        "y_pred": [classes[i] for i in y_pred],
        "prob_max": probs.max(1),
        "correct": y_true == y_pred,
    }).to_csv(out_dir / f"preds_{run}.csv", index=False)

    rep = classification_report(y_true, y_pred, labels=range(len(classes)), target_names=classes,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(out_dir / f"classification_report_{run}.csv")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / f"confusion_matrix_{run}.csv")

    m = compute_metrics(y_true, y_pred)
    row = {"run": run, "split": split, "n_classes": len(classes), "n_test": len(y_true),
           **{k: round(v, 4) for k, v in m.items()}}
    summ = out_dir / "summary_metrics.csv"
    new = pd.DataFrame([row])
    if summ.exists():
        old = pd.read_csv(summ)
        new = pd.concat([old[old.run != run], new], ignore_index=True)  # rerun replaces its row
    df = new
    df.to_csv(summ, index=False)
    return m
