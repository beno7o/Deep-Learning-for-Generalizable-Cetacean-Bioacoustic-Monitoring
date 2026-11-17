"""Classifiers on frozen Perch 2.0 embeddings (the Perch 2.0 paper's 'linear probe').

* standardisation with TRAIN statistics only (DL Lab 1)
* logistic regression, class_weight='balanced' (like EfficientNet's weighted loss)
* regularisation C chosen on VALIDATION macro-F1 over a log grid (DL Lab 3), then
  the test set is evaluated once
* few-shot: k examples per species from train (k = 4/8/16/32), 5 random draws,
  as in the paper (they report one-vs-all ROC-AUC; we add macro-F1)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluate import compute_metrics, save_evaluation

EMB_FILE = config.DATA_ROOT / "Embeddings" / "perch_v2" / "embeddings.npz"
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def load_embeddings(kind: str = "first"):
    z = np.load(EMB_FILE, allow_pickle=True)
    return pd.Series(range(len(z["filename"])), index=z["filename"]), z[f"emb_{kind}"]


def split_arrays(split_dir, kind="first"):
    pos, E = load_embeddings(kind)
    parts = {s: pd.read_csv(split_dir / f"{s}.csv") for s in ["train", "val", "test"]}
    classes = sorted(parts["train"].category.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    out = {}
    for s, p in parts.items():
        out[s] = (E[pos[p.filename].values], p.category.map(c2i).values, p.filename.values)
    return out, classes


def _fit(Xtr, ytr, C, seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def linear_probe(split: str, kind: str = "first", run: str | None = None, verbose: bool = True):
    split_dir = config.MAIN_SPLIT_DIR if split == "main" else config.GROUPED_SPLIT_DIR
    data, classes = split_arrays(split_dir, kind)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, fte) = data["train"], data["val"], data["test"]

    rows = []
    for C in C_GRID:
        sc, clf = _fit(Xtr, ytr, C)
        m = compute_metrics(yva, clf.predict(sc.transform(Xva)))
        rows.append({"C": C, "val_acc": m["accuracy"], "val_f1_macro": m["f1_macro"]})
        if verbose:
            print(f"C={C:<6} val acc {m['accuracy']:.3f} macro-F1 {m['f1_macro']:.3f}", flush=True)
    grid = pd.DataFrame(rows)
    best_C = float(grid.loc[grid.val_f1_macro.idxmax(), "C"])

    sc, clf = _fit(Xtr, ytr, best_C)
    probs = clf.predict_proba(sc.transform(Xte))
    ypred = probs.argmax(1)
    run = run or f"Perch2_LR_{kind}_{split}"
    test = save_evaluation(run, split, classes, fte, yte, ypred, probs)
    test["roc_auc_ovr_macro"] = roc_auc_score(yte, probs, multi_class="ovr", average="macro",
                                              labels=range(len(classes)))
    best = grid[grid.C == best_C].iloc[0]
    return {"run": run, "split": split, "embedding": kind, "best_C": best_C,
            "val_f1_macro": best.val_f1_macro, "val_acc": best.val_acc, **test}, grid


def few_shot(split: str, kind: str = "first", ks=(4, 8, 16, 32), repeats: int = 5, C: float = 1.0):
    """Train on k random clips per species (from train), evaluate on the full test set."""
    split_dir = config.MAIN_SPLIT_DIR if split == "main" else config.GROUPED_SPLIT_DIR
    data, classes = split_arrays(split_dir, kind)
    (Xtr, ytr, _), (Xte, yte, _) = data["train"], data["test"]
    rows = []
    for k in ks:
        for r in range(repeats):
            rng = np.random.default_rng(1000 * k + r)
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0], size=min(k, (ytr == c).sum()), replace=False)
                                  for c in range(len(classes))])
            sc, clf = _fit(Xtr[idx], ytr[idx], C, seed=r)
            probs = clf.predict_proba(sc.transform(Xte))
            # classes with fewer than k train clips still appear (with all their clips)
            m = compute_metrics(yte, probs.argmax(1))
            full = np.zeros((len(yte), len(classes)))
            full[:, clf.classes_] = probs
            rows.append({"split": split, "k": k, "repeat": r, "accuracy": m["accuracy"],
                         "f1_macro": m["f1_macro"],
                         "roc_auc": roc_auc_score(yte, full, multi_class="ovr", average="macro",
                                                  labels=range(len(classes)))})
    return pd.DataFrame(rows)
