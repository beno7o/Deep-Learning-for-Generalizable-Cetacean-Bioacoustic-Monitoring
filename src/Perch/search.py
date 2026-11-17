"""Perch-Hoplite similarity search on the Watkins Perch 2.0 embeddings.

Builds a Hoplite database (SQLite + USearch vector index, the format Hoplite's agile-modeling
notebooks use) from the existing embeddings, then asks: for a query clip, are its nearest
neighbours the same species, and are they from the same recording?

Hoplite mapping: deployment = Watkins recording (tape, first 5 chars of the filename),
recording = clip file, window = first 5 s of the clip, annotation = species label.
Embeddings are L2-normalised so Hoplite's inner-product index = cosine similarity.

Run with the Perch venv: .venv-perch\\Scripts\\python -m src.Perch.search
"""
import shutil

import numpy as np
import pandas as pd
from perch_hoplite.db import datatypes, db_loader

from src import config

EMB_PATH = config.DATA_ROOT / "Embeddings" / "perch_v2" / "embeddings.npz"
DB_DIR = config.DATA_ROOT / "Hoplite" / "watkins_perch_v2"
OUT = config.EVAL_DIR
KS = (1, 5, 10)
SEARCH_LIST = 100  # retrieve this many, then drop the query itself / its own recording


def load_embeddings():
    z = np.load(EMB_PATH, allow_pickle=True)
    df = pd.DataFrame({"filename": z["filename"], "category": z["category"]})
    df["recording"] = df.filename.str[:5]
    emb = z["emb_first"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return df, emb


def build_db(df: pd.DataFrame, emb: np.ndarray, rebuild: bool = False):
    if DB_DIR.exists() and not rebuild:
        db = _open(DB_DIR)
        if db.count_embeddings() == len(df):
            return db
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    db = db_loader.create_new_usearch_db(str(DB_DIR), emb.shape[1])
    deployment_ids = {r: db.insert_deployment(name=r, project="watkins") for r in df.recording.unique()}
    windows = []
    for row in df.itertuples():
        rec_id = db.insert_recording(filename=row.filename, deployment_id=deployment_ids[row.recording])
        db.insert_annotation(recording_id=rec_id, offsets=[0.0, 5.0], label=row.category,
                             label_type=datatypes.LabelType.POSITIVE, provenance="watkins_metadata")
        windows.append({"recording_id": rec_id, "offsets": [0.0, 5.0]})
    db.insert_windows_batch(windows, emb)
    db.commit()
    return db


def ranked(db, query: np.ndarray, k: int):
    """Hoplite search, sorted best first (TopKSearchResults is kept as an unsorted heap)."""
    res = db.search(query, search_list_size=k)
    return sorted(res.search_results, key=lambda r: -r.sort_score)


def _open(path):
    from perch_hoplite.db import sqlite_usearch_impl
    return sqlite_usearch_impl.SQLiteUSearchDB.create(db_path=str(path))


def evaluate(db, df: pd.DataFrame, emb: np.ndarray):
    """For every clip as query: species precision@k with and without its own recording."""
    # window ids were inserted in dataframe order
    win_ids = np.array(sorted(db.match_window_ids()))
    row_of = {w: i for i, w in enumerate(win_ids)}
    cat, rec = df.category.to_numpy(), df.recording.to_numpy()
    rows = []
    for q in range(len(df)):
        nn = [row_of[r.window_id] for r in ranked(db, emb[q], SEARCH_LIST) if row_of[r.window_id] != q]
        other = [i for i in nn if rec[i] != rec[q]]
        r = {"query": q, "category": cat[q], "recording": rec[q]}
        for k in KS:
            r[f"p@{k}_any"] = np.mean(cat[nn[:k]] == cat[q]) if len(nn) >= k else np.nan
            r[f"same_rec@{k}"] = np.mean(rec[nn[:k]] == rec[q]) if len(nn) >= k else np.nan
            r[f"p@{k}_other_rec"] = np.mean(cat[other[:k]] == cat[q]) if len(other) >= k else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(per_query: pd.DataFrame) -> pd.DataFrame:
    """Macro average over species (each species counts equally, like macro-F1)."""
    cols = [c for c in per_query.columns if "@" in c]
    per_species = per_query.groupby("category")[cols].mean()
    # species with < 3 recordings can never find their own species in another recording (or hardly);
    # the 34 species of the grouped split are the fair set for the "other recordings" columns
    n_rec = per_query.groupby("category").recording.nunique()
    fair = n_rec[n_rec >= config.MIN_RECORDINGS_GROUPED].index
    out = pd.DataFrame({"micro (all queries)": per_query[cols].mean(),
                        "macro (mean over species)": per_species.mean(),
                        f"macro ({len(fair)} species with >= 3 recordings)": per_species.loc[fair].mean()}).T
    return out.round(3), per_species


def example(db, df, emb, filename: str, k: int = 5) -> pd.DataFrame:
    q = int(np.flatnonzero(df.filename.to_numpy() == filename)[0])
    win_ids = np.array(sorted(db.match_window_ids()))
    row_of = {w: i for i, w in enumerate(win_ids)}
    out = [(row_of[r.window_id], r.sort_score) for r in ranked(db, emb[q], k + 1) if row_of[r.window_id] != q][:k]
    return pd.DataFrame([{"rank": i + 1, "filename": df.filename[j], "species": df.category[j],
                          "recording": df.recording[j], "same recording as query": df.recording[j] == df.recording[q],
                          "cosine": round(float(s), 3)} for i, (j, s) in enumerate(out)])


if __name__ == "__main__":
    df, emb = load_embeddings()
    db = build_db(df, emb)
    print("windows in Hoplite DB:", db.count_embeddings())
    pq = evaluate(db, df, emb)
    summary, _ = summarise(pq)
    print(summary.T)
